"""Joint carrier erasure: the ProjectionErasure class (paper Sec. 3).

The class is copied verbatim from notebooks/projection_erasure_v6.ipynb (cell 3);
only the imports and the pipeline loader below were added.
"""
from __future__ import annotations

import math

import torch
from tqdm.auto import tqdm

from .config import ModelSpec


def load_pipeline(spec: ModelSpec, device: str = "cuda", cpu_offload: bool = False):
    """Load the diffusers pipeline for a model in bfloat16 (as in notebook cell 4)."""
    if spec.family == "flux":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(spec.model_id, torch_dtype=torch.bfloat16)
    else:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(spec.model_id, torch_dtype=torch.bfloat16)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def make_eraser(spec: ModelSpec, device: str = "cuda", cpu_offload: bool = False):
    pipe = load_pipeline(spec, device, cpu_offload)
    return ProjectionErasure(pipe, spec.family, device, spec.n_steps, spec.guidance, spec.max_seq)


class ProjectionErasure:
    def __init__(self, pipe, family, device="cuda", n_steps=28, guidance=0.0, max_seq=256):
        self.pipe = pipe
        self.family = family
        self.device = device
        self.n_steps = n_steps
        self.guidance = guidance
        self.max_seq = max_seq
        self.clip_seq_len = getattr(pipe, "tokenizer_max_length", 77) if family == "sd3" else 0

    def _carriers(self):
        return ["pooled", "t5"] if self.family == "flux" else ["pooled", "clip_seq", "t5_seq"]

    @torch.no_grad()
    def _encode(self, prompt, negative_prompt=""):
        if self.family == "flux":
            pe, pp, ids = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=self.device,
                num_images_per_prompt=1, max_sequence_length=self.max_seq,
            )
            return {"pooled": pp.float(), "t5": pe.float(), "_raw": (pe, pp, ids)}

        pe, npe, pp, npp = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None,
            negative_prompt=negative_prompt or "", negative_prompt_2=None, negative_prompt_3=None,
            do_classifier_free_guidance=True, device=self.device,
            num_images_per_prompt=1, max_sequence_length=self.max_seq,
        )
        B = self.clip_seq_len
        return {
            "pooled": pp.float(),
            "clip_seq": pe[:, :B].float(),
            "t5_seq": pe[:, B:].float(),
            "neg_pooled": npp.float(),
            "neg_clip_seq": npe[:, :B].float(),
            "neg_t5_seq": npe[:, B:].float(),
            "_raw": (pe, npe, pp, npp),
        }

    def _t5_mask(self, prompt):
        tok = self.pipe.tokenizer_2 if self.family == "flux" else self.pipe.tokenizer_3
        return tok(prompt, padding="max_length", max_length=self.max_seq, truncation=True, return_tensors="pt").attention_mask.to(self.device)

    def _clip_mask(self, prompt):
        if self.family != "sd3":
            raise ValueError("_clip_mask is only defined for SD3.5")
        masks = []
        for tok in [self.pipe.tokenizer, self.pipe.tokenizer_2]:
            if tok is None:
                continue
            out = tok(prompt, padding="max_length", max_length=self.clip_seq_len, truncation=True, return_tensors="pt")
            if hasattr(out, "attention_mask") and out.attention_mask is not None:
                masks.append(out.attention_mask.to(self.device))
        if not masks:
            return torch.ones((1, self.clip_seq_len), device=self.device)
        mask = masks[0]
        for m in masks[1:]:
            mask = torch.maximum(mask, m)
        return mask

    @staticmethod
    def _mmean(act, mask):
        m = mask.to(act.device, act.dtype).unsqueeze(-1)
        return (act * m).sum(dim=(0, 1)) / m.sum(dim=(0, 1)).clamp(min=1.0)

    @torch.no_grad()
    def _concept_token_diff(self, seq_pos, seq_neg, mask_pos, mask_neg, top_tokens=8):
        T = min(seq_pos.shape[1], seq_neg.shape[1], mask_pos.shape[1], mask_neg.shape[1])
        a = seq_pos[0, :T].float(); b = seq_neg[0, :T].float()
        valid = (mask_pos[0, :T].bool() | mask_neg[0, :T].bool())
        d = a - b
        norms = d.norm(dim=-1).masked_fill(~valid, float("-inf"))
        n_valid = int(valid.sum().item())
        if n_valid <= 0:
            return d.mean(0, keepdim=True)
        M = min(top_tokens, n_valid)
        idx = norms.topk(M).indices
        return d[idx]

    @staticmethod
    def _rank_for(S, target_var=0.90, k_max=16):
        e = S ** 2
        c = torch.cumsum(e, 0) / e.sum().clamp(min=1e-8)
        k = int((c < target_var).sum().item()) + 1
        return max(1, min(k, k_max))

    @torch.no_grad()
    def learn(self, prompt_pairs, k=4, target_var=0.90, k_max_t5=16, top_tokens=8, verbose=True):
        rows = {c: [] for c in self._carriers()}
        anchors = {c: [] for c in self._carriers()}
        anchors["_tok"] = []

        for pos, neg in tqdm(prompt_pairs, desc="learn subspaces", disable=not verbose):
            ep = self._encode(pos); en = self._encode(neg)
            mp = self._t5_mask(pos); mn = self._t5_mask(neg)

            if self.family == "flux":
                rows["pooled"].append((ep["pooled"] - en["pooled"]).squeeze(0)[None])
                rows["t5"].append(self._concept_token_diff(ep["t5"], en["t5"], mp, mn, top_tokens=top_tokens))
                anchors["pooled"].append(en["pooled"].squeeze(0))
                tn = self._mmean(en["t5"], mn)
                anchors["t5"].append(tn); anchors["_tok"].append(tn)
            else:
                cp = self._clip_mask(pos); cn = self._clip_mask(neg)
                rows["pooled"].append((ep["pooled"] - en["pooled"]).squeeze(0)[None])
                rows["clip_seq"].append(self._concept_token_diff(ep["clip_seq"], en["clip_seq"], cp, cn, top_tokens=top_tokens))
                rows["t5_seq"].append(self._concept_token_diff(ep["t5_seq"], en["t5_seq"], mp, mn, top_tokens=top_tokens))
                anchors["pooled"].append(en["pooled"].squeeze(0))
                anchors["clip_seq"].append(self._mmean(en["clip_seq"], cn))
                tn = self._mmean(en["t5_seq"], mn)
                anchors["t5_seq"].append(tn); anchors["_tok"].append(tn)

        sub = {}
        for c in self._carriers():
            D = torch.cat(rows[c], dim=0).float()
            mu = D.mean(0)
            U, S, _ = torch.linalg.svd(D.T, full_matrices=False)
            kk = self._rank_for(S, target_var=target_var, k_max=k_max_t5) if c in ("t5", "t5_seq") else min(k, U.shape[1])
            ev = float((S[:kk] ** 2).sum() / (S ** 2).sum().clamp(min=1e-8))
            sub[c] = {
                "U": U[:, :kk].contiguous(),
                "anchor": torch.stack(anchors[c]).mean(0),
                "mu": mu.contiguous(),
                "sv": S[:kk].cpu().tolist(),
                "rows": int(D.shape[0]),
                "var_captured": ev,
            }
            if verbose:
                print(f"  {c:8s}: dim={U.shape[0]} k={kk} rows={D.shape[0]} var_captured={ev:.2f}")
        sub["_tok_anchor"] = torch.stack(anchors["_tok"]).mean(0)
        return sub

    @staticmethod
    def _proj(x, U):
        return (x @ U) @ U.T

    def _erase_pooled(self, pooled, sub, strength=1.0, anchor_replace=True):
        U = sub["U"].to(pooled.device); a = sub["anchor"].to(pooled.device)
        out = pooled - strength * self._proj(pooled, U)
        if anchor_replace:
            out = out + strength * self._proj(a.unsqueeze(0), U)
        return out

    def _erase_tokens(self, seq, sub, strength=1.0, token_frac=1.0, neutralize_tokens=False, tok_anchor=None, hard_frac=0.10, mask=None):
        U = sub["U"].to(seq.device)
        P = self._proj(seq, U)
        score = P.norm(dim=-1)
        B, T = score.shape
        valid = torch.ones((B, T), device=seq.device, dtype=torch.bool) if mask is None else mask[:, :T].to(seq.device).bool()
        if int(valid.sum().item()) == 0:
            return seq
        masked_score = score.masked_fill(~valid, float("-inf"))
        if token_frac < 1.0:
            gates = []
            for b in range(B):
                n_valid = int(valid[b].sum().item())
                kk = max(1, min(n_valid, int(math.ceil(token_frac * n_valid))))
                th = masked_score[b].topk(kk).values[-1]
                gates.append((masked_score[b] >= th) & valid[b])
            gate = torch.stack(gates, dim=0).to(seq.dtype).unsqueeze(-1)
        else:
            gate = valid.to(seq.dtype).unsqueeze(-1)
        out = seq - strength * gate * P
        if neutralize_tokens and tok_anchor is not None:
            hards = []
            for b in range(B):
                n_valid = int(valid[b].sum().item())
                kk = max(1, min(n_valid, int(math.ceil(hard_frac * n_valid))))
                th = masked_score[b].topk(kk).values[-1]
                hards.append((masked_score[b] >= th) & valid[b])
            hard = torch.stack(hards, dim=0).unsqueeze(-1)
            anchor = tok_anchor.to(seq.device, seq.dtype).view(1, 1, -1)
            out = torch.where(hard, anchor, out)
        return out

    def _inject_negative_sd3(self, npe, npp, sub, neg_inject_strength=1.0):
        if self.family != "sd3":
            return npe, npp
        B = self.clip_seq_len
        npe2 = npe.float().clone(); npp2 = npp.float().clone()
        if "pooled" in sub and "mu" in sub["pooled"]:
            npp2 = npp2 + neg_inject_strength * sub["pooled"]["mu"].to(npp2.device).view(1, -1)
        if "clip_seq" in sub and "mu" in sub["clip_seq"]:
            npe2[:, :B] = npe2[:, :B] + neg_inject_strength * sub["clip_seq"]["mu"].to(npe2.device).view(1, 1, -1)
        if "t5_seq" in sub and "mu" in sub["t5_seq"]:
            npe2[:, B:] = npe2[:, B:] + neg_inject_strength * sub["t5_seq"]["mu"].to(npe2.device).view(1, 1, -1)
        return npe2.to(npe.dtype), npp2.to(npp.dtype)

    @torch.no_grad()
    def generate(self, prompt, seed, sub=None, mode="all", strength=1.0, strengths=None, token_frac=1.0, neutralize_tokens=False, anchor_replace=True, negative_concept=None, neg_inject_strength=0.0, steps=None, clip_hard_frac=0.10, t5_hard_frac=0.10):
        strengths = strengths or {}
        g = torch.Generator(device=self.device).manual_seed(seed)
        def s_for(name): return strengths.get(name, strength)

        if self.family == "flux":
            e = self._encode(prompt)
            pe, pp, _ = e["_raw"]
            if sub is not None and mode != "none":
                if mode in ("pooled", "all"):
                    pp = self._erase_pooled(e["pooled"], sub["pooled"], strength=s_for("pooled"), anchor_replace=anchor_replace).to(pp.dtype)
                if mode in ("seq", "all"):
                    tm = self._t5_mask(prompt)
                    pe = self._erase_tokens(e["t5"], sub["t5"], strength=s_for("t5"), token_frac=token_frac, neutralize_tokens=neutralize_tokens, tok_anchor=sub.get("_tok_anchor"), hard_frac=t5_hard_frac, mask=tm).to(pe.dtype)
            return self.pipe(prompt_embeds=pe, pooled_prompt_embeds=pp, num_inference_steps=steps or self.n_steps, guidance_scale=self.guidance, generator=g).images[0]

        e = self._encode(prompt, negative_prompt=negative_concept or "")
        pe, npe, pp, npp = e["_raw"]
        if sub is not None and mode != "none":
            cs = e["clip_seq"]; ts = e["t5_seq"]
            if mode in ("pooled", "all"):
                pp = self._erase_pooled(e["pooled"], sub["pooled"], strength=s_for("pooled"), anchor_replace=anchor_replace).to(pp.dtype)
            if mode in ("clip_seq", "seq", "all"):
                cm = self._clip_mask(prompt)
                cs = self._erase_tokens(cs, sub["clip_seq"], strength=s_for("clip_seq"), token_frac=token_frac, neutralize_tokens=neutralize_tokens, tok_anchor=sub["clip_seq"].get("anchor"), hard_frac=clip_hard_frac, mask=cm)
            if mode in ("t5_seq", "seq", "all"):
                tm = self._t5_mask(prompt)
                ts = self._erase_tokens(ts, sub["t5_seq"], strength=s_for("t5_seq"), token_frac=token_frac, neutralize_tokens=neutralize_tokens, tok_anchor=sub["t5_seq"].get("anchor", sub.get("_tok_anchor")), hard_frac=t5_hard_frac, mask=tm)
            pe = torch.cat([cs, ts], dim=1).to(pe.dtype)
        if sub is not None and neg_inject_strength > 0:
            npe, npp = self._inject_negative_sd3(npe, npp, sub, neg_inject_strength=neg_inject_strength)
        return self.pipe(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=npp, num_inference_steps=steps or self.n_steps, guidance_scale=self.guidance, max_sequence_length=self.max_seq, generator=g).images[0]

    def save(self, sub, path):
        blob = {}
        for k, v in sub.items():
            if isinstance(v, dict):
                blob[k] = {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
            else:
                blob[k] = v.cpu() if torch.is_tensor(v) else v
        torch.save(blob, path)
        print("saved", path)

    def load(self, path):
        blob = torch.load(path, map_location=self.device)
        out = {}
        for k, v in blob.items():
            if isinstance(v, dict):
                out[k] = {kk: (vv.to(self.device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
            else:
                out[k] = v.to(self.device) if torch.is_tensor(v) else v
        return out
