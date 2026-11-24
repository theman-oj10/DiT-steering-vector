"""Residual-energy diagnostic for learned concept subspaces (paper Sec. 5.5, Fig. 4, App. E).

`residual_energy` is copied verbatim from notebooks/projection_erasure_v6.ipynb (cell 6).
"""
from __future__ import annotations

from .config import TOP_TOKENS
from .prompts import residual_probe_pair


def residual_energy(er, sub, concept, kind, anchor_mode, probe="mean", top_tokens=None):
    top_tokens = top_tokens or TOP_TOKENS
    probe_pos, probe_neg = residual_probe_pair(concept, kind, anchor_mode)
    ep = er._encode(probe_pos); en = er._encode(probe_neg)
    mp = er._t5_mask(probe_pos); mn = er._t5_mask(probe_neg)
    rows = []
    print(f"Residual energy probe='{probe}'")
    print("  pos:", probe_pos); print("  neg:", probe_neg)
    for carrier in er._carriers():
        if carrier == "pooled":
            h = ep["pooled"].squeeze(0); v = (ep["pooled"] - en["pooled"]).squeeze(0)
        elif carrier in ("t5", "t5_seq"):
            if probe == "token":
                D = er._concept_token_diff(ep[carrier], en[carrier], mp, mn, top_tokens=top_tokens)
                v = D.mean(0); h = er._mmean(ep[carrier], mp)
            else:
                h = er._mmean(ep[carrier], mp); v = h - er._mmean(en[carrier], mn)
        else:
            cp = er._clip_mask(probe_pos); cn = er._clip_mask(probe_neg)
            if probe == "token":
                D = er._concept_token_diff(ep[carrier], en[carrier], cp, cn, top_tokens=top_tokens)
                v = D.mean(0); h = er._mmean(ep[carrier], cp)
            else:
                h = er._mmean(ep[carrier], cp); v = h - er._mmean(en[carrier], cn)
        v = v / (v.norm() + 1e-8)
        U = sub[carrier]["U"].to(h.device)
        before = float(h @ v)
        h1 = h - (h @ U) @ U.T
        after = float(h1 @ v)
        in_span = float(((v @ U) ** 2).sum())
        removed = 1 - after / before if abs(before) > 1e-6 else float("nan")
        rows.append({"probe": probe, "carrier": carrier, "in_span": in_span, "before": before, "after": after, "removed_fraction": removed, "rank": sub[carrier]["U"].shape[1]})
        print(f"{carrier:9s} | in_span={in_span:5.2f} | before={before:8.2f} after={after:8.2f} | removed={100*removed:6.1f}% | rank={sub[carrier]['U'].shape[1]}")
    return rows
