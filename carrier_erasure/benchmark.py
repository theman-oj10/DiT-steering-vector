"""Benchmark helpers shared by the scripts: subspace cache, generation with a
selected configuration, and the operating-point search candidates.

Logic copied from notebooks/projection_erasure_v6.ipynb (cell 14), with module
globals (er, MODEL_FAMILY, SUB_DIR) turned into explicit arguments.
"""
from __future__ import annotations

import json
import os

from . import config as C
from .prompts import make_pairs, negative_concept_prompt


# ---------------------------------------------------------------------------
# Operating-point candidates (paper Sec. 4.4 / App. A)
# ---------------------------------------------------------------------------

def object_configs(family: str):
    cfgs = [("empty_s1.5", dict(anchor="empty_scene", strength=1.5, strengths=None,
                                token_frac=1.0, neg=False)),
            ("empty_prot", dict(anchor="empty_scene", strength=1.0,
                                strengths={"pooled": 0.5, "clip_seq": 1.5, "t5_seq": 1.5, "t5": 1.5},
                                token_frac=0.35, neg=False)),
            ("hyper_s1.5", dict(anchor="hypernym", strength=1.5, strengths=None,
                                token_frac=1.0, neg=False))]
    if family == "sd3":
        cfgs.append(("empty_prot_neg", dict(anchor="empty_scene", strength=1.0,
                                            strengths={"pooled": 0.5, "clip_seq": 1.5, "t5_seq": 1.5},
                                            token_frac=0.35, neg=True)))
    return cfgs


def style_configs():
    return [(f"style_s{s}", dict(anchor="hypernym", strength=s, strengths=None,
                                 token_frac=1.0, neg=False)) for s in (1.0, 1.5, 2.0)]


def candidate_configs(kind: str, family: str):
    return object_configs(family) if kind == "object" else style_configs()


# ---------------------------------------------------------------------------
# Subspaces and generation
# ---------------------------------------------------------------------------

def anchor_for(kind: str, cfg: dict) -> str:
    """Styles always learn with the hypernym-style pairs; objects use the config's anchor."""
    return cfg["anchor"] if kind == "object" else "hypernym"


def get_sub(er, paths: C.Paths, concept: str, kind: str, anchor_mode: str):
    """Load the cached subspace for (concept, anchor), or learn and cache it."""
    sp = paths.subspace(concept, anchor_mode)
    if os.path.exists(sp):
        return er.load(sp)
    prs = (make_pairs(concept, kind, n=C.N_PAIRS, anchor_mode=anchor_mode)
           if kind == "object" else make_pairs(concept, kind, n=C.N_PAIRS))
    s = er.learn(prs, k=C.K_RANK, target_var=C.TARGET_VAR, k_max_t5=C.K_MAX_T5,
                 top_tokens=C.TOP_TOKENS, verbose=False)
    er.save(s, sp)
    return s


def ours_generate(er, prompt: str, seed: int, sub, cfg: dict, concept: str, kind: str):
    """Generate with joint carrier erasure under one operating-point configuration."""
    neg = (negative_concept_prompt(concept, kind)
           if (cfg.get("neg") and er.family == "sd3") else None)
    return er.generate(prompt, seed, sub=sub, mode="all",
                       strength=cfg.get("strength", 1.0), strengths=cfg.get("strengths"),
                       token_frac=cfg.get("token_frac", 1.0),
                       neutralize_tokens=False, negative_concept=neg)


# ---------------------------------------------------------------------------
# Selected operating points (written by scripts/opsearch.py)
# ---------------------------------------------------------------------------

def save_selection(paths: C.Paths, selection: dict) -> None:
    with open(paths.selected_configs, "w") as f:
        json.dump(selection, f, indent=2)


def load_selection(paths: C.Paths) -> dict:
    """Return {"object": {"name":..., "cfg":...}, "style": {...}}; run scripts/opsearch.py first."""
    if not os.path.exists(paths.selected_configs):
        raise SystemExit(f"{paths.selected_configs} not found: run scripts/opsearch.py for this model first")
    with open(paths.selected_configs) as f:
        return json.load(f)
