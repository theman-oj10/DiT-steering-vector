"""UA / IRA / CRA / OTR and the operating-point composite score (paper Sec. 4.2, 4.4).

Logic copied from notebooks/projection_erasure_v6.ipynb (cells 14, 17 and 19).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ("UA", "IRA", "CRA", "OTR")


def norm_label(x) -> str:
    """Canonicalize a label so 'Van Gogh' == 'Van_Gogh' == 'van gogh'; NaN/None -> ''."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().lower().replace("_", " ")


def pct(mask) -> float:
    """Percentage of True values; NaN for an empty selection."""
    s = pd.Series(mask)
    return float(100.0 * s.mean()) if len(s) else float("nan")


def add_norm_columns(df: pd.DataFrame, cols=("pred_style", "pred_obj", "style", "obj", "concept", "kind", "method")):
    for col in cols:
        if col in df.columns:
            df[col + "_n"] = df[col].map(norm_label)
    return df


def metr(g: pd.DataFrame, concept: str, kind: str) -> dict:
    """UA/IRA/CRA/OTR for one concept on rows that already have *_n columns."""
    cn = norm_label(concept)
    if not len(g):
        return {m: float("nan") for m in METRICS}
    if kind == "object":
        t = g[g.obj_n == cn]
        rt = g[g.obj_n != cn]
        return dict(UA=pct(t.pred_obj_n != cn),
                    IRA=pct(rt.pred_obj_n == rt.obj_n),
                    CRA=pct(rt.pred_style_n == rt.style_n),
                    OTR=pct(t.pred_style_n == t.style_n))
    t = g[g.style_n == cn]
    rt = g[g.style_n != cn]
    return dict(UA=pct(t.pred_style_n != cn),
                IRA=pct(rt.pred_style_n == rt.style_n),
                CRA=pct(rt.pred_obj_n == rt.obj_n),
                OTR=pct(t.pred_obj_n == t.obj_n))


def composite(m: dict) -> float:
    """Operating-point score: 0.4 UA + 0.2 IRA + 0.2 CRA + 0.2 OTR (NaN counted as 0)."""
    parts = [m[k] for k in METRICS if not np.isnan(m[k])]
    if not parts:
        return float("nan")
    return round(0.4 * np.nan_to_num(m["UA"]) + 0.2 * np.nan_to_num(m["IRA"])
                 + 0.2 * np.nan_to_num(m["CRA"]) + 0.2 * np.nan_to_num(m["OTR"]), 1)


def results_table(df: pd.DataFrame, targets, kind: str, seeds) -> pd.DataFrame:
    """Per-concept table (mean over seeds, ± population std of per-seed values) + Ours average.

    `targets` is a list of concept names of this kind. Mirrors notebook cell 17.
    """
    df = add_norm_columns(df.copy())
    body = []
    for concept in targets:
        for method, label in (("ours", "Ours"), ("pswap", "Prompt-swap"), ("negp", "Neg-prompt")):
            g = df[(df.concept_n == norm_label(concept)) & (df.kind_n == norm_label(kind))
                   & (df.method_n == norm_label(method))]
            if not len(g):
                continue
            per_seed = [metr(g[g.seed == sd], concept, kind) for sd in seeds]
            agg = metr(g, concept, kind)
            row = {"Concept": concept, "Method": label}
            for m in METRICS:
                vals = [p[m] for p in per_seed if not np.isnan(p[m])]
                std = np.std(vals) if vals else 0.0
                row[m] = f"{agg[m]:.1f} ± {std:.1f}" if not np.isnan(agg[m]) else "n/a"
                row["_" + m] = agg[m]
            body.append(row)
    T = pd.DataFrame(body)
    if not len(T):
        return T
    display_cols = [c for c in T.columns if not c.startswith("_")]
    ours = T[T.Method == "Ours"]
    if not len(ours):
        return T[display_cols]
    avg = {"Concept": "AVERAGE", "Method": "Ours"}
    for m in METRICS:
        vals = ours["_" + m].dropna()
        avg[m] = f"{vals.mean():.1f}" if len(vals) else "n/a"
    return pd.concat([T[display_cols], pd.DataFrame([avg])], ignore_index=True)
