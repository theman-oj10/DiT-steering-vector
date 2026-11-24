#!/usr/bin/env python3
"""Concept-subspace coverage diagnostics (paper Sec. 5.5, Fig. 4, App. E).

For Dogs (object) and Van_Gogh (style), measures on a held-out probe pair how
much of the concept direction lies in each carrier's learned subspace and how
much of its coefficient a unit-strength projection removes. Uses the anchors of
the selected operating points. Mirrors notebook cell 21.
Writes tables/diag.csv and figs/diag.png. Example:
    python scripts/subspace_diagnostics.py --model flux-dev
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _common import parser, setup  # noqa: E402
from carrier_erasure import benchmark as B  # noqa: E402
from carrier_erasure.diagnostics import residual_energy  # noqa: E402
from carrier_erasure.erasure import make_eraser  # noqa: E402


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    spec, paths, device = setup(args)
    selection = B.load_selection(paths)
    er = make_eraser(spec, device, args.cpu_offload)

    rows = []
    for concept, kind in (("Dogs", "object"), ("Van_Gogh", "style")):
        am = B.anchor_for(kind, selection[kind]["cfg"])
        sub_u = B.get_sub(er, paths, concept, kind, am)
        for r in residual_energy(er, sub_u, concept, kind, anchor_mode=am, probe="token"):
            rows.append(dict(concept=concept, kind=kind, carrier=r["carrier"], in_span=r["in_span"],
                             before=r["before"], after=r["after"],
                             removed=100.0 * r["removed_fraction"], rank=r["rank"]))
    ddf = pd.DataFrame(rows)
    ddf.to_csv(os.path.join(paths.tables, "diag.csv"), index=False)
    print(ddf.to_string(index=False))

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for i, mcol in enumerate(("in_span", "removed")):
        ddf.pivot_table(index="carrier", columns="concept", values=mcol, aggfunc="mean").plot.bar(ax=axs[i], rot=0)
        if mcol == "in_span":
            axs[i].set_title("Concept direction captured by subspace", fontsize=10)
            axs[i].set_ylabel("in-span energy"); axs[i].set_ylim(0, 1.05)
        else:
            axs[i].set_title("Concept coefficient removed", fontsize=10)
            axs[i].set_ylabel("% removed"); axs[i].set_ylim(0, 105)
        axs[i].set_xlabel("carrier"); axs[i].legend(title="Concept", fontsize=8); axs[i].grid(axis="y", alpha=0.3)
    plt.suptitle("Residual-energy diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(paths.figs, "diag.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
