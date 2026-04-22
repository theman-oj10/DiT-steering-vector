# DiT Steering Vector Experiments

This repository contains iterative experiments for activation steering in diffusion transformer models.

## Layout

- `sd3/`: Existing SD3 notebook and generated results. Left unchanged.
- `experiments/flux_concept_sweep/`: Initial FLUX concept-vector extraction, steering sweeps, metrics, visualization workflow, and its `flux_steering/` Python package.
- `experiments/flux_unlearncanvas/`: Later FLUX object/style unlearning notebooks using UnlearnCanvas-style evaluation.
- `experiments/sd35_new_method/`: Newer steering-method notebooks. The SD3 steering class lives here; one evaluation pipeline still contains FLUX-based code.
- `docs/FILE_MAP.md`: File-by-file explanation and current organization notes.

## FLUX Toolkit

The reusable Python modules live in `experiments/flux_concept_sweep/flux_steering/` because they belong to the initial concept-sweep branch.

Notebooks that reuse those helpers should add `experiments/flux_concept_sweep` to `sys.path` before importing `flux_steering`.
