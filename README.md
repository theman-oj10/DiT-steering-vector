# DiT Steering Vector Experiments

This repository collects several iterations of activation-steering experiments for diffusion transformer models, mainly FLUX and Stable Diffusion 3 / 3.5.

The folders are organized by experiment branch. Some notebooks are polished benchmark runs; others are scratch iterations kept because they capture useful intervention ideas and diagnostic tests.

## Repository Layout

- `experiments/flux/`: Initial modular FLUX concept-sweep experiment.
  - `main.ipynb`: End-to-end concept-vector extraction, layer/weight/stream sweep, image generation, metric logging, and visualization.
  - `utilities/`: Python utilities that belong to this initial FLUX sweep branch.
- `flux_unlearncanvas/`: Later FLUX object/style unlearning experiments using UnlearnCanvas-style prompts and evaluation.
  - `ObjectUnlearnFlux.ipynb`: Object-unlearning benchmark iteration.
  - `StyleUnlearnFlux.ipynb`: Style-unlearning benchmark iteration.
- `unlearn_canvas_benchmark/`: Earlier/scratch UnlearnCanvas benchmark notebook.
  - `unlearning_canvas.ipynb`: Style/object suppression tests, classifier experiments, multilayer sweeps, and reconstruction diagnostics.
- `experiments/sd3.5/`: SD3.5/new-method steering experiments.
  - `Stable Diffusion Steering.ipynb`: `SD3Steering` class for SD3/SD3.5 MM-DiT steering.
  - `Main Eval Pipeline.ipynb`: Evaluation pipeline iteration; currently still contains FLUX-based code in parts.
- `sd3/`: Separate SD3 experiment branch and logs. This folder was intentionally kept separate from the other organization work.
- `docs/FILE_MAP.md`: More detailed file map and intervention summary.

## FLUX Utility Code

The reusable Python files are not at the repository root. They live with the initial FLUX sweep:

```text
experiments/flux/utilities/
```

Those files cover model setup, configuration, hooks, vector extraction/application, caching, image metrics, file naming, and plotting.

When a notebook outside `experiments/flux/` needs these helpers, it should add the FLUX experiment folder to `sys.path` before importing from `utilities`.

## Intervention Summary

- `experiments/flux/`: Broad FLUX layer sweep. Hooks transformer block outputs across double-stream and single-stream layers, with steering targets for image, text, or both streams.
- `flux_unlearncanvas/`: Later FLUX object/style unlearning. The notebook-local `FluxSteering` variants move away from broad block-output hooks toward entry-point and attention-projection hooks such as `context_embedder`, `time_text_embed`, `add_k_proj`, and `add_q_proj`.
- `unlearn_canvas_benchmark/`: Exploratory UnlearnCanvas tests, including single-layer sweeps, multilayer sweeps, and activation-difference reconstruction.
- `experiments/sd3.5/`: SD3/SD3.5 MM-DiT steering, mainly targeting joint attention output projections such as `attn.to_out[0]`.
- `sd3/`: Separate SD3 branch with its own notebooks, logs, and result images.

## Generated Files

Generated outputs, model weights, caches, local environment files, notebook checkpoints, and result folders are ignored through `.gitignore`. This includes common artifacts such as `.env`, `.DS_Store`, `__pycache__/`, `*.pt`, `*.pth`, `results/`, `outputs/`, `images/`, `plots/`, and `sd3/results/`.
