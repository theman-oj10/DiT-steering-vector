# FLUX UnlearnCanvas Experiments

This folder contains later FLUX unlearning experiments using UnlearnCanvas-style object/style prompts and evaluation.

## Files

- `ObjectUnlearnFlux.ipynb`: Full benchmark notebook focused on object unlearning.
- `StyleUnlearnFlux.ipynb`: Full benchmark notebook focused on style unlearning.
- `unlearning_canvas.ipynb`: Scratch/iteration notebook for style/object suppression, ViT classifier tests, layer sweeps, and activation-difference reconstruction.

## Intervention Locations

`ObjectUnlearnFlux.ipynb` and `StyleUnlearnFlux.ipynb` define a notebook-local `FluxSteering` class with two modes:

- `hybrid`: style-oriented branch. Hooks `context_embedder`, `time_text_embed`, and text-side `add_k_proj` / `add_q_proj` inside the 19 double-stream FLUX blocks.
- `pincer_v2`: object-oriented branch. Uses entry-point-only steering:
  - CLIP path: `time_text_embed` input, via a pre-hook on the 768-d pooled CLIP embedding before the SiLU path.
  - T5 path: `context_embedder` output, using mask-aware pooling.
  - It avoids image-patch hooks to reduce background/object entanglement.

`unlearning_canvas.ipynb` reuses the initial concept-sweep toolkit from `../flux_concept_sweep/flux_steering/` for several exploratory tests:

- single-layer style/object suppression, often around layer 18;
- multi-layer weight sweeps across layers 11-25;
- exact reconstruction tests that capture full token activations across all FLUX layers and inject activation differences.

## Relationship To `flux_concept_sweep`

This folder depends on the concept-sweep toolkit for some scratch cells, but the full object/style benchmark notebooks also contain their own `FluxSteering` implementation because the intervention locations changed.
