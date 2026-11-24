# Experiment Index

## `flux_concept_sweep/`

Contains the main FLUX autosteering workflow:

- `main.ipynb`: Builds contrastive concept vectors, caches them, runs layer/weight/stream sweeps, evaluates generated images, and visualizes sweep results.
- `flux_steering/`: Python toolkit for this initial sweep branch. It is stored here because these utilities belong to the concept-sweep implementation.
- `README.md`: Notes the files and intervention location.

## `flux_unlearncanvas/`

Contains UnlearnCanvas-oriented FLUX unlearning experiments:

- `ObjectUnlearnFlux.ipynb`: Full object-unlearning benchmark flow with FLUX steering, LLaVA/CLIP-style classification, quality metrics, and comparison tables.
- `StyleUnlearnFlux.ipynb`: Same benchmark structure focused on style unlearning.
- `unlearning_canvas.ipynb`: Scratch/iteration notebook for style or object suppression, ViT classifier experiments, multilayer sweeps, and exact activation-difference reconstruction tests.
- `README.md`: Notes the notebook-local `FluxSteering` variants and how their intervention points differ from the concept-sweep toolkit.

## `sd35_new_method/`

Contains newer method exploration:

- `Stable Diffusion Steering.ipynb`: Defines the `SD3Steering` implementation for Stable Diffusion 3 / 3.5 MM-DiT models.
- `Main Eval Pipeline.ipynb`: Evaluation pipeline notebook. Despite the folder theme, this notebook currently imports `FluxPipeline` and defines `FluxSteering`, so treat it as a FLUX evaluation bridge or older pipeline until it is converted to SD3.
- `README.md`: Local notes for the new-method experiments.
