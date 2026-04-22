# FLUX Concept Sweep

This is the initial modular FLUX steering experiment. The reusable Python code for this branch lives here in `flux_steering/`.

## Files

- `main.ipynb`: End-to-end workflow for loading FLUX, extracting contrastive vectors, caching vectors, sweeping layers/weights/streams, evaluating images, and plotting results.
- `flux_steering/`: Utility package extracted from this iteration.
  - `config.py`: Sweep settings, concept definitions, generation defaults, and Drive paths.
  - `hook_utils.py`: FLUX layer lookup and steering hooks.
  - `vector_utils.py`: Activation capture, contrastive vector creation, and steered generation.
  - `cache_utils.py`: Vector cache and manifest.
  - `image_utils.py`, `file_utils.py`, `visualization_utils.py`, `model_utils.py`: Evaluation, saving, plotting, and model setup helpers.

## Intervention Location

This branch intervenes at FLUX transformer block outputs.

- `hook_utils.get_layer()` maps a global layer index onto FLUX double-stream blocks first, then single-stream blocks.
- Double-stream block outputs are treated as `(text_stream, image_stream)`.
- Single-stream block outputs are treated as image-only.
- `vector_utils.contrastive_steering_vector()` captures mean activations at selected global layer outputs.
- `vector_utils.generate_steered_image()` applies vectors over a layer range with target `image`, `text`, or `both`.

This is the broad layer-sweep version of the project: it is useful for discovering which layer ranges and streams matter, but it is less specialized than the later entry-point/object-unlearning experiments.
