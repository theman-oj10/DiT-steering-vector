# SD3.5 Steering: New Method

This folder contains the core implementation and evaluation pipeline for steering Stable Diffusion 3.5 using the MM-DiT architecture.

## 📂 Primary Files
- **`Stable Diffusion Steering.ipynb`**: Implementation of the `SD3Steering` class. This is where the steering vectors are learned (via differences in means) and applied through forward hooks.
- **`Main Eval Pipeline.ipynb`**: Benchmark/evaluation iteration. Note: this notebook currently imports `FluxPipeline` and defines `FluxSteering`, so it appears to be a FLUX evaluation bridge or older pipeline rather than a fully converted SD3.5 pipeline.

## 🪝 Modifying Steering Hooks
To change *where* the model is steered, modify the `__init__` or `learn_vectors` methods in the `SD3Steering` class:

1.  **Locate the Target Layers**: In `__init__`, we currently find `SD3TransformerBlock` modules and target `block.attn.to_out`.
2.  **Update Hook Registration**:
    - Change `self.proj_layers` to point to different internal modules (e.g., `attn.add_k_proj` or `ff.net.2`).
    - The `apply_vectors` context manager will automatically attach hooks to whatever is stored in `self.proj_layers`.
3.  **Adjust Activation Handling**: If the new target layer returns a different structure (e.g., a tuple of image/text features), update the `hook` function within `learn_vectors` and `apply_vectors` to correctly extract the activation tensor.

## ⚙️ How it Works
The method currently targets the **Joint Attention Projection layers** (the output of the fused image-text attention). By steering this unified representation, we can erase concepts from the generative process more efficiently than steering text-only or image-only streams separately.

## Current Intervention Points

- `Stable Diffusion Steering.ipynb`: finds SD3/SD3.5 joint transformer blocks (`SD3TransformerBlock`, `JointTransformerBlock`, or `SD35TransformerBlock`) and hooks attention output projections, usually `block.attn.to_out[0]`, with `block.proj_out` as a fallback.
- `Main Eval Pipeline.ipynb`: currently uses a FLUX `FluxSteering` implementation, not SD3. It hooks double-stream `attn.to_out[0]` and single-stream `proj_out`.
