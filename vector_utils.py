"""
Utilities for creating and manipulating steering vectors
"""

import torch
from hook_utils import get_layer, make_steering_hook, register_steering_range

def random_steering_vector(embed_dim: int, seed: int = 0):
    """Return random [EMBED_DIM] vector (reproducible via seed)."""
    g = torch.Generator(device="cuda").manual_seed(int(seed))
    return torch.randn(embed_dim, generator=g)


def _mean_feature(x: torch.Tensor) -> torch.Tensor:
    """
    Reduce activation to a 1D feature vector by averaging across all dims except the last.
    Works for [B,T,D] (text, t -> tokens), [B,H,W,D] (image), etc.
    """
    if x.ndim == 1:
        return x
    reduce_dims = tuple(range(x.ndim - 1))  # we want to average out all the dims except the last one (D) to maintain the same size
    return x.detach().to(torch.float32).mean(dim=reduce_dims).cpu()


def _capture_layer_means_stream(
    transformer,
    layer_indices: list[int],
    prompt: str,
    stream: str,              # 'image' | 'text' | 'both'
    pipe,                     # diffusion pipeline
    steps: int = 4,
    H: int = 512,
    W: int = 768,
    seed: int | None = None,
):
    """
    Capture mean activations at many layers in a single forward pass.
    Returns:
      stream == 'image' -> dict[int, torch.Tensor [D]]
      stream == 'text'  -> dict[int, torch.Tensor [D]]
      stream == 'both'  -> dict[int, {'image': Tensor|None, 'text': Tensor|None}]
    """
    assert stream in {"image", "text", "both"}
    results_img: dict[int, torch.Tensor] = {}
    results_txt: dict[int, torch.Tensor] = {}

    # Build hooks for all layers up front
    handles = []
    try:
        for li in layer_indices:
            layer_type, mod, _ = get_layer(transformer, li)

            def make_hook(layer_id: int, is_double_stream: bool):
                def cap_hook(module, inputs, output):
                    # All Flux layers return (text, image) tuples
                    # Note: Single-stream layers return (text_passthrough, processed_image)
                    # where text is just carried through from the last double-stream layer
                    enc_h, img_h = output
                    
                    if stream in {"image", "both"}:
                        results_img[layer_id] = _mean_feature(img_h)
                    
                    # Only capture text if it's actually a double-stream layer
                    # (single-stream layers just pass text through unchanged)
                    if is_double_stream and stream in {"text", "both"}:
                        results_txt[layer_id] = _mean_feature(enc_h)
                    
                    return output
                return cap_hook

            is_double = (layer_type == "double")
            handles.append(mod.register_forward_hook(make_hook(li, is_double)))

        # One forward pass for this prompt across all layers being watched
        gen_device = getattr(getattr(pipe, "unet", None), "device", torch.device("cpu"))
        generator = torch.Generator(device=str(gen_device)).manual_seed(1234 if seed is None else int(seed))

        _ = pipe(
            prompt=prompt,
            guidance_scale=0.0,
            height=H,
            width=W,
            num_inference_steps=steps,
            max_sequence_length=256,
            generator=generator,
        )

    finally:
        for h in handles:
            try: 
                h.remove()
            except: 
                pass

    if stream == "image":
        return results_img
    if stream == "text":
        return results_txt

    # both
    out = {}
    for li in layer_indices:
        out[li] = {
            "image": results_img.get(li, None),
            "text":  results_txt.get(li, None),
        }
    # sanity: at least one capture happened
    if not any((v["image"] is not None or v["text"] is not None) for v in out.values()):
        raise RuntimeError("No activations captured for either stream at the requested layers.")
    return out


def contrastive_steering_vector(
    transformer,
    layer_indices: list[int] | int,  # Accept both single layer and list
    pos_prompt: str,
    neg_prompt: str,
    pipe,                            # diffusion pipeline
    stream: str = "both",
    steps: int = 4,
    H: int = 512,
    W: int = 768,
    seed: int | None = None,
    num_seeds: int = 1,              # NEW: Number of seeds to average over
):
    """
    Build contrastive steering vector(s) with automatic multi-seed averaging.
    
    IMPROVED API: Pass num_seeds > 1 to automatically average over multiple seeds.
    This handles the Normalize-Average-Normalize pattern internally.
    
    Args:
        layer_indices: Single layer index or list of layer indices
        seed: Base seed for reproducibility (default: None, uses 0)
        num_seeds: Number of different seeds to average over (default: 1)
                   Recommended: 30-50 for production, 1 for quick testing
                   Uses deterministic sequence: [seed, seed+1, seed+2, ..., seed+num_seeds-1]
        
    Seed Behavior:
        - Same seed + num_seeds = identical seed sequence (fully reproducible)
        - Different concepts with same seed params use same seeds (fair comparison)
        - Example: seed=42, num_seeds=3 → uses seeds [42, 43, 44]
        
    Returns:
      For single layer_idx (int):
        stream == 'image' -> Tensor[D]
        stream == 'text'  -> Tensor[D]
        stream == 'both'  -> {'image': Tensor|None, 'text': Tensor|None}
      For multiple layer_indices (list):
        stream == 'image' -> dict[int, Tensor[D]]
        stream == 'text'  -> dict[int, Tensor[D]]
        stream == 'both'  -> dict[int, {'image': Tensor|None, 'text': Tensor|None}]
    """
    assert stream in {"image", "text", "both"}
    
    # Single seed case - original behavior
    if num_seeds == 1:
        out = _contrastive_steering_single_seed(
            transformer, layer_indices, pos_prompt, neg_prompt,
            pipe, stream, steps, H, W, seed
        )
        return out
    
    # Multi-seed case - collect and average (Normalize-Average-Normalize)
    # Seeds are deterministic: uses [base_seed, base_seed+1, ..., base_seed+num_seeds-1]
    # This ensures reproducibility and fair comparison across different concepts
    vectors_list = []
    base_seed = seed if seed is not None else 0
    
    for seed_idx in range(num_seeds):
        vec = _contrastive_steering_single_seed(
            transformer, layer_indices, pos_prompt, neg_prompt,
            pipe, stream, steps, H, W, base_seed + seed_idx  # Deterministic seed sequence
        )
        vectors_list.append(vec)
    
    # Average the vectors (inline logic from removed average_steering_vectors function)
    # Each vector in vectors_list is already normalized (from _contrastive_steering_single_seed)
    layers = list(vectors_list[0].keys())
    averaged = {}
    
    for layer_idx in layers:
        averaged[layer_idx] = {'image': None, 'text': None}
        
        # Collect image vectors for this layer (already normalized)
        image_vecs = [v[layer_idx]['image'] for v in vectors_list 
                     if layer_idx in v and v[layer_idx]['image'] is not None]
        
        # Collect text vectors for this layer
        text_vecs = [v[layer_idx]['text'] for v in vectors_list 
                    if layer_idx in v and v[layer_idx]['text'] is not None]
        
        # Average and re-normalize image vectors
        if image_vecs:
            stacked = torch.stack(image_vecs)
            mean_vec = stacked.mean(dim=0)
            # Re-normalize the averaged vector
            averaged[layer_idx]['image'] = mean_vec / (mean_vec.norm() + 1e-8)
        
        # Average and re-normalize text vectors
        if text_vecs:
            stacked = torch.stack(text_vecs)
            mean_vec = stacked.mean(dim=0)
            # Re-normalize the averaged vector
            averaged[layer_idx]['text'] = mean_vec / (mean_vec.norm() + 1e-8)
    
    return averaged


def _contrastive_steering_single_seed(
    transformer,
    layer_indices: list[int] | int,
    pos_prompt: str,
    neg_prompt: str,
    pipe,
    stream: str,
    steps: int,
    H: int,
    W: int,
    seed: int | None,
):
    """Internal function for single-seed vector generation."""
    # Handle single layer case
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]
        single_layer = True
    else:
        single_layer = False

    with torch.no_grad():
        pos = _capture_layer_means_stream(transformer, layer_indices, pos_prompt, stream, pipe, steps, H, W, seed)
        neg = _capture_layer_means_stream(transformer, layer_indices, neg_prompt, stream, pipe, steps, H, W, seed)

        if stream in {"image", "text"}:
            out = {}
            for li in layer_indices:
                p = pos.get(li)
                n = neg.get(li)
                if p is None or n is None:
                    out[li] = None
                    continue
                vec = p - n
                out[li] = vec / (vec.norm() + 1e-6)
            
            if single_layer:
                return out[layer_indices[0]]
            return out

        # both
        out = {}
        for li in layer_indices:
            p_img = pos[li]["image"] if li in pos else None
            p_txt = pos[li]["text"]  if li in pos else None
            n_img = neg[li]["image"] if li in neg else None
            n_txt = neg[li]["text"]  if li in neg else None

            item = {"image": None, "text": None}
            if p_img is not None and n_img is not None:
                v = p_img - n_img
                item["image"] = v / (v.norm() + 1e-6)
            if p_txt is not None and n_txt is not None:
                v = p_txt - n_txt
                item["text"] = v / (v.norm() + 1e-6)
            out[li] = item

        # Ensure not all Nones
        if not any(v["image"] is not None or v["text"] is not None for v in out.values()):
            raise RuntimeError("Could not construct contrastive vectors for either stream at any layer.")
            
        if single_layer:
            return out[layer_indices[0]]
        return out


def generate_steered_image(
    pipe,
    generator,
    prompt: str,
    start_layer: int,
    end_layer: int,
    weight: float = 1.0,
    steering_vector: torch.Tensor | dict | None = None,
    height: int = 768,
    width: int = 1360,
    inference_steps: int = 4,
    steer_target: str = "both",
    scale_mode: str = "absolute"
):
    """
    Generate an image with steering applied.
    
    Args:
        steering_vector: Can be:
            - Single vector (Tensor or dict with 'image'/'text' keys): Applied to all layers in range
            - Dict mapping layer_idx -> vector: Layer-specific steering (each layer gets its own vector)
    """
    if end_layer < start_layer:
        raise ValueError("end_layer must be >= start_layer")
    
    # If weight is 0, skip hooks entirely for exact baseline reproduction
    stats = {}
    # Determine if we need layer-specific hooks
    layer_specific_hooks = None
    hook = None
    
    # Check if steering_vector is a dict with integer keys -> layer-specific mapping
    if isinstance(steering_vector, dict) and any(isinstance(k, int) for k in steering_vector.keys()):
        layer_specific_hooks = {}
        for layer_idx in range(start_layer, end_layer + 1):
            if layer_idx in steering_vector: # create a layer-specific hook for each layer
                layer_specific_hooks[layer_idx] = make_steering_hook(
                    steering_vector=steering_vector[layer_idx],
                    w=weight,
                    steer_target=steer_target,
                    scale_mode=scale_mode,
                    stats=stats
                )
    else:
        # Single vector (Tensor or dict with 'image'/'text' keys) -> apply to all layers
        hook = make_steering_hook(
            steering_vector=steering_vector,
            w=weight,
            steer_target=steer_target,
            scale_mode=scale_mode,
            stats=stats
        )
    
    with register_steering_range(pipe.transformer, start_layer, end_layer, hook, layer_specific_hooks):
        img = pipe(
            prompt=prompt,
            guidance_scale=0.0,
            height=height,
            width=width,
            num_inference_steps=inference_steps,
            generator=generator,
        ).images[0]
    
    return img, stats