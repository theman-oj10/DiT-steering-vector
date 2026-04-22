"""
Hook utilities for layer management and steering in Flux models
"""

import gc
import torch
from contextlib import contextmanager


def get_layer(transformer, global_idx: int):
    """Return ('double'|'single', module_ref, local_idx) for a global block index."""
    cfg = transformer.config
    num_double = getattr(cfg, "num_layers")
    num_single = getattr(cfg, "num_single_layers")

    double_blocks = None
    single_blocks = None
    for attr_name in dir(transformer):
        attr = getattr(transformer, attr_name, None)
        if isinstance(attr, torch.nn.ModuleList):
            if len(attr) == num_double:
                double_blocks = attr
            elif len(attr) == num_single:
                single_blocks = attr

    if double_blocks is None or single_blocks is None:
        raise AttributeError("Could not locate double/single block ModuleLists on transformer.")

    if not (0 <= global_idx < (num_double + num_single)):
        raise IndexError(f"Global idx {global_idx} out of range [0, {num_double + num_single - 1}]")

    if global_idx < num_double:
        return "double", double_blocks[global_idx], global_idx
    li = global_idx - num_double
    return "single", single_blocks[li], li


def _expand_vec(vec: torch.Tensor | None, target_dim: int, *, dtype=None, device="cuda"):
    """Normalize vec to shape [1,1,D]; if None, create random [1,1,D]."""
    if vec is None:
        return torch.randn(1, 1, target_dim, dtype=dtype or torch.float32, device=device or "cpu")
    if vec.ndim == 1:
        v = vec[None, None, :]
    elif vec.ndim == 2:
        v = vec[None, :, :]
    elif vec.ndim == 3:
        v = vec
    else:
        raise ValueError("steering_vector must have ndim in {1,2,3}")
    if v.shape[-1] != target_dim:
        raise ValueError(f"steering_vector last dim {v.shape[-1]} != target_dim {target_dim}")
    return v


def make_steering_hook(
    steering_vector: torch.Tensor | dict | None = None,
    w: float = 1.0,
    steer_target: str = "both",
    *,
    scale_mode: str = "absolute",   # "absolute" = use w as-is; "relative" = interpret w as ratio ε
    stats: dict | None = None       # optional dict to collect logs
):
    """
    Create a steering hook for model intervention
    
    Args:
        steering_vector: Can be:
            - torch.Tensor: Single vector applied based on steer_target
            - dict: {"image": tensor, "text": tensor} for stream-specific vectors
            - None: Random vector
    
    - Measures avg L2 norm of latents (per stream) on the *original* output.
    - If scale_mode == "relative", treats w as ε and scales push by ε * avg_norm.
    - Logs avg_norm and realized ratio (w_eff / avg_norm) into `stats`, if provided.
    """
    steer_target = steer_target.lower()
    assert steer_target in {"image", "text", "both"}, "steer_target must be 'image', 'text', or 'both'"

    # Extract stream-specific vectors if dict is provided
    if isinstance(steering_vector, dict):
        image_vec = steering_vector.get("image", None)
        text_vec = steering_vector.get("text", None)
    else:
        # Use same vector for both streams (random vector)
        image_vec = steering_vector
        text_vec = steering_vector

    def hook(module, inputs, output):
        # Unpack streams (double-stream: (encoder_hidden_state, hidden_state); single-stream: tensor)
        if isinstance(output, tuple) and len(output) == 2:
            encoder_hidden_state, hidden_state = output
        else:
            encoder_hidden_state, hidden_state = None, output  # single-stream behaves like image-only

        # ---- 1) Measure avg norms BEFORE modification ----
        avg_norm_img = None
        avg_norm_txt = None
        with torch.no_grad():
            if hidden_state is not None:
                # hidden_state: [B,T,D] or [B,H,W,D] or [B,D]
                avg_norm_img = hidden_state.detach().float().norm(dim=-1).mean()  # scalar tensor
            if encoder_hidden_state is not None:
                avg_norm_txt = encoder_hidden_state.detach().float().norm(dim=-1).mean()

        # ---- 2) Decide effective weights ----
        if scale_mode == "relative":
            # interpret w as ε (ratio). Convert to absolute per-stream.
            w_img_eff = float(w) * (float(avg_norm_img) if avg_norm_img is not None else 0.0)
            w_txt_eff = float(w) * (float(avg_norm_txt) if avg_norm_txt is not None else 0.0)
        else:
            w_img_eff = float(w)
            w_txt_eff = float(w)

        # ---- 3) Apply steering ----
        new_encoder_hidden_state = encoder_hidden_state
        new_hidden_state = hidden_state

        if hidden_state is not None and steer_target in {"image", "both"}:
            image_steering_vector = _expand_vec(image_vec, hidden_state.shape[-1], dtype=hidden_state.dtype, device=hidden_state.device)
            new_hidden_state = hidden_state + (w_img_eff * image_steering_vector.to(hidden_state.device, hidden_state.dtype))

        if encoder_hidden_state is not None and steer_target in {"text", "both"}:
            text_steering_vector = _expand_vec(text_vec, encoder_hidden_state.shape[-1], dtype=encoder_hidden_state.dtype, device=encoder_hidden_state.device)
            new_encoder_hidden_state = encoder_hidden_state + (w_txt_eff * text_steering_vector.to(encoder_hidden_state.device, encoder_hidden_state.dtype))

        # ---- 4) Optional logging (no extra pass) ----
        if stats is not None:
            if avg_norm_img is not None:
                stats.setdefault("avg_norm_image", []).append(float(avg_norm_img))
                stats.setdefault("w_image_eff", []).append(float(w_img_eff))
                stats.setdefault("ratio_image", []).append(float(w_img_eff) / (float(avg_norm_img) + 1e-6))
            if avg_norm_txt is not None:
                stats.setdefault("avg_norm_text", []).append(float(avg_norm_txt))
                stats.setdefault("w_text_eff", []).append(float(w_txt_eff))
                stats.setdefault("ratio_text", []).append(float(w_txt_eff) / (float(avg_norm_txt) + 1e-6))

        return (new_encoder_hidden_state, new_hidden_state) if encoder_hidden_state is not None else new_hidden_state

    return hook


@contextmanager
def register_steering_range(transformer, start_layer: int, end_layer: int, hook = None, layer_specific_hooks: dict = None):
    """
    Register hook(s) on global layers [a..b], then remove on exit.
    
    Args:
        transformer: The transformer model
        a: Start layer index
        b: End layer index
        hook: Single hook to apply to all layers (if layer_specific_hooks is None)
        layer_specific_hooks: Optional dict mapping layer_idx -> hook_fn for layer-specific steering
    """
    handles = []
    try:
        for gi in range(start_layer, end_layer + 1):
            _, mod, _ = get_layer(transformer, gi)
            # Use layer-specific hook if available, otherwise use the shared hook
            hook_to_use = layer_specific_hooks.get(gi, hook) if layer_specific_hooks else hook
            handles.append(mod.register_forward_hook(hook_to_use))
        yield
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

