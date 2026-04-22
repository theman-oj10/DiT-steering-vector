"""
Model loading and device setup utilities for Flux Autosteering
"""

import torch
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from google.colab import userdata


def setup_device():
    """Setup and return the appropriate device for computation"""
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_flux_model(repo_id="black-forest-labs/FLUX.1-schnell", device=None):
    """
    Load Flux-schnell model and scheduler
    
    Args:
        repo_id: Model repository ID
        device: Device to load model on (auto-detected if None)
    
    Returns:
        Loaded pipeline
    """
    if device is None:
        device = setup_device()
    
    hf_token = userdata.get("HF_TOKEN")
    
    pipe = DiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    print("✅ Model and scheduler loaded successfully.")
    return pipe


def model_embed_dim(transformer) -> int:
    """Get the embedding dimension of the transformer model"""
    return transformer.config.num_attention_heads * transformer.config.attention_head_dim