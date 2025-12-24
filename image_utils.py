"""
Image processing and evaluation utilities
"""

import torch
import numpy as np
from PIL import Image
import torchmetrics
import torchmetrics.functional
from typing import Dict, Any, Tuple, List, Optional
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import lpips
from hashlib import sha1

def ensure_same_size(a: Image.Image, b: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Ensure two images have the same size by resizing the second to match the first"""
    if a.size == b.size:
        return a, b
    # Resize steered to baseline's size (BICUBIC for better quality)
    return a, b.resize(a.size, Image.BICUBIC)


def compute_clip_similarity(image: Image.Image, prompt: str) -> float:
    """
    Compute CLIP similarity between a PIL image and a text prompt.
    Loads CLIP model only once and caches it.

    Args:
        image (PIL.Image.Image): Input image.
        prompt (str): Text prompt.

    Returns:
        float: Cosine similarity between image and text.
    """
    # One-time setup
    if not hasattr(compute_clip_similarity, "_init"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        compute_clip_similarity.device = device
        compute_clip_similarity.dtype = dtype
        compute_clip_similarity.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(device)
        compute_clip_similarity.model.eval()
        compute_clip_similarity.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        compute_clip_similarity._text_cache = {}
        compute_clip_similarity._image_cache = {}
        compute_clip_similarity._init = True

    device = compute_clip_similarity.device
    dtype = compute_clip_similarity.dtype
    model: CLIPModel = compute_clip_similarity.model
    processor: CLIPProcessor = compute_clip_similarity.processor
    text_cache = compute_clip_similarity._text_cache
    image_cache = compute_clip_similarity._image_cache

    # Cache image features by content
    image = image.convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor(text=[prompt], return_tensors="pt")

    # Ensure inputs are on the same device as the model
    device = compute_clip_similarity.device  # "cuda" or "cpu"
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        similarity = (image_features @ text_features.T).item()

    return similarity

  
def compute_lpips_score(
    img1: Image.Image, img2: Image.Image, net: str = "alex"
) -> float:
    """
    Compute the LPIPS distance between two images, applying a mask only to img1.

    Args:
        img1:     PIL image (original)
        img2:     PIL image (edited).
        net:      LPIPS backbone: 'alex', 'vgg', or 'squeeze'.

    Returns:
        float: LPIPS distance (lower = more perceptually similar).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LPIPS model once
    if not hasattr(compute_lpips_score, "model"):
        compute_lpips_score.model = lpips.LPIPS(net=net).to(device)

    # Helper: PIL → LPIPS tensor in [-1,1] without torchvision
    def pil_to_lpips_tensor(x: Image.Image):
        x = x.convert("RGB")
        arr = np.array(x).astype(np.float32) / 255.0  # H x W x C
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # C x H x W
        tensor = tensor.unsqueeze(0)  # 1 x C x H x W
        return (tensor * 2.0 - 1.0).to(device)  # scale to [-1,1]

    t1 = pil_to_lpips_tensor(img1)
    # Also resize img2 to match img1 size
    t2 = pil_to_lpips_tensor(img2)

    with torch.no_grad():
        dist = compute_lpips_score.model(t1, t2)

    return dist.item()


def compute_dinov2_similarity(image1: Image.Image, image2: Image.Image) -> float:
    """
    Compute DINOv2 similarity between two PIL images.
    Loads DINOv2 model only once and caches it.

    Args:
        image1 (PIL.Image.Image): First input image.
        image2 (PIL.Image.Image): Second input image.

    Returns:
        float: Cosine similarity between the two images.
    """
    # One-time setup
    if not hasattr(compute_dinov2_similarity, "_init"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        compute_dinov2_similarity.device = device
        compute_dinov2_similarity.dtype = dtype

        # Load DINOv2 model
        from transformers import AutoImageProcessor, AutoModel

        compute_dinov2_similarity.model = AutoModel.from_pretrained(
            "facebook/dinov2-large"
        ).to(device)
        compute_dinov2_similarity.model.eval()
        compute_dinov2_similarity.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-large"
        )
        compute_dinov2_similarity._image_cache = {}
        compute_dinov2_similarity._init = True

    device = compute_dinov2_similarity.device
    dtype = compute_dinov2_similarity.dtype
    model = compute_dinov2_similarity.model
    processor = compute_dinov2_similarity.processor
    image_cache = compute_dinov2_similarity._image_cache

    # Convert images to RGB and prepare inputs
    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")

    # Process both images
    inputs1 = processor(images=image1, return_tensors="pt")
    inputs2 = processor(images=image2, return_tensors="pt")

    # Ensure inputs are on the same device as the model
    inputs1 = {k: v.to(device) for k, v in inputs1.items()}
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}

    with torch.no_grad():
        # Get features for both images
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

        # Extract CLS token features (global image representation)
        features1 = outputs1.last_hidden_state[:, 0]  # [1, 1024] for dinov2-large
        features2 = outputs2.last_hidden_state[:, 0]  # [1, 1024] for dinov2-large

        # Normalize features for cosine similarity
        features1 = F.normalize(features1, p=2, dim=-1)
        features2 = F.normalize(features2, p=2, dim=-1)

        # Compute cosine similarity
        similarity = (features1 @ features2.T).item()

    return similarity


def evaluate_image_edits(
    steered_image: Image.Image,
    baseline_image: Image.Image,
    target_prompt: str,
) -> Tuple[float, float, float]:
    """
    Evaluate an edited image against a baseline.

    Metrics:
      - delta_clip: CLIP(image, text) difference: steered - baseline (higher is better)
      - lpips: perceptual distance between steered and baseline (lower is better)
      - dino_similarity (optional): cosine similarity between images (higher is better)

    Args:
        steered_image: edited image
        baseline_image: original image
        target_prompt: text prompt for CLIP
        compute_dino: also compute DINOv2 image-image similarity
        return_tuple: if True, return (delta_clip, lpips, dino_similarity|None)

    Returns:
        dict with keys {"delta_clip", "lpips", ["dino_similarity"]} OR a tuple.
    """
    # 1) Ensure common spatial size for fair comparisons
    steered_image, baseline_image = ensure_same_size(steered_image, baseline_image)
    # 2) ΔCLIP (re-uses your cached CLIP model)
    steered_clip = compute_clip_similarity(steered_image, target_prompt)
    baseline_clip = compute_clip_similarity(baseline_image, target_prompt)
    delta_clip = float(steered_clip - baseline_clip)
    # 3) LPIPS distance (lower is better)
    lpips_score = float(compute_lpips_score(steered_image, baseline_image))
    # 4) DINOv2 similarity (higher is better)
    dino_sim = float(compute_dinov2_similarity(steered_image, baseline_image))
    return (delta_clip, lpips_score, dino_sim)


def pil_to_uint8(img):
    """Convert PIL image to uint8 format"""
    arr = np.array(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


def generate_baseline_image(pipe, config):
    """Generate baseline image for comparison"""
    img = pipe(
        prompt=config.base_prompt,
        guidance_scale=config.guidance_scale,
        height=config.default_height,
        width=config.default_width,
        num_inference_steps=config.default_steps,
        generator=torch.Generator(device="cpu").manual_seed(config.default_seed),
    ).images[0]
    print("Baseline image generated.")
    if config.save_baseline_image:
        pil_to_uint8(img).save(config.baseline_path)
    return img