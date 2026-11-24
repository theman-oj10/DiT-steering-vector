"""
File management and saving utilities
"""

import re
import hashlib
from pathlib import Path
from PIL import Image


def _slug(s: str) -> str:
    """Convert string to filesystem-safe slug"""
    s = str(s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s[:80]


def _fmt_weight_for_key(w) -> str:
    """Format weight for use in filenames"""
    try:
        return f"{float(w):.9g}"
    except Exception:
        return str(w)


def _fmt_target(t) -> str:
    """Format target string"""
    return str(t)


def build_image_filename(concept, extracted_layer, start_layer, end_layer, weight, target, seed, ext="png"):
    """Build standardized filename for saved images"""
    c = _slug(concept)
    t = _slug(target)
    w = _fmt_weight_for_key(weight)
    x = f"ext{int(extracted_layer)}_" if extracted_layer is not None else ""
    return f"{c}_{x}L{int(start_layer)}-{int(end_layer)}_w{w}_t{t}_seed{int(seed)}.{ext}"


def save_image_unique(img: Image.Image, out_dir: Path, filename: str) -> Path:
    """Save image with unique filename, adding hash if file exists"""
    out_path = out_dir / filename
    if out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix
        h = hashlib.sha1(str(out_path).encode("utf-8")).hexdigest()[:6]
        out_path = out_dir / f"{stem}_{h}{suffix}"
    img.save(out_path)
    return out_path


def _row_key(concept, extracted_layer, start_layer, end_layer, weight, target):
    """Generate unique row key for checkpointing"""
    return f"{concept}|{extracted_layer}|{int(start_layer)}|{int(end_layer)}|{_fmt_weight_for_key(weight)}|{_fmt_target(target)}"


def setup_directories(base_dir: Path, create_images_dir: bool = True):
    """Setup required directories for the workflow"""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = base_dir / "concept_vectors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    results_csv = base_dir / "concept_sweep_results.csv"
    baseline_path = base_dir / "baseline.png"
    
    dirs = {
        "base": base_dir,
        "cache": cache_dir,
        "results_csv": results_csv,
        "baseline_path": baseline_path,
    }
    
    if create_images_dir:
        images_dir = base_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        dirs["images"] = images_dir
    
    return dirs


def mean_or_none(xs):
    xs = xs or []
    return float(sum(xs)/len(xs)) if len(xs) > 0 else None



