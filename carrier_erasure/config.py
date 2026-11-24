"""Model settings, hyperparameters and benchmark definitions used in the paper.

Values are copied from notebooks/projection_erasure_v6.ipynb (cells 1, 5 and 12).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str        # short name used on the command line and in file names
    model_id: str   # Hugging Face model id
    family: str     # "flux" | "sd3"
    n_steps: int
    guidance: float  # FLUX: embedded guidance; SD3.5: classifier-free guidance scale
    max_seq: int


MODELS = {
    "flux-schnell": ModelSpec("flux-schnell", "black-forest-labs/FLUX.1-schnell", "flux", 4, 0.0, 256),
    "flux-dev": ModelSpec("flux-dev", "black-forest-labs/FLUX.1-dev", "flux", 28, 3.5, 512),
    "sd3.5-medium": ModelSpec("sd3.5-medium", "stabilityai/stable-diffusion-3.5-medium", "sd3", 28, 4.5, 256),
}


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise SystemExit(f"unknown model {key!r}; choose from {', '.join(MODELS)}")


# ---------------------------------------------------------------------------
# Subspace learning (paper Sec. 3.2 / App. A)
# ---------------------------------------------------------------------------
K_RANK = 4          # fixed rank for pooled and CLIP-seq carriers
TARGET_VAR = 0.95   # adaptive-rank target for T5 carriers
K_MAX_T5 = 24       # rank cap for T5 carriers
TOP_TOKENS = 10     # top-m concept-aligned token rows per pair
N_PAIRS = 50        # contrastive prompt pairs per concept

# ---------------------------------------------------------------------------
# Benchmark grid (paper Sec. 4.1)
# ---------------------------------------------------------------------------
EVAL_STYLES = ["Van_Gogh", "Watercolor", "Cartoon", "Pop_Art", "Winter"]
EVAL_OBJECTS = ["Dogs", "Cats", "Towers", "Birds", "Horses", "Flowers", "Trees", "Sandwiches"]
EVAL_SEEDS = [188, 288]

# Label sets LLaVA classifies over: BLOCK's 10 UnlearnCanvas styles + the 20 UnlearnCanvas objects.
FULL_STYLES = ["Van_Gogh", "Watercolor", "Cartoon", "Cubism", "Winter",
               "Pop_Art", "Ukiyoe", "Impressionism", "Byzantine", "Bricks"]
FULL_OBJECTS = ["Architectures", "Bears", "Birds", "Butterfly", "Cats",
                "Dogs", "Fishes", "Flame", "Flowers", "Frogs",
                "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches",
                "Sea", "Statues", "Towers", "Trees", "Waterfalls"]

OBJECT_TARGETS = ["Dogs", "Cats", "Towers", "Birds", "Flowers"]
STYLE_TARGETS = ["Van_Gogh", "Watercolor", "Cartoon"]
BASELINE_TARGETS = ["Dogs", "Cats", "Towers", "Van_Gogh", "Watercolor"]  # controls subset

# Operating-point search (paper Sec. 4.4 / App. A)
PROXY = [(st, ob, sd) for st in ("Van_Gogh", "Cartoon") for ob in ("Dogs", "Cats", "Towers") for sd in (188,)]
PROXY_TARGET = {"object": "Dogs", "style": "Van_Gogh"}
COHERENCE_MIN = 80.0

# Operating curves (paper Fig. 3)
CURVE_STRENGTHS = [0.5, 1.0, 1.5, 2.0, 3.0]


def benchmark_grid():
    return [(st, ob, sd) for st in EVAL_STYLES for ob in EVAL_OBJECTS for sd in EVAL_SEEDS]


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------


class Paths:
    """All outputs for one model live under <out_dir>/<model key>/."""

    def __init__(self, out_dir: str, model_key: str):
        self.root = os.path.join(out_dir, model_key)
        self.model_key = model_key
        self.subspaces = os.path.join(self.root, "subspaces")
        self.img = os.path.join(self.root, "img")
        self.tables = os.path.join(self.root, "tables")
        self.figs = os.path.join(self.root, "figs")
        for d in (self.subspaces, self.img, self.tables, self.figs):
            os.makedirs(d, exist_ok=True)

    def subspace(self, concept: str, anchor_mode: str) -> str:
        return os.path.join(self.subspaces, f"bench_{concept}_{anchor_mode}.pt")

    def image(self, method: str, concept: str, style: str, obj: str, seed: int) -> str:
        return os.path.join(self.img, f"{self.model_key}__{method}__{concept}__{style}__{obj}__{seed}.jpg")

    @property
    def selected_configs(self) -> str:
        return os.path.join(self.tables, "selected_configs.json")

    @property
    def classifications(self) -> str:
        return os.path.join(self.tables, "classifications.csv")
