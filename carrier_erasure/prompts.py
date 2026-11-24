"""Contrastive prompt pairs, anchors and benchmark prompts.

Copied from notebooks/projection_erasure_v6.ipynb (cells 2, 2b and 12).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

IMAGENET_CLASSES = [
    "tench", "goldfish", "tiger shark", "hammerhead", "electric ray",
    "hen", "ostrich", "brambling", "goldfinch", "house finch",
    "junco", "indigo bunting", "robin", "bulbul", "jay",
    "magpie", "chickadee", "water ouzel", "kite", "bald eagle",
    "vulture", "great grey owl", "mud turtle", "box turtle", "banded gecko",
    "common iguana", "whiptail lizard", "agama", "frilled lizard", "alligator lizard",
    "green mamba", "thunder snake", "ringneck snake", "king snake", "garter snake",
    "vine snake", "trilobite", "scorpion", "black widow", "tarantula",
    "centipede", "grouse", "peacock", "quail", "partridge",
    "macaw", "lorikeet", "coucal", "bee eater", "hornbill",
]

OBJECT_CONTEXTS = [
    "a photo of {X} in a park",
    "a photo of {X} on a city street",
    "{X} in a garden",
    "{X} on a beach at sunset",
    "{X} in the snow",
    "a close-up photo of {X}",
    "{X} next to a red bench",
    "{X} near a lake",
    "{X} in a forest",
    "{X} in a living room",
    "{X} on a mountain trail",
    "a photo of {X} at night",
    "{X} in the rain",
    "{X} in a field of grass",
    "{X} beside a wooden fence",
    "{X} in front of a house",
    "{X} on a bridge",
    "{X} in an old town square",
    "{X} under a tree",
    "{X} by a river",
    "a professional photo of {X}",
    "{X} in bright sunlight",
    "{X} on a quiet road",
    "{X} near a fountain",
    "{X} in a courtyard",
]

EMPTY_SCENE_CONTEXT_PAIRS = [
    ("a photo of {X} in a park", "a photo of an empty park"),
    ("a photo of {X} on a city street", "an empty city street"),
    ("{X} in a garden", "an empty garden"),
    ("{X} on a beach at sunset", "an empty beach at sunset"),
    ("{X} in the snow", "an empty snowy scene"),
    ("a close-up photo of {X}", "a close-up photo of an empty background"),
    ("{X} next to a red bench", "an empty red bench"),
    ("{X} near a lake", "an empty lakeside scene"),
    ("{X} in a forest", "an empty forest path"),
    ("{X} in a living room", "an empty living room"),
    ("{X} on a mountain trail", "an empty mountain trail"),
    ("a photo of {X} at night", "an empty night scene"),
    ("{X} in the rain", "an empty rainy scene"),
    ("{X} in a field of grass", "an empty field of grass"),
    ("{X} beside a wooden fence", "an empty wooden fence scene"),
    ("{X} in front of a house", "the front of a house with no foreground subject"),
    ("{X} on a bridge", "an empty bridge"),
    ("{X} in an old town square", "an empty old town square"),
    ("{X} under a tree", "an empty area under a tree"),
    ("{X} by a river", "an empty riverbank"),
    ("a professional photo of {X}", "a professional photo of an empty studio setting"),
    ("{X} in bright sunlight", "an empty sunny outdoor scene"),
    ("{X} on a quiet road", "an empty quiet road"),
    ("{X} near a fountain", "an empty fountain scene"),
    ("{X} in a courtyard", "an empty courtyard"),
]

HYPERNYMS = {
    "Bears": "an animal", "Birds": "an animal", "Butterfly": "an animal",
    "Cats": "an animal", "Dogs": "an animal", "Fishes": "an animal",
    "Frogs": "an animal", "Horses": "an animal", "Jellyfish": "an animal",
    "Rabbits": "an animal", "Human": "a figure",
    "Architectures": "a building", "Towers": "a building",
    "Statues": "an object", "Sandwiches": "some food",
    "Flowers": "a plant", "Trees": "a plant",
    "Sea": "a natural landscape", "Waterfalls": "a natural landscape",
    "Flame": "a soft glow",
}

ANIMAL_CONCEPTS = {
    "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Frogs",
    "Horses", "Jellyfish", "Rabbits"
}

VOWELS = set("aeiou")

def _singular(concept: str) -> str:
    c = concept.replace("_", " ").strip().lower()
    if c.endswith("ies"):
        return c[:-3] + "y"
    if c.endswith("ses"):
        return c[:-2]
    if c.endswith("s") and c not in ("grass", "glass"):
        return c[:-1]
    return c

def _article_for(phrase: str) -> str:
    phrase = phrase.strip().lower()
    if phrase.startswith(("a ", "an ", "the ", "some ")):
        return ""
    return "an " if phrase[:1] in VOWELS else "a "

def object_phrase(concept: str) -> str:
    s = _singular(concept)
    return f"{_article_for(s)}{s}"

def _obj_templates(n=50):
    outs = []
    for t in OBJECT_CONTEXTS:
        outs.append(t)
        outs.append("an image of " + t.replace("a photo of ", "").replace("an image of ", ""))
    return outs[:n]

def anchor_phrase(concept: str, anchor_mode: str) -> str:
    if anchor_mode == "hypernym":
        return HYPERNYMS.get(concept, "something")
    if anchor_mode == "something":
        return "something"
    if anchor_mode == "no_animal":
        return "no animal" if concept in ANIMAL_CONCEPTS else "something"
    raise ValueError(f"anchor_phrase does not handle {anchor_mode}; empty_scene uses paired templates")

def make_pairs(concept: str, kind: str, n=50, anchor_mode="hypernym") -> List[Tuple[str, str]]:
    if kind == "style":
        style = concept.replace("_", " ")
        return [(f"a painting of {base} in {style} style", f"a painting of {base}") for base in IMAGENET_CLASSES[:n]]

    if kind != "object":
        raise ValueError("kind must be 'object' or 'style'")

    x = object_phrase(concept)

    if anchor_mode == "empty_scene":
        pairs = [(pos.format(X=x), neg) for pos, neg in EMPTY_SCENE_CONTEXT_PAIRS]
        aug = []
        for pos, neg in pairs:
            aug.append((pos, neg))
            aug.append(("an image of " + pos.replace("a photo of ", ""), "an image of " + neg))
        return aug[:n]

    neg_x = anchor_phrase(concept, anchor_mode)
    return [(t.format(X=x), t.format(X=neg_x)) for t in _obj_templates(n)]

def make_eval_prompts(concept: str, kind: str) -> List[str]:
    if kind == "object":
        x = object_phrase(concept)
        return [f"A {x} in a park.", f"A {x} in a desert."]
    style = concept.replace("_", " ")
    return [f"A painting of a mountain in {style} style.", f"A painting of a village in {style} style."]

def negative_concept_prompt(concept: str, kind: str) -> Optional[str]:
    if kind == "object":
        return f"a photo of {object_phrase(concept)}"
    if kind == "style":
        return f"{concept.replace('_', ' ')} style"
    return None

def residual_probe_pair(concept: str, kind: str, anchor_mode: str) -> Tuple[str, str]:
    if kind == "object":
        x = object_phrase(concept)
        if anchor_mode == "empty_scene":
            return f"a photo of {x} in a park.", "a photo of an empty park."
        if anchor_mode == "no_animal":
            neg = "a photo of no animal in a park." if concept in ANIMAL_CONCEPTS else "a photo of something in a park."
            return f"a photo of {x} in a park.", neg
        if anchor_mode == "something":
            return f"a photo of {x} in a park.", "a photo of something in a park."
        return f"a photo of {x} in a park.", f"a photo of {HYPERNYMS.get(concept, 'something')} in a park."
    style = concept.replace("_", " ")
    return f"a painting of a mountain in {style} style.", "a painting of a mountain."

def make_presence_pairs(concept: str, n=50):
    """Generic-subject vs empty-scene pairs. Uses the concept's HYPERNYM as the
    subject so the presence direction is concept-agnostic (no dog identity)."""
    hyper = HYPERNYMS.get(concept, "something")
    pairs = [(pos.format(X=hyper), neg) for pos, neg in EMPTY_SCENE_CONTEXT_PAIRS]
    aug = []
    for pos, neg in pairs:
        aug.append((pos, neg))
        aug.append(("an image of " + pos.replace("a photo of ", ""),
                    "an image of " + neg))
    return aug[:n]


# ---------------------------------------------------------------------------
# Benchmark prompts (paper Sec. 4.1 and 4.3)
# ---------------------------------------------------------------------------

def bench_prompt(style: str, obj: str) -> str:
    return f"A {obj.replace('_', ' ')} image in {style.replace('_', ' ')} style."


def swap_prompt(style: str, obj: str, concept: str, kind: str) -> str:
    """Prompt-swap reference: objects -> hypernym; styles -> drop the style phrase."""
    if kind == "object" and obj == concept:
        return f"A {HYPERNYMS.get(concept, 'something')} image in {style.replace('_', ' ')} style."
    if kind == "style" and style == concept:
        return f"A {obj.replace('_', ' ')} image."
    return bench_prompt(style, obj)
