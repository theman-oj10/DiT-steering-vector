"""Shared command-line setup for the scripts in this folder."""
from __future__ import annotations

import argparse
import gc
import os
import sys

# Make `carrier_erasure` importable when a script is run from anywhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from carrier_erasure import config as C  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", required=True, choices=sorted(C.MODELS), help="model to run")
    p.add_argument("--out-dir", default="outputs", help="root folder for all outputs (default: outputs/)")
    p.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu)")
    p.add_argument("--cpu-offload", action="store_true", help="enable diffusers model CPU offload")
    return p


def setup(args):
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = C.get_model(args.model)
    paths = C.Paths(args.out_dir, spec.key)
    print(f"model={spec.key} ({spec.model_id}) | steps={spec.n_steps} | guidance={spec.guidance} "
          f"| max_seq={spec.max_seq} | device={device} | outputs={paths.root}")
    return spec, paths, device


def free_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
