# One Carrier Isn't Enough: Redundancy-Aware Concept Suppression in Diffusion Transformers

Code for the AI4GOOD Workshop @ NeurIPS 2026 paper
([OpenReview](https://openreview.net/forum?id=QJ8g4adkOT)).

Amna Jamal\*, Rajendran Ramesh Babu Manoj Narender\*, Samina Ali, Tarun Chandra Nagaraju,
Kevin Zhu, Antonio Mari

<sub>\* Equal contribution.</sub>

**Joint carrier erasure** is a training-free way to stop a Diffusion Transformer from
generating a concept (an object or a style). It learns a low-rank concept subspace in
every text-conditioning *carrier* (FLUX: `pooled`, `T5`; SD3.5: `pooled`, `CLIP-seq`,
`T5-seq`) from contrastive prompt pairs, then projects the concept out of all carriers
jointly before denoising. No model weights are changed.

## Repository layout

```
carrier_erasure/      the method and evaluation code
  config.py           model settings, hyperparameters, benchmark grid, output paths
  prompts.py          contrastive prompt pairs, anchors, benchmark prompts
  erasure.py          ProjectionErasure: learn / erase / generate / save / load
  diagnostics.py      residual-energy subspace diagnostic
  llava_eval.py       LLaVA-1.6 classifier, yes/no probes, four-way audit
  metrics.py          UA / IRA / CRA / OTR and the operating-point score
  benchmark.py        subspace cache, operating-point candidates, generation helpers
scripts/              one runnable script per step (see below)
notebooks/
  projection_erasure_v6.ipynb   the original notebook behind the paper (outputs kept)
archive/              earlier steering-vector experiments, kept for reference only
```

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # FLUX.1-dev and SD3.5 require accepting their licenses on Hugging Face
```

A single GPU with about 40 GB of memory runs everything. The diffusion model is moved to
CPU while LLaVA runs; pass `--cpu-offload` to any script on smaller GPUs.

## Reproducing the paper

Every script takes `--model {flux-schnell,flux-dev,sd3.5-medium}` and `--out-dir`
(default `outputs/`), and writes to `<out-dir>/<model>/{subspaces,img,tables,figs}`.
To run everything for one model:

```bash
scripts/run_all.sh flux-schnell
```

Or step by step:

| Step | Script | Produces | Paper |
|---|---|---|---|
| 1 | `opsearch.py` | `tables/opsearch_*.csv`, `tables/selected_configs.json` | Sec. 4.4, App. A |
| 2 | `generate.py` | baseline, ours, prompt-swap (and SD3.5 negative-prompt) images | Sec. 4.1, 4.3 |
| 3 | `evaluate.py` | `tables/classifications.csv` (LLaVA labels + audit buckets) | Sec. 4.2 |
| 4 | `tables.py` | `tables/results_{object,style}.csv`, `tables/controls_{object,style}.csv` | Tables 1–4 |
| 5 | `figures.py` | `figs/qual.png`, `figs/audit.png`, `figs/audit_examples.png` | Figs. 2, 6–10 |
| 6 | `curves.py` | `figs/curve.png` | Figs. 3, 11, 12 |
| 7 | `subspace_diagnostics.py` | `figs/diag.png` | Fig. 4, App. E |
| 8 | `probes.py` | `figs/recoverability.png`, `figs/hard_concept.png` | App. F, G |
| 9 | `redundancy_ablation.py` | `figs/redundancy_<concept>_<anchor>.png` | Figs. 1, 5 |

Steps 2–9 read the operating points chosen in step 1. Generation and evaluation are
resume-safe: re-running skips images and classifications that already exist.
`redundancy_ablation.py` also takes `--concept`, `--kind`, `--anchor` and `--strength`
to ablate any concept.

## Settings

| | FLUX-schnell | FLUX-dev | SD3.5-medium |
|---|---|---|---|
| Steps | 4 | 28 | 28 |
| Guidance | none | 3.5 (embedded) | 4.5 (CFG) |
| Max sequence length | 256 | 512 | 256 |

Subspaces use 50 contrastive pairs per concept; pooled and CLIP-seq carriers use rank 4;
T5 carriers use the smallest rank reaching 95% of singular-value energy (capped at 24) and
keep the top 10 concept-aligned token rows per pair. Seeds are 188 and 288. The evaluator
is `llava-hf/llava-v1.6-vicuna-7b-hf` with greedy decoding; its prompts are in
`carrier_erasure/llava_eval.py`.

## Notes

- The scripts reproduce the notebook's logic; the class and prompt code are copied from it
  verbatim. The one structural change: outputs are kept per model
  (`outputs/flux-schnell/`, `outputs/flux-dev/`, …) rather than in one Colab Drive folder.
- `notebooks/projection_erasure_v6.ipynb` also contains exploratory cells not used in the
  paper (protected-subspace learning, scene-protected configurations, anchor ablation,
  a manual audit sheet).

## Citation

Amna Jamal and Rajendran Ramesh Babu Manoj Narender contributed equally.

```bibtex
@inproceedings{jamal2026onecarrier,
  title     = {One Carrier Isn't Enough: Redundancy-Aware Concept Suppression in Diffusion Transformers},
  author    = {Jamal, Amna and Narender, Rajendran Ramesh Babu Manoj and Ali, Samina and
               Nagaraju, Tarun Chandra and Zhu, Kevin and Mari, Antonio},
  booktitle = {NeurIPS 2026 Workshop on Trustworthy AI for Good (AI4GOOD)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=QJ8g4adkOT}
}
```
