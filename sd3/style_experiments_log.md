# SD3 Steering Vector — Style Unlearning Experiments Log

---

## Background — Lessons from FLUX Object vs Style Steering

### Where each concept lives in FLUX (diagnostic zeroing results)
- **Object identity** flows through **CLIP pooled → `time_text_embed` → modulation**. Zeroing `time_text_embed` produces pure noise. Zeroing `context_embedder` (T5) still produces the object — T5 is irrelevant for object identity in FLUX.
- **Style** is diffuse across **all real T5 tokens** and painted throughout the full denoising trajectory. CLIP carries identity, not style.

### Steering strategy differences

| | Object | Style |
|---|---|---|
| CLIP alpha | Small (prevents over-removal) | 0 (CLIP = identity, don't touch) |
| T5 alpha | Moderate, top-k concept tokens only | Moderate–high, all real tokens, no gating |
| Step range | First 1–2 steps only | All steps |
| `clip_negative` | Yes | Yes |

### SD3-specific consideration
SD3 has a third parameter FLUX lacks: **CLIP-G pooled** (`pooled_projections[768:]`, 1280-dim). Its role in style vs identity is unknown — this is what the ablation below is designed to answer.

---

## Experiment #1 — Baseline: "a landscape in watercolor style"

**Date:** 2026-04-22

### Config
| Parameter | Value |
|---|---|
| `CONCEPT_PROMPT` | `"watercolor style"` |
| `NEUTRAL_PROMPT` | `"a realistic landscape"` |
| `GEN_PROMPT` | `"a landscape in watercolor style"` |
| `NUM_STEPS` | `28` |
| `GUIDANCE_SCALE` | `7.0` |
| `SEED` | `42` |

### Approach
Per-step directions + `clip_negative` on both CLIP-L and T5 context tokens. Additional CLIP-G pooled direction captured but held at `α=0`. Swept CLIP-G alpha `[0, 10, 11, 12, 13, 14]` with CLIP-L and T5 held at 0.

### Observations
- TBD — sweep results pending.

---

## Experiment #2 — Full 3-parameter ablation (CLIP-L × T5 × CLIP-G)

**Date:** 2026-04-22

### Goal
Determine the role of each encoder in style removal for SD3. Based on FLUX findings, CLIP-L should be held near 0, T5 is the primary driver, and CLIP-G is unexplored.

### Parameter ranges
| Parameter | Values | Rationale |
|---|---|---|
| `ALPHAS_C` (CLIP-L) | `[0, 2, 4]` | Keep low — CLIP carries identity, high values degrade image quality |
| `ALPHAS_T` (T5) | `[0, 2, 4, 6, 8]` | Main style driver; FLUX used 2.0, going wider to find SD3's range |
| `ALPHAS_G` (CLIP-G) | `[0, 5, 10, 15]` | Unknown — broader sweep to characterise its role |

**Total combinations: 3 × 5 × 4 = 60**

### Config
| Parameter | Value |
|---|---|
| `CONCEPT_PROMPT` | `"watercolor style"` |
| `NEUTRAL_PROMPT` | `"a realistic landscape"` |
| `GEN_PROMPT` | `"a landscape in watercolor style"` |
| `NUM_STEPS` | `28` |
| `GUIDANCE_SCALE` | `7.0` |
| `SEED` | `42` |

### Observations
- All sweeps show steering toward **another art style**, not toward realism — the direction is wrong.
- Last 2 rows (`a_clip=4, a_t5=6` and `a_clip=4, a_t5=8`) show the strongest style shift, pointing to CLIP-L as the driver of the unintended style transfer.
- Last image on every row (`a_clipg=15`) destroys the image — CLIP-G=15 is too aggressive.
- Paradoxically, the watercolor baseline image looks **most realistic** in the last row — steering is actively moving away from realism, not toward it.

### Diagnosis
The neutral prompt `"a realistic landscape"` is too stylistically specific. The steering direction `"watercolor style" − "a realistic landscape"` points toward the mathematical opposite of a realistic landscape in embedding space, which is some other style — not "no style." Subtracting this direction steers the image away from watercolor but lands in unintended style territory.

CLIP-L at `a_clip > 0` amplifies this effect since CLIP encodes global style/identity. FLUX found CLIP should be 0 for style unlearning — confirmed here.

### Key takeaway
- **CLIP-L must be 0** for style unlearning (confirms FLUX finding).
- **CLIP-G=15 destroys images** — ceiling is somewhere below 15.
- **Neutral prompt is the root problem** — need a style-free neutral (e.g. `"a landscape"` with no style descriptor).

---

## Experiment #3 — Style-free neutral prompt: "a landscape"

**Date:** 2026-04-22

### Hypothesis
Replacing `"a realistic landscape"` with `"a landscape"` removes the style bias from the direction, so subtracting the watercolor component steers toward "no style" rather than "opposite of realism."

### Config changes from #2
| Parameter | Value |
|---|---|
| `NEUTRAL_PROMPT` | `"a landscape"` (was `"a realistic landscape"`) |
| `ALPHAS_C` | `[0]` (CLIP-L fixed at 0 — confirmed harmful above) |
| `ALPHAS_T` | `[0, 2, 4, 6, 8]` |
| `ALPHAS_G` | `[0, 5, 10]` (dropped 15 — destroys image) |

**Total combinations: 1 × 5 × 3 = 15**

### Observations
- All images identical across all alphas — steering had zero effect.
- Root cause: `clip_negative` clipped everything to zero. `"watercolor style"` vs `"a landscape"` are content-mismatched, so the direction captures content differences. The generation activations project negatively onto it, so `clamp(min=0)` kills every subtraction.

### Key takeaway
Prompts must be **content-matched**. The generation activations need to sit on the positive side of the direction for `clip_negative` to have anything to subtract.

---

## Experiment #4 — Content-matched prompt pair

**Date:** 2026-04-22

### Hypothesis
Using the full stylised prompt as concept and the bare content prompt as neutral isolates the style delta. Generation activations (from the same prompt as concept) should project positively onto the direction, making `clip_negative` effective.

### Config
| Parameter | Value |
|---|---|
| `CONCEPT_PROMPT` | `"a landscape in watercolor style"` |
| `NEUTRAL_PROMPT` | `"a landscape"` |
| `GEN_PROMPT` | `"a landscape in watercolor style"` |
| `ALPHAS_C` | `[0]` |
| `ALPHAS_T` | `[0, 2, 4, 6, 8]` |
| `ALPHAS_G` | `[0, 5, 10]` |

### Observations
- Steering visible — content-matched prompts confirmed working.
- Last column (`CLIP-G=10`) destroys images on all rows — ceiling is between 5 and 10.
- `CLIP-G=5` functional; finer resolution needed in this range.

---

## Experiment #5 — Finer CLIP-G sweep below destruction threshold

**Date:** 2026-04-22

### Config changes from #4
| Parameter | Value |
|---|---|
| `ALPHAS_G` | `[0, 2, 4, 6, 8]` (was `[0, 5, 10]`) |

**Total combinations: 5 × 5 = 25**

### Observations
- Last column (`CLIP-G=8`) working but steering to a different art style, not realistic.
- CLIP-G hook was missing `clip_negative` — doing unconstrained projection removal in both directions.
- Fixed: added `clip_negative` to CLIP-G hook.

---

## Experiment #6 — T5 only (CLIP-L=0, CLIP-G=0)

**Date:** 2026-04-22

### Goal
Isolate T5's contribution to style removal. Determine whether T5 alone can steer toward realism before adding CLIP-G back in.

### Config
| Parameter | Value |
|---|---|
| `ALPHAS_C` | `[0]` |
| `ALPHAS_T` | `[0, 2, 4, 6, 8, 10, 12]` |
| `ALPHAS_G` | `[0]` |

### Observations
- TBD

---
