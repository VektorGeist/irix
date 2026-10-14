# Irix benchmarks — honest, reproducible, CPU-only

Every number here is produced by code in the repo, on CPU, with **no model and
no GPU**. Reproduce them yourself:

```
irix bench                 # synthetic GUI-grounding, 12 screens
irix bench --no-blur       # crisp render (upper bound)
irix bench --jsonl gt.jsonl   # score real ScreenSpot-style data
```

## What is measured

The metric is **ScreenSpot's**: given a target widget, does the predicted click
point land inside its bounding box (*click-in-bbox*)? That is the number GUI-agent
grounding is judged on. Alongside it: box recall, exact-label rate, and parse
latency per screen.

The synthetic screens (`irix.bench.make_screen`) render real UIs with exact
ground truth — light/dark themes, 12–20px fonts, distractor art, and an optional
anti-alias blur that approximates real on-screen rendering. **Honest scope:**
crisp synthetic text is *easier* than a real ClearType screen, so treat the crisp
row as an upper bound and the anti-aliased row as the realistic figure. Numbers on
truly in-the-wild screenshots belong in the `--jsonl` path and are not claimed
here yet.

## Results (12 screens, 96 targets, single CPU core)

Measured 2026-09-17 in this environment (`irix bench`, seed 0):

| Condition        | Grounding acc | Box recall | Exact label | Parse / screen |
|------------------|:-------------:|:----------:|:-----------:|:--------------:|
| Crisp            | **0.938**     | 1.00       | 0.625       | ~0.31 s        |
| Anti-aliased     | **0.865**     | 0.99       | 0.49        | ~0.47 s        |

**Drift note (honest):** this file previously documented 1.00 / 0.85 grounding
with 0.91 / 0.67 exact-label. Those numbers do not reproduce here: the
pre-extension `elements` (preserved byte-for-byte as
`src/irix/elements_legacy.py`) measures 0.938 / 0.865 on the same 12 screens
in this environment. The drift is pre-existing — font/glyph-gallery and
environment differences in text rendering, not a regression from any code
change. The table above is what the code actually produces; treat the old
1.00/0.85 as stale, not as a target.

## Cross-interface widget detection (new)

`elements` was extended beyond crisp-desktop rectangles (see module docstring:
low-threshold Canny for soft borders, MSER for rounded/low-contrast/icon
targets, link-styled text detection, and text-context suppression that drops
word-fragment and text-block false positives via OCR-label + geometry fusion).
Benchmark: `~/workspace/irix-elements-x/bench/bench_widgets.py` — 114
hand-verified targets (2 real phone screenshots + 6 synthetic desktop
screens). Recall = target hit at IoU ≥ 0.5 or detection centre inside the
target; a detection is a false positive only if it matches *no* target
(redundant boxes on one target are not FPs — either would tap correctly).

| Detector                          | Recall        | Detections | FP | Precision |
|-----------------------------------|:-------------:|:----------:|:--:|:---------:|
| Before (`elements_legacy`)        | 0.412 (47/114)| 66         | 16 | 0.76      |
| After, stages A–D (no suppression)| 1.000 (114/114) | 228     | 72 | 0.68      |
| After, stages A–E (full)          | **1.000 (114/114)** | **189** | **38** | **0.80** |

Every category at 1.00 recall after the extension, including the previously
missed Cancel button, all 10 phone nav icons, both info icons, the sliders
button, low-contrast buttons, icon-only buttons, and links. The suppression
stage (E) cut false positives nearly in half (72 → 38) with zero recall loss;
remaining FPs are mostly tiny glyph fragments and a few non-clickable text
labels (e.g. section headings). Cost of the recall gain: parse latency roughly
2–3× on these screens (see table above) — more stages plus per-candidate OCR
in suppression.

**Methodology fix (honest):** the first version of this benchmark only credited
the *first* detection matching each target, which inflated FP counts whenever
several boxes covered one target (reported precision then was 0.71 → 0.50).
The counter was fixed to match its own documented definition (a detection is
an FP iff it matches no target); all numbers above use the fixed counter.
(Correction 2026-09-18: the A–D precision row previously read 0.61, which was
inconsistent with the table's own definition — (228−72)/228 = 0.68. Fixed.)

**Determinism fix (2026-09-18):** while validating, `detect_boxes` was found
to vary by ±1 box depending on what had been processed earlier in the process.
Root cause was a real bug in `irix.text.GlyphGallery.match_multi`: the match
cache was keyed on the squared glyph shape alone, but the match cost includes
an aspect-ratio term — so two windows with the same squared shape but
different aspects shared one cache entry and the second got the first's
ranking (order-dependent OCR labels). The key is now
`(shape_hash, aspect)`. Verified: identical output regardless of call order,
full test suite still 116 passed / 1 skipped, official benchmark numbers
below unchanged.

Component latency (measured, `irix.text` / `irix.elements`):

| Operation                         | Before | Now      |
|-----------------------------------|:------:|:--------:|
| OCR one UI line                   | 25.3 s | ~24 ms   |
| Full 4K desktop element parse     | >120 s | ~2.5 s   |
| 1080p phone-screenshot parse (945×2048) | ~0.03 s (19 boxes — phone UI it can't read) | ~3.1 s (49 boxes) |

## How this compares to the field

The reference GPU screen parser, **Microsoft OmniParser V2**, reports ~0.6 s/frame
on an A100 and ~0.8 s/frame on an RTX 4090 (its own model card). It needs a GPU
and a vision-language model.

Irix is not trying to win open-ended "what is this picture" — a VLM wins that.
Irix owns a different quadrant, and it is a quadrant nobody else ships:

- **CPU-only, zero VRAM** — runs on any machine, including the ones OmniParser can't.
- **Deterministic** — same pixels always give the same answer; no sampling, no drift.
- **Private** — the screen never leaves the machine; no API, no upload.
- **No hallucination** — it reports detected geometry + read text, never invents a
  control that isn't there.
- **Injection-safe by construction** — `scan_injection` reads hidden text and flags
  prompt-injection *before* any model sees the frame; a VLM agent ingests the
  screen first and is attackable through it.
- **Improves with use, not with retraining** — recognition is a growing exemplar
  gallery (faces, objects, glyph shapes), not frozen weights.
- **Exact where the OS knows** — on Windows the accessibility-tree fusion
  (`irix fuse`) returns OS-exact rectangles + names for free; pixel CV covers
  everything the tree can't (games, canvas, streamed screens).

## Reproducibility notes

- Deterministic per seed; `irix bench --seed N` varies the sample.
- Runs on the bundled glyph gallery (`src/irix/glyphs.irix`), seeded from the
  host's system fonts at crisp and small (13/17px, anti-aliased) sizes.
- No network, no GPU. Since 0.5.0 the measured path includes the tiny-brain
  reader weights (136 KB, pure-numpy CPU inference, deterministic) behind the
  gallery-first arbiter; the eye's geometry/detection stages remain model-free.

## 0.5.0: tiny-brain arbiter wired in (2026-09-19)

Eidetrix product shape: the gallery/eye reads first; the Kaggle-trained
tiny-brain reader (tb_read_v1, 33k params, 136 KB, pure-numpy CPU) arbitrates.
Measured in `~/workspace/eidetrix-study/tinybrain/eval_hybrid.py`, same eye
segmentation and same grade4 metric throughout.

Study battery text recall (114 GT texts, 12 screenshots):

| matcher | recall@0.8 | recall@0.5 |
|---|---|---|
| gallery alone (eye as shipped in 0.4.0) | 102/114 = 0.895 | 106/114 = 0.930 |
| reader alone | 34/114 = 0.298 | 96/114 = 0.842 |
| hybrid: plain uncertainty fallback (T=0.20) | 102/114 = 0.895 | 106/114 = 0.930 |
| hybrid: + arbiter (reader overrules iff conf>=0.99 and disagrees) | **103/114 = 0.904** | **107/114 = 0.939** |

Honest notes:

- A naive "fall back when the gallery is unsure" does NOT improve the battery:
  it ties at best (T=0.20) and collapses at aggressive thresholds (T=0.10 ->
  0.605), because the battery's failures are gallery-confident-but-wrong, not
  gallery-uncertain.
- The arbiter fixed one real case ('Wi-Fj' -> 'Wi-Fi': gallery sure it was
  'j', reader sure it was 'i') with zero regressions. Stable across C=0.99
  and C=0.999.
- The reader's big win is elsewhere: degraded out-of-gallery glyphs
  (synthetic held-out), reader 0.668 vs gallery 0.441. The battery doesn't
  exercise that regime; the wiring keeps the fallback for it anyway.
- Measurement hygiene: an early calibration passed 0/255 windows into
  `_square` (uint8 overflow -> blank squares) and understated the gallery.
  Caught by checking the pixels; re-ran at the pipeline's 0/1 scale.

Determinism preserved: the reader is pure numpy, no RNG at inference, same
pixels always give the same answer. `irix bench` grounding numbers unchanged
(the arbiter touches text labels, not boxes).
