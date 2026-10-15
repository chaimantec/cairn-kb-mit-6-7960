# KB build — MIT 6.7960 (Deep Learning, Fall 2024)

Catalog id `04a24924-529a-4661-bce4-8237464254fa`. Course site:
<https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/>. Slide decks are linked from
`lists/lecture-notes/` as `mit6_7960_f24_lecN.pdf`; there is **no lecture 22** in either the
playlist or the deck list. Catalog positions 22 and 23 are therefore *Lec 23* and *Lec 24*, and
position 24 is the PyTorch tutorial. Files use the lecture's own number (`23-…`, `24-…`), not
the catalog position.

Images are opted in: every figure-bearing slide, rendered per Step 1c.

**This file holds open work only.** Each run adds its own checklist here and, once committed and pushed, moves
the finished lines to [`BUILD_LOG.md`](BUILD_LOG.md#build-checklist-lectures-117), whose last sections hold every finished checklist so far.

## Lecture 23 run (catalog position 22; started 2026-10-15)

### Transcripts
- [x] 23 Metrized Deep Learning — video zBvsoxC6tAo (OCW human captions; verbatim in original/; light copy-edit inline, checked by script: 87 markers and 18 numbers identical, ratios 0.991–1.010; 7 restorations, 8 [Ed:] notes; the recording covers the whole deck, and OCW's transcript PDF ends at the same words)

### Slides
- [x] 23 — mit6_7960_f24_lec23.pdf (49 pages; typed Google Slides, not handwritten; slide N = page N, bottom-centre numbers 1–49 match pages, page 49 the OCW end page, stored rotated, which slide_number_map.py reports as printing none; thirty pages also print a larger bottom-right number from an earlier talk deck) — read by Sonnet in two halves; headings 1–49 checked by script
- [x] 23 — OCW notices from the slide file (3–7, 24, 26–28, 31, 37, 42, 45–47), cross-checked with the text layer (same fifteen) and a vector scan (no outlined small print)
- [x] 23 — Opus figure audit on 44 pages applied (26 corrected, 15 precisions only; split between three agents, all finished; 123 bolded symbols on slides 9–24 made plain as printed); renders of slides 14 and 34 checked

### Wiki
- [x] wiki/23-metrized-deep-learning.md (whole recording, with timestamps)
- [x] Topic pages for lecture 23: new modular-norm page; steepest-descent, norms, lipschitz-continuity, second-order-methods, scaling-rules, gradient-descent, adversarial-examples, normalization-layers, loss-landscapes and skip-connections extended; course-map extended; lecture 7's pointer updated

### Images
- [x] raw/images/23-metrized-deep-learning/ — 8 images (2, 13, 14, 16, 17, 19, 34, 41); 15 OCW-excluded slides never rendered; sweep both directions with the withheld list brought up to date, validated on lectures 11 and 14; full re-sweep over lectures 1–21 withdrew lecture 13's slide 3 (lecture 23's slide 7 excludes its chart) and cleared lecture 11's slide 60; wiki images hand-placed, slide-file images by script

### Publish
- [x] Lecture 23: INDEX, AGENTS (coverage, image-table and withheld-table rows, handwritten-deck prediction corrected, lessons: two numberings on one page, rotated end page, symbols as printed, a withheld slide's other images, typed decks' fonts), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 23: check_math (offline and --github) and verify_kb.py clean, review read; commit and push
- [ ] Move this run's finished lines to BUILD_LOG.md

## Remaining lectures
- [ ] 24 Inference Methods for Deep Learning — video mbgFTqKxR7A
- [ ] PyTorch Tutorial — video o5gPABcGZwc (no deck)
