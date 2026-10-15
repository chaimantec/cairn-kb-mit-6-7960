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

## Open lectures

- [ ] PyTorch Tutorial — video o5gPABcGZwc (no deck)

## Lecture 24 run (catalog position 23), started 2026-10-16

### Transcripts
- [x] 24 Inference Methods for Deep Learning — video mbgFTqKxR7A (OCW human captions; verbatim in original/; light copy-edit inline, checked by script: 107 markers and 69 numbers identical, ratios 0.983–1.008; 20 restorations in 31 places, 10 [Ed:] notes; the recording covers the whole deck, and OCW's transcript PDF ends at the same words)

### Slides
- [x] 24 — mit6_7960_f24_lec24.pdf (56 pages, Keynote; slide N = page N, 1–55 bottom centre; page 56 the OCW end page, stored rotated, prints 56 but the script reports none) — Sonnet transcription in two halves; headings 1–56 checked by script
- [x] 24 — OCW notices from the slide file (7, 8, 27–31, 39, 40, 46, 47, 52–54), cross-checked with the text layer (same fourteen) and a vector scan (no outlined small print)
- [x] 24 — Opus figure audit on 48 pages applied (15 corrected, 22 precisions only, 11 right; split between four agents, all finished); slide 43's plus-sign slip flagged by the parent; renders of slides 9 and 18 checked

### Wiki
- [x] wiki/24-inference-methods-for-deep-learning.md (whole recording, with timestamps)
- [x] Topic pages for lecture 24: new test-time-compute page; autoregressive-models, foundation-models-and-prompting, meta-learning, language-models, scaling-laws, self-supervised-learning, generative-models-as-data, diffusion-models and variational-autoencoders extended; course-map extended (table row, pointers, the office-hours announcement)

### Images
- [x] raw/images/24-inference-methods-for-deep-learning/ — 19 images (3, 5, 9, 12, 14, 15, 17–20, 22, 26, 33–38, 51); 14 OCW-excluded slides never rendered; sweep both directions, validated on lectures 11 and 14, withheld slide 4 (lecture 20's Sutton essay) and cleared slides 1 and 26; a citation check withheld slides 42 (lecture 21's Zhao et al. example, as a screenshot) and 55 (slide 54's STaR example, as crops); 15 wiki images hand-placed, slide-file images by script

### Publish
- [x] Lecture 24: INDEX, AGENTS (coverage, image-table and withheld-table rows, lessons: reuse the hash sweep cannot see, OCR text over screenshots, LaTeXiT glyphs as text), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 24: check_math (offline and --github) and verify_kb.py clean, review read; commit and push
- [ ] Move this run's finished lines to BUILD_LOG.md
