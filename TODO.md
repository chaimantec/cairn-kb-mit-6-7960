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

## Build checklist, PyTorch tutorial

### Transcripts
- [x] PyTorch Tutorial — video o5gPABcGZwc (OCW human captions; verbatim in original/pytorch-tutorial.md; light copy-edit inline, checked by script: 38 markers and 75 numbers identical, ratios 0.985–1.010; 7 changes in 8 places, 4 [Ed:] notes; OCW's transcript PDF ends at the same words)
- [x] OCW resource page checked for other materials: video and transcript only; no deck, and the notebook was on Piazza

### Wiki
- [x] wiki/pytorch-tutorial.md (whole recording, timestamps only; transcript-only notice at the top)
- [x] Topic pages: tensors-and-batching, differentiable-programming, gradient-descent and multilayer-perceptron extended; course-map extended (table row, sources line, the recorded tutorial)

### Publish
- [x] INDEX (coverage, entry, four concept entries, raw-materials lines), AGENTS (coverage, numbering, images line, no-deck lesson), BUILD_LOG (pointers, transcript edit), kb.json (24 of 24, caveat, images) updated
- [x] check_math (offline and --github) and verify_kb.py clean, review read; commit and push
- [ ] Move this run's finished lines to BUILD_LOG.md
