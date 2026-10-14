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

## Lecture 21 run (2026-10-15)

### Transcripts
- [x] 21 Language Models — video 9GWd3SAWLbA (OCW human captions; verbatim in original/; light copy-edit inline, checked by script: 100 markers and 276 numbers identical, ratios 0.993–1.008; 4 restorations, 13 [Ed:] notes; the recording covers the whole deck)

### Slides
- [x] 21 — mit6_7960_f24_lec21.pdf (92 pages; slide N = page N, since the deck prints its own number only on pages 1–7 and slide_number_map.py reads the source decks' corner numbers as a numbering with gaps; page 92 the OCW end page, no number; read by Sonnet in two halves; headings 1–92 checked by script)
- [x] 21 — OCW notices from the slide file (26–29, 31–35, 52–56, 58–60, 63, 86), cross-checked with the text layer (same nineteen) and a vector scan (no outlined small print)
- [x] 21 — Opus figure audit on 70 pages applied (19 corrected, 21 precisions only; split between four agents, all finished); renders of slides 72 and 81 checked

### Wiki
- [x] wiki/21-language-models.md (whole recording, with timestamps)
- [x] Topic pages for lecture 21: new language-models page; foundation-models-and-prompting, autoregressive-models, self-supervised-learning, transfer-learning, scaling-laws, inductive-bias, meta-learning, knowledge-distillation, generative-models-as-data, transformers and recurrent-neural-networks extended; course-map extended; lecture 6's and 16's pointers linked

### Images
- [x] raw/images/21-language-models/ — 18 images, rendered by page; 19 OCW-excluded slides never rendered; hash sweep both directions clean (sweep fixed to hash indexed-colour images, 347 of which its first version skipped), validated on lectures 11 and 14; no earlier image withdrawn

### Publish
- [x] Lecture 21: INDEX, AGENTS (coverage, image-table row, three lessons: decks numbered on a few pages only, white boxes in reused decks, indexed-colour images in the sweep), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 21: check_math (offline and --github, 24 changed files) and verify_kb.py clean, review read; commit and push
- [ ] Move this run's finished lines to BUILD_LOG.md

## Transcripts
- [ ] 23 Metrized Deep Learning — video zBvsoxC6tAo
- [ ] 24 Inference Methods for Deep Learning — video mbgFTqKxR7A
- [ ] PyTorch Tutorial — video o5gPABcGZwc (no deck)
