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

## Transcripts
- [ ] 21 Language Models — video 9GWd3SAWLbA
- [ ] 23 Metrized Deep Learning — video zBvsoxC6tAo
- [ ] 24 Inference Methods for Deep Learning — video mbgFTqKxR7A
- [ ] PyTorch Tutorial — video o5gPABcGZwc (no deck)

## Lecture 20 run

### Transcripts
- [x] 20 Scaling Laws — video 7hbf4klU3ks (OCW human captions; verbatim in original/; light copy-edit inline, checked by script: 50 markers and 45 numbers identical, ratios 1.0–1.008; the recording ends at 38:01, after slide 17)

### Slides
- [x] 20 — mit6_7960_f24_lec20.pdf (52 pages, 1:1, page 52 the OCW end page prints 52; read by Sonnet in two halves; --verify OK)
- [x] 20 — OCW notices from the slide file (3, 24, 40, 45, 47, 48), cross-checked with the text layer (same six) and a vector scan (one hit, slide 42's red brush stroke, not small print)
- [x] 20 — Opus figure audit on 43 pages applied (22 corrected, 10 precisions only; split between four agents, all finished); renders of slides 13 and 42 checked

### Wiki
- [x] wiki/20-scaling-laws.md (recorded part, slides 1–17, with timestamps; slides 18–51 marked as from the deck alone)
- [x] Topic pages for lecture 20: scaling-laws, gradient-descent, representation-learning, representational-power, lipschitz-continuity, kernel-methods, foundation-models-and-prompting, course-map extended; no new page

### Images
- [x] raw/images/20-scaling-laws/ — 20 images; 6 OCW-excluded slides never rendered; hash sweep both directions (no reuse; slide 32's and 47's coarse matches cleared on sight), validated on lectures 11 and 14; no earlier image withdrawn

### Publish
- [x] Lecture 20: INDEX, AGENTS (coverage, image-table row, three lessons: recordings that stop short, drop-shadow coarse matches, get_image_info coverage), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 20: check_math and verify_kb.py clean, review read; commit and push
- [ ] Move this run's finished lines to BUILD_LOG.md
