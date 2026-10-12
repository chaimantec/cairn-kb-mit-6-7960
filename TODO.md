# KB build — MIT 6.7960 (Deep Learning, Fall 2024)

Catalog id `04a24924-529a-4661-bce4-8237464254fa`. Course site:
<https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/>. Slide decks are linked from
`lists/lecture-notes/` as `mit6_7960_f24_lecN.pdf`; there is **no lecture 22** in either the
playlist or the deck list. Catalog positions 22 and 23 are therefore *Lec 23* and *Lec 24*, and
position 24 is the PyTorch tutorial. Files use the lecture's own number (`23-…`, `24-…`), not
the catalog position.

Images are opted in: every figure-bearing slide, rendered per Step 1c.

**This file holds open work only.** Each run adds its own checklist here and, once committed and pushed, moves
the finished lines to [`BUILD_LOG.md`](BUILD_LOG.md#build-checklist-lectures-117), which holds every finished checklist so far.

## Transcripts
- [x] 18 Transfer Learning: Models — video tNfuZ9Imt3M (OCW human captions; light copy-edit inline, checked by script; verbatim in original/)
- [ ] 19 Transfer Learning: Data — video RUdQMHV-7KM
- [ ] 20 Scaling Laws — video 7hbf4klU3ks
- [ ] 21 Language Models — video 9GWd3SAWLbA
- [ ] 23 Metrized Deep Learning — video zBvsoxC6tAo
- [ ] 24 Inference Methods for Deep Learning — video mbgFTqKxR7A
- [ ] PyTorch Tutorial — video o5gPABcGZwc (no deck)

## Lecture 18 run (Transfer Learning: Models)

### Slides
- [x] 18 — mit6_7960_f24_lec18.pdf (77 pages, 1:1, page 77 the OCW end page prints 77; read by Sonnet in two halves)
- [x] 18 — OCW notices from the slide file, cross-checked with the text layer (21 pages) and a vector scan (no outlined small print)
- [x] 18 — Opus figure audit on 62 pages applied (19 corrected; split between four agents, all finished)

### Wiki
- [x] wiki/18-transfer-learning-models.md
- [x] Topic pages for lecture 18: knowledge-distillation, domain-adaptation, foundation-models-and-prompting new; transfer-learning, contrastive-learning, softmax-and-cross-entropy, out-of-distribution-generalization, adversarial-examples, scaling-laws, self-supervised-learning, representation-learning, autoregressive-models, course-map extended

### Images
- [x] raw/images/18-transfer-learning-models/ — 26 images; 21 OCW-excluded slides and slide 74 (reuses slide 63's excluded encoder graphics) never rendered; hash sweep both directions, validated on lectures 11 and 14
- [x] Withdrawn by lecture 18's sweep: lecture 2's slide 70 (same encoder graphics) and lecture 12's slide 35 (lecture 18's excluded classroom strip); images deleted, embeds removed, counts updated

### Publish
- [x] Lecture 18: INDEX, AGENTS (coverage, image-table row, withheld rows, sweep lesson), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Repaired three pre-existing formulas whose `\to` and `\times` an earlier run's script had turned into tabs (representation-learning, softmax-and-cross-entropy)
- [ ] Lecture 18: check_math and verify_kb.py clean, review read; commit and push
- [ ] Move this run's finished lines to BUILD_LOG.md
