# KB build — MIT 6.7960 (Deep Learning, Fall 2024)

Catalog id `04a24924-529a-4661-bce4-8237464254fa`. Course site:
<https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/>. Slide decks are linked from
`lists/lecture-notes/` as `mit6_7960_f24_lecN.pdf`; there is **no lecture 22** in either the
playlist or the deck list. Catalog positions 22 and 23 are therefore *Lec 23* and *Lec 24*, and
position 24 is the PyTorch tutorial. Files use the lecture's own number (`23-…`, `24-…`), not
the catalog position.

Images are opted in: every figure-bearing slide, rendered per Step 1c.

## Transcripts
- [x] 01 Introduction to Deep Learning — video 6FkRvTtUc-o (OCW human captions; light copy-edit, verbatim in original/)
- [x] 02 How to Train a Neural Net — video vidCX_dMCu0 (OCW human captions; light copy-edit, verbatim in original/)
- [x] 03 Approximation Theory — video ySaoWrv3T_Q (OCW human captions; light copy-edit, verbatim in original/)
- [x] 04 Architectures: Grids — video bxVkZ4M-hIE (OCW human captions; light copy-edit, verbatim in original/)
- [x] 05 Architectures: Graphs — video 0niIwb37nF0 (OCW human captions; light copy-edit, verbatim in original/)
- [x] 06 Generalization Theory — video EiO8BBa-xdc (OCW human captions; light copy-edit, verbatim in original/)
- [x] 07 Scaling Rules for Optimization — video VcGPE4s_oNw (OCW human captions; light copy-edit, verbatim in original/)
- [x] 08 Architectures: Transformers — video Q1HOKrNeh2M (OCW human captions; light copy-edit, verbatim in original/)
- [ ] 09 Hacker's Guide to Deep Learning — video DC2Hw9DiLCg
- [ ] 10 Architectures: Memory — video IiHknRHA-Gk
- [ ] 11 Representation Learning: Reconstruction-Based — video QxOzQRtd440
- [ ] 12 Representation Learning: Similarity-Based — video yUh1fEGGdl4
- [ ] 13 Representation Learning: Theory — video -eC0-5mXHQg
- [ ] 14 Generative Models: Basics — video hJlrAHqGOS8
- [ ] 15 Generative Models: Representation Learning Meets Generative Modeling — video 8zzfcYIELdo
- [ ] 16 Generative Models: Conditional Models — video zaMcHuJwe1w
- [ ] 17 Generalization: Out-of-Distribution (OOD) — video tjD9LIzIIek
- [ ] 18 Transfer Learning: Models — video tNfuZ9Imt3M
- [ ] 19 Transfer Learning: Data — video RUdQMHV-7KM
- [ ] 20 Scaling Laws — video 7hbf4klU3ks
- [ ] 21 Language Models — video 9GWd3SAWLbA
- [ ] 23 Metrized Deep Learning — video zBvsoxC6tAo
- [ ] 24 Inference Methods for Deep Learning — video mbgFTqKxR7A
- [ ] PyTorch Tutorial — video o5gPABcGZwc (no deck)

## Crawl
- [x] Fetch course site index: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/
- [x] Download linked PDFs and slides (23 decks, 5 psets, hw5 solution, notation handout; 206MB, gitignored)
- [x] Write sources.md

## Slides (raw/slides/)
- [x] 01 Introduction to Deep Learning — mit6_7960_f24_lec1.pdf (81 pages; figure audit 6 pages)
- [x] 02 How to Train a Neural Net — mit6_7960_f24_lec2.pdf (81 pages; figure audit 8 pages)
- [x] 03 Approximation Theory — mit6_7960_f24_lec3.pdf (43 pages, handwritten; read by Sonnet, audited by Opus on 19 pages)
- [x] 04 Architectures: Grids — mit6_7960_f24_lec4.pdf (84 pages; read by Sonnet, audited by Opus on 24 pages)
- [x] 05 Architectures: Graphs — mit6_7960_f24_lec5.pdf (47 pages; read by Sonnet, audited by Opus on 27 pages)
- [x] 06 Generalization Theory — mit6_7960_f24_lec6.pdf (66 pages; read by Sonnet, audited by Opus on 30 pages)
- [x] 07 Scaling Rules for Optimization — mit6_7960_f24_lec7.pdf (32 pages, handwritten; read by Sonnet, audited by Opus on 24 pages)
- [x] 08 Architectures: Transformers — mit6_7960_f24_lec8.pdf (55 pages; read by Sonnet)
- [x] 08 — Opus figure audit on 33 pages applied (19 corrected)

## Wiki
- [x] wiki/01-introduction.md
- [x] wiki/02-how-to-train-a-neural-net.md
- [x] wiki/03-approximation-theory.md
- [x] wiki/04-architectures-grids.md
- [x] wiki/05-architectures-graphs.md
- [x] wiki/06-generalization-theory.md
- [x] wiki/07-scaling-rules-for-optimization.md
- [x] wiki/08-architectures-transformers.md
- [x] Topic pages (cross-lecture concepts) for lecture 1
- [x] Topic pages for lecture 2: backpropagation, loss-landscapes, differentiable-programming new; seven pages extended
- [x] Topic pages for lecture 3: lipschitz-continuity, scaling-laws new; representational-power rewritten; MLP, activations, generalization, course-map extended
- [x] Topic pages for lecture 4: convolution, inductive-bias, skip-connections, neural-fields-and-positional-encoding new; MLP, activations, generalization, representation-learning, tensors-and-batching, differentiable-programming, representational-power, course-map extended
- [x] Topic pages for lecture 5: graph-neural-networks new; inductive-bias, convolution, representational-power, neural-fields-and-positional-encoding, multilayer-perceptron, backpropagation, course-map extended
- [x] Topic pages for lecture 6: generalization-and-double-descent and inductive-bias extended in depth; gradient-descent, loss-landscapes, representation-learning, multilayer-perceptron, convolution, graph-neural-networks, neural-fields-and-positional-encoding, representational-power, course-map extended
- [x] Topic pages for lecture 7: steepest-descent, second-order-methods, norms, scaling-rules new; gradient-descent, loss-landscapes, skip-connections, lipschitz-continuity, backpropagation, multilayer-perceptron, differentiable-programming, scaling-laws, course-map extended
- [x] Topic pages for lecture 8: transformers new; graph-neural-networks, convolution, neural-fields-and-positional-encoding, inductive-bias, skip-connections, multilayer-perceptron, softmax-and-cross-entropy, tensors-and-batching, representation-learning, course-map extended
- [x] INDEX.md table of contents

## Images
- [x] raw/images/01-introduction/ — 20 images; OCW-excluded slides never rendered; slide 51 withdrawn at lecture 6 (same image as lecture 6's excluded slide 30)
- [x] raw/images/02-how-to-train-a-neural-net/ — 45 images; OCW-excluded slides never rendered
- [x] raw/images/03-approximation-theory/ — 18 images; the deck has no OCW exclusion notices
- [x] raw/images/04-architectures-grids/ — 31 images; 35 OCW-excluded slides and slides 50–52 (reused excluded photo) never rendered
- [x] raw/images/05-architectures-graphs/ — 19 images; 14 OCW-excluded slides never rendered
- [x] raw/images/06-generalization-theory/ — 28 images; 8 OCW-excluded slides and slide 62 (reuses lecture 4's excluded bird photo) never rendered
- [x] raw/images/07-scaling-rules-for-optimization/ — 9 images; 3 OCW-excluded slides (7, 25, 26) never rendered
- [x] raw/images/08-architectures-transformers/ — 34 images; 7 OCW-excluded slides (4, 6, 7, 8, 31, 45, 53) never rendered; no reuse of excluded images found by pixel hash across decks 1–8
- [x] AGENTS.md — the Images conventions section

## Publish
- [x] LICENSE.md — OCW CC BY-NC-SA 4.0 attribution
- [x] kb.json — coverage, materials.method, provenance caveats
- [x] SEE_ALSO.md, if a sibling KB is genuinely relevant
- [x] verify_kb.py clean, and its review section read
- [x] Commit and push
- [x] PATCH kbUrl onto the catalog entry (https://github.com/chaimantec/cairn-kb-mit-6-7960)
- [x] Lecture 2: INDEX, AGENTS, kb.json updated
- [x] Lecture 2: verify_kb.py clean, review read; commit and push
- [x] Lecture 3: apply the Opus figure audit
- [x] Lecture 3: INDEX, AGENTS, kb.json, SEE_ALSO updated
- [x] Lecture 3: verify_kb.py clean, review read; commit and push
- [x] Lecture 4: apply the Opus figure audit
- [x] Lecture 4: INDEX, AGENTS, kb.json, SEE_ALSO updated
- [x] Lecture 4: verify_kb.py clean, review read; commit and push
- [x] Lecture 5: apply the Opus figure audit
- [x] Lecture 5: INDEX, AGENTS, kb.json, SEE_ALSO updated
- [x] Lecture 5: verify_kb.py clean, review read; commit and push
- [x] Lecture 6: apply the Opus figure audit
- [x] Lecture 6: INDEX, AGENTS, kb.json updated
- [x] Lecture 6: verify_kb.py clean, review read; commit and push
- [x] Lecture 7: apply the Opus figure audit
- [x] Lecture 7: INDEX, AGENTS, kb.json, SEE_ALSO updated
- [x] Lecture 7: verify_kb.py clean, review read; commit and push
- [x] Lecture 8: apply the Opus figure audit
- [x] Lecture 8: INDEX, AGENTS, kb.json, SEE_ALSO updated
- [x] Lecture 8: verify_kb.py clean, review read; commit and push
