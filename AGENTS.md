# How this knowledge base is organized

This repo is the knowledge base for **MIT 6.7960 — Deep Learning, Fall 2024** (Phillip Isola, Sara
Beery, Jeremy Bernstein), built from the course's MIT OpenCourseWare release. It is read by
Cairn's in-extension AI chat, which fetches files over raw.githubusercontent.com and follows
relative markdown links.

**Coverage is partial: lecture 1 of 24.** [`TODO.md`](TODO.md) is the build state.

## Layout

| Path                         | Contents                                                      |
| ---------------------------- | ------------------------------------------------------------- |
| `INDEX.md`                   | Entry point. Course summary + annotated table of contents.    |
| `wiki/`                      | Lecture pages (`NN-<slug>.md`) and cross-lecture concept pages. |
| `raw/transcripts/`           | Copy-edited transcripts with `[MM:SS]` paragraph marks — **the ones to read**. |
| `raw/transcripts/original/`  | The unedited caption text, for reference.                     |
| `raw/slides/`                | Every slide as text, `## Slide N` = PDF page N.               |
| `raw/images/NN-<slug>/`      | Whole-slide renders of figure slides (see [Images](#images)). |
| `sources.md`                 | Every OCW course document with its canonical URL.             |
| `SEE_ALSO.md`                | Sibling KBs worth reading.                                    |
| `LICENSE.md`                 | CC BY-NC-SA 4.0, following OCW; third-party exclusions.       |
| `kb.json`                    | Machine-readable description of this KB.                      |
| `TODO.md`                    | Build tracker. Unchecked boxes are outstanding work.          |

`raw/pdfs/` is gitignored: the decks stay on the build machine and are cited by their OCW URL.

## What is particular to this course

- **Source.** Everything comes from the OCW release,
  <https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/>, licensed CC BY-NC-SA 4.0.
  OCW publishes a slide deck per lecture (`mit6_7960_f24_lecN.pdf`), five problem sets with a
  solution to the fifth, and a Math Notation handout, summarized at
  [`wiki/notation.md`](wiki/notation.md). Use that notation in every page.
- **Numbering.** There is **no lecture 22**, in the playlist or among the decks. Files use the
  lecture's own number, so the catalog's positions 22 and 23 are `23-…` and `24-…`. The PyTorch
  tutorial has no deck.
- **The decks' "Lecture N" banners do not always match the recorded schedule.** Lecture 1's deck
  sends transformers to "Lecture 9" (recorded: 8), RNNs to "Lecture 11" (recorded: 10, Memory),
  generalization theory to "Lecture 7" (recorded: 6), and so on. The mapping is in
  [`wiki/course-map.md`](wiki/course-map.md#the-decks-lecture-pointers). Always cite the
  *recorded* lecture number, and check later decks for the same drift.
- **Reused deck.** Lecture 1's title slide says "6.S898 Deep Learning … Fall 2022", the course's
  earlier number and term. The transcription keeps what is printed.
- **Slide numbers are printed at bottom centre** and equal the PDF page number, so slide N is
  page N. Each deck ends with an OCW end page (page 81 in lecture 1), which is not lecture
  content.
- **OCW excludes some figures from its licence**, with a notice on the slide: "© … All rights
  reserved. This content is excluded from our Creative Commons license." Those slides are
  transcribed with an `*OCW notice: …*` line and are **never rendered into `raw/images/`**.

## Conventions

- **INDEX.md is the front door.** The chat reads it first on every conversation. Every wiki
  page must appear there with a one-line description of what it holds. An unindexed page is
  effectively invisible.
- **Relative links only** — a wiki page links a sibling page by its bare filename. Absolute GitHub URLs
  break when the repo is renamed or forked. Decks are linked at their OCW URL, never at
  `raw/pdfs/`.
- **Cite everything.** Written content cites a slide ("slide 41"); spoken content cites a
  transcript timestamp ("≈34:14").
- **Never invent course content.** If the transcript is unclear at some point, say so on the
  page. Do not fill the gap from outside knowledge — the chat presents these pages as
  authoritative material from this course. Lecture 1, for example, never writes out the softmax
  formula, so no page here does either.
- **Concept pages say what they cover so far** ("Covered so far: lecture 1"). When a later
  lecture develops a concept, extend the page and update that line.
- **Prose over fragments.** The chat quotes these pages to learners; bullet fragments quote badly.
- **Math in LaTeX**: `$...$` inline, `$$...$$` displayed, never inside a code fence. Define each
  symbol on first use. It must render both in the chat and on github.com, so: a space after any
  `_` that follows `}` or `)` (`\mathbf{w}_ j`), `\lbrace \rbrace` not `\{ \}`, `\cr` not `\\`,
  `\thinspace` not `\,`, `\ast` not `*`, `\lt \gt` in inline math, and no math inside italics.
  `check_math.mjs` (in the cairn-kb skill) checks all of this.
- **Never rank lectures against each other** ("the richest deck"). State the measurement for
  the lecture in front of you; the tables here hold the comparison.

## Transcripts

OCW's captions are **human-made**: punctuated, with speaker labels (`SARA BEERY:`,
`AUDIENCE:`) and sound tags. So the edit is light rather than the full rewrite auto-captions
need. It is a list of restorations, each checked against the slide deck, plus `[Ed: …]` notes
where the captions are ambiguous. Every change is listed in the file's header. When the lecturer
names a speaker from the audience (Jeremy Bernstein, twice in lecture 1), the `AUDIENCE:` label is
annotated.

Each edit is verified against `original/` by script. The `[MM:SS]` marker sequence must be
identical, the inventory of numbers identical once `[Ed: …]` notes are removed, and every
paragraph's word ratio within 0.72–1.10. Lecture 1: 79 markers identical, 225 numbers identical,
maximum word ratio 1.02.

## Slides

Each deck is transcribed from page images — a vision model read every page — at Sonnet, and
checked by script (`slide_number_map.py --verify`) for one heading per page in order. Where a
pale-yellow "Lecture N" banner hides slide content, the hidden prose was recovered from the PDF
text layer and the slide says so. Equations and tables are never taken from the text layer.

**Figure audit.** Lecture 1's six chart- and figure-heavy pages (21, 28, 51, 61, 63, 72) were
checked against the PDF by an independent reader. All six agreed. Three small corrections
followed: on 21, where the curve stops; on 28, a missing bump; on 72, where the arrows point. The
reader worked from full pages with bar lengths measured in pixels, not from the high-resolution
crops it was asked to use. Nothing on those pages depended on fine print. The renders of slides
23, 45 and 70 were also checked against their descriptions.

**Two clipped pages.** Slides 45 and 70 of lecture 1 are animation frames that OCW's export
caught mid-build. The heat maps (45) and batch matrices (70) are cut off, and a fragment of a
photo and some coloured boxes from the next frame show at the bottom edge. The transcript is the
record of what those slides were meant to show.

## Images

**Lecture 1 has images; no other lecture does yet.** They are committed rather than hotlinked,
and they are the only part of this KB that redistributes course material rather than describing
it.

| Lecture | Files | Where they came from |
| --- | --- | --- |
| 1 Introduction to Deep Learning | 21 of 81 pages | rendered from `mit6_7960_f24_lec1.pdf` |

Each is a whole slide at 1400px, JPEG q85 or PNG, whichever is smaller, named `slide-N` by
PDF page number.

### Using them

**Use an image path you have actually read in a file. Never construct one from the pattern, and
never assume a slide has an image because a neighbouring one does.** Most pages of lecture 1 were
deliberately not rendered (next section), so `slide-36.jpg` existing tells you nothing about
`slide-37`. Reading a path that is not in the repo returns an error rather than a URL, which
costs a turn; a guessed path is never worth it.

Links are **relative**, like every other link here: `../raw/images/01-introduction/slide-41.png`
from `wiki/`, `../images/01-introduction/slide-41.png` from `raw/slides/`. To show one, read the
path and use the URL that comes back. Do not write an absolute `raw.githubusercontent.com` URL
into a file.

All 21 of lecture 1's images appear both in
[`wiki/01-introduction.md`](wiki/01-introduction.md), in the passage that cites each slide, and
under the matching `## Slide N` heading of
[`raw/slides/01-introduction.md`](raw/slides/01-introduction.md). To list them:
`grep -o 'raw/images/[^)]*' wiki/01-introduction.md`. The concept pages embed none; they cite
slides, and the lecture page carries the pictures.

- **Prefer the transcription for numbers and formulas.** The slide file reproduces every
  equation and table as text; use the image to *show*, not to read values off.
- **Show one image, not a gallery.**
- **Keep the citation**, so the reader can find the rest of that slide in `raw/slides/`.

### What was rendered, and what was not

Rendered (21): slides 14, 21, 22, 23, 28, 33, 34, 35, 36, 38, 40, 41, 42, 43, 45, 51, 55, 57, 58,
59, 70 — the XOR plot, the enthusiasm curves, the loss surface, the linear-layer and perceptron
diagrams, the three activation plots, stacked layers, the two-layer classification network,
double descent, the classifier and loss diagrams, cross-entropy, and the start of the batched
build.

Not rendered, and why:

- **Excluded from OCW's licence (22 slides): 1, 8, 9, 11, 13, 16, 19, 20, 54, 60, 61, 62, 63,
  64, 65, 66, 68, 69, 72, 73, 75, 77.** Each carries an "All rights reserved" notice — AlexNet
  and LeNet figures, photos and book covers, the clown fish / bear / chameleon photos, the Serre
  and Donahue hierarchy figures, and others. OCW includes them under its own fair-use
  determination, which does not pass to copies. They are described in full in the slide file, and
  that description is the only representation this KB has. The bar-chart slides 61 and 63 among
  them had a figure audit.
- **Build steps superseded by a rendered slide:** 7, 10, 12, 15, 18 (the enthusiasm curve before
  its final form on 21–23); 37 (slide 36 without its labels); 39 (slide 38's plot with bullet
  text); 56 (the loss diagram before 57 and 58).
- **No figure worth a picture:** signposts, agendas and text slides; slide 27 (an equation the
  file reproduces exactly); slide 32 (two empty boxes); slide 44 (blank); slide 81 (OCW end page).
- **Mostly hidden behind a "Lecture N" banner:** 30, 47, 49, 52, 79. Their visible text is
  transcribed.

### Provenance and attribution

Rendered slides are from *MIT 6.7960 Deep Learning, Fall 2024*, MIT OpenCourseWare
(<https://ocw.mit.edu>), CC BY-NC-SA 4.0, by Phillip Isola, Sara Beery and Jeremy Bernstein;
lecture 1's deck is by Sara Beery. Two rendered slides contain material from elsewhere that OCW
did not flag. **Slide 51** reproduces the double-descent figure of Belkin, Hsu, Ma and Mandal,
"Reconciling modern machine-learning practice and the classical bias–variance trade-off"
(PNAS, 2019), credited on the slide. **Slides 45 and 70** show, at their clipped bottom edge, a
fragment of an uncredited bird photograph from the next animation frame. If a rights holder or the
course asks for a page to come down, delete the image file and every
image embed that points at it.

## Rebuilding

Built and updated by the `cairn-kb` skill; [`TODO.md`](TODO.md) lists the remaining lectures.
Notes for the next run, from lecture 1:

- Download the deck and run `slide_number_map.py`. As of this build it reads bottom-centre
  numbers, which OCW decks use; before that it read axis labels as slide numbers.
- Transcripts: run the light edit described above, not a full rewrite.
- Images: render figure slides that carry **no** OCW exclusion notice. Find the notices with a
  text-layer grep for "excluded from our Creative Commons license", or the `*OCW notice` lines
  in the slide file.
- After writing, run `check_math.mjs` and `verify_kb.py`.
