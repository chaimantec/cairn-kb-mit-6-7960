# How this knowledge base is organized

This repo is the knowledge base for **MIT 6.7960 — Deep Learning, Fall 2024** (Phillip Isola, Sara
Beery, Jeremy Bernstein), built from the course's MIT OpenCourseWare release. It is read by
Cairn's in-extension AI chat, which starts at `INDEX.md`, fetches files over raw.githubusercontent.com and follows
relative markdown links. This file is for whoever builds or maintains the KB; the chat does not need it.

**Coverage is partial: lectures 1–21 and 23 of 24.** [`TODO.md`](TODO.md) is the build state.

**This file holds the rules every lecture's build follows. The per-lecture record is in
[`BUILD_LOG.md`](BUILD_LOG.md):** each deck's lecture pointers and end page, its OCW notices, its transcript edits,
what its figure audit found, its printed slips, and which slides were rendered and why the others were not. Read it
for precedent when building or revisiting a lecture. When a lecture teaches something new, write the lesson here in
general form and the lecture's details there; keep this file free of per-lecture narrative.

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
| `LICENSE.md`                 | CC BY-NC-SA 4.0, following OCW; third-party exclusions; attribution of the rendered images. |
| `kb.json`                    | Machine-readable description of this KB.                      |
| `TODO.md`                    | Build tracker. Unchecked boxes are outstanding work.          |
| `BUILD_LOG.md`               | Per-lecture build record. Not needed to answer questions.     |

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
- **The decks' "Lecture N" numbers do not always match the recorded schedule.** Lecture 1's deck
  sends transformers to "Lecture 9" (recorded: 8), RNNs to "Lecture 11" (recorded: 10), and so on, and later
  decks' outline slides repeat such numbers ("9. Transformers" in lecture 8, "12. Representation Learning I" in
  lecture 11). Always cite the *recorded* lecture number. For each new lecture, record what its deck and recording
  say about other lectures in [`wiki/course-map.md`](wiki/course-map.md#the-decks-lecture-pointers) and in the
  build log.
- **Reused deck.** Lecture 1's title slide says "6.S898 Deep Learning … Fall 2022", the course's
  earlier number and term, and lecture 5's footer prints "6.S898 Deep Learning" with "Fall 2024".
  The transcription keeps what is printed. Lecture 21, a guest lecture by Jacob Andreas, is assembled from MIT 6.861's decks, and its
  title page hides 6.861's "Lecture 8: Prompting" under white boxes; cite it as lecture 21.
- **Slide numbers are printed at bottom centre** and equal the PDF page number, so slide N is
  page N. Each deck ends with an OCW end page, usually a smaller page, which is not lecture content.
  `slide_number_map.py` often reports that page as printing no number although it prints one (an end page stored rotated 90° hides
  its number from the script, lecture 23); say so in the slide file rather than trusting the script. A deck assembled from another course's slides can print its own number on only a few pages and
  carry the source decks' numbers in a corner (lecture 21: pages 1–7 only). The script then reports a numbering with gaps; slide N is
  still page N, and since `extract_slide_images.py` resolves through that map, render such a deck by page with the same settings. A deck
  built from an earlier talk can instead print both on most pages, its own small number at bottom centre and the talk's larger one in a
  corner (lecture 23); the script reads the bottom-centre number correctly, but the slide file must say which is which.
- **Handwritten decks.** Lectures 3, 7 and 13 are Jeremy Bernstein's iPad notes; his lecture 23, against the expectation recorded
  here, is a typed Google Slides deck with a usable text layer, so check a deck before assuming its format. Their text layer is OCR noise: take nothing from it but typed OCW notices. The ink is vector
  paths, so the raster test for figure pages misses it; and lectures 7 and 13 paint one background image on every page, so
  the raster test flags every page there. Choose figure pages from the slide file.
- **OCW excludes some figures from its licence**, with a notice on the slide: "© … All rights
  reserved. This content is excluded from our Creative Commons license." (sometimes without "All rights
  reserved"). Those slides are transcribed with an `*OCW notice: …*` line and are **never rendered into
  `raw/images/`**. A notice binds even where it looks wrong (lecture 10's photographs of the lecturer's own cat). A
  notice can be text or **glyph outlines**: 35 of lecture 12's 37 notices are filled vector paths that `get_text()`
  cannot see. So take the excluded set from the slide file's `*OCW notice` lines, which a vision model wrote, and
  use the text layer and a vector scan only to cross-check (see Rebuilding).
- **An excluded image can reappear without its notice**, in the same deck, another deck or a problem set, and such a
  slide is not rendered either. A deck can also paste its own pictures over a third-party raster, so that the page shows
  little of it; the raster is still the third party's figure. A notice binds even where an earlier deck credits the same
  picture under CC (lecture 12's slide 35), and a slide already rendered is withdrawn when a later deck excludes one of
  its images. Slides withheld this way so far:

  | Lecture | Slide | Reuses |
  | --- | --- | --- |
  | 1 | 45, 70 | lecture 4's excluded heron photograph (slide 43 there), cut off at the bottom edge of both pages (withdrawn in the lecture 21 run) |
  | 1 | 51 | lecture 6's excluded slide 30 (Belkin et al.'s double-descent figure; withdrawn when lecture 6 was added) |
  | 2 | 70 | lecture 18's excluded slide 63 (the CLIP "Text Encoder" and "Image Encoder" graphics; withdrawn when lecture 18 was added) |
  | 4 | 50–52 | slide 25's clown fish crop |
  | 4 | 74, 75 | lecture 10's excluded slides 9 and 14 (the video frames and the space–time cube; withdrawn in the lecture 21 run) |
  | 6 | 62 | lecture 4's excluded heron photograph (slides 43, 53–55, 66–68, 70, 71) |
  | 8 | 15 | lecture 10's excluded slide 9 (the audio waveform; withdrawn in the lecture 21 run) |
  | 11 | 13 | lecture 1's excluded slide 73 (the CLIP figure, at another resolution) |
  | 12 | 35 | lecture 18's excluded slide 33 (the classroom photograph and depth map, credited CC BY-NC-SA here; withdrawn when lecture 18 was added) |
  | 13 | 3 | lecture 23's excluded slide 7 (the NanoGPT speedrun chart, credited only "@kellerjordan0" here; withdrawn in the lecture 23 run) |
  | 14 | 49, 50 | Homework 5's excluded Figure 3, Ho, Jain and Abbeel's diffusion diagram, under the deck's pixel-art overlays |
  | 16 | 12 | slide 11's excluded photograph (also lecture 4's excluded slides 12–24) |
  | 16 | 18–31 | the label map and building photographs of slide 73, which repeats slide 30 image for image under a notice |
  | 18 | 74 | slide 63's excluded encoder graphics (the slide is lecture 2's slide 70 again) |
  | 19 | 23 | slide 42's excluded volcano grid |
  | 19 | 29–31 | slide 28's cat photograph (29, 30), and the generator's reconstruction and edits of it (31) |
  | 19 | 36 | lecture 18's excluded tiger photographs (slides 12 and 13 there), credited CC BY here |

  A shared image is not always a shared figure, so check size, placement and which image a notice sits beside
  before withholding. Rendered after such a check: lecture 5's slides 12, 15, 21 and 29 (the shared object is a
  33×31 px node glyph), lecture 9's slide 57 (it repeats slide 4's lower row, not the X-ray the notice covers), and
  lecture 11's slides 31, 39 and 42 (the shared images are listed in the page's resources but drawn off the page or not
  at all). The method is under Rebuilding. Placement is not the only evidence of what a notice covers: when its "©" names the
  authors of the paper another picture on the slide comes from, treat that picture as covered too, and its reuses as withheld
  (lecture 19's slide 28, whose "© Lang et al." sits by a row of portraits but names the source of the cat beside them).

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
- **Printed slips and spoken slips** are transcribed as printed or said, and flagged twice: at the slide in the slide
  file (or with an `[Ed: …]` note in the transcript), and wherever the wiki cites them, with both versions given and
  neither silently corrected. Outside knowledge used to explain a slip is marked as outside the course material.

## Transcripts

OCW's captions are **human-made**: punctuated, with speaker labels (`SARA BEERY:`,
`AUDIENCE:`) and sound tags. So the edit is light rather than the full rewrite auto-captions
need. It is a list of restorations, each checked against the slide deck, plus `[Ed: …]` notes
where the captions are ambiguous. Every change is listed in the file's header. When the lecturer
names a speaker from the audience, the `AUDIENCE:` label is annotated; a colleague named in passing gets an inline
note ("Phil" is Phillip Isola). The edit is small enough to do inline rather than through a subagent.

Each edit is verified against `original/` by script. The `[MM:SS]` marker sequence must be
identical, the inventory of numbers identical once `[Ed: …]` notes are removed, and every
paragraph's word ratio within 0.72–1.10 once those notes are removed (a paragraph carrying a note runs higher). A
deliberate change to a number (lecture 11's "4DA" → "Fourier") is listed in the header.

## Slides

Each deck is transcribed from page images — a vision model read every page — at Sonnet, and
checked by script (`slide_number_map.py --verify`) for one heading per page in order. Where a
pale-yellow "Lecture N" banner hides slide content, the hidden prose was recovered from the PDF
text layer and the slide says so. Reused decks hide whole blocks under white boxes: the source course's title, the next build step's rows, an earlier draft (lecture 21).
Overlays pasted over a figure can hide content its embedded image still holds (lecture 19's
slide 45 covers three lines of MAML's algorithm); the audit can read it from the raster at native resolution, and the slide
file gives it as hidden, never as what the page shows. Equations and tables are never taken from the text layer. Symbols are
transcribed in the type the page prints: lecture 23's transcriber set w, x and y in bold "for consistency with the course notation",
which the page prints plain, and 123 of them had to be undone; the course notation belongs in the wiki, not the slide file.

Every deck is then audited against the PDF: from lecture 3 on by Opus, a different model from the transcriber, on its
chart-, diagram-, table- and equation-heavy pages; lectures 1 and 2 by Sonnet. Each slide file's front matter carries a
`figure_audit:` summary, and kb.json's `figureAudit` the per-deck detail.

## Images

**Lectures 1–21 and 23 have images; no other lecture does yet.** They are committed rather than hotlinked,
and they are the only part of this KB that redistributes course material rather than describing
it.

| Lecture | Files | Where they came from |
| --- | --- | --- |
| 1 Introduction to Deep Learning | 18 of 81 pages | rendered from `mit6_7960_f24_lec1.pdf` |
| 2 How to Train a Neural Net | 44 of 81 pages | rendered from `mit6_7960_f24_lec2.pdf` |
| 3 Approximation Theory | 18 of 43 pages | rendered from `mit6_7960_f24_lec3.pdf` |
| 4 Architectures: Grids | 29 of 84 pages | rendered from `mit6_7960_f24_lec4.pdf` |
| 5 Architectures: Graphs | 19 of 47 pages | rendered from `mit6_7960_f24_lec5.pdf` |
| 6 Generalization Theory | 28 of 66 pages | rendered from `mit6_7960_f24_lec6.pdf` |
| 7 Scaling Rules for Optimization | 9 of 32 pages | rendered from `mit6_7960_f24_lec7.pdf` |
| 8 Architectures: Transformers | 33 of 55 pages | rendered from `mit6_7960_f24_lec8.pdf` |
| 9 Hacker's Guide to Deep Learning | 11 of 72 pages | rendered from `mit6_7960_f24_lec9.pdf` |
| 10 Architectures: Memory | 32 of 69 pages | rendered from `mit6_7960_f24_lec10.pdf` |
| 11 Representation Learning: Reconstruction-Based | 25 of 65 pages | rendered from `mit6_7960_f24_lec11.pdf` |
| 12 Representation Learning: Similarity-Based | 7 of 70 pages | rendered from `mit6_7960_f24_lec12.pdf` |
| 13 Representation Learning: Theory | 9 of 28 pages | rendered from `mit6_7960_f24_lec13.pdf` |
| 14 Generative Models: Basics | 28 of 60 pages | rendered from `mit6_7960_f24_lec14.pdf` |
| 15 Generative Models: Representation Learning Meets Generative Modeling | 22 of 42 pages | rendered from `mit6_7960_f24_lec15.pdf` |
| 16 Generative Models: Conditional Models | 29 of 84 pages | rendered from `mit6_7960_f24_lec16.pdf` |
| 17 Generalization: Out-of-Distribution (OOD) | 11 of 63 pages | rendered from `mit6_7960_f24_lec17.pdf` |
| 18 Transfer Learning: Models | 26 of 77 pages | rendered from `mit6_7960_f24_lec18.pdf` |
| 19 Transfer Learning: Data | 15 of 52 pages | rendered from `mit6_7960_f24_lec19.pdf` |
| 20 Scaling Laws | 20 of 52 pages | rendered from `mit6_7960_f24_lec20.pdf` |
| 21 Language Models | 18 of 92 pages | rendered from `mit6_7960_f24_lec21.pdf` |
| 23 Metrized Deep Learning | 8 of 49 pages | rendered from `mit6_7960_f24_lec23.pdf` |

Each is a whole slide at 1400px, JPEG q85 or PNG, whichever is smaller, named `slide-N` by
PDF page number. Render only figure slides that carry no OCW notice and reuse no excluded image; skip build steps
superseded by a rendered slide, and text, equation, table and code slides the slide file reproduces exactly. Which
slides were rendered for each lecture, and why each other slide was not, is in the build log.

### Using them

**Use an image path you have actually read in a file. Never construct one from the pattern, and
never assume a slide has an image because a neighbouring one does.** Many pages of each deck were
deliberately not rendered, so lecture 1's `slide-36.jpg` existing tells you nothing about `slide-37`. The extension
differs from slide to slide too (`.jpg` or `.png`, whichever was smaller), so copy the whole path. Reading a path that
is not in the repo returns an error rather than a URL, which costs a turn; a guessed path is never worth it.

Links are **relative**, like every other link here: `../raw/images/01-introduction/slide-41.png`
from `wiki/`, `../images/01-introduction/slide-41.png` from `raw/slides/`. To show one, read the
path and use the URL that comes back. Do not write an absolute `raw.githubusercontent.com` URL
into a file.

Every image appears under its `## Slide N` heading in `raw/slides/` and in the wiki lecture page's passage that cites
that slide (the one exception: lecture 2's slides 13 and 54, near-duplicates of their neighbours, are in the slide
file only). To list a lecture's images: `grep -o 'raw/images/[^)]*' wiki/NN-*.md`. The concept pages embed none;
they cite slides, and the lecture pages carry the pictures.

- **Prefer the transcription for numbers and formulas.** The slide file reproduces every
  equation and table as text; use the image to *show*, not to read values off.
- **Show one image, not a gallery.**
- **Keep the citation**, so the reader can find the rest of that slide in `raw/slides/`.

Attribution of the third-party material in rendered images, lecture by lecture, is in
[`LICENSE.md`](LICENSE.md#attribution-of-the-rendered-slide-images); add each new lecture's there. If a rights holder
or the course asks for a page to come down, delete the image file and every image embed that points at it.

## Rebuilding

Built and updated by the `cairn-kb` skill; [`TODO.md`](TODO.md) lists the remaining lectures.
Notes for the next run, from lectures 1–21 and 23:

- Download the deck and run `slide_number_map.py`. As of this build it reads bottom-centre
  numbers, which OCW decks use; before that it read axis labels as slide numbers.
- Transcripts: run the light edit described above, not a full rewrite. Check that the recording covers the whole deck: compare
  its last paragraph with the deck, and with the end of OCW's transcript PDF (lecture 20's video and transcript both stop
  mid-sentence after slide 17 of 51). When it does not, say so at the top of the transcript, the slide file and the wiki page,
  and mark every wiki section that comes from the deck alone.
- **Find the OCW notices in the slide file**, from its `*OCW notice` lines. Cross-check with the text layer, grepping
  each page's text with newlines joined, or for "All rights reserved" and "excluded from our Creative" alone (the
  sentence wraps across lines; a single-line grep for the whole sentence found one of lecture 2's seven notices). Then
  scan the vector data for notices drawn as outlines: short, wide, filled paths of many segments (`page.get_drawings()`,
  fill set, 40 or more items, under 60 pt high) mark outlined small print, and every page with them should be either a
  notice or a "Courtesy … Used under CC" credit. Lecture 12's text layer showed two of its 37 notices.
- **Check every candidate image for reuse of an excluded one**, in its own deck and across all decks and problem sets
  so far, and every image already rendered against the new deck's excluded ones. xrefs are local to one PDF, so compare by a hash of the
  pixels (`hashlib.md5(fitz.Pixmap(doc, xref).samples)`), and also by a coarse 16×16 perceptual hash, mirrored too.
  Convert indexed (palette), CMYK and ICC images to RGB before the coarse hash, and count the images that fail to hash: lecture 21's
  first sweep silently skipped 347 indexed-colour rasters across all decks and still found the known reuses, so only the count showed it.
  Report every candidate page an image is drawn on, not each image once, and rerun the whole sweep over every lecture already built
  whenever the sweep or the excluded set changes: lecture 21's full re-sweep found five rendered slides (lecture 1's 45 and 70, lecture
  4's 74 and 75, lecture 8's 15) reusing lecture 4's and lecture 10's excluded images, all rendered before their decks were compared.
  An exact match between images of different sizes is no evidence: a uniform fill drawn through a soft mask hashes like every other
  uniform fill of the same byte count (lecture 16's slide 33 against lecture 9's slide 61), so skip images whose samples hold one value.
  Skip coarse hashes with fewer than 8 or more than 248 bits set (near-uniform masks match each other), and settle
  any coarse match by resizing both images to 256 × 256 and averaging the absolute grey difference over pixels darker
  than 200 in either: about 1 in 256 is the same figure, 50 or more is a different one. In between, and for sparse line art
  or text, the number does not decide: lecture 18 found one figure at 9.2 under another colour profile, and unrelated
  white-background drawings at 3.9. Keynote's drop shadows are black rasters drawn through a soft mask; they look like black
  rectangles and match one another coarsely at 3 to 6, so clear them on sight. Look at both images before withholding or clearing. A withheld slide puts all of its images in the excluded set, not only the one
  it was withheld for, so expect matches against its other images and clear them by looking (lecture 11's slide 60 shares only a
  photograph with lecture 8's slide 15, withheld for its audio waveform). Keep the script's withheld list equal to the table above: the
  copy the lecture 23 run inherited lacked the lecture 21 run's five withdrawals. Use only images actually drawn
  on the page: `page.get_images()` lists everything a page's resources hold, including images drawn off the page or
  not at all, while `page.get_image_info(xrefs=True)` gives where each is drawn. Leave out page backgrounds: an image
  drawn on more than a third of a deck's pages, or a raster covering more than 90% of a page. The handwritten decks share
  one background image, so without that filter every page matches every excluded page of another handwritten deck,
  exactly. But a raster drawn on one page only is not a background however much of it it covers (lecture 14's DALL-E
  screenshot is drawn larger than its page), so sweep those too. The excluded set the sweep uses must be the slide
  file's, plus the withheld slides in the table above, plus the images on every problem-set page with a notice. Before
  trusting a clean result, rerun the sweep on lectures whose reuse is known (lecture 11's CLIP figure, lecture 14's
  diffusion chain); a sweep that does not find them is broken.
- `page.get_image_info()` reports vector shadings as images with xref 0 and gives each image's unclipped bounding box, so a page
  can measure near-full raster coverage with no large picture on it (a clipped three-chart image; gradient squares). Count only
  xrefs above 0, intersect with the clip, and take figure pages from the slide file.
- `embed_slide_images.py` anchors a wiki image at the slide's **first** citation, and a range such
  as "slides 15–20" places every slide in it at that spot. Write the wiki so that each slide's first
  citation is the passage it belongs in, and avoid ranges in overview paragraphs. It matches any
  "slide N" in the page, so do not cite *another* lecture's slide numbers in a lecture page (lecture 4
  avoided writing "lecture 3's slide 42", which would have pulled in its own slide 42). It inserts after
  the citing paragraph, so a paragraph ending in a colon before a `$$` block gets the image between the
  colon and the equation; cite the slide after the equation instead. It also reads only the first
  number of a list: "slides 36 and 39" did not count as citing slide 39 in lecture 5, which had to be
  hand-placed.
- Hand-placed wiki images (to write a better caption) must copy the path from disk: five of lecture 4's
  first hand-placed links guessed `.png` for files that are `.jpg`, and three of lecture 19's, written before rendering, guessed
  wrong too. Write image links only after rendering. The script skips images a page
  already links, so hand-place the key figures first and let it place the rest. Its default caption is
  the slide title, which reads badly for parenthetical or question titles; rewrite those, and keep math
  out of the italic captions. It also copies the slide title into the slide file's alt text, so a
  title with math in it (lecture 4's slides 7, 8 and 10) leaves `$` in an alt text, which github.com
  then fails to parse; strip it.
- **Handwritten decks** (lectures 3, 7 and 13): do not grep the text layer for anything but typed notices;
  read every page, and choose figure pages from the slide file. Script and letter ambiguities (ℓ against L,
  subscripts against superscripts) need the audit, and the lecturer usually reads each formula aloud, so check the
  formulas against the transcript too. Where one background image is painted on every page, drop any xref that
  appears on every page before measuring raster coverage. Ink painted in the background colour can hide typed text,
  plot labels and icons that the text layer and image list still contain; transcribe what the page shows and say what
  is hidden.
- **Audit with a different model from the transcriber.** A same-model audit catches misreadings
  caused by resolution but not ones the two runs share. Have Opus check the chart-, diagram-, table-, equation- and
  photo-heavy pages from 150–600 dpi crops, the embedded rasters at native resolution and the vector data, and tell it
  it may count nodes, edges, dots and markers from `page.get_drawings()`. On a typed deck the text layer's font names and fill colours
  settle bold against plain and colour claims (lecture 23). The audits have found errors on roughly a
  third to a half of the pages checked: mostly counts, positions, colours and arrow directions; now and then an object
  or relation described that is not on the page; almost never a formula. Split a long audit between agents and have
  each append its report page by page: session limits stopped auditors in lectures 6, 11 and 12, and the reports on disk
  let fresh agents finish only the remaining pages. Ask the auditor to flag names the slide and the recording never use
  (a transcriber named a bird's species from outside knowledge in lecture 15, and an art style in lecture 16) and typos in
  screenshots corrected without a flag (lecture 16's "enalbling"). Then look at two rendered images against
  the corrected text.
- **Typed decks with LaTeXiT equations** (lecture 8): the text layer holds each equation as base64 junk, so
  it is good for titles, labels, citations, notices and code but never for an equation.
- **Stray marks in an equation** (lecture 12's "ˆ ˜ ˙ ˇ" for ⊤, τ, ∼, Σ) can be real glyphs of an embedded subset font
  with nothing drawn behind them. No crop shows the intended symbol; transcribe the reading the glyph's size and
  position support, and say so.
- **Garbled labels may decode.** Text that prints as punctuation (`!"#"$"%&`) can be a subset font that numbers its
  glyphs in order of first use. `page.get_texttrace()` gives the glyph ids; if repeated letters map consistently and
  the lengths match a known string, the decode is sound (lecture 12's slide 37). Put such strings in code spans in the
  slide file, since their `$` and `&` otherwise parse as math.
- The transcriber tends to write a quoted label that starts with math (`"$N$ tokens"`), which github.com
  will not render: inline math must not open straight after a quote mark. `check_math.mjs` warns; drop the
  quotes.
- A `$$` block written on a line touching prose renders as inline math on github.com, and the offline `check_math.mjs`
  run does not flag it; only `--github` (which `verify_kb.py` runs) does. Put a blank line either side of every block.
- **PyMuPDF on a downloaded deck**: `python3 -I` hides the user site-packages where PyMuPDF is
  installed, and this machine's Python 3.9 has no `-P`. Keep scripts in the session scratchpad and run them
  from there with the PDF path as an argument.
- After writing, run `check_math.mjs` and `verify_kb.py`. `verify_kb.py` reads only tracked files, so `git add` new
  pages first or it reports their images as unreferenced.
