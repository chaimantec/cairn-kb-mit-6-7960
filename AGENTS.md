# How this knowledge base is organized

This repo is the knowledge base for **MIT 6.7960 — Deep Learning, Fall 2024** (Phillip Isola, Sara
Beery, Jeremy Bernstein), built from the course's MIT OpenCourseWare release. It is read by
Cairn's in-extension AI chat, which fetches files over raw.githubusercontent.com and follows
relative markdown links.

**Coverage is partial: lectures 1–4 of 24.** [`TODO.md`](TODO.md) is the build state.

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
  *recorded* lecture number, and check later decks for the same drift. Lecture 2's, lecture
  3's and lecture 4's decks have no pointers to other lectures; their own "Lecture 2", "Lecture 3"
  and "Lecture 4" match the recording. Lecture 3's last slide previews "Inductive biases" with no
  lecture number, and lecture 4's recording points ahead to "the transformers lecture" (8) without
  one.
- **Reused deck.** Lecture 1's title slide says "6.S898 Deep Learning … Fall 2022", the course's
  earlier number and term. The transcription keeps what is printed.
- **Slide numbers are printed at bottom centre** and equal the PDF page number, so slide N is
  page N. Each deck ends with an OCW end page (page 81 in lectures 1 and 2, page 43 in lecture
  3, page 84 in lecture 4), which is not lecture content.
- **Lecture 3's deck is handwritten** — Jeremy Bernstein's iPad notes, in several ink colours on a
  dark background. Its PDF text layer is OCR of the handwriting and is useless (it reads
  "Hongenoucin" for a handwritten credit), so nothing was taken from it. The handwriting is vector
  ink, not a raster, so the raster-coverage test for figure pages finds almost nothing on such a
  deck; the slide file's descriptions decided which pages were rendered. Expect the same of
  Bernstein's later decks (lectures 7 and 23 are his on the schedule).
- **OCW excludes some figures from its licence**, with a notice on the slide: "© … All rights
  reserved. This content is excluded from our Creative Commons license." Those slides are
  transcribed with an `*OCW notice: …*` line and are **never rendered into `raw/images/`**.
  Lecture 4's deck carries 35 such notices in 84 pages, among them every figure built on Fredo
  Durand's stork and heron photographs (patch classification, segmentation, feature maps, receptive
  fields, encoder–decoder, U-net, ResNet), so those figures exist here only as prose.
- **An excluded image can reappear without its notice.** Lecture 4's slides 50–52 print no notice,
  but their photo crop is the same embedded image object as slide 25's clown fish, which slide 25
  marks "© source unknown. All rights reserved". They are treated as excluded and not rendered.
  Before rendering, compare each candidate page's image xrefs (`page.get_images()`) against the
  excluded pages'; for lectures 1, 2 and 4 the rendered set shares none.

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
annotated. A colleague named in passing gets an inline note ("Phil" in lecture 2 is Phillip Isola).

Each edit is verified against `original/` by script. The `[MM:SS]` marker sequence must be
identical, the inventory of numbers identical once `[Ed: …]` notes are removed, and every
paragraph's word ratio within 0.72–1.10. Lecture 1: 79 markers identical, 225 numbers identical,
maximum word ratio 1.02. Lecture 2: 103 markers identical, all 29 numerals identical (counted as
digit strings outside the markers), word ratios 1.00–1.01. Lecture 2's edit is 25 restorations —
eleven of them "differential" → "differentiable" — and five `[Ed: …]` notes. Lecture 3: 106
markers identical, all 125 digit strings identical, word ratios 1.00–1.01; ten restorations (four
"value" → "ReLU", two "kicks" → "kinks", two "4D" → "4d", "Chinchilla" capitalized, "really
nonlinearity" → "ReLU nonlinearity") and two `[Ed: …]` notes on unclear student remarks. Lecture
4: 109 markers identical, all 79 digit strings identical, word ratios 0.98–1.01 (the low end is "VGG
16" → "VGG16" joining two words); six restorations ("wait" → "weight", "acts" → "x", "C sub L" → "C
sub l", "VGG 16, ResNet 18" → "VGG16, ResNet18", "differential" → "differentiable", "resonance" →
"ResNets"), three slashes restored to question marks, and two `[Ed: …]` notes ("tool blocks",
"ladder"). The edit was small enough to do inline rather than through a subagent.

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

**Lecture 2's figure audit.** Eight chart-, diagram- and number-heavy pages (10, 12, 14, 19, 51,
52, 79, 80) were checked against the PDF by an independent reader working from cropped renders.
Seven agreed. Slide 19 needed two corrections: the cusp is not symmetric, and the optimizer path's
last swing reaches about −0.55, not −0.8. The render of slide 19 was then checked against the
corrected description, and the render of slide 53 against its own.

**Lecture 2's printed slips**, transcribed as printed and flagged where they are cited: slide 36
drops a $\partial$ from one denominator; slide 75 writes the loss with $\mathbf{x}_ 2$ where
$\mathbf{x}_ 3$ is meant; slide 79's last line multiplies by $-0.1186$ where
$\partial \mathcal{L} / \partial \mathbf{x}_ 3 = -0.1869$ is meant (its printed result uses the
right value; the audit confirmed the misprint is on the page); slides 8, 10, 21, 22 and 39 leave a
blank where the assignment sign of an update rule goes; and slide 10's plots call the momentum
coefficient $\mu$ where its equation says $\alpha$. Slides 48, 49, 76 and 80 write weight updates
with $+\eta$ and a negative learning rate (slide 72 explains why), where the gradient-descent
slides write $-\eta$; the wiki explains the convention rather than changing it.

**Lecture 3's figure audit** was the first done by a different model from the transcriber: Sonnet
read the handwritten deck, and Opus checked 19 equation-, diagram- and chart-heavy pages (5, 6, 10,
12–17, 27–33, 35, 38, 39) from 250–1200 dpi crops. Every formula agreed. Corrections followed on
slide 6 (where the curve's features sit), 12 (13 strips of unequal width, not "about twelve" equal
ones), 27 (the first kink is below the axis), 29 (six green dots, not nine; the other three kinks
share the pink dots), 30–33 (the layer symbol is the lecturer's capital L throughout, not a script
ℓ) and 38 (several values, and start points for the 6- and >6-layer series that the first reading
had invented), plus minor colour and arrow-direction fixes. The renders of slides 29 and 38 were
then checked against the corrected text.

**Lecture 3's printed slips and oddities**, transcribed as written: slide 5's two-layer formula
ends in $+ \beta_ i$ with no brackets, so whether the bias is inside the sum is unclear; slide 13
says "The triangles has area"; slide 30 gives the base case as $\text{KINKS}_ 0 = 1$, which a
student corrects to 0 in the recording (≈1:03:17) and the lecturer leaves unresolved; slide 39 cites
the Chinchilla paper as "Hoffmann, Borgeau, Mensch et al (2020)"; and slide 1 prints the email
address "jbernstein@mit.edub".

**Lecture 4's figure audit**, also cross-model: Sonnet read the deck, and Opus checked 24 chart-,
equation- and diagram-heavy pages (7, 8, 10, 18, 25, 31, 32, 37, 39–41, 48–52, 58–60, 62, 71, 77–79)
from 600-dpi crops. Every equation agreed. Corrections followed on slides 7, 8 and 10 (five training
points in the middle panels, not "about six"; slide 8's single-point fit has eight peaks, not eleven;
where the learned curves go), 32 (a centre-surround band, not a blur), 37 (twelve nodes per column, and
where $\mathbf{x}_ L[j]$ sits), 41 (which group each brace labels), 51–52 (the edge moves right, past
the window), 58 (three outputs in all), 59 and 60 (grid sizes). It also identified slides 50–52's photo
crop as slide 25's excluded clown fish (above) and confirmed that slides 9, 78 and 79 print a lowercase
$l$ for image intensity. The renders of slides 42 and 74 were checked against their descriptions.

**Lecture 4's printed slips and oddities**, transcribed as written: slide 40's last line prints
$\mathbf{x}_ {	ext{out}}[C, :]$ on the left where its right side, indexed from 0, means $C - 1$;
slide 39's multichannel formula ends in $+ b[c]$ with no brackets, so whether the bias is inside the sum
is unclear; slides 48–49 use $j$ both as the output index and as the index over the window
$\mathcal{N}(j)$; slide 3 says "Embarassingly"; slide 19 says "not easy to recognize content in small
each patch"; slide 46 says "an filter"; and slide 72 repeats slide 69 unchanged. Slide 25's filter
detects vertical edges, while the lecturer calls them "horizontal edges" (≈23:05); the wiki reports
both.

## Images

**Lectures 1, 2, 3 and 4 have images; no other lecture does yet.** They are committed rather than hotlinked,
and they are the only part of this KB that redistributes course material rather than describing
it.

| Lecture | Files | Where they came from |
| --- | --- | --- |
| 1 Introduction to Deep Learning | 21 of 81 pages | rendered from `mit6_7960_f24_lec1.pdf` |
| 2 How to Train a Neural Net | 45 of 81 pages | rendered from `mit6_7960_f24_lec2.pdf` |
| 3 Approximation Theory | 18 of 43 pages | rendered from `mit6_7960_f24_lec3.pdf` |
| 4 Architectures: Grids | 31 of 84 pages | rendered from `mit6_7960_f24_lec4.pdf` |

Each is a whole slide at 1400px, JPEG q85 or PNG, whichever is smaller, named `slide-N` by
PDF page number.

### Using them

**Use an image path you have actually read in a file. Never construct one from the pattern, and
never assume a slide has an image because a neighbouring one does.** Many pages of each deck were
deliberately not rendered (next section), so lecture 1's `slide-36.jpg` existing tells you nothing
about `slide-37`, lecture 2's `slide-38` tells you nothing about `slide-39`, lecture 3's
`slide-17` tells you nothing about `slide-18`, and lecture 4's `slide-49` tells you nothing about
`slide-50`. The extension differs from slide to slide too (`.jpg` or `.png`, whichever was smaller), so
copy the whole path. Reading a path that is not in the repo returns an error rather than a URL, which
costs a turn; a guessed path is never worth it.

Links are **relative**, like every other link here: `../raw/images/01-introduction/slide-41.png`
from `wiki/`, `../images/01-introduction/slide-41.png` from `raw/slides/`. To show one, read the
path and use the URL that comes back. Do not write an absolute `raw.githubusercontent.com` URL
into a file.

All 21 of lecture 1's images appear both in
[`wiki/01-introduction.md`](wiki/01-introduction.md), in the passage that cites each slide, and
under the matching `## Slide N` heading of
[`raw/slides/01-introduction.md`](raw/slides/01-introduction.md). To list them:
`grep -o 'raw/images/[^)]*' wiki/01-introduction.md`. Of lecture 2's 45 images, 43 appear in
[`wiki/02-how-to-train-a-neural-net.md`](wiki/02-how-to-train-a-neural-net.md); slides 13 and 54,
near-duplicates of their neighbours 12 and 55, are only under their headings in
[`raw/slides/02-how-to-train-a-neural-net.md`](raw/slides/02-how-to-train-a-neural-net.md). All
18 of lecture 3's images appear both in
[`wiki/03-approximation-theory.md`](wiki/03-approximation-theory.md) and under their headings in
[`raw/slides/03-approximation-theory.md`](raw/slides/03-approximation-theory.md). All 31 of
lecture 4's images appear both in [`wiki/04-architectures-grids.md`](wiki/04-architectures-grids.md)
and under their headings in [`raw/slides/04-architectures-grids.md`](raw/slides/04-architectures-grids.md).
The concept pages embed none; they cite slides, and the lecture pages carry the pictures.

- **Prefer the transcription for numbers and formulas.** The slide file reproduces every
  equation and table as text; use the image to *show*, not to read values off.
- **Show one image, not a gallery.**
- **Keep the citation**, so the reader can find the rest of that slide in `raw/slides/`.

### What was rendered, and what was not — lecture 1

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

### What was rendered, and what was not — lecture 2

Rendered (45): slides 7, 10–22, 24–27, 30, 34, 36–38, 41–44, 46, 48–55, 60, 63, 66–68, 70, 72, 73
and 75 — the loss surface, the momentum runs and Goh's momentum article, the six-landscape quiz and
the six landscape cases, evolution strategies and gradient clipping, the ReLU and GELU plots,
computation graphs, the chain-rule shapes, every backpropagation diagram (generic layer, linear
layer, whole MLP, one-iteration graph, merge and branch, parameter sharing), the human-versus-backprop
graph, unit visualizations, DeepDream, CLIP+GAN, and the worked example's network before and after.

Not rendered, and why:

- **Excluded from OCW's licence (7 slides): 4, 57, 58, 59, 64, 65, 69** — the clown fish and
  chameleon photos, the LeCun and Dietterich posts, the Neural Module Networks figure, Karpathy's
  Software 2.0 figure, and the CLIP figure. Described in full in the slide file only.
- **Build steps superseded by a rendered slide:** 8 (slide 5 plus the update rule), 29 (slide 30
  without its inset), 39 and 40 (slide 38's diagram with equations the file reproduces), 47 (a
  condensed slide 46), 61 and 62 (earlier frames of 63).
- **No figure worth a picture:** the title, announcements, agenda and divider slides (1–3, 71,
  74); equation and text slides (5, 6, 9, 23, 28, 32, 33, 45, 76–80); the lone-parenthesis frames
  31 and 35; slide 81 (OCW end page).
- **Slide 56**: its two-box diagram is fully described in prose, and the rest of the slide is the
  PyTorch and TensorFlow logos and an uncredited screenshot of code.

### What was rendered, and what was not — lecture 3

The deck carries **no** OCW exclusion notice on any page, so nothing was withheld for licence
reasons. Rendered (18): slides 2, 5, 6, 8, 12–17, 19, 27–30, 32, 38 and 42 — the width-or-depth
networks, the staircase data set, the Weierstrass plot, the Lipschitz bow tie, the rectangle and
hyperrectangle approximations, the four-ReLU rectangle, the wall-plus-wall surface plots, the
assembled construction, the training-points-on-rectangles sketch, the kink diagrams, the layer
recursion, the triangle map and its compositions, the Kaplan et al. scaling figures, and the
audio-and-image preview.

Not rendered, and why:

- **Identical to a rendered slide:** 23 (slide 2 again, opening the width-versus-depth section).
- **Handwritten text and equations the slide file reproduces exactly:** 3, 4, 7, 9, 10, 11, 18
  (slide 10's theorem again, beside a small unlabelled scribble), 20, 21, 24–26, 31, 33–35, 37, 39
  and 41.
- **Title and dividers:** 1, 22, 36, 40; and slide 43, the OCW end page.

### What was rendered, and what was not — lecture 4

Rendered (31): slides 3, 5–8, 10, 26–32, 34, 37, 39–42, 44, 48, 49, 56–60, 73–75 and 77 — the MLP
pros-and-cons diagram, the hypothesis-space pictures, the ReLU-net, exact-model and sine-net fits,
the fully connected, locally connected, convolutional and weight-sharing layers, the dense and banded
matrices and the Toeplitz matrix, stacked convolutions, multichannel inputs and outputs, the general
filter-bank tensor diagram, the bank of two filters, the parameter quiz, max and mean pooling,
downsampling, strided and dilated filters, convolution in time, the video cube and 3D convolution, and
the positional-encoding diagram.

Not rendered, and why:

- **Excluded from OCW's licence (35 slides): 9, 12–25, 35, 43, 47, 53–55, 61–63, 66–72, 78–81** —
  the SIREN image-fitting figure, every stork and heron figure by Fredo Durand (patch classification,
  segmentation, feature maps, pooling across channels, the classification-network chain, receptive
  fields, encoder–decoder, image-to-image, U-net, ResNet), the clown fish filter and the three filtered
  photos, the neural-field and NeRF figures, and Yen-Chen Lin's generated image. Described in full in
  the slide file only.
- **Reusing an excluded image without a notice:** 50–52 (slide 25's clown fish crop; see above).
- **Thumbnails of excluded material on a text slide:** 36 (the five views, with slide 25's filtered
  clown fish and a copy of slide 29's diagram).
- **Build steps superseded by a rendered slide:** 4 (slides 5 and 6 each add to it), 33 (slide 32's
  picture without its text), 45 (slide 44's figure with the second quiz question).
- **No figure worth a picture:** the title, agenda, dividers and text slides (1, 2, 11, 38, 46, 64, 65,
  76, 82, 83) and slide 84, the OCW end page.

### Provenance and attribution

Rendered slides are from *MIT 6.7960 Deep Learning, Fall 2024*, MIT OpenCourseWare
(<https://ocw.mit.edu>), CC BY-NC-SA 4.0, by Phillip Isola, Sara Beery and Jeremy Bernstein;
lecture 1's deck is by Sara Beery, and lecture 2's names her as speaker. Two rendered lecture 1
slides contain material from elsewhere that OCW did not flag. **Slide 51** reproduces the double-descent figure of Belkin, Hsu, Ma and Mandal,
"Reconciling modern machine-learning practice and the classical bias–variance trade-off"
(PNAS, 2019), credited on the slide. **Slides 45 and 70** show, at their clipped bottom edge, a
fragment of an uncredited bird photograph from the next animation frame.

Lecture 2's rendered slides include third-party material that the slides credit under an open
licence: **slide 11** is a screenshot of Gabriel Goh's Distill article "Why Momentum Really Works"
("Courtesy of Gabriel Goh, 2017. License: CC-BY."), and **slides 66–67** show feature
visualizations from Olah et al.'s Distill article ("Courtesy of Olah, et al. Used under CC BY.").
**Slide 68** collects DeepDream images whose credit line reads "Images created using a network
trained on places by MIT Computer Science and AI Laboratory", from the Google blog post it links;
OCW did not flag it. **Slide 70** includes a small generated image as the CLIP+GAN output.

Lecture 3's deck is Jeremy Bernstein's handwritten notes, and three of its rendered slides paste in
figures from elsewhere, each credited by hand on the slide and none flagged by OCW. **Slide 6** is a
plot titled "A pathological function of Weierstrass", credited "Hrothgar, Chebfun". **Slide 16**
shows three 3D surface plots credited "Hongzhou Lin". **Slide 38** reproduces two figures from
Kaplan, McCandlish et al. (2020), captioned "From Kaplan, McCandlish et al (2020)".

Lecture 4's deck names Sara Beery as speaker. Its rendered slides are diagrams and plots; four carry
material from elsewhere that OCW did not flag. **Slide 10**'s plots are credited "[Sitzmann\*,
Martel\*, Bergman, Lindell, Wetzstein, NeurIPS 2020]" (the SIREN paper); **slides 7 and 8** use
the same plot layout and print no credit. **Slide 42** is credited "[Figure modified from Andrea
Vedaldi]". **Slides 74 and 75** show frames of an uncredited video of people walking past a stone
building, stacked into a space–time cube.

If a rights holder or the course asks for a page to come down, delete the image file and every
image embed that points at it.

## Rebuilding

Built and updated by the `cairn-kb` skill; [`TODO.md`](TODO.md) lists the remaining lectures.
Notes for the next run, from lectures 1–4:

- Download the deck and run `slide_number_map.py`. As of this build it reads bottom-centre
  numbers, which OCW decks use; before that it read axis labels as slide numbers.
- Transcripts: run the light edit described above, not a full rewrite.
- Images: render figure slides that carry **no** OCW exclusion notice. Find the notices with a
  text-layer grep for "excluded from our Creative Commons license", or the `*OCW notice` lines
  in the slide file.
- The notice text wraps across lines in the text layer, so grep a page's text with newlines
  joined, or for "All rights reserved" alone. A single-line grep for the full sentence found one of
  lecture 2's seven notices.
- `embed_slide_images.py` anchors a wiki image at the slide's **first** citation, and a range such
  as "slides 15–20" places every slide in it at that spot. Write the wiki so that each slide's first
  citation is the passage it belongs in, and avoid ranges in overview paragraphs. It matches any
  "slide N" in the page, so do not cite *another* lecture's slide numbers in a lecture page (lecture 4
  avoided writing "lecture 3's slide 42", which would have pulled in its own slide 42). It inserts after
  the citing paragraph, so a paragraph ending in a colon before a `$$` block gets the image between the
  colon and the equation; cite the slide after the equation instead.
- Hand-placed wiki images (to write a better caption) must copy the path from disk: five of lecture 4's
  first hand-placed links guessed `.png` for files that are `.jpg`. The script skips images a page
  already links, so hand-place the key figures first and let it place the rest. Its default caption is
  the slide title, which reads badly for parenthetical or question titles; rewrite those, and keep math
  out of the italic captions. It also copies the slide title into the slide file's alt text, so a
  title with math in it (lecture 4's slides 7, 8 and 10) leaves `$` in an alt text, which github.com
  then fails to parse; strip it.
- **Handwritten decks** (lecture 3; likely lectures 7 and 23 too): the text layer is OCR noise,
  so do not grep it for notices or strings; read every page, and choose figure pages from the slide
  file, since hand-drawn ink is vector paths and the raster test misses it. Script and letter
  ambiguities (ℓ against L, subscripts against superscripts) need the audit, and the lecturer
  usually reads each formula aloud, so check the formulas against the transcript too.
- **Audit with a different model from the transcriber.** A same-model audit catches misreadings
  caused by resolution but not ones the two runs share. Lecture 3 was read at Sonnet and audited at
  Opus from 600-dpi crops, and lecture 4 the same at 600 dpi; lectures 1 and 2 were Sonnet audited
  by Sonnet. Lecture 4's audit found errors on 13 of 24 pages, mostly counts (points, peaks, nodes,
  grid cells) on charts and diagrams.
- After writing, run `check_math.mjs` and `verify_kb.py`.
