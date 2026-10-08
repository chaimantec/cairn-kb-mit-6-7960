# How this knowledge base is organized

This repo is the knowledge base for **MIT 6.7960 — Deep Learning, Fall 2024** (Phillip Isola, Sara
Beery, Jeremy Bernstein), built from the course's MIT OpenCourseWare release. It is read by
Cairn's in-extension AI chat, which fetches files over raw.githubusercontent.com and follows
relative markdown links.

**Coverage is partial: lectures 1–9 of 24.** [`TODO.md`](TODO.md) is the build state.

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
  *recorded* lecture number, and check later decks for the same drift. The decks of lectures 2
  to 6 have no pointers to other lectures, and each one's own "Lecture N" matches the recording;
  lecture 6's title slide, "Lecture 6: NN Generalization", settles lecture 1's "Lecture 7" banner.
  Lecture 3's last slide previews "Inductive biases" with no lecture number; lecture 4's recording
  points ahead to "the transformers lecture" (8) without one, lecture 5's to transformers "in a
  week or so", and lecture 6's to "Thursday's lecture" on optimization (7), the representation
  learning lectures (11–13), image-to-image models "later in the course" and language models
  "later". Lecture 7's title slide, "6.7960 :: Lecture 7", settles lecture 1's "Lecture 6: Scaling Rules
  for Optimization" banner; its recording points back to lectures 3 and 6 and to the RMS norm "from my
  other lecture" (3), and ahead to implementing a transformer "later in the class" without a number.
  Lecture 8's title slide, "Lecture 8: Transformers", settles lecture 1's "Lecture 9" banner, but its
  outline (slide 2) is still headed "9. Transformers"; cite it as lecture 8. Its recording points back to
  "the GNN lecture" (5), ResNets "in the CNN lecture" (4) and "what Jeremy was talking about" (7), and
  ahead to the next problem set (Homework 3) and to autoregressive models and language models "later".
  Lecture 9's title slide, "Lecture 9: Hacker's guide to DL", and its outline, "9. Hacker's guide to DL",
  both match the recording. Its deck prints no other lecture number (slide 23 points to "the generative
  modeling lectures" without one); its recording points back to "the generalization lecture" (6), the RMS
  norm "Jeremy introduced" (3 and 7) and "Pset 1", and ahead to generative models (14–16) and to
  pre-training and transfer learning (18–19) without numbers.
- **Reused deck.** Lecture 1's title slide says "6.S898 Deep Learning … Fall 2022", the course's
  earlier number and term, and lecture 5's footer prints "6.S898 Deep Learning" with "Fall 2024".
  The transcription keeps what is printed.
- **Slide numbers are printed at bottom centre** and equal the PDF page number, so slide N is
  page N. Each deck ends with an OCW end page (page 81 in lectures 1 and 2, page 43 in lecture
  3, page 84 in lecture 4, page 47 in lecture 5, page 66 in lecture 6, page 32 in lecture 7, page 55 in lecture 8, page 72 in lecture 9), which is
  not lecture content. Lecture 5's is a 4:3 page and lecture 6's a 792×612 one, smaller than the slides,
  and `slide_number_map.py` reports each as printing no number although they print 47 and 66; it says
  the same of lecture 7's page 32, whose render shows a small "32", of lecture 8's page 55, a
  792×612 page that prints "55", and of lecture 9's page 72, a 792×612 page that prints "72".
- **Lecture 3's deck is handwritten** — Jeremy Bernstein's iPad notes, in several ink colours on a
  dark background. Its PDF text layer is OCR of the handwriting and is useless (it reads
  "Hongenoucin" for a handwritten credit), so nothing was taken from it. The handwriting is vector
  ink, not a raster, so the raster-coverage test for figure pages finds almost nothing on such a
  deck; the slide file's descriptions decided which pages were rendered. Lecture 7's deck is handwritten
  the same way, and fails the raster test in the opposite direction: every page paints the same
  512×512 one-channel image twice, side by side, as its background (dark on most pages, white on the
  title, dividers, references and slides 5 and 16–18), so every page covers about twice its area in
  raster and every page "has a figure". Its typed OCW notices are in the text layer and were found by
  grep; its handwriting OCR is noise ("Selond order"). Expect the same of lecture 23, Bernstein's other
  lecture on the schedule.
- **OCW excludes some figures from its licence**, with a notice on the slide: "© … All rights
  reserved. This content is excluded from our Creative Commons license." Those slides are
  transcribed with an `*OCW notice: …*` line and are **never rendered into `raw/images/`**.
  Lecture 4's deck carries 35 such notices in 84 pages, among them every figure built on Fredo
  Durand's stork and heron photographs (patch classification, segmentation, feature maps, receptive
  fields, encoder–decoder, U-net, ResNet), so those figures exist here only as prose. Lecture 5's
  carries 14 in 47 pages: the application examples (Pinterest, molecules, polypharmacy, Google Maps,
  learned physics simulation), J. Leskovec's tree-view illustrations, the two example
  architectures, and the positional-encoding results. Lecture 6's carries 8 in 66 pages: the Paul
  the octopus article (The Daily Beast), the pix2pix training pairs and network (Xie and Tu; Isola
  et al.), Chris Hesse's edges2cats demo, all four Belkin et al. figures (the bias-variance U-curve,
  double descent, the MNIST result and the random-Fourier-feature norms), and slide 63's NeRF and
  polypharmacy figures ("© sources unknown"). Lecture 7's carries 3 in 32 pages: slide 7's two plots from a
  post by @kellerjordan0 on X ("optimal learning rate drifts" and "deeper performs worse"), and slides 25
  and 26, which paste one of those plots each beside the lecturer's handwritten width claim and depth
  recipe. Those two slides are not rendered either; their handwriting is transcribed in full.
  Lecture 8's carries 7 in 55 pages: Fredo Durand's bird photograph on slides 4, 6, 7 and 8 (the CNN
  limitation and the three attention examples), the DINO attention maps on slide 31 ("© AI at Meta"),
  the ScaleMAE figure on slide 45 ("© Reed, et al."), and slide 53's "Attention Is All You Need" figure
  ("© Vaswani, et al."), whose notice sits under the architecture diagram and is taken to cover the
  screenshot of the paper's first page beside it.
  Lecture 9's carries 21 in 72 pages: the Zhang et al. paper and fast.ai screenshots (slide 3), DeGrave, Janizek
  and Lee's chest X-ray (4), the coffee-cup photos loaded by DeCAF and Caffe (7), the clown fish, grizzly and
  chameleon augmentation photos (13), the domain-randomization images and OpenAI's robot-hand table, renders
  and chart (16–18, "© Tobin, et al." and "© OpenAI, Tobin, et al."), the Stable Diffusion and AlphaFold
  screenshots (28), every colorization photograph (30–33 and 35–39, "© Zhang, Isola, and Efros" and "© source
  unknown"), a facade-generation grid (58, "© IEEE"), the Weights & Biases site (60), the spice rack (61) and the
  Scale ML web page (71). Its slide 4's notice sits beside the X-ray only; see the next item for slide 57.
- **An excluded image can reappear without its notice.** Lecture 4's slides 50–52 print no notice,
  but their photo crop is the same embedded image object as slide 25's clown fish, which slide 25
  marks "© source unknown. All rights reserved". They are treated as excluded and not rendered.
  Before rendering, compare each candidate page's image xrefs (`page.get_images()`) against the
  excluded pages'; for lectures 1, 2 and 4 the rendered set shares none. **A shared xref is not
  always a shared figure, though.** Lecture 5's slides 15, 21 and 29 share one image object with
  excluded slide 3, but it is a 33×31 px node circle used 19 times to draw a graph. The figure audit
  placed slide 3's credit "(illustration: J. Leskovec)" and its notice under the slide's Pinterest
  figure, not under that graph, which the deck draws itself and reuses with no notice on slides 12,
  15, 21 and 29. Those four are rendered. Check an image's size and placement
  (`page.get_image_rects()`) before treating a page as reusing excluded content.
- **Reuse crosses decks, and xrefs do not.** An xref is local to one PDF, so compare images across
  decks by a hash of their pixels (`hashlib.md5(fitz.Pixmap(doc, xref).samples)`). Lecture 6 found
  two such reuses. Its slide 62 prints no notice, but its beak photograph is the same 800×533 image
  as lecture 4's excluded slides 43, 53–55, 66–68, 70 and 71 (Fredo Durand's heron), mirrored and
  cropped to its beak, so slide 62 is not rendered. And its excluded slide 30, "© Belkin, et al. All rights reserved", is pixel for
  pixel the double-descent figure of lecture 1's slide 51, which carries no notice in lecture 1 and
  had been rendered; that image was withdrawn when lecture 6 was added. A sweep of every rendered
  image in lectures 1–6 against every excluded image in all six decks found no other match. Lecture
  7's embedded images, hashed against all images in decks 1–6, match only once: its title slide's logo is
  lecture 3's, which carries no notice. Slides 25 and 26's plots are different image objects from slide
  7's, each under its own notice. Lecture 8's bird photograph (slides 4, 6–8) is pixel for pixel lecture 4's
  excluded stork image; those four slides carry their own notices. Every other lecture 8 image was
  hashed against every excluded image in decks 1–8, exactly and with a coarse 16×16 perceptual hash
  (mirrored too), and none matched. Slide 47's Laplacian-eigenvector figures come from the same paper
  (Kreuzer et al.) as lecture 5's excluded slide 43 but are different figures, and lecture 8 prints them
  "Courtesy of Kreuzer, et al. Used under CC BY-NC-SA.", so slide 47 is rendered. Lecture 9's slide 57 prints no
  notice and shares two image objects with its own slide 4 (the spiky loss plot and the grid of generated
  samples), plus a near-identical copy of slide 4's creature-simulation frame. Slide 4's notice, "© DeGrave,
  Janizek, and Lee", sits beside its chest X-ray, a separate image object that slide 57 does not carry, so
  slide 57 is rendered. Every other unnoticed lecture 9 image was hashed against every excluded image in decks
  1–9, exactly and with the coarse perceptual hash, and none matched. The coarse hash needs one guard: a
  near-uniform image (a mask, a blank panel) hashes to almost all zeros and "matches" every other such image,
  so skip hashes with fewer than 8 or more than 248 bits set. One remaining coarse match, lecture 9's creature
  frame against a 120×124 crop on lecture 4's slide 25, differs by a mean of 90 grey levels in 256 and is not
  the same image.

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
"ladder"). The edit was small enough to do inline rather than through a subagent. Lecture 5: 105
markers identical, all 34 digit strings identical, word ratios 0.99–1.09 (the high end is a short
paragraph carrying an `[Ed: see slide 40]` note); restorations are "Sarah" → "Sara" four times,
"a graph that can operate" → "a graph net can operate", two "graph. Net" sentence breaks, and "some"
→ "sum" three times where the lecturer reads slide 40–41's sum of MLPs, plus four punctuation fixes
and seven `[Ed: …]` notes (two naming Sara Beery and Jeremy Bernstein, four on unclear phrasing,
one pointing a restoration to slide 40).
Also done inline. Lecture 6: 104 markers identical, all 86 digit strings identical, word ratios
1.00–1.27, and 1.00 throughout once its two `[Ed: …]` notes are removed (the high end is the 16:11
paragraph carrying the note on slide 10's printed n and m). Its restorations are the student's
"approximations supposed" → "approximation's supposed" (5:24) and "Sarah" → "Sara" (1:17:40, with an
`[Ed: …]` note naming Sara Beery), plus two restored full stops, two lower-cased words and one
restart marked with the captions' "--". Also done inline. Lecture 7: 104 markers identical, all 64 digit
strings identical, word ratios 1.00–1.20, and 1.00–1.01 once its `[Ed: …]` notes are removed (the high end
is the 17:59 paragraph, whose note sets the spoken "up to first order" beside slide 11's "to
second-order"). Its restorations are "sine" → "sign" five times (slide 18's sign gradient descent),
"paths" → "pairs" (3:51), "lost" → "loss" (10:04), "matrix n" → "matrix M" (1:01:37), "operating norm" →
"operator norm" (1:03:10) and a student's "code" → "claim" (1:07:07, slide 25's "claim"), plus two restored
full stops and six `[Ed: …]` notes (naming Phillip Isola, identifying "my other lecture" as lecture 3, and
four on unclear or unexpected wording). Also done inline. Lecture 8: 96 markers identical, all 50 digit
strings identical, word ratios 1.00–1.11, and 1.00–1.01 once its `[Ed: …]` notes are removed (the high end
is the 55:04 paragraph, whose note sets the spoken "queries and values" beside slides 29 and 34's queries
and keys). Its restorations are "X7" → "x7" (4:38, slide 5), "combined function" → "combine function"
(22:24, slide 22's COMBINE) and "Noem" → "Noam" Chomsky (1:07:25), plus three restored full stops, two
possessive apostrophes turned into commas ("the keys, queries and values") and five `[Ed: …]` notes (naming
Jeremy Bernstein; a student's unclear "human case", probably "Q and K"; and three places where the spoken
wording differs from the slide: "queries and values", "dividing by the variance" and "permutation
invariant"). Also done inline. Lecture 9: 98 markers identical, all 95 digit strings identical, word ratios 1.00–1.22,
and 1.00–1.01 once its `[Ed: …]` notes are removed (the high end is the 4:44 paragraph, whose note sets the spoken
"breast cancer" beside slide 4's chest X-ray). Its restorations are "N-gram error" → "N-gram era" (44:51), "into
access possible" → "into x as possible" (55:40), a student's "CNs" → "CNNs" (1:01:06) and "such landscape" →
"search landscape" (1:14:09), plus five punctuation fixes, a question mark and one lower-cased word, and seven
`[Ed: …]` notes (naming Jeremy Bernstein, Richard Zhang and Antoine de Saint Exupéry; two unclear phrases; and two
places where the spoken wording differs from the slide, "breast cancer" and "divide by the variance"). Also done
inline.

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
$\mathbf{x}_ {\text{out}}[C, :]$ on the left where its right side, indexed from 0, means $C - 1$;
slide 39's multichannel formula ends in $+ b[c]$ with no brackets, so whether the bias is inside the sum
is unclear; slides 48–49 use $j$ both as the output index and as the index over the window
$\mathcal{N}(j)$; slide 3 says "Embarassingly"; slide 19 says "not easy to recognize content in small
each patch"; slide 46 says "an filter"; and slide 72 repeats slide 69 unchanged. Slide 25's filter
detects vertical edges, while the lecturer calls them "horizontal edges" (≈23:05); the wiki reports
both.

**Lecture 5's figure audit**, cross-model again: Sonnet read the deck, and Opus checked 27 graph-,
chart- and equation-heavy pages (3, 5, 8–13, 15–20, 22, 23, 27, 28, 30, 35–37, 39–41, 43, 44) from
250–600 dpi crops, counting nodes, edges and bars from the PDF's vector data (`page.get_drawings()`)
where that was more reliable than eyeballing. Every equation agreed. Corrections followed on 11
pages: counts on slides 8 (nine red dots, not eight), 9 (16 edges, two unlabelled) and 41 (eight
curves, not "roughly seven", with two green series, and where each callout's arrow ends); arrow ends
and directions on 16 and 43; blob membership on 12 and 15; dot positions on 10; the boxed region on
30; the notice's position on 3; and slide 18's missing closing brace. Slide 36 agreed, and its second
equivalence class is now described more fully (a triangular prism and $K_{3,3}$). It also established that slides 3, 12, 15, 21 and 29 share one 19-node, 24-edge graph
drawing (above). The renders of slides 36 and 41 were then checked against the corrected text.

**Lecture 6's figure audit**, cross-model: Sonnet read the deck, and Opus checked 30 chart-, diagram-,
table- and equation-heavy pages (3, 8, 10, 11, 19, 25–28, 30–32, 34, 37–39, 44, 49–54, 56–60, 62, 63)
from 150–600 dpi crops, measuring curves, dots and circles from `page.get_drawings()` and charts from
their embedded rasters. Every equation agreed, and every cell of slide 44's CIFAR10 table. Corrections
followed on 16 pages, mostly where things sit: slide 8 (stems on nine dots, not ten; the green
arrows land between the dots), 11 (eight blueberries), 25 (where the four off-curve samples are), 27
and 28 (the model's swings and spikes; slide 28's 17th sample gets no spike), 31 and 32 (both are
log-scaled, and the x positions of their markers), 37 and 39 (which circles are purple), 50, 51, 57,
58, 60, 62 (24 edges, the blob, the bar angles) and 63 (the layout, and that the notice sits under
the NeRF figure). It also identified slide 62's photograph as lecture 4's excluded heron (above).
The renders of slides 8 and 28 were then checked against the corrected text.

**Lecture 5's printed slips and oddities**, transcribed as written: slide 18's max aggregation has
no closing brace; slide 30's polypharmacy update writes the first normalizing constant $c^{vu}$
with no $r$ while the second is $c_r^v$; slides 40 and 41 print $g_1$ with no opening bracket before
the sum, only the closing one; slide 39's theorem writes a plain $h^{(t)}$, not bold and with no
node subscript; and slide 39 abbreviates the "Jegelka" that slide 36 prints in full as "J"
("Xu-Hu-Leskovec-J 19"), while slide 43's citation ends the same way
("Lim-Robinson-Zhao-Smidt-Sra-Maron-J 22"). Slide 19's GNN line marks "sum or
max pooling" where the lecturer says Bellman-Ford's aggregate is a min (≈41:54); the wiki reports
both.

**Lecture 6's printed slips and oddities**, transcribed as written: slides 10 and 11 print "n = 30,
m = 10" for a 30-fruit vocabulary and 10-fruit lists, the reverse of slide 9's definitions (n words
from a vocabulary of size m), and slide 10's "s = 100 trillion" at p = 1 equals neither $10^{30}$ nor
$30^{10}$ (it is close to $30 \times 29 \times \cdots \times 21$, ordered lists of ten different fruits);
the lecturer says "10 to the 30th" (≈16:11), and the wiki gives all of it without resolving it.
Slide 9 asks "What about much big fancy modern nets?" and writes m^n and p\*m^n as plain text, as
slide 43 does d = 2^n; slide 50 says "weights in biases"; and slide 44's screenshot of Zhang et al.'s
table clips the caption at its right edge. Slides 41 and 43 define the "VC dimension" $d$ as the
number of dichotomies; the wiki follows the slides and adds one marked note that textbook usage
differs. The lecturer says Paul the octopus got "6 out of 6" (≈9:16) where slide 7's article says
"all eight (!) German matches"; the wiki reports both.

**Lecture 7's figure audit**, cross-model: Sonnet read the handwritten deck, and Opus checked 24 pages
(1, 4–7, 10–19, 22–26, 28–31) from 150–800 dpi crops, the vector ink (`page.get_drawings()`, counting dots
by connected components) and the native rasters of slide 7's two plots. Every formula agreed, and every
reading the transcriber had flagged was settled: slide 4's error symbol is a lowercase ℓ; slide 11's
coefficient is $\lambda/2$ and slide 18's subscript a 1, both also confirmed by the recording; slide 19's
superscript is a dagger; slides 23–24's spectral-norm subscript is $\ast$, with $d_{\text{in}}$ on top in
slide 24's factor. Corrections followed on 15 pages: slide 7's left plot (width 32's minimum is half-way
across in the lower third, the right branches are near-parallel and cross low down) and slide 25, which
embeds the same image; dot counts on slides 6 (3, 10, 7, 4, 1) and 22 (2, 4, 1); slide 14's "stray mark",
which is the opening bracket's own top bar, not a transpose; arrow directions on 13 and 16; which pages
are white (1, 5, 16–18); slide 19's "right had side" (below); and colours on 1, 4, 11, 12, 14, 17, 18, 19
and 26. Slide 31's fifth and sixth book icons are drawn but hidden under its lilac blob. The renders of
slides 22 and 24 were checked against the text; slide 22's dot count was caught there too.

**Lecture 7's printed slips and oddities**, transcribed as written: slide 1 prints the email address
"jbernstein@mit.edub", as lecture 3's did; slide 10 writes the quadratic term with no transpose on the first
$\Delta w$, where slides 11 and 16 have one; slide 11's boxed Newton step $\Delta w = -H^{-1} g$ drops the
$\lambda$ that its own previous line carries (the wiki notes it is the $\lambda = 1$ case); slides 13–15
write the Gauss-Newton product as two identical $\partial f / \partial w$ factors with no transpose; slide
19 says "right had side" and prints its dual formulation with no minus sign, which problem set 2's statement
of the same identity has; and slide 6 writes $f(x; w)$ with a semicolon where slide 4 has a comma. In the
recording, the lecturer says "up to first order" for slide 11's second-order expansion (≈17:59), says
depth increases "as I go down" the curves of slide 7's depth plot, the reverse of the plot (≈10:04), and
says $(1 + x/L^2)^L$ tends to 0 (≈11:01); the wiki reports each, the last with one note marked as outside
the course material that the limit is 1.

**Lecture 8's figure audit**, cross-model: Sonnet read the deck, and Opus checked 33 diagram-, equation-,
code- and photo-heavy pages (4, 5, 11, 13–17, 19, 21–24, 26, 28–31, 34–36, 38, 39, 41–44, 46, 47, 49, 51, 52,
54) from 150–600 dpi crops, counting circles, boxes and arrows from `page.get_drawings()` and nodes and edges
in slide 24's raster by connected components. Every equation agreed, and slide 39's code character for
character. Corrections followed on 19 pages, mostly counts and where arrows go: slides 4 (ten birds), 13
(ten circles), 24 (eleven nodes, not ten, and a pixel check finds one of the 55 pairs unjoined), 35 (the
conv wiring has seven and six circles and a 6 × 7 matrix, not eight and seven and a square; the attn graph's
pink and orange arrows were swapped), 34 (two arrows' directions), 54 (the decoder's columns sit one to the
right of the input words, so its first causal layer goes strictly forward, not "same-or-later"), 23, 28,
29, 30, 49, 51 and 52 (where arrows, labels, rules and "=" sit; slide 29 prints no "×"), 47 (the right
molecule has five rings), 38, 42 and 46 (a font, a shade, where titles sit, and a gloss from outside the
slide removed). It settled every reading the transcriber had flagged: slide 31's frames are one image with
no attention overlay drawn on it, slide 42's w is a bold lowercase, and slide 44's $\mathbf{p}$ has ten
cells. The renders of slides 35 and 54 were then checked against the corrected text.

**Lecture 8's printed slips and oddities**, transcribed as written: slide 2's outline is headed "9.
Transformers" (above); slide 9 calls the third idea "Positional Codes" where slide 2 says "Positional
encoding"; slide 29's four key-query scores (1, 0.2, 0.9, 0.1) do not match the attention weights printed
above the values (0.1, 0.2, 1, 0.1), which sum to 1.4, and the lecturer calls the first score a typo for 0.1
(≈39:27); slide 29 writes $\mathbf{q}^{T}$ with an italic T where its matrices use an upright one;
slides 26–28 label the output a plain $t_{\text{out}}$ beside a bold $\mathbf{t}_ {\text{in}}$; slide 31
cites "Caron et all. 2021"; slide 34 scales by $\sqrt{m}$ while its diagram labels the dimension $M$;
slide 37 sizes $\bar{\mathbf{T}}_ {\text{out}}$ as $N \times kv$ with no $v$ defined; slide 39's code uses
`K` for both the patch size and the key matrix and divides by `sqrt(d)` outside `nn.softmax(…, dim=0)`,
where slide 34 divides inside; and slides 11–13 are titled "A New Data Type" and then "A new data
structure". In the recording the lecturer says attention is "permutation invariant" where slide 41 says
equivariant (≈1:02:02), that layer norm divides "by the variance" where slide 38 divides by its square root
(≈57:23), and that the attention weights "are coming from queries and values" (≈55:04); the wiki reports
each beside the slide.

**Lecture 9's figure audit**, cross-model: Sonnet read the deck, and Opus checked 34 figure-, chart-, table-,
equation- and code-heavy pages (4, 7, 9, 10, 12–18, 20, 22–24, 27, 28, 30, 31, 34, 35, 38, 39, 43, 47, 50, 51, 54, 57,
58, 60, 61, 68, 71) from 60–600 dpi crops, the embedded rasters at native resolution, the vector data and the text
layer. Every equation and code line agreed, every cell of the tables on slides 18, 51 and 71, and every word of slide
43's screenshot of the lecturer's batch-norm text. Corrections followed on 13 pages, mostly counts and positions: six
spikes and four-limbed creatures on slides 4 and 57, 39 blue and 7 red circles on slide 14 (from the vector data), a
15 × 15 patch on slides 38–39, where slide 50's two leader lines end, two red cubes and three green pyramids in slide
16's photograph, slide 18's log axis (it starts at 0.5) and blue curve, slide 13's two crops (the same size, shifted),
and the black frame that swaps with the lock on slide 20. Slide 30's first reading described a spoon that is not on
the page and a tail fin that the training-data box hides. Slide 58's column headings were first explained as an
old-style font; they are live text that prints "Ll", "Olayers", "llayers" and "61ayers". The renders of slides 10 and
14 were checked against the corrected text.

**Lecture 9's printed slips and oddities**, transcribed as written: slide 10's layernorm mean prints $1/k$ in front of
$\sum_k$, with $k$ as both normalizer and index; slide 12's `reshape(X, (X.shape(0)*X.shape(1))` and slide 30's
argmin formula have unbalanced parentheses; slide 54 leaves a blank where the assignment sign of its EMA update goes
(the audit found no glyph, drawing or image in the gap); slide 17 says "where we actual use our model", slide 20
"then other way around", slide 27 "if you have text problem", slide 41 "an numerical measure" and "an standard
architecture", and slide 71's screenshot "inquires" and "LLMse"; slide 58's headings swap letters and digits
("Ll", "Olayers", "61ayers"); and slide 70 writes `torch.cudnn.benchmark`, which the wiki gives as printed with one
note marked as outside the course material. Slide 49 prints the log-loss reference values as log probabilities,
such as $-0.69 = \ln(0.5)$, and the wiki adds that the course's cross-entropy at chance is their negative. In the recording the
lecturer calls slide 4's chest X-ray example a breast-cancer classifier (≈4:44) where the slide names no disease, and
says "divide by the variance" (≈16:24) where slide 9 divides by its square root; the wiki gives both, the first with
one note marked as outside the course material on what the cited paper studies.

## Images

**Lectures 1–9 have images; no other lecture does yet.** They are committed rather than hotlinked,
and they are the only part of this KB that redistributes course material rather than describing
it.

| Lecture | Files | Where they came from |
| --- | --- | --- |
| 1 Introduction to Deep Learning | 20 of 81 pages | rendered from `mit6_7960_f24_lec1.pdf` |
| 2 How to Train a Neural Net | 45 of 81 pages | rendered from `mit6_7960_f24_lec2.pdf` |
| 3 Approximation Theory | 18 of 43 pages | rendered from `mit6_7960_f24_lec3.pdf` |
| 4 Architectures: Grids | 31 of 84 pages | rendered from `mit6_7960_f24_lec4.pdf` |
| 5 Architectures: Graphs | 19 of 47 pages | rendered from `mit6_7960_f24_lec5.pdf` |
| 6 Generalization Theory | 28 of 66 pages | rendered from `mit6_7960_f24_lec6.pdf` |
| 7 Scaling Rules for Optimization | 9 of 32 pages | rendered from `mit6_7960_f24_lec7.pdf` |
| 8 Architectures: Transformers | 34 of 55 pages | rendered from `mit6_7960_f24_lec8.pdf` |
| 9 Hacker's Guide to Deep Learning | 11 of 72 pages | rendered from `mit6_7960_f24_lec9.pdf` |

Each is a whole slide at 1400px, JPEG q85 or PNG, whichever is smaller, named `slide-N` by
PDF page number.

### Using them

**Use an image path you have actually read in a file. Never construct one from the pattern, and
never assume a slide has an image because a neighbouring one does.** Many pages of each deck were
deliberately not rendered (next section), so lecture 1's `slide-36.jpg` existing tells you nothing
about `slide-37`, lecture 2's `slide-38` tells you nothing about `slide-39`, lecture 3's
`slide-17` tells you nothing about `slide-18`, lecture 4's `slide-49` tells you nothing about
`slide-50`, lecture 5's `slide-23` tells you nothing about `slide-24`, lecture 6's `slide-28` tells
you nothing about `slide-29`, lecture 7's `slide-18` tells you nothing about `slide-19`, lecture 8's `slide-17` tells you nothing about
`slide-18`, and lecture 9's `slide-24` tells you nothing about `slide-25`. The extension differs from slide to slide too (`.jpg` or `.png`, whichever was smaller), so
copy the whole path. Reading a path that is not in the repo returns an error rather than a URL, which
costs a turn; a guessed path is never worth it.

Links are **relative**, like every other link here: `../raw/images/01-introduction/slide-41.png`
from `wiki/`, `../images/01-introduction/slide-41.png` from `raw/slides/`. To show one, read the
path and use the URL that comes back. Do not write an absolute `raw.githubusercontent.com` URL
into a file.

All 20 of lecture 1's images appear both in
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
All 19 of lecture 5's images appear both in [`wiki/05-architectures-graphs.md`](wiki/05-architectures-graphs.md)
and under their headings in [`raw/slides/05-architectures-graphs.md`](raw/slides/05-architectures-graphs.md).
All 28 of lecture 6's images appear both in [`wiki/06-generalization-theory.md`](wiki/06-generalization-theory.md)
and under their headings in [`raw/slides/06-generalization-theory.md`](raw/slides/06-generalization-theory.md).
All 9 of lecture 7's images appear both in [`wiki/07-scaling-rules-for-optimization.md`](wiki/07-scaling-rules-for-optimization.md)
and under their headings in [`raw/slides/07-scaling-rules-for-optimization.md`](raw/slides/07-scaling-rules-for-optimization.md).
All 34 of lecture 8's images appear both in [`wiki/08-architectures-transformers.md`](wiki/08-architectures-transformers.md)
and under their headings in [`raw/slides/08-architectures-transformers.md`](raw/slides/08-architectures-transformers.md).
All 11 of lecture 9's images appear both in [`wiki/09-hackers-guide-to-deep-learning.md`](wiki/09-hackers-guide-to-deep-learning.md)
and under their headings in [`raw/slides/09-hackers-guide-to-deep-learning.md`](raw/slides/09-hackers-guide-to-deep-learning.md).
The concept pages embed none; they cite slides, and the lecture pages carry the pictures.

- **Prefer the transcription for numbers and formulas.** The slide file reproduces every
  equation and table as text; use the image to *show*, not to read values off.
- **Show one image, not a gallery.**
- **Keep the citation**, so the reader can find the rest of that slide in `raw/slides/`.

### What was rendered, and what was not — lecture 1

Rendered (20): slides 14, 21, 22, 23, 28, 33, 34, 35, 36, 38, 40, 41, 42, 43, 45, 55, 57, 58, 59,
70 — the XOR plot, the enthusiasm curves, the loss surface, the linear-layer and perceptron
diagrams, the three activation plots, stacked layers, the two-layer classification network, the
classifier and loss diagrams, cross-entropy, and the start of the batched build.

Not rendered, and why:

- **Excluded from OCW's licence (22 slides): 1, 8, 9, 11, 13, 16, 19, 20, 54, 60, 61, 62, 63,
  64, 65, 66, 68, 69, 72, 73, 75, 77.** Each carries an "All rights reserved" notice — AlexNet
  and LeNet figures, photos and book covers, the clown fish / bear / chameleon photos, the Serre
  and Donahue hierarchy figures, and others. OCW includes them under its own fair-use
  determination, which does not pass to copies. They are described in full in the slide file, and
  that description is the only representation this KB has. The bar-chart slides 61 and 63 among
  them had a figure audit.
- **Reusing an excluded image without a notice:** 51, the Belkin et al. double-descent figure, the
  same image as lecture 6's excluded slide 30 (see above). It was rendered until lecture 6 was added.
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

### What was rendered, and what was not — lecture 5

Rendered (19): slides 9–13, 15, 16, 19–23, 29, 35, 36, 39, 41, 42 and 44 — the shortest-path and
linear-program graphs, the two goals (node and graph embeddings), the adjacency matrix and its
permutation, the grid-versus-graph comparison, a CNN as a GNN over a grid graph, the two-step GNN
idea, the message-passing diagram, Bellman-Ford beside a GNN, the learned aggregation and update,
the readout, message passing unrolled, the MLP as a one-node GNN, the training data points, the
equivalence classes and their theorem, the neighbourhood trees and the two indistinguishable pairs,
color refinement beside a GNN, the PROTEINS training-accuracy chart, the structural-properties
lemma, and the positional-encoding diagram.

Not rendered, and why:

- **Excluded from OCW's licence (14 slides): 3–8, 24–28, 30, 31, 43** — the node-classification and
  Pinterest example, the molecule and Cell cover, the polypharmacy network, the Google Maps flow
  diagram and world map, the learned-simulator figure, the generalizations slide with its copy of the
  polypharmacy network, J. Leskovec's tree-view and weight-sharing illustrations, the polypharmacy and
  Google Maps architectures, and the Laplacian-eigenvector panels and bar chart. Described in full in
  the slide file only.
- **Build steps superseded by a rendered slide:** 34 (slide 35 without its theorem), 37 and 38
  (slide 39 without the GNN line and the theorem).
- **Equations the slide file reproduces exactly:** 17, 18, 40 and 46.
- **No figure worth a picture:** the title, roadmap, connections and summary slides (1, 2, 14, 32, 33,
  45) and slide 47, the OCW end page.

### What was rendered, and what was not — lecture 6

Rendered (28): slides 8, 11, 14–19, 21, 25–28, 34, 37–39, 49–54 and 56–60 — the filing cabinet
against a ReLU MLP, the image-generation counting experiment, the edges2cats creations and the
two-, three- and eight-eyed cat sequence, the oval-to-eye ConvNet diagram, Occam's razor, the four
polynomial fits, the two networks of the parameter-count example, the three VC-theory circle
diagrams, the version space, the simplicity-bias plot and diagram, the low-rank-bias scatter plots
and kernels, the parameter-kernel map, the effective-rank distributions and the parameter landscape.

Not rendered, and why:

- **Excluded from OCW's licence (8 slides): 7, 12, 13, 23, 30, 31, 32, 63** — the Paul the octopus
  article, the pix2pix training pairs, the edges2cats demo, the four Belkin et al. figures (the
  bias-variance U-curve, double descent, the MNIST result, the random-Fourier-feature norms), and the
  NeRF and polypharmacy figures. Described in full in the slide file only.
- **Reusing an excluded image without a notice:** 62 (lecture 4's heron photograph; see above).
- **Build steps superseded by a rendered slide:** 29 (slide 28's plot reduced, beside the
  "simple + spiky" text the file reproduces), 55 (slide 54's figure with a new caption line).
- **Code, screenshots and tables the slide file reproduces exactly:** 6 and 9 (the filing-cabinet
  code), 10 (the fruit list and a text-only ChatGPT exchange), 40 and 42 (the dichotomy and label
  tables), 44 (Zhang et al.'s CIFAR10 table, transcribed cell by cell).
- **No figure worth a picture:** the title, outline, dividers and text slides (1–5, 20, 22, 24, 33,
  35, 36, 41, 43, 45–48, 61, 64, 65) and slide 66, the OCW end page.

### What was rendered, and what was not — lecture 7

The raster test cannot choose here (every page carries the background tile; see above), so the slide
file's descriptions did. Rendered (9): slides 5, 6, 12, 17, 18, 22, 23, 24 and 28 — the loss curve with
"START HERE" and "FIND THIS POINT", the network beside what makes optimization hard, the gradient and
Hessian drawn as a $d \times 1$ rectangle and a $d \times d$ square, the Euclidean and infinity balls with
their steepest-descent results, the neural, tensor and spectral perspectives, the spectral-norm and
RMS-RMS operator-norm cartoons, and the module diagram.

Not rendered, and why:

- **Excluded from OCW's licence (3 slides): 7, 25, 26** — the two plots from a post by @kellerjordan0 on
  X, and the two slides that paste them again beside the width claim and the depth recipe. Described in
  full, with all their handwriting, in the slide file only.
- **Handwritten text and equations the slide file reproduces exactly:** 2, 3, 4, 9, 10, 11, 13, 14, 15,
  16, 19, 21, 29 and 30. Slide 14's only embedded image is a small upside-down smiley; slide 15's red
  cross through the curvature-of-the-model term is described in prose.
- **Title, dividers and references:** 1, 8, 20, 27 and 31 (four handwritten references beside book
  icons); and slide 32, the OCW end page.

### What was rendered, and what was not — lecture 8

Rendered (34): slides 5, 11–17, 19, 21–24, 26, 28–30, 33–36, 38, 41–44, 46–52 and 54 — the CNN
receptive-field diagram, arrays and sets of neurons and tokens, tokenizing images, text and audio, the token
notation, linear combinations of neurons and tokens, the token-wise nonlinearity, the neural net, token net
and GNN, the unrolled GNN, the fully connected graph, the fc and attention layers, the animal-counting and
impala-colour attention, query-key-value attention, self-attention, the self-attention layer and its
expansion, the family of linear layers, the MLP beside the vanilla transformer, the ViT block, permutation
equivariance, positional encoding for filters and for tokens, Fourier positional codes, the spherical-harmonic
and Laplacian encodings, autoregressive models, training and sampling, GPT, its causal mask over one and two
layers, and the image-to-text architecture.

Not rendered, and why:

- **Excluded from OCW's licence (7 slides): 4, 6, 7, 8, 31, 45, 53** — Fredo Durand's bird photograph (the
  CNN limitation and the three attention examples), the DINO attention maps, the ScaleMAE figure, and the
  "Attention Is All You Need" figure with the paper's first page. Described in full in the slide file only.
- **Build steps superseded by a rendered slide:** 20 (slide 21's neural net without the token net), 27
  (slide 28's photograph and first question, before the attention is drawn), 32 (the fc layer that slide 33
  redraws with attention, and that slide 26 already shows beside it).
- **Text, equations and code the slide file reproduces exactly:** 9 (the three ideas), 18 (the token-wise
  nonlinearity's two equations, which slide 19 repeats beside its diagram), 37 (multihead self-attention)
  and 39 (the pseudocode).
- **Title, outline, dividers and end page:** 1, 2, 3 ("*Don Quixote* by Pierre Menard" alone), 10, 25, 40
  and slide 55, the OCW end page.

### What was rendered, and what was not — lecture 9

Rendered (11): slides 10, 14, 15, 19–24, 34 and 57 — the RMS-norm and layernorm diagrams in two dimensions,
the data-space picture of augmentation, the three training curves, the fixed-data and fixed-learner pipelines,
the three next-word prompts, the drug-effectiveness $X$ and $Y$, the diagram of uncertainty over $Y$, StyleGAN2
against DALL-E, the colour gamut quantized into classes, and the three output pictures (loss curve, generated
samples, creature simulation).

Not rendered, and why:

- **Excluded from OCW's licence (21 slides): 3, 4, 7, 13, 16, 17, 18, 28, 30–33, 35–39, 58, 60, 61, 71** — listed
  under "OCW excludes some figures" above. Described in full in the slide file only.
- **Text, equations, code and screenshots the slide file reproduces exactly:** 2, 5, 6, 8, 9, 11, 12 (with a
  code-cell screenshot and the einops logo), 25–27 (27 with code screenshots and the PyTorch and Hugging Face
  logos), 29, 40–56 (43 is a screenshot of the lecturer's batch-norm text, 46–47 add assistant logos and a code
  screenshot, 51 a table transcribed cell by cell), 59 and 62–70 (68's small creature frame is a variant of the
  one rendered on slide 57).
- **Title and end page:** 1 and slide 72, the OCW end page.

### Provenance and attribution

Rendered slides are from *MIT 6.7960 Deep Learning, Fall 2024*, MIT OpenCourseWare
(<https://ocw.mit.edu>), CC BY-NC-SA 4.0, by Phillip Isola, Sara Beery and Jeremy Bernstein;
lecture 1's deck is by Sara Beery, and lecture 2's names her as speaker. Two rendered lecture 1
slides contain material from elsewhere that OCW did not flag: **slides 45 and 70** show, at their
clipped bottom edge, a fragment of an uncredited bird photograph from the next animation frame.
Lecture 1's slide 51, which reproduces the double-descent figure of Belkin, Hsu, Ma and Mandal
(PNAS, 2019), was rendered until lecture 6's deck marked the same image "All rights reserved"; it
is no longer rendered.

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

Lecture 5's deck names Phillip Isola as speaker. Its rendered slides are diagrams, equations and
plots, and some paste in figures that OCW did not flag. **Slide 41**'s PROTEINS training-accuracy
chart is a crop of a wider published multi-panel figure and prints no credit; the theorem it
illustrates is cited on slides 36 and 39 to Xu, Hu, Leskovec and Jegelka (2019). **Slides 35 and
36** paste in small pictures of graphs beside theorems credited to their authors. **Slide 42** shows
an uncredited chemical-structure drawing. **Slides 12, 15, 21 and 29** reuse the 19-node graph of
excluded slide 3 (see "An excluded image can reappear without its notice" above for why they are
rendered).

Lecture 6's deck names Phillip Isola as speaker. Several of its rendered slides carry material from
elsewhere that OCW did not flag. **Slide 11** is a ChatGPT screenshot ("Created with ChatGPT.") whose
reply is a generated image. **Slide 14** shows images made with the edges2cats demo, credited "Ivy
Tasi @ivymyt" and "Vitaly Vidmirov @vvid" ("Images created using pix2pix and edges2cats."). **Slides
15–19** show the same demo's input and output panels with the lecturer's sketches and pix2pix's
outputs; the demo's own screenshot on slide 13 is excluded ("© Chris Hesse"), but these slides print
no notice and share no image with it, so they are rendered, and they are the nearest case in this KB
to that line. **Slide 21**'s drawing is credited "Image is in the public domain." **Slides 25–28**
are uncredited polynomial-fit plots. **Slide 50**'s plot and **slide 51**'s diagram cite Valle
Pérez, Camargo and Louis (ICLR 2019). **Slides 52–60** are from Huh, Mobahi, Zhang, Cheung, Agrawal
and Isola (TMLR 2023), a paper of the lecturer's; slides 53, 54 and 60 print "Images courtesy of Huh,
et al. Used under CC BY-NC-SA."

Lecture 7's deck is Jeremy Bernstein's handwritten notes. All nine rendered slides are his own handwriting
and drawings, with no pasted material: the deck's third-party plots are on the excluded slides 7, 25 and
26, and its other embedded images (the title slide's logo, slide 14's emoji and slide 31's book icons) are
on slides that were not rendered.

Lecture 8's deck names Phillip Isola as speaker. Its rendered slides are mostly the lecturer's diagrams, but
several carry material from elsewhere that OCW did not flag. **Slides 14, 15, 28, 29, 30 and 44** use an
uncredited photograph of a savannah with two giraffes, a zebra and an impala, whole or as patches. **Slide
54** shows three uncredited photo crops (branches, an orange-yellow bird, foliage). **Slide 46** is credited "Courtesy
of Rußwurm, et al. Used under CC BY." (geographic location encoding with spherical harmonics), and **slide
47** "Courtesy of Kreuzer, et al. Used under CC BY-NC-SA." (Laplacian positional encodings).

Lecture 9's deck names Phillip Isola as speaker, and slide 2 credits Evan Shelhamer's "DIY Deep Learning: Advice
on Weaving Nets", Andrej Karpathy, Isolab members and Dylan Hadfield-Menell. Most of its rendered slides are the
lecturer's diagrams, but some carry material from elsewhere that OCW did not flag. **Slide 24** shows a StyleGAN2
face and a DALL-E illustration ("an illustration of a baby daikon radish in a tutu walking a dog"), uncredited.
**Slide 34**'s colour-gamut plots print no credit; the colorization project they belong to is cited on slides
30–32 as "[Zhang, Isola, Efros, ECCV 2016]", a paper of the lecturer's, and the photographs on those slides are
excluded. **Slide 57**'s loss plot, sample grid and creature frame are uncredited; in the recording the lecturer
describes the samples as a GAN's and the creatures as "little ants that I trained years ago" (≈7:48–8:34).

If a rights holder or the course asks for a page to come down, delete the image file and every
image embed that points at it.

## Rebuilding

Built and updated by the `cairn-kb` skill; [`TODO.md`](TODO.md) lists the remaining lectures.
Notes for the next run, from lectures 1–9:

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
  colon and the equation; cite the slide after the equation instead. It also reads only the first
  number of a list: "slides 36 and 39" did not count as citing slide 39 in lecture 5, which had to be
  hand-placed.
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
  usually reads each formula aloud, so check the formulas against the transcript too. On lecture 7 the raster
  test failed the other way, flagging every page, because the background is itself an image object
  painted on every page; drop any xref that appears on every page before measuring. Two of lecture 7's
  three OCW notices sit on slides that are mostly the lecturer's own handwriting (25 and 26); they are
  still not rendered, and their handwriting is transcribed in full.
- **Audit with a different model from the transcriber.** A same-model audit catches misreadings
  caused by resolution but not ones the two runs share. Lecture 3 was read at Sonnet and audited at
  Opus from 600-dpi crops, and lectures 4, 5, 6, 7, 8 and 9 the same; lectures 1 and 2 were Sonnet audited
  by Sonnet. Lecture 4's audit found errors on 13 of 24 pages, mostly counts (points, peaks, nodes,
  grid cells) on charts and diagrams, lecture 5's on 11 of 27, again mostly counts, lecture 6's
  on 16 of 30, mostly positions and counts, lecture 7's on 15 of 24, mostly colours, counts and
  arrow directions, lecture 8's on 19 of 33, mostly counts and where arrows go, with no formula wrong, and lecture 9's on 13 of 34,
  mostly counts and positions, plus one object described that is not on the page (slide 30's spoon). Lecture 6's audit agent was stopped by a session limit
  after 28 pages; it had appended each page to a report file, so a second agent did only the last
  two. For decks of
  graph drawings, tell the auditor it may count nodes and edges from `page.get_drawings()`.
- **Typed decks with LaTeXiT equations** (lecture 8): the text layer holds each equation as base64 junk, so
  it is good for titles, labels, citations, notices and code but never for an equation. Lecture 8's code
  slide (39) was confirmed against it character by character.
- The transcriber tends to write a quoted label that starts with math (`"$N$ tokens"`), which github.com
  will not render: inline math must not open straight after a quote mark. `check_math.mjs` warns; drop the
  quotes.
- **PyMuPDF on a downloaded deck** (lecture 9): `python3 -I` hides the user site-packages where PyMuPDF is
  installed, and this machine's Python 3.9 has no `-P`. Keep scripts in the session scratchpad and run them
  from there with the PDF path as an argument.
- **A slide can repeat part of an excluded slide without its notice** (lecture 9's slide 57 repeats slide 4's
  lower row, but not the X-ray the notice sits beside). Hash-matching flags it, as it should; then check which
  image object the notice belongs to before deciding.
- After writing, run `check_math.mjs` and `verify_kb.py`.
