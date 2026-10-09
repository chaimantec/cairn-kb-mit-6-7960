# Build log — MIT 6.7960 knowledge base

The per-lecture record of how this knowledge base was built and checked: what each deck's lecture pointers say, which
slides carry OCW licence notices, how each transcript was edited, what each figure audit found, each deck's printed slips,
and which slides were rendered as images and why the others were not. It is a build record, **not needed to answer
questions**: every fact a learner needs is on the wiki page, slide file or transcript it concerns. The rules every build
follows are in [`AGENTS.md`](AGENTS.md); attribution of the rendered images is in [`LICENSE.md`](LICENSE.md). Read this file
when building or revisiting a lecture, for precedent.

## Lecture pointers printed in each deck

Lecture 1's deck sends later topics to "Lecture N" banners that do not always match the recorded schedule; the mapping is in [`wiki/course-map.md`](wiki/course-map.md#the-decks-lecture-pointers). What each later deck and recording says:

- The decks of lectures 2
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
  Lecture 10's title slide, "Lecture 10: Memory and sequence modeling", settles lecture 1's "Lecture 11: RNNs"
  banner, but its outline (slide 2) and the outline's repeat (slide 67) are headed "11. Memory and sequence
  modeling", and the last outline (slide 68) "9. Memory and sequence modeling"; cite it as lecture 10. It prints
  no other lecture number. Its recording points back to CNNs over time (4), backpropagation over shared parameters
  (2), the spectral norm (7), "the transformer lecture last week" (8) and "the hacker lecture" (9), and names no
  later lecture.
  Lecture 11's title slide, "Lecture 11: Representation Learning I", matches the recording and lecture 1's slide 73
  banner, but its outline (slide 2) is headed "12. Representation Learning I"; cite it as lecture 11. It prints no
  other lecture number. Its recording points back to problem set 2's spectral descent, to the normalization picture
  "in one of the last lectures" (9), to "the colorization problem I showed you before" (9), to the vision transformer
  (8) and to autoregression "which we talked about before" (8 and 10), and ahead to "the next lecture" on metric or
  contrastive learning (12), generative modeling "in a few weeks" (14–16), "a few lectures on transfer learning"
  (18–19), and variational autoencoders, linear probes and low-rank fine-tuning, and CLIP, all "later" without numbers.
  It also assigns the coloured-shapes experiments to "your p set 3"; on OCW they are in Homework 4.
  Lecture 12's title slide, "Lecture 12: Similarity-based Representation Learning", matches the recording, and its roadmap
  (slides 2 and 26) carries no number; the deck prints no other lecture number. Its slide 45 points to a "geometric DL lecture",
  which is not the title of any lecture in the schedule (lecture 9 is where data augmentation is set against geometric deep
  learning). Its recording points back to "the geometric deep learning lecture" (≈52:23) without a number and ahead only to
  "Thursday" (13).

## End pages

- **Slide numbers are printed at bottom centre** and equal the PDF page number, so slide N is
  page N. Each deck ends with an OCW end page (page 81 in lectures 1 and 2, page 43 in lecture
  3, page 84 in lecture 4, page 47 in lecture 5, page 66 in lecture 6, page 32 in lecture 7, page 55 in lecture 8, page 72 in lecture 9, page 69 in lecture 10, page 65 in lecture 11, page 70 in lecture 12), which is
  not lecture content. Lecture 5's is a 4:3 page and lecture 6's a 792×612 one, smaller than the slides,
  and `slide_number_map.py` reports each as printing no number although they print 47 and 66; it says
  the same of lecture 7's page 32, whose render shows a small "32", of lecture 8's page 55, a
  792×612 page that prints "55", of lecture 9's page 72, a 792×612 page that prints "72", of lecture 10's
  page 69, a 792×612 page that prints "69", of lecture 11's page 65, a 792×612 page that prints "65", and of lecture 12's
  page 70, a 792×612 page that prints "70".

## Handwritten decks

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

## OCW exclusion notices, by lecture

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
  Lecture 10's carries 20 in 69 pages: the classroom video frame and its questions (slides 3–8), the video strip,
  audio waveform, space–time cube, its slices and the 3D filter drawn on it (9–12, 14), the hidden-state diagram
  over the video strip (19), the cat photographs of the Frank example (15–18, 55, 56; "© source unknown",
  although the lecturer introduces the cat as her own), the Reformer figure (60, "© Kitaev, et al.") and the
  Longformer attention patterns (61, "© Beltagy, et al.").
  Lecture 11's carries 22 in 65 pages: the car photograph of the x2vec and probing examples (slides 4, 15, 16), Serre's
  visual-cortex diagram (14 and 21, "© Springer Science+Business Media"), Zeiler and Fergus's patch mosaics (17–21, "©
  Zeiler and Fergus"), the crossed-out chalkboard (24), the angelfish photographs of learning via compression (36–38)
  and of colorization (53, 54), the orange bird of the $L_2$ autoencoder, the pretext tasks and imputation (40, 41, 57,
  58), the bird, parrot and temple of slide 45, the conv5 stimulus photographs (55), and Yann LeCun's cake slide (63,
  "© Yann LeCun, IEEE"). Slide 21's one notice names both holders ("Left © Springer … Right © Zeiler and Fergus").
  Lecture 12's carries 37 in 70 pages: nearly every photograph (the dog and monkey of the title and of slides 28–34, the
  instructors' faces, the moths, the horse and monkey, the bicycle, basketball and horse photographs of the "views" slides),
  Ng et al.'s scatter plots (14), Olivier Moindrot's triplet network (20), Fu et al.'s similarity triplets (24), Wang and
  Isola's hypersphere and alignment figures and the CIFAR-10 heat maps (31, 41, 42), the augmentation chart (48, "Manipulated
  images © Chen, et al."), Chuang et al.'s debiased negatives (52), Khosla et al.'s supervised contrastive figure (53), the
  iNaturalist mosaic (54) and all fourteen of Elijah Cole's case-study slides (55–68: "Slide © Elijah Cole. Image © Cole, et
  al." or "Image © source unknown"; slide 58's prints "Duagran" for "Diagram"). **Only two of the 37 are text** (slides 14
  and 59). The other 35 are drawn as glyph outlines, filled vector paths of hundreds of segments each, so a text-layer grep
  finds two notices where the deck has 37. The vision-read slide file found all of them, and a scan for short, wide filled
  paths with many segments (`page.get_drawings()`, fill set, at least 40 items, under 60 pt high) confirms them: every page
  with such paths is either a notice page or one of the five slides whose small print is a "Courtesy of … Used under CC BY"
  or "CC BY-NC-SA" credit (7, 8, 22, 23, 35). Slides 25 and 33 carry both kinds of line: a CC credit for one figure and an
  "All rights reserved" notice for the "Other images".

## Excluded images reused without a notice, and the hash sweeps

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
  the same image. Lecture 10's unnoticed images were hashed the same way against every excluded image in decks
  1–10, and none matched. Its slide 59 reproduces Table 1 of "Attention Is All You Need" with no notice; lecture 8's
  excluded slide 53 shows the same paper's architecture figure and first page, a different image, so slide 59 is not
  excluded by it (it is not rendered anyway: its table is transcribed cell by cell). Lecture 11 found one reuse across
  decks: its slide 13, the CLIP "ViT block x3" figure, prints no notice but is the figure of lecture 1's excluded slide 73
  ("© Torralba, Isola, and Freeman"), at a different resolution (2928 against 3715 px) and the same position on the page;
  over its inked pixels the two differ by a mean of 0.7 grey levels in 256. So slide 13 is not rendered. Three in-deck
  matches turned out not to be visible reuse, because `page.get_image_info()` shows where an image is actually drawn:
  slide 31 lists the three image objects that slide 4 lists beside its car photograph, but neither page draws them (slide
  31's building, road and car cards are vector drawings); slide 39 draws a small copy of slides 40–41's excluded bird
  photograph only outside the page (x 1950 on a 1920-wide page, y negative); and slides 40 and 41 draw slide 42's
  coloured-shapes grid only below the page (y 1840–2391 on a 1080-high page), so their notices cannot cover it. Slides
  31, 39 and 42 are rendered. The coarse hash also flagged slide 9's near-black 938×938 masks against lecture 4's
  120×124 crop and slides 10–12's near-white plots against lecture 1's slide 73; measured over inked pixels they differ by
  about 55 grey levels, and they are different images.
  Lecture 12's excluded slides were taken from its slide file, not the text layer (above). Its unnoticed images were hashed
  against every excluded image in decks 1–12, and every image rendered for lectures 1–11 against lecture 12's excluded ones,
  using only images actually drawn on the page (`page.get_image_info()`). The only hits were slide 18's 780×66 equation strip,
  which excluded slide 19 also draws (slide 19's notice covers its photographs, and slide 18 is not rendered anyway), and two
  coarse matches of lecture 3's slide 12 and lecture 5's slide 20 against a near-uniform 359×207 mask on slides 28–32, which
  differ by about 255 grey levels in 256. Slides 55–58 and 60–66 draw the same two images as slide 59, pixel for pixel, but
  each carries its own notice.

## Transcript edits, by lecture

Each lecture's edit was verified against `original/` by script (identical `[MM:SS]` markers, identical numbers once `[Ed: …]` notes are removed, paragraph word ratios within 0.72–1.10). Every change is also listed in its transcript's header.

Lecture 1: 79 markers identical, 225 numbers identical,
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
inline. Lecture 10: 95 markers identical, all 81 digit strings identical, word ratios 1.00–1.22,
and 1.00 throughout once its `[Ed: …]` notes are removed (the high end is the 30:19 paragraph, whose note sets the
spoken "quadratic relationship" beside the powers of W she has just written). Its restorations are "input beta" →
"input data" (13:09), "LCM" → "LSTM" (37:15), "skipped connection" → "skip connection" (40:22) and "retro" →
"RETRO" (1:06:07, slide 62), plus one lower-cased "And", one restored full stop and two stray commas removed, and
eight `[Ed: …]` notes (identifying lecture 2, lecture 7 as Jeremy Bernstein's, slide 31's CMU reading and slide 64's
Mangalam et al.; and four places where the spoken wording differs from the slides or the board: "v" for slide 23's
W, "quadratic", "maximizing cross-entropy" against slide 50's "minimize", and the Reformer against slide 60). Also
done inline. Lecture 11: 105 markers identical, all 65 digit strings identical but one, word ratios 0.99–1.27, and
0.99–1.01 once its `[Ed: …]` notes are removed (the high end is the 1:18:30 paragraph, which carries two notes). The one
digit change is deliberate: "Think of the 4DA transform" → "Fourier transform" (42:36), which "makes convolution just into a
product". Its other restorations are "RMSE norm" → "RMS norm" (10:49), "three-vision transformer blocks. In the transforming"
→ "three vision transformer blocks. In the transformer" (17:04, slide 13), "beta prime" → "b prime" (34:10, slide 29's
$\mathbf{b}'$), "CoLab" → "Colab", "date A" → "data A" (38:45), "sufficient in explanatory" → "sufficient and explanatory"
(45:43), "asked" → "ask" (53:36), "Are autoencoder is" → "Are autoencoders" (54:21), "little f in little g" → "and" (1:04:28),
"Imagenet" → "ImageNet" and "math prediction" and "mass prediction" → "masked prediction" (1:18:30), plus three restored full
stops and one stray one removed, and nine `[Ed: …]` notes (identifying lectures 8 and 9 and Kaiming He; the captions'
"0.10" and "0.01" for the one-hot points (1, 0) and (0, 1); a student's "two vectors", probably "2vecs"; "z is equal to the
encoder applied to f" against slide 22's $\mathbf{z} = f(\mathbf{x})$; a dropped "like"; and a student's inaudible
reference to the autoencoder). Also done inline. Lecture 12: 98 markers identical, all 27 digit strings identical, word ratios
0.99–1.21, and 0.99–1.00 once its `[Ed: …]` notes are removed (the high end is the 52:23 paragraph, whose note sets "the
geometric deep learning lecture" beside slide 45; the low end is "Celeb A" → "CelebA" joining two words). Its restorations are
"Sarah" → "Sara" (10:50, the lecturer of herself), "crossentropy" → "cross-entropy" five times, "projection had improved" →
"projection head improved" (56:15, slide 50), "Moco" → "MoCo" (58:41, slide 51) and "Celeb A" → "CelebA" (1:12:41, a spoken
aside the deck does not print), plus two punctuation fixes and one question mark, and five `[Ed: …]` notes (naming Phillip
Isola and the paper of slide 24; "zi and xj" against slide 12's $\mathbf{z}_ i$ and $\mathbf{z}_ j$; "push to similar samples
apart", probably "dissimilar"; and the "geometric deep learning lecture"). Also done inline.

## Slide transcription: figure audits and printed slips, by lecture

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

**Lecture 10's figure audit**, cross-model: Sonnet read the deck, and Opus checked 44 figure-, diagram-, chart-, table-
and equation-heavy pages (3, 5, 9–14, 16, 18, 19, 23–28, 31, 33, 34, 36–39, 41–45, 47, 49–53, 56, 58–62, 64, 65 and
69) from 50–600 dpi renders, the embedded rasters at native resolution, the vector data and the text layer. Every
equation agreed symbol for symbol, and all 16 cells of slide 59's table. Corrections followed on 10 pages. Two were
descriptions of things that are not on the page: slide 26's red backward arrows "beside U" (there are none; the red
arrows sit beside each V and under each W), and slide 33's large "A" on all three cells (the middle cell has none and
shows its wiring). The rest were counts and positions: slide 64 has eight video frames, not seven, and LVU sits near
17, not 8; slide 61's panel (d) has three global rows and columns; slide 60's grids (b)–(d) carry reordered labels;
slide 41's "a" over "time" is full-size; slide 34's outer cells carry an "A"; and smaller fixes on 3, 5, 11, 12, 16 and
25. The renders of slides 26 and 33 were then checked against the corrected text.

**Lecture 10's printed slips and oddities**, transcribed as written: slides 2 and 67 head the outline "11. Memory and
sequence modeling" and slide 68 "9. Memory and sequence modeling" (above); slide 4 says "kindergarden"; slide 29 says
"depedences" and its repeat, slide 54, "depedencies" (the titles' "dependences" is as printed too); slides 47–51 say
"enchances" and slide 52 "enhances"; slide 45 says "K is the size the vocabulary"; slide 41 prints a full-size "a" over
the output word "time" (it reads "taime"), as lecture 8's slide 48 does; slide 43 prints the first factor's subscripts
in bold; slide 39's output gate has no dot between $W_o$ and the bracket, where slides 36 and 37 have one; and slide 26's
upper summation limit is an upright sans-serif T. In the recording the lecturer calls the hidden-to-hidden weights "v"
(≈13:56; slide 23 has W), calls the powers of W a "quadratic relationship" before calling it exponential (≈30:19), says
"maximizing cross-entropy" where slide 50 says minimize (≈53:37), describes byte pairs as "2 character pairs", about
26 times 26 of them (≈51:16), where lecture 8 has one for "I-N-G", and closes on "another picture of Frank" that the OCW
deck does not contain (≈1:12:21); the wiki gives each beside the slide.

**Lecture 11's figure audit**, cross-model: Sonnet read the deck, and Opus checked 43 figure-, diagram-, chart-, table- and
equation-heavy pages (3–5, 7–14, 16–21, 25, 26, 28, 31, 32, 35, 38–40, 42–49, 52, 53, 55, 57–61 and 63) from 55–600 dpi renders,
the embedded rasters at native resolution, the vector data and the text layer. The first auditor was stopped by a session limit
after six pages; it had appended each page to its report, so a second agent did the other 37. Every equation agreed symbol for
symbol, including slide 9's softmax, which prints a minus sign and $\tau$ in both exponents, and all 18 cells of slide 35's
table; both charts (slides 43 and 61) are vector, and their values were measured from the marker positions. Corrections followed
on 13 pages, all counts, positions and drawing order, none an object described that is not on the page: slide 3 has three sheets
and five dotted paths, not four and four; slide 9's mapping column has a grey-grid picture beside the red cloud in every row;
slide 10 adds the planes' transformed frames; slide 14's left red arrow does not start at the fox's red dot; slide 16's two lines
reach the frame's top-right corner only; slide 19 has 108 patches, not 324, and its second row of blocks begins with posts and
railings, not vehicles; slide 20's upper-right quadrant has four women and two dogs; on slides 4 and 31 the car silhouette is in
front of the road card; slide 38's fifth column has ten shades, not nine; slide 46 has 500 dots, not about 400; slide 58's
channel-imputation cubes have three channel slabs, one observed below and two above; and slide 60's curved arrow ends in the gap
of the X. Smaller fixes followed on 5, 8, 11, 12, 35, 39, 40, 43, 55, 57, 61 and 63. The renders of slides 9 and 61 were then
checked against the corrected text.

**Lecture 11's printed slips and oddities**, transcribed as written: slide 2's outline is headed "12. Representation Learning I"
(above); slides 25–28 print "Neural" where "Neutral" is meant; slides 16 and 54 cite "Torralba., ICLR 2015" with a stray full
stop; slide 9's softmax writes the exponent as $e^{-\tau x_{\texttt{in}}[i]}$ while its wiring-graph node writes
$e^{-\mathbf{x}_ {\texttt{in}_ i}}$, with no $\tau$ and a bold $\mathbf{x}$; slide 60 cites "[He, Chen, Xie, et al. 2021]", the
masked-autoencoder paper of slide 59, under its BERT figure; and slide 63, Yann LeCun's slide pasted whole, carries its own footer
and page number "59". In the recording the lecturer says "z is equal to the encoder applied to f" (≈27:08) where slide 22 writes
$\mathbf{z} = f(\mathbf{x})$; names expectation maximization as k-means' optimizer (≈1:03:42) where slide 48 says "Block coordinate
descent"; works the linear-autoencoder-equals-PCA derivation from equations the OCW deck does not contain (slide 41 prints only
its conclusion); and assigns the coloured-shapes experiments to "p set 3" (≈54:21), which OCW publishes as Homework 4. The wiki
gives each beside the slide, and gives the PCA argument in the lecturer's words without reconstructing his equations.

**Lecture 12's figure audit**, cross-model: Sonnet read the deck, and Opus checked 46 figure-, diagram-, chart-, equation- and
photo-heavy pages (1, 6–8, 11–14, 17–25, 28–37, 39, 41–43, 48–55, 58–60, 64 and 66–68) from 100–600 dpi renders, the embedded
rasters at native resolution, the vector data, the fonts and the text layer. A session limit stopped both first auditors, after
15 and 10 pages; each had appended every page to its report, so two more agents did the other 21. Every equation agreed symbol
for symbol but one: slide 39's mutual-information bound prints the loss as $\mathcal{L}(f)$, with no "cont" subscript. The
stray marks in slides 28–30, 32 and 39's contrastive loss were settled as real glyphs of the embedded STIX fonts, with nothing
drawn behind them; their size and position support the readings given (transpose, temperature, "distributed as", sum, arrow),
and slide 39's script L is a breve, a different mark from the sum's caron. Charts were measured from vector markers or native
rasters: slide 43's 306 and 108 encoders were counted marker by marker, and the values of slides 48, 50, 52 and 60 agree with
the file to within half a point. Corrections followed on 26 pages. Three were relations described that are not on the
page: slides 18, 19 and 21's captions were said to sit over the other colour's underlined term (they are a header line above
the equation), slide 36's circle was said to have changed (only its arrowheads are hidden), and slide 52's label was said to
point at a photograph. The rest were counts, positions and wording: slide 43's triangles and second row of "+" markers, slide
54's 57 tiles, slide 22's grid and woodpeckers, slide 24's complementary badges, slide 31's wedge, slide 35's trapezoids (narrow
at the top), slide 48's notice (nothing cut off; words missing from the print), slide 58's box, slides 67–68's query outline
and repeated photograph, and slide 59's empty chart, which is slide 60's masked by a white panel. Slide 51's bars agree to within 0.2; they rise with batch size only up to about 2048. It also decoded slide
37's garbled anchor and heading (above). The renders of slides 35 and 50 were then checked against the corrected text.

**Lecture 12's printed slips and oddities**, transcribed as printed: slide 14 titles its plot "Porjected 2–class data" and
credits the figure to "Ng et al 2003", while slide 13 gives the paper as Xing, Ng, Jordan and Russell; slide 58's notice says
"Duagran" for "Diagram"; slide 13 prints empty boxes for $\in$ and plain S and D for the calligraphic set letters; slide 12
prints its Mahalanobis norm bars as "˜" marks; slides 28–30, 32 and 39 print the contrastive loss's transpose, temperature,
"distributed as" and summation signs as "ˆ", "˜", "˙" and "ˇ", slide 39 its script L as "˘", and slide 39's bound drops the
loss's subscript; slide 28's symmetry and matching-marginal lines are a separate raster in which "pos" is roman and "data"
typewriter, where the main equation sets both in italic; slide 24's labels and slide 37's anchor and heading print as
punctuation; slide 19 clips its label "x⁻"; slide 44 asks "What do the selection of positive and negative pairs encourage?";
slide 48's notice skips the words between "Chen, et al." and "content is excluded"; slide 64 prints "Imagenet"; and slide 45
points to a "geometric DL lecture" that no lecture is titled. In the recording the lecturer says "the distance between zi and
xj" (≈16:14) where slide 12 has $\mathbf{z}_ i$ and $\mathbf{z}_ j$; puts the crop-only augmentation loss at "like 15% almost"
(≈54:42), where slide 48's chart shows about 27.6 points for SimCLR and 13.1 for BYOL; puts the 256-to-8192 batch-size gap
early in training at "close to 10%" (≈57:51), where slide 51 shows about 7.2 points at 100 epochs; and defines alignment,
uniformity and the infinite-negatives decomposition of the contrastive loss from slides the OCW deck does not contain
(≈47:38–50:49). The wiki gives each beside the slide, and gives those definitions in the lecturer's words without
reconstructing formulas.

## Audit yields

Lecture 4's audit found errors on 13 of 24 pages, mostly counts (points, peaks, nodes,
  grid cells) on charts and diagrams, lecture 5's on 11 of 27, again mostly counts, lecture 6's
  on 16 of 30, mostly positions and counts, lecture 7's on 15 of 24, mostly colours, counts and
  arrow directions, lecture 8's on 19 of 33, mostly counts and where arrows go, with no formula wrong, and lecture 9's on 13 of 34,
  mostly counts and positions, plus one object described that is not on the page (slide 30's spoon), and lecture 10's on 10 of 44,
  two of them details described that are not on the page (slide 26's arrows, slide 33's label), with no formula wrong, and lecture 11's
  on 13 of 43, all counts, positions and drawing order, with no formula wrong, and lecture 12's on 26 of 46, mostly counts,
  positions and wording, three of them relations described that are not on the page (slides 18, 19 and 21's captions, slide 36's
  circle, slide 52's pointer) and one formula correction (slide 39's missing subscript). Lecture 6's audit agent was stopped by a session limit
  after 28 pages; it had appended each page to a report file, so a second agent did only the last
  two. Lecture 11's first auditor stopped after six of 43 pages the same way, and a second agent finished the other 37.
  Lecture 12 split its 46 pages between two auditors from the start; a session limit stopped both, after 15 and 10 pages,
  and two more finished the other 21 from the reports on disk. 

## Where each lecture's images appear

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
All 32 of lecture 10's images appear both in [`wiki/10-architectures-memory.md`](wiki/10-architectures-memory.md)
and under their headings in [`raw/slides/10-architectures-memory.md`](raw/slides/10-architectures-memory.md).
All 25 of lecture 11's images appear both in [`wiki/11-representation-learning-reconstruction-based.md`](wiki/11-representation-learning-reconstruction-based.md)
and under their headings in [`raw/slides/11-representation-learning-reconstruction-based.md`](raw/slides/11-representation-learning-reconstruction-based.md).
All 8 of lecture 12's images appear both in [`wiki/12-representation-learning-similarity-based.md`](wiki/12-representation-learning-similarity-based.md)
and under their headings in [`raw/slides/12-representation-learning-similarity-based.md`](raw/slides/12-representation-learning-similarity-based.md).
The concept pages embed none; they cite slides, and the lecture pages carry the pictures.

## What was rendered, and what was not, by lecture

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

### What was rendered, and what was not — lecture 10

Rendered (32): slides 13, 20, 22–28, 33–39, 41, 42, 44–53, 58, 62, 64 and 65 — the convolution in time over two rows
of circles, the RNN grid, the recurrent loop, the RNN with its weights W, U and V, the deep RNN, backpropagation
through time, the loss summed over time, parameter sharing, the shared W and its gradient, the standard RNN and LSTM
cells after Chris Olah, the cell state, the forget gate, the input gate, the update and the output gate,
autoregressive models with their training and sampling, the next-word bar chart and the word and character
vocabularies, the molecule-to-text sequence (circles, LSTM boxes, maximum likelihood, cross-entropy, teacher forcing,
testing, beam search), recurrence, convolution and attention side by side, the RETRO diagram, Mangalam et al.'s
certificate-length figure, and the fast-and-slow memory diagram.

Not rendered, and why:

- **Excluded from OCW's licence (20 slides): 3–12, 14–19, 55, 56, 60, 61** — listed under "OCW excludes some
  figures" above. Described in full in the slide file only.
- **Build steps superseded by a rendered slide:** 21 (slide 23 adds the weight labels and the weighted equations to
  the same column) and 30 (slide 25's diagram and equation again, with three bullets the file reproduces).
- **Text, equations and tables the slide file reproduces exactly:** 29 and 54 (the same text), 32, 43 (the
  factorization, with braces over "Once upon a time"), 57, 59 (Table 1 of "Attention Is All You Need", transcribed
  cell by cell), 63 and 66.
- **Decoration:** 31, the optional-reading pointer, whose picture is Peter Morville's streetlight cartoon ("CC BY-NC
  2.0") and carries no course content.
- **Title, outline, divider and end page:** 1, 2, 40, 67, 68 and slide 69, the OCW end page.

### What was rendered, and what was not — lecture 11

Rendered (25): slides 3, 5, 7–12, 25–28, 31, 39, 42–44, 46, 47, 50–52 and 59–61 — the layered prism of representation
learning and generative modeling, the x2vec data-space and representation-space cartoon, a function as a plot and as a
mapping, four scalar layers as mappings, the wiring graph, equation and mapping of the linear, relu, L2-norm and softmax
layers, the MLP training layer by layer and its stack, SGD against steepest descent in the spectral norm, training and
testing on different tasks, linear adaptation, finetuning, pretraining–adapting–testing, the properties of a good
representation, the autoencoder diagram, the coloured-shapes data, the nearest neighbours and the shape and colour
accuracy chart, clustering as an encoder, k-means as a scatter plot and as an encoder and decoder, data compression,
label prediction and data prediction, the masked autoencoder, BERT, and masked prediction against autoencoding.

Not rendered, and why:

- **Excluded from OCW's licence (22 slides): 4, 14–21, 24, 36–38, 40, 41, 45, 53–55, 57, 58, 63** — listed under "OCW
  excludes some figures" above. Described in full in the slide file only.
- **Reusing an excluded image without a notice:** 13, the CLIP figure of lecture 1's excluded slide 73 (see above).
- **Build steps superseded by a rendered slide:** 6 (slide 7's left plot alone).
- **Text, equations and tables the slide file reproduces exactly:** 22, 29, 32–35 (32–34's learner diagrams are three
  boxes and arrows, and 35 is a table transcribed cell by cell), 48 and 49 (the k-means and VQ boxes), 56, 62 and 64.
- **Title, outline, dividers and end page:** 1, 2, 23, 30 and slide 65, the OCW end page.

### What was rendered, and what was not — lecture 12

Rendered (8): slides 8, 22, 23, 35, 43, 49, 50 and 51 — the CIFAR-10 t-SNE plots under true and random labels, the
triplet-trained bird embedding and its nearest neighbours, the cross-channel "views" diagram, the alignment-and-uniformity
scatter plots of 306 and 108 encoders, the SimCLR projection-head diagram, the projection-head bar chart and the batch-size
bar chart.

Not rendered, and why:

- **Excluded from OCW's licence (37 slides): 1, 11, 14, 17, 19, 20, 24, 25, 28–34, 36, 37, 41, 42, 48, 52–68** — listed
  under "OCW excludes some figures" above, and drawn as glyph outlines on all but two. Described in full in the slide file
  only. The iNaturalist case study (54–68) is among them, so the lecture's last fifteen minutes have no image here.
- **Build steps superseded by a rendered slide:** 7 (slide 8 without its labels and text).
- **Text, equations and screenshots the slide file reproduces exactly:** 6 (the competition paper's title block), 10 (two
  arrows and two words), 12, 13 (equations and the Xing et al. title block), 16, 18, 21 and 39.
- **Title, roadmap, dividers, text slides and end page:** 2–5, 9, 15, 26, 27, 38, 40, 44–47, 69 and slide 70, the OCW end
  page.
