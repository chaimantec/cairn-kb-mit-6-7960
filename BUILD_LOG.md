# Build log — MIT 6.7960 knowledge base

The per-lecture record of how this knowledge base was built and checked: what each deck's lecture pointers say, which
slides carry OCW licence notices, how each transcript was edited, what each figure audit found, each deck's printed slips,
and which slides were rendered as images and why the others were not; its last sections hold the finished build checklists moved out of
`TODO.md`. It is a build record, **not needed to answer
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
  Lecture 13's title slide reads "6.7960 :: Lecture 13", matching the recording; the "6.7960" is handwritten on an opaque pink
  blob painted over a typed "6.S898", which only the text layer still shows. Its slide 4 points to "(Lecture 12)" for
  similarity-based representation learning and slide 5 to "(Lecture 7)" for the perspectives on neural computation, both matching
  the recorded schedule, and slide 2 to "(HW2)", Homework 2's question 4. Its recording points back to "last time" (12, ≈4:49), to
  the perspectives "in even the earlier lectures" (7, ≈5:36), to "the approximation theory lecture, which was maybe lecture two"
  (it was 3, ≈10:13) and to whether "Phillip talked about" overparameterization "in one of his lectures" (≈55:23); it names no
  later lecture, only a reading to be posted on Piazza.
  Lecture 14's title slide reads "Lecture 14: Deep Generative Models I", matching the recording ("Generative Models: Basics"). Its
  slide 2 lists "Lecture 14: fundamentals, a tour of popular models", "Lecture 15: generative modeling meets representation
  learning" and "Lecture 16: conditional models, data prediction", all matching the recorded schedule; it prints no other lecture
  number. Its recording points back to lectures 11–13 (≈0:45), to "that hacker's guide lecture" (9, ≈9:59), to the filing cabinet
  "from the lecture on generalization" (6, ≈37:18), to the colorization examples (9 and 11, ≈1:02:49) and to RNNs (10, ≈1:07:34),
  and ahead to variational autoencoders "next week" (15, ≈1:31, ≈59:43) and to conditional models and the applications of
  generative models "in lecture 16" (≈1:31).
  Lecture 15's title slide reads "Lecture 15: Deep Generative Models II", matching the recording ("Generative Models:
  Representation Learning Meets Generative Modeling"); the deck prints no other lecture number. Its recording points back to "the
  noise that we talked about last class", fitting a Gaussian "last Thursday", "the example I gave last class" (the procedural
  river) and GANs "covered … very briefly last lecture" (all 14; ≈2:19, ≈10:08, ≈18:01, ≈1:16:20), to the representation-learning
  lectures and their cartoons (11–13, ≈0:47, ≈3:06) and to the layer-by-layer diagrams (11, ≈3:06), and closes on "the last two
  lectures" (14 and 15, ≈1:19:23); it names no later lecture, only "the reading for today" and "the problem set".
  Lecture 16's title slide reads "Lecture 16: Deep Generative Models III", matching the recording ("Generative Models: Conditional
  Models"); the deck prints no other lecture number. Its recording points back to "last lecture", "a little bit math-heavy" (15,
  ≈0:00), to "the hacker's guide lecture" (9, ≈9:13), to GANs shown "very briefly in one of the previous lectures" (14, ≈24:12), to
  "the VAEs we saw last class" (15, ≈42:03) and to "the transformers lecture" (8, ≈52:15), and ahead to "another lecture on language
  models" (21, ≈51:27) and to "some future lectures" on grounding, which it does not name (≈1:20:56).
  Lecture 17's title slide reads "Lecture 17: Out-of-distribution generalization", matching lecture 1's banner and the recording
  ("Generalization: Out-of-Distribution (OOD)"); Sara Beery is the speaker. The deck's only pointer is "Domain adaptation (next
  lecture!)" on slides 58, 60 and 61 (lecture 18, Transfer Learning: Models). Its recording points back to image generation "the last
  couple of days" (14–16, ≈0:50) and to hard example mining learned "a while ago" (lecture 12's hard negative mining, ≈25:45), and
  ahead to meta learning and domain adaptation "next lecture" (18, ≈30:27, ≈56:55).
  Lecture 18's title slide reads "Lecture 18: Transfer Learning I", matching lecture 1's banner ("Lecture 18: Models") and the recording
  ("Transfer Learning: Models"); Sara Beery is the speaker, and the deck prints no other lecture number. Its outline (slide 2) and
  banner slides (8, 9, 29) list three kinds of knowledge, of which the third, about the inputs, is left to "next Tuesday" (19, ≈1:31),
  as is meta-learning (≈1:11:28). Its recording also points ahead to "a lecture from Phil around some of these scaling laws" next
  Thursday (20, ≈2:19), and back to distribution shifts "last week" (17, ≈0:00), to similarity-based representation learning and
  rotation invariance (12, ≈16:19, ≈49:35), to catastrophic forgetting "when we were talking about sequences" (probably 10, whose
  recording discusses forgetting in RNNs but never says "catastrophic forgetting", ≈25:31), to adversarial noise (17, ≈1:18:27) and to
  plugging pretrained models together, "I think we showed this early on" (≈1:23:48): lecture 2's slide 70, CLIP+GAN, is the same slide
  as this deck's slide 74.
  Lecture 19's title slide reads "Lecture 19: Transfer Learning II", matching lecture 1's banner ("Lectures 18, 19: Models, Data") and
  the recording ("Transfer Learning: Data"); Sara Beery is the speaker, it is her last lecture of the course (≈0:46), and the deck prints
  no other lecture number. Slide 3 is lecture 18's banner with "3. Knowledge about the inputs" in bold. Its recording points back to the
  generative-model lectures "with Phil" (14–16, ≈2:21), to "the lecture from VAEs" (15, ≈20:20; slide 18 is lecture 15's slide 40 again),
  to Dataset Interfaces, "I showed this work actually before" (17's slide 59, ≈36:41; this deck's slide 34), to learning a prior "when we
  were talking about distribution shift" (17's slide 55, ≈1:12:25), to the forward and backward pass unfolded (probably 2, whose slides 50
  and 51 draw slide 47's network, ≈1:09:21) and to in-context translation in "autoregressive modeling for large language models"
  (probably 18's slide 54, ≈1:13:12). It names no later lecture.
  Lecture 20's title slide reads "Lecture 20: Scaling Laws", matching the recording and settling lecture 1's "Lecture 22: Scaling laws"
  banner (slide 79); Phillip Isola is the speaker, and the deck prints no other lecture number. Its recording ends at ≈38:01, after slide
  17 of 51, and names no earlier lecture; it points ahead to "the next few lectures", which "will be of that flavor" (≈0:00), and, within
  the lecture, to the batch size "at the end of this lecture" (≈15:33) and to a theoretical model of power laws (≈24:45), both past the
  end of the recording. Lecture 3's slides 37–39 showed Kaplan et al. and Chinchilla first, and lecture 18 previewed this lecture (≈2:19).
  Lecture 21 is a guest lecture by Jacob Andreas, introduced by Phillip Isola. Its title slide reads "Language models" and "Jacob Andreas
  (many slides by Chris Tanner & Yoon Kim)", and the deck prints no lecture number of this course; lecture 1's deck has no banner for it.
  The deck is assembled from MIT 6.861's NLP lectures (≈1:34), and its title page still holds, under opaque white boxes, that course's
  title block, "Lecture 8: Prompting / Chain-of-thought and more! / 6.861* Fall 2024 / Jacob Andreas and Chris Tanner"; its outline slides
  (3, 22, 23, 44, 45, 61, 62, 84, 85) are 6.861's, and the second outline lists RLHF, DPO and "Remaining Issues/Ideas" sections that have
  no slides here. The recording points to other courses (6.8610 and "Yoon's special topics on language models class", ≈1:34) rather
  than to lectures, and to this course only generally ("throughout this semester", ≈5:25; "as a result of being in this course", ≈6:13;
  masked prediction "for vision tasks", ≈13:52, which is lecture 11). It answers lecture 16's "another lecture on language models"
  (≈51:27) and lecture 8's "We'll talk more about those later" (≈1:07:25).
  Lecture 23 is Jeremy Bernstein's. Its title slide reads "Metrized deep learning" and "Jeremy Bernstein", and the deck prints no lecture
  number of this course, its own or another's; it does not use lecture 1's banner title "Lecture 23: Automatic gradient descent" (slide
  79), so that row of the course map stays unsettled. Thirty pages print, beside OCW's small bottom-centre number, a larger number in the
  bottom-right corner from an earlier talk the deck was built from (page 16 prints 18, page 45 prints 50, page 47 prints 53). The recording
  points back without numbers to the scaling issues "in one of the earlier lectures in the class" (≈6:15) and "in the earlier lecture"
  (≈45:05), and to "optimization theory earlier in the class" (≈8:36), all lecture 7, whose slide 30 left open how modules' norms combine,
  which this lecture answers (its slide 23); and repeatedly to "the homework problem" on steepest descent under the spectral norm (≈7:02,
  ≈8:36, ≈10:09, ≈53:45, ≈59:15, ≈1:03:15), Homework 2's first question. It names no later lecture. Lecture 7's recording, in turn, does
  not mention lecture 23.

## End pages

- **Slide numbers are printed at bottom centre** and equal the PDF page number, so slide N is
  page N. Each deck ends with an OCW end page (page 81 in lectures 1 and 2, page 43 in lecture
  3, page 84 in lecture 4, page 47 in lecture 5, page 66 in lecture 6, page 32 in lecture 7, page 55 in lecture 8, page 72 in lecture 9, page 69 in lecture 10, page 65 in lecture 11, page 70 in lecture 12, page 28 in lecture 13, page 60 in lecture 14, page 42 in lecture 15, page 84 in lecture 16, page 63 in lecture 17, page 77 in lecture 18, page 52 in lectures 19 and 20, page 92 in lecture 21), which is
  not lecture content. Lecture 5's is a 4:3 page and lecture 6's a 792×612 one, smaller than the slides,
  and `slide_number_map.py` reports each as printing no number although they print 47 and 66; it says
  the same of lecture 7's page 32, whose render shows a small "32", of lecture 8's page 55, a
  792×612 page that prints "55", of lecture 9's page 72, a 792×612 page that prints "72", of lecture 10's
  page 69, a 792×612 page that prints "69", of lecture 11's page 65, a 792×612 page that prints "65", of lecture 12's
  page 70, a 792×612 page that prints "70", of lecture 13's page 28, a 792×612 page that prints "28", of lecture 14's page 60, a 792×612 page that prints "60", of lecture 15's page 42, a 792×612 page that prints "42", of lecture 16's page 84, a 792×612 page that prints "84", and of lecture 17's page 63, a 792×612 page that prints "63", and of lecture 18's page 77, a 792×612 page that prints "77", and of lecture 19's page 52, a 792×612 page that prints "52", and of lecture 20's page 52, a 792×612 page that prints "52". Lecture 21's page 92 is a 792×612 page that prints no number; that
  deck prints its own number only on pages 1–7 and carries the source decks' numbers in the bottom-right corner of eleven pages, which
  `slide_number_map.py` reads as a numbering with gaps (it reported 81 pages printing none and 7 numbers absent). Slide N is PDF page N.
  Lecture 23's page 49 is a 792×612 end page stored rotated 90° (/Rotate 90), which prints "49" at bottom centre; the script, reading
  unrotated coordinates, reports it as printing none. Its other 48 pages print the small bottom-centre number, equal to the page, and
  thirty of them also a larger bottom-right number from an earlier talk deck, which the script rightly ignores, so it reported a 1:1
  mapping. Slide N is PDF page N.

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
  grep; its handwriting OCR is noise ("Selond order"). The build expected the same of lecture 23, Bernstein's other
  lecture on the schedule; it is typed (below). Lecture 13's deck is handwritten the same way and paints the same 512×512 background twice on every page
  but the end page, plus a full-page 1168×876 grey image on slides 2–4. Its typed OCW notices (slides 4 and 22) are in the text
  layer. It also hides things under ink painted in the background colour, which the text layer and the image list still contain
  but the page does not show: the title slide's typed "6.S898" under a pink blob, slide 22's matplotlib axis labels
  ($f(x_1)$, $f(x_2)$ on both plots) under dark strokes, and three of slide 27's six book icons under lilac ones.
- **Lecture 23's deck is typed**, not handwritten: a Google Slides export (PDF creator "Google") with a full text layer in Lexend,
  Merriweather, ProximaNova and IBM Plex Mono, whose font names and fill colours the audit used to settle bold against plain and colour
  claims. Its equations are typed text whose text layer flattens them onto one line, so they were read from the page; its code (slides 32,
  41 and 43) came from the text layer and was checked against the page. The deck prints w, x, y and W in plain upright Lexend, its
  multiplication sign is the full-width asterisk "＊" (U+FF0A) and its composition sign "⚬" (U+26AC).

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
  Lecture 13's carries 2 in 28 pages, both typed and both in the text layer: slide 4's puppy and monkey photographs ("Dog ©
  source unknown. Monkey © San Diego Zoo."), the same pair as lecture 12's title slide, and slide 22's CIFAR-10 truck photographs
  ("Images © source unknown."), whose notice names no figure and sits at the bottom of the page under the slide's
  observations; the whole slide, the lecturer's own scatter plots included, is treated as excluded. Slide 3's pasted NanoGPT
  speedrun plot, from Keller Jordan's posts (handwritten credit "@kellerjordan0"), carries no notice. A vector scan for outlined
  small print finds only handwriting on this deck, as on lectures 3 and 7; the audit found no missed notice or credit.
  Lecture 14's carries 7 in 60 pages, all typed and all in the text layer: slide 6's DiffDock and MRI-to-CT figures ("Above ©
  Corse, et al. Below © Wolterink, et al.", the notice misspelling the citation's "Corso") and the flamingo images of the GAN
  slides 54–59 ("© source unknown"); on slides 56–58 the notice sits at the lower left under the equation, with the photograph at
  the top centre. Slide 5's DALL-E 2 screenshot and slide 42's WaveNet figure carry none. A vector scan for outlined small print
  found nothing, and the audit found no missed notice. Slides 49 and 50 carry no notice but reuse a figure Homework 5 excludes
  (next section).
  Lecture 15's carries 3 in 42 pages, all "© Torralba, Isola, and Freeman. All rights reserved. …", typed and in the text layer:
  slide 1's and slide 39's cover the same embedded image, a 5 × 5 grid of BigGAN images of a small bird (on slide 39 the notice sits
  at the lower left under the latent circle, far from the grid, but the grid is the page's only picture), and slide 6's covers its
  five bird photographs (the top row's orange bird, drawn twice, and four at the lower right). A vector scan for outlined small
  print found nothing, and the audit found no missed notice or credit.
  Lecture 16's carries 16 in 84 pages, all typed and all in the text layer: slide 3's four example pictures ("© Long et al., Glass,
  Reed et al., AlphaFold, Jumper et al."), slide 7's drum photograph ("© painmakers"), slide 8's colourized photograph ("©
  color_byangelina"), slide 11's photograph of flying birds ("© Fred Durand", where its credit line prints "Fredo Durand"), slide 70's
  VQGAN figure ("© Esser, et al."), slide 71's 1968 cartoon ("Cartoon © Science and Technology"), the CycleGAN material of slides
  72–79 ("© Zhu, et al."; slide 71 says the section "[Includes slides from Jun-Yan Zhu, Taesung Park]"), slide 81's plots ("© Galanti,
  Wolf, Benaim") and slide 82's figure ("© Lample, et al."). Slides 10 and 34's Andrew Ng chart ("[Slide credit: Andrew Ng]"), slide
  42's frames (Babaeizadeh et al.), slide 49's BicycleGAN facades and slide 69's LLaVA figure carry none. A vector scan for outlined
  small print found nothing, and the audit found no missed notice or credit. Slides 12 and 18–31 carry no notice but reuse excluded
  images (next section).
  Lecture 17's carries 31 in 63 pages, all typed and all in the text layer, and the slide file's set matches the text layer's exactly:
  slide 2's pictures ("Robotic hand © OpenAI. Images © source unknown", beside a placeholder "Image removed to copyright
  restrictions."), slide 6's posts ("© @hardmaru on X, @DimaKrotov on X, CNN.com"), slides 10 and 21's pig picture ("© Aleksander Mądry
  and Ludwig Schmidt"; slide 21 draws it three times, twice clipped to one pig), slide 11's traffic signs (Papernot et al.), slides
  12–15's sticker video (Brown et al.), slides 16–17's turtle (Athalye et al.), slide 18's speech diagram (Nicholas Carlini, David
  Wagner), slide 20's timeline (Battista Biggio, Fabio Roli), slide 26's digits (Hongzhou Lin), slides 27 and 36's feature strip ("©
  Aleksander Madry", without the accent), slides 28, 29 and 33 ("© sources unknown"), slides 30 and 34 (Geirhos et al.), slide 32 (AI
  Weirdness), slide 35 (Liu et al.), slide 37 ("©Tsipras et."), slides 42–43 (Koh et al.), slide 44 (Beery et al, Koh et al.), slide
  46 (Weinstein et al.), slide 55 (Mac Aodha et al.; it covers the equation, which is an image, as well as the diagram), slide 57 (He
  et al.) and slide 59 (Vendrow et al.). Slides 1 and 41 carry "Courtesy of … Used under CC BY" credits (Mannion et al.; Sara Beery).
  Slide 7's paper page (Amodei et al., "Concrete Problems in AI Safety"), slide 38's charts (Salman et al. 2020) and slide 53's
  figure (Hashimoto et al. 2018) carry a credit or none, and no notice. A vector scan for outlined small print found nothing, and the
  audit found no missed notice.
  Lecture 18's carries 21 in 77 pages, all typed and in the text layer, and the slide file's set matches the text layer's exactly:
  slides 4 and 5 (Lake, Salakhutdinov and Tenenbaum's characters and vehicles, "© AAAS"), slide 6 ("© source unknown"), slides 12 and 13
  ("Tigers © source unknown"), slides 14–17 ("Images © source unknown", the crop-field photographs), slide 19 ("Images © He, et al."),
  slide 20 (DANN, "© Ganin, et al."), slide 33 (the classroom photograph and false-colour image, "© Couprie, et al."), slide 34 (SoundNet,
  "© Aytar, et al."), slide 52 (the in-context learning figure, "© Brown, et al."), slide 53 ("Icons © source unknown"), slide 57 ("Left ©
  Li and Liang. Right © IEEE."), slides 60–63 (CLIP, "© Radford, et al.") and slide 69 (visual prompt tuning, "© Jia, et al."). Slides
  27, 28, 38, 39, 43 (CC BY-NC-SA), 58, 59, 65–68, 70 and 75 carry "Courtesy of … Used under CC BY" credits, and slide 40's painting is
  "in the public domain". Slide 3 ("[Slide credit: Andrew Ng]"), slide 32 (Hinton, Vinyals and Dean's table, typeset), slide 41
  (OpenAI's chart), slide 42 (Ahmed and Thompson) and slides 45, 54, 56 and 71 carry a credit or none, and no notice. A vector scan for
  outlined small print found nothing, and the audit found no missed notice. Slide 63's notice begins inside the bounding box of its
  "Image Encoder" graphic, whose raster crop shows the notice's first characters at its edge. Slide 74 carries no notice but reuses
  slide 63's encoder graphics (next section).
  Lecture 19's carries 20 in 52 pages, all typed and in the text layer, and the slide file's set matches the text layer's exactly:
  slides 7–9, 16 and 17 ("© Brock et al.", BigGAN's birds and interpolations), 12 ("© Chai et al."), 13, 14, 33 and 42 ("© source
  unknown"), 15 and 19 ("© Karras et al.", StyleGAN), 20 ("© Wu et al.", StyleSpace), 24–26 ("© Zhang  et al.", DatasetGAN, with two
  spaces as printed), 27 ("© Tritrong  et al."), 28 ("© Lang et al."), 34 ("© Vendrow et al.") and 49 ("© AAAI"). Slide 27's X-ray figure
  carries "Courtesy of Li et al. Used under CC BY.", slides 35–39 "Courtesy of Jahanian et al. Under under CC BY.", and slides 6, 10 and
  11 a credit for a StyleGAN2 cat photo "Made available under the Nvidia Source Code License-NC". OCW removed slide 32's picture ("Image of
  people removed due to copyright restrictions."). Slides 18 (lecture 15's own plot), 40 (Jahanian et al.'s diagram) and 45 (MAML's
  algorithm and figure, cited) carry a citation or none, and no notice. A vector scan for outlined small print found nothing, and the
  audit found no missed notice. Slide 28's notice sits under its row of portraits, but names the authors of the paper whose cat is on the
  same slide (next section).

  Lecture 20's carries 6 in 52 pages, all typed and in the text layer, and the slide file's set matches the text layer's exactly: slide 3
  ("© Rich Sutton", his essay "The Bitter Lesson"), 24 ("Dinosaur © Universal Pictures. Chinchilla © source unknown.", after a "Pocket
  gopher courtesy of LeonardoWeiss on Wikipedia Commons. Used under CC BY." credit in the same block), 40 ("Left © OpenAI. Right © OpenAI,
  McCandlish, et al.", the Dota 2 graphic and the large-batch paper), 45 ("© OpenAI, McCandlish, et al.", both charts) and 47 and 48
  ("Icons © source unknown", the apple, orange, elephant and giraffe icons of the kernel slides). Slide 50 carries "Courtesy of Huh, et al.
  Used under CC BY." The paper and OpenAI charts on slides 1, 2, 5, 6, 10–14, 16–20, 26, 28, 29, 35, 37, 44 and 49 carry a citation, a
  URL or nothing, and no notice. A vector scan for outlined small print found one filled path of 1040 segments, 288 × 17 pt, on slide 42;
  the audit identified it as the red brush-stroke underline of the gradient-noise label, not small print. The audit found no missed
  notice.
  Lecture 21's carries 19 in 92 pages, all typed and in the text layer, and the slide file's set matches the text layer's exactly: slides
  26–28, 34 and 35 ("© Brown et al.", GPT-3's zero-, one- and few-shot prompt figures, each one raster with its heading and description,
  and the SuperGLUE charts), 29, 31–33 and 63 ("© Zhao et al.", the sentiment prompt), 52–56 ("© Kojima et al.", zero-shot
  chain-of-thought figures and tables), 58 and 59 ("© source unknown", the 2009/2022 dog meme), 60 ("© Google, Forbes, and Business
  Insider", the prompt-engineering screenshots) and 86 ("© MIT; image by Alex Shipps/MIT CSAIL", the MIT News article; the notice sits
  under the left column, and both columns, including the small photographs, are treated as excluded). Slides 37–39 ("Zhang, Feng, and
  Tan"; "Min et al."), 43 and 46–51 ("Wei et al.") carry "Courtesy of … Used under CC BY." credits, and slide 16 "Image is in the public
  domain." Slides 6, 7, 13, 19, 20, 36, 40, 57, 72 and 81 reproduce third-party charts and tables with a citation or none, and no
  notice. A vector scan for outlined small print found nothing, and the audit found no missed notice.
  Lecture 23's carries 15 in 49 pages, all typed and in the text layer, and the slide file's set matches the text layer's exactly: slides
  3, 5, 6 and 45 ("© source unknown", the LEGO brick renderings and the photographed heap of bricks; the bricks are two embedded images
  drawn on slides 3, 4, 5 and 45, and the heap one image drawn on 6 and 45), 4 ("Images © source unknown", the loss surface and the
  adversarial pig, beside the excluded brick), 7, 28, 42 and 46 ("© @kellerjordan0 on X", the hand-drawn learning-rate plots, the NanoGPT
  speedrun charts and a photograph; slide 28's reads "Graphs©"), 24 ("© Li, et al.", the loss-landscape renderings), 26 ("Fruit fly ©
  Sanjay Acharya. Other images © source unkown", the four animal photographs, with OCW's slip "unkown"), 27 ("© Carlos Jones / Oak Ridge
  National Laboratory", the supercomputer photographs), 31 ("Photos © source unknown", the author portraits of the modular-norm paper's
  title block), 37 ("Image © "ilbusca" at iStock", the mirror picture) and 47 ("Images © sources unknown"). No slide carries a "Courtesy
  … Used under CC" credit. Slide 2's photographs, slide 34's chart (from the lecturer's paper with his collaborators), slide 40's screenshot
  of the Shampoo algorithm and slide 41's chart carry no credit and no notice. A vector scan for outlined small print found nothing, and the
  audit found no missed notice.

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
  Lecture 13's sweep first matched every page of the deck, exactly, to lecture 7's excluded slides 7, 25 and 26: the two
  handwritten decks paint the same 512×512 background image, so every excluded page of either deck "contains" it. With images
  drawn on more than a third of a deck's pages, and rasters covering more than 90% of a page, left out of the comparison, the
  sweep was clean in both directions. Slide 3's 708×455 speedrun plot matches none of the excluded images in decks 1–13 by pixel
  hash or by coarse hash (including lecture 7's excluded plots from the same author), and no image rendered for lectures 1–12
  matches slide 4's or slide 22's photographs.
  Lecture 14's sweep was clean in both directions across decks 1–14: no image drawn on its unnoticed pages matches an excluded
  image, and no image rendered for lectures 1–13 matches slide 6's or the GAN slides' images. Slide 5's DALL-E 2 screenshot
  (2842×1366) is drawn larger than its page, so the 90% coverage filter had dropped it; checked without that filter it matches
  nothing. The sweep was then extended to the problem sets, whose excluded images had never been compared: slides 49 and 50's
  chain (one raster, 2608×514) matched Homework 5's Figure 3 (784×155, page 4, and page 5 of the solutions) at 7 bits in 256, and a
  mean grey difference of 6.2 over 11,301 dark pixels settles it as the same figure at another resolution: Ho, Jain and Abbeel's
  diffusion diagram, "© Jonathan Ho, Ajay Jain, and Pieter Abbeel. All rights reserved." in the homework. On the slides the deck's
  pixel-art birds cover its face photographs and white boxes its labels (slide 50 uncovers the labels), so a page render shows
  little of it; both slides are withheld. No image rendered for lectures 1–13 matches the problem sets' excluded images (Homework
  3's page 5 and Homework 5's figure).
  Lecture 15's sweep, against the slide files' excluded slides of decks 1–15, the withheld slides and the problem sets' excluded
  pages, was clean in both directions: none of the 15 images drawn on its unnoticed pages matches any of 452 excluded images, and
  none of the 1,930 images drawn on pages rendered for lectures 1–14 matches its 10 excluded ones, by pixel hash or coarse hash. The
  sweep was rewritten for this lecture, so it was first rerun on lectures 14 and 11: it found Homework 5's chain on lecture 14's
  slides 49 and 50 (4 bits, grey difference 5.5) and lecture 1's CLIP figure on lecture 11's slide 13 (0 bits, 0.4), as before. Slide
  10's resources hold copies of slide 6's excluded photographs (the orange bird and the four lower birds), but every one is drawn
  entirely off the page (y 2662–3819 pt on a 1080-pt page), so slide 10 is rendered, as lecture 11's slides 31, 39 and 42 were. Its
  river image is a 4 × 4 grid clipped to 3 × 3, the image slides 35 and 36 also draw; slides 37 and 38 share another river grid; and
  slides 31, 32, 40 and 14's plots are the lecturer's Colab output. None carries a notice or a credit.
  Lecture 16's sweep, lecture 15's script unchanged, against the slide files' excluded slides of decks 1–16, the withheld slides and the
  problem sets' excluded pages, found two reuses inside the deck. Slide 12 draws slide 11's excluded photograph of flying birds (the
  same image object), which is also, at another resolution, the photograph of lecture 4's excluded slides 12–24, 61 and 62 and lecture
  8's excluded slides 4 and 6–8 (coarse hash 3 bits, grey difference 0.6); slide 12 is withheld. And slide 73, under its "© Zhu, et
  al." notice, repeats slide 30 image for image: its only pictures are the label map and the red brick building of the conditional-GAN
  build, which every slide from 18 to 31 draws; slides 18–31 are withheld. Two other exact matches were false. Slide 1's three matches
  with lecture 15's excluded title slide are the course's title-banner graphics, while lecture 15's notice there covers its BigGAN grid;
  and slide 33's match with lecture 9's excluded slide 61 is two uniform black fills of the same byte count drawn through different soft
  masks (a 140 × 135 colour image and a 378 × 150 grey one). The remaining coarse matches had grey differences of 47.6 to 74.4, between
  images of unrelated shape (a 22 × 288 strip against a 461 × 361 photograph). No image rendered for lectures 1–15 matches lecture 16's
  47 excluded images.
  Lecture 17's sweep, lecture 16's script with lecture 16's withheld slides added, against the slide files' excluded slides of decks
  1–17, the withheld slides and the problem sets' excluded pages, was clean in both directions. None of the 54 images drawn on its
  unnoticed pages matches any of 618 excluded images: the one exact match is slide 1's 600 × 120 title-banner graphic against lecture
  15's excluded slide 1, the course's banner again (lecture 15's notice covers its BigGAN grid, and slide 1 is not rendered anyway),
  and the one coarse match (slide 41 against lecture 9's slide 28) has a grey difference of 110. None of the 2,016 images drawn on
  pages rendered for lectures 1–16 matches lecture 17's 84 excluded images; its coarse matches have grey differences of 56.5 to 199. The
  sweep was first rerun on lectures 14 and 11: it found Homework 5's chain on lecture 14's slide 50 (4 bits, grey difference 5.5) and
  lecture 1's CLIP figure on lecture 11's slide 13 (0 bits, 0.4), as before.
  Lecture 18's sweep, lecture 17's script with nothing withheld for lecture 17, against the slide files' excluded slides of decks 1–18,
  the withheld slides and the problem sets' excluded pages, was first rerun on lectures 11 and 14 and found the CLIP figure (0 bits, grey
  0.4) and Homework 5's chain (4 bits, 5.5) as before. It found three reuses. **Slide 74** (CLIP+GAN) draws the two raster graphics, a
  286 × 348 "Text Encoder" trapezoid and a 180 × 224 "Image Encoder" one, that excluded slide 63 draws twice and once under its "© Radford,
  et al." notice, byte for byte; they are crops in the CLIP figure's colours and font, and slide 63's notice starts inside the Image
  Encoder's box. Slide 74 is withheld. **Lecture 2's slide 70**, rendered since lecture 2, is the same slide, with the same two rasters
  (exact matches), and was withdrawn. **Lecture 12's slide 35**, rendered since lecture 12, draws the strip of classroom photograph, depth
  map and segmentation (1518 × 382) that lecture 18's slide 33 draws at 1196 × 300 under "© Couprie, et al."; the coarse hashes differ by 8
  bits and the grey difference is 9.2 (the two copies carry different colour profiles), and side by side they are the same picture. Lecture
  12's slide credits its figure "Courtesy of Tian, et al. Used under CC BY-NC-SA.", but the notice binds, and slide 35 was withdrawn.
  Cleared by eye: slide 28's ALDI card raster against lecture 5's slide 3 text excerpt (0 bits, grey 3.9: sparse dark marks on white, a
  different figure) and against three soft-masked uniform fills in lectures 6, 8 and 9 (grey 8.6–8.7); lecture 16's 284 × 284 purple
  square against slide 63's Image Encoder (53.9); and slide 1's title banner against lecture 15's excluded slide 1, the course banner
  again. The other coarse matches had grey differences of 74.7 to 217.3.
  Lecture 19's sweep, lecture 18's script with lecture 2's slide 70, lecture 12's slide 35 and lecture 18's slide 74 added to the withheld
  list, was first rerun on lectures 11 and 14 and found the CLIP figure (0 bits, grey 0.4) and Homework 5's chain (4 bits, 5.5) as before.
  It found three reuses, all exact. **Slide 23** draws slide 42's 1482 × 1482 volcano grid, which slide 42 draws under "© source unknown";
  withheld. **Slides 29 and 30** draw slide 28's 256 × 256 cat photograph. Slide 28's "© Lang et al." sits under the row of portraits, at
  the right, not beside the cat at the lower left; but Lang et al. are the authors of the StylEx paper the cat comes from, so the notice
  was read as covering it, and both slides are withheld. **Slide 31** draws four other 256 × 256 rasters, stacked: the generator's
  reconstruction of that photograph and three edits of it (open mouth, darker eyes, lowered ears; grey differences of about 34 against the
  photograph, so no hash match). They are the same paper's versions of the same photograph, and slide 31 is withheld with 29 and 30; this
  is a judgment call, and the slide can be rendered if the cat is judged uncovered. **Slide 36** draws two 256 × 256 tiger photographs
  byte for byte the same as lecture 18's excluded slides 12 and 13 ("Tigers © source unknown"); this deck credits them "Courtesy of
  Jahanian et al. Under under CC BY.", but the notice binds, and slide 36 is withheld. Cleared: the earlier-rendered direction matched
  lecture 9's slide 57 against slide 34's plate photographs coarsely (grey 49.9 and 72.6; side by side, a dark plot of four markers
  against an empty plate and a plate of food), and lecture 8's slides 14, 15, 28 and 29 against a slide 28 portrait (grey 65.0–71.6); and
  slide 1's footer banner matched lecture 15's excluded slide 1 again. No earlier image was withdrawn.
  Lecture 20's sweep, lecture 19's script with lecture 19's slides 23, 29–31 and 36 added to the withheld list, was first rerun on lectures
  11 and 14 and found the CLIP figure (0 bits, grey 0.4) and Homework 5's chain (4 bits, 5.5), its output identical to lecture 19's
  validation runs. It found no exact match and no reuse. Cleared after looking: slide 32's 3398 × 1207 raster, the drop shadow under the
  Sharma and Kaplan title block (black, drawn through a soft mask), matched coarsely the drop shadows of earlier decks' excluded paper
  screenshots (lecture 5's slide 3 at 0 bits and grey 6.2; lecture 6's slide 23, lecture 8's slide 53 and lecture 9's slide 3 at 3.5–3.7),
  black rectangles side by side; slide 1's footer banner matched lecture 15's excluded slide 1 (grey 12.3), the course banner again; and,
  in the earlier-rendered direction, slide 47's heat-map colour bar matched lecture 8's slide 44 (grey 37.3), a green gradient, and images
  of lectures 2 and 16 at 83–105. The other coarse matches had grey differences of 95.6 to 220.0. No earlier image was withdrawn.
  Lecture 21's sweep was rewritten this run (no sweep script is kept in the repo or the skill), to the method in AGENTS.md, with a
  coarse match at 12 bits or fewer. Its first version silently skipped every raster it could not convert for the coarse hash: indexed
  (palette) colour spaces, which PyMuPDF will not write as PNG. That was 347 images across all decks and problem sets, 60 of them in
  lecture 21, and the validation runs still found lecture 11's CLIP figure and lecture 14's diffusion chain, so validation alone would not
  have caught it; counting the failures did. With every image converted to RGB before the coarse hash, the validation runs found the CLIP
  figure (0 bits, grey 1.8; 0.4 before, a different conversion path) and Homework 5's chain (4 bits, 5.5), plus the course banner of
  lectures 11 and 14 against lecture 15's excluded slide 1 (exact), cleared before. Lecture 21's 141 candidate images against 836 excluded
  images, and the 2,197 images of earlier rendered slides against lecture 21's 63 excluded ones, gave no exact and no coarse match. That
  check withdrew nothing, but the re-sweep below did.
  **The fixed sweep, rerun over lectures 1–20**, with a second fix (it had reported each matching image once, at the first page drawing
  it, so a reused image drawn on several pages showed on one), found five rendered slides that reuse excluded images, all rendered before
  the deck that excludes them was compared in that direction. **Lecture 4's slides 74 and 75** draw lecture 10's excluded video-frame
  strip (1920 × 246) and space–time cube (934 × 858), byte for byte (lecture 10's slides 9 and 14, "© source unknown"); lecture 10's
  sweep compared only its own images against the excluded set. **Lecture 8's slide 15** draws, five times, the 560 × 420 audio waveform of
  lecture 10's slide 9, byte for byte; that slide's notice is a bare "© source unknown" beside neither picture, and lecture 10's slide
  file takes it to cover the video frames "and, possibly, the waveform". Withdrawing slide 15 treats it as covering both, a judgment
  call; the slide can be rendered again if the waveform is judged uncovered. **Lecture 1's slides 45 and 70** draw lecture 4's excluded
  heron photograph (800 × 533, slide 43 there, "© Fredo Durand"; grey 0.6), most of it off the page, with its top 18% and 11% showing
  at the bottom edge, the next animation frame's picture; the slide file and LICENSE.md had described it as an uncredited bird. All five
  were withdrawn. Cleared after looking: lecture 5's slide 9 and lecture 18's slide 28 against lecture 5's slide 3 (grey 17.7 and 13.2),
  Keynote drop shadows; lecture 5's slide 12 against lecture 6's slide 62 (exact), a 308 × 220 image on a slide that prints no notice and
  is withheld only for its heron beak; lecture 9's slide 57 against slide 4, cleared before; and the course banners.
  **Lecture 23's sweep** compared its 17 candidate images (on its non-excluded pages) against 904 excluded images, with the withheld list
  brought up to date first: the script copied from the lecture 21 run still lacked that run's five withdrawals (lecture 1's 45 and 70,
  lecture 4's 74 and 75, lecture 8's 15). No image failed to hash. Lecture 23's candidates matched nothing. In the other direction, one
  rendered slide matched an image lecture 23 excludes: **lecture 13's slide 3** pastes the "NanoGPT speedruns" chart (708 × 455) that
  lecture 23's slide 7 draws (669 × 430) under "© @kellerjordan0 on X. All rights reserved" (0 bits, grey 4.8, the same figure at another
  resolution on looking at both). It had been rendered because only a notice excludes and lecture 13 prints none; it was withdrawn, and the
  withheld table in AGENTS.md gains its row. The validation runs found lecture 11's CLIP figure (0 bits, grey 1.8) and Homework 5's chain
  in lecture 14 (4 bits, 5.5). **The full re-sweep over lectures 1–21**, rerun because the excluded set changed, found nothing else to
  withdraw. One match was new because the withheld list now held lecture 8's slide 15: lecture 11's slide 60 draws, byte for byte, a
  500 × 355 photograph of guineafowl and four small images that slide 15 also draws, mostly under lecture 11's BERT figure or off the page.
  Lecture 8's slide 15 is withheld for its audio waveform (a different, 560 × 420 image), which lecture 11's slide 60 does not draw, so the
  match was cleared. The other coarse matches were at grey 53.8 or more (lecture 11's slides 10 and 12 against lecture 1's slide 73, lecture
  2's slide 15 against lecture 4's slide 47, lecture 14's and 16's figure strips against lecture 6's slide 23, lecture 5's slide 9 and
  lecture 18's slide 28 against lecture 17's slide 2), different figures, or were cleared before.

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
apart", probably "dissimilar"; and the "geometric deep learning lecture"). Also done inline. Lecture 13: 96 markers identical, all
55 digit strings identical, word ratios 1.00–1.35, and 1.00–1.00 once its `[Ed: …]` notes are removed (the three outliers, 49:48,
1:06:24 and 1:12:44, are the paragraphs carrying notes). Its one restoration is "LLM.C" → "llm.c" (3:17, slide 3's legend), with
nine `[Ed: …]` notes: lecture 3 for "maybe lecture two"; "covariate structure", probably "covariance", twice; "what Zach was
saying", probably the question at 30:01; "sigma; x, x-prime" as $\Sigma(x, x')$; Phillip Isola; Zach as the `AUDIENCE` speaker
at 59:17; a student's "as the weight goes to infinity", probably "width"; and, marked as outside the course material, that
Xavier initialization is usually defined with variance 2/(fan-in + fan-out). Also done inline. Lecture 14: 105 markers
identical, all 76 digit strings identical, word ratios 1.00–1.17, and 1.00–1.01 once its `[Ed: …]` notes are removed (the five
outliers, 9:11, 48:06, 51:12, 54:17 and 1:18:26, are the paragraphs carrying notes). Three restorations: "representation, learning"
→ "representation learning" (0:45), "this data center" → "the Stata Center" (5:21, slide 5's prompt) and "energy ration" → "energy
ratio" (48:06); and nine `[Ed: …]` notes: the simplex for two classes, lecture 6 for "the lecture on generalization", a student's
"quantum difference", two spoken slips about where the energy is raised and lowered (51:12, 54:17), "x2 times x1" for "x2 given
x1", lectures 9 and 11 for the colorization examples, a student's unclear "correspondence lower", and slide 54's printed typo.
Also done inline. Lecture 15: 104 markers identical, every number identical, and word ratios 1.00–1.01 once its `[Ed: …]` notes are
removed. The edit was delegated to Sonnet, with the finished slide file as the authority on terms, and checked in the parent by
the three scripts and a word diff. Five restorations: "times z integrated over z" → "times pz" (27:22, slide 10's prior), "n of x"
→ "N of x" (33:39, the normal density), "log of e" → "log of E" (51:31, the expectation in Jensen's inequality), "backdropping"
→ "backpropping" (1:07:50) and "Again, doesn't have that" → "GAN doesn't have that" (1:16:20); and two `[Ed: …]` notes, on the
garbled parameter count of a mixture of Gaussians (12:27) and the spoken Gaussian exponent (33:39). No student is named. The
lecturer's mix-up of variance and standard deviation (22:42–25:49) and his remark that slide 14's 1/M "should be inside the log"
(31:18) are left as said; the header lists them. Lecture 16: 105 markers identical, all 44 numbers identical, and word ratios 0.99–1.01 once its `[Ed: …]` notes are removed (with
them, 56:59 and 58:32 reach 1.11 and 1.15). Done inline, by a script of exact replacements, after reading the whole transcript for the
wiki. Seven restorations: "imagine in" → "image in" (2:16), "Dall-E" → "DALL-E" (3:02, 1:02:29), "whereas, again," → "whereas a GAN will"
(38:59), "across a tent" → "Cross-attend" (1:06:19), "LLaVa" → "LLaVA" and "threw" → "through" (1:07:07), "Cézanne on paintings" →
"Cézanne paintings" (1:16:19) and "to solved" → "to solve" (1:20:56); and seven `[Ed: …]` notes: the five lectures referred to (9, 14, 15,
21 and 8), slide 57's printed prompt (56:59) and slide 61's title against its direction (58:32). No student is named. Lecture 17: 83
markers identical, all 27 numbers identical, and word ratios 1.00–1.01 once its `[Ed: …]` notes are removed (with them, 7:03, 7:48, 11:38,
56:06 and 56:55 reach 1.11–1.30). Done inline, by a script of exact replacements, after reading the whole transcript for the wiki. Four
restorations: "strips" → "shifts" (13:13, slide 8's "Distribution Shifts"), "adversarial trend models" → "adversarially trained models"
(42:48, slide 38), "Wilds" → "WILDS" (44:20, slide 42's logo) and "GELU Or" → "GELU or" (1:00:47); and ten `[Ed: …]` notes: the lectures
referred to (14–16, 12 and 18 twice), "lie on" as probably a dataset's name, possibly LAION (outside the course material), "this data
center" (lecture 14's captions hear "the Stata Center" so, but no slide names the building), "data poisoning" for what is usually called
data leakage (outside the course material), slide 55's credit for an [INAUDIBLE] name, "a naturalist" as probably iNaturalist, and
"striving" as probably "stratifying". No student is named. Restorations were held to names the deck prints; LAION, the Stata Center and
iNaturalist are printed nowhere, so they stay as notes. Lecture 18: 110 markers identical, all 40
numbers identical, and word ratios 1.00–1.01 once its `[Ed: …]` notes are removed (with them, seven paragraphs reach 1.10–1.27). Done
inline, by a script of exact replacements, after reading the whole transcript for the wiki and the slide file. The captions spell
"pretained" and "pretaining" 24 times (and "pretrain" once); all 24 were restored to "pretrained" and "pretraining", as the deck prints
them. Three more restorations: "optimal brain compression" → "Optimal Brain Compression" (54:16, slide 38), "Dalle-2" → "DALL-E 2" (1:07:31
and 1:24:36, slide 75; slide 44 prints "Dalle-2") and "colorless green ideas sleep, furiously." → "Colorless green ideas sleep furiously."
in quotes (1:09:52, slide 48). Eighteen `[Ed: …]` notes: the lectures referred to (17 twice, 19 twice, 20 with "Phil" as Phillip Isola, 12
twice, probably 10 for catastrophic forgetting, probably 2 for "I think we showed this early on"), a missing word at 13:11, slide 32's
citation for "Jeff Hinton", "cloud Gemini" as probably Claude (outside the course material), "Bill" unidentified (58:08), and five spoken
slips or doubtful words: "CIFAR10" where slide 37 prints CIFAR100 (52:42), the 176B model "from Microsoft" that slide 43 names BLOOM
(1:01:16, BigScience outside the course material), "portability" for potability (1:03:35), Google for the Three Mile Island restart
(1:04:24, Microsoft's deal outside the course material) and "primary numbers" where slide 51 prints prime numbers (1:10:41). No student is
named. The lecturer's spending, water and Amazon/eBay figures appear on no slide and are kept unchecked, as the header says.
Lecture 19: 97 markers identical, all 15 numbers identical, and word ratios 1.00–1.01 once its `[Ed: …]` notes are removed (with them,
six paragraphs reach 1.12–1.22). Done inline, by a script of exact replacements, after reading the whole transcript for the wiki and the
slide file. Three restorations: "topK" → "top-K" (31:15, slide 31), "the tech space" → "the text space" (1:06:13, from the sentence's list
of modalities and the question it answers) and a comma moved in "Given some initialization, we update it with SGD" (1:08:34, slide 44).
Sixteen `[Ed: …]` notes: the lectures referred to ("Phil" as Phillip Isola twice, 15, 17 twice, probably 2, probably 18), slide 4's
attribution of the Stable Diffusion quotation, "Mathcad X" (slide 5's double-struck X), "walking to walk" where slide 13 prints "walked",
"the fairness of the equity", a missing word at 48:08, the garbled passage at 1:00:44 (slide 42 gives the wording), "driven data", and the
first names "Chelsea Finn" and "Gabi Tseng" against slides 44–46's and 49's citations. No student is named. SAM and DINOv2, British
shorthairs, American robins, the oriole, toucan and star-shaped baseball, the ECCV paper on "instance augmented generation", "Bigfoot with
a crow texture" and model collapse appear on no slide and are kept unchecked, as the header says.
Lecture 20: 50 markers identical, all 45 numbers identical, and word ratios 1.00–1.01 once its `[Ed: …]` notes are removed (with them,
six paragraphs reach 1.13–1.75; the 1.75 is the short last paragraph, carrying the note on where the recording ends). Done inline, by a
script of exact replacements, after reading the whole transcript and the slide file. The recording ends at 38:01, mid-sentence, after
slide 17 of 51; OCW's transcript PDF (`mit6_7960f24_lec20_transcript.pdf`) ends at the same words, so the cut is OCW's, not the caption
fetch's, and OCW's video description ("power laws, their limitations, theoretical foundations, and the concept of critical batch size")
describes the deck rather than the recording. Three restorations: "skill" → "scale" (8:39, from the paragraph and slide 5), "webtext" →
"WebText" (15:33, slide 8's "WebText2") and "Von Neumann" → "von Neumann" (31:48). Twelve `[Ed: …]` notes: slide 8's "WebText2",
slide 2's 3.4-month doubling against "every three months", slide 5's unlabelled compute axis, the paper's year, slide 8's factor of 6,
slide 9's 2^−α, slide 10's two forms of the law, the lecturer's unresolved "I might be saying that wrong" (27:53; slides 15 and 21
answer it), slide 13's four constants against "there's five", slide 15's 0.73 against "0.7", slide 16's paper for "a paper on vision
power laws", and the end of the recording. No student is named. Big-O notation, GPT-4's training cost, the self-driving example,
Anthropic's founding, context-length scaling laws and the "five parameters … elephant" saying (a student attributes it to Anderson)
appear on no slide and are kept unchecked, as the header says.
Lecture 21: 100 markers identical, all 276 digit strings identical once the `[Ed: …]` notes are removed, word ratios 0.993–1.008; done
inline. Four restorations: "mass prediction tasks" → "masked" and "-tune" → "fine-tune" (14:40), "-of-thought" → "chain-of-thought"
(45:59), "truthful QA" → "TruthfulQA" (1:06:12, slide 20's panel title). Thirteen `[Ed: …]` notes: Yoon Kim; slide 4's 107.5 for "107";
slide 6's 2007 for the paper the audience dates to 1995; slide 20's "Persian QA" for "Farsi"; slide 54's 17.7 and 78.7 for "18%" and
"80%"; slide 64's "worth thinking", printed so; slides 64–70's "Created with InstructGPT." for "GPT-3 or 2"; slide 81's 350M, 1.3B, 6.7B and
175B for the sizes said as 300 million, 13 billion and 7 billion; a word missing in the captions at 1:08:30; slides 89–91's $P_{\theta}$ for
"the parameters, P"; "SLiCK", which no slide names; and slide 11 for "this one" at 1:15:27. No student is named. The Enigma decoding,
"wreck a nice beach", Sewon Min, GPT-4 attending to flipped labels, learning algorithms inside transformers, MIT's 2019 attempts at chain of
thought, context distillation and a model not knowing the year appear on no slide and are kept unchecked, as the header says.
Lecture 23: 87 markers identical, all 18 digit strings identical once the `[Ed: …]` notes are removed, word ratios 0.991–1.010 (one
paragraph reaches 1.19 with its note); done inline, by a script of exact replacements, after reading the whole transcript against the deck's
text layer. Seven restorations: "Jax" → "JAX" (5:27, slide 33); the norm "and" → "on" network outputs (14:09, slide 11); "series,
combination" (24:56, a stray comma); "well-known that" → "well-normed" (33:25, slide 21); "it's" → "its" second derivatives (42:44);
"loose band" → "bound" (45:51); "at argmax" → "an argmax" (53:45). Eight `[Ed: …]` notes: "1 norm" as "one norm" (17:12); the diamond as
slides 20 and 23's ◇ (33:25); the captions' own "[? functions. ?]" (38:04); "Philip" as Phillip Isola, a co-author of slide 31's paper
(48:14); "solving this argument" as probably "argmin" (1:00:02); llm.c as slide 7's "llm.c baseline" (7:02); Keller Jordan as
"@kellerjordan0" (1:04:02); and the "industry benchmarking competition", not named on the slides (1:02:29). No student is named. The
recording ends at 1:07:15 on slide 48's website, and OCW's transcript PDF (`mit6_7960f24_lec23_transcript.pdf`) ends at the same words.

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

**Lecture 13's figure audit**, cross-model: Sonnet read the handwritten deck, and Opus checked 25 pages (1–11, 13–18 and 20–27),
split between two agents from the start, from 100–600 dpi crops, the vector ink, the native rasters of slides 3, 4 and 22, and the
captions; both finished. Every formula agreed symbol by symbol, and the flagged letters were settled: slide 2's subscript is a
hand-drawn asterisk, the N of slides 18 and 25 a calligraphic $\mathcal{N}$, and the L of slides 20 and 25 a slanted capital L,
not ℓ. Nine pages agreed and 16 were corrected. Two corrections were things described that are not on the page: slide 17's
"nested contours" are sketched scatter clouds, and slide 27's "faint specks" are slivers of three icons painted over, not empty
entries. The rest were counts, colours and positions: four random curves, not five or six, on slides 10 and 13; 17 cells and 12 and
11 stems on slide 14; slide 9's formula in green with only $\alpha_i$ and $x_i$ pink, and no underline under "weights"; slide 3's
baseline meeting the target line at about 42 minutes, not 45 (measured from the native raster); slide 4's dot positions, monkey,
opaque ellipse and notice placement; and slide 22's input 2 (the same framing with mild noise), its painted-out plot labels and its
notice placement. The renders of slides 3 and 17 were then checked against the corrected text.

**Lecture 13's printed slips and oddities**, transcribed as written: slide 1's email "jbernstein@mit.edub", as on lecture 7's
title slide, and its "6.7960" painted over a typed "6.S898"; slide 4's open "a" that reads "necrby"; slide 15's "radom vector";
slide 21's $f_1, \ldots, f_{100}$ in its text against $f_{1000}$ in its matrix, and its inputs named $x$ and $x'$ in the text
but $x_1$ and $x_2$ in the matrix; and slide 27's "Processes", whose second "s" is a bare stroke that can be read as an undotted
"i". In the recording the lecturer calls lecture 3 "maybe lecture two" (≈10:13), calls variance-one-over-fan-in "Xavier
initialization" (≈1:12:44), and says "where L minus 1 is the depth" (≈1:09:28) of the $L - 1$ applications of $h$, where slide
25's $L$ counts the weight matrices. He does not write out the Gaussian process's conditional mean and standard deviation
("you can just look up what the formulae are", ≈44:19), and the wiki does not either. The last part of the lecture, on
initialization (≈1:12:44–1:15:05), is at a board, on no page of the deck.

**Lecture 14's figure audit**, cross-model: Sonnet read the typed deck in two halves, and Opus checked 52 pages (1, 2, 5, 6, 8–16,
18, 19, 21–26, 28–44 and 46–59), split between three agents, from 150–600 dpi crops, the native rasters and the vector data. A
session limit stopped two of them, but only after both had appended every page but slide 45, which was then checked against its
1400px render. Every equation agreed but slide 16's covariance, a bold Σ. 32 pages were corrected, none for an object described
that is not on the page. The corrections were counts (slide 19's eight sampled dots, not seven; slide 39's six filmstrip images, not
eight; slide 1's six grid rows; slides 52 and 53's pixel counts), positions and arrowheads (slide 29's and 33's red arrows and
wells; slide 18's trapezoid, which narrows the other way; slide 36's labels, which stand over rules, not braces; slide 49's dotted
curves, which carry no arrowheads), the identity of slide 35's overprinted glyph, and the placement of the GAN slides' notices. The
auditors also found what the pages hide: slide 16's tiles are the top-left nine of slide 18's sixteen, slides 43 and 44 keep a
"Classifier" label and a "c" in the text layer under the drawings, and slides 49 and 50's chain is Ho, Jain and Abbeel's own raster
with the deck's pixel art laid over its photographs, which led to the problem-set sweep above. The renders of slides 29 and 45 were
then checked against the corrected text.

**Lecture 14's printed slips and oddities**, transcribed as written: slide 5's "Created with DallE."; slide 6's notice "Corse"
against the citation's "Corso"; slide 8's $p(x) \in \mathcal{X}$ and its script $\mathcal{R}$ codomains; slide 27's "generalization
error" for the test log-likelihood; slide 31's first line, with no minus sign on the right, and its fourth, with a minus between
$\frac{1}{Z(\theta)}$ and the integral where a product is meant (the lecturer: "this is a times negative 1", ≈58:11); slide 32's
$E_{\theta(\mathbf{x})}$; slide 35's "taime", two animation steps printed at once; slide 37's bold subscripts; slide 51's extra
closing parenthesis; slide 54's "g tries to identify the fakes", called a typo in the recording (≈1:18:26); and slide 59's
"synthetic (0.9)" and "real (0.1)" against slide 55's "fake (0.1)" and "real (0.9)". In the recording the lecturer says "increase
energy where the model places high energy" (≈51:12) and "It'll place high energy where the data lives" (≈54:17), meaning low energy
and high probability, and "probability of x2 times x1" for "given x1" (≈1:01:17). Homework 5's header prints "Fall 2025".

**Lecture 15's figure audit**, cross-model: Sonnet read the typed deck in two halves, and Opus checked 39 pages (1, 3–32 and 34–41),
split between three agents, all of which finished, from 110–600 dpi crops, the native rasters and the vector data. Most of this
deck's equations are typeset text rather than LaTeXiT images, which let the auditors cross-check them against the text layer; every
one agreed, and slide 41's table agreed cell by cell, its ✓ and ✗ read from their glyph fonts. 21 pages were corrected and three given
small precisions. The corrections were counts (slide 28's 15 circles, not "about 14"; slide 31's 27 dots in 8 rows; slide 32's full
4 × 4 grid and 27 data points; slide 40's 26 square markers), positions and arrowheads (slides 5 and 23's lines without heads; slide 6's
⇓ under the Decoder, not at the line's centre; slide 19's arrow ending at the circle's centre dot; slide 22's brace, which ties the
formula to the disc; slide 27's trapezoid, which widens the other way; slide 28's $x$ and $\mu_z$), the placement of slides 6's and
39's notices, slide 38's background colours, slide 32's being four images rather than one, and two things described that are not on
the page: a lake behind slide 6's pelican, and grey dashed rims on slides 8 and 9's circles, which are the black target outline
showing through them. The transcriber had also named the bird of slides 1 and 39 by species, which neither the slide nor the
lecturer does; the name was removed. The renders of slides 16 and 40 were then checked against the corrected text.

**Lecture 15's printed slips and oddities**, transcribed as written: slide 11's reparameterization line, with a plain $x$ and
$g^{\sigma}_ {\theta}$ where the equation above has a bold $\mathbf{x}$ and the covariance $g^{\Sigma}_ {\theta}$ (the standard deviation is
meant, as the lecturer confirms, ≈25:49); slide 13's sum index $i = 1$ over terms indexed by $j$; slide 25's last line, with
$\theta^{\ast}, \psi^{\ast}$ on the left and $\phi$ for $\psi$ on the right (Homework 5 calls the encoder's parameters $\phi$); and slide 41's
footnote asterisk with no anchor. In the recording the lecturer calls slide 14's $\frac{1}{M}$ a typo that "should be inside the log"
(≈31:18), but the slide already prints it inside; both are recorded. Slide 41 marks VAEs ✗ for a density while slide 11 lists a
density among a VAE's outputs; the wiki notes the difference without resolving it.

**Lecture 16's figure audit**, cross-model: Sonnet read the typed deck in two halves, and Opus checked 71 pages (1, 3–15, 18–33, 35–38,
40–49, 54, 56–58 and 60–82), split between four agents, all of which finished, from 55–600 dpi crops, the native rasters and the vector
data. No equation was wrong; slide 70's code grids and slide 81's two matrices agreed entry by entry. 19 pages were corrected, 37 given
precisions only, and 15 were right as written. The corrections were counts (slide 11's ten birds, not "about 9"; slide 33's four stacked
squares, not three; slide 56's four dotted lines; slide 65's nine images, not eight), positions and arrows (slide 10's crossing at a third,
not a quarter; slide 21's orange arrow, which stops short of the label map; slide 26's rectangles, the same size, only lower; slide 36's
curve heights; slide 68's arrow to the answer, which is not drawn; slide 70's discriminator, which straddles its panel), the colours of
slides 12's and 68's birds, slide 66's two noisy inputs, which are one image, slide 82's panels, and slides 61–64 and 67's ℓ, which is
bold. Two were wording: slide 57's screenshot prints "enalbling", which the transcriber had corrected without a flag, and slide 72's
painting had been called "expressionist", a label neither the slide nor the lecturer uses; slide 54's printed underscores also needed
escaping. Slide 42's raster holds three more rows of its source figure, clipped away. The renders of slides 36 and 65 were then checked
against the corrected text.

**Lecture 16's printed slips and oddities**, transcribed as written: slide 5's plots lettered a, b, d, c in reading order; slide 13's
plain $y_i$ where its data are bold, and its title "Sat2Map" over a map-to-photo $\mathbf{x} \to \mathbf{y}$, the direction the lecture
runs; slide 13's $f_{\theta}(\mathbf{x})_ i$, pasted over an earlier $f_{\theta}(x_i)$ that the text layer keeps; slide 57's "enalbling";
slide 61's title "Text-to-image" over an image-to-text diagram, which the lecturer also calls "text-to-image" (≈58:32); slide 67's
"Ouput"; and slide 74's capital $G$ and $D$ under a diagram labelled $g_{\theta}$ and $d_{\phi}$. The deck writes the discriminator as the
probability of "synthetic" (slide 20), the reverse of lecture 14's slides 55–58 and the same as lecture 14's slide 59; the GAN page
states both. Slide 81's right-hand matrix may be a slip in its source figure (Galanti, Wolf and Benaim), on the outside assumption that
the maps are affine; the slide states no form, so it is transcribed as printed and the entry says so.

**Lecture 17's figure audit**, cross-model: Sonnet read the typed deck in two halves, and Opus checked 49 pages (1, 2, 4–7, 10–18,
20–30, 32–38, 40–47, 49–53, 55, 57, 59), every page with a figure, chart, diagram or equation, split between four agents, all of which
finished. No equation was wrong. 23 pages were corrected, 25 given precisions only, and slide 18 was right as written. Slide 38's twelve
bar charts were re-measured: the transcription had given the printed ticks as axis ranges, so values outside them looked impossible,
and its Cars values were wrong by up to 6 points; the slide file now carries the measured table. Slide 35's curves were re-read (its
legend has seven entries, not eight, and the scratchpad curve does not fall slowest), as were slide 45's bars and slide 47's brightest
cells. Counts were wrong on slides 27 (nine tiles, not ten), 57 (sixteen shapes per domain, not 13 and 14) and 40 (three boats);
positions on slides 21, 30 and 51; both arrows on slide 53 point down, not up; slide 26's "unmarked" cells carry their white pixel at
the corner itself; slide 33's top-right numbers are negative. Three things were described that are not on the page: slide 2's hand
holding the cube (it floats beyond the fingertips), slide 20's bracket joining two lines, and water behind slide 28's angler. Printed
slips were missed or misquoted on slides 2, 20, 34, 50 and 51. The renders of slides 38 and 40 were then checked against the corrected
text.

**Lecture 17's printed slips and oddities**, transcribed as written: slide 2's "Image removed to copyright restrictions." (no "due");
slide 20's "evalution"; slide 30's "lables"; slide 34's "generaliszation" in its left heading and its citation "Jacobson 2019" (outside
the course material, the name is usually spelled Jacobsen); slide 40's stray "Fr" (slide 5's text layer holds another, which the page
does not show); and on slides 50 and 51 the empirical distribution written both $\mathbb{P}_ n$ and $\hat{\mathbb{P}}_ n$, with the
circle diagram's $d$, $\leq$ and $\varepsilon$ against the objective's $D$, $\lt$ and $\epsilon$; the wiki states that they are the
same objects. Slide 27's and 36's notices print "Madry" where slides 10 and 21 print "Mądry", and slide 37's reads "©Tsipras et.".
Slides 32, 33 and 46 hide part of a pasted figure under the deck's own shapes (slide 32's hidden captions name the captioning systems;
slide 46's hide a sixth aerial image and six site labels); the entries describe what the page shows. Slides 7 and 42 are full-page
rasters under an OCR text layer that misreads them ("Barkan" for "Borkan").

**Lecture 18's figure audit**, cross-model: Sonnet read the typed deck in two halves, and Opus checked 62 pages (3–6, 8–10, 12–17,
19–43, 45, 46, 48, 49, 52–54, 56–63, 65–70, 73–75), every page with a figure, chart, table, diagram or equation, split between four
agents, all of which finished. No equation was wrong: slide 38's Algorithm 1 and slides 30 and 31's losses were checked character by
character, and slides 32, 43 and 56's tables cell by cell. 19 pages were corrected, 37 given precisions only, and 6 were right as written.
Counts: slide 5 has eight vehicle pictures, not nine (one tall picture had been described twice); slide 20 three blue block arrows and a
five-cell label block; slide 41 fifteen dots, not about sixteen; slide 57 seven input cells in its first column; slide 60's "…" column
and "⋮" row are the fourth, not the middle; slide 69 draws one token row between layers, and one outline round both backbones; slide 75's
columns have eight cells, not nine, and its prior and decoder three and five circles per column. Chart readings: slide 37's KD map has no
lower-half excess; slide 41's Xception sits near 5, not 1, and its trend line was re-measured; slide 45's right red box spans about 36 to
65, with the average-human line through it, and the axis tops out near 95; slide 61's x-axis is linear, not log-like. Directions: both of
slide 52's braces were the wrong way round, and slide 34's two conv8 connectors are nested. Pictures misdescribed: slide 21's network
drawing, slide 33's false-colour image (described as a smooth gradient) and slide 70's edge maps (light lines on black). Names from
outside the page: slide 39's "hammer-head" shark, slide 45's 1.75 × 10¹¹ for GPT-3 (the chart's point sits at about 2 × 10¹¹) and slide
75's "autoregressive-style" prior. Printed details: slide 10's "Neural" was unflagged and its shading given a meaning the page does not
state, slide 53's numerals are only partly bold, and the transcription had introduced a typo into slide 74's Colab URL and called its
mixed quotes straight. The renders of slides 45 and 75 were then checked against the corrected text.

**Lecture 18's printed slips and oddities**, transcribed as written: slide 10's "Neural" between "Like" and "Dislike", in both panels
(lecture 11's slides 25–28, the same example, print it too); slide 21's slide number printed twice in two fonts; slides 30 and 31's
"giraffee"; slide 36's "students view" beside "teacher’s view"; slide 43's "231gCO2eq/kWh" with no space; "SimCLR" twice in slides 46 and
73's grid; slide 49's "1." for the adaptor, repeating slide 48's numeral for the learner; slide 61's "Linear classifer"; and slide 74's
quotation, opened with a straight quote and closed with a curly one. Slide 44 prints "Dalle-2" where slide 75 prints "DALL-E 2". Hidden
content: slides 4 and 5 cover the pasted figure's panel label "i)" with a white rectangle; slide 3's number is black on black; white,
invisible letters sit in the text layer of slides 8, 9, 29 ("z") and 10, 31, 33, 36 ("c"); slide 27's raster runs on into a table and slide
42's holds three more panels, both cropped off the page; slide 65 clips slide 66's raster to its first panel. The entries describe only
what the pages show.

**Lecture 19's figure audit**, cross-model: Sonnet read the typed deck in two halves, and Opus checked 44 pages (3–31, 33–40, 42, 43,
45–49), every page with a figure, chart, diagram or equation, split between four agents, all of which finished. No equation was wrong:
slide 5's five formulas, slide 35's and 40's sets and transformations, and slide 43's nested datasets were checked character by character,
slide 33's sixteen numbers one by one, and slide 26's chart within half a point at every marker. 19 pages were corrected, 20 given
precisions only, and 5 were right as written. Counts and positions: slides 4 and 6's network has 16 arrows, two short of fully connected;
slide 5's two citations sit by the Manipulation row, not one by each of two rows; slides 8 and 9's arrow points at the photographs, not
the heading; slide 13's "Zoomed out" and "Zoomed in" label the arrows' tails and heads; slide 17's number sits over the fourth photograph;
slide 18 has exactly 26 and 16 dots, and its tick labels were misread; slide 23's dotted lines are curves from other dots; slide 24's
face row has five faces; slide 36 has one green dot, not two; slide 40's panels are not boxed; slide 45's gradient arrows have one head
each; slide 49's Random Forest has five points, not eight. Pictures misdescribed: slide 18 is four rasters under a visible OCR layer, not
one picture; slide 19's panel (c) is a notched shape; slide 20's codes are white on black; slide 23's volcano grid changes differently
along rows and columns; slide 35 has no second point; slide 38's lower latent tile is the original photograph again; slide 39's fourth
variant looks like the original. Names from outside the page: slide 14's church, mosque and fox. The audit also read the three MAML lines
slide 45's overlays hide, from the embedded image, and found slide 49's legend cropped from another panel. The renders of slides 40 and
48 were then checked against the corrected text.

**Lecture 19's printed slips and oddities**, transcribed as written: slide 5's "et al. ," with a space before the comma; slide 24's
"StyleGan" and "see  also"; "Zhang  et al." with two spaces in the notices of slides 24–26, and "Tritrong  et al." in slide 27's; "Under
under CC BY" in the credit of slides 35–39; slide 43's "Datastets"; and slide 44's "such just". Hidden content: white "z" letters on slide
3 and "c" letters on slides 36 and 40 in the text layer; earlier builds of the cat stacked under slides 30 and 31's visible one; charts and
image grids drawn off slide 34; clipped copies of slides 47 and 48's diagrams; slide 45's overlays over lines 5, 6 and 8 of the pasted
algorithm (given in the entry as hidden); and slide 18's whited-out captions and tick labels under a visible OCR text layer. The entries
describe what the pages show.

**Lecture 20's figure audit**, cross-model: Sonnet read the typed deck in two halves, and Opus checked 43 pages (1–3, 5–20, 23–30,
32–35, 37, 39–45, 47–50), every page with a figure, chart, table, diagram or equation, split between four agents, all of which finished.
No equation was wrong: slides 7, 9–16, 27, 33, 34 and 43's formulas were checked character by character against the rasters or the
typed math, slide 13's and 23's tables cell by cell. 22 pages were corrected, 10 given precisions only, and 11 were right as written.
Chart positions and counts: slide 1's envelope (black, ending near 0.3, not at 1) and 13 curves; slide 5's five dot positions and its
arrow at 70μ, not 100μ; slide 11's first dot at about 6·10^3; slide 17's green dots from 0.53; slide 28's GPT-3 and Gopher stars below
the Kaplan line, not on it; slide 35's GPT arrow; slide 40's ring of 12 dots and its border colours; slide 49's four grid lines and four
arrowheads. A value off by a factor of ten: slide 45's yellow curve flattens at 0.08 dollars, not 0.8. Stacking and placement: slide
24's gopher photograph lies over the dinosaur, not under it; slide 41's three red arrows are chained head to tail; slides 44 and 45's
labels and arrow; slide 47's elephant behind the orange. Misread type: slide 43's note is typed, not handwritten; slide 48's subscript is
sans-serif. Slides 26 and 29's dashed lines cross, and the caption's colours and thirds do not match them; slide 37's chart is slide 10's,
unidentified; slide 23's pasted table hides a Chinchilla row under its crop; slide 1's "97% raster" is a clipped three-chart image whose
unclipped box was measured, and slide 41's "99%" is vector shadings and drop shadows. Printed slips left unflagged: the citation's "of"
(slides 10–14, 18 and 19, against slide 8's title block), slide 16's exponent, slide 42's sign and the caption of slides 26 and 29. The
renders of slides 13 and 42 were then checked against the corrected text.

**Lecture 20's printed slips and oddities**, transcribed as written: the citation "Scaling Laws of Neural Language Models" on slides
10–14, 18 and 19, where slide 8's title block reads "for"; slide 8's "shape (depth, with, attention heads, …)"; slide 9's missing
closing parenthesis; slide 12's compute constant, 2.3·10^8 in the legend and 3.1·10^8 in the equation; slide 13's legend order, 3M
before 25M; slide 16's $b(N)/N^{-\alpha_2(D)}$, a minus sign in a denominator's exponent; slide 24's "Wikipedia Commons" (outside the
course material: the repository is Wikimedia Commons); slide 25's "4× more more data"; the caption of slides 26 and 29; slide 37's
Kaplan et al. chart under a Sorscher et al. citation; slide 39's "Which is yields"; slide 41's "as a function given"; slide 42's label
$\nabla\theta - \nabla_{\text{sgd}}\theta$ against its arrow and equation; and slide 43's "Plug the equation learning rate back into the
equation". Hidden content: slides 23 and 25 crop their pasted images (a Chinchilla row; the abstract's first and last lines); slide
33's 3D raster holds its two summands out of view; slide 20's raster holds a "RoBERTa Quantization" chart off the page; slide 44 holds a
contour plot outside its clip; slide 45 crops its right chart's title, "Atari Breakout - Pareto Fronts"; slide 47's icons cover the
typed arguments $f(x_i)$ and $f(x_j)$; slide 50 holds another version of its figure out of view. The entries describe what the pages
show. In the recording the lecturer gives compute's doubling as "every three months" (slide 2: 3.4 months), the parameter exponent as
0.7 (slide 15: 0.73) and the fit's constants as five (slide 13: four), and first says the data should grow faster, then that he "might
be saying that wrong"; the transcript and the wiki give both versions.

**Lecture 21's figure audit**, cross-model: Sonnet read the typed deck in two halves, and Opus checked 70 pages (1, 4–13, 15, 16, 19–21,
25–29, 31–43, 46–60, 63, 64, 66, 67, 69, 70, 72, 74, 75, 77, 79–83 and 86–91), split between four agents, all of which finished. No
equation was wrong; the formulas of slides 8–12, 25, 29, 47, 74 and 88–91 were checked symbol by symbol, and slides 13, 40, 54, 56 and 57's
tables cell by cell. 19 pages were corrected, 21 given precisions only, and 30 were right as written. Chart readings: slide 72's calibration
panel read upside down on its reversed log axis (Letter choices given as 7–55 where the page shows about 14, 8, 3 and 2% ECE); slide 20's
IPA panel blended GPT-3 and LaMDA into one description, its word-unscramble panel given a second series, and its word-in-context panel
the wrong models; slide 51's LaMDA panel given tick labels that are not printed, and its unlabelled lines named as if a legend said so;
slide 35's axis called evenly spaced. Things called unreadable or misdescribed: slide 7's faded chart, which the audit read as the
"Training FLOPS (Sevilla 2021)" chart that slide 6 hides under its table; slide 58's photograph and painted picture of dogs called
cartoons, with a meme name neither slide nor lecturer uses; slide 82's jigsaw globe named as a logo from outside the course; a quote
misattributed in slide 86's article summary. Build steps: slide 9 drops slide 8's highlight and label, slide 31 drops slide 29's braces,
and slide 32 highlights all six "Review:" and "Answer:" labels. Positions and arrows: slide 8's dashed rule and label, slide 29's math labels
(inside the box), slide 89's arrow (on $\mathbf{x}$, not on the bar). One printed slip was unflagged (slide 87's "instructions-tuned"),
and slide 28's [sic] was explained wrongly (only the English half's "girafe" is a slip). Hidden content under white boxes: slide 1's 6.861
title block, slide 4's GPT-2 and GPT-3 rows, slide 6's training-compute chart, slide 29's header, rules and test answer, and slide 89's
earlier draft of slide 90. Printed slips, kept and flagged in the slide file: slide 2's "Chat GPT"; slide 28's "plush girafe"; slide 47's
unbalanced parentheses; slide 55's "GMS8K"; slide 57's cropped caption; the review that is "worth seeking" (29, 31–33, 63), "worth
thinking" (64–66) and "worth watching" (67); slide 72's clipped labels; slide 75's "MIT is located Cambridge"; slide 79's unclosed quote;
slide 80's "ChatGPT.."; slide 82's "response are"; slide 87's "instructions-tuned data". The renders of slides 72 and 81 were checked
against the corrected text and agree.

**Lecture 23's figure audit**, cross-model: Sonnet read the typed deck in two halves, and Opus checked 44 pages (1–7, 9–14, 16–24, 26–34,
36–43 and 45–49), split between three agents, all of which finished. No equation was wrong in structure: slides 9–13's Gauss-Newton terms
(no transposes in the orange and mauve boxes, confirmed), slide 12's α and δ (the Lipschitz condition squares both sides), slides 20, 23
and 24's definitions, slide 36's identity, slide 40's pseudocode and slide 42's Newton–Schulz recurrence were checked symbol by symbol, and
slides 32, 41 and 43's code character by character. But the transcriber had set w, x, y and W in bold on slides 9–24 "for consistency with
the course notation", and the text layer's fonts show them plain throughout; all 123 were unbolded, and slide 9's blackboard-bold R's made
plain. 26 pages were corrected, 15 given precisions only, and 3 were right as written. Pictures: slide 2's 25 bricks (not "about twenty"),
its tower's order (six labels in two columns, not four) and its legible stickers, and the label "encoder-decoder", which neither page nor
recording uses; one path on slide 4's loss surface, not two; slide 45's side-by-side bricks, which do not overlap; slide 47's banknotes
called US dollars. Charts: slide 7's baseline entering at 5.5 minutes and meeting the target at about 42, and kinks, not bumps; slide 34's
top-left and top-middle panels (width 32 never shoots up; the modulaGPT minima do not line up); slide 41's end values in all three panels;
slide 42's Adam curve, last visible at about 12 minutes; slide 46's two "fastest" curves. Diagrams: slide 14's and 38's arrow ends (only
the outer two activation-space arrows reach boxes; the tensor-space arrows stop at the network's edge), slide 38's node counts (4, 6, 6, 6,
4, read from the vector data; the drawing is slide 14's), and slide 13's 120 edges. Text: slide 24's last bullet, a complete conditional
clause, not an unfinished sentence; slide 21's "stray 1", the subscript of $\Vert \cdot \Vert_ 1$; slide 23's module box, which drops slide
20's "three vector spaces:"; slide 29's subscript, the output space's script Y, not "a calligraphic y"; slide 41's indentation (7 spaces,
not 6); slide 43's line 18, both calls on one line, the strike through "(grad)" too; fonts called monospace on slides 17 and 21; the
logo's comma, grey not white. Small page numbers placed wrongly: slide 6's "6", printed over the photograph and not missing; slides 13, 17
and 21's. One printed slip was unflagged: slide 47's "We believe are questions". Hidden content: slide 7's and 28's plot image carries
hand-lettered titles under the panel title bars, and slide 7 clips away a whole depth plot that the recording never mentions; slide 28's
network image has its own "width" and "depth" labels masked out; slide 45 hides a yellow "w₁" under a brick. Slide 49's "49" is in the text
layer, on a page stored rotated 90°. Printed slips, kept and flagged in the slide file: slide 21's "L.norm(W) ≤ 1" in the embedding
module's box (a character-for-character copy of the linear module's line), slide 47's "We believe are questions", and OCW's "unkown" in slide
26's notice. The renders of slides 14 and 34 were checked against the corrected text and agree.

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
  and two more finished the other 21 from the reports on disk. Lecture 13's audit found errors on 16 of 25 pages of a
  handwritten deck, mostly counts and colours, two of them things described that are not on the page (slide 17's contours, slide
  27's empty entries), with no formula wrong; its two auditors, on 13 and 12 pages, both finished. Lecture 14's audit found errors
  on 32 of 52 pages, mostly counts, positions and arrowheads, with one formula correction (slide 16's bold Σ) and nothing described
  that is not on the page; a session limit stopped two of its three auditors after they had reported every page but one. Lecture 15's audit found errors on 21 of 39 pages, mostly counts and positions, with no formula wrong and two things
  described that are not on the page (a lake, and circle rims that are an outline showing through); all three of its auditors
  finished. Lecture 16's audit found errors on 19 of 71 pages, mostly counts and
  positions, with no formula wrong, one description of two images as different when they are one (slide 66), and two pieces of wording
  that were not the slide's (an unflagged correction of a printed typo, and an art-style label); all four of its auditors finished. 
  Lecture 17's audit found errors on 23 of 49 pages, mostly counts, positions and chart values, with no formula wrong and three things
  described that are not on the page (a hand holding a cube, a bracket, water); all four of its auditors finished.
  Lecture 18's audit found errors on 19 of 62 pages, mostly counts, chart readings and directions, with no formula wrong, three names
  taken from outside the page and one typo introduced into a URL; all four of its auditors finished.
  Lecture 19's audit found errors on 19 of 44 pages, mostly counts and positions, with no formula wrong, seven pictures misdescribed (one
  of them a second point that is not on the page) and three names taken from outside the page; all four of its auditors finished.
  Lecture 20's audit found errors on 22 of 43 pages, mostly chart positions, counts and stacking, with no formula wrong, one value off by
  a factor of ten, a repeated chart unidentified, a table row hidden by a crop, and four printed slips left unflagged; all four of its
  auditors finished.
  Lecture 21's audit found errors on 19 of 70 pages, mostly chart readings, build-step differences and positions, with no formula wrong,
  one chart read upside down on a reversed log axis, two names taken from outside the page, one faded chart wrongly called unreadable, and
  one printed slip left unflagged; all four of its auditors finished.
  Lecture 23's audit found errors on 26 of 44 pages, mostly chart values, counts, arrow ends and fonts, with no formula wrong in structure
  but every symbol of slides 9–24 bolded against the page, one label ("encoder-decoder") taken from outside the page, one bullet wrongly
  called unfinished, and one printed slip left unflagged; all three of its auditors finished.

## Where each lecture's images appear

All 18 of lecture 1's images appear both in
[`wiki/01-introduction.md`](wiki/01-introduction.md), in the passage that cites each slide, and
under the matching `## Slide N` heading of
[`raw/slides/01-introduction.md`](raw/slides/01-introduction.md). To list them:
`grep -o 'raw/images/[^)]*' wiki/01-introduction.md`. Of lecture 2's 44 images, 42 appear in
[`wiki/02-how-to-train-a-neural-net.md`](wiki/02-how-to-train-a-neural-net.md); slides 13 and 54,
near-duplicates of their neighbours 12 and 55, are only under their headings in
[`raw/slides/02-how-to-train-a-neural-net.md`](raw/slides/02-how-to-train-a-neural-net.md). All
18 of lecture 3's images appear both in
[`wiki/03-approximation-theory.md`](wiki/03-approximation-theory.md) and under their headings in
[`raw/slides/03-approximation-theory.md`](raw/slides/03-approximation-theory.md). All 29 of
lecture 4's images appear both in [`wiki/04-architectures-grids.md`](wiki/04-architectures-grids.md)
and under their headings in [`raw/slides/04-architectures-grids.md`](raw/slides/04-architectures-grids.md).
All 19 of lecture 5's images appear both in [`wiki/05-architectures-graphs.md`](wiki/05-architectures-graphs.md)
and under their headings in [`raw/slides/05-architectures-graphs.md`](raw/slides/05-architectures-graphs.md).
All 28 of lecture 6's images appear both in [`wiki/06-generalization-theory.md`](wiki/06-generalization-theory.md)
and under their headings in [`raw/slides/06-generalization-theory.md`](raw/slides/06-generalization-theory.md).
All 9 of lecture 7's images appear both in [`wiki/07-scaling-rules-for-optimization.md`](wiki/07-scaling-rules-for-optimization.md)
and under their headings in [`raw/slides/07-scaling-rules-for-optimization.md`](raw/slides/07-scaling-rules-for-optimization.md).
All 33 of lecture 8's images appear both in [`wiki/08-architectures-transformers.md`](wiki/08-architectures-transformers.md)
and under their headings in [`raw/slides/08-architectures-transformers.md`](raw/slides/08-architectures-transformers.md).
All 11 of lecture 9's images appear both in [`wiki/09-hackers-guide-to-deep-learning.md`](wiki/09-hackers-guide-to-deep-learning.md)
and under their headings in [`raw/slides/09-hackers-guide-to-deep-learning.md`](raw/slides/09-hackers-guide-to-deep-learning.md).
All 32 of lecture 10's images appear both in [`wiki/10-architectures-memory.md`](wiki/10-architectures-memory.md)
and under their headings in [`raw/slides/10-architectures-memory.md`](raw/slides/10-architectures-memory.md).
All 25 of lecture 11's images appear both in [`wiki/11-representation-learning-reconstruction-based.md`](wiki/11-representation-learning-reconstruction-based.md)
and under their headings in [`raw/slides/11-representation-learning-reconstruction-based.md`](raw/slides/11-representation-learning-reconstruction-based.md).
All 7 of lecture 12's images appear both in [`wiki/12-representation-learning-similarity-based.md`](wiki/12-representation-learning-similarity-based.md)
and under their headings in [`raw/slides/12-representation-learning-similarity-based.md`](raw/slides/12-representation-learning-similarity-based.md).
All 9 of lecture 13's images (10 until the lecture 23 run withdrew slide 3) appear both in [`wiki/13-representation-learning-theory.md`](wiki/13-representation-learning-theory.md)
and under their headings in [`raw/slides/13-representation-learning-theory.md`](raw/slides/13-representation-learning-theory.md).
All 28 of lecture 14's images appear both in [`wiki/14-generative-models-basics.md`](wiki/14-generative-models-basics.md)
and under their headings in [`raw/slides/14-generative-models-basics.md`](raw/slides/14-generative-models-basics.md).
All 22 of lecture 15's images appear both in [`wiki/15-generative-models-representation-learning-meets-generative-modeling.md`](wiki/15-generative-models-representation-learning-meets-generative-modeling.md)
and under their headings in [`raw/slides/15-generative-models-representation-learning-meets-generative-modeling.md`](raw/slides/15-generative-models-representation-learning-meets-generative-modeling.md).
All 29 of lecture 16's images appear both in [`wiki/16-generative-models-conditional-models.md`](wiki/16-generative-models-conditional-models.md)
and under their headings in [`raw/slides/16-generative-models-conditional-models.md`](raw/slides/16-generative-models-conditional-models.md).
All 11 of lecture 17's images appear both in [`wiki/17-generalization-out-of-distribution.md`](wiki/17-generalization-out-of-distribution.md)
and under their headings in [`raw/slides/17-generalization-out-of-distribution.md`](raw/slides/17-generalization-out-of-distribution.md).
All 26 of lecture 18's images appear both in [`wiki/18-transfer-learning-models.md`](wiki/18-transfer-learning-models.md)
and under their headings in [`raw/slides/18-transfer-learning-models.md`](raw/slides/18-transfer-learning-models.md).
All 15 of lecture 19's images appear both in [`wiki/19-transfer-learning-data.md`](wiki/19-transfer-learning-data.md)
and under their headings in [`raw/slides/19-transfer-learning-data.md`](raw/slides/19-transfer-learning-data.md).
All 20 of lecture 20's images appear both in [`wiki/20-scaling-laws.md`](wiki/20-scaling-laws.md)
and under their headings in [`raw/slides/20-scaling-laws.md`](raw/slides/20-scaling-laws.md).
All 18 of lecture 21's images appear both in [`wiki/21-language-models.md`](wiki/21-language-models.md)
and under their headings in [`raw/slides/21-language-models.md`](raw/slides/21-language-models.md).
All 8 of lecture 23's images appear both in [`wiki/23-metrized-deep-learning.md`](wiki/23-metrized-deep-learning.md)
and under their headings in [`raw/slides/23-metrized-deep-learning.md`](raw/slides/23-metrized-deep-learning.md).
The concept pages embed none; they cite slides, and the lecture pages carry the pictures.

## What was rendered, and what was not, by lecture

### What was rendered, and what was not — lecture 1

Rendered (20, of which the lecture 21 run withdrew 45 and 70; see the sweeps above): slides 14, 21, 22, 23, 28, 33, 34, 35, 36, 38,
40, 41, 42, 43, 45, 55, 57, 58, 59, 70 — the XOR plot, the enthusiasm curves, the loss surface, the linear-layer and perceptron
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

Rendered (44): slides 7, 10–22, 24–27, 30, 34, 36–38, 41–44, 46, 48–55, 60, 63, 66–68, 72, 73
and 75 — the loss surface, the momentum runs and Goh's momentum article, the six-landscape quiz and
the six landscape cases, evolution strategies and gradient clipping, the ReLU and GELU plots,
computation graphs, the chain-rule shapes, every backpropagation diagram (generic layer, linear
layer, whole MLP, one-iteration graph, merge and branch, parameter sharing), the human-versus-backprop
graph, unit visualizations, DeepDream, and the worked example's network before and after.

Not rendered, and why:

- **Excluded from OCW's licence (7 slides): 4, 57, 58, 59, 64, 65, 69** — the clown fish and
  chameleon photos, the LeCun and Dietterich posts, the Neural Module Networks figure, Karpathy's
  Software 2.0 figure, and the CLIP figure. Described in full in the slide file only.
- **Withdrawn when lecture 18 was added: 70** (CLIP+GAN). Its "Text Encoder" and "Image Encoder" graphics are, byte for byte, the
  images lecture 18's slide 63 draws under "© Radford, et al."; lecture 18's slide 74 is the same slide and is withheld too.
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

Rendered (31, of which the lecture 21 run withdrew 74 and 75; see the sweeps above): slides 3, 5–8, 10, 26–32, 34, 37, 39–42, 44,
48, 49, 56–60, 73–75 and 77 — the MLP
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

Rendered (34, of which the lecture 21 run withdrew 15; see the sweeps above): slides 5, 11–17, 19, 21–24, 26, 28–30, 33–36, 38,
41–44, 46–52 and 54 — the CNN
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

Rendered (7): slides 8, 22, 23, 43, 49, 50 and 51 — the CIFAR-10 t-SNE plots under true and random labels, the
triplet-trained bird embedding and its nearest neighbours, the alignment-and-uniformity
scatter plots of 306 and 108 encoders, the SimCLR projection-head diagram, the projection-head bar chart and the batch-size
bar chart.

Not rendered, and why:

- **Excluded from OCW's licence (37 slides): 1, 11, 14, 17, 19, 20, 24, 25, 28–34, 36, 37, 41, 42, 48, 52–68** — listed
  under "OCW excludes some figures" above, and drawn as glyph outlines on all but two. Described in full in the slide file
  only. The iNaturalist case study (54–68) is among them, so the lecture's last fifteen minutes have no image here.
- **Withdrawn when lecture 18 was added: 35** (the cross-channel "views" diagram). Its classroom photograph and depth map are the
  strip lecture 18's slide 33 draws under "© Couprie, et al.", although this slide credits the figure "Courtesy of Tian, et al. Used
  under CC BY-NC-SA."
- **Build steps superseded by a rendered slide:** 7 (slide 8 without its labels and text).
- **Text, equations and screenshots the slide file reproduces exactly:** 6 (the competition paper's title block), 10 (two
  arrows and two words), 12, 13 (equations and the Xing et al. title block), 16, 18, 21 and 39.
- **Title, roadmap, dividers, text slides and end page:** 2–5, 9, 15, 26, 27, 38, 40, 44–47, 69 and slide 70, the OCW end
  page.

### What was rendered, and what was not — lecture 13

Rendered (9): slides 5, 6, 9, 10, 11, 13, 14, 17 and 20 — lecture 7's three perspectives recalled, the network as a map through vector spaces, the bump-function construction, random
functions through the data, the triangle of correspondences, the Gaussian process picture, the random vector plotted as a function,
the sketched scatter clouds of nearby and distant inputs, and the network with random weight matrices.

Not rendered, and why:

- **Reuses an image a later deck excludes: 3** — the spectral-norm update with its tricks and the NanoGPT speedrun plot. The plot
  carries no notice here, but lecture 23's slide 7 draws the same image under "© @kellerjordan0 on X. All rights reserved"; the
  render, made in the lecture 13 run, was withdrawn in the lecture 23 run.
- **Excluded from OCW's licence (2 slides): 4 and 22** — slide 4's puppy and monkey photographs and slide 22's CIFAR-10
  photographs. Slide 22's scatter plots, the lecturer's own, go unshown with it; the slide file gives their axes, ranges and
  spreads.
- **Drawings a rendered slide repeats:** 7 (the network of slide 20, without its weight labels), 8 (the four data points of slides
  9 and 10, without their fits), 18 (its plot is slide 13's last row) and 25 (slide 20's network beside formulas the slide file
  reproduces).
- **Text and equations the slide file reproduces exactly:** 2, 15, 16, 21, 23, 24 and 26.
- **Title, dividers, references and end page:** 1, 12, 19, 27 (three titles beside book icons) and slide 28, the OCW end page.

### What was rendered, and what was not — lecture 14

Rendered (28): slides 2, 5, 10, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 26, 29, 36, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 52 and
53 — the three lectures beside the data-and-embedding box, the DALL-E 2 images, the classifier, the generator with dice, the
dice's questions, the knob, the procedural river and its code, Concept #1's map tiles, the direct and indirect approaches, the
one-hump and two-hump densities, the maximum-likelihood update and derivation, the filing cabinet's delta functions, the
contrastive-divergence panels, the word predictor, the next-word classifier, the pixel model, its loss, its training and sampling,
WaveNet, noise to images, images to noise, denoising, the denoiser's training pairs, three dice rolls, and diffusion beside
autoregression with Concept #3. All 28 wiki captions are hand-written: the twelve key figures were placed by hand, and the other
sixteen re-captioned after the script placed them.

Not rendered, and why:

- **Excluded from OCW's licence (7 slides): 6 and 54–59** — slide 6's DiffDock and MRI-to-CT figures and the GAN slides' flamingo
  images. The GAN slides' equations are in the slide file and the wiki.
- **Reuse of an excluded image (2 slides): 49 and 50** — Ho, Jain and Abbeel's diffusion diagram, excluded in Homework 5, under the
  deck's pixel-art overlays (see the hash sweeps above). Their equations are in the slide file and the wiki.
- **Build steps and repeats a rendered slide supersedes:** 11 (slide 12 adds the dice to it), 21 (slide 22 adds the curve), 33 (it
  redraws slide 29's figure) and 44 (slide 45 repeats its diagram and adds the filmstrip).
- **Text, equations and code the slide file reproduces exactly:** 4, 7, 8, 9 (the notation handout's excerpt), 17, 20, 25 (the
  filing-cabinet code), 27, 28, 30, 31, 32, 34, 35, 37 and 51 (the training algorithm).
- **Title, outline and end page:** 1 (its title image is DALL-E's, by its signature strip), 3 and slide 60, the OCW end page.

### What was rendered, and what was not — lecture 15

Rendered (22): slides 3, 4, 5, 7, 8, 9, 10, 11, 14, 16, 19, 22, 26, 27, 28, 29, 30, 31, 32, 36, 38 and 40 — the data-and-embedding
box, the prior transformed into the target, the encoder and the generator, an autoencoder's lumpy code space, the mixture of
Gaussians, the infinite mixture, the latent variable model with its river tiles, the VAE as a learner, the Monte Carlo objective with
its Colab frame, random sampling with its zero terms, random against importance sampling, the encoder's Gaussian, the two halves of
one importance sample, "Looks like an autoencoder!", autoencoder and VAE, the two training frames, training and sampling on river
tiles, the two latent directions, and the coloured map. All 22 wiki captions are hand-written: eight were placed by hand, after the
equation each slide's first citation introduces (the script would have put them between a colon and its equation), and the other
fourteen re-captioned after the script placed them.

Not rendered, and why:

- **Excluded from OCW's licence (3 slides): 1, 6 and 39** — the BigGAN images of slides 1 and 39 and slide 6's bird photographs.
- **Build steps and repeats a rendered slide supersedes:** 15 (slide 16 shades its useful sample and writes the sum), 18 (slide 19
  repeats its figure and equation beside random sampling), 20 and 21 (slide 22 completes the encoder's picture), 35 (it repeats
  slide 10's generator) and 37 (slide 38 adds the second direction).
- **Text, equations and the table the slide file reproduces exactly:** 2, 12, 13, 17, 23, 24 and 25 (whose small diagrams are
  copies of slides 22's and 27's), 33, 34 and 41.
- **End page:** slide 42, the OCW end page.

### What was rendered, and what was not — lecture 16

Rendered (29): slides 5, 6, 10, 13, 14, 32, 33, 36, 38, 40, 41, 42, 43, 45, 46, 49, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69 and 80
— the shirt's colour distributions, the per-pixel predictions, Andrew Ng's chart, the Sat2Map learner, the blurry map-to-aerial output,
labels to facades, the patch discriminator, the structured-objectives cartoon, the conditional GAN's aerial output, multiple possible
outputs and the generator with a latent, the moving-square frames, the conditional VAE in training, its caption and its sampling,
BicycleGAN's facades, the captioning transformer, ChatGPT's reading of it, the domain-translation diagram and its two directions, the
text-to-image VAE in training and sampling, the text-to-image diffusion slides, VQA, LLaVA and cycle consistency. All 29 wiki images
were placed by hand with hand-written captions; the script then placed all 29 in the slide file. Slide 69 is a screenshot of the LLaVA
paper's figure with no notice, rendered by the rule that only a notice excludes.

Not rendered, and why:

- **Excluded from OCW's licence (16 slides): 3, 7, 8, 11, 70–79, 81 and 82.**
- **Reuse of an excluded image (15 slides): 12** (slide 11's photograph) **and 18–31** (the pictures of slide 73, which repeats slide 30
  under a notice). The conditional-GAN build is therefore described in the slide file and the wiki in prose only.
- **Build steps and repeats a rendered slide supersedes:** 34 (it repeats slide 10), 35 (slide 36 adds the third curve), 37 (slide 14's
  two images under a new heading), 44 (slide 45 repeats its target and adds the caption), 47 and 48 (slide 46 with other samples).
- **Text, equations, screenshots of text and section slides the slide file reproduces exactly:** 2, 4, 9 and 15 (the learner, a diagram
  of text boxes), 16, 17, 39, 50, 51, 52, 53, 54 (GPT-3's review), 55, 58 (ChatGPT's answer), 59 and 83.
- **Title and end page:** slide 1 (the title, with an edges2cats drawing) and slide 84, the OCW end page.

### What was rendered, and what was not — lecture 17

Rendered (11): slides 4, 5, 22, 38, 40, 41, 45, 47, 49, 51 and 53: the standard setting and the shifted one, projected gradient ascent,
Salman et al.'s transfer charts, the two settings side by side, biodiversity against species records, the per-location class
distributions, the cross-city heat maps, perturbing points against perturbing the distribution, the DRO ball, and Hashimoto et al.'s
DRO distribution. All 11 wiki images were placed by hand with hand-written captions; the script then placed all 11 in the slide file.
Slides 38 and 53 reproduce third-party figures with a credit and no notice, and are rendered by the rule that only a notice excludes.

Not rendered, and why:

- **Excluded from OCW's licence (31 slides): 2, 6, 10–18, 20, 21, 26–30, 32–37, 42–44, 46, 55, 57 and 59.** The adversarial examples,
  the shortcut examples, WILDS and the remedies' figures therefore exist here as prose only.
- **Reuse of an excluded image:** none.
- **A repeat a rendered slide supersedes:** 50 (its equations are in the slide file, and slide 51 redraws its circle diagram with the true
  distribution added).
- **Text, equations, screenshots of text and section slides the slide file reproduces exactly:** 3, 7 (the first page of "Concrete
  Problems in AI Safety"), 8, 9, 19, 23, 24, 25, 31, 39, 48, 52, 54, 56, 58, 60, 61 and 62.
- **Title and end page:** slide 1 (the title, with the two maps slide 41 shows) and slide 63, the OCW end page.

### What was rendered, and what was not — lecture 18

Rendered (26): slides 3, 8, 10, 21, 26, 27, 28, 30, 31, 35, 37–43, 45, 46, 59, 66, 67, 68, 70, 73 and 75: Andrew Ng's chart, the
inputs-to-outputs box, pretrain–adapt–test on music, unsupervised domain adaptation, the finished student–teacher recipe, ALDI and its
five comparisons, the cross-entropy and distillation bars, the ensemble, contrastive representation distillation's heat maps, Optimal
Brain Compression, the shark specialist, Poussin's painting, the compute and parameter charts, the BLOOM life cycle and emissions table,
performance on 58 tasks, learner and adaptor, the chain-of-thought bars, visual prompting, the pink pixel, adaptation on inputs, mapping
or outputs, inpainting prompts, VQGAN with CLIP and DALL-E 2. Seventeen wiki images were placed by hand with hand-written captions
(3, 8, 10, 26, 28, 30, 31, 37, 41, 42, 43, 45, 46, 66, 67, 68, 75), the script placed the other nine, and three of its captions (27, 39,
73) were rewritten; the script placed all 26 in the slide file. Slides 3, 37, 41, 42 and 45 reproduce third-party charts with a citation
or none, and no notice, and are rendered by the rule that only a notice excludes.

Not rendered, and why:

- **Excluded from OCW's licence (21 slides): 4, 5, 6, 12–17, 19, 20, 33, 34, 52, 53, 57, 60–63 and 69.** The few-shot figures, the
  pretraining photographs, DANN, cross-modal distillation, in-context learning and CLIP therefore exist here as prose only.
- **Reuse of an excluded image:** 74 (CLIP+GAN), whose encoder graphics are slide 63's.
- **Build steps a rendered slide supersedes:** 9 and 29 (slide 8 with one item bolded), 22–25 (earlier stages of slide 26), 36 (slide
  37 without its heat maps) and 65 (slide 66's first panel).
- **Text, tables and screenshots of text the slide file reproduces exactly:** 2, 7, 11, 18, 32, 44, 47–51, 54, 56, 58, 71 and 76.
- **Section slides, title and end page:** 55, 64 and 72; slide 1; slide 77, the OCW end page.

### What was rendered, and what was not — lecture 19

Rendered (15): slides 3, 4, 6, 10, 18, 22, 35, 37, 38, 39, 40, 43, 45, 47 and 48: the inputs-to-outputs banner, data to data++, the
generative model with its latent controls, Data2Data++, the VAE's coloured latent map, labels in latent space, the multiview paper's
dataset against generator, a positive pair made in latent space, SimCLR views against latent views (two slides), the contrastive results
(43.9%, 35.7%, 42.6%), the three levels of meta-learning, MAML's algorithm and figure, and the forward and backward pass with its meta
backward (two slides). Thirteen wiki images were placed by hand with hand-written captions (3, 4, 6, 10, 18, 22, 35, 37, 39, 40, 43, 45,
48), three of them first with guessed extensions, fixed from disk; the script placed the other two (38, anchored at first in the notation
paragraph by the range "slides 35–40" and moved by hand, and 47), and both captions were rewritten; the script placed all 15 in the slide
file. Slides 6 and 10 show the StyleGAN2 cat under its Nvidia Source Code License-NC credit; slides 40 and 45 reproduce third-party
figures with a citation and no notice, and are rendered by the rule that only a notice excludes.

Not rendered, and why:

- **Excluded from OCW's licence (20 slides): 7–9, 12–17, 19, 20, 24–28, 33, 34, 42 and 49.** BigGAN's birds, the latent directions, the
  StyleGAN and StyleSpace figures, DatasetGAN, StylEx, the retinal images, Dataset Interfaces, the volcano grid and TIML therefore exist
  here as prose only.
- **Reuse of an excluded image (5 slides):** 23 (slide 42's volcano grid), 29 and 30 (slide 28's cat photograph), 31 (the generator's
  versions of it) and 36 (lecture 18's excluded tigers). See the sweep above for the reasoning on 29–31.
- **Build steps a rendered slide supersedes:** 11 (the z-to-x picture of slide 6, after slide 10) and 21 (slide 22 without its labels).
- **Text, equations and code the slide file reproduces exactly:** 2, 5, 32 (OCW's placeholder), 41, 44, 46, 50 and 51.
- **Title and end page:** slide 1; slide 52, the OCW end page.

### What was rendered, and what was not — lecture 20

Rendered (20): slides 2, 5, 6, 12, 13, 14, 16, 17, 18, 19, 20, 26, 28, 33, 35, 41, 42, 44, 49 and 50: compute growth, GPT-4's predicted
loss and coding pass rate, Kaplan et al.'s three power laws, the joint law in data and parameters, the law in parameters and steps,
Rosenfeld et al.'s error surfaces and extrapolation, model shape, sample efficiency, train large and compress, Chinchilla's training-curve
envelopes and its predicted parameter counts, the toy model, intrinsic dimension against the exponent, GD against SGD, gradient noise,
the critical batch size, cross-modal alignment and the Platonic representation diagram. All 20 wiki images were placed by hand with
hand-written captions, after rendering, with paths copied from disk (all JPEG); the script placed all 20 in the slide file. Slide 50 is
credited CC BY; the others reproduce third-party charts with a citation, a URL or nothing, and no notice, and are rendered by the rule
that only a notice excludes.

Not rendered, and why:

- **Excluded from OCW's licence (6 slides): 3, 24, 40, 45, 47 and 48.** Sutton's essay, the gopher, dinosaur and chinchilla slide,
  OpenAI's Dota 2 graphic and the large-batch paper, the compute-against-time charts and the two kernel slides with their icons
  therefore exist here as prose only.
- **Charts a rendered slide already shows:** 10 and 11 (both in slide 12), 29 (slide 26's figure) and 37 (slide 10's chart).
- **Text, equations and tables the slide file reproduces exactly:** 4, 7, 8, 9, 15, 21, 22, 23, 25, 27, 30, 32, 34, 39, 43 and 51.
- **Section dividers:** 31, 36, 38 and 46.
- **Title and end page:** slide 1 (which also carries two Kaplan et al. charts shown on slides 12 and 14); slide 52, the OCW end page.

### What was rendered, and what was not — lecture 21

Rendered (18): slides 5, 12, 16, 19, 20, 36, 37, 38, 39, 41, 42, 47, 51, 72, 74, 79, 81 and 82: the perplexity table and the path to LLMs,
BERT's pretraining and fine-tuning, the probability of a sequence, the arms race, emergent abilities, sensitivity to ordering, label balance
and coverage, gold against random labels (two charts), one model per task against one prompted model, chain-of-thought prompting against
few-shot prompting, chain of thought's results and scale, the emergence of four methods with scale, instruction tuning's data, the
test-time failures, the smashed mirror, and why alignment is needed. The deck numbers only its first seven pages, so the renders were made
by PDF page with a scratchpad script using `extract_slide_images.py`'s settings (1400px, JPEG q85 or PNG, whichever is smaller), since
that script resolves through the printed-number map, which is wrong for this deck. `embed_slide_images.py` placed all 18 in the slide
file and in the wiki, after the wiki's citations were rewritten so that each image's first citation is its passage (no ranges, no
citation in the notation paragraph, list items turned into paragraphs); all 18 captions were then rewritten by hand. Slides 37–39, 47 and
51 are credited CC BY; slide 16's picture is "in the public domain"; slides 19, 20, 36, 72 and 81 reproduce third-party charts with a
citation or nothing, and no notice, and are rendered by the rule that only a notice excludes.

Not rendered, and why:

- **Excluded from OCW's licence (19 slides): 26–29, 31–35, 52–56, 58–60, 63 and 86.** Zhao et al.'s sentiment prompt, GPT-3's prompt
  figures and SuperGLUE charts, Kojima et al.'s zero-shot chain-of-thought figures and tables, the meme, the salary screenshots and the
  MIT News article therefore exist here as prose only.
- **Text, equations, tables and screenshots of text the slide file reproduces exactly (35):** 2, 6, 8–11, 13, 14, 17, 18, 21, 24, 25, 40,
  43, 46, 48–50, 57, 64–71, 80, 83 and 87–91.
- **Build steps or repeats a rendered slide supersedes:** 4 (in 5), 15 (slide 12's left panel with text), 73 (in 74), 75–78 (in 79).
- **Slide 7:** overlay text on faded copies of slide 6's images.
- **Outline slides:** 3, 22, 23, 44, 45, 61, 62, 84 and 85; **section break:** 30.
- **Title and end page:** slide 1; slide 92, the OCW end page.

### What was rendered, and what was not — lecture 23

Rendered (8): slides 2, 13, 14, 16, 17, 19, 34 and 41: the LEGO transformer, loose and assembled; the Lipschitz conditions over a network
and its Taylor expansion; norms induced from activation spaces to tensor spaces to the full weight space; modules, composition and
concatenation; addition, scalar multiplication and the residual block; a module's sensitivity to its inputs and weights; learning-rate
transfer across width and depth; and Shampoo's primitive in Modula, with its training curves. Rendered by `extract_slide_images.py` (the deck
is numbered 1:1); all eight wiki images were placed by hand with hand-written captions, because the page's overview cites slide ranges, and
`embed_slide_images.py` then placed all eight in the slide file. Slide 2's photographs, slide 34's chart (from the lecturer's paper with his
collaborators) and slide 41's chart carry no credit and no notice, and are rendered by the rule that only a notice excludes.

Not rendered, and why:

- **Excluded from OCW's licence (15 slides): 3–7, 24, 26–28, 31, 37, 42 and 45–47.** The LEGO renderings and heap, the loss surface and
  adversarial pig, Keller Jordan's plots, charts and photograph, Li et al.'s loss landscapes, the animal and supercomputer photographs, the
  paper's portraits, the mirror picture and slide 47's pictures therefore exist here as prose only. Slide 2's photographs, which are
  rendered, are not any of these images (the sweep compared them).
- **Text, equations, code and logos the slide file reproduces (19):** 9–12, 18, 20–23, 29, 30, 32, 33, 36, 39, 40 (a screenshot of
  pseudocode, beside equations), 43 and 48. Slide 39's two small diagrams repeat slide 16's series and parallel shapes.
- **A repeat of a rendered slide:** 38 (slide 14's drawing, item for item).
- **Title, dividers and end page:** slide 1; slides 8, 15, 25, 35 and 44; slide 49, the OCW end page.


## Build checklist, lectures 1–17

The finished lines of `TODO.md` for lectures 1–17, moved here verbatim on 2026-10-12 so that `TODO.md` holds open work only.

### Transcripts

- [x] 01 Introduction to Deep Learning — video 6FkRvTtUc-o (OCW human captions; light copy-edit, verbatim in original/)
- [x] 02 How to Train a Neural Net — video vidCX_dMCu0 (OCW human captions; light copy-edit, verbatim in original/)
- [x] 03 Approximation Theory — video ySaoWrv3T_Q (OCW human captions; light copy-edit, verbatim in original/)
- [x] 04 Architectures: Grids — video bxVkZ4M-hIE (OCW human captions; light copy-edit, verbatim in original/)
- [x] 05 Architectures: Graphs — video 0niIwb37nF0 (OCW human captions; light copy-edit, verbatim in original/)
- [x] 06 Generalization Theory — video EiO8BBa-xdc (OCW human captions; light copy-edit, verbatim in original/)
- [x] 07 Scaling Rules for Optimization — video VcGPE4s_oNw (OCW human captions; light copy-edit, verbatim in original/)
- [x] 08 Architectures: Transformers — video Q1HOKrNeh2M (OCW human captions; light copy-edit, verbatim in original/)
- [x] 09 Hacker's Guide to Deep Learning — video DC2Hw9DiLCg (OCW human captions; light copy-edit, verbatim in original/)
- [x] 10 Architectures: Memory — video IiHknRHA-Gk (OCW human captions; light copy-edit, verbatim in original/)
- [x] 11 Representation Learning: Reconstruction-Based — video QxOzQRtd440 (OCW human captions; light copy-edit, verbatim in original/)
- [x] 12 Representation Learning: Similarity-Based — video yUh1fEGGdl4 (OCW human captions; light copy-edit, verbatim in original/)
- [x] 13 Representation Learning: Theory — video -eC0-5mXHQg (OCW human captions; light copy-edit, verbatim in original/)
- [x] 14 Generative Models: Basics — video hJlrAHqGOS8 (OCW human captions; light copy-edit, verbatim in original/)
- [x] 15 Generative Models: Representation Learning Meets Generative Modeling — video 8zzfcYIELdo (OCW human captions; light copy-edit by Sonnet, checked in the parent; verbatim in original/)
- [x] 16 Generative Models: Conditional Models — video zaMcHuJwe1w (OCW human captions; light copy-edit inline, checked by script; verbatim in original/)
- [x] 17 Generalization: Out-of-Distribution (OOD) — video tjD9LIzIIek (OCW human captions; light copy-edit inline, checked by script; verbatim in original/)

### Crawl

- [x] Fetch course site index: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/
- [x] Download linked PDFs and slides (23 decks, 5 psets, hw5 solution, notation handout; 206MB, gitignored)
- [x] Write sources.md

### Slides (raw/slides/)

- [x] 01 Introduction to Deep Learning — mit6_7960_f24_lec1.pdf (81 pages; figure audit 6 pages)
- [x] 02 How to Train a Neural Net — mit6_7960_f24_lec2.pdf (81 pages; figure audit 8 pages)
- [x] 03 Approximation Theory — mit6_7960_f24_lec3.pdf (43 pages, handwritten; read by Sonnet, audited by Opus on 19 pages)
- [x] 04 Architectures: Grids — mit6_7960_f24_lec4.pdf (84 pages; read by Sonnet, audited by Opus on 24 pages)
- [x] 05 Architectures: Graphs — mit6_7960_f24_lec5.pdf (47 pages; read by Sonnet, audited by Opus on 27 pages)
- [x] 06 Generalization Theory — mit6_7960_f24_lec6.pdf (66 pages; read by Sonnet, audited by Opus on 30 pages)
- [x] 07 Scaling Rules for Optimization — mit6_7960_f24_lec7.pdf (32 pages, handwritten; read by Sonnet, audited by Opus on 24 pages)
- [x] 08 Architectures: Transformers — mit6_7960_f24_lec8.pdf (55 pages; read by Sonnet)
- [x] 08 — Opus figure audit on 33 pages applied (19 corrected)
- [x] 09 Hacker's Guide to Deep Learning — mit6_7960_f24_lec9.pdf (72 pages; read by Sonnet)
- [x] 09 — Opus figure audit on 34 pages applied (13 corrected)
- [x] 10 Architectures: Memory — mit6_7960_f24_lec10.pdf (69 pages; read by Sonnet)
- [x] 10 — Opus figure audit on 44 pages applied (10 corrected)
- [x] 11 Representation Learning: Reconstruction-Based — mit6_7960_f24_lec11.pdf (65 pages; read by Sonnet)
- [x] 11 — Opus figure audit on 43 pages applied (13 corrected; first auditor stopped by a session limit after 6 pages, a second did the other 37)
- [x] 12 Representation Learning: Similarity-Based — mit6_7960_f24_lec12.pdf (70 pages; read by Sonnet; 37 OCW notices, 35 of them drawn as glyph outlines the text layer cannot see)
- [x] 12 — Opus figure audit on 46 pages applied (26 corrected; both first auditors stopped by a session limit after 15 and 10 pages, two more did the other 21)
- [x] 13 Representation Learning: Theory — mit6_7960_f24_lec13.pdf (28 pages, handwritten; read by Sonnet; OCW notices on slides 4 and 22)
- [x] 13 — Opus figure audit on 25 pages applied (16 corrected; split between two agents, both finished)
- [x] 14 Generative Models: Basics — mit6_7960_f24_lec14.pdf (60 pages; read by Sonnet in two halves; OCW notices on slides 6 and 54–59)
- [x] 14 — Opus figure audit on 52 pages applied (32 corrected; split between three agents, two stopped by a session limit after every page but slide 45, which was checked against its render)
- [x] 15 Generative Models: Representation Learning Meets Generative Modeling — mit6_7960_f24_lec15.pdf (42 pages; read by Sonnet in two halves; OCW notices on slides 1, 6 and 39)
- [x] 15 — Opus figure audit on 39 pages applied (21 corrected; split between three agents, all finished)
- [x] 16 Generative Models: Conditional Models — mit6_7960_f24_lec16.pdf (84 pages; read by Sonnet in two halves; 16 OCW notices: slides 3, 7, 8, 11, 70–79, 81, 82)
- [x] 16 — Opus figure audit on 71 pages applied (19 corrected; split between four agents, all finished)
- [x] 17 Generalization: Out-of-Distribution (OOD) — mit6_7960_f24_lec17.pdf (63 pages; read by Sonnet in two halves; 31 OCW notices, all typed)
- [x] 17 — Opus figure audit on 49 pages applied (23 corrected; split between four agents, all finished)

### Wiki

- [x] wiki/01-introduction.md
- [x] wiki/02-how-to-train-a-neural-net.md
- [x] wiki/03-approximation-theory.md
- [x] wiki/04-architectures-grids.md
- [x] wiki/05-architectures-graphs.md
- [x] wiki/06-generalization-theory.md
- [x] wiki/07-scaling-rules-for-optimization.md
- [x] wiki/08-architectures-transformers.md
- [x] wiki/09-hackers-guide-to-deep-learning.md
- [x] wiki/10-architectures-memory.md
- [x] wiki/11-representation-learning-reconstruction-based.md
- [x] wiki/12-representation-learning-similarity-based.md
- [x] wiki/13-representation-learning-theory.md
- [x] wiki/14-generative-models-basics.md
- [x] wiki/15-generative-models-representation-learning-meets-generative-modeling.md
- [x] wiki/16-generative-models-conditional-models.md
- [x] wiki/17-generalization-out-of-distribution.md
- [x] Topic pages (cross-lecture concepts) for lecture 1
- [x] Topic pages for lecture 2: backpropagation, loss-landscapes, differentiable-programming new; seven pages extended
- [x] Topic pages for lecture 3: lipschitz-continuity, scaling-laws new; representational-power rewritten; MLP, activations, generalization, course-map extended
- [x] Topic pages for lecture 4: convolution, inductive-bias, skip-connections, neural-fields-and-positional-encoding new; MLP, activations, generalization, representation-learning, tensors-and-batching, differentiable-programming, representational-power, course-map extended
- [x] Topic pages for lecture 5: graph-neural-networks new; inductive-bias, convolution, representational-power, neural-fields-and-positional-encoding, multilayer-perceptron, backpropagation, course-map extended
- [x] Topic pages for lecture 6: generalization-and-double-descent and inductive-bias extended in depth; gradient-descent, loss-landscapes, representation-learning, multilayer-perceptron, convolution, graph-neural-networks, neural-fields-and-positional-encoding, representational-power, course-map extended
- [x] Topic pages for lecture 7: steepest-descent, second-order-methods, norms, scaling-rules new; gradient-descent, loss-landscapes, skip-connections, lipschitz-continuity, backpropagation, multilayer-perceptron, differentiable-programming, scaling-laws, course-map extended
- [x] Topic pages for lecture 8: transformers new; graph-neural-networks, convolution, neural-fields-and-positional-encoding, inductive-bias, skip-connections, multilayer-perceptron, softmax-and-cross-entropy, tensors-and-batching, representation-learning, course-map extended
- [x] Topic pages for lecture 9: normalization-layers, data-augmentation new; softmax-and-cross-entropy, tensors-and-batching, gradient-descent, inductive-bias, scaling-laws, generalization-and-double-descent, norms, convolution, transformers, course-map extended
- [x] Topic pages for lecture 10: recurrent-neural-networks, autoregressive-models new; transformers, convolution, backpropagation, loss-landscapes, skip-connections, softmax-and-cross-entropy, activation-functions, norms, multilayer-perceptron, graph-neural-networks, inductive-bias, course-map extended
- [x] Topic pages for lecture 11: autoencoders, self-supervised-learning, transfer-learning new; representation-learning, normalization-layers, activation-functions, softmax-and-cross-entropy, multilayer-perceptron, steepest-descent, transformers, autoregressive-models, skip-connections, convolution, course-map extended
- [x] Topic pages for lecture 12: metric-learning, contrastive-learning new; representation-learning, self-supervised-learning, data-augmentation, transfer-learning, softmax-and-cross-entropy, generalization-and-double-descent, inductive-bias, transformers, normalization-layers, autoencoders, course-map extended
- [x] Topic pages for lecture 13: gaussian-processes, kernel-methods new; inductive-bias, representation-learning, steepest-descent, scaling-rules, multilayer-perceptron, generalization-and-double-descent, representational-power, contrastive-learning, course-map extended
- [x] Topic pages for lecture 14: generative-models, energy-based-models, diffusion-models, generative-adversarial-networks new; autoregressive-models, softmax-and-cross-entropy, generalization-and-double-descent, representation-learning, contrastive-learning, autoencoders, inductive-bias, course-map extended
- [x] Topic pages for lecture 15: variational-autoencoders new; autoencoders, generative-models, representation-learning, generative-adversarial-networks, diffusion-models, energy-based-models, autoregressive-models, course-map extended
- [x] Topic pages for lecture 16: conditional-generative-models new; generative-models, generative-adversarial-networks, variational-autoencoders, diffusion-models, autoregressive-models, transformers, softmax-and-cross-entropy, inductive-bias, course-map extended
- [x] Topic pages for lecture 17: adversarial-examples, out-of-distribution-generalization new; generalization-and-double-descent, data-augmentation, transfer-learning, norms, transformers, gradient-descent, course-map extended
- [x] INDEX.md table of contents

### Images

- [x] raw/images/01-introduction/ — 20 images; OCW-excluded slides never rendered; slide 51 withdrawn at lecture 6 (same image as lecture 6's excluded slide 30)
- [x] raw/images/02-how-to-train-a-neural-net/ — 45 images; OCW-excluded slides never rendered
- [x] raw/images/03-approximation-theory/ — 18 images; the deck has no OCW exclusion notices
- [x] raw/images/04-architectures-grids/ — 31 images; 35 OCW-excluded slides and slides 50–52 (reused excluded photo) never rendered
- [x] raw/images/05-architectures-graphs/ — 19 images; 14 OCW-excluded slides never rendered
- [x] raw/images/06-generalization-theory/ — 28 images; 8 OCW-excluded slides and slide 62 (reuses lecture 4's excluded bird photo) never rendered
- [x] raw/images/07-scaling-rules-for-optimization/ — 9 images; 3 OCW-excluded slides (7, 25, 26) never rendered
- [x] raw/images/08-architectures-transformers/ — 34 images; 7 OCW-excluded slides (4, 6, 7, 8, 31, 45, 53) never rendered; no reuse of excluded images found by pixel hash across decks 1–8
- [x] raw/images/09-hackers-guide-to-deep-learning/ — 11 images; 21 OCW-excluded slides never rendered; slide 57 reuses slide 4's output pictures but not its excluded X-ray, so it is rendered
- [x] raw/images/10-architectures-memory/ — 32 images; 20 OCW-excluded slides never rendered; no reuse of excluded images found by pixel hash across decks 1–10
- [x] raw/images/11-representation-learning-reconstruction-based/ — 25 images; 22 OCW-excluded slides and slide 13 (lecture 1's excluded slide 73 figure) never rendered
- [x] raw/images/12-representation-learning-similarity-based/ — 8 images (slides 8, 22, 23, 35, 43, 49, 50, 51); 37 OCW-excluded slides never rendered (35 of their notices are glyph outlines the text layer misses); no reuse of excluded images found by pixel hash across decks 1–12 in either direction
- [x] raw/images/13-representation-learning-theory/ — 10 images (slides 3, 5, 6, 9, 10, 11, 13, 14, 17, 20); 2 OCW-excluded slides (4, 22) never rendered; hash sweep clean once page backgrounds are left out
- [x] raw/images/14-generative-models-basics/ — 28 images; 7 OCW-excluded slides (6, 54–59) and slides 49–50 (Homework 5's excluded diffusion diagram, found by extending the hash sweep to the problem sets) never rendered
- [x] raw/images/15-generative-models-representation-learning-meets-generative-modeling/ — 22 images; 3 OCW-excluded slides (1, 6, 39) never rendered; hash sweep clean in both directions, validated by rerunning it on lectures 11 and 14
- [x] raw/images/16-generative-models-conditional-models/ — 29 images; 16 OCW-excluded slides, slide 12 (slide 11's excluded photograph) and slides 18–31 (slide 73's excluded pictures, which slide 73 repeats from slide 30) never rendered
- [x] raw/images/17-generalization-out-of-distribution/ — 11 images; 31 OCW-excluded slides never rendered; hash sweep clean in both directions, validated by rerunning it on lectures 11 and 14
- [x] AGENTS.md — the Images conventions section

### Publish

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
- [x] Lecture 9: INDEX, AGENTS, kb.json, SEE_ALSO updated
- [x] Lecture 9: verify_kb.py clean, review read; commit and push
- [x] Lecture 10: INDEX, AGENTS, kb.json, SEE_ALSO updated
- [x] Lecture 10: verify_kb.py clean, review read; commit and push
- [x] Lecture 11: INDEX, AGENTS, kb.json, SEE_ALSO updated
- [x] Lecture 11: verify_kb.py clean, review read; commit and push
- [x] Lecture 12: INDEX, AGENTS, kb.json, SEE_ALSO updated
- [x] Lecture 12: verify_kb.py clean, review read; commit and push
- [x] AGENTS.md cut to the rules every build follows (101 KB to about 27 KB); the per-lecture record moved verbatim to BUILD_LOG.md, image attribution to LICENSE.md, the chat's image rules into INDEX.md
- [x] Lecture 13: INDEX, AGENTS (image-table row), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 13: verify_kb.py clean, review read; commit and push
- [x] From lecture 13 on: per-lecture details go to BUILD_LOG.md and attribution to LICENSE.md; AGENTS.md gets only new general lessons and the image-table row
- [x] Lecture 14: INDEX, AGENTS (coverage, image-table and withheld rows, problem-set sweep lesson), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 14: verify_kb.py clean, review read; commit and push
- [x] Lecture 15: INDEX, AGENTS (coverage, image-table row, two general lessons), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 15: verify_kb.py clean, review read; commit and push
- [x] Lecture 16: INDEX, AGENTS (coverage, image-table and withheld rows, two general lessons), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 16: verify_kb.py clean, review read; commit and push
- [x] Lecture 17: INDEX, AGENTS (coverage, image-table row), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 17: verify_kb.py clean, review read; commit and push

## Build checklist, lecture 18

The finished lines of `TODO.md` for lecture 18, moved here verbatim on 2026-10-13 once the run was committed and pushed.

### Transcripts
- [x] 18 Transfer Learning: Models — video tNfuZ9Imt3M (OCW human captions; light copy-edit inline, checked by script; verbatim in original/)

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
- [x] Lecture 18: check_math and verify_kb.py clean, review read; commit and push
- [x] Move this run's finished lines to BUILD_LOG.md

## Build checklist, lecture 19

The finished lines of `TODO.md` for lecture 19, moved here verbatim on 2026-10-13 once the run was committed and pushed.

### Transcripts
- [x] 19 Transfer Learning: Data — video RUdQMHV-7KM (OCW human captions; verbatim in original/; light copy-edit inline, checked by script)

### Slides
- [x] 19 — mit6_7960_f24_lec19.pdf (52 pages, 1:1, page 52 the OCW end page prints 52; read by Sonnet in two halves)
- [x] 19 — OCW notices from the slide file, cross-checked with the text layer (20 pages) and a vector scan (no outlined small print)
- [x] 19 — Opus figure audit on 44 pages applied (19 corrected; split between four agents, all finished)

### Wiki
- [x] wiki/19-transfer-learning-data.md
- [x] Topic pages for lecture 19: generative-models-as-data and meta-learning new; transfer-learning, generative-models, generative-adversarial-networks, contrastive-learning, data-augmentation, foundation-models-and-prompting, backpropagation, variational-autoencoders, out-of-distribution-generalization, recurrent-neural-networks, inductive-bias, autoregressive-models, representation-learning, course-map extended

### Images
- [x] raw/images/19-transfer-learning-data/ — 15 images; 20 OCW-excluded slides and 5 reusing excluded images (23, 29–31, 36) never rendered; hash sweep both directions, validated on lectures 11 and 14; no earlier image withdrawn

### Publish
- [x] Lecture 19: INDEX, AGENTS (coverage, image-table row, withheld rows, two lessons), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 19: check_math and verify_kb.py clean, review read; commit and push
- [x] Move this run's finished lines to BUILD_LOG.md

## Build checklist, lecture 20

The finished lines of `TODO.md` for lecture 20, moved here verbatim on 2026-10-14 once the run was committed and pushed.

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
- [x] Move this run's finished lines to BUILD_LOG.md

## Build checklist, lecture 21

The finished lines of `TODO.md` for lecture 21, moved here verbatim on 2026-10-15 once the run was committed and pushed.

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
- [x] Move this run's finished lines to BUILD_LOG.md

## Build checklist, lecture 23

The finished lines of `TODO.md` for lecture 23 (catalog position 22), moved here verbatim on 2026-10-15 once the run was committed and pushed.

### Transcripts
- [x] 23 Metrized Deep Learning — video zBvsoxC6tAo (OCW human captions; verbatim in original/; light copy-edit inline, checked by script: 87 markers and 18 numbers identical, ratios 0.991–1.010; 7 restorations, 8 [Ed:] notes; the recording covers the whole deck, and OCW's transcript PDF ends at the same words)

### Slides
- [x] 23 — mit6_7960_f24_lec23.pdf (49 pages; typed Google Slides, not handwritten; slide N = page N, bottom-centre numbers 1–49 match pages, page 49 the OCW end page, stored rotated, which slide_number_map.py reports as printing none; thirty pages also print a larger bottom-right number from an earlier talk deck) — read by Sonnet in two halves; headings 1–49 checked by script
- [x] 23 — OCW notices from the slide file (3–7, 24, 26–28, 31, 37, 42, 45–47), cross-checked with the text layer (same fifteen) and a vector scan (no outlined small print)
- [x] 23 — Opus figure audit on 44 pages applied (26 corrected, 15 precisions only; split between three agents, all finished; 123 bolded symbols on slides 9–24 made plain as printed); renders of slides 14 and 34 checked

### Wiki
- [x] wiki/23-metrized-deep-learning.md (whole recording, with timestamps)
- [x] Topic pages for lecture 23: new modular-norm page; steepest-descent, norms, lipschitz-continuity, second-order-methods, scaling-rules, gradient-descent, adversarial-examples, normalization-layers, loss-landscapes and skip-connections extended; course-map extended; lecture 7's pointer updated

### Images
- [x] raw/images/23-metrized-deep-learning/ — 8 images (2, 13, 14, 16, 17, 19, 34, 41); 15 OCW-excluded slides never rendered; sweep both directions with the withheld list brought up to date, validated on lectures 11 and 14; full re-sweep over lectures 1–21 withdrew lecture 13's slide 3 (lecture 23's slide 7 excludes its chart) and cleared lecture 11's slide 60; wiki images hand-placed, slide-file images by script

### Publish
- [x] Lecture 23: INDEX, AGENTS (coverage, image-table and withheld-table rows, handwritten-deck prediction corrected, lessons: two numberings on one page, rotated end page, symbols as printed, a withheld slide's other images, typed decks' fonts), BUILD_LOG, LICENSE, kb.json, course-map updated
- [x] Lecture 23: check_math (offline and --github) and verify_kb.py clean, review read; commit and push
- [x] Move this run's finished lines to BUILD_LOG.md
