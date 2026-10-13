# Licence and attribution

This repository is a study aid compiled from **MIT 6.7960 Deep Learning, Fall 2024**, as
published on MIT OpenCourseWare:
<https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/>.

Instructors: Phillip Isola, Sara Beery, and Jeremy Bernstein.

## The short version

Everything in this repository is offered under **Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International** (`CC BY-NC-SA 4.0`),
<https://creativecommons.org/licenses/by-nc-sa/4.0/>, which is the licence MIT
OpenCourseWare publishes this course under. Reuse it non-commercially, with attribution,
and share adaptations under the same licence.

**Except** the third-party figures that OCW itself excludes from its licence (next
section). None of those images is reproduced in this repository.

## What comes from the course, and on what terms

- `raw/transcripts/` — adapted from OCW's published caption files for the lecture videos.
  `raw/transcripts/original/` is the caption text regrouped into timestamped paragraphs;
  the files directly in `raw/transcripts/` add a light copy-edit with every change marked.
- `raw/slides/` — a slide-by-slide transcription of the lecture decks, with figures
  described in prose.
- `raw/images/` — whole-slide renders of deck pages, unmodified apart from scaling.

These are adaptations of CC BY-NC-SA material and carry the same licence. Attribution:
*MIT 6.7960 Deep Learning, Fall 2024, MIT OpenCourseWare, <https://ocw.mit.edu>, licensed
CC BY-NC-SA 4.0.* OCW's citation and terms of use: <https://ocw.mit.edu/terms>.

## Third-party content that is excluded

OCW marks some figures inside the decks with a notice such as *"© source unknown. All
rights reserved. This content is excluded from our Creative Commons license"*. OCW
includes those under its own fair-use determination
(<https://ocw.mit.edu/help/faq-fair-use/>); that determination is OCW's and does not pass
to anyone who copies the deck.

So **no slide carrying such a notice is rendered into `raw/images/`**. Those slides are
transcribed in `raw/slides/` as text with the figure described in prose, and each carries
an `*OCW notice: …*` line naming the rights holder. Which slides they are, lecture by lecture, is listed in
[`BUILD_LOG.md`](BUILD_LOG.md#ocw-exclusion-notices-by-lecture).

## The compilation's own writing

`INDEX.md`, `AGENTS.md`, `BUILD_LOG.md`, `TODO.md`, `kb.json`, `sources.md`, `SEE_ALSO.md`, this file, and
the explanatory prose in `wiki/` are original to this compilation (© 2026 chaimantec).
Because `wiki/` explains, quotes and adapts the course, it is released under the same
CC BY-NC-SA 4.0 licence rather than a more permissive one.

This repository is not affiliated with or endorsed by MIT or the course staff.

## Attribution of the rendered slide images

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
OCW did not flag it. Slide 70 (CLIP+GAN) was rendered until lecture 18's deck excluded its "Text Encoder" and "Image
Encoder" graphics ("© Radford, et al.", lecture 18's slide 63); it is no longer rendered.

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

Lecture 10's deck names Sara Beery as speaker. Several of its rendered slides carry material from elsewhere. **Slides
33–39** are the LSTM diagrams, each credited "[Slide derived from Chris Olah:
http://colah.github.io/posts/2015-08-Understanding-LSTMs/]"; OCW did not flag them. **Slide 62** is credited "Image
courtesy of J. Alammar. Used under CC BY-NC-SA." (the RETRO diagram), and **slide 64** "Courtesy of Mangalam, et al.
Used under CC BY." (certificate lengths of video datasets). **Slides 47–53** show an uncredited chemical-structure
drawing, which the lecturer calls a caffeine molecule (≈52:03). **Slide 27** is a version of lecture 2's
parameter-sharing diagram, and **slides 41–42** repeat lecture 8's autoregressive-model slides. Every photograph in the
deck is on an excluded slide.

Lecture 11's deck names Phillip Isola as speaker. Its rendered slides are mostly the lecturer's diagrams and plots, and a
few carry material from elsewhere. **Slide 59** is the masked-autoencoder figure, credited "Courtesy of He, et al. Used
under CC BY." **Slide 60**'s left panel is an uncredited figure of BERT pre-training; the citation printed on the slide,
"[He, Chen, Xie, et al. 2021]", repeats slide 59's masked-autoencoder paper. **Slide 61**'s chart cites "[Zhang, Isola,
Efros, ECCV 2016]", a paper of the lecturer's, whose photographs on slides 53 and 54 are excluded. **Slides 9–12**'s
mapping figures print no credit, and slide 12 links the Colab notebook that made them; slide 13's figure in the same style
is the one lecture 1's deck credits to Torralba, Isola and Freeman, and it is not rendered. **Slides 42–43**'s coloured
shapes and **slides 46–47**'s scatter plots are uncredited. Every photograph in the deck is on an excluded slide.

Lecture 12's deck names Sara Beery as speaker. Its seven rendered slides all carry material from published papers, four of
them under an open-licence credit. **Slide 8** prints "Courtesy of Chuang, et al. Used under CC BY." (Chuang et al., Measuring
generalization with optimal transport, 2021). **Slides 22 and 23** print "Courtesy of Song, et al. Used under CC BY-NC-SA."
(the lifted-structure embedding of CUB-200-2011 birds). Slide 35, which prints "Courtesy of Tian, et al. Used under CC BY-NC-SA."
(CMC), was rendered until lecture 18's deck excluded the classroom photograph and depth map it shows ("© Couprie, et al.",
lecture 18's slide 33); it is no longer rendered. **Slide 43**'s two scatter plots are credited "figures: Wang & Isola, 2020" with no licence line and no notice, while
slides 31 and 41, also from Wang and Isola, are excluded ("© Wang and Isola" and "© source unknown"); slide 43 is rendered by
the rule that only a notice excludes, and it is the nearest case in this KB to that line. **Slides 49 and 50** show the SimCLR
framework diagram and a projection-head bar chart with no credit printed, and **slide 51**'s batch-size chart is credited
"(Figure from Chen et al. 2020)", also with no licence line. Every photograph in the deck is on an excluded slide.

Lecture 13's deck is Jeremy Bernstein's handwritten notes, and nine of its ten rendered slides are entirely his handwriting and
drawings. **Slide 3** pastes a plot titled "NanoGPT speedruns", from Keller Jordan's NanoGPT speedrun posts, credited on the slide
only by the handwritten handle "@kellerjordan0" and carrying no OCW notice; it is rendered by the rule that only a notice
excludes. Lecture 7's deck reproduces other plots from the same author's posts, and OCW excluded those, so this is the nearest
case in this KB to that line after lecture 12's slide 43. The photographs in the deck are on its two excluded slides, 4 and 22.

Lecture 14's deck names Phillip Isola as speaker, and most of its 28 rendered slides are his own diagrams and pixel art: the
cartoon birds, dice, knob and trapezoids, the procedural map tiles, the density and energy plots, and the pixel-art robin of the
autoregressive and diffusion slides. Slide 2's figure is the one lecture 11's deck uses for the two directions through a network.
Two rendered slides carry material from elsewhere, with no OCW notice. **Slide 5** is a screenshot of eight images from OpenAI's
DALL-E 2 page, credited "https://openai.com/dall-e-2/" and "Created with DallE.". **Slide 42** reproduces WaveNet's diagram of its
layers of nodes, credited "[Wavenet, https://deepmind.com/blog/wavenet-generative-model-raw-audio/]". Both are rendered by the rule
that only a notice excludes. The deck's other third-party figures are on its excluded slides, 6 (DiffDock, Corso et al., and an
MRI-to-CT model, Wolterink et al.) and 54–59 (the GAN slides' flamingo images), and on slides 49 and 50, which are withheld because
their chain is Ho, Jain and Abbeel's diffusion diagram, which OCW excludes where Homework 5 prints it.

Lecture 15's deck names Phillip Isola as speaker, and every one of its 22 rendered slides is his own material: the blob, circle and
trapezoid diagrams of the encoder, generator and mixture components, the Colab plots of an infinite mixture of Gaussians and of a
two-dimensional VAE in training (slides 14, 31, 32 and 40, by the transcript the lecturer's own Colab), and the river map tiles drawn
by the procedural program he wrote for lecture 14 and the samples of a VAE trained on them (slides 10, 36 and 38). Slide 3's figure is
the one lectures 11 and 14 use for the two directions through a network. No rendered slide carries a credit or reproduces a published
figure. The deck's third-party images are all on its excluded slides: the bird photographs of slide 6 and the BigGAN images (Brock et
al., 2018) of slides 1 and 39, all under "© Torralba, Isola, and Freeman". Slide 10's resources also hold slide 6's photographs, drawn
entirely off the page, so its render shows none of them.

Lecture 16's deck names Phillip Isola as speaker, and many of its 29 rendered slides are his own diagrams: the T-shirt colour plots
(slides 5 and 6), the cartoon chart (36), the conditional-VAE diagrams (43, 45, 46), the captioning transformer (56), the
domain-translation and text-to-image diagrams (60–64), the pixel-art robins of the diffusion slides (65–67), VQA (68) and cycle
consistency (80). Several rendered slides reproduce material from elsewhere with no OCW notice, and are rendered by the rule that only a
notice excludes. **Slide 10** is Andrew Ng's "Why deep learning" chart ("[Slide credit: Andrew Ng]"). **Slides 13, 14 and 38** carry
map and aerial tiles, which the lecturer says come from Google Maps (≈20:08), with pix2pix's outputs (Isola et al. 2017, the lecturer's
own paper). **Slides 32 and 33** show pix2pix's facades, the labels' data credited "Data from [Tylecek, 2013]". **Slide 42** reproduces
frames from Babaeizadeh et al. (ICLR 2018), with predictions labelled "Finn et al. (2016)", and slides 43, 45 and 46 reuse its frames.
**Slide 49** shows BicycleGAN's facades ("[BiCycleGAN, Zhu et al., NeurIPS 2017]"), and slides 40 and 41 draw its label map. **Slide 57**
is a ChatGPT conversation ("Created with ChatGPT."). **Slide 69** is a screenshot of LLaVA's architecture figure ("Visual Instruction
Tuning", Liu, Li et al. 2023). The deck's other third-party figures are on its excluded slides, 3 (Long et al., Glass, Reed et al.,
AlphaFold), 7 (painmakers), 8 (color_byangelina), 11 (Fredo Durand's photograph), 70 (Esser et al.), 71 (the 1968 Licklider and Taylor
cartoon), 72–79 (Zhu et al.), 81 (Galanti, Wolf and Benaim) and 82 (Lample et al.), and on slides 12 and 18–31, which are withheld
because they reuse slide 11's and slide 73's excluded images.

Lecture 17's deck names Sara Beery as speaker. Of its 11 rendered slides, the distribution diagrams (slides 4, 5 and 40, built from
icons, with street photographs under "Training" and "Inference" on slides 4 and 5 that carry no credit), the projected-gradient sketch
(22), the DRO sketches (49 and 51) and the class-distribution charts of slides 45 and 47 carry no credit line. **Slide 41**'s two maps are
credited on the slide "Left: Courtesy of Mannion et al. Used under CC BY. Right: Courtesy of Sara Beery. Used under CC BY." Two rendered
slides reproduce material from papers with a credit and no OCW notice, and are rendered by the rule that only a notice excludes:
**slide 38**, the transfer-accuracy charts of Salman et al. (2020), "Do adversarially robust ImageNet models transfer better?", and
**slide 53**, a figure from Hashimoto et al. (2018), "Fairness without demographics in repeated loss minimization". The deck's other
third-party figures are on its 31 excluded slides: 2 (OpenAI and others), 6 (@hardmaru, @DimaKrotov, CNN.com), 10 and 21 (Aleksander
Mądry and Ludwig Schmidt), 11 (Papernot et al.), 12–15 (Brown et al.), 16–17 (Athalye et al.), 18 (Carlini and Wagner), 20 (Biggio and
Roli), 26 (Hongzhou Lin), 27 and 36 (Aleksander Madry), 28, 29 and 33 (sources unknown), 30 and 34 (Geirhos et al.), 32 (AI Weirdness),
35 (Liu et al.), 37 (Tsipras et al.), 42–44 (Koh et al.; Beery et al.), 46 (Weinstein et al.), 55 (Mac Aodha et al.), 57 (He et al.)
and 59 (Vendrow et al.); none of them is rendered.

Lecture 18's deck names Sara Beery as speaker. Of its 26 rendered slides, the deck's own diagrams carry no credit line (slides 8, 10, 30,
31, 35, 46 and 73), and slides 21 and 26 show crops of sonar images with no credit printed (the lecturer describes the sonar project as
her own research, ≈33:16). **Slide 3** is Andrew Ng's chart ("[Slide credit: Andrew Ng]"). Slides credited under an open licence:
**slides 27 and 28** ("Courtesy of Kay, et al. Used under CC BY.", Align and Distill), **slide 38** ("Courtesy of Frantar, et al. Used under
CC BY.", Optimal Brain Compression), **slide 39** ("Courtesy of Kuznedelev, et al. Used under CC BY."), **slide 43** ("Courtesy of Luccioni
et al. Used under CC BY-NC-SA."), **slide 59** ("Courtesy of Wei, et al. Used under CC BY.", chain-of-thought prompting), **slides 66–68**
("Courtesy of Bahng, et al. Used under CC BY.", visual prompting), **slide 70** ("Courtesy of Bar et al. Used under CC BY.") and **slide
75** ("Courtesy of Ramesh et al. Used under CC BY.", DALL-E 2). **Slide 40** shows Nicolas Poussin's *Blind Orion Searching for the Rising
Sun* (1658), "in the public domain". Four rendered slides reproduce material with a citation or none and no OCW notice, and are rendered by
the rule that only a notice excludes: **slide 37**, the heat maps of Tian, Krishnan and Isola (2020); **slide 41**, OpenAI's "AI and
compute" chart; **slide 42**, Ahmed and Thompson's (2022) chart; and **slide 45**, a chart of performance on 58 tasks with no source
printed. The deck's other third-party figures are on its 21 excluded slides: 4 and 5 (AAAS), 6, 12–17 and 53 (sources unknown), 19 (He
et al.), 20 (Ganin et al.), 33 (Couprie et al.), 34 (Aytar et al.), 52 (Brown et al.), 57 (Li and Liang; IEEE), 60–63 (Radford et al.)
and 69 (Jia et al.); none of them is rendered, and nor is slide 74, which reuses slide 63's excluded encoder graphics.

Lecture 19's deck names Sara Beery as speaker. Of its 15 rendered slides, the deck's own diagrams carry no credit line (slides 3, 4, 22
and 43; slide 4 also quotes Emad Mostaque on Stable Diffusion, with the Stability AI blog post as its source), and slides 47 and 48 draw
the forward and backward pass of lecture 2's deck. **Slide 18** is the two-dimensional VAE plot of lecture 15's slide 40, Phillip
Isola's own Colab, with no credit printed. **Slides 6 and 10** show a cat generated by StyleGAN2, credited on the slide: "Cat photo from
"StyleGAN2 — Official TensorFlow Implementation - Analyzing and Improving the Image Quality of StyleGAN", by Tero Karras, Samuli Laine,
Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila (https://github.com/NVlabs/stylegan2). Made available under the Nvidia Source
Code License-NC (https://nvlabs.github.io/stylegan2/license.html)." That licence is non-commercial, as this repository is. **Slides 35,
37, 38 and 39** are credited "Courtesy of Jahanian et al. Under under CC BY." (the slip as printed), from "Generative Models as a Data
Source for Multiview Representation Learning"; slide 35 includes the authors' portraits, and slide 39 includes SimCLR's grid of
augmentations. Two rendered slides reproduce material with a citation and no OCW notice, and are rendered by the rule that only a notice
excludes: **slide 40**, Jahanian et al.'s diagram and results, which prints no credit; and **slide 45**, the algorithm and figure of
MAML, cited "[Finn, Abbeel, Levine 2017]". The deck's other third-party figures are on its 20 excluded slides: 7–9, 16 and 17 (Brock et
al.), 12 (Chai et al.), 13, 14, 33 and 42 (sources unknown), 15 and 19 (Karras et al.), 20 (Wu et al.), 24–26 (Zhang et al.), 27
(Tritrong et al.; its X-ray figure is credited "Courtesy of Li et al. Used under CC BY."), 28 (Lang et al.), 34 (Vendrow et al.) and 49
(AAAI); none of them is rendered. Nor are five slides that reuse an excluded image: 23 (slide 42's volcano grid), 29 to 31 (slide 28's
cat photograph and the generator's versions of it) and 36 (the tiger photographs of lecture 18's excluded slides 12 and 13, credited CC
BY here). OCW removed slide 32's picture ("Image of people removed due to copyright restrictions.").

If a rights holder or the course asks for a page to come down, delete the image file and every
image embed that points at it.
