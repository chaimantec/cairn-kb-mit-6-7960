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

Lecture 12's deck names Sara Beery as speaker. Its eight rendered slides all carry material from published papers, five of
them under an open-licence credit. **Slide 8** prints "Courtesy of Chuang, et al. Used under CC BY." (Chuang et al., Measuring
generalization with optimal transport, 2021). **Slides 22 and 23** print "Courtesy of Song, et al. Used under CC BY-NC-SA."
(the lifted-structure embedding of CUB-200-2011 birds). **Slide 35** prints "Courtesy of Tian, et al. Used under CC BY-NC-SA."
(CMC). **Slide 43**'s two scatter plots are credited "figures: Wang & Isola, 2020" with no licence line and no notice, while
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

If a rights holder or the course asks for a page to come down, delete the image file and every
image embed that points at it.
