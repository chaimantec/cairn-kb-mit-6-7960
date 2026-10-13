# Generative models as data

A generative model trained on a dataset can be treated as more than a copy of it. Together with its latent space and a map back
into that space, it is "a new mechanism to access data, going beyond the capabilities of the initial training data set", which
lecture 19 calls **data++** ([lecture 19](19-transfer-learning-data.md), ≈3:08). Data++ can be interpolated, manipulated, composed and
optimized; moving in its latent space answers counterfactual questions; its samples can be labelled cheaply; it can explain what a
classifier has learned; and it can make the positive pairs of contrastive learning. In lecture 19 this is the course's answer to how
to transfer "knowledge about the inputs", the third kind of knowledge [lecture 18](18-transfer-learning-models.md) set out (see
[transfer learning](transfer-learning.md)). Covered so far: [lecture
15](15-generative-models-representation-learning-meets-generative-modeling.md), slides 37–40, ≈1:13:15–1:19:23 (latent directions
and seams); [lecture 17](17-generalization-out-of-distribution.md), slide 59, ≈57:40–58:27 (counterfactual images to diagnose a
classifier); lecture 19, slides 3–42, ≈1:33–1:06:58.

**Notation** (lecture 19's slides). A data point is $x$ and a latent variable $z$; $G$ maps latent variables to data and $G^{-1}$ maps
data back; $\mathbb{X}$ is the data++ object. $\mathcal{Z}$ is a GAN's input latent space and $\mathcal{W}$ StyleGAN's intermediate one.
$T_{\mathbf{x}}$ and $T_{\mathbf{z}}$ are transformations in data space and in latent space, and $F$ an encoder being learned.

## Why it could be worth more than the data

"You had data, and then you used it to train a model that mimics the data that you already had … so how does it actually help?"
(lecture 19, ≈2:21). Lecture 19's first answer is compression: Stable Diffusion was released as "a single file that compresses the
visual information of humanity into a few gigabytes" (slide 4, quoting Emad Mostaque), and the storage of all the images on the
internet against that of the model is "a significant difference in terms of what you're capturing" (≈3:57). Its second is that the
model can be operated on. Slide 5 defines

$$\mathbb{X} = \lbrace x, z, G, G^{-1} \rbrace$$

and four operators on such objects: interpolation, $\alpha \mathbb{X}_ 1 + (1 - \alpha)\mathbb{X}_ 2 \rightarrow \mathbb{X}_ 3$;
manipulation, $\mathbb{X}_ 1 + w \rightarrow \mathbb{X}_ 2$; composition,
$\mathbb{X}_ 1[m] + \mathbb{X}_ 2[1 - m] \rightarrow \mathbb{X}_ 3$; and optimization, $\arg\min_{\mathbb{X}} f(\mathbb{X})$, each with a paper, for "graphics, visualization, data aug,
counterfactual reasoning" (≈4:45–7:02).

Whether a generator can hold *more* information than its training data is, the lecturer says, an open "meta argument" in the
community. Information theory says no: "how could you possibly create something from nothing?" Against that, architectures and
training systems build in knowledge, as a convolutional network builds in local structure, and even random flipping "is an image
generation algorithm" that keeps the class (lecture 19, ≈39:45–42:01; see [inductive bias](inductive-bias.md) and [data
augmentation](data-augmentation.md)).

## Directions in latent space

A generator's latent variables are "the controls" (lecture 19, slide 6, ≈7:48). Lecture 15 first showed that directions in a
generator's latent space can match independent, nameable factors of variation, which it calls **disentanglement**: in a VAE trained on
river maps, one direction bends the river and another changes the grass colour, and in BigGAN one direction turns a bird and another
changes the background (lecture 15, slides 37–39, ≈1:13:15–1:17:06). Lecture 19 repeats the BigGAN example (slides 8 and 9) and adds
how such directions are found: map an image and the same scene at night into latent space, and take the vector between them; average
over many images for a direction that works on any input (≈14:02–14:50). Its examples are zoom, shift, brighten and darken (slide 13),
winter to spring, turning on the lights, day to night and a volcano erupting (slide 14), and word2vec's direction from "walking" to
"walked" (slide 13).

Such directions are easiest to find in a clean, low-dimensional case. A student asked whether they are linearly independent; in
practice, "particularly as these latent spaces get higher dimensional, being able to really explicitly disentangle these dimensions of
variation in a clean way can be very difficult and can be highly heuristic", and published directions are often "cherry picked"
(lecture 19, ≈9:25–10:13).

They also expose **bias in the training data**. Darkening a volcano makes it erupt, "because clearly, people are not taking pictures of
volcanoes in the dark that aren't erupting" (lecture 19, ≈14:50).

## Where to interpolate

Interpolating linearly between two images in data space leaves the manifold of natural images, giving overlays of the two (lecture 19,
slide 16). Interpolating in a latent space that maps onto the manifold stays on it, though the frames in between can be strange: "this
half bird, half fly thing" (slide 17, BigGAN, ≈18:48–19:34). But the input latent space is not always the best place. Lecture 15 showed
a VAE's latent space opening **seams**: "it's like you're mapping this complicated topology onto a sphere, and you're going to have cuts
and seams", so navigation hits "weird jumps" (lecture 15, slide 40, ≈1:18:37–1:19:23). Lecture 19 shows the same figure (slide 18) and
draws StyleGAN's conclusion: an intermediate layer's space, StyleGAN's $\mathcal{W}$, is less distorted than its input $\mathcal{Z}$,
and so "might be the right place to interpolate" (slides 15 and 19, ≈21:55–22:40). StyleSpace finds cleanly disentangled directions in
it, such as a car's hood style, headlights and body colour (slide 20).

## Counterfactuals and ensembles

With $G^{-1}$, a real image can be put into latent space and moved: "What would it look like if…?" (lecture 19, slides 10–11, ≈10:59).
Ensembling with Deep Generative Views (slide 12) classifies a real image together with generated variations of it and combines the
predictions, "almost like test time augmentation within the generator itself" (≈11:45–12:30). The assumption is that nearby variations
keep the category, which holds for "cat" but may not for a cat's breed (≈12:30–13:17).

## Labelling generated data

If nearby latent points share a label, a few labelled points can label the rest (lecture 19, slides 21–23, ≈23:29–25:02). DatasetGAN
(slides 24–26) trains a StyleGAN, labels a few generated images with part segmentations, trains a small classifier on the generator's
per-pixel features, and samples a large labelled synthetic dataset to train a segmentation model that is tested on real images: "1
labeled GAN image is worth ~100 labeled regular images!" (slide 26, ≈25:02–28:51). Slide 27 shows related work with one to ten manual
labels, and for chest X-rays.

## Explaining a classifier

StylEx (lecture 19, slides 28–31) maps an image into a generator's StyleSpace and finds "the top-K StyleSpace directions … that will
most affect which class the model predicts": for a cat-or-dog classifier, opening the mouth, the size of the pupils and the pointiness
of the ears (≈30:26–32:02). For a perceived-age classifier the directions found include eyebrows, skin tone, glasses and grey hair,
some of which raise questions of fairness; for a retinal fundus classifier, four attributes that doctors themselves look for (slides
32–33, ≈32:50–35:53). The catch is labour: every direction has to be found and interpreted by hand (≈35:53–36:41).

Text-conditioned diffusion models make counterfactuals easier to ask for. Dataset Interfaces (Vendrow et al., 2023) generates images
of plates with utensils and finds a classifier calling empty plates "tray" or "bucket" (lecture 17, slide 59, ≈57:40–58:27; lecture
19, slide 34, ≈36:41–37:28). Lecture 19 offers it as a way to probe a dataset's biases without looking through all the data
(≈1:02:16–1:03:03).

## Generated data for representation learning

Contrastive learning needs positive pairs, two views of the same thing. Instead of crops, flips and colour jitter in data space,
Jahanian, Puig, Tian and Isola (ICLR 2022) make views by moving a little in a generator's latent space (lecture 19, slides 35–39,
≈42:47–46:36; see [contrastive learning](contrastive-learning.md)). Trained only on generated data, an encoder reached 35.7% top-1
accuracy on ImageNet1000 linear transfer with data-space transformations alone and 42.6% with latent transformations added, against
43.9% from real data (slide 40, ≈56:02–56:48). "When the generative model is high quality, can even outperform learning from real
data" (slide 41).

Three caveats run through lecture 19's discussion:

- **Boundaries.** A small move can cross a class boundary. As long as that is rare, "it's almost like the noise will come out. It's
  almost a mechanism for regularization" (≈44:18). How far is safe is usually a hyperparameter tuned on test data (≈49:40), and
  generators make things "realistic looking but impossible", such as a robin with a grey chest (≈50:32).
- **Diversity against fidelity.** Views that are nearly identical teach less than views that differ, but asking a generator for more
  varied poses or scenes costs fidelity: "almost like a Pareto frontier" (≈46:36–48:55).
- **Priors.** Generators are "much better at changing texture than shape", and hard to push off their priors: ask for orioles and "it'll
  give you baseball players" (≈57:37–58:25).

Rare categories are hardest: "Because the thing is rare, it's hard to train a generator that does a good job of generating it"
(≈42:47). And training on generated data can feed back on itself: **model collapse**, where models trained on more generated than real
data lose "parts of your distribution that were captured in the real data" (≈1:06:13–1:06:58).

## Summed up

"Samples from implicit generative models act like decorated data, *data++*, with extra functionality. With special operators, you can
*interpolate*, *extrapolate*, *manipulate*, *compose*, *optimize*, and *label* these datapoints. Everything you can do with regular data,
you can also try with data++. It might work better!" (lecture 19, slide 42). "We need generators that can learn with less data, and we
need to better understand how these things are being learned" (≈1:01:30).

## See also

- [Lecture 19](19-transfer-learning-data.md) — the lecture, with its figures.
- [Generative models](generative-models.md) — the models themselves, and latent variables as control knobs.
- [Generative adversarial networks](generative-adversarial-networks.md) — BigGAN, StyleGAN and their latent spaces.
- [Variational autoencoders](variational-autoencoders.md) — disentanglement and seams in a VAE's latent space.
- [Contrastive learning](contrastive-learning.md) — positive pairs from latent views.
- [Data augmentation](data-augmentation.md) — hand-designed transformations, and generation as augmentation.
- [Transfer learning](transfer-learning.md) — knowledge about the inputs as one of three things to transfer.
- [Out-of-distribution generalization](out-of-distribution-generalization.md) — diagnosing failures with generated counterfactuals.
