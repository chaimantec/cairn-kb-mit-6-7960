# Lecture 19 — Transfer Learning: Data

**Lecturer:** Sara Beery ·
**Video:** [youtube.com/watch?v=RUdQMHV-7KM](https://www.youtube.com/watch?v=RUdQMHV-7KM) (76 min) ·
**Slides:** [`mit6_7960_f24_lec19.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec19.pdf)
(52 pages; the deck is titled "Lecture 19: Transfer Learning II"; transcribed slide by slide in [`raw/slides/19-transfer-learning-data.md`](../raw/slides/19-transfer-learning-data.md)) ·
**Transcript:** [`raw/transcripts/19-transfer-learning-data.md`](../raw/transcripts/19-transfer-learning-data.md)

## What this lecture establishes

The second half of the course's "short series on transfer learning" (≈0:00) takes up the third kind of knowledge that
[lecture 18](18-transfer-learning-models.md) left aside: knowledge about the **inputs**, "the distribution of inputs you're going
to expect" (≈1:33). Its main idea is that a good generative model is more than a copy of its training set. Together with its
latent space and an inverse map into it, the model is "a new mechanism to access data, going beyond the capabilities of the
initial training data set", which the lecture calls **data++** (≈3:08). Data++ can be interpolated, manipulated, composed and
optimized; directions in its latent space act as controls for counterfactual questions ("what would it look like if…"); its
samples can be labelled cheaply, because nearby latent points should share a label; it can explain what a classifier has learned;
and it can supply the positive pairs of [contrastive learning](contrastive-learning.md). Each use rests on assumptions about what
moving in latent space preserves, and the lecture keeps returning to where they break: class boundaries, biases baked into the
training data, and the trade-off between the diversity and the fidelity of what a generator makes.

The last part is about **meta-learning**, "models that learn to learn" (≈1:33): instead of asking how to adapt a pretrained model
quickly, it asks whether a model can be trained to be good at adapting. MAML learns an initialization from which a few steps of
SGD work well, by backpropagating through the SGD steps themselves, which is possible but expensive. Sequence models offer another
route: a recurrent network's state, or an autoregressive model's context, can act as a learner, and in-context learning "is kind of
just another mechanism of meta-learning" (≈1:13:58).

**Notation on this page** follows the slides. A data point is $x$ and a latent variable $z$ (bold, $\mathbf{x}$ and $\mathbf{z}$, on slides 6–11 and 35–40);
$G$ is a generator mapping latent variables to data, $G^{-1}$ the inverse map from data back to latent variables, and $\mathbb{X}$
the data++ object made of all four. $\mathcal{Z}$ is a GAN's input latent space and $\mathcal{W}$ StyleGAN's intermediate latent
space, whose points are $w$ (slide 5 also uses $w$, for a manipulation added to $\mathbb{X}$). In the contrastive-learning slides,
$F$ is the encoder being learned, $T_{\mathbf{x}}$ a transformation in data space and $T_{\mathbf{z}}$ one in latent space. In the
meta-learning slides, $\theta$ is a model's parameters and $f_{\theta}$ the model.

## Where this lecture sits

[Lecture 18](18-transfer-learning-models.md) sorted what can be transferred into knowledge about the mapping, the outputs and the
inputs, and covered the first two; slide 3 brings the list back with the third in bold. The generative models this lecture puts to
use were taught by Phillip Isola in lectures [14](14-generative-models-basics.md), [15](15-generative-models-representation-learning-meets-generative-modeling.md)
and [16](16-generative-models-conditional-models.md) ("we spent quite a few lectures with Phil", ≈1:33–2:21), and lecture 15 already
showed BigGAN's latent directions and the seams of a VAE's latent space, which this lecture builds on. The contrastive learning it
feeds is [lecture 12](12-representation-learning-similarity-based.md)'s; the Dataset Interfaces paper and the learned spatial prior
are [lecture 17](17-generalization-out-of-distribution.md)'s; the forward and backward pass it unrolls is
[lecture 2](02-how-to-train-a-neural-net.md)'s. This is the lecturer's "last lecture of the semester" (≈0:46). See [transfer
learning](transfer-learning.md) for the thread across lectures and the [course map](course-map.md) for the schedule.

## Knowledge about the inputs

Slide 3 is the transfer-learning banner of lecture 18 with "3. Knowledge about the inputs" in bold: a box from "Inputs" to
"Outputs", the mapping a network learns. "If you know something about the distribution of inputs you're going to expect, maybe
there's a way to actually use that to direct or accelerate your learning or even directly target a specific type of learning"
(≈1:33). The tool for capturing a distribution of inputs is a generative model, trained on a dataset to "generate data from that
same distribution" (≈2:21).

![Slide 3: the box from inputs to outputs beside the three kinds of knowledge to transfer, with knowledge about the inputs in bold](../raw/images/19-transfer-learning-data/slide-3.jpg)

*Slide 3 — The third kind of knowledge to transfer, which lecture 18 left for this lecture.*

## Generative models as data++

### Why a copy of the data could be worth more than the data

At first glance this seems pointless: "You had data, and then you used it to train a model that mimics the data that you already
had. But you already had the data, so how does it actually help?" (≈2:21). The answer the lecture develops is that a very good
generative model is a different way to *access* data. Slide 4 draws a dataset going through "Generative modeling" to a model, with
"data" under the first and "data++" under the second, and quotes Emad Mostaque on the release of Stable Diffusion: "This release is
the culmination of many hours of collective effort to create a single file that compresses the visual information of humanity into
a few gigabytes." A generative model "is a really cool form of data compression": compare "the storage capacity of all the images on
the internet versus the storage capacity of a Stable Diffusion model" (≈3:08–3:57).

![Slide 4: a dataset goes through generative modeling to a model, with data under the dataset and data++ under the model, above the Stable Diffusion quotation](../raw/images/19-transfer-learning-data/slide-4.png)

*Slide 4 — A dataset is data; a generative model trained on it is data++.*

### Data++ as an object

Slide 5, "Data++: making data a first class object", defines the object as four things together (≈4:45):

$$\mathbb{X} = \lbrace x, z, G, G^{-1} \rbrace$$

$x$ is the original training data; $z$ the latent variables that correspond to it; $G$ "the mapping that goes from some latent
variable to a generated data point", the generative model; and $G^{-1}$ "this inverse mapping that lets you go from data to its
associated latent variable". With these, "you can … define operators over those objects that enable us to do things to the
generator and the system of data plus generator of data that we couldn't easily do to the data itself" (≈5:30). Slide 5 lists four,
each with a paper:

- **Interpolation**, $\alpha \mathbb{X}_ 1 + (1 - \alpha)\mathbb{X}_ 2 \rightarrow \mathbb{X}_ 3$, between two data++ objects, for
  instance two datasets or two generators (≈5:30–6:17).
- **Manipulation**, $\mathbb{X}_ 1 + w \rightarrow \mathbb{X}_ 2$: adding "some bias term" to get "some whole new kind of object
  for accessing and interacting with data" (≈6:17). The two papers cited for interpolation and manipulation are Goetschalckx,
  Andonian, Oliva and Isola (ICCV 2019) and Jahanian, Chai and Isola (ICLR 2020).
- **Composition**, $\mathbb{X}_ 1[m] + \mathbb{X}_ 2[1 - m] \rightarrow \mathbb{X}_ 3$, which "is, of course, different from
  interpolating between two things" (≈6:17); Chai, Wulff and Isola (ICLR 2021).
- **Optimization**, $\arg\min_{\mathbb{X}} f(\mathbb{X})$: explicitly finding the data++ object "that's optimal for some given
  dimension of optimality" (≈7:02); Lin, Florence, Barron et al. (IROS 2021).

The applications slide 5 lists are "Graphics, visualization, data aug, counterfactual reasoning, …", and "all of this work is
kind of somewhat in progress" (≈7:02).

### Latent variables as controls

Slides 6 to 9 redraw the picture of a generative model from the generative-model lectures: latent variables
$\mathbf{z} \sim \mathcal{N}(0, 1)$, "the controls", go through $G$ to a synthesized image (≈7:48). Sampling a different latent point gives a different image, so
"given the relationship between dimensionality of movement in latent space and the image that's generated, you can think about this
as a mechanism of control" (≈8:37). Slides 8 and 9 show this with BigGAN's birds: one direction in latent space, "Bird orientation",
turns the bird from facing left to facing right, and another, "Background color", changes the background, so a grid of latent
points gives a grid of images. Those slides' photographs are excluded from OCW's licence, so they have no image here; lecture 15's
[disentanglement section](15-generative-models-representation-learning-meets-generative-modeling.md#disentanglement) discusses the
same figure. Directions like these correspond to "disentangled factors of variation" (≈8:37).

![Slide 6: latent variables z drawn from a standard normal feed a network G that synthesizes an image x of a cat](../raw/images/19-transfer-learning-data/slide-6.jpg)

*Slide 6 — The generative model as a map from latent controls to an image; the cat is a StyleGAN2 sample, credited on the slide.*

A student asked whether the two directions are assumed to be linearly independent, and whether they usually are (≈9:25). "The
assumption here is that there is some decomposable directions that are maybe some projection of this where you can find
orthogonality, and that orthogonality then corresponds to some very clean specific variables of variation in the real image space.
In practice, sometimes, particularly for certain types or certain categories of generative models, this is true, but it's not
necessarily easy to find." Papers tend to report a "cherry picked" direction that turns day into night, and "particularly as these
latent spaces get higher dimensional, being able to really explicitly disentangle these dimensions of variation in a clean way can
be very difficult and can be highly heuristic" (≈9:25–10:13).

### Counterfactuals

With $G^{-1}$, a real image can be mapped into the latent space (slide 10, "Data2Data++"), and then "you have the ability to ask
these interesting counterfactual questions. What would it look like if…?" — what happens to the image as you move around in latent
space (slides 10 and 11, ≈10:59).

![Slide 10: a photograph x of a cat is mapped by Data2Data++ to a point z in the latent disc, under the question "What would it look like if…"](../raw/images/19-transfer-learning-data/slide-10.jpg)

*Slide 10 — Data2Data++: the inverse map takes a real image into latent space, where counterfactuals can be explored.*

Slide 12 (Chai, Zhu, Shechtman, Isola and Zhang, CVPR 2021, "Ensembling with Deep Generative Views") uses this to classify better.
A real input image is mapped into latent space, the generator makes variations from nearby latent points, "different poses,
different orientations, slightly different realistic manipulations of that input image", a classifier $C$ scores each, and the
scores are combined into one prediction, with "Improved accuracy and robustness" (≈11:45). It is "almost like test time
augmentation within the generator itself" (≈12:30). The slide's photographs are excluded from OCW's licence.

The lecturer points out the assumption baked in: that "something nearby in that image space should be … maybe the same category,
not actually break a category boundary in some significant way". For "something like cat, this is probably very reasonable. If
you're trying to understand something about maybe a breed of cat, these dimensions of variation might start to get more confusing":
to her eye, two of the slide's variations "actually look like different breeds", differing in the length of the ears. "There's
always some fundamental assumptions about what you're actually trying to do that are built in to the dimensions of variation that
are good or bad for a given problem" (≈12:30–13:17).

### Finding directions in latent space

How do you discover such a direction? "You take an image of something. You take an image of the same thing at night, for example,
and then you look at where those both mapped to in latent space", and the vector from one to the other "should correspond to this
specific type of counterfactual". Averaging the directions found for many images gives one that is "input agnostic" (≈14:02–14:50).
Slide 13 shows directions found this way for zooming in and out, shifting, brightening and darkening (citing DCGAN, Radford, Metz
and Chintala, 2015), next to the classic example from word2vec (Mikolov et al., 2013), where one direction changes a verb's tense:
"walking" to "walked" and "swimming" to "swam" (≈15:35). Slide 14, "Color transformation vectors", adds "Winter to spring",
"Turning on the lights", "Day to night" and "Volcano!": "There is a latent space 'vector' for each of these transformations — the
'spring vector', the 'volcano exploding vector'". Both slides' photographs are excluded from OCW's licence.

The volcano is the lecture's running example of **bias in the training data**. Darkening an image of a volcano makes it erupt:
"Clearly, people are not taking pictures of volcanoes in the dark that aren't erupting. So when we try to make it dark, it thinks it
needs to erupt" (≈14:50–15:35); and "you're more likely to get a more aggressive looking eruptions at nighttime than during the
day" (≈16:23).

### Where to walk: the input latent space or an intermediate one

Walking in a straight line in a GAN's latent space "kind of works, but it can be really cherry picking": some directions give
"stuff that just looks like trash or is just not easily disentangled" (≈16:23–17:11). StyleGAN (Karras, Laine and Aila, 2018)
noticed that "any layer of the model, any of those intermediate representations, also represents a latent space for the model", and
that earlier layers are "better, quote, unquote, latent representations than the z representation" (≈17:11). Slide 15 sets a
traditional GAN, where the latent $z \in \mathcal{Z}$ enters the first layer, against StyleGAN, where a "Mapping network $f$" of
eight fully connected layers turns $z$ into $w \in \mathcal{W}$, which then steers every layer of a "Synthesis network $g$" through
"AdaIN" blocks, with noise added along the way.

Slides 16 and 17 explain why the choice matters. Interpolating linearly in **data space** between a photograph of a bird and one of
a fly leaves the manifold of natural images: the frames in between are "these obvious kind of just additive images", which do not
"match the statistics of real natural images" (slide 16, ≈18:00–18:48). Interpolating linearly in a **latent space** that maps onto
the manifold is a nonlinear path in data space that stays on it "the whole way" (slide 17, BigGAN, Brock et al. 2018). The frames
look more like natural images, "but they might not really map to reality. You get some really weird things like this half bird,
half fly thing where, basically, the model is just doing its best" (≈18:48–19:34). Both slides' photographs are excluded from OCW's
licence.

But "that z space … might not be the best way to organize the data to interpolate along" (≈19:34). Slide 18 goes back to "the
lecture from VAEs" ([lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md#seams-and-tendrils)): it
colours the data distribution and the latent distribution to show which parts correspond, and the recording plays the training
(≈20:20). The map is "almost like you've taken something and tried to crumple it into a ball", and although everything in the
latent space should land on the natural-image manifold, "there's danger zones" and "seams" where "things that are nearby in latent
space could potentially be quite far away from each other in data space", so a linear interpolation across a seam "could be
unnatural" (≈21:07–21:55).

![Slide 18: four panels of the two-dimensional example: data points on a Y-shaped region mapped to a disc, latent grid points mapped back, and both regions coloured by position](../raw/images/19-transfer-learning-data/slide-18.jpg)

*Slide 18 — The VAE example from lecture 15, coloured so that training shows which parts of the data and latent spaces correspond, and where seams open.*

This motivates StyleGAN's finding: an intermediate representation, "not quite fully convex, but maybe like a little bit less
distorted", where "you're possibly less likely to jump across a seam or fall into some danger zone", "might be the right place to
interpolate" (≈21:55). Slide 19 shows StyleGAN's figure: the distribution of features in the training set, the warped way $\mathcal{Z}$
maps onto it, and the less warped way $\mathcal{W}$ does. "Empirically and qualitatively, that's sometimes true. It's our favorite way
to say things, right? Sometimes" (≈22:40). Slide 20, StyleSpace (Wu, Lischinski and Shechtman, 2020), uses the $\mathcal{W}$ space
"to very cleanly isolate and disentangle these dimensions of variation" in pictures of cars: hood style, headlight, wheel angle,
background, sky colour and body colour (≈22:40–23:29). Slides 19 and 20 are excluded from OCW's licence.

## Labeling data++

If the real data is labelled, what about the generated data around it? "By construction, … things that are close together in this
manifold, should be semantically related. And so if you don't move too far on the manifold, you should argue that the category
should stay the same, or at least this is the assumption." Then "the normal inductive bias … in machine learning that assigns
similar labels to nearby points" can label the generated data too: small moves change "the context, … the scene, the
visualization, but it's not changing the category" (≈23:29–25:02). Slides 21 to 23 draw this as dots in a latent disc, a few
labelled and the rest taking the label of their cluster.

![Slide 22: a red latent disc with three clusters of black dots, one dot in each cluster coloured to show its label](../raw/images/19-transfer-learning-data/slide-22.png)

*Slide 22 — A few labelled points in latent space; slide 23 spreads each label to the rest of its cluster.*

**DatasetGAN** (Zhang, Ling, Gao, Yin, Lafleche, Barriuso, Torralba and Fidler, CVPR 2021; slides 24 to 26) does this for
semantic segmentation of the parts of a car, labels that "can be quite expensive" (≈25:02). "How many of you have ever actually
tried to label data for semantic segmentation? … It sucks. It's super slow" (≈26:33). The recipe on slide 24 has four steps: train
a StyleGAN; manually annotate a few generated images; use the labelled generator to sample "a large synthetic dataset" of images
with their part labels, "weak labels, but … explicitly based on this well-structured, well-aligned latent space"; and train a part
segmentation model on the synthetic data and test it on real images (≈25:48). Slide 25 shows how the labels are produced: the
"Style-Interpreter" stacks the generator's feature maps, upsampled, into a "Pixel feature vector" for each pixel, and a small
ensemble of classifiers maps those features to a part label. It relies on "the fact that training this generator forced the model
to learn a really well-behaved, well-aligned representation space", so that a lightweight model on its features works far better
than one on pixels (≈26:33–28:06). The result, printed on slide 26: "1 labeled GAN image is worth ~100 labeled regular images!"
(≈28:06). Slide 27 shows two related papers, segmentation with one to ten manual labels (Tritrong, Rewatbowornwong and
Suwajanakorn, CVPR 2021) and the same idea for lungs in chest X-rays (Li, Yang, Kries, Torralba and Fidler, CVPR 2021). Slides 24
to 27 are excluded from OCW's licence.

The same question, how to train efficiently on a well-learned representation space, has also been explored for foundation models
that are not generative. "These days, if you're trying to do semantic segmentation, most people start from something that's called
SAM, the Segment Anything Model", built on a large dataset made with "cute hacks for self-supervision" such as copy-paste
augmentation, and on ideas from DINOv2 (≈28:51–29:38; neither model is on a slide).

## Explaining a classifier

A classifier says "cat"; can data++ say *why*? Slides 28 to 31 follow "Explaining in Style: Training a GAN to Explain a
Classifier" (Lang, Gandelsman, Yarom, Wald, Elidan, Hassidim, Freeman, Isola, Globerson, Irani and Mosseri, ICCV 2021). The image is
mapped through an encoder into the generator's latent space and back (slide 30, $E$ and $G$), and then the latent variables are
perturbed "to understand how that would cause your prediction to change". **StylEx** "tries to find the top-K StyleSpace directions,
so these latent space directions of variation, that will most affect which class the model predicts" (slide 31, ≈30:26–31:15).
For a cat-or-dog classifier, opening the cat's mouth "is a really strong dimension of variation for moving the prediction from cat
to dog, probably because dogs tend to pant a lot more than cats do"; so are the size of the pupils and how pointy the ears are
(≈31:15–32:02). These also reveal bias in the training data: "more often than not, something that has long, pointy ears is a cat,
and maybe something with a shorter or folded over ears might be a dog", although plenty of cats have short ears (≈32:50). The
photographs on slides 28 to 31 are not rendered here (see the slide file).

For a "Perceived Age Classifier" (slide 32, whose picture OCW removed), the directions found were thicker eyebrows and lighter
skin for a younger age, and glasses or grey hair for an older one; some of these "would not be something you would want a model to
necessarily learn" and bear on fairness (≈32:50–34:22). For a "Retinal Fundus Classifier" (slide 33), the four attributes found,
"Exudates", "Cotton Wool", "Hemorrhages" and "Clustered Exudates", match "the types of even very fine grained features that a doctor
might use", which makes the method a way "to build trust in a model" (≈35:08–35:53). Its limitation is labour: "you have to
discover those latent space variations and then manually analyze … what they might correspond to", which surely "took a lot of
handcrafting and a lot of time" (≈35:53–36:41).

Recent work uses text conditioning of diffusion models instead, "really simple and interpretable mechanisms for control", though
"there's not always perfect alignment between maybe your text control and what actually gets generated" (≈36:41). Slide 34 shows
Dataset Interfaces (Vendrow et al., 2023), which [lecture 17](17-generalization-out-of-distribution.md) showed too: images of plates
"with utensils", generated to test a classifier, which it then labels "tray" or "bucket" when the plate is empty. Generating
human-chosen counterfactuals and measuring a model on them is "a really nice way to do, essentially, interfaces with your data
sets" (≈36:41–37:28). Slides 33 and 34 are excluded from OCW's licence.

## Learning representations from generated data

### Can a generator hold more information than its data?

"It is also useful for representation learning?" (≈37:28–38:14). Slide 35 introduces "Generative Models as a Data Source for
Multiview Representation Learning" (Jahanian, Puig, Tian and Isola, ICLR 2022): a dataset gives samples
$\mathbf{x} \sim \lbrace \mathbf{x}_ i \rbrace_{i=1}^{N}$, while a generative model gives $\mathbf{x} \sim G(T_{\mathbf{z}}(\mathbf{z}))$,
images made from a latent point moved by a transformation $T_{\mathbf{z}}$.

![Slide 35: a dataset of real images beside a generative model whose latent point z is moved by T_z, producing pictures of a shaggy dog in different poses](../raw/images/19-transfer-learning-data/slide-35.jpg)

*Slide 35 — Sampling from a dataset against sampling from a generator, where a transformation in latent space makes new pictures from one latent point.*

A student asked whether the generator is trained from scratch or is itself transfer learning (≈38:59–39:45). Assume from scratch,
the lecturer answers, "because otherwise, it gets even more complicated to understand", and this is the crux of "this kind of broader
meta argument going on in the community of whether it is possible to get more information out of a generator trained on data than
the underlying data itself" (≈39:45). From information theory, "how could you possibly create something from nothing?" The
counterargument is that "our architecture design, and the design of the training systems builds in knowledge": a convolutional
network says "we expect there to be local structure in images" (≈40:30–41:15). And "you could argue that random flipping is an image
generation algorithm", one guaranteed to keep the class; copy-paste augmentation, putting foreground objects on new backgrounds, is
closer still to a learned generator. "How is that any different from something like learning a generator that can interpolate maybe
in something closer to the image manifold? So it's a bit semantic" (≈41:15–42:01). See [inductive bias](inductive-bias.md) and [data
augmentation](data-augmentation.md).

The lecturer's own interest is rare categories, which puts generation in "a chicken and egg problem. Because the thing is rare,
it's hard to train a generator that does a good job of generating it". Papers that improve rare categories with generative models
"cheat, and they train the generative models on a lot of data"; gains so far come "in very specific constrained scenarios" (≈42:01–42:47).

### Positive pairs from latent space

Classical contrastive learning makes positive pairs with hand-designed transformations, "random cropping, flipping, … color jitter,
in a way that we know doesn't necessarily break the category boundary". Here, instead, "we're going to move just a little bit in
some latent space", so that the pair stays on the manifold of real images (≈42:47–43:32). Slide 36 recalls contrastive learning (two
views of a tiger close together in representation space, an elephant far away; its tiger photographs are the ones lecture 18's deck
excludes from OCW's licence, so it has no image here), and slide 37 makes the change: "Create positive pairs by transformations in latent space, rather than in data
space".

![Slide 37: two nearby points in a latent disc map to two photographs of a flower, whose encodings land on the same point in representation space](../raw/images/19-transfer-learning-data/slide-37.jpg)

*Slide 37 — A positive pair made by moving in latent space: two nearby latent points, two images, one representation.*

Slides 38 and 39 set "SimCLR views", crops, flips and colour distortions of one photograph, against "Latent views", images
generated from a ball of latent points around the real image's: "These are all different views of the same thing" (≈46:36).

![Slide 38: SimCLR views, two hand-transformed crops of one dog photograph, beside latent views of another dog, one generated from a nearby latent point and one the original photograph; both pairs are "two views of the same thing"](../raw/images/19-transfer-learning-data/slide-38.jpg)

*Slide 38 — SimCLR views against latent views: the objective is the same, two views of the same thing, but the second pair comes from the generator.*

![Slide 39: SimCLR's crops and colour distortions of one dog photograph, beside latent views of three subjects in different poses and backgrounds](../raw/images/19-transfer-learning-data/slide-39.jpg)

*Slide 39 — Hand-designed views against latent views; both are treated as views of the same thing.*

A student asked whether a small move can cross a class boundary; with a two-dimensional latent space and ten classes "there has to
exist a boundary somewhere" (≈43:32–44:18). Yes, the lecturer agrees, but "as long as, most of the time, it's not crossing a class
boundary … by the way we train these models, it's almost like the noise will come out. It's almost a mechanism for regularization",
as long as the area where small moves keep the category is large relative to the area where they cross it (≈44:18–45:05). It is like
the false negatives of ordinary contrastive learning, two dogs in one batch treated as different: removing them helped, "but the
model's still able to learn useful representations even when those do exist" (≈45:05–45:51).

### Diversity against fidelity

"It's probably more useful to know that two things that are more different from each other are, in fact, the same than it is to know
that two things that are almost identical to each other are, in fact, the same." In slide 39's middle row of latent views, the robin's
pose is "nearly identical" in every image, so "from a learning signal perspective, this might be less useful, arguably, than something
that also does something like random flipping" (≈46:36–47:23). Generators face "almost like a Pareto frontier … between diversity and
fidelity": forced to put the robin in a new pose or scene, the model complies, "but it probably won't actually be very robin-like
anymore" (≈48:08–48:55).

How far can you move before the class changes, and is it the same distance for every class? "In a lot of these papers, they're using
this as a hyperparameter": a radius in something like cosine distance, chosen by what "does the best on our test data". "That's a bit
disappointing, because I actually think that's a really interesting question" (≈48:55–49:40). Generators make things "realistic looking
but impossible": a mother with a third eye in the middle of her forehead (is it still your mom?), a robin with a grey chest instead of a
red one, "not a real bird" (≈50:32–51:19). Knowing where a class ends needs "representative data from all the dimensions of real
variation for that class", which is also "one of the reasons that few shot learning is so hard" (≈51:19).

A student suggested this is a shortcoming of classification itself, and that CLIP, which embeds text descriptions near the images they
describe, "can capture more truth about the world": an albino robin is still a robin (≈51:19–52:04). The lecturer connects this to the
philosophy of categorization, the idea that an object is "a list of all its attributes", which she calls "the CLIP approach, in a way";
her view is that "if you give me a … fixed list of attributes, and you try to map that to a category, you can basically come up with a
counterexample that breaks it", and that where to draw boundaries "has to be related to some goal": a green apple, a red apple, a green
bell pepper and a red bell pepper can be split by colour or by kind (≈52:53–54:30). "What do we really want from representations?"
(≈55:16).

### The result

Slide 40 gives the numbers, as top-1 accuracy of ImageNet1000 linear transfer (≈56:02–56:48):

| Training data | Transformations | Top-1 accuracy |
| --- | --- | --- |
| Real data (SimCLR, etc) | $T_{\mathbf{x}}$ | 43.9% |
| Generated data | only $T_{\mathbf{x}}$ | 35.7% |
| Generated data | $T_{\mathbf{z}}$ + $T_{\mathbf{x}}$ | 42.6% |

![Slide 40: contrastive learning from real data, with data-space transformations, against contrastive learning from a frozen generator with latent and data-space transformations, and their accuracies](../raw/images/19-transfer-learning-data/slide-40.jpg)

*Slide 40 — Contrastive learning from real data (43.9%) against from generated data, with data-space transformations only (35.7%) or latent ones as well (42.6%).*

"Neither of these are as good as just training on the real data themselves", but adding latent transformations is "a pretty
significant boost in performance" (≈56:48). Slide 41 concludes: "Deep generative views improve contrastive learning beyond only using
standard data augmentation. When the generative model is high quality, can even outperform learning from real data", which "of course,
means that we need the generator to be high quality, which it's not for many things that are uncommon" (≈56:48–57:37).

### What generators will not do

Generators have priors that are hard to break. Ask for orioles and "it'll give you baseball players"; ask for a toucan with a short beak
and "it just won't". They are "much better at changing texture than shape": a furry baseball is easy, a star-shaped one is not, and "it's
interesting to think about why shape might be encoded in a stronger way than texture" (≈57:37–58:25). A paper the lecturer describes as
just out at ECCV keeps a real scene's structure and swaps each object for another instance of its class ("instance augmented generation");
because its dataset has one class for all birds, it squashes an eagle into the shape of "a little round bird" (≈59:10–1:00:44). None of
these examples is on a slide.

## Data++, summed up

Slide 42: "Samples from implicit generative models act like decorated data, *data++*, with extra functionality. With special operators,
you can *interpolate*, *extrapolate*, *manipulate*, *compose*, *optimize*, and *label* these datapoints. Everything you can do with regular
data, you can also try with data++. It might work better!" (≈1:00:44–1:01:30; the recording's wording here is garbled, see the transcript).
"But I think there's still a lot of research that needs to be done … We need generators that can learn with less data, and we need to
better understand how these things are being learned" (≈1:01:30). The slide's grid of volcano images is excluded from OCW's licence, and
slide 23 draws the same grid, so it is not rendered either.

The questions that followed draw out the open problems. Biases too subtle to see in a large dataset can be probed by asking a generator
for counterfactuals in text, as Dataset Interfaces does (≈1:02:16–1:03:03). Unrealistic data can still help: guiding a generator with a
classifier can map out that classifier's boundaries, and "you get better performance on crows when you add in images where it's like
Bigfoot with a crow texture", but whether better test numbers mean better results "in the real world" is hard to say; "what is useful is
going to be directly related to your task" (≈1:03:03–1:05:25). The image domain is not the only one: there is work on generated text and
audio as training data, and older work trained on carefully handcrafted synthesizers of medical images (≈1:05:25–1:06:13). And there is
**model collapse**: as more of the internet's text is generated, models trained on more generated than real data can fall into "a vicious
cycle", losing "parts of your distribution that were captured in the real data" (≈1:06:13–1:06:58).

## Meta-learning

### Learning to learn

The last part asks "how do you learn a model that is good at transfer learning, a model that adapts quickly?" (≈1:06:58–1:07:46). Slide
43, "Meta-learning: learning to learn fast", draws three levels. **Inference** maps a data point $x^{(i)}$ through $f_{\theta}$ to a
prediction $y^{(i)}$. **Learning** maps a dataset $\lbrace x^{(i)}, y^{(i)} \rbrace_{i=1}^{N}$ through a learner to parameters $\theta$.
**Meta-learning** adds a level on top: many datasets $\lbrace \lbrace x^{(ij)}, y^{(ij)} \rbrace_{i=1}^{N_ j} \rbrace_{j=1}^{M}$ go
into a meta-learner, which outputs hyperparameters for the learner (≈1:07:46). The slide's label over the datasets is misspelled
"Datastets".

![Slide 43: three bands, meta-learning from many datasets to hyperparameters, learning from one dataset to parameters theta, and inference from a data point through f theta to a prediction](../raw/images/19-transfer-learning-data/slide-43.png)

*Slide 43 — Meta-learning adds a level above learning: a meta-learner trained on many datasets sets the learner's hyperparameters.*

### MAML

"One of the seminal works in this space" is **MAML**, model-agnostic meta-learning (Finn, Abbeel and Levine, 2017; ≈1:07:46). Slide 44
sets it up: "So far we saw transfer learning as: given some prior model/representation, how can we quickly adapt it? Could we *learn* to
do transfer learning? Consider finetuning: given an init, update it with SGD. MAML: *learn* an init such just a few steps of SGD will
update it very effectively" (the slide's "such just" as printed; ≈1:08:34). Algorithmically it is "a double loop in your optimization"
(slide 45, which pastes the paper's Algorithm 1 and covers its inner lines with two plain-language steps): sample a batch of tasks from a
distribution over tasks; for each task, "1. Starting with init, run a few steps of SGD on task i" and "2. Compute loss after these steps";
then update the initialization by backpropagating through those steps, "back to your initial meta-learning initial set of parameters"
(≈1:08:34–1:09:21). The paper's figure on the slide draws meta-learning as a thick curve ending at $\theta$; from a point on it, the
gradients of three tasks, $\nabla \mathcal{L}_ 1$, $\nabla \mathcal{L}_ 2$ and $\nabla \mathcal{L}_ 3$, point in different directions, and
from $\theta$ dashed lines of "learning/adaptation" lead to each task's own parameters $\theta_{1}^{\ast}$, $\theta_{2}^{\ast}$ and
$\theta_{3}^{\ast}$.

The overlays hide the algorithm's formulas, but the pasted image still holds them (the slide file gives them as the audit read them
from the image; they are not visible on the page). Line 6 adapts the parameters to task $\mathcal{T}_ i$ with one gradient step of
size $\alpha$ on its loss $\mathcal{L}_ {\mathcal{T}_ i}$, and line 8 updates the initialization with step size $\beta$ on the sum of
the adapted parameters' losses:

$$\theta_ i' = \theta - \alpha \nabla_{\theta} \mathcal{L}_ {\mathcal{T}_ i}(f_{\theta}) \qquad \theta \leftarrow \theta - \beta \nabla_{\theta} \sum_{\mathcal{T}_ i \sim p(\mathcal{T})} \mathcal{L}_ {\mathcal{T}_ i}(f_{\theta_ i'})$$

The gradient in the second depends on $\theta$ through $\theta_ i'$, which is why the outer loop has to backpropagate through the inner one.

![Slide 45: MAML's Algorithm 1 with its inner steps covered by the overlays "run a few steps of SGD on task i", "compute loss after these steps" and "init backprop through", beside the paper's diagram of a meta-learned theta and three task gradients](../raw/images/19-transfer-learning-data/slide-45.jpg)

*Slide 45 — MAML's double loop: an inner loop of SGD on each task, and an outer loop that backpropagates through it to the initialization.*

### Backpropagating through backward

"How to backprop through" the inner loop? (slide 46, ≈1:09:21). One step of SGD is a forward pass with parameters that were themselves
produced by a backward pass:

`forward(x, backward(forward(x, θ)) + θ)`

and two steps nest it again, `forward(x, backward(forward(x, backward(forward(x, θ)) + θ), θ) + θ)`. The lecturer recalls the forward
and backward pass through a small network (slide 47: linear, relu, linear and an $L_2$ loss, the example
[lecture 2](02-how-to-train-a-neural-net.md) used): "you could basically unfold it, and it was essentially one big forward pass of
going through forward and then backward of a model" (≈1:09:21–1:10:09). So "you can backprop through backward!" (slides 47 and 48).
Slide 48 lays the forward pass and the backward pass end to end as one long chain and adds a "meta backward" through all of it.

![Slide 47: the nested forward and backward expression, above the forward pass through linear, relu, linear and an L2 loss with its matrix pictures, and the backward pass with its gradients g](../raw/images/19-transfer-learning-data/slide-47.png)

*Slide 47 — The forward and backward pass through a small network, the pieces that one SGD step nests.*

![Slide 48: the forward pass through linear, relu, linear and L2 loss, followed by the backward pass drawn as its reflection, with a red arrow labelled meta backward running back through both](../raw/images/19-transfer-learning-data/slide-48.png)

*Slide 48 — Forward and backward together form one deeper forward computation, which a "meta backward" can differentiate.*

"So it's possible. It's just expensive": stretching out the nested loops gives "a very, very deep network that you're going to optimize
through" (≈1:10:09–1:10:55). See [backpropagation](backpropagation.md).

### In practice

"In practice, the gains of this method are not often worth the additional computational cost at training time (worse than just
pretraining?)" (slide 49, ≈1:10:55–1:11:40). But task-informed meta-learning helped one real problem: Tseng et al. (TIML, 2022)
"introduce a geospatial embedding that encodes task-specific context into meta-learning", "a bit of a task-specific tokenizer", for
classifying crop types such as maize in many countries. On the slide's chart for Kenya, maize against the rest, TIML scores the highest
AUC ROC at every training-set size, about 0.73 to 0.76, against about 0.70 for a pretrained model and about 0.65 rising to 0.69 for
MAML: meta-learning alone and pretraining are "about the same", and the task conditioning is what helps (≈1:11:40–1:12:25). The
lecturer links it to [lecture 17](17-generalization-out-of-distribution.md)'s learned prior: "a different mechanism of trying to learn
something like a geospatial prior" (≈1:12:25). The slide is excluded from OCW's licence.

### Meta-learning by sequence modeling

Slide 50 defines learning as a map from a sequence $x_0, y_0, x_1, y_1, x_2, y_2, \ldots$ to a function $f$ from $x$ to $y$. Given
$x_0, y_0, x_1, y_1, x_2, y_2, x_3$, what comes next? "RNN state + params is f, apply that f to x3." This is like the English-to-Spanish
prompt of [lecture 18](18-transfer-learning-models.md), where a language model shown a few translations translates the next sentence:
"sequence based in-context learning" (≈1:12:25–1:13:12). "The hidden state of that RNN … is a model that arguably learns to learn", and
"this really is, I think, what GPT and a lot of these modern autoregressive models are doing. It's kind of a version of meta-learning at
large scale" (≈1:13:58). Trained on many tasks and sequences, such models learn to answer new ones quickly, so meta-learning through
sequence modeling may have "a much better actual output" than through "this kind of inner and outer loop of training" (≈1:14:43). See
[foundation models and prompting](foundation-models-and-prompting.md) and [recurrent neural networks](recurrent-neural-networks.md).

## Summary

Slide 51 repeats the outline: transferring knowledge about the inputs, with generative models as data++, and meta-learning. "We learned
today about generative models as data and some of the pros and cons of that, and we also talked about learning to learn, or
meta-learning" (≈1:14:43).

## Pointers to other lectures

The deck prints no lecture number but its own, on the title slide ("Lecture 19: Transfer Learning II"). The recording points back to the
generative-model lectures "with Phil" (14–16, ≈2:21), to "the lecture from VAEs" (15, ≈20:20), to Dataset Interfaces, "I showed this work
actually before" (17, ≈36:41), to learning a prior "when we were talking about distribution shift" (17, its spatiotemporal prior,
≈1:12:25), to the forward and backward pass "unfolded" (probably 2, ≈1:09:21), and to in-context translation in "autoregressive
modeling for large language models" (probably 18, its GPT-3 prompt, ≈1:13:12). It names no later lecture. See the [course map](course-map.md).

## See also

- [Generative models as data](generative-models-as-data.md) — data++, latent directions, labelling and explaining with a generator.
- [Meta-learning](meta-learning.md) — MAML, backpropagating through backward, and in-context learning as meta-learning.
- [Transfer learning](transfer-learning.md) — the three kinds of knowledge to transfer, across lectures.
- [Generative adversarial networks](generative-adversarial-networks.md) — StyleGAN, BigGAN and their latent spaces.
- [Contrastive learning](contrastive-learning.md) — positive pairs from data-space and latent-space transformations.
- [Data augmentation](data-augmentation.md) — flips and crops as generation, and test-time augmentation through a generator.
- [Variational autoencoders](variational-autoencoders.md) — the latent space whose seams slide 18 shows.
- [Foundation models and prompting](foundation-models-and-prompting.md) — in-context learning.
