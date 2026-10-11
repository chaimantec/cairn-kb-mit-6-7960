# Lecture 16 — Generative Models: Conditional Models

**Lecturer:** Phillip Isola ·
**Video:** [youtube.com/watch?v=zaMcHuJwe1w](https://www.youtube.com/watch?v=zaMcHuJwe1w) (81 min) ·
**Slides:** [`mit6_7960_f24_lec16.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)
(84 pages; the deck is titled "Lecture 16: Deep Generative Models III"; transcribed slide by slide in [`raw/slides/16-generative-models-conditional-models.md`](../raw/slides/16-generative-models-conditional-models.md)) ·
**Transcript:** [`raw/transcripts/16-generative-models-conditional-models.md`](../raw/transcripts/16-generative-models-conditional-models.md)

## What this lecture establishes

The last of the course's three lectures on generative models is about using them to **predict** one thing from another:
"This one, I hope, will be a bit more fun and lightweight. We're going to get to see some applications of that math"
(≈0:00). Its claim is that generative models are not a frivolous art form but "fundamental. They're core to almost
everything", because every problem of predicting "high-dimensional structured data … out of a network" is one they are
suited to (≈1:31). Such problems are called **structured prediction**: the output is a high-dimensional object whose
dimensions are not independent of each other.

The argument runs in three steps. First, the older way of predicting such outputs, regression or per-pixel
classification, implicitly treats every output dimension as independent and so averages over the possibilities, which is
why its pictures come out blurry and desaturated. Second, the fix is a change of **objective**, not of hypothesis space:
a generative model's objective can represent a multimodal distribution with joint dependencies, and every family the
course has met can be made **conditional** on an input, usually by one small change. The lecture shows it for each
pairing of images and text: image-to-image with conditional GANs (pix2pix) and conditional VAEs, text-to-text with
autoregressive language models, image-to-text with a captioning transformer and cross-attention, and text-to-image with
VAEs (DALL-E 1), diffusion models (Stable Diffusion) and a mixture of all three (VQGAN). Third, it asks whether such
mappings can be learned **without paired data**, and answers yes, for GANs, by cycle consistency (CycleGAN): "you don't
always need paired data to solve conditional prediction" (≈1:20:56).

**Notation on this page** follows the slides. The input, or conditioning variable, is $\mathbf{x}$ and the output to be
predicted is $\mathbf{y}$, drawn from paired data $p_{\texttt{data}}(\mathbf{x}, \mathbf{y})$. In a conditional GAN the
generator is $g_{\theta}$ with parameters $\theta$ and the discriminator $d_{\phi}$ with parameters $\phi$;
$d_{\phi}$ outputs the probability that its input is **synthetic** (slide 20). $\mathbb{E}_ {\mathbf{x},\mathbf{y}}$ is an
expectation over the training pairs. $\mathbf{z}$ is a latent variable, sampled at test time as
$z \sim \mathcal{N}(\vec{0}, 1)$, the unit Gaussian. In the text-and-image slides an image is $\boldsymbol{\ell}$ (a bold script ell), a text is
$\mathbf{t}$, a hat marks a prediction, $f$ is an encoder and $g$ a decoder. See also the course's [notation](notation.md).

## Where this lecture sits

The outline has two items: **structured prediction**, with four sub-items, image-to-image, text-to-text, image-to-text and
text-to-image, and **unpaired translation** (slide 2). "All the domains will be images and text because those are the two
domains where this kind of technology is best developed", though it "is currently starting to be used in other domains,
too" (≈0:45). Lecture 14 introduced the families of [generative models](generative-models.md) and lecture 15 went deep on
the [variational autoencoder](variational-autoencoders.md), "a little bit math-heavy … a little bit of a slog" (≈0:00);
this lecture applies them. It was given on Halloween (≈0:00).

## Structured prediction

The deck opens with four "data prediction problems", each mapping structured data in to structured data out (slide 3):
semantic segmentation (a photograph to a mask that gives every pixel a class), automatic speech recognition (a waveform to
the sentence "computers that talk"), text-to-photo (a description of a bird to a picture of it) and protein folding (an
amino-acid sequence to a 3D structure). "You're not just trying to give one label to, what is this protein? You're trying
to output a structured object" (≈2:16–3:48). Slide 3's pictures are excluded from OCW's licence.

"In statistics and other areas, this is sometimes called structured prediction" (≈2:16), and the lecturer finds the term
odd: "It's a weird historical artifact that we had this term, structured prediction. This is just everything" (≈2:16).
Slide 4 defines it. Structured prediction is to "model *joint* distribution of high-dimensional data",

$$P(\mathbf{X} \vert \mathbf{Y} = \mathbf{y}),$$

where $\mathbf{X}$ is high-dimensional, and "in deep learning this is usually what we are interested in". The
**unstructured** alternative models each dimension on its own,

$$\prod_i p(X_i \vert \mathbf{Y} = \mathbf{y}).$$

(Slide 4 writes the output as $\mathbf{X}$ and the condition as $\mathbf{Y}$; the rest of the deck predicts $\mathbf{y}$
from $\mathbf{x}$.) "Unstructured prediction was if I want to model some high-dimensional data, I will assume that all the
dimensions are independent statistically … the joint distribution equals a product over each dimension. And if I don't
have independence, I don't have a product. And then we call it structured prediction" (≈3:48). Independence almost never
holds: a robot arm's trajectory or a self-driving car's route through a city is a sequence of decisions that depend on
each other: "If I turned left, that affects what my next decision should be" (≈4:34). The unstructured approach is still
worth seeing first, "because even though it seems like a crazy idea now, it was simple. And it's not terrible" (≈4:34).

## Why unstructured prediction fails: colourization

The example is colourization, a problem the lecturer loves "because it captures all the key concepts in machine learning"
(≈5:19). Given the grey version of a shirt, what colour is it? Suppose the shirt is made in two colours, teal and pink, with
equal probability, so the true distribution of colour given the grey value has two spikes (slide 5, plot a). Slide 5 then
shows three ways to predict it.

![Slide 5: a grey T-shirt observation and four plots over a colour axis running from green to pink: the true distribution with two spikes, a point prediction with one spike in the middle, a Gaussian centred in the middle, and a ten-bar softmax with two tall bars](../raw/images/16-generative-models-conditional-models/slide-5.png)

*Slide 5 — Predicting the colour of a grey shirt: the true distribution has two modes; regression predicts their mean, a Gaussian puts a bump on the mean, and only the softmax over quantized colours captures both modes. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

- **Point prediction, or regression** (plot b), outputs a single colour, a delta function that "doesn't express anything
  about your uncertainty … And it can't capture this multimodal distribution." Least-squares regression is minimized by the
  **mean** of the data: "the average of a set of data points is a minimizer of the squared deviation from all the data
  points. So that's something to … slot away in your brain." But "the mean has zero density under the true data
  distribution. So the mean is a terrible estimate of the color" (≈6:52–8:26).
- **A Gaussian predictive distribution** (the plot the slide letters d) fitted by maximum likelihood models the uncertainty,
  "but … this Gaussian is insufficient because it's unimodal distribution. And the true data distribution that I'm trying
  to model is bimodal" (≈8:26–9:13).
- **A softmax distribution, or classification** (the plot lettered c), over quantized colours does work: "the hack that I
  mentioned in the hacker's guide lecture" ([lecture 9](09-hackers-guide-to-deep-learning.md)). "The categorical
  distribution … can express bimodal distributions … You can represent any number of bumps", so it solves the problem "up
  to quantization error" (≈9:13–10:44). See [softmax and cross-entropy](softmax-and-cross-entropy.md).

(Slide 5 letters its plots a, b, d, c in reading order; the slide file flags it.)

Classification fails in a second way, once the output has many dimensions. Colourizing the whole shirt means predicting a
colour for every pixel, and the per-pixel probabilities, almost exactly even, will vary a little from pixel to pixel with
shading. Where they cross the threshold the most likely colour flips, so the top of the shirt comes out teal and the bottom
pink (slide 6). "I'm basically failing to capture the joint dependency between this prediction and that prediction. The
joint dependency says that they should be the same color" (≈11:32–12:17).

![Slide 6: a large T-shirt coloured green over most of it and pink at the lower right, with two marked pixels each linked to its own ten-bar colour histogram, one peaking at green and one at pink, above the two properties of generative models](../raw/images/16-generative-models-conditional-models/slide-6.png)

*Slide 6 — Independent per-pixel predictions flip between the two modes across the shirt. The fix is a model with the two properties listed underneath. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

Slide 6 states what generative models bring: "Generative models have two important properties for structured prediction:
1. They can model a multimodal distribution. 2. They can model joint dependences between multidimensional predictions."
"This is almost like the definition — when we talk about generative AI, generative models, this is what we mean … the VAEs
and GANs and diffusion models all can do this" (≈13:04). "It was only in the last few years that people really had models
that can do both of these things in practice. In theory, we've thought about this for decades" (≈13:51).

**Real or fake?** Two photographs test the class (slides 7 and 8, both excluded from OCW's licence). Most hands go up for
"fake" on the first, a stack of drums; a student points to pink and blue on the same surface: "It's oscillating. Exactly …
This system is making independent predictions per pixel. And it crossed the threshold of probability. And it switched to
the other mode" (≈14:39). The second is a colourized photograph of Margaret Hamilton, "who was working at MIT doing
programming in the '60s for the Apollo project" (≈15:25). Students spot a watermark ("COLOR_BYANGELINA") and a mottled
white-to-blue door, the same flipping again (≈16:12). The lecturer adds one more tell, desaturation: "we're basically doing
some averaging … we'll collapse the distribution toward the mean because we're not sure what to output." AI-generated work,
"at least from a few years ago", looks "a little bit blurry, a little bit desaturated", and generated music sounds fuzzy:
"These are the effect of averaging that you get when you're fitting a distribution that doesn't have the capacity to
approximate the true distribution" (≈16:58–17:46).

## The objective, not the hypothesis space

"Let's go back to 2012." Deep learning then was "about deep learning as a new hypothesis space with high capacity"
(≈17:46–18:35). Slide 9 draws the learner, data in and a function $f$ out, with three parts: objective, hypothesis space and
optimizer; "Deep learning in 2012" puts a red box around the **hypothesis space**: "Use a hypothesis space that can model
complex structure (e.g., a CNN, nearest-neighbor)". Slide 10 is "a very popular slide in 2012 or a little bit after from
Andrew Ng" (≈18:35).

![Slide 10: a white-on-black chart, Performance against Amount of data, where a blue Deep learning curve keeps rising while a red Older learning algorithms curve levels off](../raw/images/16-generative-models-conditional-models/slide-10.jpg)

*Slide 10 — Andrew Ng's "Why deep learning" chart: older algorithms plateau as data grows, deep learning keeps improving. The lecture later argues that deep learning with an unstructured objective plateaus too. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

Its argument is about capacity: a linear support vector machine "can't fit nonlinear data", so with more data "I'll find
the best linear fit. But then I won't ever be able to model the nonlinear parts of the data. And so deep learning will just
keep scaling with more data, but other methods don't." "So this is an argument about the hypothesis space. But I want to
make an argument about the objective function" (≈18:35–19:20).

The objective is the expected loss of a function over the training pairs (slide 11),

$$\arg\min_{\mathcal{F}} \mathbb{E}_ {\mathbf{x},\mathbf{y}} \left[ L(\mathcal{F}(\mathbf{x}), \mathbf{y}) \right],$$

with the hypothesis space under the arg min and the loss $L$ as the objective. Slide 12's learner does semantic
segmentation, $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ to one-hot labels $\mathbf{y} \in \mathbb{R}^{H \times W \times K}$,
with objective $f^{\ast} = \arg\min_{f \in \mathcal{F}} \sum_{i=1}^{N} H(\mathbf{y}_ i, \hat{\mathbf{y}}_ i)$, a
convolutional net and stochastic gradient descent; the lecturer calls it "the per-pixel classification" (≈19:20). Slide 11's
photograph is excluded from OCW's licence, and slide 12 reuses it.

The better example, he says, is mapping a map to the satellite view of the same place (slide 13, titled "Sat2Map"; its
$\mathbf{x}$ is the map and its $\mathbf{y}$ the aerial photograph). "I have paired data. I can just predict the y's from the
x's. But I'm regressing or outputting an entire image" (≈20:08). The objective is the squared error over the output,

$$\theta^{\ast} = \arg\min_{\theta} \sum_{i=1}^{N} \left( f_{\theta}(\mathbf{x})_ i - y_i \right)^2,$$

"least squares regression for outputting a high-dimensional object. And this is implicitly treating all of the output
dimensions as independent." Squared error "is like log probability under a Gaussian probability model. Sum of logs is like
the product of those probabilities. So this is implicitly a unstructured objective where I've assumed that the world
factorizes independently over dimensions of the output space" (≈20:08–21:00).

![Slide 13: the Sat2Map learner, two pairs of a map tile and an aerial photograph as data, the squared-error objective, a convolutional net and stochastic gradient descent](../raw/images/16-generative-models-conditional-models/slide-13.jpg)

*Slide 13 — The same learner with a squared-error objective, mapping street-map tiles to aerial photographs. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

Train "the biggest, highest capacity convolutional network I want" with SGD, and the output looks OK but blurry (slide 14):
"This park has trees in it. Now, I don't know if it's an oak tree or a pine tree. I don't know where the trees are. If I
average over all the different trees and all the different grass and all the different locations, I'm just going to make an
average green in the park … And that's a blurry image. So that was deep learning 2012 to 2016" (≈21:00–21:46).

![Slide 14: a street map with buildings and a park as input, and beside it the deep net's output, a blurry dark aerial-style image with the same layout and no detail](../raw/images/16-generative-models-conditional-models/slide-14.jpg)

*Slide 14 — Squared-error regression from a map to an aerial view: the layout is right, but the park is an average green with no trees. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

"And then deep learning 2016 to 2024 has been about that, to some degree, but more about modifying now the objective
function. And that's where the generative AI comes in" (≈22:37). Slide 15 moves slide 9's red box from the hypothesis space
to the objective: "Use an objective that can model structure! (e.g., a graphical model, a GAN, etc)", one that can
"penalize errors in structure, in high joint structure between pixels and the deviations from a multimodal posterior"
(≈22:37).

## Image-to-image with conditional GANs

The first application is image to image, "with conditional GANs" and "with conditional VAEs" (slide 16). The running
example is an architectural label map, where "each different color … is going to be indicating what is the architectural
element — is it a door, is it a window", mapped to a photograph of the building: "I could be an architect trying to design
new buildings and use this tool" (≈23:25). The generator $g_{\theta}$ maps the label map $\mathbf{x}$ to a picture
$g_{\theta}(\mathbf{x})$ (slide 18, "For example: pix2pix [Isola et al. 2017]"). It "could be a ConvNet in this project. But
it could be a ViT in the modern era" (≈23:25).

### The generator, the discriminator and their game

A GAN, as lecture 14 drew it, maps Gaussian noise through a generator to an image that a discriminator cannot tell from a
real one. "Here, the noise is replaced with a conditioning variable. So there doesn't have to be noise. In this example,
there is just a mapping from an x to a y as opposed to a z to an x. So it's the same exact math as the GAN" (≈24:12). "g
tries to synthesize fake images that fool d. d tries to identify the fakes" (slide 19).

The discriminator is "a binary cross-entropy objective … a synthetic data classifier". It assigns high log probability of
being synthetic to the generator's outputs and low probability, "1 minus the probability", to the real data $\mathbf{y}$
(≈24:59–25:47; slide 20, where the generated facade scores "synthetic (0.9)" and the real photograph "real (0.1)"):

$$d^{\ast}_ {\phi} = \arg\max_{\phi} \mathbb{E}_ {\mathbf{x},\mathbf{y}} \left[ \log d_{\phi}(g_{\theta}(\mathbf{x})) + \log(1 - d_{\phi}(\mathbf{y})) \right].$$

The generator does the opposite, "minimize the log probability … of a synthetic image being called synthetic", and "we will
backprop from d through the parameters of g" (≈26:32; slide 21):

$$g^{\ast}_ {\theta} = \arg\min_{\theta} \mathbb{E}_ {\mathbf{x},\mathbf{y}} \left[ \log d_{\phi}(g_{\theta}(\mathbf{x})) \right].$$

Together they make "this minimax game, this adversarial game", in which "we don't want to fool just any old random
discriminator. We want to fool the best possible discriminator" (≈26:32–27:18; slide 22, "g tries to synthesize fake images
that fool the best d"):

$$\arg\min_{\theta} \max_{\phi} \mathbb{E}_ {\mathbf{x},\mathbf{y}} \left[ \log d_{\phi}(g_{\theta}(\mathbf{x})) + \log(1 - d_{\phi}(\mathbf{y})) \right].$$

"At equilibrium, you can prove that this will converge to being — samples from g are identically distributed as samples from
y, the true data-generating process", provided the networks have "actually found … the Nash equilibrium of this game and that
there's sufficient capacity" (≈27:18). See [generative adversarial networks](generative-adversarial-networks.md).

### The discriminator as a learned loss

"The perspective that I want to give in this lecture is that from g's perspective, d is like a loss function" (≈27:18–28:06;
slide 23: "g's perspective: d is a loss function. Rather than being hand-designed, it is *learned* and *highly
structured*"). "It's like the squared error loss function, except it has a new property, which is that it's learned as
opposed to being hand-defined. So it's a loss function that can look at structure in the output and not just say, did I get
every pixel independently correct, but also, did I get the edges correct? And did I get the higher order statistics of the
output correct? … it's very hard to hand-define such a loss function. But it can be learned via this game" (≈28:06).

### Conditioning: the discriminator sees pairs

The unconditional objective has "one little error" for a conditional GAN. Show the discriminator a real photograph of a
building that does not match the label map (slides 24 and 25): "It is a real photo … But it doesn't match the input. So the
one little trick to make a GAN conditional is you need to let the discriminator look at not only is the output real, but
does the output match the input? So it's, is the pair x, y a real pair sampled from the data-generating process"
(≈28:54). The discriminator takes the input as well, and asks "real or synthetic *pair*?" (slides 26 to 29), which gives the
conditional GAN's objective (slide 30):

$$\arg\min_{\theta} \max_{\phi} \mathbb{E}_ {\mathbf{x},\mathbf{y}} \left[ \log d_{\phi}(\mathbf{x}, g_{\theta}(\mathbf{x})) + \log(1 - d_{\phi}(\mathbf{x}, \mathbf{y})) \right].$$

Now it can penalize an output that is blurry, "but it can also penalize whether the output has the windows in the wrong
place that don't match where the architect has drawn the windows" (≈29:43). Asked where the label maps come from: "you have
a data set of diagrams with photos. So you have paired data. You have supervised data, x, y pairs" (≈30:29). Asked when the
loss is applied: you run $\mathbf{x}$ through the generator, run the prediction through the discriminator, "and then you
backprop through that entire computation graph. So the computation graph is d of g of x" (≈30:29–31:15).

The slides of this build, 18 to 31, are not rendered: slide 73 repeats slide 30 under an OCW notice ("© Zhu, et al."), so
their pictures are excluded from the licence.

### Adding an L1 term

"Training Details: Loss function" adds a regression term to the GAN's (slide 31; "[c.f. Pathak et al. CVPR 2016]"):

$$g^{\ast}_ {\theta} = \arg\min_{\theta} \max_{\phi} \mathcal{L}_ {\texttt{cGAN}}(\theta, \phi) + \lambda \mathcal{L}_ {\texttt{L1}}(\theta),$$

where the last term, drawn as the norm of the difference between the output $g_{\theta}(\mathbf{x})$ and the target
$\mathbf{y}$, is "a regular old regression objective … And that will produce these blurry outputs because … it treats every
pixel as independent, conditioned on the input. But it can be added to this GAN objective for a little bit better
performance." GANs are "not as popular now. And the main reason was they weren't very stable … This adversarial game is very
hard to optimize. But this regular regression loss can very much stabilize that process" (≈31:15–32:00; slide 31, "Stable
training + fast convergence").

![Slide 32: four pairs of a coloured facade label map and the building photograph generated from it](../raw/images/16-generative-models-conditional-models/slide-32.jpg)

*Slide 32 — Labels to facades: each label map, with its colour-coded windows, balconies and doors, and the conditional GAN's photograph of a matching building. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

Slide 32 shows the results ("Labels → Facades", data from Tylecek, 2013). "But this is not impressive. By now, we've seen that
you can generate all kinds of photos conditioned on all kinds of things. But a few years ago, it was magic" (≈32:00).

### The patch discriminator

"The architecture of the discriminator critically matters in what types of errors it is sensitive to" (≈32:00–32:46). The
project used a **patch discriminator**: rather than judging the whole photograph at once, "we look at every patch of the photo
and say whether that patch looks like a real patch or a fake patch … it's going to be convolutional. So it'll just slide
across the input image" (≈32:46; slide 33). That allows "a very small network", and gives more supervision: "if I have 10
images, I might have 10 million patches" (≈33:32). "I think that this architecture is still part of Stable Diffusion to this
day. But maybe in the very latest Stable Diffusions, it disappeared" (≈33:32).

![Slide 33: a sliding window labelled D with N pixels on a side scanning an image y and outputting 1 or 0, beside four facade outputs for discriminator patch sizes 1 by 1, 16 by 16, 70 by 70 and 286 by 286](../raw/images/16-generative-models-conditional-models/slide-33.jpg)

*Slide 33 — The patch discriminator, and what its patch size does: the outputs for patch sizes of 1, 16, 70 and 286 pixels on a side. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

The patch size sets what the loss can see (slide 33's four outputs, $1 \times 1$ to $286 \times 286$). With a $1 \times 1$
patch "I get back to almost the same thing as least squares regression … factorizing my model into a set of independent
predictions that are not jointly penalized." At $16 \times 16$ it can see blur: "Blurriness cannot be seen in a single pixel …
It's the second-order statistic of the data." Bigger patches, $70 \times 70$, model "higher and higher order statistics". At
the whole image it "worked worse because … discriminator had too many parameters relative to the number of data points … This
was trained on 400 images", though "in the more modern era, this might not be such an issue because we just have infinite
data" (≈34:20–35:52).

### Why structured objectives

Slide 34 repeats Andrew Ng's chart, and slides 35 and 36 redraw it as a "cartoon". "Deep learning with unstructured objectives
actually does saturate. So it could be that, yes, in the hypothesis space, there's the correct answer. It's a universal
approximator … But if I'm only incentivizing with my objective function that it's good enough just to model the factorized
distribution … there'll be no incentive to model the joint dependencies" (≈35:52–36:38).

![Slide 36: a cartoon chart of Performance against Amount of data with three curves: older learning algorithms plateau lowest, deep learning with an unstructured objective plateaus higher, and deep learning with a structured objective starts slower but rises highest](../raw/images/16-generative-models-conditional-models/slide-36.png)

*Slide 36 — The cartoon: deep learning with an unstructured objective (least squares) saturates too; with a structured objective (conditional generative models) it keeps improving. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

"This is a cartoon. But these little swaps of the ordering are meant to be indicative of the fact that with more capacity, you
might require more data. So less capacity and a simpler set of modeling assumptions might mean that you can use less data to
get a good result. But if you have more data, then you'll get better results if you have more capacity and a more flexible
model class" (≈36:38).

The map example again, with each pixel "penalized with … an L1 loss per pixel" (slide 37, "Unstructured prediction (L1)"),
against the conditional GAN with a patch discriminator (slide 38, "Structured Prediction (cGAN)"), which "says, it was blurry.
That looked unrealistic … I better add a bunch of trees. But we enter into the regime of hallucination, where these trees are
just a guess. They're just a sample from the true distribution over where the trees might be" (≈37:27).

![Slide 38: the map input beside the conditional GAN's output, a sharp aerial-style image with roofs and dense trees in the park](../raw/images/16-generative-models-conditional-models/slide-38.jpg)

*Slide 38 — The same map through a conditional GAN: sharp roofs and trees, plausible rather than true; compare slide 14's blurry regression output. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

Two questions follow. Would a Gaussian with a full covariance, enforcing structure between nearby pixels, do as well? "I think
it'll improve things over this result. But I would be pretty surprised if it makes a dramatic difference. I think that that
still can only model Gaussian structure" (≈38:14). And the L1 output seems more faithful to the map: a road visible in it has
disappeared from the GAN's. "This is a common trade-off — that the L1 prediction is taking the average. It's very
conservative … all the modes of the distribution must share that so that the average will also share that road, whereas a GAN
will select a mode … I think it's kind of like an ensemble." "The problem with generative AI stuff is that you often will get
these hallucinations that are samples from how something might be, but not all aspects of that sample might actually be
high-probability" (≈38:59–39:45).

## Image-to-image with conditional VAEs

"Every generative model that we've talked about can be made conditional. Oftentimes, it's quite easy" (≈39:45). The reason to
want a VAE is **mode collapse**: a GAN's generator "will output one mode of the distribution that looks realistic. But … it will
maybe miss other modes … So conditional VAEs are a method that is a little better at maybe capturing all the modes of the
posterior" (≈39:45–40:31). The label map does not say what the building is made of: "this schematic could match that building.
But it also could match a brick building" (slide 40).

![Slide 40: a facade label map as input and a grid of six possible outputs, one a generated building and five empty squares with question marks](../raw/images/16-generative-models-conditional-models/slide-40.jpg)

*Slide 40 — One input, many possible outputs: the label map fixes the windows and doors but not the material. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

So a latent variable $\mathbf{z}$ is "injected into the network in order to model all of the uncertainty over the possible
variations in the predictions" (≈40:31; slide 41, where $z \sim \mathcal{N}(\vec{0}, 1)$ enters the generator $G$ beside the
input).

![Slide 41: the label map x passes through a three-layer generator G to an output G(x), with a Gaussian surface below feeding z into the generator](../raw/images/16-generative-models-conditional-models/slide-41.jpg)

*Slide 41 — A conditional generator with a latent input: z, drawn from a unit Gaussian, carries what the input leaves open. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

### The moving square

The worked example is a small video: a pink square that can move in any of eight directions (slide 42, an example from
Babaeizadeh et al., ICLR 2018). Predict the next frame from the current one with a least-squares objective and "my prediction
will be an average of all eight possible directions, average of the eight modes … The average has zero probability under the
data distribution" (≈41:17–42:03). Slide 42's lower row, labelled "Finn et al. (2016)", shows such predictions as blurry smears
spreading out from the square.

![Slide 42: a context frame with a magenta square on grey, three ground-truth frames of it moving up and to the right, and below them three blurry predictions that spread into a ring](../raw/images/16-generative-models-conditional-models/slide-42.jpg)

*Slide 42 — Predicting a moving square: the ground truth moves one way, while a deterministic prediction averages over the directions it might go. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

The conditional VAE is built in two steps (slide 43). First, a VAE on the target alone: "take my target. This is the prediction
of the future. We're going to try to compress it into a bottleneck, a Gaussian variable z, and then reconstruct that
prediction." Second, "to make it a conditional VAE, I'm going to simply let my reconstruction network, my decoder, observe the
data x" (≈42:03–42:50).

![Slide 43: the observation x, a grey frame with a green border and the square at its centre, and the target y with the square at the upper right; arrows run from the target and the observation into z, and from z and the observation to the reconstruction](../raw/images/16-generative-models-conditional-models/slide-43.jpg)

*Slide 43 — A conditional VAE in training: the target passes through the bottleneck z, and the decoder sees the observation x as well. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

The decoder is shown the answer, but only through the bottleneck: "I'm going to force the answer, y, to go through an
information bottleneck, z … z will learn to encode what direction this thing goes. But z is compressed. Z is trying to be a unit
Gaussian. So it's trying to remove information. And so rather than encoding the color in z, the color can be given by x … z
ends up only encoding the direction of motion, which is unobserved in the observation" (≈42:50–43:37; slide 44 shows a downward
target). Slide 45 says it in a caption: "z learns to encode the missing information necessary to predict y from x, i.e. the
direction in which the purple box moves".

![Slide 45: the conditional VAE with a downward-moving target and reconstruction, and a caption from z: z learns to encode the missing information necessary to predict y from x](../raw/images/16-generative-models-conditional-models/slide-45.jpg)

*Slide 45 — What the latent learns: only what the observation does not already give, here the direction of motion. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

"Now, at sampling time, I can sample from this model by replacing … my inference of z given a target image with z as a unit
Gaussian because I've trained it as a VAE. And I've imposed that it should be close to a unit Gaussian" (≈43:37–44:24; slide
46). Each sample of $\mathbf{z}$ gives one future: "Maybe I sample the down z vector. That will cause me to predict that the
object goes down … Now I sample up. It goes up. It goes right" (≈44:24; slides 46 to 48).

![Slide 46: at test time a Gaussian surface replaces the target, feeding z; with the observation, the prediction shows the square moving down](../raw/images/16-generative-models-conditional-models/slide-46.jpg)

*Slide 46 — Sampling: z drawn from the unit Gaussian picks one direction, and the decoder predicts that future. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

"In unconditional VAEs, z has to encode everything about the data. In conditional VAEs, z only encodes information that's not
given in the conditioning variable, x" (≈44:24–45:10). Slide 49 shows it on facades: one label map and six sampled buildings,
from BicycleGAN (Zhu et al., NeurIPS 2017).

![Slide 49: a facade label map x and a Gaussian z feeding a generator G, which outputs six buildings with the same layout in different colours and materials](../raw/images/16-generative-models-conditional-models/slide-49.jpg)

*Slide 49 — Multiple outputs from one input: six samples of z give six facades that all match the label map. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

### Explicit inputs or latent variables

"Concept: **You can control your data either via explicit inputs or latent variables**" (slide 50). In the previous lectures
the latent variables were knobs for "the lighting and the materials"; "now we also have conditioning variables, x. And these
also can control the way that the data looks." And "for a lot of modern generative models, actually, the latent variables might
play a very small role because you're just conditioning on more and more data. So in text-to-image models, if you tell it a
whole string of text, now you don't need a lot of latent variables because the string almost — very precisely could pin down
what you're going to see" (≈45:10–45:58).

## Text-to-text with autoregressive models

"Luckily, text-to-text is basically nothing new because … language models, are typically autoregressive" (≈45:58; slide 51).
"Autoregressive models are already, by construction, conditional generative models", since they model the data as a product of
conditionals, $P(x_2 \vert x_1)$, $P(x_3 \vert x_1, x_2)$ and so on: "autoregressive models are natively conditional" (≈45:58–46:43).
See [autoregressive models](autoregressive-models.md).

Slides 52 and 53 give the two uses. **Training** is next-word prediction: "Colorless green ideas sleep ____" goes into the language
model, which predicts "furiously". **Predicting** uses the same model on a conditional problem by giving it the input: a review
followed by "The sentiment in this review is ____" yields "positive". "So it's P of y given x, where x is the review and y is the
sentiment" (≈46:43–48:20). The model is still stochastic, though it has no explicit latent variables, "because every next word I
predict will be sampled from a categorical distribution", and the conditioning variables "are now made explicit". "This is often
called prompting language models … This is what you do when you just talk with your favorite LLM, Claude or ChatGPT" (≈47:33).

This "became evident, I would say, in the GPT-3 era": language models "are not just about writing paragraphs from scratch … They're
actually most powerful for conditional prediction" (≈48:20–49:08). Slide 54 shows GPT-3 (Brown et al., 2020) completing an airline
review with "The sentiment in this review is negative."

From the floor: the underscores around "five" in the review just mark italics ("I think I probably made this one up myself",
≈49:56); retrieval-augmented generation is "related to this. But I would say for now, treat that as a separate concept" (≈49:56).
And how is such a model trained? Either a language model trained on "all the words on the internet" learns such conditionals "if
they exist in the data", or, if not, "you can collect your own data set of reviews and sentiment labels. And then you could
fine-tune on that" (≈49:56–51:27). "We'll have another lecture on language models", [lecture 21](course-map.md) (≈51:27).

## Image-to-text: captioning with cross-attention

Mapping between data types uses [transformers](transformers.md), "a way of putting all data types onto the same equal footing.
Just turn everything into tokens, and then just do token-to-token mappings … molecules, financial data, spreadsheets — if you just
can convert it to tokens … you'll be able to run all the same technology" (≈51:27–52:15; slide 55). The task is image captioning,
outputting "the caption that somebody would have written underneath the image online … or maybe the alt text" (≈52:15).

![Slide 56: an image encoder, three image patches through self-attention layers, beside a text decoder whose tokens A yellow bird pass through cross-attention to the image tokens and causal self-attention, predicting yellow bird sitting](../raw/images/16-generative-models-conditional-models/slide-56.jpg)

*Slide 56 — Image-to-text with a transformer: image tokens from a self-attention encoder, and a causal text decoder that cross-attends to them while predicting each next word. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

Slide 56 draws the architecture. The **image encoder** is a vision transformer, which, as in the transformers lecture
([lecture 8](08-architectures-transformers.md)), chops the image into patches, embeds each and runs self-attention over the
tokens (≈52:15–53:01). The **text decoder** "is just a GPT … an autoregressive language model with a transformer" with causal
attention: "this word here, 'sitting,' only depends on all the previous stuff. 'Bird' does not depend on 'sitting'" (≈53:01–53:47).
To join them, "all I have to do is prefix the set of language tokens with the set of image tokens", and one simple way of stitching
the networks together is **cross-attention**: the language model's tokens issue queries that are answered by keys and values from
the image tokens. "The query is emitted from the word 'yellow.' That will then match the key for the bird because it's yellow …
So cross-attention is just attention, but … taking keys from — and values from one set and queries from a different set"
(≈53:47–54:38).

Asked when the caption is provided: the training data are pairs of an image $\mathbf{x}$ and its caption $\mathbf{y}$, and "at
inference time, you do it sequentially. But you do it in parallel at training time" (≈55:26). On integrating the two streams,
"typically, you have a query matrix which applies to the text and another query matrix that applies to the visual data. And then
you can submit keys, values, and queries from both modalities" (≈56:13–56:59).

"One of the trending industrial things this year is multimodal chatbots" (≈56:59). The lecturer gave ChatGPT a screenshot of
slide 56 and asked what it shows and how it relates to the way it works (slide 57, "Created with ChatGPT"): "it's pretty good. You
have an image encoder, cross-attention layer. But it did get this wrong." Its answer (slide 58, "Does ChatGPT know how its own brain
works?") says "I don't process visual data directly"; "it was looking at the visual data. It had to be in order to make the correct
assessment … It doesn't know how its own brain or itself works" (≈56:59–57:45).

![Slide 57: a ChatGPT conversation with a small, garbled copy of slide 56's diagram, the question what is this slide showing and how does it relate to the way you work, ChatGPT, and the start of its answer](../raw/images/16-generative-models-conditional-models/slide-57.jpg)

*Slide 57 — Asking ChatGPT about slide 56: it reads the architecture correctly, then (slide 58) denies that it processes images. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

## Text-to-image with all models combined

"Text-to-image models of the current generation, or maybe one generation back, actually are a combination of GANs, VAEs, and
diffusion models … People call them diffusion models. But they have all three components in them" (≈57:45; slide 59, "with all
models combined!").

### A general strategy for domain translation

Slide 60 gives "the more general strategy for translating between two different modalities or two domains": an encoder $f$ maps
a point $\mathbf{x}$ of one space $\mathcal{X}$ to a latent $\mathbf{z}$ in $\mathcal{Z}$, a decoder $g$ maps it on to a
prediction $\hat{\mathbf{y}}$ in another space $\mathcal{Y}$, and training penalizes "the difference between my prediction and …
the ground truth, y". "If it's in the VAE autoencoder style, then I decode from that latent space now not back to in autoencoding,
like back to x, but back to another domain, y" (≈58:32).

![Slide 60: a blue blob X with a point x, an encoder trapezoid f to a point z in an orange circle Z, a decoder trapezoid g to a point y-hat in a yellow blob Y, and a red dotted prediction error between y-hat and y](../raw/images/16-generative-models-conditional-models/slide-60.jpg)

*Slide 60 — Domain translation in general: encode from one space into a latent space, decode into another, and penalize the prediction error there. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

Slide 61 fills it in from images to text, with a photograph of a bird captioned "a yellow bird sitting on a tree branch". It is
titled "Text-to-image", and the lecturer says the same ("here's what that looks like in text-to-image … I go from image space to
latent space to text space"), but its diagram runs from image space to text space, the captioning direction just covered; there,
"where do you define the latents is not clear because it's autoregressive" (≈58:32–59:21). Slide 62, with the same title, runs it
the other way, from the text "A yellow bird sitting on a tree branch." to the image, and "this one is going to follow this diagram
a little bit more precisely" (≈59:21).

![Slide 61: the domain-translation diagram labelled from image space through latent space to text space, beside a small stack mapping a bird photograph through an encoder and decoder to the caption a yellow bird sitting on a tree branch](../raw/images/16-generative-models-conditional-models/slide-61.jpg)

*Slide 61 — The strategy with images on the left and text on the right; the slide is titled "Text-to-image" but runs image to text. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

![Slide 62: the same diagram reversed, from text space through latent space to image space, beside a small stack mapping the caption A yellow bird sitting on a tree branch through an encoder and decoder to the bird photograph](../raw/images/16-generative-models-conditional-models/slide-62.jpg)

*Slide 62 — Text to image: the caption is encoded to a latent and decoded into a picture. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

### A text-to-image VAE: DALL-E 1

The conditional VAE, conditioned on text instead of a frame of video (slide 63, "Text-to-image VAE", "For example: DALL-E 1 [Ramesh
et al. 2021]"). "Rather than conditioning on the observation of that moving pink block, I'm going to condition on a scarlet macaw
sitting on a perch." The decoder sees the text and a latent $\mathbf{z}$ that "has to explain everything about the data to make
this a deterministic process … This input does not say that the scarlet macaw is facing to the left. So therefore, the latent
variable has to learn to encode that the macaw is facing to the left. The latent variable is trained by inferring the latent
variable from the target prediction" (≈59:21–1:00:07). The text encoder $f$ is a language model that "encodes text into tokens",
and the decoder "could be a ViT", a transformer decoder over tokens, "and the latent variables might also be tokens" (≈1:00:07–1:00:55).

![Slide 63: the text A scarlet macaw sitting on a perch goes through a text encoder f into an image decoder g, while a target macaw picture goes through a dotted image encoder to a latent z, also fed to the decoder, which outputs a predicted macaw](../raw/images/16-generative-models-conditional-models/slide-63.jpg)

*Slide 63 — A text-to-image VAE in training: the image encoder, drawn dotted and gone from the test-time slide 64, infers the latent from the target; the decoder sees the latent and the encoded text. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

"At inference time, I simply sample from the unit Gaussian because the VAE is trained to make the latent distribution dense and
compact. And therefore, samples from the Gaussian are going to all map to valid predictions" (≈1:00:55). Slide 64 draws
$\mathbf{z}^{(i)} \sim \mathcal{N}(0, 1)$ for the prompt "A profile photo of a robin, facing left."

![Slide 64: at test time the text A profile photo of a robin, facing left goes through the text encoder into the image decoder, with z sampled from a unit Gaussian, and the prediction is a robin facing left](../raw/images/16-generative-models-conditional-models/slide-64.jpg)

*Slide 64 — Sampling from the text-to-image VAE: the image encoder is gone, and z comes from the unit Gaussian. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

How should the text and the latent be combined? "The vanilla, most standard thing right now seems to be concatenation, like convert
everything into a set of tokens or a set of embedding vectors, and then just concatenate and do more cross-attention between them,
and then repeat … You could also sum them together, take an average, multiply … how do you combine different streams in a transformer
in a more intelligent way? That is an active area of research — so maybe a good final project" (≈1:00:55–1:01:42).

### A text-to-image diffusion model

"This is roughly the way that the DALL-E 1 model works … But then Stable Diffusion, DALL-E 2, Midjourney, all these other things came
out. And they switched to using a diffusion model" (≈1:02:29). A [diffusion model](diffusion-models.md) starts from Gaussian noise
and denoises step by step, each step "a supervised learning problem" of undoing noise that was added to a data point. "And to make it
conditional on text, all I have to do is let my denoiser observe the text" (≈1:02:29–1:03:14; slide 65, where $f_{\theta}$ takes the
text $\mathbf{x}$ with a noisy image $\mathbf{y}_ t$ and outputs the less noisy $\mathbf{y}_ {t-1}$).

![Slide 65: a strip of pixel-art images denoising from random noise, y T, on the left to a clean robin, y 0, on the right, and below it f theta taking the text A profile photo of a robin, facing left with a noisy image y t and outputting y t-1](../raw/images/16-generative-models-conditional-models/slide-65.png)

*Slide 65 — A text-to-image diffusion model: each denoising step sees the text as well as the noisy image. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

Why that works: "because the text will be informative toward what the true image I started with is. So if the text says 'A robin on
a yellow background,' well, here's a noisy image … it might be a little bit of a yellow background, or maybe it's a white background
or a pink background. The text disambiguates that", to the degree there is a dependency between text and image in the training data,
"therefore, f will learn to use the text and learn to denoise in the direction guided by the text" (≈1:03:14–1:04:00; slide 66, "Text
provides information as to what the noiseless image should look like").

![Slide 66: two rows, A robin on a yellow background and A robin on a pink background, each a noisy pixel-art robin denoised to a clean robin on a background of the named colour](../raw/images/16-generative-models-conditional-models/slide-66.png)

*Slide 66 — The text tells the denoiser what the clean image looks like: the same noisy robin denoises to a yellow or a pink background. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

The lecture sums up how each family is made conditional (≈1:04:47): "It's really easy to make a diffusion model into a conditional
diffusion model. Just let the denoiser take text as input. It's really easy to make a GAN into a conditional GAN. Just let the
discriminator take the text as the input. It's really easy to make an autoregressive model into a conditional model. It already is a
conditional model. The VAE was the only one that was a little bit tricky."

Slide 67 shows how Stable Diffusion worked "a few years ago" ("Stable Diffusion has five or six different versions. So I don't know
which version this is"): at every denoising step the network was a **U-Net**, "just like a ConvNet with these skip connections"; "now
people … use what's called diffusion transformers, where this is a ViT" (≈1:04:47–1:05:32). The text came in "not by just concatenating
tokens, because this was not like a token architecture … And I think you modulated the batch normalization or something … scaling each
of the activations by, somehow, the outcoming vectors here. So you could look up the details" (≈1:05:32); the lecture does not settle
the mechanism.

![Slide 67: inputs, the text A profile photo of a robin, facing left and a square of noise; a text encoder f feeding three U-Nets in a row inside an image decoder g, the image growing less noisy after each; and the output, a robin](../raw/images/16-generative-models-conditional-models/slide-67.png)

*Slide 67 — Stable Diffusion's shape: the encoded text feeds every step of a chain of U-Net denoisers that turn noise into the picture. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

### Visual question answering and LLaVA

"Once you have this ability to plug together different modalities and solve general conditional prediction problems … this is a very
general thing" (≈1:05:32–1:06:19). **Visual question answering** (slide 68, Antol, Agrawal et al., ICCV 2015) encodes a question with a
text encoder and an image with an image encoder, and a decoder predicts the answer: "if this is trained on tuples of the form question,
image, answer, then it will learn to maximize the conditional probability of the answer, given the question and the image. And that's
general-purpose image understanding" (≈1:06:19).

![Slide 68: two copies of a diagram in which a question and a photograph of a yellow bird are encoded and decoded into an answer: What animal is this, Golden palm weaver; What color is the bird, yellow](../raw/images/16-generative-models-conditional-models/slide-68.jpg)

*Slide 68 — Visual question answering: the question and the image are encoded separately and decoded together into the answer. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

"Just to show you that I'm not … hiding too much under the hood here", slide 69 shows LLaVA ("Visual Instruction Tuning", Liu, Li et al.
2023), one of the current popular VQA models: an image encoder makes a set of tokens, the question, "or they call it instruction", makes
another, "they cross-attend. And they out — through a transformer, a GPT, and the output of language response. That's pretty simple"
(≈1:07:07).

![Slide 69: the LLaVA architecture figure: an image through a vision encoder and a projection W to tokens H v, a language instruction to tokens H q, and a language model producing the language response](../raw/images/16-generative-models-conditional-models/slide-69.jpg)

*Slide 69 — LLaVA: image tokens and instruction tokens go into one language model, which writes the response. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

### Mixing GANs, VAEs and autoregressive models

Slide 70 ("Taming Transformers", Esser, Rombach and Ommer 2020; excluded from OCW's licence) is "one of the important text-to-image models
from a few years ago. This is still the core of the Stable Diffusion line of work … up to 2023, let's say." "You actually first trained
what's called a VQGAN. That's a VAE with a patch GAN discriminator that tries to make the output look sharp … So it's like a combination of
a VAE and a GAN. And then in the latent space, I have either a diffusion model, or here it's actually an autoregressive model over the
latent variables" (≈1:07:53).

How is such a system trained? "You can potentially backprop through the whole system", or pre-train parts separately, for example the
VQGAN on images without text, "and then backprop through a interface with a language model only on the paired data." The "vector
quantization in the latent space in this particular model … makes backprop hard", which the lecture does not go into (≈1:08:40–1:09:25).

Two more questions. Do modern chatbots that take and give both images and text use one model or several? "Usually, they take image and
text as input and output image and text … I don't know how they work because these are all now companies that don't tell us"; Llama 3.2
"does have a paper that describes the details" (≈1:09:25–1:10:12). Do more modalities make a model more data-hungry? "I think it can go in
either direction": the mapping may be more complicated and need more data, but modalities share information, so "a pure language model
that is just trained on language might be worse at language modeling than a language model that's trained on language and videos"
(≈1:10:58–1:11:44).

## Unpaired translation

Everything so far modelled $P(Y \vert X)$ from data "that comes paired" (≈1:11:44). "What happens if I actually just have data x and data
y, but I have no examples of x paired with y? … oftentimes, you can do conditional modeling with unpaired data. And this is sometimes
called unpaired translation" (≈1:12:31). Slide 71 opens the section with a 1968 cartoon from Licklider and Taylor's "The Computer as a
Communication Device", a rough heart drawn on one console appearing as a neat valentine on another; the section "[Includes slides from
Jun-Yan Zhu, Taesung Park]". Slides 71 to 79, 81 and 82 are excluded from OCW's licence.

Paired data are easy to get for colourization, "I just take a color image. I remove the color channels", but not for "what would Cézanne
have painted if he had been standing at this river? I can't ask Cézanne to go and paint a million different rivers" (≈1:12:31–1:13:16;
slide 72, "Paired" grey and colour photographs against "Unpaired" sets of photographs and paintings). Unpaired data is "potentially more
common": paired data needs "both the x's and the y's" and "the pairing between them" (≈1:13:16).

### The GAN loss checks membership of a set

"Here's a really cool thing about GANs … With a GAN, the pairing between the input and the output is only coming through the
discriminator, which looks at, does the output go with the input? But I could drop that" (≈1:14:02). Slide 73 is the conditional GAN with
its pair discriminator, and slide 74 the unpaired case, where the output is unknown: "No input-output training pairs!" Drop the pairing,
and the discriminator asks only "does this look like a Cézanne painting or not? Now this is the traditional unconditional GAN objective"
(≈1:14:47; slide 75):

$$\arg\min_{\theta} \max_{\phi} \mathbb{E}_ {\mathbf{x},\mathbf{y}} \left[ \log d_{\phi}(g_{\theta}(\mathbf{x})) + \log(1 - d_{\phi}(\mathbf{y})) \right].$$

Slide 75 states the property: "Usually loss functions check if output matches a target *instance*. GAN loss checks if output is part of an
admissible *set*." "So it's suitable for unpaired learning." But it does not tie the output to the input (≈1:15:34): a painting of the
river scene is "Real!" (slide 76), and an unrelated painting of a house is "Real too!", with "Nothing to force output to correspond to
input" (slide 77). (The lecturer notes twice that "the animations are a little off" here, ≈1:14:02–1:15:34.)

### CycleGAN and cycle consistency

"The trick that works in this case is to use this idea of cycle consistency" (≈1:15:34–1:16:19; slides 78 and 79, "CycleGAN, or there and
back aGAN", Zhu, Park et al. 2017, with Yi et al. 2017 and Kim et al. 2017). One network maps photographs to paintings, checked by a
discriminator $d_{\phi}^{Y}$ that asks whether the output looks like a Cézanne; a second maps paintings back to photographs, checked by
$d_{\phi}^{X}$. "The discriminators will make sure I map to the domain of Cézanne paintings and the domain of photos. But I'll train these
things to be what's called cycle-consistent, which just means that if I go from domain X to domain Y and then I go back, I should arrive
where I started. And if I go from domain Y to domain X and I go back, I should arrive where I started" (≈1:16:19–1:17:07).

![Slide 80: two halves, x mapped by g theta to y-hat and by f back to x-hat, and y mapped by f to x-hat and by g theta back to y-hat, each with a discriminator on the middle image, and below each a cycle-consistency loss between the start and the return point](../raw/images/16-generative-models-conditional-models/slide-80.png)

*Slide 80 — Cycle consistency: go from one domain to the other and back, and penalize the distance from where you started, in both directions. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec16.pdf)*

Slide 80 draws the loss: $\mathbf{x}$ goes by $g_{\theta}$ to $\hat{\mathbf{y}}$ and by $f$ back to $\hat{\mathbf{x}}$, the distance between
$\mathbf{x}$ and $\hat{\mathbf{x}}$ is the "cycle-consistency loss", and the same runs from $\mathbf{y}$. Language has the same property, the
lecturer says, recalling "an old Mark Twain story about trying to translate between a French speaker and an English speaker": translate
into French and back, and "you know that the system is consistent with itself if you arrive where you started", since natural languages
"tend to have a bijection between the different languages … not perfectly" (≈1:17:07–1:17:53). "That actually works decently well", and
turns a photograph into a synthetic Cézanne "without having paired data" (≈1:17:53).

### Is the mapping unique?

"There must be many cycle-consistent mappings between any set of images x and any set of images y. And indeed, that's true." But "under the
constraints of a particular neural net architecture in a particular optimizer, only some of the possible mappings from inputs to outputs are
going to be found … and neural networks do find simple mappings. We've talked about that" (≈1:17:53–1:18:38). Slide 81 ("Unique solution?",
Galanti, Wolf and Benaim, ICLR 2018) maps a segment A to a segment B: "There's infinite possible mappings. But a ReLU network trained with
gradient descent with two hidden units can only find two mappings in this case … One is the identity mapping. And the other is the flipping
of the lines, like negative identity." Wiggly mappings that send each point to a random point "are very hard … to be found by the
architectures that we use and the optimizers that we use", so the simple mapping, green to green, wins (≈1:18:38–1:19:23). See
[inductive bias](inductive-bias.md).

### Unpaired translation in language, and grounding

The same ideas appear in unpaired machine translation, as "back translation or cycle consistency between the mapping from one language to
the other and back" (≈1:19:23; slide 82, Lample et al. 2018). Another family aligns distributions: "if I have a distribution of data in one
modality … and the data distribution of the blue modality. Well, I can just align their shapes. It's like a rigid transformation problem"
(≈1:20:09).

The lecture ends on a concept (slide 83): "**paired data is not always necessary for grounding** (finding a unique mapping between two
domains)". "One name for pairing is grounding, grounding one modality in another … I could potentially identify the meaning of a word
without ever having seen a paired example of what that word corresponds to in the physical world. And I'll have more to say on that in some
future lectures" (≈1:20:56).

## Pointers to other lectures

The deck prints no lecture number but its own, on the title slide. The recording points back to "last lecture", which "was a little bit
math-heavy" (lecture 15, ≈0:00), to "the hacker's guide lecture" and its quantization hack (lecture 9, ≈9:13), to GANs shown "very briefly
in one of the previous lectures" (lecture 14, ≈24:12), to "the VAEs we saw last class" (lecture 15, ≈42:03) and to "the transformers lecture"
(lecture 8, ≈52:15); and ahead to "another lecture on language models" (lecture 21, ≈51:27) and, for grounding, to "some future lectures"
that it does not name (≈1:20:56). Lecture 14's outline had announced this one as "Lecture 16: conditional models, data prediction". See the
[course map](course-map.md).

## See also

- [Generative models](generative-models.md) — the hub page: what a generative model is, the families, and (from this lecture) structured
  prediction and conditional models.
- [Generative adversarial networks](generative-adversarial-networks.md) — the conditional GAN, the discriminator as a learned loss, the patch
  discriminator, and CycleGAN.
- [Variational autoencoders](variational-autoencoders.md) — the conditional VAE, whose latent encodes only what the input leaves out.
- [Diffusion models](diffusion-models.md) and [autoregressive models](autoregressive-models.md) — conditioned by giving the denoiser the text,
  and conditional by construction.
- [Transformers](transformers.md) — cross-attention, the way image and text tokens are joined.
- [Softmax and cross-entropy](softmax-and-cross-entropy.md) — why a categorical distribution captures two modes that regression averages.
