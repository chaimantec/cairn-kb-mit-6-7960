# Diffusion models

A **diffusion model** generates data by starting from random noise and removing a little of it at a time. It is learned by
running the opposite, easy process on the training data — adding noise step by step until each data point is pure Gaussian
noise — and training a network by supervised learning to undo each step. "Diffusion models convert data to noise and then just
learn to reverse the process" ([lecture 14](14-generative-models-basics.md), ≈1:09:54). Covered so far: lecture 14, slides
43–53, ≈1:07:34–1:17:40, and Homework 5's "Diffusion Models" section; [lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md), ≈8:37, ≈1:17:06,
≈1:18:37–1:20:09 and slide 41 (diffusion as a kind of VAE, its latent structure, and the comparison of families); [lecture 16](16-generative-models-conditional-models.md), slides 65–67
and 70, ≈1:02:29–1:05:32 and ≈1:07:53 (conditioning on text, Stable Diffusion's U-Net). See also
[generative models](generative-models.md). [Lecture 24](24-inference-methods-for-deep-learning.md) (slide 16, ≈22:31–24:05) asks whether step-by-step denoising finds the
most likely image.

## The observation: data to noise is easy

"It's very hard to convert noise, random variables, into data, structured objects, but it's really easy to convert structured
objects into noise" (lecture 14, ≈1:08:21). Pop a balloon animal full of coloured gas and the particles spread out until they
are uniformly distributed, "Physically, that's called a diffusion process"; a diffusion model does the same in a
high-dimensional data space, perturbing a data point at random until it has a simple distribution, typically Gaussian
(≈1:08:21–1:09:09). Slide 45 shows a pixel-art bird dissolving into coloured noise: "Diffusion: Just add noise."

## Reversing it with supervised learning

Generating data in one step from noise is hard, but each small step back is easy: "just remove a little bit of noise, right?
It's a very easy thing to model" (≈1:09:54). So noise the training images, reverse the sequences, cut them into pairs of a
noisier image $\mathbf{x}_ t$ and a less noisy one $\mathbf{x}_ {t-1}$, with $t$ indexing the noise level, and fit a denoiser
$f$ from a function class $\mathcal{F}$ (slides 46–47):

$$\underset{f \in \mathcal{F}}{\arg\min} \sum_{i=1}^{N} \mathcal{L}(f(\mathbf{x}_ t), \mathbf{x}_ {t-1})$$

with $\mathcal{L}$ as simple as the squared loss. This "converts generative modeling into a bunch of supervised prediction
problems" (slide 47). The denoiser is usually one network told which step it is on, "you condition on t", though a different
function per step would also do (≈1:11:26).

To sample, draw a unit Gaussian vector $\mathbf{z} \sim \mathcal{N}(0, 1)$ and apply the denoiser over and over. Each draw is
a different set of latent variables and gives a different image (slide 48, ≈1:12:13). That repetition is the cost: "When these
first came out, t was large, like, a thousand. Now people are showing how to do this with t being pretty small, like, even
going down to 1. But you typically have to do a few time steps" (≈1:12:13–1:12:58).

## Gaussian diffusion

Following Ho, Jain and Abbeel (2020), lecture 14's slides 49–50 make the two processes Gaussian. The **forward process** adds
unit Gaussian noise $\epsilon_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ scaled by a schedule $\beta_t$:

$$\mathbf{x}_ t = \sqrt{(1 - \beta_t)} \thinspace \mathbf{x}_ {t-1} + \sqrt{\beta_t} \thinspace \epsilon_t, \qquad q(\mathbf{x}_ t \mid \mathbf{x}_ {t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} \thinspace \mathbf{x}_ {t-1}, \beta_t)$$

chosen so that, as $t$ grows, the result approaches unit Gaussian noise (≈1:13:45). The **reverse process** treats each step as
"a small Gaussian max likelihood density modeling problem": the network predicts the mean of a Gaussian over the less noisy
image, which is then sampled (≈1:13:45–1:14:33):

$$\mathbf{x}_ {t-1} \sim \mathcal{N}(f_{\theta}(\mathbf{x}_ t, t), \sigma^2), \qquad p_{\theta}(\mathbf{x}_ {t-1} \mid \mathbf{x}_ t) = \mathcal{N}(f_{\theta}(\mathbf{x}_ t, t), \sigma^2)$$

"The variances, beta and sigma, are modeling choices" (slide 49). The letters $q$ for the forward process and $p$ for the
learned reverse one connect "to the standard notation and things like called variational inference" (≈1:14:33). Slide 51's
"stripped down training algorithm" generates the noised sequences for every training example and every $t = 1, \ldots, T$,
then fits $\theta^{\ast} = \arg\min_{\theta} \sum_{i} \sum_{t} \mathcal{L}(f_{\theta}(\mathbf{x}_ t^{(i)}, t), \mathbf{x}_ {t-1}^{(i)})$; the lecturer's Colab implements it ([lecture 14](14-generative-models-basics.md#gaussian-diffusion)).

Homework 5 continues from here: it derives the closed form of $q(\mathbf{x}_ t \mid \mathbf{x}_ 0)$, a denoising objective
from the KL divergence to the true reverse process and its noise-prediction form, and trains a model in a Colab
([lecture 14](14-generative-models-basics.md#the-problem-set)).

## Diffusion and autoregression

Both diffusion and [autoregressive models](autoregressive-models.md) turn generation of a structured object into "a sequence
of very, very simple modeling problems" (lecture 14, ≈1:15:21–1:16:07). A diffusion model adds noise globally, to every pixel,
until the data point is a simple Gaussian; an autoregressive model removes one pixel at a time until nothing is left; each
sequence, reversed, supervises the model (slides 52–53). The lecturer goes further: "an autoregressive model is a small
variation on a diffusion model", and "a diffusion model is a type of variational autoencoder" (≈3:03–3:48), the subject of
lecture 15. Diffusion also appears among lecture 14's examples, in DiffDock's "reverse diffusion over translations, rotations
and torsions" for docking a ligand on a protein (slide 6).

## Diffusion models and VAEs (lecture 15)

[Lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md) repeats the claim in its own words: "many of the models we've seen, like diffusion models, can be cast as a certain kind
of VAE or autoencoders or a degenerate VAE" (≈8:37); see [variational autoencoders](variational-autoencoders.md). It does not derive the
connection. It also counts diffusion models among the generative models whose internal variables learn structured representations, "like in
diffusion models, maybe the noise", or directions in the activation space of a layer (≈1:17:06), and warns that navigating "diffusion model
activation space on some layer of the network" can hit seams where the output "suddenly … switches to a different category" (≈1:19:23). In
the closing comparison (slide 41) marks diffusion models as having latent variables ("high-dimensional") and a generator ("slow") but no density, and the lecturer calls them "the most popular right now" (≈1:19:23).

## Conditioning on text (lecture 16)

"To make it conditional on text, all I have to do is let my denoiser observe the text" ([lecture 16](16-generative-models-conditional-models.md), slide 65, ≈1:03:14). It works
"because the text will be informative toward what the true image I started with is": a noisy image might have a yellow, white or pink
background, and "A robin on a yellow background" settles it, so the denoiser "will learn to use the text and learn to denoise in the
direction guided by the text", to the degree the training data link text and image (slide 66, ≈1:03:14–1:04:00). "It's really easy to make
a diffusion model into a conditional diffusion model. Just let the denoiser take text as input" (≈1:04:47). DALL-E 2, Stable Diffusion and
Midjourney took this route after DALL-E 1's VAE (≈1:02:29).

In the version of Stable Diffusion that slide 67 draws, every denoising step was a **U-Net**, "just like a ConvNet with these skip
connections", into which the text came by modulating the network's activations rather than as tokens ("I think you modulated the batch
normalization or something … So you could look up the details", ≈1:04:47–1:05:32); "now people … use what's called diffusion transformers,
where this is a ViT". The Stable Diffusion line was built on VQGAN, with a diffusion or autoregressive model in its latent space (≈1:07:53).
See [conditional generative models](conditional-generative-models.md).

## Greedy denoising (lecture 24)

[Lecture 24](24-inference-methods-for-deep-learning.md) asks of diffusion models what it shows for autoregressive ones, that taking the most likely step at each stage need not give the most
likely whole (slide 16: "What about for VAEs? What about for diffusion models?"). The answer: "it's the same exact thing with diffusion
models". Each sampling step tries "to maximize the conditional probability of slightly less noisy given slightly more noisy", and "that
won't necessarily result in a sequence of denoising steps that maximizes the probability at the end of the sequence for the same kind of
reason" (≈22:31–23:19). "There's not as much work in trying to do intelligent inference through diffusion models", so how to "take higher
likelihood samples from a diffusion model than you'd get with the regular denoising process" is offered as an open topic (≈24:05). See
[search and test-time compute](test-time-compute.md).
