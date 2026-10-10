# Generative models

A **generative model**, in this course, is "an algorithm that generates data" — images, proteins, stories — rather than
one that classifies or decides about data ([lecture 14](14-generative-models-basics.md), slide 4). It is the inverse of
[representation learning](representation-learning.md): representation learning goes from data to a simple,
low-dimensional embedding, and generative modeling goes "from simple low-dimensional embeddings to data" (lecture 14,
≈0:45). The lecturer thinks "they're really the same topic" (≈2:18). Covered so far: lecture 14, slides 2–59,
≈0:00–1:20:47 (fundamentals, density and energy models, autoregressive models, diffusion models and GANs). Lecture 15
covers variational autoencoders and lecture 16 conditional models and applications (lecture 14, slide 2; see the
[course map](course-map.md)).

## Two definitions

Lecture 14's slide 4 gives two definitions in use: (1) an algorithm that generates data, and (2) a statistical model of the
joint distribution of some data, $p(x, y, \ldots)$. The second is the "slightly older definition" of statistics classes;
the course adopts the first, because "our real goal in generative modeling, especially in this area that we would call
generative AI, is we want algorithms that make things that look like data" (≈3:48–4:33). Most generative models do it "by
either explicitly or implicitly modeling a distribution" (≈5:21).

This changes how a network is viewed. Instead of a function $f_{\theta} : \mathcal{X} \to \mathcal{Y}$ that returns one
output per input, a network becomes a map $f_{\theta} : \mathcal{X} \to \mathcal{P}(\mathcal{Y})$ into the probability
distributions over outputs — which softmax classification already is (lecture 14, slide 7; see
[softmax and cross-entropy](softmax-and-cross-entropy.md)). A network can represent a distribution in two ways: it can
output samples from it, or it can output its parameters, such as a Gaussian's mean and variance (≈16:10).

## Latent variables: "noise" as control knobs

A generator is a deterministic function, so to produce many outputs for one input it is fed random numbers as well —
"dice" (lecture 14, slide 12). These are often called noise, but "they're not noise in the sense that they're meaningless or
something that you want to get rid of". They specify "everything about the data that's not specified by your input command"
— the colour, angle and size of the bird (slide 13, ≈18:28–19:14) — and can be set deliberately, like knobs, to steer the
output (slide 14, ≈20:00). Their formal name is **latent variables**: "any variable that determines how the data looks that
is not directly observed" (slide 16, ≈23:03). Lecture 14's first printed concept is "noise is latent variables". The
procedural-graphics river of slides 15 and 16, driven by coin flips, is a hand-written generative model of this kind.

## Direct and indirect: samplers, densities and energies

There are two ways to learn a generator (lecture 14, slide 17). The **direct approach** learns a function
$G : \mathcal{Z} \to \mathcal{X}$ from latent variables to data, trained on examples and then run to sample (slide 18); it is,
"confusingly, sometimes called an 'implicit generative model'", because the sampler implies a distribution without writing
it down. The **indirect approach** learns a function that scores data, $E : \mathcal{X} \to \mathbb{R}$, and generates by
finding or sampling points that score highly, with a separate sampling algorithm such as Markov chain Monte Carlo (slide 19,
≈27:48–28:36). The score can be a **density** $p_{\theta} : \mathcal{X} \to [0, \infty)$, which integrates to 1, or an
**energy** $E_{\theta} : \mathcal{X} \to \mathbb{R}$, which need not ([energy-based models](energy-based-models.md)).
Lecture 14's second concept: "you can represent the data generating process directly or indirectly" (slide 34). Many deep
generative models have several of these forms at once (≈58:58).

| Family (lecture 14) | Represents | Trained by |
| --- | --- | --- |
| [Autoregressive models](autoregressive-models.md) | a density, as a product of next-element conditionals | maximum likelihood, one classification per element |
| [Energy-based models](energy-based-models.md) | an energy | contrastive divergence |
| [Diffusion models](diffusion-models.md) | a generator from Gaussian noise, step by step | supervised denoising |
| [GANs](generative-adversarial-networks.md) | a generator, judged by a discriminator | a min-max game |

The lecturer places the autoregressive model among density models (≈1:05:11) and calls diffusion models and GANs direct
(≈1:08:21, ≈1:18:26), while allowing that "these things are a little fuzzy".

## The goal: new data that looks real, measured by likelihood

"What does it mean to look like the training data? … We've kind of converged on one main one in machine learning": the
data should have high probability under a density model fit to real data (lecture 14, slide 20, ≈29:24). Fitting a density
$p_{\theta}$ to the unknown data distribution $p_{\texttt{data}}$ by minimizing the KL divergence reduces to **maximum
likelihood** (slide 24):

$$p^{\ast}_ {\theta} = \underset{p_{\theta}}{\arg\min} \thinspace \texttt{KL}(p_{\texttt{data}}, p_{\theta}) = \underset{p_{\theta}}{\arg\max} \thinspace \mathbb{E}_ {\mathbf{x} \sim p_{\texttt{data}}} \left[ \log p_{\theta}(\mathbf{x}) \right] \approx \underset{p_{\theta}}{\arg\max} \thinspace \frac{1}{N} \sum_{i=1}^{N} \log p_{\theta}(\mathbf{x}^{(i)})$$

where $\lbrace \mathbf{x}^{(i)} \rbrace_ {i=1}^{N}$ is the training set. The term $\mathbb{E}_ {\mathbf{x} \sim p_{\texttt{data}}}[\log p_{\texttt{data}}(\mathbf{x})]$ drops out because it does not depend on the model — fortunately,
since nobody can evaluate $p_{\texttt{data}}$ (≈34:57–35:44). Because a density has constant mass, raising it where the data
are lowers it everywhere else (slides 22–24, ≈30:57–31:43). The [lecture page](14-generative-models-basics.md#density-models-and-maximum-likelihood)
works through the derivation.

## Overfitting: the filing cabinet

A model that memorizes its training set and samples a stored example at random — the filing cabinet of
[lecture 6](06-generalization-theory.md), recast as a generative model — achieves the highest likelihood on the training
data possible: a delta function on every training point (lecture 14, slides 25–26). But it puts zero probability on new
samples from the same process. "It's exactly the same as overfitting in classical machine learning" (≈38:04). So the goal "is
not to replicate the training data but to make *new* data that is *realistic*", and one way to measure it is the
likelihood of held-out test data (slide 27). That is easy to cross-validate for density models and hard for direct
samplers, which give no density (≈40:25). The usual remedies — early stopping, regularization, architectures with inductive
biases such as convolution's spatial locality — all apply (≈41:57–43:29; see
[generalization and double descent](generalization-and-double-descent.md)).

## Turning generation into supervised learning

Lecture 14's third concept: "A common strategy is to turn generative modeling into a sequence of supervised learning
problems" (slide 53). An autoregressive model takes a data point apart one element at a time and learns to classify each next
element; a diffusion model adds noise until the data point is a simple Gaussian and learns to undo each step. "You change
generative modeling of a high-dimensional structured object into a sequence of very, very simple modeling problems"
(≈1:15:21–1:16:52). The lecturer also says that "all generative models use the same principles … A diffusion model is a type
of variational autoencoder, and an autoregressive model is a small variation on a diffusion model" (≈3:03–3:48).

## Uses

Generative models make the text-to-image pictures of DALL-E 2 (lecture 14, slide 5), but also scientific data: DiffDock, a
diffusion model that docks a ligand on a protein, and an image-to-image model that predicts the CT scan an MRI scan would
have been (slide 6, ≈6:06–6:53). "Generative models are not just for entertainment" (≈6:53). Their applications are lecture
16's subject (≈1:31).
