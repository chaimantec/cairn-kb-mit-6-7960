# Generative models

A **generative model**, in this course, is "an algorithm that generates data" — images, proteins, stories — rather than
one that classifies or decides about data ([lecture 14](14-generative-models-basics.md), slide 4). It is the inverse of
[representation learning](representation-learning.md): representation learning goes from data to a simple,
low-dimensional embedding, and generative modeling goes "from simple low-dimensional embeddings to data" (lecture 14,
≈0:45). The lecturer thinks "they're really the same topic" (≈2:18). Covered so far: lecture 14, slides 2–59,
≈0:00–1:20:47 (fundamentals, density and energy models, autoregressive models, diffusion models and GANs);
[lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md), slides 3–5, 10–11 and 35–41, ≈0:47–3:55, ≈17:10–21:06 and ≈1:11:40–1:20:09 (generative models as
distribution transformers, latent variable models and the marginal likelihood, what their latent spaces learn, and the families
compared); [lecture 16](16-generative-models-conditional-models.md), slides 3–6, 15, 50, 59 and 83, ≈1:31–4:34, ≈13:04, ≈39:45, ≈45:10–45:58, ≈57:45 and ≈1:04:47 (generative models
for structured prediction, conditioning each family, and explicit inputs against latent variables). Lecture 15's model, the
[variational autoencoder](variational-autoencoders.md), has its own page, and lecture 16's conditional models have
[theirs](conditional-generative-models.md).

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
| [Variational autoencoders](variational-autoencoders.md) (lecture 15) | a density as an infinite mixture of Gaussians, and a generator | maximizing a lower bound on the likelihood (the ELBO) |

The lecturer places the autoregressive model among density models (≈1:05:11) and calls diffusion models and GANs direct
(≈1:08:21, ≈1:18:26), while allowing that "these things are a little fuzzy".


## Distribution transformers and latent variable models (lecture 15)

[Lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md) restates the picture: "Deep generative models are distribution transformers" (slide 4), mapping a simple
**prior** $p(z)$, usually a Gaussian, onto a complicated data distribution $p(x)$, the network's layers pushing and pulling the
prior "geometrically" into the data's shape (≈2:19–3:06). In high dimensions a Gaussian prior is "roughly a soap bubble … Most of the
probability density of a Gaussian in high dimensions is right near the surface of the hypersphere" (≈2:19). The generator runs the
other way from a representation learner's encoder, and "these are sometimes called generators. Other times, they're called decoders"
(slide 5, ≈3:55).

Its latent variables are lecture 14's control knobs given a probabilistic reading (slide 10). As a graphical model, the latent $z$
generates the observed $x$, and only $x$ is seen; the generator is a conditional distribution $p_{\theta}(\mathbf{x} \vert \mathbf{z})$,
and the probability of a data point marginalizes the latents out:

$$p_{\theta}(\mathbf{x}) = \int_{\mathbf{z}} p_{\theta}(\mathbf{x} \vert \mathbf{z}) p_{\mathbf{z}}(\mathbf{z}) d\mathbf{z}$$

"So sometimes we call this marginal likelihood. So a latent variable model, I'm going to fit it to data by maximizing the marginal
likelihood that it assigns to data when I marginalize over all the latent variables" (≈18:46–19:34). A model of this kind gives both a
**density** $p_{\theta} : \mathcal{X} \to [0, \infty)$ and a **sampler** $g_{\theta} : \mathcal{Z} \to \mathcal{X}$ (slide 11, ≈21:06);
the catch is that the integral usually cannot be computed, which is what the [variational autoencoder](variational-autoencoders.md)'s
tricks are for.

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


## What their latent spaces learn (lecture 15)

"Generative modeling forces you to explain the world" ([lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md), ≈1:17:52). A VAE trained on lecture 14's procedural river
tiles learns latent axes that bend the river and change the grass colour (slides 37–38); BigGAN's latent space has directions for a
bird's orientation and the background colour (slide 39); and diffusion models' noise and a layer's activations show the same kind of
structure (≈1:17:06). The factors are **disentangled** but need not be the true causes of the data, and squeezing a complicated shape
onto a Gaussian leaves seams where a smooth walk through the latent space suddenly changes category (slide 40, ≈1:18:37–1:19:23). See
[representation learning](representation-learning.md#both-directions-at-once-and-disentanglement-lecture-15).

## The families compared (lecture 15)

Lecture 15 closes with "a rough categorization of vanilla versions of each model" by three questions (slide 41): does it have latent
variables, does it give a density or energy, and does it give a generator?

| Method | Latents? | Density/Energy? | Generator? |
| --- | --- | --- | --- |
| [Energy-based models](energy-based-models.md) | ✗ | ✓ (energy only) | ✗ |
| Gaussian | ✗ | ✓ | ✗ |
| [Autoregressive models](autoregressive-models.md) | ✗ | ✓ | ✓ (slow) |
| [Diffusion models](diffusion-models.md) | ✓ (high-dimensional) | ✗ | ✓ (slow) |
| [GANs](generative-adversarial-networks.md) | ✓ | ✗ | ✓ |
| [VAEs](variational-autoencoders.md) | ✓ | ✗ | ✓ |

"I think the most popular right now is diffusion models. The most complete, like full picture, that encompasses everything is VAEs.
The most cute and fun is GANs. The ones that are popular in language modeling is autoregressive. And they all have different
trade-offs" (≈1:19:23–1:20:09). The VAE row's ✗ for a density differs from slide 11, which lists a density among a VAE's outputs; the
[lecture page](15-generative-models-representation-learning-meets-generative-modeling.md#comparing-the-families) notes it.

## Conditional models and structured prediction (lecture 16)

[Lecture 16](16-generative-models-conditional-models.md) turns the families to prediction: given an input $\mathbf{x}$, model the whole distribution of
outputs $\mathbf{y}$. Problems whose output is a high-dimensional object with dependent dimensions are **structured prediction**
(slide 4), and the lecturer finds generative models suited to almost all of them: "I'll frame all problems that are predicting
high-dimensional structured data or information out of a network as being appropriate to be treated with generative modeling"
(≈1:31). Its working definition of the term comes from what regression and per-pixel classification lack: "Generative models
have two important properties for structured prediction: 1. They can model a multimodal distribution 2. They can model joint
dependences between multidimensional predictions" (slide 6, ≈13:04). Each family is made conditional by a small change, the
discriminator, the denoiser or the decoder seeing the input, and an autoregressive model is conditional already (≈1:04:47); a
conditional model's latent variables then encode only what the input leaves out, so "you can control your data either via explicit
inputs or latent variables" (slide 50, ≈45:10–45:58). Text-to-image systems combine GANs, VAEs and diffusion models (≈57:45). See
[conditional generative models](conditional-generative-models.md).

## Uses

Generative models make the text-to-image pictures of DALL-E 2 (lecture 14, slide 5), but also scientific data: DiffDock, a
diffusion model that docks a ligand on a protein, and an image-to-image model that predicts the CT scan an MRI scan would
have been (slide 6, ≈6:06–6:53). "Generative models are not just for entertainment" (≈6:53). Their applications are lecture
16's subject (≈1:31); see [conditional generative models](conditional-generative-models.md).
