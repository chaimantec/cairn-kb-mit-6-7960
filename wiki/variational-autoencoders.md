# Variational autoencoders

A **variational autoencoder** (VAE) is an [autoencoder](autoencoders.md) whose encoder maps each data point not to a single code
but to a Gaussian in a latent space, trained so that the codes together fill a unit Gaussian. That one change makes it a
[generative model](generative-models.md) as well as a [representation learner](representation-learning.md): sample a vector from
the unit Gaussian, decode it, and you have new data. The course calls it "one model that really puts the two ideas together and
solves them both at the same time" ([lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md),
≈0:00), and "mathematically, … a framework that encompasses almost everything that we've seen", though "in practice, they're not
as popular these days as diffusion models" (≈9:22). Covered so far: lecture 15, slides 3–41, ≈0:00–1:20:09; the model is promised
in [lecture 11](11-representation-learning-reconstruction-based.md) (≈47:20–48:06) and placed in lecture 15 by
[lecture 14](14-generative-models-basics.md) (slide 2, ≈1:31); and its conditional form in [lecture 16](16-generative-models-conditional-models.md), slides 39–50 and
63–64, ≈39:45–45:58 and ≈59:21–1:01:42; and its seams again in [lecture 19](19-transfer-learning-data.md), slide 18, ≈20:20–21:55.

**Notation**, from the slides: a data point $\mathbf{x}$, a latent vector $\mathbf{z}$ with prior $p_{\mathbf{z}}$, the unit
Gaussian; a decoder or generator $g_{\theta}$ with parameters $\theta$, outputting a mean $g^{\mu}_ {\theta}(\mathbf{z})$ and a
covariance $g^{\Sigma}_ {\theta}(\mathbf{z})$; an encoder $f_{\psi}$ with parameters $\psi$, outputting $f^{\mu}_ {\psi}(\mathbf{x})$
and $f^{\Sigma}_ {\psi}(\mathbf{x})$. $\mathcal{N}(\mathbf{x}; \mu, \mathbf{\Sigma})$ is a Gaussian density and
$\mathtt{KL}(q \parallel p)$ the KL divergence. Homework 5 writes the encoder's parameters as $\phi$.

## The problem it solves

An ordinary autoencoder's decoder already maps codes to data, so why not sample a code and decode it? Because nothing says what
distribution to sample from: "there's no guarantee that set of embeddings is going to form a simpler shape or easier to sample from
distribution than the original data itself … It might be some weird distribution full of holes. And if I go to a point that has
never been … observed via f of x, then I won't know how to decode that. So the whole point of a variational autoencoder is to fix
this problem" (lecture 15, slide 6, ≈5:30–6:18). The first, mechanical view: "a VAE is an autoencoder with one change, which is that
rather than the encoder mapping to a complicated shape, it will map to a simple shape. It will map to a Gaussian" (≈7:04–7:49).

## The probabilistic model: an infinite mixture of Gaussians

The second view is that a VAE fits a mixture of Gaussians to the data by maximum likelihood — an **infinite** one (slides 8–9). A
finite mixture $\sum_{i=1}^{k} w_i \mathcal{N}(\mathbf{x}; \mu_i, \mathbf{\Sigma}_ i)$ needs parameters for every component; an
infinite one is parameterized instead by a function $g_{\theta}$ that maps each point $\mathbf{z}$ of a continuous base distribution to
the mean and covariance of one component, "a finite number of parameters to model an infinite mixture" (≈14:47). The prior weights the
components, and the density is a **marginal likelihood**, an integral over the unobserved latent variables (slides 10–11, ≈18:46–19:34):

$$p_{\theta}(\mathbf{x}) = \int_{\mathbf{z}} \underbrace{\mathcal{N}(\mathbf{x}; g^{\mu}_ {\theta}(\mathbf{z}), g^{\Sigma}_ {\theta}(\mathbf{z}))}_ {p_{\theta}(\mathbf{x} \vert \mathbf{z})} \underbrace{\mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})}_ {p_{\mathbf{z}}(\mathbf{z})} d\mathbf{z}$$

The model gives both a density $p_{\theta}$ and a sampler $g_{\theta}$ (slide 11). Nothing here needs a neural network: "a VAE doesn't
have to have anything to do with neural networks. g could be any type of mapping", though in this course it is one (≈13:12). Nor need
the components be Gaussian (≈16:24–17:10).

**Sampling, and the reparameterization trick.** To draw $\mathbf{x}$ from one component, sample unit Gaussian noise and transform it:
$x = g^{\mu}_ {\theta}(\mathbf{z}) + g^{\sigma}_ {\theta}(\mathbf{z}) \epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$ (slide 11).
The noise is scaled by the standard deviation, not the variance, as the lecturer settles after a slip of the tongue (≈22:42–25:49).
Because the randomness enters "through the side", $\mathbf{x}$ is a deterministic function of the network's outputs, and gradients
can flow through the mean and the spread (≈25:03, ≈1:07:50–1:08:36).

## Three tricks for an intractable integral

Maximizing $\sum_{i=1}^{N} \log p_{\theta}(\mathbf{x}^{(i)})$ (slide 12) would mean computing the integral at every step, which has no
closed form; "one step of backprop is running my neural network forward many, many, many times" (≈24:17). Lecture 15 approximates it
with three tricks (slide 33).

1. **Monte Carlo.** The integral is an expectation over the prior, so average $p_{\theta}(\mathbf{x} \vert \mathbf{z}^{(j)})$ over
   $M$ samples $\mathbf{z}^{(j)} \sim p_{\mathbf{z}}$ (slides 13–14), an unbiased estimator as $M$ grows (≈28:57–29:44). This works for
   a two-dimensional example but "in practice, this doesn't work in high dimensions" (≈38:16); the lecturer's intuition is that the
   number of samples needed grows exponentially with the dimension, a point he leaves to discuss "offline" (≈36:45–38:16).
2. **Importance sampling.** For a given $\mathbf{x}$, almost every prior sample lands on a component far away and contributes nearly
   zero (slides 15–16). Sampling instead from a density $q_{\mathbf{z}}$ that is non-zero everywhere and reweighting by
   $p_{\mathbf{z}}(\mathbf{z}) / q_{\mathbf{z}}(\mathbf{z})$ leaves the expectation unchanged (slide 17). The lecturer asserts, without
   proof, that the best choice is the posterior $p_{\theta}(Z \vert \mathbf{x})$, with which one sample does the work of many (slides
   18–19, ≈41:19–42:55).
3. **An encoder predicts the sampling distribution.** The posterior is unknown, so a second network learns it: $f_{\psi}$ maps
   $\mathbf{x}$ to $q_{\psi}(Z \vert \mathbf{x}) = \mathcal{N}(f^{\mu}_ {\psi}(\mathbf{x}), f^{\Sigma}_ {\psi}(\mathbf{x}))$, trained
   to minimize $\mathtt{KL}(q_{\psi}(Z \vert \mathbf{x}) \parallel p_{\theta}(Z \vert \mathbf{x}))$ (slides 20–23). "That's an
   encoder" (≈45:13). Predicting the latents that explain a data point is what "we call … inference" (≈48:21).

## The evidence lower bound

Taking the log of a Monte Carlo estimate is biased, so the log is moved inside the expectation by Jensen's inequality, which gives a
lower bound on the log likelihood (slide 25, ≈50:43–52:18). Rearranged, it is the **evidence lower bound**, or ELBO:

$$J = \mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \vert \mathbf{x})}\left[\log p_{\theta}(\mathbf{x} \vert \mathbf{z})\right] - \mathtt{KL}\left(q_{\psi}(Z \vert \mathbf{x}) \parallel p_{\mathbf{z}}\right)$$

The encoder's KL objective equals the same $J$ up to a term that does not depend on $\psi$, so "p and q share the same exact objective"
and both networks maximize $J$ jointly (slides 23–25, ≈53:54). The first term asks how much probability the decoder places on the data;
the second pays for having sampled from $q$ instead of the prior (≈54:43).

**Reconstruction and squashing.** For one-dimensional $x$ and $z$ and one sample, the first term is
$\log \frac{1}{\sigma_x \sqrt{2\pi}} - \frac{(x - \mu_x)^2}{2\sigma_x^2}$, a squared **reconstruction error** plus normalization, and the
KL term has the closed form $\frac{1}{2}\left(\mu_z^2 + \sigma_z^2 - \log(\sigma_z^2) - 1\right)$, subtracted (slide 28). With the
variances fixed, the first is exactly an autoencoder's mean squared error (≈59:23) and the second just "squash[es] the means toward the
origin" (≈1:01:43). Reconstruction stops the codes collapsing to a point — "that's the fundamental tension of the variational autoencoder"
(≈1:04:00) — and the squashing stops them flying apart and leaving gaps the decoder has never seen (≈1:10:08–1:11:40). The result packs
the encoder's Gaussians to "evenly tile the unit Gaussian" (slides 31–32, ≈1:06:18).

## Training

Slide 34's loop: sample a batch from the training set, encode it, add noise to each code from the encoder's Gaussian, decode, compute the
squared error and the KL term, and backpropagate to update $\theta$ and $\psi$. "Usually people just take one" importance sample per data
point (≈1:07:50). In practice simpler variants — an autoencoder with noise in the latent space, with or without the KL penalty — can work,
"but your mileage will vary" (≈1:08:36–1:09:21).

## What the latent space learns

Trained on the river tiles of lecture 14's procedural program, a VAE's latent axes come to control **disentangled** factors: one axis bends
the river and another changes the grass colour (slides 35–38, ≈1:13:15). The colour is a true cause, a normal variable in the program; the
curvature is not, since the program drew the river from a sequence of coin flips, so the VAE found "this abstracted, intuitive physics"
(≈1:14:00–1:15:34). GANs show the same structure without an encoder (slide 39). And mapping a complicated shape onto a Gaussian leaves
**seams**, where a smooth walk through latent space suddenly switches category, and **tendrils** of probability (slide 40,
≈1:18:37–1:19:23). See [representation learning](representation-learning.md).

## Relation to other models

Slide 41 credits a vanilla VAE with latent variables and a generator. It marks them ✗ for a density or energy, although slide 11 lists a density
$p_{\theta}$ among a VAE's outputs; in the lecture that density is an integral that cannot be computed, and training uses a lower
bound on it. The lecturer calls VAEs "the most complete, like full
picture, that encompasses everything" (≈1:20:09), and diffusion models "can be cast as a certain kind of VAE" (≈8:37; also lecture 14,
≈3:03–3:48); see [diffusion models](diffusion-models.md). Homework 5's first section implements one on FashionMNIST
([lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md#the-problem-set)).

## Conditional VAEs (lecture 16)

[Lecture 16](16-generative-models-conditional-models.md) makes the VAE conditional to predict an output $\mathbf{y}$ from an input $\mathbf{x}$, because a GAN tends to
collapse onto one mode and conditional VAEs are "a little better at maybe capturing all the modes of the posterior" (≈39:45–40:31).
Start from a VAE on the target $\mathbf{y}$ alone, then "let my reconstruction network, my decoder, observe the data x" (≈42:03–42:50).
The target still reaches the decoder only through the Gaussian bottleneck $\mathbf{z}$, which is "trying to remove information. And so
rather than encoding the color in z, the color can be given by x" (≈42:50). So "z learns to encode the missing information necessary to
predict y from x", in the lecture's example the direction a square moves (slide 45). At test time $\mathbf{z}$ is sampled from the unit
Gaussian, and each sample gives one plausible output (slides 46–49, ≈43:37–44:24). "In unconditional VAEs, z has to encode everything
about the data. In conditional VAEs, z only encodes information that's not given in the conditioning variable, x" (≈45:10).

Conditioned on text, this is "roughly the way that the DALL-E 1 model works" (slides 63 and 64, ≈59:21–1:02:29): the prompt does not say
which way the scarlet macaw faces, "so therefore, the latent variable has to learn to encode that the macaw is facing to the left". Of the
four families, the VAE "was the only one that was a little bit tricky" to make conditional (≈1:04:47). VQGAN is "a VAE with a patch GAN
discriminator" (≈1:07:53). See [conditional generative models](conditional-generative-models.md).

## Seams, again (lecture 19)

Lecture 19 brings back lecture 15's coloured two-dimensional example to explain where to interpolate in a generator (slide 18, "the
lecture from VAEs", ≈20:20). Training maps the data onto the latent space as if "you've taken something and tried to crumple it into a
ball", leaving "danger zones" and seams where "things that are nearby in latent space could potentially be quite far away from each
other in data space" (≈21:07–21:55); this is why StyleGAN interpolates in an intermediate, less distorted space. See [generative models
as data](generative-models-as-data.md).

## See also

- [Autoencoders](autoencoders.md) — compression and reconstruction, without the Gaussian.
- [Generative models](generative-models.md) — latent variables, maximum likelihood, and the families compared.
- [Representation learning](representation-learning.md) — disentanglement, and generation as its inverse.
- [Diffusion models](diffusion-models.md) and [generative adversarial networks](generative-adversarial-networks.md).
