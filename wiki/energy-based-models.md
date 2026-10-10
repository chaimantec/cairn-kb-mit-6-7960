# Energy-based models

An **energy-based model** scores data with an **energy** $E_{\theta} : \mathcal{X} \to \mathbb{R}$, a function from a data
point to any real number, with low energy meaning likely data. It is an "unnormalized probability model" (lecture 14, slide
28): unlike a density, an energy need not integrate to anything, which frees the network from having to output a normalized
distribution. It is one of the indirect ways to represent a data-generating process in the course's taxonomy of
[generative models](generative-models.md). Covered so far: [lecture 14](14-generative-models-basics.md), slides 28–33,
≈44:16–58:11.

## From energy to probability

Any energy implies a normalized density through the Boltzmann form (lecture 14, slide 28, ≈45:02):

$$p_{\theta}(\mathbf{x}) = \frac{e^{-E_{\theta}(\mathbf{x})}}{Z(\theta)}, \qquad Z(\theta) = \int_{\mathbf{x}} e^{-E_{\theta}(\mathbf{x})} \thinspace d\mathbf{x}$$

where $\theta$ are the network's parameters and $Z(\theta)$ is the normalizing constant, "sometimes called the partition
function" (≈55:03). The minus sign is "just a definition", so that low energy means high probability (≈48:53). The logits
of a softmax classifier are an energy, and softmax is exactly this exponentiate-and-normalize step over a finite set of
classes (≈45:48; see [softmax and cross-entropy](softmax-and-cross-entropy.md)).

## Why energies are often enough

Computing $Z(\theta)$ means integrating over every possible data point, which is intractable for a network that outputs an
arbitrary number per data point; that is why restricting a model to normalized densities is "very restrictive" (≈46:33). But
the normalizer cancels in any ratio of probabilities (slide 28):

$$\frac{p_{\theta}(\mathbf{x}_ 1)}{p_{\theta}(\mathbf{x}_ 2)} = \frac{e^{-E_{\theta}(\mathbf{x}_ 1)}}{e^{-E_{\theta}(\mathbf{x}_ 2)}}$$

"Relative probabilities are often all you need (e.g., for sampling)": for a decision such as whether snow or a hurricane is
likelier tomorrow, and for samplers such as Markov chain Monte Carlo, which "only requires relative probabilities"
(≈47:20–48:53). Normalized probabilities still have the advantage of a common scale: "If you go to the hospital and they tell
you your energy of having cancer is 10 million, what are you going to do with that, right?" (≈49:39–50:26).

## Training by contrastive divergence

Lowering the energy at the training data is not enough on its own, since nothing stops the model from lowering it everywhere
(≈51:12). The fix is to raise the energy where the model currently puts its samples, drawn for instance by Markov chain Monte
Carlo, which needs no $Z(\theta)$. Data push the energy down (green arrows on the slides), model samples push it up (red), and
"at convergence, green (data) and red (model) samples are identical and model update (green-red) cancels out" (slide 29,
≈51:57–52:44). This is **contrastive divergence**, "the contrast between where the data lives and where the model has placed
high probability"; it is "even connected to contrastive learning, but a little bit indirectly" (≈52:44–53:32; see
[contrastive learning](contrastive-learning.md)).

The gradient follows from maximum likelihood (slides 30–32). Writing $p_{\texttt{data}}$ for the data distribution, the
gradient of the expected log-likelihood splits into a **positive term**, from the data, and a **negative term**, from the
normalizer:

$$\nabla_{\theta} \mathbb{E}_ {\mathbf{x} \sim p_{\texttt{data}}} \left[ \log p_{\theta}(\mathbf{x}) \right] = -\mathbb{E}_ {\mathbf{x} \sim p_{\texttt{data}}} \left[ \nabla_{\theta} E_{\theta}(\mathbf{x}) \right] - \nabla_{\theta} \log Z(\theta)$$

Differentiating under the integral turns the negative term into an expectation under the model itself (slide 31):

$$-\nabla_{\theta} \log Z(\theta) = -\int_{\mathbf{x}} \frac{e^{-E_{\theta}(\mathbf{x})}}{Z(\theta)} \nabla_{\theta} E_{\theta}(\mathbf{x}) \thinspace d\mathbf{x} = -\mathbb{E}_ {\mathbf{x} \sim p_{\theta}} \left[ \nabla_{\theta} E_{\theta}(\mathbf{x}) \right]$$

so that both terms can be estimated from samples, training examples $\mathbf{x}^{(i)} \sim p_{\texttt{data}}$ and model
samples $\hat{\mathbf{x}}^{(i)} \sim p_{\theta}$ (slide 32):

$$\nabla_{\theta} \mathbb{E}_ {\mathbf{x} \sim p_{\texttt{data}}} \left[ \log p_{\theta}(\mathbf{x}) \right] \approx -\frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} E_{\theta}(\mathbf{x}^{(i)}) + \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} E_{\theta}(\hat{\mathbf{x}}^{(i)})$$

"Push down energy where the data lives. Push up energy where the model currently places high probability. Eventually, those
cancel out, and the model has fit the data" (≈58:11). The [lecture page](14-generative-models-basics.md#fitting-an-energy-contrastive-divergence)
gives every step of the derivation and notes two lines of slide 31 that are printed loosely. Like any maximum-likelihood
model, an energy-based model trained to convergence overfits, to a delta function on each training point, without early
stopping or regularization (≈52:44).
