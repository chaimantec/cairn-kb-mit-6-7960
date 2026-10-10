# Generative adversarial networks

A **generative adversarial network** (GAN; Goodfellow et al., 2014) trains a generator against a second network. The
**generator** $g_{\theta}$ maps a vector of random numbers $\mathbf{z}$ to a synthetic data point $g_{\theta}(\mathbf{z})$;
the **discriminator** $d_{\phi}$, with its own parameters $\phi$, is a classifier that tries to tell real data from the
generator's output. The generator is trained to fool it. Covered so far: [lecture 14](14-generative-models-basics.md), slides
54–59, ≈1:17:40–1:20:47, the last three minutes of the lecture. See also [generative models](generative-models.md).

## Direct or indirect?

A GAN's generator maps latent variables straight to samples, so the lecturer "would have called it direct", in the course's
split between learning a sampler and learning a scoring function; "but with d, it becomes a little more confusing. So maybe
either answer could have an interpretation which is correct" (lecture 14, ≈1:18:26).

## The objectives

Why should it work? "If g is able to make output data points which are indistinguishable from your training data points, then
it must be that you're outputting things that are from the same distribution. You can prove that, in fact" (≈1:18:26–1:19:13).
The discriminator is "just a binary softmax classifier" (see [softmax and cross-entropy](softmax-and-cross-entropy.md)) whose
output is the probability that its input is real. Writing $p_{\texttt{data}}$ for the data distribution and $p_{\mathbf{z}}$
for the distribution of the latent vector, it maximizes the log probability it gives real data and the log probability that
the generator's samples are fake (slide 55):

$$d^{\ast}_ {\phi} = \underset{\phi}{\arg\max} \thinspace \mathbb{E}_ {\mathbf{x} \sim p_{\texttt{data}}} \left[ \log d_{\phi}(\mathbf{x}) \right] + \mathbb{E}_ {\mathbf{z} \sim p_{\mathbf{z}}} \left[ \log (1 - d_{\phi}(g_{\theta}(\mathbf{z}))) \right]$$

The generator minimizes "the exact same objective that d was maximizing", through the term that depends on it (slide 56,
≈1:20:01):

$$\underset{\theta}{\arg\min} \thinspace \mathbb{E}_ {\mathbf{z} \sim p_{\mathbf{z}}} \left[ \log (1 - d^{\ast}_ {\phi}(g_{\theta}(\mathbf{z}))) \right]$$

Since it must fool the best discriminator, the two together are a min-max game (slide 57):

$$\arg \min_{\theta} \max_{\phi} \thinspace \mathbb{E}_ {\mathbf{x} \sim p_{\texttt{data}}} \left[ \log d_{\phi}(\mathbf{x}) \right] + \mathbb{E}_ {\mathbf{z} \sim p_{\mathbf{z}}} \left[ \log (1 - d_{\phi}(g_{\theta}(\mathbf{z}))) \right]$$

In practice training alternates between the discriminator and the generator, each by backprop, and the global optimum is
reached "when g reproduces data distribution" (slide 58).

## A student and a teacher

"This is a min-max game. It's an adversarial game. But think of it more like a student and a teacher. d is the teacher, and g
is the student. The teacher is saying, hey, you need to make a realistic painting, and you didn't get it quite right. The
shadows are wrong. So g will update its parameters to make realistic shadows. And then d will say, oh, that looks
indistinguishable from what I consider realistic art" (lecture 14, ≈1:20:01–1:20:47).

## Two things printed on the slides

Slide 54's second line reads "g tries to identify the fakes"; the lecturer calls it a typo for d (≈1:18:26), and slide 58
prints "d tries to identify the fakes". And slide 59's summary figure labels the discriminator's outputs "synthetic (0.9)"
and "real (0.1)", the reverse of slide 55's "fake (0.1)" and "real (0.9)"; the lecture ends before discussing it. Every GAN
slide carries an OCW notice for its flamingo images, so the knowledge base reproduces none of them; the slide file describes
them.
