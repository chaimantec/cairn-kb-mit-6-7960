# Generative adversarial networks

A **generative adversarial network** (GAN; Goodfellow et al., 2014) trains a generator against a second network. The
**generator** $g_{\theta}$ maps a vector of random numbers $\mathbf{z}$ to a synthetic data point $g_{\theta}(\mathbf{z})$;
the **discriminator** $d_{\phi}$, with its own parameters $\phi$, is a classifier that tries to tell real data from the
generator's output. The generator is trained to fool it. Covered so far: [lecture 14](14-generative-models-basics.md), slides
54–59, ≈1:17:40–1:20:47, the last three minutes of the lecture; [lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md), slides 39 and 41,
≈1:16:20–1:17:06 and ≈1:19:23–1:20:09 (BigGAN's latent directions, and GANs among the other families); [lecture 16](16-generative-models-conditional-models.md), slides
16–38 and 71–81, ≈22:37–39:45 and ≈1:11:44–1:19:23 (conditional GANs, the patch discriminator, and CycleGAN); [lecture 19](19-transfer-learning-data.md), slides 6–31, ≈7:48–32:50
(BigGAN and StyleGAN latent spaces as controls, labels and explanations). See also
[generative models](generative-models.md).

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

## Latent directions without an encoder (lecture 15)

[Lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md) returns to GANs when it asks what generative models' latent spaces learn. "GANs didn't have any encoder that maps
from data to latents", unlike a [variational autoencoder](variational-autoencoders.md), "but nonetheless, the latent variables, if you sample
random latent variables or sample along a grid of latent variables, they do show that they actually model structure in a similar way to the
VAE" (≈1:16:20). Its example is BigGAN [Brock et al. 2018], with "one direction in latent space that controls the orientation of the bird and
one direction that controls the background color" (slide 39, ≈1:16:20–1:17:06); slide 39's images are excluded from OCW's licence, so the
knowledge base describes them only in the slide file. See [representation learning](representation-learning.md#both-directions-at-once-and-disentanglement-lecture-15).
In the lecture's closing comparison GANs are "the most cute and fun" of the families (≈1:20:09), and slide 41 marks GANs as having latent variables and a generator but no density or energy.

## Conditional GANs (lecture 16)

[Lecture 16](16-generative-models-conditional-models.md) makes the GAN predict: "the noise is replaced with a conditioning variable … there is just a mapping from an x
to a y as opposed to a z to an x. So it's the same exact math as the GAN" (≈24:12). Its example is pix2pix (Isola et al. 2017), which
maps an architectural label map to a photograph of a building. Its slides write the discriminator the other way round from lecture
14's: $d_{\phi}$ outputs the probability that its input is **synthetic** (slide 20, "synthetic (0.9)" and "real (0.1)", as on lecture
14's summary figure), so the discriminator maximizes $\log d_{\phi}(g_{\theta}(\mathbf{x})) + \log(1 - d_{\phi}(\mathbf{y}))$ and the
generator minimizes $\log d_{\phi}(g_{\theta}(\mathbf{x}))$; the two conventions give the same game. "The one little trick to make a
GAN conditional" is that the discriminator must judge the **pair**, "is the pair x, y a real pair sampled from the data-generating
process" (≈28:54), which gives the conditional GAN's objective (slide 30):

$$\arg\min_{\theta} \max_{\phi} \mathbb{E}_ {\mathbf{x},\mathbf{y}} \left[ \log d_{\phi}(\mathbf{x}, g_{\theta}(\mathbf{x})) + \log(1 - d_{\phi}(\mathbf{x}, \mathbf{y})) \right]$$

**The discriminator as a learned loss.** "From g's perspective, d is like a loss function … it's learned as opposed to being
hand-defined. So it's a loss function that can look at structure in the output and not just say, did I get every pixel independently
correct" (slide 23, ≈28:06). Its architecture decides what it penalizes: pix2pix's **patch discriminator** judges every patch of
the output, so a $1 \times 1$ patch is "almost the same thing as least squares regression", a $16 \times 16$ one can see blur, larger
ones "higher and higher order statistics", and the whole image worked worse on a training set of 400 images (slide 33, ≈32:46–35:52).

**Training.** GANs are "not as popular now. And the main reason was they weren't very stable", so pix2pix adds an L1 regression term,
$\lambda \mathcal{L}_ {\texttt{L1}}(\theta)$, that "can very much stabilize that process" (slide 31, ≈31:15–32:00). And a GAN's
generator "tends to exhibit what's called **mode collapse** … it will output one mode of the distribution that looks realistic. But …
it will maybe miss other modes", which is the reason the lecture gives for conditional VAEs (≈39:45–40:31; see
[variational autoencoders](variational-autoencoders.md#conditional-vaes-lecture-16)).

**Unpaired translation and CycleGAN.** "With a GAN, the pairing between the input and the output is only coming through the
discriminator" (≈1:14:02). Drop it, and the loss "checks if output is part of an admissible *set*" rather than whether it "matches a target
*instance*" (slide 75), which suits unpaired data but lets any realistic output pass (slides 76 and 77). CycleGAN (Zhu, Park et al. 2017)
adds a second generator and discriminator for the way back and a **cycle-consistency loss**: going from $X$ to $Y$ and back, "I should
arrive where I started" (slides 78–80, ≈1:16:19–1:17:07). "Here's a really cool thing about GANs and why they're still a neat model,
even though they're less flashy these days" (≈1:14:02). See [conditional generative models](conditional-generative-models.md).

## Latent spaces as data (lecture 19)

Lecture 19 uses GANs' latent spaces for transfer learning. Walking in a straight line in a GAN's latent space "kind of works, but it
can be really cherry picking" (≈16:23–17:11). StyleGAN (Karras, Laine and Aila, 2018) puts a "Mapping network $f$" of eight fully
connected layers between the input latent $z \in \mathcal{Z}$ and an intermediate latent $w \in \mathcal{W}$, which steers each layer of
the "Synthesis network $g$" through AdaIN (slide 15), and found the intermediate space better to walk in: the input space is "almost
like you've taken something and tried to crumple it into a ball", with seams where nearby latent points map far apart, while
$\mathcal{W}$ is "a little bit less distorted" (slides 18–19, ≈21:07–22:40). StyleSpace finds cleanly disentangled directions there,
such as a car's hood style, headlights and body colour (slide 20). BigGAN's interpolation from a bird to a fly stays on the image
manifold but passes through "this half bird, half fly thing" (slide 17, ≈19:34). DatasetGAN labels a StyleGAN's images from a few
annotations (slides 24–26), and StylEx explains a classifier through StyleSpace directions (slides 28–31). See [generative models as
data](generative-models-as-data.md).

## Two things printed on the slides

Slide 54's second line reads "g tries to identify the fakes"; the lecturer calls it a typo for d (≈1:18:26), and slide 58
prints "d tries to identify the fakes". And slide 59's summary figure labels the discriminator's outputs "synthetic (0.9)"
and "real (0.1)", the reverse of slide 55's "fake (0.1)" and "real (0.9)"; the lecture ends before discussing it. Every GAN
slide carries an OCW notice for its flamingo images, so the knowledge base reproduces none of them; the slide file describes
them.
