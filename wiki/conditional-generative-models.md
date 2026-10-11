# Conditional generative models and structured prediction

A **conditional generative model** predicts one high-dimensional thing from another, an image from a label map, a caption from
an image, an image from a caption, by modelling the whole distribution of possible outputs given the input rather than a single
best guess. [Lecture 16](16-generative-models-conditional-models.md) argues that this is not a niche use of generative models but
the reason they matter: "I'll frame all problems that are predicting high-dimensional structured data or information out of a
network as being appropriate to be treated with generative modeling" (≈1:31). Covered so far: lecture 16, slides 2–83,
≈0:00–1:20:56. The families being conditioned were introduced in [lecture 14](14-generative-models-basics.md) and, for the VAE,
[lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md); see [generative models](generative-models.md).

**Notation**, from the slides: an input or conditioning variable $\mathbf{x}$, an output $\mathbf{y}$, paired training data
$p_{\texttt{data}}(\mathbf{x}, \mathbf{y})$, and a latent variable $\mathbf{z}$ sampled from the unit Gaussian
$\mathcal{N}(\vec{0}, 1)$. See also the course's [notation](notation.md).

## Structured prediction

Predicting "something that's not just a single number or a single decision, but is actually some high-dimensional structured
object" is called **structured prediction** (lecture 16, ≈2:16). Slide 3's examples are semantic segmentation, speech
recognition, text-to-photo and protein folding. Slide 4 defines it: model the *joint* distribution of the high-dimensional output
given the input, $P(\mathbf{X} \vert \mathbf{Y} = \mathbf{y})$, where an **unstructured** model would treat the output's
dimensions as independent and model $\prod_i p(X_i \vert \mathbf{Y} = \mathbf{y})$ (the slide writes the output as $\mathbf{X}$
and the condition as $\mathbf{Y}$). Independence almost never holds, which is why the lecturer calls the term "a weird
historical artifact … This is just everything" (≈2:16–4:34).

## What goes wrong without it

Lecture 16's example is colourization (slides 5 and 6, ≈5:19–13:04), with a shirt that is teal or pink with equal probability.

- **Regression** predicts one colour, and least squares is minimized by the mean of the data, which here "has zero density under
  the true data distribution" (≈7:41).
- **A Gaussian** models uncertainty but is unimodal, and the truth is bimodal (≈9:13).
- **Classification** over quantized colours, a softmax, captures both modes, the trick of [lecture 9](09-hackers-guide-to-deep-learning.md)
  (≈9:13–10:44; see [softmax and cross-entropy](softmax-and-cross-entropy.md)). But applied to every pixel independently it
  flips between the modes wherever the near-even probabilities cross, so one shirt comes out in two colours: it fails "to capture
  the joint dependency between this prediction and that prediction" (≈12:17).

The visible symptoms are blur, desaturation and colour that oscillates, "the effect of averaging that you get when you're fitting
a distribution that doesn't have the capacity to approximate the true distribution" (≈17:46). A squared-error loss on a whole
output image is "implicitly treating all of the output dimensions as independent", since squared error is a Gaussian log
likelihood and a sum of logs is a product of probabilities (≈20:08–21:00); a map turned into an aerial view this way paints a park
"an average green" (slide 14, ≈21:46).

Generative models fix both failures. "Generative models have two important properties for structured prediction: 1. They can
model a multimodal distribution 2. They can model joint dependences between multidimensional predictions" (slide 6). "When we
talk about generative AI, generative models, this is what we mean" (≈13:04).

## The objective, not the hypothesis space

Deep learning's story around 2012 was a higher-capacity **hypothesis space** (slides 9 and 10, Andrew Ng's chart of performance
against data). Lecture 16's is about the **objective**: "Use an objective that can model structure! (e.g., a graphical model, a
GAN, etc)" (slide 15). With an unstructured objective, "deep learning … actually does saturate … there'll be no incentive to model
the joint dependencies", however large the network (≈35:52–36:38; the cartoon of slide 36). In a conditional GAN the
discriminator *is* such an objective: "from g's perspective, d is like a loss function … it's learned as opposed to being
hand-defined. So it's a loss function that can look at structure in the output" (≈28:06).

The price is **hallucination**. A conditional GAN fills a park with trees that are "just a sample from the true distribution over
where the trees might be", and may drop a road that an L1 prediction, which averages and is "very conservative", keeps (≈37:27–39:45).

## Making each family conditional

"Every generative model that we've talked about can be made conditional. Oftentimes, it's quite easy" (≈39:45). The lecture's
summary (≈1:04:47) and its examples:

| Family | How it is conditioned | Lecture 16's example |
| --- | --- | --- |
| [GAN](generative-adversarial-networks.md) | the generator maps $\mathbf{x}$ instead of noise to $\mathbf{y}$, and the discriminator judges the pair $(\mathbf{x}, \mathbf{y})$: "just let the discriminator take the text as the input" | pix2pix, label maps to facades and maps to aerial photographs (slides 18–38) |
| [VAE](variational-autoencoders.md) | the decoder sees $\mathbf{x}$ as well as $\mathbf{z}$, so $\mathbf{z}$ encodes only what $\mathbf{x}$ leaves out; "the only one that was a little bit tricky" | the moving square, BicycleGAN's facades, DALL-E 1 (slides 42–49, 63–64) |
| [Autoregressive model](autoregressive-models.md) | already conditional: the input is the prefix; "it already is a conditional model" | prompting a language model, GPT-3's sentiment (slides 52–54); captioning (slide 56) |
| [Diffusion model](diffusion-models.md) | the denoiser sees the text with the noisy image: "just let the denoiser take text as input" | DALL-E 2 and Stable Diffusion (slides 65–67) |

The conditional VAE carries the lecture's general point about latents: "In unconditional VAEs, z has to encode everything about
the data. In conditional VAEs, z only encodes information that's not given in the conditioning variable, x" (≈45:10). Hence
"Concept: You can control your data either via explicit inputs or latent variables" (slide 50), and with long text prompts "the
latent variables might play a very small role" (≈45:58).

Real text-to-image systems combine the families: "people call them diffusion models. But they have all three components in them"
(≈57:45). VQGAN, "a VAE with a patch GAN discriminator", carries a diffusion or autoregressive model in its latent space (slide 70,
≈1:07:53).

## Joining modalities: tokens and cross-attention

Images and text meet in [transformers](transformers.md), which turn "everything into tokens, and then just do token-to-token
mappings" (≈51:27). A captioning model prefixes a causal language model's tokens with a vision transformer's image tokens and joins
them by **cross-attention**, "taking keys from — and values from one set and queries from a different set" (slide 56, ≈53:47–54:38).
Slide 60's "general strategy for domain translation" encodes from one space into a latent space and decodes into another,
penalizing the prediction error there. Visual question answering and LLaVA condition on an image and a question together (slides 68
and 69). The standard way to combine streams "seems to be concatenation … and then just concatenate and do more cross-attention",
and better ways are "an active area of research" (≈1:01:42).

## Without paired data

Everything above needs pairs. **Unpaired translation** learns a conditional mapping from two unpaired sets, photographs and
Cézanne paintings (slides 71–83, ≈1:11:44–1:20:56). A GAN's loss "checks if output is part of an admissible *set*" rather than
matching "a target *instance*" (slide 75), so it can be trained without pairs, but nothing then ties the output to the input
(slides 76 and 77). **Cycle consistency** does: map $X$ to $Y$ and back, and "I should arrive where I started" (CycleGAN, slides
78–80, ≈1:16:19–1:17:07). Many cycle-consistent mappings exist, but networks trained by gradient descent find simple ones (slide 81,
≈1:17:53–1:19:23; see [inductive bias](inductive-bias.md)). In language the same idea is back translation (slide 82). "Concept:
paired data is not always necessary for grounding (finding a unique mapping between two domains)" (slide 83).

## See also

- [Lecture 16](16-generative-models-conditional-models.md) — the whole argument, with its figures.
- [Generative models](generative-models.md) — the families and what makes a model generative.
- [Generative adversarial networks](generative-adversarial-networks.md), [variational autoencoders](variational-autoencoders.md),
  [diffusion models](diffusion-models.md) and [autoregressive models](autoregressive-models.md) — each family's own page, with
  its conditional form.
