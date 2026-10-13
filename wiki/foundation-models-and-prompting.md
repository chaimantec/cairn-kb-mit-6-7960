# Foundation models and prompting

A **foundation model** is a model so broadly useful that it serves as the starting point for many downstream tasks, often
with no further training at all. **Prompting** is the way such a model is adapted without changing its weights: by choosing,
or learning, what to put in front of it. Lecture 18 presents the pair as the newest form of transfer learning, "a new paradigm"
of "'Deep Learning' with *No* Data" in which you "just ask", and adds that "of course, this isn't really machine *learning*
anymore" (slide 47). Covered so far: [lecture 18](18-transfer-learning-models.md), slides 40–75, ≈55:48–1:24:36; prompting a
language model as conditioning, in [lecture 16](16-generative-models-conditional-models.md), ≈47:33; and optimizing a GAN's input
through CLIP, in [lecture 2](02-how-to-train-a-neural-net.md), slide 70, ≈1:06:05–1:06:52. [Lecture 19](19-transfer-learning-data.md) (slide 50, ≈1:12:25–1:14:43) calls in-context learning a form of meta-learning.
Language models are lecture 21's subject, which this knowledge base does not yet cover (see the [course map](course-map.md)).

**Notation** (lecture 18's slides). In CLIP, an image encoder produces image embeddings $I_1, \ldots, I_N$ and a text encoder text
embeddings $T_1, \ldots, T_N$; $I_i \cdot T_j$ is the dot product of an image's embedding with a text's. In the banana and CLIP+GAN
examples, $e_1$, $e_2$, $e_3$ are embeddings and $z$ is the input, or latent vector, of an image generator.

## What makes a model a foundation

"Foundation models" is "a term that was coined by the computer science department of Stanford" (slide 40 cites Bommasani et al.
2021) for "the idea that you have some models that are so generally useful that they end up being really effective for a bunch of
different downstream tasks" (lecture 18, ≈55:48–56:35). The lecturer sorts them loosely into buckets. Some are representation
learners, where the question is whether they need finetuning at all; and "as foundation models become more and more general, you
need to fine-tune them less and less", because the task is more likely to fall inside their broad training distribution. A model
that needs no finetuning "becomes more accessible": ChatGPT, CLIP and image generators are usable "off the shelf", and "their use
is directly related to what percent of the things people want to do, this model can do off the shelf without any … specialized
training" (≈56:35–58:08). Slide 40's image for it is Newton's "If I have seen further it is by standing on the shoulders of Giants":
"the bigger the foundation, maybe the further up we get as a starting point" (≈58:08). [Lecture 9](09-hackers-guide-to-deep-learning.md)
gave the same advice in practical form: "stand on the shoulders of giants. use pretrained models (but be aware of their flaws and
limitations)" (slide 28).

## Learner and adaptor

Lecture 18 draws the paradigm in two halves (slide 46). A **learner** turns "Tons of data" into foundation models: the slide names
BERT, SimCLR, CLIP, GPT, AlexNet, DALL-E, WaveNet, BigGAN, StyleGAN and VQGAN. An **adaptor** turns a foundation model, with "Little
or no data", into an app. The adaptor can be an AI agent or as simple as pulling in CLIP, and "a lot of the most recent set of startups
in Silicon Valley have been some version of this" (≈1:09:04). For GPT-3 (Brown et al., 2020) the learner is generative pretraining,
predicting the next characters ("Colorless green ideas sleep ____" → "furiously", slide 48), and the adaptor is prompt engineering
("[Review] + 'The sentiment in this review is ____'" → "positive", slide 49). For CLIP the learner is contrastive pretraining (slide 60),
and there are two adaptors, below.

## The cost of the foundation

Lecture 18 does not let the paradigm pass as free. Training compute grew "300,000x" from AlexNet to AlphaGo Zero (slide 41), and
industry, not academia, is where the parameter counts are rising, because "they're often the only people who can afford it" (slide 42,
≈59:40–1:00:29). The models carry climate and water costs: a study of BLOOM's carbon footprint shows a smaller model can emit far more
if trained on a grid without renewable power (slide 43, ≈1:01:16–1:02:02), and the lecturer recalls estimates of the water used to cool
data centers (≈1:02:48–1:03:35). Slide 44 counts the cost of a few models ("Dalle-2: 650m training examples"; "GPT-3: ~$4.6m to train
*once*"; "PaLM: 67 authors") and offers the counter-argument that "these numbers are small if we think of the result as foundational
infrastructure", like the interstate highway system. Slide 45 is the argument for scale: on 58 tasks, small models are "Not working at
all" and only the largest PaLM models are "Working as well as an average human" (≈1:08:17). The lecturer's conclusion: "Now, let's do
research into efficiency to get these back down" (≈1:09:04). See [scaling laws](scaling-laws.md).

## Prompting a language model

"New capabilities by just asking": a list of top dishes at a restaurant, a list of prime numbers continued (slides 50 and 51). Whether
the model is "learning complex patterns and learning to mathematically reproduce them, or … pattern matching", "essentially just a
kernel machine" over its training data, "depending on who you talk to" (≈1:10:41).

**In-context learning** (slide 52, from the GPT-3 paper) has two loops. The outer loop is "Learning via SGD during unsupervised
pre-training"; in the inner loop, each sequence teaches its own task from the examples in front of the model, sums, unscrambled words or
translations: "here's some examples of what I'd like you to do, and then ask it to continue to do them". It "starts to feel flavored,
like meta-learning" (≈1:11:28). It extrapolates a count (slide 53) and translates from three examples (slide 54). See [autoregressive
models](autoregressive-models.md).

**Prompts are brittle.** "Hand-crafting good prompts is tricky…" (slide 56): on SST-2 sentiment, the template "It was [MASK]." scores
92.7 with the label words "great/terrible" and 83.2 with "terrible/great"; on SNLI, rearranging a template drops accuracy from 77.2 to
62.9. "It was a little bit like working with a genie in a lamp. You're like, if you asked the genie for no pain on Earth, it might kill
everybody" (≈1:13:02). Ensembling many prompts helps and is expensive.

**Soft prompts** (slide 57, prompt tuning and prefix-tuning) go "back to the normal learning framework": optimize "a small continuous
task-specific vector while having language model parameters frozen", learned tokens prepended to the input that "might be somewhat
gibberish, but they're really good at finding the place in the distribution" (≈1:13:02–1:13:50).

**Chain-of-thought prompting** (slides 58 and 59, Wei, Wang, Schuurmans et al. 2022) shows the model a worked answer before the question.
On GSM8K math word problems PaLM 540B goes from 18% with standard prompting to 57%, above finetuned GPT-3 175B (33%) and the prior best
(55%). Telling a model "you're a scientist" works for the same reason, the lecturer suggests: it is "finding the right part of the
distribution", text "maybe more likely to be correct" (≈1:14:35–1:15:20).

## Prompting CLIP

CLIP (Radford et al., 2021) learns an image encoder and a text encoder contrastively, so that each image's embedding $I_i$ is most similar
to its own caption's $T_i$ among the $N$ in a batch (slide 60; see [contrastive learning](contrastive-learning.md)). As a representation
learner its adaptor is a linear classifier on the image embeddings, which beats BiT-M, SimCLRv2 and ResNet50 at every number of labelled
examples on slide 61's chart. As a foundation model its adaptor is "Just ask" (slide 62): embed "A photo of a {object}." for each class
name and pick the class whose text embedding is most similar to the image's. Zero-shot CLIP, on slide 61, matches the linear probe trained
with four labelled examples per class.

It is brittle in the same way: "a photo of a dog" against "photo of dog" can "massively change the performance" (≈1:16:08–1:16:54). It can
also do new things: "A sketch of a banana" and "A photo of a banana" embedded as $e_1$ and $e_2$, a drawing of bananas as $e_3$, and the
larger of $e_1 \cdot e_3$ and $e_2 \cdot e_3$ says the drawing is a sketch (slide 63). And it has limits: CLIP models "are very good at
simple concepts, but they're really bad at composition … they're good at ors, and they're bad at ands", like a bag of words, and bad at
negation, so "not a photo of a bald eagle" finds bald eagles (≈1:17:40–1:18:27).

## Prompting images

The same idea applies to the input image (slide 64). **Visual prompting** (Bahng, Jahanian, Sankaranarayanan and Isola, 2022) learns a
border of pixels that, added to every image, makes a frozen CLIP classify better: "learning beneficial noise", the counterpart of the
[adversarial noise](adversarial-examples.md) of lecture 17 (slide 65, ≈1:18:27–1:19:59). With a vision model and a hard-coded mapping from
its classes to new ones it is **adversarial reprogramming** (slide 66). A single learned pink pixel, at the same place in every satellite
image, gives "+3% acc boost" (slide 67), which the lecturer reads as showing "how fragile and complex the statistical representations
are" (≈1:21:30). **Visual prompt tuning** learns prompt tokens for a frozen vision transformer (slide 69), and **visual prompting via image
inpainting** gets segmentation, edge detection, colorization and more from an inpainting model by asking it to complete a grid of example
and query images (slide 70). Because prompting only touches the input, it needs no access to the weights or the logits (≈1:22:15).

Slide 68 places prompting beside the other adaptations: a pretrained model can be adapted on its **inputs** (prompting), its **mapping**
(finetuning) or its **outputs** (a linear probe on the frozen model). See [transfer learning](transfer-learning.md).

## Plugging foundation models together

Foundation models are also building blocks (slide 72). Text-to-image can be had by linking VQGAN and CLIP (slide 73). In CLIP+GAN
(slide 74; the same construction as lecture 2's slide 70) the generator's input $z$ is optimized so that the image it generates,
embedded by CLIP's image encoder as $e_2$, maximizes $e_1 \cdot e_2$ with the embedding $e_1$ of a text prompt; every network stays fixed
(≈1:23:48–1:24:36). DALL-E 2 (slide 75, Ramesh et al. 2022) is built from the same parts: a CLIP objective between text and image
embeddings, and "this prior that helps them do decoding slightly more efficiently" (≈1:24:36). See [conditional generative
models](conditional-generative-models.md).

## In-context learning as meta-learning (lecture 19)

Lecture 19 returns to in-context learning from the side of meta-learning. Learning maps a sequence $x_0, y_0, x_1, y_1, \ldots$ to a
function from $x$ to $y$; a recurrent network or an autoregressive model reading such a sequence does this in its state ("RNN state +
params is f, apply that f to x3", slide 50), like the English-to-Spanish prompt of lecture 18 (≈1:12:25–1:13:12). "This really is, I
think, what GPT and a lot of these modern autoregressive models are doing. It's kind of a version of meta-learning at large scale"
(≈1:13:58). See [meta-learning](meta-learning.md).

## See also

- [Lecture 18](18-transfer-learning-models.md) — the lecture, with its figures.
- [Transfer learning](transfer-learning.md) — finetuning, linear probes and the other ways to reuse a model.
- [Contrastive learning](contrastive-learning.md) — how CLIP is trained.
- [Autoregressive models](autoregressive-models.md) — the next-word predictors behind GPT-3.
- [Adversarial examples](adversarial-examples.md) — the harmful noise that visual prompting turns around.
- [Scaling laws](scaling-laws.md) — compute, parameters and performance.
