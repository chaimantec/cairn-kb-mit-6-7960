# Transfer learning

Transfer learning reuses what a network learned on one task to learn another faster, from less data. Lecture 11
gives it as the answer to "why learn representations?": "Maybe the simplest is to do more learning, right? So we learn
to learn … the main use of trained representations is to accelerate future learning"
([lecture 11](11-representation-learning-reconstruction-based.md), ≈27:54–28:41). The course has two lectures on it,
**18 (models)** and **19 (data)**, both covered below (see the [course map](course-map.md)). Covered so far: [lecture 1](01-introduction.md), slides 76–77, ≈57:19–58:53 (reusing
weights); [lecture 2](02-how-to-train-a-neural-net.md), ≈56:42 and ≈1:10:48–1:12:21 (pretrained parameters inside a
larger network); [lecture 9](09-hackers-guide-to-deep-learning.md), slides 27–28, ≈56:26–58:43 (start from a pretrained
model); lecture 11, slides 23–29, ≈27:54–39:32; [lecture 12](12-representation-learning-similarity-based.md), slides 3, 30 and 60–68, ≈2:20–3:05 and ≈1:03:23–1:10:21
(transfer as a reason to learn representations, self-supervised against supervised pretraining, and linear probes on
iNaturalist); [lecture 17](17-generalization-out-of-distribution.md), slides 38 and 57, ≈42:48 and ≈56:06–56:55 (adversarially trained models transfer better, and domain
adaptation previewed); [lecture 18](18-transfer-learning-models.md), slides 2–39 and 46–68, ≈0:00–56:35 and ≈1:09:04–1:23:48
(the three kinds of knowledge to transfer, finetuning in practice, domain adaptation, distillation, and adapting foundation models);
[lecture 19](19-transfer-learning-data.md), slides 2–51, ≈0:00–1:14:43 (knowledge about the inputs, through generative models as
"data++", and meta-learning); [lecture 21](21-language-models.md), slides 8–11 and 41–42, ≈10:03–13:06 and ≈35:50–37:26 (from
pretrain-and-transfer to pretrain-and-prompt).

## A good representation makes the next task easier

Lecture 11's slide 24 is titled "To do more learning! (aka **Transfer learning**)", and quotes the *Deep Learning* book
(Goodfellow et al. 2016): "Generally speaking, a good representation is one that makes a subsequent learning task
easier."

Lecture 1 gives the intuition. If the lower layers of a network learn general building blocks, lines and orientations
and structures, they stay useful when the task changes: a network that categorized photos of animals can keep them when
you turn to satellite images of damaged buildings (slide 77, ≈57:19–58:53). Reuse "is really valuable if you don't have
big data or big compute", because "you can pre-generate representations that then you can learn on top of without needing
to learn everything from scratch" (≈58:06).

## Not blank slates

Lecture 11 calls it "a strange misconception in the early days of deep learning" that deep nets would come to each new
problem as blank slates, which is where the complaint that they are data hungry, while "humans are much more sample
efficient", came from (≈28:41). "But it wasn't apples to apples": a student arrives after "billions of years of
evolution" and "20 plus years of pre-training in your lifetime". "And deep nets also should generally not be used as blank
slates. This is something that's changed over the last decade … We start with pretrained representations" (≈29:28).

The lecturer turns the usual story around: "A lot of people think of deep learning as the thing you do when you have a ton
of data. But I think the real point of deep learning is it's the thing you do that enables learning from little data …
And the way you learn from little data is by pre-training on massive data" (≈34:10–34:56). Nobody quite expected that:
people had hoped for "some algorithm that's just better at learning. You don't have to pre-train that algorithm, but this
works better" (≈34:56).

## Pretrain, adapt, test

Lecture 11's example is a network trained to recognize musical genre, an encoder $f$ to a representation $\mathbf{z}$
followed by a readout $\mathbf{W}$ (slide 25). Any layer can be taken as the representation, with the layers before it as
the encoder and those after as the readout (≈30:14). The company later needs to predict whether users like the music:
"Often, what we will be 'tested' on is not what we were trained on". There are two ways to adapt.

- **Linear adaptation** (slide 26): "freeze f, train a new linear map to new target data". A linear readout trained on a
  frozen representation is a **linear probe**; the readout "could be MLP, or something else" (≈31:01).
- **Finetuning** (slide 27): "initialize f’ as f, then continue training on new target data", backpropagating into the
  encoder "to find some fine-tuned perturbation of your parameters that will solve the new problem" (≈31:48–32:36).

Slide 29 states the recipe: pretrain on task A, giving parameters $\mathbf{W}$ and $\mathbf{b}$; initialize a second network
with some or all of them; train it on task B, giving $\mathbf{W}'$ and $\mathbf{b}'$. It is called fine-tuning "because we
assume the W prime is just a minor modification of W", and which weights to copy is open to "all kinds of interesting network
surgery" (≈33:22–34:10).

The resulting paradigm, "what is typically done for real-world problems", has three phases instead of two (slide 28):
**pretraining** on "A lot of data", **adapting** on "A little data", and **testing** (≈32:36–33:22). The phases now also go
by "pre-training and post-training" (≈31:48). Pretraining can afford a lot of data because it "doesn't require as much knowledge
about what the final task is going to be"; methods that learn without labels, such as [autoencoders](autoencoders.md) and
contrastive learning, are built for it (≈33:22). See [self-supervised learning](self-supervised-learning.md).

## How much data, and from where

The ratio between the phases "is always becoming more extreme": a language model pretrained on "10 trillion tokens", fine-tuned
on "a million or less" in "a few minutes or hours" on Colab, "so the ratio is many orders of magnitude" (lecture 11,
≈35:42–36:27). Linear probes and low-rank fine-tuning methods "will come in a future lecture".

Pretraining need not be on the same kind of data. Asked about a specialized scientific domain with no pretrained model of its
own, the lecturer answers that a language model pretrained on internet text "often actually works decently well", that a very
large gap "might not work as well", and that "if you fine-tune a language model trained on the internet on almost anything, it
will help … it will be better than initializing the network from scratch" (≈37:14–38:00).

## When does it work?

"The big question is, when and why does training on task A on data A help on task B and data B? What is the property of A and
B that has to be alike for this to actually work? So empirically, it just generally works pretty well, more than people maybe
thought. Theoretically, I would say it's very much an open question" (lecture 11, ≈38:45). The lecturer points to Sanjeev
Arora's papers and talks, which phrase the question this way, "but I would say it's in its early days". Lecture 1 frames it
the same way, as the open question of "what do we really need to learn from scratch versus what is generally useful"
(≈57:19–58:53).

## The benchmark decides what transfers (lecture 12)

Lecture 12 lists "To do more learning (transfer learning)" among the reasons to learn representations: a representation "that
then is more easily finetuned ... maybe requiring less training data in that finetuning step" (slide 3, ≈2:20–3:05). Its case
study measures a self-supervised representation through a **linear probe**, "taking a representation space and then just learning
a linear projection that's going to actually be our categorizer", trained with labels (≈1:03:23). On iNaturalist 2021, SimCLR and
MoCo probes nearly match a supervised model on coarse levels of the taxonomy but trail it by about 30 points on species, where on
ImageNet the gap is about 7 (slides 60–66). "The benchmarks that we use really influence our idea of what's good enough": on
ImageNet's coarse categories, self-supervised representations look "very, very close to as good as supervised representations",
but that is "contextualized on the task of interest" (≈1:09:36–1:10:21). See [contrastive learning](contrastive-learning.md).

## In practice

Lecture 9's advice is to "stand on the shoulders of giants. use pretrained models (but be aware of their flaws and
limitations)": "You generally don't want to be training your systems from scratch" (slide 28, ≈57:57–58:43). Lecture 2 places
pretrained weights inside the [differentiable programming](differentiable-programming.md) picture: a part of a network that
looks hand-specified "might have just actually previously been programmed by backprop", and deciding what to adapt "is just a
matter of defining … where you want to freeze your gradients" (≈1:10:48–1:12:21). Lecture 11: "If you're ever going to play with
deep nets, you'll download a pre-trained system, and you'll fine-tune it" (≈33:22).

## Robust models transfer better; domain adaptation (lecture 17)

Lecture 17 reports that adversarially trained ImageNet models transfer better to other datasets than standard ones (slide
38, Salman et al. 2020, "Do adversarially robust ImageNet models transfer better?"; ≈42:48), and it previews **domain
adaptation**, which learns from a labelled source domain and an unlabelled or sparsely labelled target domain "how to
align their representations during training so that you can enforce generalization or adaptation to a new target domain"
(slide 57, ≈56:06–56:55). Domain adaptation is promised for lecture 18 (slide 58), which teaches it; see [domain adaptation](domain-adaptation.md). The lecture's closing answer separates
being robust to any shift from adapting to a known one, for which fine-tuning and domain adaptation are "much more
efficient" (≈1:02:25–1:03:59). See [out-of-distribution generalization](out-of-distribution-generalization.md).

## What can be transferred (lecture 18)

Lecture 18 opens the course's two lectures on transfer learning by asking what exactly is transferred. A model maps inputs to
outputs, and the lecture sorts the answers into "1. Knowledge about the mapping", "2. Knowledge about the outputs" and "3.
Knowledge about the inputs" (slide 8); it covers the first two and leaves the inputs to lecture 19 (≈1:31). Its thesis is lecture
11's: "One view: the **point** of deep learning is to enable problem solving with little data" (slide 7). Humans learn from few
examples because they bring prior knowledge, and the simplest way to give a network prior knowledge is to "'pretrain' them on prior
tasks", which the lecturer calls sending "your neural network to school" (≈7:48). Few examples are also ambiguous: one Segway does not
say whether the concept is Segways, two-wheelers or wheeled things (≈4:41–5:28).

**Knowledge about the mapping** is transferred by finetuning. Slides 10 and 11 repeat lecture 11's music example and recipe: "The
'learned representation' is the *encoder* (its weights and biases) so that's what we transfer". The final layer is usually replaced,
being "pretty specific to the pretraining task" (≈13:57), and after finetuning the early layers of image or audio models often barely
change, "which does seem to argue that some of that is just universal in some sense" (≈15:30). When the new task's inputs or outputs
have other dimensions, $\mathcal{X}' \neq \mathcal{X}$ and $\mathcal{Y}' \neq \mathcal{Y}$, new blocks are added as "glue" that map
them to the intermediate representations the pretrained middle expects (slides 14–17, ≈17:04–19:23): an extra encoding for an RGB-D
camera's depth channel, PCA down to three channels for hyperspectral satellite data, or a new output layer for an extra class
(≈20:11–21:42). New blocks can even be initialized by copying, such as a new depth channel starting from the blue channel's weights
(≈23:59). Finetuned too long on too little data, a model forgets; slide 18's remedies are to freeze most of the network, use a small
learning rate, stop early, or keep training on the original data, all of which "assume that you have a good, representative
validation set" (≈25:31–27:50). **Domain adaptation** transfers the mapping while aligning a source domain's representations with an
unlabelled target's (slides 19–28; see [domain adaptation](domain-adaptation.md)).

**Knowledge about the outputs** is transferred by **distillation**: a student trained to match a teacher's whole distribution of
predictions, which carries information about similarity that a one-hot label lacks (slides 29–39; see [knowledge
distillation](knowledge-distillation.md)).

Finally, **foundation models** push transfer to its limit: models general enough to be adapted with little or no data, by a linear
probe, by finetuning, or by changing only the input (slides 40–75; see [foundation models and
prompting](foundation-models-and-prompting.md)). Slide 68 sets the three side by side: "Adaptation can occur on inputs, mapping, or
outputs", by prompting, finetuning or a linear probe.

## Knowledge about the inputs, and learning to transfer (lecture 19)

Lecture 19 takes up the third kind: "if you know something about the distribution of inputs you're going to expect, maybe there's a
way to actually use that to direct or accelerate your learning" (≈1:33). Its tool is a generative model trained on the data, which
with its latent space and inverse map becomes **data++**, "a new mechanism to access data, going beyond the capabilities of the
initial training data set" (≈3:08): its latent directions answer counterfactual questions, its samples can be labelled from a few
examples (DatasetGAN: "1 labeled GAN image is worth ~100 labeled regular images!", slide 26), it can explain a classifier, and it can
supply contrastive learning's positive pairs (slides 4–42; see [generative models as data](generative-models-as-data.md)). The
lecture then asks whether a model can be trained to be good at transfer itself: **meta-learning**, by MAML, which learns an
initialization that a few SGD steps adapt well, or by sequence modelling, where a model's state learns in context (slides 43–50; see
[meta-learning](meta-learning.md)).

## From pretrain and transfer to pretrain and prompt (lecture 21)

[Lecture 21](21-language-models.md) tells the history of NLP as a history of transfer. Until 2014 a language model was a component tacked
onto a task-specific system; from 2014 the task itself was posed as conditional next-word prediction; from about 2017 "you could get
nontrivial transfer effects between different tasks", pretraining on next-word or masked-word prediction and fine-tuning, "a way to get
away with much, much less data for any specific downstream task"; and from 2020 a single model is prompted rather than fine-tuned at all
(slides 8–11, ≈10:03–13:06). Slides 41 and 42 draw the change: from collecting data and training a model for every task to a few examples
per task and "Single model for all tasks!" That works only because the pretraining data "already contains evidence and information about
how to perform most of these specific downstream things": scrape the internet and "you find examples for translation tasks and
summarization tasks and entity recognition tasks and sentiment analysis tasks" (≈35:50–37:26). Fine-tuning still wins for "some totally
novel thing" with "lots and lots" of examples, but "most people don't have the ability to fine-tune a very large model" (≈50:36–51:21).
See [foundation models and prompting](foundation-models-and-prompting.md).

## See also

- [Representation learning](representation-learning.md) — what is being transferred.
- [Self-supervised learning](self-supervised-learning.md) — pretraining without labels.
- [Lecture 11](11-representation-learning-reconstruction-based.md) — linear adaptation, fine-tuning and the three phases.
- [Lecture 18](18-transfer-learning-models.md) — the lecture on transfer learning from the model side.
- [Domain adaptation](domain-adaptation.md) — transferring the mapping to an unlabelled target domain.
- [Knowledge distillation](knowledge-distillation.md) — transferring knowledge about the outputs.
- [Foundation models and prompting](foundation-models-and-prompting.md) — adapting a fixed model with little or no data.
- [Lecture 19](19-transfer-learning-data.md) — the lecture on transfer learning from the data side.
- [Generative models as data](generative-models-as-data.md) — transferring knowledge about the inputs.
- [Meta-learning](meta-learning.md) — learning to transfer quickly.
