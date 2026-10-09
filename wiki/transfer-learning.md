# Transfer learning

Transfer learning reuses what a network learned on one task to learn another faster, from less data. Lecture 11
gives it as the answer to "why learn representations?": "Maybe the simplest is to do more learning, right? So we learn
to learn … the main use of trained representations is to accelerate future learning"
([lecture 11](11-representation-learning-reconstruction-based.md), ≈27:54–28:41). The course has two lectures on it
later, **18 (models)** and **19 (data)**, which this knowledge base does not yet cover (see the
[course map](course-map.md)). Covered so far: [lecture 1](01-introduction.md), slides 76–77, ≈57:19–58:53 (reusing
weights); [lecture 2](02-how-to-train-a-neural-net.md), ≈56:42 and ≈1:10:48–1:12:21 (pretrained parameters inside a
larger network); [lecture 9](09-hackers-guide-to-deep-learning.md), slides 27–28, ≈56:26–58:43 (start from a pretrained
model); lecture 11, slides 23–29, ≈27:54–39:32; [lecture 12](12-representation-learning-similarity-based.md), slides 3, 30 and 60–68, ≈2:20–3:05 and ≈1:03:23–1:10:21
(transfer as a reason to learn representations, self-supervised against supervised pretraining, and linear probes on
iNaturalist).

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

## See also

- [Representation learning](representation-learning.md) — what is being transferred.
- [Self-supervised learning](self-supervised-learning.md) — pretraining without labels.
- [Lecture 11](11-representation-learning-reconstruction-based.md) — linear adaptation, fine-tuning and the three phases.
