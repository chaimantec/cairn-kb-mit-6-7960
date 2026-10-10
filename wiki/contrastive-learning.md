# Contrastive learning

**Contrastive learning** trains an encoder by contrast: representations of a **positive pair**, two inputs that should be
similar, are pulled together, and representations of **negatives** are pushed apart. Its supervised form grew out of
[metric learning](metric-learning.md); its **self-supervised** form makes the pairs without labels, typically from two
augmentations of the same image, and is "Ideas from metric learning and self-supervision" combined
([lecture 12](12-representation-learning-similarity-based.md), slide 27). Lecture 12 calls it "a pretty important point
throughout this entire course because it turns out that many of our modern architectures are built on top of some form,
or some component of contrastive learning and contrastive representations" (≈0:46). Covered so far: lecture 12, slides
17–69, ≈21:45–1:15:46; CLIP, an image–text contrastive model, in [lecture 2](02-how-to-train-a-neural-net.md), slide 69,
and [lecture 11](11-representation-learning-reconstruction-based.md), slide 13. [Lecture 13](13-representation-learning-theory.md), the
course's representation-learning theory lecture, recaps contrastive learning (slide 4, ≈4:49–5:36) and then asks what similarity
an architecture expresses before any training; see [Gaussian processes](gaussian-processes.md).

**Notation.** The encoder $f$ maps data onto the unit hypersphere, $f : \mathcal{X} \rightarrow \mathbb{S}^{d-1}$. An anchor
$\mathbf{x}$ has a positive $\mathbf{x}^+$, drawn with it from the positive-pair distribution $p_{pos}$, and $N$
negatives $\mathbf{x}_ 1^-, \ldots, \mathbf{x}_ N^-$ drawn from the data distribution $p_{data}$. $\tau$ is the
temperature that divides the inner products.

## From triplets to many negatives

The first contrastive loss is the triplet loss: the anchor's distance to its negative must exceed its distance to its
positive by a margin (lecture 12, slide 18; see [metric learning](metric-learning.md#the-triplet-loss-and-its-descendants)).
Comparing each positive pair with many negatives at once, as the lifted structured loss does over a whole batch (slide 21),
uses the batch better and finds more hard negatives, the only ones that give a gradient (≈25:44–26:30).

## The loss: a softmax over similarities

The self-supervised setup normalizes representations onto a hypersphere and uses a "Cross-entropy for softmax 'classifier'
to discriminate 'classes' defined by similarities" (lecture 12, slide 28):

$$\min_f \thinspace \mathbb{E}_ {(\mathbf{x}, \mathbf{x}^+) \sim p_{pos}, \thinspace \lbrace \mathbf{x}_ i^- \rbrace_ {i=1}^{N} \sim p_{data}} \left[ -\log \frac{e^{f(\mathbf{x})^{\top} f(\mathbf{x}^+)/\tau}}{e^{f(\mathbf{x})^{\top} f(\mathbf{x}^+)/\tau} + \sum_{i=1}^{N} e^{f(\mathbf{x})^{\top} f(\mathbf{x}_ i^-)/\tau}} \right]$$

The numerator "pull[s] positive pair together" and the sum over negatives "push[es] negative pairs apart" (slide 28). It is a
"cross-entropy loss to distinguish data points" (slide 39). On the sphere an inner product measures angle: "we want to make sure that the inner
product is as close as possible to 1 because that would mean that they were as close as possible on this sphere" (≈32:43).
The positive-pair distribution is assumed symmetric, $p_{pos}(\mathbf{x}, \mathbf{x}^+) = p_{pos}(\mathbf{x}^+, \mathbf{x})$,
and to have the data distribution as its marginal, $\int p_{pos}(\mathbf{x}, \mathbf{x}^+) \thinspace d\mathbf{x}^+ = p_{data}(\mathbf{x})$
(slide 28). The family includes noise-contrastive estimation (NCE; Gutmann and Hyvärinen 2010) and the **InfoNCE** loss
(van den Oord et al. 2018), with "similar losses also in metric learning" (slide 29).

The sphere matters for two reasons: "more stable training (logistic regression needs regularization)", and
"well-clustered classes on hypersphere are linearly separable (cut off caps)" (slide 31), so "you can always build a linear
classifier, just a slice through that hypersphere" (≈35:54). See [softmax and cross-entropy](softmax-and-cross-entropy.md).

## Making pairs without labels

Negatives are "randomly uniformly drawn from data"; positives are "perturbations that keep semantic meaning, data
augmentation" (slide 33): crops, flips, colour distortion, rotation, cutout, noise, blur and Sobel filtering, after Chen,
Kornblith, Norouzi and Hinton (2020). **SimCLR** generates two random augmentations of each data point in a batch as its
positive pair and uses "all other 2(B-1) augmented samples in the batch (of size B)" as negatives (slide 34). Trained this
way, a representation "can outperform supervised pre-training (for some tasks)" (slide 30, citing He et al. 2020 and Misra
and van der Maaten 2020), because a supervised signal that is "not well-aligned with the actual task" can teach invariances
"that are explicitly unhelpful" (≈35:07).

Positives can also be different **views** of the same scene, encoded by two encoders $f^x$ and $f^y$ (slides 35–37): two
channels such as depth and colour (CMC; Tian, Krishnan and Isola 2020), frames of the same video, or an image and its
caption (Karpathy, Joulin and Fei-Fei 2014; CLIP, Radford, Kim et al. 2021). Lecture 2 shows CLIP's training grid, in which
every image in a batch is scored against every caption and the matching pairs lie on the diagonal (lecture 2, slide 69).
"A lot of modern work actually really is just people coming up with creative ways to supervise models without requiring
labels" (lecture 12, ≈42:52).

Random negatives are sometimes not negative at all. With several pictures of one thing in the data set, "these types of
positive negatives, almost in a way, are confusing" (≈38:12). Over many batches the signal still averages out, since "more
often than not, you're going to get two augmentations of a golden retriever and a boat, not two augmentations of a golden
retriever and another golden retriever" (≈41:21). Removing the false negatives with labels raises accuracy (slide 52,
Chuang et al.), and **supervised contrastive learning** goes further, using "images from same class as positive pairs
(multiple positive pairs)" (slide 53, Khosla et al. 2020), with labelled points as anchors when only some are labelled
(≈1:01:02).

## What the loss does: alignment and uniformity

The contrastive loss "maximizes a lower bound on mutual information between 'views'" (Poole et al. 2019),
$\text{MI}(f(\mathbf{x}), f(\mathbf{x}^+)) \geq \log(N) - \mathcal{L}(f)$, where $\mathcal{L}(f)$ is the contrastive loss
(slide 39). But maximizing mutual
information alone "actually worsened downstream performance", so "the contrastive loss has to be doing something else"
(≈44:28). Lecture 12's answer, after Wang and Isola (2020), is two properties:

- **Alignment**: positives map close together. It "is really built into that loss from the beginning" (≈45:15).
- **Uniformity**: features spread over the whole sphere. Pushing all negatives apart equally is best done by covering the
  sphere, "because if we don't use part of the hypersphere, then there's actually a way we could have pushed even further
  apart things that are different" (≈46:03); its caption is "Uniformity: Preserve maximal information" (slide 41).

As the number of negatives goes to infinity, the loss splits into two terms, the first minimized "if and only if f is
perfectly aligned" and the second "if we get perfect uniformity" (≈49:11–50:49). The lecturer defines both as measurable
quantities, alignment as the expected distance between the features of a positive pair and uniformity as "the logarithm of
the expected pairwise Gaussian potential, which is essentially e to the minus squared distance", minimized by the uniform
distribution on the sphere (≈47:38–49:11); the OCW deck prints neither formula. On CIFAR-10 encoders with a circle as feature
space, contrastive learning spreads the features uniformly where supervised learning does not (slide 42), and across 306
STL-10 and 108 BookCorpus encoders the most accurate are those with the lowest alignment and uniformity losses (slide 43).

## What the data does: learned invariance

The loss gives alignment and separation; the **choice of pairs** gives robustness. "positive pairs = augmentations of the
same data point should be close", so the "learned representation is invariant to perturbations induced by data
augmentations: learned invariance" (slide 45). That makes augmentation a design decision: a model that must tell left shoes
from right shoes cannot be trained with random flips (≈39:00–39:47), and molecules, text and video each need their own
domain knowledge. Slide 45 asks when learned invariances beat hard-coded ones; augmentation builds in invariances that "can
be really difficult to define mathematically", such as invariance "to pose within a scene" (≈53:10). See
[data augmentation](data-augmentation.md) and [inductive bias](inductive-bias.md).

## The SimCLR ingredients

Slide 47 lists what makes self-supervised contrastive learning work: heavy data augmentation, projection heads, large batch
size (many negatives), and the choice of data pairs and hard negatives.

- **Heavy augmentation.** Removing transformations costs accuracy, most of all for SimCLR: with crop alone it drops by
  roughly 27 points in Grill et al.'s comparison (slide 48).
- **Projection head.** The loss is applied to $g(\mathbf{h})$, with $g$ "linear or small MLP", and $\mathbf{h}$ is kept for the
  downstream task (slide 49). It "improves performance", "Possibly because representation $\mathbf{h}$ then need not be
  completely invariant to augmentations, can retain some information" (slide 50). The lecturer compares the separate
  projections a transformer uses for similarity and for passing information (≈57:03; see [transformers](transformers.md)).
- **Large batches.** In SimCLR the batch supplies the negatives, so accuracy rises with batch size, most of all early in
  training, though by 1000 epochs the sizes from 2048 up perform alike (slide 51). That is expensive, and MoCo (He et al. 2020) makes it more efficient (slide 51, ≈58:41).
- **Good negatives.** Hard negatives give signal, and false negatives hurt (slides 25 and 52).

## When it falls short

Contrastive learning learns whatever similarity its pairs encode. On iNaturalist 2021 (Cole et al., CVPR 2022), SimCLR and
MoCo with a linear probe nearly match a supervised model at the kingdom, phylum and class levels but trail it by about 30
points at the species level, where on ImageNet the gap is about 7 (slides 60–66, ≈1:04:55). A query bird held in a hand
retrieves other birds held in hands: the augmentations taught "contextual similarity ... than it is giving us information
about species similarity" (slides 67–68, ≈1:08:02). "For contrastive learning to work well, you need to have a good
similarity measure for your problem of interest" (≈1:07:16), and the benchmark decides how good "good enough" looks: on
ImageNet, whose categories are coarse, "without any labels at all, we're still building representations that are very, very
close to as good as supervised representations" (≈1:09:36). For open-set problems such as re-identifying unseen people,
though, contrastive and triplet objectives "often will generalize better" than a classification loss (≈1:12:41–1:13:28).

## The problem set

Homework 4's first section, "Similarity-Based Learning: Self-Supervised and Supervised" (13 points), analyses a contrastive
loss over groups of similar samples: what encoders classify perfectly, how that connects alignment, margin and uniformity,
the role of normalization and temperature, and supervised cross-entropy as a dual-encoder contrastive method. Its second
section's question 8 (6 points) implements the standard loss with a temperature and asks which invariances three sets of
augmentations would teach. See [lecture 12](12-representation-learning-similarity-based.md#the-problem-set).

## See also

- [Lecture 12 — Representation Learning: Similarity-Based](12-representation-learning-similarity-based.md), where this
  material comes from.
- [Metric learning](metric-learning.md) — the supervised origin: Mahalanobis distances, triplet loss, hard negatives.
- [Self-supervised learning](self-supervised-learning.md) — the predictive route to learning without labels.
- [Representation learning](representation-learning.md) — what makes a representation good.
- [Data augmentation](data-augmentation.md) — the transformations that define positives.
- [Transfer learning](transfer-learning.md) — linear probes on a pretrained representation.
