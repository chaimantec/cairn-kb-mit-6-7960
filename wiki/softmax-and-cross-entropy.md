# Softmax and cross-entropy

How a classifier turns its last layer into a prediction, and how it is scored against the label.
Lecture 1 lists "softmax, cross-entropy loss" among what the course **expects you to have seen**
(slide 53), and reviews them on a running example: classifying photos into animal classes — clown
fish, grizzly bear, chameleon and so on. Covered so far: [lecture 1](01-introduction.md), slides
53–66, ≈49:40–53:30; [lecture 2](02-how-to-train-a-neural-net.md), ≈1:03:45–1:04:32 (logits
versus probabilities as optimization targets); [lecture 8](08-architectures-transformers.md), slides 29, 34 and 49, ≈35:36,
≈48:51 and ≈1:07:25 (the softmax inside attention, and next-word prediction as classification);
[lecture 9](09-hackers-guide-to-deep-learning.md), slides 29–41 and 49, ≈58:43–1:03:24 and ≈1:10:21–1:11:53 (classification as the
default formulation, colorization turned into classification, and the loss at chance); [lecture 10](10-architectures-memory.md), slides 44–46 and 49–50, ≈48:08–51:16 and ≈53:37 (the next-word classifier, the size of
its vocabulary, and maximum likelihood as cross-entropy); [lecture 11](11-representation-learning-reconstruction-based.md), slides 9–10, ≈11:35–14:39 (the
softmax as a map onto the simplex); [lecture 12](12-representation-learning-similarity-based.md), slides 28–39 and 53, ≈31:11–35:54 and ≈43:43 (the contrastive loss as
a softmax cross-entropy over similarities).

## From last layer to prediction

A classifier's **last layer** has one unit per class (slide 55). The simplest readout is the
**argmax**: whichever unit is most active is the prediction — "the thing that's darkest on here is
clownfish" (≈50:26).

To learn, the output must be compared with the **ground-truth label** by a **loss**, which should
be small when the prediction is right (slide 57) and large when it is wrong (slide 58, where the
network says clown fish and the label says grizzly bear). What is wanted is "some loss function
that does a good job of punishing you when you're wrong and not punishing you when you're right"
(≈51:12).

## Outputs as a distribution

If the network's output is a normalized vector, it "can think of this as a probability
distribution over different possible object classes" (≈51:12). On the slides this normalization
is labelled **softmax**: the last layer's arrows pass through a "softmax" to give $\hat{y}$ (slide
59). Lecture 1 treats softmax as background and **does not write out its formula**; it says only
that the output is normalized.

The label is encoded as a **one-hot** vector $y$, "a vector that's 0 everywhere except for the
place where it's correct, where it's 1" (≈51:12).

The lecturer adds a caution worth keeping: she puts "probability" in quotes because "these are not
true probabilities. They're scores" (≈51:59). They are normalized to sum to one, but nothing
guarantees they are calibrated.

## Cross-entropy

With $K$ classes, the **cross-entropy** between the label $y$ and the prediction $\hat{y}$ is
(slide 59):

$$H(y, \hat{y}) = -\sum_{k=1}^{K} y_k \log \hat{y}_ {k}$$

Because $y$ is one-hot, every term but one vanishes, and $H$ is just $-\log$ of the score the
model gave the true class. The lecture describes it as indicating "the distance between what the
model believes the output distribution should be and what the input distribution actually is"
(≈51:12). Slide 59 annotates it as the "probability of the observed data under the model".

Minimizing it "is telling the network to maximize the probability of the training data, which
forces the output to be a reasonably good probability model of the object class, given the image,
assuming you have enough training data" (≈51:59).

## Reading the score

Slides 60–63 lay this out as three bar charts per example: the prediction $\log \hat{y}$ per
class (an axis running from $-\infty$ to 0), the one-hot ground truth, and their element-wise
product. The product is the score $-H(y, \hat{y}) = \sum_{k} y_k \log \hat{y}_ {k}$, nonzero only in
the true class's row. The gap between that score and 0 is drawn in red and labelled **"How much
better you could have done"**; it is "telling you the direction of the gradient you want to go
in" (≈51:59–52:45).

- **Clown fish, labelled clown fish** (slide 61): correct, but the model's score for the true
  class is modest, so a sizeable red gap remains.
- **Grizzly bear, labelled grizzly bear** (slide 62): confident and correct, so the gap is small.
- **Chameleon, labelled chameleon** (slide 63): the model puts most of its mass on *iguana*.
  Because the output is normalized, the true class is left with little, and the gap — the loss —
  is large (≈52:45).

## The training setup

Put together, training is the [gradient descent](gradient-descent.md) objective with this loss:

$$\theta^{\ast} = \arg\min_\theta \sum_{i=1}^{N} L\big(f_\theta(x^{(i)}), y^{(i)}\big)$$

Slides 64–66 draw it for three examples in turn — $x^{(1)}$ a clown fish, $x^{(2)}$ a grizzly bear,
$x^{(i)}$ a chameleon — each flowing through six layers with parameters
$\theta_1, \ldots, \theta_6$, marked "Learned", into the loss. The lecturer's summary: "you're taking all these
different weights in your massive model, and then you're fiddling around with them until you
match the desired output for all your training data", repeating "over and over and over again
until the model gets everything right, or as close to everything right as possible" (≈52:45–53:30).
Because those per-example losses are simply summed, they can be computed in parallel — see
[tensors and batching](tensors-and-batching.md).

## Logits or probabilities, when optimizing toward a class

Lecture 2 uses gradients to change an *input* so that one class's score rises (see
[differentiable programming](differentiable-programming.md)), and makes a point about where to
push. Because softmax normalizes, "the maybe easiest way to increase the probability softmax given
to a class is often to make the alternatives unlikely, rather than to make the class of interest
likely. Whereas if you optimize pre softmax logits, this tends to actually be a bit more stable"
(≈1:03:45–1:04:32). Its "cat" visualization therefore maximizes the output neuron "maybe pre
softmax".

## The softmax inside attention, and next-word classification (lecture 8)

Lecture 8 uses the softmax away from the output layer. In query-key-value attention, the dot products
of a query with every key form a score vector $\mathbf{s}$, and $\mathbf{A} = \text{softmax}(\mathbf{s})$
turns the scores into weights for a weighted sum of values, "just normalizing that vector so it sums to
1" (slide 29, ≈35:36). In the full self-attention layer the softmax is applied to the scaled query-key
products, "to ensure that everything sums to 1. So that has some nice normalization and numerical
advantages" (slide 34, ≈48:51). And a language model is trained as a classifier: given a sequence, "use
supervised learning to try to classify what is the next word in a vocabulary of possible words" (slide
49, ≈1:07:25). See [transformers](transformers.md).

## Softmax regression as the default formulation (lecture 9)

Lecture 9's advice is to "formulate your problem as **softmax regression** (a.k.a. classification)",
with a cross-entropy loss and one-hot labels (slide 40). It gives three reasons. First, "no restriction on
shape of predictive distribution [up to quantization]": the predicted categorical distribution can put
its mass anywhere over the classes, "the most general way of quantifying uncertainty or modeling a
distribution over a discrete variable", while least-squares regression "assumes Gaussian predictions"
(≈1:01:51–1:02:38). Second, discrete classes are easy to label and to talk about. Third, "all labels are
equidistant under 1-hot representation", which "removes inductive bias about how we represent target
variables" (≈1:02:38–1:03:24). Slide 41's "recipe for deep learning in a new domain", one of its "good
default choices ca 2024", starts the same way: turn the data into one-hot vectors and the goal into a
cross-entropy loss, then use Adam and a transformer.

The lecture's worked case is image colorization, a task that looks like regression (predict a colour for
every pixel) and is turned into classification (slides 29–39, from Zhang, Isola and Efros, ECCV 2016).
The continuous plane of colour values is quantized into $K$ cells, so that each pixel's target becomes a
one-hot vector over $K$ colour classes, $\mathbf{y} \in \mathbb{R}^{H 	imes W 	imes K}$ in place of
$\mathbf{y} \in \mathbb{R}^{H 	imes W 	imes 2}$ (slide 34). "Rather than cat, dog, elephant, it will be
the blue class, the yellow class" (≈1:00:18). A ConvNet slid over the image then classifies every pixel.
See [lecture 9](09-hackers-guide-to-deep-learning.md#transform-your-problem-into-a-solved-problem).

**Know the loss at chance.** Slide 49 says to sanity-check the loss "against a suitable reference value";
for classification with cross-entropy, the uniform distribution. "Get to know log loss numbers": the slide
prints $-0.69 = \ln(0.5)$, chance on binary classification, and $-2.3 = \ln(0.1)$, chance on 10-way
classification. These are log probabilities; with the cross-entropy defined above as a negative log
probability, the loss at chance is 0.69 and 2.3. "If you get that number as your loss, is that good? …
No, that's chance" (≈1:11:07).

## Next-word classification over a vocabulary (lecture 10)

Lecture 10 models each factor of a sentence's probability, such as $p(\texttt{time} \mid \texttt{Once, upon, a})$, as a
next-word classifier: a network and a softmax that "will squish the outputs into a probability mass function", trained by
cross-entropy (slide 44, ≈48:55–49:41). The classes are a choice: words, with $K$ around 100,000, which needs a lot of data
and "can be quite unstable"; characters, $K = 26$, which makes the sequence much longer to predict; or byte pairs in between
(slides 45–46, ≈49:41–51:16). With one-hot targets, maximizing the likelihood of each target word is minimizing the
cross-entropy $\sum_i H(\mathbf{y}_ i, \hat{\mathbf{y}}_ i)$ between targets and outputs (slides 49–50, ≈53:37). See
[autoregressive models](autoregressive-models.md).

## The softmax as a map onto the simplex (lecture 11)

Lecture 11 draws the softmax as a map of a two-dimensional cloud of points, and asks why its output lies on a line (slide 9). A
student answers that it is the line $x_1 + x_2 = 1$: "that's the simplex … the set of points in Rd, where the dimensions sum to 1.
And the output of a softmax is going to be a point in the simplex" (≈11:35–12:21). Slide 9 prints the softmax as
$x_{\texttt{out}}[i] = e^{-\tau x_{\texttt{in}}[i]} / \sum_{k=1}^{K} e^{-\tau x_{\texttt{in}}[k]}$, with a minus sign in the exponent and
a coefficient $\tau$ coloured as a learnable parameter; the lecture does not comment on either. In a width-2 MLP trained with
cross-entropy (slide 10), the last layer's softmax "clamps that back down to this simplex", and each class's points head to its
corner, "the one-hot labels for your data" (≈13:53–14:39); in a classifier with many classes, training pushes the classes "to the
vertices of the simplex" (≈17:50).

## A softmax over similarities (lecture 12)

Lecture 12's self-supervised contrastive loss is a "Cross-entropy for softmax 'classifier' to discriminate 'classes' defined by
similarities" (slide 28). The logits are inner products between unit-norm representations divided by a temperature $\tau$, the
numerator holds the positive pair and the denominator adds the negatives:

$$-\log \frac{e^{f(\mathbf{x})^{\top} f(\mathbf{x}^+)/\tau}}{e^{f(\mathbf{x})^{\top} f(\mathbf{x}^+)/\tau} + \sum_{i=1}^{N} e^{f(\mathbf{x})^{\top} f(\mathbf{x}_ i^-)/\tau}}$$

Here $f$ is the encoder, $\mathbf{x}$ an anchor, $\mathbf{x}^+$ its positive and $\mathbf{x}_ i^-$ its $N$ negatives. It "looks to
us maybe close to a softmax" (≈31:57), and slide 39 calls it a "cross-entropy loss to distinguish data points". Mapping onto a
hypersphere keeps the logits bounded, which the lecture compares to regularizing logistic regression (slide 31, ≈35:54). Lecture
12 also sets cross-entropy against contrastive objectives: "Contrastive learning provides more geometric and robustness feedback
than cross-entropy loss" (slide 53), and for open-set problems a cross-entropy loss "based on a fixed number of categories" says
"almost nothing" about new categories (≈1:13:28). See [contrastive learning](contrastive-learning.md).
