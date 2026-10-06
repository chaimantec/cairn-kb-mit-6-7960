# Softmax and cross-entropy

How a classifier turns its last layer into a prediction, and how it is scored against the label.
Lecture 1 lists "softmax, cross-entropy loss" among what the course **expects you to have seen**
(slide 53), and reviews them on a running example: classifying photos into animal classes — clown
fish, grizzly bear, chameleon and so on. Covered so far: [lecture 1](01-introduction.md), slides
53–66, ≈49:40–53:30.

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
