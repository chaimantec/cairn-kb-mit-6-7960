# Gradient descent

Gradient descent is how a deep network's parameters are learned. Lecture 1 lists it first among
the things the course **expects you to have seen** (slide 26), and defers the details —
backpropagation, how gradients reach every parameter, the choice of step size — to lecture 2,
which reviews the algorithm and its stochastic and momentum variants. Covered so far:
[lecture 1](01-introduction.md), slides 26–30 and 64–66, ≈24:05–25:37;
[lecture 2](02-how-to-train-a-neural-net.md), slides 4–11, ≈0:45–10:54; [lecture 6](06-generalization-theory.md),
slide 61 and ≈35:34–40:10, on what the optimizer adds to generalization;
[lecture 7](07-scaling-rules-for-optimization.md), slides 4–7 and 16–18, ≈3:06–13:55 and
≈33:37–39:53, which derives gradient descent as steepest descent in the Euclidean norm. For the
symbols see [notation](notation.md).

## Learning as optimization

The model is a function $f_\theta$ with parameters $\theta$, trained on $N$ examples
$(x^{(i)}, y^{(i)})$. A per-example **loss** $L$ measures how wrong the model's output is on one
example. Learning means finding the parameters that minimize the total loss over the training
set (slide 27):

$$\theta^{\ast} = \arg\min_\theta \sum_{i=1}^{N} L\big(f_\theta(x^{(i)}), y^{(i)}\big)$$

The sum is the **cost**, $J(\theta)$, so slide 28 writes the same objective as
$\theta^{\ast} = \arg\min_\theta J(\theta)$. In the lecturer's words, $\theta^{\ast}$ is "your
optimal set of model parameters that minimize your cost function" (≈24:05).

## The algorithm

"Maybe you start somewhere random, and then you calculate your gradient. Now you update the
weights of that model in the direction of the gradient, and you step through, et cetera, et
cetera" (≈24:05). Slide 30 gives one iteration:

$$\theta^{t+1} = \theta^{t} - \eta_t \frac{\partial J(\theta)}{\partial \theta}\Big|_ {\theta=\theta^{t}}$$

Here $\theta^{t}$ is the parameter vector after $t$ steps, and $\eta_t$ is the **learning rate**,
the step size at step $t$. The minus sign means each step moves *against* the gradient, downhill
on $J$. Slide 28 draws exactly this: a loss surface over two parameters $\theta_1, \theta_2$, a
start marked X on a peak, and a chain of arrows walking down into a deep well.

The lecture describes the result as "at least, local optimal" (≈1:35). Gradient descent follows
the slope from wherever it starts, so it finds *a* minimum, not necessarily the best one.

## Why it requires differentiability

Every step needs $\partial J / \partial \theta$, so every part of the model the gradient passes
through must be differentiable, with gradients that are not trivially zero. This is the lecture's
argument against the step-function perceptron (≈28:47): its gradient is zero almost everywhere, so
"you're not going to move because the gradient is zero". It is also why saturating non-linearities,
whose gradients vanish far from the centre, train slowly (see
[activation functions](activation-functions.md)).

## What can be computed about the cost

Lecture 2 sorts optimizers by what they can evaluate (slide 6, ≈3:08–3:54): $J(\theta)$ alone is
**black-box optimization**; adding the gradient $\nabla_\theta J(\theta)$ makes it
**first-order**; adding the Hessian $H_\theta(J(\theta))$ makes it **second-order**. Deep learning
lives almost entirely in the first-order case: second-order methods are "not something that we often
actually do. Usually, we really just focus on … first order optimization, where we're just looking
at that linear approximation" (≈3:54). Lecture 2 writes the step with an iteration index $k$ and a
fixed rate (slide 8, ≈4:41):

$$\theta^{k+1} = \theta^{k} - \eta \nabla_\theta J(\theta^{k})$$

## Stochastic gradient descent

Summing the loss over every example is "often not possible to calculate, mostly just due to the
time", with datasets now "in the billions" (lecture 2, ≈5:29–6:15). **Stochastic gradient
descent** computes each step's gradient on a **batch**, a subset of the data (slide 9). A batch of
1 updates after every example; a batch of the whole dataset is ordinary gradient descent
(≈6:15–7:02). The batch gradient is a noisy estimate of the full one. That noise is "an implicit
regularizer" that "can actually bounce you out of local minima", and SGD is faster, but the updates
have high variance. In the lecturer's experience they are "more unstable the more unbalanced your
training data set is over the set of categories" (≈7:02–8:35).

## Momentum

Momentum biases each step "to continue in direction of previous update", like "a heavy ball rolling
down a hill" (lecture 2, slide 10, ≈8:35–10:06):

$$\theta^{t+1} = \theta^{t} - \eta \nabla f(\theta^{t}) - \alpha m^{t}$$

Here $m^{t}$ is the momentum term carrying the direction of past movement and $\alpha$ its strength,
a hyperparameter. It "can help or hurt": on slide 10's V-shaped loss, momentum 0.5 reaches the
minimum in about half the steps of plain gradient descent, while momentum 0.95 overshoots and is
still oscillating 90 steps in (≈10:06). The lecturer names Adam as "a popular example of this type
of momentum in gradient descent", and recommends Goh's Distill article "Why Momentum Really Works"
(slide 11) for intuition.

## When the landscape is hard

Gradient descent only knows the local slope, so the shape of $J$ decides how it behaves: it stalls
on flat regions, oscillates around a cusp, and stops in whichever local minimum it reaches first.
Lecture 2 shows each failure, plus two remedies — evolution strategies, which sample perturbations
instead of following the gradient, and gradient clipping, which caps each gradient component (slides
12–22). See [loss landscapes](loss-landscapes.md).

## Getting the gradient

Lecture 1 frames the course's version of this topic as **differentiable programming**: "what
does it actually look like to build programming languages that are structured around the idea of
optimality and optimizing through gradient descent", and "what are you actually optimizing for,
and what are you pushing the gradients through to?" (≈24:50). Autograd frameworks such as PyTorch
and TensorFlow make this practical by implementing "the chain rule in software" (≈20:15). Lecture 2
derives the algorithm they implement — see [backpropagation](backpropagation.md) and
[differentiable programming](differentiable-programming.md).

**A sign convention.** Lecture 2's linear-layer and worked-example slides write the update as
$\mathbf{W} \leftarrow \mathbf{W} + \eta (\partial J / \partial \mathbf{W})^{\mathsf{T}}$ with a
negative learning rate, "η = -0.2 (because we used positive increments)" (slide 72). That is the
same downhill step as the minus form above with a positive rate.

**When to stop.** "Often using something like a validation set and looking for some sort of
plateauing of change on that validation set" (lecture 2, ≈41:12). Lecture 6 adds that the number of
gradient-descent steps "is kind of a hyperparameter … And if you do too many, then you might be
overfitting. And that's kind of a classical idea. But what we'll see is that in this neural net era, it
tends to be just train forever" (≈36:20–37:07), because test error shows double descent in compute as
well as in parameters: "we're past the first peak, and we're in the part where you just train longer,
and longer, and longer, and it smoothly will get better" (≈39:25–40:10). See [generalization and double
descent](generalization-and-double-descent.md).

For classification, the loss $L$ in the objective above is usually the cross-entropy — see
[softmax and cross-entropy](softmax-and-cross-entropy.md).

## What the optimizer prefers (lecture 6)

Gradient descent does not pick a random solution among those that fit the data, and lecture 6 makes
that one of the candidate reasons deep nets generalize (slide 61, ≈1:12:56–1:15:19). **Weight decay**
"acts like an L2 regularizer on weights, shrinking them toward zero all else being equal", so weights
the network does not use go to zero and the solution has lower norm. **Initialization** near zero "biases
solutions toward low norm. GD initialized near zero converges to minimum norm solution for linear models
[Zhang et al. 2017, Gunasekar et al. 2017]". And **SGD, and GD with finite step size**, "converge to
'flat' minima; they will tend to overshoot or bounce out of minima that are too narrow" (see [loss
landscapes](loss-landscapes.md)). Because the optimizer is random (random initialization, random
mini-batches), clean statements about which solution it reaches are hard; one makes empirical ones, "if
we run this neural network over and over again with the Adam optimizer, or the SGD optimizer, here's what
we observe", or probabilistic ones (≈35:34–36:20).

## Gradient descent as steepest descent (lecture 7)

Lecture 7 writes the problem with an average rather than a sum,
$\mathcal{L}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^{N} \ell(f(\mathbf{x}^{(i)}, \mathbf{w}), y^{(i)})$, where $\ell$ is an error measure such as
cross-entropy or square error and $\mathbf{w}$ the weights (slide 4), and the step as
$\mathbf{w} \to \mathbf{w} - \eta \thinspace \partial \mathcal{L} / \partial \mathbf{w}$ (slide 5). Of
the three things that make it hard (a lot of weights, a lot of layers, and a lot of data, which forces
mini-batches), the lecture sets the noise aside and studies **full-batch** optimization, "it's already
interesting" (slide 6, ≈6:11–7:45).

It then asks where the update rule comes from. Expand the loss to first order around the current
weights, replace everything beyond the linear term by a penalty
$\frac{\lambda}{2} \Vert \Delta \mathbf{w} \Vert^2$, and minimize. With the Euclidean norm the answer is
$\Delta \mathbf{w} = -\frac{1}{\lambda} \mathbf{g}$, "vanilla gradient descent!" with learning rate
$1/\lambda$ (slide 17, ≈37:35). With the infinity norm the same recipe gives **sign gradient descent**,
$\Delta \mathbf{w} = -\frac{\Vert \mathbf{g} \Vert_ 1}{\lambda} \operatorname{sign}(\mathbf{g})$ (slide 18).
So the gradient direction is the best direction only if distances in weight space are Euclidean: "when
you say the best direction is the gradient direction, I'm trying to say that you're thinking Euclidean"
(≈47:38). See [steepest descent](steepest-descent.md).

Second-order alternatives exist, Newton's method and Gauss-Newton among them, "but then in practice, we
just use Adam to train neural networks" (≈21:05); see [second-order methods](second-order-methods.md).
And the learning rate that works best is not stable as a network grows: naively, the optimal learning
rate drifts as width increases, so it has to be retuned at every size (slide 7, ≈8:30–10:04). Lecture 7's
fix is to measure and normalize each update in the right norm; see [scaling rules](scaling-rules.md).
