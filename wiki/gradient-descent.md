# Gradient descent

Gradient descent is how a deep network's parameters are learned. Lecture 1 lists it first among
the things the course **expects you to have seen** (slide 26), and defers the details —
backpropagation, how gradients reach every parameter, the choice of step size — to lecture 2,
which reviews the algorithm and its stochastic and momentum variants. Covered so far:
[lecture 1](01-introduction.md), slides 26–30 and 64–66, ≈24:05–25:37;
[lecture 2](02-how-to-train-a-neural-net.md), slides 4–11, ≈0:45–10:54. For the
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
plateauing of change on that validation set" (lecture 2, ≈41:12).

For classification, the loss $L$ in the objective above is usually the cross-entropy — see
[softmax and cross-entropy](softmax-and-cross-entropy.md).
