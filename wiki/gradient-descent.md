# Gradient descent

Gradient descent is how a deep network's parameters are learned. Lecture 1 lists it first among
the things the course **expects you to have seen** (slide 26), and defers the details —
backpropagation, how gradients reach every parameter, the choice of step size — to lecture 2.
Covered so far: [lecture 1](01-introduction.md), slides 26–30 and 64–66, ≈24:05–25:37. For the
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

## Where it goes next

Lecture 1 frames the course's version of this topic as **differentiable programming**: "what
does it actually look like to build programming languages that are structured around the idea of
optimality and optimizing through gradient descent", and "what are you actually optimizing for,
and what are you pushing the gradients through to?" (≈24:50). Slide 30's banner assigns this —
"Backprop and Differentiable Programming" — to **lecture 2**. Stochastic gradient descent and
its variants are also promised "in an upcoming lecture" (≈24:50). Autograd frameworks such as
PyTorch and TensorFlow are what make this practical today, by implementing "the chain rule in
software" (≈20:15).

For classification, the loss $L$ in the objective above is usually the cross-entropy — see
[softmax and cross-entropy](softmax-and-cross-entropy.md).
