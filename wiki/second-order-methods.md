# Second-order methods: Newton and Gauss-Newton

A **second-order** optimization method uses the second derivatives of the loss, the Hessian, as well
as the gradient. [Lecture 2](02-how-to-train-a-neural-net.md) places them in its sorting of
optimizers and notes that deep learning rarely uses them; [lecture
7](07-scaling-rules-for-optimization.md) derives two, Newton's method and the Gauss-Newton method,
from a Taylor expansion of the loss, names the assumption each one makes, and asks why neither is
used in practice. Covered so far: lecture 2, slide 6, ≈3:08–3:54; lecture 7, slides 9–15,
≈13:55–33:37.

**Notation** follows lecture 7: $\mathcal{L}(\mathbf{w})$ is the loss over weights $\mathbf{w}$,
$\mathbf{g} = \partial \mathcal{L} / \partial \mathbf{w}$ the gradient,
$\mathbf{H} = \partial^2 \mathcal{L} / \partial \mathbf{w}^2$ the Hessian, $\Delta \mathbf{w}$ a step,
and $d$ the number of weights. For a composite loss, $f$ is the network and $\ell$ the error measure.

## First order and second order

Lecture 2 sorts optimizers by what they can compute about the cost (slide 6): only its value
(black-box), its gradient as well (first-order), or its Hessian too (second-order). Second-order
methods are "not something that we often actually do. Usually, we really just focus on … first order
optimization, where we're just looking at that linear approximation" (lecture 2, ≈3:54).

Lecture 7 makes the same split (slide 9) and then explains it. Every classical method "start[s] by
Taylor expanding the loss function" (≈15:32), around the current weights (slide 10):

$$\mathcal{L}(\mathbf{w} + \Delta \mathbf{w}) = \mathcal{L}(\mathbf{w}) + \mathbf{g}^{T} \Delta \mathbf{w} + \frac{1}{2} \Delta \mathbf{w}^{T} \mathbf{H} \thinspace \Delta \mathbf{w} + \cdots$$

The first two terms are the **linearization**, the rest the **non-linear part**. $\mathbf{g}$ is a
vector in $\mathbb{R}^d$; $\mathbf{H}$ is a $d \times d$ matrix of all pairs of second derivatives
(≈14:43, ≈17:08–17:59). The methods differ in what they do with the non-linear part. First-order
steepest descent replaces it with a norm (see [steepest descent](steepest-descent.md)); the
second-order methods below keep the Hessian term.

## Newton's method

Keep the expansion to second order, written on slide 11 with a $\lambda$ on the quadratic term, and
minimize it over $\Delta \mathbf{w}$:

$$\mathcal{L}(\mathbf{w} + \Delta \mathbf{w}) \approx \mathcal{L}(\mathbf{w}) + \mathbf{g}^{T} \Delta \mathbf{w} + \frac{\lambda}{2} \Delta \mathbf{w}^{T} \mathbf{H} \thinspace \Delta \mathbf{w}.$$

Setting the derivative to zero gives $\mathbf{g} + \lambda \mathbf{H} \Delta \mathbf{w} = 0$, and the
slide boxes **Newton's method**,

$$\Delta \mathbf{w} = -\mathbf{H}^{-1} \mathbf{g},$$

the $\lambda = 1$ case (the box carries no $\lambda$; the slide does not comment). It is described as
"pre-condition the gradient with the inverted Hessian": invert the Hessian and multiply it into the
gradient (lecture 7, ≈18:46–19:33).

**Its problems** (slide 12, ≈20:20–21:54):

- **The Hessian is too big.** With $d$ parameters it is $d \times d$, "too expensive even for 'small'
  networks". "d may be billions in a large neural network, so you can't even store such a large
  matrix."
- **It may head for a maximum.** Setting the derivative to zero "is just finding a critical point of the
  quadratic form. If it's close to a max, it could find a maximum … It could be doing gradient ascent."
  **Cubic regularization**, adding a cubic penalty to the quadratic form, fixes this, "but people don't
  use that".
- **Practice wins.** "There's always like attempts to make it practical. And like there's a literature on
  that method. But then in practice, we just use Adam to train neural networks."

## The Gauss-Newton decomposition

A deep learning loss is a **composite**, $\mathcal{L} = \ell \circ f$: an error measure applied to a
network's output (slide 13). The chain rule gives the gradient,

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial \mathbf{w}},$$

and differentiating that product again, by the product rule and the chain rule, splits the Hessian in
two:

$$\frac{\partial^2 \mathcal{L}}{\partial \mathbf{w}^2} = \frac{\partial f}{\partial \mathbf{w}} \cdot \frac{\partial^2 \ell}{\partial f^2} \cdot \frac{\partial f}{\partial \mathbf{w}} + \frac{\partial \ell}{\partial f} \frac{\partial^2 f}{\partial \mathbf{w}^2}.$$

This is the **Gauss-Newton decomposition**. The first term carries the **curvature of the error**,
$\partial^2 \ell / \partial f^2$, and the second the **curvature of the model**,
$\partial^2 f / \partial \mathbf{w}^2$ (slide 14): "it's like decomposing curvature into one piece from
the error and one piece from the model" (≈24:18).

## The Gauss-Newton method

Two assumptions turn the decomposition into an algorithm (slide 14, ≈24:18–26:41):

1. For the square loss $\ell = \frac{1}{2}(f - y)^2$, the curvature of the error is
   $\partial^2 \ell / \partial f^2 = 1$.
2. Ignore the curvature of the model. "It's just an assumption that people make. They say, hey, I don't
   really know how to deal with this term, so let's just ignore it."

The Hessian is then a product of two derivatives of the model, and Newton's method becomes the
**Gauss-Newton method**:

$$\Delta \mathbf{w} = - \left[ \frac{\partial f}{\partial \mathbf{w}} \frac{\partial f}{\partial \mathbf{w}} \right]^{-1} \mathbf{g}.$$

The slides write both factors alike, with no transpose; "all of these things are tensors and you need to
be careful about all the indexing" (≈25:54). These are derivatives of the model's output, "not the same
thing as the regular gradient" (≈26:41).

**Its problems** (slide 15, ≈26:41–28:12): it requires computing the extra derivatives
$\partial f / \partial \mathbf{w}$; it is not clear that ignoring the curvature of the model is safe
(the slide crosses that term out); and it still forms a matrix and inverts it, which may be low rank, so
in practice people "add a little bit of identity matrix to it to give it better conditioning".
"People are not going to want to do that in practice unless they're really convinced that there's a
really big improvement … And basically, nobody's convinced them of that."

## Why the extra derivatives are expensive

Backpropagation computes $\mathbf{g}$ without ever forming $\partial f / \partial \mathbf{w}$: "you start
from the end of the network and work backwards", tracking only derivatives of the loss, "which is a
single number" (≈28:59, ≈32:49). The network's output may be a tensor, and
$\partial f / \partial \mathbf{w}$ holds the derivative of every output component with respect to every weight, "a bigger
tensor than the gradient". Forward-mode automatic differentiation would produce it, "but it's much more
expensive" (≈28:59). See [backpropagation](backpropagation.md).

Inverting a $d \times d$ matrix is the other cost. The lecturer, "not an expert on all these", thinks it
is "basically equivalent to computing a singular value decomposition", far more expensive than a forward
or backward pass, which are "somehow linear in the number of layers"; and GPUs are built for the matrix
multiplications of forward and backward passes, "not designed to invert matrices or do singular value
decomposition" (≈29:45–31:17).

## Where this leaves deep learning

Both methods come with assumptions about the part of the loss they model, and both cost more per step
than the gradient. Lecture 7's verdict is practical rather than theoretical: there is "a lot of research
trying to make this type of thing practical" (≈28:12), but networks are trained with first-order
methods. The rest of the lecture keeps first-order steps and asks in which norm to measure them (see
[steepest descent](steepest-descent.md) and [scaling rules](scaling-rules.md)).
