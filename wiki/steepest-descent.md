# Steepest descent: choosing a norm chooses the algorithm

**Steepest descent** derives an optimization step by replacing everything the loss does beyond its
linear term with a squared norm penalty on the step. Which norm you pick decides which algorithm
comes out: the Euclidean norm gives ordinary gradient descent, the infinity norm gives sign gradient
descent, and every norm has a solution with the same two-part shape, a step size times a step
direction. The course develops it in [lecture 7](07-scaling-rules-for-optimization.md) as one of
three classical methods that start from a Taylor expansion, and problem set 2 works out the details.
Covered so far: lecture 7, slides 16–19, ≈33:37–50:45; [lecture 11](11-representation-learning-reconstruction-based.md), slide 12, ≈15:30–16:17 (SGD against steepest
descent in the spectral norm on a small MLP).

**Notation** follows lecture 7. $\mathcal{L}(\mathbf{w})$ is the loss as a function of the weights
$\mathbf{w}$ (the [course notation](notation.md)'s total cost $J$),
$\mathbf{g} = \partial \mathcal{L} / \partial \mathbf{w}$ its gradient, $\Delta \mathbf{w}$ a step, and $\lambda \gt 0$ a
number. $\Vert \cdot \Vert$ is a norm; [norms](norms.md) collects the ones used here.

## The model

Every classical method in lecture 7 starts by Taylor expanding the loss around the current weights
(slide 10):

$$\mathcal{L}(\mathbf{w} + \Delta \mathbf{w}) = \mathcal{L}(\mathbf{w}) + \mathbf{g}^{T} \Delta \mathbf{w} + \frac{1}{2} \Delta \mathbf{w}^{T} \mathbf{H} \thinspace \Delta \mathbf{w} + \cdots$$

The first two terms are the **linearization**; the rest, starting with the Hessian $\mathbf{H}$, is the
**non-linear part**. Newton's method keeps the Hessian term (see [second-order
methods](second-order-methods.md)). Steepest descent throws the whole non-linear part away and puts a
norm penalty in its place (slide 16):

$$\mathcal{L}(\mathbf{w} + \Delta \mathbf{w}) \approx \mathcal{L}(\mathbf{w}) + \mathbf{g}^{T} \Delta \mathbf{w} + \frac{\lambda}{2} \Vert \Delta \mathbf{w} \Vert^2.$$

The step is the $\Delta \mathbf{w}$ that minimizes this model's right-hand side. The lecturer is frank
about the move: "I don't even know what that nonlinear part is. Let me just replace it with something
which I know very well. And actually, a lot of optimization methods actually have that flavor"
(≈33:37–34:22). Since there are many norms (the class suggested Euclidean, Frobenius, infinity and
$\ell_p$), "there's a lot of steepest descent methods because there's a lot of norms" (≈35:15–36:03).
A student's suggestion of the KL divergence was set aside: it is a measure of distance, but "it's
actually not a norm … It's not symmetric".

## Euclidean norm: gradient descent

With the squared Euclidean norm (slide 17), differentiating and setting the derivative to zero gives
$\mathbf{g} + \lambda \Delta \mathbf{w} = 0$, so

$$\Delta \mathbf{w} = -\frac{1}{\lambda} \mathbf{g}.$$

This is "vanilla gradient descent!" with learning rate $1/\lambda$: a heavier penalty means a smaller
step. "It's kind of interesting that there's some connection between regular old gradient descent and
L2 geometry on your optimization space" (≈37:35–38:22). See [gradient descent](gradient-descent.md).

## Infinity norm: sign gradient descent

The infinity norm is the largest absolute coordinate,
$\Vert \Delta \mathbf{w} \Vert_ \infty = \max_i |\Delta w_i|$, and its balls are squares rather than circles (slide 18). The minimization is
"not quite as simple as just differentiating" because of the max, and is left to the homework; the
result is

$$\Delta \mathbf{w} = -\frac{\Vert \mathbf{g} \Vert_ 1}{\lambda} \operatorname{sign}(\mathbf{g}),$$

the $\ell_1$ norm of the gradient over $\lambda$, times the coordinate-wise sign of the gradient, $+1$
or $-1$ (≈39:07–39:53). This is **sign gradient descent**, and every weight moves by the same amount:
"the sign of a vector maps a vector to another vector, where all the components have the same
magnitude" (≈45:18–46:05). Problem set 2 has an optional question arguing that Adam, with its
hyperparameters $\beta_1 = \beta_2 = \epsilon = 0$, takes the same direction.

## Any norm: step size times step direction

For a general norm, the minimization has a "dual formulation" (slide 19):

$$\underset{\Delta \mathbf{w}}{\operatorname{argmin}} \left[ \mathbf{g}^{T} \Delta \mathbf{w} + \frac{\lambda}{2} \Vert \Delta \mathbf{w} \Vert^2 \right] \equiv \frac{\Vert \mathbf{g} \Vert^{\dagger}}{\lambda} \thinspace \underset{\mathbf{t} : \Vert \mathbf{t} \Vert = 1}{\operatorname{argmax}} \thinspace \mathbf{g}^{T} \mathbf{t}$$

Here $\Vert \cdot \Vert^{\dagger}$ is the **dual norm** of $\Vert \cdot \Vert$. The fraction is the
**step size** and the argmax over unit vectors $\mathbf{t}$ the **step direction**, so the solution
separates into "computing the step size" and "computing the direction" (≈40:40–41:27). Problem set 2
states the same identity with a minus sign in front of the right-hand side, which the slide does not
print, and defines the dual norm as
$\Vert \mathbf{a} \Vert^{\dagger} = \max_{\Vert \mathbf{b} \Vert = 1} \mathbf{a}^{\top} \mathbf{b}$. Its questions derive the dual norms of the Euclidean and infinity norms
and recover the two cases above from this formula.

The same recipe works for a weight matrix, with the Frobenius inner product
$\operatorname{trace}(\mathbf{G}^{\top} \Delta \mathbf{W})$ in place of
$\mathbf{g}^{T} \Delta \mathbf{w}$ and a matrix norm as the penalty. Problem set 2 asks for steepest descent under the
**spectral norm**, expressed through the singular value decomposition of the gradient matrix
$\mathbf{G}$. Lecture 7 does not solve it; it argues for matrix norms on other grounds (see [scaling
rules](scaling-rules.md)).

## Does the model hold?

No guarantee comes with it. "I said take your loss function, throw away the nonlinear part, and just
replace it with your favorite norm. So that doesn't sound like the kind of thing that's going to give
you a guarantee. It sounds kind of random. But it's a framework for thinking about deriving different
optimization algorithms" (≈42:14). A larger $\lambda$ keeps the step short, so the linear term is more
likely to be accurate; but "you wouldn't know how big you need to make lambda to be for that to work"
(≈43:00).

The route to a guarantee is to choose $\lambda$ and the norm so that the model is a genuine **upper
bound** on the loss, so that minimizing it can only help (≈43:46–44:33). Problem set 2's bonus question
does this for the square loss of a linear predictor, and the lecturer's research asks the same of
two-layer and five-layer networks: "that's essentially what I'm trying to do in my research at the
moment" (≈44:33).

## Why not always the Euclidean norm?

Going straight down the gradient feels natural, but "implicitly, when you say the best direction is the
gradient direction … you're thinking Euclidean" (≈47:38). The lecturer's picture is a map of the US
stretched in Photoshop, so that a centimetre east-west is thousands of miles and a centimetre
north-south is ten. Going along the gradient assumes "a very isotropic map". A network's
weight space "may not have that property that different directions in the weight space are equal to
each other — may be a very non-isotropic space", "because of the architecture of the network and so on"
(≈47:38–48:25).

Two ways to read the choice of norm:

- **As a reweighting of directions.** A Euclidean norm with a different positive constant on each
  coordinate gives a steepest descent that reshapes the gradient, as a preconditioner would. That is
  "one example"; the infinity and Frobenius norms are others, all "saying that different coordinate
  directions or different directions more generally have different weightings" (≈49:12–49:58).
- **As how far the linear term can be trusted.** "If I move too far in that direction, the linear piece
  broke down. And it may break down at a different rate going in different directions. So the idea of
  building a norm is to capture at what rate does the linear piece break down if I move in this
  direction or this direction" (≈49:58).

Choosing the norm for a whole network is the open problem behind the lecturer's modular theory, where
each layer type carries its own norm and composing layers is meant to compose their norms (lecture 7,
slides 28–30); see [scaling rules](scaling-rules.md#a-modular-theory).

## SGD against spectral descent, on a small MLP (lecture 11)

Lecture 11 trains the same width-2 MLP twice, "just for fun", with SGD (learning rate 0.01) and with "the spectral descent method
that you worked out in your problem set 2", steepest descent in the spectral norm (learning rate 0.002), where "you are normalizing
the updates by their spectral norm" (slide 12, ≈15:30). "It's a little bit hard to read too much into this. Well, the spectral
descent is descending faster, but that depends on the learning rate, so there's some caveats. But it is getting to a lower loss",
separating the two classes better: "different optimization schemes actually do perform differently". The lecturer notices that the
first linear layer looks "almost like a rotation, which is a orthogonal transformation. And if your updates are orthogonalized, then
maybe that somehow relates to this first layer finding this orthogonal transformation … it's a little complicated" (≈15:30–16:17).
Slide 12 links the Colab code that made the figures.
