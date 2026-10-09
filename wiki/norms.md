# Norms: measuring the size of vectors, matrices and updates

A **norm** assigns a size to a vector or a matrix. The course keeps returning to the choice of one,
because "there's many different possible ways to measure the size of something" (lecture 3,
≈13:57), and the choice changes the answer. [Lecture 3](03-approximation-theory.md) introduces the
RMS norm to define Lipschitz functions of many inputs. [Lecture 6](06-generalization-theory.md) asks
whether the norm of a network's parameters measures its complexity. [Lecture
7](07-scaling-rules-for-optimization.md) makes norms the centre of optimization: the norm chosen for
a step decides the optimization algorithm, and a matrix norm decides how to scale a network. Its
rule of thumb: "Whenever anyone talks about too big or too small, you should always ask in which
norm?" (≈51:32). Covered so far: lecture 3, slide 9, ≈12:20–13:57; lecture 6, slides 33 and 61; lecture 7,
slides 16–19 and 21–25, ≈35:15–1:04:46; lecture 9, slide 10, ≈21:02–22:34; lecture 10, ≈31:06–31:53 (whether the spectral norm can stop an RNN's gradients vanishing).

**Notation.** $\mathbf{v}$ is a vector in $\mathbb{R}^d$ with entries $v_i$; $\mathbf{M}$ is a matrix.
A subscript names the norm.

## Why it matters which norm

Lecture 7's analogy is a map: "If someone just says something's really big, you have no idea what
they're talking about, because you always need some kind of reference, like a scale. And if you look at
a map, there's the scale at the bottom … And for tensors, the notion of scale is what's the norm"
(≈51:32–52:20). So "we want the update to not be too big or too small" is only a meaningful goal once a
norm is chosen (slide 21). And the step that steepest descent takes depends on the norm it is
penalized in, which is why there are as many steepest descent methods as norms (see [steepest
descent](steepest-descent.md)).

## Vector norms

- **Euclidean**, $\Vert \mathbf{v} \Vert_ 2 = \sqrt{\sum_i v_i^2}$. Its balls are circles or spheres
  (lecture 7, slide 17). Penalizing a step in it gives gradient descent.
- **RMS**, the root-mean-square size of the entries,
  $\Vert \mathbf{v} \Vert_{\text{RMS}} = \sqrt{\frac{1}{d} \sum_{i=1}^{d} v_i^2} = \frac{1}{\sqrt{d}} \Vert \mathbf{v} \Vert_ 2$.
  Lecture 3 calls the Euclidean norm "kind of a dimensional object" and the RMS norm "a kind of
  non-dimensional analog": a vector of all ones has RMS norm 1 whatever its length, but Euclidean norm
  $\sqrt{d}$ (≈13:09–13:57). Lecture 7 adds that "if $v$ RMS is 1, it implies that each $v_i$ is around
  1", and that transformers already enforce it on their activations under the names "RMS normalization,
  or layer normalization, or layer norm" (≈1:00:05–1:01:37). See [Lipschitz
  continuity](lipschitz-continuity.md).
- **Infinity (max)**, $\Vert \mathbf{v} \Vert_ \infty = \max_i |v_i|$, the largest coordinate. Its unit
  ball "is more of a kind of square" (lecture 7, slide 18, ≈38:22). Penalizing a step in it gives sign
  gradient descent.
- **$\ell_1$**, the sum of absolute values, appears in that result as $\Vert \mathbf{g} \Vert_ 1$ (slide
  18).
- **$\ell_p$**, of which the Euclidean norm is $p = 2$, offered by the class (≈35:15).
- **Weighted Euclidean**, with a positive constant scaling each coordinate, the lecturer's example of a
  norm that treats directions unequally, like a preconditioner (≈49:12).

The KL divergence, also suggested, "is actually not a norm … It's not symmetric", though it measures a
kind of distance (≈36:03).

## Dual norms

Every norm has a **dual norm**, written $\Vert \cdot \Vert^{\dagger}$ (lecture 7, slide 19). It sets the
step size in the general solution of steepest descent: the step is
$\Vert \mathbf{g} \Vert^{\dagger} / \lambda$ times a unit-norm direction. Problem set 2 defines it as
$\Vert \mathbf{a} \Vert^{\dagger} = \max_{\Vert \mathbf{b} \Vert = 1} \mathbf{a}^{\top} \mathbf{b}$ and asks for the duals of the Euclidean
and infinity norms. The lecture's sign-descent result, a step size proportional to
$\Vert \mathbf{g} \Vert_ 1$ under the infinity norm, is one instance (slides 18–19).

## Matrix norms

"A neural net is built out of weight matrices", so "perhaps we could try a matrix norm?" (lecture 7,
slide 21). "There's actually a lot of ways to measure how big a matrix is, which kind of makes sense
because it's got a lot of different coordinates" (≈53:51). The lecture names:

- the **Frobenius** norm, the one the class offered first ("I think I pretty much knew just the
  Frobenius norm until maybe three years ago", ≈53:06);
- the **spectral** norm, the largest singular value (below);
- the **nuclear** norm;
- the **Schatten $p$-norms**, "a general class of norms".

## Induced operator norms

The **spectral norm** measures a matrix by what it can do to a vector (slide 23):

$$\Vert \mathbf{M} \Vert_ \ast = \max_{\mathbf{v} \neq 0} \frac{\Vert \mathbf{M} \mathbf{v} \Vert_ 2}{\Vert \mathbf{v} \Vert_ 2}.$$

It answers "how much can a matrix scale up the Euclidean norm of a vector?", and it equals the largest
singular value, which the slide states as a fact without proof (≈57:00–58:32). The lecturer likes it
because "when you train a neural network, you're putting vectors into matrices", so it is "descriptive of
what's actually happening during training".

The Euclidean norms on the input and output are "an arbitrary decision". Any norm on the input space and
any norm on the output space **induce** a norm on the matrix, an **induced operator norm** (≈58:32–59:18).
With the RMS norm on both sides, slide 24 defines the **RMS-RMS operator norm**:

$$\Vert \mathbf{M} \Vert_{\text{RMS-RMS}} = \max_{\mathbf{v} \neq 0} \frac{\Vert \mathbf{M} \mathbf{v} \Vert_{\text{RMS}}}{\Vert \mathbf{v} \Vert_{\text{RMS}}}$$

It "constrains how much a matrix can change the RMS norm of its input". As an exercise, it is a rescaled
spectral norm, for a matrix mapping a $d_{\text{in}}$-dimensional input to a $d_{\text{out}}$-dimensional
output:

$$\Vert \mathbf{M} \Vert_{\text{RMS-RMS}} = \sqrt{\frac{d_{\text{in}}}{d_{\text{out}}}} \thinspace \Vert \mathbf{M} \Vert_ \ast.$$

This is the norm lecture 7's scaling recipe uses: initialize every weight matrix, and scale every update,
to have RMS-RMS operator norm about 1 (slide 25). See [scaling rules](scaling-rules.md).

## The norm of a network's parameters

Lecture 6 tries the opposite use of a norm, as a measure of how complex a trained network is. Parameter
count fails, and "parameter norm: maybe": it tracks the double-descent picture better, "but parameter
norm is also not everything" (slide 33, ≈46:27). Weight decay, which shrinks weights toward zero, and
initialization near zero both bias training toward low-norm solutions (slide 61). See [generalization and
double descent](generalization-and-double-descent.md) and [gradient descent](gradient-descent.md).

## Norms for a whole network

Lecture 7 ends with an open question. Linear layers have a good norm (the RMS-RMS operator norm) and a
ReLU has no weights and needs none, but how should the norms of two modules combine when the modules are
composed? The lecturer's research aims to make that automatic, so that building an architecture also
builds the norm its optimizer should use (slides 28–30, ≈1:17:55–1:19:28). See [scaling
rules](scaling-rules.md#a-modular-theory).

## The RMS norm as a normalization layer (lecture 9)

Lecture 9 uses the RMS norm as a layer. RMS-norm divides an activation vector by its RMS norm, which
"will tend to map data points onto the unit hypersphere"; layer norm first subtracts the vector's mean.
"In high dimensions, these things behave almost identically", but in two dimensions RMS-norm maps points
onto a circle and layer norm onto just two points (slide 10, ≈21:02–22:34). See
[normalization layers](normalization-layers.md).

## The spectral norm and recurrence (lecture 10)

In [lecture 10](10-architectures-memory.md) a student asked whether the spectral norm of lecture 7 would cure a recurrent network's vanishing and exploding
gradients. Normalizing the weights could keep them near 1, but the recurrence raises the weight matrix to a power: "You
could take a spectral norm over that exponential, but you're still going to take the exponential first. So that might help
you with the exploding gradients problem, but it's not going to change the fact that the small things are going to go to 0"
(≈31:06–31:53). See [recurrent neural networks](recurrent-neural-networks.md).
