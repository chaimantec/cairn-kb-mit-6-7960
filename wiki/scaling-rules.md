# Scaling rules: width, depth and transferring the learning rate

A **scaling rule** says how to set up a network's initialization and updates so that training keeps
working as the network is made wider or deeper. Without one, a wider network needs its learning rate
retuned and a deeper one trains worse. [Lecture 7](07-scaling-rules-for-optimization.md) presents a
heuristic rule for width, built on the RMS-RMS operator norm, a more tentative one for depth, built on
residual block multipliers, and the lecturer's research programme for a general theory. Problem set 2
implements the width rule. Covered so far: lecture 7, slides 7 and 20–31, ≈7:45–13:55 and
≈50:45–1:20:14; [lecture 13](13-representation-learning-theory.md), ≈1:12:44–1:15:05 (a doubt about initializing at variance
one over fan-in).

Not to be confused with **scaling laws** (see [scaling laws](scaling-laws.md)), which describe how a
model's loss falls as it is given more parameters, data and compute.

**Notation** follows lecture 7. $\mathbf{W}_ \ell$ is the weight matrix of layer $\ell$ in a network of
$L$ layers, and $\Delta \mathbf{W}_ \ell$ an update to it; in the depth section $L$ is the number of
residual blocks. $\Vert \cdot \Vert_{\text{RMS-RMS}}$ is the RMS-RMS operator norm and
$\Vert \cdot \Vert_ \ast$ the spectral norm (see [norms](norms.md)).

## The problem

Slide 7 shows two "nuisances" that appear when a network is scaled naively (≈8:30–10:04). Both of its
plots, screenshots from a post by @kellerjordan0 on X that OCW excludes from its licence, show training
loss against learning rate, one curve per network size from 32 to 1024.

- **Width: the optimal learning rate drifts.** Wider networks reach lower loss, but the learning rate that
  gives the best loss moves as width grows. In practice "you don't run this full sweep of learning rate.
  You just pick a learning rate, and then maybe you try to scale your model … And you find, oh, my
  training isn't working … you have to retune the learning rate at the large scale" (≈9:17–10:04).
- **Depth: deeper performs worse.** The best learning rate stays put, but deeper networks reach a higher
  loss. This "is basically pre, let's say, 2015", when the maximum depth was "let's say, 16 or something";
  techniques since then allow "thousands of layers" (≈11:36).

The practical stake is **hyperparameter transfer**: "if you're trying to scale massive transformers, you
may not even have the resources to tune all the hyperparameters of a really big model. So you may be like
in this regime where you can only tune things at small scale and then try to transfer them" (≈13:07). If
the problems are solved fully, "the optimal learning rate always transfers across scale" (≈12:21). The
drift persisted "even with all the latest fixes prior to 2021", and "if you know all the literature from
2021 until now, then you can fix it for sure" (≈13:07).

## How big should an update be, and in which norm?

The heuristic starts from a "Goldilocks" update size, "not too big, not too small" (slide 21): too big
breaks the network, too small trains slowly. But "always ask: in which norm?" Because a network is built
out of weight matrices, the lecture reaches for a matrix norm, and from the matrix norms on offer it
argues for an **induced operator norm**, which measures how much a matrix can change the size of a vector
passing through it (slides 23–24). A layer's activations are naturally measured in the **RMS norm**, which
is what layer norm enforces in transformers; inducing from it on both sides gives the RMS-RMS operator
norm, equal to $\sqrt{d_{\text{in}} / d_{\text{out}}}$ times the spectral norm. See [norms](norms.md).

The spectral picture behind this (slide 22): write every weight matrix by its singular value
decomposition, and imagine training as changing the singular values. "Probably the singular values should
not change too drastically from step to step. That would be bad. But if they change too little from step
to step, that would be also bad because I would be training too slow" (≈55:26–56:11).

## The width rule

Slide 25's claim: to remove drift in the optimal learning rate as width is varied, for every layer
$\ell = 1, \ldots, L$,

1. initialize weights so that $\Vert \mathbf{W}_ \ell \Vert_{\text{RMS-RMS}} \sim 1$;
2. scale updates so that $\Vert \Delta \mathbf{W}_ \ell \Vert_{\text{RMS-RMS}} \sim 1$.

**Why the initialization.** If the input features are "coordinate wise 1 or on average 1", a matrix of
RMS-RMS norm 1 can "only get out features that are at most coordinate wise 1", so the condition "controls
RMS norms as you move through the network" (≈1:04:01).

**Why the updates.** It bounds "the amount that the activation vectors can change from step to step … in
precisely the same way", which the lecturer calls **feature learning** (≈1:04:01–1:04:46).

**Why this stops the drift.** The RMS norm is "in some sense … a non-dimensional norm": a norm of 1 means
every activation is around 1, whatever the width. Control Euclidean norms instead and, as width changes,
"either the individual coordinates are growing or shrinking". The RMS norm "keeps the size of the
individual coordinates actually invariant to the width, or the amount that the coordinates are changing is
also invariant to the width" (≈1:07:07–1:07:54). So the dimension dependence is packaged "in the right way
into the norm", and once updates are normalized in it, "the learning rate, I don't need to change it with
width anymore". Without that normalization, "the learning rate itself needs to account for the dimension
dependence" (≈1:08:40).

**What it does not say.** It controls the network only at initialization and between consecutive steps;
"over many steps, things can change much more", which is probably why it does not limit expressivity
(≈1:04:46–1:05:34). And it is an upper bound: "it says that the features of the particular layer cannot
change more than by this amount … But it doesn't tell you the features definitely will change by this
amount." Theory that addresses that "usually makes infinite width limits", and the question is "still a
not fully resolved" (≈1:05:34–1:06:19).

**In problem set 2.** Its hyperparameter-transfer questions, which open "We saw in lecture 7 that the
learning rate did not transfer well across architectures of different width if we are not careful",
implement a version of the rule for sign gradient descent: each layer's weights start as a random
semi-orthogonal matrix scaled by $\sqrt{d_k / d_{k-1}}$, and each update is the sign of the gradient,
divided by its spectral norm and multiplied by the same factor and the learning rate. The student tunes
the learning rate at small width and checks whether it transfers to a large one. The questions before it
measure how the spectral norm of random Gaussian and orthogonal matrices scales with size, and estimate a
spectral norm by power iteration.

## Depth: the residual block multiplier

For depth the lecture is openly tentative: "I kind of wimped out a little bit. And also it's like an open
research topic" (≈1:09:26). Slide 26: "the trick seems to be to parameterize your residual block the
'right' way". The analogy is

$$\lim_{L \to \infty} \left(1 + \frac{x}{L}\right)^{L} = \exp(x),$$

a product of many terms scaled so that, with infinitely many of them, it neither blows up nor collapses:
"that kind of well-scaled depth limit" (≈1:10:13–1:11:01). A residual network is a block applied $L$ times
over, so the suggestion is to divide each block by the number of blocks,

$$\mathbf{x} \longrightarrow \mathbf{x} + \frac{1}{L} \operatorname{layer}(\mathbf{x}) \thinspace ?$$

where the block "could be like a little MLP, or it could be an attention layer" (≈1:11:47). The slide adds
"needs more research", and the lecturer notes that "there's different papers saying different things".

The lecturer contrasts it with common practice: "a standard transformer block actually doesn't put a 1
over L multiplier. The standard thing is actually to put 1 over square root L", on the argument that the
blocks are "incoherent and random with respect to each other at initialization", so their sum behaves
like a random walk, which moves "a distance like square root number of time steps". "It's unclear whether
that's the right thing to do or whether just to divide by L is the right thing to do. But it's really a
thing, which is part of a transformer code base, is what block multiplier does your residual block have?"
(≈1:13:20–1:14:05). See [skip connections](skip-connections.md).

## A modular theory

The lecturer's own research, presented with "so be skeptical" (slide 27), aims at a theory that covers
any architecture. "Someone can always produce a new architecture", so instead of a theory per
architecture, "build the theory with the neural net" (slide 28, ≈1:14:52–1:15:37).

A **module** has weights, inputs and outputs, an abstraction that covers anything from a ReLU (with an
empty weight space) to "a full transformer" (≈1:16:23). Each module carries three methods (slide 29):
`M.forward` computes the function, `M.backward` its derivatives, and `M.norm` maps its weights to a
number. A library of **atomic modules** (Linear, Embedding, Conv2D, ReLU) has all three written by hand;
for a linear layer "the good norm is this RMS to RMS operator norm", and a ReLU needs none (≈1:17:55).
**Combination rules** then build bigger modules (slide 30): to compose $M = M_2 \circ M_1$, compose the
forwards and apply the chain rule for the backward. How to combine $M_2$.norm with $M_1$.norm is the open
question (≈1:18:42).

The payoff would be an answer to steepest descent's question of which norm to use: "you can have any
architecture that someone can come to you with, and how are you supposed to give them a norm so that they
can do steepest descent? But what if there was an automatic way?" (≈1:19:28). See [steepest
descent](steepest-descent.md).

## Initializing at one over fan-in (lecture 13)

[Lecture 13](13-representation-learning-theory.md)'s neural network–Gaussian process correspondence assumes weights sampled with
variance one over the fan-in, "what we would call somehow the standard parameterization in PyTorch" (slide 25, ≈1:08:41), and the
lecturer closes the lecture at the board on why that may not be the best choice (≈1:12:44–1:15:05; the deck has no page for it). He
calls it "Xavier initialization"; outside the course material, Xavier, or Glorot, initialization is usually defined with variance
$2 / (\text{fan-in} + \text{fan-out})$, and one over fan-in is usually credited to LeCun.

The initialization is popular because "it's the initialization that is good at initialization": it preserves the magnitude of
activations through random weights. But "if you have a matrix where the fan-in is much larger than the fan-out", it "has a huge null
space, meaning that a lot of the inputs are going to get mapped to 0". Random inputs at initialization mostly fall in that null space,
so preserving their magnitude means scaling up the part that does not. Once training is under way, "the inputs to the layer will kind
of align with the non-null space. And then this principle is kind of bad because then things are much too large" (≈1:13:30–1:14:18).
The lecturer relates this to "maximal update parameterization or MUP, which is something that if you're trying to train giant
networks, people care about this a lot" (≈1:14:18). Lecture 7's width rule, which sizes weights by the RMS-RMS operator norm rather
than by fan-in alone, is the course's other answer to how weights should scale with width.

## Further reading

Slide 31 lists two papers on infinite-width limits, "Feature Learning in Infinite Width Neural Networks"
(Yang & Hu, 2020) and "Infinite Limits of Multi-Head Transformer Dynamics" (Bordelon, Chaudhury &
Pehlevan, 2024); the lecturer's collaborators' "Scalable Optimization in the Modular Norm" (Large et al,
2024), "where we're not trying to do those limits"; and "Universal Majorization-Minimization Algorithms"
(Streeter, 2023) (≈1:20:14).
