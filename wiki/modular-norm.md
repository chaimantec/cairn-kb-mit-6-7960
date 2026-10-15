# Modules and the modular norm

The **modular theory** is Jeremy Bernstein's research programme for building the theory of a network the way the network is built:
give each layer type its own description, including a norm on its weights, and define rules for combining layers so that the
description of any architecture, including a norm on all of its weights (the **modular norm**), follows automatically. The lecturer
presents it as his own research both times, with a warning to be skeptical. [Lecture 7](07-scaling-rules-for-optimization.md)
introduces modules with a norm and leaves open how two modules' norms should combine; [lecture 23](23-metrized-deep-learning.md)
answers with explicit combination rules, extends them to second order, and uses the result for scaling and for optimization.
Covered so far: lecture 7, slides 27–31, ≈1:14:52–1:20:14; lecture 23, slides 2–24 and 29–43, ≈0:00–1:05:37.

**Notation** follows lecture 23's slides. A module $\mathrm{M}$ has input space $\mathcal{X}$, weight space $\mathcal{W}$ and output space
$\mathcal{Y}$, with input $\mathbf{x}$, weights $\mathbf{w}$ and output $\mathbf{y}$; $\Delta \mathbf{x}$, $\Delta \mathbf{w}$ and
$\Delta \mathbf{y}$ are changes to them. Attributes are written as in code: `M.forward`, `M.sensitivity`, `M.mass`, `M.norm`. The slides print
the symbols in plain type; they are bold here, as in the [course notation](notation.md).

## Why build the theory this way

"There's such a zoo of architectures … how can you build an optimization theory that covers all of them? Because someone can always
produce a new architecture" (lecture 7, ≈1:14:52). The answer on lecture 7's slide 28 is to "build the theory with the neural net".
Lecture 23 puts the same idea as LEGO: "if that's how we build neural networks, why don't we also build our theoretical understanding of
them in exactly the same way?" A deep learning library should be "a collection of layers each with its own theory" and "a system of rules
for combining layers", so that you can "build whatever you want!" and have "a theoretical understanding of that thing automatically"
(lecture 23, slides 2 and 6, ≈0:46–1:33 and ≈5:27).

What the theory should describe is **sensitivity**: how much a module's output changes when its inputs or its weights change. That
question underlies optimization, which changes the weights, and robustness, which is about changes to the inputs (lecture 23, slide 4,
≈2:20–3:52).

## Where the norm comes from: steepest descent

Steepest descent picks a step by minimizing the loss's linearization plus a squared norm penalty, so it needs a norm on the weights, and
the right norm is one in which the loss's second-order term is tightly bounded (see [steepest descent](steepest-descent.md)). Lecture 7
asks the question; lecture 23 reduces it. By the Gauss-Newton decomposition and a norm on the network's output, it is enough to find a
norm on the weights in which the network is **Lipschitz** and **Lipschitz smooth** in its weights:
$\Vert \nabla_ {\mathbf{w}} f \thinspace \Delta \mathbf{w} \Vert^{2} \le \delta \Vert \Delta \mathbf{w} \Vert^{2}$ and
$\Vert \Delta \mathbf{w} \thinspace \nabla_ {\mathbf{w}}^{2} f \thinspace \Delta \mathbf{w} \Vert \le \alpha \Vert \Delta \mathbf{w} \Vert^{2}$,
where $f$ is the network (lecture 23, slides 10–12, ≈10:57–16:26). Such a norm would "predict how much the output of the network can change
up to first and second order" whenever the weights change (slide 13, ≈18:46). It can be built in stages: simple norms on the activation
spaces, induced operator norms on the weight matrices, and a rule that combines the layer norms "into one meganorm on the whole neural net"
(slide 14, ≈21:48–23:21). See [second-order methods](second-order-methods.md) and [Lipschitz continuity](lipschitz-continuity.md).

## What a module is

A module "has inputs weights and an output", and "can be an individual layer in a neural network, or it can be a whole neural network"
(lecture 23, ≈24:10); a ReLU is a module with a trivial weight space, and "a full transformer is also a module" (lecture 7, ≈1:16:23).
Lecture 7's slide 29 gives a module three methods: `M.forward`, the function; `M.backward`, its derivatives; and `M.norm`, a norm on the
weights. Lecture 23's slide 20 gives it a different set of four attributes:

1. a function `M.forward`, $\mathcal{X} \times \mathcal{W} \to \mathcal{Y}$;
2. a number `M.sensitivity` $\in \mathbb{R}^{+}$, "one of the Lipschitz constants", the input-output one;
3. a number `M.mass` $\in \mathbb{R}^{+}$, "a kind of hyperparameter which sets how important this module is compared to other modules,
   how much it will contribute to learning";
4. a norm `M.norm`, $\mathcal{W} \to \mathbb{R}^{+}$ (≈28:46).

A module is **well-normed** if its input and output spaces have norms $\Vert \cdot \Vert_ {\mathcal{X}}$ and $\Vert \cdot \Vert_ {\mathcal{Y}}$
and its first derivatives satisfy

$$\Vert \nabla_ {\mathbf{w}} \mathrm{M} \diamond \Delta \mathbf{w} \Vert_ {\mathcal{Y}} \le \mathrm{M.norm}(\Delta \mathbf{w}) \qquad \text{and} \qquad \Vert \nabla_ {\mathbf{x}} \mathrm{M} \diamond \Delta \mathbf{x} \Vert_ {\mathcal{Y}} \le \mathrm{M.sensitivity} \ast \Vert \Delta \mathbf{x} \Vert_ {\mathcal{X}}$$

that is, it is 1-Lipschitz in its weights, measured in its own norm, and Lipschitz in its inputs with its sensitivity as the constant
(slide 20, ≈29:35). The diamond means "contract over any tensor indices which are shared": a dot product for two vectors, a matrix-vector
product for a matrix and a vector (≈33:25). Only well-normed modules go in the library: "if the LEGO brick satisfies these constraints,
we'll put it in our library. And if it doesn't, we'll just throw it out" (≈30:21).

## Atoms, bonds and compounds

Lecture 23's slide 18 sorts modules into **atoms**, with hand-declared attributes (Linear, Conv2d, Embed); **bonds**, with hand-declared
attributes and no weights (ReLU, and "FunctionalAttention", attention without its weights); and **compounds**, built by combination, such as
an MLP, Linear ∘ ReLU ∘ Linear. Lecture 7's slide 29 lists its atomic modules as Linear, Embedding, Conv2D and ReLU.

Two atoms are defined on lecture 23's slide 21 (≈30:21–33:25):

- The **linear module** computes $\mathbf{W} \mathbf{x}$, with sensitivity 1, mass 1 and norm
  $\Vert \cdot \Vert_ {\text{spectral}} \ast \sqrt{\text{fan-in}/\text{fan-out}}$, "this rescaled spectral norm", which is lecture 7's
  RMS-RMS operator norm ("for a linear layer, the good norm is this RMS to RMS operator norm", lecture 7, ≈1:17:55). It is well-normed, with
  the RMS norm on input and output, if its inputs have norm at most 1 and its weights norm at most 1. Hence a reason for normalization
  layers: "if you want to have Lipschitz guarantees for a linear module, the inputs and the weights must be bounded" (≈31:52).
- The **embedding module** also computes $\mathbf{W} \mathbf{x}$, but its input is a one-hot vector that selects a column, so its norm is
  the largest RMS norm of a column, $\max_ i \Vert \text{column}_ i(\mathbf{W}) \Vert_ {\text{RMS}}$, and its input norm is $\ell_ 1$, as for "a
  probability vector" (≈32:38). The slide prints its second condition as "L.norm(W) ≤ 1", a slip for the embedding module.

See [norms](norms.md).

## Combination rules

Lecture 7 composed forwards and backwards (the chain rule) and asked: "How should we combine $\mathrm{M}_ 2$.norm and $\mathrm{M}_ 1$.norm?"
(lecture 7, slide 30, ≈1:18:42). Lecture 23's slide 23 answers, for the **composition** $\mathrm{M} = \mathrm{M}_ 2 \circ \mathrm{M}_ 1$
(modules in series):

- `M.forward = M2.forward ∘ M1.forward`;
- `M.sensitivity = M1.sensitivity * M2.sensitivity`;
- `M.mass = M1.mass + M2.mass`;
- `M.norm = max(p * M1.norm, q * M2.norm)`, with `p = M.mass / M1.mass * M2.sensitivity` and `q = M.mass / M2.mass` (the slide prints no
  brackets in $p$).

For the **concatenation** $\mathrm{M} = (\mathrm{M}_ 1, \mathrm{M}_ 2)$ (modules in parallel, sharing the input) the forwards are paired, the
sensitivities and masses add, and the norm is the same weighted max with `p = M.mass / M1.mass` and `q = M.mass / M2.mass`. "The important
thing is that we use a max norm", weighted by the mass ratios; in a composition the first module's weight also carries the second's
sensitivity, because "if you're going to change the weights of the first one, you hopefully want to know that the second one is really,
really sensitive or it's very insensitive" (lecture 23, ≈37:19–38:04).

The point of the definitions is three propositions (slide 22): combination is associative; it preserves well-normedness, so that "if you
compose two things that are well-normed, then the composition is also well-normed" (≈41:58); and "feature learning is apportioned by mass".
"Come up with really, really clever definitions. And then everything else is easy. So the definitions are kind of the hard part" (≈36:31).
With an addition module and a scalar-multiplication module, composition and concatenation already build a residual block,
`a * Identity + b * M`, and hence a residual network (slide 17, ≈24:56–25:41).

**Mass** is the degree of freedom the theory could not remove. In a transformer with millions of residual blocks but one embedding layer and
one output layer, it is "not obvious" whether the input and output layers should still be trained or "become negligible": the mass lets you
choose (≈38:50–39:37). Only ratios matter (≈39:37–40:24). In the lecturer's experiments the blocks share a total mass $m$, $m/l$ each for
$l$ blocks, and the input and output layers have mass 1 (≈41:12).

## Second order

A module is **$(\alpha, \beta, \gamma)$-sharp** if its second derivatives in the weights, in weights and inputs, and in the inputs are bounded
by $\alpha$, $\beta$ and $\gamma$ times the corresponding norms (lecture 23, slide 24). The three numbers are "generalized top eigenvalues": the
top eigenvalue of the Hessian is "like a Euclidean constant", and "if you don't believe that the geometry of the weight space is a Euclidean
geometry, then you shouldn't necessarily even care about the top eigenvalue" (≈42:44–43:32). The sharpness tuple also combines by rules, so
that neural network losses are "Lipschitz smooth in the modular norm", "with non-dimensional Lipschitz constants", so long as the error
measure is smooth in the module output (slide 24). See [loss landscapes](loss-landscapes.md).

## Using it: scaling

The thesis of lecture 23's slide 29: if every module's Lipschitz constants are independent of width and depth and its bounds stay tight
across scale, then controlling $\mathrm{M.norm}(\Delta \mathbf{w})$ controls $\Vert \Delta \mathbf{y} \Vert_ {\mathcal{Y}}$, and
"if you normalize the updates in the corresponding norm, that will give you good scaling", "a thesis because it's not really a theorem"
(≈45:51–46:41). The lecturer's library **Modula** builds the modular norm automatically as modules are composed and uses it to normalize the
gradient (`mlp.normalize(grad)`, slide 32), and its experiments show the optimal learning rate transferring across width and depth in GPT
training once updates are normalized (slide 34, ≈49:47–52:10). The paper is "Scalable Optimization in the Modular Norm" (Large et al.,
2024), cited on lecture 7's slide 31 and shown on lecture 23's slide 31. See [scaling rules](scaling-rules.md).

## Using it: modular duality

Steepest descent's step is a step size, the dual norm of the gradient over $\lambda$, times a **duality map**, an argmax over unit vectors
(lecture 23, slide 36). Because the gradient is a dual vector, "gradient descent does not type check": the update should be
`weight - LR * dualize(gradient)` (slide 37). **Modular dualization** (slide 39) solves the duality map for each layer and combines the
layer maps recursively over the architecture, as the norms were combined; the lecturer says the recursive rule exists but did not put it on
the slides (≈58:29–59:15). For a linear layer under the spectral norm, the duality map sets the gradient's singular values to one, which is
the core of the Shampoo optimizer and, computed by a Newton-Schulz iteration, of a NanoGPT training-speed record (slides 40–42). The
lecturer concludes that normalizing, as Modula did, "is actually not optimal. We should actually do this dualization procedure", and that
the package still needs updating (slide 43, ≈1:04:51–1:05:37). See [steepest descent](steepest-descent.md).

## Open ends

Lecture 23 is candid about what is unfinished. The boundaries of the well-normed conditions are not enforced in practice, "we just let
everything go outside the boundary", and manifold optimization that keeps weights well-normed is a plan (≈34:10–34:55). The recursive
duality map and dual norm are not written down (≈59:15, ≈1:00:02). And the lecturer would like the same framework to address low-precision
training and adversarial robustness, as "questions of module sensitivity" (slide 47, ≈1:06:26–1:07:15).
