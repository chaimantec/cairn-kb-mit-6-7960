# Lecture 23 — Metrized Deep Learning

**Lecturer:** Jeremy Bernstein ·
**Video:** [youtube.com/watch?v=zBvsoxC6tAo](https://www.youtube.com/watch?v=zBvsoxC6tAo) (68 min) ·
**Slides:** [`mit6_7960_f24_lec23.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec23.pdf)
(49 pages, typed; transcribed slide by slide in [`raw/slides/23-metrized-deep-learning.md`](../raw/slides/23-metrized-deep-learning.md)) ·
**Transcript:** [`raw/transcripts/23-metrized-deep-learning.md`](../raw/transcripts/23-metrized-deep-learning.md)

**Slide numbers on this page are PDF page numbers**, which is also the small number printed at the bottom centre of every page.
Thirty pages carry a second, larger number in the bottom-right corner, left over from an earlier talk the deck was built from (slide 16
prints 18 there, slide 45 prints 50); it is not the slide number.

## What this lecture establishes

The lecturer presents "some of my research", with a warning: "you don't have to fully trust me in this lecture. And maybe you should be
skeptical" (≈0:00). The idea is that "the way that we build neural networks is kind of like a LEGO set" (slide 2), so "why don't we also
build our theoretical understanding of them in exactly the same way?" A theory of deep learning that "really applies to any neural network
that you actually find in the wild" had better be built the way the networks are, from a characterization of each piece and rules for what
happens when pieces are combined (≈0:46–1:33). The property of a piece that matters is its **sensitivity**: how much a layer's output
changes when its inputs or its weights change. That is the question behind optimization, since "training a neural network involves changing
the weights", and behind robustness to adversarial examples, which change the inputs; "no one understands this. It's a really basic question
about neural networks, but no one teaches you about it" (slide 4, ≈3:06–3:52). The lecture does not gloss its title; its subject throughout
is how to measure size and distance in a network's weight space, that is, which norm to give it.

The lecture has four parts. **Part I** returns to the steepest descent of [lecture 7](07-scaling-rules-for-optimization.md), which needs a
norm on the weights in which the loss's second-order term is tightly bounded, and reduces that need, through the Gauss-Newton decomposition,
to two conditions on the network itself: that it be Lipschitz and Lipschitz smooth in its weights, in that norm (slides 9–13). Such a norm
can be built layer by layer and combined (slide 14). **Part II** makes this a formal **theory of modules**: a module is anything with
inputs, weights and an output (a layer, or a whole network), carrying a forward function, a sensitivity, a mass and a norm; a module is
**well-normed** if it is Lipschitz in its weights and inputs in those norms; and rules for composing modules in series and in parallel
are defined so that well-normed pieces make a well-normed whole (slides 16–24). **Part III** applies the theory to scaling: the thesis
that tight Lipschitz constants that do not change with scale give learning rates that transfer across width and depth, the lecturer's
library Modula, and experiments on GPT (slides 26–34). **Part IV**, **modular duality**, argues that the gradient is a different kind of
object from the weights, so that "gradient descent does not type check", and that a gradient must be turned into a weight update by a
**duality map**, computed layer by layer and combined over the architecture; for a linear layer under the spectral norm that map is the
core of the Shampoo optimizer and of a NanoGPT training-speed record (slides 36–43). The conclusion proposes that low precision and
adversarial robustness are also "questions of module sensitivity" (slide 47).

**Notation on this page** follows the slides. $\mathcal{L}(\mathbf{w})$ is the loss as a function of the weights $\mathbf{w}$, with
$\nabla_ {\mathbf{w}} \mathcal{L}$ its gradient and $\nabla_ {\mathbf{w}}^{2} \mathcal{L}$ its Hessian; $\Delta \mathbf{w}$ is a change of
the weights; $\lambda$ is a **sharpness** constant; $\Vert \cdot \Vert$ is a norm. The loss is a composite of an **error measure** $\ell$
and the **network** $f$, so $\mathcal{L} = \ell \circ f$ (slide 10), and $f(\mathbf{x}; \mathbf{w})$ is the network's output on input
$\mathbf{x}$ (slide 13). In Part II a module $\mathrm{M}$ has input space $\mathcal{X}$, weight space $\mathcal{W}$ and output space
$\mathcal{Y}$, with input $\mathbf{x}$, weights $\mathbf{w}$ and output $\mathbf{y}$, and attributes written as in code, `M.forward`,
`M.sensitivity`, `M.mass` and `M.norm` (slide 20). The slides print $\mathbf{w}$, $\mathbf{x}$ and $\mathbf{y}$ in plain type; they are
bold here, as in the [course notation](notation.md). $\mathbf{W}$ is a weight matrix and $\mathbf{G}$ its gradient.

## Where this lecture sits

This is the lecturer's third lecture. [Lecture 3](03-approximation-theory.md) introduced [Lipschitz continuity](lipschitz-continuity.md)
and the RMS norm. [Lecture 7](07-scaling-rules-for-optimization.md) derived [steepest descent](steepest-descent.md) from a Taylor
expansion with a norm in place of the non-linear part, introduced the Gauss-Newton decomposition (see [second-order
methods](second-order-methods.md)), measured weight matrices in the RMS-RMS operator norm to fix how the learning rate drifts with width
(see [scaling rules](scaling-rules.md)), and ended on the lecturer's modular theory with an open question: how to combine the norms of two
modules when they are composed. Homework 2 asked for steepest descent under the spectral norm, and [lecture
13](13-representation-learning-theory.md) opened with its answer and the Muon optimizer built on it. This lecture answers lecture 7's open
question with explicit combination rules, extends the theory to second order, and connects it to scaling and to the optimizer, in the
spirit of "building neural networks like a LEGO set" (≈8:36). See [modules and the modular norm](modular-norm.md) for the theory across
lectures 7 and 23, and [norms](norms.md) for the norms it uses.

## Build the theory the way you build the network (slides 2–7)

Slide 2 shows LEGO bricks labelled with the components of a transformer, loose and then assembled into its tower: "we have all these
little pieces, all these building blocks, and we piece them together, and we can build a transformer, for example. But we could take those
pieces, and we could build something else" (≈0:46). If the theory is to describe such networks it should be built the same way. Imagine "a
full theoretical characterization" of a single piece, "which is maybe reasonable because it's a simple little piece", and then ask what
happens under the operations that combine pieces: a **series combination**, "putting one on top of the other", and a **parallel
combination**, "putting them side by side" (slide 3, ≈1:33–2:20).

![Slide 2: two photographs of LEGO bricks stickered with transformer labels such as Multi-Head Attention, Add & Norm and Positional Encoding, scattered on a table at left and built into one tall tower at right](../raw/images/23-metrized-deep-learning/slide-2.jpg)

*Slide 2 — LEGO bricks stickered with the parts of a transformer diagram, loose (left) and assembled into the tower (right): the lecture's picture of building the theory the way the networks are built. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec23.pdf)*

What should a piece's characterization say? A layer has an input $\mathbf{x}$, weights $\mathbf{w}$ and an output $\mathbf{y}$, and "the
thing that we're really going to care about is how sensitive is the output of the layer ... to both changing the input ... or what if we
perturb the weights?" (slide 4, ≈2:20–3:06). The change in output "gets contributions both from the change in the inputs and the change in
the weights and potentially the interactions of those two things". Slide 4 illustrates the two reasons to care with a loss surface, for
optimization, and an adversarial example, for robustness: a photograph of a pig captioned "pig", plus 0.005 times a square of noise,
equals the same-looking photograph captioned "airliner". "People talk about adversarial examples, where you change the input to the
network and the output changes dramatically. So it's another question of understanding the sensitivity of the network" (≈3:52; see
[adversarial examples](adversarial-examples.md)).

The question then is whether the sensitivity of single pieces determines the sensitivity of combinations (slide 5). In series each piece
has its own weight space, "if we combine them, we're somehow combining the weight space"; in parallel the input is duplicated, "sending it
to both of the pieces, and then we get two outputs now", and "the series combination is somehow a bit more coupled than the parallel
combination" (≈3:52–4:39). If this can be done, "you should really think of" a library like PyTorch or JAX "like a LEGO set, where you've
got all these layers, and each layer should have its own theoretical description. And then we should have a system of rules for if you
combine layers, the new object that you get has these properties" (slide 6, ≈5:27). Then "you can go and take this library and build
whatever fanciful creation you like ... And you should have a theoretical understanding of that thing automatically."

Slide 7 gives "the practical payoff ... so far". The first is fixing scaling issues, the drift of the optimal learning rate with width,
which "we talked a bit about ... in one of the earlier lectures in the class" (≈6:15; lecture 7). The second is speed: a chart of NanoGPT
training runs, from posts by @kellerjordan0, compares the "llm.c baseline", "like Andrej Karpathy's extremely high-performance C
implementation", with newer records that use "steepest descent algorithms based on norms and spectral descent-type things", and shows "a
huge wall clock speedup in training" (≈7:02–7:49). The chart, which OCW excludes from its licence here, is the one lecture 13 pastes into its opening
aside: on it the baseline reaches the target validation loss of about 3.28 after about 42 minutes, and the "+Distributed Muon" run after
about 13. Asked whether anyone knows why it is so much faster, the lecturer said "we partly understand why": the faster curves use "steepest
descent under the spectral norm, which was in the homework problem. And if you believe that the geometry of an individual linear layer is
somehow described by the spectral norm, then that would explain why" (≈7:49–8:36).

## Part I — Optimization theory: what norm for the weights? (slides 9–14)

### Steepest descent, recalled (slide 9)

Slide 9 recalls lecture 7's starting point. Taylor expand the loss in the weights:

$$\mathcal{L}(\mathbf{w} + \Delta \mathbf{w}) = \mathcal{L}(\mathbf{w}) + \nabla_ {\mathbf{w}} \mathcal{L}^{\top} \Delta \mathbf{w} + \tfrac{1}{2} \Delta \mathbf{w}^{\top} \nabla_ {\mathbf{w}}^{2} \mathcal{L} \thinspace \Delta \mathbf{w} + \cdots$$

and ask whether the second-order term can be bounded by a squared norm, so that

$$\mathcal{L}(\mathbf{w} + \Delta \mathbf{w}) \le \mathcal{L}(\mathbf{w}) + \nabla_ {\mathbf{w}} \mathcal{L}^{\top} \Delta \mathbf{w} + \tfrac{1}{2} \lambda \Vert \Delta \mathbf{w} \Vert^{2} + \cdots$$

The slide's question: "can we find a **norm** $\Vert \cdot \Vert$ and a **sharpness** $\lambda$ to make this inequality hold tightly?" If
so, an optimization step can be chosen by minimizing the right-hand side,
$\arg\min_ {\Delta \mathbf{w}} \thinspace \nabla_ {\mathbf{w}} \mathcal{L}^{\top} \Delta \mathbf{w} + \tfrac{1}{2} \lambda \Vert \Delta \mathbf{w} \Vert^{2}$,
which is steepest descent in that norm; "if I pick this norm, I get this optimization algorithm. If I pick this other norm, I get this
other optimization algorithm" (≈9:22–10:57). A tight bound would be worth more than an algorithm, because "we can actually think of using
this bound as an actual description of the loss function" (≈10:09). So: "What is a good norm for a neural network? Why would we favor one
norm over a different norm? I think that's an important question" (≈10:57).

### Three steps from the loss to the network (slides 10–12)

The task is to bound $\Delta \mathbf{w}^{\top} \nabla_ {\mathbf{w}}^{2} \mathcal{L} \thinspace \Delta \mathbf{w} \le \lambda \Vert \Delta \mathbf{w} \Vert^{2}$
(slides 10–12, each headed "We need to bound").

**Step 1** (slide 10). In deep learning the loss is a **composite**, $\mathcal{L}(\mathbf{w}) = \ell \circ f(\mathbf{w})$, an error measure
$\ell$ applied to the network $f$, and "whenever you have a composite loss function where both parts of it are suitably differentiable"
the second-order term splits by the **Gauss-Newton decomposition** (≈10:57–11:44):

$$\Delta \mathbf{w}^{\top} \nabla_ {\mathbf{w}}^{2} \mathcal{L} \thinspace \Delta \mathbf{w} = \Delta \mathbf{w} \thinspace \nabla_ {\mathbf{w}}^{2} f \thinspace \Delta \mathbf{w} \thinspace \nabla_ {f} \ell + \Delta \mathbf{w} \thinspace \nabla_ {\mathbf{w}} f \thinspace \nabla_ {f}^{2} \ell \thinspace \nabla_ {\mathbf{w}} f \thinspace \Delta \mathbf{w}$$

"The first one involves the curvature of the neural network, and the second piece involves the curvature of the error measure" (≈11:44).
The slide writes the products on the right without transpose signs, the factors simply side by side. This is lecture 7's decomposition
of the Hessian, applied to a step; see [second-order methods](second-order-methods.md#the-gauss-newton-decomposition).

**Step 2** (slide 11). "Now let's suppose that we know a really good norm on the output of the network". That is reasonable, because the
output is usually "a simple thing like a vector ... or it outputs a real number", for which there are few candidates: "maybe you give the
output the infinity norm, the L2 norm, the RMS norm, the L1 norm. There's not many other obvious choices" (≈11:44–13:20). With such a norm
both terms can be bounded:

$$\Delta \mathbf{w}^{\top} \nabla_ {\mathbf{w}}^{2} \mathcal{L} \thinspace \Delta \mathbf{w} \le \Vert \Delta \mathbf{w} \thinspace \nabla_ {\mathbf{w}}^{2} f \thinspace \Delta \mathbf{w} \Vert \thinspace \Vert \nabla_ {f} \ell \Vert + \Vert \nabla_ {f}^{2} \ell \Vert \thinspace \Vert \nabla_ {\mathbf{w}} f \thinspace \Delta \mathbf{w} \Vert^{2}$$

The first term "as being a dot product between the curvature of the model and the linearization of the error measure", bounded by the
Cauchy-Schwarz inequality, so that the norm on the error measure's derivative is "a dual norm"; the second as "a vector times a matrix
times a vector", bounded with an operator norm (the slide labels $\Vert \nabla_ {f} \ell \Vert$ "dual norm" and
$\Vert \nabla_ {f}^{2} \ell \Vert$ "operator norm"). "So no approximations, just an upper bound" (≈13:20–14:54).

**Step 3** (slide 12). "Therefore, our problem reduces to the following": can we produce a norm on the network's weights such that

$$\Vert \Delta \mathbf{w} \thinspace \nabla_ {\mathbf{w}}^{2} f \thinspace \Delta \mathbf{w} \Vert \le \alpha \Vert \Delta \mathbf{w} \Vert^{2} \qquad \text{and} \qquad \Vert \nabla_ {\mathbf{w}} f \thinspace \Delta \mathbf{w} \Vert^{2} \le \delta \Vert \Delta \mathbf{w} \Vert^{2}$$

for constants $\alpha$ and $\delta$? The first says the network is "Lipschitz smooth" in its weights and the second that it is "Lipschitz"
(continuous) in its weights; plugged into step 2, they bound the curvature of the whole loss (≈14:54–16:26). "So we've kind of reduced the
problem which we didn't really know how to solve now into at least a different problem."

A student asked whether being Lipschitz and Lipschitz smooth guarantees such constants exist. "You could take these actually as the
definitions of Lipschitz smooth and Lipschitz continuous and once you have them in one norm, you can automatically produce them in any
other norm because all the norms are equivalent but it may be a much tighter guarantee in one norm than another norm. So we really want to
find a norm on the weight space such that these conditions hold very, very tightly" (≈17:12).

### What the conditions mean (slide 13)

Forget the optimization argument and Taylor expand the network itself in its weights (slide 13):

$$f(\mathbf{x}; \mathbf{w} + \Delta \mathbf{w}) = f(\mathbf{x}; \mathbf{w}) + \nabla_ {\mathbf{w}} f \thinspace \Delta \mathbf{w} + \Delta \mathbf{w} \thinspace \nabla_ {\mathbf{w}}^{2} f \thinspace \Delta \mathbf{w} + \ldots$$

(printed without a factor of one half). The two conditions bound the first- and second-order terms in a chosen norm: "We seek a weight norm
that controls the network's Taylor expansion". "Wouldn't it be great if we had a norm on the weights that could predict the magnitude of
these terms in the Taylor expansion? If we had such a norm, it would mean that whenever you were going to change the weights of your
neural network, you would have a direct way to predict how much the output of the network can change up to first and second order"
(≈18:00–18:46). Optimization is one use; "they're useful anytime you want to be able to just predict what changing the weights of your
network is going to do to the overall function" (≈18:46–19:32). The difficulty: "how on Earth would we produce a norm ... because it seems
that neural networks are really complicated" (≈19:32).

![Slide 13: the Lipschitz-smooth and Lipschitz conditions in coloured boxes above a fully connected network with input x and output f(x;w), and the network's Taylor expansion in its weights below, its first- and second-order terms boxed](../raw/images/23-metrized-deep-learning/slide-13.png)

*Slide 13 — the two conditions of slide 12 above a fully connected network, and the network's Taylor expansion in its weights: the boxed first- and second-order terms are what a good weight norm should bound. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec23.pdf)*

A student asked whether this approach has worked for other problems. "A lot of classical optimization, once you start to get into it, is
all about finding a distance measure for your problem. And once you've got a really good distance measure, you can start to design clever
algorithms." People have the idea in deep learning, he said, "but it's not really borne fruit", and he pointed to "composite optimization
theory" (≈20:17). "That's why we have different norms. Why do we have the L1 norm and L2 norm? Because somehow, they're supposed to
characterize the structure of different problems. If they didn't, we would just only have one of them" (≈21:03).

### Inducing a norm layer by layer (slide 14)

This is "where the whole LEGO thing comes back again" (≈21:03). Slide 14 draws a fully connected network with three tiers of norm boxes
above it. First, the **activation spaces**, the feature vectors between layers, get simple norms, "because there's not that many options,
like L2, L infinity, L1, RMS". Next the **tensor spaces**, such as the weight matrices: "once we've picked norms on the activation spaces,
there's actually a way to induce something called an induced operator norm", "a procedure for taking two norms on the input and output of
a linear operator and coming up with a norm on the operator" (≈21:48–22:34; see [norms](norms.md#induced-operator-norms)). On the slide,
arrows join each pair of neighbouring activation-space norms to the tensor-space norm between them. Finally, all the layer norms are
combined into a norm on the **full weight space**: "we really need a procedure for taking all of these layer-wise norms and combining them
into one meganorm on the whole neural net" (≈22:34–23:21).

![Slide 14: a fully connected network overlaid with five blue norm boxes for the activation spaces, four green norm boxes for the tensor spaces between them, magenta arrows combining them into one red norm box, and a label Full weight space](../raw/images/23-metrized-deep-learning/slide-14.png)

*Slide 14 — norms on the activation spaces (blue) induce norms on the weight tensors between them (green), which combine into one norm on the full weight space (red). [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec23.pdf)*

A student pointed out that activation functions have no weights. A nonlinearity is still a LEGO block: "it will just be a LEGO block with
... the weight space is just the 0 vector space, or it just has a trivial weight space" (≈22:34).

## Part II — The theory of modules (slides 16–24)

### Modules and combinators (slides 16–18)

The formal version borrows the **combinator pattern** from programming, quoted on slide 16 from the Haskell Wiki: "complex structures are
built by defining a small set of very simple "primitives", and a set of "combinators" for combining them into more complicated
structures". "There's an abstract type, and you can build very simple things which are members of that type, and then you combine them and
you get new things which belong to that type" (≈24:10). Here the type is the **module**, which "has inputs weights and an output", and "can
be an individual layer in a neural network, or it can be a whole neural network". Two modules $\mathrm{M}_ 1$ and $\mathrm{M}_ 2$ combine by
**composition**, $\mathrm{M}_ 2 \circ \mathrm{M}_ 1$, modules in series, and by **concatenation**, $(\mathrm{M}_ 1, \mathrm{M}_ 2)$,
modules in parallel, which share an input and produce a pair of outputs (≈24:10–24:56).

![Slide 16: the Haskell Wiki quotation on combinators, a module box with input x, weights w and output y, and two diagrams, composition of M1 and M2 in series and concatenation of M1 and M2 in parallel](../raw/images/23-metrized-deep-learning/slide-16.png)

*Slide 16 — a module has an input, weights and an output; two modules combine by composition (in series) or concatenation (in parallel). [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec23.pdf)*

Those two rules are enough for real architectures (slide 17). Adding two modules is a concatenation followed by an "Add" module; multiplying
a module by a scalar $a$ is a composition with a "Mul" module; these are "special "utility modules"". Together they build a **residual
block**, `a * Identity + b * M`, and "you can think about taking that little circuit and composing it with itself a bunch of times to build
a residual network". "Those simple kind of binary combination rules are actually enough to build all the neural networks that you want to
build" (≈24:56–25:41; see [skip connections](skip-connections.md)). Asked about modules with two inputs: "a module having two inputs we
would think of as the input as being a single tuple. So the input space can have whatever structure you want ... it just has to be a vector
space, but in the abstract sense" (≈26:28).

![Slide 17: circuit diagrams for adding two modules through an Add box, multiplying a module by a scalar through a Mul box, and a residual block whose identity and module branches are scaled by a and b and added](../raw/images/23-metrized-deep-learning/slide-17.png)

*Slide 17 — addition, multiplication by a scalar, and a residual block, all built from composition and concatenation with the utility modules Add and Mul. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec23.pdf)*

Slide 18 names three kinds of module. **Atoms** have hand-declared attributes, "the simple layers that you write by hand, like a linear
layer, a convolution layer, an embedding layer". **Bonds** have hand-declared attributes and no weights, like ReLU, and
"FunctionalAttention", "the part of attention without the weights", which "you could compose ... with some linear modules" to build an
attention module. **Compounds** are combinations of atoms and bonds, such as an MLP, Linear ∘ ReLU ∘ Linear (≈26:28–27:14).

### Sensitivity (slide 19)

The goal, stated on slide 19, is to predict the size of $\Delta \mathbf{y}$ from the size of $\Delta \mathbf{x}$ (perturbing the inputs),
and from the size of $\Delta \mathbf{w}$ (perturbing the weights), "for any module". "If we can do this for atoms and bonds, what about
compounds?" "Can we do it for any compound module or anything built from our LEGO set is the analogy we were making" (≈27:14–28:00).

![Slide 19: three module diagrams, unperturbed, with the input perturbed by delta x, and with the weights perturbed by delta w, each changing the output by delta y, above the goal of predicting the size of delta y for any module](../raw/images/23-metrized-deep-learning/slide-19.png)

*Slide 19 — perturb a module's inputs or its weights and its output changes; the goal is to predict the size of that change for any module. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec23.pdf)*

### Modules and well-normed modules (slide 20)

The definitions come from work with "a pure mathematician, basically. So it's quite an abstract or formal way of going about solving this
problem" (≈28:00). A **module** $\mathrm{M}$ has three vector spaces, the input space $\mathcal{X}$, the weight space $\mathcal{W}$ and the
output space $\mathcal{Y}$, and four attributes (slide 20):

1. a function `M.forward`, $\mathcal{X} \times \mathcal{W} \to \mathcal{Y}$;
2. a number `M.sensitivity` $\in \mathbb{R}^{+}$, "supposed to measure the input-output sensitivity. So it's one of the Lipschitz constants";
3. a number `M.mass` $\in \mathbb{R}^{+}$, "a kind of hyperparameter which sets how important this module is compared to other modules, how
   much it will contribute to learning";
4. a norm `M.norm`, $\mathcal{W} \to \mathbb{R}^{+}$, on the weight space (≈28:46).

"Anything with these attributes is a valid module", but the library should hold only "nice functions", hence a second definition (≈28:46).
A module is **well-normed** if its input space has a norm $\Vert \cdot \Vert_ {\mathcal{X}}$, its output space a norm
$\Vert \cdot \Vert_ {\mathcal{Y}}$, and its first derivatives satisfy

$$\Vert \nabla_ {\mathbf{w}} \mathrm{M} \diamond \Delta \mathbf{w} \Vert_ {\mathcal{Y}} \le \mathrm{M.norm}(\Delta \mathbf{w}) \qquad \text{and} \qquad \Vert \nabla_ {\mathbf{x}} \mathrm{M} \diamond \Delta \mathbf{x} \Vert_ {\mathcal{Y}} \le \mathrm{M.sensitivity} \ast \Vert \Delta \mathbf{x} \Vert_ {\mathcal{X}}$$

That is, "it's one Lipschitz in its weights" and "Lipschitz in its inputs where the sensitivity number assigned to the module is Lipschitz
constant" (≈29:35). The diamond, a student asked about, means "contract over any tensor indices which are shared ... If there are two
vectors, it's a dot product. If it's a matrix and a vector, it's a matrix vector multiplication": objects in a network "have a whole zoo of
different shapes", and the symbol is "a shorthand for figure out what kind of multiplication you actually want to do here" (≈33:25–34:10).
In LEGO terms, a brick "has a theory associated with it" when it has these attributes, and the second definition constrains them against
each other: "if the LEGO brick satisfies these constraints, we'll put it in our library. And if it doesn't, we'll just throw it out because
we don't like it" (≈30:21).

### Two atoms: the linear and embedding modules (slide 21)

The **linear module** $\mathrm{L}$ computes $\mathbf{W} \mathbf{x}$, with sensitivity 1, mass 1, and "this kind of funky norm, this
rescaled spectral norm" (≈30:21–31:07):

$$\mathrm{L.norm} = \Vert \cdot \Vert_ {\text{spectral}} \ast \sqrt{\text{fan-in}/\text{fan-out}}$$

This is the RMS-RMS operator norm of [lecture 7](07-scaling-rules-for-optimization.md), the spectral norm times
$\sqrt{d_ {\text{in}}/d_ {\text{out}}}$ (see [norms](norms.md#induced-operator-norms)). With the RMS norm on both input and output, "which
is kind of a natural choice for feature vectors inside a neural network", the linear module is well-normed if
$\Vert \mathbf{x} \Vert_ {\mathcal{X}} \le 1$ and $\mathrm{L.norm}(\mathbf{W}) \le 1$: bounded inputs and bounded weights (slide 21,
≈31:07). The lecturer draws a lesson about normalization layers: "one of the basic things that, in an actual neural net, that you do is you
normalize all the feature vectors everywhere explicitly. People don't really know why they're doing that other than somehow, intuitively,
it's going to make things more stable and stop things blowing up. But this is pointing out that if you want to have Lipschitz guarantees
for a linear module, the inputs and the weights must be bounded. If they're not, it can be arbitrarily sensitive" (≈31:07–31:52; see
[normalization layers](normalization-layers.md)).

The **embedding module** $\mathrm{E}$ "looks like it's kind of the same thing": also $\mathbf{W} \mathbf{x}$, sensitivity 1, mass 1. But its
input is a one-hot vector that "picks out one of the columns of the matrix", so "an embedding layer has a very different semantics to a
linear layer", and that shows in its norms (≈31:52–32:38). Its weight norm is the largest RMS norm of a column,
$\mathrm{E.norm} = \max_ i \Vert \text{column}_ i(\mathbf{W}) \Vert_ {\text{RMS}}$, and its input gets the $\ell_ 1$ norm, because a one-hot
vector "is a bit like a probability vector", with the RMS norm on the output. It is well-normed "under the same conditions of like unit
inputs and unit weights" (≈32:38–33:25). Slide 21 prints the second condition as "L.norm(W) ≤ 1" in the embedding module's box too, where
the module is $\mathrm{E}$: a slip, kept as printed.

A student asked what happens at the boundary, when the gradient pushes the inputs or weights outside the unit ball. "The short answer is
we've not actually implemented this part really faithfully yet ... we just let everything go outside the boundary." The plan is "manifold
optimization algorithms where the weights stay on this like well-normed manifold", which would make the guarantees consistent from layer to
layer; "there's not like a complete version, which is where everything is consistent in practice" (≈34:10–34:55).

### Combination rules (slides 22–23)

The abstract definitions exist so that combinations "automatically preserve those properties": "if you take two well-normed modules in your
library and compose them, the new thing that you get is automatically satisfies the definition of being well normed" (≈35:44). Slide 22
states three propositions: module combination is associative; module combination preserves well-normed-ness; and "feature learning is
apportioned by mass", the mass being what lets you "tune how much different modules contribute relative to each other", like "tune
learning rates at different layers" (≈35:44–36:31). "The kind of style of what this is, is come up with really, really clever definitions.
And then everything else is easy. So the definitions are kind of the hard part" (≈36:31).

The rules, from slide 23, for the **composite** $\mathrm{M} = \mathrm{M}_ 2 \circ \mathrm{M}_ 1$:

- `M.forward = M2.forward ∘ M1.forward`;
- `M.sensitivity = M1.sensitivity * M2.sensitivity`: "if you compose things, you multiply that Lipschitz constant";
- `M.mass = M1.mass + M2.mass`;
- `M.norm = max(p * M1.norm, q * M2.norm)`, where `p = M.mass / M1.mass * M2.sensitivity` and `q = M.mass / M2.mass` (the slide prints no
  brackets, so how the denominator of $p$ groups is not marked).

For the **concatenation** $\mathrm{M} = (\mathrm{M}_ 1, \mathrm{M}_ 2)$ the forward functions are paired, the sensitivities and masses
add, and the norm is the same weighted max with `p = M.mass / M1.mass` and `q = M.mass / M2.mass`.

"The important thing is that we use a max norm": the norm on the joint weight space is a maximum of the two modules' norms, weighted by
coefficients that "depend on the mass ratios", so that the mass sets "the importance of the two different modules in the composition"
(≈37:19). In a composition, $p$ also carries the second module's sensitivity: "if you compose two things, but the second one is very, very
sensitive to its inputs. Somehow, the first one should know about that because if you're going to change the weights of the first one, you
hopefully want to know that the second one is really, really sensitive or it's very insensitive" (≈37:19–38:04). Concatenation "doesn't have
the coupling because they don't compose with each other. So it's a little bit simpler" (≈41:58). With these definitions "you can prove that
if you compose two things that are well-normed, then the composition is also well-normed".

**Why mass?** Think of a transformer, with an embedding layer, a body of residual blocks and a linear output layer, and scale the number of
blocks into the millions: "Should you even still be training the input layer and the output layer? Should they now become negligible
because you want all the learning to happen in the middle? Or actually, the embedding layer plays a really interesting and different
functional role to the blocks, so even if you have a lot of blocks, you still want to train the embeddings. It's not obvious what the answer
to that question is. I find it difficult to know." The mass is the degree of freedom that lets you choose (≈38:50–39:37). Only ratios
matter: doubling every module's mass changes nothing, while doubling some doubles their contribution to learning relative to the rest
(≈39:37–40:24). It is "a degree of freedom that we couldn't really see a theoretical way to get rid of, so it's something you have to tune"; in
the experiments, "if all of the blocks have mass m and there's l blocks, each block gets mass m divided by l", and the input and output
layers get mass 1, though "maybe it would work better if you tune them separately" (≈40:24–41:58).

### Second order: sharpness (slide 24)

The theory "works to second order" (slide 24). A module is **$(\alpha, \beta, \gamma)$-sharp** if its second derivatives obey

$$\Vert \Delta \mathbf{w} \thinspace \nabla_ {\mathbf{w}} \nabla_ {\mathbf{w}} \mathrm{M} \thinspace \Delta \tilde{\mathbf{w}} \Vert_ {\mathcal{Y}} \le \alpha \ast \mathrm{M.norm}(\Delta \mathbf{w}) \ast \mathrm{M.norm}(\Delta \tilde{\mathbf{w}})$$

$$\Vert \Delta \mathbf{x} \thinspace \nabla_ {\mathbf{x}} \nabla_ {\mathbf{w}} \mathrm{M} \thinspace \Delta \mathbf{w} \Vert_ {\mathcal{Y}} \le \beta \ast \mathrm{M.norm}(\Delta \mathbf{w}) \ast \Vert \Delta \mathbf{x} \Vert_ {\mathcal{X}}$$

$$\Vert \Delta \mathbf{x} \thinspace \nabla_ {\mathbf{x}} \nabla_ {\mathbf{x}} \mathrm{M} \thinspace \Delta \tilde{\mathbf{x}} \Vert_ {\mathcal{Y}} \le \gamma \ast \Vert \Delta \mathbf{x} \Vert_ {\mathcal{X}} \ast \Vert \Delta \tilde{\mathbf{x}} \Vert_ {\mathcal{X}}$$

where $\Delta \tilde{\mathbf{w}}$ and $\Delta \tilde{\mathbf{x}}$ are second perturbations. "Because a module has two arguments, we can give
it three sharpness coefficients, which correspond to varying just the weights, varying the weights and the input, or varying just the
input" (≈42:44). The slide's gloss is "Generalized top eigenvalues": "when people talk about the sharpness of the Hessian in deep learning,
what they're meaning is how big is the largest eigenvalue of the Hessian. But somehow, that's like a Euclidean constant. And if you don't
believe that the geometry of the weight space is a Euclidean geometry, then you shouldn't necessarily even care about the top eigenvalue"
(≈42:44–43:32; see [loss landscapes](loss-landscapes.md)). As with the first-order attributes, "there's an automatic way to determine what
the sharpness coefficients are of compositions and concatenations" (the slide: the "sharpness tuple $(\alpha, \beta, \gamma)$ obeys
associative combination laws"), and that shows the loss is Lipschitz smooth. Slide 24's bullets: neural net loss functions are "Lipschitz
smooth in the modular norm", "with non-dimensional Lipschitz constants!", "so long as the error measure is smooth in the module output"
(≈43:32).

## Part III — Scaling (slides 26–34)

### Bigger is better, until it is not (slides 26–28)

"Even in the natural world, ... big things with bigger brains are supposed to be more intelligent" (≈44:19). Slide 26 captions four photographs with neuron counts, from 302 neurons to 250 billion; "the African elephant is, according to Wikipedia, ... the organism with the
most neurons" (≈44:19–45:05). Slide 27's "Recipe for AGI?" is "get the biggest supercomputer you can", "scrape as much data as you can
(don't get caught!)" and "train the biggest transformer you can". But, as in "the earlier lecture" (≈45:05; lecture 7), scaling can hurt
(slide 28): as width grows, "the performance is getting better, but the optimal step size, which is on the x-axis, is just drifting", and
"if you scale badly in depth, the performance can get worse as you make the network deeper" (≈45:05). See [scaling
rules](scaling-rules.md#the-problem).

### The thesis (slide 29)

"Our thesis is scaling well is actually not difficult. It's just about really understanding the Lipschitz properties of the system, which
is the thing that people don't really understand" (≈45:51). Slide 29 puts it as non-dimensional, tight Lipschitzness implying
learning-rate transfer. If for a generic module one can achieve **Lipschitz constants independent of width, depth and so on**
("non-dimensional ... as you change the scale of the system, the Lipschitz constant is staying fixed") and **bounds that stay tight across
scale** ("not like a really loose bound that's kind of useless"), then controlling $\mathrm{M.norm}(\Delta \mathbf{w})$ controls the size of
$\Delta \mathbf{y}$. "Formally, we want"

$$\Vert \Delta \mathbf{y} \Vert_ {\mathcal{Y}} \le \mathrm{M.norm}(\Delta \mathbf{w})$$

"to hold tightly". "If you normalize the updates in the corresponding norm, that will give
you good scaling. You kind of call it a thesis because it's not really a theorem. It's just like what we think" (≈45:51–46:41). The main
example is a network that is 1-Lipschitz: "one is a really great number, which is an example of a number, which is nondimensional", and then
"if we control the weight norm of our weight updates, we'll be able to predict that output norm of the change in outputs" (≈47:27).

### Breaking up the problem (slides 30–31)

The problem splits in two (≈48:14). For an **individual layer** (slide 30), a linear layer's weights and updates get "spectral conditions",
which "we put a little bit on the homework":

$$\sqrt{\text{fan-in}/\text{fan-out}} \ast \Vert \mathbf{W} \Vert_ {\text{spectral}} = 1 \qquad \sqrt{\text{fan-in}/\text{fan-out}} \ast \Vert \Delta \mathbf{W} \Vert_ {\text{spectral}} = \text{LR}$$

the weights at RMS-RMS operator norm 1 and the update at the learning rate. This is lecture 7's width rule (see [scaling
rules](scaling-rules.md#the-width-rule)). Slide 30 cites "On the distance between two neural networks and the stability of learning"
(Bernstein, Vahdat, Yue, Liu, NeurIPS 2020) and "A spectral condition for feature learning" (Yang, Simon, Bernstein, arXiv 2023). For
**composition and concatenation** (slide 31), the slide shows the title block of "Scalable Optimization in the Modular Norm" (Tim Large, Yang
Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, Jeremy Bernstein), "this paper that I worked on with Philip and also with some other
collaborators, which is about the idea of combining things" (≈48:14).

### Modula (slides 32–33)

"We actually built this software package, which is still a work in progress, but it's the idea that all of those rules, everything that I
was showing you, are things that can be programmed" (≈48:14–49:00). Slide 32's example builds an MLP from atoms and bonds,
`Linear(10,10000) @ ReLU() @ Linear(10000, 1000)`, a compound module, and trains it with a loop whose update is

```python
grad = weights.grad()
mlp.normalize(grad)
weights -= 0.1 * grad
```

"When we compose things, we automatically build a norm on the weight space. And then we can use that norm to normalize ... And the claim is
that because we did this clever normalization, the training will now scale better without needing to retrain the learning rate"
(≈49:00–49:47). Slide 33 lists versions on PyTorch (github.com/jxbz/modula/), JAX (github.com/GallagherCommaJack/modulax/) and NumPy (a
Colab notebook, "best place to start").

### Learning rates that transfer (slide 34)

Slide 34's experiment trains GPT for 10k steps on OpenWebText at a sweep of widths (top row) and depths (bottom row), plotting test loss
against learning rate (≈49:47–52:10). Its three columns:

- **Adam nanoGPT**: "if you just take like nanoGPT, which is this GitHub repository, and you try changing the width, the optimal learning
  rate is going to drift."
- **Adam modulaGPT**: "our own transformer implementation, which has some tweaks to the architecture to make it actually Lipschitz".
- **normed Adam modulaGPT**: "adding normalization on top of the optimization", which fixes the drift: the curves' minima line up across
  widths.

A student observed that normalizing a gradient is like changing the learning rate, "but in a way that's functionally dependent on the
gradient itself"; the lecturer agreed, adding that "it changes the learning rate in a way which depends on what the architecture of the
network is ... different layers will get gradients rescaled by different amounts" (≈51:21). In depth, nanoGPT "works reasonably well, but
it's a little bit messier"; without normalization, "Adam coincidentally does pretty well in terms of depth scaling, and then adding the
normalization on top, things stay pretty much the same" (≈52:10). He is frank that the result was expected: "we knew these things were
going to work because we had an intuitive understanding of how to do scaling, and then we kind of built the theory in order to do this"
(≈52:10). The slide adds two results from the paper: the normalization "enables training GPT using SGD" and "transfers LR across context
length".

![Slide 34: six panels of test loss against learning rate for Adam nanoGPT, Adam modulaGPT and normed Adam modulaGPT, the top row varying width from 32 to 1024 and the bottom row depth from 2 to 64 blocks](../raw/images/23-metrized-deep-learning/slide-34.jpg)

*Slide 34 — test loss against learning rate across widths (top) and depths (bottom): under Adam the best learning rate drifts with width, and with normalized updates on modulaGPT (right column) the minima line up. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec23.pdf)*

## Part IV — Modular duality (slides 36–43)

### Steepest descent as a step size times a duality map (slide 36)

This part, "the thing which led to the big training speed-up", starts again from a norm in which the loss obeys
$\mathcal{L}(\mathbf{w} + \Delta \mathbf{w}) \le \mathcal{L}(\mathbf{w}) + \nabla_ {\mathbf{w}} \mathcal{L}^{\top} \Delta \mathbf{w} + \tfrac{1}{2} \lambda \Vert \Delta \mathbf{w} \Vert^{2}$
(≈52:59–53:45). As in the homework, the steepest-descent step "decouple[s]" into two pieces (slide 36):

$$\arg\min_ {\Delta \mathbf{w}} \thinspace \nabla_ {\mathbf{w}} \mathcal{L}^{\top} \Delta \mathbf{w} + \tfrac{1}{2} \lambda \Vert \Delta \mathbf{w} \Vert^{2} = - \frac{\Vert \nabla_ {\mathbf{w}} \mathcal{L} \Vert}{\lambda} \ast \arg\max_ {\Vert \Delta \mathbf{w} \Vert = 1} \nabla_ {\mathbf{w}} \mathcal{L}^{\top} \Delta \mathbf{w}$$

The first factor, a dual norm over $\lambda$, is the **step size**; the second, an argmax over unit vectors, "we can actually call it a
**duality map**" (≈53:45). The slide prints the gradient's norm with no mark that it is the dual norm, which lecture 7's slides write
$\Vert \cdot \Vert^{\dagger}$; the lecturer calls it "a dual norm" (see [steepest descent](steepest-descent.md#any-norm-step-size-times-step-direction)).

### Gradient descent does not type check (slide 37)

"There's this idea in more classical optimization theory that the gradient is a dual vector and the weights are a regular vector",
where "the dual vector is a linear map that takes a regular vector and returns a real number" (≈54:37). What a gradient is "really good for
is taking dot products with weights and describing the linearization of the function. That's the only thing that you should ever really
trust the gradient will tell you ... The idea that you can take a gradient and just do gradient descent with it is generically a bad idea
and something that we should all forget" (≈54:37–55:25). The two have the same dimensions, so "technically you could add and subtract them",
"it can still be a bad idea": what is needed is "a procedure for converting dual vectors, like the gradient, back to the primal space such
that we can add and subtract them" (≈55:25). The argmax is such a procedure: "I want to take a step of unit norm which maximizes the linear
improvement of my objective function. And just subtracting the gradient from the weights may not solve that problem" (≈55:25–56:11).
Slide 37 puts it as two lines of code: `weight - LR * gradient` is crossed out, and `weight - LR * dualize(gradient)` ticked.

A student asked whether the objection is that a gradient step need not improve a nonlinear objective. "For a sufficiently small step size,
it will give you an improvement. It may just be really suboptimal ... Just moving in the gradient direction is equivalent to saying that
this is the Euclidean norm. And so the optimization space in neural networks is really unlikely to have a Euclidean structure" (≈56:11–56:57).
Slide 37's picture is a man facing a distorted reflection in a mirror, labelled "weight space" and "gradient space": the things in the
gradient space "are stretched and distorted. So we need some procedure to undistort them before we're going to subtract them from the
weights" (≈56:57–57:43). "In a sense, this should not be surprising because everyone tells you, if you're going to do deep learning, you
should just use the Adam optimizer. No one ever tells you to use regular gradient descent, so ... intuitively, we all know there's something
bad about regular gradient descent, we just don't quite know what it is" (≈57:43–58:29). See [gradient descent](gradient-descent.md).

### Dualizing a whole network (slides 38–39)

"How do we come up with such a dualization procedure for a neural net?" The recursive construction of the weight norm (slide 38 repeats
slide 14) suggests "a corresponding procedure for recursively constructing a duality map" (≈58:29). **Modular dualization** (slide 39):
first solve the duality map for each layer, $\text{dualize}(\mathbf{G}) = \arg\max_ {\Vert \mathbf{A} \Vert = 1} \langle \mathbf{A}, \mathbf{G} \rangle$,
"and that's what you did on the homework because we said that we're going to give the linear layers a spectral norm, and then we're going to
solve the duality map that corresponds to the spectral norm"; then "there is a valid recursive procedure for combining them to come up with
a duality map on the full architecture", for modules in series and in parallel, "which I didn't write it down in the slides, but maybe it's
something fun to think about" (≈58:29–59:15). The dual norm, the step size, has a recursive procedure too, also not on the slides
(≈1:00:02).

Asked whether dualization is a projection, the lecturer was unsure: "Definitely the one on the homework was a projection. Yeah, I think, for
norms, it probably usually is a projection. So yeah, I need to just think through that", "taking the gradient and projecting it to the unit
norm ball but also distorting the direction" (≈1:00:53). His intuitive summary: the Lipschitz guarantees tell you "when the second-order
changes and the network output become important relative to the first order changes, so a sensible thing to do is to take an optimization
step which squeezes as much kind of juice out of the linearization as possible while making sure that the second-order network effects don't
dominate the first-order effects, because the gradient really only tells you about the first-order effects" (≈1:01:44).

### Shampoo's core primitive (slides 40–41)

The backstory: Shampoo, an optimizer "that people at Google proposed", "won this speed challenge at this industry benchmarking competition"
(not named on the slides), but "the method that they actually use is totally different to the method in the paper" (≈1:02:29). Slide 40
shows the paper's algorithm for a matrix, which accumulates $\mathbf{L}_ t = \mathbf{L}_ {t-1} + \mathbf{G}_ t \mathbf{G}_ t^{\top}$ and
$\mathbf{R}_ t = \mathbf{R}_ {t-1} + \mathbf{G}_ t^{\top} \mathbf{G}_ t$ and updates
$\mathbf{W}_ {t+1} = \mathbf{W}_ t - \eta \mathbf{L}_ t^{-1/4} \mathbf{G}_ t \mathbf{R}_ t^{-1/4}$. Stripping away what is not used in practice, its
core primitive for a linear layer is

$$\Delta \mathbf{W} = -\eta \ast (\mathbf{G} \mathbf{G}^{\top})^{-1/4} \thinspace \mathbf{G} \thinspace (\mathbf{G}^{\top} \mathbf{G})^{-1/4} = -\eta \ast \arg\max_ {\Vert \mathbf{A} \Vert \le 1} \langle \mathbf{G}, \mathbf{A} \rangle$$

"i.e. "steepest descent under the spectral norm"" (slide 40): "this funny gradient transformation is actually equivalent to that homework
problem that you all did, but when you read the paper, they don't frame it that way" (≈1:03:15). Slide 41 adds that the transformation is
$\mathbf{G}^{0}$, "i.e. set all singular values to one", the semi-orthogonal $\mathbf{U} \mathbf{V}^{\top}$ of lecture 13's solution.

Modula made it easy to try: "we just override the normalize function of a linear module to do this spectral operation" (≈1:04:02). Slide
41's `ShampooLinear` class computes an SVD of the gradient and returns `U @ Vt * target_norm`. Its chart, from "a simple notebook, just as a
proof of concept", compares training loss for five learning rates under vanilla gradient descent, normalizing by $\mathbf{G} / \Vert \mathbf{G} \Vert_ \ast$,
and normalizing to $\mathbf{G}^{0}$: "doing this special normalization, it actually leads to training much faster than the thing in the
middle, which is just like a naive form of spectral normalization. And both of them are much better than just vanilla gradient descent"
(≈1:04:02).

![Slide 41: the ShampooLinear code that overrides normalize with an SVD, and three log-scale charts of training loss over 1000 iterations for five learning rates, under vanilla gradient descent and two kinds of modular normalization](../raw/images/23-metrized-deep-learning/slide-41.jpg)

*Slide 41 — Shampoo's primitive implemented by overriding Linear, and training loss for five learning rates under vanilla gradient descent, normalizing by the spectral norm, and setting every singular value to one. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec23.pdf)*

### Newton–Schulz and the speed record (slides 42–43)

The catch is cost: "it's so expensive to do these spectral normalization operations that it's practically useless". But "this open-source
neural network speedrunning community" found a fast way. Keller Jordan (@kellerjordan0) "made a really, really fast implementation of this
spectral dualization procedure. It turns out, at the moment, it's the fastest way to train nanoGPT on this speedrun", using "an iterative
method" (≈1:04:02–1:04:51). Slide 42 announces "New NanoGPT speed record!", with a chart comparing optimizers by wallclock time (Adam,
DistributedShampoo at two update frequencies, SOAP and Muon), and names the iteration, "Newton-Schulz", which does `Linear.dualize` fast:

$$\mathbf{X}_ {t+1} = a \mathbf{X}_ t - b \mathbf{X}_ t \mathbf{X}_ t^{\top} \mathbf{X}_ t + c \mathbf{X}_ t \mathbf{X}_ t^{\top} \mathbf{X}_ t \mathbf{X}_ t^{\top} \mathbf{X}_ t$$

The slide gives no values for $a$, $b$ and $c$. The lecturer then corrected his own library (slide 43, the code of slide 32 with
`mlp.normalize(grad)` struck out and `mlp.dualize(grad)` written beside it): "when we wrote this modular package, we actually kind of did it
wrong because when we said that we should normalize, that is actually not optimal. We should actually do this dualization procedure. And it
should recursively compute the dualization based on the architecture of the network. So we still need to update the package to do that"
(≈1:04:51–1:05:37).

## Conclusion (slides 45–48)

The conclusion repeats the opening (slides 45 and 46): build a network like LEGO, "and you should be able to have a theoretical understanding
of what you build, and it should be based on thinking about just how you combine individual bricks"; so far the payoffs are "these scaling
issues and now, apparently, also these training speed records" (≈1:05:37). The lecturer thinks the idea "can be a more generally useful idea"
(slide 47, "The future: Robust, low-precision models"). "If you want to save energy and money and time, a good way to do that is to make your
models really low precision", yet "no one has a theoretical understanding of how precision in a neural network should behave ... all the
research is just trying stuff". "Precision and continuity are very closely, very intimately related concepts because if you want to quantize
as much as you can, you need to know what discretization scale the space is kind of continuous on" (≈1:06:26). Similarly, adversarial
robustness "should also be related to having a good understanding of the continuity properties of the space". Slide 47: "We believe are
questions of module sensitivity" (as printed). He wants "to turn this into an open-source project", at https://modula.systems/ (slide 48),
which "doesn't have very much information there yet" (≈1:07:15).

## Pointers to other lectures

The deck prints no lecture number of this course, its own or another's; its title slide reads only "Metrized deep learning" and "Jeremy
Bernstein". The larger numbers in the bottom-right corner of thirty pages belong to an earlier talk (see the [slide
file](../raw/slides/23-metrized-deep-learning.md)). The recording points back without numbers: to the scaling issues "in one of the earlier
lectures in the class" (≈6:15) and "in the earlier lecture" (≈45:05), and to "optimization theory earlier in the class" (≈8:36), all
[lecture 7](07-scaling-rules-for-optimization.md); and, repeatedly, to "the homework problem" on steepest descent under the spectral norm
(≈7:02, ≈8:36, ≈10:09, ≈53:45, ≈59:15, ≈1:03:15), which on OCW is Homework 2's first question. Lecture 1's preview banner "Lecture 23:
Automatic gradient descent" names a lecture 23, but neither this deck nor this recording uses that phrase (see the [course
map](course-map.md#the-decks-lecture-pointers)). The lecture names no later lecture.

## See also

- [Modules and the modular norm](modular-norm.md) — the theory across lectures 7 and 23: modules, well-normed modules, combination rules,
  mass, sharpness and modular dualization.
- [Steepest descent](steepest-descent.md) — the step size times duality map, the type-check argument, and Shampoo's primitive.
- [Norms](norms.md) — induced operator norms, the rescaled spectral norm, the embedding norm and the weighted max norm.
- [Lipschitz continuity](lipschitz-continuity.md) — Lipschitz and Lipschitz smooth in the weights, sensitivity and well-normed modules.
- [Second-order methods](second-order-methods.md) — the Gauss-Newton decomposition used to bound the loss's curvature.
- [Scaling rules](scaling-rules.md) — the thesis for good scaling and learning-rate transfer in Modula.
- [Gradient descent](gradient-descent.md) — why gradient descent "does not type check".
- [Adversarial examples](adversarial-examples.md) — sensitivity to inputs, and robustness as a question of module sensitivity.
- [Normalization layers](normalization-layers.md) — why bounded inputs matter for Lipschitz guarantees.
- [Loss landscapes](loss-landscapes.md) — sharpness as a generalized top eigenvalue.
