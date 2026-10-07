# Representational power: what a network can approximate

Which functions a neural network can represent, and at what cost. Lecture 1 previews this as
"why we can approximate" (slides 46–47, ≈40:25–42:42), and lecture 3, Approximation Theory, gives
the full treatment: one universal approximation theorem proved in full, a depth-separation result,
and the limits of both. Covered so far: [lecture 1](01-introduction.md),
[lecture 3](03-approximation-theory.md), and [lecture 4](04-architectures-grids.md)'s slides 3, 8 and
9 (universality weighed against inductive bias, and the SIREN preview).

Approximation is only one piece of the puzzle. Lecture 3's slide 4 splits machine learning into
three questions: **approximation** ("Does there exist a neural net in my model family that fits the
training data?"), **optimization** ("If it does exist, can I find it?") and **generalization**
("Does it work well on unseen data?"). Everything on this page answers the first question only.

## One layer versus two or more (lecture 1)

Slide 47 of lecture 1 states it in three lines:

- **One layer** gives a **linear decision surface**. A single perceptron can only split its input
  space with a line or hyperplane, which is enough for linearly separable data but never for XOR
  (see [multilayer perceptrons](multilayer-perceptron.md)).
- **Two or more layers** can, "in theory, … represent any function", **assuming a non-trivial
  non-linearity** between the layers. The assumption is essential: without it "a linear layer and
  another linear layer, this is just a linear combination. It's still linear" (≈40:25–41:11).
- **"But issue is efficiency."**

Lecture 3 opens with the same point (slide 5). A single $\text{relu}(\mathbf{w}^T \mathbf{x} + b)$
cannot separate data whose class boundary is a staircase, because "there's only one hyperplane
defined by w-transpose x. And ReLU does not distort the shape of the hyperplane", whereas a sum of
ReLUs plausibly can (≈3:50–4:36).

## The intuition: a Riemann sum (lecture 1)

Lecture 1's lecturer is explicit that this is "a rough, hand-wavy intuition thing" (≈41:11). Given
enough capacity, meaning enough units each contributing one narrow piece, a network can build up any
curve the way a Riemann sum builds up an integral from thin rectangles. "As you're shrinking the
size of these different vertical things", the approximation gets as good as you like, "as long as
you have enough capacity in terms of the number of small things that you're building up". Slide 47
shows a curve beside a staircase of bars approximating it, though a banner covers most of the
figure in the published deck. Lecture 3 turns exactly this picture into a proof.

## Formalizing the question (lecture 3)

Slide 7 of lecture 3 poses approximation precisely. Fix a **family of curves** $G$, the functions
you want to approximate, chosen for instance to "exclude pathological functions" such as
Weierstrass's continuous but nowhere-differentiable fractal (slide 6). Fix a **family of networks**
$F$, such as all five-layer ReLU MLPs: "a neural architecture could specify a family of functions".
Then ask whether, for every $g \in G$, some $f \in F$ has $\text{error}(f, g) \lt \epsilon$ for a
chosen small $\epsilon$. The error must be chosen too: the $L_\infty$ error
$\max_x |f(x) - g(x)|$, the largest gap anywhere, or the $L_1$ error
$\int dx \thinspace |f(x) - g(x)|$, the total gap (≈6:08–8:30).

## A universal approximation theorem (lecture 3)

The theorem lecture 3 proves (slide 10, ≈14:43–15:30) takes $G$ to be the
[Lipschitz](lipschitz-continuity.md) functions on the unit hypercube:

> Let $g: [0,1]^d \to \mathbb{R}$ be any $L$-Lipschitz function. Then for any error
> $\epsilon \gt 0$ there exists a 3-layer ReLU network $f$ with $N = 4d(L/\epsilon)^d$ units such
> that $\int_{[0,1]^d} |f(\mathbf{x}) - g(\mathbf{x})| \thinspace d\mathbf{x} \lt 2\epsilon$.

Here $L$ is the Lipschitz constant, $d$ the input dimension and $N$ the total number of neurons. A
"3-layer ReLU network" has three weight matrices, with a ReLU after the first two (≈17:48).

**The proof is the Riemann sum made exact,** in three steps:

1. **Rectangles** (slides 12–13). Approximate $g$ on $[0,1]$ by $N$ flat strips. Lipschitzness
   bounds each strip's error by a triangle of area $\frac{1}{2} L/N^2$, so the total error is at most
   $\frac{1}{2} L/N$, and error $\epsilon$ needs about $L/\epsilon$ strips.
2. **Hyperrectangles** (slide 14). In $d$ dimensions the strips become boxes, and error $\epsilon$
   needs $N = (L/\epsilon)^d$ of them.
3. **ReLUs make boxes** (slides 15–17). Four ReLUs with slopes scaled by a constant $c$ make a
   rectangle as $c \to \infty$. Adding $d$ such rectangles, one per axis, and thresholding the sum
   at $d - 1$ with another ReLU keeps only the box where all are "on". A final linear layer weights
   the boxes by their heights. So each box costs $4d$ neurons, which gives $N = 4d(L/\epsilon)^d$
   (≈42:53).

The full derivation, with every formula, is on the [lecture page](03-approximation-theory.md#the-proof).

**Its weaknesses, by the lecturer's own account** (slide 18, ≈39:45–41:22):

- It needs exponentially many neurons in the dimension $d$: "really bad dimension dependence. And we
  don't, in practice, really want to do that ever" (≈21:39).
- Taking $c \to \infty$ "feels unrealistic". Weights that blow up are usually "a sign that
  something's going wrong in your neural network".
- Building rectangles "feels like a trick". Why not use rectangles as basis functions directly?
- **It would not generalize** (slide 19). Trained on data, each point would pull up its own
  rectangle while the rest never moved, so "the training performance is going to be really good,
  but the generalization performance is going to be really bad" (≈43:38–45:13). See
  [generalization and double descent](generalization-and-double-descent.md).

**Other results** (slide 20). Barron's theorem shows "smooth functions can be approximated with
fewer neurons", using the Fourier representation. A classic result shows "2 layers are enough",
for example Hornik, Stinchcombe and White (1989), using the Stone–Weierstrass theorem, which in
the lecturer's understanding is "a more powerful result" than the three-layer one (≈45:13). When a
student recalled lecture 1's claim that a single ReLU layer suffices, the lecturer called that
"a different result" from the one lecture 3 proves (≈18:36).

## Is universal approximation enough?

Lecture 3's slide 21 asks whether universal function approximation ("UFA") is sufficient or
necessary for learning (≈46:48–49:59). **Not sufficient**: "there are many UFAs that we usually
don't do ML with", such as Fourier series, polynomials and "the space of Python programs". "Just
because you've installed Python, you can't start doing machine learning." **Maybe not necessary**:
"My guess is probably we don't need to have a universal function approximator to do machine
learning. But I think it's a little unclear how to answer that."

## Width versus depth

### What lecture 1 says

The catch is efficiency. "In theory, you could approximate any complicated function with a
sufficiently wide two-layer network", where wide means "maybe thousands or millions of dimensions",
"but in practice, that's actually very inefficient" (≈41:57). A **narrow, deep** model, with much
smaller layers but "many, many, many more stacked layers", may approximate the same function "with
a lot less parameters". The lecturer's empirical summary: "In practice, we do find that's true.
More layers helps."

Two follow-ups from lecture 1's discussion of this trade-off (≈48:08–49:40):

- **Interpretability doesn't favour either shape.** A student asked whether wide-and-shallow or
  narrow-and-deep is easier to interpret. Neither is: two near-infinite layers live in "really,
  really high dimensional space", and many narrow layers are just as opaque. "Anytime you get to
  larger numbers of parameters in either dimension, that interpretability gets more difficult."
  That is why fields such as ecology still use simple additive models of linear terms.
- **There is no known recipe.** Jeremy Bernstein noted that the width-versus-depth recipe of
  frontier language models is "a closely guarded secret", and the lecturer agreed: "there's no
  perfect prescription or recipe for width versus depth and what's optimal."

### Arguments for width (lecture 3)

Lecture 3's slide 24 lists them (≈50:47–53:53): three-layer (or even two-layer) networks are
universal approximators; "width is inherently parallelisable, depth is sequential", so a wide
shallow network makes better use of parallel hardware; and "width is easier to train, depth leads to
'compound problems'": a vanilla MLP made 50 layers deep is "very difficult to get it to train".
Hence, provocatively, "So, scaling width is obviously better!"

### Depth separation (lecture 3)

"Or is it?" A **depth separation** result constructs a deep network that a shallow one could only
match with exponentially more units (slides 25–26). Lecture 3 proves one by counting **kinks**,
the places where a piecewise linear function's slope changes (slide 27):

- ReLU networks are piecewise linear, since ReLU is, and sums, compositions and scalings of
  piecewise linear functions are too (slide 27).
- Adding functions at most adds their kinks (slide 28). Applying a ReLU at most doubles them,
  because each linear piece can be split in two where it crosses zero (slide 29).
- So for a network of width $n$, the kinks per unit grow by at most a factor $2n$ per layer, and
  after $L$ layers $\text{KINKS}_ L \le (2n)^L$: "polynomially in width but exponentially in depth"
  (slides 30–31, ≈1:01:43–1:05:41). Here $L$ is the depth. The base case of this induction was
  disputed in the lecture; see the [lecture page](03-approximation-theory.md#the-bound-polynomial-in-width-exponential-in-depth).
- The bound is attained (slide 32). The triangle map
  $g(x) = \text{relu}[2 \cdot \text{relu}(x) - 4 \cdot \text{relu}(x - \tfrac{1}{2})]$ doubles its
  number of linear regions every time it is composed with itself.
- So $g$ composed 500 times, a network of 1000 layers of width 2, has $2^{500} - 1$ kinks. A
  three-layer network would need width $n \ge \frac{1}{2} (2^{500} - 1)^{1/3} \approx 7 \times 10^{49}$
  to match (slide 33, ≈1:07:14–1:08:51).

**What this does not say** (slide 34): not that very deep networks are easy to train, and not that
they generalize well. "It only tells us something about approximation!" Nor is depth everything:
Lu, Pu, Wang, Hu and Wang (2017) show a **minimum width** is needed to be a universal approximator
even at large depth, "if the input space has dimension $n$, you need a width of at least $n$"
(slide 35, ≈1:09:37–1:10:24).

### In practice (lecture 3)

Experiments do not settle it either. Kaplan, McCandlish et al. (2020) found that, within a wide
range, test loss depends on the number of parameters and hardly on how they are split between
width and depth, while the Chinchilla paper showed that changing a detail like the learning-rate
schedule can change such conclusions (slides 38–39). See [scaling laws](scaling-laws.md).

## Where it goes next

Lecture 3 closes on **inductive biases** (slide 42). An MLP is a universal approximator, so one
could flatten audio or an image into a vector and fit an MLP, but it ignores the structure of the
data: "with an image, there's an inherent two-dimensional structure. And with a waveform, there's an
inherent sequential structure" (≈1:20:32). Matching the architecture to that structure is the subject
of the architecture lectures: grids (lecture 4), graphs (5), transformers (8) and memory (10). See
the [course map](course-map.md). How networks *generalize* from data, rather than merely fit it, is
a separate question; see [generalization and double descent](generalization-and-double-descent.md).

Lecture 4, the first architecture lecture, picks the thread up (see [inductive bias](inductive-bias.md)).
It credits the MLP's universality as a strength and its weak inductive biases as the cost: an
architecture that can represent the true function "and is otherwise minimal" learns from far less data
(slides 3 and 8). Its preview result is that "better architectures can approximate important function
classes more efficiently" (slide 9): SIREN, with sine activations, fits a photograph faster than ReLU
or tanh networks. The slide is careful about what that shows: the result "may be due to improved
approximation ability but it might also be due to improved optimization ability; these two effects
are typically coupled in experiments", the same approximation-versus-optimization split as lecture 3's
puzzle.
