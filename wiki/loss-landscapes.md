# Loss landscapes: what makes a function easy or hard to optimize

The **loss landscape** is the cost $J(\theta)$ viewed as a surface over the parameters $\theta$.
Gradient descent walks downhill on it, so its shape decides whether training is fast, slow,
unstable or stuck. Lecture 1 draws one (slide 28); lecture 2 takes six one-parameter toy losses
apart, one failure at a time, and shows two fixes. Covered so far: [lecture 1](01-introduction.md),
slide 28; [lecture 2](02-how-to-train-a-neural-net.md), slides 7 and 10–25, ≈3:54–26:31; [lecture 6](06-generalization-theory.md),
slide 61, ≈1:14:30–1:15:19, on flat minima.

## Differentiable is not the same as easy

Lecture 2 quizzes the class on six curves: a smooth bowl, a zig-zag with two minima, a flat line,
a step, a cusp, and a line with a jump (slides 12–14, ≈10:54–13:12). Three different questions get
three different answers:

- **Differentiable everywhere?** Only the bowl and the flat line.
- **Defined gradients in PyTorch?** All six, "because PyTorch has autograd", which assumes "things
  like directional derivatives around discontinuities" (≈11:40, ≈17:08).
- **Hard to optimize?** Four: the zig-zag, the flat line, the step and the cusp.

The lecturer's summary: "PyTorch can make anything differentiable. That doesn't mean it will be
easy to optimize. … it's computationally possible to calculate a gradient there, but the gradient
will still be 0" (≈17:08).

## The six cases

Each case slide plots the loss beside a run of gradient descent started at $\theta \approx 0.5$,
with optimizer steps running down the page (slides 15–20, ≈14:01–16:22).

| Case | Shape | What the gradient does | What gradient descent does |
| --- | --- | --- | --- |
| Convex | one smooth bowl | points to the minimum everywhere, "gracefully" going to 0 there | settles at the minimum by about step 50 |
| Discontinuous | a jump, with "well-defined one-sided derivatives" | fine on both sides | reaches the jump, where the low side is; "not a problem … for PyTorch" |
| Vanishing gradient | almost flat | tiny everywhere | barely moves; "noise in your batches … might end up dominating" |
| Zero gradient | a step, flat on both sides | exactly 0 | never moves; the low region is never reached |
| Exploding gradient | a cusp | "goes to infinity as minimizer is approached" | overshoots back and forth and never settles |
| Multiple local minima | two valleys | correct locally | stays in whichever valley it starts in |

Two cautions on the labels. On the quiz slide (slide 14) the step is labelled "Vanishing
gradient"; its own slide (18) calls it zero gradient. And on the cusp the problem is the size of
the steps near the minimum: "the gradient is really large, so then it's going to bounce you
somewhere really far away" (≈12:26–13:12).

For the multiple-minima case, "where you initialize matters a lot". The practical response is to
"try experimentally a bunch of different random seeds", and "the variance in your performance,
based on different random seeds, will tell you something about how unstable your loss landscape
is" (≈16:22). A picture for the easy case is water: on the bowl, "no matter where you started, you
would actually end up getting to that minimum" (≈13:12).

## Fixes

**Evolution strategies** (slide 21, ≈17:54–18:39) do without the gradient. They sample random
perturbations $\epsilon_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ of the parameters, evaluate
$s_i = J(\theta + \sigma \epsilon_i)$ at each, and step toward the perturbations that scored better:

$$\theta^{k+1} = \theta^{k} - \eta \frac{1}{\sigma M} \sum_{i=1}^{M} s_i \epsilon_i$$

Here $\sigma$ sets the perturbation size, $M$ is the number of samples and $\eta$ the step size.
Because they sample the function rather than its slope, they cross the step function that freezes
gradient descent. The lecturer says they help with "things like vanishing gradients", likens them to
regularization (≈21:46), and has no rule of thumb for $\sigma$ beyond starting from values a
published paper used (≈25:43–26:31).

**Gradient clipping** (slide 22, ≈18:39–20:13) caps each gradient component at magnitude $m$:
"If gradients exceed a magnitude m, scale them to magnitude m" — a "useful, and commonly used
hack". It tames the cusp: "you won't let your model move too far in any direction", and the clipped
run settles at the minimum where the unclipped one oscillated.

**Noise and momentum** change the walk too. Stochastic gradient descent's batch noise is "an
implicit regularizer" that "can actually bounce you out of local minima" (slide 9, ≈7:47).
Momentum speeds descent at moderate strength, and at high strength overshoots and oscillates
(slide 10, ≈10:06). Both are on the [gradient descent](gradient-descent.md) page.

## What a good function looks like

Lecture 2 distils this into three properties — everywhere **continuous**, everywhere
**differentiable**, everywhere **smooth** (slide 23, ≈20:13) — and applies them to activation
functions, since the network's non-linearities shape its landscape. ReLU is continuous,
differentiable "(Almost!)" and not smooth; GELU, $z \ast \Phi(z)$ with $\Phi$ the Gaussian
cumulative distribution function, is all three (slides 24–25). The claim is a trend, not a proof:
"we don't precisely know experimentally or theoretically, which properties are needed", but
"trends do seem to be moving towards functions which satisfy all three" (≈21:00). Whether
monotonicity also matters — GELU is not monotonic — is something "the jury's a little bit still out"
on (≈24:57). See [activation functions](activation-functions.md).

Lecture 1 meets the same problems from the activation side: saturating tanh and sigmoid units give
vanishing gradients, and the step function's zero gradient is why the original perceptron cannot
be trained by gradient descent ([activation functions](activation-functions.md)).

## Flat minima and generalization (lecture 6)

Lecture 6 adds a reason the *shape* of a minimum matters beyond reaching it (slide 61,
≈1:14:30–1:15:19). "Let's say that I have a really, really sharp well in my objective function, in my
energy landscape. Well, gradient descent can't go into a really sharp well if I'm using fixed step
sizes. It'll just jump right over it." So SGD, and gradient descent with a finite step size, tend to
settle in **flat minima**, basins "that are low loss, but are low loss over kind of a large region of the
parameter space". These "can be argued to generalize better than these narrow minima. The narrow minima
might be more like just a weird solution that got lucky". The slide points to Vardi (2022) for a review,
and the lecturer leaves the arguments to the papers. See [generalization and double
descent](generalization-and-double-descent.md).
