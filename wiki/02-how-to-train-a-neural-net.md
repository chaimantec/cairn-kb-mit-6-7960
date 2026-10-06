# Lecture 2 — How to Train a Neural Net

**Lecturer:** Sara Beery ·
**Video:** [youtube.com/watch?v=vidCX_dMCu0](https://www.youtube.com/watch?v=vidCX_dMCu0) (79 min) ·
**Slides:** [`mit6_7960_f24_lec2.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec2.pdf)
(81 pages; transcribed slide by slide in [`raw/slides/02-how-to-train-a-neural-net.md`](../raw/slides/02-how-to-train-a-neural-net.md)) ·
**Transcript:** [`raw/transcripts/02-how-to-train-a-neural-net.md`](../raw/transcripts/02-how-to-train-a-neural-net.md)

## What this lecture establishes

Training a network means choosing parameters that minimize a cost, and every method the lecture
considers needs the gradient of that cost with respect to every parameter. The lecture first
reviews gradient descent and its stochastic, momentum, clipped and gradient-free variants, and
uses one-dimensional toy losses to show what makes a function easy or hard to optimize. It then
recasts a network as a **computation graph** — a directed acyclic graph of differentiable
functions — and derives **backpropagation** as the efficient way to get every gradient at once:
run the graph forward, then send gradients backward, computing each shared term of the chain rule
only once. It works this out for a generic layer, a linear layer, a ReLU, a whole MLP, and then
any DAG, which needs only two more rules, for merging and for branching. It closes on
**differentiable programming**: once any node can be optimized against any scalar cost, the same
machinery can optimize a network's *input* instead of its weights, which is how feature
visualizations, DeepDream and text-to-image generation by steering a GAN with CLIP are made.

The lecturer expects some of this "might be review for some of you", but hopes the way training is
framed "might be a slightly reformulated from something you've seen before" (≈0:00). The agenda
is a review of gradient descent and SGD, computation graphs, backprop through chains, through MLPs
and through DAGs, and differentiable programming (≈0:45).

The deck ends with a fully worked numerical example of one backpropagation step. The recording
does not reach it: when a student asked for a worked example, the lecturer said it is "in the
slides. And so you can actually work through it. And there are the solutions at the end"
(≈48:12). It is written out [at the end of this page](#worked-example-one-iteration-of-backpropagation).

The announcements slide (slide 2) says Pset 1 was out, due 9/24, with office hours starting and
PyTorch tutorials running that week.

## The training problem

Slide 4 redraws lecture 1's picture of training (≈0:45–1:32): an input $\mathbf{x}^{(i)}$ (a
clown fish photograph) with its label $\mathbf{y}^{(i)}$ ("clown fish"), a chain of six layers
with parameters $\theta_1, \ldots, \theta_6$, and a loss box computing
$\mathcal{L}(f_\theta(\mathbf{x}^{(i)}), \mathbf{y}^{(i)})$. The parameters are what is learned,
and "when you say getting learned, essentially, we just mean we wiggle the values around in
numerical space until we get something that is at least reasonably close to optimal, based on a
set of data that we actually have access to" (≈1:32). The goal is

$$\theta^{\ast} = \arg\min_\theta \sum_{i=1}^{N} \mathcal{L}(f_\theta(\mathbf{x}^{(i)}), \mathbf{y}^{(i)})$$

where $f_\theta$ is the network with parameters $\theta$, $N$ is the number of training examples
and $\mathcal{L}$ is the per-example loss. Slide 5 names the sum the **cost**, $J(\theta)$, so
training is $\theta^{\ast} = \arg\min_\theta J(\theta)$ (≈2:22). Asked late in the lecture whether
"loss" and "cost" are interchangeable, the lecturer called her own usage "a little fast and
loose" and said "it's not the end of the world to think of them somewhat interchangeably"
(≈1:10:02–1:10:48). The course's [notation](notation.md) keeps $\mathcal{L}$ and $J$ distinct.

### What you can know about J

Slide 6 sorts optimization methods by what can be computed about $J$ (≈3:08–3:54). If you can
only evaluate $J(\theta)$, you are doing **black-box optimization**. If you can also evaluate the
gradient $\nabla_\theta J(\theta)$, it is **first-order optimization**; add the Hessian
$H_\theta(J(\theta))$ and it is **second-order**. Second-order methods are possible, "in practice,
this isn't something that we often actually do. Usually, we really just focus on … first order
optimization, where we're just looking at that linear approximation to the gradient at a given
point" (≈3:08–3:54).

## Gradient descent

Gradient descent starts somewhere on the **loss landscape** — the surface $J(\theta)$ over the
parameters — and repeatedly steps downhill. Slide 7 draws one over two parameters,
$\theta_1$ and $\theta_2$: a path starts at a peak marked x and runs down into the shallower of two basins, not
the deep pit beside it. That is the point the lecturer makes: on a non-convex landscape "this may
not be guaranteed to be the global minimum. It might just be a local minimum. And how bad that
local minimum is relative to the global minimum, often, we don't know" (≈3:54–4:41).

![Slide 7: a 3D loss surface J(θ) over θ1 and θ2, with a tall peak marked x and a dashed path running down into the shallower of two basins, beside a deeper pit](../raw/images/02-how-to-train-a-neural-net/slide-7.jpg)

*Slide 7 — gradient descent follows the slope from where it starts, and here it ends in a local minimum, not the deepest one.*

One iteration (slide 8, ≈4:41–5:29) is

$$\theta^{k+1} = \theta^{k} - \eta \nabla_\theta J(\theta^{k})$$

where $\theta^{k}$ is the parameter vector after $k$ steps and $\eta$ is the **learning rate**,
which "is going to say how far we're going to step in the direction of that gradient" (≈4:41).
The slide leaves a blank where the assignment sign goes; the full-algorithm slide later writes
the same update with $\leftarrow$. See [gradient descent](gradient-descent.md).

## Stochastic gradient descent

The full gradient sums over every training example, and that is "often not possible to calculate,
mostly just due to the time, the computational complexity" (≈5:29). The lecturer's measure of
how things changed: "when I started in machine learning, we were lucky if we had ImageNet scale
data sets of a million images. Now we're dealing with image data sets in the billions and text
data sets even larger than that" (≈5:29–6:15).

**Stochastic gradient descent** instead computes the gradient on a subset of the data, a
**batch**, on the assumption that it is "a reasonable approximation to taking the gradient over
the entire data set" (slide 9, ≈6:15). With batch size 1 the parameters are updated after every
single example, which "might be suboptimal in terms of the computational complexity of doing
those updates"; with batch size $N$, the whole dataset, it is ordinary gradient descent (≈6:15–7:02).

The batch gradient is a noisy estimate of the full one. The lecturer's own experience is that the
noise depends on the data: with a strong imbalance across categories, some categories go unseen
for several batches, "and then you see them. And at that point, it actually massively shifts your
gradient. So this instability is actually often related to how uniformly distributed your data
set is" (≈7:02–7:47). The slide's ledger:

- **Advantages:** it is faster; it "approximates total gradient with small sample"; and its noise
  is "an implicit regularizer" — "it can actually bounce you out of local minima and help you
  reasonably find closer to what might be a global minimum" (≈7:47–8:35).
- **Disadvantages:** high variance and unstable updates, which she has "found experimentally …
  are more unstable the more unbalanced your training data set is" (≈8:35).

## Momentum

Momentum borrows the physical idea: "a heavy ball rolling down a hill, gains speed" (slide 10,
≈8:35–9:20). Each step is biased "to continue in direction of previous update":

![Slide 10: the momentum update rule above four panels — a V-shaped loss J(θ), and heat maps of θ over 100 optimizer steps for μ = 0, 0.5 and 0.95, the last oscillating around the minimum](../raw/images/02-how-to-train-a-neural-net/slide-10.png)

*Slide 10 — momentum 0.5 reaches the minimum in about half the steps of momentum 0; momentum 0.95 overshoots and is still oscillating at step 90.*

$$\theta^{t+1} = \theta^{t} - \eta \nabla f(\theta^{t}) - \alpha m^{t}$$

Here $f$ is the function being minimized, $m^{t}$ is the momentum term, "capturing the direction
that you moved in the past", and $\alpha$ is its strength, a hyperparameter (≈9:20). The slide
gives no separate formula for $m^{t}$. Momentum "can help or hurt", and the lecturer names Adam as
"a popular example of this type of momentum in gradient descent" (≈9:20–10:06).

The plots on slide 10 show both outcomes on a V-shaped loss with its minimum at $\theta = 0$, each
run starting from $\theta \approx 0.5$. (The plots label momentum $\mu$, where the equation uses
$\alpha$.) With $\mu = 0$ the parameter walks straight to the minimum, arriving at about step 50.
With $\mu = 0.5$ it arrives at about step 26, roughly "half the time the number of training
steps". With $\mu = 0.95$ it overshoots and oscillates, "bouncing past", and is still wobbling at
step 90: "it ends up taking longer to optimize your function" (≈10:06).

**How to read these plots** (the lecturer explained them when asked, ≈14:49–15:36). The left
panel is the loss $J$ against a single parameter $\theta$. In the right panel the horizontal axis
is again $\theta$, the background colour is the loss at that $\theta$, and the vertical axis is
optimizer steps, from 0 at the top to about 100 at the bottom. The white dashed line is where
$\theta$ sits at each step, so a line that runs straight down has stopped moving.

For intuition, the lecturer recommends Gabriel Goh's interactive Distill article "Why Momentum
Really Works" (2017), shown on slide 11: "I found this really helpful for building intuition"
(≈10:54).

![Slide 11: screenshot of Gabriel Goh's Distill article "Why Momentum Really Works" — an orange optimization path zig-zagging down a curved valley toward the optimum, with step-size and momentum sliders](../raw/images/02-how-to-train-a-neural-net/slide-11.jpg)

*Slide 11 — Goh's interactive article (Distill, 2017, CC BY), which the lecturer recommends for building intuition about momentum.*

## What makes a function easy or hard to optimize

The class was quizzed on six toy losses (≈10:54–13:12): a smooth bowl, a zig-zag with two minima,
a flat line, a step, a cusp, and a line with a jump in it.

**Which are differentiable?** Slide 12 highlights two: the bowl and the flat line. The flat one
counts because "it's flat, but it does have a derivative" (≈10:54).

![Slide 12: six small loss curves — a bowl, a zig-zag, a flat line, a step, a cusp, and a line with a jump — with the bowl and the flat line highlighted as differentiable](../raw/images/02-how-to-train-a-neural-net/slide-12.png)

*Slide 12 — only the bowl and the flat line are differentiable everywhere.*

**Which have defined gradients in PyTorch?** All six (slide 13), "because PyTorch has autograd. It
has this nice mechanism that helps us calculate gradients for any function" (≈11:40). That is not
the same as being easy to optimize, as she stressed in answer to a later question: "PyTorch can
make anything differentiable. That doesn't mean it will be easy to optimize. … it's
computationally possible to calculate a gradient there, but the gradient will still be 0", and
around discontinuities PyTorch assumes "things like directional derivatives" (≈17:08).

**Which will be hard to optimize?** Slide 14 marks four, and labels them: the zig-zag ("Local
minima"), the flat line ("Vanishing gradient"), the step ("Vanishing gradient") and the cusp
("Exploding gradient"). The bowl and the line with a jump are not marked (≈11:40–13:12). A useful
picture is flowing water: on the bowl, "no matter where you started, you would actually end up
getting to that minimum" — though the lecturer notes this intuition only rules out the case with
several minima (≈13:12).

![Slide 14: the same six curves, with the zig-zag labelled "Local minima", the flat line and the step each labelled "Vanishing gradient", and the cusp labelled "Exploding gradient"](../raw/images/02-how-to-train-a-neural-net/slide-14.png)

*Slide 14 — being differentiable is not the same as being easy to optimize: four of the six are hard, for three different reasons.*

### Six landscapes, one at a time

The next six slides take each shape in turn, each with a run of gradient descent from
$\theta \approx 0.5$ (≈14:01–16:22).

**Convex** (slide 15) is "maybe the simplest case … if you're a theoretician [it] is really great
because you can assume convexity. You have a single minimum. All the gradients point to it
everywhere. And the gradient will gracefully go to 0 as the minimum is approached" (≈14:01). The
run overshoots slightly and settles at the minimum by about step 50.

![Slide 15: a smooth bowl-shaped loss with its minimum at θ = 0, and a heat map in which the optimizer's path settles at the minimum by about step 50](../raw/images/02-how-to-train-a-neural-net/slide-15.png)

*Slide 15 — the convex case: one minimum, and gradients that point to it from everywhere.*

**Discontinuous** (slide 16): the function jumps, "but it's well defined in terms of the
derivatives on both sides of that discontinuity. And this is not a problem at all for PyTorch"
(≈14:01). The run reaches the jump at $\theta \approx 0$, where the low side is, and stays.

![Slide 16: a loss with a jump at θ = 0, lower on the right, and a heat map in which the path reaches the jump and stays there](../raw/images/02-how-to-train-a-neural-net/slide-16.jpg)

*Slide 16 — a discontinuity with well-defined one-sided derivatives is no problem for PyTorch.*

**Vanishing gradient** (slide 17): an almost flat line. "The progress would be really slow, and
noise in your batches, for example, might end up dominating over the signal" (≈14:49). The run
barely moves in 100 steps.

![Slide 17: an almost flat loss line, and a heat map in which the optimizer's path barely moves in 100 steps](../raw/images/02-how-to-train-a-neural-net/slide-17.jpg)

*Slide 17 — vanishing gradient: progress is slow, and batch noise may dominate.*

**Zero gradient** (slide 18): a step function, flat on both sides. The gradient is "completely
uninformative as how to make progress, and so it's really difficult to ever hit any low loss
region" (≈14:49). The run does not move at all. Slide 14 labels this same shape "Vanishing
gradient"; slide 18 is the one that calls it zero gradient.

![Slide 18: a step-function loss, flat on both sides, and a heat map in which the path is a straight vertical line — θ never moves](../raw/images/02-how-to-train-a-neural-net/slide-18.jpg)

*Slide 18 — zero gradient: the low-loss region on the left is never reached.*

**Exploding gradient** (slide 19): a cusp, where "the gradient goes to infinity as the minimizer
is approached, so you have really unstable updates, and you can really overshoot" (≈16:22). "As
you get close to that minimum, the gradient is really large, so then it's going to bounce you
somewhere really far away" (≈12:26–13:12). The run swings out to $\theta \approx -0.8$ by step 25
and is still swinging across the minimum at step 99.

![Slide 19: a cusp-shaped loss with a sharp minimum at θ = 0, and a heat map in which the path swings far to either side of the minimum without settling](../raw/images/02-how-to-train-a-neural-net/slide-19.png)

*Slide 19 — exploding gradient: the steps grow as the minimum gets close, so the run overshoots again and again.*

**Multiple local minima** (slide 20): "where you initialize matters a lot", and gradient descent
reaches *a* local minimizer, not necessarily the global one. Starting at 0.5, the run sits in the
shallower minimum for all 100 steps and never finds the deeper one at $\theta \approx -0.5$. This
is why practitioners "try experimentally a bunch of different random seeds", and "the variance in
your performance, based on different random seeds, will tell you something about how unstable
your loss landscape is" (≈16:22).

![Slide 20: a wavy loss with a deeper minimum near θ = −0.5 and a shallower one near 0.5, and a heat map in which the path stays in the shallower one](../raw/images/02-how-to-train-a-neural-net/slide-20.png)

*Slide 20 — multiple local minima: started at 0.5, gradient descent never finds the deeper minimum.*

These cases are collected on the [loss landscapes](loss-landscapes.md) page.

### Two fixes: evolution strategies and gradient clipping

**Evolution strategies** are "gradient-like: [they find] a locally loss-minimizing direction in
parameter space" without using the gradient. They "sample small perturbations of θ and move toward
perturbations that achieved lower loss" (slide 21, ≈17:54–18:39):

![Slide 21: the evolution-strategies update rule above the step-function loss, with a heat map in which the path drifts left across the step into the low region](../raw/images/02-how-to-train-a-neural-net/slide-21.png)

*Slide 21 — evolution strategies minimize the step function on which plain gradient descent never moves.*

$$\epsilon_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \qquad s_i = J(\theta + \sigma \epsilon_i), \qquad \theta^{k+1} = \theta^{k} - \eta \frac{1}{\sigma M} \sum_{i=1}^{M} s_i \epsilon_i$$

Each $\epsilon_i$ is a random direction drawn from a standard normal, $\sigma$ scales how far the
perturbation reaches, $s_i$ is the loss at the perturbed point, and the update averages the $M$
directions weighted by their losses. On the step function, where plain gradient descent never
moves, it "successfully minimizes this function": the run drifts across the step at about
iteration 65 and ends in the low region. In the lecturer's words, "because we're perturbing the
values of the function, it might bounce us over into a better value" (≈18:39). Asked to explain it
again, she described it as randomly sampling "around your current parameter values, and then you
move in the random direction that … minimizes your cost", helpful for "things like vanishing
gradients", and "almost like regularization in your loss" (≈21:46). How large to make the
perturbations has no universal answer: "one generally good strategy is you take a paper that used
it, and you look at the values they used, and you start there" (≈25:43–26:31).

**Gradient clipping** handles "peakiness or pointiness in our loss function" (slide 22,
≈18:39–20:13): "If gradients exceed a magnitude m, scale them to magnitude m", a "useful, and
commonly used hack". With $\mathbf{v} = \nabla_\theta J(\theta^{k})$, whose components are
$v_1, \ldots, v_M$,

![Slide 22: the gradient-clipping update rule above the cusp-shaped loss, with a heat map in which the path settles at the minimum by about step 60](../raw/images/02-how-to-train-a-neural-net/slide-22.png)

*Slide 22 — clipping each gradient component to at most m in size tames the cusp that kept the unclipped run oscillating.*

$$\theta^{k+1} = \theta^{k} - \eta \left[\texttt{clip}(v_1, -m, m), \ldots, \texttt{clip}(v_M, -m, m)\right]^{\mathsf{T}}$$

"So you won't let your model move too far in any direction. And this, then, helps us with not
oscillating back and forth across something" (≈19:26). On the same cusp that defeated plain
gradient descent on slide 19, the clipped run reaches the minimum by about step 60 and stays.

Are these tricks combined in practice? "Everything's a bit of a big, soup pot these days." Top
models "often … incorporate a lot of these tricks", but "I don't think we actually have a really
great recipe for given a certain data set, given a certain architecture, what is the right way to
optimize that architecture" (≈22:31–23:19).

### Continuous, differentiable, smooth

Slide 23 asks "What is important in a loss function?" and answers: everywhere continuous,
everywhere differentiable, everywhere smooth (≈20:13). The examples that follow are activation
functions, as the lecturer confirmed when a student asked (≈24:09). ReLU, $\max(0, z)$, is
continuous, differentiable "(Almost!)" and not smooth, because of "this kink" at 0 (slide 24).
**GELU**, the Gaussian error linear unit, is all three (slide 25):

![Slide 24: the checklist everywhere continuous (ticked), everywhere differentiable ("Almost!"), everywhere smooth (crossed), beside a plot of ReLU with its corner at 0](../raw/images/02-how-to-train-a-neural-net/slide-24.png)

*Slide 24 — ReLU fails the smoothness test because of its kink at 0.*

![Slide 25: the same checklist with all three ticked, beside a plot of GeLU — a smooth curve with a shallow dip below zero just left of the origin](../raw/images/02-how-to-train-a-neural-net/slide-25.png)

*Slide 25 — GELU is continuous, differentiable and smooth everywhere; the small dip left of zero is what makes it non-monotonic.*

$$\mathrm{GELU}(z) = z \ast \Phi(z)$$

where $\Phi$ is "the cumulative distribution function for the Gaussian distribution" (≈21:00);
the slide itself does not define it, and cites arXiv 1606.08415. The claim is a trend, not a
theorem: "even if we don't precisely know experimentally or theoretically, which properties are
needed for training neural networks, trends do seem to be moving towards functions which satisfy
all three" (≈21:00), and "I'm not saying here that GeLU is much better than ReLU all the time"
(≈23:19).

A student pointed out that GELU is not monotonic — it dips below zero just left of the origin.
"Is it better to be smooth or to be monotonic? And that's where I think the jury's a little bit
still out" (≈24:57). See [activation functions](activation-functions.md).

## Computation graphs

The second half moves "to thinking of this all from this perspective of differentiable
programming" (≈27:16). A **computation graph** is "a graph of functional transformations, nodes,
that when strung together perform some useful computation" (slide 26). The lecturer's everyday
example is a decision tree for getting to class: go into building 45 (correct) or building 32
(incorrect); in 45, take the elevator or the stairs. Each decision transforms inputs into an
output, which makes it "a really, really simple heuristic version of a functional transformation"
(≈27:16–28:04). In a network a node might be one layer "or even an entire neural network", with
no restriction on size (≈28:04–28:50).

![Slide 26: a DAG of eight beige boxes with arrows pointing upward — two inputs at the bottom, one node branching in two, two paths merging near the top, and two outputs](../raw/images/02-how-to-train-a-neural-net/slide-26.png)

*Slide 26 — a computation graph: a directed acyclic graph whose nodes are differentiable functions.*

"Deep learning deals (primarily) with computation graphs that take the form of **directed acyclic
graphs** (DAGs), and for which each node is differentiable" (slide 26). Directed: "the information
only goes one direction on any given edge". Acyclic: "you don't have loops or cycles". The
differentiability requirement is softer than it sounds, because PyTorch "can calculate a
derivative for almost anything" (≈28:50).

Even an MLP is "easy to represent as a computation graph" (slide 27, ≈29:36): input
$\mathbf{x}$, a `linear` node giving the pre-activation $\mathbf{z}$, a `relu` node giving the
hidden state $\mathbf{h}$, and a second `linear` node giving the output $\mathbf{y}$. See
[multilayer perceptrons](multilayer-perceptron.md).

![Slide 27: an MLP drawn as columns of neurons x, z, h, y with weights W1 and W2, shown as equivalent to the chain x → linear → z → relu → h → linear → y](../raw/images/02-how-to-train-a-neural-net/slide-27.png)

*Slide 27 — the same MLP as a network diagram and as a computation graph.*

### Forward pass, and what learning needs

A forward pass through any node takes an input and the node's parameters and produces an output
(slide 28, ≈29:36–30:24):

$$\mathbf{x}_ {\texttt{out}} = f(\mathbf{x}_ {\texttt{in}}, \theta)$$

Chain $L$ of them and the output of each "becomes the input of the next component" (slide 29,
≈30:24): $\mathbf{x}_ 0 \to f_1 \to \mathbf{x}_ 1 \to f_2 \to \cdots \to f_L \to \mathbf{x}_ L$,
with parameters $\theta_l$ feeding node $f_l$, and the final output feeding the loss
$\mathcal{L}$, which gives the cost $J$. Learning then needs the gradient of $J$ "with respect to
all of those model parameters that happen all the way through the network", and this is possible
because, "by design, every single layer will be differentiable with respect to its inputs" —
where a layer's inputs are both its data and its parameters (slide 30, ≈31:10).

![Slide 30: a chain of boxes f1 … fL, each fed its parameters θl, ending in a loss box and J, with an inset of a loss surface showing one step from θt to θt+1](../raw/images/02-how-to-train-a-neural-net/slide-30.jpg)

*Slide 30 — learning needs the gradient of the cost with respect to every layer's parameters.*

## Matrix calculus, briefly

"Just in case any of you are not brushed up on your matrix calculus" (≈31:55), slides 32 and 33
fix the shapes. These match the course's [notation handout](notation.md#matrix-calculus).

- A vector $\mathbf{x}$ is a column, $[n \times 1]$.
- For a scalar $y = f(\mathbf{x})$, the derivative $\partial y / \partial \mathbf{x}$ is a **row**
  vector, $[1 \times n]$ (slide 32).
- For a vector $\mathbf{y}$ of size $[m \times 1]$, $\partial \mathbf{y} / \partial \mathbf{x}$
  is the **Jacobian**, $[m \times n]$: "m rows, which is the size of y, by n columns [which] is the
  size of x" (slide 32, ≈32:42).
- For a scalar $y$ and a matrix $\mathbf{X}$ of size $[n \times m]$,
  $\partial y / \partial \mathbf{X}$ is $[m \times n]$: "by taking that derivative, we flipped the
  rows and columns. So it's been transposed" (slide 33, ≈33:29).

Derivatives of vectors by matrices, matrices by vectors and matrices by matrices have no agreed
notation, slide 33 notes, quoting Wikipedia, and the course does not use them (≈33:29).

The **chain rule** (slide 34, ≈33:29–36:33): for $h(\mathbf{x}) = f(g(\mathbf{x}))$,
$h'(\mathbf{x}) = f'(g(\mathbf{x}))\thinspace g'(\mathbf{x})$. Writing $\mathbf{z} = f(\mathbf{u})$
and $\mathbf{u} = g(\mathbf{x})$,

![Slide 34: the chain rule for vectors with the shapes m×n, m×p and p×n marked under its three factors, and a block picture of a 1×4 row equal to a 1×2 row times a 2×4 grid](../raw/images/02-how-to-train-a-neural-net/slide-34.png)

*Slide 34 — the chain rule as a matrix product: the inner dimensions have to match.*

$$\left.\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right|_ {\mathbf{x}=\mathbf{a}} = \left.\frac{\partial \mathbf{z}}{\partial \mathbf{u}}\right|_ {\mathbf{u}=g(\mathbf{a})} \cdot \left.\frac{\partial \mathbf{u}}{\partial \mathbf{x}}\right|_ {\mathbf{x}=\mathbf{a}}$$

with shapes $[m \times n] = [m \times p] \cdot [p \times n]$, where $p = \lvert \mathbf{u} \rvert$,
$m = \lvert \mathbf{z} \rvert$ and $n = \lvert \mathbf{x} \rvert$. The class worked these shapes
out together (≈35:46–36:33). In the slide's example, $\lvert \mathbf{z} \rvert = 1$,
$\lvert \mathbf{u} \rvert = 2$ and $\lvert \mathbf{x} \rvert = 4$, so a $1 \times 4$ row equals a
$1 \times 2$ row times a $2 \times 4$ matrix. "We're bringing this up because we will see a lot of
matrices times matrices in the next bit" (≈36:33).

## Backpropagation

### The trick: compute shared terms once

By the chain rule, the gradient for the first layer's parameters factors through every later
layer (slide 36, ≈37:20–38:06):

![Slide 36: the layer chain with two braces underneath for the gradients of J with respect to θ1 and θ2, and the two chain-rule expansions with their shared factors in a grey box](../raw/images/02-how-to-train-a-neural-net/slide-36.jpg)

*Slide 36 — the grey box is the same in both gradients; backpropagation computes it once.*

$$\frac{\partial J}{\partial \theta_1} = \left[\frac{\partial J}{\partial \mathbf{x}_ L} \frac{\partial \mathbf{x}_ L}{\partial \mathbf{x}_ {L-1}} \cdots \frac{\partial \mathbf{x}_ 3}{\partial \mathbf{x}_ 2}\right] \frac{\partial \mathbf{x}_ 2}{\partial \mathbf{x}_ 1} \frac{\partial \mathbf{x}_ 1}{\partial \theta_1}$$

$$\frac{\partial J}{\partial \theta_2} = \left[\frac{\partial J}{\partial \mathbf{x}_ L} \frac{\partial \mathbf{x}_ L}{\partial \mathbf{x}_ {L-1}} \cdots \frac{\partial \mathbf{x}_ 3}{\partial \mathbf{x}_ 2}\right] \frac{\partial \mathbf{x}_ 2}{\partial \theta_2}$$

(The slide prints the factor $\partial \mathbf{x}_ 2 / \partial \mathbf{x}_ 1$ without the second
$\partial$.) The bracketed product, shaded grey on the slide, is the same in both. "We could
separately compute all of the derivatives using the chain rule, but because these terms in the
gray box are shared, we only need to compute them once. So back propagation is a pretty simple
algorithm for propagating shared terms through the computation graph. It's basically an
efficiency trick, but it's one that makes it computationally practical for very large models"
(≈38:06). The slide's subtitle gives the other name for this: "aka dynamic programming".

### Forward, then backward

The **forward pass** sends data through the network, "computing outputs and calculating loss";
the **backward pass** sends "error signals (gradients) backwards through the network, from
outputs and loss back to inputs and parameters" (slide 37, ≈38:06–38:52). In the backward
diagram each node $f_l$ is replaced by its derivative $f_l'$; a 1 enters the loss's derivative
$\mathcal{L}'$; and each $f_l'$ passes a gradient $\mathbf{g}_ {l-1}$ down to the node below and
also emits $\partial J / \partial \theta_l$, the gradient for its own parameters.

![Slide 37: the forward pass as a chain of boxes with green arrows running left to right, and the backward pass as the mirrored chain of derivative boxes with red arrows running right to left, each emitting a parameter gradient](../raw/images/02-how-to-train-a-neural-net/slide-37.png)

*Slide 37 — forward computes the outputs and the loss; backward sends gradients from the loss back to every input and parameter.*

### One layer: the arrays L and g

For a generic layer the lecture tracks "two kinds of arrays of partial derivatives" (slide 38,
≈38:52–39:39):

![Slide 38: one layer f(x_in, θ) between x_in and x_out, with braces marking L under the layer and g_out and g_in under the path to J](../raw/images/02-how-to-train-a-neural-net/slide-38.png)

*Slide 38 — the two arrays backprop keeps for each layer: L, the layer's own derivative, and g, the cost's derivative at each activation.*

$$\mathbf{L} \triangleq \frac{\partial \mathbf{x}_ {\texttt{out}}}{\partial [\mathbf{x}_ {\texttt{in}}, \theta]}, \qquad \mathbf{g} \triangleq \frac{\partial J}{\partial \mathbf{x}}$$

$\mathbf{L}$ is the gradient of the layer's outputs with respect to its inputs, a matrix, split
into $\mathbf{L}^{\mathbf{x}} = \partial \mathbf{x}_ {\texttt{out}} / \partial \mathbf{x}_ {\texttt{in}}$
and $\mathbf{L}^{\theta} = \partial \mathbf{x}_ {\texttt{out}} / \partial \theta$. $\mathbf{g}$ is
the gradient of the cost with respect to an activation, a row vector:
$\mathbf{g}_ {\texttt{out}}$ at the layer's output and $\mathbf{g}_ {\texttt{in}}$ at its input.

With both in hand, "the parameter update is easy" (slide 39, ≈39:39–40:25):

$$\frac{\partial J}{\partial \theta} = \mathbf{g}_ {\texttt{out}} \mathbf{L}^{\theta}, \qquad \theta^{i+1} = \theta^{i} - \eta \left(\frac{\partial J}{\partial \theta}\right)^{\mathsf{T}}$$

The transpose is there because the gradient is a row vector. Where do the two arrays come from?
$\mathbf{L} = f'(\mathbf{x}_ {\texttt{in}}, \theta)$ comes from the layer's derivative function,
"which we assume is provided"; $\mathbf{g}$ comes from a recurrence that runs down the network
(slide 40, ≈40:25–41:12):

$$\mathbf{g}_ {\texttt{in}} = \mathbf{g}_ {\texttt{out}} \mathbf{L}^{\mathbf{x}}$$

"This is, essentially, the back propagation of the error signal back through the model." Slide 41
packs the three lines into a single `backward` box, under the caption "All this machinery is to
compute parameter update directions".

![Slide 41: a single "backward" box containing L = f′(x_in, θ), g_in = g_out L^x and ∂J/∂θ = g_out L^θ, with arrows for its inputs and outputs](../raw/images/02-how-to-train-a-neural-net/slide-41.png)

*Slide 41 — everything one layer does in the backward pass, in one box.*

**The full algorithm** (slide 42, ≈41:12): forward, then backward, then update, "and repeat".
"You do a forward pass, calculate loss. You take that loss, propagate it backward, update all your
parameters, and do it again, and again, and again". When to stop is deferred to later lectures,
but "it's often using something like a validation set and looking for some sort of plateauing of
change on that validation set" (≈41:12).

![Slide 42: the forward chain in green, the backward chain of derivative boxes in red with a parameter gradient under each box, and the update rule followed by "... and repeat"](../raw/images/02-how-to-train-a-neural-net/slide-42.png)

*Slide 42 — the whole algorithm: forward, backward, update, repeat.*

### The view from layer l

The next slides are "essentially cheat sheets … intended to be a pretty nice resource if you need
to refer back" (≈41:59). During training, layer $l$ has three inputs — the activation
$\mathbf{x}_ {l-1}$ from below, the gradient $\partial J / \partial \mathbf{x}_ l$ from above, and
its parameters $\theta_l$ — and three outputs (slide 43, ≈41:59–43:30):

![Slide 43: layers l−1, l and l+1 stacked vertically, with green forward arrows going up, red gradient arrows coming down, and θl entering layer l from the side; on the right, the layer's three inputs and three outputs](../raw/images/02-how-to-train-a-neural-net/slide-43.jpg)

*Slide 43 — the view from one layer: three inputs in, three outputs out.*

$$\mathbf{x}_ l = f_l(\mathbf{x}_ {l-1}, \theta_l)$$

$$\frac{\partial J}{\partial \mathbf{x}_ {l-1}} = \frac{\partial J}{\partial \mathbf{x}_ l} \cdot \frac{\partial f_l}{\partial \mathbf{x}_ {l-1}}$$

$$\frac{\partial J}{\partial \theta_l} = \frac{\partial J}{\partial \mathbf{x}_ l} \cdot \frac{\partial f_l}{\partial \theta_l}$$

So each layer only has to be able to evaluate $f_l$, $\partial f_l / \partial \mathbf{x}_ {l-1}$
and $\partial f_l / \partial \theta_l$. Slide 44 summarizes the procedure (≈43:30): a forward
pass computing every $\mathbf{x}_ l$; a backward pass computing the loss derivatives "iteratively
from top to bottom"; and a parameter update from each $\partial J / \partial \theta_l$.

![Slide 44: the three-step backpropagation summary beside a vertical stack of layers from the input x0 up to the loss, with forward values going up and gradients coming down](../raw/images/02-how-to-train-a-neural-net/slide-44.jpg)

*Slide 44 — the backpropagation cheat sheet.*

### Over a batch

Training minimizes the average cost over many data points (slide 45, ≈43:30–44:17):

$$J = \frac{1}{N} \sum_{i=1}^{N} J_i(\mathbf{x}^{i}, \theta), \qquad \frac{\partial J}{\partial \theta} = \frac{1}{N} \sum_{i=1}^{N} \frac{\partial J_i(\mathbf{x}^{i}, \theta)}{\partial \theta}$$

where $J_i$ is the cost on example $\mathbf{x}^{i}$. The gradient of the average is the average
of the gradients, "because when you take a derivative, it can move inside the sum". With large
GPUs, "that batch size could be thousands" (≈44:17). See
[tensors and batching](tensors-and-batching.md).

## Backprop through a linear layer

For a linear layer "everything gets really nice and simple" (≈44:17). The forward pass is a
matrix product (slide 46),

![Slide 46: a linear layer with its forward product drawn as blocks (a 3-cell column equals a 3×4 grid times a 4-cell column) and its backward product (a 4-cell row equals a 3-cell row times the same 3×4 grid)](../raw/images/02-how-to-train-a-neural-net/slide-46.png)

*Slide 46 — forward multiplies the input by W; backward multiplies the gradient row by the same W from the other side.*

$$\mathbf{x}_ {\texttt{out}} = f(\mathbf{x}_ {\texttt{in}}, \mathbf{W}) = \mathbf{W} \mathbf{x}_ {\texttt{in}}$$

with $\mathbf{W}$ of size $\lvert \mathbf{x}_ {\texttt{out}} \rvert \times \lvert \mathbf{x}_ {\texttt{in}} \rvert$
— $3 \times 4$ in the slide's block picture.

**Backprop to the input** (slide 46, ≈45:04–45:51). The $i$-th output's derivative with respect
to the $j$-th input is $W_{ij}$, so the Jacobian $\mathbf{L}^{\mathbf{x}}$ is $\mathbf{W}$ itself:

$$\frac{\partial x_{\texttt{out}_ i}}{\partial x_{\texttt{in}_ j}} = W_{ij} \quad\Longrightarrow\quad \mathbf{g}_ {\texttt{in}} = \mathbf{g}_ {\texttt{out}} \cdot \mathbf{W}$$

A row of 3 times a $3 \times 4$ matrix gives a row of 4. "In the opposite direction, you multiply
the gradient by the weights to get the gradient coming in. Cool, so your forward and backward
pass are just multiplying by your weight matrix, just in a different order" (≈45:51).

**Backprop to the weights** (slide 48, ≈45:51–47:25). The weight $W_{ij}$ affects only the $i$-th
output, and $\partial x_{\texttt{out}_ i} / \partial W_{ij} = x_{\texttt{in}_ j}$, so

![Slide 48: the derivation of ∂J/∂W for a linear layer, ending in a block picture of a 4×3 grid equal to a 4-cell column times a 3-cell row, and the weight-update rule](../raw/images/02-how-to-train-a-neural-net/slide-48.jpg)

*Slide 48 — the weight gradient is an outer product of the layer's input and its output gradient.*

$$\frac{\partial J}{\partial W_{ij}} = \frac{\partial J}{\partial x_{\texttt{out}_ i}} \cdot x_{\texttt{in}_ j} \quad\Longrightarrow\quad \frac{\partial J}{\partial \mathbf{W}} = \mathbf{x}_ {\texttt{in}} \cdot \mathbf{g}_ {\texttt{out}}$$

an outer product: a column of 4 times a row of 3 gives a $4 \times 3$ matrix, the transpose of
$\mathbf{W}$'s shape, as the course's convention for a scalar's derivative by a matrix requires.
"When you want to calculate the update to the weights, all you're doing is you're multiplying the
input to the function by the output gradient. Again, it's a really nice simplification that comes
out just because everything is linear" (≈47:25). The update transposes it back:

$$\mathbf{W}^{k+1} \leftarrow \mathbf{W}^{k} + \eta \left(\frac{\partial J}{\partial \mathbf{W}}\right)^{\mathsf{T}}$$

**Watch the sign.** Slides 48, 49, 76 and 80 write the weight update with a plus sign, where
slides 8, 39 and 42 subtract. The worked example at the end of the deck explains the convention: its learning
rate is "η = -0.2 (because we used positive increments)". A negative $\eta$ in the plus form is
the same downhill step as a positive $\eta$ in the minus form.

The cheat sheet for the linear layer (slide 49, ≈48:12–48:58) is three lines:

![Slide 49: a linear-layer summary box with the forward and backward products inside, the weight gradient in a separate box fed by both x_in and g_out, and the weight-update rule](../raw/images/02-how-to-train-a-neural-net/slide-49.png)

*Slide 49 — the linear-layer cheat sheet: three products cover forward, backward and the weight gradient.*

$$\mathbf{x}_ {\texttt{out}} = \mathbf{W} \mathbf{x}_ {\texttt{in}}, \qquad \mathbf{g}_ {\texttt{in}} = \mathbf{g}_ {\texttt{out}} \cdot \mathbf{W}, \qquad \frac{\partial J}{\partial \mathbf{W}} = \mathbf{x}_ {\texttt{in}} \cdot \mathbf{g}_ {\texttt{out}}$$

## Backprop through a whole MLP

Slide 50 puts the pieces together (≈48:58): $\mathbf{x}$ (4 entries) through
$\mathbf{z} = \mathbf{W}_ 1 \mathbf{x}$ with $\mathbf{W}_ 1$ of size $3 \times 4$, then
$\mathbf{h} = \texttt{relu}(\mathbf{z})$, then $\hat{\mathbf{y}} = \mathbf{W}_ 2 \mathbf{h}$ with
$\mathbf{W}_ 2$ of size $2 \times 3$, and finally an L2 loss
$J = \lVert \hat{\mathbf{y}} - \mathbf{y} \rVert_ 2^2$ against the target $\mathbf{y}$.

![Slide 50: an MLP as linear → relu → linear → L2 loss → J, with W1 (3×4) and W2 (2×3) drawn as blocks under the linear boxes](../raw/images/02-how-to-train-a-neural-net/slide-50.png)

*Slide 50 — the small MLP whose backward pass the next slide derives.*

For the backward pass the lecture switches convention: "instead of representing gradients as row
vectors, we're going to transpose them and treat them as column vectors" (≈49:44). Since
$(AB)^{\mathsf{T}} = B^{\mathsf{T}} A^{\mathsf{T}}$ (slide 51),

![Slide 51: the MLP's backward pass from right to left, with transposed weight matrices under the linear boxes and, under relu, a 3×3 diagonal matrix with a, b, c on the diagonal and zeros elsewhere](../raw/images/02-how-to-train-a-neural-net/slide-51.png)

*Slide 51 — backward through a linear layer multiplies by W transposed; backward through a ReLU multiplies by a diagonal gate.*

$$\mathbf{g}_ {\texttt{in}}^{\mathsf{T}} = (\mathbf{g}_ {\texttt{out}} \mathbf{W})^{\mathsf{T}} = \mathbf{W}^{\mathsf{T}} \mathbf{g}_ {\texttt{out}}^{\mathsf{T}}$$

and this "reveals this interesting connection between forward and backward. So backward for a
linear layer is the same operation as forward but with the weights transposed" (≈49:44). The
backward chain on slide 51 runs from the loss to the input:

$$\mathbf{g}_ 1^{\mathsf{T}} = 2(\hat{\mathbf{y}} - \mathbf{y}) \cdot 1, \qquad \mathbf{g}_ 2^{\mathsf{T}} = \mathbf{W}_ 2^{\mathsf{T}} \mathbf{g}_ 1^{\mathsf{T}}, \qquad \mathbf{g}_ 3^{\mathsf{T}} = \mathbf{H}'^{\mathsf{T}} \mathbf{g}_ 2^{\mathsf{T}}, \qquad \mathbf{g}_ 4^{\mathsf{T}} = \mathbf{W}_ 1^{\mathsf{T}} \mathbf{g}_ 3^{\mathsf{T}}$$

**The ReLU becomes a gate.** "A ReLU wouldn't be a ReLU on the backwards pass", because "you don't
want to re-pass them through ReLU" (≈49:44–50:31). Instead it becomes "roughly like a gating
matrix": $\mathbf{H}'^{\mathsf{T}}$ on slide 51 is a $3 \times 3$ diagonal matrix with entries
$a, b, c$ and zeros elsewhere, "parameterized by … the activations from that forward pass". Its
job is "making sure that you're not passing gradients for the components … that were in that 0
part of the ReLU. You don't want to send gradients for parts that should be masked out" (≈50:31).
It is diagonal "so that it's operating on each element independently" (≈51:18). This fits
lecture 1's ReLU derivative, 0 for negative input and 1 otherwise
([activation functions](activation-functions.md)).

**So the backward pass is linear.** "Backprop is still a linear model. Because even that ReLU,
this is still a linear operation" (≈51:18). The lecturer passes on an intuition from co-instructor
Phillip Isola: however curved the loss landscape, a first-order method fits a plane to it at the
current point "and then we're moving the direction of the plane. So by definition, it must be
linear" (≈51:18–52:05).

Slide 52 draws a complete iteration as one network. The forward half is
linear–relu–linear–relu–linear into an L2 loss, with
$\mathbf{W}_ 1$, $\mathbf{W}_ 2$ and $\mathbf{W}_ 3$ feeding the linear boxes. The backward half is six boxes all labelled "linear", fed
$\mathbf{W}_ 3^{\mathsf{T}}, \mathbf{W}_ 2^{\mathsf{T}}, \mathbf{W}_ 1^{\mathsf{T}}$, and ending in
$(\partial J / \partial \mathbf{x})^{\mathsf{T}}$. Three further linear boxes combine forward
activations with backward gradients to output $\partial J / \partial \mathbf{W}_ 1$,
$\partial J / \partial \mathbf{W}_ 2$ and $\partial J / \partial \mathbf{W}_ 3$. Thick lines carry
values from the loss and from both ReLUs into the backward half. The legend colours parameters
forward blue, parameters backward yellow, data forward green and data backward red. The whole
thing "can be thought of as a single forward pass, in a way, even though the actual forward pass
is only the first half, and the backward pass is the second half" (≈52:05).

![Slide 52: one training iteration drawn as a single network — forward linear and relu boxes into an L2 loss, a backward chain of six linear boxes fed the transposed weights, and three linear boxes that output the gradients for W1, W2 and W3](../raw/images/02-how-to-train-a-neural-net/slide-52.png)

*Slide 52 — forward and backward together are one big computation; the thick lines carry forward values into the backward half.*

That reuse is also where forward and backward differ. A student observed that the forward pass can
discard each activation once the next layer has used it, while backprop must keep them all. "Yes.
So you do need to save the intermediate representation so that you can efficiently compute back
propagation", and "there's this asymmetry in where the information is flowing" (≈52:51–53:36).
For a ReLU, "you basically need to save off what the value of the activations were for each
component of your input" to build the gating matrix (≈54:22). The slides leave out bias terms
"just for simplicity … in practice, there would be bias terms floating around everywhere"
(≈53:36).

## Backprop through any DAG: merge and branch

Real networks are not chains: they share weights, split, and concatenate outputs "from different
heads or being split into different objectives" (≈54:22–55:09). "There's actually only two
operations that you need to make everything we just talked about work for any arbitrary
directed-acyclic graph. And those are merging and branching" (slide 53, ≈55:09).

![Slide 53: merge — two inputs combined on the way forward and the gradient split back to each; branch — one input copied on the way forward and the two incoming gradients summed on the way back](../raw/images/02-how-to-train-a-neural-net/slide-53.png)

*Slide 53 — the only two rules needed to backpropagate through any DAG.*

**Merge.** Two values $\mathbf{x}^a$ and $\mathbf{x}^b$ are combined — "it could be addition. It
could be concatenation, what have you" — and pass forward as $[\mathbf{x}^a, \mathbf{x}^b]$.
Backward, the incoming gradient $\partial J / \partial [\mathbf{x}^a, \mathbf{x}^b]$ is split, and
each input gets only its own part, $\partial J / \partial \mathbf{x}^a$ or
$\partial J / \partial \mathbf{x}^b$: "you just need to make sure that you track the gradient with
respect to the correct input variables" (≈55:09–55:55).

**Branch.** One value is sent two ways, $\mathbf{x}^a = \mathbf{x}$ and $\mathbf{x}^b = \mathbf{x}$.
Backward, "it's as simple as a sum. You just take the gradients from both, and you sum them":

$$\frac{\partial J}{\partial \mathbf{x}} = \sum_i \frac{\partial J}{\partial \mathbf{x}^{i}}$$

The superscripts are deliberately open-ended: a branch "could literally just be a duplicate …
Or it could be some splitting of the embedding vector … any differentiable branching operation"
(≈1:18:35). Asked *why* branching sums, the lecturer deferred the explanation to office hours
(≈1:14:38–1:15:24), and the lecture points to "a much more detailed derivation in the lecture
notes" (≈55:55). Asked how backprop for an MLP differs from backprop for a DAG: "there's no
difference" — a DAG is "just chains, where sometimes they merge, and sometimes they branch"
(≈1:13:07–1:14:38).

**Parameter sharing** is a branch in disguise (slides 54–55, ≈56:42). In a chain where every layer
has its own parameters $\theta_1, \theta_2, \theta_3$, each gets its own gradient. If the layers
instead share one $\theta$ — "maybe you're reusing parameters that were pre-trained on ImageNet or
something, and you're using them in multiple parts of your network" — then $\theta$ branches to
every place it is used, so its gradient is the sum of the gradients from each use. The slide puts
it in four words: "Parameter sharing —> sum gradients".

![Slide 55: three layers sharing one parameter θ, beside a branch diagram in which θ is copied to two uses and their gradients are summed, captioned "Parameter sharing —> sum gradients"](../raw/images/02-how-to-train-a-neural-net/slide-55.png)

*Slide 55 — a shared parameter is a branch, so its gradient is the sum over its uses.*

The complete treatment is on the [backpropagation](backpropagation.md) page.

## Differentiable programming

The last section is a "meta concept": "why neural network training is, essentially, just
differentiable programming" (≈57:27). Slide 56 sets a deep-learning layer $f(\mathbf{x}_ {\texttt{in}}, \mathbf{W})$
beside a differentiable-programming one, $f(\mathbf{x}_ {\texttt{in}}, \theta)$, with the same
forward and backward arrows. The perspective is the same, "but now we basically just are actually
programming that model". PyTorch, TensorFlow ("if anyone still uses it") and JAX "are all
basically libraries that enable us to really efficiently do differentiable programming"
(≈57:27–58:16).

Slide 57 gives two reasons deep nets are popular: they are "easy to optimize (differentiable)"
and "compositional", which the slide calls "block based programming". "An emerging term for
general models with these properties is **differentiable programming**" (≈58:16). It quotes Yann
LeCun ("Deep Learning est mort. Vive Differentiable Programming!") and Thomas Dietterich ("DL is
essentially a new style of programming … and the field is trying to work out the reusable
constructs in this style. We have some: convolution, pooling, LSTM, GAN, VAE, memory units,
routing units, etc."). The lecturer finds the tweets "a bit old now, but I still think they're
funny" (≈58:16–59:02).

Slide 58 is an example of a model built from mixed parts: **Neural Module Networks** (Andreas et
al., 2017 on the slide). A question such as "Where is the dog?" goes to a parser that "might not
be learned at all. It might just be a standard parser", which assembles modules such as `where`
and `dog`. A CNN, whose weights are learned from data, reads the image, and the network answers
"couch" (≈59:02–59:48).

Slide 59 is Andrej Karpathy's **Software 2.0** picture. In the space of all programs, Software 1.0
— explicitly written code — is "just that little, red dot", while "Software 2.0 is actually
defining a space of possible software systems, and then you're actually optimizing within that
space to build the system that you need to solve your problem" (≈59:48).

So a real system mixes nodes **programmed by a human** with nodes **programmed by backprop**
(slide 60, ≈1:00:38). The lecturer's example of the first kind: "we often explicitly program the
normalization values that we want to use for natural images directly into the pre-processing of
our data. … We don't learn what those normalization values should be." The second kind are
"programmed by tuning behavior to match training examples".

![Slide 60: the eight-node DAG with seven grey boxes and one beige, annotated "Programmed by a human" and "Programmed by backprop"](../raw/images/02-how-to-train-a-neural-net/slide-60.png)

*Slide 60 — a real system mixes hand-written nodes with nodes tuned by backprop.*

Which parts should a human write? "Anytime a human is programming part of this system, it's
constraining the system. And that constraint can be useful, but it could also be unuseful"
(≈1:02:10). Her example is the era of hand-built feature engineering feeding very simple models:
"our best ideas were not as good as just a much more complex, larger models that were learned
somewhat end-to-end". The jury is still out, and the one thing a human always defines is the cost
(≈1:02:10–1:03:45). The boundary moves in practice, too. Switching what is optimized "is just a
matter of defining … where you want to freeze your gradients", and a human-programmed part "might
have just actually previously been programmed by backprop", as when pretrained weights are plugged
in (≈1:10:48–1:12:21).

A student asked what happens when a forward pass calls an operation that is not a PyTorch
operation (≈1:15:24–1:18:35). To sit inside the network an operation has to be a torch operation
with a gradient. Otherwise, put it in pre-processing — "the pre-processing steps don't necessarily
need to be differentiable, things like data augmentation". The lecturer has built statistical
models (an "occupancy model of species") into networks "with varying levels of success. And often
it's just like, OK, it's technically possible, but it's intractable to actually learn"
(≈1:16:12–1:18:35). See [differentiable programming](differentiable-programming.md).

## Optimizing anything with respect to anything

"Backprop lets you optimize any node (function) or edge (variable) in your computation graph
w.r.t. to any scalar cost" (slides 61–63, ≈1:01:23). Slide 63 highlights two choices: how the cost
changes when a node's weights change, and how it changes "when the input data changes". "You don't
actually need to specifically do this for weights" (≈1:01:23).

![Slide 63: a DAG with a red cost node at the top, one inner node and one input edge highlighted in yellow, and the derivative of the cost with respect to each](../raw/images/02-how-to-train-a-neural-net/slide-63.png)

*Slide 63 — backprop can differentiate the cost with respect to any node's weights, or with respect to the input data itself.*

That turns training around. The usual gradient $\partial J / \partial \theta$ asks "how much the
total cost is increased or decreased by changing the parameters" (slide 64). Holding the
parameters fixed, $\partial y_j / \partial \mathbf{x}$ asks "how much the 'chameleon' score is
increased or decreased by changing the image pixels" (slide 65, ≈1:03:45). Which output to push on
matters: "the maybe easiest way to increase the probability softmax given to a class is often to
make the alternatives unlikely, rather than to make the class of interest likely. Whereas if you
optimize pre softmax logits, this tends to actually be a bit more stable" (≈1:04:32). See
[softmax and cross-entropy](softmax-and-cross-entropy.md).

**Unit visualization** (slide 66) finds an image that maximizes one output neuron by gradient
*ascent* on the input:

![Slide 66: the unit-visualization objective and its gradient-ascent update beside a synthetic image that maximizes the "cat" output — a dense collage of cat faces and fur](../raw/images/02-how-to-train-a-neural-net/slide-66.jpg)

*Slide 66 — what a trained classifier finds most cat-like (Olah et al., Distill 2017, CC BY).*

$$\arg\max_{\mathbf{x}} \thinspace y_j + \lambda R(\mathbf{x}), \qquad \mathbf{x}^{k+1} \leftarrow \mathbf{x}^{k} + \eta \left.\frac{\partial (y_j(\mathbf{x}) + \lambda R(\mathbf{x}))}{\partial \mathbf{x}}\right|_ {\mathbf{x}=\mathbf{x}^{k}}$$

Here $y_j$ is the output score of class $j$. The slide does not define the extra term
$\lambda R(\mathbf{x})$, and the lecture does not discuss it. Maximizing the "cat" neuron gives a
dense collage of cat faces and fur, which is "what a given trained model thinks is most cat like"
(≈1:04:32). Maximizing a hidden neuron $j$ in layer $l$ instead (slide 67) is "a mechanism to probe
what the model is paying attention to" (≈1:05:19). Both images are from Olah et al.'s Distill
article on feature visualization (2017). **DeepDream** (slide 68) is the same idea, producing
images the lecturer calls "psychedelic and beautiful" (≈1:05:19).

![Slide 67: the same objective for a hidden neuron, beside a synthetic image of a repeating network-like texture in orange, yellow, blue and black](../raw/images/02-how-to-train-a-neural-net/slide-67.jpg)

*Slide 67 — maximizing one hidden unit shows the pattern it responds to (Olah et al., Distill 2017, CC BY).*

![Slide 68: a collage of six DeepDream images — landscapes, arches and pagodas dissolving into repeated temple-like textures](../raw/images/02-how-to-train-a-neural-net/slide-68.jpg)

*Slide 68 — DeepDream, made with a network trained on Places by MIT CSAIL.*

**CLIP** trains a text encoder and an image encoder so that "things that are similar,
semantically, from text to images, are quite close together in the learned embedding space"
(slide 69, ≈1:06:05). Its training grid scores every image in a batch against every caption, and
the matching pairs lie on the diagonal. Asked what an embedding is, the lecturer described an
encoder mapping a high-dimensional input "to a low-dimensional representation or a
low-dimensional embedding", for example "a vector of length 2048 … or 1024. Those are both common
embedding sizes" (≈1:09:15–1:10:02).

**CLIP+GAN** (slide 70, ≈1:06:05–1:06:52) chains three trained networks: CLIP's text encoder
turns a prompt — on the slide, "What is the answer to the ultimate question of life, the universe,
and everything?" — into an embedding $\mathbf{e}_ 1$; an image generator turns a latent input
$\mathbf{z}$ into an image; and CLIP's image encoder turns that image into $\mathbf{e}_ 2$. The
slide marks $\mathbf{z}$ "Optimize this" and $\mathbf{e}_ 1 \cdot \mathbf{e}_ 2$ "To maximize
this". Every network weight stays fixed: "the only thing you're optimizing is this hidden
parameter that's going to be the input to the image generator", although the gradient still has
to be backpropagated through the frozen networks to reach $\mathbf{z}$ (≈1:07:39–1:09:15). Asked
whether that product is an attention block, the lecturer called it "a cosine similarity", a
component of attention. A full attention block also has the projections into the shared space and
out of it (≈1:12:21). The takeaway: "all these trapezoids are neural networks. You can plug them
together. You can take the components trained in one way and use them in another way. Really, the
idea is you can optimize modules with respect to all the other modules, and the world's your
oyster" (≈1:06:52).

![Slide 70: CLIP+GAN — a text prompt through CLIP's text encoder to e1, a latent z through an image generator and CLIP's image encoder to e2, with z marked "Optimize this" and the product of e1 and e2 marked "To maximize this"](../raw/images/02-how-to-train-a-neural-net/slide-70.jpg)

*Slide 70 — text-to-image by optimizing only the generator's input; every network weight stays fixed.*

## Worked example: one iteration of backpropagation

The last section of the deck is an exercise with its solution: "run one iteration of back
propagation" (slide 72). The recording does not go through it (see the top of this page).

![Slide 72: a five-node network — inputs 1 and 2, tanh nodes 3 and 4, a linear output node 5 — with weights 1, 0.2, −3 and 1 into the hidden layer and 1, −1 into the output, plus the learning rate, loss and training example](../raw/images/02-how-to-train-a-neural-net/slide-72.png)

*Slide 72 — the exercise: one iteration of backpropagation on this network.*

**The network** (slide 72). Two input nodes, 1 and 2; two hidden nodes, 3 and 4, each applying
$\tanh$; and one linear output node, 5. The weights are $w_{13} = 1$, $w_{14} = 0.2$,
$w_{23} = -3$, $w_{24} = 1$, $w_{35} = 1$ and $w_{45} = -1$. The loss is Euclidean, the one
training example has input $(1.0, 0.1)$ and desired output $0.5$, and the learning rate is
"η = -0.2 (because we used positive increments)".

**In block notation** (slide 75):
$\mathbf{x}_ 0 \to \mathbf{W}_ 0 \mathbf{x}_ 0 = \mathbf{x}_ 1 \to \tanh(\mathbf{x}_ 1) = \mathbf{x}_ 2 \to \mathbf{W}_ 1 \mathbf{x}_ 2 = \mathbf{x}_ 3$,
with loss $\mathcal{L} = \frac{1}{2} \lVert \mathbf{x}_ 3 - \mathbf{y} \rVert_ 2^2$. (Slide 75
prints $\mathbf{x}_ 2$ inside the norm; every later slide uses $\mathbf{x}_ 3$.) The goal is the
two updates of slide 76,

![Slide 75: the same network as a chain of blocks W0x0 → tanh → W1x2 → loss, each block listing the derivatives it must supply, with gradient arrows running back underneath](../raw/images/02-how-to-train-a-neural-net/slide-75.png)

*Slide 75 — the worked example in the lecture's block notation.*

$$\mathbf{W}_ 0^{k+1} = \mathbf{W}_ 0^{k} + \eta \left(\frac{\partial \mathcal{L}}{\partial \mathbf{W}_ 0}\right)^{\mathsf{T}}, \qquad \mathbf{W}_ 1^{k+1} = \mathbf{W}_ 1^{k} + \eta \left(\frac{\partial \mathcal{L}}{\partial \mathbf{W}_ 1}\right)^{\mathsf{T}}$$

starting from

$$\mathbf{W}_ 0^{k} = \begin{pmatrix} 1 & -3 \cr 0.2 & 1 \end{pmatrix}, \qquad \mathbf{W}_ 1^{k} = \begin{pmatrix} 1 & -1 \end{pmatrix}$$

Row 1 of $\mathbf{W}_ 0$ holds the weights into node 3 ($w_{13}, w_{23}$), and row 2 those into
node 4 ($w_{14}, w_{24}$).

**The equations** (slide 77), derived "working *backwards*" by the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_ 3} = \mathbf{x}_ 3 - \mathbf{y}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{x}_ 2} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_ 3} \mathbf{W}_ 1, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{x}_ 1} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_ 2} (1 - \tanh^2(\mathbf{x}_ 1))$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_ 0} = \mathbf{x}_ 0 \frac{\partial \mathcal{L}}{\partial \mathbf{x}_ 1}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{W}_ 1} = \mathbf{x}_ 2 \frac{\partial \mathcal{L}}{\partial \mathbf{x}_ 3}$$

The slide warns about the order in the last two: "The notation hides the details but you can
write out all the indices to see that this is the correct ordering — or just check that the
dimensions work out."

**Forward pass** (slide 78), with $\mathbf{x}_ 0 = (1.0, 0.1)^{\mathsf{T}}$ and $\mathbf{y} = 0.5$:

$$\mathbf{x}_ 1 = \begin{pmatrix} 0.7 \cr 0.3 \end{pmatrix}, \qquad \mathbf{x}_ 2 = \tanh(\mathbf{x}_ 1) = \begin{pmatrix} 0.604 \cr 0.291 \end{pmatrix}, \qquad \mathbf{x}_ 3 = 0.313, \qquad \mathcal{L} = \frac{1}{2}(\mathbf{x}_ 3 - \mathbf{y})^2 = 0.017$$

**Backward pass** (slide 79):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_ 3} = -0.1869, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{x}_ 2} = \begin{pmatrix} -0.1869 & 0.1869 \end{pmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_ 1} = \begin{pmatrix} -0.1869 & 0.1869 \end{pmatrix} \begin{pmatrix} 1 - \tanh^2(0.7) & 0 \cr 0 & 1 - \tanh^2(0.3) \end{pmatrix} = \begin{pmatrix} -0.1186 & 0.171 \end{pmatrix}$$

The matrix is diagonal "because tanh is a pointwise operation" — the same reason the ReLU's
backward matrix was diagonal on slide 51. Then

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_ 0} = \begin{pmatrix} 1.0 \cr 0.1 \end{pmatrix} \begin{pmatrix} -0.1186 & 0.171 \end{pmatrix} = \begin{pmatrix} -0.1186 & 0.171 \cr -0.01186 & 0.0171 \end{pmatrix}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{W}_ 1} = \begin{pmatrix} 0.604 \cr 0.291 \end{pmatrix} (-0.1869) = \begin{pmatrix} -0.113 \cr -0.054 \end{pmatrix}$$

**A misprint to know about.** Slide 79's last line prints the multiplier as $-0.1186$, the
first entry of $\partial \mathcal{L} / \partial \mathbf{x}_ 1$. The factor the equation calls for is
$\partial \mathcal{L} / \partial \mathbf{x}_ 3 = -0.1869$, and the printed result,
$(-0.113, -0.054)$, is what $-0.1869$ gives ($0.604 \times -0.1869 \approx -0.113$). The version
above uses $-0.1869$.

**Updates** (slide 80), with $\eta = -0.2$:

$$\mathbf{W}_ 0^{k+1} = \begin{pmatrix} 1.02 & -3.0 \cr 0.17 & 1.0 \end{pmatrix}, \qquad \mathbf{W}_ 1^{k+1} = \begin{pmatrix} 1.02 & -0.989 \end{pmatrix}$$

The middle line of slide 80 shows the $\mathbf{W}_ 0$ gradient untransposed, but the result is
the transposed update the formula specifies: $w_{14}$ moves from 0.2 to 0.17, using the 0.171 from
the gradient's top-right entry. Slide 73 redraws the network with the new weights, "rounding to
two digits": $w_{13} = 1.02$, $w_{14} = 0.17$, $w_{23} = -3.0$, $w_{24} = 1.0$, $w_{35} = 1.02$
and $w_{45} = -0.99$.

![Slide 73: the same five-node network with its weights after one iteration: 1.02, 0.17, −3.0 and 1.0 into the hidden layer, 1.02 and −0.99 into the output](../raw/images/02-how-to-train-a-neural-net/slide-73.jpg)

*Slide 73 — the answer, rounded to two digits.*

## Closing

The recap (slide 71, ≈1:07:39) repeats the agenda: gradient descent and SGD, computation graphs,
backprop through chains, MLPs and DAGs, and differentiable programming. The rest of the recording
is questions, folded into the sections above. The last is about the problem set, which "will be
there soon" (≈1:18:35).
