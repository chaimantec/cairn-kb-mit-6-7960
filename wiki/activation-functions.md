# Activation functions (non-linearities)

An activation function $g$ is the **pointwise non-linearity** applied after each linear layer of a
neural network: $h = g(z)$, applied to each component of $\mathbf{z}$ separately. Without it a
stack of linear layers collapses to a single linear map (see
[multilayer perceptrons](multilayer-perceptron.md)). The course treats the common choices as
background (slide 31, "MLPs, Nonlinearities (ReLu)"). Covered so far:
[lecture 1](01-introduction.md), slides 36–41, ≈28:01–38:49;
[lecture 2](02-how-to-train-a-neural-net.md), slides 23–25 and 51, ≈20:13–25:43 and ≈49:44–51:18
(GELU, the continuous–differentiable–smooth criterion, and the ReLU on the backward pass);
[lecture 3](03-approximation-theory.md), slides 15, 27–29 and 32 (what sums and compositions of
ReLUs can build); [lecture 4](04-architectures-grids.md), slides 7–10, ≈7:40–13:04 (sine activations
as an inductive bias); [lecture 10](10-architectures-memory.md), slides 36–39, ≈36:29–37:15 and ≈41:53 (sigmoid and tanh inside the LSTM);
[lecture 11](11-representation-learning-reconstruction-based.md), slides 8–9, ≈7:46–10:49 (the ReLU and sigmoid as maps of a distribution).

## The four in lecture 1

| | Formula | Range | Gradient behaviour | Verdict in lecture 1 |
| --- | --- | --- | --- | --- |
| Step | $1$ if $z \gt 0$, else $0$ | $\lbrace 0, 1 \rbrace$ | zero almost everywhere | not differentiable, so it cannot be trained by gradients |
| tanh | $\frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ | $[-1, 1]$ | vanishes for large positive or negative $z$ | bounded and zero-centred, but saturates |
| Sigmoid | $\frac{1}{1 + e^{-z}}$ | $[0, 1]$ | vanishes for large positive or negative $z$ | "not used in practice" |
| ReLU | $\max(0, z)$ | $[0, \infty)$ | 0 for $z \lt 0$, 1 for $z \gt 0$ | "our default choice" |

### Step (slides 36–37)

The step function turns a linear unit into Rosenblatt's perceptron. It is the wrong choice for
learning, as the class itself points out (≈28:01–28:47). It is not differentiable, which "hinders
backpropagation", and away from $z = 0$ its gradient is zero: "If you're anywhere on this graph,
you don't know which way to go". A gradient step simply does not move.

### tanh (slides 38–39)

$$g(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$

One of the earliest non-linearities tried (≈31:56). What is good about it: it is bounded between
−1 and 1, so "none of the values are super huge", and its outputs are centred at 0. What is bad:
it **saturates**. For very large or very small inputs the curve is flat, the gradient goes to 0,
and a unit that starts "really far from the center" gets little training signal (≈32:42). Slide
39 also gives its relation to the sigmoid $\sigma$:

$$\tanh(z) = 2\thinspace\sigma(2z) - 1$$

### Sigmoid (slide 40)

$$g(z) = \frac{1}{1 + e^{-z}}$$

Historically read as the **firing rate of a neuron** (≈33:28). It is bounded between 0 and 1, so
never negative, and it saturates exactly as tanh does. Its outputs are centred at 0.5 rather than
0, which the slide calls "poor conditioning" and the lecturer "slightly biased" — "In practice,
we don't actually use this." **Note:** slide 40 prints the formula with $h$ in the exponent,
$1/(1 + e^{-h})$. The lecturer corrects it in the lecture: "the notation is wrong. It should be
minus z" (≈32:42). The formula above is the corrected one.

### ReLU (slide 41)

$$g(z) = \max(0, z)$$

The **rectified linear unit** (≈33:28–34:59):

- **Unbounded on the positive side**, so values can grow large, which "can result in … exploding
  gradients".
- **Efficient to implement**: the derivative is just a step (given below the list).
- **Seems to help convergence**: the AlexNet paper (Krizhevsky et al.) reported "something like a
  6x speed-up using a ReLU over using something like a tanh".
- **Drawback: dead units.** "If you're strongly in the negative region, the unit's what we call
  dead. There is no gradient", and that component of the vector stops learning.
- **The default**: "widely used in current models", with "lots of slight tweaks to this".

The derivative, which is what makes ReLU cheap:

$$\frac{\partial g}{\partial z} = \begin{cases} 0 & \text{if } z \lt 0 \cr 1 & \text{if } z \geq 0 \end{cases}$$

## GELU, and what makes a non-linearity easy to train through (lecture 2)

Lecture 2 gives three properties that make a function easier to optimize through: everywhere
**continuous**, everywhere **differentiable**, everywhere **smooth** (slide 23, ≈20:13). The slide
is titled "What is important in a loss function?", but its examples are activation functions, as
the lecturer confirmed when asked (≈24:09). ReLU passes the first, passes the second "(Almost!)",
and fails the third because of "this kink" at 0 (slide 24).

The **Gaussian error linear unit** passes all three (slide 25):

$$\mathrm{GELU}(z) = z \ast \Phi(z)$$

Here $\Phi$ is "the cumulative distribution function for the Gaussian distribution" (≈21:00), which
the slide leaves undefined, citing arXiv 1606.08415. Its curve is close to zero for negative $z$,
dips slightly below zero just left of the origin, and approaches the line $z$ for positive inputs.
The lecture's claim is about a direction of travel, not a ranking: "even if we don't precisely know
experimentally or theoretically, which properties are needed for training neural networks, trends do
seem to be moving towards functions which satisfy all three", and "I'm not saying here that GeLU is
much better than ReLU all the time" (≈21:00–23:19). Asked about GELU's dip, which makes it
non-monotonic: "Is it better to be smooth or to be monotonic? And that's where I think the jury's a
little bit still out" (≈24:57).

**ReLU on the backward pass.** In backpropagation a ReLU becomes a diagonal "gating matrix" built
from the forward activations, which passes the gradient for components where the ReLU was active
and blocks it for those that "were in that 0 part of the ReLU" (lecture 2, slide 51, ≈50:31–51:18).
That is the derivative above at work: 1 where $z \gt 0$, 0 where $z \lt 0$. See
[backpropagation](backpropagation.md). For the wider picture of which functions are hard to
optimize, see [loss landscapes](loss-landscapes.md).

## What ReLUs can build (lecture 3)

Lecture 3 uses the ReLU as a building block for approximation, and three of its constructions are
worth knowing.

**Four ReLUs make a rectangle** (slide 15). With a constant $c$ scaling the slopes,

$$f_c(x) = \text{relu}(cx) - \text{relu}(cx - 1) - \text{relu}(c(x-1) - 2) + \text{relu}(c(x-1) - 3),$$

which is the slide's vector form written out. The first two ReLUs make a ramp up to height 1, the
last two a ramp back down, and as $c \to \infty$ the ramps become vertical, leaving a rectangle that
is 1 on $[0, 1]$ and 0 elsewhere. The lecturer built it live on a graphing website: with $c = 1$ it
is a trapezoid, and raising $c$ "is making the slopes slopier" (≈33:27–35:01).

**A ReLU thresholds a sum** (slide 16). Adding $d$ one-dimensional rectangles, one per axis, gives a
surface that exceeds $d - 1$ only where all of them are on, so $\text{relu}(\text{sum} - (d-1))$ keeps
just the $d$-dimensional box.

**A ReLU at most doubles the kinks** (slide 29). Every ReLU network is piecewise linear (slide 27),
and applying a ReLU to a piecewise linear function can split each linear piece in two where it
crosses zero. In the slide's example a function with five kinks becomes one with nine. The triangle
map $g(x) = \text{relu}[2 \cdot \text{relu}(x) - 4 \cdot \text{relu}(x - \tfrac{1}{2})]$ of slide 32
doubles its linear regions every time it is composed with itself. This is the engine of lecture 3's
depth-separation result; see [representational power](representational-power.md#depth-separation-lecture-3).

## Sine activations as an inductive bias (lecture 4)

Lecture 4 compares activations not for trainability but for what they assume about the function
(slides 7–10, ≈7:40–12:17). Fitted to a few points of a wiggly one-dimensional function, a 5-layer
ReLU network fits the data and goes flat or straight beyond it, while a 5-layer **sin-net**, with
sinusoidal activations (the SIREN of Sitzmann et al., 2020), starts "seeing some periodicity that
maybe starts to match the periodicity in the function." Asked why its output is periodic, a student
answered that it is built from sines, and the lecturer agreed: "we've just added an inductive bias in
our model architecture that says that the distribution should be periodic." On images, SIREN fits a
photograph faster than ReLU or tanh networks, because "a Fourier basis is a good basis for
representing images, and the fundamental building blocks of Fourier basis are sinusoids" (slide 9,
≈10:45). This is the exception lecture 1's answer below allows for: an activation chosen to match
known structure in the data. See [inductive bias](inductive-bias.md) and
[neural fields and positional encoding](neural-fields-and-positional-encoding.md).

## Sigmoid and tanh in the LSTM (lecture 10)

Lecture 10's LSTM uses the sigmoid as a gate. Its forget, input and output gates are sigmoids of a linear function of the
previous hidden state and the input, so each entry lies between 0 and 1 and multiplies the cell state element-wise, about 1
to remember and about 0 to forget (slides 36–39). "The reason you use a sigmoid function here is because it's bounded
between 0 and 1. So you want large values to map to 1 and small values to map to 0" (≈41:53); a ReLU "would be unbounded"
(≈37:15). The candidate values and the output go through tanh (slides 37 and 39). See [recurrent neural networks](recurrent-neural-networks.md).

## What a non-linearity does to a distribution (lecture 11)

Lecture 11 draws functions not as graphs but as maps from input points to output points, which shows what a layer does to a whole
distribution of data (slides 7–9). On a line (slide 8), a ReLU "takes all of your data in the negative half space and maps it to
0", so "You'll usually have a spike of density at 0", and a sigmoid "will map most of your inputs to either 0 or 1, but in a soft
way" (≈7:46). In two dimensions (slide 9), the ReLU "maps all data to the positive orthant", and "most of the points get mapped to
these axes. You get this what's called a sparse representation": in each dimension about half the values are negative and become
0, and everything in the strictly negative orthant lands on the origin. "And in high dimensions, I think this effect becomes even
more extreme" (≈10:02–10:49). See [representation learning](representation-learning.md).

## How do you choose one?

A student asked whether the application should dictate the activation (≈36:31). The lecturer's
answer is that there are no reliable rules of thumb. Fields converge on whatever has worked
experimentally and build on it — "sometimes that's called grad student gradient descent". Jeremy
Bernstein added that it is "not a hard science", and that the choice is usually about making the
network **trainable** rather than about modelling a particular kind of data (≈38:02). The
exception the lecturer gives is **known structure**: if you know your features are sinusoidal,
working in Fourier space with a sinusoidal activation may make sense. She closes by calling this
an open problem that needs "a lot more research on the theoretical side" (≈38:49).
