# Lecture 1 — Introduction to Deep Learning

**Lecturer:** Sara Beery, with co-instructor Jeremy Bernstein speaking from the audience twice ·
**Video:** [youtube.com/watch?v=6FkRvTtUc-o](https://www.youtube.com/watch?v=6FkRvTtUc-o) (61 min) ·
**Slides:** [`mit6_7960_f24_lec1.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec1.pdf)
(81 pages; transcribed slide by slide in [`raw/slides/01-introduction.md`](../raw/slides/01-introduction.md)) ·
**Transcript:** [`raw/transcripts/01-introduction.md`](../raw/transcripts/01-introduction.md)

## What this lecture establishes

The lecture defines deep learning as two things working together — **neural networks**, built from
stacks of linear maps interleaved with pointwise non-linearities, and **differentiable
programming**, where parts of a program are parameterized and tuned by gradient-based
optimization. It then does three jobs: a short history of neural networks told as a curve of
enthusiasm over time; a review of what the course *assumes you already know* (gradient descent,
MLPs and non-linearities, softmax and cross-entropy, batching and tensors); and a preview of what
the course *will teach* (backprop, approximation, architectures, generalization, representation
learning, generative models, transfer, scale). The lecturer is explicit that this is "not an intro
to deep learning class. This is advanced graduate-level deep learning" (≈5:24) — so the review
half is a checklist to brush up on, not a tutorial.

The course logistics, policies and full schedule are on the [course map](course-map.md).

## What deep learning is

Slide 2 gives the definition the course works from (≈0:49–1:35):

1. **Neural nets** — "a class of machine learning architectures that use stacks of linear
   transformations interleaved with pointwise nonlinearities". The lecturer calls them "a
   building block for a lot of the progress" in the field.
2. **Differentiable programming** — "a programming paradigm where [we] parameterize parts of the
   program and let gradient-based optimization tune the parameters". Gradient descent then finds
   "at least, local optimal" settings for that program (≈1:35).

The course philosophy (slide 3, ≈1:35–2:20) is that breakthroughs have come from "a mixture of
theory and practice", so the course offers both "theoretical grounding in important deep learning
building blocks" and "practice implementing, understanding, and using those blocks".

## Logistics in brief

Grading is **65% problem sets** (five, each one to two weeks, mixing pen-and-paper work with code)
and **35% a final project**, a research project proposed by the student and delivered as a
**blog post** "that's going to demonstrate novel experimentation and visualization", in groups of
at most two (≈2:20–3:54). The staff cannot promise compute, so projects should not depend on
reaching state of the art at scale: "You're not going to be able to outcompete OpenAI on this
research project" (≈4:39). The **AI-assistant policy** (slide 4, ≈10:00–11:34) is that AI
assistants are treated exactly like human collaborators — use them as discussion partners, never
to produce the answer, and say which AI you used and how at the top of the problem set. Details,
including the collaboration rules and the PyTorch tutorials, are on the
[course map](course-map.md#coursework-and-policies).

## Why deep learning (slide 5)

The goal offered is to "model complex phenomena in the real world" — natural language, images
and video, DNA, ecosystems, climate change. Why is that hard? "They're complex" (≈12:20). The
slide proposes the human brain as an "existence proof for deep learning as a solution", which the
lecturer frames as: everyone in the room models complex phenomena with a brain every day.

## A brief history, as a curve of enthusiasm

From ≈13:10 to ≈20:15 the lecture tells the history of neural networks by drawing a curve of
"enthusiasm" against "time", one event at a time (slides 7–23). The curve rises and falls twice
before the present peak:

**1958 — the perceptron** (slides 8–10, ≈13:58). Rosenblatt's perceptron, published in
*Psychological Review*, is "arguably" the first neural network. It was meant to categorize
images: combine the inputs (pixels), sum them, pass the sum through a non-linearity, and read
off a category. The lecturer stresses that this unit "still forms the building block for most of
the deep learning that we do". Enthusiasm peaks.

**1972 — Minsky and Papert's *Perceptrons*** (slides 11–12, ≈14:44). The book took "a very
critical lens" to the perceptron as a model of the brain and characterized mathematically what
it cannot represent — so carefully that "enthusiasm for machine learning really took a dip".

**1986 — *Parallel Distributed Processing*** (slides 13–15, ≈15:32). The PDP book introduced
backpropagation, which made *multilayer* perceptrons trainable by passing a gradient back
through several layers. That solves problems no single layer can, the classic one being
**XOR**: its two classes cannot be split by one line, so it needs "multiple layers of a
perceptron" (slide 14 gives the truth table: inputs $(0,0)$ and $(1,1)$ map to 0, $(1,0)$ and
$(0,1)$ to 1). Enthusiasm returns.

![Slide 14: the XOR truth table beside a 2×2 grid in which the two output-0 corners are shaded and the output-1 corners are white — no single line separates them.](../raw/images/01-introduction/slide-14.png)

*Slide 14 — XOR is the standard example of a function a single-layer network cannot represent.*

**1998 — LeCun's convolutional networks** (slide 16, ≈16:20), with the LeNet-5 architecture
and the demos on LeCun's web page.

**2000 — the AI winter** (slides 17–18, ≈17:07). At NeurIPS 2000 the title words most
predictive of *acceptance* were "belief propagation" and "Gaussian"; those most predictive of
*rejection* were "neural" and "network". The lecturer's diagnosis: "even though we had the
theory, we had the building blocks, we didn't actually have the ability to train them
efficiently. We didn't have the right programming perspective, and we didn't have the right
hardware" (≈17:54).

**2012 — AlexNet** (slides 19–21, ≈17:54–19:29). Krizhevsky, Sutskever and Hinton's network was
another convolutional net, but Krizhevsky "figured out how to program GPUs" — hardware built
for graphics and games — to train it. It beat "every single other possible method" on
ImageNet, "a shot heard around the world". The lecturer names three ingredients behind the
turn: the theory, the programming that let networks be fit efficiently, and — "the other really
vital component" — **data**: "Machine learning does not work without large-scale, curated,
labeled data in many capacities still" (≈19:29). ImageNet was "the first very large-scale data
set of its kind".

![Slide 21: the enthusiasm curve rising and falling twice — peaks at Perceptrons 1958 and PDP book 1986, troughs at Minsky and Papert 1972 and AI winter 2000 — and climbing toward Krizhevsky, Sutskever, Hinton 2012, with two "28 years" arrows marking the spacing.](../raw/images/01-introduction/slide-21.png)

*Slide 21 — the whole history on one curve: two booms 28 years apart, and a third beginning in 2012.*

The two booms are spaced about **28 years** apart (1958 → 1986 → 2012), which the slides mark
with two "28 years" arrows. Projecting the pattern forward lands on **2028** — four years after
the lecture — and the last two slides offer three futures (≈19:29–20:15): another bust on the
same cycle, which the lecturer thinks "probably not true"; enthusiasm going "off the deep end…
out in the stratosphere" (the steep line on slide 22); or "some new plane of enthusiasm", still
oscillating but at a higher level (slide 23).

![Slide 22: the same curve with two alternative futures after 2012 — a green curve dipping into another trough, and a steep blue line shooting off the top of the slide — and a red "2028 ?" on the time axis.](../raw/images/01-introduction/slide-22.png)

*Slide 22 — two of the three futures: another bust on the 28-year cycle, or enthusiasm going "out in the stratosphere".*

![Slide 23: the grey enthusiasm curve with peaks labelled Perceptrons 1958, PDP book 1986 and Krizhevsky, Sutskever, Hinton 2012, troughs labelled Minsky and Papert 1972 and AI winter 2000, two "28 years" arrows, a red "2028 ?" on the time axis, and a green continuation oscillating at a higher level than the grey curve.](../raw/images/01-introduction/slide-23.png)

*Slide 23 — the "new plane of enthusiasm" future: the green continuation keeps oscillating, but its troughs sit near the old peaks.*

Asked whether AI could take over society by 2028, the lecturer answered, "let's see" (≈20:15).
At the end of the lecture she adds that "everyone's pretty clear that we're currently in a really
hypey part of the hype cycle" (≈59:39).

## What deep learning is today (slide 24)

The lecture's inventory of the present (≈20:15–23:19):

- **Autograd** — frameworks like PyTorch and TensorFlow that "implement the chain rule in
  software, making good use of available hardware". Almost nobody now programs GPUs from scratch
  the way Krizhevsky did.
- **Billion-plus data point datasets**, such as LAION; **parallel training on thousands of
  GPUs**; **billion-plus parameter architectures**; and **million-plus dollar training costs**.
- **"Shockingly good results"** — results the lecturer "didn't anticipate seeing in my lifetime",
  which she calls both exciting and "a bit overwhelming".
- **"Massive isn't necessary"** — Stable Diffusion is "pretty lightweight" and was "incredibly
  impactful". Here she also notes the carbon cost of training large models, which works against
  her own field of using machine learning on biodiversity loss.
- **Open source and modular reuse** — weights trained by one group become a module in someone
  else's system. She adds that this openness is receding: closed models may offer an API but
  not the weights or architecture.

## The two signposts

From slide 25 on, the lecture alternates between two kinds of slide, marked by a signpost graphic
(≈23:19): **looking back**, what the course *expects you to have seen before*, and **looking
forward** (in green), what the course *will cover*. The bulleted signpost slides (26, 29, 31, 46,
48, 50, 53, 67, 71, 74, 76, 78) grow one bullet at a time; the full lists are on slide 67
(background) and slide 78 (coverage).

The background items below are review: "It's OK if you haven't seen these things before, but we
would expect you then to go and brush up on them" (≈13:10).

## Background: gradient descent (slides 26–28)

Learning is posed as optimization (≈24:05). With a model $f_\theta$, $N$ training pairs
$(x^{(i)}, y^{(i)})$ and a per-datapoint loss $L$, the optimal parameters are

$$\theta^{\ast} = \arg\min_\theta \sum_{i=1}^{N} L\big(f_\theta(x^{(i)}), y^{(i)}\big)$$

and the sum is the cost $J(\theta)$ (slide 27). Gradient descent starts "somewhere random",
computes the gradient, steps the weights, "and you step through, et cetera" (≈24:05). Slide 28
draws this as a path of arrows walking down a loss surface over two parameters into a deep well.

![Slide 28: a 3-D loss surface J(θ) over axes θ1 and θ2, with a tall yellow peak at the back, smaller bumps, and a deep blue well at the front right; an X on the peak marks the start and a chain of arrows descends into the well.](../raw/images/01-introduction/slide-28.jpg)

*Slide 28 — gradient descent as a walk downhill on the loss surface, from the start marked X into the well.*

Stochastic gradient descent and its relatives are deferred to a later lecture. See
[gradient descent](gradient-descent.md).

## Coming in lecture 2: backprop and differentiable programming (slides 29–30)

The first "will cover" item is "backprop and differentiable programming": "what does it actually
look like to build programming languages that are structured around the idea of … optimizing
through gradient descent", and "what are you actually optimizing for, and what are you pushing the
gradients through to?" (≈24:50). Slide 30 shows one iteration of gradient descent,

$$\theta^{t+1} = \theta^{t} - \eta_t \frac{\partial J(\theta)}{\partial \theta}\Big|_ {\theta=\theta^{t}}$$

with $\eta_t$ labelled "learning rate", under a banner deferring it all to **Lecture 2** (≈25:37).

## Background: MLPs and non-linearities (slides 31–45)

This is the longest review section (≈25:37–40:25). The full treatment is on
[multilayer perceptrons](multilayer-perceptron.md) and
[activation functions](activation-functions.md); the outline follows.

**Computation in a neural net** maps a vector in to a vector out (slide 32), and networks reuse
the same simple units "over and over again" (≈26:23). The basic unit is a **linear layer**: each
output component $z_j$ is a weighted sum of the inputs, $z_j = \sum_i w_{ij} x_i$ (slide 33), plus a
**bias** $b_j$, a term that "does not actually take in any of those input components" (slide 34).

![Slide 33: a column of input units with the top one labelled x_i, each connected to one output unit z_j; the connection from x_i is labelled w_ij, and beside it z_j = Σ_i w_ij x_i.](../raw/images/01-introduction/slide-33.jpg)

*Slide 33 — one output unit of a linear layer: a weighted sum of every input.*

![Slide 34: the same unit with an extra constant-1 input connected to z_j through the bias b_j, and z_j = Σ_i w_ij x_i + b_j with "weights" and "bias" labelled.](../raw/images/01-introduction/slide-34.jpg)

*Slide 34 — the bias enters as a weight on a constant input of 1.*

In vector form this is

$$z_j = \mathbf{x}^T \mathbf{w}_ j + b_j$$

(slide 35, ≈27:13). The lecturer flags this notation as the one the class will "be sticking with".
The parameters $\theta$ are all the weights and all the biases.

![Slide 35: a column of input units bracketed as x feeding one output unit z_j through a weight vector w_j, plus a constant-1 unit feeding it through the bias b_j; beside it, z_j = x^T w_j + b_j and θ = {W, b}, "parameters of the model".](../raw/images/01-introduction/slide-35.jpg)

*Slide 35 — the linear layer in vector form; this is the notation the course keeps.*

**A non-linearity makes it a neural net** (≈28:01): a pointwise function $g$ applied to each
$z_j$. The first one shown is a step, $g(z) = 1$ if $z \gt 0$ and $0$ otherwise — linear unit
plus step is "roughly … called a perceptron" (slide 36). Asked whether it is a good choice, the
class answers that it is not differentiable, which "hinders backpropagation": wherever you are,
"the gradient is zero", so a gradient step does not move (≈28:47).

![Slide 36: a perceptron diagram — inputs x, weights w, bias b and a constant-1 unit feeding z, then a pointwise non-linearity g(z) — beside the step function g(z) = 1 if z > 0, else 0, plotted for z from −4 to 4.](../raw/images/01-introduction/slide-36.jpg)

*Slide 36 — the perceptron: a linear unit followed by a step. The step's zero gradient is why it is not used for learning.*

**A perceptron can classify linearly separable data** (≈29:34–31:56). With two inputs
$x_1, x_2$, the pre-activation $z = \mathbf{x}^T \mathbf{w} + b$ is a plane over the input space —
"almost a ramp" — and thresholding the plane is a linear classifier. The lecturer then walks
through fits of increasing quality on separable data: a bad fit with seven misclassifications,
an OK fit after a step, and a good fit with none. Those plots are not in the published deck; the
transcript is the only record of them.

**Better non-linearities** (slides 38–41, ≈31:56–34:59). Three smooth alternatives follow.

**tanh** (slides 38–39) is bounded in $[-1, 1]$ and centred at 0, so no value gets "super huge".
But it saturates for large positive or negative inputs, so gradients go to zero far from the
centre: start far out and "you don't have a lot of training signal". It equals
$2\thinspace\sigma(2z) - 1$, where $\sigma$ is the sigmoid.

![Slide 38: the perceptron diagram beside the tanh formula g(z) = (e^z − e^−z)/(e^z + e^−z) and its plot, an S-curve from −1 to 1 crossing 0 at z = 0.](../raw/images/01-introduction/slide-38.jpg)

*Slide 38 — tanh: bounded and zero-centred, but flat at both ends, where the gradient vanishes.*

The **sigmoid** (slide 40) can be read as a neuron's firing rate and is bounded in $[0, 1]$. It
saturates in the same way, and its outputs are centred at 0.5, which is "poor conditioning" and
"slightly biased" — "In practice, we don't actually use this." The slide prints the formula as
$1/(1+e^{-h})$, and the lecturer corrects it aloud: "the notation is wrong. It should be minus z"
(≈32:42).

![Slide 40: sigmoid properties — firing-rate interpretation, bounded in [0,1], saturation, vanishing gradients, outputs centred at 0.5 (poor conditioning), not used in practice — beside the formula and its S-curve from 0 to 1.](../raw/images/01-introduction/slide-40.png)

*Slide 40 — the sigmoid. The formula on this slide has h in the exponent; the lecturer corrects it to z.*

The **ReLU**, $g(z) = \max(0, z)$ (slide 41), is unbounded above, which can contribute to
exploding gradients, but "super efficient to implement" because its derivative is just a step.
It also seems to speed up convergence: the AlexNet paper reported about a $6\times$ speed-up over tanh.
Its drawback is that a unit stuck strongly in the negative region is "dead", with no gradient. It
is "our default choice, and it's widely used in current models" (≈34:14).

![Slide 41: ReLU properties — unbounded output, a derivative of 0 below zero and 1 above, about 6x faster convergence than tanh in Krizhevsky et al., dead units with no gradient, the default choice — beside the plot of g(z) = max(0, z).](../raw/images/01-introduction/slide-41.png)

*Slide 41 — ReLU, the default non-linearity. Compare slide 38 (tanh) and slide 40 (sigmoid) for the saturating alternatives.*

**How to choose a non-linearity?** (≈36:31–38:49). There are no good general rules. Practice
converges on whatever has worked experimentally in a line of work — "sometimes that's called grad
student gradient descent". Jeremy Bernstein adds that it is "not a hard science", and the choice
is usually about making the network trainable rather than modelling the data. The exception the
lecturer gives is a known structure: if the features are sinusoidal, a sinusoidal activation can
make sense.

**Stacking layers** (slides 42–43, ≈34:59–36:31). Feed one layer's output into another and the
middle quantities $\mathbf{z}$ and $\mathbf{h}$ become **hidden units**. Stacking all the units of
a layer turns the per-unit dot products into a matrix product:

![Slide 42: input units x feed hidden pre-activations z, each passed through g to give h = g(z), which feed the output units y; weights W1j, W2j and biases b1j, b2j label one unit's connections, and z and h are captioned "hidden units".](../raw/images/01-introduction/slide-42.jpg)

*Slide 42 — stacking two layers: z and h are the hidden units, seen at neither the input nor the output.*

$$\mathbf{h} = g(\mathbf{W}_ 1 \mathbf{x} + \mathbf{b}_ 1) \qquad \mathbf{y} = g(\mathbf{W}_ 2 \mathbf{h} + \mathbf{b}_ 2)$$

and $\theta = \lbrace \mathbf{W}_ 1, \ldots, \mathbf{W}_ L, \mathbf{b}_ 1, \ldots, \mathbf{b}_ L \rbrace$, every
layer's weights and biases (slide 43).

![Slide 43: a fully connected two-layer network — input x, hidden layer h, output y, with weights W1, W2 and biases b1, b2 on the connections — beside h = g(W1 x + b1), y = g(W2 h + b2) and θ = {W1…WL, b1…bL}.](../raw/images/01-introduction/slide-43.jpg)

*Slide 43 — two stacked layers in matrix form, and the parameter set of an L-layer net.*

**Non-linear classification with a two-layer net** (slide 45, ≈38:49–40:25). In the same
two-input setting, each hidden unit is its own ramp; combining two ramps through a non-linearity
gives "a kind of triangular, almost a pyramidal component that will be higher than everything
else", and thresholding that gives a *non-linear* decision region. The slide's network is
$x_1, x_2 \to z_1, z_2 \to h_1, h_2 \to z_3 \to y$, with $y = \mathbf{1}(z_3 \gt 0)$. In the
published PDF the four heat-map panels ($h_1$, $h_2$, $z_3$, $y$) are a clipped animation frame
— only part of the $h_1$ panel is drawn — so the lecturer's description is the record of the
result.

![Slide 45: the two-input, two-hidden-unit network and its equations, above four heat-map panels titled h1, h2, z3 and y, of which only part of h1 is drawn in the published PDF.](../raw/images/01-introduction/slide-45.jpg)

*Slide 45 — the network for the non-linear classification example. The heat maps are an animation frame clipped in OCW's PDF, so only the diagram and equations are reliable here.*

## Coming later: approximation, architectures, generalization (slides 46–52)

**Why we can approximate** (slides 46–47, ≈40:25–42:42; **Lecture 3**). One layer gives a linear
decision surface. Two or more layers can in theory represent any function, provided there is a
non-trivial non-linearity between them, since two linear layers compose to a linear map. The
intuition offered is a Riemann sum: enough narrow pieces can build up any curve. The catch is
**efficiency**: a sufficiently *wide* two-layer net could approximate anything in principle, but
"in practice, that's actually very inefficient", and a narrow, deep model can approximate the
same function with far fewer parameters — "In practice, we do find that's true. More layers
helps." See [representational power](representational-power.md).

**Architectures** (slides 48–49, ≈42:42–43:29). A deep net is "some cascade of repeated simple
computations" whose representations grow more abstract with each layer. A shallow net might
efficiently find edges; a deep one can recognize "this clownfish" or translate English to
Chinese. Written as a function, it is a chain:

$$f(\mathbf{x}) = f_L(f_{L-1}(\ldots f_2(f_1(\mathbf{x}))))$$

The course covers CNNs, GNNs, transformers and RNNs. The deck's banner gives lecture numbers for
these that differ from the published schedule — see the
[course map](course-map.md#the-decks-lecture-pointers).

**When and why can we generalize** (slides 50–52, ≈43:29–50:26). Deep nets have enough
parameters to act as lookup tables that regurgitate their training data, "but instead, they seem
to learn rules that generalize. And this actually defies classical theory." Classical theory
predicts a U-shaped test-risk curve against capacity; the **double descent** curve (slide 51,
citing Belkin, Hsu, Ma and Mandal, PNAS 2019) shows test risk falling *again* once capacity passes
the **interpolation threshold**, in the over-parameterized "modern" regime. The **simplicity
hypothesis** (slide 52) restates the contrast: classically "big models learn complicated
functions, and overfit"; the emerging theory is that "deep nets learn *simple* functions that
generalize".

![Slide 51: two risk-versus-capacity plots — (A) the classical U-shaped test-risk curve with a "sweet spot" between under- and over-fitting, and (B) double descent, where test risk peaks at the interpolation threshold and falls again in the over-parameterized "modern" interpolating regime while training risk stays at zero.](../raw/images/01-introduction/slide-51.jpg)

*Slide 51 — the classical U-curve (A) against double descent (B), from Belkin et al., PNAS 2019.*

Four student questions sharpen this section (≈45:01–49:40), and are written up on
[generalization and double descent](generalization-and-double-descent.md):

- *What is capacity?* "The number of parameters in the model" — width and depth together.
- *How does dataset size relate?* It is a different axis from the one plotted, but closely
  related. With too few data points a model cannot learn to interpolate between them, "because you
  don't have enough coverage of the data distribution", and that is "one of the reasons that none
  of this was possible until we started building big enough data sets".
- *Overfitting versus over-parameterized?* Overfitting is the classical failure, where a too-flexible
  function threads three points and is "too spiky". "Over-parameterized" is the new axis, and its
  message is that you perhaps cannot be *too* over-parameterized. But that ignores resources: what
  you really want is "some optimal point on this curve" that you can afford.
- *Wide-and-shallow or narrow-and-deep for interpretability?* Neither is easy to interpret once
  parameter counts grow, which is why fields like ecology still favour simple additive models.
  Jeremy Bernstein adds that the width-versus-depth recipe of frontier models is "a closely guarded
  secret", and the lecturer agrees there is "no perfect prescription".

## Background: softmax, cross-entropy and the training setup (slides 53–66)

(≈49:40–53:30.) See [softmax and cross-entropy](softmax-and-cross-entropy.md) for the full
treatment.

A classifier's **last layer** has one unit per class, and the simplest readout is the
**argmax**: the most active unit is the prediction (slide 55). A **loss** compares the output
with the ground-truth label. It should be small when the prediction is right (slide 57) and large
when it is wrong (slide 58): "some loss function that does a good job of punishing you when you're
wrong and not punishing you when you're right" (≈51:12).

![Slide 55: the last layer of a classifier, one unit per class (dolphin, cat, grizzly bear, angel fish, chameleon, clown fish, iguana, elephant, …) shaded by activation, with argmax picking the darkest unit, clown fish.](../raw/images/01-introduction/slide-55.jpg)

*Slide 55 — the classifier's last layer, read out by argmax.*

![Slide 57: the network output, with clown fish darkest, compared against the ground-truth label "clown fish"; the loss is small.](../raw/images/01-introduction/slide-57.jpg)

*Slide 57 — prediction and label agree, so the loss should be small.*

![Slide 58: the classifier's output column, with the clown fish unit darkest, compared against the ground-truth label "grizzly bear"; the loss is large.](../raw/images/01-introduction/slide-58.jpg)

*Slide 58 — the network still says clown fish, the label says grizzly bear, so the loss should be large (contrast slide 57, where they match).*

If the network's output is normalized — the slides route it through a "softmax" — it can be read
as a distribution over classes, $\hat{y}$. The label becomes a **one-hot** vector $y$, "0
everywhere except for the place where it's correct". The **cross-entropy** (slide 59) is

$$H(y, \hat{y}) = -\sum_{k=1}^{K} y_k \log \hat{y}_ {k}$$

over $K$ classes. Minimizing it is "telling the network to maximize the probability of the
training data", which "forces the output to be a reasonably good probability model of the object
class, given the image, assuming you have enough training data" (≈51:59). The lecturer puts
"probability" in quotes: "these are not true probabilities. They're scores."

![Slide 59: the network output ŷ after a softmax beside a one-hot ground-truth column y with only "grizzly bear" set, and the cross-entropy H(y, ŷ) = −Σ y_k log ŷ_k, "probability of the observed data under the model".](../raw/images/01-introduction/slide-59.jpg)

*Slide 59 — cross-entropy against a one-hot label. The lecture never writes out the softmax function itself; it is part of the assumed background.*

Slides 60–63 show this as bar charts. The score $-H$ is the log-probability the model gave the
true class, and the gap up to 0 is "how much better you could have done" — the direction of the
gradient (≈51:59). A confident correct answer leaves a small gap (the grizzly bear, slide 62). A
confusion between similar classes — the network says iguana for a chameleon — leaves a large one
(slide 63, ≈52:45).

Training then means taking "all these different weights in your massive model, and then you're
fiddling around with them until you match the desired output for all your training data"
(≈52:45). Slides 64–66 draw it as one training example after another — clownfish, grizzly bear,
chameleon — flowing through six layers with parameters $\theta_1, \ldots, \theta_6$ into a loss,
all under the same $\arg\min$ objective as slide 27.

## Background: batching and tensors (slides 67–70)

![Slide 70: the two-input network and its equations, the heading "Tensor processing with batch size = 3:", and the input matrix X with columns x1 and x2 and N_batch rows; the rest of the build is cut off in the published PDF.](../raw/images/01-introduction/slide-70.jpg)

*Slide 70 — the start of the batched build; the remaining matrices are cut off in OCW's PDF.*

(≈53:30–55:02.) The per-example losses are summed anyway, so they can be computed **in
parallel**: stack a batch of examples and push them through the same layers together (slide
68). Each layer's output is then a matrix of features by examples, and in general a **tensor**, "a
multi-dimensional matrix", with every layer "some representation of the input data" (slide 69). The
two-layer network becomes a chain of matrix multiplications over a batch dimension, with a
pointwise non-linearity between them. That is why GPUs, which "can do so many multiplications in
parallel", mattered (≈55:02). Slide 70 begins this build for a batch of three, but in the published
PDF it stops after the input matrix $\mathbf{X}$ ($N_{\text{batch}}$ rows, one column each for
$x_1$ and $x_2$). See [tensors and batching](tensors-and-batching.md).

## Coming later: representations, generative models, reuse, scale (slides 71–79)

**How deep networks represent data** (slides 71–73, ≈55:02–57:19; **Lectures 11–13**). Deep nets
are "a more compact way of representing knowledge", built on reusable lower-level parts. The
example is a classifier for the letter T: rather than learn one from scratch, reuse a line
detector, rotate it, and learn only how the two lines join. Analyses of trained convolutional
networks find low-level features early and more complex ones later. Slide 72 pairs a
visual-cortex hierarchy (Serre, 2014) with feature clusterings from early and late layers of a
network (Donahue, 2013): early layers do not cluster by category, late ones do. See
[representation learning](representation-learning.md).

**Generative models** (slides 74–75, ≈57:19; **Lectures 14–16**) — text- and image-generation,
covering basics, representations and conditional models.

**Reusing weights** (slides 76–77, ≈57:19–58:53; **Lectures 18–19**). The edge and orientation
features learned for animal photos are also useful for satellite imagery, so you may not need to
learn everything from scratch — "really valuable if you don't have big data or big compute".

**Scale** (slides 78–79, ≈58:53–59:39). The comparison is with brains: a worm (302 neurons), a
fruit fly (15,000), a human (about 100 billion), an elephant (250 billion). The course will cover
scaling rules for optimization, scaling laws, and "automatically learning how to best optimize a
model".

## Closing

The recap (slide 80, ≈59:39) restates the three parts: the history, the expected background, and
the course's scope. Asked how long the first lecture would take to reach a million views, the
lecturer predicted the intro "is probably going to be the least watched of all the lectures …
because it's just a bunch of course logistics" (≈1:00:25).
