# Course notation

6.7960 publishes a six-page **Math Notation** handout that fixes the symbols used across the
lectures and problem sets. This page restates it, so that a formula on any other page of this
wiki can be read against it. When a lecture slide departs from these conventions, the lecture
page says so.

Source: [`mit6_7950_f24_notation.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7950_f24_notation.pdf)
(the filename really does say `6_7950`, and the handout's running header says "6.S898 Deep
Learning, Fall 2024" — 6.S898 is the course's earlier number). The handout says the course
will "stick to the following conventions throughout most of this course, and note it when we
deviate from these rules".

## Scalars, vectors, matrices and tensors

**Bold means not-a-scalar.** Scalars are plain ($x, y, z$); vectors, matrices and tensors are
bold. Vectors are lowercase bold ($\mathbf{x}, \mathbf{y}, \mathbf{z}$) and matrices uppercase
bold ($\mathbf{X}, \mathbf{Y}, \mathbf{Z}$).

**An indexed entry is a scalar, so it is not bold.** The $i$-th entry of a vector is written
$x_i$ or $\mathbf{x}[i]$. The entry of a matrix in row $i$, column $j$ is $X_{ij}$ or
$\mathbf{X}[i, j]$ — rows first, then columns.

**Slices** use the bracket form: $\mathbf{X}_ i$ or $\mathbf{X}[i, :]$ is row $i$, and
$\mathbf{X}[:, j]$ is column $j$. Indexing starts at 1. The handout's own example is a
$3 \times 2$ matrix with rows $(1, 2)$, $(3, 4)$, $(5, 6)$, for which $\mathbf{X}[2, :]$ is
$(3, 4)$.

**Tensors** — multidimensional arrays — are usually written as *lowercase* bold, like
vectors: $\mathbf{x}$. The handout gives two reasons. A tensor can have any number of
dimensions, and many operators in the course are defined to work on an array of any
dimensionality, so a notation that does not commit to a shape is the honest one. Uppercase is
used for a tensor only to distinguish tensors of different shapes, and the lecture will say so
when it does. A tensor is indexed or sliced the same way as a matrix: $\mathbf{x}[c, i, j, k]$,
$\mathbf{x}[:, :, k]$.

**A dataset of $N$ points** is $\lbrace \mathbf{x}^{(i)} \rbrace_{i=1}^{N}$ — the superscript in
parentheses indexes datapoints, never powers or layers.

## Products

| Operation | Notation |
| --- | --- |
| Dot product | $\mathbf{x}^T \mathbf{y}$ |
| Matrix product | $\mathbf{A}\mathbf{B}$ |
| Hadamard (element-wise) product | $\mathbf{x} \odot \mathbf{y}$, $\mathbf{A} \odot \mathbf{B}$ |
| Product of two scalars | $ab$ or $a \ast b$ |

## Learning

- $L$ is the **loss**, and usually means the loss on a single datapoint.
- $J$ is the **total cost** over all datapoints.
- $\theta$ stands for the learnable **parameters**, whatever they are.

So the training objective of [lecture 1](01-introduction.md) reads, in this notation, as a sum
of per-datapoint losses $L$ making up a cost $J(\theta)$, minimized over $\theta$.

## Neural networks

- **Parameters** $\theta$ include the weights $\mathbf{W}$ and biases $\mathbf{b}$, and any
  other learnable parameter of the network.
- **"Data" is any representation of the signal**, not just the input: inputs, hidden
  activations and outputs are all data, and all can be written $\mathbf{x}$. When they need
  telling apart, the input is $\mathbf{x}$, a hidden layer $\mathbf{h}$ and the output
  $\mathbf{y}$.
- **Pre- and post-activation.** Where it matters, $\mathbf{z}$ is a hidden layer *before* the
  non-linearity and $\mathbf{h}$ is the same layer *after* it. This is the
  $\mathbf{z} \to \mathbf{h} = g(\mathbf{z})$ pattern of [lecture 1's MLP](multilayer-perceptron.md).
- **Layers.** $\mathbf{x}_ l$ is the vector of neuron values on layer $l$, and
  $\mathbf{x}_ l[n]$ is its $n$-th neuron. For a layer with a spatial layout, such as a
  convolutional feature map, $\mathbf{x}_ l[n, m]$ indexes the neuron at position $(n, m)$. The
  representation at layer $l$ of the $i$-th datapoint is $\mathbf{x}_ {l}^{(i)}$.
- **Batches.** In code-like settings an activation may be written $\mathbf{x}_ l[b, c, n, m]$:
  $b$ indexes the element of the batch, $c$ the channel, and $n, m$ are spatial coordinates.
- **Modules.** $\mathbf{x}_ {\text{in}}$ and $\mathbf{x}_ {\text{out}}$ are a particular layer's or
  module's input and output, when keeping track of layer indices would only get in the way.
- **Channels come first.** For a signal with several channels the first dimension of the
  tensor indexes the channel: $\mathbf{x} \in \mathbb{R}^{C \times N \times M \times \cdots}$,
  where $C$ is the number of channels.
- **Transformers are the exception.** To match standard notation, a set of tokens is an
  $N \times d$ matrix — $N$ tokens, each of dimension $d$ — rather than channels-first.

### Diagram conventions

Circles are **neurons**, which are scalar nodes; an edge between two neurons is an
$\mathbb{R} \to \mathbb{R}$ map, usually parameterized by one weight on that edge. Squares or
rectangles are **vectors of neurons**; an edge from a node of dimension $C_1$ to one of
dimension $C_2$ is an $\mathbb{R}^{C_1} \to \mathbb{R}^{C_2}$ map, which may have many weights. A
**token** — a vector of neurons used in the particular way the transformer lectures define — gets
its own rectangle symbol. Networks are drawn flowing either left-to-right or bottom-to-top; the
two mean the same thing.

## Probability

Random variables and their realizations are normally not distinguished. When the distinction
matters, a non-bold capital ($X$) is the random variable and a lowercase letter ($x$) a
realization, taking values in a set $\mathcal{X}$.

- $a = p(X = x \mid \ldots)$ is the probability of the realization $X = x$, possibly
  conditioned on observations. It is a scalar.
- $f = p(X \mid \ldots)$ is the whole distribution over $X$ — a function
  $f : \mathcal{X} \to \mathbb{R}$, a mass function if $X$ is discrete and a density if it is
  continuous.
- $p(x \mid \ldots)$ is shorthand for $p(X = x \mid \ldots)$.
- For a named distribution such as $p_\theta$, writing $p_\theta$ alone means $p_\theta(X)$.

For continuous variables everything above refers to densities, and "probability distribution"
should be read as "probability density function".

## Matrix calculus

These are definitions, not results. The handout is explicit that "there is no right or wrong to
it", and that they were chosen because they make the equations — and therefore the code —
simpler. Lecture 2 restates them on slides 32–33 and builds [backpropagation](backpropagation.md)
on them; for an MLP's backward pass it then switches to column-vector gradients by transposing
(slide 51), and says so.

**Vectors are columns**, of shape $N \times 1$:

$$\mathbf{x} \triangleq \begin{bmatrix} x_1 \cr x_2 \cr \vdots \cr x_N \end{bmatrix}$$

**The gradient of a scalar with respect to a vector is a row**, of shape $1 \times N$. For a
scalar $y$ and an $N$-dimensional $\mathbf{x}$:

$$\frac{\partial y}{\partial \mathbf{x}} \triangleq \begin{bmatrix} \frac{\partial y}{\partial x_1} & \frac{\partial y}{\partial x_2} & \cdots & \frac{\partial y}{\partial x_N} \end{bmatrix}$$

**The gradient of a vector with respect to a vector is the Jacobian**, of shape $M \times N$ for
an $M$-dimensional $\mathbf{y}$ and an $N$-dimensional $\mathbf{x}$ — one row per output, one
column per input:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \triangleq \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_N} \cr \vdots & \ddots & \vdots \cr \frac{\partial y_M}{\partial x_1} & \cdots & \frac{\partial y_M}{\partial x_N} \end{bmatrix}$$

**The gradient of a scalar with respect to a matrix is transposed.** If $\mathbf{W}$ is
$N \times M$ and $L$ is a scalar, $\frac{\partial L}{\partial \mathbf{W}}$ is $M \times N$: its
entry in row $j$, column $i$ is $\frac{\partial L}{\partial W_{ij}}$.

$$\frac{\partial L}{\partial \mathbf{W}} \triangleq \begin{bmatrix} \frac{\partial L}{\partial W_{11}} & \cdots & \frac{\partial L}{\partial W_{N1}} \cr \vdots & \ddots & \vdots \cr \frac{\partial L}{\partial W_{1M}} & \cdots & \frac{\partial L}{\partial W_{NM}} \end{bmatrix}$$

The handout flags this as the surprising one — "the dimensions are transposed from what you
might have expected; this makes the math simpler later". It is consistent with the row-vector
rule above: treating $\mathbf{W}$ as a list of numbers, the derivative of a scalar is laid out
the transposed way round from the thing being differentiated.

## Loose conventions

The handout lists two conventions that "will not be strictly adhered to": $x$ is often a
function's input and $y$ its output, and $f$, $g$, $h$ are usually functions, with function
spaces $\mathcal{F}$, $\mathcal{G}$, $\mathcal{H}$.

It also warns that **"dimension" is used in two senses**: a coordinate of a data structure ("the
$i$-th dimension of a vector", "a 128-dimensional feature space") and the number of axes of an
array ("a 4D tensor"). Both appear in the course, and context says which.
