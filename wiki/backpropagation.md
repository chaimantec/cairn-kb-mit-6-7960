# Backpropagation

Backpropagation is how a network gets the gradient of its cost with respect to every parameter,
which is what [gradient descent](gradient-descent.md) needs at every step. Lecture 1 names it as a
topic the course will teach, and credits frameworks such as PyTorch with implementing "the chain
rule in software" (≈20:15). Lecture 2 derives it from the chain rule, for a chain of layers, a
linear layer, a ReLU, a whole MLP and any directed acyclic graph. Covered so far:
[lecture 1](01-introduction.md) in passing; [lecture 2](02-how-to-train-a-neural-net.md), slides
26–55 and the worked example on slides 72–80, ≈27:16–57:27; [lecture 5](05-architectures-graphs.md)
in passing, for graph neural networks (≈53:31–54:20). For the symbols, see
[notation](notation.md).

## The setting: a computation graph

Lecture 2 treats a network as a **computation graph**, "a graph of functional transformations,
nodes, that when strung together perform some useful computation". Deep learning deals mainly with
graphs that are **directed acyclic graphs** (DAGs) "for which each node is differentiable" (slide
26, ≈27:16–28:50). The simplest is a chain:

$$\mathbf{x}_ 0 \xrightarrow{f_1(\cdot,\thinspace \theta_1)} \mathbf{x}_ 1 \xrightarrow{f_2(\cdot,\thinspace \theta_2)} \cdots \xrightarrow{f_L(\cdot,\thinspace \theta_L)} \mathbf{x}_ L \xrightarrow{\mathcal{L}} J$$

Here layer $l$ computes $\mathbf{x}_ l = f_l(\mathbf{x}_ {l-1}, \theta_l)$ from the previous
activation and its own parameters $\theta_l$, and the loss $\mathcal{L}$ turns the final output
into the scalar cost $J$. Training needs $\partial J / \partial \theta_l$ for every $l$. Each layer
is differentiable "with respect to its inputs (the inputs are the data and parameters)" (slide 30,
≈31:10), so these gradients exist. Backpropagation is the efficient way to compute them all.

## Shapes first

Every gradient below is a matrix product, and the shapes are fixed by convention (slides 32–34,
≈31:55–37:20; the [notation handout](notation.md#matrix-calculus) uses the same rules). A vector is
a column. The derivative of a scalar by an $n$-vector is a $1 \times n$ **row**. The derivative of
an $m$-vector by an $n$-vector is the $m \times n$ **Jacobian**. The derivative of a scalar by an
$n \times m$ matrix is $m \times n$, transposed. The chain rule then multiplies Jacobians, and the
inner dimensions have to match:

$$\underbrace{\frac{\partial \mathbf{z}}{\partial \mathbf{x}}}_ {m \times n} = \underbrace{\frac{\partial \mathbf{z}}{\partial \mathbf{u}}}_ {m \times p} \underbrace{\frac{\partial \mathbf{u}}{\partial \mathbf{x}}}_ {p \times n}$$

## The idea: compute shared terms once

Expand $\partial J / \partial \theta_1$ and $\partial J / \partial \theta_2$ by the chain rule and
the two expansions share a long product — everything from the loss back to $\mathbf{x}_ 2$ (slide
36). "We could separately compute all of the derivatives using the chain rule, but because these
terms in the gray box are shared, we only need to compute them once. So back propagation is a
pretty simple algorithm for propagating shared terms through the computation graph. It's basically
an efficiency trick, but it's one that makes it computationally practical for very large models"
(≈38:06). The slide's other name for it is "dynamic programming".

So training alternates two sweeps (slide 37). The **forward pass** sends data up the graph and
computes the loss. The **backward pass** sends "error signals (gradients) backwards through the
network, from outputs and loss back to inputs and parameters", starting from
$\partial J / \partial J = 1$ at the top.

## What each layer does

For one layer with input $\mathbf{x}_ {\texttt{in}}$, output $\mathbf{x}_ {\texttt{out}}$ and
parameters $\theta$, backprop tracks two arrays (slides 38–41, ≈38:52–41:12):

- $\mathbf{L} = \partial \mathbf{x}_ {\texttt{out}} / \partial [\mathbf{x}_ {\texttt{in}}, \theta]$,
  the layer's own derivative — a matrix — in two parts, $\mathbf{L}^{\mathbf{x}}$ with respect to
  the input and $\mathbf{L}^{\theta}$ with respect to the parameters. It "comes from the derivative
  function, f′, of the layer (which we assume is provided)".
- $\mathbf{g} = \partial J / \partial \mathbf{x}$, the cost's derivative at an activation — a row
  vector — written $\mathbf{g}_ {\texttt{out}}$ at the layer's output and $\mathbf{g}_ {\texttt{in}}$
  at its input.

Given the gradient arriving from above, the layer computes the gradient to pass down and the
gradient for its own parameters:

$$\mathbf{g}_ {\texttt{in}} = \mathbf{g}_ {\texttt{out}} \mathbf{L}^{\mathbf{x}}, \qquad \frac{\partial J}{\partial \theta} = \mathbf{g}_ {\texttt{out}} \mathbf{L}^{\theta}$$

The first line is the recurrence the slide calls "backpropagation of error signals". The second
gives the update,
$\theta \leftarrow \theta - \eta (\partial J / \partial \theta)^{\mathsf{T}}$, with $\eta$ the
learning rate and the transpose turning the row-vector gradient back into the parameters' shape.
Seen from layer $l$ (slide 43), there are three inputs during training — $\mathbf{x}_ {l-1}$,
$\partial J / \partial \mathbf{x}_ l$ and $\theta_l$ — and three outputs — $\mathbf{x}_ l$,
$\partial J / \partial \mathbf{x}_ {l-1}$ and $\partial J / \partial \theta_l$. A layer therefore
only has to know how to evaluate $f_l$, $\partial f_l / \partial \mathbf{x}_ {l-1}$ and
$\partial f_l / \partial \theta_l$.

The full algorithm is "forward, then backward", then update, "and repeat" (slide 42). Over a
batch of $N$ examples the cost is the average of the per-example costs, so its gradient is the
average of the per-example gradients (slide 45, ≈43:30–44:17).

## The linear layer

For $\mathbf{x}_ {\texttt{out}} = \mathbf{W} \mathbf{x}_ {\texttt{in}}$ the Jacobian with respect
to the input is $\mathbf{W}$ itself, since $\partial x_{\texttt{out}_ i} / \partial x_{\texttt{in}_ j} = W_{ij}$,
and the weight $W_{ij}$ touches only output $i$, with
$\partial x_{\texttt{out}_ i} / \partial W_{ij} = x_{\texttt{in}_ j}$ (slides 46–49,
≈44:17–48:58). The result is three matrix products:

$$\mathbf{x}_ {\texttt{out}} = \mathbf{W} \mathbf{x}_ {\texttt{in}}, \qquad \mathbf{g}_ {\texttt{in}} = \mathbf{g}_ {\texttt{out}} \mathbf{W}, \qquad \frac{\partial J}{\partial \mathbf{W}} = \mathbf{x}_ {\texttt{in}} \mathbf{g}_ {\texttt{out}}$$

"Your forward and backward pass are just multiplying by your weight matrix, just in a different
order" (≈45:51). The weight gradient is an outer product of the layer's input column and its
output-gradient row.

**A sign convention to watch.** The lecture's linear-layer and worked-example slides write the
update as $\mathbf{W} \leftarrow \mathbf{W} + \eta (\partial J / \partial \mathbf{W})^{\mathsf{T}}$
with a negative learning rate — "η = -0.2 (because we used positive increments)" (slide 72) —
where its gradient-descent slides write $\theta - \eta \nabla_\theta J$ with a positive one. The two
are the same step.

## Through a whole MLP, and why the backward pass is linear

Transposing the gradients into columns (slide 51, ≈49:44) gives

$$\mathbf{g}_ {\texttt{in}}^{\mathsf{T}} = \mathbf{W}^{\mathsf{T}} \mathbf{g}_ {\texttt{out}}^{\mathsf{T}}$$

so "backward for a linear layer is the same operation as forward but with the weights transposed".
A ReLU on the backward pass becomes a diagonal **gating matrix** whose entries come from the
forward activations. It blocks the gradient for every component that sat in the ReLU's zero
region, and it is diagonal "so that it's operating on each element independently" (≈50:31–51:18).
A pointwise $\tanh$ likewise becomes a diagonal matrix of $1 - \tanh^2$ values in the worked
example (slide 79). For the L2 loss $J = \lVert \hat{\mathbf{y}} - \mathbf{y} \rVert_ 2^2$, where
$\hat{\mathbf{y}}$ is the prediction and $\mathbf{y}$ the target, the first backward gradient is
$2(\hat{\mathbf{y}} - \mathbf{y})$ (slide 51).

So every backward step is a matrix product: "backprop is still a linear model" (≈51:18). The
lecture passes on co-instructor Phillip Isola's reason: however curved the loss, a first-order
method fits a plane to it at the current point and moves along that plane (≈51:18–52:05). Slide
52 draws a whole training iteration as one computation graph, with a forward half, a backward half
made only of linear boxes using $\mathbf{W}^{\mathsf{T}}$, and three linear boxes producing the
weight gradients.

**Memory.** The backward pass reuses forward values — every layer's input for its weight gradient,
and the ReLU activations for its gating matrix — so "you do need to save the intermediate
representation so that you can efficiently compute back propagation". The forward pass alone could
discard them, which makes the two passes asymmetric (≈52:51–54:22).

## Any DAG: merge, branch, and shared parameters

Two more rules extend all this from chains to any DAG (slide 53, ≈54:22–56:42):

- **Merge** (concatenation, addition, …): the gradient arriving at the merged value is split, and
  each input receives only the part that belongs to it.
- **Branch** (one value used in several places): the gradients coming back from every use are
  **summed**, $\partial J / \partial \mathbf{x} = \sum_i \partial J / \partial \mathbf{x}^{i}$.

A DAG is then "just chains, where sometimes they merge, and sometimes they branch" (≈1:14:38).
**Parameter sharing** is a branch: when several layers use the same $\theta$, its gradient is the
sum of the gradients from every use — "Parameter sharing —> sum gradients" (slide 55, ≈56:42). The
lecture defers the derivation of why branching sums to the lecture notes (≈55:55) and office
hours (≈1:15:24).

Graph neural networks (lecture 5) are trained the same way. Their message passing is the forward
pass, not an optimization: "So how would you train that? You just backpropagate through this
computation graph. Nothing different" (≈53:31–54:20). Every node in a layer uses the same aggregation
and update functions, a case of parameter sharing. See [graph neural networks](graph-neural-networks.md).

## A worked example

The deck's last section runs one iteration by hand (slides 72–80). The network takes the input
$\mathbf{x}_ 0 = (1.0, 0.1)^{\mathsf{T}}$ through a $\tanh$ hidden layer and a linear output, with
target $\mathbf{y} = 0.5$ and loss $\frac{1}{2}(\mathbf{x}_ 3 - \mathbf{y})^2$, where $\mathbf{x}_ 3$
is the output. Its weights before the step are

$$\mathbf{W}_ 0 = \begin{pmatrix} 1 & -3 \cr 0.2 & 1 \end{pmatrix}, \qquad \mathbf{W}_ 1 = \begin{pmatrix} 1 & -1 \end{pmatrix}$$

The forward pass gives $\mathbf{x}_ 3 = 0.313$ and a loss of $0.017$. The backward pass starts from
$\partial \mathcal{L} / \partial \mathbf{x}_ 3 = -0.1869$. With $\eta = -0.2$, the weights after
one step are

$$\mathbf{W}_ 0 = \begin{pmatrix} 1.02 & -3.0 \cr 0.17 & 1.0 \end{pmatrix}, \qquad \mathbf{W}_ 1 = \begin{pmatrix} 1.02 & -0.989 \end{pmatrix}$$

Every intermediate value, and a misprint on slide 79, is on the
[lecture 2 page](02-how-to-train-a-neural-net.md#worked-example-one-iteration-of-backpropagation).

## Beyond the weights

Nothing in the algorithm is specific to weights. "Backprop lets you optimize any node (function)
or edge (variable) in your computation graph w.r.t. to any scalar cost" (slides 61–63), including
the input itself. That is the basis of feature visualization and CLIP-guided image generation — see
[differentiable programming](differentiable-programming.md). In practice frameworks do the
bookkeeping: PyTorch "has autograd", which calculates gradients "for any function" built from
torch operations, each of which must define its gradient (≈11:40, ≈1:16:12).
