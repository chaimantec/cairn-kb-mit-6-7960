# Multilayer perceptrons

A multilayer perceptron (MLP) is the network lecture 1 builds up from scratch: linear layers
alternating with pointwise non-linearities. The course treats MLPs as **background it expects you
to have seen** (slide 31), and reviews them so that its notation is fixed before anything new is
built on it. Covered so far in this knowledge base: [lecture 1](01-introduction.md), slides 32–45,
≈25:37–40:25. See also [activation functions](activation-functions.md) and
[representational power](representational-power.md).

## The linear layer

"Computation in a neural net" maps a vector in to a vector out (slide 32), by repeating the same
simple units "over and over again" (≈26:23). The first unit is the **linear layer**. Each output
component $z_j$ is a weighted sum of the input components $x_i$:

$$z_j = \sum_i w_{ij} x_i$$

(slide 33). A **bias** $b_j$ is then added, a term that "does not actually take in any of those
input components" (≈27:13). Slide 34 draws it as the weight on an extra input fixed at 1:

$$z_j = \sum_i w_{ij} x_i + b_j$$

Collecting the weights into $\mathbf{w}_ j$, the vector form is

$$z_j = \mathbf{x}^T \mathbf{w}_ j + b_j$$

(slide 35) — the notation the lecturer says the class "will be sticking with … for the rest of
the class" (≈26:23). The **parameters** $\theta$ are "the set of all the weight terms and all the
bias terms". Slide 35 writes this as $\theta = \lbrace \mathbf{W}, \mathbf{b} \rbrace$.

## The non-linearity, and the perceptron

"What actually makes a neural net a neural net is the fact that it's not just linear combinations
of inputs" (≈28:01). A function $g$ is applied to each pre-activation, **pointwise** — to each
component separately — giving $h = g(z)$. In the course's [notation](notation.md), $z$ is always
the value before the non-linearity and $h$ the value after.

A linear unit followed by a step,

$$g(z) = \begin{cases} 1 & \text{if } z \gt 0 \cr 0 & \text{otherwise} \end{cases}$$

is "roughly … called a perceptron" (slide 36, ≈28:47). It is the unit Rosenblatt introduced in
1958, which the lecture calls the building block for most deep learning since (≈13:58). As a
*learning* unit the step is a poor choice, because it is not differentiable and its gradient is
zero almost everywhere: "If you're anywhere on this graph, you don't know which way to go … you're
not going to move because the gradient is zero" (≈28:47). That is why smooth
[activation functions](activation-functions.md) replace it.

## What a single layer can classify

Take two inputs $x_1, x_2$, weights $w_1, w_2$, and $z = \mathbf{x}^T \mathbf{w} + b$, with
output $y = g(z)$ (≈29:34). Over the input plane, $z$ is itself a plane — "almost a ramp" — and
thresholding a plane splits the inputs along a straight line. So "even a single-layer neural
network can perform linear classification", provided the data are **linearly separable**
(≈30:21).

Learning then means finding the weights and bias that minimize a loss measuring the gap between
the true label (here 0 or 1) and the output. The lecturer walks through three decision lines on
separable data (≈31:08–31:56). A *bad fit* misclassifies seven points: "You're wrong about more
things than you're right". An *OK fit*, perhaps one gradient step later, misclassifies fewer. A
*good fit* misclassifies none. These plots are described in the lecture but are not in the
published deck.

What one layer cannot do is draw a non-linear boundary. **XOR** is the classic case (slide 14):
the inputs $(0,0)$ and $(1,1)$ map to 0, and $(1,0)$ and $(0,1)$ map to 1, so no single line
separates the classes. Solving it needs "multiple layers of a perceptron" (≈16:20). Backpropagation
made such multilayer networks trainable, which is why the 1986 PDP book revived the field.

## Stacking layers

Feed one layer's output into another, and the middle quantities $\mathbf{z}$ and $\mathbf{h}$
become **hidden units**. They are "somewhere in the middle of the model", seen at neither the
inputs nor the outputs (slide 42, ≈34:59). Writing out every unit of a layer at once turns the
per-unit dot products $\mathbf{x}^T \mathbf{w}_ j$ into one matrix product, giving "this cleaner
notation" (≈35:45):

$$\mathbf{h} = g(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \qquad \mathbf{y} = g(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)$$

The parameters of an $L$-layer network are every layer's weights and biases (slide 43):

$$\theta = \lbrace \mathbf{W}_1, \ldots, \mathbf{W}_L, \mathbf{b}_1, \ldots, \mathbf{b}_L \rbrace$$

**The non-linearity is what makes stacking worth doing.** Without it, "a linear layer and another
linear layer, this is just a linear combination. It's still linear" (≈40:25–41:11): two stacked
linear maps are one linear map, and depth adds nothing.

## What two layers can classify

Slide 45 builds the smallest interesting case (≈38:49–40:25):

$$\mathbf{z} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1, \quad \mathbf{h} = g(\mathbf{z}), \quad z_3 = \mathbf{W}_2 \mathbf{h} + b_2, \quad y = \mathbf{1}(z_3 \gt 0)$$

with two inputs $x_1, x_2$, two hidden units $h_1, h_2$, and a thresholded output. The lecturer's
geometric account: each hidden unit is its own ramp over the input plane, because each is a
different linear model. Combining two ramps through the non-linearity gives "a kind of
triangular, almost a pyramidal component that will be higher than everything else", and
thresholding that region gives a **non-linear decision boundary**. "Of course, you can expand this
into many more dimensions", but this is the intuition for "why you start getting non-linear models
when you're taking linear combinations, but with these non-linearities" (≈39:38). The slide's
heat maps of $h_1$, $h_2$, $z_3$ and $y$ are a clipped animation frame in the published PDF, so
the spoken description is the record here.

## Deep nets as composition

Stack $L$ such layers and the network is a chain of functions (slide 54):

$$f(\mathbf{x}) = f_L(f_{L-1}(\ldots f_2(f_1(\mathbf{x}))))$$

Each $f_l$ is a linear map followed by a non-linearity, and the last layer classifies (≈43:29).
With depth "we're getting more abstracted representations", which is the subject of
[representation learning](representation-learning.md). How much a stack like this can represent,
and how efficiently, is [representational power](representational-power.md).

In practice the same computation runs on whole batches of inputs at once, as matrix products over
a batch dimension — see [tensors and batching](tensors-and-batching.md).
