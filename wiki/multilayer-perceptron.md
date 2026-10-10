# Multilayer perceptrons

A multilayer perceptron (MLP) is the network lecture 1 builds up from scratch: linear layers
alternating with pointwise non-linearities. The course treats MLPs as **background it expects you
to have seen** (slide 31), and reviews them so that its notation is fixed before anything new is
built on it. Covered so far in this knowledge base: [lecture 1](01-introduction.md), slides 32–45,
≈25:37–40:25; [lecture 2](02-how-to-train-a-neural-net.md), slides 27 and 50–52 (the MLP as a
computation graph, and its backward pass); [lecture 3](03-approximation-theory.md), slides 15–17
and 27–33 (what ReLU MLPs can approximate, and depth versus width); [lecture 4](04-architectures-grids.md),
slides 3, 26 and 30 (the MLP's strengths and weaknesses as an architecture, and the fully connected
layer that a convolution constrains); [lecture 5](05-architectures-graphs.md), slides 11, 22 and 23 (why an MLP
on an adjacency matrix is not permutation invariant, and the MLP as a graph net over a single node);
[lecture 6](06-generalization-theory.md), slide 8 and ≈40:10–40:58 (how an MLP fits between its
training points, and its last layer as a weighted sum of features); [lecture 7](07-scaling-rules-for-optimization.md),
slide 22 and ≈53:51–56:11 (the neural, tensor and spectral perspectives); [lecture 8](08-architectures-transformers.md),
slides 17–21 and 36, ≈16:59–22:24 and ≈55:49 (token nets as MLPs over vectors, and the token-wise MLP
inside a transformer); [lecture 10](10-architectures-memory.md), slide 23, ≈14:44–15:30 (an RNN without its recurrence); [lecture 11](11-representation-learning-reconstruction-based.md),
slides 10–12, ≈13:08–16:17 (a width-2 MLP's layers drawn as it trains); [lecture 13](13-representation-learning-theory.md), slides 20–25, ≈45:07–1:10:13
(an infinitely wide MLP with random weights as a Gaussian process). See also [activation functions](activation-functions.md),
[representational power](representational-power.md) and [convolution](convolution.md).

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

$$\mathbf{h} = g(\mathbf{W}_ 1 \mathbf{x} + \mathbf{b}_ 1) \qquad \mathbf{y} = g(\mathbf{W}_ 2 \mathbf{h} + \mathbf{b}_ 2)$$

The parameters of an $L$-layer network are every layer's weights and biases (slide 43):

$$\theta = \lbrace \mathbf{W}_ 1, \ldots, \mathbf{W}_ L, \mathbf{b}_ 1, \ldots, \mathbf{b}_ L \rbrace$$

**The non-linearity is what makes stacking worth doing.** Without it, "a linear layer and another
linear layer, this is just a linear combination. It's still linear" (≈40:25–41:11): two stacked
linear maps are one linear map, and depth adds nothing.

## What two layers can classify

Slide 45 builds the smallest interesting case (≈38:49–40:25):

$$\mathbf{z} = \mathbf{W}_ 1 \mathbf{x} + \mathbf{b}_ 1, \quad \mathbf{h} = g(\mathbf{z}), \quad z_3 = \mathbf{W}_ 2 \mathbf{h} + b_2, \quad y = \mathbf{1}(z_3 \gt 0)$$

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

## The MLP as a computation graph, and how it is trained

Lecture 2 redraws the MLP as a chain of nodes, $\mathbf{x} \to$ `linear` $\to \mathbf{z} \to$
`relu` $\to \mathbf{h} \to$ `linear` $\to \mathbf{y}$, and calls it "easy to represent as a
computation graph" (slide 27, ≈29:36). Training it means backpropagating through that chain. For a
linear layer the forward pass is $\mathbf{W}\mathbf{x}$, and the backward pass multiplies the
gradient by the same $\mathbf{W}$ — "just in a different order", or, written with column-vector
gradients, by $\mathbf{W}^{\mathsf{T}}$. The ReLU becomes a diagonal gating matrix, so the whole
backward pass is a chain of matrix products (slides 46–52, ≈44:17–52:51). See
[backpropagation](backpropagation.md).

## What ReLU MLPs can approximate (lecture 3)

Lecture 3 asks what functions an MLP can express, and gives a precise meaning to "a three-layer
ReLU network": "the thing with three weight matrices and two ReLU functions that follow the first
weight matrix and the second weight matrix. And then the third weight matrix doesn't have a ReLU"
(≈17:48). Its theorem (slide 10) says such a network, with $N = 4d(L/\epsilon)^d$ neurons in
total, can approximate any $L$-Lipschitz function on the $d$-dimensional unit hypercube to $L_1$
error below $2\epsilon$. In the construction the first two layers build boxes out of ReLUs and the
third weights them (slides 15–17). The cost is exponential in $d$.

Two facts about ReLU MLPs from the same lecture are worth keeping:

- **A ReLU MLP is piecewise linear** in its input (slide 27): ReLU is piecewise linear, and sums,
  compositions and scalar multiples of piecewise linear functions are piecewise linear too.
- **Depth multiplies, width adds.** Counting the places where the slope changes, a network of
  width $n$ and depth $L$ has at most $(2n)^L$ per unit, so a deep narrow network can compute functions a
  shallow one needs exponentially many neurons for (slides 30–33).

Both are developed on [representational power](representational-power.md).

## The MLP as an architecture, and what replaces it (lecture 4)

Lecture 4 opens the course's architecture lectures by weighing the MLP as a design (slide 3,
≈1:32–4:36). In its favour: it is **universal**; it is **simple**, "one of the only models that we
actually have really elegant theory for"; and it is **embarrassingly parallel**, since the pointwise
non-linearity is independent for every neuron and the linear layer independent for every example in a
batch. Against it: **weak inductive biases** ("there's not a lot of structure or intuition baked into
this model"); being **sample inefficient**, or data hungry; and **dense layers that take a lot of
compute**, since a high-resolution image flattened into a vector has thousands of inputs, each
connected to every output (≈4:36).

The lecture's first alternative keeps the linear layer and constrains it. A fully connected layer,
$\mathbf{x}_ {\text{out}} = \mathbf{W}\mathbf{x}_ {\text{in}} + \mathbf{b}$, multiplies by a dense
matrix in which "every single one of those values matters" (slides 26 and 30). A convolutional layer
is the same product with a matrix that is zero except for a band of shared weights along the diagonal
(slide 31): fewer parameters, the same function at every position, and applicable to inputs of any
size, which an MLP is not: applied to a larger image, "You just wouldn't have weights for some of that
size" (≈28:31). The MLP's own lack of bias is what lets lecture 4's 5-layer ReLU network fit training
points perfectly and extrapolate badly beyond them (slide 7). See [convolution](convolution.md) and
[inductive bias](inductive-bias.md).

Lecture 5 adds the graph view. Feeding a graph's flattened adjacency matrix and node attributes to an
MLP makes the output depend on how the nodes happen to be numbered, so it is not permutation
invariant (slide 11, ≈17:51–22:31). A graph neural network is the alternative, and the lecture shows
the MLP as its simplest case: "an MLP is just a graph net that's a single node". With no neighbours
the aggregation is trivial and the update $\sigma(\mathbf{W}_ {\text{self}} \mathbf{h})$ is a linear
layer and a pointwise non-linearity, repeated (slide 23, ≈54:20–55:51). Conversely, unrolled message
passing "looks a lot like a neural network MLP", except that its neurons are vectors: aggregation is
akin to a linear layer and the update to a pointwise one (slide 22, ≈48:53–49:42). See
[graph neural networks](graph-neural-networks.md).

## How an MLP fits between the data (lecture 6)

Lecture 6 uses a 3-layer ReLU MLP as its first example of generalization (slide 8, ≈10:48–12:21). Fit to
the same scalar data as a "filing cabinet" that memorizes each training point and predicts 0 elsewhere,
the MLP also passes through every point, but between and beyond them it draws a continuous piecewise-linear
curve rather than dropping to zero. Both have zero training error; what differs is "how you interpolate
and extrapolate from the training data", and the MLP's smooth kind is the better bet because "most
functions in the world are going to be smooth as opposed to spiky".

Asked how polynomial regression relates to deep nets, the lecturer described an MLP's last layer as
regression on features: "you can think of the last layer of an MLP as a linear combination of some basis
functions, of some features. So every neuron on the previous layer is a feature of the data." Polynomial
features give polynomial regression, sines and cosines give Fourier features, "and in deep learning,
they'll be learned functions" (≈40:10–40:58). See [generalization and double
descent](generalization-and-double-descent.md).

## Three perspectives on a network (lecture 7)

Lecture 7 describes three ways of seeing the same layers (slide 22, ≈53:51–55:26). The **neural
perspective** sees nodes connected by edges. The **tensor perspective** recognizes a weight matrix, then a
ReLU, acting on a vector, "and now, I'd start to describe my neural network as like matrix
multiplications". The **spectral perspective** writes every weight matrix through its singular value
decomposition, drawn as orthogonal, diagonal and semi-orthogonal factors: "not every matrix has
eigenvalue decomposition, but every matrix has a singular value decomposition". The point is not to train
in that form, but to picture training as changing the singular values, by neither too much nor too little
each step. See [norms](norms.md) and [scaling rules](scaling-rules.md).

## Token nets: the MLP over vectors (lecture 8)

Lecture 8 builds the transformer by lifting the MLP from scalars to vectors. Networks over tokens have
"two basic operations", just as MLPs do: a linear combination and a unit-wise nonlinearity (≈16:59–17:47).
The linear combination of neurons $\mathbf{x}_ {\text{out}} = \mathbf{W} \mathbf{x}_ {\text{in}}$
becomes one of tokens, $\mathbf{T}_ {\text{out}} = \mathbf{W} \mathbf{T}_ {\text{in}}$, which scales whole
token vectors and is "a low rank transformation over the neurons" (slide 17, ≈19:20). The pointwise ReLU
becomes a token-wise function $F_\theta$ applied to every token, and "F is typically an MLP", so
"transformers are like a meta architecture. Like, the units of the architecture are other neural
networks" (slide 18, ≈20:06–20:52). Slide 21 draws a neural net and a token net side by side ("The motif
is the same", ≈22:24), and slide 36 the MLP beside the vanilla transformer, with self-attention in place
of the linear layers and a token-wise MLP in place of the ReLU (≈55:49). One more difference: an MLP's
ReLU has no parameters, while a transformer's token-wise MLP is learned (≈25:28). See
[transformers](transformers.md).

## An RNN without its recurrence (lecture 10)

Lecture 10's simplest recurrent network (slide 23) is

$$\mathbf{h}_ t = \sigma_1(\mathbf{W}\mathbf{h}_ {t-1} + \mathbf{U}\mathbf{x}_ {\texttt{in}}[t] + \mathbf{b}), \qquad \mathbf{x}_ {\texttt{out}}[t] = \sigma_2(\mathbf{V}\mathbf{h}_ t + \mathbf{c})$$

where $\mathbf{x}_ {\texttt{in}}[t]$ is the input at time $t$, $\mathbf{h}_ t$ the hidden state, $\mathbf{W}$, $\mathbf{U}$
and $\mathbf{V}$ the recurrent, input and output weights, $\mathbf{b}$ and $\mathbf{c}$ biases and $\sigma_1$, $\sigma_2$
nonlinearities. Remove $\mathbf{W}$, and the map from input to output, beyond scalars, "starts looking a little bit like that
multi-layer perceptron that we're very familiar with. So the only difference here is that recurrence" (≈14:44–15:30). See
[recurrent neural networks](recurrent-neural-networks.md).

## Watching an MLP's layers train (lecture 11)

Lecture 11 draws every layer of a small MLP as a map from one cloud of points to the next (slide 10). The network is "a
three-layer MLP because there's three linear layers, so our convention is three layers", with ReLUs between them and a softmax on
top, trained with cross-entropy; the data are a red cloud at the origin surrounded by blue points, which no hyperplane separates
(≈13:08–13:53). Every layer has width 2, so "There's nothing hidden … It's just the raw activation values at each of these layers"
(≈14:39). As it trains the red points move away from the blue: a linear layer shifts them, "the ReLU snaps it back onto the axes",
the next linear layer skews, until they spread out on a line and the softmax puts each class at its one-hot label (≈13:53–14:39).
"Each of the layers now can be understood as a different representation of the data distribution and a better and better
representation" for the task (≈14:39). Slide 12 trains it with SGD and with steepest descent in the spectral norm; see
[steepest descent](steepest-descent.md) and [representation learning](representation-learning.md).

## An infinitely wide MLP with random weights (lecture 13)

[Lecture 13](13-representation-learning-theory.md) studies the MLP before any training. Sample its weight matrices
$\mathbf{W}_ 1, \ldots, \mathbf{W}_ L$ at random and it computes a random function (slide 20); sample 1,000 times and scatter-plot
the outputs on two inputs, and the cloud shows how the architecture relates them (slide 21). For a three-layer MLP of hidden width
1000, the outputs on a CIFAR-10 truck and a slightly noised copy lie along a thin diagonal line, and on a heavily noised copy they
spread into a wider ellipse (slide 22, ≈47:28–49:01). As the width goes to infinity, the outputs on any finite set of inputs become
jointly Gaussian, so the random MLP is a [Gaussian process](gaussian-processes.md) whose covariance function depends on the depth
and the non-linearity (slide 23). With non-linearity $\sqrt{2} \thinspace \operatorname{relu}$ and weight variance one over fan-in,
that covariance is the compositional arccosine kernel, which applies one fixed function $L - 1$ times to the inputs' scaled dot product
(slide 25, ≈1:07:55–1:10:13).
