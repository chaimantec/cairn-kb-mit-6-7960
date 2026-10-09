# Normalization layers

A **normalization layer** rescales a network's activations, usually to zero mean and unit size, so
that every layer sees inputs on a standard scale. The course meets three. **RMS normalization** divides
a vector by its RMS norm. **Layer norm**, which the transformer lecture also calls **token norm**,
subtracts the vector's mean and then divides by the RMS of what is left. **Batch norm** does the same
over the examples of a batch instead of over the entries of one vector. Covered so far: the RMS norm in
[lecture 3](03-approximation-theory.md) (≈13:09–13:57) and [lecture 7](07-scaling-rules-for-optimization.md)
(≈1:00:05–1:01:37); layer norm in the transformer block of [lecture 8](08-architectures-transformers.md)
(slide 38, ≈57:23–58:56); and the practical advice of [lecture 9](09-hackers-guide-to-deep-learning.md)
(slides 10, 42 and 43, ≈21:02–24:06 and ≈1:04:10–1:05:43); and the L2 norm drawn as a map of a whole distribution in
[lecture 11](11-representation-learning-reconstruction-based.md) (slide 9, ≈10:49–11:35).

**Notation.** $x_{\text{in}}$ is the vector a layer normalizes, $x_{\text{in}}[k]$ its $k$-th entry, $d$
its dimension, and $x_{\text{out}}$ the result. The RMS norm of a vector $\mathbf{v} \in \mathbb{R}^d$ is
$\Vert \mathbf{v} \Vert_{\text{RMS}} = \frac{1}{\sqrt{d}} \Vert \mathbf{v} \Vert_ 2$, the root-mean-square
size of its entries (see [norms](norms.md)).

## RMS normalization

Lecture 3 introduced the RMS norm as "a kind of non-dimensional analog" of the Euclidean norm: a vector
of all ones has RMS norm 1 whatever its length (≈13:09–13:57). Lecture 7 adds that a unit RMS norm means
"each $v_i$ is around 1", and that "we often make a big effort to normalize all of the activations of
the layer in the RMS norm. And this is referred to either as RMS normalization, or layer
normalization, or layer norm" (≈1:00:05–1:00:51). "It means that the feature vectors or the activation
vectors are all well-behaved" (≈1:01:37). Lecture 7 then builds its RMS-RMS operator norm and width rule
on that norm; see [scaling rules](scaling-rules.md).

Lecture 9 separates the two names. RMS-norm divides by the RMS norm and nothing else. Layer norm first
subtracts the mean of the vector. "In high dimensions, these things behave almost identically" (≈21:02).
Dividing by the RMS norm puts every output on the unit sphere of the RMS norm: "RMS-norm will tend to map
data points onto the unit hypersphere" (≈21:02–21:48).

## Layer norm, or token norm

In lecture 8's vision transformer block (slide 38), a **token norm** comes before multihead
self-attention and again before the token-wise MLP. It "is often more commonly called in the literature
LayerNorm", and it normalizes each token vector "to have zero mean and unit variance" (≈57:23). Slide 38
writes it as

$$x_{\text{out}}[k] = \frac{x_{\text{in}}[k] - \mathbb{E}[x_{\text{in}}[k]]}{\sqrt{\text{Var}[x_{\text{in}}[k]]}}$$

with the mean and variance taken over the entries of the one token. (The lecturer says "dividing by the
variance"; the formula divides by its square root.) Lecture 9's slide 10 writes the same operation in
two steps. First $x[k] = x_{\text{in}}[k] - \frac{1}{k} \sum_k x_{\text{in}}[k]$, which subtracts the
mean (the slide prints $1/k$ in front of $\sum_k$, using $k$ for both). Then
$x_{\text{out}} = \text{RMS-norm}(x)$.

Why normalize at all? Lecture 8 ties it to lecture 7. "It's good to normalize activations and weight
updates in your network for a lot of reasons", and normalizing activations affects the size of the weight
updates, because backpropagating through the normalization makes the update "a function of the scale of
the activations" (≈58:09–58:56). "Exactly what those are is basically open science. And it's not quite
clear why LayerNorm is right thing to do, but it's what people do" (≈58:56).

## Normalization misbehaves in low dimensions

Lecture 9's slide 10, "Beware of low dimensions", shows what layer norm does to two-dimensional inputs.
Subtracting the mean of a 2D vector loses one degree of freedom, and dividing by the RMS norm loses
another. "So now the outputs have 0 degrees of freedom, and they end up being on these just two points as
opposed to being on a one-dimensional manifold, the circle for RMS-norm" (≈21:48–22:34). The slide's
caption: "In high dimensions, normalization layers can make entries ~N(0,1), whose typical set is
~surface of hypersphere. ← Not so in low dimensions. Many normalization layers behave badly in low
dimensions."

The same holds along the batch axis. "Normalization layers, like batch norm, won't work well if your
batch size is small. Layernorm won't work well if your layer dimensionality is small" (≈22:34). The
extreme case is the lecturer's own bug in pix2pix's first code release: batch norm over a batch of size 1
for a baseline. Subtracting the mean over one example leaves zero, "so that baseline was trivially
beaten" (≈22:34–23:20). Slide 10's advice: "Avoid low dimensions! All tensor dimensions should be big
numbers", which in the recording means "at least 10 and above" (≈24:06).

## The case against batch norm

Lecture 9 lists "Don't use batch norm" among its "good default choices ca 2024" (slide 42). Its reasons:

- It "introduces a strong dependency on batch size (now batch size becomes an even more critical
  hyperparameter)". Batch norm normalizes by the statistics of the batch, so "if I change my batch size, I
  have to retune my whole model" (≈1:04:10–1:04:56).
- It behaves differently at training and test time. At test time there is often one example, and learned
  statistics replace the batch's.
- It "makes distributed computing hard — requires communication between all elements in a batch".
- "Use layer norm instead", "the standard one right now" (≈1:04:56).

Slide 43, "a longer rant I wrote a few years ago", adds three more. A large batch whose activations
happen to be identical is zeroed out as a batch of one is; this is the "bug" the SPADE paper tried to fix.
Batch elements are no longer processed i.i.d., "one reason why the theory of why batchnorm works is still
not really resolved". And the usual fixes add complexity: running batch norm separately on each machine
makes "your results change dramatically depending on how many machines are in your cluster". Slide 56's
"switch to evaluation mode by model.eval() (PyTorch) … no, really" is the everyday form of the
train-test difference.

## What a normalization does to a distribution (lecture 11)

Lecture 11 draws each layer as a map from a cloud of input points to a cloud of output points (slide 9). The L2 norm,
$x_{\texttt{out}}[i] = x_{\texttt{in}}[i] / \lVert \mathbf{x}_ {\texttt{in}} \rVert_ 2$, "and same with RMS norm, and same
with LayerNorm, which is a variation on this", takes all of the data and "will map it to vectors that have norm 1": in two
dimensions onto the circle, in high dimensions onto the hypersphere (≈10:49–11:35). The lecturer had shown this picture "in one
of the last lectures, but I didn't really fully explain it", lecture 9's drawing of the RMS norm and layer norm. One reason the
map is nice: "the numerics are going to be bounded somehow. The vectors will not go to infinity or not go to 0" (≈11:35). The
same lecture lists low-dimensional embeddings' weird interactions "with BatchNorm and LayerNorm" among the reasons an
autoencoder's bottleneck is hard to work with (≈1:17:41); see [autoencoders](autoencoders.md).

## See also

- [Norms](norms.md) — the RMS norm and the RMS-RMS operator norm.
- [Scaling rules](scaling-rules.md) — why unit-RMS activations make the width rule work.
- [Transformers](transformers.md) — the token norm inside the transformer block.
- [Tensors and batching](tensors-and-batching.md) — keeping every tensor dimension large.
