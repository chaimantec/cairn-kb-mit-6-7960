# Tensors and batching

Deep learning computes on whole batches of examples at once, as multiplications of
multi-dimensional arrays — **tensors**. Lecture 1 lists "parallel processing, tensors" as
**expected background** (slide 67) and gives the core idea in about two minutes (slides 68–70,
≈53:30–55:02). Covered so far: [lecture 1](01-introduction.md), plus the course's
[notation](notation.md) handout; [lecture 2](02-how-to-train-a-neural-net.md), slides 9 and 45 (batches
in stochastic gradient descent and in backpropagation); [lecture 4](04-architectures-grids.md),
slides 39–47, 64 and 74 (channels, filter banks, implementing convolution as a batched matrix product,
and video as a four-dimensional input); [lecture 8](08-architectures-transformers.md), slides 16, 34 and 39 (a set of tokens
as an $N \times d$ matrix, and attention as matrix products); [lecture 9](09-hackers-guide-to-deep-learning.md), slides 7–12, ≈10:52–30:17
(inspecting tensors, prime-sized dummy dimensions, dtype casts, einops, and keeping every dimension large).

## Why batch

Training sums a loss over examples (see [gradient descent](gradient-descent.md)). Each example's
loss is computed the same way, through the same layers, and "it's all going to be summed anyway.
So you can do it in parallel" (≈53:30). Slide 68 draws three copies of the pipeline — a clownfish,
a chameleon and a grizzly bear, each running through the same layers to its own loss — with the
three losses meeting in a $\Sigma$. In its corner the batch becomes a single grid of **features by
images**: "you can stack them, and now essentially you get this matrix that's features multiplied
by images" (≈53:30–54:16).

## Everything is a tensor

A tensor is "just a multi-dimensional matrix", and in a deep network "everything is a tensor"
(≈54:16). Slide 69 makes the point that **each layer is a representation of the data**: a grid
whose rows are examples (clownfish, chameleon, grizzly bear, …) and whose columns are features
such as "Furry?", "Is a fish?", "Size", "# Stripes". The shadings on the slide are illustrative,
not real values.

The two-layer network from the [MLP](multilayer-perceptron.md) review then becomes a chain of
whole-batch operations (≈54:16–55:02). Stack the input vectors along a batch dimension and
multiply by the weight matrix. Apply the pointwise non-linearity to the result, multiply again,
and apply the non-linearity once more to get the output. Slide 70 sets this up for a batch of
three: the input matrix $\mathbf{X}$ has one row per example ($N_{\text{batch}}$ rows) and one
column per input feature ($x_1$, $x_2$). The rest of that build is cut off in the published PDF.
Its text layer names the matrices of the full sequence — $\mathbf{X}$, $\mathbf{W}_ 1$,
$\mathbf{Z}_ 1$, $\mathbf{H}_ 1$, $\mathbf{W}_ 2$, $\mathbf{Z}_ 2$, $\mathbf{Y}$ — but not how they
are laid out, so this page does not reconstruct the matrix equations.

## Why it mattered

"This is why the fact that GPUs can do so many multiplications in parallel was important, because
we can rewrite all of these models as just essentially sets of multiplications" (≈55:02). It is
the same point the history makes about AlexNet in 2012: GPUs, built "to handle massive-scale
parallel multiplications for graphics", were repurposed for training (≈17:54–18:44). The lecture
says this "in a lot more detail as well in the future".

## Index conventions

The course's [notation](notation.md) handout fixes how tensors are written and indexed:

- Tensors are usually **lowercase bold**, $\mathbf{x}$, whatever their number of dimensions.
- In code-like settings an activation on layer $l$ may be indexed $\mathbf{x}_ l[b, c, n, m]$:
  $b$ is the element of the batch, $c$ the channel, and $n, m$ are spatial coordinates.
- **Channels come first**: $\mathbf{x} \in \mathbb{R}^{C \times N \times M \times \cdots}$.
- Transformers are the exception: a set of tokens is an $N \times d$ matrix.
- "Dimension" is used both for one coordinate of a vector and for the number of axes of an
  array ("a 4D tensor").

## Batches in training

Lecture 2 adds what batching does to the gradient. Stochastic gradient descent takes each step on
a batch rather than the whole dataset. A batch of 1 updates after every example, and a batch of the
whole dataset is ordinary gradient descent. The batch gradient is a noisy estimate of the full one,
and the noise is worse when categories are unevenly represented (slide 9, ≈6:15–8:35). For
backpropagation the batch is easy to handle. The cost is the average of per-example costs,
$J = \frac{1}{N} \sum_{i=1}^{N} J_i$, so its gradient is the average of the per-example gradients,
"because when you take a derivative, it can move inside the sum" (slide 45, ≈43:30–44:17). With
large-memory GPUs "that batch size could be thousands" (≈44:17). See
[gradient descent](gradient-descent.md) and [backpropagation](backpropagation.md).

## Channels and grids (lecture 4)

Lecture 4 is where the channel axis earns its place. A colour image has three channels, and every
convolutional layer turns $C_{\text{in}}$ channels into $C_{\text{out}}$:
$\mathbf{x}_ {\text{in}} \in \mathbb{R}^{C_{\text{in}} \times H \times W} \to \mathbf{x}_ {\text{out}} \in \mathbb{R}^{C_{\text{out}} \times H \times W}$
(slide 42), with each layer "a set of C **feature maps** aka **channels**" (slide 43). The parameter
count follows from the shapes: mapping $C_l$ channels to $C_{(l+1)}$ with $K_1 \times K_2$ filters
takes $C_{(l+1)}$ filters of $K_1 \times K_2 \times C_l$ parameters each (slide 46). One slide of the
deck, slide 47, writes shapes channels last, $[H \times W \times 3]$, against the course convention.
A video adds a time axis, "a four dimensional input" of channels, two spatial axes and time (slide 74,
≈1:09:53). See [convolution](convolution.md).

Slide 64 gives the standard way a convolution becomes a batched matrix product, the operation GPUs are
fast at: `im2col` rearranges the input into one row per patch, `bmm` multiplies the rows by the kernel,
and `col2im` puts the result back into an image. The recording does not discuss it.

## Tokens as a matrix (lecture 8)

Lecture 8 writes a set of $N$ tokens, each a vector in $\mathbb{R}^d$, as a matrix
$\mathbf{T} \in \mathbb{R}^{N \times d}$ whose rows are the transposed tokens, $N$ tokens by $d$
channels (slide 16, ≈16:14–16:59). This is the exception to channels-first noted above. With that
layout, a transformer layer is a handful of matrix products. The queries of every token are
$\mathbf{T}_ {\text{in}} \mathbf{W}_ q^{\mathsf{T}}$, the attention matrix comes from the product of
queries and transposed keys, and the output is the attention matrix times the values (slide 34): "everything
just becomes these matrix multiplies in a very simple, notational form" (≈46:32). The pseudocode of slide 39
is a loop of `nn.matmul` calls, which "maps wonderfully onto modern compute that loves matrix multiplies"
(≈1:00:29). The lecturer also names the fit to GPU hardware as a reason tokens all have the same size
(≈13:56). See [transformers](transformers.md).

## Inspecting and reshaping tensors (lecture 9)

Lecture 9 treats tensor bookkeeping as a main source of bugs. "The data as it is loaded is not always the
data as it is stored" (slide 7). The recipe is to inspect the tensor right before the forward pass, with
slide 8's `inspect_data`, which prints its type, shape, `requires_grad`, range, mean and variance.
"The shape of the tensors in deep learning are super critical" (≈14:49).

Slide 11 adds three checks. **Summary statistics** catch values in $[0, 255]$ where the model expects
$[0, 1]$. **Shape**: test with "dummy data of prime dimensions: there are no common factors, so mistaken
reshaping/flattening/permuting will be more obvious", since "a 64x64x64x64 array can be permuted without
knowing". With a different size on every axis, a wrong permutation throws a shape-mismatch error
(≈24:51–26:22). **Type**: "check for casting, especially to lower precision. What's -1 for a byte?" A
standardized tensor cast to uint8 loses its negative values (≈26:22–27:10).

"A lot of your code will just be reshaping tensors" (slide 12): "transposing, permuting, reshaping,
flattening, unflattening, unsqueezing, squeezing. This is like half the code in PyTorch" (≈27:10). PyTorch's
reshape is row-contiguous, as a student answers (≈27:56). The lecturer recommends einops, whose
`rearrange(ims, 'b h w c -> h (b w) c')` names every axis and flattens batch and width into one, and whose
reverse pattern undoes it (≈28:43–30:17).

Slide 10 asks for large tensors along every axis: "All tensor dimensions should be big numbers:
[BxNxMxC] data batches, [NxM] weights", "at least 10 and above" (≈24:06), because normalization layers
misbehave in low dimensions; see [normalization layers](normalization-layers.md). For batch size and GPU
utilization, slide 50 says "use biggest batch size that will fit in memory", and slide 69 "increase batch
size until ~100% utilization".
