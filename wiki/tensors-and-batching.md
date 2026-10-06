# Tensors and batching

Deep learning computes on whole batches of examples at once, as multiplications of
multi-dimensional arrays — **tensors**. Lecture 1 lists "parallel processing, tensors" as
**expected background** (slide 67) and gives the core idea in about two minutes (slides 68–70,
≈53:30–55:02). Covered so far: [lecture 1](01-introduction.md), plus the course's
[notation](notation.md) handout.

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
