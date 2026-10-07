# Convolution and convolutional layers

A **convolutional layer** applies one small set of learned weights, a **filter** or **kernel**, to
every local patch of a grid-structured input and writes one output per position. It is the
architecture the course builds for grids such as images, in [lecture 4](04-architectures-grids.md).
Covered so far: lecture 4, slides 12–64 and 73–75, ≈13:04–59:38 and ≈1:08:14–1:10:38, plus the
end-of-lecture questions on pooling and video (≈1:16:51 and ≈1:20:43–1:23:48). The architectures
built from convolutional layers are on [skip connections](skip-connections.md); why the layer's
assumptions help is on [inductive bias](inductive-bias.md).

## Where it comes from: classifying overlapping patches

Lecture 4 reaches convolution from a practical problem. A photo of storks in the sky is not one
thing, so classify each patch of it instead (slides 13–17). Fixed patches give a coarse map whose
cells may hold only a wing tip (slide 18), and tiny patches have too little context to classify
(slide 19). The answer is "large but *overlapping* patches": slide a window over the image and, at
each position, predict the class of its centre pixel from the context around it (slides 19–21,
≈16:55–18:27). Labelling every pixel this way is **semantic segmentation** (slide 22). If the
per-patch classifier is just a weighted sum of the patch's pixels, its weights are a convolutional
kernel applied to the whole image (slide 24, ≈20:46).

## Definition

Convolution is a "Linear, shift-invariant transformation" (slide 25), the filtering operation of
signal processing. For a filter with weights $w[k_1, k_2]$, $k_1, k_2 \in \lbrace -K, \ldots, K \rbrace$,
and a bias $b$, the output at position $[n, m]$ is

$$x_{\text{out}}[n, m] = b + \sum_{k_1, k_2 = -K}^{K} w[k_1, k_2] \thinspace x_{\text{in}}[n + k_1, m + k_2]$$

The course writes a convolutional layer with a star (slides 28–29),

$$\mathbf{z} = \mathbf{w} \star \mathbf{x} + b$$

"We are going to use a five-point star in our lectures and in the notes" (≈25:26). Two remarks from
the lecture:

- **It is cross-correlation, strictly.** The definition above "is not quite the same definition as
  familiarly used in signal processing for convolution. Usually, it has like a flipped value." Deep
  learning calls it convolution anyway, partly because "it's very easy to learn how to flip a sign,
  so it doesn't matter so much" (≈24:41–25:26).
- **A filter responds most where the input looks like it.** Applied to a photo, an edge filter
  gives large outputs along edges that match it and the most negative outputs along edges of the
  opposite polarity (slide 25, ≈22:19–23:55).

## Why it suits images

**Locality and weight sharing.** A fully connected layer makes every output depend on every input
(slide 26). A convolutional layer assumes "output is a **local** function of input", and uses "the
same weights (**weight sharing**) to compute each local function" (slides 27–29, ≈23:55–26:13).

**Translation equivariance.** Because every patch is processed the same way, shifting the input
shifts the output and changes nothing else (slide 23):

$$f(\texttt{translate}(x)) = \texttt{translate}(f(x))$$

"you could shift the input, then apply the convolution. Or you could apply the convolution and then
shift the output, and the results would be the same" (≈23:55). This matches a symmetry of the world:
"A bird should look the same no matter where it is in an image" (≈20:46).

## A constrained linear layer

Written as a matrix, a convolutional layer is a linear layer whose matrix is mostly zeros (slides
30–31, ≈26:13–27:00). A filter that looks at three neighbouring inputs puts its three weights
$w[-1]$, $w[0]$, $w[1]$ along the three central diagonals, the same values all the way down, and
zeros everywhere else. A matrix that is constant along every diagonal is a **Toeplitz matrix**
(slide 32):

$$\begin{pmatrix} a & b & c & d & e \cr f & a & b & c & d \cr g & f & a & b & c \cr h & g & f & a & b \cr i & h & g & f & a \end{pmatrix}$$

So a convolutional layer is "a constraint on a standard linear layer", with consequences listed on
slides 32 and 34:

- **Fewer parameters** — "easier to learn, less overfitting".
- **Any input size** — the banded pattern extends to any length, so "Conv layers can be applied to
  arbitrarily-sized inputs (generalizes beyond the training data due to an architectural
  structure!)". An MLP trained on one image size has no weights for a larger one (≈27:45–28:31).
- **Parallelism** — the same small weight set is applied to many patches at once (≈26:13). The
  banded matrix is a way to think about the layer, not how it is computed: "often, we just really do
  compute it patch wise" (≈34:44).

Slide 36 sums up **five views on convolutional layers**: equivariant with translation; patch
processing; an image filter; parameter sharing; and a way to process variable-sized tensors.

## Stacking layers, and receptive fields

Two convolutions with a pointwise non-linearity between them make each output depend on a wider
window of the input, through a function that is the same at every position: "The whole CNN acts like
a (nonlinear) convolutional filter!" (slide 37, ≈30:04–31:37). Each layer has its own filter (≈33:57).

The part of the input that can influence an output is that output's **receptive field**: "what parts
of the initial input have any influence on this part of the output" (slides 61–62, ≈56:30–58:03).
Nothing outside it can affect the output. It grows with filter size and with depth (≈33:10), and a
stack of layers with growing receptive fields is a **spatial pyramid** (≈31:37). Deeper layers
therefore see more of the image, and their feature maps look "much more diffuse", "more semantically
meaningful but less affected by subtle texture" (slide 63, AlexNet, VGG16 and ResNet18,
≈58:03–59:38).

The lecture reports a trend in how networks get large receptive fields: AlexNet used large filters
and fewer layers, while more recent networks "often see filters that are reasonably small, 7 or 5
spatial extent. But maybe then they're actually stacked quite deep" (≈51:51–52:37).

## Channels and filter banks

Images have channels, and so do hidden layers. Each layer "can be thought of as a set of C **feature
maps** aka **channels**", each an $N \times M$ image (slide 43).

- **Multichannel input** (slide 39): one set of weights per input channel, summed into one output
  channel, $\mathbf{x}_ {\text{out}} = \sum_{c} \mathbf{w}[c, :] \star \mathbf{x}_ {\text{in}}[c, :] + b[c]$
  as the slide prints it (with no brackets, so whether the bias is inside the sum is left open).
- **Multichannel output** (slide 40): a **filter bank**, several filters applied in parallel to the
  same input, one output channel each.
- **The general layer** (slide 41): $C_{\text{out}}$ filters, each $K \times K$ spatially with
  $C_{\text{in}}$ channels, mapping
  $\mathbf{x}_ {\text{in}} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ to
  $\mathbf{x}_ {\text{out}} \in \mathbb{R}^{C_{\text{out}} \times H \times W}$ (slide 42):

$$\mathbf{x}_ {\text{out}}[c_2, :, :] = \sum_{c_1 = 1}^{C_{\text{in}}} \mathbf{w}[c_1, c_2, :, :] \star \mathbf{x}_ {\text{in}}[c_1, :, :] + b[c_2]$$

**Counting parameters** (slide 46). Mapping $\mathbf{x}_ l \in \mathbb{R}^{C_l \times N \times M}$ to
$\mathbf{x}_ {(l+1)} \in \mathbb{R}^{C_{(l+1)} \times N \times M}$ with filters of spatial extent
$K_1 \times K_2$ takes $K_1 \times K_2 \times C_l$ parameters per filter and $C_{(l+1)}$ filters. The
lecture's quiz (slides 44–45): an RGB input of $3 \times 128 \times 128$ and a bank of "3x3" filters
producing $96 \times 128 \times 128$ means **27** parameters per filter and **96** filters. The trap is
convention: "When we talk about filter size, we often only talk about the spatial extent of the
filters", because standard layers do not convolve across channels (≈41:45). Filters are normally
square, but need not be (≈44:05).

Nothing forces a layer's feature maps to differ, but they usually do, "because that gives it more
capacity"; when they don't, it is "feature collapse" (≈39:24–40:09).

## Pooling

Pooling summarizes a local window of a layer's outputs without learned weights: **max pooling** takes
the largest value, **mean pooling** the average (slides 48–49). The lecture calls it "a constrained
version, a nonlearned version of a convolutional layer" (≈1:16:51). It has two uses:

- **Across space**, it "achieves stability w.r.t. small translations": a max over a window still
  reports an edge when the edge moves a little (slides 50–52, ≈46:23).
- **Across channels**, it "can achieve other kinds of invariances": a max over edge filters of several
  orientations responds to "any edge, regardless of its orientation" (slide 53, ≈47:10).

## Downsampling, strides and dilation

When only a label is wanted at the end, as in classification, the network **downsamples** as it goes,
so each block is smaller than the last (slides 54–57, ≈47:10–48:47). A **strided** operation folds
the downsampling into the convolution or pooling itself by moving the window more than one position
at a time: "**Strided operations** combine a given operation (convolution or pooling) and downsampling
into a single operation" (slides 58–59). Strides are usually smaller than the filter, so windows still
overlap: "maybe your input kernel is like 7 by 7. Your stride might be 5" (≈49:34). A **dilated**
filter spaces its weights out with zeros between them, so it "Covers a large receptive field with
fewer parameters" (slide 60, ≈50:20). Choosing among these is a trade-off "between the amount of
computation and the receptive field" (≈52:37).

## Implementation

Slide 64, not discussed in the recording, gives a "Basic implementation": `im2col` rearranges the
input into one row per patch, `bmm` does a batched matrix multiplication with the kernel, and `col2im`
rearranges the result back; "or: fft signal processing stuff…". It recommends the timm library. See
[tensors and batching](tensors-and-batching.md).

## Beyond images

Convolution works on any grid. In **time**, a filter slides along a one-dimensional signal (slide 73).
A **video** is a four-dimensional input, channels by two spatial axes by time, and a **3D
convolution** slides a cube-shaped filter "over space and time", often for action recognition (slides
74–75, ≈1:09:04–1:10:38). The lecturer added that temporal convolutions are rarely worth their cost,
and that video models often downsample in time instead (≈1:21:29–1:23:48).

Equivariance is not always wanted, and a **positional encoding** removes it on purpose; see
[neural fields and positional encoding](neural-fields-and-positional-encoding.md).

The lecture closes on the idea in its plainest form (slide 82): convolution "just means: chop up the
image into patches and apply the same function to each patch. This concept appears in almost all
modern architectures, such as CNNs, transformers, NeRFs, and more." In transformers, the lecturer
added, "the patch-wise operation isn't necessarily convolution, use attention instead" (≈1:15:19).
