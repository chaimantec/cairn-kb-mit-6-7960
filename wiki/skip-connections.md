# Skip connections: encoder–decoders, U-net and ResNet

A **skip connection** carries a layer's input past one or more layers and combines it with their
output, usually by adding or concatenating. The course introduces it in
[lecture 4](04-architectures-grids.md)'s tour of convolutional architectures, as the fix for the
information an encoder–decoder's bottleneck throws away (U-net) and as the defining feature of
ResNet. Covered so far: lecture 4, slides 66–72, ≈59:38–1:08:14. Every figure in that part of the
deck is excluded from OCW's licence, so this knowledge base describes them in prose only (in the
[slide file](../raw/slides/04-architectures-grids.md)).

## The problem: what a bottleneck loses

An **encoder–decoder** (slide 67) compresses an image through repeated convolution, non-linearity
and subsampling "to get down to some very low dimensional representation", a vector $\mathbf{z}$, and
then expands it again through convolution, non-linearity and **upsampling**. Trained so that "the
decoded output [matches] the encoded input as close as possible", it learns a compact representation
of images; the same structure is built into variational autoencoders, and masked autoencoders work
similarly with attention in place of convolution (≈1:00:25–1:02:00). The two halves can be used
separately: the encoder to represent images, the decoder to generate them (≈1:02:00). See
[representation learning](representation-learning.md).

The same shape can produce a spatial output, such as a segmentation map. But "this bottleneck in the
middle" is "both a blessing and a curse. It's forcing the model to learn something that's kind of
semantically useful, but it also means that often it's not possible with an architecture that just
has downsampling, then upsampling to get a really precise output that's the same size as the input
because… You've lost a lot of the information about the fine-grained details in that spatial
structure" (≈1:02:46).

Dropping the bottleneck altogether — convolution, ReLU, convolution, softmax, all at full resolution
(slide 69) — keeps every detail but "can be really computationally expensive" (≈1:03:33).

## U-net

U-net gets "the best of both worlds" (slides 68 and 70, ≈1:03:33–1:05:09). It keeps the encoder and
decoder, and adds **skip connections** across the "U": each encoder stage's output is passed,
unchanged, by the identity, to the decoder stage of the same size. As a result, the network "is not
forced to explicitly try to shove a bunch of fine-grained spatial detail into some very low rank,
non-spatially dimensional embedding, which is somewhat impossible. It's able to learn the semantic
details through this information bottleneck, but then it also gets a lot of the spatial detail
through those skipped connections." The lecture calls U-net "a really old architecture" and "still one
of the most widely used architectures for semantic segmentation", for example in remote sensing and
land-cover mapping.

## ResNet

The skip connection is also "one of the fundamental underpinnings of what we call the ResNet
architecture" (slide 71, ≈1:05:09–1:06:42), which the lecture credits to Kaiming He and calls "like
10 years old at this point" and still in constant use. In a ResNet every layer has one, a **residual
connection**. With $F$ the transformation a layer applies,

$$\mathbf{x}_ {\text{out}} = F(\mathbf{x}_ {\text{in}}) + \mathbf{x}_ {\text{in}}$$

and, "if you want to change dimensionality" — for instance to downsample inside the network — the skip
path gets a learned linear map $\mathbf{W}$:

$$\mathbf{x}_ {\text{out}} = F(\mathbf{x}_ {\text{in}}) + \mathbf{W}\mathbf{x}_ {\text{in}}$$

**Learning its own depth.** The lecturer's intuition: the identity path lets the network decide, layer
by layer, whether to transform its input or pass it on, "so then the model itself can implicitly learn
the optimal depth for the task that you want to learn", without "explicitly training a bunch of
different architectures side by side" (≈1:05:57).

**Add, not choose.** Asked whether the skip becomes an extra channel or a choice between paths: the
output of the layer and the residual "are the same size. And then you're just adding them together",
though "there's different implementations of this that will have some of learned weight." Even plain
addition loses nothing that matters, since "there's some version of the next layer that could
basically learn how to ignore certain dimensions", and when the dimensionality changes, the
$\mathbf{W}$ on the skip path "could be learned to be all zeros" (≈1:07:27–1:08:14).

## Where it goes next

Lecture 4 says "that same type of skipped connection is also something that gets surfaced via
self-attention in transformers, which we'll talk about more in the transformer architecture"
(≈1:07:27). In the recorded schedule that is lecture 8; see the [course map](course-map.md).
