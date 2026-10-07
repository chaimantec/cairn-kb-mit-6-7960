# Lecture 4 — Architectures: Grids

**Lecturer:** Sara Beery ·
**Video:** [youtube.com/watch?v=bxVkZ4M-hIE](https://www.youtube.com/watch?v=bxVkZ4M-hIE) (84 min) ·
**Slides:** [`mit6_7960_f24_lec4.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec4.pdf)
(84 pages; the deck is titled "Architectures for Grids"; transcribed slide by slide in [`raw/slides/04-architectures-grids.md`](../raw/slides/04-architectures-grids.md)) ·
**Transcript:** [`raw/transcripts/04-architectures-grids.md`](../raw/transcripts/04-architectures-grids.md)

## What this lecture establishes

This is the first of the course's architecture lectures. It argues that an architecture is a
**hypothesis about the function to be learned**: a multilayer perceptron can approximate anything,
but it assumes almost nothing, so it needs a great deal of data, whereas an architecture that
builds the right structure in can find a good solution from little data and generalize outside
the range the data covered. The lecture then builds the convolutional layer for grid-structured
data such as images. It starts from classifying overlapping image patches and arrives at the
convolution: a linear layer whose weights are local and shared across positions, which makes it
equivariant to translation, cheap, and applicable to inputs of any size. From there it covers
multiple channels and filter banks, pooling, downsampling, strides and dilation, receptive fields,
and a tour of architectures built from convolutions: classification networks, encoder–decoders,
U-net and ResNet. It ends with convolution in time and over video, positional encoding for when
translation invariance is unwanted, and neural fields (SIREN and NeRF).

The lecturer says early on that the convolutional material "might be a review for some of you,
but we're still going to spend some time on it" (≈0:46). Positional encoding gets "a first
touch" here and is developed in the transformers lecture (≈1:32, ≈1:15:19); the
[course map](course-map.md) lists it as lecture 8.

**What is in the deck but not in the picture here.** OCW excludes 35 of the deck's slides from its
licence, among them every figure built on the stork and heron photographs: the patch
classification, semantic segmentation, feature maps, receptive fields, and the encoder–decoder,
U-net and ResNet diagrams. Those slides are described in prose below and in the slide file, but
have no image in this knowledge base. The figures shown here are the deck's own diagrams and plots.

**Notation on this page.** The slides write convolution with a star, $\mathbf{w} \star \mathbf{x}$;
asked whether it is "always a five-point star", the lecturer said "we are going to use a
five-point star in our lectures and in the notes" (≈25:26). A lowercase bold $\mathbf{w}$ is a
convolutional filter (kernel) and a capital $\mathbf{W}$ a dense weight matrix, as on the slides.
$\mathbf{x}_ {\text{in}}$ and $\mathbf{x}_ {\text{out}}$ are a layer's input and output, and
tensors are indexed channels first, $\mathbf{x}[c, n, m]$, following the
[course notation](notation.md). One slide departs from that: slide 47 writes image and feature-map
shapes channels last, $[H \times W \times 3]$.

## Why build better architectures?

### What the MLP gives you, and what it doesn't

The lecture opens on the architecture used so far, the [multilayer perceptron](multilayer-perceptron.md):
a linear combination of neurons, a neuron-wise non-linearity, and another linear combination
(slide 3, ≈1:32–4:36). Slide 3 lists its pros and cons.

![Slide 3: a two-layer network drawn bottom to top, dense connections, a neuron-wise non-linearity, dense connections, beside the MLP's pros (universal, simple, embarrassingly parallel) and cons (weak inductive biases, data hungry, expensive dense layers)](../raw/images/04-architectures-grids/slide-3.jpg)

*Slide 3 — the multilayer perceptron weighed as an architecture.*

- **Universal.** "You can build any model and prove that you can build any model with this type of
  architecture" — the result of [lecture 3](03-approximation-theory.md).
- **Simple, with elegant theory.** "This is one of the only models that we actually have really
  elegant theory for", because it "maps well into our mathematical" tools.
- **Embarrassingly parallel.** The pointwise non-linearity is computed independently for every
  neuron, and the linear layer independently for every example in a batch, so GPUs can do it all at
  once (≈2:17–3:03).
- **Weak inductive biases.** "There's not a lot of structure or intuition baked into this model."
- **Sample inefficient, or data hungry.** Learning a complex function with an MLP "might require so
  much data that it's not actually tractable, like more data than exists on Earth" (≈3:48).
- **Dense layers are expensive.** A high-resolution photo flattened into a vector has thousands of
  input dimensions, and a dense layer multiplies all of them by all of its outputs (≈4:36).

### The hypothesis space

Slides 4 to 6 draw learning as a picture of function space (≈4:36–7:40). A grey box is the set
of "All mappings $\mathcal{X} \to \mathcal{Y}$", and somewhere in it is the true solution. The
training data rules out most of the box: what remains is a green ellipse of functions that "Fit
the data". Optimization picks one of those, the learned solution, which "will fit the data, but
it won't necessarily be very close to the true solution."

Asked what more data would do, a student answered that the green region shrinks, and the lecturer
agreed: "there will be fewer models that fit all of the data", so the learned solution lands
closer to the truth (slide 5, "Effect of adding more data", ≈6:08). "But what if you don't have
more data?" The alternative is to **define a hypothesis space** $\mathcal{F}$, a region of the
box that the architecture allows, which "is essentially going to shine a spotlight on the space of
all possible models". It is "a stronger inductive bias or a stronger prior over what we think the
model is going to take the form of", and the solution is then sought where $\mathcal{F}$ meets
the region that fits the data (slide 6, ≈6:55).

![Slide 5: the hypothesis-space box again, with the green region of functions that fit the data now a thin sliver, and the true and learned solutions close together inside it](../raw/images/04-architectures-grids/slide-5.png)

*Slide 5 — the effect of adding more data: fewer functions fit, so the learned solution lands nearer the truth.*

![Slide 6: a grey box of all mappings from X to Y, a green ellipse of functions that fit the data, and a glowing circle F, the hypothesis space, overlapping its top; the true and learned solutions sit close together in the overlap](../raw/images/04-architectures-grids/slide-6.jpg)

*Slide 6 — "Less data, better architecture": the hypothesis space F cuts down the functions that fit the data, so the learned solution lands near the true one.*

Slide 6's conclusion: "We can pin down truth *either* by adding more data, or by using a more
constrained architecture." In the lecture's words, "more data helps. Better architecture or
stronger inductive bias helps. And those two things combined is ideal" (≈7:40). See
[inductive bias](inductive-bias.md).

### A one-dimensional example

Slides 7, 8 and 10 make the picture concrete with a scalar function $y = f(x)$ on $-4 \le x \le 4$,
fitted three times with more and more training data (≈7:40–12:17). The true function, a dashed
green curve, wiggles: a peak near $x = -3$, a trough near $x = -2$, and a tall peak near $x = 3$.

**A 5-layer ReLU network** (slide 7), "densely connected layers with point-wise ReLU-based
activations". Fitted to one point, the learned solution "is just this flat line". With a few more
points it fits "slightly better in the region where we add data", but it is "still not a very good
solution". With a lot of data it fits "really, really nice[ly] in the distribution where you have
data, but you get really poor generalization out of distribution" (≈7:40–8:26).

![Slide 7: three panels of a wiggly dashed green true function with a blue learned curve from a 5-layer ReLU network, fitted to one, a few and many training points; the fit follows the data where there is data and goes flat or straight beyond it](../raw/images/04-architectures-grids/slide-7.jpg)

*Slide 7 — a 5-layer ReLU network fits where the training points are, and nowhere else.*

**The right model** (slide 8). Now the learner is told the form of the answer,
$y = ax + \sin(bx^2)$, and has only to find $a$ and $b$: "that's very specific. It turns out that
it's exactly the correct model for this function." With a single point even this model cannot
tell where in its family the truth lies. But "even with very, very, very few data points, we start
getting a really nice fit", and the fit holds over the whole range, far outside the data (≈8:26–9:13).
The slide's two conclusions:

> Architectures enable us to generalize *outside the training distribution*.

> A good architecture is one that can represent the true function and is otherwise minimal (and is
> also easy to search over via gradient-based learning, easy to parallelize, fast on GPU, etc).

![Slide 8: the same three panels for a model of the form y = ax + sin(bx²); with one point the blue curve oscillates wildly, with a few points it matches the green curve across the whole range](../raw/images/04-architectures-grids/slide-8.jpg)

*Slide 8 — with the exactly right hypothesis, a few points pin down the whole function, including where there is no data.*

**A 5-layer sin-net** (slide 10). In between the two is a network whose activations are sinusoids
rather than ReLUs, the SIREN of Sitzmann et al. (2020). It does not recover the true function, but
"even just with a couple data points, we start seeing some periodicity that maybe starts to match
the periodicity in the function." Asked why the output would be periodic, a student answered that
it is built from sines, and the lecturer agreed: "we've just added an inductive bias in our model
architecture that says that the distribution should be periodic. And then it tries to find
solutions that are periodic outside of the distribution where we have training data" (≈11:31–12:17).
See [activation functions](activation-functions.md).

![Slide 10: the same three panels for a 5-layer network with sine activations; the blue curve fits the training points and then wanders in waves rather than going flat](../raw/images/04-architectures-grids/slide-10.jpg)

*Slide 10 — sine activations build in periodicity, so the network extrapolates in waves rather than flat lines.*

### Preview: fitting an image

Slide 9 previews a result from "another MIT professor, Vincent Sitzmann": SIREN, with sinusoidal
activations, "fits images more efficiently" (≈9:59–11:31). The task is to fit one photograph,
treated as a function from pixel position to intensity ("a function x,y —> l"), by sampling pixels
as training data, and the figure compares a ReLU network, a tanh network, a ReLU network with
positional encoding ("ReLU P.E."), an "RBF ReLU" network and SIREN. The lecturer's reason why the
sine network should win comes from signal processing: "A Fourier basis is a good basis for
representing images, and the fundamental building blocks of Fourier basis are sinusoids." She
describes the fits as they develop: "you start getting a pretty good approximation from SIREN much,
much faster than you do from some of these other models. And, of course, after sampling enough
points, eventually, all the models are getting reasonably good." In the PDF the figure is a single
frame in which none of the five fits yet shows anything recognisable. Slide 9 is excluded from
OCW's licence, so it has no image here.

The slide adds a caveat worth keeping: "this result may be due to improved approximation ability
but it might also be due to improved optimization ability; these two effects are typically coupled
in experiments." This is the approximation-versus-optimization split of
[lecture 3](03-approximation-theory.md), and see [representational power](representational-power.md).

A student asked how the sinusoid's period relates to the number of pixels. The lecturer's answer:
the Fourier basis uses sines of many periods, and the image is rebuilt from different periodicities
and weights, "one of the reasons that we can compress images really well with things like the fast
Fourier transform" (≈12:17–13:04).

## From image patches to convolution

### One label is not enough

The section on convolutional networks opens on a photograph of storks in flight against the sky
(slide 12, ≈13:04–14:37). Asked what is in it, students answered "birds", "sky", "birds and sky",
and "nature". The lecturer liked the last answer for what it shows, a "trade-off between generality
and specificity", but her point was that "a single label is usually insufficient to represent the
semantic concepts that are captured in images", and this is "a pretty simple image" compared with
Times Square or a photo of the lecture hall. In complex scenes, "there's often maybe different
categories or different bits of information that are somewhat locally constrained", which suggests
breaking the problem into one that looks at location as well as semantics.

### Classify each patch

The first attempt cuts the image into a grid of patches and classifies each one separately
(slides 13 to 17, ≈15:22–16:08). Slide 13 cuts the photo into an 8-by-5 grid; one patch at a
time goes to a "Classifier", which says "Bird" or "Sky"; and the labels fill an output grid of the
same shape. The result "is actually giving us an image output. It's a bit like a low-resolution
image."

The problem is on slide 18: "What if objects don't fit neatly into these patches? How to increase
the resolution of the output map?" A patch holding only "the tip of the bird's wing" is ambiguous:
should it be "bird" if there is any bird at all, or whatever most of its pixels show (≈16:08)?
Smaller patches give a finer map, but "at the far extreme, the patches are a single pixel. From a
single pixel, you don't have enough context to determine what the semantic category is" (slide 19,
≈16:55).

### Large, overlapping patches

"The winning idea here with convolutional neural networks is we use large but overlapping
patches" (slide 19, ≈16:55). A window large enough to see the whole bird slides across the image,
and for each position the classifier predicts **the class of the centre pixel**, given the context
around it (slide 20, "What's the object class of the center pixel?", ≈17:41). The training data
become pairs of a patch $\mathbf{x}$ and the label $y$ of its centre pixel (slide 21).

Run over the whole image, this gives one label per pixel, and "that starts to look like,
semantically, the boundaries of one category and another." Labelling every pixel instead of the
whole image "is called **semantic segmentation**" (slide 22, ≈18:27). The lecturer mentions a
related problem, instance segmentation, in which separate instances of a category are also told
apart.

### Translation equivariance

Because every patch is processed "independently and identically", the mapping treats every
position the same (slide 23, ≈19:13). Slide 23 calls this "Translation invariance: process each
patch in the same way" and writes it as an **equivariant** mapping:

$$f(\texttt{translate}(x)) = \texttt{translate}(f(x))$$

Shifting the input and then applying $f$ gives the same result as applying $f$ and then shifting
the output. So "it doesn't matter if the bird is in the top corner, the center, the bottom left,
those pixels are still going to be categorized as bird." Asked why that suits images, a student
said each pixel's answer depends on its local context, and the lecturer added that the world has
this symmetry: "if you took an image of a bird and then you moved that portion of the image… to a
different part of the image or… you shifted your camera a little bit, the object you're taking a
photo of is the same… A bird should look the same no matter where it is in an image." She flagged
that this glosses over pose (≈20:00–20:46).

### The weighted sum is a convolution

The simplest version of the patch classifier computes "a weighted sum of the pixels in each patch",
and learns the weights (slide 24, ≈20:46). That set of weights $\mathbf{W}$ "is a convolutional
kernel that then we can apply to the entire image."

Slide 25 defines **convolution** as a "Linear, shift-invariant transformation" of grid-structured
data, the operation signal processing uses for filtering (≈21:32). For a filter $w$ with
$(2K+1) \times (2K+1)$ weights $w[k_1, k_2]$, $k_1, k_2 \in \lbrace -K, \ldots, K \rbrace$, and a
bias $b$, the output at position $[n, m]$ is

$$x_{\text{out}}[n, m] = b + \sum_{k_1, k_2 = -K}^{K} w[k_1, k_2] \thinspace x_{\text{in}}[n + k_1, m + k_2]$$

the bias plus "a linear combination of the filter weight for that coordinate times the pixel value
for that coordinate in the image, and then summed over all the filter weights and all the pixels in
the patch" (≈22:19). A filter produces its largest outputs where the image looks like it. Slide
25's example applies an edge filter to a photo of a clown fish: the output is "highlighting areas
in the image that match the filter the most", with high values along boundaries from dark to light
and low values along boundaries from light to dark, "the place that's going to be maximally opposite
to that filter" (≈23:05). The lecturer calls them "horizontal edges", but the filter drawn on slide 25
has its dark and light lobes side by side, and in the output it is the vertical edges that stand out:
the sides of the fish's white stripes, the eye ring and the fins, while the near-horizontal outline of
the body barely registers.

A student asked what "shift-invariant" means here. The answer: shift the clown fish over and run the
filter, and "the output would be this shifted over. And so you could shift the input, then apply
the convolution. Or you could apply the convolution and then shift the output, and the results
would be the same" (≈23:05–23:55).

**Convolution or cross-correlation?** The lecturer noted that the operation defined here "is not
quite the same definition as familiarly used in signal processing for convolution. Usually, it has
like a flipped value." In neural networks it is called convolution anyway, though the
signal-processing name would be cross-correlation, partly because "in these networks, it's very
easy to learn how to flip a sign, so it doesn't matter so much" (≈24:41–25:26). See
[convolution](convolution.md).

## Convolution as a constrained linear layer

### Fully connected, locally connected, shared

Slide 26 starts from a **fully connected layer**: every output depends on every input, through a
dense weight matrix $\mathbf{W}$ and bias $\mathbf{b}$, followed by a non-linearity, $g(\mathbf{z})$
(≈23:55). "This can be quite computationally expensive when things get very large."

![Slide 26: a fully connected layer: five inputs x and a bias input 1, every one wired to every pre-activation z through W and b, each z passed through g](../raw/images/04-architectures-grids/slide-26.jpg)

*Slide 26 — a fully connected layer: every output depends on every input.*

A **locally connected** layer (slide 27) assumes instead that each "output is a **local** function
of input": the fifth output depends only on the fourth, fifth and sixth inputs, through a small
weight vector $\mathbf{w}$ and a bias $b$. "If we use the same weights (**weight sharing**) to
compute each local function, we get a convolutional neural network" (slide 28, ≈24:41).

![Slide 27: the same layout with eight inputs, where only one output is connected, to three neighbouring inputs through a small weight box w and a bias b](../raw/images/04-architectures-grids/slide-27.jpg)

*Slide 27 — a locally connected layer: each output depends on a small window of the input.*

![Slide 28: the locally connected layer with grey-shaded activations and the boxed equation z = w ⋆ x + b](../raw/images/04-architectures-grids/slide-28.jpg)

*Slide 28 — use the same weights at every window and the layer is a convolution.*

With the shared filter written as a convolution, the layer is

$$\mathbf{z} = \mathbf{w} \star \mathbf{x} + b$$

the boxed equation of slides 28 and 29. Slide 29 draws the same $\mathbf{w}$ repeated at every
position: "it's going to be the same weights that are used across the entire input vector" (≈25:26). That brings back the embarrassing
parallelism of the MLP, because the same small weight matrix is applied to many small patches at
once (≈26:13).

![Slide 29: every interior output connected to its three neighbouring inputs through a stack of overlapping boxes, all labelled w](../raw/images/04-architectures-grids/slide-29.jpg)

*Slide 29 — weight sharing: one filter reused at every position.*

### The matrix picture: a Toeplitz matrix

Slides 30 and 31 put the two layers side by side as matrix–vector products (≈26:13–27:00). A
fully connected layer, $\mathbf{x}_ {\text{out}} = \mathbf{W}\mathbf{x}_ {\text{in}} + \mathbf{b}$
(slide 30), multiplies by a dense matrix, "where every single one of those values matters."

![Slide 30: a fully connected linear layer drawn as a column vector equal to a dense 9-by-9 matrix W times a column vector, beside the same layer as a network in which every input connects to every output](../raw/images/04-architectures-grids/slide-30.jpg)

*Slide 30 — a fully connected layer: a dense matrix, every input wired to every output.*

A convolutional layer, $\mathbf{x}_ {\text{out}} = \mathbf{w} \star \mathbf{x}_ {\text{in}} + b$
(slide 31), is the same product with a very particular matrix. For a filter that looks at three
neighbouring inputs, "those three elements… above and below that diagonal, those are going to be
the only components that have values, and the values are going to be exactly the same along the
entire diagonal of that matrix." The three filter weights, $w[-1]$, $w[0]$ and $w[1]$, run down
the three central diagonals, and every entry off that band is zero (≈27:00).

![Slide 31: a convolutional layer drawn as a column vector equal to a 9-by-9 matrix that is zero except on a three-diagonal band carrying w[-1], w[0] and w[1], beside the network view in which each output connects only to its three neighbouring inputs through the same three weights](../raw/images/04-architectures-grids/slide-31.png)

*Slide 31 — the same layer as convolution: three shared weights down a diagonal band, zeros everywhere else.*

A student asked how the three weights become a matrix. The lecturer: "you're only going to have
three values. And those three values, if you transfer this into a linear layer, those are going to
be the values above, along, and below the diagonal for the whole matrix" (≈28:31–29:18).

A matrix that is constant along each diagonal is a **Toeplitz matrix** (≈27:00), for example

$$\begin{pmatrix} a & b & c & d & e \cr f & a & b & c & d \cr g & f & a & b & c \cr h & g & f & a & b \cr i & h & g & f & a \end{pmatrix}$$

Slide 32 prints that example. A convolutional layer is "a specific form of a Toeplitz matrix", and
so "a constraint on a standard linear layer" (slide 32: "Constrained linear layer"; "Fewer parameters —> easier to learn, less
overfitting"). There are fewer parameters, "and they're going to be repeated. And so that means you
can have a lot less overfitting. There's less open flexibility" (≈27:45).

![Slide 32: a 5-by-5 Toeplitz matrix of letters a to i, constant along each diagonal, beside a picture of y equal to a banded greyscale matrix times a pixel-image vector x](../raw/images/04-architectures-grids/slide-32.jpg)

*Slide 32 — a convolutional layer is a Toeplitz matrix: a constrained linear layer with fewer parameters.*

### Any input size

Because the matrix is the same small pattern repeated down a band, it can be extended to any
length: "you can take that same weight matrix and you can scale it. It's not going to change
anything about the model you've learned" (≈27:45). Slide 34: "Conv layers can be applied to
arbitrarily-sized inputs (generalizes beyond the training data due to an architectural
structure!)". An MLP trained on one image size cannot be applied to a larger image: "You just
wouldn't have weights for some of that size." The same edge filter can be run on any photo and
will always pick out the same kind of structure (≈28:31). Slide 35 shows it on a chameleon, a bear
cub and the clown fish.

![Slide 34: a column vector y equal to a large banded matrix, drawn as a greyscale image with a bright diagonal and dark background, times a column vector x labelled "e.g., pixel image"; the caption says conv layers can be applied to arbitrarily-sized inputs](../raw/images/04-architectures-grids/slide-34.jpg)

*Slide 34 — the banded matrix extends to any input length, so one learned filter serves every image size.*

### Five views on convolutional layers

Slide 36 collects the ways of thinking about a convolutional layer (≈29:18–30:04):

1. **Equivariant with translation**, $f(\texttt{translate}(x)) = \texttt{translate}(f(x))$, "a
   built-in hypothesis".
2. **Patch processing**, which "builds in explicit parallelism into our computation".
3. **An image filter**, "highlighting specific structures in the input data in some useful way
   that's learned".
4. **Parameter sharing**, "using the same parameters across the entire input data in a
   computationally efficient way".
5. **A way to process variable-sized tensors**, "because you have this built-in generalization
   across the input size".

## Stacking convolutional layers

Slide 37 asks "What happens when you stack convolutional layers?" (≈30:04–31:37). With a
convolution, a pointwise non-linearity, and a second convolution, each output depends on a window
of the input that is wider than either filter: in the slide's one-dimensional drawing, with
three-wide filters, an output $\mathbf{x}_ L[i]$ depends on five inputs. And the function that maps
that window to the output is the same at every position. So "the entire CNN is a convolutional
filter, but one that's nonlinear" (slide 37: "The whole CNN acts like a (nonlinear) convolutional
filter!").

![Slide 37: two copies of a three-layer one-dimensional network; on the left, coloured arrows trace how two outputs depend on three hidden units and five inputs each; on the right, each dependency is replaced by a grey trapezoid F](../raw/images/04-architectures-grids/slide-37.png)

*Slide 37 — two stacked convolutions with a non-linearity between them act as one non-linear filter with a wider window.*

Asked why it is non-linear, the lecturer pointed to the activation between the two layers, which the
diagram does not draw: "Maybe that's something that in future years I'll just put that in the
diagram" (≈32:23).

Depth widens that window: "as you're adding more depth, you're increasing what we call the
**receptive field** of the output", and a stack of layers whose windows grow this way is "what we
call a spatial pyramid" (≈31:37). Asked what the growing dependence is called, she named the
receptive field again and gave the two ways to enlarge it: a larger filter, "or you can just add
more depth" (≈33:10).

Three more answers from the questions here (≈33:10–35:29):

- **Images are convolved in two dimensions, not flattened.** The slides draw vectors because "they're
  really clean and easy to look at", but flattening an image and convolving it in one dimension
  would give "this weird like stripe-wise structure. So, actually, we do take those convolutions in
  two dimensions."
- **Each layer has its own filter.** "In the second layer, would you use the same kernel?" "No. And
  that's why the colors are different. I mean, you could. But usually, we don't. Those weights are
  learned, but they are shared across the entire layer."
- **So few parameters per layer is a real restriction**, and that is why convolutional networks
  use "quite a few different filters that are happening in parallel" or "a lot more layers". Also,
  the sparse matrix is a way to think about the layer, not how it is computed: "often, we just
  really do compute it patch wise." Experimentally, "for things where this is [an] appropriate
  inductive bias, you can learn more efficiently. You get a better fit with less data."

## Channels

### Multichannel inputs and outputs

"What if we have color? (aka multiple input channels?)" (slide 38, ≈35:29). A colour image is a
tensor with a depth as well as a spatial extent. For **multichannel inputs**, such as red, green
and blue, the layer learns a separate set of weights for each input channel and sums the results
into one output channel (≈35:29–36:16):

$$\mathbf{x}_ {\text{out}} = \sum_{c} \mathbf{w}[c, :] \star \mathbf{x}_ {\text{in}}[c, :] + b[c]$$

This is slide 39's formula. Its input is $\mathbf{x}_ {\text{in}} \in \mathbb{R}^{3 \times N}$, three channels of length
$N$, the output $\mathbf{x}_ {\text{out}} \in \mathbb{R}^{1 \times N}$, and the filter
$\mathbf{w} \in \mathbb{R}^{3 \times 3}$ has one row of weights per channel. The per-channel weights
"could arguably" be forced to be shared, "but often, we let them be separate." As printed, the
$+ b[c]$ follows the sum with no brackets, so the slide leaves open whether the bias is inside the sum
(one bias per input channel) or outside it, although it uses the summation index $c$.

![Slide 39: red, green and blue input columns whose three-neighbour windows are combined through one filter w into a single greyscale output column, with the multichannel-input formula](../raw/images/04-architectures-grids/slide-39.jpg)

*Slide 39 — multichannel input: a weight row per channel, summed into one output channel.*

For **multichannel outputs**, the layer learns "a filter bank or a set of different output
convolutions": several filters applied to the same input in parallel, each producing its own
output channel (slide 40, ≈36:16–37:05). With a bank of $C$ filters, output channel $c$ is
$\mathbf{w}[c, :] \star \mathbf{x}_ {\text{in}} + b[c]$, counting from $c = 0$. The slide's last
line has a slip: it prints $\mathbf{x}_ {\text{out}}[C, :]$ on the left of
$\mathbf{w}[C - 1, :] \star \mathbf{x}_ {\text{in}} + b[C - 1]$, where its own indexing, which starts
at 0, makes the last output channel $C - 1$.

![Slide 40: one greyscale input column feeding two filters, a red w[0,:] and a blue w[1,:], each writing its own output column, with the per-channel equations](../raw/images/04-architectures-grids/slide-40.jpg)

*Slide 40 — multichannel output: a filter bank, one filter per output channel (the last equation's left side has a slip, see text).*

### The general form

Most convolutional layers have many input and many output channels (slide 41, ≈37:05–37:52). A
layer with $C_{\text{in}}$ input channels and $C_{\text{out}}$ output channels has $C_{\text{out}}$
filters, each of spatial size $K \times K$ and each with $C_{\text{in}}$ channels "to match your
input data". Output channel $c_2$ sums the convolutions of every input channel $c_1$ with that
filter's slice for it:

$$\mathbf{x}_ {\text{out}}[c_2, :, :] = \sum_{c_1 = 1}^{C_{\text{in}}} \mathbf{w}[c_1, c_2, :, :] \star \mathbf{x}_ {\text{in}}[c_1, :, :] + b[c_2]$$

![Slide 41: a filter bank drawn as two groups of three stacked K-by-K grids, C_out filters of C_in channels each, convolved with a stack of C_in input grids to give a stack of C_out output grids, above the general multi-input, multi-output formula](../raw/images/04-architectures-grids/slide-41.png)

*Slide 41 — the general convolutional layer: one filter per output channel, each spanning every input channel.*

Slide 42 draws a bank of two filters on a two-channel input (≈37:52–38:39): each filter, $F^1$
and $F^2$, takes a $3 \times 3$ patch through both input channels, sums the weighted values, and
writes one value into its own output channel. "Each of these sets of weights, F1 and F2, those are
learned separately, but they're shared across the entire image." In tensor shapes,
$\mathbf{x}_ {\text{in}} \in \mathbb{R}^{C_{\text{in}} \times H \times W} \to \mathbf{x}_ {\text{out}} \in \mathbb{R}^{C_{\text{out}} \times H \times W}$.
The figure is credited "modified from Andrea Vedaldi".

![Slide 42: a two-sheet input slab with a blue and a red 3-by-3-by-2 patch, each summed by its own filter F1 or F2 into one value on its own sheet of a two-sheet output slab](../raw/images/04-architectures-grids/slide-42.jpg)

*Slide 42 — a bank of two filters: each reads a patch through all input channels and writes to its own output feature map.*

### Feature maps

"Every layer can be thought of as a set of C **feature maps** or channels", and each feature map
is itself an image, $N \times M$ (slide 43, ≈38:39–39:24). Slide 43 shows them for a heron photo
passed through AlexNet: after the first convolutional layer, the 64 feature maps mostly trace edges
of the bird and ground, and some show its silhouette; after the second, the maps are smaller,
blobbier, and "can often be quite difficult to interpret". Slide 47 draws the first two layers of
such a network with their filters: an RGB input, $C_1$ layer-1 filters of three channels each
producing $C_1$ feature maps, and $C_2$ layer-2 filters of $C_1$ channels each. "Even just one or two
layers of depth, when you have convolution, it can map really complex functions" (≈44:05–45:37).
Both slides are excluded from OCW's licence. See [representation learning](representation-learning.md).

A student asked what makes the feature maps differ from one another. Nothing forces it, said the
lecturer; regularization such as dropout helps, but mostly "the model will learn each of them
uniquely because that gives it more capacity." Sometimes it does not: "you can have feature collapse
sometimes where some stuff just gets ignored that you might not want to be ignored" (≈39:24–40:09).

### Counting parameters

Slides 44 and 45 are a quiz (≈40:09–43:19). The input is an RGB image,
$\mathbf{x}_ {\text{in}} \in \mathbb{R}^{3 \times 128 \times 128}$; a "Filter Bank with 3x3 filters"
produces $\mathbf{x}_ {\text{out}} \in \mathbb{R}^{96 \times 128 \times 128}$.

![Slide 44: a thin 3-by-128-by-128 input slab, an arrow through a "Filter Bank with 3x3 filters", and a thick 96-by-128-by-128 output slab, above the question of how many parameters each filter has](../raw/images/04-architectures-grids/slide-44.png)

*Slide 44 — the parameter-counting quiz: 3 input channels, 3-by-3 filters, 96 output channels.*

"How many parameters does each *filter* have?" The first answer from the room was 9, "3 by 3". The
right answer is **27**: $3 \times 3$ spatially, times 3 input channels. The lecturer explained the
trap: "When we talk about filter size, we often only talk about the spatial extent of the filters",
and the channel depth is assumed, because standard layers do not convolve across channels. One
could ("maybe you have something like a hyperspectral input. You actually want convolution across
the spectral bands"), but by default each input channel gets its own learned weights.

"How many filters are in the bank?" **96**, one per output channel, each with 27 parameters.

Slide 46 states the rule. Mapping $\mathbf{x}_ l \in \mathbb{R}^{C_l \times N \times M}$ to
$\mathbf{x}_ {(l+1)} \in \mathbb{R}^{C_{(l+1)} \times N \times M}$ with filters of spatial extent
$K_1 \times K_2$ takes

- $K_1 \times K_2 \times C_l$ parameters per filter, and
- $C_{(l+1)}$ filters (≈43:19).

Asked about rectangular filters: "normally, the filters are square. That is totally standard. They
don't have to be" (≈44:05).

## Pooling

A convolutional layer — weights, bias, pointwise non-linearity — can be followed by **pooling**
over its outputs, partly because "the dimensionality can get really large, really fast"
(≈45:37–46:23). **Max pooling** takes the largest value in a local window
$\mathcal{N}(j)$:

$$y_j = \max_{j \in \mathcal{N}(j)} h_j$$

**Mean pooling** takes the average over the window:

$$y_j = \frac{1}{\lvert \mathcal{N} \rvert} \sum_{j \in \mathcal{N}(j)} h_j$$

Slide 48 gives max pooling, drawn as a "max" box after a convolutional layer, and slide 49 adds mean
pooling, with a $\Sigma$ in the box. The slides use $j$ both for the output position and for the positions inside the window; the
intended reading is that $y_j$ pools the values $h$ over the neighbourhood $\mathcal{N}(j)$ of
position $j$.

![Slide 48: a convolutional layer x to z to h followed by a max box that pools three neighbouring h values into one y, with the max-pooling formula](../raw/images/04-architectures-grids/slide-48.jpg)

*Slide 48 — max pooling after a convolutional layer.*

![Slide 49: the same diagram with a sigma in the pooling box, and the mean-pooling formula beside max pooling](../raw/images/04-architectures-grids/slide-49.jpg)

*Slide 49 — mean pooling, the average over the window.*

**Why pool over space?** "Pooling across spatial locations achieves stability w.r.t. small
translations" (slides 50 to 52, ≈46:23). If an edge filter fires on an edge, the maximum over a
window still reports the edge when the edge moves a little within the window: "large response
regardless of exact position of edge." In the three build steps, the edge moves right across a fixed
$3 \times 3$ window and the max over the window is drawn as filled only in the last. These three
slides reuse slide 25's clown fish photograph, which OCW excludes from its licence there, so they have
no image here.

**Why pool over channels?** "Pooling across feature channels (filter outputs) can achieve other
kinds of invariances" (slide 53, ≈47:10). Take the maximum over the outputs of several edge filters
of different orientations and the result is "a filter that was finding edges and was invariant to
the orientation of the edge": "large response for any edge, regardless of its orientation".

Asked at the end how pooling relates to convolution, the lecturer described pooling as "the same as
running a convolutional layer. But now you're not learning weights. You're just taking, for
example, like the maximum function or that average function… it's a constrained version, a
nonlearned version of a convolutional layer" (≈1:16:51).

## Downsampling, strides and dilation

### The classification network

A typical convolutional network stacks a building block — filter, ReLU, pool — over and over, and
ends in a classifier, $f(\mathbf{x}) = f_L(\ldots f_2(f_1(\mathbf{x})))$ (slide 54, "heron",
≈47:10–47:59). For classification, where all that comes out is something like "a word or a
specific one hot encoding", the full spatial resolution is not needed at every layer, so the pool
step becomes **downsampling** and each block is smaller than the last (slide 55). Slide 56 pools
and downsamples; slide 57 simply keeps every second value, halving the length (≈47:59–48:47).

![Slide 56: a convolutional layer whose three neighbouring outputs connect directly to one value of the next column, headed Filter and Pool and downsample](../raw/images/04-architectures-grids/slide-56.jpg)

*Slide 56 — pool and downsample.*

![Slide 57: the same convolutional layer, with every second output kept and passed to a column half as long](../raw/images/04-architectures-grids/slide-57.jpg)

*Slide 57 — downsampling by keeping every second value.*

### Strided operations

"One way that you can actually do both pooling and downsampling together is through something
called a strided operation" (slide 58, ≈48:47). Instead of sliding the filter one position at a time,
it moves by the **stride**: "you might move two or you might move five." Slide 58: "**Strided
operations** combine a given operation (convolution or pooling) and downsampling into a single
operation." Slide 59 draws a strided $3 \times 3$ filter in two dimensions. In practice the windows
usually still overlap, "so you don't want to lose any of the context from the original input… maybe
your input kernel is like 7 by 7. Your stride might be 5, something like that" (≈49:34).

![Slide 58: a convolutional layer that computes only three outputs, at every second input position, with a bracket labelled Stride 2](../raw/images/04-architectures-grids/slide-58.jpg)

*Slide 58 — a strided convolution computes the filter only at every second position, folding downsampling into the layer.*

![Slide 59: a 3-by-3 filter w convolved with an 11-by-11 input grid, with windows drawn at positions a stride apart, giving a small 3-by-3 output](../raw/images/04-architectures-grids/slide-59.png)

*Slide 59 — a strided convolution in two dimensions.*

### Dilated filters

A **dilated filter** "is a filter where we've set some of the values to 0 and baked that into our
model" (slide 60, ≈49:34–50:20). It spreads its weights out with gaps between them, so it "Covers a
large receptive field with fewer parameters": "There's less flexibility… but you're kind of getting
that larger context window. And that's kind of a trade-off that you can optimize for."

![Slide 60: a 5-by-5 dilated filter whose nine weights sit on alternate cells with gaps between them, convolved with a 10-by-10 input](../raw/images/04-architectures-grids/slide-60.png)

*Slide 60 — a dilated filter covers a large receptive field with few parameters.*

### Choosing among them

A student asked what the norms are for when to use a stride (≈51:05–52:37). The space of
architectures "is vast", and there is a research field, **neural architecture search**, that tries
to optimize over it, at great computational cost. But some trends have held up experimentally: "in
the original AlexNet paper, the filters were quite big… and then there were less depth in the
layers. One of the things that we've seen more and more is now we actually often see filters that
are reasonably small, 7 or 5 spatial extent. But maybe then they're actually stacked quite deep. So
then you end up with really large receptive fields." Striding is "that trade off between the amount
of computation and the receptive field." Her advice was to compare the architecture diagrams of
standard models.

Does the complexity of the images matter? "Experimentally, yes." Simple images such as CIFAR's need
a less complex network. When fine detail matters, as with species images in the lecturer's own
work, "one of the things that actually tends to matter more than almost anything else is the input
image size that you use", and it can be a better trade-off to keep a larger input and spend less
computation elsewhere (≈53:22–54:08).

Asked whether a downsampled representation can be brought back to the input size, she said it can,
"but you might not do it very well", and returned to it with the encoder–decoder below (≈50:20–51:05).

### Learned filters versus hand-crafted ones

A student asked why the filters are learned when signal processing designs them by hand for a
purpose (≈54:08–56:30). The lecturer called this "this massive paradigm shift that happened maybe
10, 15 years ago", when computer vision moved from **feature engineering**, hand-crafted filters
combined into larger systems, to filters learned end to end. When learned filters proved more
predictive, "it was a bit of like a catastrophe emotionally for many researchers." The trade-off,
she said, "often comes down to how much data you have. So if you don't have enough data to learn
the filters, well, then more handcrafting, more knowledge, more inductive bias tends to be better…
But if you're in a space where you can get huge amounts of data to learn from, often, we don't know
as much as we think we know about what optimality might be." And hand-design does not scale: "how
would you handcraft like the seventh convolutional layer in a 128 layer network for a single
channel?" See [inductive bias](inductive-bias.md) and
[differentiable programming](differentiable-programming.md).

## Receptive fields

"Receptive field essentially just says what parts of the initial input have any influence on this
part of the output" (slides 61 and 62, ≈56:30–58:03). Slide 62 draws a one-dimensional
conv–ReLU–conv network over a stork crop: the output $x_2[3]$, labelled "Bird", depends on five
input positions, and an output of the first layer, $x_1[5]$, on three. "Outside of that receptive
field, this model doesn't have any input information. So if there was something just outside of
that field of view that was really informative… there's no way for the model to get information"
about it.

Slide 63 shows what this does to the representation, with feature maps from three networks on the
heron photo: AlexNet's conv1 to conv5, VGG16's conv1 to conv13, and ResNet18's conv1 to block8
(≈58:03–59:38). Early layers are "very fine grained or very detailed"; later ones "much more
diffuse", because each output "is actually collecting really complex information" from a large
part of the input, so they may be "more semantically meaningful but less affected by subtle texture
or variation". The maps are resized to the same size for display; in a ResNet the spatial extent by
block 8 "is much lower. And then there's a lot more channels." Asked whether this resembles a
Gaussian blur, she said the intuition "is a little similar, but computationally, it's quite
different." Slides 61 to 63 are excluded from OCW's licence.

## Implementing convolution

Slide 64, which the recording does not discuss, gives the "Basic implementation" of a convolutional
layer as three steps: `im2col`, which rearranges an $N \times M \times C$ input into one row per
patch, $N_{\text{patches}} \times (K \times K \times C)$; `bmm`, a batched matrix multiplication with
the kernel; and `col2im`, which rearranges the result back to $N \times M \times C$. It adds "or: fft
signal processing stuff…" and recommends the library timm,
<https://github.com/rwightman/pytorch-image-models>. See [tensors and batching](tensors-and-batching.md).

## An architecture zoo

The last part of the convolutional material is "a pretty high level" tour of architectures and
components "that have tended to continue to be used over time" (≈59:38). Every figure in it is
built on the heron or stork photographs and is excluded from OCW's licence (slides 66 to 72), so
this section has no images.

**Classification networks** (slide 66): filter, pointwise non-linearity and downsampling, repeated,
then a classifier (≈1:00:25).

**Encoder–decoder** (slide 67, ≈1:00:25–1:02:00). The encoder repeats convolution, non-linearity
and subsampling "to get down to some very low dimensional representation", a vector $\mathbf{z}$.
The decoder repeats convolution, non-linearity and **upsampling**, "increasing the spatial extent of
each of the layers going forward", and "the training signal here is you want the decoded output to
match the encoded input as close as possible." This is one way to get a low-dimensional
representation of an image, and the structure is built into variational autoencoders; masked
autoencoders are similar but "use attention as opposed to convolution as a way to share information
over space." The parts are useful on their own: train the pair, then use the encoder to represent
images or the decoder to generate them, the [differentiable programming](differentiable-programming.md)
idea of separable components (≈1:02:00). See [representation learning](representation-learning.md).

An encoder–decoder can also produce a spatial output such as a segmentation map. But the
**bottleneck** is "both a blessing and a curse": it forces the network to learn something
semantically useful, but "you've lost a lot of the information about the fine-grained details in
that spatial structure", so the output cannot be precise at the input's resolution (≈1:02:46).

**Image-to-image without a bottleneck** (slide 69): convolution, ReLU, convolution, softmax, all at
full resolution. "You don't lose any spatial structure", but "this can be really computationally
expensive" (≈1:03:33). (Slide 72 repeats slide 69 unchanged.)

**U-net** (slides 68 and 70, ≈1:03:33–1:05:09). U-net gets "the best of both worlds" with **skip
connections** from each encoder stage to the decoder stage of the same size, which pass the
features across by the identity. The network "is not forced to explicitly try to shove a bunch of
fine-grained spatial detail into some very low rank, non-spatially dimensional embedding, which is
somewhat impossible. It's able to learn the semantic details through this information bottleneck,
but then it also gets a lot of the spatial detail through those skipped connections." U-net "is a
really old architecture, and it is still one of the most widely used architectures for semantic
segmentation", for example in remote sensing and land-cover mapping.

**ResNet** (slide 71, ≈1:05:09–1:08:14). The skip connection is also "one of the fundamental
underpinnings of what we call the ResNet architecture", invented by Kaiming He, "who is also a
professor now here at MIT"; it is "like 10 years old at this point" and still used all the time.
Every layer has a skip connection, a **residual connection**:

$$\mathbf{x}_ {\text{out}} = F(\mathbf{x}_ {\text{in}}) + \mathbf{x}_ {\text{in}}$$

or, "if you want to change dimensionality", for example to downsample inside the network,

$$\mathbf{x}_ {\text{out}} = F(\mathbf{x}_ {\text{in}}) + \mathbf{W}\mathbf{x}_ {\text{in}}$$

where $F$ is the layer and $\mathbf{W}$ a learned linear map on the skip path. The lecturer's
intuition is that the identity path lets the network choose, layer by layer, whether to transform
its input or pass it through, so "the model itself can implicitly learn the optimal depth for the
task", without "explicitly training a bunch of different architectures side by side."

Asked whether the skip becomes an extra channel or a choice between paths: "it's kind of choosing
one or the other". The layer's output and the residual are the same size and are added, though some
implementations learn a weight on the skip path, and "even if you just add them together, there's
some version of the next layer that could basically learn how to ignore certain dimensions"; the
$\mathbf{W}$ "could be learned to be all zeros." The same kind of connection "gets surfaced via
self-attention in transformers" (≈1:07:27–1:08:14). See [skip connections](skip-connections.md).

## Beyond images: time and video

"So far, we've talked about two-dimensional or three-dimensional grids", but real data has other
dimensions, and "one really simple one is time" (≈1:08:14–1:09:04). A **convolution in time**
slides a filter along a one-dimensional signal, such as the readings of a temperature sensor
(slide 73).

![Slide 73: two rows of seventeen grey-shaded circles along a time arrow, with a blue filter box w joining three neighbouring input circles to one output circle above them](../raw/images/04-architectures-grids/slide-73.png)

*Slide 73 — a filter sliding along a one-dimensional time series.*

A video is a stack of RGB frames over time, "a four dimensional input": channels, two spatial axes
and time (slide 74, ≈1:09:04–1:09:53). Slide 74 stacks the frames of a street scene into a cube
with axes $n$, $m$ and $t$.

![Slide 74: eight consecutive frames of people walking past a stone building, and below them the same video stacked into a cube with spatial axes n and m and time axis t, whose top and side faces show the motion as streaks](../raw/images/04-architectures-grids/slide-74.jpg)

*Slide 74 — a video as a space–time volume, the grid a 3D convolution runs over.*

A **3D convolution** slides a small cube-shaped filter through that volume, convolving "over space
and time" (slide 75, ≈1:09:53). 3D convolutions are often used for action recognition in video.

![Slide 75: the faded video cube with a small wireframe cube marking a 3D filter window at one corner, and an arrow to an output volume with the matching output cell](../raw/images/04-architectures-grids/slide-75.jpg)

*Slide 75 — a 3D convolution slides a cube-shaped filter through space and time.*

Asked at the end whether one could stack a video's frames as extra input channels instead of using
a 3D convolution, the lecturer answered that "if you add channels for all of the different frames in
the video, essentially, that is 3D convolution", but that it is expensive, and "very rarely are those
convolutions in the temporal space worth it computationally" (≈1:20:43–1:23:48). State-of-the-art
video models often downsample in time, because most actions can be recognized from "a bag of images";
she cited a paper of Alyosha Efros's measuring an action's complexity by how many frames it takes to
recognize it, and "most actions are… a single frame." Many models instead run a "fast and slow"
design: a low-resolution stream running often, and a high-resolution stream run only periodically,
"one model is maybe modeling something like the motion and the other one is modeling some
fine-grained semantic details." Her summary: "video is fascinating and quite complicated and
expensive… Yeah, it's a mess."

## When you don't want shift invariance: positional encoding

Equivariance in time is not always wanted: "You don't want like picking a cup up to be the same
action as putting a cup down… you want the fact that one happened before the other to matter"
(≈1:09:53–1:10:38). Slide 76 gives two ways out (≈1:10:38–1:11:26):

1. "Use an architecture that is not shift invariant (e.g., MLP)." There "the position in time would
   actually matter", but MLPs are hard to train on such complex functions.
2. "Add location information to the *input* to the convolutional filters — this is called
   **positional encoding**." This is the option that has "really shown to be a winner".

A positional encoding is an extra input built alongside the signal: "This is not something that's
learned. It's actually constructed and added to the input. And you learn an additional weight that
incorporates that positional encoding as part of your convolution" (slide 77, ≈1:11:26). In slide
77's diagram the filter reads three neighbouring values of the signal together with a "pos" input,
a column whose shade runs from pink to dark purple with position.

![Slide 77: a "pos" column shaded pink to dark purple beside a grey-shaded "signal" column; a filter box w takes three signal values and the position encoding and writes one output value](../raw/images/04-architectures-grids/slide-77.png)

*Slide 77 — positional encoding: the filter sees where it is as well as what is there, so the layer is no longer shift invariant.*

Asked how to feed side information such as the GPS location where a photo was taken, the lecturer
listed common options: process it with its own layer or encoding, then concatenate it to the input
image or introduce it later in the network (≈1:19:12–1:20:43). See
[neural fields and positional encoding](neural-fields-and-positional-encoding.md).

## Neural fields

A field, from physics, is "a varying physical quantity of both spatial and temporal coordinates",
and a **neural field** "is a field that is parameterized maybe fully or in part by a neural
network" (≈1:11:26–1:12:12). Slide 78 draws one: the input is a pair of coordinates $(x, y)$, shown
as two gradient images, and a function $\Phi : \mathbb{R}^2 \to \mathbb{R}$ maps them to a
greyscale photograph. The slide writes the photo's value at a point as an italic lowercase $l$,
$l = \Phi(x, y)$; slide 9's heading writes the same mapping as "a function x,y —> l".

**SIREN** (slide 79, ≈1:12:12–1:13:46) is such a field, the model from the start of the lecture.
The slide calls it a "CNN applied *per-pixel* to map from a coordinate grid to a color": the
network learns one image by learning "a functional mapping from a position in the image to a color",
using its sinusoidal activations. Its input is just the position, "so now you run these inputs
through a model that's just taking x and y. And then you're outputting the color of that value."
Because the input is a coordinate rather than a grid cell, it "Can take continuous coordinates as
input!", which means "we can then do something like think about learning these functional
representations of input space as opposed to only gridwise representations."

**NeRF** (slide 80, ≈1:13:46–1:14:32) builds on the same idea. It is a network applied "to a map
from a five-dimensional coordinate grid, which is a position and direction of the camera relative
to the scene" — on the slide, $(x, y, z, \theta, \phi)$ — "and then it gives you an output of color
and volumetric density", $(RGB\sigma)$. That lets it render "what that same object would look like
from different positions than what was in the original training data", so "you can take some images
of a scene and then generate these essentially walkthroughs of scenes." The lecturer called the
right positional encodings one of NeRF's "fundamental underpinnings". Slide 81, an image "made by
Yen-Chen Lin", follows with no explanation on the slide; in the PDF it shows only a blurry,
unrecognisable scene. Slides 78 to 81 are excluded from OCW's licence. NeRFs "are really fun to
play with"; the lecturer mentioned a NeRF built at Woods Hole "for under the sea that handles the
way that light bends under water" (≈1:14:32).

From the questions at the end (≈1:16:05–1:19:57):

- **Training a NeRF.** "You take a bunch of different images of a scene from different positions",
  often from a video of someone walking around an object. For each image you estimate the camera's
  position and direction relative to the scene; for any point, that is the input, "and the output is
  the color of the image at that pixel."
- **What a NeRF is not.** "The output of NeRF is not a 3D model of the scene. It's what the image
  would look like taken from a different direction, which is not the same as a 3D model of the
  scene." Work bringing the two together, which she attributed to Ben Mildenhall "I think", had only
  just started.
- **Limits.** NeRFs "are also really not super robust. You need a lot of input data", it is hard to
  fit one to a few sparse images, and a scene that moves, such as a waving tree, breaks the
  optimization.
- **Upscaling.** Adaptations of NeRF have been used to raise spatial resolution, for instance of a
  point cloud.

## Concluding remarks

Slide 82 (≈1:14:32–1:15:19):

> Convolution is a fundamental operation for image processing.
>
> It just means: chop up the image into patches and apply the same function to each patch.
>
> This concept appears in almost all modern architectures, such as CNNs, transformers, NeRFs, and
> more.

The lecturer widened the first line to "processing anything that has some structure", such as time,
or hyperspectral satellite imagery of the whole Earth taken weekly, and added that in transformers
"the patch-wise operation isn't necessarily convolution, use attention instead." Slide 83 repeats
the agenda: why build better architectures, convolutional layers, pyramids, the architecture zoo,
and neural fields and positional encodings.

## See also

- [Convolution](convolution.md) — the convolutional layer in full: definition, equivariance, the
  Toeplitz view, channels and parameter counts, pooling, strides, dilation, receptive fields.
- [Inductive bias](inductive-bias.md) — the hypothesis-space picture, architectures as priors, and
  data versus hand-built structure.
- [Skip connections](skip-connections.md) — the encoder–decoder bottleneck, U-net and ResNet.
- [Neural fields and positional encoding](neural-fields-and-positional-encoding.md) — SIREN, NeRF,
  and breaking shift invariance on purpose.
- [Multilayer perceptrons](multilayer-perceptron.md) — the architecture this lecture starts from.
- [Lecture 3 — Approximation Theory](03-approximation-theory.md), which ends by previewing the
  inductive biases this lecture builds.
