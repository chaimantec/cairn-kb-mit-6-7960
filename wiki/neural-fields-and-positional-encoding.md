# Neural fields and positional encoding

A **positional encoding** gives a network an explicit input saying *where* each value sits, so that
the network can treat positions differently. A **neural field** goes further and takes the position
as its whole input: a network that maps coordinates to values, such as pixel position to colour. The
course gives both "a first touch" in [lecture 4](04-architectures-grids.md) and says positional
encodings return in the transformers lecture (≈1:32, ≈1:15:19), which is lecture 8 in the recorded
schedule (see the [course map](course-map.md)). Covered so far: lecture 4, slides 9–10 and 76–81,
≈9:59–12:17 and ≈1:09:53–1:14:32, and the end-of-lecture questions (≈1:16:05–1:20:43); [lecture 5](05-architectures-graphs.md),
slides 43–44 and 46, ≈1:09:47–1:10:32 and ≈1:19:02–1:20:36, on positional encodings for graphs; and
[lecture 6](06-generalization-theory.md), slide 63, ≈1:17:40–1:18:26, on NeRF's built-in physics as a
reason it generalizes; and [lecture 8](08-architectures-transformers.md), slides 41–47 and 53,
≈1:00:29–1:05:51 and ≈1:12:15, on positional encodings for transformers.

## Why break shift invariance

A [convolutional layer](convolution.md) is equivariant to translation: it applies the same function
at every position, which is the right bias for "a bird should look the same no matter where it is in
an image." It is the wrong one when position carries meaning. Applied over time to video, it would
make "picking a cup up… the same action as putting a cup down… you want the fact that one happened
before the other to matter" (lecture 4, ≈1:10:38). Slide 76, "What if you *don't* want to be shift
invariant?", gives two options:

1. "Use an architecture that is not shift invariant (e.g., MLP)." There "the position in time would
   actually matter", but MLPs are hard to train on complex functions.
2. "Add location information to the *input* to the convolutional filters — this is called
   **positional encoding**." The lecture calls this the option "really shown to be a winner".

## Positional encoding

The encoding is built, not learned: "you have your input signal. And then you also have this separate
positional encoding, which is something you construct. This is not something that's learned. It's
actually constructed and added to the input. And you learn an additional weight that incorporates
that positional encoding as part of your convolution" (slide 77, ≈1:11:26). The filter then sees where
it is as well as what is there, so the same pattern at two positions can produce different outputs.
Slide 77 draws it in one dimension: the filter reads three neighbouring values of the signal together
with a "pos" input, a column whose shade runs from pink to dark purple with position.

A related question at the end of the lecture: how to feed side information, such as the GPS location
where a photo was taken. The lecturer's options were to give it its own layer or encoding, and then
concatenate it to the input image or introduce it later in the network (≈1:19:57).

### On graphs

Lecture 5 carries the idea to graph neural networks, which are permutation invariant rather than
shift invariant. Giving each node an input that says "where" it is in the graph breaks that symmetry,
and with it the equivalence classes of graphs a graph net cannot tell apart (slide 43,
≈1:19:02–1:19:50). The crudest version, raised by a student, is a one-hot encoding of each node's
index; the lecturer's caution was that "by breaking that symmetry, you break the permutation
invariance. So it's a trade-off" (≈1:09:47–1:10:32). The version on slide 43 is the eigenvectors of
the graph Laplacian $\mathbf{L} = \mathbf{D} - \mathbf{A}$, with $\mathbf{D}$ the diagonal matrix of
node degrees and $\mathbf{A}$ the adjacency matrix: "a generalized coordinate system for your
location of a node within a graph," as sinusoids are for an image (≈1:19:50). They add "global
structural information", with the challenge that eigenvectors are ambiguous up to sign flips and
repeated eigenvalues. "Now you might not generalize to new permutations, but you will be able to
discriminate things you couldn't discriminate before … that's just like positional encoding in
CNNs" (≈1:19:50–1:20:36). See [graph neural networks](graph-neural-networks.md).

## Neural fields

"A field from physics, it can be a varying physical quantity of both spatial and temporal
coordinates. And a neural field is a field that is parameterized maybe fully or in part by a neural
network" (≈1:11:26–1:12:12). Slide 78 draws one: a function $\Phi : \mathbb{R}^2 \to \mathbb{R}$ from
the coordinates $(x, y)$ of a pixel to the greyscale photograph's value there, written
$l = \Phi(x, y)$ with an italic lowercase $l$ (slide 9's heading: "a function x,y —> l").

### SIREN

SIREN, by Sitzmann, Martel et al. (2020), is a neural field with **sinusoidal activations** (slides
9, 10 and 79). The lecture introduces it twice.

- **As an inductive bias** (slides 9–10, ≈9:59–12:17). Fitting one image by sampling pixels as
  training data, SIREN gets "a pretty good approximation… much, much faster" than ReLU or tanh
  networks, because "a Fourier basis is a good basis for representing images, and the fundamental
  building blocks of Fourier basis are sinusoids." On a one-dimensional function, a 5-layer sine
  network extrapolates periodically where a ReLU network goes flat. The slide notes that the gain
  "may be due to improved approximation ability but it might also be due to improved optimization
  ability; these two effects are typically coupled in experiments." See
  [activation functions](activation-functions.md) and [inductive bias](inductive-bias.md).
- **As a neural field** (slide 79, ≈1:12:12–1:13:46). The slide calls it a "CNN applied *per-pixel*
  to map from a coordinate grid to a color": the network learns a single image as "a functional
  mapping from a position in the image to a color". Its input is just $x$ and $y$. Because that
  input is a coordinate rather than a grid cell, it "Can take continuous coordinates as input!", so
  the image is represented as a function rather than "only gridwise".

### NeRF

NeRF, by Mildenhall, Srinivasan, Tancik et al. (ECCV 2020), applies the idea to 3D scenes (slide 80,
≈1:13:46–1:14:32). It maps "a five-dimensional coordinate grid, which is a position and direction of
the camera relative to the scene" — $(x, y, z, \theta, \phi)$ on the slide — to "color and volumetric
density", $(RGB\sigma)$. Trained on images of a scene, it can render "what that same object would
look like from different positions than what was in the original training data", producing
"walkthroughs of scenes". The lecturer named appropriate positional encodings as one of its
"fundamental underpinnings".

From the questions (≈1:16:05–1:19:57):

- **Training data.** "You take a bunch of different images of a scene from different positions",
  often a video of someone walking around an object, estimate each camera's position and direction,
  and train the network to output "the color of the image at that pixel."
- **Not a 3D model.** "The output of NeRF is not a 3D model of the scene. It's what the image would
  look like taken from a different direction."
- **Limits.** NeRFs "are also really not super robust. You need a lot of input data", struggle with
  a few sparse images, and break when the scene moves, "like a waving tree".
- **Upscaling.** Adaptations of NeRF have been used to raise spatial resolution, such as that of a
  point cloud.

Slides 9 and 78–81 are excluded from OCW's licence, so this knowledge base has no images of them;
the [slide file](../raw/slides/04-architectures-grids.md) describes them.

## NeRF as built-in structure (lecture 6)

Lecture 6 cites NeRF as an example of domain-specific constraints, one of its candidate explanations of why
deep nets generalize (slide 63, whose figure is excluded from OCW's licence). Recalling Sara Beery's
example from lecture 4, the lecturer says NeRF "is a neural architecture, but it uses equations of
perspective projection, and it uses equations of light transport. And these are built into the process,
and so it will generalize to new viewpoints because it's using structures. It's not just fitting to data.
It's data plus structure, data plus constraints" (≈1:17:40–1:18:26). See [inductive
bias](inductive-bias.md).

## Positional encodings in transformers (lecture 8)

Lecture 8 brings positional encoding back as the third of the transformer's ideas. A transformer is
permutation equivariant, "a set-to-set mapping" in which "the ordering of the tokens doesn't matter"
(slide 41, ≈1:01:16). When order matters, the remedy is lecture 4's. Slide 42 repeats its picture of a
filter that sees a position input beside the signal, and slide 43 does the same for tokens: "Add location
information to the token code vectors" (≈1:02:02–1:02:48). The "vanilla" choice is a **Fourier positional
code** (slide 44): rather than the Cartesian coordinates of a patch, the values of $\sin(x)$, $\sin(x/B)$,
up to $\sin(x/B^4)$, and the same in $y$, at that location. Why a Fourier basis helps is left to the book
chapter and called "an open science question" (≈1:03:33–1:04:18).

The positional encoding is where domain knowledge enters a transformer: "that's the part where you tell
the system what it means to be local for your domain" (≈1:04:18). Lecture 8's examples are ScaleMAE's
ground-sample-distance encoding for satellite images at different scales (slide 45); spherical harmonics,
"sinusoids on the sphere", for positions on the globe (slide 46); and, for graphs, the eigenvectors of the
graph Laplacian, as in lecture 5 (slide 47, ≈1:04:18–1:05:51). "Attention Is All You Need" used a
sinusoidal positional encoding (slide 53, ≈1:12:15). See [transformers](transformers.md).
