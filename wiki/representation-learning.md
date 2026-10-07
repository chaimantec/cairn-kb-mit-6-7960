# Representation learning

What deep networks learn internally, and why those internal representations can be reused.
Lecture 1 previews it twice: as "how deep networks represent data" (slides 71–73, ≈55:02–57:19),
assigned to **lectures 11–13**, and as "reusing weights" (slides 76–77, ≈57:19–58:53), assigned
to **lectures 18–19 on transfer learning**. Covered so far: [lecture 1](01-introduction.md); [lecture 2](02-how-to-train-a-neural-net.md),
≈1:04:32–1:10:02 (what an embedding is, and visualizing what a unit responds to);
[lecture 4](04-architectures-grids.md), slides 43, 47, 63 and 67 (feature maps, how they change with
depth, and the encoder–decoder).

## Compact, compositional representations

The lecture's thesis is that "deep networks are a more compact way of representing knowledge in
data" (≈55:02). Earlier methods were "more like lookup tables". A deep network instead assumes
there are "lower-level building blocks that are useful for many different possible downstream
tasks".

The example is a classifier for the letter **T** (≈55:47). You could train a T-detector from
scratch. Or you could take a classifier for *lines*, apply it once as it is and once rotated, and
learn only how the two lines meet. "So that line classifier is reusable in some combination to get
a T classifier. And that's the basic idea here."

## The hierarchy, observed

Analyses of trained convolutional networks find this structure after the fact (≈55:47–56:34).
Earlier layers learn low-level features, which later layers combine into more and more complex
ones. Slide 72 sets two figures side by side:

- **The visual cortex** (Serre, 2014): a hierarchy from oriented edges (V1/V2), through corners and
  curves (V2/V4) and more complex shapes (V4/PIT, PIT/AIT), up to classification units for whole
  objects — a deer, a bird, a fox, a snake.
- **A deep network** (Donahue, 2013): image features from an early layer and from a late layer,
  each clustered and plotted in two dimensions and coloured by category. Early-layer features are
  mixed, with no category structure. Late-layer features fall into separated clusters by class.

The lecturer's reading: early layers are categorizing "things like, does it have directional
orientations of light and dark gradients in the pixels?", so they "are not going to cluster well
by category". Towards the end of the network "it's learning more concepts, and that's where you
start getting these clusters" (≈56:34).

How these representations are learned and how they are structured is the subject of three
lectures: **reconstruction-based** (11), **similarity-based** (12) and **theory** (13), per the
banner on slide 73 (≈57:19).

## Reuse and transfer

If the lower layers learn general-purpose features, they can be kept when the task changes
(slide 77, ≈57:19–58:53). The example: a network learned to categorize photos of animals, and now
you want to work with satellite imagery. "A lot of those initial components about lines and
orientations and structures, those are useful for both of these." Slide 77 draws the same
hierarchy twice. On the left it ends in animal classes. On the right (Gandour, 2018) the lower
layers are unchanged, the input is aerial photos of buildings, and the classification units become
a check and a cross — building intact or damaged.

The open question is "what do we really need to learn from scratch versus what is generally
useful". The payoff is practical: reuse is "really valuable if you don't have big data or big
compute", because "you can pre-generate representations that then you can learn on top of without
needing to learn everything from scratch" (≈58:06). Whether representations really are
"generalizable or transferable" is the subject of the transfer-learning lectures, **18 (models)**
and **19 (data)**.

The same theme appears in lecture 1's account of deep learning today: an open-source culture of
**modular reuse**, where "people take weights that were trained by one person with one architecture"
and use them "as a module within another, larger system" (≈22:33–23:19).

## Embeddings, and looking inside

Lecture 2 pins down the word **embedding**, in answer to a student. A network, sometimes called an
**encoder**, maps "a high-dimensional input or a complex input, something like an image, to a
low-dimensional representation or a low-dimensional embedding of that input data" — for example "a
vector of length 2048 … or 1024. Those are both common embedding sizes". "Representation" is used
for the same thing (≈1:09:15–1:10:02). CLIP is the lecture's example of two encoders, one for text
and one for images, trained so that "things that are similar, semantically, from text to images,
are quite close together in the learned embedding space" (slide 69, ≈1:06:05).

The same lecture shows one way to see what a unit has learned: optimize the input image to maximize
it. For the "cat" output this gives "what a given trained model thinks is most cat like"; for a
hidden neuron it is "a mechanism to probe what the model is paying attention to" (slides 66–67,
≈1:04:32–1:05:19). See [differentiable programming](differentiable-programming.md).

## Feature maps in a convolutional network (lecture 4)

Lecture 4 shows the hierarchy inside a [convolutional network](convolution.md). Each layer "can be
thought of as a set of C **feature maps** aka **channels**", each an $N \times M$ image (slide 43,
≈38:39–39:24). On a heron photograph passed through AlexNet, the 64 maps after the first layer mostly
trace edges, and a few show the bird's silhouette; after the second, they are smaller and blobbier and
"can often be quite difficult to interpret". Slide 63 follows the same photo deeper through AlexNet,
VGG16 and ResNet18: early maps are "very fine grained or very detailed", later ones "much more diffuse",
because each later unit's **receptive field** covers more of the image. They become "more semantically
meaningful but less affected by subtle texture or variation in the initial input image" (≈58:03–58:51).
Nothing forces the maps of a layer to differ, but they usually do, "because that gives it more
capacity"; when some are ignored, it is "feature collapse" (≈39:24–40:09).

The same lecture gives the **encoder–decoder** (slide 67, ≈1:00:25–1:02:00): an encoder compresses an
image to a low-dimensional vector $\mathbf{z}$, a decoder expands it back, and training makes the
output match the input. It is "one way to get a low-dimensional representation of an image", the
structure of variational autoencoders, and close to masked autoencoders, which use attention instead
of convolution. See [skip connections](skip-connections.md) for what its bottleneck costs.
