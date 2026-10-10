# Representation learning

What deep networks learn internally, and why those internal representations can be reused.
Lecture 1 previews it twice: as "how deep networks represent data" (slides 71–73, ≈55:02–57:19),
assigned to **lectures 11–13**, and as "reusing weights" (slides 76–77, ≈57:19–58:53), assigned
to **lectures 18–19 on transfer learning**. Covered so far: [lecture 1](01-introduction.md); [lecture 2](02-how-to-train-a-neural-net.md),
≈1:04:32–1:10:02 (what an embedding is, and visualizing what a unit responds to);
[lecture 4](04-architectures-grids.md), slides 43, 47, 63 and 67 (feature maps, how they change with
depth, and the encoder–decoder); [lecture 6](06-generalization-theory.md), slides 52–60 (kernels of a
network's output representation, and the low-rank bias of depth); [lecture 8](08-architectures-transformers.md), slides 11 and
31, ≈8:32–10:04 and ≈41:48–42:36 (tokens as representations at every layer, and attention maps in a
trained transformer); [lecture 11](11-representation-learning-reconstruction-based.md), slides 3–35 and 42–43, ≈0:00–45:43
and ≈54:21–58:19 (what a representation is, layers as transformations of a distribution, probing a network like a brain,
why representations are learned, what makes one good, and the trade-offs any one makes). Lecture 11's methods for learning
one without labels have their own pages: [autoencoders](autoencoders.md), [self-supervised learning](self-supervised-learning.md)
and, for what a representation is used for, [transfer learning](transfer-learning.md). [lecture 12](12-representation-learning-similarity-based.md), slides 3–9, 40–46 and
69, ≈1:33–11:37, ≈45:15–54:42 and ≈1:14:13–1:15:46, lists what a good representation should be and learns one from similarity;
its methods are on [metric learning](metric-learning.md) and [contrastive learning](contrastive-learning.md). [lecture 13](13-representation-learning-theory.md),
slides 4–7 and 20–26, ≈4:49–8:41 and ≈45:07–1:12:44, asks what similarity an untrained architecture already builds in.

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

## Kernels, and why deeper representations cluster (lecture 6)

Lecture 6 characterizes a network's output representation by its **kernel**, the matrix of similarities
between the representations of every pair of inputs: "every row is a data point and the column is another
data point" (slide 53, ≈1:06:45–1:08:18). A block-structured kernel means the network has clustered the
data, "all the red points have grouped together into one block", which makes the classes easy to separate.
The lecturer says these kernels will return "in the representation learning lectures" (lectures 11–13). Lecture 13 also
studies networks with random weights through a similarity of outputs, the covariance function of the infinite-width limit (see
below); its recording does not connect that to lecture 6's kernels.

Deeper networks give blockier kernels, and the "common understanding" is that they "have greater capacity
to organize the data" (slides 52–53). But deep *linear* networks show the same effect though depth adds
them no capacity (slide 54), so lecture 6, following Huh et al. (TMLR 2023), explains it differently:
"products of matrices tend to be low rank" (slide 55). For random weights, a deeper network is more likely
to map the data to a low-rank kernel, measured by its effective rank $\rho(K)$, and the distribution of
effective rank moves lower as depth grows from 1 to 16 (slides 56–59, ≈1:09:04–1:12:56). Lower rank means
the data have been organized into "a simpler format", which lecture 6 offers as one inductive bias behind
generalization; see [generalization and double descent](generalization-and-double-descent.md).

## Tokens, and attention maps (lecture 8)

Lecture 8 defines a token as "a vector of neurons" with the connotation of "an encapsulated bundle of
information", and uses the word for the representation of the data at *any* layer, not only for the
units of the input vocabulary as natural language processing usually does (slide 11, ≈8:32–10:04). A
token starts as the flattened pixels of a patch, "but later on, it could be more abstract" (≈29:20).

What a trained transformer's tokens attend to says something about what it has learned. In DINO (Caron
et al., 2021), shown on slide 31 as a video, the self-attention of a token on a horse falls on the horse's
other patches, "So it's solving the segmentation problem. It's like, if I want to understand something
about this part of the image, I should attend to other things that are related to it, and I shouldn't
cross object boundaries" (≈41:48–42:36). The lecturer is careful about what that shows. The strategy was
found by backpropagation, it "often has an intuitive interpretation … but it doesn't have to", and
whether it always arises is "more empirical science … it's not provable" (≈43:22–44:55). See
[transformers](transformers.md).

## What a representation is (lecture 11)

Lecture 11 opens the course's "next third" with this perspective (≈0:00). "Deep nets transform datapoints, layer by layer",
and "Each layer is a different *representation* of the data" (slide 3). The forward direction, "from observed data to latent
embeddings", is **representation learning**; the reverse, "from latent embeddings to observed data", is **generative
modeling**, which "can roughly be thought of as" its inverse (slide 3, ≈2:23). One name for the forward map is **x2vec**,
after word2vec, for any modality (slide 4, ≈3:08). Another, a student points out, is feature extraction: "another name for the
same thing" (≈3:55–4:40).

Slide 22 makes it precise, restricting attention mainly to vector embeddings. "A representation of a data domain
$\mathcal{X}$ is a function $f : \mathcal{X} 	o \mathbb{R}^d$ that assigns a feature vector to each input in that domain. This
function is called an **encoder**." And "A representation of a datapoint $\mathbf{x}$ is a vector
$\mathbf{z} \in \mathbb{R}^d$ with $\mathbf{z} = f(\mathbf{x})$." For a neural network, $f$ is parameterized by its weights and
biases, so "the representation, by this definition, is its weights and biases"; "When we say the learned representation, we
usually mean f" (≈27:08).

The cartoon the lecture reuses draws the data space as a blob, "some complicated object, high-dimensional, weird topology", and
the representation space as a circle, "because … the representation is, oftentimes, a simpler space", whose distribution "will
typically be a Gaussian distribution" (slide 5, ≈5:27). A representation need not be smaller than its input, though "Often, the
representation you want to be a smaller object" (≈4:40).

## Layers as transformations of a distribution (lecture 11)

To see what each layer does to the data, lecture 11 draws a function not as a graph but as a **mapping** from points on an input
line or plane to points on an output one (slides 6–9). Seen this way a linear layer rotates, squishes and scales; a ReLU maps
everything into the positive orthant and onto the axes, a sparse representation; an L2, RMS or layer norm puts every point on
the unit hypersphere; and a softmax puts it on the simplex (≈7:46–12:21). See [activation functions](activation-functions.md),
[normalization layers](normalization-layers.md) and [softmax and cross-entropy](softmax-and-cross-entropy.md).

Stacking these, a width-2 MLP trained to separate a red cloud from a surrounding blue one moves the two classes apart layer by
layer, and "each of the layers now can be understood as a different representation of the data distribution and a better and
better representation" for the task (slide 10, ≈13:53–14:39); see [multilayer perceptron](multilayer-perceptron.md). In CLIP,
a large vision network with its embeddings reduced to two dimensions by PCA, the classes of a data set separate as depth grows
(slide 13, ≈16:17–17:50). The general picture: from "entangled, complicated data at the input" a network, "layer by layer,
gradually morphing, disentangling it through these geometric transformations", reaches "clean separation of your semantics of
interest", and "the output space is a simpler object than the input space" (≈17:50–18:36).

## Probing a network like a brain (lecture 11)

Lecture 11 returns to the visual-cortex hierarchy of lecture 1 (Serre, 2014; slide 14): five or six layers of linear filtering
and pointwise non-linearity, edges first, then conjunctions of edges, up to a layer called IT "where the semantics are
segregated" (≈19:23–20:09). It then probes an artificial network the way a neuroscientist probes a brain, **deep net
"electrophysiology"** (slides 15–16): record a unit's activation and find the inputs that turn it on. "One of the modern names for
this type of work is interpretability or mechanistic interpretability … but it's an old problem" (≈20:55).

The results are Zeiler and Fergus's (2014), the image patches that most activate units at each layer of a convolutional network
(slides 17–20): oriented edges and colour at layer 1 ("like a greenness detector"), conjunctions of edges such as crosses, circles
and gradients at layer 2, a crude face template at layer 3, and dog faces separated from human faces at layer 5 (≈21:43–24:47).
"The rough story is that deep nets and the neuroscience models of the brain are quite in alignment here" (slide 21, ≈24:47).
Modern networks with "dozens or hundreds of layers" have "very precise detectors deep in the network that code for very specific
things", in language models too, as in the paper "The Sentiment Neuron" (≈24:47). One difference: the brain has only about seven
layers of filters, "But the brain has recurrence, so it uses those seven layers over and over again" (≈25:32).

The same probe on a network trained only to colorize grayscale images finds units for faces, dog faces and flowers at layer 5
(slide 55). Whatever the training task, "the units that carve the world at its joints … turn out to be objects and semantics and
the words that humans have" (≈1:09:08); see [self-supervised learning](self-supervised-learning.md).

## Why learn one, and what makes one good (lecture 11)

The main reason to learn a representation is "to do more learning": to adapt it to new tasks with little data (slide 24,
≈27:54); see [transfer learning](transfer-learning.md). Slide 31 lists what a good one is: **compact** (minimal), **explanatory**
(sufficient), **disentangled** (independent factors), **interpretable**, and above all one that will "Make subsequent problem
solving easy". Minimal and sufficient go together: "very low-dimensional or low-information, but sufficient, that they are
sufficient statistics for solving your tasks", like a scene reduced to "Just three things and where they are, but not all the
weird, photometric details" (≈40:17–41:04). Compactness also has "an Occam's razor generalization theory type of" justification
(≈40:17). The class adds unit variance, context awareness and robustness to small changes in the input (≈41:51–42:36). The classic
example of the last property is the Fourier transform: "It's a representation of data that makes convolution really easy. It
makes convolution just into a product" (≈42:36–43:23).

## Every representation trades something off (lecture 11)

An autoencoder trained on coloured shapes (slides 42–43) organizes them sensibly, with a query's nearest neighbours in its code
being the same shape in about the same colour (≈55:12). But measured layer by layer, shape gets easier to read off with depth and
colour harder. Pixels are a representation too, and "color is very superficial and explicit in pixel space … Shape is not
explicitly represented in pixel space" (≈57:32). So "every representation is good at some things and bad at other things, and
there's trade-offs … autoencoding doesn't strictly result in better representations. It results in different representations.
That's a fundamental principle" (≈58:19).

## Learning without labels (lecture 11)

A representation can come from supervised training on any task, but the lecture's interest is in methods that "just try to learn
good representations generically", from data with no labels (≈43:23). The output can be embeddings, clusters or metrics (slide
34), and there are "two general principles": **compression**, the route of [autoencoders](autoencoders.md), PCA, k-means and
vector-quantized autoencoders, and **prediction** of held-out data, the route of [self-supervised learning](self-supervised-learning.md)
(slide 35, ≈44:11–44:58). Clustering is itself representation learning, an encoder to integers, and in the lecturer's view "the
problem of making up new words for things", words being the atoms of language, "the best representation of the world that humans
have discovered" (slide 45, ≈1:00:38–1:01:23). Lecture 11 ends on LeCun's cake, whose bulk is representation learning: "the bulk
of intelligence, and I agree with that point" (slide 63, ≈1:20:05). Metric (similarity-based) learning is lecture 12's subject; see the next section.

## Five properties, and similarity as the signal (lecture 12)

Lecture 12 starts from the same question, "Why learn representations?", with five answers: to improve generalization, to do
more learning (transfer), to exploit geometric similarity for new data or queries (has this face been seen before; which items
are similar to a query), to improve clustering with side information, and dimensionality reduction (slide 3, ≈1:33–3:51). It then
builds up a list of what a good representation is (slides 5 and 9):

1. compact (*minimal*): "We don't want to have more capacity than we actually need";
2. explanatory (*sufficient*): it keeps "the dimensions that matter", which depends on the downstream task;
3. concentration: data from the same class is close together;
4. separation: classes are well separated;
5. robustness to irrelevant perturbations: "if you saw me from this angle, you'd want to be robust to me looking the other
   direction" (≈6:12–10:50).

The evidence for the middle three is empirical. The winning entries of a NeurIPS 2020 competition on predicting generalization
looked at the geometry of the representation (consistency and separation) and at robustness to perturbations (slide 6), and a
CIFAR-10 network trained on true labels gives tight, separated clusters, while one trained on random labels memorizes into
fuzzy, overlapping ones (slides 7–8, ≈8:30–10:50; see [generalization](generalization-and-double-descent.md)).

To get these properties, the lecture trains on "feedback in terms of similarity: pairs of similar/dissimilar inputs" (slide 10):
first with labels ([metric learning](metric-learning.md)), then without ([contrastive learning](contrastive-learning.md)). Its
account of what the contrastive recipe does splits the list in two: the loss encourages concentration (**alignment**) and
separation (**uniformity** over a hypersphere), and the data, through the choice of positive pairs, encourages robustness, as
**learned invariance** (slides 40–46, ≈45:15–53:56). The summary: "Good representations capture relevant similarity/dissimilarity
information", with "well-clustered, compact and separated/spread out classes" that preserve relevant information and "teach
relevant invariances ('forget' irrelevant information)" (slide 69).

What counts as relevant is set by the task. A contrastive representation trained on iNaturalist without labels groups a bird held
in a hand with other birds held in hands rather than with its species, so "for contrastive learning to work well, you need to have a
good similarity measure for your problem of interest" (slides 67–68, ≈1:07:16), the same lesson as lecture 11's trade-offs.

## An architecture's built-in similarity (lecture 13)

Lectures 11 and 12 learn a representation by training. [Lecture 13](13-representation-learning-theory.md) asks what similarity a
network expresses before training. It recaps lecture 12's goal, an objective "that maps similar data to nearby embeddings" (slide 4,
≈4:49), and sets aside lecture 7's neural, tensor and spectral perspectives for "a more abstract perspective": "a neural net is a map
through a sequence of vector spaces", one per layer, and the aim is "a good representation of the data at the final layer", for
example a linearly separable one (slides 5–6, ≈5:36–7:55). Its claim is that "a neural architecture (even without training)
already expresses an opinion about data similarity" (slide 7).

The opinion is read off the infinite-width limit. With iid random weights, a very wide network's outputs on any finite set of
inputs are jointly Gaussian, and their covariance function $\Sigma(x, x')$, which "depends on the architecture and non-linearity",
is large for inputs the network treats as similar and small for ones it treats as different (slide 23, ≈49:48–52:12). The lecture's
experiment shows it on random three-layer MLPs: a CIFAR-10 truck and a slightly noised copy give tightly correlated outputs, a heavily
noised copy much less so (slide 22). In a Gaussian process, "the covariance structure is some kind of measure of similarity between
the inputs" (≈38:47), so the covariance function is to an untrained architecture what a learned metric is to a trained encoder. The
lecturer is clear that this has had "very little impact on what people actually do in deep learning" (≈1:00:54). See
[Gaussian processes](gaussian-processes.md) and [inductive bias](inductive-bias.md#the-architectures-opinion-about-similarity-lecture-13).
