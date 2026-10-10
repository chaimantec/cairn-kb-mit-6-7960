# Autoencoders

An autoencoder learns a representation by **compression**: an encoder $f$ maps a data point $\mathbf{x}$ to a
smaller code $\mathbf{z} = f(\mathbf{x})$, a decoder $g$ maps the code back, and the two are trained so that the
reconstruction $\hat{\mathbf{x}} = g(f(\mathbf{x}))$ matches $\mathbf{x}$. Lecture 11 calls it "representation
learning algorithm 101, just the most basic vanilla one, and also my favorite, and probably the best one. And I
think it will just win out in the end" (≈46:31), and builds a family around it: with linear maps it is PCA, with
an integer bottleneck it is k-means, with deep nets on both sides of that bottleneck it is a vector-quantized
autoencoder, and with part of the input hidden it becomes a masked autoencoder. Covered so far:
[lecture 4](04-architectures-grids.md), slide 67, ≈1:00:25–1:02:00 (the convolutional encoder–decoder);
[lecture 11](11-representation-learning-reconstruction-based.md), slides 36–49 and 59–62, ≈44:58–1:05:15 and
≈1:11:26–1:20:05. Variational autoencoders, which make the code's distribution Gaussian, are promised for "later"
(≈47:20–48:06); [lecture 14](14-generative-models-basics.md) places them in lecture 15 (slide 2, ≈1:31).

## Compress, then reconstruct

The idea is to "Just map the data to a simpler form such that you can decode the original data from that simpler
form" (≈46:31). Mechanically, the encoder is a network "where the width decreases as a function of depth", so each
layer is lower-dimensional than the last (≈45:43). A compact code alone is easy and useless: "I don't want to just
map my input to 0, right? I want to map it to something that actually still can reconstruct the data". The decoder
supplies that requirement (lecture 11, slides 36–38).

Lecture 4 met the same structure as a convolutional **encoder–decoder**: convolution, non-linearity and
subsampling "to get down to some very low dimensional representation" $\mathbf{z}$, then convolution, non-linearity
and upsampling back, trained so that "you want the decoded output to match the encoded input as close as possible"
(slide 67, ≈1:00:25–1:02:00). Once trained, either half is useful on its own: the encoder to represent images, the
decoder to generate them.

## The objective, and why it is not trivial

Lecture 11 writes the training problem as

$$f^{\ast}, g^{\ast} = \arg\min_{f, g} \mathbb{E}_ {\mathbf{x}} \lVert \mathbf{x} - g(f(\mathbf{x})) \rVert_2^2$$

(slide 39): encode, decode, and minimize the squared reconstruction error in expectation over the data. In its
learning-problem form (slide 40), the **$L_2$ autoencoder** has objective
$\mathcal{L}(F(\mathbf{x}), \mathbf{x}) = \lVert F(\mathbf{x}) - \mathbf{x} \rVert_2^2$ over the hypothesis space
$F = g \circ f : \mathbb{R}^N \to \mathbb{R}^M \to \mathbb{R}^N$, "Typically, M\<N".

The objective alone asks for the identity: the composite $F$ "should ideally just be an identity. It should do
nothing … If I had no constraints on the architecture, on the hypothesis space, the autoencoder would be a trivial
thing to learn, and it would have no utility. It would just spit out an exact copy of the data. But the key thing is
the autoencoder works because you put constraints over the architecture" (≈48:55). For a vanilla autoencoder the
constraint is the bottleneck, $M \lt N$; other variants "impose simplicity on the embeddings in other ways than
dimensionality reduction" (≈49:41).

That is also why a shortcut ruins it. Asked what would be wrong with an autoencoder whose $f$ and $g$ had residual
connections between them, a student answers that it would not force the low-dimensional representation: "It skips
the bottleneck" (lecture 11, ≈1:17:41). Lecture 4 makes the converse point about U-nets, where skip connections are
added precisely so that detail the bottleneck would lose can pass around it (see
[skip connections](skip-connections.md)).

## Linear autoencoders are PCA

If $f$ and $g$ are both linear, "Then the embedding spans the same M-dimensional subspace as PCA" (lecture 11, slide
41). PCA "can be understood as trying to maximize the variance I'm capturing in the signal via some linear orthogonal
transformation … And if I'm maximizing the variance, I'm able to best reconstruct. And that actually is exactly
equivalent to the L2 reconstruction objective" (≈50:27–51:15). In the lecturer's spoken derivation, the linear encoder
and decoder are matrices $\mathbf{W}_ f$ and $\mathbf{W}_ g$; PCA is the variant where encoder and decoder are one
orthogonal matrix $\mathbf{W}$; and the reconstruction error then splits into the variance of the data, which does not
depend on $\mathbf{W}$, minus the variance captured, so minimizing one is maximizing the other (≈51:15–52:50). The
equations he worked are not in the OCW deck.

So "an autoencoder … is just nonlinear PCA. That's a generalization of PCA to nonlinear representations" (≈52:50).

## What it learns, and what it gives up

On a data set of coloured triangles, circles and squares (lecture 11, slides 42–43), an autoencoder's code puts a
triangle's nearest neighbours among triangles of about the same colour, organizing the data "in a way that seems
meaningful and seems to match human perception" (≈55:12–55:59). Measured layer by layer with a one-nearest-neighbour
classifier, shape gets more decodable with depth and colour less (slide 43). Colour was already explicit in the
pixels, the input's own representation, while shape was not; so "autoencoding doesn't strictly result in better
representations. It results in different representations … All representation learning is making trade-offs"
(≈57:32–58:19). See [representation learning](representation-learning.md).

## k-means: an integer bottleneck

"Autoencoders are the deep learning version of PCA … another model is the deep learning version of k-means" (≈58:19).
Clustering is an encoder to integers, $f : \mathcal{X} \to \lbrace 1, \ldots, k \rbrace$, or equivalently to one-hot
codes (slides 44–45). Give it a decoder that is "like a lookup table … Think of g as a matrix W applied to a one-hot
code. It selects a row of that matrix" (slide 47, ≈1:02:08–1:02:54). Every member of a cluster gets the same code and so
the same reconstruction, and the vector closest in squared distance to a set of points is their mean. "So that means
that k-means is an L2 autoencoder" (≈1:02:54).

Slide 48 writes it in the same form as slide 40: the same objective, the hypothesis space
$F = g \circ f : \lbrace \mathbf{x} \rbrace_{i=1}^{N} \to \lbrace 1, \ldots, k \rbrace \to \mathbb{R}^M$ with "f and g
are both lookup tables", and the optimizer "Block coordinate descent". "The only difference … is that the hypothesis space
doesn't have a low-dimensional bottleneck. It has an integer bottleneck." It is not differentiable, so it is optimized
by a method that exploits its structure rather than by SGD; the lecturer names expectation maximization (≈1:03:42).

## Vector-quantized autoencoders

"What if f and g are both deep nets? Then we call this a **'Vector Quantized' Autoencoder** (e.g., VQVAE, VQGAN)"
(slide 49, citing van den Oord, Vinyals and Kavukcuoglu, 2017). It keeps the integer bottleneck, adds terms to the
objective (the slide's "+ …"), and is trained by "Backprop w/ approximations". "These have bells and whistles, but this
is the gist of it … it's just k-means with deep nets" (≈1:04:28). The number of codes $k$ is a hyperparameter the user
sets (≈1:05:15).

## Masked autoencoders

Hide part of the input and ask the decoder for the hidden part, and the autoencoder becomes a predictor. The **masked
autoencoder** (slide 59, He, Chen, Xie, et al. 2021) masks random patches of an image, encodes only the visible ones with
a vision transformer, and lets a second transformer fill blank tokens with the missing pixels; attention scales with the
number of tokens it is given, so the masking ratio can vary freely (≈1:11:26–1:13:01). BERT is the same idea on text
(slide 60). Lecture 4 had already set masked autoencoders beside the convolutional encoder–decoder, as similar but using
"attention as opposed to convolution as a way to share information over space" (≈1:00:25–1:02:00). See
[self-supervised learning](self-supervised-learning.md).

## Reconstruction against prediction

Empirically, "masked prediction just tends to always work better than autoencoding" (lecture 11, ≈1:15:20): on slide 61,
a colorization network's layers support a far better linear ImageNet classifier than an autoencoder's from the middle
layers on. Slide 62's three hypotheses: a dimensional bottleneck is hard to control and low-dimensional embeddings
"interact with optimization in weird ways"; autoencoders take shortcuts, copying local information for a decent loss; and
prediction is closer to the downstream problems people care about. "Still an open question!"

The lecturer still bets on autoencoders, from first principles he says he cannot prove: if compression is somehow
equivalent to prediction, and "the most compressed representation will make the most accurate predictions about the
future", then "if compression is all you need, then autoencoders are all you need" (≈1:18:30–1:20:05).

## The problem set

Homework 4's question on autoencoders (6 points) trains one on $64 \times 64$ images of coloured shapes, asks what a
perfect reconstruction on a finite training set does and does not determine about the code, and has you visualize the
encoder by nearest neighbours and explain any clusters, since "The Autoencoder objective alone … doesn't enforce any
grouping or smoothness of the representation space". Homework 5 has a section on variational autoencoders (14 points). See
[sources](../sources.md).

## Toward variational autoencoders (lecture 14)

Lecture 14 introduces [generative models](generative-models.md) as the inverse of representation learning, and sends variational
autoencoders to lecture 15, "a model that does both directions jointly" (≈1:31; slide 2's "Lecture 15: generative modeling meets
representation learning"). Its view of a generator's random inputs as latent variables, "control knobs that specify all the attributes
of the data" (≈23:03), is the role an autoencoder's code plays for its decoder. The lecturer also says that "a diffusion model is a type
of variational autoencoder" (≈3:03–3:48); see [diffusion models](diffusion-models.md).

## See also

- [Representation learning](representation-learning.md) — what a representation is, and how to tell a good one.
- [Self-supervised learning](self-supervised-learning.md) — learning by prediction rather than compression.
- [Contrastive learning](contrastive-learning.md) — learning from similarity rather than reconstruction (lecture 12).
- [Skip connections](skip-connections.md) — what a bottleneck loses, and why a skip around it defeats an autoencoder.
- [Lecture 11](11-representation-learning-reconstruction-based.md) — the lecture these sections come from.
