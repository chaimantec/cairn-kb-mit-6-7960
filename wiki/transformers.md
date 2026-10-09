# Transformers

A **transformer** is a network that operates on a set of **tokens**, vectors that each carry a chunk of
the input. It alternates two operations. **Self-attention** replaces each token with a weighted
combination of all the tokens, with weights computed from the tokens themselves. A **token-wise MLP**
then transforms each token separately. A **positional encoding** added to each token records where it
came from, because attention by itself ignores order. The course presents it as "the architecture that
you should use today", while warning that "next year, there'll be a new architecture" (lecture 8,
≈0:00). Covered so far: [lecture 8](08-architectures-transformers.md), slides 1–54, ≈0:00–1:13:49; masked autoencoders and BERT in
[lecture 11](11-representation-learning-reconstruction-based.md) (slides 59–60, ≈1:11:26–1:15:20); with
the graph-net view previewed in [lecture 5](05-architectures-graphs.md) (slide 24, ≈49:42, ≈56:38), and
the transformer as part of the default recipe in [lecture 9](09-hackers-guide-to-deep-learning.md) (slides 41, 42 and 61); and attention set against recurrence and convolution, with the efforts to lengthen its context, in [lecture 10](10-architectures-memory.md) (slides 57–64, ≈58:19–1:10:02).

**Notation.** A token $\mathbf{t}_ i \in \mathbb{R}^d$ is a vector of $d$ neurons, and $N$ tokens are
stacked as the rows of $\mathbf{T} \in \mathbb{R}^{N \times d}$, with $\mathbf{T}_ {\text{in}}$ and
$\mathbf{T}_ {\text{out}}$ a layer's input and output ([notation](notation.md)). $\mathbf{W}_ q$,
$\mathbf{W}_ k$, $\mathbf{W}_ v$ are learned projections to queries, keys and values;
$\mathbf{Q}_ {\text{in}}, \mathbf{K}_ {\text{in}}, \mathbf{V}_ {\text{in}}$ stack them for every token; and
$\mathbf{A} \in \mathbb{R}^{N \times N}$ is the attention matrix.

## Three ideas, one of them new

Lecture 8 builds the transformer from three ideas: tokens, attention and positional encoding. Only
attention is new: "Everything else will actually be just new names for old ideas" (≈0:45). The lecture
frames this with Borges's story of Pierre Menard, who rewrote *Don Quixote* word for word, so that the
same words take on new meaning in a new context: "everything old is new again" (slide 3, ≈1:30–2:17).

## Why attention: the limit of locality

A ConvNet's filters are local, and stacking them widens what each unit sees only slowly. In lecture 8's
two-layer example, no output unit depends on both the first and the seventh input, so "far apart image
patches do not interact" (slide 5, ≈4:38–6:11). A fully connected layer sees everything but costs $n^2$
parameters for a map from $\mathbb{R}^n$ to $\mathbb{R}^n$. Attention looks "globally, but … sparsely
globally": for each question it weights the parts of the input that matter, as a person counting birds
looks at the birds (slides 4–8, ≈6:11–7:43). See [convolution](convolution.md) and
[inductive bias](inductive-bias.md).

## Tokens

"A token is just a vector of neurons", the node attribute vector of a graph net under a new name, with
the connotation of "an encapsulated bundle of information" (slide 11, ≈8:32). The course uses the word
for the representation at *any* layer. In natural language processing it more often means a discrete
unit of the vocabulary, present only at the input and output (≈10:04). Transformers operate on sets of
tokens rather than arrays (slides 11–13, ≈10:50–11:35).

Turning data into tokens is the one domain-specific step: "First, the domain expert turns data into
tokens, and then generic transformer processes tokens" (≈12:22). An image is cut into non-overlapping
patches, each flattened and multiplied by a matrix $\mathbf{W}_ {\text{tokenize}}$ to give a token in
$\mathbb{R}^d$ (slide 14). Text is cut into byte pairs and audio into snippets (slide 15, ≈14:42–16:14).
The chunks should be local, "because locality is a very good inductive bias" (≈15:29).

## Token nets

A network over tokens has the MLP's two operations, lifted from scalars to vectors (slides 17–21,
≈17:47–22:24). A **linear combination of tokens**, $\mathbf{T}_ {\text{out}} = \mathbf{W} \mathbf{T}_ {\text{in}}$,
scales whole tokens by one weight each, which is "a low rank transformation over the neurons" (≈19:20).
A **token-wise nonlinearity** applies one function $F_\theta$, "typically an MLP", to every token
independently and identically, equivalent to a $1 \times 1$ convolution over the token sequence (slide 18,
≈20:06–21:37). So "transformers are like a meta architecture. Like, the units of the architecture are
other neural networks" (≈20:06). See [multilayer perceptrons](multilayer-perceptron.md).

Unrolled message passing in a [graph neural network](graph-neural-networks.md) has the same shape, with
AGGREGATE and COMBINE. The difference is who decides the weights of the combination. In a GNN it is the
graph's adjacency matrix, and the aggregate may be nonlinear. A transformer connects every token to every
other and computes the weights by attention. It is "a graph net over a fully connected graph, typically
without weight-sharing in depth" (slides 22–24, ≈22:24–23:55).

## Attention

An attention layer is a linear combination of tokens, $\mathbf{T}_ {\text{out}} = \mathbf{A} \mathbf{T}_ {\text{in}}$,
whose weights $\mathbf{A}$ are not free parameters but "a function of some input data. The data tells us
which tokens to attend to" (slide 26, ≈27:02–27:49). Hand-built examples show the idea. Weighting the
animal-head patches by 1 and summing an "is an animal head" entry counts the animals; averaging the
impala's patches reads out its colour (slides 27–28, ≈28:34–30:56).

**Query-key-value attention** (slide 29, ≈31:43–36:22) computes the weights. Each token emits a query
$\mathbf{q} = \mathbf{W}_ q \mathbf{t}$, a key $\mathbf{k} = \mathbf{W}_ k \mathbf{t}$ and a value
$\mathbf{v} = \mathbf{W}_ v \mathbf{t}$. A query's dot product with each key scores how relevant that
token is, a softmax turns the scores into weights that sum to 1, and the output is the weighted sum of
the values. The names come from a database lookup: "when the query matches the key, you open that part
of the database … and you pull out the values" (≈34:50). The three projections are the layer's only
learned parameters (≈37:55).

In **self-attention** the queries come from the tokens themselves, so every token attends to every
token, itself included (slide 30, ≈41:01). For a whole layer (slide 34, ≈45:45–48:51),

$$\mathbf{Q}_ {\text{in}} = \mathbf{T}_ {\text{in}} \mathbf{W}_ q^{\mathsf{T}}, \qquad \mathbf{K}_ {\text{in}} = \mathbf{T}_ {\text{in}} \mathbf{W}_ k^{\mathsf{T}}, \qquad \mathbf{V}_ {\text{in}} = \mathbf{T}_ {\text{in}} \mathbf{W}_ v^{\mathsf{T}},$$

$$\mathbf{A} = \text{softmax}\left( \frac{\mathbf{Q}_ {\text{in}} \mathbf{K}_ {\text{in}}^{\mathsf{T}}}{\sqrt{m}} \right), \qquad \mathbf{T}_ {\text{out}} = \mathbf{A} \mathbf{V}_ {\text{in}},$$

where $m$ is the dimension of the queries and keys, which need not equal $d$. Everything "just becomes
these matrix multiplies" (≈46:32). In a trained vision transformer, a token's attention tends to stay
inside its object: "the horse token attends to the horse patches. So it's solving the segmentation
problem" (DINO, slide 31, ≈41:48–42:36). The lecturer stresses that nothing forces this. The mechanism is
found by backpropagation, and its intuitive reading is an empirical observation, "not provable"
(≈43:22–44:55).

**Why not learn $\mathbf{A}$ directly?** The recipe "just seems to work the best", and learning
$\mathbf{A}$ itself would take $n^2$ parameters that depend on the sequence length $n$, while the
projections depend only on the token dimension, "much smaller" than sequences of up to "length million"
(≈49:39–51:13).

**Multihead self-attention** runs $k$ attention layers in parallel, each with its own query, key and
value functions, and combines their outputs, so that different heads can attend to different things,
"shapes" and "textures" for example (slide 37, ≈55:49–56:35).

## Attention in the family of linear layers

Slide 35 compares three linear layers (≈52:45–55:49). A fully connected layer has a separate weight on
every edge, $N^2$ parameters, and a fixed input size. A convolution has a Toeplitz matrix with $k + 1$
parameters for a kernel of size $k$ (with the bias), any input size, and translation equivariance. An
attention layer, seen as a matrix over the neurons, is a pattern of diagonal blocks, because each token
is scaled by one weight. Its parameters are those of $\mathbf{W}_ q$, $\mathbf{W}_ k$ and $\mathbf{W}_ v$.
It takes any number of tokens and is permutation equivariant. The lecturer calls it "this low-rank sparse
transformation".

## The transformer block

The "vanilla" transformer alternates self-attention with a token-wise MLP, as an MLP alternates linear
layers with ReLUs (slide 36, ≈55:49). The vision transformer (ViT) block adds two things around each
half (slide 38, ≈56:35–58:56):

- a **residual connection**, "an identity pass around all this processing", as in ResNets
  ([skip connections](skip-connections.md));
- a **token norm**, "more commonly called in the literature LayerNorm", which normalizes each token to
  zero mean and unit variance,
  $x_{\text{out}}[k] = (x_{\text{in}}[k] - \mathbb{E}[x_{\text{in}}[k]]) / \sqrt{\text{Var}[x_{\text{in}}[k]]}$.

The lecturer links layer norm to lecture 7's argument for normalizing weight updates. Normalizing
activations changes the size of the weight updates too, and "exactly what those are is basically open
science" (≈58:09–58:56; see [scaling rules](scaling-rules.md) and [norms](norms.md)). The block repeats
$L$ times, and slide 39's pseudocode is a dozen lines of matrix products (≈58:56–1:00:29).

## Positional encoding

Self-attention and the token-wise MLP both treat tokens without regard to order, so a transformer is
**permutation equivariant**: permuting the input tokens permutes the output tokens the same way, "a
set-to-set mapping" (slide 41, ≈1:01:16–1:02:02), like a graph net. When order matters, as in a sentence
whose earlier words shape the later ones, each token is concatenated with a code for its position
(slides 42–43, ≈1:02:02–1:02:48). This is the trick lecture 4 used to stop a ConvNet being translation
equivariant ([neural fields and positional encoding](neural-fields-and-positional-encoding.md)).

The "vanilla" code is a **Fourier positional code**: the values of sines of several frequencies,
$\sin(x / B^j)$ and $\sin(y / B^j)$, at the token's location (slide 44, ≈1:02:48–1:04:18). The
positional encoding is also where domain knowledge goes in, since it tells the system "what it means
to be local for your domain" (≈1:04:18). Lecture 8's examples are ground-sample-distance encodings for
satellite images at different scales (ScaleMAE), spherical harmonics for positions on the globe, and
Laplacian eigenvectors for nodes of a graph (slides 45–47, ≈1:04:18–1:05:51).

## Language models: causal attention

An **autoregressive** model predicts the next word of a sequence, appends it, and repeats. Trained by
supervised classification of the next word over a vocabulary, it is "how ChatGPT works" (slides 48–49,
≈1:06:38–1:07:25). **GPT**, the Generative Pre-trained Transformer, is a transformer trained this way
(slide 50, ≈1:08:17). To stop the network reading the answer from its input, the attention matrix is
**causally masked**: "every output token can only attend to earlier tokens in the sequence", with the
masked weights set to zero. The model is then no longer permutation equivariant, and every position can
be supervised at once (slides 51–52, ≈1:09:03–1:10:38).

"Attention Is All You Need" (Vaswani et al.) defined the architecture, "the worst diagram in all of
science, but it's also the best paper in all of science". In the lecturer's reading, its "Feed Forward"
is the token-wise MLP, its masked attention over shifted outputs belongs to autoregressive modelling
rather than to transformers, and its "Add & Norm" is the residual connection (slide 53, ≈1:11:27–1:13:02).

**Cross-attention** lets one set of tokens attend to a different set. In an image-captioning model, a
text decoder attends causally to the words so far and, through cross-attention, to the tokens of an
image encoder (slide 54, ≈1:13:02–1:13:49).

## Why transformers are everywhere

The lecture gives three reasons. They are agnostic to the domain, so once data is tokenized "you just
use a transformer" (≈11:35). They are made of matrix multiplications, which "maps wonderfully onto
modern compute" (≈1:00:29; also ≈13:56). And homogeneity pays: "everything runs on the same commodity
hardware. The code bases are all the same. The lessons are transferable between different modalities",
traded against what specializing to a modality would buy (≈51:59). Domain knowledge enters in two
places, the tokenizer and the positional encoding.

## The default architecture (lecture 9)

Lecture 9's "recipe for deep learning in a new domain", one of its "good default choices ca 2024", ends:
"Use a generic optimizer (Adam) and a standard architecture (transformer) to solve the learning problem"
(slide 41, as printed "an standard architecture"). The next slide says to use layer norm, the
transformer's token norm, rather than batch norm (slide 42; see
[normalization layers](normalization-layers.md)). The lecture also cautions against reading the
transformer's origin paper as a recipe: "attention is not all you need. Attention is one thing that works
pretty well and has certain effects"; tuning is choosing a combination of "spices" (slide 61, ≈1:14:09).

## Attention against recurrence, and longer contexts (lecture 10)

Lecture 10 sets attention beside the other two ways of modeling arbitrarily long sequences (slide 58). Recurrence
shares its weights across time and passes a hidden state forward; convolution shares its weights across time but
sees nothing outside its window; attention's weights are "dynamically determined as a function of the data"
(≈59:56–1:00:41; see [recurrent neural networks](recurrent-neural-networks.md)). Slide 59 reproduces Table 1 of "Attention Is All You Need": for sequence length $n$
and representation dimension $d$, self-attention costs $O(n^2 \cdot d)$ per layer with $O(1)$ sequential operations
and an $O(1)$ maximum path length, where a recurrent layer costs $O(n \cdot d^2)$ with $O(n)$ of each. The title, in
the lecturer's reading, says "we don't need memory. All we need is attention", and the price is the $n^2$, which
"takes a lot of memory" (≈1:01:29–1:02:17).

Three lines of work stretch the context within limited memory (slides 60–62): sparsifying attention, such as the
Reformer's hashing, which reaches $O(n \log n)$, or the low-rank approximations of Performers and Linformers, which
reach $O(n)$, often at "the
expense of a little bit of performance"; combining local and global attention, as in Transformer XL, the Longformer
and Big Bird; and retrieval, as in RETRO, which separates a lighter language model from a trillion-token database of
facts (≈1:03:02–1:06:07). Context windows grew from BERT's 512 tokens to GPT-4's 8,000, with a 32K version, and about
100K for an Anthropic model (slide 63). But Mangalam et al. found that most video benchmarks need only about two
seconds of context, so the benefit of a longer one may simply go unmeasured (slide 64, ≈1:06:52–1:10:02; see
[lecture 10](10-architectures-memory.md)).

## Masking tokens: masked autoencoders and BERT (lecture 11)

Lecture 11 uses the transformer to learn representations without labels. The **masked autoencoder** (slide 59, He, Chen, Xie, et
al. 2021) takes a vision transformer, which "already tokenized the image" into patches, removes some of the tokens and predicts the
missing ones (≈1:12:14). Attention makes this easy: "I can only keep four of these tokens, but then the attention mechanism will scale
in a way that is proportional to the number of tokens … So it has this nice kind of architectural invariance to the number of tokens
you put in". The decoder is another transformer given "some blank tokens", trained so that they are filled in with the missing
pixels (≈1:13:01). "Masked autoencoders are just a new name for another model which was very popular called BERT", which masks tokens
of text (slide 60). Autoregressive language models are the same idea with only the final token masked, which fits generation and is
why, in the lecturer's account, BERT has gone out of fashion (≈1:13:47–1:15:20). See [self-supervised learning](self-supervised-learning.md).
Lecture 11 also shows a trained vision transformer, CLIP, separating the classes of a data set layer by layer (slide 13, ≈16:17–17:50).

## Where it goes next

The problem set announced in lecture 8, Homework 3 on OCW, implements a transformer, a vision transformer and a small
GPT ([lecture 8](08-architectures-transformers.md#the-problem-set)). Autoregression continues in [lecture 10](10-architectures-memory.md) (see [autoregressive models](autoregressive-models.md)), and generative
models return in the generative-model lectures (14–16), and language models in lecture 21; see the
[course map](course-map.md).
