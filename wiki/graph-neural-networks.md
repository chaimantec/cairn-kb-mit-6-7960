# Graph neural networks

A **graph neural network** (GNN, or "graph net") is an architecture for data that lives on a graph:
a set of nodes, each with an attribute vector, joined by edges. It processes the graph by **message
passing**. In each layer every node gathers the vectors of its neighbours with a function that
ignores their order, and updates its own vector from what it gathered. The course builds it in
[lecture 5](05-architectures-graphs.md), the second of its architecture lectures, as a
generalization of the convolutional network of [lecture 4](04-architectures-grids.md) and a step
towards transformers. Covered so far: lecture 5, slides 2–46, ≈0:48–1:20:36; [lecture 6](06-generalization-theory.md),
≈23:57 and ≈1:16:06, on permutation symmetry as a source of generalization.

**Notation.** A graph $G$ has $n$ nodes, an adjacency matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$
($A_{ij} = 1$ when nodes $i$ and $j$ share an edge) and a feature matrix
$\mathbf{X} \in \mathbb{R}^{n \times d}$ holding one $d$-dimensional attribute vector per node.
$\mathcal{N}(v)$ is the set of node $v$'s neighbours, $\mathbf{h}_ v^{(k)}$ its vector after $k$
layers ($\mathbf{h}_ v^{(0)}$ being its attributes), and $\mathbf{m}_ {\mathcal{N}(v)}^{(k)}$ the
message it receives from its neighbours in layer $k$. These follow lecture 5's slides.

## What graphs are for

Graph nets suit "problems that naturally can be specified with a graph or data types that exist on a
graph" (lecture 5, ≈3:52): social networks, molecules (atoms and bonds), drug–protein interaction
networks, road networks, particles exerting forces on each other, and combinatorial optimization
problems written as graphs of constraints and variables (slides 3–9). A question can concern a
node (does this user like jazz?), an edge (link prediction, as in recommender systems), or the whole
graph (is this molecule toxic?). Slide 10 names the two outputs: **node embeddings**, one vector per
node, and a **graph embedding**, one vector for the graph. Where a classical algorithm already
solves the problem, a learned one can be faster on the data it was trained on, at the cost of any
guarantee on other data (≈12:26–13:59).

## The symmetry: permutation invariance and equivariance

The nodes of a graph can be numbered in any order, and the numbering is arbitrary. Renumbering by a
permutation matrix $\mathbf{P}$ turns the inputs into $\mathbf{P}\mathbf{A}\mathbf{P}^\top$ and
$\mathbf{P}\mathbf{X}$ without changing the graph. So a network $f$ on graphs should be (slide 11):

- **permutation invariant** when it outputs something about the whole graph,
  $f(\mathbf{P}\mathbf{A}\mathbf{P}^\top, \mathbf{P}\mathbf{X}) = f(\mathbf{A}, \mathbf{X})$;
- **permutation equivariant** when it outputs one vector per node,
  $f(\mathbf{P}\mathbf{A}\mathbf{P}^\top, \mathbf{P}\mathbf{X}) = \mathbf{P} f(\mathbf{A}, \mathbf{X})$:
  "if I shuffle my nodes, then I should end up shuffling my predictions" (≈23:16).

An MLP fed the flattened adjacency matrix and attributes is neither, because its input depends on
how the nodes happen to be indexed (≈20:12–22:31). This is the graph counterpart of the
convolutional layer's translation equivariance: "Where convolutional networks are invariant or
equivariant to translation, graph nets are going to be invariant or equivariant to permutations of
the inputs" (≈22:31), and "translation invariance is one type of permutation invariance, but
permutation invariance is a more general version of that" (≈41:08). Whether a problem calls for it
depends on the domain; see [inductive bias](inductive-bias.md).

## Message passing

One GNN layer, in the general form of slide 16, aggregates a message from the neighbours and then
updates the node:

$$\mathbf{m}_ {\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}^{(k)} \left( \lbrace \mathbf{h}_ u^{(k-1)} : u \in \mathcal{N}(v) \rbrace \right)$$

$$\mathbf{h}_ v^{(k)} = \text{UPDATE}^{(k)} \left( \mathbf{h}_ v^{(k-1)}, \mathbf{m}_ {\mathcal{N}(v)}^{(k)} \right)$$

The aggregate must not depend on the order of the neighbours. It is a **multiset function**: it maps
a set of vectors, possibly with duplicates and of any size, to one vector (slide 18, ≈35:39–36:26).
Common choices are the sum $\sum_{u \in \mathcal{N}(v)} \mathbf{h}_ u$, the average (the same sum
divided by $|\mathcal{N}(v)|$), a sum normalized by
$\sqrt{|\mathcal{N}(v)||\mathcal{N}(u)|}$, and a coordinate-wise max or min. A typical update is a
one-layer MLP (slide 20):

$$\mathbf{h}_ v^{(k)} = \sigma \left( \mathbf{W}_ {\text{self}} \mathbf{h}_ v^{(k-1)} + \mathbf{W}_ {\text{neigh}} \mathbf{m}_ {\mathcal{N}(v)}^{(k)} + b \right)$$

with learned $\mathbf{W}_ {\text{self}}$ and $\mathbf{W}_ {\text{neigh}}$ and a pointwise non-linearity
$\sigma$.

Each layer widens a node's view by one hop. After $k$ layers a node has heard from everything within
$k$ edges: its view is a tree, its neighbours below it, their neighbours below them, and so on
(slides 25–27, ≈57:24–59:00). That is the graph version of a growing receptive field: "the way you
make a global decision … is just by stacking those local operations" (≈29:28). For a graph-level
output, a final **readout** pools all the node vectors with another multiset function,
$\mathbf{h}_ {\mathcal{G}} = \text{READOUT}(\lbrace \mathbf{h}_ v^{(K)} : v \in \mathcal{G} \rbrace)$
(slide 21).

The aggregate is deliberately constrained: "aggregate is not a universal function, and it can't
represent everything" (≈37:12). But within the multiset functions there is a universal family,
$\text{MLP}_ 2 \left( \sum_{u} \text{MLP}_ 1 (\mathbf{h}_ u, \mathbf{h}_ v) \right)$ (slide 20): the
sum keeps it permutation invariant, and the two MLPs make it able to approximate any multiset
function (≈42:40–44:13). A min-aggregation reproduces a classical algorithm, Bellman-Ford shortest
paths, $d_v^{(k)} = \min_{u \in \mathcal{N}(v)} d_u^{(k-1)} + \text{cost}(u, v)$ (slides 18–19,
≈37:59–42:40).

## How it relates to the other architectures

- **A ConvNet is a GNN on a grid graph** (slide 13). Each pixel's column of channels is a node's
  attribute vector, and the filter's window is the node's neighbourhood. What changes on a general
  graph is that neighbourhoods vary in size and have no left or right; what stays is local
  operations, globalizing through depth, weight sharing, and inputs of any size (slide 12,
  ≈26:24–30:16). See [convolution](convolution.md).
- **An MLP is a GNN over a single node** (slide 23): with no neighbours, the update is a linear layer
  and a non-linearity, repeated (≈54:20–55:51). See [multilayer perceptrons](multilayer-perceptron.md).
- **Unrolled, a GNN is an MLP whose neurons are vectors** (slide 22): each round is a layer,
  AGGREGATE is "akin to a linear layer" and UPDATE "akin to a pointwise layer" (≈46:31–49:42).
  It is trained like any network, by backpropagating through the unrolled computation graph
  (≈53:31–54:20; see [backpropagation](backpropagation.md)).
- **A transformer is a GNN whose aggregation is attention**,
  $\mathbf{m}_ {\mathcal{N}(v)} = \sum_{u \in \mathcal{N}(v)} \alpha_{v,u} \mathbf{h}_ u$ with weights
  that depend on the node vectors (slide 24, ≈49:42 and ≈56:38). The course develops this in the
  transformers lecture, lecture 8 on the [course map](course-map.md).

## Weight sharing and graph size

Every node in a layer uses the same aggregate and update, so the parameters do not depend on the
graph's size or shape: a GNN can "generate encodings for previously unseen nodes & graphs" (slide 28),
as a ConvNet trained on small images runs on large ones. Functionally it handles any size;
statistically it "might only generalize to graphs of a slightly different size" than it was trained
on (≈1:01:19–1:02:04). Parameters may also be shared *across* layers, which lets the message passing
run for as many rounds as wanted, or kept separate per layer, which fixes the depth (≈51:13–51:58).

## Training

A data point is a graph with its adjacency structure and either a label per node or one label for
the graph. You specify the aggregate, update and readout functions and a loss, for example
cross-entropy on each node's prediction, and train with backpropagation and stochastic gradient
descent (slide 29, ≈1:02:04–1:04:24). The graph's connectivity is part of the data, not something
learned (≈59:48).

## What a GNN can and cannot distinguish

Because a node sees only its neighbourhood tree, two graphs whose nodes have the same trees look
identical to every GNN, and a GNN must give them the same output, however different they are
(slide 36, ≈1:07:28–1:09:47). Lecture 5's example is a pair of six-node graphs, one built from two
closed four-sided regions and one from two triangles, that no GNN can tell apart. So GNNs cannot
approximate a function that separates such graphs; conversely, slide 35's theorem says any function
that respects these equivalence classes can be approximated by message-passing GNNs. See
[representational power](representational-power.md).

The classes are those of a classical graph-isomorphism heuristic, the 1-dimensional
**Weisfeiler-Leman** test, or color refinement, which repeatedly recolours each node by hashing its
colour with its neighbours' (slide 37). "Any GNN can at best distinguish the same graphs as the
1-dim WL algorithm", and for any graph size some GNN matches it exactly (slides 36 and 39,
≈1:11:17–1:12:48). Matching it needs an aggregation that is injective on multisets, which the
sum-of-MLPs form $g_1\left(\sum g_2(\mathbf{h})\right)$ provides (slide 40). The choice matters in
practice: on lecture 5's slide 41, sum with an MLP fits a protein data set perfectly, sum with a
single linear-and-ReLU layer plateaus lower, and mean or max do worse, because the mean discards how
many neighbours a node has (≈1:14:23–1:18:15). Plain message passing also cannot compute, in
general, a graph's longest or shortest cycle, its diameter, or how often a motif occurs (slide 42).

The lecturer's view of how much this matters: indistinguishable graphs are "a worst-case thing",
and whether they occur in real data, such as a benign and a toxic molecule, is something he said he
does not know (≈1:13:36–1:14:23).

## Positional encodings on graphs

To distinguish more graphs, give each node an input saying where it sits in the graph: a
**positional encoding**. A one-hot node index does it but discards permutation invariance entirely
(≈1:09:47–1:10:32). Slide 43's example is the eigenvectors of the graph Laplacian
$\mathbf{L} = \mathbf{D} - \mathbf{A}$, where $\mathbf{D}$ is the diagonal matrix of node degrees,
"a generalized coordinate system for your location of a node within a graph" (≈1:19:50), with the
challenge that eigenvectors are ambiguous up to sign flips and repeated eigenvalues. It is a
trade-off: "now you might not generalize to new permutations, but you will be able to discriminate
things you couldn't discriminate before" (≈1:19:50–1:20:36), the same trade as positional encoding
in a ConvNet. See [neural fields and positional encoding](neural-fields-and-positional-encoding.md).

## Symmetry as generalization (lecture 6)

Lecture 6 returns to the graph net's symmetry as an example of how an architecture generalizes beyond its
data. Because a graph net is permutation invariant or equivariant, "I don't have to have seen every
permutation. I can just have seen some permutations, and then it will generalize to other permutations.
That's baked into the architecture. It's not something you have to learn from the data" (≈23:57). Its
slide 62 lists the equivariance among the architectural symmetries the lecturer considers the most
important explanation of why deep nets generalize: "If I permute the labeling of the nodes, then I will
permute the predictions" (≈1:16:06). Its slide 63 cites a polypharmacy network, the one lecture 5 shows,
as an example of domain knowledge built into an architecture. See [inductive bias](inductive-bias.md).
