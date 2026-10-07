# MIT 6.7960 — Deep Learning, Fall 2024

A graduate course in deep learning from MIT, taught by **Phillip Isola, Sara Beery and Jeremy
Bernstein** and published on [MIT OpenCourseWare](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/).
It treats deep learning as neural networks plus differentiable programming, and covers how
networks are trained, what they can approximate, the main architectures (grids, graphs,
transformers, memory), generalization, representation learning, generative models, transfer
learning, and scaling. It is explicitly "not an intro to deep learning class"; it assumes
gradient descent, MLPs, softmax and cross-entropy, and tensors as background.

> **Coverage: lectures 1–5 of 24 only.** This knowledge base currently holds the first five
> lectures (Introduction to Deep Learning; How to Train a Neural Net; Approximation Theory;
> Architectures: Grids; Architectures: Graphs) and the concept pages they support. For anything
> taught in lectures 6–24, it can tell you *which* lecture covers it — see the
> [course map](wiki/course-map.md) — but not *what* that lecture says. Do not cite it as the course
> beyond lecture 5. Build progress is in
> [TODO.md](TODO.md).

## Lecture pages

- [Lecture 1 — Introduction to Deep Learning](wiki/01-introduction.md) — Sara Beery. The
  definition of deep learning (neural nets plus differentiable programming); the history of
  neural networks as a curve of enthusiasm (perceptron 1958, Minsky and Papert 1972, PDP and
  backprop 1986, the AI winter, AlexNet 2012, the 28-year cycles and the "2028?" question); what
  deep learning is today; a review of the assumed background (gradient descent, the linear layer
  and perceptron, tanh/sigmoid/ReLU, stacking layers, softmax and cross-entropy, batching); and a
  preview of the course. Includes the student Q&A on choosing activations, capacity, data size,
  and width versus depth.
- [Lecture 2 — How to Train a Neural Net](wiki/02-how-to-train-a-neural-net.md) — Sara Beery.
  Gradient descent reviewed (black-box, first- and second-order; the update rule), stochastic
  gradient descent and batch noise, momentum; six toy loss landscapes (convex, discontinuous,
  vanishing, zero and exploding gradients, local minima) with evolution strategies and gradient
  clipping as fixes; continuous, differentiable and smooth (ReLU versus GELU); computation graphs
  and DAGs; the matrix-calculus shapes and chain rule; backpropagation through a generic layer, a
  linear layer, a ReLU, a whole MLP and any DAG (merge, branch, parameter sharing); differentiable
  programming (Software 2.0, Neural Module Networks); optimizing inputs instead of weights (unit
  visualization, DeepDream, CLIP+GAN); and the deck's worked one-iteration backprop example, with
  a misprint on slide 79 flagged.
- [Lecture 3 — Approximation Theory](wiki/03-approximation-theory.md) — Jeremy Bernstein,
  handwritten deck. Would you rather scale width or depth? The approximation–optimization–
  generalization puzzle; formalizing approximation ($L_\infty$ and $L_1$ error); Lipschitz
  functions and the RMS norm; a full proof that a 3-layer ReLU network with $4d(L/\epsilon)^d$
  units approximates any $L$-Lipschitz function on the hypercube (rectangles, hyperrectangles, four
  ReLUs per rectangle, thresholding), the live graphing demo, and why the construction would not
  generalize; Barron, Hornik et al. and Stone–Weierstrass; whether universal approximation matters;
  arguments for width; a depth separation by counting kinks ($(2n)^L$ bound, the triangle map,
  $7 \times 10^{49}$ units for a 3-layer match); Kaplan et al.'s scaling laws and Chinchilla as
  confounders; and a preview of inductive biases.
- [Lecture 4 — Architectures: Grids](wiki/04-architectures-grids.md) — Sara Beery. Why build
  better architectures: the MLP's pros and cons, the hypothesis-space picture (more data or a more
  constrained architecture), and ReLU-net, exact-model and sine-net (SIREN) fits of a 1D function;
  convolutional networks from classifying overlapping patches to semantic segmentation; translation
  equivariance; the convolution formula (and why it is really cross-correlation); fully connected,
  locally connected and weight-shared layers; the Toeplitz matrix; five views on convolutional
  layers; stacking and receptive fields; multichannel inputs and outputs, filter banks, feature maps
  and the 27-parameter quiz; max and mean pooling; downsampling, strides and dilated filters;
  learned versus hand-crafted filters; AlexNet, VGG16 and ResNet18 feature maps; `im2col`; the
  architecture zoo (encoder–decoder, U-net, ResNet's residual connection); convolution in time and
  3D convolution over video; positional encoding; and neural fields (SIREN, NeRF). 35 of its slides
  are excluded from OCW's licence and described in prose only.
- [Lecture 5 — Architectures: Graphs](wiki/05-architectures-graphs.md) — Phillip Isola. Graph
  neural networks: learning tasks on graphs (node classification, link prediction, molecules,
  polypharmacy, Google Maps, physics simulation, combinatorial optimization); node and graph
  embeddings; why an MLP on the adjacency matrix fails, and permutation invariance and equivariance;
  a CNN as a GNN over a grid graph; message passing with AGGREGATE and UPDATE; sum, mean,
  normalized and max aggregation, Bellman-Ford as min-aggregation, and the universal
  $\text{MLP}_ 2(\sum \text{MLP}_ 1)$ aggregator; readout; GNNs unrolled as an MLP of vectors, an MLP
  as a one-node GNN, and attention as the aggregation that makes a transformer; the tree view and
  weight sharing; training; what GNNs can distinguish (equivalence classes, the 1-dim
  Weisfeiler-Leman bound, injective aggregation, sum versus mean on PROTEINS, cycles and diameter);
  and Laplacian-eigenvector positional encodings. 14 of its slides are excluded from OCW's licence.

## Course pages

- [Course map](wiki/course-map.md) — the schedule of recorded lectures, with what is in this KB.
  Also the table translating lecture 1's "Lecture N" banners into the recorded lecture numbers
  (several differ — e.g. transformers are lecture 8, not 9). Covers grading (65% problem sets,
  35% blog-post final project), compute, PyTorch, the collaboration rules and the AI-assistant
  policy; and which problem set goes with which lecture where a lecture says (lecture 5's
  graph-network questions are Homework 2's).
- [Course notation](wiki/notation.md) — the course's Math Notation handout: bold for
  vectors/matrices/tensors, $L$ versus $J$, $\mathbf{z}$ (pre-activation) versus $\mathbf{h}$
  (post-activation), channels-first tensors, probability notation, and the matrix-calculus
  conventions (row-vector gradients, transposed $\partial L / \partial \mathbf{W}$).

## Concept pages

Each explains one idea in full, from the course material so far, and links back to the lecture
passages it draws on.

- [Multilayer perceptrons](wiki/multilayer-perceptron.md) — the linear layer
  $z_j = \mathbf{x}^T \mathbf{w}_ j + b_j$, the perceptron, why one layer is a linear classifier
  and fails on XOR, stacking layers into matrix form, why the non-linearity is essential, and the
  "two ramps make a pyramid" picture of non-linear classification; the MLP as a computation graph
  and how it is backpropagated (lecture 2); what "three-layer ReLU network" means, and ReLU MLPs as
  piecewise linear functions (lecture 3); the MLP weighed as an architecture, and the fully
  connected layer that a convolution constrains (lecture 4); an MLP on an adjacency matrix is not
  permutation invariant, and an MLP is a graph net over a single node (lecture 5).
- [Activation functions](wiki/activation-functions.md) — step, tanh, sigmoid and ReLU compared:
  ranges, saturation and vanishing gradients, dead ReLUs, the $6\times$ AlexNet speed-up, the
  sigmoid typo on lecture 1's slide 40, and how (not) to choose one; GELU and the continuous,
  differentiable and smooth criterion (lecture 2); the ReLU as a gate on the backward pass;
  what ReLUs can build: rectangles from four ReLUs, thresholded sums, and kink doubling (lecture 3);
  sine activations (SIREN) as an inductive bias for periodic functions and images (lecture 4).
- [Gradient descent](wiki/gradient-descent.md) — the training objective
  $\theta^{\ast} = \arg\min_\theta \sum_i L$, the cost $J(\theta)$, the update rule and learning
  rate, why differentiability matters, black-box versus first- and second-order optimization,
  stochastic gradient descent and batch size, momentum (and Adam), the plus-sign update convention,
  and when to stop.
- [Backpropagation](wiki/backpropagation.md) — computation graphs, the shape rules and chain
  rule, the "compute shared terms once" trick, the per-layer arrays $\mathbf{L}$ and $\mathbf{g}$
  and the recurrence $\mathbf{g}_ {\texttt{in}} = \mathbf{g}_ {\texttt{out}} \mathbf{L}^{\mathbf{x}}$,
  the linear layer's three products, the ReLU as a gating matrix, why the backward pass is linear,
  memory, merge and branch rules for DAGs, parameter sharing, and the worked example's numbers;
  graph neural networks trained by backpropagating through unrolled message passing (lecture 5).
- [Loss landscapes](wiki/loss-landscapes.md) — differentiable versus "has a PyTorch gradient"
  versus easy to optimize; the six toy cases (convex, discontinuous, vanishing, zero and exploding
  gradient, local minima) and what gradient descent does on each; random seeds; evolution
  strategies and gradient clipping; continuous, differentiable and smooth as a design criterion.
- [Differentiable programming](wiki/differentiable-programming.md) — programs as computation
  graphs, the LeCun and Dietterich posts, human-programmed versus backprop-programmed parts
  (Neural Module Networks, Software 2.0, feature engineering), what PyTorch needs from an operation,
  and optimizing inputs: unit visualization, DeepDream and CLIP+GAN; learned versus hand-crafted
  filters, and using an encoder or decoder on its own (lecture 4).
- [Softmax and cross-entropy](wiki/softmax-and-cross-entropy.md) — argmax readout, one-hot
  labels, $H(y, \hat{y}) = -\sum_k y_k \log \hat{y}_ {k}$, the "how much better you could have done"
  reading of the loss, the clown fish / grizzly / chameleon examples, and the "scores, not
  probabilities" caution; why to optimize toward a class through its logits, not its softmax
  probability (lecture 2).
- [Tensors and batching](wiki/tensors-and-batching.md) — why losses are computed in parallel,
  each layer as a features-by-examples representation, the network as batched matrix products,
  why GPUs mattered, and the course's tensor index conventions; batches in stochastic gradient
  descent and why the batch gradient is the average of per-example gradients (lecture 2);
  channels, filter-bank shapes, `im2col` and video as a 4D input (lecture 4).
- [Representational power](wiki/representational-power.md) — what networks can approximate,
  across lectures 1 and 3: one layer gives a linear surface; the Riemann-sum intuition; the
  formal question (families $G$ and $F$, error $\epsilon$); lecture 3's universal approximation
  theorem and proof sketch, with its weaknesses; Barron and Hornik et al.; whether universal
  approximation is sufficient or necessary; width versus depth, from lecture 1's discussion to
  lecture 3's depth separation and minimum-width result; inductive biases, and lecture 4's preview
  that better architectures approximate important function classes more efficiently; graph neural
  networks as deliberately non-universal, universal within multiset functions, and bounded by the
  Weisfeiler-Leman test (lecture 5).
- [Lipschitz continuity](wiki/lipschitz-continuity.md) — $|g(x + \Delta x) - g(x)| \le L |\Delta x|$,
  the bounded-slope intuition and the "bow tie" picture, the multi-input version with the RMS norm
  (and how it differs from the Euclidean norm), and how Lipschitzness bounds approximation error
  in lecture 3's proof.
- [Scaling laws](wiki/scaling-laws.md) — as lecture 3 presents them: Kaplan, McCandlish et al.
  (2020)'s power laws in compute, data and parameters, their finding that width versus depth
  barely matters at fixed parameter count, and Chinchilla as an example of confounders. Previews
  lecture 20.
- [Generalization and double descent](wiki/generalization-and-double-descent.md) — why
  over-parameterized nets don't just memorize, the classical U-curve against double descent
  (Belkin et al., 2019), the interpolation threshold, capacity versus data, and the simplicity
  hypothesis; previewing lectures 6 and 17. Lecture 3's rectangle network as a model that fits
  but would not generalize; lecture 4's case that architecture lets a model generalize with less
  data and outside the training distribution.
- [Representation learning](wiki/representation-learning.md) — compact, compositional
  representations (the letter-T example), the early-to-late feature hierarchy in brains and
  networks, reuse/transfer of lower layers, what an embedding is, and visualizing what a unit
  responds to (lecture 2); convolutional feature maps and how they change with depth, and the
  encoder–decoder (lecture 4); previewing lectures 11–13 and 18–19.
- [Inductive bias](wiki/inductive-bias.md) — the structure an architecture assumes before seeing
  data: why an MLP is data hungry, the hypothesis-space picture (more data or a more constrained
  architecture), how a bias decides what a model does outside its training data (ReLU-net, exact
  model and sine-net fits), translation equivariance as the convolutional bias, positional encoding
  as removing it, and hand-crafted versus learned structure (lectures 3 and 4); permutation
  invariance and equivariance as the graph bias, and "universality is actually not what we're after
  in architecture design" (lecture 5).
- [Convolution](wiki/convolution.md) — the convolutional layer in full (lecture 4): from classifying
  overlapping patches to the formula, cross-correlation and the $\star$ notation, locality, weight
  sharing and translation equivariance, the Toeplitz-matrix view, fewer parameters and any input
  size, the five views, stacking and receptive fields, channels and filter banks with the parameter
  count rule, max and mean pooling, downsampling, strides and dilation, `im2col`, and convolution in
  time and over video; a ConvNet as a graph net over a grid (lecture 5).
- [Skip connections](wiki/skip-connections.md) — what an encoder–decoder's bottleneck loses, U-net's
  skip connections across the "U", and ResNet's residual connection
  $\mathbf{x}_ {\text{out}} = F(\mathbf{x}_ {\text{in}}) + \mathbf{x}_ {\text{in}}$, including
  learning its own depth (lecture 4); previews transformers.
- [Neural fields and positional encoding](wiki/neural-fields-and-positional-encoding.md) — why and
  how to break shift invariance with a constructed positional encoding, neural fields as networks
  from coordinates to values, SIREN (sine activations) and NeRF (5D position and direction to colour
  and density), with what NeRF is not and its limits (lecture 4); previews transformers. Positional
  encodings on graphs: one-hot node indices and Laplacian eigenvectors, and what they cost in
  invariance (lecture 5).
- [Graph neural networks](wiki/graph-neural-networks.md) — lecture 5's architecture in full:
  graph tasks, permutation invariance and equivariance, message passing (AGGREGATE, UPDATE,
  READOUT), multiset aggregations and the universal sum-of-MLPs form, Bellman-Ford, how GNNs relate
  to ConvNets, MLPs and transformers, weight sharing and graph size, training, the
  neighbourhood-tree and Weisfeiler-Leman limits, and positional encodings.

## Raw materials

- [`raw/transcripts/`](raw/transcripts/) — lecture transcripts in ~45-second paragraphs, each
  opening with an `[MM:SS]` timestamp into the video. **Read the files directly in this folder**:
  they are MIT OpenCourseWare's human-made captions with a light, fully listed copy-edit. The
  unedited caption text is kept in [`raw/transcripts/original/`](raw/transcripts/original/) for
  reference.
- [`raw/slides/`](raw/slides/) — every slide of each lecture deck as text, headed `## Slide N`
  (slide N is PDF page N), with equations in LaTeX and every figure described in prose. Slides
  whose figures OCW excludes from its licence carry an `*OCW notice*` line.
- [`raw/images/`](raw/images/) — whole-slide renders of figure slides, embedded in the slide file
  and in the wiki passage that cites them. Lectures 1–5 only; see [AGENTS.md](AGENTS.md#images)
  for which slides have images and which deliberately do not.
- [`sources.md`](sources.md) — every course document on OCW (slide decks, problem sets, the
  notation handout) with its canonical URL. The PDFs are not committed; cite those URLs.
- [`SEE_ALSO.md`](SEE_ALSO.md) — sibling knowledge bases that cover the same ground from another
  course.
- [`LICENSE.md`](LICENSE.md) — CC BY-NC-SA 4.0, following OCW, with the third-party exclusions.
