# MIT 6.7960 — Deep Learning, Fall 2024

A graduate course in deep learning from MIT, taught by **Phillip Isola, Sara Beery and Jeremy
Bernstein** and published on [MIT OpenCourseWare](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/).
It treats deep learning as neural networks plus differentiable programming, and covers how
networks are trained, what they can approximate, the main architectures (grids, graphs,
transformers, memory), generalization, representation learning, generative models, transfer
learning, and scaling. It is explicitly "not an intro to deep learning class"; it assumes
gradient descent, MLPs, softmax and cross-entropy, and tensors as background.

> **Coverage: lectures 1–8 of 24 only.** This knowledge base currently holds the first eight
> lectures (Introduction to Deep Learning; How to Train a Neural Net; Approximation Theory;
> Architectures: Grids; Architectures: Graphs; Generalization Theory; Scaling Rules for
> Optimization; Architectures: Transformers) and the concept pages they support. For anything taught in lectures 9–24, it can
> tell you *which* lecture covers it — see the [course map](wiki/course-map.md) — but not *what*
> that lecture says. Do not cite it as the course beyond lecture 8. Build progress is in
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
- [Lecture 6 — Generalization Theory](wiki/06-generalization-theory.md) — Phillip Isola. Why do
  neural networks generalize? Empirical versus population risk; bad data and bad models (the
  "filing cabinet" that memorizes, Paul the octopus); memorization versus generalization (a filing
  cabinet and a ReLU MLP fit the same points); a counting experiment showing GPT-4o is no filing
  cabinet; pix2pix and edges2cats generalizing to sketches with three and eight eyes, and the
  ConvNet's compositional bias; Occam's razor and the shortest program (Solomonoff); bias and
  variance; polynomial fits of degree 1, 3, 20 and 1000, the simple + spiky hypothesis and double
  descent (Belkin et al.), with when to stop training; why parameter count, parameter norm and the
  number of functions all fail as complexity measures; Vapnik-Chervonenkis theory, dichotomies and
  the $\sqrt{d / n}$ bound, made vacuous by networks that fit random labels (Zhang et al.'s CIFAR10
  table); the version space; candidate inductive biases — simplicity bias of the parameter-function
  map, the low-rank bias of depth, implicit regularization by optimizers (weight decay,
  initialization, flat minima), architectural symmetries, domain constraints; and Ilya Sutskever's
  "anything finite will look small". 8 of its slides are excluded from OCW's licence.
- [Lecture 7 — Scaling Rules for Optimization](wiki/07-scaling-rules-for-optimization.md) — Jeremy
  Bernstein, handwritten deck. The optimization piece of the puzzle; the loss as an average of an
  error measure composed with a network; size, depth and noise as what makes it hard, and full-batch
  optimization; the scaling woes (the optimal learning rate drifts with width, deeper performs worse);
  classical methods from a Taylor expansion — Newton's method ($-\mathbf{H}^{-1}\mathbf{g}$, too big,
  may find a maximum), the Gauss-Newton decomposition into curvature of the error and of the model and
  the Gauss-Newton method, why backpropagation never forms $\partial f / \partial \mathbf{w}$, and
  steepest descent (Euclidean norm gives gradient descent, infinity norm gives sign gradient descent,
  the dual-norm formula); why a non-Euclidean norm (the squeezed map); "in which norm?"; the neural,
  tensor and spectral perspectives; the spectral norm and the RMS-RMS operator norm; the width rule
  (RMS-RMS norm about 1 at initialization and for every update); depth and the $1/L$ versus
  $1/\sqrt{L}$ residual multiplier; the lecturer's modular theory (modules with a norm); references;
  and what problem set 2 asks. 3 of its slides are excluded from OCW's licence.
- [Lecture 8 — Architectures: Transformers](wiki/08-architectures-transformers.md) — Phillip Isola.
  Transformers as three ideas, only one of them new: tokens, attention and positional encoding;
  "everything old is new again" (Pierre Menard's *Don Quixote*); the locality limit of CNNs (receptive
  fields that grow slowly, far-apart patches that never interact) against the fully connected layer's
  $n^2$ parameters; tokens as vectors of neurons, arrays and sets of tokens, and tokenizing images
  (patches), text (byte pairs) and audio; the $N \times d$ token matrix; token nets (linear combinations
  of tokens and token-wise MLPs) as MLPs over vectors and as graph nets over fully connected graphs;
  attention as weights computed from the data, the animal-counting and impala-colour intuitions,
  query-key-value attention with its database analogy (and the typo on slide 29), self-attention and
  DINO's attention maps; the self-attention layer
  $\text{softmax}(\mathbf{Q} \mathbf{K}^{\mathsf{T}} / \sqrt{m}) \mathbf{V}$ and why $\mathbf{A}$ is not
  learned directly; fc, conv and attention as a family of linear layers; the vanilla transformer,
  multihead self-attention, the ViT block with token norm (layer norm) and residual connections, and its
  pseudocode; permutation equivariance and positional encodings (Fourier codes, ScaleMAE, spherical
  harmonics, Laplacian eigenvectors); autoregressive models, GPT and causal masking; "Attention Is All
  You Need" read against the lecture; cross-attention for image captioning; and Homework 3. 7 of its
  slides are excluded from OCW's licence.

## Course pages

- [Course map](wiki/course-map.md) — the schedule of recorded lectures, with what is in this KB.
  Also the table translating lecture 1's "Lecture N" banners into the recorded lecture numbers
  (several differ — e.g. transformers are lecture 8, not 9). Covers grading (65% problem sets,
  35% blog-post final project), compute, PyTorch, the collaboration rules and the AI-assistant
  policy; and which problem set goes with which lecture where a lecture says (lecture 5's
  graph-network questions and lecture 7's steepest-descent and hyperparameter-transfer questions are
  Homework 2's, which went out at lecture 6; lecture 8's transformer and GPT implementation is Homework 3).
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
  permutation invariant, and an MLP is a graph net over a single node (lecture 5); how an MLP
  interpolates between training points where a memorizing "filing cabinet" cannot, and its last
  layer as regression on features (lecture 6); the neural, tensor and spectral perspectives on a
  network (lecture 7); token nets as MLPs over vectors, and the token-wise MLP inside a transformer
  (lecture 8).
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
  and when to stop ("just train forever", lecture 6); what the optimizer prefers — weight decay,
  initialization near zero and flat minima (lecture 6); gradient descent derived as steepest descent
  in the Euclidean norm, sign gradient descent, and full-batch optimization (lecture 7).
- [Steepest descent](wiki/steepest-descent.md) — lecture 7's first-order method: replace the
  non-linear part of the Taylor expansion with $\frac{\lambda}{2} \Vert \Delta \mathbf{w} \Vert^2$
  and minimize; the Euclidean norm gives gradient descent ($-\mathbf{g}/\lambda$), the infinity norm
  sign gradient descent, any norm a step size $\Vert \mathbf{g} \Vert^{\dagger} / \lambda$ times a
  step direction (the dual norm); steepest descent for matrices and the spectral norm in problem set 2;
  no guarantee unless the model is an upper bound; and why another norm — the squeezed map,
  preconditioning, where the linear term breaks down.
- [Second-order methods](wiki/second-order-methods.md) — first- versus second-order (lecture 2); the
  Taylor expansion, linearization and non-linear part; Newton's method $-\mathbf{H}^{-1}\mathbf{g}$ and
  its problems (a $d \times d$ Hessian, heading for a maximum, cubic regularization); the Gauss-Newton
  decomposition into curvature of the error and of the model, the Gauss-Newton method and its
  problems; and why the extra derivatives and the matrix inversion are expensive (lecture 7).
- [Norms](wiki/norms.md) — "in which norm?": Euclidean, RMS, infinity, $\ell_1$, $\ell_p$ and weighted
  norms (and why KL is not one); dual norms; Frobenius, spectral, nuclear and Schatten $p$-norms;
  induced operator norms, the spectral norm and the RMS-RMS operator norm
  ($\sqrt{d_{\text{in}}/d_{\text{out}}}$ times the spectral norm); parameter norm as a complexity
  measure (lecture 6); composing norms across modules (lecture 7). Spans lectures 3, 6 and 7.
- [Backpropagation](wiki/backpropagation.md) — computation graphs, the shape rules and chain
  rule, the "compute shared terms once" trick, the per-layer arrays $\mathbf{L}$ and $\mathbf{g}$
  and the recurrence $\mathbf{g}_ {\texttt{in}} = \mathbf{g}_ {\texttt{out}} \mathbf{L}^{\mathbf{x}}$,
  the linear layer's three products, the ReLU as a gating matrix, why the backward pass is linear,
  memory, merge and branch rules for DAGs, parameter sharing, and the worked example's numbers;
  graph neural networks trained by backpropagating through unrolled message passing (lecture 5); why
  backpropagation never forms the network's output Jacobian $\partial f / \partial \mathbf{w}$, and
  forward mode (lecture 7).
- [Loss landscapes](wiki/loss-landscapes.md) — differentiable versus "has a PyTorch gradient"
  versus easy to optimize; the six toy cases (convex, discontinuous, vanishing, zero and exploding
  gradient, local minima) and what gradient descent does on each; random seeds; evolution
  strategies and gradient clipping; continuous, differentiable and smooth as a design criterion;
  why fixed-step gradient descent finds flat minima, argued to generalize better (lecture 6);
  linearization and non-linear part, Newton's method heading for a maximum, a non-isotropic weight
  space, and the loss-versus-learning-rate curves that drift with width (lecture 7).
- [Differentiable programming](wiki/differentiable-programming.md) — programs as computation
  graphs, the LeCun and Dietterich posts, human-programmed versus backprop-programmed parts
  (Neural Module Networks, Software 2.0, feature engineering), what PyTorch needs from an operation,
  and optimizing inputs: unit visualization, DeepDream and CLIP+GAN; learned versus hand-crafted
  filters, and using an encoder or decoder on its own (lecture 4); modules that carry a norm as well
  as a forward and a backward, in the lecturer's modular theory (lecture 7).
- [Softmax and cross-entropy](wiki/softmax-and-cross-entropy.md) — argmax readout, one-hot
  labels, $H(y, \hat{y}) = -\sum_k y_k \log \hat{y}_ {k}$, the "how much better you could have done"
  reading of the loss, the clown fish / grizzly / chameleon examples, and the "scores, not
  probabilities" caution; why to optimize toward a class through its logits, not its softmax
  probability (lecture 2); the softmax that normalizes attention scores, and next-word prediction as
  classification (lecture 8).
- [Tensors and batching](wiki/tensors-and-batching.md) — why losses are computed in parallel,
  each layer as a features-by-examples representation, the network as batched matrix products,
  why GPUs mattered, and the course's tensor index conventions; batches in stochastic gradient
  descent and why the batch gradient is the average of per-example gradients (lecture 2);
  channels, filter-bank shapes, `im2col` and video as a 4D input (lecture 4); a set of tokens as an
  $N \times d$ matrix, and a transformer layer as a handful of matrix products (lecture 8).
- [Representational power](wiki/representational-power.md) — what networks can approximate,
  across lectures 1 and 3: one layer gives a linear surface; the Riemann-sum intuition; the
  formal question (families $G$ and $F$, error $\epsilon$); lecture 3's universal approximation
  theorem and proof sketch, with its weaknesses; Barron and Hornik et al.; whether universal
  approximation is sufficient or necessary; width versus depth, from lecture 1's discussion to
  lecture 3's depth separation and minimum-width result; inductive biases, and lecture 4's preview
  that better architectures approximate important function classes more efficiently; graph neural
  networks as deliberately non-universal, universal within multiset functions, and bounded by the
  Weisfeiler-Leman test (lecture 5); approximation set beside generalization, and why parameter
  count is not the capacity that matters (lecture 6).
- [Lipschitz continuity](wiki/lipschitz-continuity.md) — $|g(x + \Delta x) - g(x)| \le L |\Delta x|$,
  the bounded-slope intuition and the "bow tie" picture, the multi-input version with the RMS norm
  (and how it differs from the Euclidean norm), and how Lipschitzness bounds approximation error
  in lecture 3's proof; the RMS norm reused for the RMS-RMS operator norm (lecture 7).
- [Scaling laws](wiki/scaling-laws.md) — as lecture 3 presents them: Kaplan, McCandlish et al.
  (2020)'s power laws in compute, data and parameters, their finding that width versus depth
  barely matters at fixed parameter count, and Chinchilla as an example of confounders. Previews
  lecture 20. Not the same as lecture 7's scaling *rules* (below).
- [Scaling rules](wiki/scaling-rules.md) — lecture 7's answer to "the optimal learning rate drifts"
  and "deeper performs worse": hyperparameter transfer, the Goldilocks update and "in which norm?",
  the width rule ($\Vert \mathbf{W}_ \ell \Vert_{\text{RMS-RMS}} \sim 1$ at initialization and
  $\Vert \Delta \mathbf{W}_ \ell \Vert_{\text{RMS-RMS}} \sim 1$ for updates) with why it works and
  what it does not guarantee, problem set 2's version for sign gradient descent, the $1/L$ residual
  multiplier and $(1 + x/L)^L \to e^x$ against the $1/\sqrt{L}$ of standard transformers, the
  modular theory, and the references.
- [Generalization and double descent](wiki/generalization-and-double-descent.md) — why
  over-parameterized nets don't just memorize, the classical U-curve against double descent
  (Belkin et al., 2019), the interpolation threshold, capacity versus data, and the simplicity
  hypothesis (lecture 1). Lecture 3's rectangle network as a model that fits but would not
  generalize; lecture 4's case that architecture lets a model generalize with less data and outside
  the training distribution. Lecture 6's full treatment: empirical and population risk,
  memorization versus generalization, double descent on polynomial fits and the simple + spiky
  hypothesis, why parameter count, norm and VC dimension fail (random labels make the VC bound
  vacuous), and the candidate inductive biases; previews lecture 17.
- [Representation learning](wiki/representation-learning.md) — compact, compositional
  representations (the letter-T example), the early-to-late feature hierarchy in brains and
  networks, reuse/transfer of lower layers, what an embedding is, and visualizing what a unit
  responds to (lecture 2); convolutional feature maps and how they change with depth, and the
  encoder–decoder (lecture 4); kernels of a network's output representation, and why deeper (even
  linear) networks give lower-rank, more clustered ones (lecture 6); tokens as representations at every
  layer, and DINO's attention maps that stay inside objects (lecture 8); previewing lectures 11–13 and
  18–19.
- [Inductive bias](wiki/inductive-bias.md) — the structure an architecture assumes before seeing
  data: why an MLP is data hungry, the hypothesis-space picture (more data or a more constrained
  architecture), how a bias decides what a model does outside its training data (ReLU-net, exact
  model and sine-net fits), translation equivariance as the convolutional bias, positional encoding
  as removing it, and hand-crafted versus learned structure (lectures 3 and 4); permutation
  invariance and equivariance as the graph bias, and "universality is actually not what we're after
  in architecture design" (lecture 5); why generalization requires inductive bias, the ConvNet's
  compositional bias (the eight-eyed cat), and invariances, equivariances, compositionality and
  domain constraints as the lecturer's favoured explanation of why deep nets generalize (lecture 6);
  the transformer's few built-in biases, with locality left to the tokenizer and domain knowledge to the
  positional encoding (lecture 8).
- [Convolution](wiki/convolution.md) — the convolutional layer in full (lecture 4): from classifying
  overlapping patches to the formula, cross-correlation and the $\star$ notation, locality, weight
  sharing and translation equivariance, the Toeplitz-matrix view, fewer parameters and any input
  size, the five views, stacking and receptive fields, channels and filter banks with the parameter
  count rule, max and mean pooling, downsampling, strides and dilation, `im2col`, and convolution in
  time and over video; a ConvNet as a graph net over a grid (lecture 5); patch-wise processing as a
  reason ConvNets generalize to new arrangements (lecture 6); locality as a limitation, the $1 \times 1$
  convolution as a token-wise MLP, and the Toeplitz matrix beside attention (lecture 8).
- [Skip connections](wiki/skip-connections.md) — what an encoder–decoder's bottleneck loses, U-net's
  skip connections across the "U", and ResNet's residual connection
  $\mathbf{x}_ {\text{out}} = F(\mathbf{x}_ {\text{in}}) + \mathbf{x}_ {\text{in}}$, including
  learning its own depth (lecture 4); previews transformers. How much each residual block should
  contribute, $1/L$ or $1/\sqrt{L}$ (lecture 7). The residual connections of the transformer block
  (lecture 8).
- [Neural fields and positional encoding](wiki/neural-fields-and-positional-encoding.md) — why and
  how to break shift invariance with a constructed positional encoding, neural fields as networks
  from coordinates to values, SIREN (sine activations) and NeRF (5D position and direction to colour
  and density), with what NeRF is not and its limits (lecture 4); previews transformers. Positional
  encodings on graphs: one-hot node indices and Laplacian eigenvectors, and what they cost in
  invariance (lecture 5). NeRF's built-in projection and light transport as a reason it generalizes
  to new viewpoints (lecture 6). Positional encodings for transformers: why permutation equivariance
  needs them, Fourier codes, ScaleMAE, spherical harmonics and Laplacian eigenvectors (lecture 8).
- [Graph neural networks](wiki/graph-neural-networks.md) — lecture 5's architecture in full:
  graph tasks, permutation invariance and equivariance, message passing (AGGREGATE, UPDATE,
  READOUT), multiset aggregations and the universal sum-of-MLPs form, Bellman-Ford, how GNNs relate
  to ConvNets, MLPs and transformers, weight sharing and graph size, training, the
  neighbourhood-tree and Weisfeiler-Leman limits, and positional encodings; permutation symmetry as
  a source of generalization (lecture 6); transformers as graph nets over fully connected graphs
  (lecture 8).
- [Transformers](wiki/transformers.md) — lecture 8's architecture in full: why attention (the limit of
  locality), tokens and tokenizing, token nets, attention as data-dependent weights, query-key-value
  attention and self-attention, why $\mathbf{A}$ is not learned directly, multihead attention, attention
  in the family of linear layers, the ViT block (token norm, residual connections), permutation
  equivariance and positional encodings, autoregressive models, GPT and causal masking, "Attention Is All
  You Need", cross-attention, and why transformers are everywhere.

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
  and in the wiki passage that cites them. Lectures 1–8 only; see [AGENTS.md](AGENTS.md#images)
  for which slides have images and which deliberately do not.
- [`sources.md`](sources.md) — every course document on OCW (slide decks, problem sets, the
  notation handout) with its canonical URL. The PDFs are not committed; cite those URLs.
- [`SEE_ALSO.md`](SEE_ALSO.md) — sibling knowledge bases that cover the same ground from another
  course.
- [`LICENSE.md`](LICENSE.md) — CC BY-NC-SA 4.0, following OCW, with the third-party exclusions.
