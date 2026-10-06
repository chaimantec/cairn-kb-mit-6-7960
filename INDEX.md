# MIT 6.7960 — Deep Learning, Fall 2024

A graduate course in deep learning from MIT, taught by **Phillip Isola, Sara Beery and Jeremy
Bernstein** and published on [MIT OpenCourseWare](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/).
It treats deep learning as neural networks plus differentiable programming, and covers how
networks are trained, what they can approximate, the main architectures (grids, graphs,
transformers, memory), generalization, representation learning, generative models, transfer
learning, and scaling. It is explicitly "not an intro to deep learning class"; it assumes
gradient descent, MLPs, softmax and cross-entropy, and tensors as background.

> **Coverage: lecture 1 of 24 only.** This knowledge base currently holds the first lecture
> (Introduction to Deep Learning) and the concept pages it supports. For anything taught in
> lectures 2–24, it can tell you *which* lecture covers it — see the
> [course map](wiki/course-map.md) — but not *what* that lecture says. Do not cite it as the
> course beyond lecture 1. Build progress is in [TODO.md](TODO.md).

## Lecture pages

- [Lecture 1 — Introduction to Deep Learning](wiki/01-introduction.md) — Sara Beery. The
  definition of deep learning (neural nets plus differentiable programming); the history of
  neural networks as a curve of enthusiasm (perceptron 1958, Minsky and Papert 1972, PDP and
  backprop 1986, the AI winter, AlexNet 2012, the 28-year cycles and the "2028?" question); what
  deep learning is today; a review of the assumed background (gradient descent, the linear layer
  and perceptron, tanh/sigmoid/ReLU, stacking layers, softmax and cross-entropy, batching); and a
  preview of the course. Includes the student Q&A on choosing activations, capacity, data size,
  and width versus depth.

## Course pages

- [Course map](wiki/course-map.md) — the schedule of recorded lectures, with what is in this KB.
  Also the table translating lecture 1's "Lecture N" banners into the recorded lecture numbers
  (several differ — e.g. transformers are lecture 8, not 9). Covers grading (65% problem sets,
  35% blog-post final project), compute, PyTorch, the collaboration rules and the AI-assistant
  policy.
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
  "two ramps make a pyramid" picture of non-linear classification.
- [Activation functions](wiki/activation-functions.md) — step, tanh, sigmoid and ReLU compared:
  ranges, saturation and vanishing gradients, dead ReLUs, the $6\times$ AlexNet speed-up, the
  sigmoid typo on slide 40, and how (not) to choose one.
- [Gradient descent](wiki/gradient-descent.md) — the training objective
  $\theta^{\ast} = \arg\min_\theta \sum_i L$, the cost $J(\theta)$, the update rule and learning
  rate, why differentiability matters, and what lecture 2 takes further.
- [Softmax and cross-entropy](wiki/softmax-and-cross-entropy.md) — argmax readout, one-hot
  labels, $H(y, \hat{y}) = -\sum_k y_k \log \hat{y}_ {k}$, the "how much better you could have done"
  reading of the loss, the clown fish / grizzly / chameleon examples, and the "scores, not
  probabilities" caution.
- [Tensors and batching](wiki/tensors-and-batching.md) — why losses are computed in parallel,
  each layer as a features-by-examples representation, the network as batched matrix products,
  why GPUs mattered, and the course's tensor index conventions.
- [Representational power](wiki/representational-power.md) — one layer gives a linear surface;
  two or more can represent any function (given a non-linearity); the Riemann-sum intuition; wide
  versus deep efficiency; previewing lecture 3.
- [Generalization and double descent](wiki/generalization-and-double-descent.md) — why
  over-parameterized nets don't just memorize, the classical U-curve against double descent
  (Belkin et al., 2019), the interpolation threshold, capacity versus data, and the simplicity
  hypothesis; previewing lectures 6 and 17.
- [Representation learning](wiki/representation-learning.md) — compact, compositional
  representations (the letter-T example), the early-to-late feature hierarchy in brains and
  networks, and reuse/transfer of lower layers; previewing lectures 11–13 and 18–19.

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
  and in the wiki passage that cites them. Lecture 1 only; see [AGENTS.md](AGENTS.md#images) for
  which slides have images and which deliberately do not.
- [`sources.md`](sources.md) — every course document on OCW (slide decks, problem sets, the
  notation handout) with its canonical URL. The PDFs are not committed; cite those URLs.
- [`SEE_ALSO.md`](SEE_ALSO.md) — sibling knowledge bases that cover the same ground from another
  course.
- [`LICENSE.md`](LICENSE.md) — CC BY-NC-SA 4.0, following OCW, with the third-party exclusions.
