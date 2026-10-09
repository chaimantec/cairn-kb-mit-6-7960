# Metric learning

**Metric learning** learns a distance between data points from examples of which points belong together. Instead of
measuring similarity in the input space, where "looking at Euclidean distance between all the pixels in one image and
another might not be ideal", it learns "some metric, some representation that respects some desired properties": data
points that belong together end up close and data points that differ end up far apart, with "similarity information" as
the only supervision ([lecture 12](12-representation-learning-similarity-based.md), slide 11, ≈14:43). It is the
origin of the [contrastive learning](contrastive-learning.md) that most modern representation learners use. Covered so
far: lecture 12, slides 10–25, ≈11:37–30:25. Lecture 11 names "metrics" among the outputs of representation learning
(slide 34) and promises metric learning for the next lecture (≈1:35).

**Notation.** Data points are $\mathbf{x}_ 1, \ldots, \mathbf{x}_ n$; $\mathcal{S}$ is the set of similar pairs and
$\mathcal{D}$ the set of dissimilar ones; the learned map is $\mathbf{z} = \mathbf{W}\mathbf{x}$ (linear) or
$\mathbf{z} = f(\mathbf{x})$ (a neural network). In a triplet, $\mathbf{x}$ is the anchor, $\mathbf{x}^+$ a positive and
$\mathbf{x}^-$ a negative.

## Learning from relative judgements

The supervision is weak on purpose. It is "not exactly the distances between things or how similar or dissimilar they
are", only which pairs are in the same category and which are not (≈15:28). Lecture 12 argues this is how people learn
too. Describing an elephant from scratch leaves "almost an infinite set of things" that fit; describing it as a rhinoceros
with a long trunk and big ears works, "because I understand similarity and then these small contrastive differences"
(≈12:23–13:55). And a judgement of relative similarity is easy even when the category is not: of three white moths, it is
hard to say whether two are the same kind, but "very easy, regardless of what you want your downstream task and the
dimension of granularity to be to say those two are definitely more similar than that one" (slide 17, ≈23:19).

## The linear case: a Mahalanobis distance

With a linear map $\mathbf{z} = \mathbf{W}\mathbf{x}$, Euclidean distance between representations is

$$\Vert \mathbf{z}_ i - \mathbf{z}_ j \Vert^2 = (\mathbf{x}_ i - \mathbf{x}_ j)^\top \mathbf{W}^\top \mathbf{W} (\mathbf{x}_ i - \mathbf{x}_ j),$$

a **Mahalanobis distance** $d_{\mathbf{A}}(\mathbf{x}_ i, \mathbf{x}_ j) = \Vert \mathbf{x}_ i - \mathbf{x}_ j \Vert_ {\mathbf{A}}$ with
the positive semidefinite matrix $\mathbf{A} = \mathbf{W}^\top \mathbf{W}$ (slide 12, ≈16:14–17:00). Learning the metric
is learning $\mathbf{A}$.

Xing, Ng, Jordan and Russell (2003), the paper that "introduced the term and problem" as "distance metric learning", pose
it as a constrained optimization (slide 13, ≈17:51):

$$\min_{\mathbf{A} \succeq 0} \sum_{(i,j) \in \mathcal{S}} d_{\mathbf{A}}(\mathbf{x}_ i, \mathbf{x}_ j)^2 \quad \text{s.t.} \quad \sum_{(k,\ell) \in \mathcal{D}} d_{\mathbf{A}}(\mathbf{x}_ k, \mathbf{x}_ \ell)^2 \geq 1$$

Similar points are brought "as close together as possible, while enforcing that there is at least some margin" from
dissimilar ones (≈18:36). The objective and constraint can be swapped, and follow-ups such as information-theoretic metric
learning (Davis et al. 2007) preserve "distribution information (relative entropy between Gaussians)" under similar bounds
(slide 13). In the paper's own example (slide 14), each of two classes is split into two clusters lying side by side with
the other class's; the learned projection simply removes a dimension, and the classes become separable (≈19:24–20:11).

## Deep metric learning

Three developments followed (slide 15): nonlinear transformations, through kernels or neural networks; contrastive losses;
and normalizing representations so that similarity is an angle rather than a distance. **Deep metric learning** learns
$\mathbf{z} = f(\mathbf{x})$ with a neural network $f$, so it "optimize[s] not over psd matrices but weights of a neural
network", by stochastic gradient descent (slide 16, ≈21:45). Normalizing onto a hypersphere means "distance would be
essentially just equivalent to angle", measured by inner products, and removes "the scale complexity" (≈20:57).

## The triplet loss and its descendants

The loss for deep metric learning "explicitly says that the distance between dissimilar pairs must be much larger than the
distance between similar pairs" (≈23:19). The **triplet loss** (Schroff et al. 2015) penalizes any triplet in which the
negative is not farther from the anchor than the positive by a margin $\epsilon$ (slide 18):

$$\mathcal{L}_ {\text{triplet}}(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-) = \sum_{\mathbf{x} \in \mathcal{X}} \max\left(0, \thinspace \Vert f(\mathbf{x}) - f(\mathbf{x}^+) \Vert_ 2^2 - \Vert f(\mathbf{x}) - f(\mathbf{x}^-) \Vert_ 2^2 + \epsilon\right)$$

A **triplet network** runs anchor, positive and negative through one network with shared weights and back-propagates the
loss (slide 20, ≈24:54). The **lifted structured loss** (Song et al. 2015) compares each positive pair with all negatives
in a batch, which makes "the best use of the batch you've already loaded in" (slide 21, ≈25:44). Large-margin Nearest
Neighbor metric learning (LMNN; Weinberger et al. 2009) is a related method (slide 18).

Trained on birds, such an embedding groups photographs by kind (slides 22–23) but also by setting: terns flying against
blue sky cluster together, so "similarity here has may be picked up on some more stuff than just maybe what we might be
interested in" (≈27:18). People, too, judge images similar in many ways, by pose, perspective, foreground colour, number
of items or object shape (slide 24), and "what's the most useful way ... is probably going to be contextualized by some
downstream task" (≈28:53).

## Hard negatives

A triplet the model already gets right, with the margin satisfied, gives no gradient: "the easy stuff doesn't actually help
you train" (≈26:30). So training looks for **hard negatives**, "currently 'misplaced', i.e., closer to anchor than a
positive example" (slide 25). With anchor $a$, positive $p$, negative $n$ and distance $d$, slide 25 separates hard
negatives, $d(a, n) \lt d(a, p)$; semi-hard ones, $d(a, p) \lt d(a, n) \lt d(a, p) + \mathit{margin}$; and easy ones,
$d(a, p) + \mathit{margin} \lt d(a, n)$. Hard negatives are where the confounds live: moths that match in pose and
background but differ in type teach the model what to be invariant to (≈29:40–30:25).

## Where it leads

All of this assumes labelled pairs. [Contrastive learning](contrastive-learning.md) keeps the losses and makes the pairs
without labels, by augmentation. For open-set problems such as re-identifying people never seen in training, "almost all the
best methods" learn metric spaces with triplet and contrastive losses, while a cross-entropy loss over a fixed set of
categories says "almost nothing" about new ones (≈1:12:41–1:13:28).

## See also

- [Lecture 12 — Representation Learning: Similarity-Based](12-representation-learning-similarity-based.md), where this
  material comes from.
- [Contrastive learning](contrastive-learning.md) — the self-supervised descendant.
- [Representation learning](representation-learning.md) — what makes a representation good.
- [Norms](norms.md) — the vector norms these distances are built from.
