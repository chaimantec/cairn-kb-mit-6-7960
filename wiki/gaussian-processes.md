# Gaussian processes, and the neural network–Gaussian process correspondence

A **Gaussian process** is a random function whose values on any finite set of inputs are jointly
Gaussian. It is one of two classical ways of building a space of functions to fit data, the other being
[kernel methods](kernel-methods.md): instead of choosing a function, you choose a distribution over
functions, keep the ones consistent with the data, and read off a mean prediction with an uncertainty
band. The course introduces it in [lecture 13](13-representation-learning-theory.md) for one reason: an
infinitely wide neural network with randomly sampled weights *is* a Gaussian process, and its covariance
function exposes the similarity judgements its architecture builds in. Covered so far: lecture 13, slides
10–26, ≈14:55–1:12:44.

**Notation.** An input $x$ is a point of an input space $\mathcal{X}$, which may be the real line, a
space of images or anything else. $f(x)$ is the random function's value at $x$.
$\mathcal{N}(\mu, \sigma^2)$ is a normal distribution with mean $\mu$ and variance $\sigma^2$. A covariance
matrix is $\boldsymbol{\Sigma}$, with entries $\Sigma_{ij}$; a covariance function is $\Sigma(x, x')$.

## Random functions consistent with the data

Suppose you want to fit some data without choosing a function class. One way is to sample random
functions from a stochastic process, throw away those that do not pass through the data and keep those
that do. What remains is a distribution of functions that agree on the data and vary away from it
([lecture 13](13-representation-learning-theory.md), slide 10, ≈14:55–15:43). In practice you do not sample
and reject: you condition the distribution on the training data and get "the posterior distribution of
functions". That computation becomes straightforward when the random functions are Gaussian, which is
what a Gaussian process is (≈15:43–16:30).

Lecture 13 gives three descriptions, from pictorial to formal.

**The picture.** Given data, a Gaussian process gives "a distribution of consistent functions" along
with "a formula for the mean and standard deviation of this distribution" (slide 13). The mean is
smoother than the random samples; the standard deviation grows away from the data and shrinks to zero on
it. The formula "just involves some matrices and some linear algebra", and its shape depends on the
**covariance structure**, which is "what we get to pick as machine learning people using Gaussian
processes" (≈22:55–24:25).

**A random vector, plotted.** Sample a Gaussian vector with mean 0 and covariance matrix
$\boldsymbol{\Sigma}$, plot each coordinate as a height along an axis, and connect the dots. With
independent coordinates the result is jagged. Choose $\boldsymbol{\Sigma}$ so that neighbouring entries are
correlated (slide 14 writes $\Sigma_{ij} \sim \exp - (i-j)^2$) and the result looks like a continuous
function. Let the dimension of the vector go to infinity, packing the points closer together, and you
have a Gaussian process: "if you forget everything else from the lecture, that's basically what a Gaussian
process is" (slide 14, ≈25:11–28:22).

**The definition.** Let $f(x)$ be a random variable for every $x \in \mathcal{X}$, an
"infinite-dimensional random vector indexed by $x$". If for every finite collection of inputs
$x_1, \ldots, x_n$ the vector $(f(x_1), \ldots, f(x_n))$ is Gaussian, then $f$ is a Gaussian process on
$\mathcal{X}$ (slide 15, ≈28:22–30:01). In practice one constructs something that satisfies the definition
analytically, rather than testing samples for Gaussianity; that such objects exist at all is the business
of what the lecturer believes is "called Kolmogorov extension theorem", which "you actually don't need to
know … to actually use this stuff" (≈30:01–31:39).

## Covariance functions encode "nearby"

A Gaussian process generalizes a covariance matrix $\Sigma_{ij}$, $i, j = 1, \ldots, n$, to a
**covariance function** $\Sigma(x, x')$ for $x, x' \in \mathcal{X}$: the covariance between the function's
values at two inputs (slide 16, ≈34:45–35:35). "Typically, we want a covariance function that is large for
"nearby" points and small otherwise" (slide 17). Sample many functions and scatter-plot their values at
$x$ and $x'$: for nearby inputs the cloud is a thin band along the diagonal, the two values strongly correlated;
for distant inputs it is round, the values uncorrelated (≈36:24–38:02). Lecture 13's two examples are the
squared exponential and the inner product (slide 17):

$$\Sigma(x, x') = e^{-(x - x')^2}, \qquad \Sigma(x, x') = \langle x, x' \rangle.$$

"Basically, the covariance structure is some kind of measure of similarity between the inputs"
(≈38:47). Covariance functions can be defined on "weird spaces", including discrete ones, which is "one of
the reasons people really like kernel methods as well" (≈39:36). They are also where the modelling
choices go: the squared exponential is often divided by $2\sigma^2$ in the exponent, with $\sigma$ a
length-scale hyperparameter, or given one $\sigma$ per input coordinate, and papers tabulate a dozen
kernels for different kinds of structure, "similar to, oh, use a ConvNet for images and use a transformer
for this" (≈1:02:31–1:04:52).

## Prediction by conditioning

To predict the value at a test input $x_{\ast}$ from known values $f(x_1), \ldots, f(x_n)$, stack them all
into one vector. By the definition it is Gaussian, so the conditional distribution of $f(x_{\ast})$ given
the rest is Gaussian too, $\mathcal{N}(\mu, \sigma^2)$, with "simple closed form formulae" for the mean
$\mu(x_{\ast})$ and standard deviation $\sigma(x_{\ast})$ (slide 18, ≈41:12–44:19). Sweeping $x_{\ast}$ along
the axis draws the mean curve and uncertainty band of the picture above. Lecture 13 does not write the
formulae out and suggests looking them up, so this page does not give them either.

The mean of the conditioned distribution is where Gaussian processes meet kernel methods: it "turns out to
be equivalent to the kernel interpolator of minimum kernel norm, for some definition of the RKHS norm"
(≈17:15–18:04). See [kernel methods](kernel-methods.md).

## The neural network–Gaussian process correspondence

Sample a network's weights $\mathbf{W}_ 1, \ldots, \mathbf{W}_ L$ at random and you get a random function;
resample and you get another (lecture 13, slide 20, ≈45:07–45:54). The **NN-GP correspondence** says
what this distribution is in the wide limit: if the weights are sampled iid, then as the width goes to
infinity, the joint distribution of any finite collection of outputs $f(x_1), \ldots, f(x_n)$ is
Gaussian, so the random network is a Gaussian process, and "the covariance function depends on the
architecture and non-linearity" (slide 23, ≈49:48–51:24).

Lecture 13's evidence is an experiment from the lecturer's own work: 1,000 random three-layer MLPs of
hidden width 1000, evaluated on a CIFAR-10 truck, the same truck with slight pixel noise, and the same
truck with heavy noise. The outputs on the two similar images lie along a thin diagonal line; on the
less similar pair they form a wider cloud. Pairs of outputs look jointly Gaussian, and their covariance
depends on how similar the inputs are (slide 22, ≈47:28–49:01).

The proof runs through the **multivariate central limit theorem**: first show that, for a fixed input and
layer, the activations are iid (by induction on depth); then write the output vector on $k$ inputs as a
sum over iid vectors from the penultimate layer and apply the theorem again (slide 24, ≈1:04:52–1:06:24).
The width has to go to infinity because a central limit theorem needs something to be large, "and it turns
out it's the width" (≈57:42). The lecturer calls his justification of the independence "hand-wavy" and
points to the readings for the details (≈1:07:10–1:07:55).

For a ReLU network the kernel is known exactly. With non-linearity
$\phi(x) = \sqrt{2} \thinspace \operatorname{relu}(x)$, weights sampled iid from $\mathcal{N}(0, 1/\text{fan-in})$, and inputs
$\mathbf{x}, \mathbf{x}' \in \mathbb{R}^d$, the output has mean zero and covariance given by the
**compositional arccosine kernel** (slide 25, ≈1:07:55–1:10:13):

$$\mathbb{E} f(\mathbf{x}) f(\mathbf{x}') = \underbrace{h \circ \cdots \circ h}_ {L - 1 \text{ times}} \left( \frac{\mathbf{x}^{\top} \mathbf{x}'}{d} \right),
\qquad h(t) = \frac{1}{\pi} \left[ \sqrt{1 - t^2} + t \left( \pi - \arccos t \right) \right].$$

### What it means, and what it did not change

The correspondence is the precise form of lecture 13's thesis, that "a neural architecture (even without
training) already expresses an opinion about data similarity" (slide 7): take the architecture infinitely
wide and its covariance function $\Sigma(x, x')$ is that opinion (≈51:24–52:12). See
[inductive bias](inductive-bias.md#the-architectures-opinion-about-similarity-lecture-13).

The hope was that the kernel could guide how architectures are designed and weights regularized (slide
26). "It seems to have just not really happened. Instead, we just do transformer" (≈1:11:57). Asked why
not use the equivalent kernel instead of the network, the lecturer suggests that the kernel has
hyperparameters of its own, and that kernel methods cost cubically in the training-set size while network
training scales linearly, calling the question "a somewhat open problem" (≈57:42–59:17). The correspondence
covers untrained networks only; the **neural tangent kernel** is "supposed to be a Gaussian process
characterization of trained networks", which the lecture mentions and does not develop (≈1:10:13).

## Where it appears

- [Lecture 13](13-representation-learning-theory.md) — the whole development: the three definitions,
  covariance functions, conditioning, the correspondence, its proof sketch and the ReLU example.

## See also

- [Kernel methods](kernel-methods.md) — the other classical function space, and how it relates.
- [Inductive bias](inductive-bias.md) — what an architecture assumes before it sees data.
- [Representation learning](representation-learning.md) — similarity between inputs, learned and built in.
- [Multilayer perceptrons](multilayer-perceptron.md) — the networks in every example.
