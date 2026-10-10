# Kernel methods

A **kernel method** fits data with a weighted sum of bump functions, one centred on each training input.
The bump, the **kernel**, says how much one input should influence the prediction at another, so choosing
it plays the role that choosing an architecture plays in deep learning. The course introduces kernel
methods in [lecture 13](13-representation-learning-theory.md), as one corner of a triangle of
correspondences between kernel methods, [Gaussian processes](gaussian-processes.md) and neural networks.
Covered so far: lecture 13, slides 9 and 11, ≈10:13–22:06 and ≈56:56–59:17.

**Notation.** $x$ is an input and $x_1, \ldots, x_n$ the training inputs. $k(x, x_i)$ is the kernel,
$\alpha_i$ a scalar weight, and $f$ the fitted function.

## A bump on every data point

To fit $n$ data points, place a bump on each input and rescale the bumps so that their sum passes through
the data: four points, four bumps, four coefficients, "everything checks out"
([lecture 13](13-representation-learning-theory.md), ≈10:58–11:47). The functions are

$$f(x) = \sum_{i=1}^{n} \alpha_i \thinspace k(x, x_i),$$

where $k(x, x_i)$ is a bump centred on $x_i$ and the $\alpha_i$ are weights (slide 9). The bump "could just
look like a little Gaussian"; $k(x, 0)$ is the bump at the origin, and the second argument translates it
to the $i$-th data point (≈10:58, ≈12:33). The lecturer relates the construction to
[lecture 3](03-approximation-theory.md)'s approximation theory (≈10:13), whose universal approximation
theorem lecture 3 summarizes as "approximate 'bumps' then linearly combine" (see
[lecture 3](03-approximation-theory.md#how-seriously-to-take-it)).

"The freedom to choose the number of bumps $n$, the bump centres $x_i$ and the weights $\alpha_i$ leads to a
rich function space called a "reproducing kernel Hilbert space"" (slide 9), the space obtained by letting
the number of bumps, at all possible locations, go to infinity (≈13:20). The course does not develop the
space further.

## The kernel as the modelling choice

Before deep learning, "everyone was doing kernel methods": instead of choosing a neural architecture,
"you pick a kernel function. You choose a good kernel or a good bump function that somehow captures the
structure in your data" (lecture 13, ≈14:09). The kernel's hyperparameters, such as a length scale, and its
functional form are modelling decisions, and papers list "a table of 10 different kernels" for different
kinds of structure, much as one would "use a ConvNet for images and use a transformer for this"
(≈1:03:19–1:04:52). Kernels can be defined on inputs that are not real numbers, including discrete ones,
"one of the reasons people really like kernel methods" (≈39:36).

## Correspondences with Gaussian processes and networks

Lecture 13 draws three function spaces at the corners of a triangle (slide 11). Kernel methods and
[Gaussian processes](gaussian-processes.md) are related in both directions. Concretely, the mean of a
Gaussian process conditioned on the data "turns out to be equivalent to the kernel interpolator of minimum
kernel norm, for some definition of the RKHS norm" (≈16:30–18:04). Neural networks reach Gaussian
processes through the infinite-width, random-weight limit, so a network architecture has an equivalent
kernel, its covariance function; for a ReLU network it is the compositional arccosine kernel (slide 25).
See [Gaussian processes](gaussian-processes.md#the-neural-networkgaussian-process-correspondence).

On expressiveness the lecturer does not rank the three. Kernel functions, built from bumps, "may still be
reasonably smooth functions"; Gaussian process samples can be "really horrendous", though their posterior
mean is smooth; "I don't know which one is more expressive" (≈20:27–22:06).

## Why not kernel methods?

If a network has an equivalent kernel, why train the network? Lecture 13 poses the question at the start,
"Why didn't we go back to kernel methods?" (≈14:55), and returns to it when a student asks. The lecturer's
candidate reasons: the kernel inherits hyperparameters from the architecture it comes from, so it is not
"a tuning-free thing"; and kernel methods scale worse, at a cost "cubic in the size of the training set",
whereas training a network costs a forward and a backward pass per training example. "It's a super
interesting question why people use neural nets rather than kernel methods. I think it may still be a
somewhat open problem" (≈57:42–59:17).

## Not the same as a convolution kernel

The word "kernel" has other meanings in the course. A [convolution](convolution.md)'s kernel is its small
filter of weights. In [lecture 6](06-generalization-theory.md) a network's **kernel** is the matrix of
similarities between the output representations of every pair of data points, whose block structure shows
how the network has clustered the data (see
[representation learning](representation-learning.md#kernels-and-why-deeper-representations-cluster-lecture-6)).
That is the same idea as this page's in one respect: both kernels measure the similarity of two inputs.

## Where it appears

- [Lecture 13](13-representation-learning-theory.md) — bump functions, the reproducing kernel Hilbert
  space, the correspondences, and why networks won.

## See also

- [Gaussian processes](gaussian-processes.md) — random functions, covariance functions and the NN-GP
  correspondence.
- [Representational power](representational-power.md) — what networks can approximate, including by bumps.
- [Inductive bias](inductive-bias.md) — the assumptions an architecture, or a kernel, builds in.
