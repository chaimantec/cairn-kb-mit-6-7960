# Inductive bias

An **inductive bias** is the structure a model assumes about the function it is learning before it
sees any data: a prior, built into the architecture, over what form the answer will take. The
course introduces the idea at the end of [lecture 3](03-approximation-theory.md), as the reason not
to flatten every input into a vector for an MLP, and makes it the organizing idea of the
architecture lectures, starting with [lecture 4](04-architectures-grids.md). Covered so far:
lecture 3's closing preview (≈1:18:12–1:22:07); lecture 4, slides 3–10 and 23–36, ≈1:32–13:04 and
≈19:13–30:04, and its answer on hand-crafted versus learned filters, ≈54:56–56:30; [lecture 5](05-architectures-graphs.md), on
graphs: slides 11–13 and 34–44, ≈2:20–3:52, ≈17:51–30:16 and ≈1:05:56–1:20:36; [lecture 6](06-generalization-theory.md),
on why generalization needs inductive bias at all: slides 19, 47–63, ≈21:37–23:57 and
≈58:53–1:18:26.

## Why an MLP is not enough

A [multilayer perceptron](multilayer-perceptron.md) is a universal approximator (see
[representational power](representational-power.md)), so in principle it can learn anything. Lecture 3
asked whether that makes it a good choice for, say, a waveform or a photograph, and the class's
objection was that "an MLP doesn't take advantage of the structure in the input data. So with an
image, there's an inherent two-dimensional structure. And with a waveform, there's an inherent
sequential structure" (lecture 3, ≈1:20:32). The lecturer's conclusion: "perhaps we want to match the
architecture to the problem that we're trying to solve, or the structure of the data."

Lecture 4 opens on the same point (slide 3, ≈3:03–4:36). The MLP has "very weak inductive biases…
there's not a lot of structure or intuition baked into this model", and the price is data: it is
"sample inefficient or data hungry", and for a complex function might need "more data than exists on
Earth to be able to really generalize."

## The hypothesis-space picture

Lecture 4's slides 4 to 6 draw learning as a search through function space (≈4:36–7:40). All
mappings $\mathcal{X} \to \mathcal{Y}$ form a box containing the true solution. The training data
rule out most of it, leaving the functions that "Fit the data", and optimization returns one of
those, which "won't necessarily be very close to the true solution."

There are two ways to close the gap. **More data** shrinks the set of functions that fit, so the
learned solution lands nearer the truth. Or, without more data, **define a hypothesis space**
$\mathcal{F}$: a region the architecture permits, "a stronger inductive bias or a stronger prior over
what we think the model is going to take the form of", searched only where it meets the functions
that fit. Slide 6: "We can pin down truth *either* by adding more data, or by using a more
constrained architecture." And both together is best: "more data helps. Better architecture or
stronger inductive bias helps. And those two things combined is ideal" (≈7:40).

## Bias decides what happens outside the data

The one-dimensional example on lecture 4's slides 7, 8 and 10 shows what a bias buys (≈7:40–12:17).
The same wiggly function is fitted from one, a few, and many training points:

- A **5-layer ReLU network**, the weak bias, fits well where there is data and poorly everywhere
  else: "really, really nice fits in the distribution where you have data, but you get really poor
  generalization out of distribution."
- A model of exactly the right form, $y = ax + \sin(bx^2)$, recovers the whole function from a few
  points, including far from them.
- A **5-layer sin-net** (SIREN), with sine activations, sits in between: it does not find the true
  function, but its errors are periodic, because "we've just added an inductive bias in our model
  architecture that says that the distribution should be periodic."

Slide 8 draws the conclusion: "Architectures enable us to generalize *outside the training
distribution*." So "the hypothesis space you build, that's really something that helps us
generalize out of the distribution of the data we have" (≈8:26). See
[generalization and double descent](generalization-and-double-descent.md).

What makes a bias good? Slide 8: "A good architecture is one that can represent the true function and
is otherwise minimal (and is also easy to search over via gradient-based learning, easy to
parallelize, fast on GPU, etc)." A bias that excludes the truth is no help however much data there
is; one that admits too much buys little.

A caution from slide 9, on SIREN fitting images better than ReLU networks: such a result "may be due
to improved approximation ability but it might also be due to improved optimization ability; these
two effects are typically coupled in experiments."

## The convolutional bias: translation equivariance

The first concrete inductive bias the course builds is the convolutional layer's (see
[convolution](convolution.md)). It assumes two things about images. Each output depends only on a
**local** patch of the input, and the same function applies at **every position** (lecture 4, slides
27–29). The second is a symmetry, written on slide 23 as an equivariance,

$$f(\texttt{translate}(x)) = \texttt{translate}(f(x))$$

and lecture 4 justifies it from the world: "if you took an image of a bird and then you moved that
portion of the image corresponding to a bird to a different part of the image… the object you're
taking a photo of is the same. So there actually is this type of symmetry innate to categories in the
real world. And this symmetry is something we've built into our hypothesis space via the model
structure" (≈20:00–20:46). The lecturer flagged that this glosses over pose.

The bias pays off as fewer parameters ("Fewer parameters —> easier to learn, less overfitting",
slide 32) and as a kind of generalization an MLP cannot have at all: a convolutional layer "can be
applied to arbitrarily-sized inputs (generalizes beyond the training data due to an architectural
structure!)" (slide 34). And "for things where this is [an] appropriate inductive bias, you can learn
more efficiently. You get a better fit with less data" (≈35:29).

## When the bias is wrong: removing it on purpose

A bias that fits one task can be wrong for another. Shift equivariance in time would make "picking a
cup up… the same action as putting a cup down" (lecture 4, ≈1:10:38). Slide 76 gives two remedies:
use an architecture without the bias, such as an MLP, or keep the convolution and give it the
position as an extra input, a **positional encoding**. See
[neural fields and positional encoding](neural-fields-and-positional-encoding.md).

## The graph bias: permutation symmetry

Lecture 5 states the architecture-as-constraint view most bluntly: "graph nets are not universal in
the same way that MLPs are. And in fact, that's where the power comes from … another big part of
architecture design is adding constraints. So making a function approximator that actually can't fit
certain types of functions because we want to rule those types of functions out … So universality
is actually not what we're after in architecture design" (≈2:20–3:52).

For graphs the constraint is a symmetry over node orderings. The numbering of a graph's nodes is
arbitrary, so a graph-level prediction should be **permutation invariant** and a per-node prediction
**permutation equivariant** (lecture 5, slide 11); an MLP fed the adjacency matrix is neither
(≈20:12–22:31). "Where convolutional networks are invariant or equivariant to translation, graph nets
are going to be invariant or equivariant to permutations of the inputs" (≈22:31), and "translation
invariance is one type of permutation invariance, but permutation invariance is a more general version
of that" (≈41:08). A graph net builds the symmetry in by aggregating each node's neighbours with an
order-blind function such as a sum, and keeps lecture 4's other biases: local operations, globalizing
through depth, weight sharing, and inputs of any size (slide 12, ≈28:41–30:16).

The bias has a measurable cost. Two graphs whose nodes see the same neighbourhood trees get the same
output from every graph net, so some functions are out of reach, such as a graph's diameter or
longest cycle (slides 36 and 42). As with convolutions, the remedy is to remove some of the bias on
purpose with a **positional encoding**, here eigenvectors of the graph Laplacian, which trades
invariance to new orderings for the power to tell more graphs apart (slide 43, ≈1:19:02–1:20:36).
See [graph neural networks](graph-neural-networks.md).

## Hand-crafted structure versus learned structure

An inductive bias can also be built in by hand. Asked why convolutional filters are learned when
signal processing designs them, the lecturer described the field's move from **feature engineering**
to filters learned end to end as "this massive paradigm shift that happened maybe 10, 15 years ago",
and gave the trade-off: "it often comes down to how much data you have. So if you don't have enough
data to learn the filters, well, then more handcrafting, more knowledge, more inductive bias tends to
be better because you actually can put more constraints on the system so that you can optimize
better with less data. But if you're in a space where you can get huge amounts of data to learn from,
often, we don't know as much as we think we know about what optimality might be" (lecture 4,
≈55:44–56:30). See [differentiable programming](differentiable-programming.md), which lecture 2 frames
as the split between human-programmed and learned parts of a program.

## Why generalization needs inductive bias (lecture 6)

Lecture 6 turns the idea into an argument. A model can fit its training data perfectly and still
generalize badly: a "filing cabinet" that memorizes every training pair and returns 0 elsewhere has zero
training error. So "generalization requires *inductive biases*. Can't be explained by just fitting the
training data (we have to rule out the filing cabinet!)" (slide 47). And the classical measures of
complexity, the number of parameters and the VC dimension, fail to explain deep nets, so the biases must
be something else: deep learning "must have some nice inductive biases that control complexity in ways we
don't fully know how to characterize!" See [generalization and double
descent](generalization-and-double-descent.md).

Its first example is compositional (slide 19, ≈21:37–23:57). pix2pix, a ConvNet trained to turn
sketches into cat photos, draws a cat with three or eight eyes when given a sketch with three or eight
ovals, though it never saw such a cat. The explanation is "the inductive biases of convolutional nets":
the same patch function turns each oval into an eye ("When I see an oval, draw an eye"), and the
architecture stitches the patches back together, so a new arrangement of familiar parts needs no new
learning. Graph nets do the same for permutations: "That's baked into the architecture. It's not something
you have to learn from the data. So architecture is one of the main levers we have for these feats of
generalization" (≈23:57).

The lecture's candidate biases go beyond architecture (slides 48–61). Random settings of a network's
weights mostly compute simple functions, so training tends to land on simple ones (the parameter-function
map); deeper networks are biased toward low-rank representations of their data; and optimizers prefer
some solutions, through weight decay, initialization near zero and the flat minima that gradient descent
with a finite step size finds. But "I think all of what I've said so far is not actually the most
important. I think the most important is … the architectural symmetries" (≈1:15:19). Slide 62 names
three: **invariances** (max pooling over oriented filters fires "regardless of what the orientation of
that bird's beak is"), **equivariances** (a ConvNet's to shifts, a graph net's to permutations) and
**compositionality**, where "the conjunction of components is just given by something that's defined by
the architecture. It's not learned" (≈1:16:06–1:17:40). The lecturer thinks this is much of why large
language models generalize: the architecture "carves the world at its joints" into words, sentences and
paragraphs, and their compositions come "not from learning, but from just how the architecture is built"
(≈1:17:40).

Domain knowledge is the strongest form (slide 63). NeRF's projection and light-transport equations let
it generalize to new viewpoints, and a drug-interaction network builds in how drugs interact: "It's not
just fitting to data. It's data plus structure, data plus constraints" (≈1:17:40–1:18:26).

## Where it goes next

Lectures 4 and 5 cover grids and graphs. The [course map](course-map.md) lists the architecture
lectures that follow: transformers (8) and memory (10). Lecture 5 calls transformers "a special kind
of graph net" (≈1:35), graph nets whose aggregation is attention (≈56:38). Lecture 4 itself points ahead twice: positional
encodings return "in the transformers lecture" (≈1:15:19), and its closing slide says the idea of
applying one function to every patch "appears in almost all modern architectures, such as CNNs,
transformers, NeRFs, and more" (slide 82).
