# Representational power: what a network can approximate

Which functions a neural network can represent, and at what cost. Lecture 1 previews this as
"why we can approximate" (slides 46–47, ≈40:25–42:42) and assigns the full treatment to **lecture
3, Approximation Theory**. Covered so far: [lecture 1](01-introduction.md) only.

## One layer versus two or more

Slide 47 states it in three lines:

- **One layer** gives a **linear decision surface**. A single perceptron can only split its input
  space with a line or hyperplane — enough for linearly separable data, never for XOR (see
  [multilayer perceptrons](multilayer-perceptron.md)).
- **Two or more layers** can, "in theory, … represent any function", **assuming a non-trivial
  non-linearity** between the layers. The assumption is essential: without it "a linear layer and
  another linear layer, this is just a linear combination. It's still linear" (≈40:25–41:11).
- **"But issue is efficiency."**

## The intuition: a Riemann sum

The lecturer is explicit that this is "a rough, hand-wavy intuition thing" (≈41:11). Given enough
capacity — enough units, each contributing one narrow piece — a network can build up any curve the
way a Riemann sum builds up an integral from thin rectangles. "As you're shrinking the size of
these different vertical things", the approximation gets as good as you like, "as long as you have
enough capacity in terms of the number of small things that you're building up". Slide 47 shows a
curve beside a staircase of bars approximating it, though a banner covers most of the figure in
the published deck.

## Width versus depth

The catch is efficiency. "In theory, you could approximate any complicated function with a
sufficiently wide two-layer network" — wide meaning "maybe thousands or millions of dimensions" —
"but in practice, that's actually very inefficient" (≈41:57). A **narrow, deep** model, with much
smaller layers but "many, many, many more stacked layers", may approximate the same function "with
a lot less parameters". The lecturer's empirical summary: "In practice, we do find that's true.
More layers helps."

Two follow-ups from the discussion of this trade-off, both later in lecture 1 (≈48:08–49:40):

- **Interpretability doesn't favour either shape.** A student asked whether wide-and-shallow or
  narrow-and-deep is easier to interpret. Neither is: two near-infinite layers live in "really,
  really high dimensional space", and many narrow layers are just as opaque. "Anytime you get to
  larger numbers of parameters in either dimension, that interpretability gets more difficult."
  That is why fields such as ecology still use simple additive models of linear terms.
- **There is no known recipe.** Jeremy Bernstein noted that the width-versus-depth recipe of
  frontier language models is "a closely guarded secret", and the lecturer agreed: "there's no
  perfect prescription or recipe for width versus depth and what's optimal."

## Where it goes next

The banner on slide 47 sends the theory to **lecture 3, Approximation theory**: "the theory of
what it is possible to approximate" (≈41:57). How deep networks go on to *generalize* from data,
rather than merely fit it, is a separate question — see
[generalization and double descent](generalization-and-double-descent.md).
