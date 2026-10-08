# Data augmentation

**Data augmentation** makes more training data out of less, by applying transformations that should
not change the target, such as mirroring, cropping or darkening an image. **Domain randomization** is the
same idea pushed further: randomize the training domain so widely (colours, lighting, physics) that the
test conditions look like one more random draw. Both are ways of changing the data rather than the
learner, and the course presents them as the practitioner's alternative to building invariance into an
architecture. Covered so far: [lecture 9](09-hackers-guide-to-deep-learning.md), slides 13–18,
≈30:17–41:03, with mentions in [lecture 2](02-how-to-train-a-neural-net.md) and
[lecture 6](06-generalization-theory.md). Out-of-distribution generalization and transfer learning from
the data side are lectures 17 and 19, not yet in this knowledge base (see the [course map](course-map.md)).

**Notation.** A training pair is $(\mathbf{x}, y)$, an input and its label. $p_{\text{source}}$ is the
distribution the training data comes from and $p_{\text{target}}$ the one the model is used on.

## Label-preserving transformations

Lecture 9's slide 13 is the recap ("You did some data augmentation in Pset 1", ≈30:17). A training pair,
a clown fish labelled "Fish", fans out into four new pairs, all still labelled "Fish": "Mirror", two
"Crop"s and "Darken". (The photographs are excluded from OCW's licence and described in the slide
file.) The transformations must "not fundamentally change" the data; for classification, they must not
change the class. "So this creates bigger data from smaller data" (≈30:17–31:02).

Slide 14 gives the purpose: "Train on randomly perturbed data, so that test set just looks like another
random perturbation." Its figure spreads training points over the whole "Data space", with a few test
points inside their spread. Augmenting "just makes it so that my test data looks like it's in
distribution. My test data just looks like more training data" (≈33:20).

## Augmentation or an invariant architecture?

The lecturer takes a side (≈31:02–32:34). One community says augmentation is "a little hacky" and should
be replaced by **geometric deep learning**: rather than training a network to ignore flips, crops and
lighting, design one that is invariant by construction, as a ConvNet is equivariant to translation
and pooling makes it invariant. "But I think that that's mostly not the right advice for a good hacker in
machine learning." Augmentation is "really easy, really intuitive. And it's architecture agnostic". You
list the transformations you accept, "and I'll feed that to any architecture. I can send this to an MLP,
and it will be invariant to these operations too". It "decouples these properties from the architecture
design", though "maybe in a few years, we'll have a very clean framework for geometric deep learning".

This is the other side of the architecture lectures, which build symmetry into the network:
translation equivariance in [convolution](convolution.md) and permutation invariance in
[graph neural networks](graph-neural-networks.md). See [inductive bias](inductive-bias.md).

## Make the problem harder on purpose

Lecture 9 argues that augmentation helps because it makes learning *harder*. Slide 15's quiz shows three
loss curves. The one that falls at once and goes flat is "Bad! Your data is too easy", and the good one
is "Fitting a hard problem", still falling slowly (≈33:20–34:55). The slide's rule is
`max_data min_params loss(data, params)`: minimizing over parameters is backpropagation, but "you want to
increase the difficulty of your data until there's something interesting to learn from it". The
hardest training distribution "is going to generalize the best" (≈34:55–35:42).

## Domain randomization

In robotics, a policy trained in simulation must work in reality (slide 16, "[Sadeghi & Levine 2016]",
example from "[Tobin, Fong, Ray et al. 2017]"). Trained under one lighting condition and one set of block
colours, a robot "will only know how to pick up blocks of that color". Randomizing them "will take a lot
longer to train, but it will generalize better" (≈35:42–36:28). It will not generalize to lighting that
never appears in the randomized training data, "the short answer is, no", though with a broad enough
distribution "maybe you start to get this kind of emergent generalization" (≈36:28–37:13).

Slide 17 names the problem: a "**Domain gap** between $p_{\text{source}}$ and $p_{\text{target}}$ will
cause us to fail to generalize". Domain randomization closes it "by randomizing the source domain"
(≈37:59). OpenAI's robot hand (slide 18) randomized object sizes, masses, friction, joint damping,
actuator gains, joint limits and the gravity vector, and rendered the hand under many visual conditions.
Randomizing gravity models the noise in the robot's accelerometers, so that a slightly miscalibrated
measurement is still "within distribution" (≈39:31). The result: without randomization the hand learned
the task within a few simulated years, and with every randomization it took "like, 10 times longer to get
to the same level" of a much harder problem (≈40:18–41:03). Slide 18's lesson: "High train accuracy can
mean problem is too easy. Add more data to make problem harder."

## Elsewhere in the course

Lecture 2 gives data augmentation as an example of a pre-processing step, which, unlike an operation
inside the network, need not be differentiable (see [differentiable programming](differentiable-programming.md)). Lecture 6's table from
Zhang et al. compares models trained with and without data augmentation and weight decay (see
[generalization and double descent](generalization-and-double-descent.md)).

## See also

- [Lecture 9 — Hacker's Guide to Deep Learning](09-hackers-guide-to-deep-learning.md), where this
  material comes from, including the wider advice to change the data rather than the learner.
- [Inductive bias](inductive-bias.md) — invariance built into the architecture instead.
- [Generalization and double descent](generalization-and-double-descent.md) — why a problem that is too
  easy generalizes badly.
