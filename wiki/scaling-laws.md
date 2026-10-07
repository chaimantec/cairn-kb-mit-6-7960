# Scaling laws

Empirical laws for how a model's test loss falls as compute, data and parameters grow. Lecture 3
uses them for a narrower question: whether it matters how parameters are split between width and
depth. The course gives scaling laws a lecture of their own, **lecture 20, Scaling Laws**, which
this knowledge base does not yet cover (see the [course map](course-map.md)). Covered so far:
[lecture 3](03-approximation-theory.md), slides 37–39 only.

## Why lecture 3 brings them up

Lecture 3 proves two theoretical results about width and depth: a three-layer network can
approximate any Lipschitz function if it is wide enough, and some functions need exponentially
fewer neurons if the network is deep (see [representational power](representational-power.md)).
Neither result says what to do in practice. Slide 37 makes it concrete: "pretend you work at an LLM
startup" and you want to train the most efficient LLM possible, so "you really care about the
optimal width vs. depth", for the cost of training, the cost of inference and how well the model
works. "In a sense, it's the most basic question … But it's, to a large extent, an unsolved
problem" (≈1:11:56).

The difficulty is that approximation, optimization and generalization, which theory treats one at
a time, arrive together in a real training run. When training goes badly you cannot tell whether
the network cannot represent the target or the optimizer is failing to find it. Only
generalization is easy to diagnose, by comparing training and test error (≈1:11:10).

## Kaplan, McCandlish et al. (2020)

Slide 38 reproduces two figures from Kaplan, McCandlish et al. (2020) (≈1:12:41–1:15:01).

**Loss falls as a power law.** The first figure plots test loss on log-log axes against three
quantities: compute (in PF-days, non-embedding), dataset size (in tokens) and parameters
(non-embedding). In each panel the loss falls along a straight line, and each prints its fitted
power law:

$$L = (C_{\min} / 2.3 \cdot 10^{8})^{-0.050}, \qquad L = (D / 5.4 \cdot 10^{13})^{-0.095}, \qquad L = (N / 8.8 \cdot 10^{13})^{-0.076},$$

where here $L$ is the test loss, $C_{\min}$ the compute, $D$ the dataset size and $N$ the number of
parameters (the figure's own symbols). The lecturer's reading: "as you make GPT bigger, it gets
better", and "the test loss … is always going down as you scale any of these axes". People "got
T-shirts where they just had this figure". It cannot go on forever, since "you can't have negative
loss", so it "has to saturate eventually". But "back in 2020, they were saying, hey, the models keep
getting better. We should pay attention to this", and then "ChatGPT was created and so on"
(≈1:12:41–1:13:28).

**Width versus depth barely matters.** The figure plots against parameters, "not against width
and it's not against depth". The paper's claim, as the lecturer gives it, is that "within several
orders of magnitude, the allocation of your compute budget between width and depth doesn't
matter. All that matters is the number of parameters and number of flops. You can have a depth 20
network or a depth 10 network with a commensurate number of width, it doesn't matter" (≈1:13:28).
The second figure shows it: test loss against parameters for networks of 0, 1, 2, 3, 6 and more
than 6 layers. When the embedding parameters are left out of the count (the right-hand panel),
"beyond depth 6, it doesn't matter what the width is or what the depth is. The curves converge to
each other" (≈1:14:15). In the panel the 2-, 3-, 6- and more-than-6-layer curves nearly collapse
onto one line, and the 1-layer curve, after crossing them at small sizes, ends clearly above. The
lecturer ties this convergence to counting parameters "once they subtract the embedding
parameters", which is why he reads it off the right-hand panel.

The lecturer called this "a very, very practical perspective, but from some very careful
experimentalists about how things actually work in transformers" (≈1:14:15–1:15:01).

## Chinchilla, and confounders

Slide 39 adds the "Chinchilla scaling rules" (the slide's citation reads "Hoffmann, Borgeau, Mensch
et al (2020)"), a paper interested in the same questions that "question[s] some results in Kaplan
et al" (≈1:15:01–1:16:34). In particular, "if you use a different learning rate schedule for the
training, you can get slightly different, qualitatively different conclusions".

The lesson the lecturer draws is about **confounders**: "some aspect of the training that we don't
fully understand, but is having a bearing on the results". One experimental setup says width
versus depth doesn't matter; "then you change some detail about how you actually train the
network. And now it's like oh, no, scaling width is a lot better because I fixed that issue." So
"to really answer 'what is the optimal width versus depth'", you "need to obsess over 'minor
details' of the training pipeline" (slide 39). Resolving such questions experimentally is hard
because of these confounders, and resolving them theoretically is hard too, which is why "a lot of
these problems are unsolved" (≈1:16:34).

For a full treatment of the Kaplan-versus-Chinchilla dispute from another course, see the CS336
knowledge base listed in [SEE_ALSO](../SEE_ALSO.md).

## Not the same as scaling rules

Lecture 7, "Scaling Rules for Optimization", is about a different question: how to initialize and update a
network so that the best learning rate does not change as it is made wider, and so that it still trains
as it is made deeper. See [scaling rules](scaling-rules.md).
