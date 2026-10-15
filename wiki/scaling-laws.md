# Scaling laws

Empirical laws for how a model's test loss falls as compute, data and parameters grow. Lecture 3
uses them for a narrower question: whether it matters how parameters are split between width and
depth. The course gives scaling laws a lecture of their own, **[lecture 20, Scaling Laws](20-scaling-laws.md)**, whose
recording ends at ≈38:01, after slide 17; the rest of its deck is known here from the slides alone. Covered so far:
[lecture 3](03-approximation-theory.md), slides 37–39; and [lecture 9](09-hackers-guide-to-deep-learning.md), slides 25 and 44,
≈49:29–52:35 and ≈1:05:43–1:08:04, where scale is practical advice; and [lecture 18](18-transfer-learning-models.md), slides 3
and 41–45, ≈1:31–3:08 and ≈58:54–1:09:04 (deep learning does not saturate with data, and the growth, cost and payoff of foundation
models); and [lecture 20](20-scaling-laws.md), slides 1–51, of which the recording (≈0:00–38:01) covers slides 1–17; and [lecture
21](21-language-models.md), slides 5–7, 19–20, 35, 51, 55, 72 and 81, on emergent abilities and what else changes with a language
model's scale; and [lecture 24](24-inference-methods-for-deep-learning.md), slides 4–9, ≈8:24–14:43, on scaling
test-time compute and trading search against learning.

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

Lecture 20 takes the paper up in full ([below](#chinchilla-revises-the-split-lecture-20)). Its slide 24 pastes the paper's own title
block, "Training Compute-Optimal Large Language Models", with the second author's name spelled Sebastian Borgeaud, and its slide 25
places the paper "Two years later…" than Kaplan et al.'s 2020.

For a full treatment of the Kaplan-versus-Chinchilla dispute from another course, see the CS336
knowledge base listed in [SEE_ALSO](../SEE_ALSO.md).

## Scale as practical advice (lecture 9)

Lecture 9 turns scaling into a rule of thumb. "The easiest way to get better performance" is to scale the
data ("more (diverse) training examples"), the model ("more layers, more channels") and the compute
("train for longer"): "in the current era, I would say these are the top three factors that determine
success … but working at small scale forces efficiency, and *then* when you do scale up, you get more bang
for your buck" (slide 44). The lecturer calls it "kind of the scale is all you need recipe", "not
necessarily the entire story, but it is part of the story" (≈1:05:43). His example is a colorization
baseline left training for two weeks while he and his collaborator were away, which beat "all the kind of
crazy advanced architectures and variants that we tried" (≈1:06:31). Of the "bitter lesson": "truth might
be simple and easy. And if it is, all the better." His summary: "scaling is necessary but not sufficient"
(≈1:07:18–1:08:04).

Scale of data has three senses on slide 25: many $\lbrace x, y \rbrace$ pairs, an input $x$ that is "a big
object, replete with information", and a big output $y$. A student working on rare events (crash
detection) objects that scale alone wastes most of the data on the common case. The lecturer agrees that
"data diversity and coverage over the classes of interest, that's the key thing that matters. So scale is
kind of a proxy for, actually, coverage" (≈51:04–51:49).

## Not the same as scaling rules

Lecture 7, "Scaling Rules for Optimization", is about a different question: how to initialize and update a
network so that the best learning rate does not change as it is made wider, and so that it still trains
as it is made deeper. See [scaling rules](scaling-rules.md).

## Scale and its cost (lecture 18)

Lecture 18 opens with Andrew Ng's chart (slide 3): older learning algorithms plateau as data grows, deep learning does not, and one
analysis suggests current models have "far more capacity than data" (≈2:19–3:08). It previews lecture 20 ("a lecture from Phil around
some of these scaling laws, these relationships that people are starting to learn about the scale of the number of parameters, the
amount of available data", ≈2:19). Its section on foundation models shows the growth: training compute up "300,000x" from AlexNet to
AlphaGo Zero, a 3.4-month doubling (slide 41), and parameter counts rising in industry while academia's stay flat (slide 42). Slide 45
gives the payoff: on 58 tasks, the small models in its lower-left box, up to about $2 \times 10^{10}$ parameters, are "Not working at all", and only the largest PaLM models are
"Working as well as an average human". The lecture weighs it against the cost in money, energy and water (slides 43 and 44) and ends
the argument with "let's do research into efficiency to get these back down" (≈1:09:04). See [foundation models and
prompting](foundation-models-and-prompting.md).

## The scaling-laws lecture (lecture 20)

[Lecture 20](20-scaling-laws.md) is the course's lecture on the subject. Its recording ends at ≈38:01, mid-sentence, after slide 17;
everything below about later slides comes from the deck alone. In its notation $N$ is the number of parameters, $D$ the dataset size in
tokens, $C$ the compute in FLOPs and $L$ the test loss.

### Why predict, and how the prediction is used

Training compute has grown along a "3.4-month doubling" (slide 2), and Sutton's "Bitter Lesson" (slide 3) argues that "general methods
that leverage computation are ultimately the most effective". "The one that scales better will win" (≈4:39), so what matters is how a
method scales. The practical payoff is on slide 5: OpenAI predicted GPT-4's loss from smaller models along a fitted curve, so "they design
it at 1/1,000 the scale, or maybe in this case, it's 10,000, 1/10,000 the scale", "and then they only train once at that massive scale"
(≈7:49–8:39). Slide 6 does the same for a downstream measure, the pass rate on 23 coding problems (≈10:56).

### Power laws in data, parameters and compute

Kaplan, McCandlish et al. (2020) trained autoregressive transformers on WebText2 and counted compute as "6 x #parameters x batchsize x
#steps" (slide 8). Their first finding is that the test loss is a power law in each resource $X$, $L(X) = X^{-\alpha}$, a straight line on
log-log axes (slide 9): $L(D) = (D_c / D)^{\alpha_D}$ with $\alpha_D \sim 0.095$ (slide 10), $L(N) = (N_c / N)^{\alpha_N}$ with
$\alpha_N \sim 0.076$ (slide 11), and a law in compute with exponent about 0.050 (slide 12). The compute law is the frontier of many
training runs, not one run: "if you … take one architecture, and you just scale the compute, you won't see a power law … you need to tune
the data set size and parameters for every level of compute" (≈28:40). The lecturer is clear that this is empirical: "That could have
come out as some parabola or something crazy, but it turned out to come out as a line on a log-log plot" (≈26:17).

### Data and parameters together, and how to split compute

Performance improves predictably "as long as we scale up $N$ and $D$ in tandem", but a small model stops improving as data grows
(slide 13, ≈30:14–31:02). Kaplan et al. fit $L(N, D)$ jointly and conclude that the parameters should grow as $C^{0.73}$ and the data as
$C^{0.27}$ (slide 15): "I should train relatively big models on relatively small data" (≈34:54), though "take these numbers with a grain
of salt". Rosenfeld et al. (2020) found similar laws for vision and language and showed that a law fitted on small models predicts larger
ones (slides 16 and 17): "We're not just reading tea leaves" (≈38:01). The rest of the deck has no recording. It adds that the model's
shape hardly matters (slide 18, "Performance depends strongly on scale, weakly on model shape"), that larger models need fewer samples
(slide 19), and that the field then built "larger models, scale data only moderately" (slide 23), from GPT-3's 175 billion parameters
on 300 billion tokens to MT-NLG's 530 billion on 270 billion.

### Chinchilla revises the split (lecture 20)

"Two years later…" (slide 25), Hoffmann et al. found that "current large language models are significantly under-trained" and that
"the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training
tokens should also be doubled". All three of their methods give roughly $N \propto C^{0.49}$ and $D \propto C^{0.51}$, against Kaplan et
al.'s 0.73 and 0.27 (slide 27): "Better allocation: smaller model, more data/steps". Their 70-billion-parameter Chinchilla, given
Gopher's compute and about four times the data, outperforms Gopher (280B), GPT-3, Jurassic-1 and Megatron-Turing NLG (slides 25 and
28). Among the differences from Kaplan et al., slide 26 names "different learning rate schedules … adapted to dataset size", the
confounder lecture 3 warned about.

### Why a power law?

One hypothesis (slides 31–35, from Sharma and Kaplan) explains the exponent through the dimension of the data. Approximating a
Lipschitz function on $[0,1]^{d}$ by $N$ piecewise constant pieces gives a loss bounded by $1/N^{2/d}$, and by $N^{-4/d}$ for
piecewise linear (ReLU) pieces (slide 33). If $d$ is the intrinsic dimension of the data manifold a network maps its inputs into, this
predicts $L(N) \propto 1/N^{\alpha}$ with $\alpha \approx 4/d$, so "different models will scale similarly on the same data" (slide 34).
The measured intrinsic dimension fits for vision models; for GPT it gives only a bound (slide 35). The lecturer, asked why a power law
rather than an exponential, said the theory "comes out of just high-dimensional statistics" but "I'm not sure how much I trust the
theory behind this" (≈24:45–25:32). See [representational power](representational-power.md).

### Beyond the power law, and the critical batch size

A small exponent means "a small improvement in loss may need an order of magnitude more data/parameters"; the deck suggests that many
training examples are redundant and that pruning data intelligently can beat power-law scaling (slide 37, Sorscher et al. 2022). Its
section on the batch size (slides 38–45) derives the **critical batch size**: one SGD step's progress is proportional to
$1/(1 + \mathcal{B}_ {\text{noise}}/B)$, so doubling a small batch nearly doubles the progress per step, and beyond the noise scale
$\mathcal{B}_ {\text{noise}}$ a larger batch buys almost nothing (slide 44). See [gradient descent](gradient-descent.md).

### Representational convergence

The deck's last section (slides 46–50) asks whether representations converge as models scale. It compares a vision model and a language
model through their kernels, the similarities each assigns to pairs of inputs (slides 47–48), and shows that, across eleven language
models and four sizes of the vision model DINO, alignment to it rises with language performance and with the vision model's size
("Scaling laws for cross-modal alignment", slide 49). Slide 50 is the diagram of "The Platonic Representation Hypothesis" (Huh et al.,
2024). See [representation learning](representation-learning.md).

## Emergent abilities (lecture 21)

[Lecture 21](21-language-models.md) shows scale from the side of capabilities rather than loss. Its slide 5 ends the path from n-gram counts
to transformers with one more step, "Scaling", to large language models, while slides 6–7 warn that scale is not everything: n-gram models
of 2007 already had "300 billion n-grams", and "Number of parameters is a bad proxy for model flexibility" (≈7:45–9:16). Slide 19 charts
the "arms race between technology companies", model sizes rising to about 540 billion parameters by 2022. Slide 20 (Wei et al., 2022) shows
**emergent abilities**: on eight tasks, from modular arithmetic to word-in-context, accuracy stays at the random baseline until some model
size and then rises sharply, abilities "without having been trained in a targeted way" (≈20:56–21:42). The same shape recurs through the
lecture: in-context learning improves with model size (slide 35); chain-of-thought prompting helps only the largest models (slide 51),
because, the lecturer suggests, small models "do the copying bit, and they won't do the generalization bit" (≈44:28); so does zero-shot
chain of thought (slide 55); and slide 72 shows instruction tuning, a scratchpad and calibration likewise having no positive effect "until a
certain model scale". Scale can also hurt: on slide 81 a 175B model asked "What happens if you smash a mirror?" answers with the
superstition, because it models "in a finer grained way, the training distribution" (≈1:06:58).

## Test-time compute (lecture 24)

[Lecture 24](24-inference-methods-for-deep-learning.md) reads the scaling story through Sutton's "The Bitter Lesson": the two methods "that seem to scale arbitrarily" with computation "are
search and learning" (slide 4), and "a lot of people have focused on the learning part" (≈9:13). Slide 5 sets two charts from OpenAI's o1
announcement side by side: AIME accuracy against train-time compute, the familiar power law with a slope of "maybe 0.5", and against
**test-time compute**, which gives "also … a power law. And in fact, the slope in this particular case is even steeper" (≈10:00–10:46).
Slide 9, Jones's "Scaling Scaling Laws with Board Games" (2021), shows the two trading off: for each playing strength, a curve of test-time
compute against train-time compute, so that "I can do more search, and then I can get the same elo for less training, or I can do less
search, and then I can get the same elo for more training" (≈13:56). "So search is test time. Learning is train time" (≈14:43). See
[search and test-time compute](test-time-compute.md).
