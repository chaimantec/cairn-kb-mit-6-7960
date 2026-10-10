# Generalization and double descent

Why do deep networks generalize when they have enough parameters to memorize their training data?
Lecture 1 poses this as one of the course's central questions (slides 50–52, ≈43:29–50:26) and
assigns it to **lecture 6, Generalization Theory**, and **lecture 17, Out-of-Distribution
Generalization**. The deck's banner says "Lecture 7" for generalization theory; in the recorded
schedule it is lecture 6 (see the [course map](course-map.md#the-decks-lecture-pointers)).
Covered so far: [lecture 1](01-introduction.md); [lecture 3](03-approximation-theory.md), slides
4, 19 and 34 (generalization as one piece of the approximation–optimization–generalization
puzzle, and a network that fits but would not generalize); [lecture 4](04-architectures-grids.md),
slides 4–10, 32 and 34 (architecture as a way to generalize with less data and outside the training
distribution); and [lecture 6](06-generalization-theory.md), Generalization Theory itself, slides
3–65, ≈0:00–1:19:58, which gives the full treatment: memorization against generalization, double
descent in detail, why the classical measures of complexity fail for deep nets, and the candidate
inductive biases; and [lecture 9](09-hackers-guide-to-deep-learning.md), slides 3, 4 and 14–18, ≈1:34–7:01 and ≈33:20–41:03 (shortcuts
that will not generalize, and making the training problem hard enough to generalize); [lecture 12](12-representation-learning-similarity-based.md), slides 6–9,
≈7:45–11:37 (the geometry of representations that generalize); and [lecture 13](13-representation-learning-theory.md), ≈52:12–56:09 (whether an
infinitely wide network must overfit); [lecture 14](14-generative-models-basics.md), slides 25–27, ≈36:31–43:29 (overfitting in a generative model, and test likelihood).

## The puzzle

"Deep nets have so many parameters, they could just act like lookup tables. They would just
regurgitate the training data. But instead, they seem to learn rules that generalize. And this
actually defies classical theory" (slide 51, ≈43:29–44:15).

## Classical theory: the U-curve

The classical view says that **an over-parameterized model will overfit**. Plot test error ("risk")
against model capacity and you get a U (slide 51, panel A). With too little capacity the model
*underfits*; with too much it *overfits*. Test risk is lowest at a "sweet spot" in between, while
training risk falls steadily toward zero the whole way.

The lecture's example of overfitting is three data points fitted with a 12-dimensional function
(≈46:33–47:23). The function passes through all three points exactly but is "too spiky or too
peaky" to generalize.

## Double descent

The observation from "Double Descent" (Belkin, Hsu, Ma and Mandal, PNAS 2019, cited on slide 51)
is that the U is only the left part of the picture (panel B). Keep adding capacity, and test risk
rises to a peak at the **interpolation threshold** — the capacity at which the model first fits
the training data exactly, so training risk reaches zero — and then **falls again**. Past it, in
the "modern" interpolating regime, the models are "massively overparameterized" and can still get
"even better" (≈44:15–45:01). The panel labels the two sides **under-parameterized ("classical"
regime)** and **over-parameterized ("modern" interpolating regime)**.

So the classical axis of underfitting and overfitting gives way to a new one, under-parameterized
versus over-parameterized. Its message is that "you maybe can't be too overparameterized"
(≈47:23).

## What the questions added

Three student questions in lecture 1 refine the picture (≈45:01–48:08):

**What is capacity?** "The number of parameters in the model" — both width and depth, "basically
how many values are captured in those model weights".

**Where does dataset size come in?** Not on the plotted axes, but "definitely also related". With
few data points it is easy to overfit them. The lecturer suggests thinking of data as "almost the
capacity of your training data", though there is no direct translation. Without enough data a model
"can't learn to interpolate between those data points because you don't have enough coverage of
the data distribution". She calls this "one of the reasons that none of this was possible until we
started building big enough data sets" (≈45:47–46:33). The course will cover "tricks" for getting
models not to overfit when data are scarce.

**Overfitting versus over-parameterized?** Overfitting is the classical failure, the spiky
function. Over-parameterized is a position on the capacity axis — and double descent says it is
not, by itself, a failure. The lecturer adds a practical caveat: this view "doesn't take into
account resources". Without unlimited compute, what you really want is "some optimal point on
this curve where you get good performance" without needing memory you do not have (≈47:23–48:08).

## The simplicity hypothesis

Slide 52 states where theory is heading (≈49:40):

- **Classical theory:** "big models learn complicated functions, and overfit the data."
- **Emerging theory:** "deep nets learn *simple* functions that generalize."

The course promises "a theoretical and a more experimental lecture around generalization, both in
and out of distribution" — recorded lectures 6 and 17. Lecture 6, below, takes up the theoretical half and lands where lecture 1's slide 52
points: classical measures of complexity fail for deep nets, and the candidate explanations are biases
toward simple solutions.

## Fitting is not generalizing (lecture 3)

Lecture 3 splits machine learning into three questions: whether a network that fits the data
**exists** (approximation), whether training can **find** it (optimization), and whether it
"work[s] well on unseen data" (generalization) (slide 4). Its results are about the first only, and
it is explicit that they say nothing about the third: a deep network's extra power to approximate
"does not mean that … very deep networks would generalise well" (slide 34).

It also gives a concrete case of a network that can fit anything and would generalize badly. The
universal approximation theorem it proves builds a network out of narrow rectangles (slides 12–17).
Imagine training that representation (slide 19, ≈43:38–45:13). Each training point pulls up its own
rectangle, "but there's all these other ones which are never actually going to move", and weight
decay would suppress them. The result is a pulse at every training point and zero in between, where
"visually, the obvious way to approximate this data is just to draw a line through it". So "the
training performance is going to be really good, but the generalization performance is going to be
really bad. Because you're just fitting the data on tiny little strips". In lecture 1's terms this
is the lookup-table behaviour that deep nets, empirically, do not fall into.

Lecture 3 adds a practical asymmetry: of the three pieces, generalization is the one you can
diagnose directly, "because you could just compute the training error and the test error, and you
can see if they're different" (≈1:11:10).

## Architecture as a route to generalization (lecture 4)

Lecture 4 makes the architecture itself a generalization tool. Its picture (slides 4–6, ≈4:36–7:40):
of all the functions that fit the training data, optimization returns one that "won't necessarily be
very close to the true solution". More data shrinks that set; so does a **hypothesis space** built into
the architecture. "We can pin down truth *either* by adding more data, or by using a more constrained
architecture" (slide 6).

The lecture's one-dimensional example separates two kinds of generalization (slides 7–8,
≈7:40–9:59). A 5-layer ReLU network trained on many points "start[s] getting really, really nice fits
in the distribution where you have data, but you get really poor generalization out of distribution."
A model of the right form, $y = ax + \sin(bx^2)$, gets the whole curve from a few points. Slide 8:
"Architectures enable us to generalize *outside the training distribution*."

Convolutional layers make the point concretely. Their weight sharing gives "Fewer parameters —>
easier to learn, less overfitting" (slide 32), and because the same filter applies at any position
they "can be applied to arbitrarily-sized inputs (generalizes beyond the training data due to an
architectural structure!)" (slide 34). See [inductive bias](inductive-bias.md) and
[convolution](convolution.md).

## Approximation versus generalization (lecture 6)

Lecture 6 defines the terms with two risks (slide 3). Training minimizes the **empirical risk**, the
average loss of the model $f_ \theta$ over the $N$ training pairs,
$\widehat{\mathcal{R}}(\theta) = \frac{1}{N} \sum_ {i=1}^{N} \mathcal{L}(f_ \theta(\mathbf{x}_ i), \mathbf{y}_ i)$,
where $\mathcal{L}$ is the loss. What matters is the **population risk** (test error), the expected loss
on new samples from the data-generating distribution $\mathcal{P}$,
$\mathcal{R}(\theta) = \mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim \mathcal{P}} \thinspace \mathcal{L}(f_ \theta(\mathbf{x}), \mathbf{y})$.
**Generalization** is the question "how different is $\widehat{\mathcal{R}}(\theta)$ from
$\mathcal{R}(\theta)$?", beside approximation (the best $\mathcal{R}(\theta^\ast)$ the model can
achieve) and optimization (how well $\widehat{\mathcal{R}}$ is being minimized). "And that's all we
actually care about because we want to do well when we deploy the system in the world" (≈5:24).

Data come first: a cats-versus-dogs classifier trained only on cats cannot work, and "you can't really
understand generalization without understanding the data distribution that you're training on"
(slide 5, ≈6:58). The rest of the lecture is about the model.

## Memorization versus generalization (lecture 6)

The lecture's foil is the **filing cabinet**, a program that stores every training pair in a dictionary
and returns 0 for any input it has not seen (slide 6). Its approximation error is zero; its
generalization error is bad unless the true function is "0 almost everywhere" (≈8:30). Fitting can also
be luck: Paul the octopus picked football winners correctly, but was "some random function that happened
to fit the training data" (slide 7, ≈10:03).

Lecture 6's slide 8 separates the two ideas. A filing cabinet and a 3-layer ReLU MLP both fit the same
points exactly. **Memorization** is what a model predicts on the training points, which both get right;
**generalization** is what it does on the points in between and beyond, where the cabinet predicts 0 and
the MLP draws a smooth curve. "Most functions in the world are going to be smooth as opposed to spiky"
(≈12:21), so the MLP's kind of interpolation is the better bet.

Big modern models are not filing cabinets either. The lecturer's counting experiment (slides 9–11): a
cabinet that answered random lists of fruit as well as GPT-4o does would need on the order of 100
trillion entries for a text question and 12.5 trillion for drawing the fruits, more than such models are
thought to be trained on (≈16:57–18:30). And a network can generalize **out of distribution**: pix2pix,
trained to turn computer-generated edge maps into cat photos, also worked on hand-drawn sketches, and
put a third eye, or eight, where they were drawn (slides 12–18, ≈19:16–22:23). The lecturer credits the
ConvNet's patch-by-patch processing, which composes familiar parts into new arrangements (slide 19); see
[inductive bias](inductive-bias.md).

## Double descent in detail (lecture 6)

Lecture 6 shows the phenomenon on a regression problem (slides 25–28, ≈30:10–33:14). Twenty noisy
samples of a smooth curve are fit with a family of polynomials up to degree $d$. Degree 1 underfits,
degree 3 (the true order) fits well, and degree 20 overfits, swinging "wildly to fit the deviations".
Degree 1000 also passes through every sample, but between them it "adheres much more closely to the
true smooth solution", reaching the noisy points with narrow spikes. A student anticipated it: with
that much capacity there is "a large manifold of fits that all fit perfectly", some good and some bad,
and which one you get depends on "how you're searching over that space" (≈32:27).

The **simple + spiky hypothesis** (slide 29, citing Belkin, Rakhlin and Tsybakov 2018) reads the
result as "learned model = 'simple' + 'spiky'": a smooth predictive component plus narrow spikes that
memorize the noise, "memorization plus generalization" (≈34:01–34:47). It is provable for simple systems
but not for deep nets.

The lecturer's intuition for the second descent (≈37:53–38:39): at the **interpolation threshold**,
"the point at which I can perfectly memorize all the training data", the model must contort itself to
fit. With more capacity many solutions fit equally well, and something selects the smoothest among them;
"the pressures that select within the set of things that fit the data the one that is the smoothest are
sometimes called regularizers, or implicit regularizers, or inductive biases."

Three practical remarks from lecture 6. Double descent is "a little finicky to actually observe in
practice"; the lecturer could not reproduce it himself, and momentum, weight decay and other details
"can make this disappear", which is fine, since "all the tricks that we use in practice are just getting
rid of that spike and pushing us more out to the right" (slide 31's MNIST result, ≈40:58–42:28).
Capacity can be measured in compute as well as parameters, "so there's kind of double descent also in
compute time", and "in deep learning, we're out in this regime. We're past the first peak … you just
train longer, and longer, and longer, and it smoothly will get better" (≈39:25–40:10). And with random
Fourier features, the norm of the learned weights peaks exactly at the interpolation threshold and then
falls as features are added: "more features means lower norm parameter solutions", provably for linear
models and empirically for deep nets (slide 32, ≈42:28–44:03).

## How should we measure complexity? (lecture 6)

Occam's razor, "the simplest model that fits the data will generalize best", underlies most
generalization theory, and its rigorous form is that "the shortest program that fits the data is the
one that will generalize best" (slides 21–22, [Solomonoff 1964]). That is intractable, since it means
searching all programs, so lecture 6 tries the classical stand-ins for "simple" one at a time.

- **Number of parameters: no** (slides 24 and 33). Double descent shows more parameters need not
  overfit. Slide 34's example: $h(x) = 10^{-100} f(x) + (1 - 10^{-100}) g(x)$, with $f$ a large network
  and $g$ a small one, is fit almost exactly by $g$ alone, so "the count of parameters is not what
  matters. Something about the size has to also be what matters" (≈45:36).
- **Parameter norm: maybe** (slide 33). It tracks the double-descent picture better, "but parameter
  norm is also not everything" (≈46:27).
- **Number of distinct functions: no** (slides 35–45). This is Vapnik-Chervonenkis theory: if the
  training set dwarfs the number of functions in the class, training error matches population error with
  high probability, because few candidate functions can fit the data by luck (slide 36). Counting the
  class's **dichotomies**, the binary labelings it can realize on the $n$ training points, slide 41 calls
  that count the VC dimension $d$ and bounds the generalization error by $\sqrt{d / n}$. But neural nets
  can fit random labels (slide 42): on CIFAR10 they reach about 100% training accuracy on shuffled labels,
  with test accuracy near 10% (Zhang et al., 2017, slide 44). So they realize all $2^n$ dichotomies, the
  bound becomes $\sqrt{2^n / n}$, and it is "extremely loose (in fact, vacuous)" (slide 43). Yet the same
  networks generalize when the labels are real. The deeper problem is an assumption: "we're not just
  picking a random hypothesis that fits the data. We're using gradient descent" (≈55:48).

(Lecture 6 uses "VC dimension" for the count of dichotomies itself; textbook treatments of VC theory
define the term differently, as a number of points.)

The verdict, slide 46: "for deep learning, it's still an open question!" Slide 47's recap: deep nets
generalize; generalization "requires inductive biases", since fitting the data cannot rule out the filing
cabinet; those biases "can't just be about classical notions of complexity"; so deep learning "must have
some nice inductive biases that control complexity in ways we don't fully know how to characterize!"

## Candidate explanations (lecture 6)

Of all the hypotheses that fit the training data, the **version space** (slide 49), why does training
land on one that generalizes? Lecture 6 offers ideas that "are not yet completely standard or proven"
(≈1:02:01):

- **Simplicity bias in the parameter-function map** (slides 50–51, Valle Pérez, Camargo and Louis, ICLR
  2019). "Most random settings of the weights in biases in a neural net map to simple functions",
  simple in Lempel-Ziv complexity. Since most of parameter space maps to simple functions, learning tends
  to reach the simple corner of the version space "just by chance" (≈1:05:07).
- **Low-rank bias of depth** (slides 52–60, Huh et al., TMLR 2023). Deeper networks, even deep *linear*
  ones that gain no capacity from depth, produce blockier, lower-rank kernels of their outputs, because
  "products of matrices tend to be low rank": a greater proportion of a deep network's parameter space
  maps the data to low-rank embeddings (≈1:08:18–1:12:56).
- **Implicit regularization of optimizers** (slide 61): weight decay shrinks unused weights toward zero;
  initialization near zero biases toward low-norm solutions; and SGD or GD with a finite step size "will
  tend to overshoot or bounce out of minima that are too narrow", finding flat minima, which "can be argued
  to generalize better" (≈1:13:43–1:15:19). See [loss landscapes](loss-landscapes.md).
- **Architectural symmetries** (slide 62), which the lecturer calls "the most important": invariances
  such as max pooling over orientations, equivariances such as a ConvNet's to shifts and a graph net's to
  permutations, and compositionality, where the architecture, not learning, decides how the parts are
  combined (≈1:15:19–1:17:40). See [inductive bias](inductive-bias.md).
- **Domain-specific constraints** (slide 63): NeRF's projection and light-transport equations, and a
  drug-interaction network's built-in structure, give "data plus structure, data plus constraints"
  (≈1:17:40–1:18:26).

The lecture closes on the theory answer revised: perhaps not the shortest program but one "short enough"
(slide 64), and a remark the lecturer recalls from Ilya Sutskever: "Deep nets are finite; that is enough.
Anything finite will look small once you have enough data" (slide 65, ≈1:19:12–1:19:58).

## Practical generalization (lecture 9)

Lecture 9 opens by recalling the random-labels result as the reason for "the (temporary) success of
hacking over theory" (slide 3): the VC-style bounds are vacuous for networks that can fit random labels,
"and yet, deep nets generalize" (≈1:34–2:23).

Its first example of a model that will *not* generalize is a shortcut. A chest-scan classifier reached
very high accuracy by reading an "R" marker in the corner of the image, which revealed something like the
hospital the scan came from, rather than looking at the tissue (slide 4, ≈5:30–7:01). It was right "for
the wrong reason because it won't generalize to other hospitals".

The lecture also argues that a training problem can be too easy to generalize from. A loss curve that
drops at once and goes flat is "Bad! Your data is too easy" (slide 15). Data augmentation and domain
randomization make the problem harder, and "the last one, which is the hardest to train on, is going to
generalize the best" (≈35:42). OpenAI's robot hand, trained with every physics and visual randomization,
took about ten times longer to reach the same training performance, and was the version expected to
work in reality (slide 18): "High train accuracy can mean problem is too easy. Add more data to make
problem harder." See [data augmentation](data-augmentation.md).

## The geometry of representations that generalize (lecture 12)

Lecture 12 looks for generalization in the shape of a representation. A NeurIPS 2020 competition asked "can we build
complexity measures that accurately predict the generalization of models?", and its "3 winning strategies look at: Geometry of
representation: consistency, separation" and "Robustness to perturbations" (slide 6, ≈7:45–8:30). Slides 7 and 8 (Chuang et al.,
2021) show why. A CIFAR-10 network trained with the true labels puts its training data into ten tight, separated clusters
("generalizes"). Trained on random labels, "you still will get clusters because the model can learn to memorize", but they are
"much less concise ... much less separable" ("cannot generalize"), so a small perturbation is "pretty quickly moving into another
color" (≈9:16–10:50). Concentration, separation and robustness to irrelevant perturbations become three of the lecture's five
properties of a good representation (slide 9); see [representation learning](representation-learning.md).

## Does an infinitely wide network overfit? (lecture 13)

[Lecture 13](13-representation-learning-theory.md) takes networks to infinite width, and a student asked whether that gives them
"a good opportunity to overfit" (≈52:12). The lecturer answers from the Gaussian process picture: among the random functions that pass
through the data there is "some probability of getting a really horrible function", but it is very small, "because things can't
deviate too much from the mean with very significant probability", and the mean itself is "a nice, smooth function" (≈53:47–54:36).
Sampling the weights with a very small standard deviation makes the distribution "collapse onto the mean" (≈55:23). And, echoing
[lecture 6](06-generalization-theory.md)'s theme, networks can be very overparameterized and still represent "lots of nice, simple
functions": "the fact that the network is very, very wide does not imply that you're going to overfit. It just implies that it's
possible to overfit" (≈55:23–56:09). See [Gaussian processes](gaussian-processes.md).

## Overfitting in a generative model (lecture 14)

Lecture 14 asks whether the filing cabinet is a good generative model: store every training point, and sample by "picking a drawer
at random" (slide 25). It achieves the highest training likelihood possible, a delta function on every training point, yet places
zero probability on new samples from the same process — "exactly the same as overfitting in classical machine learning"
(slide 26, ≈38:04–38:53). So the goal is "not to replicate the training data but to make *new* data that is *realistic*", measured
by the likelihood of held-out test data (slide 27), and "you have to control capacity or regularize in order to avoid the memorization
solution" (≈39:39). Early stopping on validation likelihood, regularizers and architectures with inductive biases all apply
(≈41:57–43:29). See [generative models](generative-models.md).
