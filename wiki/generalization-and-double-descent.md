# Generalization and double descent

Why do deep networks generalize when they have enough parameters to memorize their training data?
Lecture 1 poses this as one of the course's central questions (slides 50–52, ≈43:29–50:26) and
assigns it to **lecture 6, Generalization Theory**, and **lecture 17, Out-of-Distribution
Generalization**. The deck's banner says "Lecture 7" for generalization theory; in the recorded
schedule it is lecture 6 (see the [course map](course-map.md#the-decks-lecture-pointers)).
Covered so far: [lecture 1](01-introduction.md) only.

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
and out of distribution" — recorded lectures 6 and 17.
