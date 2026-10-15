# Lecture 24 — Inference Methods for Deep Learning

**Lecturer:** Phillip Isola ·
**Video:** [youtube.com/watch?v=mbgFTqKxR7A](https://www.youtube.com/watch?v=mbgFTqKxR7A) (83 min) ·
**Slides:** [`mit6_7960_f24_lec24.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)
(56 pages; transcribed slide by slide in [`raw/slides/24-inference-methods-for-deep-learning.md`](../raw/slides/24-inference-methods-for-deep-learning.md)) ·
**Transcript:** [`raw/transcripts/24-inference-methods-for-deep-learning.md`](../raw/transcripts/24-inference-methods-for-deep-learning.md)

The recording covers the whole deck: it ends at ≈1:22:19 on slide 55, with "So I will end there."

## What this lecture establishes

Almost all of the course has been about **training**: fitting a model to data. At test time, "it's almost entirely been just a forward
pass through the trained model. … But it turns out you can do much more than that at test time, and if you do, you'll get a lot of
advantage" (≈3:48). This lecture is about that "much more", which it calls **inference methods**: ways of spending computation on each
new query, by searching through the outputs a model can produce or by changing the model's behaviour on the test data itself. Isola
calls it "a brand new lecture" on a topic that "this year, has suddenly become quite the hot topic", built from ideas that are "very
old" (≈0:01–0:46), and gives its payoff as a guess at how "the very latest models, like OpenAI's o1, presumably work. We don't know
exactly how they work, but this lecture is my best guess" (≈0:46). The deck points to Sasha Rush's tutorial for more (slide 1).

The lecture has four movements. First, **search**: Sutton's two methods that scale, the trade between search and learning in chess, Go
and board-game scaling laws, and three search methods for autoregressive models, Best-of-N, beam search and chain of thought, steered by
likelihood or by any other score, including a verifier. Second, an aside from the lecturer's own work on **steering generative models
toward human preferences** by searching their latent space for images a model of human perception scores highly, and the "just ask"
alternative of putting the desired reward in the prompt. Third, methods that **change the model at deployment**: in-context learning,
the finding that it can act like gradient descent over the context, and **test-time training**, which does the gradient descent
explicitly. Fourth, **search and learning in a loop**: use search to find better answers, train on them, and use the improved model to
search better, as in reinforcement learning, AlphaGo, STaR and, by the lecturer's speculation, o1.

**Notation.** As in the [course notation](notation.md), $\theta$ is a model's parameters, $p_\theta$ the distribution the model defines,
and bold symbols are vectors or sequences: $\mathbf{x}$ an input or query, $\mathbf{y}$ an output sequence, $\mathbf{z}$ a generator's
latent variables. The slides set some of these plain; the slide file keeps the type each slide prints.

## Where this lecture sits

The lecture draws on much of the course. Its autoregressive models are those of [lecture 8](08-architectures-transformers.md) and
[lecture 10](10-architectures-memory.md), its example captioner combines [lecture 5](05-architectures-graphs.md)'s graph network with an
LSTM, and it asks of its own claim about greedy decoding what holds for [lecture 14](14-generative-models-basics.md)'s VAEs and diffusion
models. Its image scorer is CLIP (lectures 12 and 18), its latent search recalls lecture 18's "plugging pretrained models together", and
its self-supervised test-time training uses a masked autoencoder ([lecture 11](11-representation-learning-reconstruction-based.md)). It
comes back to "The Bitter Lesson" of [lecture 20](20-scaling-laws.md), and to the in-context learning, chain-of-thought prompting and
RLHF of Jacob Andreas's [lecture 21](21-language-models.md) ("Jacob's lecture", ≈6:05). Reinforcement learning, which the deck's map
treats as search improving learning, is a topic "which we didn't really have time to cover" (≈1:15:15).

## What "inference" means

The word is a trap: "the definition of inference in statistics is different than in machine learning" (≈1:33). Slide 2 sets the two side
by side. In **statistics**, inference is "figuring out properties of a data generating process from samples from that process", which
machine learning calls "learning" or "training": "the thing that they talk about in statistics as inferring the parameters that explain
the data-generating process — we call that training a model" (≈2:18). In **machine learning**, inference is "making predictions about
new datapoints using a trained model", which statistics calls "prediction". The lecture uses the machine-learning sense, "and that's the
one that's becoming — taking over. But statisticians will still complain about this, so be careful when you talk to statisticians"
(≈3:03).

## A map: training against inference

Slide 3 is the lecture's map, and it returns three times with different parts in bold. A dotted vertical line divides **Training**,
"what you do in the factory, at the company … before you actually come across new problems in the wild", from **Inference**, "what you
do at test time, in the wild, on deployment, on device. It's what a robot would do as it walks around this room" (≈3:48–4:35). The
lecturer stresses that the categories are his own: "These are not hard boundaries. A few of the terms could go under different
categories" (≈4:35).

![Slide 3: the Training and Inference halves of a map divided by a dotted line, with Pre-training, Post-training (shaded across the line) and Search boxes, their definitions and example methods, and a curved arrow from Search back to Pre-training through a box reading Reinforcement learning, STaR, self-instruct, self-play](../raw/images/24-inference-methods-for-deep-learning/slide-3.png)

*Slide 3 — the lecture's map: training on the left, inference on the right, post-training straddling the line, and the arrow along the bottom by which search feeds back into learning. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

The two sides have other names. Training is also "statistical inference" and "learning, amortized inference"; inference is
"prediction" and "thinking, reasoning, cognition" (slide 3). **Amortized inference** is the link between them: inference "is about
solving a problem, and training learns a set of weights that directly solves the problem without having to go through the step-by-step
reasoning process. So it's amortization of the more extended-thinking inference process" (≈4:35–5:20).

Below them, three boxes run from one extreme to the other:

- **Pre-training**, "Given data, learn a model or representation" (generative modeling, representation learning): "the most extreme
  version of training", and "what most of this course has really been about" (≈5:20).
- **Post-training**, a box shaded from training's colour to inference's that straddles the line. On the training side it is "Given a
  model and new data, update the model" (fine-tuning, RLHF); on the inference side, "Given a model and data during deployment, update
  the model or its behavior" (prompting, in-context learning, test-time training, continual learning, feedback control). Feedback
  control is the robot that "is going to use that feedback to change its behavior", one of the topics the lecture has no time for
  (≈6:52–7:39).
- **Search**, "Given a model and a query, find the best answer to the query" (Best-of-N, beam search, MCTS, chain of thought): "the most
  extreme side of inference, where we're not actually updating or changing the model's behavior at all", the domain of "more classical
  AI search methods, like tree search and breadth-first search and depth-first search" (≈7:39).

A curved arrow runs along the bottom of the map from Search back to Pre-training, through a box reading "'Reinforcement learning',
STaR, self-instruct, self-play, …". It is the lecture's last topic: "you can use search to learn new things, and you can use learning to
speed up your search … a very general-purpose name for all the methods that do that are basically reinforcement learning" (≈7:39–8:24).

## Search and learning

### The bitter lesson again

Slide 4 quotes the passage of Rich Sutton's "The Bitter Lesson" (March 13, 2019) that the lecture is built on: "One thing that should be
learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased
computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search
and learning." Isola's gloss: hand-designed systems "with clever inductive biases … just don't work that well. There's only, really, two
things that work. It's simple methods at scale. And the two that we know of are search and learning. So a lot of people have focused on
the learning part. That's the deep learning idea" (≈8:24–9:13).

### Train-time and test-time compute

What is happening "in industry right now" is "this move away from only focusing on train-time compute, but now using a lot of test-time
compute or search at test time" (≈9:13). Slide 5 shows two charts from OpenAI's o1 announcement, side by side under the labels
"Training" and "Search": o1's AIME accuracy ($\text{pass@}1$) against train-time compute, and against test-time compute, both on log-scale axes
with no numbers. The left chart is the familiar scaling law, "if I train on more and more data, I'm going to get better and better
accuracy, and there might be some kind of power law", with a slope of "maybe 0.5" (≈10:00). The right one is the new observation: scale
up "the amount of compute I apply at test time to do intelligent inference or search through my model", and "you also can get a power
law. And in fact, the slope in this particular case is even steeper than the train-time power law". So the shift is to deploying
"not just the forward pass through the model, but many, many passes through the model at test time" (≈10:46).

![Slide 5: two scatter charts of o1's AIME pass@1 accuracy, rising against train-time compute on the left, under Training, and against test-time compute on the right, under Search, both on log-scale axes without numbers](../raw/images/24-inference-methods-for-deep-learning/slide-5.jpg)

*Slide 5 — o1's accuracy rises with train-time compute (left) and, more steeply, with test-time compute (right). [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

### What search is: chess and Go

Search "is the way that AI used to be done" (slide 6, ≈10:46). Deep Blue's win over Garry Kasparov (slide 7, which prints no year; the
lecturer says "in 1994, maybe", and outside the course material the match was in 1997) used "very little or maybe no learning": it
"just did look-ahead search", following chains of possible moves "12, 10, 20, 30 moves in advance". Chess's branching factor, "however
many moves you can make at each position", turned out small enough to "enumerate all possible futures and just take the set of actions
that will lead to the best possible future" (≈11:35–12:21).

AlphaGo (slide 8, "[AlphaGo: Silver*, Huang* et al, Nature 2016]") "also heavily uses search. But AlphaGo uses both search and
learning". Go's branching factor "is much higher than the game of chess. And that's why they weren't able to do brute-force search like
they did in chess. Instead, they had to do search with heuristics — with learned heuristics on top" (≈12:21–13:07). The lecture
returns to how those heuristics are learned at the end.

### Trading search for learning

"There's these two things that just get better with more compute. One is learning, and the other is search. And sometimes they trade
off" (≈13:07). Slide 9, from Jones (2021), "Scaling Scaling Laws with Board Games", plots the trade-off for board games: test-time compute
against train-time compute, both on log axes, one downward-sloping curve for each of six playing strengths (labelled -1500 to -250 in steps of 250; the slide does not
define the numbers), each beside a dashed straight line of the same colour. The printed equation is

$$\log_{10}(\text{test}) = -1.2 \cdot \log_{10}(\text{train}) + 0.004 \cdot \text{elo} + 29$$

a straight line on the log axes for each strength: the more train-time compute, the less test-time compute the same strength needs. One can
solve chess "by taking a million chess games or a billion chess games and learning the mapping from the board
state to the best action", or "by just not having any pre-trained learning of optimal moves, but instead just, for every game position,
search ahead in time" (≈13:56). Between the two there is "a trade-off curve. It's saying that I can do more search, and then I can get
the same elo for less training, or I can do less search, and then I can get the same elo for more training. So there's two different
ways of being smart, search or learning". Which is better depends on which scales better, "and also, learning means that I don't, at
test time, have to do a lot of compute. So search is test time. Learning is train time" (≈13:56–14:43).

![Slide 9: a log-log chart of test-time compute against train-time compute for board games, six downward-sloping curves labelled -1500 to -250, each beside a dashed straight line, under the equation log10(test) = -1.2 · log10(train) + 0.004 · elo + 29](../raw/images/24-inference-methods-for-deep-learning/slide-9.jpg)

*Slide 9 — Jones's board-game scaling: for each playing strength, the more train-time compute, the less test-time compute it needs. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

## Search through an autoregressive model

The rest of the search section is about three methods for language models: Best-of-N, beam search and chain of thought (slide 10, the
map with those three in bold, ≈14:43). The first claim is that autoregressive models "can be improved by sampling multiple possible
completions and picking the best one" (≈14:43–15:32), and the reason lies in how they are trained.

The running example is a molecule captioner (slide 11): "a graph net that will process this", because "graph net makes sense on a
molecule", feeding "an LSTM that will spit out a sentence that describes" it, "A mild stimulant that enhances cognitive ability".
The molecule is caffeine, as a student answered (≈15:32). The slide "is a little bit old, so I was showing the example with an LSTM. You
can replace it with a transformer, but it doesn't really matter" (≈15:32); see [recurrent neural networks](recurrent-neural-networks.md)
and [autoregressive models](autoregressive-models.md).

### Training: maximum likelihood with teacher forcing

Training is generative modeling: "maximize the likelihood of the target data, which is just sentences from some corpus" (≈16:18). At
each step the network outputs a categorical distribution over the next token (the small bar charts of slide 12), and the
**max-likelihood objective** is to "maximize probability the model assigns to each target word":

$$\arg\max_\theta \thinspace \log p_\theta(\mathbf{y})$$

where $\mathbf{y}$ is the target sentence (slide 12 prints it plain). Written per step, each target word is a one-hot vector and the
objective is to "minimize cross-entropy between model outputs and one-hot encoded targets" (slide 13):

$$f^\ast = \arg\min_{f \in \mathcal{F}} \sum_{i=1}^{N} H(\mathbf{y}_ i, \hat{\mathbf{y}}_ i)$$

with $\mathbf{y}_ i$ the one-hot target, $\hat{\mathbf{y}}_ i$ the predicted distribution, $H$ the cross-entropy and $\mathcal{F}$ the
hypothesis space; the cross-entropy is "just doing the pointwise product between the one-hot vector and the predicted probability
vector" (≈16:18–17:05; see [softmax and cross-entropy](softmax-and-cross-entropy.md)).

![Slide 12: a molecule feeding a GNN and a row of eight LSTMs, each with a three-bar distribution over the next word, under the target words A mild stimulant that enhances cognitive ability END, with the max-likelihood objective](../raw/images/24-inference-methods-for-deep-learning/slide-12.png)

*Slide 12 — training the molecule captioner: a predicted distribution over the next word at each step, scored against the target sentence. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

Training uses **teacher forcing** (slide 14): "Condition each next word prediction on the ground-truth preceding word." "In
transformers, this stopped being a term. But in the LSTM era, we used to talk about the way of training this thing as teacher forcing"
(≈17:05). The target sentence is shifted by one, and at each step the network is told "the ground-truth previous word … whatever it was
in the training data" (≈17:05–17:53).

![Slide 14: the same network with an arrow feeding each LSTM from below, headed Teacher forcing, and the line Condition each next word prediction on the ground-truth preceding word](../raw/images/24-inference-methods-for-deep-learning/slide-14.png)

*Slide 14 — teacher forcing: during training, each step is fed the ground-truth previous word. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

### Inference: sampling, and the gap it opens

At test time there is no ground truth to condition on. The model samples a word from its predicted distribution, or takes the most
likely one, **greedy decoding**, and feeds its own output back in as the next input (slide 15: "A", then "strong", then "stimulant",
…). "You condition every next-token prediction on all the previous sampled tokens. You're just writing one word at a time" (≈18:40–19:26).

![Slide 15: the same network at inference, each sampled word, A strong stimulant that enhances cognitive ability END, looped back by a dashed arrow as the next LSTM's input](../raw/images/24-inference-methods-for-deep-learning/slide-15.png)

*Slide 15 — at test time each sampled word is fed back in as the next input, so the model conditions on its own outputs. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

This is "a deviation between how we'll use this system at test time and how it was trained" (≈17:53). Training maximizes the likelihood of
each next word given the ground-truth previous words, "which is not what you actually have at test time"; what one really wants is "to
maximize the likelihood of entire sequences sampled from the model" (≈17:53–18:40). And because the model's samples "might differ from
the ground-truth distribution", there is "this kind of drift": the model produces something a little different from the data, "and then
its next predictions will be of distribution [the captions' wording] compared to how it was trained" (≈19:26).

### Greedy decoding does not maximize likelihood

Slide 16 states the point: "Greedy sampling does not maximize likelihood!" An autoregressive model factorizes the probability of a
sequence $\mathbf{X} = (\mathbf{x}_ 1, \ldots, \mathbf{x}_ n)$ by the chain rule,

$$p(\mathbf{X}) = p(\mathbf{x}_ n \mid \mathbf{x}_ 1, \ldots, \mathbf{x}_ {n-1}) \cdots p(\mathbf{x}_ 2 \mid \mathbf{x}_ 1) \thinspace p(\mathbf{x}_ 1)$$

and the slide sets two ways of choosing a sequence against each other. The one we want is the jointly most likely sequence,
$\lbrace \mathbf{x}_ 1^\ast, \ldots, \mathbf{x}_ n^\ast \rbrace = \arg\max_{\mathbf{X}} p(\mathbf{X})$. Greedy decoding instead takes
$\mathbf{x}_ 1^\ast = \arg\max_{\mathbf{x}_ 1} p(\mathbf{x}_ 1)$, then the most likely second word given that one, and so on, each factor
maximized in turn. The greedy sequence "is not going to equal the optimal joint configuration. … You can construct counterexamples"
(≈20:13–21:00). The slide uses the same stars for both; asked whether the equality holds, the lecturer agreed that the two argmax
solutions differ, "I probably should have had a prime or something to indicate that those were different values" (≈21:00–21:45). (The
slide also prints the last factor's subscript 1 in bold, unlike the others; the slide file flags it.)

The slide ends with a question, "(What about for VAEs? What about for diffusion models?)", which the lecturer partly answers (≈21:45–
24:05). Diffusion models have "the same exact thing": sampling is "also iterative step by step", each step maximizing "the conditional
probability of slightly less noisy given slightly more noisy", which "won't necessarily result in a sequence of denoising steps that
maximizes the probability at the end of the sequence for the same kind of reason". A VAE is different: "it's not really sequential.
There's just one step. There's just encode, decode", so "there's no sequential process where my sampling could be deviating from the most
likely sample" (≈22:31–23:19). He offers the diffusion case as an open topic: there is "a lot of work on doing intelligent inference
that's better than greedy sampling for language models", "but less for diffusion models" (≈24:05). See [diffusion
models](diffusion-models.md) and [variational autoencoders](variational-autoencoders.md).

## Three search methods

Most language models do not sample greedily: "you just take random samples", and "a random sample is not necessarily going to be the most
likely sample", so the caption a model gives may be "a random description that is low probability just by chance" (≈24:05–24:52).
Better inference means finding higher-probability samples.

### Best-of-N

**Best-of-N** (slide 17) is "really simple": "Sample N sequences iid and pick the sequence with highest likelihood." The samples are random,
not maximum-likelihood, "but you can evaluate the likelihood of any sequence using the chain rule", multiplying the conditionals:

$$p_\theta(\mathbf{y}_ 1, \ldots, \mathbf{y}_ T \mid \mathbf{x}) = \prod_i p_\theta(\mathbf{y}_ i \mid \mathbf{y}_ 1, \ldots, \mathbf{y}_ {i-1}, \mathbf{x})$$

where $\mathbf{x}$ is the input (here the molecule) and $\mathbf{y}_ 1, \ldots, \mathbf{y}_ T$ the output tokens (≈24:52–25:38). The slide
scores three samples, "A stimulant that enhances cognitive function." at 0.7, "A stimulant that improves thinking." at 0.5 and "A
chemical that makes you feel more awake." at 0.8, and boxes the last as the winner.

![Slide 17: Best-of-N, three sampled captions scored by the model's likelihood at 0.7, 0.5 and 0.8, the last boxed in green, under the chain-rule formula for a sequence's likelihood](../raw/images/24-inference-methods-for-deep-learning/slide-17.png)

*Slide 17 — Best-of-N: sample several captions, score each by its likelihood under the model, keep the best. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

Three questions followed (≈26:24–28:47). Sampling draws from the categorical distribution, with a **temperature** that "can scale how flat
or spiky the distribution is": at temperature 1, "you're taking unbiased samples from the data distribution", and lower settings give
"mode-seeking behavior". Trying "all possible combinations" would find the true maximum-likelihood sample "but that would be incredibly
expensive", so one samples with the temperature turned down a little, but not to 0, where "it's going to take the greedy sample, and
that might not maximize probability". And a student's idea of editing a sample word by word to improve it is "the whole space of
optimization and statistical inference": MCMC, Gibbs sampling, hill climbing, or asking a language model to "rewrite this in 10
different ways". Still, "to my knowledge, this Best-of-N sampling is about as good as anything right now. … It's rejection sampling,
essentially" (≈28:47).

### Beam search

**Beam search** (slide 18), "a form of tree search", keeps several candidates at once: "at every step of your autoregressive generation,
you maintain a memory of the top-k best next words", expanding the tree "with some branching factor k", and at the end takes "the path
along that tree that has max probability" (≈29:32). The slide's tree grows "A" into "strong" and "mild", "mild" into "stimulant" and
"psychostimulant", "strong" into "stimulant" and "neural", and marks the path "A mild stimulant". Of the methods, "Best-of-N seems to be
the simplest and also works about as well as anything, as far as I can tell right now. And beam search is maybe the most popular,
slightly more advanced method" (≈30:19). Beam search for sequence models also appears in [lecture 10](10-architectures-memory.md).

![Slide 18: beam search as a tree of samples from A to strong and mild, then to stimulant, neural, stimulant and psychostimulant, with the path A mild stimulant in orange](../raw/images/24-inference-methods-for-deep-learning/slide-18.png)

*Slide 18 — beam search keeps the top-k continuations at each step; the orange path is the one picked. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

### Chain of thought: learned search

**Chain of thought** (slide 19, "[Wei et al. 2022]") is "an emergent or learned search" (≈30:19). The lecturer's example, "from ChatGPT
last night", asks the model itself to "use beam search to solve the following problem. Write out your tree of samples then pick the best.
Here's the problem: What is 4\*5+10". Rather than running beam search over the model's completions, "maybe if I prompt you, you can do that
beam search yourself because you've observed, in your training corpus, examples of this kind of behavior. Or maybe you've been tuned with
reinforcement learning" (≈31:05–31:51). The response, pasted on the slide, draws a tree with two branches, "20 + 10" leading to 30 and
"14 \* 5" leading to 70, and concludes that "the best solution is 30". So chain of thought "is like doing a reasoning or search process
that could be written with a classical algorithm, but it's the LLM doing it itself and writing the intermediate steps into the text
space, into its context, and then autoregressively continuing on" (≈31:51). Lecture 21 introduced chain-of-thought prompting; see
[foundation models and prompting](foundation-models-and-prompting.md).

![Slide 19: a prompt asking an LLM to use beam search on What is 4*5+10, and the pasted chat response, a text tree with branches 20 + 10 to 30 and 14 * 5 to 70, concluding that the best solution is 30](../raw/images/24-inference-methods-for-deep-learning/slide-19.jpg)

*Slide 19 — chain of thought as learned search: asked to, the model writes out its own tree of samples and picks one. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

## Searching for other scores

Everything so far searched for higher likelihood, "but it turns out you can do the same exact thing, but not for likelihood, but for any
kind of scoring function" (≈32:38). Slide 20 scores three captions of the caffeine molecule for **fun-ness**: "The molecular diagram of
caffeine" scores 0.1, "Nature's tiny cheerleader, jumping into your bloodstream with pom-poms" 0.7 ("The language model gave me that. I
didn't come up with that"), and a row of four emoji 0.8, the winner (≈32:38–33:25). Slide 21 rescores the same three for **scientific
accuracy** (1.0, 0.3, 0.1), and the dry caption wins. The scorer might be "people voting online": rating samples by humans and keeping
the best "is the basis of what's called reinforcement learning from human feedback" (≈33:25; RLHF is lecture 21's last topic).

![Slide 20: three captions of caffeine, a dry one, a cheerleader metaphor and four emoji, scored for fun-ness at 0.1, 0.7 and 0.8, the emoji row boxed in green](../raw/images/24-inference-methods-for-deep-learning/slide-20.png)

*Slide 20 — the same search with a different score: for fun-ness, the emoji win. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

What can the scorer be? "It can be absolutely anything. So something that takes in a sentence and gives you a scalar out. It could be
another model that you've trained. It could be a human brain. It could be an animal brain. It could be a formal proof checker" (≈34:14).

### Verification

A popular scorer is a **verifier** (slide 22), which "usually has the connotation that you're going to actually check that the output is
correct using some formal process", such as "a compiler or a mathematical proof checker" (≈34:14). Asked to "Write me python code to
compute the fibonacci sequence", the language model returns two programs. One, `fibonacci_3`, initializes `a, b = 1, 1` with the
comment "Incorrect initialization"; the other, `fibonacci_2`, is the recursive definition. A verifier that "knows what the Fibonacci
sequence is" runs each and marks the first "incorrect" and the second "correct", and "I can just do Best-of-N in this fashion"
(≈34:14–35:00).

![Slide 22: an LLM asked for Python code to compute the Fibonacci sequence returns two programs, and a verifier marks fibonacci_3 incorrect and fibonacci_2 correct](../raw/images/24-inference-methods-for-deep-learning/slide-22.jpg)

*Slide 22 — verification: checking a program's output is easier than writing the program. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

The slide's key idea: "verification is easier than generation". "Checking that I correctly solved a math problem or I correctly made code
that has some desirable property is just fundamentally easier than coming up with the code"; "proving a theorem is fundamentally harder
than checking that a proof is correct" (≈35:00–35:45). The lecturer ties this to problems "checkable in polynomial time" for which "we
expect that they don't have a way of solving the problem in polynomial time, as long as P does not equal NP" (the captions garble the
class's name), and to prime factorization: finding the factors of a big number is hard, "but checking that some set of factors multiply
to equal the number is an easy problem" (≈35:45–36:34).

## Steering generative models toward human preferences

The same search works "not just for language models and autoregressive models, but also for other types of generative models", and the
lecturer takes "a little aside into some work that I did, now quite a few years ago" (slide 23, ≈36:34).

### Objectives other than imitation

Slide 24 lists "Great objectives for AI systems": "1. Given lots of wonderful data, imitate it (supervised learning, SSL, generative
models, etc)", then "2. …". The first is "like, 90% of the class" (≈37:19). Students suggested winning a game and interpretability, and
the lecturer noted that such goals "usually are handcrafted" and "go under this area of reinforcement learning in some way or another …
you have some score that you design for that, and you optimize that score" (≈37:19–38:04). His suggestion (slide 25): "Make imagery that
people find meaningful", "memorable/aesthetic/evocative/…" (≈38:04).

### Searching a generator's latent space

Slide 26, "Generating data that optimizes a scoring function", sets up the search for an image generator with latent variables
$\mathbf{z}$, which "maps z through g of z to an output image": "a GAN or the decoder of a VAE, or it could be a diffusion model where z is
the noise" (≈38:51). The score comes from CLIP: a text encoder turns the prompt, "What is the answer to the ultimate question of life,
the universe, and everything?", into an embedding $\mathbf{e}_ 1$, an image encoder turns the generated image into $\mathbf{e}_ 2$, and
the search optimizes $\mathbf{z}$ to maximize the dot product $\mathbf{e}_ 1 \cdot \mathbf{e}_ 2$, which CLIP's training makes largest
"so that the caption that matches an image will have the maximal dot product" (≈39:38). (The slide's output image is an almost uniform
purple-grey texture.)

![Slide 26: a text encoder embeds the prompt as e1, an image generator turns a latent z into an output image that an image encoder embeds as e2, and z is optimized to maximize e1 · e2](../raw/images/24-inference-methods-for-deep-learning/slide-26.png)

*Slide 26 — searching a generator's latent space for the image whose CLIP embedding best matches the prompt (CLIP + VQGAN). [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

"This is kind of constrained optimization. I'm not searching over the space of all possible pixels. … I'm only searching through the
latent space." Keeping $\mathbf{z}$ at high prior probability ("maybe I just search over the unit hypersphere" of a Gaussian prior)
keeps the output "within the manifold of high-probability images according to the image generator" (≈40:23). The method, labelled
"[e.g., CLIP + VQGAN]" on the slide, drove "the first batch of text-to-image generative models that were really impressive"; it "was very
popular for a year or two in, like, 2020 [later, "or maybe it was 2021"], and then it got replaced by diffusion models" (≈40:23–41:10,
43:30). The slide links the lecturer's code. It is test-time compute in the sense of slide 5: "a costly optimization process", solving
"a search process at test time that hadn't yet been amortized into a feed-forward generative model", and it "made this weak model much
stronger" (≈42:43–43:30). See [generative models as data](generative-models-as-data.md) and [contrastive
learning](contrastive-learning.md) for CLIP.

A student asked how to trade two objectives, say likelihood and creativity. The field is **multi-objective optimization**: one can sum
the objectives with weights, "but sometimes … one objective just scaled very differently than another one", and a key concept is the
**Pareto frontier**, "where I'm maximizing every objective as much as possible" without making the others worse (≈41:58–42:43).

### A model of human perception: memorability

Replace the scorer with "a model of humans. Sometimes it's called a reward model" (≈43:30). Slide 27 puts "Human perception" (a brain
scoring an image) above a learned "Model of human" in the same pipeline: optimize $\mathbf{z}$ so the generated image maximizes the
model's score. The lecturer's example is **memorability**, from his work of around 2018 (the slides date GANalyze to 2019 and the
memorability data to 2011 and 2015): "we have a model of how much people will remember an image or not" (≈43:30–44:16). Slide 28 shows
how it was measured: people watched a stream of images and said whenever they saw a repeat, with "vigilance" repeats a few images apart and
"memory" repeats far apart; and it shows forgettable images (landscapes) beside memorable ones (people). "People are super memorable, so
that was the main finding" (≈45:03).

### GANalyze

**GANalyze** (slide 29, Goetschalckx, Andonian, Oliva and Isola, ICCV 2019; the slide's author line prints "Goetschalkx", its
photograph caption "Goetschalckx"), "work that Lore and Alex led", couples a generator, an
assessor of memorability and a search over the generator's latent space (≈45:03). Slide 30 gives the architecture: a "Transformer"
module moves the latent $\mathbf{z}$ by an amount set by a dial $\alpha$, to $T_\theta(\mathbf{z}, \alpha)$; a generator ("e.g., BigGΛN", as the slide prints it, its A without a
crossbar, for BigGAN) renders the image $G(T_\theta(\mathbf{z}, \alpha), \mathbf{y})$ for class $\mathbf{y}$; and an assessor (MemNet, for example)
scores it, $A(G(T_\theta(\mathbf{z}, \alpha), \mathbf{y}))$. "This is all differentiable. So this type of search process is, in some
ways, easier than with autoregressive language models", which need "tree search, these symbolic search methods": "you just can do
gradient descent to do the search" (≈45:49).

Turning the dial on a dog photograph (slide 30, "Probe into human perception") raises the assessed memorability from 0.58 at
$\alpha = -0.2$ to 0.77 at $\alpha = 0.2$: "the eyes get really big. That's making it more memorable. You can actually measure that, and
people do remember that photo better", while the less memorable version is "grainy and hard to interpret and small" (≈45:49–46:36).
Slide 31 repeats the experiment for **aesthetics** and **emotional valence** on a dog, a snake and a beach. "Simple symmetric structures,
big, centered objects are easy to remember"; "the most aesthetic version of the snake … has this depth-of-field effect … at a 3/4 angle";
the most aesthetic landscape is "really saturated", while the most memorable "will make everything big"; "the most aesthetic dog is
really cute, but doesn't have this big head" (≈47:22).

It was meant as "a probe into human perception", "more like a psychology paper", but "this type of method has become more popular … under
names like reinforcement learning from human feedback" (≈46:36, 48:08). From the questions: the main object, being centred and being
close up matter most (≈48:55); the assessor is "simply a neural net that is trained to predict — to regress how likely a human is to
remember this image, and it's fit to human data" (≈48:55–49:41); and the study modelled only what people share, not "individual
differences" of culture or age, which "would be a really interesting future work" (≈49:41–50:27). The lesson: "if you have a really good
generative model, then you don't have to just use its raw outputs. You can actually optimize and search … to maximize some function of
interest" (≈50:27–51:14).

## "Just ask"

There is "a new paradigm to do all of this stuff", which "Eric Jang nicely described in this blog post as the 'just ask' paradigm"
(slide 32, linking evjang.com, ≈51:14). "Rather than doing beam search, you can say — just ask the model to do beam search." Slide 32:
"Chain-of-thought is 'just ask for symbolic search'"; "Can we do 'just ask for continuous reward optimization'?" (≈51:14–52:01).

"Interestingly enough, this works." Slides 33 to 38 are the lecturer's DALL-E experiments ("DALL-E 1, I believe", ≈52:01): ten paintings
for "A painting of a mountain next to a waterfall." (slide 33), then the same prompt with "beautiful", then with one, two, six and finally
twenty-two "very"s on the next five slides, ending at slide 38 (the screenshots print "An very" from the third on). Each step is "a little more
beautiful. It's like I steered it in the direction that the model has interpreted as being related to beauty and aesthetics"; with many
"very"s "it's definitely more beautiful … I think it maxed out around there" (≈52:01–52:46). The model "did learn this conditional
distribution of conditioned on the reward I want to achieve, what should the output look like? I don't actually have to necessarily
search explicitly for reward-maximizing images" (≈52:46).

![Slide 33: a text-to-image interface with the prompt A painting of a mountain next to a waterfall and a grid of ten fairly plain, dark paintings of waterfalls](../raw/images/24-inference-methods-for-deep-learning/slide-33.jpg)

*Slide 33 — the plain prompt, "A painting of a mountain next to a waterfall." [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

![Slide 38: the same interface with the prompt's "beautiful" preceded by twenty-two "very"s, and a grid of ten saturated, dramatic paintings of waterfalls](../raw/images/24-inference-methods-for-deep-learning/slide-38.jpg)

*Slide 38 — the same prompt with twenty-two "very"s before "beautiful" (some garbled in the page's text). [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

Slides 39 and 40 show Jon Barron's "Make it more" posts: a "Level 2 Reuben", a "Level 1e15 Reuben" towering over a city, and a "Level
1e40 Reuben" in outer space, where such sequences "always [devolve] into this like picture of the universe" (≈53:33). But "these learned
abilities — they're not like the abilities that you get by doing well-understood search algorithms", since they depend "on the data
distribution on which it was trained. It's not going to be something we have guarantees or proofs for": asked for a "1e negative 40
Reuben", the model "doesn't make a tiny, tiny Reuben. It, instead, makes this microscope because it has that connotation" (≈53:33–54:21).

A student asked whether a model must be trained with a search method in mind. Generic search such as Best-of-N will improve "any
autoregressive model … It's very safe". But a prompted search like "let's think step by step" "will depend a lot on how it was trained":
it was "an empirical discovery that that works well, but you could also design your data set in conjunction with the search algorithm"
(≈54:21–55:55).

## Changing the model at deployment

"Now let's go into inference methods that actually change the model's behavior, maybe update its weights" (≈55:55): methods by which a
self-driving car "is getting smarter and smarter as it drives around the city. So it's actually a different model at the end of the trip
than it was at the beginning" (≈55:55). Slide 41 is the map with in-context learning and test-time training in bold.

### In-context learning, again

Slide 42 repeats lecture 21's sentiment example of **in-context learning** (cited "[Example from Zhao et al. '21]"): two reviews with
their answers, $\mathbf{x}^{(1)}, \mathbf{y}^{(1)}$ and $\mathbf{x}^{(2)}, \mathbf{y}^{(2)}$, then a query review $\mathbf{x}$ with its
answer left blank; the language model completes it with "Positive", modelling
$P(\cdot \mid \mathbf{x}^{(1)}, \mathbf{y}^{(1)}, \ldots, \mathbf{x}^{(N)}, \mathbf{y}^{(N)}, \mathbf{x})$. The natural completion is "to follow the pattern. And what is the name of following a
pattern of x, y example pairs? The name of that is supervised learning. So in-context learning is just the definition of supervised
learning. It's learning to do supervised learning" (≈56:42–57:30). See [meta-learning](meta-learning.md).

### Is in-context learning gradient descent?

A paper of von Oswald et al. (ICML 2023), "Transformers learn in-context by gradient descent" (slide 43), asks: "Is in-context learning a
learning algorithm like the ones we know, like gradient descent on a supervised regression loss? Or is in-context learning something
fundamentally new?" (≈57:30). Let $f_\theta$ be a transformer, $\mathbf{x}$ a query and $\mathbf{c}$ the context of example pairs. Feeding
the context in with the query can be rewritten as a context-free model with other parameters,

$$f_\theta(\mathbf{x}, \mathbf{c}) = f_{\theta'}(\mathbf{x}),$$

"baking it, instead, into a set of weights that has the same effect" (≈58:15–59:02). Such a $\theta'$ exists in principle, since "these
are universal approximators". The question is how it relates to $\theta$, and the paper "shows a case where"

$$\theta' = \theta + \lambda \nabla_\theta \mathcal{L}(f_\theta(\mathbf{x}), \mathbf{c})$$

with $\lambda$ a step size and $\mathcal{L}$ a loss on the context examples. (The slide prints a plus sign; a gradient-descent step
subtracts the gradient, as in [gradient descent](gradient-descent.md), so either $\lambda$ is meant to be negative or the sign is a slip, and
the slide does not say which.) That is, "in-context learning, in certain cases, can be expressed as gradient descent over the context examples, to update the final
mapping from query to answer" (slide 43). The new transformer, "when not showing any context at all, just via the weights, … will output
the same result as the original transformer with that context" (≈59:02–59:48). The result is for "one layer, linear self-attention. It's
not the full picture, but they showed this for a simple toy case", and the lecturer "didn't dig too much into this paper" and does not
remember whether the argument is empirical or a proof (≈59:48–1:00:34).

### Why not do gradient descent over the context?

"This result raises a tantalizing possibility: instead of doing ICL over $\mathbf{c}$, why not do gradient descent over $\mathbf{c}$?"
(slide 44). The possible advantages: "ICL is learned learning. It isn't guaranteed to work. And it often doesn't work. Gradient descent is
not learned. It is guaranteed to improve performance on the in-context examples"; and "Gradient descent can leverage more test-time compute
than ICL. You can run it forever" (slide 44). In the lecture's words, gradient descent "if I take small enough step sizes, will always find
a local minimum", while for in-context learning "there's no guarantee that if I show it more and more context, it will get better and
better"; and it is "an iterative optimization algorithm that does forward and backwards … many, many, many times", where in-context
learning is one forward pass (≈1:02:04–1:02:51). The possible disadvantage: "ICL could be a smarter learning algorithm than gradient
descent" (slide 44), though "the theory papers suggest that maybe it's not that much different … but that's still only for simple toy
models" (≈1:03:37).

Asked what the gradient descent would be on, the lecturer said: "gradient descent on queries without context and trying to use the
context examples just to improve the mapping from the query directly to the answer without the context" (≈1:04:24). A student's point,
that training cannot prepare a model for everything, is "exactly why people like in-context learning and test-time compute": at test time
"you, on the fly, learn a specialized skill" (≈1:05:12).

### Test-time training

**Test-time training** (slide 45, "[Sun et al. ICML 2020]") is to "Update the weights of your model as a function of the test query
(/queries)". "If the test query includes few-shot examples,
$\lbrace \mathbf{x}_ 1, \mathbf{y}_ 1, \mathbf{x}_ 2, \mathbf{y}_ 2, \ldots, \mathbf{x}_ q, ? \rbrace$, then this should clearly improve performance on those examples" (slide 45). But the test query may be just an
$\mathbf{x}$, or a set of unlabelled examples $\lbrace \mathbf{x}_ 1, \mathbf{x}_ 2, \ldots, \mathbf{x}_ q, ? \rbrace$: "a single photo at
test time, or maybe a movie at test time, or maybe a book" (≈1:05:57). "It turns out you can", by **self-supervised test-time training**
(≈1:06:46).

Sun et al.'s system (slide 46) splits a classifier into a shared trunk $f$, which maps an image $\mathbf{x}$ to features $\mathbf{z}$, and
a task head $g_{\text{task}}$ that outputs the label ("Bird"). The trunk is also trained with a self-supervised head
$g_{\text{self-sup}}$, here masked autoencoding: "I remove some patches, and I try to predict the missing patches" (see
[self-supervised learning](self-supervised-learning.md)). At test time "I might have a pre-trained f, and now I'm going to try to fine tune
it on the test example with a self-supervised loss. And then I will simply have a better representation", which "can actually improve
your performance on correctly identifying that this is a bird" (≈1:06:46–1:07:35). It is "a very general-purpose thing": "you go to Mars.
You want to, on the fly, adapt your representation", "and you don't have to have labels" (≈1:07:35–1:08:20). Asked, the lecturer said
only $f$ is updated and he believes $g_{\text{task}}$ stays frozen; on forgetting, "I'd have to look into the paper", but "if you don't
change f that much, you just perturb it slightly, there's a regime in which it works" (≈1:08:20). Compare [domain
adaptation](domain-adaptation.md) and [out-of-distribution generalization](out-of-distribution-generalization.md).

### Test-time training on ARC

A paper "from about a week or two ago", from MIT with Jacob Andreas and Yoon Kim as the professors, "The Surprising Effectiveness of
Test-Time Training for Abstract Reasoning" (slides 47 and 48), applies supervised test-time training to the **ARC** challenge, "supposed to
be a really hard challenge for deep learning to solve" (≈1:09:07). Each ARC task gives a few input–output grids and a test input; slide 47's
example shows a model fine-tuned in advance ("FT") getting it wrong and test-time training ("TTT") getting it right. Humans "can get 50% or
maybe higher", experts "80%, 90%" (≈1:09:07–1:09:52). The method: "pre-train a model to be generally good at this type of pattern
recognition problem", then, "given the few in-context examples, you update the model to do better at those in-context examples, gradient
descent on these supervised examples", and apply it to the test query. Fine-tuning on "the training set of similar problems" instead
"doesn't work as well because the in-context examples are the most relevant" (≈1:10:40).

Slide 48: "61.9% performance on ARC, which is current SOTA"; "Mostly with an 'old' method (GD on supervised examples; but examples that are
highly relevant to each test query)"; "Raises question: do we need fancy new search methods? Or is GD still enough?" Beside it is the ARC
Prize banner, "AGI PROGRESS HAS STALLED. NEW IDEAS ARE NEEDED." "Maybe a little bit of a new idea is needed, but actually, they got the
highest performance that's been reported so far, and they did it with old methods" (≈1:11:25). In answer to questions: no transformer
processes the examples as context; each example is a training pair for gradient descent (≈1:12:11); and it is not cheating, because the
challenge is defined as applying a pattern from a few examples to a query whose answer is never seen (≈1:12:57). At "62%" (spoken; the
slide prints 61.9%), "maybe we're not quite there yet, but that's interesting that you can go pretty far" (≈1:13:42).

## Search and learning in a loop

The last part of the lecture is the bottom arrow of the map (slide 49, with "Reinforcement learning" and STaR in bold): "use search and
inference and optimization at test time to come up with better solutions, and then distill those solutions back into … the base model
itself. So fit to those optimized solutions and then repeat" (≈1:13:42). "It's an old idea. It's been around since the beginning of AI"
(≈1:13:42–1:14:29).

### Search to improve learning

Slide 50 repeats the verifier of slide 22 under a header arrow from Search to Training: find better answers "than the LLM output by
default", by Best-of-N, tree search or verification. Then "take that better answer and call it a training pair for supervised learning"
(slide 51): the prompt $\mathbf{x}$ and the verified program $\mathbf{y}$, with more pairs like it, go through supervised learning to give an
updated model, "LLM′" (≈1:14:29). "Sometimes people call these self-improvement or recursive self-improvement" (≈1:14:29). Strictly, "the
information came in through the verifier", so it is "improving with an external verifier"; but since "it's easier to learn to verify than
to generate", the verifier can itself be learned, "and then it's purely self-improvement" (≈1:15:15).

![Slide 51: the Fibonacci prompt x and the verified program y as a training pair, with more pairs below, fed into supervised learning to give an updated model LLM′](../raw/images/24-inference-methods-for-deep-learning/slide-51.jpg)

*Slide 51 — search improving learning: a verified answer becomes a supervised training pair. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec24.pdf)*

### Learning to improve search: reinforcement learning

"You can also do it the other way around … use learning to improve search. And this is basically the fundamental concept in
reinforcement learning" (≈1:15:15). Slide 52's loop, on a video game: "1. Search for random action sequences that lead to high rewards
(winning). 2. Train a network (policy) that will tend to do those winning moves. 3. Repeat, guiding search using the policy to explore
intelligently." The search finds that jumping over the pipe scores; then "fit a neural network to directly predict that sequence … It's
called a policy that directly maps from the state to the optimal action" (the slide's policy $\pi$ puts most of its probability on "up");
then the search is guided "to be preferentially doing things that the neural network thought were good" (≈1:16:05–1:16:52). "If you have
some score in the game, like a reward function in reinforcement learning, then this will provably get to maximal reward under some
conditions" (≈1:16:52).

"This is how AlphaGo works" (slide 53, its four panels "Selection", "Expansion", "Evaluation" and "Backup"): "You do some tree search over
the possible moves you can make. You find the best move according to the search. And then you back up that solution into a heuristic,
which is able to predict — without expanding the tree … if I were to expand the tree, I would have found this solution" (≈1:17:39–1:18:24).
A student asked how this carries over from games, where self-play and a reward come with the environment, to reasoning in language; the
answer is STaR.

### STaR

**STaR** (slide 54, "[Zelikman\*, Wu\*, Mu, Goodman, 2022]") brings the loop to language models. A language model is prompted to produce "a
kind of chain of thought … they call that a rationale — and then produce an answer", and a verifier checks the answer. "If I get the
answer right, then I will reinforce that behavior": the question, rationale and answer are added to the data the model is fine-tuned on,
"to preferentially produce these good rationales, these good chains of thought that lead to the good answer" (≈1:18:24–1:19:10). The
slide's diagram also sends wrong answers through a "Hint" and a "Rationalize" step, which the lecturer mentions but does not explain ("I
won't talk about the bottom branch", ≈1:19:10). Its example: "Q: What can be used to carry a small dog?", answered "basket (b)" with the
rationale "Baskets are designed to hold things." The rationales start from "some supervised rationales or some seed" (he thinks "only … a
handful"), after which "the good rationales that led to the correct answer get added to the training set" (≈1:21:30–1:22:19).

### o1?

The lecture ends on "speculation", since "this is the latest OpenAI thing": "this is what a lot of people seem to think that this o1 model
does", and Sasha Rush "gives this as one of his candidate explanations for o1, but I feel like it's got to be something basically like
this" (≈1:19:10). Slide 55's four steps:

1. "Search for CoTs that solve the problem (randomly sample different chains)": "it's like beam search. I'm looking over all the possible
   different ways the language model could start reasoning and thinking aloud until it comes up with the answer", and keeping those that,
   "according to some scoring function, … actually got the right answer" (≈1:19:57).
2. "Finetune the LLM to preferentially output CoTs that led to high reward." The verifier "could be a learned verifier. That's another
   language model. Or it could be some kind of humans on the internet … or it could be a compiler or a proof checker. … We don't know"
   (≈1:19:57–1:20:43).
3. "Repeat, now the LLM outputs better CoTs", which "makes my search converge on a narrower space of really good chains of thoughts"
   (≈1:20:43).
4. "At deployment time, you can choose how much search to do on top of your fine-tuned LLM": "that's why if you run o1 or any of the other
   models that do this type of search, they will take longer" (≈1:20:43).

"It's just reinforcement learning applied to reasoning with a language model" (≈1:20:43). Asked how more rounds of search could be "more
efficient as they claim", the lecturer said the claim is "higher accuracy": search on every query finds better answers, distilling them
leaves a base model "just as efficient", and on top of it "you can always do as much search as you want"; it is "a way of using more
compute to get better answers" (≈1:21:30). "That's the latest, greatest thing in AI. We don't know exactly how it works, but it's very
much a cool frontier" (≈1:22:19).

## Summary

- In machine learning, **inference** means using a trained model on new data; statistics calls that prediction, and calls training
  inference (slide 2). The lecture's map runs from pre-training through post-training to pure search, with reinforcement learning as the
  loop from search back to learning (slide 3).
- **Search and learning** are the two methods that scale (slide 4); they trade off (slide 9), and o1's accuracy grows with test-time
  compute as it does with train-time compute (slide 5).
- An autoregressive model trained by maximum likelihood with teacher forcing is used differently at test time, and **greedy decoding does
  not find the most likely sequence** (slides 12–16). **Best-of-N**, **beam search** and **chain of thought** search for better sequences
  (slides 17–19); Best-of-N is "about as good as anything right now" (≈28:47).
- Search can maximize **any score**: fun, accuracy, human ratings, or a **verifier**, since verification is easier than generation
  (slides 20–22).
- A generator's latent space can be searched, by gradient descent, for images a model of human perception scores as more memorable or
  aesthetic (GANalyze, slides 26–31), or the model can be **asked** for more reward in the prompt (slide 32 and the eight after it).
- **In-context learning** can act like gradient descent over the context (slide 43); **test-time training** does the gradient descent
  explicitly, with self-supervision (slide 46) or the task's own examples, reaching 61.9% on ARC (slide 48).
- **Search improves learning and learning improves search**: verified answers become training data (slide 51), a policy guides search
  (slides 52–53), STaR fine-tunes on rationales that reached the right answer (slide 54), and o1 plausibly does the same with chains of
  thought, plus search at deployment.

## Pointers to other lectures

The deck prints no lecture number but its own, on the title slide ("Lecture 24: Inference methods for deep learning"). The recording
refers back to "The Bitter Lesson … that I mentioned a few lectures ago" (≈8:24; [lecture 20](20-scaling-laws.md)), to "Jacob's lecture"
for RLHF and in-context learning (≈6:05, 6:52) and to Jacob Andreas's talk for chain of thought (≈30:19; [lecture
21](21-language-models.md)), and to in-context learning as "we saw that before" (≈57:30). It points ahead to the next class session,
Thursday's office hours for final projects, "There won't be any talk or any presentation" (≈1:33). See the [course map](course-map.md).

## See also

- [Autoregressive models](autoregressive-models.md) — the chain-rule factorization, teacher forcing, sampling, greedy decoding and beam search.
- [Language models](language-models.md) — lecture 21's language models and RLHF, which this lecture's search and STaR build on.
- [Foundation models and prompting](foundation-models-and-prompting.md) — in-context learning, chain of thought, "just ask", and CLIP+GAN.
- [Meta-learning](meta-learning.md) — in-context learning as learning to learn, and whether it is gradient descent.
- [Scaling laws](scaling-laws.md) — train-time scaling, and this lecture's test-time and board-game counterparts.
- [Self-supervised learning](self-supervised-learning.md) — the masked-autoencoding loss of test-time training.
- [Generative models as data](generative-models-as-data.md) — searching and steering a trained generator.
- [Diffusion models](diffusion-models.md) and [variational autoencoders](variational-autoencoders.md) — the slide 16 question.
- [Softmax and cross-entropy](softmax-and-cross-entropy.md) — the training loss, and sampling temperature.
