# Meta-learning

**Meta-learning** is "learning to learn": training a system on many tasks so that it becomes good at learning a new one quickly. In
the course it is the last idea of the two lectures on [transfer learning](transfer-learning.md). Ordinary transfer learning asks,
"given some prior model/representation, how can we quickly adapt it?"; meta-learning asks, "Could we *learn* to do transfer
learning?" ([lecture 19](19-transfer-learning-data.md), slide 44). Lecture 19 gives two routes: MAML, which learns an initialization
from which a few steps of SGD go a long way, and sequence modeling, where a model's state does the learning, which the lecturer
connects to in-context learning in large language models. Covered so far: [lecture 11](11-representation-learning-reconstruction-based.md),
≈27:54 ("we learn to learn", as the point of representations); [lecture 18](18-transfer-learning-models.md), ≈1:11:28 (previewed);
lecture 19, slides 43–50, ≈1:06:58–1:14:43; [lecture 21](21-language-models.md), slides 36–40, ≈28:47–35:50 (whether in-context
learning is learning); [lecture 24](24-inference-methods-for-deep-learning.md), slides 42–48, ≈56:42–1:13:42 (in-context learning as gradient descent, and test-time training).

**Notation** (lecture 19's slides). $\theta$ is a model's parameters and $f_{\theta}$ the model; a dataset is
$\lbrace x^{(i)}, y^{(i)} \rbrace_{i=1}^{N}$; $\mathcal{T}_ i$ is a task drawn from a distribution over tasks $p(\mathcal{T})$, with
loss $\mathcal{L}_ {\mathcal{T}_ i}$; $\alpha$ and $\beta$ are step sizes.

## Three levels

Slide 43 of lecture 19, "Meta-learning: learning to learn fast", stacks three levels. **Inference** maps a data point $x^{(i)}$ through
$f_{\theta}$ to a prediction $y^{(i)}$. **Learning** maps a dataset $\lbrace x^{(i)}, y^{(i)} \rbrace_{i=1}^{N}$ through a learner to
parameters $\theta$. **Meta-learning** maps many datasets, $\lbrace \lbrace x^{(ij)}, y^{(ij)} \rbrace_{i=1}^{N_ j} \rbrace_{j=1}^{M}$,
through a meta-learner to the learner's hyperparameters (the slide's label "Datastets" is a printed slip). The lecture describes it as
adding "this meta layer on top of taking a bunch of different data sets and training some of meta learner" (≈1:07:46).

## MAML

Model-agnostic meta-learning (Finn, Abbeel and Levine, 2017) "was the first work that tried to ask, could we learn to do transfer
learning effectively?" (lecture 19, ≈1:08:34). Finetuning starts from an initialization and updates it with SGD; MAML instead learns "an
init such just a few steps of SGD will update it very effectively" (slide 44, "such just" as printed). It is "a double loop in your
optimization" (≈1:08:34). Slide 45 states the loop in two plain steps that it pastes over the paper's Algorithm 1: for each task in a
batch, "1. Starting with init, run a few steps of SGD on task i" and "2. Compute loss after these steps"; then "init <— backprop
through" those steps. The covered lines of the pasted algorithm, which the page does not show but its image holds, are the inner update
and the outer update:

$$\theta_ i' = \theta - \alpha \nabla_{\theta} \mathcal{L}_ {\mathcal{T}_ i}(f_{\theta}) \qquad \theta \leftarrow \theta - \beta \nabla_{\theta} \sum_{\mathcal{T}_ i \sim p(\mathcal{T})} \mathcal{L}_ {\mathcal{T}_ i}(f_{\theta_ i'})$$

## Backpropagating through backward

The outer update needs gradients through the inner SGD steps. Lecture 19 writes one SGD step as a forward pass whose parameters came
from a backward pass (slide 46):

`forward(x, backward(forward(x, θ)) + θ)`

and two steps by nesting it again. Recalling the forward and backward pass through a small network from
[lecture 2](02-how-to-train-a-neural-net.md), the lecturer notes that forward followed by backward "was essentially one big forward pass",
so "you can backprop through backward!" (slides 47–48, ≈1:09:21–1:10:09). Slide 48 lays the backward chain after the forward chain and
runs a "meta backward" through both. "So it's possible. It's just expensive": the unrolled loops make "a very, very deep network that
you're going to optimize through" (≈1:10:55). See [backpropagation](backpropagation.md).

## Does it pay?

"In practice, the gains of this method are not often worth the additional computational cost at training time (worse than just
pretraining?)" (lecture 19, slide 49). One application where it did help is TIML (Tseng et al., 2022), which adds "a geospatial embedding
that encodes task-specific context into meta-learning", for crop-type mapping across countries; on the slide's chart for Kenya it scores
above both MAML and a pretrained model at every training-set size, while those two are "about the same" (≈1:11:40–1:12:25). The lecturer
relates it to the spatiotemporal prior of [lecture 17](17-generalization-out-of-distribution.md) (≈1:12:25).

## Meta-learning by sequence modeling

Slide 50 defines learning as a map from a sequence $x_0, y_0, x_1, y_1, x_2, y_2, \ldots$ to a function $f: x \rightarrow y$. A
[recurrent network](recurrent-neural-networks.md) reading such a sequence can do this itself: "RNN state + params is f, apply that f to
x3". So "the hidden state of that RNN … is a model that arguably learns to learn" (lecture 19, ≈1:13:58). The same holds for
autoregressive language models shown a few examples of a task, such as English sentences with their Spanish translations ([lecture
18](18-transfer-learning-models.md)'s in-context learning): "this really is, I think, what GPT and a lot of these modern autoregressive
models are doing. It's kind of a version of meta-learning at large scale … in-context learning is kind of just another mechanism of
meta-learning" (≈1:13:12–1:13:58). Trained on many tasks and sequences, such models may get "a much better actual output" than "this kind
of inner and outer loop of training" (≈1:14:43). See [foundation models and prompting](foundation-models-and-prompting.md) and
[autoregressive models](autoregressive-models.md).

## Is in-context learning learning? (lecture 21)

[Lecture 21](21-language-models.md) asks the question this page's last section raises. An in-context prompt looks like a training set, $x$,
$y$, $x$, $y$, $x$, but "should we think of what's actually going on there as learning the way we think of gradient descent is learning,
or is this something more complicated?" (≈29:33). One hypothesis is that the prompt conveys only the task's format, with everything else
coming "from pretraining and not from this context at all" (≈30:21), and the evidence of 2022 fit it: shuffling the labels of the
examples barely hurt (slides 38–39, Min et al.), so "the conventional wisdom was that … it's really just helping models retrieve tasks that
they had previously learned how to do" (≈32:43). That has changed. Larger models "do actually start to pay attention to the association
between labels and input examples", and "in particular cases, … models are capable of implementing real learning algorithms internal to
their hidden representations", something "that looks like gradient descent", with "the parameter vector of a smaller model being stored
inside the hidden representations" and "updated across layers as more examples get ingested", at least on synthetic data. Whether big models
do this, "nobody knows. Almost certainly, the answer is, it's complicated" (≈34:15–35:50).

## In-context learning as gradient descent (lecture 24)

[Lecture 24](24-inference-methods-for-deep-learning.md) answers lecture 21's question from the other side. In-context learning, given pairs $\mathbf{x}^{(1)}, \mathbf{y}^{(1)}, \ldots$ and a
query, "is just the definition of supervised learning. It's learning to do supervised learning" (slide 42, ≈56:42–57:30). Von Oswald et
al., "Transformers learn in-context by gradient descent" (ICML 2023, slide 43), write a transformer $f_\theta$ reading a query $\mathbf{x}$
with context $\mathbf{c}$ as a context-free transformer with other weights, $f_\theta(\mathbf{x}, \mathbf{c}) = f_{\theta'}(\mathbf{x})$,
and show a case, "one layer, linear self-attention … a simple toy case", where $\theta'$ is a gradient-descent step on $\theta$ with a loss
on the context examples (≈58:15–1:00:34).

Slide 44 draws the consequence: "instead of doing ICL over $\mathbf{c}$, why not do gradient descent over $\mathbf{c}$?" "ICL is learned
learning. It isn't guaranteed to work", while gradient descent "is guaranteed to improve performance on the in-context examples" and "can
leverage more test-time compute than ICL. You can run it forever"; the possible disadvantage is that "ICL could be a smarter learning
algorithm than gradient descent". Updating the weights on the test query's examples, **test-time training**, reached 61.9% on the ARC
benchmark with "an 'old' method (GD on supervised examples; but examples that are highly relevant to each test query)" (slide 48). See
[search and test-time compute](test-time-compute.md).

## See also

- [Lecture 19](19-transfer-learning-data.md) — the lecture, with its figures.
- [Transfer learning](transfer-learning.md) — adapting a pretrained model, which meta-learning tries to make easy.
- [Backpropagation](backpropagation.md) — the forward and backward pass that MAML differentiates through.
- [Foundation models and prompting](foundation-models-and-prompting.md) — in-context learning.
- [Recurrent neural networks](recurrent-neural-networks.md) — a network whose state can act as a learner.
- [Lecture 24](24-inference-methods-for-deep-learning.md) — in-context learning as gradient descent, and test-time training.
