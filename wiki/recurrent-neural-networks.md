# Recurrent neural networks and LSTMs

A **recurrent neural network** (RNN) processes a sequence one step at a time and carries a **hidden state**
from each step to the next, so that its prediction at any time can depend on everything it has seen so far.
The same functions, with the same weights, are applied at every step. The **LSTM** (Long Short Term Memory) is
an RNN built so that its default is to remember: a cell state passed along unchanged unless learned gates
remove or add information. The course builds both in [lecture 10](10-architectures-memory.md), the last of
its architecture lectures, as the answer to the forgetting of a convolution over time, and then sets them
against convolution and attention as ways to model long sequences. Covered so far: lecture 10, slides 13–39 and
54–59, ≈6:13–41:53 and ≈58:19–1:02:17. Lecture 1's deck points to RNNs as "Lecture 11", which in the recorded
schedule is lecture 10 (see the [course map](course-map.md#the-decks-lecture-pointers)). Homework 3's first
problem, "RNNs versus transformers", works through a simple RNN by hand.

**Notation** follows lecture 10's slides: input $\mathbf{x}_ {\texttt{in}}[t]$, hidden state $\mathbf{h}_ t$ and
output $\mathbf{x}_ {\texttt{out}}[t]$ at time $t$; recurrent weights $\mathbf{W}$, input weights $\mathbf{U}$,
output weights $\mathbf{V}$, biases $\mathbf{b}$ and $\mathbf{c}$. The LSTM slides, derived from Chris Olah's
blog post, use plain $x_t$, $h_t$ and the cell state $C_t$.

## Why memory: what a convolution over time forgets

A convolution in time slides one filter along a sequence, so it handles sequences of any length, but each output
sees only the inputs inside its window (lecture 10, slide 13, ≈6:59–7:46). In the lecturer's example, a filter that
saw her cat Frank early in a home video has no access to that later on, and labels an orange cat outdoors a tiger,
"because it's orange and it's outside, and cats are usually indoors" (slides 15–16, ≈9:20). A memory unit carrying
what was seen earlier would say "it's maybe still Frank" (slides 17–18, ≈10:05). That is the motivation for the RNN.
Convolution in time is on [convolution](convolution.md).

## The recurrence

The RNN keeps a hidden unit and sends it "forward in time to the next time step", so each step takes in "both that new
temporal input and your memory from the past" (slide 19, ≈10:51–11:37):

$$\mathbf{h}_ t = f(\mathbf{h}_ {t-1}, \mathbf{x}_ {\texttt{in}}[t]), \qquad \mathbf{x}_ {\texttt{out}}[t] = g(\mathbf{h}_ t)$$

Both $f$ and $g$ are learned and **shared over time**: "it's independent of time what that function is" (slides
20–21, ≈12:23–13:09). Where earlier architectures had a fixed window, "a recurrent neural network makes that infinite"
(≈10:51). In the simplest RNN they are linear layers with nonlinearities (slide 23):

$$\mathbf{h}_ t = \sigma_1(\mathbf{W}\mathbf{h}_ {t-1} + \mathbf{U}\mathbf{x}_ {\texttt{in}}[t] + \mathbf{b}), \qquad \mathbf{x}_ {\texttt{out}}[t] = \sigma_2(\mathbf{V}\mathbf{h}_ t + \mathbf{c})$$

Without $\mathbf{W}$ this is an input-to-output map that "starts looking a little bit like that multi-layer perceptron"
([multilayer perceptrons](multilayer-perceptron.md)); the recurrence is the only difference. "The hidden state is
intended to capture the most relevant historical information. And what we're learning in W is how do we actually
understand what is relevant?" (≈14:44–15:30). Stacking several hidden layers, each with its own recurrent weights,
makes a deep RNN (slide 24).

The hidden state feeding itself makes a cycle: "This is the first time we're not seeing a DAG" (slide 22, ≈13:09).
The lecturer calls the RNN the first architecture of the course that is "quote, unquote, Turing complete" (≈0:46).

## Backpropagation through time

Backpropagation needs a DAG. "The hack is you just decide when you're training what your fixed time window that you're
going to propagate back through is. And now, you unroll it through that time window" (≈18:36). Unrolled, the network is
a DAG again, and the chain rule runs back along it (slide 25):

$$\frac{\partial \mathbf{x}_ {\texttt{out}}[t]}{\partial \mathbf{x}_ {\texttt{in}}[0]} = \frac{\partial \mathbf{x}_ {\texttt{out}}[t]}{\partial \mathbf{h}_ T} \frac{\partial \mathbf{h}_ T}{\partial \mathbf{h}_ {T-1}} \cdots \frac{\partial \mathbf{h}_ 1}{\partial \mathbf{h}_ 0} \frac{\partial \mathbf{h}_ 0}{\partial \mathbf{x}_ {\texttt{in}}[0]}$$

The window is a hyperparameter, tuned or ablated, and chosen with domain knowledge where possible (≈19:22, ≈23:16–24:04).
Training only through a window may teach the model "to only capture some of limited temporal window of information",
even though at inference the earlier information is still passed along (≈20:08). The loss is summed over the steps of
the sequence (slide 26), and because $\mathbf{W}$ is shared across steps its gradient is the sum of the gradients at
each step where it is used (slides 27–28, "Parameter sharing —> sum gradients"), the rule of
[lecture 2](02-how-to-train-a-neural-net.md); see [backpropagation](backpropagation.md). PyTorch's autograd does the
summing "as long as you don't do deep copies" (≈26:25).

## Vanishing and exploding: the problem of long-range dependencies

In theory the hidden state can carry information arbitrarily far; slide 29 lists why not simply remember everything
(memory grows with $t$, such memory is nonparametric, and RNNs make a Markov assumption, depending only on the previous
hidden state). In practice information must pass "through a long chain of dependences" (slide 30). The lecturer's
chalkboard argument (≈28:46–30:19) drops the nonlinearities and biases: then
$\mathbf{h}_ 2 = \mathbf{W}(\mathbf{W}\mathbf{h}_ 0 + \mathbf{U}\mathbf{x}_ 1) + \mathbf{U}\mathbf{x}_ 2$, and after $N$
steps the earliest terms are multiplied by $\mathbf{W}^N$. Small values go to 0 and large values to infinity; stable
training needs them close to 1, and even slightly below 1, "when you go to infinity, it will go to 0 because you have
this sort of exponential relationship" (≈30:19). So old observations are forgotten, stochastic gradients become high
variance, and gradients **vanish** or **explode** (slide 30, ≈31:06; compare [loss landscapes](loss-landscapes.md)).

The two failures differ: exploding gradients mean unstable training, vanishing ones mean forgotten memories (≈33:25).
Normalizing the weights, for example with the spectral norm of [lecture 7](07-scaling-rules-for-optimization.md), "might
help you with the exploding gradients problem, but it's not going to change the fact that the small things are going to
go to 0", since the exponential is taken first (≈31:06–31:53; see [norms](norms.md)). Exploding gradients can be
normalized, constrained, regularized or clipped, "But we can't do that when things go to 0" (≈34:11–34:57).

Whether long memory is wanted depends on the signal: temperature has daily cycles, yearly cycles and El Niño, "so
depending on what you're trying to model, there may actually be a lot of value in the longer and longer term memory"
(≈33:25–34:11).

## LSTMs

An LSTM (Hochreiter and Schmidhuber, 1997) is "A special kind of RNN designed to avoid forgetting", in which the network
"has to *learn to forget*" (slide 32). "The default is the identity. You're biased to not forget anything. And then you
learn what to forget, kind of like garbage collection" (≈34:57). Each step contains "a little memory controller that
decides what to save and what to delete" (slides 33–34, ≈35:44). Its parts, with $[h_{t-1}, x_t]$ the previous hidden
state concatenated with the input and $\sigma$ the sigmoid (slides 35–39):

| Part | Equation | What it does |
| --- | --- | --- |
| Forget gate | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ | multiplies each element of the cell state by about 1 (remember) or about 0 (forget) |
| Input gate | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ | "which indices to write to" |
| Candidate values | $\tilde{C}_ t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$ | "what to write to those indices" |
| Cell-state update | $C_t = f_t \ast C_{t-1} + i_t \ast \tilde{C}_ t$ | "Forget selected old information, write selected new information" |
| Output gate | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ | decides what to output |
| Hidden state | $h_t = o_t \ast \tanh(C_t)$ | the output passed to the next step |

The products are element-wise (≈41:07). The **cell state** $C_t$ is "kind of the tape in the Turing machine", and the
LSTM "is still Turing complete" (≈35:44–36:29). The gates are sigmoids because the sigmoid is bounded between 0 and 1,
unlike a ReLU (≈37:15, ≈41:53; see [activation functions](activation-functions.md)). The lecturer likens the output
gate, "on vibes", to the value projection of attention (≈39:36).

What fixes the vanishing gradient is the multiplicative gating of the cell state: by default it is multiplied by 1,
"this identity instead of multiplying by yourself, which removes the vanishing and exploding gradients. We no longer have
that exponential", which makes it "kind of like a skip connection or a residual connection in a ResNet" (≈39:36–40:22;
see [skip connections](skip-connections.md)).

## RNNs against convolution and attention

Lecture 10 closes the comparison on slide 58. Recurrence shares its weights across time and passes information forward;
convolution shares its weights across time but sees nothing outside its window; attention's weights are "dynamically
determined as a function of the data" (≈59:56–1:00:41). An RNN is not autoregressive by itself, since its outputs are not
fed back as inputs, "but you could totally loop it around that way" (≈45:02–45:48). Slide 59 reproduces the cost table of
"Attention Is All You Need": per layer, a recurrent layer costs $O(n \cdot d^2)$ with $O(n)$ sequential operations and a
maximum path length of $O(n)$ for sequence length $n$ and dimension $d$, where self-attention costs $O(n^2 \cdot d)$ with
$O(1)$ of each. Its title, in the lecturer's reading, says "we don't need memory. All we need is attention"
(≈1:01:29–1:02:17). See [transformers](transformers.md).

LSTM-based language and captioning models, such as lecture 10's molecule-to-text example (an LSTM decoder conditioned on a
[graph neural network](graph-neural-networks.md)), were the approach "2015-ish"; "today, what we see is often people will
use transformers instead" (slides 47–53, ≈55:56–56:45). Their training and decoding are on
[autoregressive models](autoregressive-models.md).

## Where it goes next

Homework 3's "RNNs versus transformers" problem derives the hidden states of a linear RNN after $T$ steps, asks what happens
to the first input's contribution as $T$ grows, and compares the cost of a forward pass with a self-attention layer's (see
[lecture 10](10-architectures-memory.md#the-problem-set)). Language models return in lecture 21; see the
[course map](course-map.md).
