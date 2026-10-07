# Differentiable programming

The course defines deep learning as two things working together: neural networks, and
**differentiable programming**, "a programming paradigm where [we] parameterize parts of the
program and let gradient-based optimization tune the parameters" (lecture 1, slide 2, ≈1:35).
Lecture 1 frames it as a question of "what are you actually optimizing for, and what are you
pushing the gradients through to?" (≈24:50) and assigns it to lecture 2, which builds it from
computation graphs and [backpropagation](backpropagation.md). Covered so far:
[lecture 1](01-introduction.md), slide 2; [lecture 2](02-how-to-train-a-neural-net.md), slides 26
and 56–70, ≈27:16–29:36 and ≈57:27–1:12:21; [lecture 4](04-architectures-grids.md), ≈54:56–56:30
and ≈1:02:00 (learned versus hand-crafted filters, and using an encoder or decoder on its own);
[lecture 7](07-scaling-rules-for-optimization.md), slides 28–30, ≈1:15:37–1:19:28 (modules that carry a
norm as well as a forward and a backward).

## Programs as computation graphs

A **computation graph** is "a graph of functional transformations, nodes, that when strung
together perform some useful computation" (lecture 2, slide 26). A node can be anything from a
single layer to "an entire neural network" (≈28:04–28:50). Deep learning works with graphs that
are directed and acyclic and whose nodes are differentiable. Backpropagation then delivers the
gradient of a scalar cost with respect to every parameter in the graph, and gradient descent can
tune all of them at once.

Seen that way, a deep network is one kind of differentiable program. Slide 56 draws a layer
$f(\mathbf{x}_ {\texttt{in}}, \mathbf{W})$ under "Deep learning" and the same box as
$f(\mathbf{x}_ {\texttt{in}}, \theta)$ under "Differentiable programming". The perspective is the
same, "but now we basically just are actually programming that model". PyTorch, TensorFlow and JAX
"are all basically libraries that enable us to really efficiently do differentiable programming"
(≈57:27–58:16).

## Why the term caught on

Slide 57 gives two reasons deep nets are popular — they are "easy to optimize (differentiable)"
and "compositional", which it calls "block based programming" — and calls differentiable
programming "an emerging term for general models with these properties". It quotes two posts. Yann
LeCun: "Deep Learning est mort. Vive Differentiable Programming!" Thomas Dietterich: "DL is
essentially a new style of programming … and the field is trying to work out the reusable
constructs in this style. We have some: convolution, pooling, LSTM, GAN, VAE, memory units,
routing units, etc." (≈58:16–59:02).

## Human-programmed and backprop-programmed parts

Not every node has to be learned. In **Neural Module Networks** (Andreas et al., 2017 on slide 58),
a question such as "Where is the dog?" goes to a parser that "might not be learned at all. It
might just be a standard parser", while the CNN that reads the image is trained (≈59:02–59:48). In
Andrej Karpathy's **Software 2.0** picture (slide 59), hand-written Software 1.0 is a single point
in the space of programs, and Software 2.0 defines "a space of possible software systems" and
optimizes within it (≈59:48).

Slide 60 marks the two kinds in one graph: nodes "Programmed by a human" and nodes "Programmed by
backprop", which are "programmed by tuning behavior to match training examples". The lecturer's
example of the human kind is image normalization: "we often explicitly program the normalization
values that we want to use for natural images directly into the pre-processing of our data. … We
don't learn what those normalization values should be" (≈1:00:38).

Where to draw that line is an empirical question. "Anytime a human is programming part of this
system, it's constraining the system. And that constraint can be useful, but it could also be
unuseful." Her example is the feature-engineering era, when hand-designed features fed very simple
models: "our best ideas were not as good as just a much more complex, larger models that were
learned somewhat end-to-end" (≈1:02:10–1:02:56). One part is always human-defined: the cost, which
decides what counts as optimal (≈1:02:56–1:03:45). The line also moves. A part that looks
human-programmed "might have just actually previously been programmed by backprop" — pretrained
weights plugged in — and changing what is optimized is "just a matter of defining … where you want
to freeze your gradients" (≈1:10:48–1:12:21).

In PyTorch, an operation inside the network must be a torch operation with a gradient. Anything
else belongs in pre-processing, since "the pre-processing steps don't necessarily need to be
differentiable, things like data augmentation". Building a complicated statistical model into a
network can be "technically possible, but … intractable to actually learn" (≈1:15:24–1:18:35).

Lecture 4 returns to the feature-engineering shift from the other side, in answer to a student who
asked why convolutional filters are learned when signal processing designs them (≈54:56–56:30). The
move to filters learned end to end was "this massive paradigm shift that happened maybe 10, 15 years
ago", and "a bit of like a catastrophe emotionally for many researchers". The lecturer's rule for where
the line belongs is data: "if you don't have enough data to learn the filters, well, then more
handcrafting, more knowledge, more inductive bias tends to be better… But if you're in a space where
you can get huge amounts of data to learn from, often, we don't know as much as we think we know about
what optimality might be." See [inductive bias](inductive-bias.md). The same lecture applies the
modular view to an encoder–decoder: once trained, "you could use the encoder to build some
representation of input images. But you can also use the decoder to generate images. So these
components can be used separately and can be valuable separately" (≈1:02:00).

## Optimizing inputs instead of weights

"Backprop lets you optimize any node (function) or edge (variable) in your computation graph w.r.t.
to any scalar cost" (slides 61–63, ≈1:01:23). Training uses $\partial J / \partial \theta$, the
cost's sensitivity to the parameters $\theta$. Holding $\theta$ fixed and taking
$\partial y_j / \partial \mathbf{x}$ instead asks how class $j$'s score $y_j$ changes "by changing
the image pixels" $\mathbf{x}$ (slides 64–65, ≈1:03:45). It is better to push on the logits than
on the softmax probabilities, because the easiest way to raise a probability "is often to make the
alternatives unlikely, rather than to make the class of interest likely" (≈1:04:32).

Three uses follow in the lecture:

- **Unit visualization** (slides 66–67): gradient *ascent* on the input to maximize one neuron,
  $\arg\max_{\mathbf{x}} \thinspace y_j + \lambda R(\mathbf{x})$, gives an image of "what a given
  trained model thinks is most cat like", or of what a hidden unit responds to (≈1:04:32–1:05:19).
  The slides do not define the extra term $\lambda R(\mathbf{x})$.
- **DeepDream** (slide 68): the same idea, producing images the lecturer calls "psychedelic and
  beautiful" (≈1:05:19).
- **CLIP+GAN** (slide 70): a frozen text encoder embeds a prompt as $\mathbf{e}_ 1$. A frozen image
  generator and a frozen image encoder turn a latent input $\mathbf{z}$ into an image embedding
  $\mathbf{e}_ 2$. Only $\mathbf{z}$ is optimized, to maximize $\mathbf{e}_ 1 \cdot \mathbf{e}_ 2$,
  with the gradient backpropagated through all three frozen networks (≈1:06:05–1:09:15).

The lecture's summary of the whole idea: "all these trapezoids are neural networks. You can plug
them together. You can take the components trained in one way and use them in another way. Really,
the idea is you can optimize modules with respect to all the other modules, and the world's your
oyster" (≈1:06:52).

## Modules with a norm (lecture 7)

Lecture 7's lecturer extends the module idea in his own research, presented with "so be skeptical". A
**module** takes weights and inputs to outputs, as a PyTorch module does, and covers anything from a ReLU
(with no weights) to a whole transformer. Besides `M.forward` and `M.backward`, each module gets
`M.norm`, a function from its weights to a number (slides 28–29). Atomic modules (Linear, Embedding,
Conv2D, ReLU) have all three written by hand, and combination rules build the rest: composing two modules
composes their forwards, and the chain rule gives the backward. How to compose their norms is the open
question, and answering it would let any architecture built this way come with the norm its optimizer
should use (slide 30, ≈1:18:42–1:19:28). See [scaling rules](scaling-rules.md#a-modular-theory).
