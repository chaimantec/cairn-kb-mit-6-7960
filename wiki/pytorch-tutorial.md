# PyTorch Tutorial

**Speaker:** Jamison Meindl ·
**Video:** [youtube.com/watch?v=o5gPABcGZwc](https://www.youtube.com/watch?v=o5gPABcGZwc) (29 min) ·
**Slides:** none ·
**Transcript:** [`raw/transcripts/pytorch-tutorial.md`](../raw/transcripts/pytorch-tutorial.md)

**This page is built from the transcript alone.** The tutorial has no lecture number and no slide deck. It walks through a
notebook that the speaker posted on the class Piazza (≈2:20), and OCW publishes neither that notebook nor any slides: the
tutorial's OCW page holds only the video and its transcript. So the code, the printed outputs and the plots on screen are known
here only from what is said about them. Function names are given as spoken, and where the speech does not say what a cell
computes, this page says so rather than filling it in. The recording covers the whole tutorial, which ends at ≈28:53 when its
last example fails to run.

## What this tutorial covers

A 29-minute introduction to PyTorch for students who have not used it; a show of hands at the start finds "a lot of newbies to
PyTorch" (≈2:20). It gives three reasons for the library: it is fast, because it runs on the GPU; its data operations are
similar to NumPy's; and it supports **autograd**, computing gradients automatically, which makes it "very intuitive and simple
for us to actually train and update the weights of our models" (≈0:00–1:32). It then works through four parts of the notebook:
**tensors** and the operations on them (≈2:20–16:16), **autograd** (≈16:16–19:21), **optimizers** (≈19:21–20:53) and **neural
networks** built from PyTorch's modules (≈20:53–28:53). The speaker presents it as a start rather than full coverage: "You'll go
into a lot more depth on some of the psets" (≈1:32), and at the end of the tensor part the speaker is "going a little slower than I was
hoping" (≈15:29).

## Where it sits in the course

Lecture 1 announced PyTorch tutorials "for if you're not familiar with PyTorch" or want a refresher (≈6:57–7:43), and lecture
2's announcements slide lists "Pytorch tutorials this week"; see the [course map](course-map.md#coursework-and-policies). The
tutorial refers to gradients and batches as having come up "in lecture today" (≈0:46, ≈10:04), to stochastic gradient descent
as seen "in lecture" (≈20:06) and to ReLU as "the activation that we've seen in lecture" (≈25:32). That fits the week lecture 2's
slide gives, but the recording names no lecture and does not say which of the announced tutorials it is. OCW files the video as
`mit6_7960f24_review.mp4` and describes it as "an overview of PyTorch, an open-source deep learning framework". The problem sets
use PyTorch: "you guys are going to be using PyTorch many times with all of your psets" (≈0:00).

## Tensors

"PyTorch tensors is basically where we can actually store data within PyTorch": a model's inputs, and the outputs to compare,
are represented as tensors (≈3:07). See [tensors and batching](tensors-and-batching.md) for the course's account of why deep
learning computes on them.

### Creating one

The tutorial shows four ways (≈3:07–5:27). **Directly from data**: a Python list holding 1, 2, 3 and 4 becomes a tensor "which
is the same structure as that nested list", though "you're probably not going to actually do this too often" (≈3:07). **From a
NumPy array**, with `torch.from_numpy`: tensors and NumPy arrays "store a lot of the same structure", and the tensor "can use the
same memory in the system for both the tensor and NumPy array" (≈3:54–4:41). **From another tensor's shape**, with `ones_like`,
or `rand_like`, which "will give you a random number between 0 and 1 in each element", for instance to initialize a model
(≈4:41). And **from a shape given directly**, such as 2 by 3, with `rand`, `ones` or `zeros`; the documentation lists more
initializations (≈5:27).

### Shape, data type and device

Three attributes are "good to know". A random 3 by 4 tensor reports its size through `.shape` (≈5:27). Its **data type**: most
tensors used as model inputs are float32, a 32-bit float, "because it's the fastest. And GPU hardware is optimized to use that";
integers appear "in some situations" (≈6:12). Its **device**, from `.device`, matters when both CPU and GPU are in use: "we can
only do operations between tensors that are on the same device", so a tensor must be moved to the other's device before the two
can be combined, and "you will probably be frequently be finding out which device your tensors are on and then moving them"
(≈6:12–6:58). The speech does not name the call that moves one.

### Element-wise operations, statistics and plots

The running example is a tensor of "101 equally-spaced numbers from 0 to 5", useful for "seeing the effect along a specific
range" (≈7:44). It is indexed as in NumPy: the first five entries, and the last five "by doing negative 5 to the end" (≈7:44).
`x.sin` applies the sine to every entry. A "more complex" expression follows, captioned as "x to the power of x cosine", which is
unclear; the transcript flags it, and the autograd example below squares $x$ and takes the cosine (≈8:31). Tensors can be
subtracted, summarized by their min, max, mean or standard deviation, and passed directly to matplotlib to plot (≈8:31–9:17).
`.clamp` limits values to a range, "if you, say, were worried about your outputs going too far in some direction" (≈9:17). The
general advice: vector operations apply to tensors and return tensors, and for any transformation "there's probably some ability
to do it. And you'll just have to look up the documentation" (≈9:17–10:04).

### Batches: the batch index comes first

Data usually arrives as a batch, "and generally, what we do to represent that is we include the batch as the first dimension".
For images the RGB channel comes second, "and then the rest of your data", but "the most important part there is that the batch
index is generally the first dimension" (≈10:04–10:50). The example holds random numbers in the shape of "two RGB images that
are 5 by 4", a tensor of size 2, 3, 4, 5 (≈10:50–11:35).

**Permuting** reorders the dimensions: list them in the order wanted, here keeping 0 and 1 (batch and channel) where they are and
swapping the two image axes, so that "we originally had a size of 2, 3, 4, 5, but in this case, we've swapped the 4 and the 5"
(≈11:35). **Reshaping** uses `.view`, "generally the more efficient way", or `.reshape`, which "has pretty much the same
functionality". Keeping the batch size and passing $-1$ for the rest puts "all of the rest of the data" into one dimension,
giving a shape of 2 by 60 (≈12:23). **Indexing** works "pretty much the same way as NumPy": 0 for the first row; a colon and then
0 for the first column, "the second dimension"; and $-1$ for the last. The same slicing assigns values, as when the second
column is set to 0 (≈12:23–13:09).

### Joining tensors

`torch.cat` takes a list of tensors and a dimension, and joins the tensors along it: three 4 by 4 tensors concatenated with dim
equal to 1 give a 4 by 12 (≈13:55). The speaker calls this "the first dimension". In the counting used a minute earlier, where
the batch index is the first dimension and columns the second, dim 1 is the second, and a 4 by 12 result is a join along the
columns; the transcript flags the wording. `torch.stack` does something different: "instead of concatenating along one
dimension, you're going to actually create a new dimension". To make a batch from several inputs, stack them along dimension 0,
"and you'll end up with an additional dimension at the start" (≈13:55–14:41).

### Products and aggregates

Matrix multiplication can be written several ways that "do the same thing"; "using the at symbol is probably the easiest way"
(≈14:41). The element-wise product is the multiplication symbol or `tensor.mul`, so the tensor 1, 2, 3, 4 multiplied element-wise
by itself gives its squares (≈15:29). Aggregators such as sum, mean, max and min give "the value along a specific dimension or
along the whole tensor" (≈15:29).

## Autograd

"Autograd is one of the most important parts of PyTorch because it allows us to actually compute gradients and update our
models so that they can actually learn" (≈16:16). The tutorial shows it twice: first one derivative computed explicitly, then
the way it is used in practice. For what autograd computes, see [backpropagation](backpropagation.md); for the paradigm it
serves, [differentiable programming](differentiable-programming.md).

**One derivative, explicitly.** Take the tensor $x$ of values from 0 to 5, square it and take the cosine, $\cos(x^2)$. "If we
want to take a derivative, we actually need to have a scalar as an output", so the outputs are summed, and `torch.autograd` of
that sum with respect to $x$ (called `autograd.grad` at ≈18:35) gives "dy by dx" (≈16:16–17:02). The plot shows the function,
"where we end up with a shorter period as we go to the right", and beside it "the correct gradient of that value"; a polynomial
gets the same treatment (≈17:02–17:48). The tutorial states the rule, a scalar output, and does not say why the gradient of the
sum is the derivative at each point.

**All gradients at once.** In a model, "PyTorch takes care of exactly which derivatives we need" (≈17:02). Calling `.backward`
"will store the gradient of each part within the tensor", which `.grad` then reads: "You'll never do the autograd.grad that we
had up here. You'll always use the .backward to just calculate all the gradients that you need" (≈18:35).

**Turning it off.** "If you aren't going to use the gradient, you shouldn't calculate it": `torch.no_grad` skips it and speeds up
the computation (≈18:35–19:21). The layer weights later show `requires_grad=True`, "which shows that it's calculating the
gradient as you go along" (≈22:27).

## Optimizers

With a gradient in hand, "the next thing we need to do is actually use that gradient to optimize some function, some loss
function" (≈19:21). The notebook first minimizes a toy function by naive gradient descent, which "works for this scenario, but in
a lot of cases we want to be more adaptive and speed up our training" (≈19:21). It then imports an optimizer, the stochastic
gradient descent one ("the SGD", ≈20:53), whose `optimizer.step` "takes care of the gradient update. So it will use the gradient,
and it will update the weights of our model". On the toy problem, "Maybe it got to the minimum a little faster, and then it's
jumping around a little bit", which the speaker sets aside: "This is just some little toy examples" (≈20:06). Last comes
**Adam**, "a pretty common optimizer" that "adaptively does a lot of the steps. And it generally is the most efficient", so "a
lot of people use that in real scenarios". It is used with "the same structure as the SGD", and in this example "it's not quite
doing as well as some of the simpler ways, but in general, this is a pretty good method of optimizing" (≈20:06–20:53). The
speech does not say what function the toy example minimizes. See [gradient descent](gradient-descent.md), and [lecture
9](09-hackers-guide-to-deep-learning.md) for the course's advice on choosing an optimizer.

## Neural networks

PyTorch has "a lot of modules ... that we can use to create networks from just really simple MLPs to more complex models"
(≈20:53). See [multilayer perceptron](multilayer-perceptron.md).

### The linear layer

"The first building block" maps one dimension to another "with a linear transformation", set up with a number of input features,
a number of output features and a bias term (≈21:39). With 3 inputs and 2 outputs, an input of size 3 gives an output of size 2,
and a batch of four inputs of size 3 gives four outputs of size 2: "natively, it will just work to support that" (≈21:39–22:27).
In the course's notation (see [the linear layer](multilayer-perceptron.md#the-linear-layer)), output $j$ is
$z_j = \mathbf{x}^T \mathbf{w}_ j + b_j$, for an input vector $\mathbf{x}$, a weight vector $\mathbf{w}_ j$ and a bias $b_j$. The
tutorial says the same in words: the weights are "the numbers that it's going to multiply the inputs by", and "the bias terms
are added to the outputs" (≈22:27). Setting the bias to 0 means the outputs "will not add that bias term at the end"
(≈22:27–23:13). The parameters can be listed with their values (captioned "net.name_parameters"; the transcript notes that
PyTorch's method of this kind is `named_parameters`, which is outside the course material), their shapes inspected, and the
weights saved and loaded again (≈23:13).

### Training it to output a constant

"Kind of a dumb example": train the linear layer "to always output 1 comma 1", that is, the output (1, 1) whatever the input
(≈24:00). Each pass of the loop takes a batch of inputs, computes the squared loss of the outputs against (1, 1), gets the
gradients with `loss.backward`, and updates the parameters by gradient descent (≈24:00–24:46). The loss falls to near 0, and the
learned parameters show how: "the weights within the linear layer just went to 0, and our bias went to 1", so the layer ignores
its input and outputs its bias, (1, 1), every time (≈24:46).

### Stacking layers: a classifier around a sine wave

A "sequential command" stacks layers: "This will combine different layers together." Here they are linear and ReLU layers ("ReLU is the
activation that we've seen in lecture"), making a network that is "I think ... 3 layers in size" (≈25:32). Its task: take "a 2D input in the space negative 2 to 2"
and "classify points as either above a sine wave or below a sine wave" (≈26:17). The training loop uses the Adam optimizer. Each
iteration generates a batch of random inputs, labels each as above or below the sine wave (the notebook's "classify target"),
and computes the cross-entropy, "a loss typically used for classification", which "is going to give a higher value if your
classification result is worse and a lower value if it's better". It then calls `loss.backward` and takes an optimizer step, and
reports the accuracy (≈26:17–27:05). See [softmax and cross-entropy](softmax-and-cross-entropy.md) and [activation
functions](activation-functions.md).

**The example never ran.** The cell failed because "a few cells above" had not been run, the speaker could not find which, and
the tutorial ends there: "we'll just leave that there" (≈27:59–28:53). So the recording shows neither the trained classifier nor
its accuracy.

## Summary

- **Tensors** hold a model's data. They are made from lists, from NumPy arrays (sharing memory), from another tensor's shape or
  from a given shape, and carry a shape, a data type (usually float32) and a device; tensors combine only on the same device.
- **The batch index comes first.** Permute reorders dimensions, view and reshape regroup them ($-1$ for "the rest"), `torch.cat` joins
  along an existing dimension and `torch.stack` adds a new one; `@` is the matrix product and `*` the element-wise one.
- **Autograd** gives one derivative of a scalar output through `autograd.grad`, but in practice `.backward` fills in every
  `.grad` at once; `torch.no_grad` turns it off.
- **Optimizers** apply the gradient with `optimizer.step`; SGD and Adam share one structure, and Adam is the common choice.
- **Modules**: a linear layer handles batches natively, trains by the same loop (loss, `backward`, update), and a sequential
  command stacks linear and ReLU layers into a classifier trained with cross-entropy, whose result the recording never shows, because a cell above it had not been run.

## Pointers to other lectures

The tutorial points back to lecture for gradients (≈0:46), batches (≈10:04), stochastic gradient descent (≈20:06) and ReLU
(≈25:32), naming no lecture, and ahead to the problem sets (≈0:00, ≈1:32). [Lecture 2](02-how-to-train-a-neural-net.md) covers
gradient descent, batches, backpropagation and differentiable programming, and says that PyTorch "has autograd". [Lecture
9](09-hackers-guide-to-deep-learning.md) covers the practical side: inspecting tensors before the forward pass, reshaping bugs,
dtype casts and optimizers. See the [course map](course-map.md).

## See also

- [Tensors and batching](tensors-and-batching.md) — why deep learning computes on batched tensors, and the tutorial's tensor operations.
- [Differentiable programming](differentiable-programming.md) — the paradigm autograd serves, and autograd in practice.
- [Backpropagation](backpropagation.md) — what `.backward` computes.
- [Gradient descent](gradient-descent.md) — the update `optimizer.step` applies, SGD and Adam.
- [Multilayer perceptron](multilayer-perceptron.md) — the linear layer and stacking layers.
- [Softmax and cross-entropy](softmax-and-cross-entropy.md) — the classification loss of the last example.
