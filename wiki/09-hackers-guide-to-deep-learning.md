# Lecture 9 — Hacker's Guide to Deep Learning

**Lecturer:** Phillip Isola ·
**Video:** [youtube.com/watch?v=DC2Hw9DiLCg](https://www.youtube.com/watch?v=DC2Hw9DiLCg) (76 min) ·
**Slides:** [`mit6_7960_f24_lec9.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)
(72 pages; the deck is titled "Lecture 9: Hacker's guide to DL"; transcribed slide by slide in [`raw/slides/09-hackers-guide-to-deep-learning.md`](../raw/slides/09-hackers-guide-to-deep-learning.md)) ·
**Transcript:** [`raw/transcripts/09-hackers-guide-to-deep-learning.md`](../raw/transcripts/09-hackers-guide-to-deep-learning.md)

## What this lecture establishes

This lecture is practical rather than theoretical: "a bit more on just practical advice and
heuristics and hacks, a little less on formal knowledge and theory" (≈0:00). The lecturer gives a
disclaimer at the start: "this is going to be a somewhat opinionated lecture. I'll share things that
I think are good ideas, but they're not what everybody would agree on. It's just anecdotes from my
own experience." Slide 2 prints the same disclaimer in red ("This lecture is my personal opinions and
anecdotes!"). The deck walks through five parts of the deep learning pipeline, **data, model,
optimization, evaluation/experimentation/debugging, and compute**.

The lecture argues that **the data matters most**. Look at the inputs and the outputs, not only the
loss curve. Check that the data reaching the model is what you think it is. Standardize it, keep every
tensor dimension large, and make the learning problem *hard enough*, with data augmentation and domain
randomization, so that a model trained on it generalizes. Above all, change the data rather than the
learner: more examples, and more information in each input, until predicting the target is nearly
deterministic. "Data is the most important thing in machine learning" (≈52:35). On the model side,
the advice is to keep it simple and popular, start from a pretrained model, turn a new problem into a
solved one (colorization turned into classification), formulate problems as softmax regression, avoid
batch norm, scale data, model and compute, and strip out everything nonessential once the system works.
The default recipe "ca 2024" is one-hot data, cross-entropy loss, Adam and a transformer.

**What the recording covers, and what it does not.** The talk spends more than half its time on data
("that was all about data, and that was already more than half the lecture", ≈52:35) and most of the
rest on the model. At ≈1:08:50 the lecturer says he is "a little short on time" and will skip slides.
He then covers copilots, one-data-point overfitting, log-loss reference values, exponential moving
averages and the "spices" slide in a few minutes, and ends: "The rest of the slides have various tips
that you can read on your own time. I hope that they're self-explanatory" (≈1:14:54). Most of the
optimization, evaluation, debugging and compute slides (48–71) are therefore never spoken. This page
takes them from the slides alone and says so where it does.

**Sources.** Slide 2's acknowledgements say that "lots of slides" are "adapted from Evan Shelhamer's
'DIY Deep Learning: Advice on Weaving Nets'". The deck also builds on Andrej Karpathy's recipe
(karpathy.github.io/2019/04/25/recipe), feedback from Isolab members and the MIT community, slides from
Dylan Hadfield-Menell, and Twitter feedback. In the recording, Evan Shelhamer's talk "formed the
skeleton of this talk", and the lecturer introduces him as the developer of Caffe, "the most popular
deep learning framework 5 or 10 years ago" (≈0:47). Eleven slides (5–7, 11, 48, 49, 52, 53, 56, 62 and 63) carry the credit "[slide adapted
from Evan Shelhamer]", and slide 59 carries "[slide adapted from Dylan Hadfield-Menell]".

**Notation on this page** follows the slides. $\mathbf{x}$ (or $X$) is a network's input and
$\mathbf{y}$ (or $Y$) its target. $f$ is the learned function, and $P(Y \mid X)$ the distribution of
targets given an input. For a single input dimension $x_k$, $\mathbb{E}[x_k]$ is its mean over the
data and $\mathtt{Var}[x_k]$ its variance.

**What is in the deck but not in the picture here.** OCW excludes 21 of the deck's 72 pages from its
licence: the paper and website screenshots of slide 3; the chest X-ray of slide 4; the coffee-cup photos
of slide 7; the augmentation photographs of slide 13; the domain-randomization images and results of
slides 16–18; the Stable Diffusion and AlphaFold screenshots of slide 28; every colorization photograph
(slides 30–33 and 35–39); the facade-generation grid of slide 58; the Weights & Biases screenshot of
slide 60; the spice-rack photograph of slide 61; and the Scale ML web page of slide 71. Those are
described in prose below and in the slide file but have no image in this knowledge base.

## Hacking versus theory

Slide 3 states the premise: "Part of the story of deep learning has been the (temporary) success of
hacking over theory." In the recording the lecturer calls it "a little bit of a triumph of the
practitioners over the academics and the theorists". It "doesn't mean that we won't eventually have a
very clean mathematical theory of deep learning", but historically much of the progress came from
"these things which we might call hacking" (≈1:34).

The slide sets two screenshots side by side. On the left is the first page of Zhang, Bengio, Hardt,
Recht and Vinyals, "Understanding Deep Learning Requires Re-thinking Generalization". The lecturer
recalls it from [lecture 6](06-generalization-theory.md#neural-nets-can-fit-random-labels): a deep net
can fit random labels, so the generalization bounds of classical theory such as the VC dimension "are
vacuous", "and yet, deep nets generalize" (≈1:34–2:23). On the right is the fast.ai website ("Making
neural nets uncool again"). Many people meet deep learning through courses like that, "very practical
in nature … let's just code these systems and make them run, and maybe the math and the theory is
secondary" (≈2:23). "This practical stuff is a big part of the story, and it's worth understanding."

## Look at the data

### "Become friends with every pixel"

The first story was passed down "through multiple generations" (≈2:23–3:59). Alyosha Efros, the
lecturer's postdoctoral advisor at Berkeley, was a graduate student of Jitendra Malik. He would bring
Malik a summary number or plot ("oh, I got 30% accuracy. Or maybe here's my loss"). Malik would ask
why there were spikes, Efros would not know, and Malik would say that this was not enough: "You really
have to look at the data." In a computer vision lab the data was images, so the instruction was to look
at the images and the labels, not a summary statistic. Malik's phrase was "**Become friends with every
pixel**" (≈3:59). The lecturer generalizes it: "If you're studying music, become friends with every
note. If you're studying chemistry, become friends with every molecule." He applies it to the course's
final projects: "Don't just show us a plot. Show us the data" (≈4:44).

### Look at the input: a shortcut in a medical classifier

Slide 4 has two headings, "Look at the input" and "Look at the output". Under the first is a frontal
chest X-ray with a bright "R" marker at its upper left, credited "DeGrave, Janizek, Lee, 2020" (an OCW
notice excludes the image). The lecturer describes the example as a system that classifies a scan of a
patient's chest as benign or malignant, to tell whether "this patient has breast cancer or not"
(≈4:44). The paper is "an analysis and critique of prior work" (≈5:30). A deep net in that prior work
reached very high accuracy ("I don't know — 99% accuracy"), then failed when tested.

He asks the class why. One student suggests that it says nothing is cancerous. That is a common
failure when the data set is imbalanced, "but that's not the case in this one" (≈5:30–6:15). Another
student notices the "R" in the corner, and that was the paper's point. "The deep net was not looking at
the tissue. It was looking at the R in the corner," a **spurious correlation**, a **shortcut** (≈6:15).
The R indicated something like which hospital the scan came from ("I don't remember entirely"). The
lecturer offers a hypothetical: imagine a hospital where every patient in the system had a malignant
tumour, "because it was a hospital you go to for that situation". "There was a bias in the data, and
the data gave away the answer", so the model was right for the wrong reason, and "it won't generalize
to other hospitals" (≈7:01).

*Note, from outside the course material: the paper by DeGrave, Janizek and Lee that slide 4 cites
studies deep networks that detect COVID-19 from chest radiographs. The slide names no disease, and the
lecturer says he does not remember the details.*

### Look at the output

"Don't just look at your loss curve. That's the most superficial thing" (≈7:01). Slide 4's lower row,
repeated on its own as slide 57 in the evaluation section, shows three outputs. First, a training-loss
curve against epochs, falling but broken by sharp spikes. Second, a grid of generated samples that all
look alike. Third, a frame from a simulation of small multi-limbed creatures.

The lecturer's two examples match the second and third pictures. The first is a generative model
learning to make photos of cats. Its loss "might be spiky and crazy", but samples taken during training
show that the model oscillates, "like, periodic". That is typical of a generative adversarial network,
and it is "very obvious in this output" (≈7:48–8:34). Looking at samples gives "a much
higher-dimensional, high-throughput understanding of what's going on". The second is "some little ants
that I trained years ago" to play a predator-prey game by reinforcement learning. The predators' reward
rose and then plateaued. "Learning is hard … SGD was failing." In fact "the ants fell over," and that
had to be fixed (≈8:34–9:21).

![Slide 57: look at the output: a spiky loss curve, a grid of near-identical generated samples, and a frame of the creatures simulation](../raw/images/09-hackers-guide-to-deep-learning/slide-57.jpg)

*Slide 57 — Look at the output: the spiky loss curve, the grid of near-identical samples and the creature simulation, the same three pictures as slide 4's lower row. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

### Inspect the distribution

Slides 5 and 6 turn this into a checklist. Slide 5 is headed "Look at the data!": "inspect the
distribution of inputs and targets". That means inspecting a random selection of inputs and targets "to
have a general sense", histogramming input dimensions "to see range and variability" and targets "to
see range and imbalance", and selecting, sorting and inspecting "by type of target or whatever else".
Slide 6 adds the inliers, outliers and neighbours. Visualize the data, "especially **outliers**, to
uncover dataset issues", and look at **nearest neighbours**. Its examples are "rare grayscale images in
color dataset, huge images that should have been rescaled, corrupted class labels that had been cast
to uint8".

On imbalance, the lecturer returns to the medical example. If the data is 99% benign tumours, then a
system that always says benign, "not looking at the input at all", is 99% accurate. "You have to
realize that 99% is trivial." You then need a higher number, or a different measure, "like the
conditional probability of being correct, given that it's malignant" (≈9:21–10:07). He also asks what
units the data comes in: "Are your numbers in the range 0 to 1, or in the range 0 to 256, or in some
other range? Can negative numbers be your data?" (≈10:07–10:52).

### The data as loaded is not the data as stored

Slide 7: "pre-processing: the data as it is loaded is not always the data as it is stored!" Its
instruction is to "inspect the data as it is given to the model by `output = model(data)`". The lecturer
calls this "one of the most common bugs that I come across, especially in final projects" (≈10:52). His
example is image data the code assumes lies in $[0, 1]$, loaded as unsigned 8-bit integers ("uint8") in
the range 0 to 255. If the code then clamps its inputs to $[0, 1]$, "you're making all your data points
equal to either 0 or 1" and losing all the information (≈11:40). His recipe is to inspect the data
"right before" the line that runs the network's forward pass, not only on disk or as the data loader
first produces it (≈11:40–12:26).

Slide 7's three photographs (excluded by an OCW notice) show one picture of a cup of latte three ways:
the original, the same image loaded by DeCAF and shown upside down, and loaded by Caffe with its colour
channels swapped. DeCAF and Caffe were "the precursors of PyTorch and TensorFlow". DeCAF represents an
image "upside down", and Caffe's standard colour order is BGR rather than RGB, "because of some old
convention in a library called OpenCV" (≈12:26–13:13). A model pre-trained on RGB data and fed BGR data
may call a red object a blueberry, because it reads the red channel as blue (≈13:13–14:01). "If there's
any mismatch between training and inference, or between the model architecture and the numerical
transformations you apply and the data format, then this will cause problems."

### The most important function in deep learning

Slide 8 is headed "Most important function in deep learning:". The lecturer introduces it as "one
that I wrote. So it's a very clever advanced contribution" (≈14:01):

```python
def inspect_data(X):
  print('type:', X.type())
  print('shape:', X.shape)
  print('requires grad:', X.requires_grad)
  print('numerical range: [{:.2f}, {:.2f}]'.format(X.min(), X.max()))
  print('mean and var: {:.2f}, {:.2f}'.format(X.mean(), X.var()))
```

He puts it "all over in code that I write", above all right before the forward pass (≈14:49). It
reports the data type ("Is it a 32-bit tensor? Is it a uint8?"), the shape ("the shape of the tensors in
deep learning are super critical"), whether the tensor requires gradients, and its minimum, maximum,
mean and variance. On gradients, he says that any parameter you want to train "needs to have requires
gradient equals true" (≈14:49–15:36). Slide 8 also names a library that reports this by default,
lovely-tensors (github.com/xl0/lovely-tensors). The lecturer prefers the snippet, which "will work, I
think, for at least a few years" (≈15:36).

## Pre-processing

### Standardize

Slide 9: "pre-processing: **standardize**",

$$x_k \leftarrow \frac{x_k - \mathbb{E}[x_k]}{\sqrt{\mathtt{Var}[x_k]}} \qquad \forall k$$

for every input dimension $k$. The slide's bullets say that this "squashes all your data dimensions into
the same standard range", "makes it so that, a priori, no one dimension is valued more than any other",
and is "important when different measurements have vastly different scales or units". The lecturer's
reason is unit invariance. If you are modelling geospatial data, "it doesn't matter if I've measured my
data with meters or centimeters or inches or feet" (≈16:24). He describes the operation once as "divide
by the square root of the variance" and once as "divide by the variance"; slide 9 divides by the square
root.

A student asks whether this is better than rescaling by the minimum and maximum. "Not necessarily"
(≈17:10). The aim is to remove information about the unit of measure or the mean, "and there's a lot of
ways of doing that". Standardization is the most common. The other common one is to subtract the
minimum and divide by the new maximum, so that the data lies in $[0, 1]$ (≈17:56).

Both remove "inductive bias about the unit of measurement, which might be good or might be bad". The
lecturer asks when that would be bad. A student offers horizontal and vertical distances. If the two
are physically the same quantity, scaling them separately would measure "space in the horizontal axis
differently than the vertical axis" (≈17:56–18:44). It is still a good default, "especially when you
have measurements on vastly different scales". If one measurement is 10 million times bigger than
another, the network will struggle to attend to the small one, "because the gradients will just be
much, much smaller for the data on a smaller scale" (≈18:44–19:29). See [inductive bias](inductive-bias.md).

### Check the statistics, the shape and the type

Slide 11 lists three checks.

- **Summary statistics.** "Check the min/max and mean/variance to catch mistakes like loading values in
  the range [0,255] when the model expects values in the range [0,1]."
- **Shape.** "Are you certain of each dimension and its size?" Sanity-check the network "with dummy data
  of prime dimensions: there are no common factors, so mistaken reshaping/flattening/permuting will be
  more obvious. example: a 64x64x64x64 array can be permuted without knowing". The lecturer credits this
  trick to Evan Shelhamer. If every dimension is 64, a wrong permutation still produces a tensor of the
  right shape. With "a unique dimensionality for each dimension of the tensor", a mistake "will throw a
  shape mismatch error" (≈24:51–26:22).
- **Type.** "Check for casting, especially to lower precision. What's -1 for a byte? How does
  standardization change integer data?" In the lecturer's words: if you cast to uint8 to save memory,
  negative values cannot be represented. If you have standardized data you thought of as positive, it
  now has negative values, "but it's uint8's, and it can't have negative values. OK, there's going to be
  a bug" (≈26:22–27:10).

## Beware of low dimensions

Slide 10 reverses the usual warning. Statistics classes say "beware of high dimensions", because human
intuition is built for three-dimensional space, and a high-dimensional Gaussian is "like a soap bubble,
where almost all of the probability is in a tiny little band around the surface of the sphere"
(≈19:29–20:15). "But … if we weren't human and we're just some ideal creatures of math … things are
simple in high dimensions, and they're weird in low dimensions. And deep learning is all about getting
used to thinking in high dimensions" (≈20:15–21:02).

The example is two normalization layers. The RMS norm was introduced by co-instructor Jeremy Bernstein
(see [norms](norms.md)), and layer norm is "the standard layer in transformers". "In high dimensions,
these things behave almost identically." Layer norm subtracts the mean over the vector and then takes
the RMS norm, while RMS-norm does not subtract the mean (≈21:02). Slide 10 writes layer norm as

$$x[k] = x_{\texttt{in}}[k] - \frac{1}{k} \sum_k x_{\texttt{in}}[k], \qquad x_{\texttt{out}} = \texttt{RMS-norm}(x)$$

where $x_{\texttt{in}}$ is the layer's input vector, $x[k]$ its $k$-th entry after the mean is subtracted,
and $x_{\texttt{out}}$ the output (the slide sets "in", "out" and "RMS-norm" in typewriter type). As printed, the mean carries $1/k$ in front of $\sum_k$, with $k$ as
both the normalizer and the summation index.

![Slide 10: in 2D, RMS-norm maps a cloud of inputs onto a circle, while layernorm collapses it to two points](../raw/images/09-hackers-guide-to-deep-learning/slide-10.jpg)

*Slide 10 — Beware of low dimensions: in two dimensions RMS-norm sends a cloud of input points onto a circle, but layernorm sends them all to just two points. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

The slide draws both for two-dimensional inputs. RMS-norm maps a cloud of points onto a circle, "the
unit hypersphere". Layer norm in two dimensions first subtracts the mean of a two-dimensional vector,
"I'm losing one degree of freedom", and the RMS norm then loses another, "so now the outputs have 0
degrees of freedom, and they end up being on these just two points as opposed to being on a
one-dimensional manifold" (≈21:48–22:34). The slide's caption reads: "In high dimensions,
normalization layers can make entries ~N(0,1), whose typical set is ~surface of hypersphere. ← Not so in
low dimensions. Many normalization layers behave badly in low dimensions."

The same failure hits batch norm with a small batch ("batch norm won't work well if your batch size is
small") and layer norm with a narrow layer (≈22:34). The lecturer's own example is pix2pix, "the work
that was the most popular that I've ever been involved in". Its initial code release ran batch norm
over a batch of size 1 for one of the baselines. Subtracting the mean over a batch of one data point
leaves zero, "so that baseline was trivially beaten". He corrected it, and "we still beat the baseline
in the end" (≈22:34–23:20). See [normalization layers](normalization-layers.md).

The slide's conclusion: "Avoid low dimensions! All tensor dimensions should be big numbers: [BxNxMxC]
data batches, [NxM] weights." Asked how big, the lecturer says "at least 10 and above, and ideally,
probably as big as — the bigger, the better" (≈24:06). His analogy is thermodynamics. "It's really hard
to say anything about a set of three or four molecules … But it's very easy to say what the temperature
is of a billion atoms because, asymptotically, things become very simple. We leverage the law of large
numbers" (≈24:06–24:51). The cost of size is "money, time, energy", but "for statistics and inference
and prediction, bigger is better".

## Reshaping tensors

Slide 12: "A lot of your code will just be reshaping tensors." The lecturer: "You're constantly
transposing, permuting, reshaping, flattening, unflattening, unsqueezing, squeezing. This is like half
the code in PyTorch" (≈27:10). Does flattening a two-dimensional tensor stack its rows or its columns?
"I don't even remember." A student answers that PyTorch's reshape is "row contiguous", reading off the
first row and then the next (≈27:56).

His recommendation is the einops library. It replaces reshape, flatten and unsqueeze with one operation,
`rearrange`, which takes a string describing the mapping (≈28:43). Slide 12's example is

```python
# or compose a new dimension of batch and width
rearrange(ims, 'b h w c -> h (b w) c')
```

It maps a four-dimensional tensor with axes batch, height, width and channels to a three-dimensional
one. The parentheses flatten batch and width into a single axis, giving "an h by b times w by c tensor"
(≈29:29). The pattern written the other way round is the inverse operation, so applying one after the
other "I'll get back where I started" (≈29:29–30:17). "You can still have bugs with einops. But I found
it to be very helpful." See [tensors and batching](tensors-and-batching.md).

## Data augmentation and domain randomization

### Augmentation

Slide 13 recaps data augmentation, which the class used in problem set 1 (≈30:17). Its figure (excluded
by an OCW notice) lists training pairs $(\mathbf{x}, y)$: a clown fish labelled "Fish", a grizzly bear and
a chameleon. The fish pair fans out into four augmented copies, all still labelled "Fish": "Mirror",
two "Crop"s and "Darken". The transformations "should not fundamentally change" the data; here they do
not change the target class. "So this creates bigger data from smaller data" (≈30:17–31:02).

The lecturer takes a side in a debate. One community holds that augmentation is "a little hacky" and
should be replaced by **geometric deep learning**: instead of training for invariance to flips, crops
and lighting changes, design an architecture that is invariant by construction, such as a ConvNet for
translation equivariance followed by pooling for invariance (≈31:02–31:48). "But I think that that's
mostly not the right advice for a good hacker in machine learning." Augmentation "is really easy, really
intuitive. And it's architecture agnostic. That's the power of it". You do not need "some fancy
symmetry-preserving architecture graphnet that knows how molecule chirality should be properly
propagated". You list the transformations that are fine to apply to your data, "and I'll feed that to
any architecture. I can send this to an MLP, and it will be invariant to these operations too"
(≈31:48–32:34). As software engineering, it "decouples these properties from the architecture design".
He allows that "maybe in a few years, we'll have a very clean framework for geometric deep learning, and
then we'll go to that". See [data augmentation](data-augmentation.md) and
[inductive bias](inductive-bias.md).

Slide 14 gives the idea: "Train on randomly perturbed data, so that test set just looks like another
random perturbation. This is called **domain randomization** or **data augmentation**." Its figure is a
rectangle labelled "Data space", with training points spread over the whole of it and a few test points
in one region, inside the spread of the training points. Augmenting "just makes it so that my test data
looks like it's in distribution. My test data just looks like more training data" (≈33:20).

![Slide 14: training data spread over the whole data space, so the test data falls inside it](../raw/images/09-hackers-guide-to-deep-learning/slide-14.png)

*Slide 14 — Train on randomly perturbed data, so that the test set looks like one more random perturbation: training points cover the whole data space, test points sit inside them. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

### What does a good training curve look like?

Slide 15 is a quiz with three loss-against-iterations curves. In the room about a third of the class
picked the left one, no one the middle one, and two thirds the right one (≈33:20). The labels on the
slide give the answer.

- **Left, "Bad! Your data is too easy."** The loss drops almost at once and then stays flat. Training
  in the flat region wastes flops. More generally, "you don't want to make your problem so easy that
  training on it converges really, really quickly. You want to make your problem hard enough that you're
  actually going to learn a lot" (≈34:08).
- **Middle, "Bad! You aren't fitting your data."** The loss rises as training goes on.
- **Right, "Good. Fitting a hard problem."** The loss falls slowly and smoothly with "a long way to go"
  (≈34:55).

![Slide 15: three loss curves, too easy, not fitting, and a hard problem being fitted](../raw/images/09-hackers-guide-to-deep-learning/slide-15.png)

*Slide 15 — What a good training curve looks like: the left one is too easy, the middle one is not fitting, and the right one is fitting a hard problem. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

Slide 15's rule is "You roughly want to select data and parameters as: `max_data min_params
loss(data, params)`". The minimization over parameters "is backpropagation", but "if you're going to
actually work on a largescale project, most of your time will be on this part": changing the data to
make it harder (≈34:55). "You don't necessarily want to max over data, but you want to increase the
difficulty of your data until there's something interesting to learn from it." Augmentation and domain
randomization move a run from the too-easy curve toward the good one, and the hardest problem to train
on "is going to generalize the best" (≈35:42).

### Domain randomization

Slide 16 (excluded) shows the idea in robotics, "[Sadeghi & Levine 2016]", with its example "from
[Tobin, Fong, Ray et al. 2017]". The training data is rendered simulation with flat, random colours,
and the test data is a photograph of a real table with coloured blocks. A robot trained in simulation
under one lighting condition and one set of block colours "will only know how to pick up blocks of that
color". Randomizing them makes the problem harder: "it will take a lot longer to train, but it will
generalize better" (≈35:42–36:28).

A student asks whether the robot will generalize to lighting that never occurs in training. "The short
answer is, no". The point is to cover all possible lightings, "so that the test data just looks like
another random lighting condition". With a broad enough distribution "maybe you start to get this kind
of emergent generalization", but "it's a little more subtle, and I'm not sure that there's a clear
answer" (≈36:28–37:13).

Slide 17 (excluded) names the gap being closed. "**Domain gap** between $p_{\text{source}}$ and
$p_{\text{target}}$ will cause us to fail to generalize." Here $p_{\text{source}}$ is the distribution of
the source domain (simulation) and $p_{\text{target}}$ that of the target domain, "where we actual use our
model" (as printed). Its figure shows a rendered robot hand holding a lettered cube as source data, a
real robot hand holding a cube as target data, and a double-headed arrow between them in the "Space of
images". Domain randomization minimizes that gap "by randomizing the source domain" (≈37:59).

Slide 18 (excluded; openai.com/blog/learning-dexterity) is OpenAI's robot hand. Its table, "Ranges of
physics parameter randomizations", scales object dimensions by uniform([0.95, 1.05]), masses by
uniform([0.5, 1.5]), surface friction by uniform([0.7, 1.3]), joint damping by loguniform([0.3, 3.0]) and
actuator force gains by loguniform([0.75, 1.5]). It adds $\mathcal{N}(0, 0.15)$ rad to joint limits and
$\mathcal{N}(0, 0.4) \thinspace \text{m/s}^2$ to each coordinate of the gravity vector. The lecturer asks why anyone would
randomize gravity. The students suggest going to space or Mars, and that acceleration of a moving
machine is indistinguishable from gravity (≈37:59–39:31). The answer he thinks most important is
sensor noise. The accelerometers that measure which way is down "was just a little bit noisy", possibly
"slightly miscalibrated", and a model trained with gravity always slightly randomized stays "within
distribution" whatever the miscalibration (≈39:31).

The lesson he draws is the chart. It plots consecutive goals achieved against simulated "Years of
Experience" on a logarithmic axis. "No Randomizations" climbs fast and levels off at about 50 within a
few years. "All Randomizations" starts later and is still climbing at about 40 at 100 years. "Some
engineer on this project might have seen that curve and been like, OK, we're done. We solved it." The
randomized version took "like, 10 times longer to get to the same level", to solve "a much harder
problem" under many conditions (≈40:18–41:03). This is performance on the training data, so slide 18
concludes: "**High train accuracy can mean problem is too easy**. Add more data to make problem harder."
"A bad manager would say, oh, get rid of that. Make it learn quickly. But that's wrong" (≈41:03).

## Change the data, not the learner

### Fixed data in the academy, fixed learner in industry

Slides 19 and 20 draw the same pipeline twice: Data → Learner → $f$. On slide 19, "In the academy we
typically take data as fixed, and design models that learn from it", the data carries a padlock and the
learner is highlighted. On slide 20, "In industry, it's usually then other way around [as printed]. We
use a standard learning algorithm, and get to collect data to instruct it", the lock and the highlight
swap. "You just download the latest, greatest model, and you don't change it too much. But what you do
change is the data. That's what companies can do. That's what governments can do" (≈41:03–41:50). This is
"an even more powerful set of tools than changing the learning algorithm". The lecturer adds that it is
"probably a problem in our curriculum", which, like most MIT classes, is about algorithms rather than
data (≈42:35).

![Slide 19: in the academy the data is locked and the learner is designed](../raw/images/09-hackers-guide-to-deep-learning/slide-19.png)

*Slide 19 — In the academy: the data is fixed (locked) and the learner is what gets designed. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

![Slide 20: in industry the learner is locked and the data is what changes](../raw/images/09-hackers-guide-to-deep-learning/slide-20.png)

*Slide 20 — In industry: a standard learner is fixed (locked) and the data is collected to instruct it. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

### Prediction gets easier the longer the input

Slide 21 asks "Which is the hardest prediction problem?". It shows three prompts for a network to
complete: "It __", "Call me __" and "All happy families __". They are first lines of famous novels
(≈42:35). For the first, the lecturer supplies "It was" and a student names *A Tale of Two Cities* ("It was the
best of times"), though "there's actually a lot of other ones" and "not as many knew the first one". A student completes "Call me Ishmael", which
the lecturer identifies as *Moby Dick*. For the last one, "hopefully you can hear from the volume that a
lot of people know" it (≈43:20). The slide's answer: "Prediction gets *easier* the longer the input!"

![Slide 21: three prompts of increasing length, each into a network: which is hardest to predict?](../raw/images/09-hackers-guide-to-deep-learning/slide-21.png)

*Slide 21 — Which is the hardest prediction problem? The longer the prompt, the easier the next word. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

The lecturer thinks the field once had this backwards. Early NLP used N-gram models, "and you didn't
want to make N too large". "Really, you can make your prediction problem much easier by making N really,
really, really big" (≈44:06). He adds "a little technical distinction": "in the N-gram era, you are
modeling the joint distribution of everything. Here, you're modeling just the conditional distribution
on the last word given the previous words" (≈44:51).

### Put the universe into X

Slide 22: "You can change your data to make the learning work better!" Its example is designing a
pharmaceutical drug: $X$ → NN → $Y$, with $Y$ the drug's effectiveness. "We will treat our prediction
targets as fixed (given)." Your boss assigns $Y$, "but you can change X a lot" (≈44:51–45:37). The slide
lists what could go into $X$: "Chemical formula", "Folding structure", "Patient age", "Patient biopsy",
"… the universe". In the room the students add patient data and other medicines that worked, which the
lecturer glosses as "nearest neighbor, similar medicines". "You should put the universe in, OK?"
(≈45:37–46:23).

![Slide 22: X (chemical formula, folding structure, patient age, patient biopsy, the universe) into a network predicting drug effectiveness Y](../raw/images/09-hackers-guide-to-deep-learning/slide-22.png)

*Slide 22 — Change the data: the target Y (drug effectiveness) is given, but X can include everything from the chemical formula to "the universe". [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

He flags the tension with the earlier advice. A richer input can make the problem too easy, or let the
model overfit to a shortcut "like … the R label at the top of the cancer scan". "But I think, roughly,
this is even better advice, that you should put a lot of information into the inputs to your modeling
problem and control overfitting in other ways" (≈46:23).

### Why: make P(Y|X) a point

Slide 23, "Adding info to X reduces uncertainty over Y", gives the reason. Its picture shows one point
in an oval labelled $X$, with three dashed arrows to three places in a blob inside an oval labelled $Y$:
one observation, several possible targets.

![Slide 23: one observation X maps to a spread of possible values of Y](../raw/images/09-hackers-guide-to-deep-learning/slide-23.png)

*Slide 23 — Adding information to X reduces uncertainty over Y: one observation with a whole region of possible targets, which a point-prediction network cannot represent. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

Its bullets say that "it's really hard to model a complicated *distribution*, P(Y|X), over all the
possible values of Y for some given observation X (we will get to this in the generative modeling
lectures)". "Standard NN regression outputs a single point prediction for each X." "The hack is to put so
much info in X that P(Y|X) looks like a single point!" The standard training of the course so far, an
L2 loss for regression, makes a point prediction "so it doesn't handle modeling a whole distribution of
possibilities" (≈47:10–47:56). Putting enough information in $x$ makes $P(y \mid x)$ "a delta function",
and "this is the hack that led to text-to-image models working so well".

Slide 24, "Evolution of image generation", is the evidence. In 2019, StyleGAN2 was trained on a set of
face images alone and became "a generative model that can only make frontal views of faces". In 2021,
DALL-E was trained on pairs of text and image, such as "an illustration of a baby daikon radish in a tutu
walking a dog". It "can make basically any image you can think of". The older models "just model Y …
we had no information about X, and we just have to model the entire possible space of all faces", which
only works for narrow categories (≈47:56–48:42). Systems like DALL-E, Stable Diffusion and Midjourney
work because "we switched to giving a lot of information into the input to make it so that the problem of
modeling data is almost deterministic" (≈48:42–49:29). The lecturer's reading: "I don't think we've made
that much progress in generative modeling on modeling complicated distributions. We've actually just
switched to modeling simple Gaussian distributions conditioned on a lot of information."

![Slide 24: StyleGAN2 trained on faces alone versus DALL-E trained on text-image pairs](../raw/images/09-hackers-guide-to-deep-learning/slide-24.jpg)

*Slide 24 — Evolution of image generation: StyleGAN2 (2019) learns from images alone and makes only frontal faces; DALL-E (2021) learns from text–image pairs and makes almost anything. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

### Make it big

Slide 25 sums up the data section: "Use **big** data". Big in three senses: "lots of {x,y} training
pairs", "x is a big object, replete with information (+ high-dimensional)", and "(y is a big object
too)". The last is "a little bit more subtle", left for "generative modeling and foundation models"
(≈49:29–50:16).

A student who worked on crash detection at an insurance-technology company pushes back. True positives
are rare, "five car crashes for every million miles of driving". People on roller coasters trigger false
detections, and "we end up training on 8-terabyte data sets that have 2,000 crashes" (≈50:16–51:04). The
lecturer agrees that scaling the data might not be the most efficient way to capture rare classes,
"because you'll be wasting most of your scale on the common case". "Data diversity and coverage over the
classes of interest, that's the key thing that matters. So scale is kind of a proxy for, actually,
coverage." For the roller coaster, add context to the input: if the system knows it is at an amusement
park, a crash becomes less likely. This is what language models' push to million-token contexts is
"basically trying to achieve" (≈51:49–52:35).

## Model

### Keep it simple

Slide 26 is headed "Keep it as simple as possible!", and its first line is "do your first experiment
with the simplest possible model w/ and w/o your idea". "Why keep it simple?" Simple models are "easy to
build, debug, share" and "tractable to understand, make robust, build theories around", and "*simple
models also work better* (Occam's razor, Solomonoff Induction)". Finally, "if you focus on simplicity you
will have an unfair advantage". The theory side of the course works with MLPs "because it's easier on
the simpler models" (≈52:35–53:20). "Simple models are also the most powerful models. Well, the simplest
model that fits the data … going to be the best model," by Occam's razor and its formalizations in
generalization bounds (≈53:20–54:06; see [lecture 6](06-generalization-theory.md)).

His advice to his students: "If method A gets 1% better on benchmarks than method B, but method A is
more complicated than method B by some factor, go with the simpler — publish the simpler method"
(≈54:06). "Don't fight for that last 1% by adding complexity", unless "you're, like, saving lives at a
hospital" (≈54:54). Asked how to measure that trade-off, he says he does not know: "I just think most
people are calibrated too much toward accuracy … it'd be lovely to have a way of measuring simplicity.
But again, we don't really have that" (≈54:54–55:40).

Another student asks whether to use out-of-domain data in training. "Yes." "If you have data x, y, and z
and your problem is about data z, you should train on x, y, and z," since x and y are "strictly
beneficial towards z if you're doing things right … You could always just ignore x and y if you had the
optimal algorithm" (≈55:40–56:26).

### Start with a standard, popular, pretrained model

Slide 27: "start with a standard and popular model (popularity matters more than performance)". For
image classification, `torch.hub.load('pytorch/vision:v0.9.0', 'resnet18')` (pytorch.org/hub). For a
text problem, a Hugging Face pipeline with `model='gpt2'`. Popular models and code are on
paperswithcode.com. "Popularity matters more than performance. And simplicity matters more than
anything" (≈56:26). Popular systems "interface well with the community", and you won't be "building some
amazing palace that nobody ever visits".

His rule of thumb is that GitHub stars or forks are "a good proxy for quality … Don't look at the
leaderboard. Look at the number of stars" (≈57:11). Take the leaderboard, "go down the list until you
find one that has 10,000 GitHub stars, and do that one" (≈57:57). He notes the limit: this advice does
not apply to cutting-edge research, "but if you're just trying to solve a problem at a company,
popularity matters more" (≈57:11).

Slide 28 (excluded) says "stand on the shoulders of giants. use pretrained models (but be aware of their
flaws and limitations)". It shows screenshots of the Stable Diffusion project page and the AlphaFold
repository. "You generally don't want to be training your systems from scratch." Be aware of "the flaws
and the copyright issues and the biases", though some, "like AlphaFold and AlphaFold 2", are "probably
pretty safe to use" (≈57:57–58:43). Pre-training and transfer learning come "a little bit later" in the
course.

### Transform your problem into a solved problem

Slide 29: "**Transform your problem into a 'solved' problem**", with the case study "transforming image
*colorization* to image *classification*" and the aside "[c.f. the strategy of 'polynomial-time
reduction']". The lecturer compares it to complexity theory: if you can convert your problem into a known
one, you understand it (≈58:43–59:31).

The case study is his own project from "almost 10 years ago", cited on the slides as "[Zhang, Isola,
Efros, ECCV 2016]". Every colorization image is excluded by OCW notices. Slide 30 (excluded) shows
training pairs of greyscale and colour photographs (a lionfish, blossoms with an insect, a dessert) and a
greyscale angelfish mapped by $f$ to its colourized version. The task is set up as empirical risk
minimization,

$$\arg\min_{f \in \mathcal{F}} \sum_{i=1}^{N} \mathcal{L}(f(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})$$

where $\mathcal{F}$ is the family of candidate functions, $\mathcal{L}$ the loss and
$(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})$ the $N$ training pairs. As printed, the parentheses do not
balance: the closing one of $f(\cdot)$ is missing. Slides 31 and 32 (excluded) give the shapes. The input
is the greyscale "L channel", $\mathbf{x} \in \mathbb{R}^{H \times W \times 1}$, and the output is the
colour information, the "ab channels", $\mathbf{y} \in \mathbb{R}^{H \times W \times 2}$, for an image $H$
pixels high and $W$ wide. The lightness need not be predicted, "it's already in the input", so the
predicted colours are concatenated with it to give the colour photograph (≈59:31–1:00:18).

"At first, you're thinking, this doesn't look very much like just a classification problem. I'm trying
to do something like regression" (≈59:31). The reduction takes two steps.

- **Colours into classes.** Slide 33 (excluded) replaces the colour output with a single label,
  "yellow". Slide 34, "Colors → Classes", quantizes the plane of $(a, b)$ colour values into a grid of
  cells, each a class: $\mathbf{y} \in \mathbb{R}^{H \times W \times 2}$ becomes
  $\mathbf{y} \in \mathbb{R}^{H \times W \times K}$, a "one-hot representation of K discrete classes"
  (one-hot codes such as $[0,0,1, \ldots]$). "Rather than cat, dog, elephant, it will be the blue class,
  the yellow class" (≈1:00:18). On slides 35 and 36 (excluded) a six-layer network that outputs "rockfish"
  can equally output "yellow". "It's the same math."
- **Image classification into pixel classification.** On slides 37–39 (excluded), a patch around one
  input pixel predicts the colour class of that one output pixel, and sliding it over the image colours
  every pixel. A student names the architecture: a ConvNet. "ConvNet just says, I will predict the label
  of the center pixel of a patch. And if I slide that ConvNet across the whole image, I will be
  predicting the label, the color, of every pixel" (≈1:01:06). See [convolution](convolution.md).

![Slide 34: the continuous ab colour plane quantized into a grid of colour classes, with one-hot codes](../raw/images/09-hackers-guide-to-deep-learning/slide-34.jpg)

*Slide 34 — Colors into classes: the continuous plane of ab colour values is cut into a grid of cells, and each cell becomes one class with a one-hot code. [Deck](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec9.pdf)*

### Formulate it as softmax regression

Slide 40: "Formulate your problem as **softmax regression** (a.k.a. classification)", that is, a
cross-entropy loss with one-hot labels. It gives three reasons.

1. "No restriction on shape of predictive distribution [up to quantization] (this is *not* the case for
   least-squares regression, which assumes Gaussian predictions)." The predicted categorical
   distribution is "fully expressive over the space of probability mass functions over the discrete
   vocabulary … the most general way of quantifying uncertainty". Least-squares regression "can only
   represent Gaussian distributions" (≈1:01:51–1:02:38).
2. "Discrete classes are easy to label", and easy for people to talk about and understand.
3. "All labels are equidistant under 1-hot representation." "The distance is always 1 or 0", so this
   "removes inductive bias about how we represent target variables" (≈1:02:38–1:03:24).

See [softmax and cross-entropy](softmax-and-cross-entropy.md).

### Good default choices, circa 2024

Slide 41 is marked "good default choices ca 2024". Its "Recipe for deep learning in a new domain":

1. Transform your data into numbers (one-hot vectors).
2. Transform your goal into a numerical measure (cross-entropy loss).
3. Use a generic optimizer (Adam) and a standard architecture (transformer) to solve the learning problem.

(The slide prints "an numerical measure" and "an standard architecture".) "This will work pretty well"
(≈1:03:24). See [transformers](transformers.md).

### Don't use batch norm

Slide 42, also "good default choices ca 2024": "Don't use batch norm". "I think that lesson has kind of
gotten out now" (≈1:04:10). Its reasons are these.

- It "introduces a strong dependency on batch size (now batch size becomes an even more critical
  hyperparameter)". Batch norm normalizes activations by the statistics of the batch, so "your performance
  will change dramatically when you change batch size … I have to retune my whole model" (≈1:04:10–1:04:56).
- "Different behavior at train and test time." At test time there is often a single data point, and
  learned statistics stand in for the batch.
- It "makes distributed computing hard — requires communication between all elements in a batch". You
  cannot simply shard a batch across the nodes of a cluster.
- "Use layer norm instead", "the standard one right now" (≈1:04:56).

Slide 43 reproduces "a longer rant I wrote a few years ago", for reading after the lecture (≈1:05:43). It
has six points.

1. Train and test behaviour differ ("Forgot to set model.eval()? You will have a bug").
2. Batch norm only works if the batch is big enough. With batch size 1, subtracting the batch mean gives
   $\mathbf{z} - \mathbf{z} = 0$ and the variance is undefined. This was the bug in the original pix2pix
   paper, "and it made the baseline work worse than it should have (which might not have been a bad
   thing for making the paper popular…)".
3. A large batch whose activations happen to be equal fails the same way. That is the "bug" the SPADE
   paper tried to fix.
4. Data-parallel training must communicate activations between machines after each layer.
5. Batch elements are no longer processed i.i.d., which complicates theory, "one reason why the theory
   of why batchnorm works is still not really resolved".
6. The fixes add bugs and complexity. Running batch norm separately on each machine makes "your results
   change dramatically depending on how many machines are in your cluster".

See [normalization layers](normalization-layers.md).

### Scale

Slide 44: "Remember that often the easiest way to get better performance is: 1) **Scale** your data:
more (diverse) training examples. 2) **Scale** your model: more layers, more channels. 3) **Scale** your
compute: train for longer. In the current era, I would say these are the top three factors that
determine success … but working at small scale forces efficiency, and *then* when you do scale up, you get
more bang for your buck." The lecturer calls this "kind of the scale is all you need recipe … not
necessarily the entire story, but it is part of the story" (≈1:05:43).

His story from the colorization project: he and his collaborator Richard (Richard Zhang, by the
slides' citation the paper's first author) spent a month visiting
a lab in France while a baseline, the standard convolutional classifier, kept training. "We just turned
the GPU on … it had horrible results at the time." Two weeks later "it's beautiful. And it's better than
all the kind of crazy advanced architectures and variants that we tried. So just if you're ever not sure
what to do, just train for longer. Go to France. Go to the beach. Relax" (≈1:05:43–1:07:18).

Many people call this "the bitter lesson", and some think it diminishes science. "But it's like, truth
might be simple and easy. And if it is, all the better. It's, to me, very sweet." There are reasons to
work small: fast iteration, and pressure to find algorithms that scale better. His summary is that
"scaling is necessary but not sufficient … I don't really think there's going to be any way of making
performant, human-like intelligence without massive scale" (≈1:07:18–1:08:04). See
[scaling laws](scaling-laws.md).

### Remove everything nonessential

Slide 45: "Once you get your system working, you are only halfway done. Second half is to remove
everything nonessential." It quotes Antoine de Saint Exupéry, whom the lecturer introduces as the author
of *The Little Prince*: "Perfection is finally attained not when there is no longer anything to add, but
when there is no longer anything to take away." Components that mattered early may stop mattering once
the system is scaled up or other parts work. "Add, add, add until it works. Remove, remove, remove until
it stops working. And cut off there. And then report it" (≈1:08:04–1:08:50).

### Copilots

Slide 46: coding assistants are "good for boilerplate code, visualization, getting syntax right. Ever
improving. Don't use it for your psets but you can use it for your final projects. You *should* learn
how to use these tools effectively. Think first, then ask an LLM for help. Don't trust the code without
verification." In the recording the lecturer restates the course policy: "You shouldn't use AI
assistants in a way you wouldn't use a friend." You can ask them conceptual questions, but they should
not write the code for you. The next problem set would clarify how to turn off Colab's autocomplete
(≈1:08:50). For final projects "and for life in general, you should absolutely use coding assistants",
but "don't replace thinking. Think first, and then ask the LLM" (≈1:09:36). See the
[course map](course-map.md) for the policy.

Slide 47: "the more documentation you provide, the better the completion will be". Its example is a
function stub, `backward_D_basic(self, netD, real, fake)`, whose docstring describes the GAN
discriminator's loss. An assistant's completion fills in the body, computing binary cross-entropy on
real and fake predictions. "Write a very clear docstring, and then autocomplete … will work pretty
well." "You can prompt it by saying, you're a really smart coder, if you want. But I think some of those
tricks will disappear" (≈1:09:36–1:10:21).

## Optimization

The recording covers slides 48, 49 and 54 of this section, briefly. The rest comes from the slides
alone.

**One, few, many** (slide 48, spoken at ≈1:10:21): "figure out optimization on **one/few/many
datapoints**, in that order": overfit to a data point, then fit a batch, "and finally try fitting the
dataset (or a miniature version of it)". "First make sure you can fit train set, then consider
generalization to test set." "So scaling comes last."

**Sanity-check the loss** (slide 49, ≈1:10:21–1:11:53): check it "against a suitable reference value".
For classification with cross-entropy loss, the reference is the uniform distribution. "Get to know log
loss numbers": $-0.69 = \ln(0.5)$ is chance on binary classification, and $-2.3 = \ln(0.1)$ chance on
10-way classification. "If you get that number as your loss, is that good? … No, that's chance"
(≈1:11:07). The slide prints these as log probabilities. Under the course's definition of cross-entropy
as a negative log probability, the loss at chance is their negative, 0.69 and 2.3. For regression with
squared loss, the reference is "mean of targets (or even just zero)". A squared loss "better not be
negative, because the minimizer of least squares is 0 … If you see negative 1,000 on a regression loss,
you know there's a bug" (≈1:11:53). The slide adds: "and if your loss is constant, double check for zero
initialization of the weights".

**Learning rate and batch size** (slide 50; the lecturer says only "always tune your learning rate and,
often, your batch size", ≈1:11:53). "Most important hyperparameters: **learning rate** and **batch
size**." First "use a constant rate; don't schedule until everything else is figured out". "Schedule
according to number of iterations of SGD, not epochs." "Use biggest batch size that will fit in memory."
And "always retune lr when *anything* changes in your model (most changes to model change scale of
gradients, which changes the **effective lr**)". A side note says a change "may look like your model is
training much faster, but really you just scaled the effective lr". Another reads "Until Jeremy solves
lr-free optimization", a nod to co-instructor Jeremy Bernstein, whose [lecture 7](07-scaling-rules-for-optimization.md)
was about learning rates that transfer across scale. See [gradient descent](gradient-descent.md).

**Epochs** (slide 51): "Be careful with the concept of 'epochs'". "There are no epochs in the wild",
there is a "trend toward single-epoch training in LLMs", and you should not tie the learning-rate schedule
to epochs. Doing so "makes it hard to compare learning curves between experiments", and you should "be
careful with cosine lr (looks like it is converging when it is not)". The slide shows a table from
"[Tian, Sun, Poole, et al., 2020]" comparing self-supervised methods on ResNet-50, with epochs from 200 to
1000. The InfoMin method reaches 70.1 top-1 at 200 epochs and 73.0 at 800, against SimCLR's 69.3 at
1000. The slide does not say what the table is meant to show; it sits beside the bullets about comparing
runs of different lengths.

**Checkpoints** (slide 52): "**checkpoint** features + gradients to trade space for time and fit large
models". You can then "accumulate gradients across checkpoints", "resume training if your computer
crashes", and "have a 'paper trail' to debug later".

**Live on the edge** (slide 53): "try extreme settings (but just a little bit). If optimization never
diverges, your learning rate is too low", "in the style of *Umeshism*": "If you've never missed a
flight, you're spending too much time in airports".

**Exponential moving averages** (slide 54, ≈1:11:53–1:13:23): "Use exponentially moving averages (EMA)".
The slide writes the update for a quantity $\theta$ at step $t$ as

$$\theta_{\texttt{EMA}}^{t} \leftarrow \beta \thinspace \theta_{\texttt{EMA}}^{t-1} + (1 - \beta) \thinspace \theta^{t-1}$$

with a decay factor $\beta$ between 0 and 1. The slide leaves the assignment sign blank; the arrow here
marks the assignment the bullets describe. It means "replace a quantity with a weighted average of its
previous values, with weight exponentially decaying over time". The lecturer frames it as a trade
between space and time. Averaging gradients over a batch is an average "across space", and an EMA
averages "across time, across iterates of your learning" (≈1:12:38). The slide says that "time averages
(EMA) can achieve a similar effect as 'space' averages (e.g., average gradients over batch)", and that
an EMA is useful for "gradients (where it is known as momentum), weights, data, activations, targets,
etc. (Basically for any variable in DL, try replacing it with its EMA version and it may be better)". In
PyTorch "you can … just have some wrapper" that turns a variable into its EMA version (≈1:13:23).

**Optimizers** (slide 55): "Adam (or AdamW) is good for prototyping (generally just works)". "SGD may be
slightly better for performance (but requires more tuning of hyperparameters)". "Clip gradients to
improve stability." See [lecture 2](02-how-to-train-a-neural-net.md) for momentum and gradient clipping.

## Evaluation, tuning and debugging

These slides are not spoken, apart from the spice rack, so this section is from the slides.

**Evaluation mode** (slide 56): "switch to **evaluation mode** by model.eval() (PyTorch). No, really.
And check the mode by model.training." The rant on slide 43 names forgetting it as a batch-norm bug.

**Look at the output** (slide 57) repeats slide 4's three output pictures (above). Slide 58 (excluded,
"© IEEE") is a grid of image-generation results that the slide does not identify. Rows of
building-facade label maps and their ground-truth photographs sit beside the outputs of five model
variants, whose column headings print as "Ll", "Olayers", "llayers", "3layers" and "61ayers", most likely L1 and 0,
1, 3 and 6 layers with letters and digits swapped. The first two columns are blurry and the others
progressively sharper.

**Logging** (slides 59–60): "**WandB** and **Tensorboard** can be your friends (?). Or roll your own
logs/viz. When in doubt, log it. If you're logging it, make it easy to see the results." Slide 60
(excluded) is a screenshot of the Weights & Biases website (wandb.ai/site).

**Tuning** (slide 61, ≈1:13:23–1:14:54): a photograph (excluded) of a spice rack whose jars are
relabelled "dropout", "attention", "weight decay", "skip connections", "momentum" and "relu", over the
caption "Cayenne pepper is all you need? No! Each spice has its use. But combination matters. And don't
over spice." "So attention is not all you need. Attention is one thing that works pretty well and has
certain effects." Tuning is "really like cooking … It's very multimodal in this search landscape". You
need some regularization. You need skip connections "if you have a really deep architecture. But maybe
you don't if you have an optimizer which somehow doesn't have vanishing gradients" (≈1:14:09). Many
combinations are equally valid, like "a nice, spicy chicken dish" or "a nice sushi with miso and soy
sauce". "Some ingredients regularize. Some ingredients increase capacity. Some ingredients make
optimization faster … And don't overspice. Don't overrely on one" (≈1:14:54). See
[skip connections](skip-connections.md).

**Script everything** (slide 62): "**don't be finger-bound!** script the optimization + evaluation of your
models. Every character you type is a chance to make a mistake. Also scripting makes the work
reproducible! Use config files (e.g., yaml) to manage experiments; log all arguments."

**Debugging** (slide 63): use Python's own debugger, `import pdb; pdb.set_trace()`. To find NaNs, use
`torch.autograd.set_detect_anomaly(True)`.

**Common bugs** (slides 64–67), each with its PyTorch error and fix.

- **In-place operations on a leaf** (slide 64). "RuntimeError: a view of a leaf Variable that requires
  grad is being used in an in-place operation." With `x = torch.ones(2,2, requires_grad=True)`, the
  in-place `x += 1` and `x[0,0] = 1` fail, while `x = x + 1` works. So does editing a clone,
  `y = x.clone(); y[0,0] = 1`, "(but what should the gradient be?)". "A leaf variable is one that you
  directly create, that is not the result of any differentiable operation. These are the leaves, the
  inputs, to the computation graph." See [backpropagation](backpropagation.md).
- **Out of memory** (slide 65). At inference, do not store gradients: run the forward pass inside
  `with torch.no_grad():`. Clear memory where appropriate with `torch.cuda.empty_cache()` and
  `del variable_name`.
- **Timing your code** (slide 66). "GPU calls may run asynchronously, so if you want to time an
  operation, make sure to synchronize first": call `torch.cuda.synchronize()` before starting the timer.
- **Backward twice** (slide 67). "RuntimeError: Trying to backward through the graph a second time …
  Specify retain_graph=True when calling backward the first time." "PyTorch frees computational graph
  after calling backward(). If you see this error it's likely you are doing something you don't want to
  be doing." If you do want to, pass `retain_graph=True` to the first `backward()`.

## Compute

This section is from the slides alone.

**More hardware, more problems** (slide 68): "don't parallelize immediately". Make the model work on a
single device, then try to parallelize on a single machine, "only then go to a multi machine set", "and
check that iterations/time actually improves". The slide points to the paper "Accurate, Large Minibatch
SGD: Training ImageNet in 1 Hour" "for good advice".

**Saturate your GPUs** (slide 69): check GPU utilization (memory and flops) with `nvtop` or `nvidia-smi`,
and "increase batch size until ~100% utilization".

**Speed-ups** (slide 70): "Include this at the top of your scripts: `torch.cudnn.benchmark = True`". Try
automatic mixed precision (AMP; developer.nvidia.com/automatic-mixed-precision), shown as a `GradScaler`
and an `autocast()` block around the forward pass and loss, with `scaler.scale(loss).backward()`,
`scaler.step(optimizer)` and `scaler.update()`. And try `torch.compile`.

*Note, from outside the course material: in PyTorch the cuDNN flag lives at
`torch.backends.cudnn.benchmark`; the slide's `torch.cudnn.benchmark` is transcribed as printed.*

**Scale ML** (slide 71, excluded): an "MIT group with presentations / tutorials on cutting edge practice
of training big models", scale-ml.org. The screenshot lists its seminar schedule, which includes a
hands-on session by Jeremy Bernstein, "How to scale models with Modula in NumPy".

## See also

- [Data augmentation](data-augmentation.md) — label-preserving transformations, augmentation against
  invariant architectures, domain randomization and the domain gap, and making the problem harder on
  purpose.
- [Normalization layers](normalization-layers.md) — batch norm, layer norm (token norm) and RMS norm,
  why they misbehave in low dimensions, and the case against batch norm.
- [Softmax and cross-entropy](softmax-and-cross-entropy.md) — softmax regression as the default
  formulation, colorization as classification, and the log-loss values at chance.
- [Tensors and batching](tensors-and-batching.md) — inspecting tensors, prime-sized dummy dimensions,
  dtype casts and einops.
- [Gradient descent](gradient-descent.md) — the learning rate and batch size, one data point first,
  EMAs, Adam against SGD.
- [Inductive bias](inductive-bias.md) — augmentation as an architecture-agnostic way to get invariance,
  and the biases that standardization and one-hot labels remove.
- [Scaling laws](scaling-laws.md) — scale data, model and compute, and scaling as necessary but not
  sufficient.
- [Generalization and double descent](generalization-and-double-descent.md) — why a problem that is too
  easy generalizes badly, and the random-labels paper behind "hacking over theory".
- [Lecture 6 — Generalization Theory](06-generalization-theory.md), which the lecturer cites for the
  random-labels result, and [Lecture 8 — Architectures: Transformers](08-architectures-transformers.md),
  the architecture of the default recipe.
