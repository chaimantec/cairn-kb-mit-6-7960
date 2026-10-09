# Self-supervised learning

Self-supervised learning learns a representation from unlabelled data by **predicting part of the raw data from
another part**. "It's called self-supervised learning because it's using the machinery of supervised learning,
meaning predict y from x, except that we define y as being some part of the raw data as opposed to some label"
([lecture 11](11-representation-learning-reconstruction-based.md), ≈1:06:02). Lecture 11 calls the move to it "the
big finding over the last decade that led to this revolution in how we do deep learning, which was the move from
supervised learning to self-supervised learning" (≈1:09:53). Covered so far: lecture 11, slides 33–35 and 50–64,
≈43:23–45:43 and ≈1:05:15–1:20:51; colorization as a project in [lecture 9](09-hackers-guide-to-deep-learning.md),
slides 29–39; next-word prediction in [lecture 8](08-architectures-transformers.md) and
[lecture 10](10-architectures-memory.md).

## Learning without labels

Supervised learning fits a function to example pairs (lecture 11, slide 32):

$$f^{\ast} = \arg\min_{f \in \mathcal{F}} \sum_{i=1}^{N} \mathcal{L}(f(\mathbf{x}^{(i)}), \mathbf{y}^{(i)})$$

where $\mathcal{F}$ is the hypothesis space, $\mathcal{L}$ the loss and $(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})$ the $N$
training pairs. Training on a labelled task does induce a representation, "and that's OK. That works sometimes"
(≈43:23). But it ties the representation to that task and needs humans to label it. "Learning without examples"
(slide 33), which "includes **unsupervised learning** / **self-supervised learning**", starts from data that "is not xy
pairs, but it's just x", and can learn embeddings, clusters or metrics (slide 34).

There are, in the lecturer's opinion, "two general principles for how to learn a good vector embedding without having
an explicit supervised task": compression and prediction (≈44:11–44:58). Slide 35 files autoencoding, contrastive
learning and clustering under compression, and future prediction, imputation and pretext tasks under prediction, then
asks "are these actually different?" "You can actually even see the compression prediction as fundamentally the same.
I'm not sure if there's really a difference, but at least it's useful, intuitively" (≈44:58). Compression is the
subject of [autoencoders](autoencoders.md); this page is about prediction.

## The trick: a pretext task

"Common trick: Convert 'unsupervised' problem into 'supervised' empirical risk minimization. Do so by cooking up 'labels'
(prediction targets) from the raw data itself — called **pretext task**" (slide 56). We "switch it into the mathematical
machinery of supervised learning by just predicting some fake labels, which are just raw data" (≈1:09:53). Slide 57 sets
three side by side. Predicting a class label "would be called supervised learning of representations". Predicting the
next frame of a video, or the next pixel of an image, is self-supervised "because no human had to provide the label
target" (≈1:09:53–1:10:38).

Compared with an autoencoder, "I'm predicting half of the data from the other half of the data. And interestingly, this
works really well. This tends to work a lot better than autoencoders" (≈1:06:50). Slides 50–52 draw the three cases:
data compression ($\mathbf{X}$ to $\hat{\mathbf{X}}$), label prediction ($\mathbf{X}$ to $y$) and data prediction ("Some
data" $\mathbf{X}_ 1$ to "Other data" $\hat{\mathbf{X}}_ 2$).

## Colorization, and what its units learn

The running example is colorization, the lecturer's own project ([Zhang, Isola, Efros, ECCV 2016]): predict an image's
colour channels from its grayscale channel, input $\mathbf{X} \in \mathbb{R}^{H \times W \times 1}$ and output
$\widehat{\mathbf{Y}} \in \mathbb{R}^{H \times W \times 2}$ (lecture 11, slide 53). The labels are free, "because color
images have the colors built in. I just took the color image. I split it into the luminance channel and the color
channels" (≈1:06:50–1:07:35). Lecture 9 uses the same project as a case study in turning a problem into a solved one,
recasting colour prediction as classification over quantized colours (slides 29–39).

Probing the trained network's layer 5 (slide 55), units fire on faces, dog faces and flowers: objects, which the class had
guessed because "Different classes of objects have different colors" (≈1:08:22). The lecturer draws a general lesson:
"it doesn't really matter how you train these networks. If you train them to predict classes, if you train them to
predict colors, if you train them to inpaint missing pixels … the units that carve the world at its joints, that are
predictive of everything, turn out to be objects and semantics and the words that humans have. So it's like words are not
arbitrary. We have the words we have because they're very predictive statistically of missing data" (≈1:09:08).

## Imputation: one pretext task to rule them all?

"All of these self-supervised tasks can be understood as something we call imputation, which just means take your data
— it could be a matrix, a tensor … mask part of it, and put that into an encoder. Decode it to predict the masked part"
(slide 58, ≈1:10:38–1:11:26). The masked part can be spatial (the next pixel, half of an image, scattered patches),
temporal (the next frame) or a channel (colour from grayscale). "Masked prediction or imputation is the standard pretext
tasks that people like to use these days to learn representations" (≈1:11:26).

## Masked autoencoders, BERT and next-word prediction

The **masked autoencoder** (slide 59, He, Chen, Xie, et al. 2021) applies imputation to a vision transformer's tokens:
mask random patches, encode the rest, and decode blank tokens into the missing pixels. Because "each token will attend to
each other token", the network takes any number of visible tokens, "this nice kind of architectural invariance to the
number of tokens you put in" (≈1:12:14–1:13:01). "Masked autoencoders are just a new name for another model which was
very popular called BERT", which masks tokens of text (slide 60, ≈1:13:01–1:13:47).

Next-word prediction is the same pretext task with only the final token masked: autoregressive language models "are just
the same, except they're only masking the final word as opposed to interleaving words" (≈1:13:47). "Most people still
call language models self-supervised because they're just predicting the raw data. It happens the raw data is semantic
and is words" (≈1:10:38). Masking only the future suits generation, since in a conversation "I have to answer the question
after the question has been asked"; once the biggest models all masked only the future, "they just worked the best. But
if I want to learn a sentence embedding, I bet the BERT method is still going to work better if scaled the same amount"
(≈1:14:33–1:15:20). See [autoregressive models](autoregressive-models.md) and [transformers](transformers.md).

## Why prediction beats reconstruction

Slide 61 compares, layer by layer, an autoencoder with colorization by how well a linear classifier on each layer decodes
ImageNet categories: the autoencoder "learns an OK representation, but masked prediction learns a representation which is
more semantic" (≈1:16:06). Why is "ongoing science". Slide 62's hypotheses:

1. A dimensional bottleneck is hard to control, while masked prediction controls compression "by the non-overlap between
   the outputs you're predicting and the inputs you're conditioning on", keeping only what the inputs say about the outputs
   (≈1:16:52).
2. Autoencoders take shortcuts, copying part of the input for a decent loss (≈1:17:41).
3. Prediction "is closer to the downstream problems we care about, which are mainly about prediction" (≈1:18:30).

Kaiming He, the masked autoencoder's first author, told the lecturer "at the end of the day, it's just empiricism"
(≈1:18:30). See [autoencoders](autoencoders.md#reconstruction-against-prediction).

## The cake

Yann LeCun's slide (lecture 11, slide 63) measures what each kind of learning gives the machine: "A few bits for some
samples" in pure reinforcement learning (the cherry), "10→10,000 bits per sample" in supervised learning (the icing), and
"Millions of bits per sample" in self-supervised learning, where "The machine predicts any part of its input for any
observed part" (the cake génoise). "Self-supervision is about not having labels, but it's also about not being narrow and
tied to a task. It's just generally compress the universe into something that's predictive of the future and is compact
… representation learning is the bulk of intelligence, and I agree with that point" (≈1:20:05).

## See also

- [Representation learning](representation-learning.md) — what the learned representation is and how it is probed.
- [Autoencoders](autoencoders.md) — the compression route.
- [Transfer learning](transfer-learning.md) — what a pretrained representation is for.
- [Lecture 11](11-representation-learning-reconstruction-based.md) — the lecture these sections come from.
