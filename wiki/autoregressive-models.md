# Autoregressive models

An **autoregressive model** generates a sequence one element at a time: given the beginning of a sequence it
predicts what comes next, the prediction is appended to the input, and the model is asked again. Trained as a
classifier of the next element, it is how the course describes modern language models: "This is how ChatGPT
works" ([lecture 8](08-architectures-transformers.md), ≈1:06:38). Covered so far: lecture 8, slides 48–52,
≈1:05:51–1:10:38 (autoregression, GPT and causal masking); [lecture 9](09-hackers-guide-to-deep-learning.md), slide 21,
≈42:35–44:51 (longer prompts make the next word easier to predict); [lecture 10](10-architectures-memory.md), slides
40–53, ≈41:53–58:19 (the probability model, next-word classification, the choice of vocabulary, a molecule-to-text
model, teacher forcing, sampling and beam search); [lecture 11](11-representation-learning-reconstruction-based.md), ≈1:10:38 and ≈1:13:47–1:15:20
(next-word prediction as self-supervised learning, set beside BERT's masking); [lecture 14](14-generative-models-basics.md), slides 35–42 and
52–53, ≈59:43–1:07:34 and ≈1:15:21–1:16:52 (autoregressive models as generative density models, of words, pixels and audio, and their kinship with
diffusion models); [lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md), slide 41, ≈1:20:09 (the families compared); [lecture 16](16-generative-models-conditional-models.md), slides 52–56 and 70, ≈45:58–53:47 and ≈1:07:53 (conditional by
construction, prompting, captioning, and a prior over VQGAN's codes). [Lecture 18](18-transfer-learning-models.md), slides 48–54, ≈1:09:52–1:12:14, adapts GPT-3 by prompting and in-context learning, and
[lecture 19](19-transfer-learning-data.md), slide 50, ≈1:13:12–1:14:43, calls in-context learning meta-learning.
Generative models are the subject of lectures 14–16. [Lecture 21](21-language-models.md), slides 4–17, ≈2:20–19:20, traces language
models from counted n-grams to LLMs; see [language models](language-models.md). [Lecture 24](24-inference-methods-for-deep-learning.md), slides 11–19, ≈14:43–32:38, treats decoding as
search: why greedy decoding misses the most likely sequence, Best-of-N, beam search and chain of thought.

## Predict, append, repeat

Both lectures use the same picture (lecture 8's slides 48–49, lecture 10's slides 41–42). A predictor fills in
"Once upon ___" with "time". It is trained by supervised learning on pairs of a sequence and its next word ("Once
upon a" → "time", "To be or not to" → "be"), which for language models "is just text online" (lecture 8, ≈1:07:25).
At test time it continues "Colorless green ideas sleep" with "furiously", and the prediction is fed back in, "again,
and again, and again" (lecture 10, ≈44:15–45:02). Nothing forces left-to-right order: the same framework can fill in a
blank in the middle, "Once ___ a time" (lecture 10, ≈43:29), "But that's just a detail" (lecture 8, ≈1:07:25).

Lecture 10 calls this "how we think about doing modern generative AI" (≈42:41). An RNN can predict the next word, but
it does not by itself feed its output back as the next input; "you could totally loop it around that way" (≈45:02–45:48;
see [recurrent neural networks](recurrent-neural-networks.md)).

## The probability of a sequence

Any joint distribution over a sequence factorizes into next-element conditionals (lecture 10, slide 43):

$$p(\mathbf{X}) = \prod_{i=1}^{n} p(\mathbf{x}_ i \mid \mathbf{x}_ 1, \ldots, \mathbf{x}_ {i-1})$$

"This is true for any probability distribution" (≈46:33). So
$p(\texttt{Once upon a time}) = p(\texttt{Once}) \thinspace p(\texttt{upon} \mid \texttt{Once}) \thinspace p(\texttt{a} \mid \texttt{Once, upon}) \thinspace p(\texttt{time} \mid \texttt{Once, upon, a})$,
the likelihood of the sentence (≈47:22–48:08). Lecture 9 makes the related point that conditioning on more context makes
the prediction easier ("Prediction gets *easier* the longer the input!", slide 21), and contrasts the N-gram era, which
modeled "the joint distribution of everything", with predicting "the conditional distribution on the last word given the
previous words" (≈44:06–44:51).

## Each factor is a classifier

"Just treat it as a next word classifier!" (lecture 10, slide 44): a network scores every candidate, the softmax turns the
scores into a distribution, and cross-entropy trains it (≈48:55–49:41; see
[softmax and cross-entropy](softmax-and-cross-entropy.md)). Lecture 8 says the same: the learner uses supervised learning
"to classify what is the next word in a vocabulary of possible words" (≈1:07:25).

The classes must be chosen (lecture 10, slides 45–46, ≈49:41–51:16):

- **Words**, as one-hot vectors of size $K$, "e.g., K=100,000". Classification over that many classes needs a lot of data
  and "can be quite unstable".
- **Characters**, $K = 26$ for English letters. The classification is easy, "but the sequence prediction is a lot harder.
  You have to take a lot more time steps", and a model that "starts spelling something weird" has nowhere to go.
- **Byte pairs**, the sweet spot in between, used "even in the largest scale language models we have". Lecture 10 describes
  them as "2 character pairs", about 26 times 26 of them "maybe plus a few more" (≈51:16); lecture 8's tokenizer has "a
  byte pair that represents T-H" and also one "that represents I-N-G" (≈15:29–16:14). See [transformers](transformers.md).

## Training: maximum likelihood and teacher forcing

Lecture 10's molecule-to-text model (slides 47–51) encodes a caffeine molecule with a
[graph neural network](graph-neural-networks.md) and decodes "a mild stimulant that enhances cognitive ability" with an LSTM,
ending in an END token, "the idea of stopping when you're done" (≈52:03–52:51). Training maximizes the probability the model
assigns to each target word, $\arg\max_\theta \log p_\theta(y)$ (slide 49), which with one-hot targets is minimizing the
cross-entropy $\sum_i H(\mathbf{y}_ i, \hat{\mathbf{y}}_ i)$ (slide 50, ≈53:37).

With **teacher forcing** (slide 51), each prediction is conditioned on the ground-truth preceding words rather than the
model's own: "even if you predict the wrong thing at a given time step, you enforce the correct word goes in as the input",
instead of "letting it devolve and then penalizing it for every mistake it made after it made its first mistake"
(≈53:37–54:24).

## Training a transformer: causal masking

A transformer sees the whole sequence at once, so trained naively it could read the answer from its input. GPT, the
Generative Pre-trained Transformer, masks its attention so that "every output token can only attend to earlier tokens in
the sequence" (lecture 8, slides 50–51, ≈1:08:17–1:09:49). Each output then depends only on earlier words, so the whole
sequence can be supervised in one pass, "an efficient way of training such a system" (slide 52, ≈1:10:38). See
[transformers](transformers.md).

## Decoding: sampling and beam search

At test time the model samples a word from its predicted distribution, or takes the most likely one, and feeds it back in
(lecture 10, slide 52). One bad choice compounds: the example decodes "A strong stimulant …" where the truth is "mild", and
"if you make one mistake, then often, it can take you down a path in almost this tree where you make a lot of mistakes"
(≈54:24–55:10).

**Beam search** (slide 53) keeps the $k$ most likely continuations at each step, expands each, and picks the complete
sequence with the best overall score, the model's confidence in the whole sentence,
$p_\theta(\mathbf{y}_ 1, \ldots, \mathbf{y}_ T \mid \mathbf{x})$ (≈55:10–55:56). To predict a word several steps ahead, the
lecturer would still step through the words in between, "there's signal in those intermediate words", perhaps with beam
search (≈56:45–57:32).

## Next-word prediction as masking the future (lecture 11)

Lecture 11 files language models under self-supervised learning: "the way that language models work is they predict the next word in
a sequence … And most people still call language models self-supervised because they're just predicting the raw data. It happens the
raw data is semantic and is words" (≈1:10:38). Set beside BERT, which masks tokens in the middle of a text and predicts them,
autoregressive models "are just the same, except they're only masking the final word as opposed to interleaving words. And that has
some advantages in that I can decode in sequential order" (≈1:13:47). That ordering is why the lecturer thinks BERT went out of fashion:
in conversation, "time is an axis that is important and not symmetric with other axes. I have to answer the question after the question
has been asked", so masking only the future "just fits into language models"; the biggest models did that, "And because those were the
biggest models, they just worked the best". For learning a sentence embedding, "I bet the BERT method is still going to work better if
scaled the same amount" (≈1:14:33–1:15:20). See [self-supervised learning](self-supervised-learning.md).

## Autoregressive models as generative models (lecture 14)

Lecture 14 places autoregressive models in the course's taxonomy of [generative models](generative-models.md), as "the simplest
one to understand" (≈59:43). Training is supervised learning on prefixes and next elements; sampling feeds each sampled output
back in as input (slide 36). That this is a valid probability model rests on the chain rule of probability,
$p(\mathbf{X}) = \prod_{i=1}^{n} p(\mathbf{x}_ i \mid \mathbf{x}_ 1, \ldots, \mathbf{x}_ {i-1})$, so that modeling every
conditional by a next-element classifier and maximizing the likelihood of the training sequences is maximum-likelihood
density modeling (slides 37–38, ≈1:01:17–1:02:03).

**Which kind of generative model is it?** Asked whether it is a density model, an energy model or a sampler, the class voted
mostly for density model, and the lecturer agrees: it outputs "the density over the next pixel given the previous pixels", and
"the product of all the conditional probabilities is the probability of the entire image. That's a density that integrates to
1" (≈1:04:24–1:05:58).

**Pixels and audio.** The same model draws an image pixel by pixel: quantize the colours into classes, predict a categorical
distribution over the next pixel's colour from the pixels so far, sample it, write it in, and repeat (slide 39,
≈1:02:49–1:03:36). The loss on one pixel is the log probability of the ground-truth colour, the predicted log probabilities
multiplied elementwise by the one-hot label (slide 40), and summed over all the pixels of a training image it is the image's
log-likelihood (≈1:06:47). Slide 41 shows the training pairs and the sampling loop on bird images, and slide 42 cites WaveNet,
which models raw audio the same way: "it looks exactly like language modeling" (≈1:05:58).

**Kinship with diffusion.** Read backwards, an autoregressive model's training data is a "reverse autoregressive sequence" that
erases a data point one element at a time until nothing is left, just as a [diffusion model](diffusion-models.md) noises it until
it is a simple Gaussian; each model learns to reverse its sequence one simple step at a time (slides 52–53, ≈1:15:21–1:16:52).
The lecturer: "an autoregressive model is a small variation on a diffusion model" (≈3:48).

## Among the other families (lecture 15)

[Lecture 15](15-generative-models-representation-learning-meets-generative-modeling.md)'s comparison of generative-model families (slide 41) marks autoregressive models as having no latent variables,
a density, and a generator that is slow ("✓ (slow)"); the lecturer calls them "the ones that are popular in language modeling"
(≈1:20:09). See [generative models](generative-models.md#the-families-compared-lecture-15).

## Conditional by construction (lecture 16)

"Autoregressive models are already, by construction, conditional generative models", since every factor is a conditional; "you don't
have to modify them or do something new to turn them into conditional models" ([lecture 16](16-generative-models-conditional-models.md), ≈45:58–46:43). A language model is trained
by next-word prediction and used for a conditional problem by putting the input in the prefix: a review followed by "The sentiment in
this review is" predicts the sentiment (slides 52 and 53), as GPT-3 does (slide 54). The model has no explicit latent variables but is
still stochastic, "because every next word I predict will be sampled from a categorical distribution" (≈47:33), and "this is often called
prompting language models". That they are "most powerful for conditional prediction" became evident "in the GPT-3 era" (≈48:20–49:08). A
captioning model is the same decoder prefixed with image tokens (slide 56), and VQGAN's text-to-image pipeline can put an autoregressive
transformer over its latent codes (slide 70, ≈1:07:53). See [conditional generative models](conditional-generative-models.md).

## Prompting and in-context learning (lecture 18)

Lecture 18 treats GPT-3 (Brown et al., 2020) as a foundation model whose learner is generative pretraining, "trained autoregressively to
predict the next character" ("Colorless green ideas sleep ____" → "furiously", slide 48), and whose adaptor is prompt engineering
("[Review] + 'The sentiment in this review is ____'" → "positive", slide 49). **In-context learning** (slide 52) adds an inner loop to the
outer loop of "Learning via SGD during unsupervised pre-training": each prompt's examples, of sums, unscrambled words or translations, set
the task the model continues (≈1:11:28). Whether that is learning or "pattern matching", "essentially just a kernel machine" over the
training data, "depending on who you talk to" (≈1:10:41). See [foundation models and prompting](foundation-models-and-prompting.md).

Lecture 19 calls this meta-learning. A sequence $x_0, y_0, x_1, y_1, \ldots$ defines a task, and a model that predicts what comes next
learns the task from the sequence: "this really is, I think, what GPT and a lot of these modern autoregressive models are doing. It's kind
of a version of meta-learning at large scale" (slide 50, ≈1:13:58). See [meta-learning](meta-learning.md).

## Language models, from counts to LLMs (lecture 21)

[Lecture 21](21-language-models.md) restates the definition from which everything here follows: a model that "places a distribution over
possible next words" can, "via the chain rule", give "a distribution over strings", and can be sampled from to generate (≈3:06). For
decades such models were estimated "just by getting big text corpora and counting" what follows each context of a few words (≈3:52–5:25).
Slide 9 writes the seq2seq era's **conditional** language model, the translation or summarization model of 2014–2017, in this page's
factorized form,

$$\Pr(\mathbf{Y}=\mathbf{y} \mid \mathbf{X}=\mathbf{x}) = \prod_{t=1}^{T} \Pr(Y_t = y_t \mid \mathbf{X}=\mathbf{x}, \mathbf{Y}_ {\lt t}=\mathbf{y}_ {\lt t})$$

an "autoregressive conditional language model". The lecture also draws the line between this and masked prediction: BERT predicts a
hidden word from both sides and "already knows exactly how many words are going to be in a sentence", so "BERT is NOT a language model"
and cannot generate (slide 17, ≈17:47). See [language models](language-models.md).

## Decoding as search (lecture 24)

[Lecture 24](24-inference-methods-for-deep-learning.md) returns to lecture 10's molecule captioner, a graph network feeding an LSTM ("You can replace it with a transformer, but it doesn't
really matter", ≈15:32), to ask how a trained autoregressive model should be used. It is trained by maximum likelihood with **teacher
forcing**, each next word conditioned on "the ground-truth previous words" (slides 12–14, ≈17:05–17:53), but at test time it conditions on
its own samples (slide 15). That is "a deviation between how we'll use this system at test time and how it was trained", and samples
that stray from the data put the model's next predictions off the distribution it was trained on (≈17:53–19:26).

Slide 16's point is that **greedy decoding does not maximize likelihood**. The model's probability of a sequence factorizes as

$$p(\mathbf{X}) = p(\mathbf{x}_ n \mid \mathbf{x}_ 1, \ldots, \mathbf{x}_ {n-1}) \cdots p(\mathbf{x}_ 2 \mid \mathbf{x}_ 1) \thinspace p(\mathbf{x}_ 1)$$

and taking the most likely $\mathbf{x}_ 1$, then the most likely $\mathbf{x}_ 2$ given it, and so on, does not in general give
$\arg\max_{\mathbf{X}} p(\mathbf{X})$, the jointly most likely sequence: "You can construct counterexamples" (≈20:13–21:00). Random
sampling does no better, since "a random sample is not necessarily going to be the most likely sample" (≈24:52). Hence search. **Best-of-N**
samples $N$ sequences independently, scores each by the chain rule, and keeps the most likely (slide 17); **beam search** keeps "the top-k
best next words" at every step and expands them as a tree (slide 18); and **chain of thought** asks the model to do the search itself in
text (slide 19). The lecturer's verdict: "Best-of-N seems to be the simplest and also works about as well as anything, as far as I can tell
right now. And beam search is maybe the most popular" (≈30:19). The same search can maximize any score rather than likelihood; see
[search and test-time compute](test-time-compute.md).
