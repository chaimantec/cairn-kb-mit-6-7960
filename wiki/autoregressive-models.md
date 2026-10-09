# Autoregressive models

An **autoregressive model** generates a sequence one element at a time: given the beginning of a sequence it
predicts what comes next, the prediction is appended to the input, and the model is asked again. Trained as a
classifier of the next element, it is how the course describes modern language models: "This is how ChatGPT
works" ([lecture 8](08-architectures-transformers.md), ≈1:06:38). Covered so far: lecture 8, slides 48–52,
≈1:05:51–1:10:38 (autoregression, GPT and causal masking); [lecture 9](09-hackers-guide-to-deep-learning.md), slide 21,
≈42:35–44:51 (longer prompts make the next word easier to predict); [lecture 10](10-architectures-memory.md), slides
40–53, ≈41:53–58:19 (the probability model, next-word classification, the choice of vocabulary, a molecule-to-text
model, teacher forcing, sampling and beam search); [lecture 11](11-representation-learning-reconstruction-based.md), ≈1:10:38 and ≈1:13:47–1:15:20
(next-word prediction as self-supervised learning, set beside BERT's masking). Generative models are the subject of lectures 14–16 and language
models of lecture 21 (see the [course map](course-map.md)).

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
