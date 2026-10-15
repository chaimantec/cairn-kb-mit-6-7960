# Language models

A **language model** "takes in … a sequence of words or tokens … as input and places a distribution over possible next words as
output"; "via the chain rule, you can turn this into a distribution over strings, you can sample from it to generate language, and you
can do all kinds of other things" ([lecture 21](21-language-models.md), ≈3:06). Equivalently, it "estimates the probability of any
sequence of words" (lecture 21, slide 16). This page follows that object through the course: how it is built, how it became the
large language models (LLMs) of today, how such a model is trained in three stages, and where each piece is taught. Covered so far:
[lecture 21](21-language-models.md), slides 1–91, ≈0:00–1:17:00, the guest lecture by Jacob Andreas that is about language models
throughout; next-word prediction and GPT's causal masking in [lecture 8](08-architectures-transformers.md), slides 48–52; RNN language
models, teacher forcing, sampling and beam search in [lecture 10](10-architectures-memory.md), slides 40–53; next-word prediction set
beside BERT's masking in [lecture 11](11-representation-learning-reconstruction-based.md), ≈1:10:38 and ≈1:13:47–1:15:20; GPT-3 and prompt
engineering in [lecture 18](18-transfer-learning-models.md), slides 48–49; and the scaling laws of language models in [lecture
20](20-scaling-laws.md); and search over a language model's outputs, and training on what the search finds (STaR, and a
guess at o1), in [lecture 24](24-inference-methods-for-deep-learning.md), slides 11–22 and 49–55.

**Notation.** As on lecture 21's slides: $P(\cdot \mid \ldots)$ is the model's distribution over the next token given the context after
the bar; $P_{\theta}$ is a model with parameters $\theta$; $\mathbf{x}$ is the text so far and $\mathbf{y}$ an output.

## Predicting the next word

The mechanics are in [autoregressive models](autoregressive-models.md): factor the probability of a sequence by the chain rule into
next-token predictions, each of which is a classification over the vocabulary, train by maximum likelihood with teacher forcing, and
generate by sampling or beam search (lectures 8 and 10). Lecture 21 starts further back. Predicting the next word dates to decoding
Enigma messages after the Second World War, and for decades the models were estimated "just by getting big text corpora and counting":
unigram counts, then counts of what follows "once upon", then of "every sequence of four words", which "is basically all that the field
did from, say, the '60s to the early 2000s" (≈3:52–5:25).

Lecture 21's slides 4–5 measure the progress since with perplexity, "a measure of the quality of a language model's predictions" (lower
is better, ≈6:13): about 300 for a unigram model, 141.2 for a 5-gram, 126.0 for a neural 5-gram (2000), 78.4 for an LSTM (2014), 54.5
for a transformer (2017), 35.8 for GPT-2 (2019) and 20.5 for GPT-3 (2020). Each step on the slides' path is labelled with what it fixed:
sparsity (feed-forward models), longer context (RNNs, see [recurrent neural networks](recurrent-neural-networks.md)), vanishing
gradients (LSTMs), a fixed-dimension bottleneck and parallel computation ([transformers](transformers.md)), and then scaling. Scale alone
was not the difference: a 2007 paper trained n-gram models on "2 trillion tokens" with "300 billion n-grams" (slide 6), and they "were
typically not nearly as good as the neural models that we see today", which the lecturer credits to "the inductive bias that you get from
neural sequence models" (≈8:30–9:16; slide 7: "Number of parameters is a bad proxy for model flexibility"). See [inductive
bias](inductive-bias.md).

## What language models were for

Lecture 21's slides 8–11 divide the history into four eras. In **pipelined NLP** (1970–2014) the language model was the prior
$\Pr(\mathbf{Y}=\mathbf{y})$ of a noisy-channel model, reranking a speech recognizer's or translator's candidates. In **seq2seq learning**
(2014–2017) the task itself became conditional next-word prediction. In **pretrain + transfer** (2017–2020), a language-model-like
objective learned representations that were then fine-tuned. In **pretrain + prompt** (2020–), "a single language model for everything"
is used without changing its parameters (≈10:03–13:06). See [transfer learning](transfer-learning.md).

The pretrain-and-transfer era ran on **masked** language modeling, BERT's objective: hide words and predict them from both sides
([self-supervised learning](self-supervised-learning.md); lecture 11). Lecture 21 is explicit that "BERT is NOT a language model"
(slide 17): its predictions use the words on both sides, it "already knows exactly how many words are going to be in a sentence", and
so it cannot generate text (≈17:47). "Nowadays, 'large language model' ≈ large Transformer pre-trained with a self-supervised objective
(next-word prediction)" (slide 17).

## Three stages of training

Lecture 21 describes a modern LLM as the product of three stages.

1. **Pretraining** on next-word prediction over as much text as possible. At a large enough scale this alone produces abilities nobody
   trained for, "emergent abilities" such as arithmetic, transliteration and question answering in Persian (slide 20, Wei et al., 2022),
   and the ability to learn a task from examples in the prompt (≈20:56–22:30). See [scaling laws](scaling-laws.md).
2. **Instruction tuning**: fine-tuning on instruction-following data, either existing NLP datasets turned into instructions with templates
   ("task-based") or newly collected human demonstrations ("demonstration-based"), by maximizing
   $\log P_{\theta}(\text{response} \mid \text{instruction})$ (slides 71 and 74). Without it a pretrained model given "Answer the following
   question. …" may write more questions, because that is a plausible continuation of such a document (slides 68–69, ≈53:47). Its benefit
   too appears only at scale (slide 72, ≈1:00:45).
3. **Reinforcement learning from human feedback** (RLHF). Supervised targets cannot teach a model to say "I don't know", to refuse some
   requests, or to answer truthfully when the internet usually does not, because annotators cannot see what the model knows (slides
   75–82, ≈1:02:17–1:08:30). Instead, sample outputs from the model, have people judge them, train a **reward model**
   $r_{\phi}(\mathbf{x}, \mathbf{y})$ to reproduce the judgments ("recognition problems generally are easier than generation problems",
   ≈1:13:05), and optimize the model, now called a **policy** $P_{\theta}(\mathbf{y} \mid \mathbf{x})$, to maximize the expected reward,
   by policy gradient ("normal maximum likelihood estimation where each sample is weighted by the score assigned to it by the reward
   model", ≈1:13:51). Methods such as DPO skip the reward model and "directly" raise the probability of the preferred outputs (≈1:14:38).

The objective of the third stage, as slide 90 writes it:

$$\arg\max_{\theta} \mathbb{E}_ {y \sim P_{\theta}(y \mid x)} \left[ r_{\phi}(x, y) \right]$$

The lecture's view of the result: large-scale pretraining "and then a little bit of refinement after the fact", which changes "mostly
just the format in which they produce their responses by default", gives "really flexible, really general purpose predictors" (≈1:15:27–1:16:12).

## Using a language model without training it

Once a model is trained, "the only mechanism by which we can exert control over" it "is by changing what we put into it" (≈23:17):
in-context learning, zero-shot instructions, and chain-of-thought prompting, including "Let's think step by step". These are on
[foundation models and prompting](foundation-models-and-prompting.md), with lecture 18's GPT-3 and CLIP. Whether in-context learning is
learning at all, or the retrieval of a task the model already knows, is on [meta-learning](meta-learning.md). Fine-tuning a model on the
outputs it produces with a good prompt, **context distillation**, is on [knowledge distillation](knowledge-distillation.md), and why a
model's own synthetic data cannot teach it "anything new" without an outside filter is on [generative models as
data](generative-models-as-data.md).

## Searching, and learning from the search (lecture 24)

[Lecture 24](24-inference-methods-for-deep-learning.md) is about what can be done with a trained language model beyond sampling it once. Greedy decoding does not find the most likely
sequence, so one searches: Best-of-N, beam search, and chain of thought as a search the model does itself (slides 16–19). The search can
be steered by any score instead of likelihood, including human ratings, "the basis of what's called reinforcement learning from human
feedback" (≈33:25), and a **verifier** such as a compiler or proof checker, because "verification is easier than generation" (slide 22).
Then the answers found by search can become training data: **STaR** fine-tunes a model on its own rationales that reached a correct answer
(slide 54), and the lecturer's guess at OpenAI's **o1**, after Sasha Rush, is the same loop on chains of thought, with search on top at
deployment (slide 55): "It's just reinforcement learning applied to reasoning with a language model" (≈1:20:43). See
[search and test-time compute](test-time-compute.md).

## See also

- [Lecture 21 — Language Models](21-language-models.md) — the lecture this page summarizes.
- [Autoregressive models](autoregressive-models.md) — the chain rule, teacher forcing, sampling and beam search.
- [Transformers](transformers.md) — the architecture of today's language models, and causal attention.
- [Recurrent neural networks](recurrent-neural-networks.md) — the language models before transformers.
- [Self-supervised learning](self-supervised-learning.md) — next-word and masked-word prediction as pretext tasks.
- [Foundation models and prompting](foundation-models-and-prompting.md) — in-context learning and chain of thought.
- [Scaling laws](scaling-laws.md) — how language models improve with data, parameters and compute, and emergent abilities.
- [Softmax and cross-entropy](softmax-and-cross-entropy.md) — the classifier behind each next-token prediction.
