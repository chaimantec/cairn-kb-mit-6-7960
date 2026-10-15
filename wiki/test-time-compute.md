# Search and test-time compute

What a trained model can do with more computation on each new query: search through the outputs it can produce, steer that search with a
score or a verifier, change its own weights on the test data, and feed what the search finds back into training. In the course's
words, almost everything else "has been about the training side", and at test time "it's almost entirely been just a forward pass through
the trained model. … But it turns out you can do much more than that at test time" ([lecture 24](24-inference-methods-for-deep-learning.md),
≈3:03–3:48). Covered so far: [lecture 24](24-inference-methods-for-deep-learning.md), slides 2–55, ≈1:33–1:22:19, the lecture on this topic;
with beam search in [lecture 10](10-architectures-memory.md), CLIP-guided search of a GAN's input in [lecture
2](02-how-to-train-a-neural-net.md) and [lecture 18](18-transfer-learning-models.md), and chain-of-thought prompting and RLHF in [lecture
21](21-language-models.md).

**Notation.** $\theta$ is a model's parameters and $p_\theta$ the distribution it defines; $\mathbf{x}$ is an input or query,
$\mathbf{y}$ an output sequence, and $\mathbf{z}$ a generator's latent variables, as in the [course notation](notation.md).

## Training and inference

In machine learning, **inference** means "making predictions about new datapoints using a trained model"; statistics calls that
"prediction", and uses "inference" for what machine learning calls training (lecture 24, slide 2). Lecture 24's map (its slide 3) puts the
methods on a line from training to inference. **Pre-training** learns a model from data; **post-training** updates it, before deployment
(fine-tuning, RLHF) or during it (prompting, in-context learning, test-time training, continual learning, feedback control); and
**search**, "Given a model and a query, find the best answer to the query", changes nothing in the model (Best-of-N, beam search, MCTS,
chain of thought). Training is also called **amortized inference**: it "learns a set of weights that directly solves the problem without
having to go through the step-by-step reasoning process" (≈4:35–5:20).

## Why spend compute at test time

Rich Sutton's "The Bitter Lesson" names two general methods "that seem to scale arbitrarily" with computation: "search and learning"
(lecture 24, slide 4; the essay also opens [lecture 20](20-scaling-laws.md)). Deep learning has mostly scaled learning. Lecture 24's slide 5
shows OpenAI's o1 improving with **test-time compute** along a power law "even steeper than the train-time power law" (≈10:00–10:46), and its
slide 9, Jones's board-game scaling laws, shows the two trading off: "I can do more search, and then I can get the same elo for less
training, or I can do less search, and then I can get the same elo for more training" (≈13:56). Chess fell to search alone; Go's larger
branching factor needed "search with heuristics — with learned heuristics on top" (≈12:21–13:07). See [scaling laws](scaling-laws.md).

## Searching a model's outputs

An autoregressive model trained by maximum likelihood with teacher forcing is used differently at test time, where it conditions on its own
samples, and **greedy decoding does not maximize likelihood**: picking the most likely token at each step need not give the jointly most
likely sequence (lecture 24, slides 12–16). Three searches do better (see [autoregressive models](autoregressive-models.md)):

- **Best-of-N**: "Sample N sequences iid and pick the sequence with highest likelihood", scoring each with the chain rule (slide 17). "To
  my knowledge, this Best-of-N sampling is about as good as anything right now. … It's rejection sampling, essentially" (≈28:47).
- **Beam search**: keep "the top-k best next words" at every step and expand them as a tree, then take the most probable path (slide 18;
  beam search for sequence models also appears in [lecture 10](10-architectures-memory.md)).
- **Chain of thought**: "an emergent or learned search", in which "it's the LLM doing it itself and writing the intermediate steps into
  the text space" (slide 19, ≈30:19–31:51; chain-of-thought prompting is lecture 21's, see [foundation models and
  prompting](foundation-models-and-prompting.md)).

The same question arises for other generative models. Diffusion models sample step by step and have the same issue; a VAE samples in one
step and does not (≈22:31–23:19; see [diffusion models](diffusion-models.md) and [variational autoencoders](variational-autoencoders.md)).

## What to search for: scores and verifiers

The search need not maximize likelihood: "you can do the same exact thing, but not for likelihood, but for any kind of scoring function"
(≈32:38). The scorer "can be absolutely anything … something that takes in a sentence and gives you a scalar out": another model, a human,
or "a formal proof checker" (≈34:14). Rating samples by people and keeping the best "is the basis of what's called reinforcement learning
from human feedback" (≈33:25; RLHF itself is in [language models](language-models.md)). A **verifier** checks correctness by a formal
process, a compiler or a proof checker, and is worth using because "verification is easier than generation" (lecture 24, slide 22), as
checking a proof is easier than finding one, or multiplying factors easier than factoring (≈35:00–36:34).

## Searching a generator's latent space

For an image generator $g(\mathbf{z})$, the search runs over the latent $\mathbf{z}$. With CLIP's text and image encoders as the score,
optimizing $\mathbf{z}$ to maximize the embeddings' dot product is CLIP + VQGAN (lecture 24, slide 26), the same idea as lecture 2's and
lecture 18's CLIP+GAN; it is "test-time compute to solve a problem that in 2020 you could not solve at train time" (≈42:43). With a learned
model of human perception as the score, GANalyze searches for more memorable or aesthetic images by gradient descent, since there
"everything is differentiable" (slides 29–31, ≈45:49). Or one can "just ask": a model prompted for "a very, very … beautiful painting"
gives one, but such learned abilities carry no guarantees (slides 32–40, ≈51:14–54:21). See [generative models as
data](generative-models-as-data.md).

## Changing the model at test time

**In-context learning** is "learning to do supervised learning" from examples in the prompt (lecture 24, slide 42, ≈56:42–57:30), and
von Oswald et al. show a toy case in which it equals a gradient-descent step on the context examples (slide 43). Doing that gradient
descent explicitly is **test-time training**: "Update the weights of your model as a function of the test query" (slide 45). It can use the
query's own examples, which on the ARC benchmark gave 61.9% with "an 'old' method" (slide 48), or, with no labels at all, a self-supervised
loss such as masked autoencoding on the test image (Sun et al., slide 46). Gradient descent "is guaranteed to improve performance on the
in-context examples" and "can leverage more test-time compute than ICL", though "ICL could be a smarter learning algorithm" (slide 44). See
[meta-learning](meta-learning.md) and [self-supervised learning](self-supervised-learning.md).

## Search and learning in a loop

"You can use search to improve learning and use learning to improve search. And this is basically the fundamental concept in
reinforcement learning" (lecture 24, ≈1:15:15):

- **Search improves learning**: answers that pass a verifier become supervised training pairs, giving a better model (slides 50–51).
  With an external verifier, "the information came in through the verifier"; with a learned one, "it's purely self-improvement" (≈1:15:15).
- **Learning improves search**: search for winning action sequences, train a policy on them, and guide the next search with the policy
  (slide 52); AlphaGo backs up the results of tree search into a learned heuristic (slide 53, ≈1:17:39–1:18:24).
- **STaR** fine-tunes a language model on its own rationales that reached correct answers (slide 54), and the lecturer's guess at **o1**,
  after Sasha Rush, is the same loop on chains of thought, with as much search as one likes at deployment (slide 55): "It's just
  reinforcement learning applied to reasoning with a language model" (≈1:20:43).

Lecture 21's context distillation, fine-tuning a model on what it produces with a good prompt, is a related idea on [knowledge
distillation](knowledge-distillation.md).

## See also

- [Lecture 24 — Inference Methods for Deep Learning](24-inference-methods-for-deep-learning.md) — the lecture, with its figures.
- [Autoregressive models](autoregressive-models.md) — the chain rule, teacher forcing, sampling and decoding.
- [Language models](language-models.md) — RLHF, and the training stages that search builds on.
- [Foundation models and prompting](foundation-models-and-prompting.md) — prompting, chain of thought and "just ask".
- [Meta-learning](meta-learning.md) — in-context learning and whether it is learning.
- [Scaling laws](scaling-laws.md) — train-time scaling, beside test-time scaling.
