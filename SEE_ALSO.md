# See also — related knowledge bases

Other Cairn knowledge bases whose material genuinely bears on 6.7960. Each entry says what that
KB is good for and what it does not cover, so you can decide *before* spending a fetch on it.

Pass the **repo URL** as the `kb` argument to `kb_read` / `kb_list`. The
`raw.githubusercontent.com` links are direct `web_fetch` targets that return plain markdown.

---

- **CS224N — Natural Language Processing with Deep Learning** (Stanford, Christopher Manning,
  Spring 2024). Complete: all 23 lectures.
  KB: `https://github.com/chaimantec/cairn-kb-cs224n` — pass this as `kb`.

  The closest match for 6.7960 lecture 1's **background review**. Its lecture 3 derives
  gradients by hand and then algorithmically (backpropagation), and it has concept pages on
  [activation functions](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/wiki/activation-functions.md),
  [gradient descent](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/wiki/gradient-descent.md),
  [softmax and cross-entropy](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/wiki/softmax-and-cross-entropy.md),
  [backpropagation](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/wiki/backpropagation.md)
  and [vanishing and exploding gradients](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/wiki/vanishing-and-exploding-gradients.md).
  Reach for it when a question goes past what 6.7960 lecture 1 says — especially **how
  backpropagation actually computes the gradients**, which lecture 1 defers to lecture 2 and this
  KB does not yet cover. Its examples come from NLP rather than vision, and its notation differs
  (e.g. $h = f(Wx + b)$ for a layer).
  Start at [INDEX](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/INDEX.md).

- **CS336 — Language Modeling from Scratch** (Stanford, Percy Liang and Tatsunori Hashimoto,
  Spring 2026). Complete: all 18 lectures.
  KB: `https://github.com/chaimantec/cairn-kb-cs336` — pass this as `kb`.

  The systems-and-scale side. Lecture 1 here previews **scaling** (slides 78–79: scaling rules
  for optimization, scaling laws), recorded as 6.7960 lectures 7 and 20, which this KB does not yet
  cover. CS336 has two lectures on scaling laws (its lectures 9 and 11), a page on
  [learning-rate scaling and muP](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs336/main/wiki/learning-rate-scaling-and-mup.md),
  and lectures on GPUs and parallelism — useful for "why do GPUs matter" beyond lecture 1's
  one-paragraph answer. It is about language models specifically, and measures and builds rather
  than proves.
  Start at [INDEX](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs336/main/INDEX.md).
