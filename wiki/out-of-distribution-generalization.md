# Out-of-distribution generalization

A model **generalizes out of distribution** when it works on data drawn from a different distribution from the one it was
trained on. Theory and standard practice assume the opposite, that training and test data are independent samples of one
distribution (IID, "training distribution = test distribution"), and lecture 17's point is that deployment never meets
that assumption: "This is never true in the real world" (slide 4, ≈6:14). It calls the ability to generalize beyond the
training distribution "one of the biggest open challenges for modern machine learning models" (≈0:00), on which "I can
talk about progress, but we don't have solutions" (≈14:00). Lecture 1 assigned the topic its own lecture beside
generalization theory (see [generalization and double descent](generalization-and-double-descent.md)). Covered so far:
[lecture 17](17-generalization-out-of-distribution.md), slides 1–8 and 26–62, ≈0:00–14:00 and ≈30:27–1:03:59.

**Notation.** An input is $\mathbf{x}$ and its label $y$; $f_{\theta}$ is a network with parameters $\theta$ and
$\text{Loss}$ its training loss. The training set $\lbrace (\mathbf{x}^{(i)}, y^{(i)}) \rbrace_{i=1}^{n}$ defines an
empirical distribution $\mathbb{P}_ n$ (written $\hat{\mathbb{P}}_ n$ on some slides), $\mathbb{P}$ is the true
distribution, $\mathbb{Q}$ any other distribution over pairs, and $D$ a distance between distributions.

## Why it matters

Deployed models mostly work, the lecture argues, because a human checks them: they are "high precision human verification
systems", like a chatbot whose output you edit, or a medical decision-support tool whose advice a doctor is liable for
(≈3:09–4:43). Trusting a model without that check is where trouble starts, because the world the model meets differs from
the one it was trained on in ways no one can list in advance. The training and test data can differ by **perturbed data**,
a **different label distribution**, or shifts in "sequence/graph size, weather, country/city, source of measurement"
(slide 5, ≈7:48–8:34); a dataset built for the shift you expected does not cover the one you did not (slide 6, ≈8:34–10:07).
Evaluation has the same problem. A test set that shares "strong, spatial, temporal, or contextual correlations" with the
training set gives "very falsely high expectations" (≈12:26–13:13). The lecture calls the failure "data poisoning" (≈11:38);
outside the course material, it is usually called data leakage.

## Shortcuts

Why models fail out of distribution is the subject of the first half of lecture 17, through
[adversarial examples](adversarial-examples.md). Models use whatever predicts the label, including **non-robust features**
"correlated with label, but can be flipped via perturbation" (slide 27). **Shortcuts** are "features correlated with label in
the training data, but not under realistic distribution shifts", and "models will use them and not generalize if features
are no longer correlated" (slide 30, ≈35:51–36:36). They come from the data: bow ties go with cats, ImageNet's "fish" are
fish held by people (slide 28), datasets are scraped and crowd-labelled rather than curated by experts (slide 29), and
imbalance teaches that raccoons appear at night, so daytime raccoons are misclassified (≈36:36–37:22). Examples include
X-ray models that recognize the hospital rather than the disease, and a skin-lesion model that calls a tumour malignant when
a ruler is in the picture: "not all predictive patterns are desirable" (slide 33, ≈38:58). Slide 34, from Geirhos et al.
(2020), sorts such failures into changes that keep the category for humans but not for networks, and the reverse. And slide 35
shows transformers that failed to generalize past the sequence lengths they were trained on (see
[transformers](transformers.md)). Because shortcuts live in the data, "adversarial examples transfer across models trained
on the same dataset" (slide 31).

## Distribution shift, measured

When the whole test distribution moves, accuracy falls, and the **WILDS** benchmark measures how far on realistic shifts:
hospitals, camera locations, countries, time, users, demographics and molecular scaffolds (lecture 17, slide 42,
≈44:20–45:54). Accuracy fell from 93.2% to 70.3% across hospitals, and macro F1 on wildlife cameras from 47.0% on new
images from the training cameras to 31.0% on held-out cameras (slides 43–44, ≈45:54–46:41). The lecturer contrasts these with
the artificial shifts the literature had used, such as rotated or recoloured MNIST and the background-shifted Waterbirds
(≈44:20–45:09).

A shift has two parts that often come together. **Visual shift** changes what the same category looks like (background,
camera angle, infrared at night). **Subpopulation shift** changes how often each category appears: every camera trap "has
its own unique distribution over the categories of interest" (slide 45, ≈46:41–47:26), and across North American cities
the accuracy of a tree classifier trained on one city and tested on another tracks how similar the two cities' genus
distributions are (slide 47, ≈48:17–49:03). Since models "are trained on likelihoods", they carry a prior from the training
mix, and if the test mix is known a "really simple Bayesian remapping of the predicted probabilities" can adjust for it
without retraining (≈49:03).

## What to do about it

Lecture 17's answers come in the order of slides 48–61.

**Distributionally robust optimization (DRO)** perturbs the whole training distribution rather than each point, and minimizes
the expected loss under the worst distribution within distance $\epsilon$ of the empirical one (slides 49–50, ≈49:03–50:37):

$$\min_{\theta} \max_{\mathbb{Q}, \thinspace D(\mathbb{Q}, \mathbb{P}_ n) \lt \epsilon} \mathbb{E}_ {(\mathbf{x}, y) \sim \mathbb{Q}} \left[ \text{Loss}(f_{\theta}(\mathbf{x}), y) \right]$$

If the true distribution satisfies $D(\mathbb{P}, \hat{\mathbb{P}}_ n) \lt \epsilon$, good performance is guaranteed on it (slide
51). The difficulty is the distance between distributions ($\chi^2$, Wasserstein, maximum mean discrepancy), which for large image
datasets is "pretty tricky", so it is "not definitely something that works super well in practice" (≈50:37–51:24). It works for
**class imbalance**: with 80% and 20% subgroups, "50% error on minority group makes only 10% average error", and DRO's worst case
re-weights toward the minority (slides 52–53, ≈52:55–54:33), which is why it is used for fairness.

**Learn, or use, a prior for subpopulation shift.** Combine an image classifier with a separately learned spatiotemporal prior,
$P(y \mid I, \mathbf{x}) \propto P(y \mid I) \thinspace P(y \mid \mathbf{x})$, where $I$ is the image and $\mathbf{x}$ its
longitude, latitude and day (slide 55, Mac Aodha et al. 2019; ≈55:21–56:06).

**Domain adaptation** aligns the representations of a labelled source domain and an unlabelled target domain during training
(slide 57, ≈56:06–56:55); it is lecture 18's subject (see [transfer learning](transfer-learning.md)).

**Diagnose failures**: break the evaluation down by category and condition, or generate counterfactual images, as in the
classifier that calls an empty plate a "tray" (slide 59, ≈56:55–58:27).

**Get training data that is representative of your test domain**, which "works better than any algorithm" (slide 61): "if you
really need something to work well, you have to invest in the right kind of training data" (≈58:27–59:16).

The lecture ends on the distinction between being robust to any shift one can imagine and adapting to a known one. The
second, with a target dataset in hand, can be done far more efficiently by fine-tuning, domain adaptation or human feedback
(≈1:02:25–1:03:59).

## See also

- [Lecture 17](17-generalization-out-of-distribution.md) — the lecture, with the examples and the student questions.
- [Adversarial examples](adversarial-examples.md) — robustness at the scale of single inputs, and what it reveals.
- [Generalization and double descent](generalization-and-double-descent.md) — generalization within the training
  distribution.
- [Data augmentation](data-augmentation.md) — changing the training data to cover the variation expected at test time.
- [Transfer learning](transfer-learning.md) — adapting a trained model to a new domain.
