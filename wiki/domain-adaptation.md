# Domain adaptation

**Domain adaptation** adapts a model trained on a **source domain**, where there are labels, to a **target domain**,
where there are few labels or none, by learning to **align the representations** of the two. Lecture 17 introduces it as one
answer to distribution shift: "an entire category of models" that "learn how to align their representations during training
so that you can enforce generalization or adaptation to a new target domain" (slide 57, ≈56:06–56:55). Lecture 18 teaches it as
a way of transferring knowledge about the mapping, beside finetuning. Covered so far: [lecture
17](17-generalization-out-of-distribution.md), slides 57–61, ≈56:06–56:55 and ≈1:02:25–1:03:59; [lecture
18](18-transfer-learning-models.md), slides 19–28, ≈29:24–40:20.

**Notation** (lecture 18's slide 20). An input is $\mathbf{x}$; a feature extractor $G_f(\cdot; \theta_f)$ with parameters
$\theta_f$ maps it to features $\mathbf{f}$; a label predictor $G_y(\cdot; \theta_y)$ outputs a class label $y$ with loss
$L_y$; a domain classifier $G_d(\cdot; \theta_d)$ outputs a domain label $d$ with loss $L_d$; $\lambda$ weights the domain
term.

## Aligning two domains

Lecture 18 pictures the goal as two circles of shapes, solid for the source domain and hollow for the target, which domain
adaptation merges so that each kind of shape from either domain sits together (slide 19; lecture 17's slide 57 draws
the same kind of picture). The aim is to do well on the target "but … also … make sure that the representation spaces of the source and target
domain are well aligned", and the hope is that aligned feature spaces let the mapping be learned "with much less data"
(≈29:24–30:10). One way is to learn a projection from the target domain into the source domain, "where you had a lot of data",
and reuse the source model (≈30:10–30:57).

## Domain adversarial training

"One kind of large set of them takes an adversarial approach" (lecture 18, ≈30:57). **Domain Adversarial Neural Networks**
(DANN, slide 20) put two heads on one feature extractor: a label predictor, trained to predict the category, and a domain
classifier, trained to predict which domain the input came from. A **gradient reversal layer** sits between the features and
the domain classifier. On the backward pass the domain classifier receives $\lambda \frac{\partial L_d}{\partial \theta_d}$ as
usual, but the feature extractor receives $-\lambda \frac{\partial L_d}{\partial \theta_f}$, the domain gradient with its sign
flipped, alongside $\frac{\partial L_y}{\partial \theta_f}$ from the label loss. "They flip the gradient from the domain
predictor. So you're essentially trying to train the model to be as good as possible at predicting the category, while being as
bad as possible at predicting the domain, which tries to force, essentially, the representations across the two domains to
align" (≈31:44). The lecturer's example has product photographs from two sites, clean ones from Amazon and ones taken "on their
messy desk" for eBay (≈30:57–31:44). The trick needs only domain labels on the target, so it still works with few target
category labels or none (≈31:44–32:30). It is a cousin of [adversarial examples](adversarial-examples.md): the adversary here is
a classifier to be fooled, and the features are trained to fool it.

## Unsupervised domain adaptation

In **unsupervised domain adaptation** (UDA, slide 21) the target domain has data "but no labels" (≈32:30). Lecture 18's
example is the lecturer's own: counting fish in sonar video of rivers, with labels for some rivers, for a model that has to work
on any river and keep working when the sensor moves (≈33:16). Slides 22 to 26 assemble the standard recipe from components
(≈34:02–36:25):

1. **Pseudo-labels**: run the target data through the source model and treat predictions above a threshold as ground truth
   (slide 22). On its own "it doesn't usually work very well."
2. **A student** trained on those predictions as weak supervision (slide 23), which can learn "something useful that is better
   than just taking those raw predictions", with small gains; but "this threshold is really, really finicky and hard to
   calibrate."
3. **An EMA teacher**: the predicting model becomes a teacher whose weights are an exponential moving average of the student's,
   so it updates more slowly (slide 24).
4. **Source supervision**: the student also trains on the labelled source data (slide 25).
5. **Softmax, distillation and alignment**: soft predictions instead of a threshold, a [distillation](knowledge-distillation.md)
   loss that matches "the distribution over the outputs", and explicit alignment of source and target features (slide 26).

"Basically every modern domain adaptation architecture is some combination of these components" (≈36:25). **Align and Distill**
(ALDI; slides 27 and 28, Kay et al.) writes them as one framework, with a supervised loss $L_{sup}$, an alignment loss
$L_{align}$, a distillation loss $L_{distill}$ and an EMA teacher, and redraws earlier methods (SADA, PT, AT, UMT and MIC) as
particular choices within it. Unifying how the experiments are run, "you can often be more optimal than any previous approach".
Such methods "are almost never as effective as having labeled data", but built carefully "you can actually start to get pretty
close", because the model learns "something about the statistics of the target domain" (≈36:25–37:12).

## The catch: choosing a model without labels

Asked how anyone knows a model adapted to an unlabelled domain works, the lecturer called it "the elephant in the room for most of
the domain adaptation literature": results are reported "on labeled validation data for the target domain", and "if you have to
label data to do model selection …, why don't you just use that labeled data for training?" (lecture 18, ≈37:12–37:59). Where it
makes sense is where data cannot easily reach a human labeller, such as sensors in the remote Arctic, on the moon or in the deep sea
(≈38:47–39:33). Without quality control a self-training model can "veer off in some catastrophic way", getting "worse and worse"
(≈39:33). The starting model matters a great deal, so domain adaptation complements building a good general representation: "a
mechanism to fine-tune if you have a lot of data that doesn't have labels, instead of maybe a little bit of data that does"
(≈40:20).

## Robust to any shift, or adapted to one

Lecture 17 draws the line between its own subject and this one: "the difference between trying to be robust to distribution
shifts versus trying to adapt to a specific distribution shift". When there is a specific target dataset, human-in-the-loop
learning, domain adaptation or supervised finetuning "can be much more efficient" (≈1:02:25–1:03:59). And its last remedy for
shift outranks them all: "Get training data that is representative of your test domain (works better than any algorithm)"
(slide 61). See [out-of-distribution generalization](out-of-distribution-generalization.md).

## See also

- [Lecture 18](18-transfer-learning-models.md) — the lecture, with its figures.
- [Transfer learning](transfer-learning.md) — finetuning, the other way to transfer the mapping.
- [Knowledge distillation](knowledge-distillation.md) — the teacher–student machinery the UDA recipe is built from.
- [Out-of-distribution generalization](out-of-distribution-generalization.md) — the distribution shift domain adaptation
  answers.
- [Adversarial examples](adversarial-examples.md) — adversarial optimization over inputs, where DANN's is over features.
