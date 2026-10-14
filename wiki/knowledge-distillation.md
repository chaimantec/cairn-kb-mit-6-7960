# Knowledge distillation

**Knowledge distillation** trains one network, the **student**, to reproduce the outputs of another, the **teacher**,
instead of, or as well as, the ground-truth labels. The teacher is usually a big pretrained model held fixed, and the
student is usually smaller. In the course it is the main way of transferring "knowledge about the outputs" of a model,
the second of three kinds of knowledge lecture 18 sets out to transfer. Its central idea is that a teacher's full
distribution over classes carries more information than a one-hot label: it says not just "this is a dog" but "this is a dog
that looks a bit like a cat and very different from a bear". Covered so far: [lecture 18](18-transfer-learning-models.md),
slides 22–26 and 29–39, ≈34:02–36:25 and ≈41:06–56:35; [lecture 21](21-language-models.md), ≈59:59–1:00:45 (context
distillation).

**Notation** (lecture 18's slides). An input is $\mathbf{x}$. A classifier $f$ outputs $\hat{\mathbf{y}}$, a vector of scores
over classes; $\mathbf{y}$ is the one-hot label. In distillation $f^T$ is the teacher and $f^S$ the student, with outputs
$\mathbf{y}^T$ and $\mathbf{y}^S$; $i$ indexes the classes.

## A label is already a teacher

Lecture 18 starts from ordinary training with cross-entropy (slide 30, "Recall: training a classifier with cross-entropy"):

$$L(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_i \mathbf{y}_ i \log \hat{\mathbf{y}}_ i$$

The loss pulls the predicted distribution toward the label's, "which is just 1 for cat and 0 for everything else", and the
slide already labels the two ends "Teacher" and "Student", in quotation marks: "you could argue that this is also kind of a
teacher and a student" (≈41:06–41:51). See [softmax and cross-entropy](softmax-and-cross-entropy.md).

## Matching a teacher's distribution

Distillation (slide 31, Hinton, Vinyals and Dean, 2015) swaps the label for a pretrained teacher, "a locked model", and
trains the student to make "the distance between the teacher model's predictions and the student's model's predictions …
as close as possible" (≈42:37):

$$L(\mathbf{y}^S, \mathbf{y}^T) = - \sum_i \mathbf{y}^T_ i \log \mathbf{y}^S_ i$$

It is the same cross-entropy, "but instead of cross entropy being to that binary category, it's just between the teacher and
the student outputs" (≈45:44). In slide 31's example the teacher puts high probability on both "cat" and "tiger", and the
student is trained toward that whole distribution.

## Why it helps

Two reasons came from the class (≈43:26–44:57). The practical one: the student can be "smaller, can be more computationally
efficient, and can learn the same thing". The one the lecturer was after: the teacher's spread is **information about
similarity**, "not just the similarity of the category set, it's the similarity of what's recognizable in a given input". An
orange cat in tall grass looks like the tigers the teacher has seen, "so now, what you're teaching the model is not just, this
is a cat, … this is a cat that looks a little bit like a tiger". Slide 32 calls the teacher's output a **soft target**: "The
'label' you get for each input is more informative than just the ground truth class".

The original paper's evidence, on speech recognition (slide 32's table):

| System & training set | Train Frame Accuracy | Test Frame Accuracy |
| --- | --- | --- |
| Baseline (100% of training set) | 63.4% | 58.9% |
| Baseline (3% of training set) | 67.3% | 44.5% |
| Soft Targets (3% of training set) | 65.4% | 57.0% |

With soft targets, 3% of the data gets nearly the test accuracy of all of it, with lower training accuracy than the 3%
baseline, which overfits. Soft targets of high entropy "are giving a lot more information per training case than hard
targets", with "a lot less variance in the gradient between training cases", so a small model can learn from much less data
at a much higher learning rate (≈46:29).

The lecturer adds a consequence for closed models: from a model's predictive scores alone, "you could arguably learn a model
that can match" them, "maybe one of the reasons that a lot of those closed source models don't actually give you output
predictions over the whole category set" (≈44:57–45:44).

## Variants

- **Cross-modal distillation** (slide 33, Gupta, Hoffman and Malik, 2015): a teacher on one modality, such as RGB images,
  supervises a student on another, such as depth. "Useful if you have labeled data for one domain but not the other" (≈47:17).
- **SoundNet** (slide 34, Aytar, Vondrick and Torralba, 2016): image networks for objects and scenes watch the frames of
  unlabelled video, and a 1D convolutional network on the raw waveform learns to match their object and scene distributions
  (≈47:17–48:02).
- **Distilling an ensemble** (slide 35): "Ensembling model outputs creates more accurate estimates. Why? Each model is
  imperfect in different random ways. Errors cancel out, truth is shared." Distilling the ensemble into one model gives "fast
  inference" (≈48:02–48:50), but not the corner cases every member gets wrong, since models trained on the same data share
  biases (≈48:50).
- **Contrastive representation distillation** (CRD; slides 36 and 37, Tian, Krishnan and Isola, 2020) distills the
  representation rather than the outputs: [contrastive learning](contrastive-learning.md) treats the teacher's and the
  student's embeddings of an input as two "views" to be pulled together, "teacher's view" (big net) against "students view"
  (small net), as the slide prints it. On CIFAR100, the correlations between class logits, student's minus teacher's, are
  "much, much cleaner" with CRD than with standard distillation (≈49:35–50:21, ≈52:42–53:29).

## Distillation in domain adaptation

The unsupervised [domain adaptation](domain-adaptation.md) recipe of lecture 18 is built from distillation. A model trained on
the source labels the target data; a student is trained on those predictions, "a mechanism to do distillation" (slide 23,
≈34:02); the source model becomes a teacher updated as an exponential moving average of the student (slide 24); and the
thresholded labels give way to a distillation loss, matching "the distribution over the outputs" (slide 26, ≈35:37). Align and
Distill (ALDI) names it (slide 27).

## Distillation against compression

"Both give us smaller models with similar accuracy to large ones" (slide 38). Distillation "uses SGD to train a smaller model to
match characteristics of a larger one", which can be computationally expensive; compression "uses *pruning* or *quantization* to
reduce the computational cost of a large model", which can be much cheaper and can run on a device rather than in the cloud
(≈53:29–55:02). In practice the smallest good models usually come from "some combination of all of the above" (≈51:06). How much
smaller the student can be "is really directly related to the complexity of the information you're trying to model" (≈50:21),
and specializing helps: "We almost never need a truly generalist model in practice. Specialization can enable much higher rates
of compression with the same accuracy" (slide 39, ≈55:48).

## Context distillation (lecture 21)

[Lecture 21](21-language-models.md) names a distillation in which teacher and student are the same language model, with and without a
prompt. Write "a really good prompt that really makes it very clear that what the model is supposed to be doing is instruction following",
with "a ton of few-shot examples" and explicit meta-instructions; sample data from the model with that prompt; and fine-tune the model on
the samples. "What you get there is not a model that knows how to do anything new, but just a model that behaves like that really good
prompt … is sitting there in the context all the time so that the user doesn't have to type it out." The lecturer calls it "one of the
standard things that you can do with synthetic data that works", with the caveat that "it really just moves mass around in the output
space" (≈59:59–1:00:45). There is no slide for it.

## See also

- [Lecture 18](18-transfer-learning-models.md) — the lecture, with its figures.
- [Transfer learning](transfer-learning.md) — distillation as transfer of knowledge about the outputs.
- [Domain adaptation](domain-adaptation.md) — where distillation is one of the standard components.
- [Contrastive learning](contrastive-learning.md) — the loss behind contrastive representation distillation.
- [Softmax and cross-entropy](softmax-and-cross-entropy.md) — the loss distillation reuses.
