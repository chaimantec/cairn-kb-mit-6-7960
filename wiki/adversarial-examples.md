# Adversarial examples and adversarial training

An **adversarial example** is an input changed by a perturbation too small for a person to notice, chosen so that a
trained model changes its prediction, often to a wrong answer held with high confidence. **Adversarial training** makes a
model robust to such perturbations by training it on the worst perturbation of each example. The course treats both as the
small-scale half of out-of-distribution generalization: robustness "on the level of individual data points", before
shifts of whole distributions (lecture 17, slide 8, ≈13:13). What they reveal about models is as important as how to
defend against them: models rely on features that predict the label without meaning what a human would take them to mean.
Covered so far: [lecture 17](17-generalization-out-of-distribution.md), slides 9–38, ≈13:13–43:34; [lecture
18](18-transfer-learning-models.md), slides 20 and 65–67, ≈30:57–31:44 and ≈1:18:27–1:21:30 (an adversary over features in domain
adaptation, and beneficial noise in visual prompting).

**Notation.** An input is $\mathbf{x} \in \mathbb{R}^{d}$ and its label $y$; $f_{\theta}$ is a network with parameters
$\theta$, and $P_{\theta}(y \vert \mathbf{x})$ the probability it assigns to $y$ through a softmax. A perturbation is
$\delta \in \mathbb{R}^{d}$, $\Delta$ the set of perturbations allowed, and $\text{Loss}$ the training loss. The training
set is $\lbrace (\mathbf{x}^{(i)}, y^{(i)}) \rbrace_{i=1}^{n}$.

## Small perturbations, large effects

Lecture 17's first example adds $0.005$ times a speckle pattern to a photograph of a pig: the classifier's answer moves
from "pig" at 91% confidence to "airliner" at 99%, and the two images look the same (slide 10, ≈14:00–14:49). The pattern
looks random but is "explicitly not random, it's actually optimized to be maximally adversarial" (≈14:49). The attack
works in the physical world: a printed sticker beside a banana makes a classifier answer "toaster" (slides 12–15), a
3D-printed turtle with an optimized texture is classified as a rifle from almost every angle (slides 16–17), and a
crafted waveform $0.001$ times the size of a recording makes a speech recognizer transcribe a different sentence (slide
18) (≈15:36–17:58). Models "are (mostly) accurate but can be brittle" (slide 10), to things that a human visual system is
not brittle to at all (≈14:49).

The idea predates deep learning: slide 20's timeline, from Biggio and Roli (2018), starts with evading linear classifiers
in 2004–2005, and an early motivating case was getting email past a spam filter (≈17:58–18:45).

## Creating one: optimize the input

With the model fixed, an adversarial example is found by optimizing over the input instead of the weights. For a target
class $y_{\text{target}}$ other than the true one,

$$\max_{\delta \in \Delta} P_{\theta}(y_{\text{target}} \vert \mathbf{x} + \delta), \qquad \Delta = \lbrace \delta \in \mathbb{R}^{d} \mid \lVert \delta \rVert_{\infty} \lt \epsilon \rbrace$$

(lecture 17, slide 21, ≈19:33–21:54). The $\ell_ \infty$ ball bounds the largest change to any single pixel, which tends
to keep the change imperceptible (≈21:06; see [norms](norms.md)). Without a target, the attack maximizes the loss on the
true label instead, $\max_{\delta \in \Delta} \text{Loss}(f_{\theta}(\mathbf{x} + \delta), y)$ (slide 23, ≈24:13).

**Projected gradient ascent** solves it: "flip SGD on its head" (≈21:54), take a gradient step on the perturbation,

$$\delta^{(t+1)} = \delta^{(t)} + \eta \cdot \nabla_{\delta} P_{\theta}(y_{\text{target}} \vert \mathbf{x} + \delta),$$

with step size $\eta$, project the result back into $\Delta$, and repeat (slide 22). It needs the model's weights and
architecture to compute the gradient; without them one can search more expensively by trial (≈22:41–24:13). See
[gradient descent](gradient-descent.md).

## Adversarial training

Standard training minimizes the average loss; adversarial training, or robust optimization, minimizes the average of the
worst-case loss over the allowed perturbations of each example (lecture 17, slide 24, ≈24:59–25:45):

$$\min_{\theta} \frac{1}{n} \sum_{i=1}^{n} \max_{\delta \in \Delta} \text{Loss}(f_{\theta}(\mathbf{x}^{(i)} + \delta), y^{(i)})$$

Slide 24 calls it "adaptive data augmentation" (see [data augmentation](data-augmentation.md)), and the lecturer likens it
to hard example mining. Each step of stochastic gradient descent samples a point, approximately computes its optimal
perturbation $\delta^{\ast}$ by projected gradient ascent, and updates $\theta$ with
$\nabla_{\theta} \text{Loss}(f_{\theta}(\mathbf{x} + \delta^{\ast}), y)$ (slide 25). With an inner optimization inside every
outer step, it is "quite computationally expensive", and with two conflicting objectives it can be "pretty finicky to
train", though for a reasonable perturbation set it usually converges (≈29:41–30:27).

Its protection reaches only as far as the perturbation set: it is robust to "the types of perturbations that you've
defined as within your reasonable set", which small pixel noise does not cover, and "this type of approach would probably
not work very well for those stickers" (≈27:16–28:03). "A lot of the work in … adversarial robustness, they have to make
assumptions, and usually, those assumptions are violated in real world" (≈28:49).

## What adversarial examples reveal

They show which input features the model's decision rests on (slide 26). Slide 27 sorts features into **useless**,
**robust** ("correlated with label even when perturbed") and **non-robust** ("correlated with label, but can be flipped via
perturbation"); many are predictive "beyond what humans would use" (≈31:59–32:44). The MNIST example of slide 26 makes it
concrete: put a white pixel in one corner of every 4 and beside it in every 9, and the model learns the pixel, so moving
the pixel flips the answer (≈31:12–31:59). Because these features come from the data, "adversarial examples transfer
across models trained on the same dataset" (slide 31, ≈37:22); see
[out-of-distribution generalization](out-of-distribution-generalization.md) for the shortcuts they come from.

Adversarial training "teaches invariance to non-robust features" (slide 36). Its models' input gradients look like the
object rather than noise, the adversarial examples that fool them start to look like the target class, and they transfer
better to other datasets (slides 37–38, Tsipras et al. 2019 and Salman et al. 2020; ≈41:16–42:48). The lecturer calls it
"another very targeted form of maybe regularization or a different type of domain adaptation" (≈42:48).

## Turned around (lecture 18)

Lecture 18 uses the same optimization in two other directions. In **domain adversarial training** (slide 20) the adversary is a
classifier that predicts which domain an input came from, and a gradient reversal layer trains the shared features to make it fail,
"as bad as possible at predicting the domain", so that the two domains' representations align (≈31:44; see [domain
adaptation](domain-adaptation.md)). And **visual prompting** (slides 65–67) optimizes a perturbation of the input to *help* a frozen
model: "It's kind of similar to adversarial noise. We learned about learning adversarial noise, but this is now like learning
beneficial noise" (≈1:18:27). A learned border added to every image, or even one pink pixel at the same place in every satellite
image, raises accuracy across a dataset, which the lecturer reads as one more sign of "how fragile and complex the statistical
representations are" (≈1:21:30). See [foundation models and prompting](foundation-models-and-prompting.md).

## See also

- [Lecture 17](17-generalization-out-of-distribution.md) — where all of this comes from, with the student questions.
- [Out-of-distribution generalization](out-of-distribution-generalization.md) — shortcuts, distribution shift, and
  distributionally robust optimization, the same min-max one level up.
- [Norms](norms.md) — the $\ell_ \infty$ and $\ell_ 2$ balls.
- [Data augmentation](data-augmentation.md) — fixed augmentation, of which adversarial training is the adaptive form.
