# Lecture 17 — Generalization: Out-of-Distribution (OOD)

**Lecturer:** Sara Beery ·
**Video:** [youtube.com/watch?v=tjD9LIzIIek](https://www.youtube.com/watch?v=tjD9LIzIIek) (65 min) ·
**Slides:** [`mit6_7960_f24_lec17.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec17.pdf)
(63 pages; the deck is titled "Lecture 17: Out-of-distribution generalization"; transcribed slide by slide in [`raw/slides/17-generalization-out-of-distribution.md`](../raw/slides/17-generalization-out-of-distribution.md)) ·
**Transcript:** [`raw/transcripts/17-generalization-out-of-distribution.md`](../raw/transcripts/17-generalization-out-of-distribution.md)

## What this lecture establishes

Machine learning theory, and most of the course so far, assumes that the data a model is used on is drawn from the same
distribution as the data it was trained on. This lecture is about what happens when that assumption fails, which, the
lecturer argues, is always: "Training distribution equals test distribution. This is never true in the real world"
(≈6:14). It calls **out-of-distribution generalization** "one of the biggest open challenges for modern machine learning
models, the ability to generalize to data that is not explicitly covered within the training distribution" (≈0:00), and
it is candid that "both of these are pretty active, open areas of research. So I can talk about progress, but we don't
have solutions for these" (≈14:00).

It takes the problem at two scales (slide 8). The first is **adversarial examples**: tiny, deliberately optimized
perturbations of a single input that flip a model's prediction, and **adversarial training**, which makes a model robust to
them by training on the worst perturbation of every example. What adversarial examples reveal is that models rely on
**non-robust features** and **shortcuts**, correlations with the label that hold in the training data and fail under
realistic change, and that these come from the data, not the architecture. The second scale is **distribution shift**:
the whole test distribution differs from the training one, as it does in every ecological problem the lecturer has worked
on. The **WILDS** benchmark measures how much accuracy realistic shifts cost, and the lecture surveys what to do about it:
**distributionally robust optimization** (DRO), priors for **subpopulation shift**, domain adaptation (next lecture),
diagnosing failures, and, best of all in practice, training data that represents the test domain (slide 61).

**Notation on this page** follows the slides. An input is $\mathbf{x}$ (an image, in the examples) and its label is $y$;
$f_{\theta}$ is a neural network with parameters $\theta$, and $P_{\theta}(y \vert \mathbf{x})$ the probability it assigns
to label $y$, from a softmax. A perturbation of the input is $\delta$, and $\Delta$ is the set of perturbations allowed.
The training set is $\lbrace (\mathbf{x}^{(i)}, y^{(i)}) \rbrace_{i=1}^{n}$, $\mathbb{P}_ n$ is the empirical distribution
it defines, $\mathbb{Q}$ is another distribution over pairs, and $D(\mathbb{Q}, \mathbb{P}_ n)$ a distance between
distributions.

## Where this lecture sits

"Today, we're moving on from generative models" (≈0:00). The lecture continues two threads. [Lecture
6](06-generalization-theory.md) asked why networks generalize to new data drawn from the training distribution; this one
asks what happens when the new data is drawn from somewhere else (see [generalization](generalization-and-double-descent.md)).
And it treats [data augmentation](data-augmentation.md), lecture 9's practical tool for building invariance into a model,
as one end of a spectrum whose other end is adversarial training, "adaptive data augmentation" (slide 24). The next
lecture, [transfer learning from the model side](course-map.md), is where domain adaptation and meta learning are
promised (≈30:27, ≈56:55; slide 58). The lecture was given on a United States election day ("today being voting day",
≈18:45), and it closes with "Take the extra time and go vote" (≈1:03:59).

## Are machine-learning systems ready for the real world?

The lecture opens with successes, image classification, strategy games, machine translation, robotic manipulation and
image generation (slide 2). Simple categorization on a dataset like ImageNet "is pretty much solved" apart from a few
hard, ambiguous or mislabelled corner cases (≈0:50). Then it asks the question on slide 3: "Are ML systems really ready
for the real world?"

Self-driving cars are the example of a system that works but is not trusted at scale. Regulation plays a part, but the
main reason is that "we really don't have a good mechanism to guarantee the robustness of these models. We can't
guarantee that they are not going to make potentially harmful mistakes" (≈1:38). The lecturer's characterization of the
AI that *is* deployed is that it comes down to risk: most deployed models are "high precision human verification
systems", good enough to be useful, with "a human verifier in the loop that is making corrections when something's not
right" (≈3:09). ChatGPT is the example: you prompt it, probably edit what you get, and if it is "absolute trash" you
laugh and try again, so "things like hallucination aren't so much of an issue if a human is making the corrections
themselves. It's when we start trusting these models without being human verified that we start running into issues"
(≈3:54).

Medical **decision support tools** are the same arrangement: a radiologist uses the tool for advice and remains liable
for the decision (≈3:54–4:43). The evidence on whether they help is divided. Some research finds they keep doctors from
missing things; other research finds they harm patient outcomes, "because doctors are becoming compliant, they're not
paying attention, and they're missing things in that long tail of the distribution" (≈4:43). And the bar to deploy one is
low: the lecturer recalls, without vouching for the details, something "on the order of like 100 or 1,000 patients", with
no clinical trials, because the tool only supports an expert's decision (≈5:29).

### The standard setting and the real world

The standard machine-learning setting is the one these deployments live in, and the one theory assumes: the training
data and the data the model is used on are **IID**, independently and identically distributed, samples of one
distribution, so "training distribution = test distribution" (slide 4, ≈6:14). Slide 4 draws a single distribution, a
blob of patches holding categories of objects, feeding both "Training" and "Inference".

![Slide 4: one distribution feeds both training and inference.](../raw/images/17-generalization-out-of-distribution/slide-4.jpg)
*Slide 4 — the standard setting: training and inference are drawn from one distribution.*

That is never true in practice. "There has never been any such thing as a representative data set of all reality", and
even training on billions of images scraped at scale does not fix it, "because the world is changing. So if you capture
all the data today, it's not going to necessarily be representative of a month from now, a week from now" (≈6:14–7:03).
Slide 5 draws the deployed case: the inference distribution is a distorted version of the training one, and the
photograph under "Inference" is a snowy street where the training photograph was a sunny one; "we've trained a model on
Cambridge … And now, we're trying to use that model in a different season. Now there's snow on the ground and the model
starts to fail" (≈7:48). The slide lists the ways the distributions can differ: **perturbed data** (adversarially, or
because a camera manufacturer updated the sensors in Google Street View cars), a **different label distribution** (a rare
category becomes common, or one disappears), and other shifts in "sequence/graph size, weather, country/city, source of
measurement" (slide 5, ≈7:48–8:34).

![Slide 5: training and inference distributions differ.](../raw/images/17-generalization-out-of-distribution/slide-5.jpg)
*Slide 5 — the real world: the inference distribution is a shifted version of the training one.*

### What can go wrong

Slide 6's example is a pair of posts. A robot vacuum maker, to stop its robots smearing dog mess across people's homes,
built "a diverse dataset of poop"; the reply wishes it had also built "a diverse dataset of rugs so that it didn't
confuse black stripes with cliffs" (slide 6, ≈8:34–10:07). The point: "you can collect data for the distribution shift
you do anticipate, but there's always going to be things you don't anticipate … we cannot possibly characterize and
anticipate every way that the world is going to change" (≈10:07).

This becomes a concrete problem in AI safety: a paper on the field's largest problems, "Concrete Problems in AI Safety",
lists "Robustness to Distributional Change" as one of them (slide 7, ≈10:52). It is also a problem for science. A 2022
workshop at Princeton, "The Reproducibility Crisis in Machine Learning-Based Science", looked at about 1,000 papers across
many disciplines, and one of the biggest failure modes was what the lecturer calls **data poisoning**: "the way that they
constructed the evaluation for these machine learning systems was, in some way, artificially IID in a way that wasn't
representative of reality" (≈10:52–11:38). (Outside the course material: this failure, test data that shares
correlations with the training data, is usually called *data leakage*; the transcript carries a note.)

The lecturer sees the same mismatch reviewing papers in ecology, between the data a method is evaluated on and the
claims made for it (≈11:38). "It's fine to have a limited data set, but then you have to understand what that means about
what you can claim": a system that identifies the individual macaques of one national park supports a claim about
monitoring that park, if it has been evaluated over time (≈12:26). "The way you evaluate these models needs to be
representative of the actual deployment use", and tools that split data at random by default give "very falsely high
expectations … because you've got … strong, spatial, temporal, or contextual correlations in your training and test data
that make the model think it's doing well when it's actually not" (≈12:26–13:13).

## Adversarial examples

### Small perturbations that flip a prediction

A classifier says a photograph of a pig is a "pig" with 91% confidence. Add $0.005$ times a pattern that looks like
random speckle, and it says "airliner" with 99% confidence, though the two images look identical (slide 10, ≈14:00). The
noise is "explicitly not random, it's actually optimized to be maximally adversarial … I can see no actual visible
difference between the two images" (≈14:49). Model predictions "are (mostly) accurate but can be brittle" (slide 10),
brittle to things a human visual system is not brittle to at all (≈14:49). Slide 11 shows the same for traffic signs
(Papernot et al. 2017).

It works in the physical world too. In a video, a classifier sees a banana as a banana; placing a photograph of a toaster
beside it does not change the prediction, but a small, specially constructed **adversarial sticker** makes it predict
"toaster" (slides 12–15, ≈15:36–16:25). "From a classification standpoint", the lecturer notes, the photograph of a real
toaster is the ambiguous case, while the sticker "looks nothing really like a toaster to us" (≈16:25). A 3D-printed turtle
whose texture was optimized is classified as a rifle from almost every angle and never as a turtle (slides 16–17,
≈16:25–17:11). And speech recognition falls the same way: adding $0.001$ times a crafted waveform to a recording of "it
was the best of times, it was the worst of times" makes the system transcribe "it is a truth universally acknowledged that
a single", with almost no visible change to the waveform (slide 18, ≈17:11–17:58).

Slide 19 asks the three questions the section answers: "Are our models completely useless? Why does this happen? Can one
prevent it?"

### A short history

Adversarial examples are older than deep learning. Slide 20 reproduces a timeline from Biggio and Roli (2018), "Wild
Patterns: ten years after the rise of adversarial machine learning", which starts with work in 2004–2005 on evading
linear classifiers (Dalvi et al., Lowd and Meek). One of the original motivating examples was a spam filter: from a "red
team, blue team perspective", could you build emails that get through it (≈17:58–18:45)? The lecturer's present-day
example, on election day, is misinformation filters: detecting deepfakes, and the "active antagonism of trying to actually
develop deepfakes that are undetectable by these algorithms" (≈18:45–19:33).

### How to create one

Assume a fixed model. Creating an adversarial example is then an optimization **over the data**: find "a small
perturbation that does not change meaning to a human, but to ML model" (slide 21, ≈19:33). The model outputs
$P_{\theta}(y \vert \mathbf{x})$ through a softmax, and an adversarial example for an input image $\mathbf{x}$ and a
chosen wrong class $y_{\text{target}}$ (such as "airliner") solves

$$\max_{\delta \in \Delta} P_{\theta}(y_{\text{target}} \vert \mathbf{x} + \delta)$$

(slide 21). Any wrong class will do: "you could also find the noise that made this pig look most like, I don't know, your
friend Bob" (≈20:19). The set $\Delta$ of **allowed perturbations** is commonly an $\ell_ \infty$ ball,

$$\Delta = \lbrace \delta \in \mathbb{R}^{d} \mid \lVert \delta \rVert_{\infty} \lt \epsilon \rbrace,$$

where $d$ is the number of input dimensions (slide 21). The infinity norm is the largest absolute entry, so "instead of
just looking at the aggregate difference between the image and the new one … the largest change for any given pixel has
to be less than a certain amount", which tends to give perturbations imperceptible to humans (≈21:06–21:54).

The optimization is solved by **projected gradient ascent**: "flip SGD on its head" and update the perturbation rather
than the weights (≈21:54). Repeat two steps (slide 22):

1. take a step in the direction of the gradient,
   $\delta^{(t+1)} = \delta^{(t)} + \eta \cdot \nabla_{\delta} P_{\theta}(y_{\text{target}} \vert \mathbf{x} + \delta)$,
   with step size $\eta$;
2. project the result back into the feasible set $\Delta$.

Slide 22's sketch shows a step that leaves the square $\Delta$ and the projection that brings it back to the edge. "It's
quite simple and it actually works quite well" (≈22:41).

![Slide 22: projected gradient ascent on the perturbation.](../raw/images/17-generalization-out-of-distribution/slide-22.jpg)
*Slide 22 — projected gradient ascent: step along the gradient with respect to the perturbation, then project back into the allowed set.*

A student asked whether this needs the model's gradients. It does: projected gradient ascent "assumes that you have the
model weights and the architecture so you could actually calculate the gradients". Without them you could search by brute
force, "trying a bunch of things and just picking the one greedily that the result was worst", which "might be more
expensive", and other methods exist (≈22:41–24:13).

### How to defend: adversarial training

A second way to state the attack drops the target class: find the perturbation that makes the **loss** on the correct
label $y$ as large as possible, "the noise that will make this most wrong" (slide 23, ≈24:13). Set beside standard
training, the two are a max and a min of the same loss:

$$\max_{\delta \in \Delta} \text{Loss}(f_{\theta}(\mathbf{x} + \delta), y) \qquad \text{versus} \qquad \min_{\theta} \text{Loss}(f_{\theta}(\mathbf{x}), y)$$

(slide 23). **Adversarial training**, or robust optimization, nests them: find the parameters for which, on every training
example, even the worst allowed perturbation does as little harm as possible (slide 24, ≈24:59–25:45):

$$\min_{\theta} \frac{1}{n} \sum_{i=1}^{n} \max_{\delta \in \Delta} \text{Loss}(f_{\theta}(\mathbf{x}^{(i)} + \delta), y^{(i)})$$

where standard training minimizes the same average without the inner max. Slide 24 calls this "adaptive data
augmentation": rather than augmenting the data with fixed transformations, training generates the hardest version of each
example as it goes. The lecturer compares it to hard example mining, met earlier in the course ([lecture
12](12-representation-learning-similarity-based.md)'s hard negatives): "it's maybe most informative, sometimes, if you show
models explicit examples of things are going to be hard and you want it still do well" (≈25:45).

With stochastic gradient descent the loop is: sample a data point $(\mathbf{x}, y)$; compute the optimal adversarial
perturbation $\delta^{\ast}$, approximately, by projected gradient ascent; compute the gradient
$g = \nabla_{\theta} \text{Loss}(f_{\theta}(\mathbf{x} + \delta^{\ast}), y)$; update $\theta$ with $g$; repeat until
convergence (slide 25, ≈29:41–30:27). "For every data point, you're training another model": the inner loop runs inside
the outer one, as in meta learning, which the next lecture covers, so adversarial training is "quite computationally
expensive" (≈30:27).

Three student questions marked its limits. Asked what happens when there are two very different hard examples, the
lecturer restated it as "what if the data changes in a way you didn't anticipate", and answered that the method is robust
only to "the types of perturbations that you've defined as within your reasonable set of perturbations". Small noise is
not how most real data changes, and this defence "would probably not work very well for those stickers" (≈26:31–28:03).
A student's example of a perturbation no formula models, a pedestrian in a crosswalk, drew the story of a self-driving car
behind a truck carrying stop signs, "just freaking out because what is it supposed to do?" (≈28:03). "A lot of the work in
… adversarial robustness, they have to make assumptions, and usually, those assumptions are violated in real world"
(≈28:49). And asked whether the back-and-forth between minimizing and maximizing converges: "for a reasonable set of
perturbations … it usually is often possible", but with two conflicting objectives "they can be pretty finicky to train"
(≈28:49–29:41).

## What adversarial examples reveal about models

### Features that are predictive but not robust

That adversarial examples exist "tells us something about the input 'features' that are critical for the model's
decision" (slide 26, ≈30:27–31:12). The example is MNIST training data for "4 vs 9" in which every 4 has a white pixel in
its top-left corner and every 9 a white pixel just next to it. The model learns to look at that pixel and ignore the rest
of the image, and an adversarial perturbation that does no more than move the pixel gives "the exact opposite prediction"
(slide 26, ≈31:12–31:59). Models "learn whatever signals are there, and ideally, whatever signals are easiest to pick up"
(≈31:59).

Slide 27 sorts a model's features three ways: **useless features**; **robust features**, "correlated with label even when
perturbed", such as the semantic parts of a dog; and **non-robust features**, "correlated with label, but can be flipped
via perturbation" (slide 27, ≈31:59–32:44). Many features "may be correlated with the label and hence predictive and help
with accuracy, beyond what humans would use" (slide 27). Some are worse than inhuman: "things that we might use based on
stereotypes". Early text-to-image diffusion models asked for doctors drew "a bunch of white men" and asked for homemakers
drew women (≈32:44–33:31); such models "tend to exacerbate societal biases", which is the subject of the literature on
fairness in machine learning. So "correlative predictive features are not always something we want the model to use, both
because it can make it non-robust, but also because it can exacerbate bias" (≈34:20).

### Where the correlations come from: the data

"Where do the correlations come from? They really come from the data" (slide 28, ≈34:20). A dog in a bow tie is predicted
to be a cat, because people put bow ties on cats more than dogs; and the photographs labelled "fish" in the ImageNet
training set are mostly people holding up their catch, so "the model's not learned fish. It's learned like fish in the
hands of a person" (slide 28, ≈35:05). The root is how datasets are made. In an ideal world, representative real-world
images go to expert annotators, whose perfect annotations make a meaningful benchmark; in the real world, scraped images
get automated and crowd labels, whose noisy, biased annotations make an easy-to-optimize benchmark that "is not
necessarily something that is rooted in reality" (slide 29, ≈35:05–35:51).

### Shortcuts

One argument is that "all these models are learning are shortcuts": **shortcuts** are "features correlated with label in
the training data, but not under realistic distribution shifts", and "models will use them and not generalize if features
are no longer correlated" (slide 30, ≈35:51–36:36). Slide 30's illustration, from Geirhos et al. (2020), trains on stars
labelled A and moons labelled B where each shape also sits in a typical position; on an i.i.d. test set the network agrees
with a human, and on an o.o.d. test set with the shapes moved it labels by position rather than shape.

Imbalance makes shortcuts out of context. In the lecturer's species identification from static cameras, raccoons are
mostly photographed at night, so the model learns that a raccoon is "nighttime, kind of like, fat, cat blob thing", and
on daytime raccoons, though it saw some in training, "the performance is just much worse" (≈36:36–37:22). Because
shortcuts live in the data, "adversarial examples transfer across models trained on the same dataset" even when the
architectures differ: "this is related to 'data', not models" (slide 31, ≈37:22).

The lecture's gallery of shortcuts: a captioner that sees "a herd of sheep grazing" on an empty green hillside, "a flock
of birds flying in the air" for goats standing in a tree, and dogs in the arms of people holding goats (slide 32,
≈38:09); chest X-ray models that learned "where an x-ray was acquired … and calibrate[d] predictions accordingly" (Zech et
al. 2018), and a skin-lesion model more likely to call a tumour malignant "if an image had a ruler in it" (Esteva et al.
2017) (slide 33, ≈38:58). The ruler case means the model is least likely to catch the lesions a doctor has not already
suspected and measured. "Not all predictive patterns are desirable" (slide 33). Slide 34 sorts many such failures, again
from Geirhos et al. (2020), into images that are the same category for humans but not for networks (domain shift,
adversarial examples, distortions, pose, texture, background) and images that are the same category for networks but not
for humans (excessive invariance, fooling images, natural adversarials, texturized images) (≈39:43–40:29). Transformers
are no exception: transformers trained on algorithmic reasoning tasks failed to generalize beyond the sequence lengths they
were trained on, since "parallel solutions generalize within-distribution, but not out-of-distribution" (slide 35, Liu et
al., "Transformers Learn Shortcuts to Automata"; ≈40:29).

### What adversarial training changes

Training a model to be stable under adversarial perturbations "teaches invariance to non-robust features" (slide 36,
≈40:29–41:16), within the assumption that we know which perturbations to be robust to, "which, of course, we don't often
know". It is much more expensive, but it gives features "that tend to be more human intuitive" (≈41:16). Slide 37 (Tsipras
et al. 2019) shows the loss gradients with respect to the input pixels: for a standard model they look like noise, for
$\ell_ \infty$- and $\ell_ 2$-trained models they trace the insect, the dog and the primate. The lecturer is "always a
little bit iffy on the use of things like these activation heat maps as concrete explainers", "but these do look
interesting" (≈41:16–42:02). The adversarial examples change too: to fool a robust model, the perturbation has to make the
image look like the target class, so an attack turning a photograph of birds into "dog" paints dog faces into it, "the
type of mistakes" a human finds more acceptable (slide 37, ≈42:02–42:48).

And adversarially trained models transfer better. Slide 38 (Salman et al. 2020, "Do adversarially robust ImageNet models
transfer better?") compares the transfer accuracy of standard and robust ImageNet models of four sizes on twelve datasets,
and the robust models come out ahead in 40 of the 48 comparisons, and for the largest model on all twelve datasets, by the
slide file's readings of the bars (≈42:48). The lecturer sums up adversarial training as "another very
targeted form of maybe regularization or a different type of domain adaptation", targeted at one explicitly constructed
form of perturbation (≈42:48–43:34).

![Slide 38: transfer accuracy of standard and robust models.](../raw/images/17-generalization-out-of-distribution/slide-38.jpg)
*Slide 38 — transfer accuracy on twelve datasets, standard (red) against adversarially trained (blue) ImageNet models of four sizes.*

## Distribution shifts

### Shifts of whole distributions

The second half moves from perturbing single data points to perturbing whole distributions: training and inference drawn
from "a pretty different distribution" (slide 40, ≈43:34).

![Slide 40: same distribution against different distributions.](../raw/images/17-generalization-out-of-distribution/slide-40.jpg)
*Slide 40 — the two settings side by side: one distribution for training and inference, or two that differ.*

"This is basically the world I live in." Slide 41's left map estimates alpha diversity, the number of unique species in
each place, across the globe; its right map is a heat map of the three billion species occurrence records in the Global
Biodiversity Information Facility (GBIF). They are "clearly not super well-aligned. They're almost anti-correlated", so a
model trained on the records and used where the species are faces "a pretty significant distribution shift", on top of
everything changing all the time (slide 41, ≈43:34–44:20).

![Slide 41: global biodiversity against where species records come from.](../raw/images/17-generalization-out-of-distribution/slide-41.jpg)
*Slide 41 — where species are (left) and where the data about them comes from (right).*

### Measuring realistic shift: WILDS

That gap motivated **WILDS**, a benchmark the lecturer built with many collaborators (slide 42, ≈44:20). Much of the
literature on distribution shift studied shifts that are not realistic: added noise, MNIST rotated by 45 degrees or
recoloured, or datasets like Waterbirds, a deliberate background shift (birds out of water in training, in water at test),
which is still artificial because the categories are balanced and only one dimension shifts, "and the real world is quite
messy" (≈44:20–45:09). WILDS gathers real shifts: hospitals in histopathology, camera locations in wildlife
identification, countries in poverty mapping, time in satellite imagery, users in product reviews, demographics in comment
moderation, and molecular scaffolds in bioassays (slide 42, ≈45:09–45:54). Its finding: "when you move out of distribution
in these realistic ways, basically, model performance degrades significantly" (≈45:54). Slide 43 gives four drops from
in-distribution (ID) to out-of-distribution (OOD) test data: 93.2% to 70.3% accuracy across hospitals, 48.6% to 32.3%
across time in satellite images, 63.3% to 49.6% across regions in wheat-head detection, and 34.4% to 27.2% average
precision across scaffolds.

The wildlife case is the lecturer's own (slide 44). Training data comes from 245 cameras; the in-distribution control test
set is new images from those same cameras ("not actually identical. It's just randomly sampled from within the training
data locations"); the out-of-distribution test set is held-out cameras. Macro F1 falls from 47.0% to 31.0%, "like, 16%,
which is pretty significant" (slide 44, ≈45:54–46:41). The causes are many: background, camera pose and angle, infrared
at night against colour by day (≈46:41).

### Subpopulation shift

Besides that visual shift there is **subpopulation shift**: every camera "has its own unique distribution over the
categories of interest", because it sits in a microhabitat with its own local species (slide 45, ≈46:41–47:26). Slide 45
plots the number of images per category, on a log scale, at three locations, and the three histograms barely resemble each
other. "It turns out that this is basically true for every ecological problem I've ever worked on": different ecosystems
have both subpopulation and visual shifts, as in detecting individual trees from aerial images across the National
Ecological Observatory Network, where trees look different and grow at different densities (slide 46), counting fish in
rivers of the Pacific Northwest, and categorizing urban trees across North American cities (≈47:26–48:17).

![Slide 45: per-location class distributions.](../raw/images/17-generalization-out-of-distribution/slide-45.png)
*Slide 45 — the class distribution of three camera locations: each sensor sees its own mix of species.*

For the urban trees, performance "has a very strong correlation with subpopulation distribution similarity" (slide 47,
≈48:17). The slide's left matrix is the L1 distance between the per-genus distributions of every pair of cities, the
right the accuracy of a model trained on one city and tested on another, and the two show matching block structures:
models generalize better among cities whose tree populations are similar. "These models are trained on likelihoods, so
you're almost building a prior based on the subpopulation distribution within any given training set" (≈48:17–49:03).
That suggests a cheap fix that needs no training: if you can represent a prior over the categories you expect in the test
setting, "you can just do a really simple Bayesian remapping of the predicted probabilities to take into account that
prior" (≈49:03).

![Slide 47: distribution distance against accuracy across cities.](../raw/images/17-generalization-out-of-distribution/slide-47.jpg)
*Slide 47 — genus-distribution distance between cities (left) and cross-city accuracy (right): similar cities, better transfer.*

## What to do about distribution shift

### Distributionally robust optimization

The first answer (slide 48) mirrors adversarial training one level up. "So far: allowed to perturb each datapoint by a
limited amount"; the alternative is to "perturb the entire training distribution (sample) by a certain amount, together",
moving the empirical distribution $\mathbb{P}_ n$ to a nearby $\mathbb{Q}$ (slide 49, ≈49:03–49:50).

![Slide 49: perturbing data points against perturbing the distribution.](../raw/images/17-generalization-out-of-distribution/slide-49.png)
*Slide 49 — adversarial training perturbs each point; DRO perturbs the whole training distribution.*

Standard training minimizes the average loss over the training sample, which is an expectation over its empirical
distribution:

$$\frac{1}{n} \sum_{i=1}^{n} \text{Loss}(f_{\theta}(\mathbf{x}^{(i)}), y^{(i)}) = \mathbb{E}_ {(\mathbf{x}, y) \sim \mathbb{P}_ n} \left[ \text{Loss}(f_{\theta}(\mathbf{x}), y) \right]$$

**Distributionally robust optimization** (DRO) minimizes the expected loss under the worst distribution within distance
$\epsilon$ of it (slide 50, ≈49:50–50:37):

$$\min_{\theta} \max_{\mathbb{Q}, \thinspace D(\mathbb{Q}, \mathbb{P}_ n) \lt \epsilon} \mathbb{E}_ {(\mathbf{x}, y) \sim \mathbb{Q}} \left[ \text{Loss}(f_{\theta}(\mathbf{x}), y) \right]$$

The worst case might, for instance, "re-weight or perturb training data points" (slide 50): change the densities, take
some data away. The catch is the distance $D$. Measuring how far one image is from another is easy; measuring how far
apart two distributions are is not, and slide 50 lists choices, $\chi^2$-distance, Wasserstein distance and maximum mean
discrepancy (MMD). For two simple one-dimensional distributions this is one thing; for two datasets of a million images
"it gets a lot more complicated", and people often compare learned representations of the datasets instead. "This stuff
is pretty tricky. It's not definitely something that works super well in practice" (≈50:37–51:24).

What DRO buys, in principle, is generalization. It optimizes for a whole set of training distributions. If the true data
distribution is $\mathbb{P}$, the empirical training data is $\hat{\mathbb{P}}_ n$, and
$D(\mathbb{P}, \hat{\mathbb{P}}_ n) \lt \epsilon$, then "we are guaranteed to perform well on $\mathbb{P}$ too, i.e., generalize!" (slide 51, ≈51:24–52:10).
The limitation is the same as before: you must be able to measure distances between distributions and sample the
hypothetical ones realistically, and "the more complicated your data structures are, the harder that gets" (≈52:10).
The slides write the empirical distribution two ways: $\mathbb{P}_ n$ in the objective and $\hat{\mathbb{P}}_ n$ in the
bullets and the circle diagram, which also writes $d$ and $\leq$ where the objective has $D$ and $\lt$ (slides 50 and 51).
They are the same objects.

![Slide 51: the true distribution inside the DRO ball.](../raw/images/17-generalization-out-of-distribution/slide-51.jpg)
*Slide 51 — if the true distribution lies inside the ball DRO optimizes over, good performance carries over to it.*

### DRO and class imbalance

DRO has been applied effectively to class imbalance (slide 52, ≈52:55). Suppose the population has $K$ subgroups, say
$K = 2$, such as the lecturer's raccoons photographed at night and by day. Empirical risk minimization weights each group
by its share of the data, 80% and 20% on slide 52:

$$\min_{\theta} \frac{1}{n} \left( \sum_{i \text{ in group 1}} \text{Loss}(x_i; \theta) + \sum_{j \text{ in group 2}} \text{Loss}(x_j; \theta) \right)$$

Then "50% error on minority group makes only 10% average error", and the minority's statistical patterns may differ too,
so a model "can 'ignore' minority group and still get decent loss!" (slide 52, ≈52:55–53:44). DRO's worst case
**re-weights the data automatically**, so it pays more attention to the minority class (slide 53). Empirically this puts
less emphasis on noisy features and more on useful ones, and seems to shift the distribution so the groups are more
separable; and there are adaptations of DRO that work "even if you don't know what the subgroups are or how many subgroups
there are" (≈53:44–54:33). Slide 53 shows a result from Hashimoto et al. (2018), "Fairness without demographics in
repeated loss minimization", in which the DRO distribution puts far more weight than the empirical one on the smaller of
two clusters. That is why DRO is used for fairness: a model that recognizes doctors should not be biased against female or
Black doctors because the training data is; and a worst-case distribution shift might sample only those minority groups
(≈54:33).

![Slide 53: DRO re-weights toward the minority group.](../raw/images/17-generalization-out-of-distribution/slide-53.jpg)
*Slide 53 — the DRO distribution against the empirical one (Hashimoto et al. 2018): weight moves toward the smaller group.*

### Learn, or use, a prior for subpopulation shift

The second answer is to model the shift you expect. Mac Aodha et al. (2019) learned, separately from the image
classifier, a spatiotemporal prior over places and times, and combined the two:

$$P(y \mid I, \mathbf{x}) \propto P(y \mid I) \thinspace P(y \mid \mathbf{x})$$

where $I$ is the image and $\mathbf{x}$ = (longitude, latitude, day) (slide 55, ≈55:21). "This is what ecologists do":
species that look nearly identical can be hard even for an expert to tell apart from the image alone, but easy with
"context about where, when it was taken" (≈55:21–56:06). Building that structure of the domain into the model improved
species accuracy "by like, 10%" on a species dataset (≈56:06).

### Domain adaptation

**Domain adaptation** is "an entire category of models" that, given a source domain and a target domain without labels,
or with few, learns "how to align their representations during training so that you can enforce generalization or
adaptation to a new target domain" (slide 57, ≈56:06–56:55). Slide 57 draws it as two clouds of shapes, solid for the
source and hollow for the target, merged so that each kind of shape lines up. It is next lecture's topic (slide 58).

### Diagnose failures

Creative ways to diagnose a model's failures "often teach you more about the biases in your initial training data than
anything else". The simplest is to break the evaluation down by category and along dimensions such as night against day,
or the sexes of participants, where the labels allow (≈56:55–57:40). A more elaborate one, from Aleksander Madry's group
(Vendrow et al.), uses generative models to make counterfactual images: an ImageNet-trained classifier recognizes plates
with food on them but calls an empty plate a "tray" or a "bucket", a failure then confirmed on real photographs (slide 59,
≈57:40–58:27). Diagnosis tells you what training data to collect, or how far to trust a deployed model (≈58:27).

### Get representative training data

The last answer is the one the lecturer rates highest: "Get training data that is representative of your test domain
(works better than any algorithm)" (slide 61). "In practice, a lot of these algorithmic approaches … usually, they're just
significantly worse than just more training data that's representative of your test domain. So from a practical
perspective, if you really need something to work well, you have to invest in the right kind of training data"
(≈58:27–59:16). Where that is impossible, for a highly endangered species, on another planet, under the Arctic ice or in
the deep sea, "you're probably going to have to have a human look at a lot of the data still" (≈59:16). Her advice to
students bringing deep learning to their own scientific problems: read this literature, but "if you have limited time and
effort, sometimes, it's just worth it to label more data" (≈1:00:01).

## Summary, and two questions

"Out-of-distribution generalization: big challenge, but helps understand what NNs learn" (slide 62), and what it helps
understand is "this very tight and complex relationship between the training data and where we can effectively use models"
(≈1:00:01).

A student asked whether the choice of activation function, ReLU or GELU, changes how easily a model is fooled. The
lecturer knew of no literature; her intuition was that it "wouldn't have a huge effect", perhaps more on simple datasets
like MNIST or CIFAR than on complex ones, "but that's totally just my best guess" (≈1:00:47–1:01:36). Another asked
whether it would be cheaper to collect feedback on the shifted data, through humans or reinforcement learning, than to
train for robustness example by example. "People totally do that", with human-in-the-loop learning, reinforcement
learning, Direct Preference Optimization (DPO), domain adaptation or supervised fine-tuning, and it can be much more
efficient when there is a specific target dataset. The lecture's methods aim at something else, being "robust to any
imagined hypothetical change": "the difference between trying to be robust to distribution shifts versus trying to adapt
to a specific distribution shift" (≈1:02:25–1:03:59).

## Pointers to other lectures

The deck prints no lecture number but its own, on the title slide; slides 58, 60 and 61 point to "Domain adaptation (next
lecture!)", which is lecture 18, Transfer Learning: Models. The recording points back to image generation "the last
couple of days" (lectures 14–16, ≈0:50) and to hard example mining "a while ago" (lecture 12's hard negative mining,
≈25:45), and ahead to meta learning and domain adaptation "next lecture" (lecture 18, ≈30:27, ≈56:55). Lecture 1's banner
had announced this lecture as "Lecture 17: OOD generalization". See the [course map](course-map.md).

## See also

- [Adversarial examples](adversarial-examples.md) — the attack as an optimization over the input, projected gradient
  ascent, adversarial training, and what robust models learn.
- [Out-of-distribution generalization](out-of-distribution-generalization.md) — shortcuts, distribution and
  subpopulation shift, WILDS, distributionally robust optimization and the other remedies.
- [Generalization and double descent](generalization-and-double-descent.md) — generalization within the training
  distribution, lecture 6's question.
- [Data augmentation](data-augmentation.md) — fixed augmentation, of which adversarial training is the adaptive form.
- [Norms](norms.md) — the $\ell_ \infty$ and $\ell_ 2$ balls that bound a perturbation.
- [Transfer learning](transfer-learning.md) — adversarially trained models transfer better; domain adaptation is next.
