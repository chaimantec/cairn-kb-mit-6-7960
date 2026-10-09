# Course map — MIT 6.7960 Deep Learning, Fall 2024

Instructors: Phillip Isola, Sara Beery and Jeremy Bernstein. Course site:
[MIT OpenCourseWare](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/). This page
collects what lecture 1 says about how the course runs, and maps its schedule onto the recorded
lectures. Everything here is sourced from [lecture 1](01-introduction.md), the announcements slide of
[lecture 2](02-how-to-train-a-neural-net.md), what [lecture 3](03-approximation-theory.md) says
about the problem sets, what [lecture 4](04-architectures-grids.md), [lecture 5](05-architectures-graphs.md) and
[lecture 6](06-generalization-theory.md), [lecture 7](07-scaling-rules-for-optimization.md), [lecture 8](08-architectures-transformers.md), [lecture 9](09-hackers-guide-to-deep-learning.md), [lecture 10](10-architectures-memory.md) and [lecture 11](11-representation-learning-reconstruction-based.md) say about other lectures and the problem sets, and the OCW site; as
later lectures are added to this knowledge base, their pages become the authority on their own
content.

## What the course is, and is not

Lecture 1 is explicit that this "is not an intro to deep learning class. This is advanced
graduate-level deep learning" (≈5:24). It assumes gradient descent, MLPs and non-linearities,
softmax and cross-entropy, and batching and tensors as background (slide 67), and expects anyone
who has not seen them to "go and brush up on them" — for instance from other OpenCourseWare
courses (≈13:10). Its philosophy is that progress comes from "a mixture of theory and practice",
so it teaches both (slide 3).

## The lectures

Catalog order of the recorded lectures. There is **no lecture 22**, either in the playlist or
among the OCW slide decks, and the course also recorded a PyTorch tutorial.

| # | Lecture | In this KB |
| --- | --- | --- |
| 1 | Introduction to Deep Learning | [yes](01-introduction.md) |
| 2 | How to Train a Neural Net | [yes](02-how-to-train-a-neural-net.md) |
| 3 | Approximation Theory | [yes](03-approximation-theory.md) |
| 4 | Architectures: Grids | [yes](04-architectures-grids.md) |
| 5 | Architectures: Graphs | [yes](05-architectures-graphs.md) |
| 6 | Generalization Theory | [yes](06-generalization-theory.md) |
| 7 | Scaling Rules for Optimization | [yes](07-scaling-rules-for-optimization.md) |
| 8 | Architectures: Transformers | [yes](08-architectures-transformers.md) |
| 9 | Hacker's Guide to Deep Learning | [yes](09-hackers-guide-to-deep-learning.md) |
| 10 | Architectures: Memory | [yes](10-architectures-memory.md) |
| 11 | Representation Learning: Reconstruction-Based | [yes](11-representation-learning-reconstruction-based.md) |
| 12 | Representation Learning: Similarity-Based | not yet |
| 13 | Representation Learning: Theory | not yet |
| 14 | Generative Models: Basics | not yet |
| 15 | Generative Models: Representation Learning Meets Generative Modeling | not yet |
| 16 | Generative Models: Conditional Models | not yet |
| 17 | Generalization: Out-of-Distribution (OOD) | not yet |
| 18 | Transfer Learning: Models | not yet |
| 19 | Transfer Learning: Data | not yet |
| 20 | Scaling Laws | not yet |
| 21 | Language Models | not yet |
| 23 | Metrized Deep Learning | not yet |
| 24 | Inference Methods for Deep Learning | not yet |
| — | PyTorch Tutorial | not yet |

The spoken overview in lecture 1 (≈5:24–7:43) runs through the same arc: training, approximation
theory, grids and graphs, Jeremy Bernstein on scaling rules for optimization, generalization
theory, transformers, Phillip Isola's "hacker's guide to deep learning", memory, three lectures
of representation learning, three of generative models, out-of-distribution generalization,
transfer learning from the model and the data side, large language models, guest lectures
("TBD"), scaling laws and automatic gradient descent, and class-wide project office hours near
the end. At the time of the lecture some of this was still being settled — a "past and future
of deep learning" lecture was "potentially" on the list.

## The deck's lecture pointers

Lecture 1's "what we'll cover" slides carry yellow banners naming the lecture where each theme
is taught. **Several of those numbers do not match the published schedule**, because the deck
was drafted before the schedule settled (its title slide still reads "6.S898 … Fall 2022"). Use
the right-hand column.

| Slide | Banner says | Recorded lecture |
| --- | --- | --- |
| 30 | Lecture 2: Backprop and Differentiable Programming | 2 — How to Train a Neural Net |
| 47 | Lecture 3: Approximation theory | 3 — Approximation Theory |
| 49 | Lecture 4: CNNs | 4 — Architectures: Grids |
| 49 | Lecture 5: GNNs | 5 — Architectures: Graphs |
| 49 | Lecture 9: Transformers | **8** — Architectures: Transformers |
| 49 | Lecture 11: RNNs | **10** — Architectures: Memory (lecture 11 is reconstruction-based representation learning) |
| 52 | Lecture 7: Generalization theory | **6** — Generalization Theory |
| 52 | Lecture 17: OOD generalization | 17 — Generalization: Out-of-Distribution |
| 73 | Lectures 11, 12, 13: Reconstruction-based, Similarity-based, Theory | 11, 12, 13 — Representation Learning |
| 75 | Lectures 14, 15, 16: Basics, Representations + Generation, Conditional Models | 14, 15, 16 — Generative Models |
| 77 | Lectures 18, 19: Models, Data | 18, 19 — Transfer Learning |
| 79 | Lecture 6: Scaling Rules for Optimization | **7** — Scaling Rules for Optimization |
| 79 | Lecture 22: Scaling laws | **20** — Scaling Laws |
| 79 | Lecture 23: Automatic gradient descent | not established — the recorded lecture 23 is "Metrized Deep Learning", and nothing in lecture 1 says whether that is the same lecture |

The "Lecture 9" and "Lecture 11" rows are the ones most likely to mislead. The recorded lecture 9
is the hacker's guide, and lecture 11 is representation learning, not RNNs.

Lecture 2's deck has no pointers of this kind. The only lecture number it prints is its own, on
the title slide and the agenda ("Lecture 2", "2. How to train a neural net"), and that matches the
recording.

Lecture 3's handwritten deck has none either. Its title slide reads "6.7960 :: Lecture 3", which
matches the recording, and its last slide, "Preview: Inductive biases" (slide 42), names a topic
without a lecture number. The architecture lectures that take it up are 4, 5, 8 and 10 above.

Lecture 4's deck (titled "Lecture 4: Architectures for Grids", matching the recording) prints no
other lecture number either. Its recording points ahead twice without a number: positional encodings
will be covered "a lot more… when we get into lectures on transformers later on" (≈1:32, again at
≈1:15:19), and skip connections "get surfaced via self-attention in transformers" (≈1:07:27). Both
mean lecture 8 in the table above.

Lecture 5's deck is titled "Lecture 5: Graph Neural Networks", matching the recording, though its
footer prints the course's earlier number, "6.S898". It prints no other lecture number. Its
recording points back to "the one that Sara gave on Tuesday on convolutional networks" (lecture 4)
and "the approximation lecture that Jeremy gave" (lecture 3), and ahead to transformers, "a special
kind of graph net", without a number: "we're going to hear a lot more about that in a week or so"
(≈0:48–2:20, ≈1:20:36). That is lecture 8 above.

Lecture 6's deck is titled "Lecture 6: NN Generalization", matching the recording, which settles lecture
1's "Lecture 7: Generalization theory" banner (the row above). It prints no other lecture number. Its
recording points back to Sara's NeRF example (lecture 4, ≈1:17:40) and to graph nets' permutation symmetry
(lecture 5, ≈23:57), and ahead without numbers: more on optimization "on Thursday's lecture" (≈4:37),
lecture 7; generative and image-to-image models "a little bit later in the course" (≈20:02), lectures
14–16; language models and prompting "later" (≈16:11), lecture 21; kernels "in the representation learning
lectures" (≈1:06:45), lectures 11–13; and architectural symmetries "in transformers" (≈1:15:19), lecture 8.

Lecture 7's handwritten deck reads "6.7960 :: Lecture 7" on its title slide, matching the recording, which settles
lecture 1's "Lecture 6: Scaling Rules for Optimization" banner (the row above). It prints no other lecture
number. Its recording points back to the puzzle's other two pieces, approximation and generalization
(lectures 3 and 6, ≈1:31), and to the RMS norm "from my other lecture" (lecture 3, ≈1:00:05), and ahead
without numbers to implementing a transformer and its layer norm, "I think at some point you'll do this"
(≈1:00:51), and to "a transformer code base, which I think later in the class we actually do" (≈1:13:20).
Transformers are lecture 8 above. Nothing in it says whether the lecturer's lecture 23 continues its modular
theory.

Lecture 8's deck is titled "Lecture 8: Transformers", matching the recording, which settles lecture 1's
"Lecture 9: Transformers" banner (the row above), though its outline (slide 2) is still headed "9.
Transformers". It prints no other lecture number. Its recording points back to "the GNN lecture" (lecture 5,
≈7:43, ≈8:32, ≈1:04:18), to ResNets "in the CNN lecture" (lecture 4, ≈57:23) and to "what Jeremy was
talking about" on normalizing weight updates (lecture 7, ≈58:09). It points ahead without numbers to "the
problem set coming out next week" (≈40:15), Homework 3 below; to autoregressive and generative models,
which "we're going to come back to … a little later" (≈1:05:51, again at ≈1:13:49), lectures 14–16; and to
large language models, "We'll talk more about those later" (≈1:07:25), lecture 21. Multihead attention is
left to "the reading", the book chapter the lecture says it follows "almost exactly" (≈55:49, ≈1:10:38).

Lecture 9's deck is titled "Lecture 9: Hacker's guide to DL", and its outline (slide 2) is headed "9. Hacker's
guide to DL", both matching the recording. It prints no other lecture number; slide 23 points to "the generative
modeling lectures" without one, and slide 50's aside "Until Jeremy solves lr-free optimization" names Jeremy
Bernstein. The recording points back to "the generalization lecture" for the random-labels paper (lecture 6,
≈1:34), to the RMS norm that "Jeremy introduced … in one of his lectures" (lectures 3 and 7, ≈21:02) and to data
augmentation in "Pset 1" (≈30:17). It points ahead without numbers to generative models ("we'll talk about
generative models later", ≈7:48, and "the generative modeling lectures", ≈47:10), lectures 14–16; to pre-training
and transfer learning "a little bit later" (≈57:57), lectures 18–19; and to foundation models (≈50:16). It also
says the next problem set will clarify how to turn off Colab's autocomplete (≈1:08:50).

Lecture 10's deck is titled "Lecture 10: Memory and sequence modeling", matching the recording, which settles lecture 1's
"Lecture 11: RNNs" banner (the row above). But its outline (slide 2) and the repeat of it near the end (slide 67) are headed
"11. Memory and sequence modeling", and the last outline (slide 68) "9. Memory and sequence modeling"; cite the lecture as 10.
It prints no other lecture number. The recording opens by calling it "our last of this mini series" on architectures, after
"CNNs, GNNs, things like transformers" (≈0:00), and points back to adding time to CNNs "when we introduced CNNs" (lecture 4,
≈3:54), to backpropagation over shared parameters, "It's been a couple of weeks" (lecture 2, ≈25:39), to the spectral norm "a
couple lectures ago" (lecture 7, ≈31:06), to "the transformer lecture last week" (lecture 8, ≈47:22) and to "the hacker
lecture" (lecture 9, ≈48:55). It names no later lecture.

Lecture 11's deck is titled "Lecture 11: Representation Learning I", matching the recording and lecture 1's banner on its slide 73,
but its outline (slide 2) is headed "12. Representation Learning I"; cite the lecture as 11. It prints no other lecture number. The
recording opens the course's "next third", on representation learning and generative modeling (≈0:00), and points back to the problem
set 2 spectral-descent method (≈15:30), to the normalization picture "I showed in one of the last lectures" (lecture 9, ≈10:49), to "the
colorization problem I showed you before" (lecture 9, ≈1:06:50), to the vision transformer's tokens (lecture 8, ≈1:12:14) and to
autoregressive generation "which we talked about before" (lectures 8 and 10, ≈1:14:33). It points ahead to the next lecture on "metric
learning or contrastive learning" (lecture 12, ≈1:35; metrics "in the next lectures", ≈44:11), to generative modeling "in a few weeks"
(lectures 14–16, ≈2:23), to variational autoencoders "later" (≈48:06), to "a few lectures on transfer learning" (lectures 18–19, ≈27:54),
to linear probes and low-rank fine-tuning "in a future lecture" (≈36:27) and to CLIP "a little bit later in the course" (≈17:04), the
last three without numbers.

## Coursework and policies

**Grading** (≈2:20–3:54):

- **65% problem sets.** There are five, each "about one to two weeks long", combining
  pen-and-paper work ("or, more realistically, maybe Overleaf") with code to write and submit.
  OCW publishes all five ([sources](../sources.md)).
- **35% final project.** A research project "focused on deeper understanding of some of the
  topics that we cover", starting from a proposal. The deliverable is a **blog post** "that's going
  to demonstrate novel experimentation and visualization". The format is chosen because writing a
  blog-style account of research, without dropping its technical content, is now part of doing ML
  research and "a really valuable skill". Groups of at most two. Either an applied or a theoretical
  angle is fine: "What we want is for them to be innovative" (≈5:24).

**Compute is not provided** — at most "some very limited compute, but that's TBD" (≈3:54). So a
project should not depend on "crunching a huge architecture on a bunch of huge data sets". The
lecturer presents working creatively within that limit, rather than assuming "bigger is better",
as a skill worth having as large-scale deep learning becomes more centralized (≈4:39).

**PyTorch.** Two PyTorch tutorials were offered the week after lecture 1, for anyone not familiar with PyTorch or wanting a refresher
(≈6:57–7:43). Other frameworks are fine for the final project, but some problem sets come with
PyTorch code to fill in, so familiarity with PyTorch is strongly recommended "unless you want to
rewrite all of it in JAX or something". Lecture 2's announcements slide confirms the tutorials ran
that week, alongside the release of **problem set 1**, due 9/24, and the start of office hours
(slide 2). Lecture 3 says its depth-separation result is "also on the first problem set"
(≈1:08:51), and recommends a graphing website for building functions out of ReLUs, which "is quite
helpful on some of the homework problems" (≈32:41). Lecture 5 sends the class to "your problem set"
for the Bellman-Ford construction and for more pairs of graphs a graph net cannot tell apart
(≈42:40, ≈1:09:00); on OCW that is **Homework 2**, whose ten-point part on message-passing graph
neural networks recalls AGGREGATE, UPDATE and READOUT "from lecture". At the start of lecture 5 a
problem set, "a two-week problem set", was due the following Tuesday (≈3:06–3:52). Lecture 6, that
Tuesday, confirms it: "Problem set one is due today. Problem set two goes out today", posted after the
lecture as every problem set is (≈0:00–0:46). The lecturer recommends reading problem set 2's first
question before Thursday's lecture (lecture 7), "because it can help motivate … what we'll see on
Thursday"; on OCW, Homework 2's first question is "Steepest descent" (9 points). Lecture 7, that Thursday,
refers to "the homework" throughout: the steepest-descent solutions under the infinity norm and a general
norm, the bonus upper bound for a linear predictor, and the RMS-RMS normalization are all "on the homework"
(≈39:07, ≈41:27, ≈44:33, ≈1:02:24). Those are Homework 2's "Steepest descent" part and its
"Hyperparameter transfer" part (6 points), whose learning-rate question begins "We saw in lecture 7".
Lecture 8 announces that "the problem set coming out next week" will "implement a bunch of transformers and
all the variations. It'll implement GPT" (≈40:15). On OCW that is **Homework 3**, 35 points: "RNNs versus
transformers" (8 points), "Implementing a Transformer" (11), "Vision Transformers" (6) and "DialogueGPT"
(10), with the code for the last three in a Colab notebook. Its first problem is lecture 10's material: a simple RNN worked
by hand, the fate of the first input's contribution over a long sequence, and the cost of a forward pass against a
self-attention layer's ([lecture 10](10-architectures-memory.md#the-problem-set)). Lecture 11 says the class will
run its coloured-shapes autoencoder experiments "on your p set 3" (≈54:21); on OCW they are in **Homework 4**, whose second section,
"Reconstruction and Similarities in Representation Learning" (12 points), has an autoencoder question (6 points) and a contrastive-learning
question (6 points) on $64 \times 64$ images of coloured shapes ([lecture 11](11-representation-learning-reconstruction-based.md#the-problem-set)).
Its first section, on similarity-based learning (13 points), belongs with lecture 12. Homework 5 has a section on variational autoencoders (14 points) and one on diffusion models (17 points).

**Collaboration** (≈7:43–10:00). Discussing problems with peers, TAs and instructors is
allowed, but every submission — writeup *and* code — must be your own, written separately. Do not
copy or share complete solutions, and do not ask others, in person or on Piazza or Canvas,
whether your solution is right. Write the names of everyone you worked with, other than staff, at
the top of the problem set, whatever the size of the group. The reason given: "it matters more
that you've gone through that process independently than actually getting it correct", because
the feedback teaches you more on your own attempt. You may show your work to TAs, who "will work
with you to help you hopefully figure it out on your own" (≈11:34).

**AI assistants** (slide 4, ≈10:00–11:34). The policy is "*identical* to our policy for using
human assistants". Because it is a deep learning class you *should* try the latest assistants, and
learning what they can and cannot do is "part of the content of this course". Use them as you
would office hours — questions about lecture material, clarifications of problem-set questions,
tips for getting started — but not to do the work: "just like you are not allowed to ask an expert
friend to do your homework for you, you also should not ask an expert AI". When unsure, "imagine
the AI as a human and apply the same norm". If you use one on a problem set, say which and how at
the top, in "a few sentences". Lecture 9 restates the policy for coding assistants: "You shouldn't use AI
assistants in a way you wouldn't use a friend", so not on the problem sets to write code, but "for your final
projects and for life in general, you should absolutely use coding assistants" (slide 46, ≈1:08:50–1:09:36).

## Course materials on OCW

All are listed with canonical URLs in [`sources.md`](../sources.md):

- **Slide decks** for every lecture, `mit6_7960_f24_lecN.pdf` (no lecture 22).
- **Problem sets** 1–5, `mit6_7960_f24_hwN.pdf`, and a solution for problem set 5.
- **Math Notation** handout, `mit6_7950_f24_notation.pdf`, summarized on [notation](notation.md).
