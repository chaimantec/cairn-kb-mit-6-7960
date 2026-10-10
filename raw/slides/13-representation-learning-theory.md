---
title: Lecture 13 — Representation Learning: Theory (slide deck)
lecture: 13
slides: 28
source_pdf: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec13.pdf
note: Printed slide numbers 1–27 (bottom centre) equal the PDF page numbers. Page 28 is OCW's appended end page (a smaller page), which prints "28" in the same position but is not part of the lecture deck. The deck is handwritten; the PDF text layer is OCR of the handwriting and was not used, except for OCW's typed licence notices.
figure_audit: Transcribed by Sonnet from page images; 25 pages (1–11, 13–18, 20–27) were then checked by Opus, a different model, split between two agents, from 100–600 dpi crops, the PDF's vector ink (page.get_drawings()), the native rasters of slides 3, 4 and 22, and the captions. Every formula agreed symbol by symbol; the N of slides 18 and 25 is a calligraphic 𝒩, and the L of slides 20 and 25 a slanted capital L, not ℓ. Nine pages agreed (5, 6, 8, 11, 16, 21, 23, 24, 26; slide 6's approximate dot counts were made exact) and corrections were applied on the other 16. Two described things that are not on the page: slide 17's two diagrams are sketched scatter clouds, not nested contours, and slide 27's "faint specks" are slivers of three book icons painted over in the background colour, not empty entries. The rest were counts, colours and positions: four random curves, not five or six, on slides 10 and 13; 17 cells and 12 and 11 stems on slide 14; exact dot counts on slide 6; the colours of slide 9's formula; slide 3's baseline meeting the target line at about 42 minutes, not 45; slide 4's dot positions, monkey and notice placement; slide 22's input 2 (same framing, mild noise), its painted-out plot labels and its notice placement. Slide 1's typed "6.S898" is hidden under the pink blob. Slide 21's "f100" and slide 15's "radom" are as written; slide 27's "Procesies" was downgraded to an ambiguous stroke.
---

# Lecture 13 — Representation Learning: Theory: slide-by-slide

A slide-by-slide transcription of all 28 pages of
[`mit6_7960_f24_lec13.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec13.pdf),
the handwritten (iPad) deck by Jeremy Bernstein, whose title slide reads "Architectural Bias on Representations". Cite these as "slide N" — the printed number equals the PDF page number for slides 1–27; slide 28 is OCW's appended end page. The deck is handwritten prose, equations and hand-drawn diagrams in several ink colours, on a dark background or on white with a dark band at the right edge; the title and the section dividers are white with a coloured band at the right. All of it is read visually from the page images, and diagrams and plots are described in prose since the KB is read as text.

**Images.** 10 slides carry a whole-slide render under their heading: 3, 5, 6, 9, 10, 11, 13, 14, 17 and 20. Not rendered: the two slides with an OCW exclusion notice (4 and 22); slide 7, whose network drawing slide 20 repeats with its weight labels; slide 8, whose four data points slides 9 and 10 repeat with their fits; slide 18, whose plot repeats slide 13's last row; slide 25, whose network repeats slide 20's and whose formulas this file reproduces; the text and equation slides 2, 15, 16, 21, 23, 24 and 26; the title (1), the section dividers (12, 19), the references (27) and OCW's end page (28). Slide 3's pasted plot, from Keller Jordan's NanoGPT speedrun posts, carries no OCW notice and is not any of the images OCW excluded from lecture 7, so it is rendered.

Companion pages: [wiki page for this lecture](../../wiki/13-representation-learning-theory.md) ·
[transcript](../transcripts/13-representation-learning-theory.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1 | Title: Architectural Bias on Representations |
| 2–3 | Aside: steepest descent under the spectral norm (HW2) and its use in the Muon optimizer (NanoGPT speedruns) |
| 4–7 | Framing: similarity-based representation learning (lecture 12), perspectives on neural computation (lecture 7), a neural net as a map through vector spaces, this lecture's aim |
| 8–11 | A journey to the past: fitting data with bump functions (kernel methods) and with random functions (Gaussian processes); correspondences between function spaces |
| 12–18 | Gaussian processes: pictorially, informally, formally; covariance functions; conditioning on data |
| 19–25 | NN-GP correspondence: random weights give random functions; inspecting them (3-layer MLP, width 1000); the infinite-width theorem; proof sketch; ReLU MLP example (compositional arccosine kernel) |
| 26 | Natural questions |
| 27 | References |
| 28 | MIT OpenCourseWare end page |

---

## Slide 1 — Architectural Bias on Representations

Title page: white, with a pale-pink vertical band at the right. Handwritten black title sitting on a thick black horizontal rule that runs the full width of the page: "Architectural Bias on Representations". Below the rule, typed: "Jeremy Bernstein" (black) and, in dark grey monospace, "jbernstein@mit.edub" (printed as such [sic]; probably "jbernstein@mit.edu"). A black MIT logo at right, straddling the left edge of the pink band. At the bottom left, a handwritten "6.7960" (dark grey) on an opaque pink blob, then typed grey monospace ":: Lecture" and a handwritten "13" (dark grey). The blob is painted over a typed "6.S898", the course's earlier number, which is in the PDF's text layer but hidden on the page; no part of it shows. Slide number 1 at bottom centre.

## Slide 2 — Aside: Steepest Descent (HW2)

White page with a dark band at the right edge. Handwritten yellow heading: "Aside: Steepest Descent (HW2)".

In a green hand-drawn rectangle, yellow:

$$\underset{\Delta W}{\arg\min} \quad \mathrm{Tr}\left(G^{\top} \Delta W\right) + \frac{\lambda}{2} \lVert \Delta W \rVert_{\ast}^{2}$$

The subscript on the norm is a hand-drawn asterisk $\ast$ (two crossing diagonals and a curved crossbar); a blue arrow runs from the blue words "spectral norm" (two lines, at the right edge) up to it, so it denotes the spectral norm, as Homework 2 writes it ($\Vert \cdot \Vert_ \ast$).

Yellow underlined "Solution", then yellow "for $G = U \Sigma V^{\top}$"; a blue arrow from the blue words "reduced SVD" (two lines) points at $U \Sigma V^{\top}$. Below, circled by a large lilac oval:

$$\Delta W = - \frac{\mathrm{Tr}(\Sigma)}{\lambda} U V^{\top}$$

Beneath the oval, in lilac with quote marks: "steepest descent under the spectral norm".

## Slide 3 — Neural Network Speedrunning

![Slide 3 — Neural Network Speedrunning](../images/13-representation-learning-theory/slide-3.jpg)

Dark background. Handwritten yellow heading "Neural Network Speedrunning", and at the upper right, blue with the handle underlined: "@kellerjordan0" (the last character is a plain circle, which shape alone cannot tell from a letter O; the captions spell the handle "kellerjordan0" at ≈1:39).

Within a lilac oval, yellow:

$$\Delta W = - \frac{\mathrm{Tr}(\Sigma)}{\lambda} U V^{\top}$$

Below it, lilac: "steepest descent under the spectral norm" (in quote marks). Yellow: "Add some tricks :" followed by three yellow lines: "momentum", "low precision", "fast computation of $U V^{\top}$ via iteration".

Lower left, a pasted typed plot (a matplotlib chart on a white background), with the slide number 3 just to the right of its lower right corner. Title "NanoGPT speedruns". Vertical axis "Fineweb val loss", ticks 3.3 to 3.8 in steps of 0.1. Horizontal axis "Wallclock time (minutes on 8xH100)", ticks 0, 10, 20, 30, 40, running to a little past 45. Legend (top right) with three series:

- Blue: "llm.c baseline", "140ms/step", "10.25B tokens". Starts above the top of the plot, descends smoothly and slowly, meets the dashed line at about 42 minutes (it is at about 3.281 at 40 minutes) and runs along or just under it to its end at about 46 minutes.
- Orange: "10/14/24 record", "179ms/step", "2.67B tokens". Descends faster, has a small kink near 3.41 at about 11 minutes, and reaches the dashed line at about 15 minutes.
- Green: "+Distributed Muon", "154ms/step", "2.67B tokens". Descends fastest, a similar kink at about 9.5 minutes, and reaches the dashed line at about 13 minutes.

A grey dashed horizontal line at about 3.28 (measured 3.276) marks the target loss. Three series.

To the right, a green arrow from the green words (quote marks): "Muon optimizer" (underlined) "trains nanoGPT to 3.28 val loss on "Fineweb" dataset in < 15 minutes" points at the plot. No OCW notice or credit is printed on this slide.

## Slide 4 — Similarity-Based Representation Learning (Lecture 12)

Dark background. Handwritten yellow heading "Similarity-Based Representation Learning (Lecture 12)" (the "(Lecture 12)" is as printed; see the course map for the deck's lecture pointers), and a yellow line: "training objective that maps similar data to nearby embeddings" (the "a" of "nearby" is left open, so it reads "necrby").

On the left, green capitals "DATA SPACE" above three photographs stacked vertically: a golden retriever puppy in orange flowers facing right, a second crop of a similar puppy photo (the puppy shifted right and cut off at the edge), and a monkey (a grey-faced guenon-like monkey with amber eyes and a white chin) looking at the camera. On the right, pink capitals "EMBEDDING SPACE" above a large tilted pink-outlined ellipse filled with muted pink. Three smooth lines go from the right of each photograph into the ellipse, ending in a filled dot: from the first and second puppy photos, green lines ending in two green dots close together near the upper part of the ellipse (the first dot slightly higher and to the right of the second); from the monkey photo, a lilac line ending in a lilac dot lower in the ellipse, well apart from the green ones. The figure shows similar images mapped to nearby embeddings and a dissimilar one to a distant embedding.

*OCW notice: Dog © source unknown. Monkey © San Diego Zoo. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/*

The notice is typed in white serif in two lines at the bottom, to the right of the monkey photograph; it names both the dog and the monkey photographs. The slide number 4 sits under its second line, invisible on the dark ground.

## Slide 5 — Perspectives on Neural Computation (Lecture 7)

![Slide 5 — Perspectives on Neural Computation (Lecture 7)](../images/13-representation-learning-theory/slide-5.png)

White page with a dark band at the right edge (the "(Lecture 7)" in the heading runs into the band). Handwritten yellow heading "Perspectives on Neural Computation (Lecture 7)". Three yellow numbered rows in capitals:

- "#1 NEURAL PERSPECTIVE": at right, a small network drawing: green nodes joined by magenta lines — a left column of 2 nodes, a middle column of 4, a right column of 1 node; every left node joins every middle node and every middle node joins the right node.
- "#2 TENSOR PERSPECTIVE": at right, a green rectangle holding a $2 \times 4$ grid of 8 magenta dots, then green "relu", then a tall narrow green rectangle holding a column of 4 magenta dots.
- "#3 SPECTRAL PERSPECTIVE": below, a row of green boxes with magenta dots: a small square box with a $2 \times 2$ block of 4 dots; a second square box with 2 dots (top left and bottom right); a wide box with a $2 \times 4$ grid of 8 dots; then green "relu"; then a tall box with a column of 4 dots; then two small square boxes with 1 dot each.

## Slide 6 — A More Abstract Perspective

![Slide 6 — A More Abstract Perspective](../images/13-representation-learning-theory/slide-6.png)

Dark background. Handwritten yellow heading "A More Abstract Perspective". Yellow: "A neural net is a map through a sequence of vector spaces."

A row of four green-outlined parallelograms (slanted planes, filled dark green), each holding blue and red dots, separated by pink curved arrows pointing right (four arrows, one above the gap after each of the first three planes and one after the green "…" before the last plane), and between the third and last planes three small green dots ("…"). Labels in yellow capitals under the planes: "INPUT SPACE", "FIRST HIDDEN SPACE", "SECOND HIDDEN SPACE", "LAST HIDDEN SPACE".

- Input space: 5 blue and 4 red dots, thoroughly mixed.
- First hidden space: 4 blue and 4 red dots, still mixed.
- Second hidden space: 5 blue and 4 red dots, beginning to cluster (blue drifting to the upper left, red to the right and lower).
- Last hidden space: 4 blue dots in a $2 \times 2$ cluster at top left and 4 red dots in a cluster at bottom right, cleanly separated.

Yellow at the bottom: "Want a good representation of the data at the final layer" and, indented, "e.g. linearly separable". The figure shows the dots becoming separable layer by layer.

## Slide 7 — This Lecture

White page with a dark band at the right edge. Handwritten yellow heading "This Lecture". Yellow: "We will develop an advanced tool that shows that a neural architecture (even without training) already expresses an opinion about data similarity." (the text runs into the band).

Below, a hand-drawn fully connected network: green nodes in seven columns with 2, 4, 4, 4, 4, 4 and 1 nodes, joined by magenta lines between every pair of adjacent columns, each pair completely (76 edges; the last column's single node on the edge of the dark band).

## Slide 8 — A Journey to the Past

Dark background. Handwritten yellow heading "A Journey to the Past". Yellow: "Pretend you've never heard of deep learning, and you want to fit some data." A pair of lilac axes (vertical with an open arrowhead at top, horizontal with an open arrowhead at right, crossing near the lower left) with four red crosses: three rising to the right in the upper left (low, middle, high), and one low on the right. Yellow at the bottom left: "How would you do it?"

## Slide 9 — Function Space Construction #1

![Slide 9 — Function Space Construction #1](../images/13-representation-learning-theory/slide-9.png)

Dark background. Handwritten yellow heading "Function Space Construction #1". Yellow: "Place a "bump function" on each datapoint:".

A small plot: lilac axes; green bumps, four of them, each topped by a pink cross, with pink labels $x_1$, $x_2$, $x_3$, $x_4$ under the axis at their centres. Bump heights: $x_1$ low, $x_2$ medium, $x_3$ tallest, $x_4$ low-medium; the $x_4$ bump is further right with a gap before it.

Yellow: "Formally, we consider functions of the form" and

$$f(x) = \sum_{i=1}^{n} \alpha_i \thinspace k(x, x_i)$$

(written in green, with only $\alpha_i$ and the second-argument $x_i$ in pink); a yellow arrow from the yellow words, in two lines, k is called the "kernel" (only "kernel" in quote marks) points at $k(x, x_i)$.

Yellow "where:" with two bullets: $k(x, x_i)$ is a bump (followed by a small green drawn bump, a top-hat arch with flared feet) "centred on $x_i$" ($k(x, \cdot)$ green, $x_i$ pink) and "the $\alpha_i$ are weights" ($\alpha_i$ pink, "weights" plain yellow).

Yellow at the bottom: "The freedom to choose the number of bumps $n$, the bump centres $x_i$ and the weights $\alpha_i$ leads to a rich function space called a "reproducing kernel Hilbert space"." (the symbols $n$, $x_i$, $\alpha_i$ in pink). The slide number 9 is overlaid by the last text line.

## Slide 10 — Function Space Construction #2

![Slide 10 — Function Space Construction #2](../images/13-representation-learning-theory/slide-10.png)

Dark background. Handwritten yellow heading "Function Space Construction #2". Yellow: "Draw random functions consistent with the data".

A plot with lilac axes, four pink crosses at $x_1$ to $x_4$ (pink labels), and four wiggly random curves (yellow, green, dark blue and red-orange) that all pass through the four crosses and wander between and beyond them.

Yellow: "We will look at a special way of drawing random functions that uses Gaussian random variables" and, after an arrow, "it is called a Gaussian process (GP)".

## Slide 11 — Correspondences between Function Spaces

![Slide 11 — Correspondences between Function Spaces](../images/13-representation-learning-theory/slide-11.png)

White page with a dark band at the right edge. Handwritten yellow heading "Correspondences between Function Spaces". Three filled ellipses with yellow capitals: a purple ellipse at upper left, "GAUSSIAN PROCESSES"; a brown-red ellipse with an orange outline at right (overlapping the band), "KERNEL METHODS"; a blue ellipse at lower left, "NEURAL NETWORKS". Three thick yellow arrows: from Gaussian processes curving down to kernel methods (arrowhead at kernel methods); from kernel methods curving up to Gaussian processes (arrowhead at Gaussian processes); and from neural networks up to Gaussian processes, labelled in yellow at its left "infinite width" and to its right "random weights" (arrowhead at Gaussian processes).

## Slide 12 — Gaussian Processes

Section divider: white with a pale-blue band at the right. Handwritten black capitals "GAUSSIAN PROCESSES" sitting on a thick black horizontal rule across the page.

## Slide 13 — What is a Gaussian Process Pictorially?

![Slide 13 — What is a Gaussian Process Pictorially?](../images/13-representation-learning-theory/slide-13.png)

Dark background. Handwritten yellow heading "What is a Gaussian Process Pictorially?". Three rows, each with a text at left and a plot at right (green axes with open arrowheads at top and right):

- "Given data": 3 pink crosses in the upper area: two close together at the left, rising, and one lower on the right.
- "A Gaussian process gives us a distribution of consistent functions": the same three pink crosses, with four wiggly random curves (red-orange, green, lilac and dark blue) through all three crosses.
- "Along with a formula for the" "mean" (orange) "and" "standard deviation" (blue) "of this distribution": the three crosses with one smooth orange curve through all three (the mean, rising to a peak at the middle cross then falling) and two dark-blue curves (standard deviation) that bulge away from the orange curve between the crosses and pinch together at each cross, one above and one below, and diverge beyond the outer crosses. One orange series and two blue.

## Slide 14 — What is a Gaussian Process Informally?

![Slide 14 — What is a Gaussian Process Informally?](../images/13-representation-learning-theory/slide-14.png)

Dark background. Handwritten yellow heading "What is a Gaussian Process Informally?". Yellow: "Sample a Gaussian vector :" followed by a green segmented bar (a row of 17 narrow cells) and yellow $\sim N(0, \Sigma)$, with a lilac arrow from the lilac word "mean" to the $0$ and a lilac arrow from the lilac word "covariance" to the $\Sigma$.

Yellow: "We can construct a function by plotting the components of this vector". Plot: a green horizontal axis with an arrowhead at the right, 12 lilac vertical stems of random heights above and below it (up, down, up, up, down, up, down, down, up, down, up, up), and a pink zigzag line joining their tops: jagged, with 8 sign changes in 11 gaps.

Yellow: "Idea: set up $\Sigma$ such that consecutive components are correlated" and, at right in smaller yellow, "e.g. $\Sigma_{ij} \sim \exp - (i-j)^2$" (the minus sign applies to $(i-j)^2$ as written; no brackets around the exponent). Second plot: the same green axis with 11 lilac stems, and a pink line joining the tops, but now smooth: a hump above the axis (4 stems), then a trough below (4), then a hump above again (3). Yellow at the bottom: "This leads to more "continuous looking" functions."

## Slide 15 — What is a Gaussian Process Formally?

Dark background. Handwritten yellow heading "What is a Gaussian Process Formally?". Yellow: "Consider an input space $X$" (a plain upright capital $X$, the same in all four places). "Let $f(x)$ be a random variable for every $x \in X$". Beneath it, indented, in blue: "Informally, think of $f$ as an infinite-dimensional random vector indexed by $x \in X$."

Yellow: "If for every finite collection of inputs" and, at right, $x_1, x_2, \ldots, x_n$. "the associated finite-dimensional radom vector" ("radom" [sic], for "random"). In a green rectangle divided into four cells: $f(x_1)$, $f(x_2)$, "· · · · ·", $f(x_n)$; then yellow "is Gaussian". Then "then $f$ is a "Gaussian process" on $X$."

## Slide 16 — Covariance Functions

Dark background. Handwritten yellow heading "Covariance Functions". Yellow: "A Gaussian process generalises". Two rows, each with a left phrase, a yellow arrow and a right phrase:

- "finite-dimensional Gaussian vectors" (two lines) $\longrightarrow$ "infinite-dimensional functions" (two lines).
- "The covariance matrix" and $\Sigma_{ij}$ for $i, j = 1, \ldots, n$ $\longrightarrow$ "covariance function" and $\Sigma(x, x')$ for $x, x' \in X$.

## Slide 17 — Covariance Functions

![Slide 17 — Covariance Functions](../images/13-representation-learning-theory/slide-17.png)

Dark background. Handwritten yellow heading "Covariance Functions". Yellow: "Typically, we want a covariance function that is large for "nearby" points and small otherwise".

Two small scatter diagrams side by side, each with yellow axes (open arrowheads) labelled $f(x')$ on the vertical axis and $f(x)$ on the horizontal axis:

- Left, with lilac text $x$ and $x'$ and "nearby": a sketched scatter of short green dashes and dots in a thin band along the diagonal from lower left to upper right through the origin, densest at the origin: the two function values are strongly correlated.
- Right, with lilac text $x$ and $x'$ and "distant": a sketched round scatter of short green dashes and dots centred on the origin (the outer dashes loosely ring-shaped), densest at the centre: the two values are uncorrelated.

Neither diagram has closed contour lines; both are scatter clouds of sampled pairs $(f(x), f(x'))$, as the lecturer describes them (≈37:13).

Yellow: after an arrow, "choice of covariance function encodes what we mean by "nearby"". Then two examples: "e.g. squared exponential" with $\Sigma(x, x') = e^{-(x - x')^2}$ (the exponent written as a superscript to $e$), and "inner product" with $\Sigma(x, x') = \langle x, x' \rangle$.

## Slide 18 — Conditioning on Data

Dark background. Handwritten yellow heading "Conditioning on Data". Yellow: "Suppose we are given $f(x_1), \ldots, f(x_n)$" and "And we want to predict $f(x_{\ast})$" (the subscript is a star-like asterisk). "Since $f$ is a Gaussian process, then this vector is Gaussian". In a green rectangle of five cells: $f(x_1)$, $f(x_2)$, "· · ·", $f(x_n)$, $f(x_{\ast})$.

Yellow: "Can prove: $f(x_{\ast})$ given $f(x_1), \ldots, f(x_n)$ is $\mathcal{N}(\mu, \sigma^2)$" (a calligraphic $\mathcal{N}$ with a curled left stroke). The star subscript is a six-armed asterisk in every $x_{\ast}$.

Yellow: "The mean $\mu(x_{\ast})$ and standard deviation $\sigma(x_{\ast})$ have simple closed form formulae." At the lower right, a small plot with green axes (open arrowheads), three pink crosses, one smooth orange curve through them (the mean), and two blue curves (the standard deviation band) that bulge apart between the crosses and meet at each cross: the same picture as the third plot of slide 13 (one orange series, two blue).

## Slide 19 — NN-GP Correspondence

Section divider: white with a pale-green band at the right. Handwritten black capitals "NN-GP CORRESPONDENCE" sitting on a thick black horizontal rule across the page.

## Slide 20 — Random Weights ⇒ Random Functions

![Slide 20 — Random Weights ⇒ Random Functions](../images/13-representation-learning-theory/slide-20.png)

White page with a dark band at the right edge. Handwritten yellow heading "Random Weights $\Rightarrow$ Random Functions". Yellow: "Consider a neural network:". A hand-drawn fully connected network: green nodes in seven columns with 2, 4, 4, 4, 4, 4 and 1 nodes, joined by magenta lines between every pair of adjacent columns. Under the layers, blue labels: $W_1$ under the first gap, $W_2$ under the second, $W_3$ under the third, then a blue dashed line (dash-dot, running under the middle layers) and $W_L$ under the last gap (a slanted capital $L$, with no loop; the lecturer says "W1, W2, W3, up to WL", ≈45:07). The dash-dot line runs under the fourth and fifth gaps.

Yellow at the bottom: "If we randomly sample the weights," and, on the next line, indented, "we get a random function." (the last word runs into the dark band).

## Slide 21 — Inspecting the Random Functions

Dark background. Handwritten yellow heading "Inspecting the Random Functions". Yellow: "To inspect the distribution of random functions"; "Pick two inputs $x$ and $x'$"; "Sample 1000 random networks $f_1, \ldots, f_{100}$" (written "f100" [sic]: probably $f_{1000}$, as the matrix below ends with $f_{1000}$); "Plot scatterplot of" the matrix, in large yellow brackets:

$$\begin{bmatrix} f_1(x_1), & f_1(x_2) \cr f_2(x_1), & f_2(x_2) \cr & \vdots \cr f_{1000}(x_1), & f_{1000}(x_2) \end{bmatrix}$$

(the middle is a column of dots, not a symbol). Note the text says the two inputs are $x$ and $x'$ while the matrix names them $x_1$ and $x_2$.

## Slide 22 — 3 Layer MLP, Width 1000

Dark background. Handwritten yellow heading "3 Layer MLP, Width 1000" (the L of "MLP" is drawn like the L of "Layer"; the PDF text layer misreads it as "MCP").

Three small pasted photographs (low-resolution, $32 \times 32$-like) of a white and tan lorry (truck), labelled in yellow: "input 1" at left (the lorry), "input 2" at top centre (the same lorry photograph, identically framed, with mild per-pixel colour noise added), "input 3" at top right (a noisy version of the lorry: a heavily pixel-noised image in random bright colours with the lorry faintly visible). Two pasted white scatter plots (matplotlib), blue semi-transparent dots (1000 per plot, one per sampled network), both with axes running from $-3$ to $3$ and ticks $-2$, $0$, $2$ on each axis:

- Left plot: yellow label "output 1" on the vertical axis (at left) and "output 2" under the horizontal axis. The dots lie in a very thin straight band along the diagonal from about $(-2.7, -2.6)$ to about $(2.6, 2.6)$ — almost a perfect line: 98% of the dots lie within about 0.2 of the diagonal.
- Right plot: yellow label "output 1" on the vertical axis and "output 3" under the horizontal axis. The dots form a broader tilted elliptical cloud along the same diagonal, from about $(-2.7, -2.6)$ to about $(2.9, 2.9)$, noticeably wider than the left one: 98% of the dots lie within about 0.7 of the diagonal, and none further than about 0.95.

The plots' own printed axis labels, $f(x_1)$ and $f(x_2)$ on both plots, are painted over in background colour and replaced by the handwritten "output" labels. Yellow: "Observations:" with two bullets: "joint distribution of pairs of outputs seems Gaussian" and "covariance depends on similarity of inputs".

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/*

The notice is typed in white serif at the very bottom left of the page, below the "Observations" bullets. It does not say which images it covers. The photographs are CIFAR-10 images (≈47:28); the scatter plots are from the lecturer's own run ("I actually did this", ≈47:28). The whole slide is treated as excluded and is not rendered.

## Slide 23 — Neural Network - Gaussian Process Correspondence

Dark background. Handwritten yellow heading "Neural Network - Gaussian Process Correspondence". Yellow: "If we sample iid the weights of an NN"; "then as the width $\longrightarrow \infty$"; "The joint distribution of any finite collection of network outputs" followed in lilac by $f(x_1)$   $f(x_2)$   $\ldots$   $f(x_n)$ and then "is Gaussian." Last, yellow: "The covariance function depends on the architecture and non-linearity."

## Slide 24 — Proof Sketch

Dark background. Handwritten yellow heading "Proof Sketch". Yellow: "Main tool: multivariate central limit theorem". Yellow underlined "Step 1:" "for a fixed input and fixed layer, the activations are iid random variables"; then, in blue after an arrow: "prove by induction on depth using MV-CLT". Yellow underlined "Step 2:" "for any collection of $k$ inputs $x_1, \ldots, x_k$ the network outputs $f(x_1), \ldots, f(x_k)$ are Gaussian"; then, in blue after an arrow: "prove by writing the vector $f(x_1), \ldots, f(x_k)$ as a sum over iid vectors from the penultimate layer and apply the MV-CLT."

## Slide 25 — Example: MLPs with ReLU

Dark background. Handwritten yellow heading "Example: MLPs with ReLU". The same seven-column network drawing as slide 20 (green nodes in columns of 2, 4, 4, 4, 4, 4 and 1, magenta edges between adjacent columns) with blue labels $W_1$, $W_2$, $W_3$, a blue dashed line and $W_L$ below.

Yellow, in two rows of text with the formulas at right: "set non-linearity to" with $\phi(x) = \sqrt{2} \thinspace \mathrm{relu}(x)$; "sample weights iid" with $\mathcal{N}\left(0, \frac{1}{\text{fan-in}}\right)$ (the same calligraphic $\mathcal{N}$ as slide 18). Then "For inputs $x, x' \in \mathbb{R}^d$:" with

$$\mathbb{E} f(x) = 0$$

$$\mathbb{E} f(x) f(x') = h \circ \cdots \circ h \left( \frac{x^{\top} x'}{d} \right)$$

where a brace under $h \circ \cdots \circ h$ is labelled $L - 1$ times (a slanted capital $L$, the same letter as the $W_L$ subscript; the lecturer says "We apply h L minus 1 times", ≈1:09:28). Then "where $h(t) = \frac{1}{\pi} \left[ \sqrt{1 - t^2} + t (\pi - \arccos t) \right]$" and, in green at the lower right, in quote marks: "compositional arccosine kernel". The slide number 25 is overlaid on the formula line.

## Slide 26 — Natural Questions

Dark background. Handwritten yellow heading "Natural Questions". Yellow: "How does $\Sigma(x, x')$ depend on :" with two bullets: "choice of architecture ?" and "choice of weight distribution?". Then "Can this inform:" with two bullets: "architecture design?" and "weight regularisation strategies?".

## Slide 27 — References

Reference page: pale lilac background over the whole page, with a typed black bold heading "References". Three entries, each beside a small book or notebook icon (brown book with a red bookmark; orange notebook; blue-grey notebook), with the title handwritten in black and the authors beneath it in grey handwriting:

1. "Bayesian Learning for Neural Networks" — "Neal"
2. "Kernel Methods for Deep Learning" — "Cho & Saul"
3. "Deep Neural Networks as Gaussian Processes" — "Lee, Bahri, Novak et al." (the second "s" of "Processes" is written as a bare slanted stroke with no dot, so it can be read as an undotted "i", "Procesies"; the "i" of "Gaussian" in the same title is dotted)

Three more book icons lower in the column (grey, green and blue) are painted over in the lilac background colour; a few slivers of them show as faint specks. There are no further entries. No credit lines.

## Slide 28 — MIT OpenCourseWare end page

OCW's appended end page, not lecture content: a smaller white page (landscape, 792 × 612) with typed text: "MIT OpenCourseWare", "https://ocw.mit.edu", "6.7960 Deep Learning", "Fall 2024", "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms". It prints the number "28" at bottom centre.
