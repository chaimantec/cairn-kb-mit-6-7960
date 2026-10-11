---
title: Lecture 15 — Deep Generative Models II (slide deck)
lecture: 15
slides: 42
source_pdf: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec15.pdf
note: Printed slide numbers 1–41 (bottom centre) equal the PDF page numbers exactly. Page 42 is OCW's appended end page (a smaller page), which also prints 42, although slide_number_map.py reports it as printing no number.
figure_audit: Transcribed by Sonnet from page images, in two halves; 39 figure-, diagram-, chart-, table- and equation-heavy pages (1, 3–32 and 34–41) were then checked by Opus, a different model, split between three agents (all finished), from 110–600 dpi crops, the embedded rasters at native resolution, the vector data and the text layer, and slides 16 and 40 against their rendered images. No equation was wrong; slide 41's table agreed cell by cell. 21 pages were corrected and three (29, 37, 41) given small precisions. The corrections were counts (slide 28's 15 circles; slide 31's 27 dots in 8 rows; slide 32's 16 and 27 dots and its four separate images; slide 40's 26 square markers), positions and arrowheads (slide 5's and 23's lines without heads; slide 6's ⇓; slide 19's arrow ending at the circle's centre; slide 22's brace; slide 27's trapezoid, which widens the other way; slide 28's x and μ_z), the placement of slides 6's and 39's notices, colours (slide 38's backgrounds), and two things described that are not on the page: a lake behind slide 6's pelican, and grey dashed rims on slides 8 and 9's circles, which are the target outline showing through. The bird of slides 1 and 39 had been given a species name that neither the slide nor the lecturer uses.
---

# Lecture 15 — Deep Generative Models II: slide-by-slide

Text and figures of all 42 pages of
[`mit6_7960_f24_lec15.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec15.pdf),
transcribed from the deck (speaker: Phillip Isola; the title slide reads "Lecture 15: Deep Generative Models II", and the
recording's title is "Generative Models: Representation Learning Meets Generative Modeling"). Cite these as "slide N" — the printed
number equals the PDF page number for slides 1–41; page 42 is OCW's appended end page and prints 42. Diagrams, plots and pictures
are described in prose since the KB is read as text. Most equations are typeset text (Computer Modern) and a few are LaTeXiT
images; all were read from the page, never from the text layer.

**Images.** 22 slides carry a whole-slide render under their heading: 3, 4, 5, 7, 8, 9, 10, 11, 14, 16, 19, 22, 26, 27, 28, 29, 30, 31, 32, 36, 38 and 40. Not rendered: the 3 slides with an OCW exclusion notice (1, 6 and 39); build steps or repeats that a rendered slide supersedes (15, 18, 20, 21, 35 and 37); equations, text and the table, which this file reproduces (2, 12, 13, 17, 23, 24, 25, 33, 34 and 41; slides 24 and 25 also draw small copies of slides 22's and 27's diagrams); and the end page, 42.

Companion pages: [wiki page for this lecture](../../wiki/15-generative-models-representation-learning-meets-generative-modeling.md) · [transcript](../transcripts/15-generative-models-representation-learning-meets-generative-modeling.md)

**Signposting slides you can skip.** Slide 1 is the title; slide 2 is the outline (generative modeling as inverse representation
learning; variational autoencoders; disentanglement); slide 33 lists the three tricks; slide 42 is the OCW end page. The deck prints
no lecture number but its own.

Some slides are **build steps** or repeats, transcribed individually with a note of what they add: slides 15 and 16 (random sampling,
then the one useful sample shaded and the sum written out); 18 and 19 (importance sampling, then set beside random sampling); 20 to 22
(the posterior's picture, an encoder pointing at it, then the encoder's Gaussian); 26 and 27 (encoding and sampling, then decoding); 29
and 30 (autoencoder, then variational autoencoder); 35, which repeats slide 10's generator; and 37 to 39 (one latent direction, two, then
BigGAN's). Slides with no printed title are headed here with a description: 5 to 7, 31, 32, 35 to 38 and 40.

**Printed slips and oddities**, kept as printed: slide 11's reparameterization line writes a plain $x$ and $g^{\sigma}_ {\theta}$ where the
equation above it has bold $\mathbf{x}$ and the covariance $g^{\Sigma}_ {\theta}$ (the lecturer confirms that the noise is scaled by the
standard deviation, ≈25:49); slide 13's sum prints its index as $i = 1$ while its summand and samples use $j$ (slide 14 prints $j = 1$); slide
25's last line has $\theta^{\ast}, \psi^{\ast}$ on the left but $\arg\max_{\theta, \phi}$ and $J(\mathbf{x}^{(i)}, \theta, \phi)$ on the right,
$\phi$ for the encoder's $\psi$ (Homework 5 names the encoder's parameters $\phi$); slide 41's footnote asterisk has no anchor. In the
recording the lecturer says of slide 14 "I think there's a typo here. So that 1 over M should be inside the log, I suppose" (≈31:18), but
the slide already prints the $\frac{1}{M}$ inside the log. Slide 28 switches to plain $x$ and $z$ on purpose, labelled "(eqns for 1D x and z)".
Hidden, not visible on the page: slide 10's resources draw slide 6's excluded photographs entirely off the page; its river image is a 4 × 4
grid clipped to the 3 × 3 shown (slide 35 draws the same image); slides 16 and 34 hold images placed outside the page; and slides 3, 10 and 35
keep an invisible white "z" or "c" in the text layer.

**OCW licence notices** are on slides 1, 6 and 39, all "© Torralba, Isola, and Freeman. All rights reserved. …", all typed and in the text
layer: slide 1's and slide 39's cover the same embedded image, the grid of BigGAN images, and slide 6's covers its five bird photographs. A
scan for notices drawn as glyph outlines found none, and the audit found no missed notice or credit. Each is transcribed as an
`*OCW notice: …*` line with the picture it covers.

## Contents

| Slides | Section |
| ------ | ------- |
| 1–3 | Title, outline; generative modeling against representation learning |
| 4–7 | Distribution transformers; encoder and generator; an autoencoder's decoder as a generative model |
| 8–12 | Mixture of Gaussians; the infinite mixture; latent variable models; the VAE as a learner; learning its parameters |
| 13–14 | Trick #1: estimate the integral by Monte Carlo sampling |
| 15–19 | Trick #2: importance sampling |
| 20–28 | Trick #3: an encoder predicts the sampling distribution; putting everything together; the ELBO; "Looks like an autoencoder!" |
| 29–34 | Autoencoder against VAE; training pictures; the three tricks; training a VAE step by step |
| 35–40 | Disentanglement: the river generator, training and sampling, latent directions, BigGAN; the coloured map |
| 41 | Comparing popular model types |
| 42 | OCW end page |

## Slide 1 — Lecture 15: Deep Generative Models II

*OCW notice: © Torralba, Isola, and Freeman. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower middle of the slide, immediately left of the grid's bottom row; the grid is the slide's only picture, so the notice covers it).*

Title: "Lecture 15: Deep Generative Models II" (the title sits on a white band that overlaps the top of the picture grid). Subtitle: "Speaker: Phillip Isola".

At the right (about the right half of the slide), a grid of images of the same kind of small bird on a branch, 5 columns by 5 rows (25 tiles; the top row's first four tiles are covered by the title band, with faded strips of them showing above it (the tops of the birds' heads) and just below it; the fifth tile, at the top right, is fully visible with a bird on a branch). The slide does not name the bird or say where the images come from; they are the BigGAN images of slide 39, the same embedded image. Each tile shows the same kind of bird (black cap and bib, white cheek, grey back, pale belly) in a different pose: in the leftmost column it perches on a diagonal twig facing left with the tail pointing down-right; in the second column it is similar but slightly turned; in the middle column it faces the camera head-on on a horizontal branch; in the fourth and fifth columns it is seen in three-quarter view facing right on a horizontal branch. Background colours vary from tile to tile (olive, brown, green). Moving down the rows, the backgrounds shift gradually from brownish to greener. The tiles are separated by thin dark lines. The grid is a smooth variation in pose across the columns and in background across the rows.

Footer bar (grey): MIT logo, "6.7960 Deep Learning" (red), "https://phillipi.github.io/6.7960" (underlined), right side "Fall 2024". The printed slide number "1" sits just below the end of the URL.

## Slide 2 — Deep generative models II

Title: "Deep generative models II". Three bullets:

- Generative modeling as inverse representation learning
- Variational autoencoders
- Disentanglement

## Slide 3 — Generative modeling vs Representation learning

![Slide 3 — Generative modeling vs Representation learning](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-3.jpg)

Title: "Generative modeling vs Representation learning".

Left, two blocks of text:

- **Representation learning**: mapping data to abstract representations (analysis)
- **Generative modeling**: mapping abstract representations to data (*synthesis*; the word "synthesis" is in italics)

Right, the same figure as slide 2 of the previous lecture: a tall isometric 3D box (outline only), labelled "Embedding" above its top and "Data" below its bottom, with three light-grey translucent diamond-shaped sheets stacked inside and five thin dotted curved lines, each starting at a dot near the bottom and ending in an arrowhead near the top. A black vertical arrow on the left of the box points up, with the rotated label "Representation learning". A black vertical arrow on the right of the box points down, with the rotated label "Generative modeling". (No yellow highlight on this slide.)

## Slide 4 — Deep generative models are distribution transformers

![Slide 4 — Deep generative models are distribution transformers](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-4.png)

Title: "Deep generative models are distribution transformers".

Two shapes on a white background, left to right. Left: the heading "Prior distribution" above a plain salmon-orange filled circle with a thin black outline, with $p(z)$ below it. Middle: a thick black arrow pointing right, labelled $g$ above it. Right: the heading "Target distribution" above a pale-blue filled blob with a thin black outline, shaped like a "Y" with rounded ends (two arms at the upper left and upper right, the right arm taller and larger, and a stem at the bottom), with $p(x)$ below it. (The arrow stands for the generator $g$ turning the simple prior into the target.)

## Slide 5 — (no title; the encoder and the generator drawn as maps between data space and representation space)

![Slide 5 — (no title; the encoder and the generator drawn as maps between data space and representation space)](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-5.jpg)

Two diagrams, one above the other, staggered (the first at the upper left, the second at the lower right).

Upper diagram, with the underlined heading "Representation learning": at the left, the pale-blue "Y"-shaped blob (as on slide 4) labelled "Data space" (a dotted curved arrow from that label points at the blob's left arm), with $p_{\texttt{data}}$ written inside it and a point $\mathbf{x}$ inside near the upper right arm. A thick black curved line (no arrowhead) runs from $\mathbf{x}$ into the left (wide) end of a grey trapezoid labelled $f$ (wide at the left, narrow at the right; a dotted arrow from the label "Encoder" below points up at it). A second curved arrow leaves the trapezoid and ends with an arrowhead at the label $\mathbf{z}$ inside a salmon-orange circle at the right, labelled "Representation space" (dotted arrow from the label pointing at the circle).

Lower diagram, with the underlined heading "Generative modeling": at the left, a salmon-orange circle labelled "Representation space" (dotted arrow), with $p_{\mathbf{z}}$ written inside it and a point $\mathbf{z}$ at its lower right. A thick black curved line (no arrowhead) runs from $\mathbf{z}$ into a grey trapezoid labelled $g$ (narrow at the left, wide at the right; a dotted arrow from the label "Generator" below points up at it). A second curved arrow leaves the trapezoid and ends with an arrowhead at the label $\mathbf{x}$ in the middle of a pale-blue "Y"-shaped blob at the right, labelled "Data space" (dotted arrow). So the generator runs the encoder's map in the opposite direction, from representation space to data space.

## Slide 6 — (no title; an autoencoder, and its decoder reused as a generative model)

*OCW notice: © Torralba, Isola, and Freeman. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the bottom left, just right of the "Generative model" tab; it is the slide's only notice, so it covers all its photographs: the orange bird of the top row and the four birds at the lower right).*

Two rows separated by a horizontal dotted line. A grey vertical tab at the far left of each row carries a rotated label.

Top row, tab labelled "Autoencoder". Left to right: a square photograph labelled $\mathbf{x}$ above it, of an orange-yellow bird with a dark eye and grey beak perched on a branch against dark green spiky (aloe-like) leaves and feathery green sprigs; a black arrow right into the "Encoder" block, a rectangle with a gradient background from pale blue at the left to salmon-orange at the right, holding three outlined rectangles of decreasing height (tall, medium, short), joined by black arrows (outlines only; the block's gradient shows through them); a black arrow to a small white outlined rectangle labelled $\mathbf{z}$ above it (the bottleneck); a black arrow into the "Decoder" block, a gradient rectangle running from salmon-orange at the left to pale blue at the right, holding three outlined rectangles of increasing height (short, medium, tall), joined by arrows (outlines only); a black arrow to a second square photograph, identical to the first (the same embedded image), labelled $\hat{\mathbf{x}}$ above it.

Directly under the Decoder block (not at the dotted line's centre), a double-lined downward arrow ($\Downarrow$) crosses the dotted line, pointing from the Decoder down into the lower row.

Bottom row, tab labelled "Generative model". At the centre: $\mathbf{z} \sim p_{\mathbf{z}}$, a black arrow right into a small white outlined rectangle labelled $\mathbf{z}$ above it, directly below the top row's $\mathbf{z}$ box (only the $\mathbf{z}$ box; the decoder is not drawn, so the slide does not show what sits between). At the right, a 2 × 2 grid of bird photographs: top left a dark green and blue bird feeding at a dish of fruit; top right a pelican with a pale cream-yellow head, a long pinkish-orange bill with a red pouch and a dark body, in front of rocks, a lawn, pale-barked trees and buildings; bottom left a heron-like bird with a chestnut-brown neck and head, a long grey bill and a dark body, its black-and-white streaked back plumes fanned out, resting on a branch; bottom right a deep blue macaw (yellow eye-ring, yellow at the base of the beak) leaning down from a branch to a feeding tray, a green leaf blade standing in front of it. Below the grid, a vertical ellipsis (three dots).

## Slide 7 — (no title; encoder f then generator g gives back the data shape)

![Slide 7 — (no title; encoder f then generator g gives back the data shape)](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-7.jpg)

Three shapes on a white background, left to right, linked by arrows. Left: the pale-blue "Y"-shaped blob (as on slide 4), then a thick black arrow pointing right labelled $f$ above it; middle: a salmon-orange blob of a different shape (a lumpy gourd or pear: a wide body at the bottom, a narrower waist, a bulge, and a thin curved tendril reaching up to the upper right); then a second thick black arrow pointing right labelled $g$ above it; right: the pale-blue "Y"-shaped blob again, identical to the left one. (The data blob is mapped by $f$ into a representation space and by $g$ back to the same data shape; no text on the slide says more.)

## Slide 8 — Mixture of Gaussians

![Slide 8 — Mixture of Gaussians](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-8.jpg)

Title: "Mixture of Gaussians".

Top left, a boxed 1D plot (thin black frame). Vertical axis labelled $p_{\theta}(x)$ with an arrowhead at the top; horizontal axis labelled $x$ with an arrowhead at the right. A single smooth bell-shaped black curve rises from the axis just right of the vertical axis, peaks at about the middle of the plot and returns to the axis near the right; it has one peak. On the horizontal axis, 5 black dots, each with a short green upward arrow above it: 3 close together left of the peak (the middle and right of the three nearly touching) and 2 close together right of the peak. (The dots are data points under the model density.)

Below it, the equation

$$p_{\theta}(\mathbf{x}) = \sum_{i=1}^{k} w_i \mathcal{N}(\mathbf{x}; \mu_i, \mathbf{\Sigma}_ i)$$

At the upper right: "Parameters:" followed by $\lbrace w_i, \mu_i, \mathbf{\Sigma}_ i \rbrace_{i=1}^{k}$.

Lower right, a 2D picture: 7 translucent grey filled circles with thin dark outlines, of different radii, each with a black centre dot, overlapping each other so as to trace a "Y"-like shape: a large circle at the upper right, a smaller one just below it, a medium one lower and left of it, a smaller one below that, a medium one at the bottom, the smallest at the lower left of the middle, and a large one at the middle left. The top-right circle is labelled: its centre dot is labelled $\mu_1$ and a dotted radius line from the centre to its rim is labelled $\mathbf{\Sigma}_ 1$. A thick dashed black outline follows the outer edge of the whole cluster in the Y-shaped blob of earlier slides (the target), and the circles are translucent, so the dashed outline shows grey through them where they cover it (and overlapping circles' rims show through each other). Two legend boxes at the bottom: "Model distribution" in a grey filled box (the circles) and "Target distribution" in a box with a dashed outline (the dashed shape).

## Slide 9 — Infinite Mixture of Gaussians

![Slide 9 — Infinite Mixture of Gaussians](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-9.jpg)

Title: "*Infinite* Mixture of Gaussians" (the word "Infinite" in italics here; shown in the printed title as italic).

At the upper middle: "Parameters:" followed by $\theta$.

Left, a salmon-orange filled circle with a black outline containing $p_{\mathbf{z}}$ in its middle and 7 black dots (sample points of $\mathbf{z}$), one of them labelled $\mathbf{z}_ 1$ (upper right, near the rim). From each dot a thin black curved line runs to the right, converging into the narrow left end of a grey trapezoid labelled $g_{\theta}$ (narrow at the left, wide at the right). From the right side of the trapezoid, 7 black curved arrows fan out, each ending with an arrowhead at the centre dot of one of 7 translucent grey circles of different radii at the right. The circles are arranged as on slide 8, overlapping to trace a "Y"-like shape, with a thick dashed black outline round them (it shows grey through the translucent circles where they cover it). The upper-right circle's centre is labelled $g^{\mu}_ {\theta}(\mathbf{z}_ 1)$ (below the centre) and a dotted radius line to its rim is labelled $g^{\Sigma}_ {\theta}(\mathbf{z}_ 1)$; the longest curved arrow, which arcs over the top, ends at this circle; it is the path that starts at $\mathbf{z}_ 1$ (each of the 7 lines runs continuously from one $\mathbf{z}$ dot, behind the trapezoid, to one circle, its arrowhead stopping just short of the centre dot). Bottom right, two legend boxes: "Model distribution" (grey filled box) and "Target distribution" (dashed outline box).

Bottom left, the equation

$$p_{\theta}(\mathbf{x}) = \int_{\mathbf{z}} p_{\mathbf{z}}(\mathbf{z}) \mathcal{N}(\mathbf{x}; g^{\mu}_ {\theta}(\mathbf{z}), g^{\Sigma}_ {\theta}(\mathbf{z})) d\mathbf{z}$$

## Slide 10 — Latent variable models

![Slide 10 — Latent variable models](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-10.jpg)

Title: "Latent variable models".

Top left, under a curly brace labelled "latent variables" (brace opens downward, over the list), three lines of typewriter-and-math random-variable specifications with serif parentheticals, and a vertical ellipsis below:

- $\mathbf{z}_ 1 \sim \texttt{Bernoulli}(0.5)$ (River turns)
- $\mathbf{z}_ 2 \sim \texttt{Normal}(\mu_1, \mathbf{\Sigma}_ 1)$ (Grass color)
- $\mathbf{z}_ 3 \sim \texttt{Unif}(0, 10)$ (Number trees)

A black arrow points right from this list into a large grey trapezoid labelled "Generator" (narrow at the left, taller at the right, with a white inner border). Another black arrow points right from it to a 3 × 3 grid of nine small top-down landscape pictures (green grass in several shades, a blue river with a different course in each tile, small clusters of trees in yellow, orange and light green).

Below the trapezoid, a curly brace (opening upward) joined by a dotted vertical line to a second small curly brace above the factor $p_{\theta}(\mathbf{x} \vert \mathbf{z})$ in the equation below, i.e. the Generator corresponds to that factor.

Bottom left, a small graphical model: a white circle containing $z$, a black arrow pointing down to a grey-filled circle containing $x$ (the shaded node is the observed one).

Bottom middle, the equation

$$p_{\theta}(\mathbf{x}) = \int_{\mathbf{z}} p_{\theta}(\mathbf{x} \vert \mathbf{z}) p_{\mathbf{z}}(\mathbf{z}) d\mathbf{z}$$

with a dotted curved arrow from the words "latent variables" (bottom right) pointing up at the $\mathbf{z}$ inside $p_{\theta}(\mathbf{x} \vert \mathbf{z})$.

## Slide 11 — Infinite GMM / VAE

![Slide 11 — Infinite GMM / VAE](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-11.jpg)

Title: "Infinite GMM / VAE".

A "Learner" diagram. At the left: "Data" and a right arrow ($\rightarrow$) pointing at a large black-outlined box. The box has a grey header bar "Learner", then, centred:

"Objective", followed by

$$\max_{\theta} \mathbb{E}_ {\mathbf{x} \sim p_{\texttt{data}}} \left[ \log p_{\theta}(\mathbf{x}) \right]$$

then "Hypothesis space", and, inside a red rectangle, the equation

$$p_{\theta}(\mathbf{x}) = \int_{\mathbf{z}} \underbrace{\mathcal{N}(\mathbf{x}; g^{\mu}_ {\theta}(\mathbf{z}), g^{\Sigma}_ {\theta}(\mathbf{z}))}_ {p_{\theta}(\mathbf{x} \vert \mathbf{z})} \underbrace{\mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})}_ {p_{\mathbf{z}}(\mathbf{z})} d\mathbf{z}$$

where the first underbrace is labelled $p_{\theta}(\mathbf{x} \vert \mathbf{z})$ and the second $p_{\mathbf{z}}(\mathbf{z})$. Below the red rectangle, still inside the box, the line

$$x = g^{\mu}_ {\theta}(\mathbf{z}) + g^{\sigma}_ {\theta}(\mathbf{z}) \epsilon \qquad \epsilon \sim \mathcal{N}(0, 1)$$

(sic: this line uses a lower-case $x$ and $g^{\sigma}_ {\theta}$, where the equation above uses bold $\mathbf{x}$ and $g^{\Sigma}_ {\theta}$.) A curved arrow from the quoted words "Reparameterization trick" (below the box) points up at this line.

At the right, a right arrow ($\rightarrow$) from the box to two outputs: "Density" with $p_{\theta} : \mathcal{X} \to [0, \infty)$, and "Sampler" with $g_{\theta} : \mathcal{Z} \to \mathcal{X}$.

## Slide 12 — VAEs — learning the model parameters

Title: "VAEs — learning the model parameters".

$$\theta^{\ast} = \arg\max_{\theta} L(\lbrace \mathbf{x}^{(i)} \rbrace_{i=1}^{N}, \theta)$$

$$= \arg\max_{\theta} \sum_{i=1}^{N} \log \int_{\mathbf{z}} \overbrace{\mathcal{N}(\mathbf{x}^{(i)}; g^{\mu}_ {\theta}(\mathbf{z}), g^{\Sigma}_ {\theta}(\mathbf{z}))}^{p_{\theta}(\mathbf{x}^{(i)} \vert \mathbf{z})} \overbrace{\mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})}^{p_{\mathbf{z}}(\mathbf{z})} d\mathbf{z}$$

Annotations: a brace above the first normal density labelled $p_{\theta}(\mathbf{x}^{(i)} \vert \mathbf{z})$; a brace above the second labelled $p_{\mathbf{z}}(\mathbf{z})$; and a brace below the whole integrand (from the integral sign to $d\mathbf{z}$) labelled $p_{\theta}(\mathbf{x}^{(i)})$.

## Slide 13 — Trick #1: Estimate the integral via sampling (Monte Carlo)

Title: "Trick #1: Estimate the integral via sampling (Monte Carlo)".

$$p_{\theta}(\mathbf{x}) = \int_{\mathbf{z}} p_{\theta}(\mathbf{x} \vert \mathbf{z}) p_{\mathbf{z}}(\mathbf{z}) d\mathbf{z}$$

$$= \mathbb{E}_ {\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})} \left[ p_{\theta}(\mathbf{x} \vert \mathbf{z}) \right]$$

$$\approx \frac{1}{M} \sum_{i=1}^{M} p_{\theta}(\mathbf{x} \vert \mathbf{z}^{(j)}), \qquad \mathbf{z}^{(j)} \sim p_{\mathbf{z}}$$

(sic: the sum is printed with index $i = 1$ below it while its summand and the sampling use $j$; $j = 1$ is probably meant. Slide 14 prints $j=1$.)

## Slide 14 — Fitting an infinite mixture of Gaussians via Monte Carlo

![Slide 14 — Fitting an infinite mixture of Gaussians via Monte Carlo](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-14.jpg)

Title: "Fitting an infinite mixture of Gaussians via Monte Carlo".

$$\theta^{\ast} = \arg\max_{\theta} \sum_{i=1}^{N} \log \frac{1}{M} \sum_{j=1}^{M} \overbrace{\mathcal{N}(\mathbf{x}^{(i)}; g^{\mu}_ {\theta}(\mathbf{z}^{(j)}), g^{\Sigma}_ {\theta}(\mathbf{z}^{(j)}))}^{p_{\theta}(\mathbf{x}^{(i)} \vert \mathbf{z}^{(j)})}$$

with a brace over the normal density labelled $p_{\theta}(\mathbf{x}^{(i)} \vert \mathbf{z}^{(j)})$.

Below, a matplotlib-style plot (title "p(X|Z=z)" in small type at the top; axis ticks only, no axis labels; horizontal axis from -4 to 12 in steps of 2, vertical axis from -4 to 4 in steps of 1; a thin black frame). At the left, a salmon-orange filled disc centred at about (0, 0) with radius about 2, holding a 4 × 4 regular grid of 16 small black dots (the sampled $\mathbf{z}^{(j)}$, at x and y about −1.33, −0.44, 0.44 and 1.33). From every dot a thin dotted/dashed grey line runs right to a single point at about (8.1, 0.1), the centre of a grey filled disc of radius about 1.16 (a single Gaussian, with the point at its centre). To its upper left and right, a thick dashed black closed curve of the same Y-like shape as before (with a lobe at the upper left around x 4.5 to 7, y 0 to 2, a long lobe running to the upper right, peaking at about (10.3, 3.7), and a tail at the bottom around x 6 to 8, reaching y ≈ −3.7): the target distribution; the grey disc covers the crook of the Y between the two upper lobes, and the dashed curve passes behind it there. All 16 lines converge on the one point (the figure does not label it).

## Slide 15 — Trick #2: Importance sampling

Title: "Trick #2: Importance sampling".

At the top middle: the underlined words "Random sampling" with $\mathbf{z}^{(j)} \sim p_{\mathbf{z}}$ to their right, and "7 samples" below.

Left, a salmon-orange filled circle labelled $p_{\mathbf{z}}$ above it, holding 7 black sample dots; three are labelled $\mathbf{z}^{(1)}$ (upper right), $\mathbf{z}^{(2)}$ (upper left) and $\mathbf{z}^{(3)}$ (left, below $\mathbf{z}^{(2)}$), the other four unlabelled. A thin black curve runs from each dot to the right, converging on the left (narrow) end of a grey trapezoid labelled $g_{\theta}$. From its right end 7 curved black arrows fan out, each ending with an arrowhead at the centre dot of one of 7 thin-outlined circles of different sizes, arranged as in slides 8 and 9 (overlapping, tracing the Y-like shape with a thick dashed black outline and thin grey dashed rims; the longest arrow arcs over the top to the upper-right circle). A small pale-blue circle containing $\mathbf{x}$ (a data point) sits at the upper right of the leftmost-top circle's centre dot, inside that circle's rim; no circle is shaded on this slide.

Bottom centre, the equation

$$p_{\theta}(\mathbf{x}) = \mathbb{E}_ {\mathbf{z} \sim p_{\mathbf{z}}} \left[ p_{\theta}(\mathbf{x} \vert \mathbf{z}) \right]$$

## Slide 16 — Trick #2: Importance sampling

![Slide 16 — Trick #2: Importance sampling](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-16.jpg)

Title: "Trick #2: Importance sampling". Same figure as slide 15 (the "Random sampling" heading, "7 samples", $\mathbf{z}^{(j)} \sim p_{\mathbf{z}}$, $p_{\mathbf{z}}$ with 7 samples, $g_{\theta}$, the 7 circles), with two changes: the circle that contains the data point $\mathbf{x}$ (the one at the upper left of the cluster) is now filled grey, and the equation at the bottom is replaced by two lines:

$$p_{\theta}(\mathbf{x}) \approx \frac{1}{M} \left( p_{\theta}(\mathbf{x} \vert \mathbf{z}^{(1)}) + p_{\theta}(\mathbf{x} \vert \mathbf{z}^{(2)}) + p_{\theta}(\mathbf{x} \vert \mathbf{z}^{(3)}) + \ldots \right)$$

$$p_{\theta}(\mathbf{x}) \approx \frac{1}{M} \left( 0 + p_{\theta}(\mathbf{x} \vert \mathbf{z}^{(2)}) + 0 + \ldots \right)$$

(The second line says that most of the terms are about zero, because most samples land on Gaussians far from $\mathbf{x}$; the slide itself prints only the two lines.)

## Slide 17 — Trick #2: Importance sampling

Title: "Trick #2: Importance sampling".

$$p_{\theta}(\mathbf{x}) = \mathbb{E}_ {\mathbf{z} \sim p_{\mathbf{z}}} \left[ p_{\theta}(\mathbf{x} \vert \mathbf{z}) \right] = \int_{\mathbf{z}} p_{\mathbf{z}}(\mathbf{z}) p_{\theta}(\mathbf{x} \vert \mathbf{z}) d\mathbf{z} = \int_{\mathbf{z}} q_{\mathbf{z}}(\mathbf{z}) \frac{p_{\mathbf{z}}(\mathbf{z})}{q_{\mathbf{z}}(\mathbf{z})} p_{\theta}(\mathbf{x} \vert \mathbf{z}) d\mathbf{z}$$

$$= \mathbb{E}_ {\mathbf{z} \sim q_{\mathbf{z}}} \left[ \frac{p_{\mathbf{z}}(\mathbf{z})}{q_{\mathbf{z}}(\mathbf{z})} p_{\theta}(\mathbf{x} \vert \mathbf{z}) \right]$$

Below, in sans-serif "Set" followed by

$$q_{\mathbf{z}} = p_{\theta}(Z \vert \mathbf{x})$$

(The capital $Z$ here is as printed.)

## Slide 18 — Trick #2: Importance sampling

Title: "Trick #2: Importance sampling".

At the top middle: the underlined words "Importance sampling" with $\mathbf{z}^{(j)} \sim p_{\theta}(Z \vert \mathbf{x})$ to their right, and "1 sample" below.

Left, a plain white circle with a black outline, $p_{\mathbf{z}}$ above it, containing a small salmon-orange blob (kidney-bean shaped) near the upper left, labelled $p_{\theta}(Z \vert \mathbf{x})$ below it, with a single black dot in the blob. A thin black curve runs from that dot to the left end of the grey trapezoid $g_{\theta}$, and from the trapezoid a second curve ends with an arrowhead at the pale-blue circle $\mathbf{x}$ in the upper part of a single translucent grey circle (with a black dot at its centre, below $\mathbf{x}$). A thick dashed black outline (the same one as on slide 15, now without the seven circles: a lobe at the upper right and a tail at the bottom) is drawn at the right, the grey circle sitting at its upper-left; the outline shows grey through the translucent circle.

Bottom, an arrow between two equations. Left:

$$p_{\theta}(\mathbf{x}) = \mathbb{E}_ {\mathbf{z} \sim p_{\mathbf{z}}} \left[ p_{\theta}(\mathbf{x} \vert \mathbf{z}) \right]$$

then a thick black arrow pointing right, then:

$$p_{\theta}(\mathbf{x}) = \mathbb{E}_ {\mathbf{z} \sim p_{\theta}(Z \vert \mathbf{x})} \left[ \frac{p_{\mathbf{z}}(\mathbf{z})}{p_{\theta}(\mathbf{z} \vert \mathbf{x})} p_{\theta}(\mathbf{x} \vert \mathbf{z}) \right]$$

## Slide 19 — Trick #2: Importance sampling

![Slide 19 — Trick #2: Importance sampling](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-19.jpg)

Title: "Trick #2: Importance sampling". A side-by-side comparison of slides 16 and 18, separated by a vertical dotted line.

Left half: the underlined "Random sampling" with "7 samples" below; the same figure as slide 15 at smaller size (orange circle $p_{\mathbf{z}}$ with 7 sample dots, three of them labelled $\mathbf{z}^{(1)}$, $\mathbf{z}^{(2)}$, $\mathbf{z}^{(3)}$, trapezoid $g_{\theta}$, 7 circles with arrows, the circle round $\mathbf{x}$ filled grey). Below it:

$$p_{\theta}(\mathbf{x}) = \mathbb{E}_ {\mathbf{z} \sim p_{\mathbf{z}}} \left[ p_{\theta}(\mathbf{x} \vert \mathbf{z}) \right]$$

Right half: the underlined "Importance sampling" with "1 sample" below; the same figure as slide 18 but with the prior drawn as a dashed circle (not a solid one) and without the label $p_{\mathbf{z}}$: the dashed circle holds the small salmon blob labelled $p_{\theta}(Z \vert \mathbf{x})$ with its one dot, a curve to the trapezoid $g_{\theta}$ and an arrow from it ending at the black centre dot of the grey circle, just below $\mathbf{x}$ (on slide 18 the arrow ends at $\mathbf{x}$), inside the dashed Y-like outline at the right. Below it:

$$p_{\theta}(\mathbf{x}) = \mathbb{E}_ {\mathbf{z} \sim p_{\theta}(Z \vert \mathbf{x})} \left[ \frac{p_{\mathbf{z}}(\mathbf{z})}{p_{\theta}(\mathbf{z} \vert \mathbf{x})} p_{\theta}(\mathbf{x} \vert \mathbf{z}) \right]$$

## Slide 20 — Trick #3: Predict the optimal sampling distribution for each given datapoint

Title: "Trick #3: Predict the optimal sampling distribution for each given datapoint".

The importance-sampling picture of slide 18 without the equations or captions: at the left a dashed circle containing a small salmon-orange blob labelled $p_{\theta}(Z \vert \mathbf{x})$ with a black dot; a black curve from the dot to the left end of the grey trapezoid $g_{\theta}$; a curve from the trapezoid with an arrowhead to the dot at the centre of a grey circle that contains the pale-blue $\mathbf{x}$ circle; at the right the dashed Y-like outline.

## Slide 21 — Trick #3: Predict the optimal sampling distribution for each given datapoint

Title: "Trick #3: Predict the optimal sampling distribution for each given datapoint".

Same layout as slide 20 with the arrow reversed and the generator replaced by an encoder. At the left, the dashed circle containing the salmon blob labelled $p_{\theta}(Z \vert \mathbf{x})$ (no dot in the blob now). In the middle, a grey trapezoid labelled $f_{\psi}$ (taller at its right side and narrowing to the left), with a dotted curved arrow from the word "Encoder" below pointing up at it. A black curve starts at the pale-blue circle $\mathbf{x}$ (at the right, inside the dashed Y-like outline; the grey circle is no longer drawn), runs left into the trapezoid's right side, and a second curve leaves its left side and ends with an arrowhead pointing at the salmon blob. So an encoder $f_{\psi}$ maps $\mathbf{x}$ to the distribution $p_{\theta}(Z \vert \mathbf{x})$ over $\mathbf{z}$.

## Slide 22 — Trick #3: Predict the optimal sampling distribution for each given datapoint

![Slide 22 — Trick #3: Predict the optimal sampling distribution for each given datapoint](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-22.jpg)

Title: "Trick #3: Predict the optimal sampling distribution for each given datapoint". A diagram, with a displayed formula above its left half.

Formula, with a curly brace below it, its point facing up at the formula: the brace sits directly above the grey disc (centred on it) and lies under the opening $= \mathcal{N}(f^{\mu}_ {\psi}$ of the formula, so it ties the whole formula to the disc rather than marking any one term:

$$q_{\psi}(Z \mid \mathbf{x}) = \mathcal{N}\left(f^{\mu}_ {\psi}(\mathbf{x}), f^{\Sigma}_ {\psi}(\mathbf{x})\right)$$

Diagram, from right to left. At the right, a large closed dashed black blob (an irregular, curving shape with a lobe at the top right and a long tail curving down to the lower middle) stands for the data space; a small light-blue filled circle labelled bold "x" sits just inside the top of the blob's upper-left lobe. A thin black curve leaves this point and goes left into the right (tall) side of a light-grey trapezoid labelled $f_{\psi}$ (the trapezoid is tall at the right, short at the left; the encoder). A dotted curved arrow from the text "Encoder" below points up at the trapezoid. From the trapezoid's left side a black curve with an arrowhead goes left to a black dot inside a grey filled disc; this disc sits inside a larger dashed circle (the latent space, unfilled). The dot is labelled $f^{\mu}_ {\psi}(\mathbf{x})$ (below it) and a dotted line from the dot to the disc's edge, pointing up and to the right, is labelled $f^{\Sigma}_ {\psi}(\mathbf{x})$. So the encoder maps $\mathbf{x}$ to the centre (the mean) and radius (the covariance) of a small Gaussian-like disc in latent space.

## Slide 23 — Trick #3: Predict the optimal sampling distribution for each given datapoint

Title: "Trick #3: Predict the optimal sampling distribution for each given datapoint". Two annotations over the first equation: the text "We are fitting this" (upper left) with a dotted curved line, no arrowhead, down to a yellow hand-drawn rectangle that encloses the KL's opening parenthesis and $q_{\psi}(Z \mid \mathbf{x})$, and the text "to model this" (upper right) with a dotted curved line, no arrowhead, down to $p_{\theta}(Z \mid \mathbf{x})$.

$$J_q(\mathbf{x}, \psi) = -\mathtt{KL}\left(q_{\psi}(Z \mid \mathbf{x}) \parallel p_{\theta}(Z \mid \mathbf{x})\right)$$

$$= \mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x})}\left[-\log q_{\psi}(\mathbf{z} \mid \mathbf{x}) + \log p_{\theta}(\mathbf{x} \mid \mathbf{z}) + \log p_{\mathbf{z}}(\mathbf{z})\right] - \log p_{\theta}(\mathbf{x})$$

Then the text "Our learning problem for q:" and

$$\psi^{\ast} = \arg\max_{\psi} \frac{1}{N} \sum_{i=1}^{N} J_q(\mathbf{x}^{(i)}, \psi)$$

$$= \arg\max_{\psi} \frac{1}{N} \sum_{i=1}^{N} \underbrace{\mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x}^{(i)})}\left[-\log q_{\psi}(\mathbf{z} \mid \mathbf{x}^{(i)}) + \log p_{\theta}(\mathbf{x}^{(i)} \mid \mathbf{z}) + \log p_{\mathbf{z}}(\mathbf{z})\right]}_ {J}$$

The underbrace covers the expectation and is labelled $J$ (italic).

## Slide 24 — Putting everything together

Title (centred): "Putting everything together". Left, two learning problems.

"Our learning problem for q:" (the same two lines as slide 23, with the same underbrace labelled $J$)

$$\psi^{\ast} = \arg\max_{\psi} \frac{1}{N} \sum_{i=1}^{N} J_q(\mathbf{x}^{(i)}, \psi)$$

$$= \arg\max_{\psi} \frac{1}{N} \sum_{i=1}^{N} \underbrace{\mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x}^{(i)})}\left[-\log q_{\psi}(\mathbf{z} \mid \mathbf{x}^{(i)}) + \log p_{\theta}(\mathbf{x}^{(i)} \mid \mathbf{z}) + \log p_{\mathbf{z}}(\mathbf{z})\right]}_ {J}$$

"Our learning problem for p:"

$$J_p(\mathbf{x}, \theta) = \log \mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x})}\left[\frac{p_{\mathbf{z}}(\mathbf{z})}{q_{\psi}(\mathbf{z} \mid \mathbf{x})} p_{\theta}(\mathbf{x} \mid \mathbf{z})\right]$$

$$\theta^{\ast} = \arg\max_{\theta} \frac{1}{N} \sum_{i=1}^{N} J_p(\mathbf{x}^{(i)}, \theta)$$

Right, two small diagrams, one beside each learning problem. Upper (beside the q problem): a miniature of slide 22's figure, with the dashed latent circle and grey disc at the left holding the dot labelled $f^{\mu}_ {\psi}(\mathbf{x})$ and $f^{\Sigma}_ {\psi}(\mathbf{x})$, the trapezoid $f_{\psi}$ in the middle, and the dashed data blob with the light-blue point "x" at the right; the arrow runs from x through $f_{\psi}$ to the latent disc. Lower (beside the p problem): the dashed latent circle at the left, unfilled, with a small black dot inside, above its centre; a black curve with arrowhead from that dot goes right through a trapezoid labelled $g_{\theta}$ (tall at the right, the generator) to a black dot inside a small grey disc (with a dashed grey outline) that overlaps the light-blue point "x" on the dashed data blob at the right. The arrow runs from latent space to data space.

## Slide 25 — Putting everything together

Heading text in the slide: "Improving our learning problem for p:" (the word "Improving" is in italics). Top right, a small diagram like the lower diagram of slide 24: dashed latent circle with a dot, arrow through $g_{\theta}$ to a grey disc with a dashed grey outline around the light-blue point "x" on the dashed data blob.

$$J_p(\mathbf{x}, \theta) = \log \mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x})}\left[\frac{p_{\mathbf{z}}(\mathbf{z})}{q_{\psi}(\mathbf{z} \mid \mathbf{x})} p_{\theta}(\mathbf{x} \mid \mathbf{z})\right]$$

$$\geq \mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x})}\left[\log\left(\frac{p_{\mathbf{z}}(\mathbf{z})}{q_{\psi}(\mathbf{z} \mid \mathbf{x})} p_{\theta}(\mathbf{x} \mid \mathbf{z})\right)\right]$$

This line is annotated at the right with a small left-pointing triangle and "Jensen's inequality".

$$= \mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x})}\left[-\log q_{\psi}(\mathbf{z} \mid \mathbf{x}) + \log p_{\theta}(\mathbf{x} \mid \mathbf{z}) + \log p_{\mathbf{z}}(\mathbf{z})\right]$$

$$= J$$

The line $= J$ carries the annotation (left-pointing triangle) "VAE objective".

$$= \mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x})}\left[\log p_{\theta}(\mathbf{x} \mid \mathbf{z})\right] - \mathtt{KL}\left(q_{\psi}(Z \mid \mathbf{x}) \parallel p_{\mathbf{z}}\right)$$

This last line is enclosed in a red rectangle, with a red arrow pointing left at it from the red bold text "J = Evidence Lower-Bound (ELBO)".

Below: "Now, p and q share the same exact objective!" and

$$\theta^{\ast}, \psi^{\ast} = \arg\max_{\theta, \phi} \frac{1}{N} \sum_{i=1}^{N} J(\mathbf{x}^{(i)}, \theta, \phi)$$

(sic: the left side prints $\psi^{\ast}$ and the argmax subscript and the argument of $J$ print $\phi$; $\psi$ is the encoder's parameter everywhere else on slides 22–28, so $\phi$ is probably meant to be $\psi$.)

## Slide 26 — Trick #3: Predict the optimal sampling distribution for each given datapoint

![Slide 26 — Trick #3: Predict the optimal sampling distribution for each given datapoint](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-26.jpg)

Same layout as slide 22, with these differences: the latent space is now a solid-filled salmon/orange circle with a black outline (not a dashed one); the grey disc inside it is semi-transparent, and a second black dot sits near the disc's upper right, joined to the mean dot $f^{\mu}_ {\psi}(\mathbf{x})$ by a short dotted arrow (open arrowhead at the new dot), the label $f^{\Sigma}_ {\psi}(\mathbf{x})$ sitting just left of the new dot. This shows a sample $\mathbf{z}$ drawn from the disc. The encoder $f_{\psi}$ trapezoid, the "Encoder" label with its dotted arrow, the dashed data blob and the light-blue "x" are as on slide 22 but placed higher on the page. Slide 22's formula and brace above the circles are not on this slide. At the bottom, a displayed equation:

$$p_{\theta}(\mathbf{x}) = \mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x})}\left[\frac{p_{\mathbf{z}}(\mathbf{z})}{q_{\psi}(\mathbf{z} \mid \mathbf{x})} p_{\theta}(\mathbf{x} \mid \mathbf{z})\right]$$

## Slide 27 — Trick #3: Predict the optimal sampling distribution for each given datapoint

![Slide 27 — Trick #3: Predict the optimal sampling distribution for each given datapoint](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-27.jpg)

Build step: it adds the generator half to slide 26's picture. Same title and bottom equation as slide 26:

$$p_{\theta}(\mathbf{x}) = \mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x})}\left[\frac{p_{\mathbf{z}}(\mathbf{z})}{q_{\psi}(\mathbf{z} \mid \mathbf{x})} p_{\theta}(\mathbf{x} \mid \mathbf{z})\right]$$

The solid salmon latent circle is at the left with only one black dot in it (the sampled $\mathbf{z}$, upper right of centre); the encoder trapezoid and its labels are gone. A black curve leaves that dot, goes right into the left (narrow) side of a light-grey trapezoid labelled $g_{\theta}$ (the generator; short at the left, tall at the right), and from its right side a black curve with an arrowhead ends on a black dot at the centre of a grey disc. The disc sits on the upper-left lobe of the dashed data blob, whose outline is drawn over it, and the light-blue point "x" lies inside the disc directly above the centre dot, about halfway to the disc's top edge. So the generator maps the sampled latent to a distribution over data space that is centred near, not on, x.

## Slide 28 — Looks like an autoencoder!

![Slide 28 — Looks like an autoencoder!](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-28.jpg)

Title: "Looks like an autoencoder!". At the top a displayed formula with two coloured underlines: a magenta underline beneath the first term (the expectation) and a teal underline beneath the KL term.

$$\mathbb{E}_ {\mathbf{z} \sim q_{\psi}(Z \mid \mathbf{x})}\left[\log p_{\theta}(\mathbf{x} \mid \mathbf{z})\right] - \mathtt{KL}\left(q_{\psi}(Z \mid \mathbf{x}) \parallel p_{\mathbf{z}}\right)$$

Diagram, left to right. At the left a light-blue filled irregular blob (a "Y"/boomerang shape with a rounded lobe at the upper right) is the data space; a large black dot labelled italic $x$ sits near the bottom end of its lower stem (the blob runs slightly off the page's left edge). A black curve goes from $x$ into a light-grey trapezoid labelled $f_{\psi}$ (encoder, tall at the left). Beneath it, two lines:

$$\mu_z = f^{\mu}_ {\psi}(x)$$

$$\sigma_z^2 = f^{\Sigma}_ {\psi}(x)$$

From $f_{\psi}$ a black arrow leads into the middle: a solid salmon circle (latent space) containing the label $\mu_z$ (near its centre, the arrow's head just left of it; the curve from $x$ runs behind $f_{\psi}$ and has its only arrowhead here), a short black arrow from $\mu_z$ down and to the right to the label $\mu_z + \epsilon\sigma_z$. A black curve leaves $\mu_z + \epsilon\sigma_z$ rightwards into a second light-grey trapezoid labelled $g_{\theta}$ (generator, tall at the right). Beneath it:

$$\mu_x = g^{\mu}_ {\theta}(\mu_z + \epsilon\sigma_z)$$

$$\sigma_x^2 = g^{\Sigma}_ {\theta}(\mu_z + \epsilon\sigma_z)$$

A black arrow from $g_{\theta}$ (the curve runs behind it) ends at the $\mu_x$ dot of the lowest circle in the right-hand panel: a cloud of 15 overlapping translucent grey circles of equal size that together trace the same boomerang shape as the blue blob (the data distribution covered by Gaussians). At the lowest circle, a small black dot labelled $\mu_x$ (the circle's centre) is joined by a dotted line to the circle's edge, labelled $\sigma_x^2$, and a larger black dot labelled italic "x" lies just below-left of the centre, inside the circle.

Below the diagram, a displayed expression with a magenta underline under its first two terms and a teal underline under its last term; the curly brace over the squared-error fraction is labelled "reconstruction error" (small serif text):

$$\log \frac{1}{\sigma_x \sqrt{2\pi}} - \frac{\overbrace{(x - \mu_x)^2}^{\text{reconstruction error}}}{2\sigma_x^2} \quad - \quad \frac{1}{2}\left(\mu_z^2 + \sigma_z^2 - \log(\sigma_z^2) - 1\right)$$

Captions: under the magenta underline, "data likelihood under 1 importance sample"; under the teal underline, "squash encodings toward origin". At the lower left, the small note "(eqns for 1D x and z)". In the diagram and the bottom expression $x$ and $z$ are italic and not bold; the top formula keeps the bold $\mathbf{x}$, $\mathbf{z}$ of the slides above.

## Slide 29 — Autoencoder

![Slide 29 — Autoencoder](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-29.jpg)

Title: "Autoencoder". Three shapes in a row, joined by arrows. At the left, a light-blue filled "Y"-like blob with a black outline (a rounded lobe at the upper right, a smaller lobe at the left, and a tail at the bottom). A thick black arrow pointing right, with the italic label $f$ above it. In the middle, a salmon/orange filled blob with a black outline, a narrow, wiggly, vertically elongated shape: a wide bulb at the bottom, a narrower waist, a second, taller bulb above with a wavy right edge, and a thin tail curving to the upper right. A second thick black arrow pointing right, with the italic label $g$ above it. At the right, a copy of the light-blue blob from the left. Two bullets under the picture:

- Encodes and decodes the data
- Low-dimensional bottleneck

## Slide 30 — Variational Autoencoder

![Slide 30 — Variational Autoencoder](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-30.jpg)

Title: "Variational Autoencoder". Same layout as slide 29: the light-blue blob at the left, arrow labelled $f$, arrow labelled $g$, the light-blue blob at the right. The difference is the middle shape: the salmon blob is now a plain circle (with a thin black outline). Three bullets:

- Encodes and decodes the data
- Low-dimensional bottleneck
- Gaussian bottleneck (**can sample**; **disentangled**)

The words "can sample" and "disentangled" are in bold. The slide adds the third bullet and the round latent space.

## Slide 31 — (no title; many data points encoded to one Gaussian disc, which is decoded to one disc in data space)

![Slide 31 — (no title; many data points encoded to one Gaussian disc, which is decoded to one disc in data space)](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-31.jpg)

A whole-page picture with no title and no text apart from the slide number "31". Left: a light-blue filled "Y"-shaped blob (data space; a rounded lobe at the upper right, a smaller lobe at the left, a tail at the bottom) holding 27 small black dots on a regular grid, in 8 rows of 2, 3, 5, 5, 5, 2, 3 and 2 (top to bottom; the grid dots fall inside the blob). From each dot a thin dashed grey line runs right to one common point, a small black dot at the centre of a solid grey disc with a thick black outline, in the middle of the page. A thick dashed black circle (unfilled, larger) surrounds the grey disc. So many data points are all mapped to the same disc in the latent space (the disc is the encoder's Gaussian; the dashed circle is the latent space, here with the disc nearly filling it). From the disc's centre, one thin grey line runs right to a second solid grey disc with a thin black outline and a black centre dot, at the right. This disc sits at the junction of the lobes and tail of an unfilled dashed black blob of the same shape and size as the blue blob at the left, crossing its outline at the left and upper right. So the picture shows the decoder mapping the single latent point to a disc-shaped distribution in data space. The page is a still; the recording plays it as an animation, which the lecturer introduces as "the encoder mapping at initialization … everything maps to 0. And then the decoder. So initially, all of my mixture components are near 0" (≈1:04:46).

## Slide 32 — (no title; two pictures of a latent grid decoded to a Gaussian disc, and many data points encoded to a Gaussian disc, each with a heat-map of a Gaussian)

![Slide 32 — (no title; two pictures of a latent grid decoded to a Gaussian disc, and many data points encoded to a Gaussian disc, each with a heat-map of a Gaussian)](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-32.jpg)

Four pictures and no text apart from the slide number "32": two diagrams side by side across the top, and a square heat-map under each (four separate embedded images).

Left figure: a solid salmon circle (latent space) at the left containing a regular 4 × 4 grid of 16 small black dots. Thin dashed grey lines from every dot converge on one small black dot at the centre of a solid grey disc at the right. That disc sits inside an unfilled dashed black blob (the data-space blob from slide 22), over the notch between the blob's left lobe and its tall upper-right lobe, its edge meeting the dashed outline on its left and its upper right. Under the figure, a square heat-map (purple, viridis-style colours) with a single round blob at its centre, yellow in the middle and fading through green and teal to the purple background: a Gaussian bump. The square is at the lower middle-left of the page.

Right figure: a light-blue filled "Y"-shaped blob (data space) holding 27 black dots on a grid (the same 27 as on slide 31), with thin dashed grey lines from each dot converging on one dot at the centre of a solid grey disc, which is enclosed by a thick dashed black circle (unfilled), at the right. Under it, a second purple square heat-map with the same yellow-green Gaussian bump at its centre, at the lower right of the page, slightly higher than the first.

The picture suggests the two directions: sampled latent points all pointing to a Gaussian in data space, and data points all pointing to a Gaussian in latent space. The slide prints no caption saying so. In the recording the left diagram is the decoder and the right the encoder at initialization (≈1:04:46), and the heat-maps are introduced as "what the densities look like if I plot them as a heat map … And here's the density over the latent space" (≈1:06:18); by their placement the left is most likely the density in data space and the right the density over the latent space, but neither the page nor the transcript says which is which.

## Slide 33 — VAE — three tricks

Title: "VAE — three tricks". Three bullets:

- Trick #1: approximate an infinite mixture with samples
- Trick #2: sample efficiently via importance sampling
- Trick #3: predict the optimal sampling distribution for each datapoint

## Slide 34 — Training a VAE — Step by step

Title: "Training a VAE — Step by step". Five bullets (typeset in the serif LaTeX font, with some words in italics):

- Sample one or more $\mathbf{x} \sim \lbrace \mathbf{x}^{(i)} \rbrace_{i=1}^{N}$
- *Encode* the data with a forward pass through $f_{\psi}$
- For each datapoint, create one or more noisy latent codes using the distribution parameterized by the encoder
- *Decode* the data by passing the noisy latent codes through the $g_{\theta}$
- Compute the losses and backprop to update $\theta$ and $\psi$

(The word "Encode" and the word "Decode" are in italics. The third bullet is printed with a hyphenated line break, "pa-" / "rameterized".)

## Slide 35 — (no title; a generator turns random variables into a grid of river images)

Left, three lines of sampled variables, each with a typewriter-font distribution and a parenthetical in serif type, then a vertical ellipsis (three dots):

$$\mathbf{z}_ 1 \sim \mathtt{Bernoulli}(0.5) \quad \text{(River turns)}$$

$$\mathbf{z}_ 2 \sim \mathtt{Normal}(\mu_1, \mathbf{\Sigma}_ 1) \quad \text{(Grass color)}$$

$$\mathbf{z}_ 3 \sim \mathtt{Unif}(0, 10) \quad \text{(Number trees)}$$

$$\vdots$$

A black arrow points right into a large trapezoid with a thick black outline and a grey fill, labelled "Generator" (tall at the right, narrow at the left); a second black arrow points right to a 3 × 3 grid of nine small square images separated by black lines (these nine are the top-left 3 × 3 of the 4 × 4 data grid on slide 36: the same image, cropped). Each image is a pixel-art-like top-down map tile with a green grass background, a winding dark-blue river, and small clusters of trees (dark green, light green and orange-tan blobs). The shades of green differ from tile to tile (darker and bluer at the top left, bright light green in the middle row). Rivers: top row, a river that enters at the bottom left and winds up and bends right, a river that curves along the lower left, a tile with only a bit of river at the bottom right corner; middle row, a river that winds like an "S" from the top edge near the left corner down to the bottom edge, a vertical river down the middle, a river that bends from the left edge to the right then down; bottom row, a thin river running down the left half, a river running from top left to bottom centre, a short river curling at the lower right. The slide prints no title or credit.

## Slide 36 — (no title; training a learner on river images, then sampling from the generator)

![Slide 36 — (no title; training a learner on river images, then sampling from the generator)](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-36.jpg)

A two-row diagram split by a dotted horizontal line across the page. Each row has a grey rotated label box at the far left: "Training" (upper row) and "Sampling" (lower row), both in serif type reading bottom to top.

Upper row (Training): the heading "Data" above a 4 × 4 grid of 16 river-map tiles (the same pixel-art style as slide 35: green grass of varying shades, a dark-blue winding river, small tree clusters). A thin arrow points right into a white square box labelled "Learner"; a second thin arrow points right from the box to the text $f_{\psi}\thinspace, g_{\theta}$.

A thick dashed black arrow runs straight down from under $g_{\theta}$, across the dotted dividing line, to just above the top of the trapezoid in the lower row (meaning the trained $g_{\theta}$ is carried into the sampling stage).

Lower row (Sampling), at the right: a bold $\mathbf{z}$, an arrow pointing right into a white trapezoid labelled $g_{\theta}$ (tall at the right), and an arrow pointing right to the heading "Samples" over a 4 × 4 grid of 16 tiles. These generated tiles are blurrier than the data tiles: each shows a smooth green background (shades from dark green to bright lime) with one blurred blue river segment, some curling into a hook or loop; no tree clusters are visible.

## Slide 37 — (no title; moving along one latent direction changes river curvature)

Heading at upper right: "Synthesized Images". Left: the heading "Latent variables" with "(controls)" under it in smaller type, and below it a solid salmon circle (latent space) with a thick black outline. Inside it, a horizontal dashed double-headed arrow in the circle's middle, from the circle's centre-left to a bold $\mathbf{u}$ at its right end, with 5 black dots evenly spaced along it. A thin black arrow runs from the circle to the right panel.

Right panel: a dotted vertical double-headed arrow at the left labelled bold $\mathbf{v}$ at its top; a dotted horizontal double-headed arrow at the bottom, labelled bold $\mathbf{u}$ at its right end and "River curvature" in quotation marks beneath it. A single row of 5 square tiles, side by side, sits between the axes, each a flat green background with one blurred dark-blue river. Left to right the river bends progressively: tile 1 curves strongly to the left at the bottom and leans to the upper right; tile 2 has a milder left-hand bend; tile 3 (outlined slightly thicker) is a nearly straight vertical river; tile 4 mirrors tile 2, bending to the right; tile 5 mirrors tile 1, a strong bend to the right at the top. There is no variation along $\mathbf{v}$ yet; the $\mathbf{v}$ axis is empty. The slide is the first of a build of three (37, 38, 39). (The row is the middle row of the 5 × 5 grid on slide 38: the same image, cropped.)

## Slide 38 — (no title; two latent directions control river curvature and background colour)

![Slide 38 — (no title; two latent directions control river curvature and background colour)](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-38.jpg)

Build step from slide 37: adds the second latent direction $\mathbf{v}$ and four more rows of tiles. Heading "Synthesized Images" at the upper right; "Latent variables (controls)" at the upper left.

Left: the salmon circle now holds a 5 × 5 grid of 25 black dots in its middle, with a dashed double-headed arrow through the central row labelled bold $\mathbf{u}$ at its right end and a dashed double-headed vertical arrow through the central column labelled bold $\mathbf{v}$ at its top. A thin arrow points right to the right panel.

Right: axes as on slide 37: a dotted vertical double-headed arrow labelled bold $\mathbf{v}$ at its top, with the rotated label "Background color" in quotation marks to its left, and a dotted horizontal arrow labelled bold $\mathbf{u}$ with "River curvature" below. Between them a 5 × 5 grid of 25 tiles. Along each row (the $\mathbf{u}$ direction) the river bends as on slide 37 (strong left bend, mild left bend, straight, mild right bend, strong right bend). Down each column (the $\mathbf{v}$ direction) the background shifts from teal (blue-green) in the top row to pure green in the bottom two rows, which are almost the same colour; the bottom row's rivers are also fainter and grainier. The slide prints no further labels.

## Slide 39 — BigGAN [Brock et al. 2018]

*OCW notice: © Torralba, Isola, and Freeman. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower left of the page, under the latent circle; the page's only picture is the 5 × 5 grid of BigGAN-generated bird images at the right, the same embedded image as on slide 1).*

Title at the upper left: "BigGAN" followed by the citation "[Brock et al. 2018]". Heading at the upper right: "Synthesized Images". The layout is that of slide 38, with the same salmon latent circle holding a 5 × 5 grid of 25 dots with dashed axes labelled bold $\mathbf{u}$ (horizontal) and bold $\mathbf{v}$ (vertical), under "Latent variables (controls)", and an arrow to the right panel.

Right panel: dotted axes labelled bold $\mathbf{v}$ (vertical, with the rotated label "Background color" in quotation marks) and bold $\mathbf{u}$ (horizontal, with "Bird orientation" in quotation marks below). Between them, a 5 × 5 grid of 25 synthesized images of the same kind of small bird (black cap and bib, white cheek, grey back, pale belly) perched on a branch; neither the slide nor the lecturer names the bird. Along each row the bird turns: at the left it faces left and is seen in three-quarter view on a diagonal twig, in the middle column it faces the camera on a horizontal branch, and at the right it faces right with its tail towards the lower left. Down each column the background changes from a pale blue-grey and brown at the top to a darker, green-brown at the bottom.

## Slide 40 — (no title; four-panel plot of the encoder q(Z|X=x) and decoder p(X|Z=z))

![Slide 40 — (no title; four-panel plot of the encoder q(Z|X=x) and decoder p(X|Z=z))](../images/15-generative-models-representation-learning-meets-generative-modeling/slide-40.jpg)

A whole-page picture of four matplotlib plots with no text layer (a screenshot), no title, and the slide number "40" at the bottom. A 2 × 2 grid. All four panels have x-axes labelled with ticks from -4 to 12 (step 2) and y-axes with ticks from -4 to 4 (step 1); the axes carry no axis names.

Top left, titled "q(Z|X=x)": a blue filled "Y"-shaped blob (data space, spanning about x from -3.5 to 3.5 and y from -3.5 to 3.5, a lobe at the upper right, a smaller lobe at the left, and a tail at the lower left) holding 26 small black square markers on a grid (8 rows of 1, 3, 5, 5, 5, 2, 3 and 2, top to bottom); thin dashed grey lines run from every dot to a single point at about (8, 0.1), the centre of a grey disc with a black outline (radius about 1.25), which sits inside a larger solid salmon-red disc (radius about 2, centred near (8, 0)) which is the latent space.

Top right, titled "p(X|Z=z)": a solid salmon-red disc at left (centred near (0, 0), radius about 2) holding a 4 × 4 grid of 16 black dots, with dashed grey lines from each dot converging on a point at about (8, 0), the centre of a grey disc with a black outline (radius about 1.25). The disc lies over a blue filled blob of the same Y shape as in the top-left panel, here placed with its left lobe near x = 5.5 and its large upper-right lobe near x = 10.

Bottom left: a black rectangular panel (same axes) showing the blob from the top-left panel filled with a colour gradient that varies with position: green at the left lobe, yellow at the upper-right lobe, orange in the middle of the right arm, and dark red at the tail; a tiny white speck sits at about (8, 0.2). Bottom right: a black panel with a disc at (0, 0), radius about 2, filled with a similar gradient (green at the upper left, yellow at the upper right, orange to red at the bottom), and a tiny white speck at about (8, 0.2). The colour runs from dark to red left to right and from dark to green bottom to top, so each point is coloured by its coordinates; the slide prints no caption saying so, but the lecturer: "here, I'm coloring the input points according to their coordinates within this space. And here's what happens as I train it" (≈1:17:52–1:18:37). The page is the untrained frame: every coloured point still maps to the one speck at x ≈ 8.

## Slide 41 — Comparing popular model types

A table with a header row, a thin horizontal rule under it, and thin vertical rules between its four columns. Each cell below was read from the image.

| Method | Latents? | Density/Energy? | Generator? |
| --- | --- | --- | --- |
| Energy-based models | ✗ | ✓ (energy only) | ✗ |
| Gaussian | ✗ | ✓ | ✗ |
| Autoregressive models | ✗ | ✓ | ✓ (slow) |
| Diffusion models | ✓ (high-dimensional) | ✗ | ✓ (slow) |
| GANs | ✓ | ✗ | ✓ |
| VAEs | ✓ | ✗ | ✓ |

Footnote at the lower left: "* rough categorization of vanilla versions of each model". (No row, cell, header or the title carries an asterisk; the footnote has no anchor.)

## Slide 42 — (OCW end page)

A smaller page (landscape, not the 16:9 slide size) with black text on white: "MIT OpenCourseWare", "https://ocw.mit.edu" (blue, underlined), "6.7960 Deep Learning", "Fall 2024", and "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms" (the URL blue and underlined). The page prints the number "42" at bottom centre, in larger type than the numbers on the slides (the script that reported none was wrong). It is not lecture content.
