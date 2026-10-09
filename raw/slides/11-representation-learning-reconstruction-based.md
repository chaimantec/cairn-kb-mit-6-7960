---
title: Lecture 11 — Representation Learning I (slide deck)
lecture: 11
slides: 65
source_pdf: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec11.pdf
note: Printed slide numbers 1–64 (bottom centre) equal the PDF page numbers exactly. Page 65 is OCW's appended end page (a smaller page), which prints 65.
figure_audit: Transcribed by Sonnet from page images; 43 figure-, diagram-, chart-, table- and equation-heavy pages (3–5, 7–14, 16–21, 25, 26, 28, 31, 32, 35, 38–40, 42–49, 52, 53, 55, 57–61 and 63) were then checked by Opus, a different model, from 55–600 dpi renders, the embedded rasters at native resolution, the vector data and the text layer. Every equation agreed; corrections were applied on 13 pages (counts, positions and drawing order) and smaller fixes on 12 more.
---

# Lecture 11 — Representation Learning I: slide-by-slide

Text and figures of all 65 pages of
[`mit6_7960_f24_lec11.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec11.pdf),
transcribed from the deck (speaker: Phillip Isola; the deck's title slide reads "Lecture 11: Representation Learning I", while its outline slide 2 is headed "12. Representation Learning I"). Cite these as "slide N" — the printed number equals the PDF page number for slides 1–64; page 65 is OCW's appended end page. Diagrams, plots, screenshots and photographs are described in prose since the KB is read as text.

**Images.** 25 slides carry a whole-slide render under their heading: 3, 5, 7–12, 25–28, 31, 39, 42–44, 46, 47, 50–52 and 59–61. Not rendered: the 22 slides with an OCW "All rights reserved" notice (4, 14–21, 24, 36–38, 40, 41, 45, 53–55, 57, 58 and 63); slide 13, whose CLIP figure prints no notice here but is the figure that lecture 1's deck marks "© Torralba, Isola, and Freeman. All rights reserved" on its own slide 73; slide 6, a build step that slide 7 completes; text, equation and table slides this file reproduces (22, 29, 32–35, 48, 49, 56, 62 and 64); and the title, outline, dividers and end page (1, 2, 23, 30 and 65). Slides 31, 39 and 42 share image objects with excluded slides but do not draw them on the page, or draw only an unnoticed one (see `AGENTS.md`).

Companion pages: [wiki page for this lecture](../../wiki/11-representation-learning-reconstruction-based.md) · [transcript](../transcripts/11-representation-learning-reconstruction-based.md)

**Signposting slides you can skip.** Slide 1 is the title; slide 2 is the outline (Nets learn representations; Why learn representations?; Autoencoders; Clustering and VQ; Self-supervised learning by reconstruction); slides 23 and 30 are section dividers ("Why learn representations?" and "How do you learn a good representation?"); slide 64 is the summary; slide 65 is the OCW end page.

Some slides are **build steps** or near-repeats, transcribed individually with a note of what they add: slides 4 and 5 (the x2vec picture of a layered code, then its abstract data-space and representation-space version); slides 6 and 7 (the identity function as a plot, then as a mapping between number lines); slides 11 and 12 (the MLP stack, then SGD beside steepest descent in the spectral norm); slides 15 and 16 (the car-photograph network, then its circles removed and a recorded neuron added), with slide 54 repeating slide 16's trace on the colorization network; slides 17 to 20 (patch mosaics for layers 1, 2, 3 and 5), reused in miniature on slide 21; slides 25 to 28 (training and testing on a new task, linear adaptation with a frozen encoder, finetuning, and pretraining–adapting–testing); slides 32 to 34 (supervised, no-example and representation learning); slides 36 to 38 (a fish photograph with its compact mental representation, a compressed code, and the full autoencoder); slides 40 to 42 (the $L_2$ autoencoder, its linear case and the shapes experiment); slides 46 to 49 (k-means as a picture, as an encoder and decoder, as a box of objective and optimizer, and as a vector-quantized autoencoder); slides 50 to 52 (data compression, label prediction, data prediction); and slides 59 and 60 (masked autoencoder and BERT, both citing "[He, Chen, Xie, et al. 2021]"). Slides with no printed title are headed here with a description: 3, 9, 12, 14, 18 to 21, 25 to 28, 40, 41, 43, 53 and 57.

Printed slips and oddities, kept as printed: the outline on slide 2 is numbered "12." while the title slide says Lecture 11; the output labels of the music-preference examples read "Neural" where "Neutral" is meant (slides 25 to 28, in the same place on each); the citation on slides 16 and 54 prints "Torralba., ICLR 2015" with a stray full stop; slide 9's softmax equation and wiring-graph node write the exponent as $e^{-\tau x_{\texttt{in}}[i]}$ and $e^{-\mathbf{x}_ {\texttt{in}_ i}}$, the second without the $\tau$ the first carries; slide 12's URL is overlapped by the printed slide number; slide 63 is a pasted slide that carries its own page number "59" and footer; slide 7's left plot has eight dots and its right number line nine; and slide 29 gives the primes of $\mathbf{W}'$ and $\mathbf{b}'$ as typographic apostrophes, as does slide 27's "f’".

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and outline |
| 3–5 | Nets learn representations: the forward direction as representation learning and the reverse as generative modeling; x2vec, embeddings and encoders |
| 6–13 | Layers as data transformations: a function as a plot and as a mapping of points; linear, relu, L2-norm and softmax; an MLP and its training; SGD against steepest descent in spectral norm; CLIP |
| 14–21 | Deep nets as brains: the visual cortex hierarchy; deep net "electrophysiology"; Zeiler and Fergus's patches for layers 1, 2, 3 and 5 |
| 22 | What is a representation? (the encoder $f : \mathcal{X} \to \mathbb{R}^d$) |
| 23–29 | Why learn representations?: transfer learning, linear adaptation, finetuning |
| 30–31 | How do you learn a good representation? Properties of good representations |
| 32–35 | Learning from examples, without examples, and representation learning; compression and prediction as two basic approaches |
| 36–43 | Autoencoders: learning via compression, the autoencoder and its objective, the PCA case, a shapes experiment with nearest neighbours and layer-wise accuracy |
| 44–49 | Clustering and VQ: clustering as an encoder, a rep learning perspective, k-means, vector-quantized autoencoders |
| 50–52 | Compression, label prediction and data prediction compared |
| 53–58 | Self-supervised learning by prediction: colorization, neurons that respond to faces, dog faces and flowers, pretext tasks, imputation |
| 59–62 | Masked autoencoders and BERT; masked prediction against autoencoding, and three hypotheses |
| 63 | LeCun's slide on how much information the machine is given during learning |
| 64 | Summary |
| 65 | OCW end page |

---

## Slide 1 — Lecture 11: Representation Learning I

Title: "Lecture 11: Representation Learning I". Subtitle: "Speaker: Phillip Isola".

At the right, a small diagram: a light-grey blob shaped like a three-armed "Y" or boomerang (one arm up, one to the left, one down-left) with a bold $\mathbf{x}$ inside its upper arm; a curved black arrow runs from $\mathbf{x}$ to a trapezoid (wider on the left, narrower on the right) labelled with an italic $f$; from the trapezoid a second curved arrow with an arrowhead points to a bold $\mathbf{z}$ beside a grey circle at the far right, shaded from white at the upper left to darker grey at the lower right.

Footer bar (grey): MIT logo, "6.7960 Deep Learning" (red), "https://phillipi.github.io/6.7960" (underlined), right side "Fall 2024". The printed slide number "1" sits just below the end of the URL.

## Slide 2 — 12. Representation Learning I

Title as printed: "12. Representation Learning I" (the number 12 is the deck's own; the title slide says Lecture 11).

- Nets learn representations
- Why learn representations?
- Autoencoders
- Clustering and VQ
- Self-supervised learning by reconstruction

## Slide 3 — (no title; deep nets transform datapoints layer by layer)

![Slide 3 — (no title; deep nets transform datapoints layer by layer)](../images/11-representation-learning-reconstruction-based/slide-3.jpg)

No title is printed. Left, four bullets:

- Deep nets transform datapoints, layer by layer
- Each layer is a different *representation* of the data (the word "representation" in italics)
- In the forward direction, the mapping goes from observed data to latent embeddings — this direction is called **representation learning** (the last two words in bold)
- In the reverse direction, the mapping goes from latent embeddings to observed data — this direction is called **generative modeling** (the last two words in bold)

Right, a figure: a tall hexagonal prism drawn as an isometric 3D box (outline only), labelled "Embedding" above its top and "Data" below its bottom. Its top and bottom faces are outline-only diamonds, and inside it three light-grey translucent parallelogram sheets are stacked at intervals (the layers); where the lowest sheet lies over the box's back edges, those edges can look like a fourth sheet. Five thin dotted curved lines, each starting at a dot and ending in an arrowhead near the top, rise through the sheets: three start at black dots near the bottom of the box and two at dots drawn under the lowest sheet, so they show grey through it; all five end near the top of the prism. A black vertical line with an upward arrowhead on the left of the prism carries the rotated label "Representation learning" (reading bottom to top); a black vertical line with a downward arrowhead on the right carries the rotated label "Generative modeling" (reading bottom to top, arrow pointing down).

## Slide 4 — x2vec

Title: "x2vec". Left: a photograph, labelled with a bold capital $\mathbf{X}$ above and "Image" below, of a small orange-yellow vintage hatchback car (a Fiat 500 type) parked in profile facing left on a cobbled street in front of a pale stone building with grey roller shutters and an arched window. A thick black arrow points right to three boxed columns of circles, each joined to the next by a short black arrow:

- First column (a tall box): 10 circles, from top to bottom light grey, white, black, mid grey, mid grey, mid grey, light grey, light grey, white, mid grey.
- Second column (a shorter box): 7 circles, mid grey, light grey, black, mid grey, white, white, near-black.
- Third column (a shorter box still): 5 circles, white, light grey, light grey, dark grey, light grey.

A dotted curved arrow from the text "layer 1 representation of image" (below, right of the first column) points up to the first column's bottom. A dotted curved arrow from the text "layer 3 representation of image" (top right) points to the third column's top.

At the right, three overlapping flat shapes: a pale beige card at the back labelled "building" in typewriter type, a dark-grey card in front of it at the bottom labelled "road", and, frontmost, an orange car silhouette labelled "car", which overlaps both cards (its wheels lie over the road card). (This stands for what the representation might encode.)

Bottom text: "Represent data as a neural **embedding** — a vector/tensor of neural activations" (the word "embedding" in bold) and below it, smaller, "(perhaps representing a vector of detected texture patterns or object parts)".

The notice sits under the photograph and its "Image" label, at the lower left: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the car photograph). All rights reserved — excluded from the CC license.*

## Slide 5 — x2vec

![Slide 5 — x2vec](../images/11-representation-learning-reconstruction-based/slide-5.jpg)

Title: "x2vec". An abstract version of the previous slide. At the left, a blue (light sky-blue) filled blob shaped like a three-armed "Y" with black outline, bold $\mathbf{x}$ inside its upper arm, headed "Data space" in bold above. A curved black line (no arrowhead) runs from $\mathbf{x}$ into a trapezoid (wide at left, narrow at right) containing an italic $f$; a dotted arrow from the word "Encoder" (below) points up to the trapezoid. From the trapezoid a black arrow with a filled head points to a bold $\mathbf{z}$ inside a salmon (pink-orange) filled circle with black outline, headed "Representation space" in bold above, with a dotted arrow from that heading to the circle's top edge. A dotted arrow from the word "Embedding" (below the circle) points up to the $\mathbf{z}$.

## Slide 6 — Two different ways to represent a function

Title: "Two different ways to represent a function". A single plot at the left of centre: axes drawn with arrowheads, vertical axis labelled with a bold $\mathbf{y}$ and ticks printed "0.5" and "1", horizontal axis labelled with a bold $\mathbf{x}$ and ticks "0.5" and "1". A thin black straight line runs from the origin diagonally to the point (1, 1), the identity function. The right half of the slide is empty.

## Slide 7 — Two different ways to represent a function

![Slide 7 — Two different ways to represent a function](../images/11-representation-learning-reconstruction-based/slide-7.png)

Title: "Two different ways to represent a function". Left, the plot of slide 6 with axes now labelled with typewriter-bold $\mathbf{x}_ {\texttt{out}}$ (vertical) and $\mathbf{x}_ {\texttt{in}}$ (horizontal), ticks 0.5 and 1 on each. Eight black dots sit on the horizontal axis at x = 0.1, 0.2, …, 0.8. From each dot a thin dashed vertical line goes up to the diagonal line, and from there a dashed horizontal line goes left to the vertical axis, ending in a left-pointing black arrowhead on the axis.

In the middle, a double right arrow, $\Rightarrow$. Right: the same function drawn as a mapping between two number lines. A horizontal line at the bottom with arrowheads at both ends, with ticks "0", "0.5", "1" and the label $\mathbf{x}_ {\texttt{in}}$ beneath, carries nine black dots at 0.1, 0.2, …, 0.9. A horizontal line at the top, with an arrowhead at its right end only, ticks "0", "0.5", "1" and the label $\mathbf{x}_ {\texttt{out}}$ above, receives nine dashed vertical lines going straight up from the dots, each ending in an upward arrowhead on the top line. (Eight dots and nine dots, as drawn.)

## Slide 8 — Data transformations for a variety of neural net layers

![Slide 8 — Data transformations for a variety of neural net layers](../images/11-representation-learning-reconstruction-based/slide-8.png)

Title: "Data transformations for a variety of neural net layers". Four panels in a 2 × 2 grid. Each panel has a bottom number line (labelled $\mathbf{x}_ {\texttt{in}}$, arrowheads at both ends, five black dots) and a top number line (labelled $\mathbf{x}_ {\texttt{out}}$, arrowhead at right), with a dashed arrow from each bottom dot to a point on the top line, and a boxed equation to the left of the arrows.

- Top left, boxed equation $\mathbf{x}_ {\texttt{out}} = 2\mathbf{x}_ {\texttt{in}}$. Ticks −1, −0.5, 0, 0.5, 1 on both lines. The five dots sit at −0.5, −0.25, 0, 0.25, 0.5 on the bottom; the dashed arrows fan outward to land at −1, −0.5, 0, 0.5, 1 on the top.
- Top right, boxed equation $\mathbf{x}_ {\texttt{out}} = \frac{\mathbf{x}_ {\texttt{in}} + 1}{2}$. Ticks −1, −0.5, 0, 0.5, 1. The dots sit at −1, −0.5, 0, 0.5, 1; the arrows slant right and squeeze together, landing at 0, 0.25, 0.5, 0.75, 1.
- Bottom left, boxed equation $\mathbf{x}_ {\texttt{out}} = \texttt{relu}(\mathbf{x}_ {\texttt{in}})$. Ticks −1, −0.5, 0, 0.5, 1. The dots sit at −1, −0.5, 0, 0.5, 1; the arrows from −1, −0.5 and 0 all land at 0 on the top line (the first two slant to it), and the arrows from 0.5 and 1 go straight up to 0.5 and 1.
- Bottom right, boxed equation $\mathbf{x}_ {\texttt{out}} = \texttt{sigmoid}(\mathbf{x}_ {\texttt{in}})$. The top ticks are −1, −0.5, 0, 0.5, 1 and the bottom ticks −5, −2.5, 0, 2.5, 5. The five bottom dots are evenly spaced but more tightly than the printed labels, so the outer ones sit inside the "−5" and "5" labels (at about ±4.3 and ±2.2 on the printed scale). The arrows land at about 0, 0.07, 0.5, 0.93 and 1: the first two close together at and just right of 0, the middle one at 0.5, and the last two close together just left of and at 1.

## Slide 9 — (no title; wiring graph, equation and mapping of four layers)

![Slide 9 — (no title; wiring graph, equation and mapping of four layers)](../images/11-representation-learning-reconstruction-based/slide-9.jpg)

No title is printed. A table-like figure with three column headings in serif type, "Wiring graph", "Equation" and "Mapping", and four rows, one per layer type. At the top left a legend of two coloured boxes: a salmon-red box labelled "Activations" and a sky-blue box labelled "Parameters". Each row has a rotated rounded label (typewriter type) at the left naming the layer. Activations are written in red and parameters in blue.

The Mapping column holds two pictures in every row, with a black vertical arrow labelled $\mathbf{x}_ {\texttt{in}}$ (bottom) and $\mathbf{x}_ {\texttt{out}}$ (top) between them: at the left, a regular grid of grey points on the lower plane and where the layer sends it on the upper plane; at the right, the same for a red Gaussian cloud of points. The wiring graphs' circles are white with black outlines; only the $\mathbf{x}_ {\texttt{in}}$ and $\mathbf{x}_ {\texttt{out}}$ labels are red.

- Row "linear". Wiring graph: three circles in a column on the left (labelled $\mathbf{x}_ {\texttt{in}}$ above) and three on the right ($\mathbf{x}_ {\texttt{out}}$ above), every left circle joined to every right circle by an arrow, with a box labelled with a blue bold $\mathbf{W}$ in the middle of the crossing arrows; below, a fourth circle labelled "1" on the left, joined by arrows to the right circles through a box labelled with a blue bold $\mathbf{b}$. Equation: $\mathbf{x}_ {\texttt{out}} = \mathbf{W}\mathbf{x}_ {\texttt{in}} + \mathbf{b}$ (x terms red, $\mathbf{W}$ and $\mathbf{b}$ blue). Mapping: two pictures of two stacked diamond (square-in-perspective) planes joined by dotted lines. At left, the grey grid on the lower plane is mapped to a large sheared quadrilateral that reaches far beyond the upper plane to the left. At right, red dotted lines from a cloud of red points on the lower plane run to a skewed cluster of red points on the upper plane, with a dashed grey sheared outline of the transformed frame.
- Row "relu". Wiring graph: four circles in a column labelled $\mathbf{x}_ {\texttt{in}}$ and four in a column labelled $\mathbf{x}_ {\texttt{out}}$, each input joined to its own output by one horizontal arrow. Equation: $x_{\texttt{out}}[i] = \max(x_{\texttt{in}}[i], 0)$ (the $x$ terms red). Mapping: the grey grid and the red cloud are each folded into a corner, a V-shaped crease at the upper plane's back corner.
- Row "L2-norm". Wiring graph: four input and four output circles with a horizontal arrow from each input to its output, plus curved arrows from every input into an extra circle at the bottom labelled $\lVert \mathbf{x}_ {\texttt{in}} \rVert$ (red), and curved arrows from that circle to each output. (The node's norm bars are black and carry no subscript 2.) Equation: $x_{\texttt{out}}[i] = \frac{x_{\texttt{in}}[i]}{\lVert \mathbf{x}_ {\texttt{in}} \rVert_ 2}$ (italic $x$ in the numerator, bold $\mathbf{x}$ with the subscript 2 in the denominator). Mapping: the grey grid and the red cloud are each projected onto a ring of points on the upper plane.
- Row "softmax". Wiring graph: as for L2-norm (four inputs, four outputs, an extra circle at the bottom with curved arrows in and out), the extra circle labelled with a sum, $\sum_i e^{-\mathbf{x}_ {\texttt{in}_ i}}$, with a bold $\mathbf{x}$, the summation index $i$ under the $\Sigma$ and no upper limit, a minus sign and no $\tau$; only the $\mathbf{x}_ {\texttt{in}_ i}$ is red. Equation: $x_{\texttt{out}}[i] = \frac{e^{-\tau x_{\texttt{in}}[i]}}{\sum_{k=1}^{K} e^{-\tau x_{\texttt{in}}[k]}}$, with a minus sign and $\tau$ in both exponents, the $\tau$ in blue, $x_{\texttt{in}}$ red and italic (not bold). Mapping: the grey grid and the red cloud are each squashed onto a short straight row of points near the upper plane's back edge.

## Slide 10 — MLP

![Slide 10 — MLP](../images/11-representation-learning-reconstruction-based/slide-10.jpg)

Title: "MLP" (top left). Four tall stacks side by side, showing the same network at four moments of training, labelled at the bottom with "Training iteration" and a long black arrow pointing right, with "…" breaks in the arrow at about one third and two thirds of its length (between the second and third stacks and between the third and fourth). Each stack is seven stacked diamond planes (the input and the output of each of six layers) with dashed lines joining the point clouds on successive planes. At the far left of the first stack, a vertical column of six rounded boxes with upward arrows, from the bottom to the top: "linear", "relu", "linear", "relu", "linear", "softmax". The dots are red or blue (two classes). In the first stack the clouds are the initial ones: red points concentrated at the centre, blue ones around them, and the top plane's red and blue points mixed on a short line segment. In the later stacks the dashed lines bend and shift as the layers stretch the data: by the third stack the red and blue points on the top line are partly separated (red at the left, blue at the right), and in the fourth stack they are mostly separated, with the blue points' dashed lines fanning out far to the right (the longest reaching beyond the plane's right corner), while the red points stay clustered towards the left, grading through purple to blue. In stacks 2–4, grey dashed lines with small grey and black corner markers also trace each plane's transformed frame, swinging far out to the left (furthest in the fourth stack) and to the right.

## Slide 11 — MLP

![Slide 11 — MLP](../images/11-representation-learning-reconstruction-based/slide-11.jpg)

Title: "MLP" (top left). One tall stack, centred, of the same seven diamond planes with red and blue point clouds and dashed lines as in the first stack of slide 10, at about the same size (the embedded plot is about 4% wider than slide 10's) and drawn in a plain plotting style. Small text at the top: "Loss: 0.71". To the right of the stack, small black labels in plain sans-serif type, set in the gaps between planes, from the top to the bottom: "softmax", "linear", "relu", "linear", "relu", "linear". The top plane shows a short line of mostly blue points with some purple, in the middle.

## Slide 12 — (no title; SGD against steepest descent in spectral norm)

![Slide 12 — (no title; SGD against steepest descent in spectral norm)](../images/11-representation-learning-reconstruction-based/slide-12.jpg)

No title is printed. Two stacks of the same seven diamond planes, left and right, both with "Loss: 0.71" printed small at the top and the layer labels "softmax", "linear", "relu", "linear", "relu", "linear" (top to bottom) to the right of each stack.

- Left, large text at the upper left: "SGD" and "(lr=0.01)".
- Right, large text at the top centre: "Steepest descent" "in spectral norm" "(see pset 2)" "(lr=0.002)", set to the left of the right stack, level with its top two planes.

At the bottom: "Code to make these: https://colab.research.google.com/drive/1VBw_HOQg6J2HCgozEO-ktUM_KSFaYVUD?usp=sharing". The printed slide number "12" overprints the "e/" at the end of "drive/" in this URL.

## Slide 13 — CLIP

Title: "CLIP", with "[Radford\*, Kim\* et al., ICML 2021]" beneath it. Centre: a stack of five diamond planes (the input, then four outputs) with many coloured points (about ten or more different colours, not just red and blue) joined between planes by thin dotted coloured lines. On the bottom plane the points are bunched at the centre; moving up, the lines wander and the points spread and, on the top planes, cluster by colour (colours grouped at the upper left, upper middle and upper right). To the left of the stack, four rounded boxes in typewriter type, each reading "ViT block x3", with a black up arrow through each, one per gap between planes, from the bottom to the top. Right: a salmon filled circle at the top, a sky-blue filled "Y"-shaped blob at the bottom, and a long black upward arrow from the blob to the circle, with text beside the arrow: "maps from complex data space to simple embedding space".

## Slide 14 — (no title; a brain and the visual hierarchy)

No title is printed. Left: a black line drawing of the side profile of a human head facing left, with the brain drawn inside it. Right, a labelled diagram of the visual cortex hierarchy, with the citation "[Serre, 2014]" at the lower right. From top to bottom it shows:

- At the top, four small illustrations in a 2 × 2 block with a label at the left, "Classification units": a deer on grass, a bird on a twig, a fox on grass and a coiled snake.
- A layer labelled "PIT/AIT" in cyan: two dashed cyan circles, empty, each with an arrow up toward the fox picture.
- A layer labelled "V4/PIT" in green: two solid green circles, empty, both with arrows up to the right cyan circle.
- A layer labelled "V2/V4" in yellow: five small circles with yellow outlines (icons: a squarish pinwheel, a clock-hand shape, a three-pronged star, a three-pronged star, a dark swirl) and, above them, two dashed green circles holding a three-pronged star and a black-and-white spiral; arrows lead up.
- A layer labelled "V1/V2" in red: four dashed red circles holding oriented bars (horizontal, rising diagonal, vertical, falling diagonal), above two groups of four solid red-outlined circles with the same four bars; dotted arrows join them.
- At the bottom, an illustration of a fox among grass, with a large red dot on its shoulder and chest and two red arrows up to the two groups of oriented-bar circles: the right arrow starts at the red dot and ends at the right group's second circle (rising diagonal); the left arrow starts on the fox's back near its hindquarters, well left of the dot, and ends at the left group's third circle (vertical bar).

The arrows converge on one path up the hierarchy: dotted arrows run from the fourth (falling-diagonal) circle of each solid-red group to the fourth dashed red circle; solid arrows from the third and fourth dashed red circles go to the fourth yellow circle; the third and fourth yellow circles feed the dashed green three-pronged-star circle; both dashed green circles feed the right solid green circle; both solid green circles feed the right cyan circle; and both cyan circles point up to the fox picture.

The notice sits at the lower left, below the brain drawing and far from the diagram: "© Springer Science+Business Media, LLC, part of Springer Nature. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It is taken to cover the Serre diagram (the citation "[Serre, 2014]" is beside it); it does not sit beside either figure.

*OCW notice: © Springer Science+Business Media, LLC, part of Springer Nature (the Serre visual-hierarchy diagram). All rights reserved — excluded from the CC license.*

## Slide 15 — What do deep nets internally learn?

Title: "What do deep nets internally learn?". Left: the car photograph of slide 4 (the orange car on a cobbled street), labelled with a bold capital $\mathbf{X}$ above and "Image" below. A thick black arrow points right to the three columns of circles of slide 4 (10, 7 and 5 circles, with the same shades), joined by short arrows; after the third column, an arrow to a "…" and then a further arrow to the word "Car" in typewriter type. Added over slide 4: the "…" and "Car" output; the right-hand building/car/road cards, the dotted callouts and the bottom text are gone.

The notice sits under the photograph and its "Image" label, at the lower left: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the car photograph). All rights reserved — excluded from the CC license.*

## Slide 16 — Deep Net "Electrophysiology"

Title: Deep Net “Electrophysiology” (with curly quotation marks). Left: the car photograph of slides 4 and 15 (the small orange car, cropped slightly differently, with no label). A thick black arrow points right to three empty (white, circle-free) boxed columns, a tall one, a shorter one and a short one, joined by short arrows as on slide 15, followed by "…", an arrow and "Car" in typewriter type. In the lower half of the second column there is a small empty circle; two thin black lines run from this circle down and to the left to the top-right corner of a framed rectangle at the bottom left (one ends just inside the frame below its top edge, the other on its right edge; neither reaches the top-left corner). The frame holds a thin grey-black trace of a neural recording: a wobbling noisy baseline near the bottom with seven tall thin vertical spikes rising from it at irregular spacing (the fifth spike is fainter and the last two are spaced further apart, the sixth and seventh standing alone to the right). At the lower right, two citations, right-aligned: "[Zeiler & Fergus, ECCV 2014]" and "[Zhou, Khosla, Lapedriza, Oliva, Torralba., ICLR 2015]" (with the stray full stop after "Torralba" as printed).

The notice sits under the car photograph at the lower left, above the framed trace: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It sits beside the photograph; whether it also covers the spike trace is not stated.

*OCW notice: © source unknown (the car photograph, and possibly the spike-train inset). All rights reserved — excluded from the CC license.*

## Slide 17 — Visualizing and Understanding CNNs

Title: "Visualizing and Understanding CNNs", with "[https://arxiv.org/pdf/1311.2901]" beneath it at the right of centre. Left, centred text: "Image patches that activate several of the **layer 1** neurons most strongly" (the words "layer 1" in bold). Right: a square mosaic on a black background, made of nine blocks in a 3 × 3 arrangement, each block a 3 × 3 grid of small low-resolution colour patches (81 patches in all). Each block shows what one neuron responds to: row one, a block of grey and black diagonal edges, a block of white-and-red or pink-and-grey diagonal edges, and a block of yellow-green vertical stripes; row two, a block of orange and amber flat colour, a block of dark diagonal edges in pink, brown and black, and a block of light and dark diagonal edges with blue, green and white; row three, a block of checkered and diagonal grey, green and black edges, a block of flat dark and lighter greens, and a block of red, orange and pink flat colour.

The notice sits at the lower left of the mosaic, beside it: "© Zeiler and Fergus. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Zeiler and Fergus (the layer 1 patch mosaic). All rights reserved — excluded from the CC license.*

## Slide 18 — (no title; "[Zeiler and Fergus, 2014]" and the layer 2 patches)

No title is printed; the text at the top centre is the citation "[Zeiler and Fergus, 2014]". Left, centred text: "Image patches that activate several of the **layer 2** neurons most strongly" ("layer 2" in bold). Right: a square mosaic of 16 blocks in a 4 × 4 arrangement on a dark grey background, each block a 3 × 3 grid of colour image patches (144 patches). The patches are larger than on slide 17 and show textures and simple shapes: the first row of blocks shows mesh and wavy dark-and-teal patterns, vertical pleats and gold stripes, horizontal bands of brown, and red and brown stripes; the second row orange gradient surfaces, circular rings, round dark discs (lens, eye-like), and curved and striped arcs in blue, grey and yellow; the third row pale-blue sky with thin lines, cyan and black blobby textures, grey angled planes, and olive and white curved shapes; the fourth row grey fur, solid yellow, horizontal ribbed and striped surfaces, and window frames with reds and blues.

The notice sits at the lower left, beside the mosaic: "© Zeiler and Fergus. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Zeiler and Fergus (the layer 2 patch mosaic). All rights reserved — excluded from the CC license.*

## Slide 19 — (no title; "[Zeiler and Fergus, 2014]" and the layer 3 patches)

No title is printed; the top centre text is "[Zeiler and Fergus, 2014]". Left, centred text: "Image patches that activate several of the **layer 3** neurons most strongly" ("layer 3" in bold). Right: a wide mosaic of 12 blocks in a 4 × 3 arrangement (four across, three down), each block a 3 × 3 grid of image patches (108 patches), larger than slide 18's. Row one of blocks: lattices and honeycomb patterns (wire mesh, leopard-like spots, a red-and-black honeycomb); hands, legs and objects on white backgrounds; people's skin, shoulders and a daisy, with two birds' heads (a green bee-eater and a black bird); birds and animals on green backgrounds. Row two: vertical posts, railings and window bars on coloured grounds (an orange-and-red wall with a pole, a blue-and-white tile, a bollard, green frames); wheels and the sides of cars and trucks; vehicle fronts and street scenes; and signs and lettering ("STRIKE FACE", "morning", a red sign with the word "ORDINANCE"). Row three: orange objects, a corn tin and a pink-and-grey object; ladybirds and persimmon-like orange fruits; people, one playing a saxophone, men and women outdoors and by a beach; white, lumpy textures such as coral and foam.

The notice sits at the lower left, beside the mosaic: "© Zeiler and Fergus. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Zeiler and Fergus (the layer 3 patch mosaic). All rights reserved — excluded from the CC license.*

## Slide 20 — (no title; "[Zeiler and Fergus, 2014]" and the layer 5 patches)

No title is printed; the top centre text is "[Zeiler and Fergus, 2014]". Left, centred text: "Image patches that activate several of the **layer 5** neurons most strongly" ("layer 5" in bold). Right: a square mosaic of 36 colour photographs in a 6 × 6 grid. The upper left 3 × 3 quadrant holds spiky or radial flowers and seed-heads (orange, yellow, pink, purple, red, a daisy); the upper right quadrant holds three cars, four women (two in tops, black and red, and two smiling faces) and two dogs on grass; the lower rows are almost all dogs' faces and heads (collies, terriers, small dogs, a white wolf-like dog, a dark dog), with a few exceptions: a black wrought-iron bench at row five, column six, and a purple object beside a terrier at row six, column one.

The notice sits at the lower left, beside the mosaic: "© Zeiler and Fergus. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Zeiler and Fergus (the layer 5 patch mosaic). All rights reserved — excluded from the CC license.*

## Slide 21 — (no title; the visual hierarchy beside the CNN's patches)

No title is printed. Left: the Serre diagram of slide 14 without the brain (Classification units: deer, bird, fox, snake; PIT/AIT, V4/PIT, V2/V4 and V1/V2 layers in cyan, green, yellow and red; the fox with its red dot), with "[Serre, 2014]" beside its lower right. Right: three small image grids stacked vertically with a thick black up arrow between each pair. From the bottom: a 3 × 3 grid of layer 1 patches (the white-and-red diagonal-edge block of slide 17's first row); a 3 × 3 grid of layer 2 patches (rings, swirls, striped and yellow shapes, from slide 18's second row, fourth block); a 3 × 3 grid of dogs' faces at the top. "[Zeiler & Fergus, ECCV 2014]" sits at the lower right.

One notice block, at the bottom centre between the two figures and overlapping the printed slide number: "Left © Springer Science+Business Media, LLC, part of Springer Nature. Right © Zeiler and Fergus. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". The two holders are named in this single block: "Left" is the Serre diagram (Springer Nature) and "Right" the three patch grids (Zeiler and Fergus).

*OCW notice: Left © Springer Science+Business Media, LLC, part of Springer Nature (the Serre diagram); Right © Zeiler and Fergus (the three patch grids). All rights reserved — excluded from the CC license.*

## Slide 22 — What is a representation?

Title: "What is a representation?". Three paragraphs:

Mainly, we will restrict our attention to **vector embeddings**

A representation of a data domain $\mathcal{X}$ is a function $f : \mathcal{X} \to \mathbb{R}^d$ that assigns a feature vector to each input in that domain. This function is called an **encoder**.

A representation of a datapoint $\mathbf{x}$ is a vector $\mathbf{z} \in \mathbb{R}^d$ with $\mathbf{z} = f(\mathbf{x})$.

(The words "vector embeddings" and "encoder" are in bold; the math is set in a serif font.)

## Slide 23 — Why learn representations?

Section divider: the single line "Why learn representations?" centred on a white slide.

## Slide 24 — To do more learning! (aka Transfer learning)

Title: "To do more learning! (aka **Transfer learning**)" (the words "Transfer learning" in bold, "(aka" smaller). A quotation: "“Generally speaking, a good representation is one that makes a subsequent learning task easier.” — *Deep Learning*, Goodfellow et al. 2016" (the title "Deep Learning" in italics).

Below, centred: a photograph of an empty dark chalkboard with a wooden frame, leaning against a turquoise wooden wall, standing on a wooden floor, with a large hand-drawn red X over it from corner to corner (the red strokes overshoot the photograph's edges).

The notice sits under the photograph at its lower left: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the chalkboard photograph). All rights reserved — excluded from the CC license.*

## Slide 25 — (no title; Training on genre recognition, Testing on preference prediction)

![Slide 25 — (no title; Training on genre recognition, Testing on preference prediction)](../images/11-representation-learning-reconstruction-based/slide-25.jpg)

No title is printed. Two panels side by side.

- Left panel: a grey box "Training" at the top, then "Genre recognition". A black music-note icon (two beamed quavers), then a large trapezoid (taller at the left, narrower at the right) containing an italic $f$, then a small grey vertical bar labelled with a bold $\mathbf{z}$, then an arrow to a square box containing a bold $\mathbf{W}$, then an arrow to a boxed column of six circles with typewriter labels: "classical" (mid grey circle), "hip hop" (light grey), "rock" (white), "metal" (dark slate grey), "alternative" (mid grey), "rap" (black). A dotted arrow from the word "Encoder" points to the trapezoid, and a dotted arrow from "Prediction Head" (two lines) points to the $\mathbf{W}$ box.
- Right panel: a grey box "Testing", then "Preference prediction". The same chain of note icon, trapezoid $f$, bar $\mathbf{z}$, arrow, box $\mathbf{W}$, arrow, with a boxed column of three circles labelled "Like" (light grey), "Neural" (dark grey; spelled as printed) and "Dislike" (white). No dotted labels.

Bottom, centred: "Often, what we will be “tested” on is not what we were trained on."

## Slide 26 — (no title; linear adaptation)

![Slide 26 — (no title; linear adaptation)](../images/11-representation-learning-reconstruction-based/slide-26.jpg)

No title is printed. As slide 25, but the right grey box now reads "Adapting" in place of "Testing". Left panel ("Training", "Genre recognition"), as on slide 25 (note, $f$, $\mathbf{z}$, $\mathbf{W}$, the six genre circles, "Encoder" and "Prediction Head" callouts), now with a turquoise padlock icon drawn over the top of the trapezoid. Right panel ("Adapting", "Preference prediction"): note icon, trapezoid $f$ with a turquoise padlock over its top, bar $\mathbf{z}$, arrow, a box containing a bold $\mathbf{W}'$ drawn with a thick yellow outline, arrow, and the three circles "Like" (light grey), "Neural" (dark grey), "Dislike" (white). Added over slide 25: the two padlocks, the yellow $\mathbf{W}'$ box and the prime.

Bottom: "**Linear adaptation**: freeze f, train a new linear map to new target data" ("Linear adaptation" in bold).

## Slide 27 — (no title; finetuning)

![Slide 27 — (no title; finetuning)](../images/11-representation-learning-reconstruction-based/slide-27.jpg)

No title is printed. As slide 26 but without any padlock. Left panel ("Training", "Genre recognition"): as on slide 25. Right panel ("Adapting", "Preference prediction"): note icon, a trapezoid with a thick yellow outline containing an italic $f'$, bar $\mathbf{z}$, arrow, a yellow-outlined box with $\mathbf{W}'$, arrow, and the three circles "Like" (light grey), "Neural" (dark grey), "Dislike" (white). Added over slide 26: the trapezoid now holds $f'$ with a yellow outline in place of the padlocked $f$.

Bottom: "**Finetuning**: initialize f’ as f, then continue training on new target data" ("Finetuning" in bold; the prime is a typographic apostrophe).

## Slide 28 — (no title; pretraining, adapting and testing)

![Slide 28 — (no title; pretraining, adapting and testing)](../images/11-representation-learning-reconstruction-based/slide-28.png)

No title is printed. Three columns separated by two vertical dotted lines. Each column has a grey box heading, a task name, a note icon, a trapezoid, an arrow and a boxed column of circles with typewriter labels.

- Column 1: "Pretraining", "Genre recognition". Trapezoid with italic $f$; boxed column of six circles: "classical" (mid grey), "hip hop" (light grey), "rock" (white), "metal" (dark slate grey), "alternative" (mid grey), "rap" (black). Beneath, in italics: "A lot of data".
- Column 2: "Adapting", "Preference prediction". Trapezoid with $f'$; three circles: "Like" (light grey), "Neural" (dark grey), "Dislike" (white). Beneath, in italics: "A little data".
- Column 3: "Testing", "Preference prediction". Trapezoid with $f'$; three circles: "Like" (black), "Neural" (light grey), "Dislike" (dark grey). No caption beneath.

There is no $\mathbf{z}$ bar or $\mathbf{W}$ box on this slide.

## Slide 29 — Finetuning

Title: "Finetuning". Three bullets:

- Pretrain a network on task A, resulting in parameters $\mathbf{W}$ and $\mathbf{b}$
- Initialize a second network with some or all of $\mathbf{W}$ and $\mathbf{b}$
- Train the second network on task B, resulting in parameters $\mathbf{W}'$ and $\mathbf{b}'$

(The symbols are in bold sans-serif type; the primes are typographic.)

## Slide 30 — How do you learn a good representation?

Section divider: the single line "How do you learn a good representation?" centred on a white slide.

## Slide 31 — Representation learning

![Slide 31 — Representation learning](../images/11-representation-learning-reconstruction-based/slide-31.jpg)

Title: "Representation learning". Left, a numbered list headed "Good representations are:":

1. Compact (*minimal*)
2. Explanatory (*sufficient*)
3. Disentangled (*independent factors*)
4. Interpretable
5. *Make subsequent problem solving easy* (the whole item in italics)
6. …?

Right: the three flat shapes of slide 4: a pale beige card labelled "building" (typewriter type) at the back, a dark grey card labelled "road" in front of it, and, frontmost, an orange car silhouette labelled "car", whose wheels and lower body lie over the road card's top edge. They are vector drawings. At the bottom right: "[See “Representation Learning”, Bengio 2013, for more commentary]".

## Slide 32 — Learning from examples

Title: "Learning from examples", with the subtitle "(aka **supervised learning**)" (the words "supervised learning" in bold). A grey box labelled "Training data" at the upper left. Below it, a list of pairs in plain italic type: $\lbrace x^{(1)}, y^{(1)} \rbrace$, $\lbrace x^{(2)}, y^{(2)} \rbrace$, $\lbrace x^{(3)}, y^{(3)} \rbrace$ and "…". An arrow to a square box labelled "Learner"; an arrow from the box to $f : X \to Y$ (italic, plain). At the bottom, centred:

$$f^{\ast} = \arg\min_{f \in \mathcal{F}} \sum_{i=1}^{N} \mathcal{L}(f(\mathbf{x}^{(i)}), \mathbf{y}^{(i)})$$

(The pairs in the list are in plain italic $x$ and $y$, while the equation has bold $\mathbf{x}$ and $\mathbf{y}$, as printed.)

## Slide 33 — Learning without examples

Title: "Learning without examples", with the subtitle "(includes **unsupervised learning** / **self-supervised learning**)" (the two terms in bold). A grey box "Data" at the upper left; below it $\lbrace x^{(1)} \rbrace$, $\lbrace x^{(2)} \rbrace$, $\lbrace x^{(3)} \rbrace$ and "…"; an arrow to a square box "Learner"; an arrow from the box to a large question mark "?" in sans-serif type.

## Slide 34 — Representation Learning

Title: "Representation Learning". As slide 33: grey box "Data", $\lbrace x^{(1)} \rbrace$, $\lbrace x^{(2)} \rbrace$, $\lbrace x^{(3)} \rbrace$, "…", an arrow, the "Learner" box and an arrow, now to a stacked list at the right: "embeddings", "clusters", "metrics", "…". Added over slide 33: the output list replaces the question mark.

## Slide 35 — Two basic approaches: 1) compression, 2) prediction

Title: "Two basic approaches: 1) compression, 2) prediction". A table in serif type with three rules (booktabs style: a heavy top rule, a lighter rule under the header row, and a heavy bottom rule), three columns (bold headers: "Learning Method", "Learning Principle", "Short Summary"):

| Learning Method | Learning Principle | Short Summary |
| --- | --- | --- |
| Autoencoding | Compression | Remove redundant information |
| Contrastive | Compression | Achieve invariance to viewing transformations |
| Clustering | Compression | Quantize continuous data into discrete categories |
| Future prediction | Prediction | Predict the future |
| Imputation | Prediction | Predict missing data |
| Pretext tasks | Prediction | Predict abstract properties of your data |

Below the table, centred: "(Question: are these actually different?)".

## Slide 36 — Learning via compression

Title: "Learning via compression". Left: a bold capital $\mathbf{X}$ above a photograph, with "Image" below, of a yellow-and-black angelfish (yellow head, front and tail fin, black body) swimming over a coral reef with red and orange sponges. A thick black arrow points right to a diagram, at the centre: a salmon-red square labelled “Coral” (at its top) and overlapping its lower left a yellow fish-shaped silhouette labelled “Fish”, outlined in black. Below it: "Compact mental representation" (two lines).

The notice sits at the lower left, below the photograph: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the fish photograph). All rights reserved — excluded from the CC license.*

## Slide 37 — Learning via compression

Title: "Learning via compression". The fish photograph with bold $\mathbf{X}$ above and "Image" below, as on slide 36, then a thick black arrow to the three columns of circles of slide 4 (10, 7 and 5 circles, the same shades), each joined to the next by a short arrow. Above the third column, the text "compressed image code (vector $\mathbf{z}$)" ("z" bold), with a thick dotted arrow pointing down to the third column. Added over slide 36: the three columns and the label; the Coral/Fish cards and their caption are gone.

The notice sits at the lower left, below the photograph: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the fish photograph). All rights reserved — excluded from the CC license.*

## Slide 38 — Learning via compression

Title: "Learning via compression". Left, the fish photograph with bold $\mathbf{X}$ above and "Image" below; a thick arrow to five columns of circles in boxes, joined by short arrows: 10, 7, 5, 7 and 10 circles. The first three are as on slide 37. The fourth column (7 circles), top to bottom: light grey, black, dark slate grey, light grey, white, black, mid grey. The fifth column (10 circles), top to bottom: white, dark slate grey, black, black, light grey, mid grey, dark slate grey, very pale grey, dark slate grey, light grey. The text "compressed image code (vector $\mathbf{z}$)" sits above the third (middle) column with a thick dotted arrow to it. A thick arrow leads from the fifth column to a second copy of the fish photograph at the right, labelled with a bold capital X with a hat, $\hat{\mathbf{X}}$, above and "Reconstructed image" (two lines) below. Beneath the columns, in large type: “Autoencoder” (in quotation marks). At the lower right: "[e.g., Hinton & Salakhutdinov, Science 2006]".

The notice sits at the lower left, below the left photograph: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the fish photograph, shown twice). All rights reserved — excluded from the CC license.*

## Slide 39 — Autoencoder

![Slide 39 — Autoencoder](../images/11-representation-learning-reconstruction-based/slide-39.jpg)

Title: "Autoencoder". A diagram in serif type, left to right: a sky-blue "Y"-shaped blob (labelled "Data space" above, with a dotted arrow to it) holding a bold $\mathbf{x}$ in its upper arm; a curved line (no arrowhead) from $\mathbf{x}$ into a trapezoid labelled $f$ (dotted arrow from the bold word "Encoder" below); an arrow from the trapezoid to a bold $\mathbf{z}$ at the left edge of a salmon circle (labelled "Representation space" above, with a dotted arrow to it); a long curved arrow from $\mathbf{z}$ through a second trapezoid labelled $g$ (with a dotted arrow from the bold word "Decoder" below), which then runs on, curving up, to a bold $\hat{\mathbf{x}}$ in the upper arm of a second blue "Y" blob (labelled "Data space" above, with a dotted arrow to it). In that blob, just below $\hat{\mathbf{x}}$, is a bold $\mathbf{x}$, with a short red dotted bracket (an I-beam with solid end caps) between the two, labelled "Reconstruction error" (two lines) at the right with a dotted arrow to the bracket. At the bottom, the objective:

$$f^{\ast}, g^{\ast} = \arg\min_{f, g} \mathbb{E}_ {\mathbf{x}} \lVert \mathbf{x} - g(f(\mathbf{x})) \rVert_2^2$$

## Slide 40 — (no title; the L2 autoencoder)

No title is printed. The notice is at the top left of the slide. Centre: a framed box with a grey banner reading $L_2$ autoencoder (only "autoencoder" is bold; the $L$ is an ordinary italic with an upright subscript 2), and inside it, centred, in serif type:

Objective

$$\mathcal{L}(F(\mathbf{x}), \mathbf{x}) = \lVert F(\mathbf{x}) - \mathbf{x} \rVert_2^2$$

Hypothesis space

$$F = g \circ f : \mathbb{R}^N \to \mathbb{R}^M \to \mathbb{R}^N$$

At the left, "Data" and $\lbrace \mathbf{x}^{(i)} \rbrace_{i=1}^{n}$ with an arrow into the box; at the right of the box, an arrow to an italic $f$. Beneath the box, "Typically, M<N" in sans-serif type with a dotted arrow pointing up to the middle $\mathbb{R}^M$.

At the far right, a vertical stack: a photograph of an orange-yellow bird perched on a branch against dark green spiky leaves, with a bold $\hat{\mathbf{x}}$ above it; below it a trapezoid, wide at the top, labelled $g$; then a small grey horizontal bar labelled with a bold $\mathbf{z}$ at its right; then a trapezoid, wide at the bottom, labelled $f$; then a second copy of the bird photograph with a bold $\mathbf{x}$ below it.

The notice, at the top left: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It sits far from the bird photographs, which are at the right, and is taken to cover them.

*OCW notice: © source unknown (the bird photograph, shown twice). All rights reserved — excluded from the CC license.*

## Slide 41 — (no title; the L2 autoencoder and PCA)

No title is printed. As slide 40 (the same $L_2$ autoencoder box with the same objective and hypothesis space, "Data" $\lbrace \mathbf{x}^{(i)} \rbrace_{i=1}^{n}$, the arrows to $f$, and the stack of the bird photograph, $g$, $\mathbf{z}$ bar, $f$ and the second photograph at the far right), but everything is shifted slightly up and the dotted arrow and "Typically, M<N" are gone. Added over slide 40: at the bottom, two lines of text: "What if f and g are both linear?" and "Then the embedding spans the same M-dimensional subspace as PCA".

The notice is at the top left, as on slide 40: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the bird photograph, shown twice). All rights reserved — excluded from the CC license.*

## Slide 42 — Quick experiment

![Slide 42 — Quick experiment](../images/11-representation-learning-reconstruction-based/slide-42.png)

Title: "Quick experiment". Left: "Data" above a 5 × 5 grid of black square tiles separated by thin white lines, each tile holding one small solid-coloured shape. Row 1: a pink triangle, a green circle, a salmon-red triangle, a magenta-pink circle, an orange tilted square. Row 2: a blue square, a small pink-red circle, a teal diamond, a purple tilted square, a purple diamond. Row 3: a blue triangle, a small teal circle, an orange circle, a small salmon triangle, a blue-violet square. Row 4: a small lavender tilted square, an orange diamond, a small dark-olive diamond, an olive-green circle, a teal triangle. Row 5: a small green triangle, a green triangle, a blue circle, an olive square, a purple square.

An arrow leads to a framed box, as on slide 40: banner reading $L_2$ autoencoder, Objective $\mathcal{L}(F(\mathbf{x}), \mathbf{x}) = \lVert F(\mathbf{x}) - \mathbf{x} \rVert_2^2$, Hypothesis space $F = g \circ f : \mathbb{R}^N \to \mathbb{R}^M \to \mathbb{R}^N$; an arrow leaves the box to an italic $f$. This slide prints no notice.

## Slide 43 — (no title; nearest neighbours in z-space and the accuracy of colour and shape)

![Slide 43 — (no title; nearest neighbours in z-space and the accuracy of colour and shape)](../images/11-representation-learning-reconstruction-based/slide-43.jpg)

No title is printed. Three parts.

Top left: a bold $\mathbf{x}$ above a small black tile holding an olive-green tilted square; a curved arrow into a trapezoid $f$; an arrow to a bold $\mathbf{z}$ in a salmon circle.

Bottom left: the headings "query" and "nearest-neighbors in z-space". Under "query", a column of five query tiles: a pink triangle, a red-orange circle (the same size as the fourth), a purple-blue tilted square, a red-orange circle, a teal triangle. To the right, a 5 × 5 grid of tiles, one row of five neighbours for each query: row 1 triangles (pink, orange, magenta, salmon-red, red); row 2 circles (orange, orange, pink-red, magenta-purple, orange-brown); row 3 squares and a diamond (grey-blue, blue, blue, teal, blue-violet); row 4 circles (brown-orange, pink, brown-red, red-orange, orange); row 5 triangles (green, blue, green, green, teal).

Right: a line chart. The vertical axis is labelled "accuracy (%)" with ticks 60, 70, 80, 90, 100; the horizontal axis is labelled "layer" with ticks 0, 1, 2, 3, 4, 5, 6. Two series, each with dot markers, labelled by boxed words on the curves:

- A dashed curve labelled "shape" (dotted box), rising: about 75 at layer 0, 76 at 1, 81.5 at 2, about 82.5 at 3 (the marker is hidden behind the label), 85 at 4, 89.5 at 5, 99 at 6.
- A solid curve labelled "color" (solid box), falling: about 73 at layer 0, 69 at 1, 66 at 2, about 68.5 at 3 (marker hidden behind the label), 67.5 at 4, 63.5 at 5, 58 at 6.

Beneath the chart, a trapezoid labelled $f$ divided by five dotted vertical lines into six sections; six curved arrows leave its upper edge, one from each section, and point up to the axis ticks 1, 2, 3, 4, 5 and 6 respectively (the arrow to tick 3 ends below the word "layer").

## Slide 44 — Clustering

![Slide 44 — Clustering](../images/11-representation-learning-reconstruction-based/slide-44.png)

Title: "Clustering". Two rows separated by a horizontal rule. Row 1, with the rotated label "Training" at the left: a framed box labelled "Data" above, holding 18 black dots in 2D (six in an upper-left group, two in the upper right, ten in a lower-right group); an arrow to a square "Learner" box; an arrow to "Encoder" above $f : \mathcal{X} \to \lbrace 1, \ldots, k \rbrace$. A dotted vertical arrow runs from this expression down across the rule to the expression in the lower row. Row 2, with the rotated label "Inference": the same framed "Data" box (label below it) with the same 18 dots; a long arrow to $f : \mathcal{X} \to \lbrace 1, \ldots, k \rbrace$; an arrow to a framed box labelled "Clusters" below, holding the same 18 points coloured by group: six blue (upper left), two green (upper right), ten red (lower right).

## Slide 45 — Clustering — a rep learning perspective

Title: "Clustering — a rep learning perspective". Left, three rows, each with a bold-subscripted input label, a photograph, an arrow, a tall box with an italic $f$, an arrow, and a column of five small boxes with one filled black, and a quoted word:

- $\mathbf{x}_ 1$: a photograph of an orange bird on a branch against dark spiky leaves; output column labelled $a_1$ above, with the second box filled; the word “bird”.
- $\mathbf{x}_ 2$: a photograph of a blue parrot on a branch; output column labelled $a_2$, second box filled; the word “bird”.
- $\mathbf{x}_ 3$: a photograph of a golden pavilion beside a lake with trees and rocks; output column labelled $a_3$, the fifth (bottom) box filled; the word “temple”.

Right, four bullets: "What’s the best representation that humans have come up with so far?"; "Language!"; "Words are the atoms of language"; "Clustering is the problem of making up new words for things".

The notice sits at the lower left, under the third photograph: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It sits beside the photographs and is taken to cover all three.

*OCW notice: © source unknown (the three photographs). All rights reserved — excluded from the CC license.*

## Slide 46 — k-means

![Slide 46 — k-means](../images/11-representation-learning-reconstruction-based/slide-46.jpg)

Title: "k-means". Two bullets:

- Map datapoints to integers (i.e. cluster)
- In such a way that each datapoint is as close as possible to the mean of the cluster it is assigned to

Below, two scatter plots joined by a black arrow pointing right. Both have axes labelled $x_1$ (horizontal) and $x_2$ (vertical) with ticks −7.5, −5.0, −2.5, 0.0, 2.5, 5.0, 7.5 on each axis.

- Left: 500 black dots in four visible blobs (about 200 in the upper-left one and about 100 in each of the others): a large loose one in the upper left (centred near (−5, 5)), a tight one at the lower left (near (−4, −6)), a tight one in the middle (near (2, 0)) and a looser one at the right (near (6, 4)).
- Right: the same data, coloured into five clusters, each with a large cross marker (a colour-filled X with a black outline inside a thin white halo) at its centre; about 100 points per cluster. The large upper-left blob is split into a blue cluster (cross near (−4, 6)) and a yellow cluster (cross near (−6, 4)). The lower-left blob is orange (cross near (−4, −6)), the middle blob dark red (maroon; cross near (2, 0)) and the right blob red (cross near (6, 4)).

## Slide 47 — k-means

![Slide 47 — k-means](../images/11-representation-learning-reconstruction-based/slide-47.jpg)

Title: "k-means". Two bullets:

- Map datapoints to integers (i.e. cluster)
- In such a way that each datapoint is as close as possible to its cluster’s code mean

Below, a row of figures, left to right:

- The black scatter plot of slide 46 (axes $x_1$ and $x_2$, ticks −7.5 to 7.5).
- A trapezoid (tall at the left, narrow at the right) labelled with an italic $f$, with a dotted arrow from the bold word "Encoder" below it.
- Three columns of five stacked empty boxes, each with one box filled black, at different heights: the first (upper) column has its second box filled; the second (lowest) column has its first (top) box filled; the third (middle-height) column has its fourth box filled. They stand for one-hot codes.
- A trapezoid (narrow at the left, tall at the right) labelled with an italic $g$, with a dotted arrow from the bold word "Decoder" below it.
- A scatter plot with the same axes holding only the five cluster-centre crosses of slide 46: blue near (−4, 6), yellow near (−6, 4), red near (6, 4), maroon near (2, 0) and orange near (−4, −6).

## Slide 48 — k-means

Title: "k-means" (top left). A framed box with a grey banner reading "**k-means** ($L_2$)" (the word "k-means" in bold), and, inside, centred, in serif type:

Objective

$$\mathcal{L}(F(\mathbf{x}), \mathbf{x}) = \lVert F(\mathbf{x}) - \mathbf{x} \rVert_2^2$$

Hypothesis space

$$F = g \circ f : \lbrace \mathbf{x} \rbrace_{i=1}^{N} \to \lbrace 1, \ldots, k \rbrace \to \mathbb{R}^M$$

Optimizer

Block coordinate descent

At the left of the box, "Data" and $\lbrace \mathbf{x}^{(i)} \rbrace_{i=1}^{n}$ with an arrow into the box; at its right, an arrow to an italic $f$. At the bottom left: "f and g are both lookup tables".

## Slide 49 — VQ nets

Title: "VQ nets" (top left). A framed box with a grey banner "VQ Autoencoder ($L_2$)" and, inside:

Objective

$$\mathcal{L}(F(\mathbf{x}), \mathbf{x}) = \lVert F(\mathbf{x}) - \mathbf{x} \rVert_2^2 + \ldots$$

Hypothesis space

$$F = g \circ f : \lbrace \mathbf{x} \rbrace_{i=1}^{N} \to \lbrace 1, \ldots, k \rbrace \to \mathbb{R}^M$$

Optimizer

Backprop w/ approximations

At the left, "Data" and $\lbrace \mathbf{x}^{(i)} \rbrace_{i=1}^{n}$ with an arrow into the box; at the right, an arrow to an italic $f$. Below the box, two lines of text: "What if f and g are both deep nets?" and "Then we call this a **“Vector Quantized” Autoencoder** (e.g., VQVAE, VQGAN)" (the words “Vector Quantized” Autoencoder in bold). At the lower right: "[see e.g., Oord, Vinyals, Kavukcuoglu, 2017]".

## Slide 50 — Data compression

![Slide 50 — Data compression](../images/11-representation-learning-reconstruction-based/slide-50.png)

Title: "Data compression". Left, a large square frame holding a bold capital $\mathbf{X}$ with "Data" beneath it; a thick arrow right; five empty rectangles standing in a row as in slide 38 but with no circles (tall, shorter, shortest, shorter, tall), joined by thin arrows; a thick arrow right; a second square frame holding a bold capital $\hat{\mathbf{X}}$ with "Data" beneath it.

## Slide 51 — Label prediction

![Slide 51 — Label prediction](../images/11-representation-learning-reconstruction-based/slide-51.png)

Title: "Label prediction". Left, a square frame holding a bold capital $\mathbf{X}$, with "Data" beneath it; a thick arrow right; three empty rectangles in decreasing height (tall, shorter, shortest) joined by thin arrows; a thin arrow to an italic $y$ with "Label" beneath it.

## Slide 52 — Data prediction

![Slide 52 — Data prediction](../images/11-representation-learning-reconstruction-based/slide-52.png)

Title: "Data prediction", with the subtitle: aka “self-supervised learning”. Left, a square frame whose upper-left triangle (above the diagonal from top right to bottom left) is filled blue and holds a bold $\mathbf{X}_ 1$; the diagonal is a dotted line; the lower-right triangle is white. "Some data" beneath. A thick arrow; the five empty rectangles of slide 50 (tall, shorter, shortest, shorter, tall) joined by thin arrows; a thick arrow; a second square frame whose lower-right triangle is filled pale yellow and holds a bold $\hat{\mathbf{X}}_ 2$ (with a hat), the upper-left triangle white and a dotted diagonal between them. "Other data" beneath.

## Slide 53 — (no title; colorization, L channel to ab channels)

No title is printed. Top left: a grayscale photograph of the angelfish and coral reef of slide 36, captioned "Grayscale image: L channel" and $\mathbf{X} \in \mathbb{R}^{H \times W \times 1}$. At the centre, a large outlined right-pointing block arrow labelled with a script $\mathcal{F}$. Top right: a blurry pastel image, captioned "Color information: ab channels" and $\widehat{\mathbf{Y}} \in \mathbb{R}^{H \times W \times 2}$ (bold capital Y with a wide hat). It is mostly lilac-grey with an olive-yellow crescent where the fish's head and front are, an olive-yellow blob at the lower left where the tail fin is, teal-blue at the lower right, orange-pink at the upper left and green-grey at the upper right.

At the bottom centre: a box containing a bold sans-serif "L", an arrow, a network schematic of nine grey 3D slabs (a tall thick slab at each end, with the slabs in between thinning from the left to four very thin flat ones and growing again towards the right), an arrow, and a box containing a bold sans-serif "ab". At the lower right: "[Zhang, Isola, Efros, ECCV 2016]".

The notice sits at the lower left, below the grayscale photograph and its caption and beside the network schematic: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It is taken to cover the two photographs.

*OCW notice: © source unknown (the grayscale fish photograph and the colour-channel image). All rights reserved — excluded from the CC license.*

## Slide 54 — Deep Net "Electrophysiology"

Title: Deep Net “Electrophysiology” (curly quotation marks). Left: the grayscale fish photograph of slide 53; a thick arrow right; five empty rectangles (tall, shorter, shortest, shorter, tall) joined by thin arrows; a thick arrow right; the pastel ab-channel image of slide 53. In the lower part of the shortest, middle rectangle, a small empty circle with two thick black lines running down to the top corners of a framed rectangle at the bottom, which holds the spike trace of slide 16: a wobbling noisy baseline with seven tall thin vertical spikes at irregular spacing. At the top right, in the corner, a small portrait-format photograph: an orange and yellow microscope image of nerve cells, with dark branching cell bodies and fine fibres, and a dark needle-like electrode entering from the top right. At the lower right, two citations: "[Zeiler & Fergus, ECCV 2014]" and "[Zhou, Khosla, Lapedriza, Oliva, Torralba., ICLR 2015]".

The notice sits at the lower left, beside the trace and below the grayscale fish photograph: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It is far from the nerve-cell micrograph at the top right; which of the slide's pictures it covers is not stated.

*OCW notice: © source unknown (the pictures on the slide: the fish photographs, the spike trace and the nerve-cell micrograph). All rights reserved — excluded from the CC license.*

## Slide 55 — Stimuli that drive selected neurons (conv5 layer)

Title: "Stimuli that drive selected neurons (conv5 layer)". A grid of photographs, three rows of five, with a row label at the left of each: "faces", "dog faces" (two lines) and "flowers". Each photograph is darkened except for one bright, roughly circular region (the part of the image that drives the neuron).

- Row "faces": two smiling women side by side; a woman with long brown hair holding a cello; a boy with glasses (holding a cat, outside the bright patch); a woman in a dark jacket (a second small bright circle at the lower right); a bald man (holding a large fish, outside the bright patch).
- Row "dog faces": a white chow-like dog with its tongue out; a white dog with a golden-cream head on grass; a shaggy cream dog; a white Samoyed-like dog; two small white terriers (two bright patches, one per dog).
- Row "flowers": a white daisy with a yellow centre; a yellow daisy-like flower; a pink aster with a butterfly behind; a white daisy; a spiky orange-and-cream flower with dark spines.

The notice sits at the lower left, under the row label "flowers": "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the grid of stimulus photographs). All rights reserved — excluded from the CC license.*

## Slide 56 — Self-supervised learning

Title: "Self-supervised learning". The left half of the slide is empty. At the right:

Common trick:

- Convert “unsupervised” problem into “supervised” empirical risk minimization
- Do so by cooking up “labels” (prediction targets) from the raw data itself — called **pretext task**

(The words "pretext task" in bold.)

## Slide 57 — (no title; pretext tasks and model schematics)

No title is printed. A table with two row headings at the left in bold serif type, "Pretext task:" (top row) and "Model schematic:" (second row), and three columns separated by vertical dotted lines, under a horizontal rule. Column headings: "Class prediction", "Future frame prediction", "Next pixel prediction". Each schematic reads bottom to top.

- Column "Class prediction": a bold $\mathbf{x}$ at the bottom beneath the orange-bird photograph (an orange-yellow bird on a branch among dark green leaves); above the photograph a trapezoid (narrow at the top) labelled $f$; above it a small grey horizontal bar with a bold $\mathbf{z}$ at its right; above that a white bar with an italic $g$ at its left; above that the typewriter word "Bird"; above that a bold $\mathbf{y}$.
- Column "Future frame prediction": a bold $\mathbf{x}$ beneath the same bird photograph; above it a trapezoid $f$, a grey bar with $\mathbf{z}$, an inverted trapezoid (wide at the top) labelled $g$, and then a second photograph of the same bird, turned in profile towards the viewer's right with its beak pointing right and slightly down, with a bold $\mathbf{y}$ above it.
- Column "Next pixel prediction": a bold $\mathbf{x}$ beneath a grid of coloured squares, four across and three down: row one dark grey, pale yellow, yellow, dark orange; row two dark green, yellow, gold, orange; row three black, yellow, orange, and a dashed empty square at the bottom right (the missing pixel). Above the grid a trapezoid $f$, a grey bar with $\mathbf{z}$, a white bar with $g$, one golden-yellow square, and a bold $\mathbf{y}$.

The notice sits at the lower left, below the "Model schematic:" heading and across the first dotted line: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It is taken to cover the bird photographs.

*OCW notice: © source unknown (the bird photographs). All rights reserved — excluded from the CC license.*

## Slide 58 — Imputation: one pretext task to rule them all?

Title: "**Imputation**: one pretext task to rule them all?" (the word "Imputation" in bold). A table like slide 57's, with "Pretext task:" and "Model schematic:" row headings in bold serif type, and three columns separated by vertical dotted lines, headed "Spatial imputation", "Spatial imputation" and "Channel imputation". At the lower left of the row-heading column a legend: a green square labelled "Observed" and a grey square labelled "Masked". Each column is a vertical stack, from the top: $V_2(\mathbf{X})$ (serif, bold X), an image beside a cube diagram, an inverted trapezoid $g$, a grey bar with a bold $\mathbf{z}$, a trapezoid $f$, then another image beside a cube diagram, and $V_1(\mathbf{X})$ at the bottom. Each cube is a Rubik-style block of small cubes, with the vertical axis labelled "N", and "M" and "C" labels along its bottom edges; observed cells are green and masked cells grey.

- Column 1 (Spatial imputation): the top image is the bird photograph with its left half black; the cube above has green cells in its right-hand half and grey cells in its left-hand half. The bottom image is the bird photograph with its right half black; its cube has green cells in its left half and grey cells in its right half.
- Column 2 (Spatial imputation): the top image is the bird photograph with scattered black rectangular blocks hiding some patches; its cube has a scattered green and grey pattern; the bottom image shows the complementary blocks hidden, with a complementary pattern on its cube.
- Column 3 (Channel imputation): each cube is 4 (M) × 4 (N) × 3 (C) cells. The top image is the bird in pastel, colour-channel-like colours (grey-blue background, orange bird); in its cube the front channel slab is grey and the back two channel slabs are green. The bottom image is the bird in grayscale; in its cube only the front channel slab is green and the back two are grey. The two cubes are complements: one channel observed below (the grayscale image), two predicted above (the colour channels).

The notice sits at the lower left, under the first column's bottom image, and overlaps the label $V_1(\mathbf{X})$ of that column: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It is taken to cover the bird photographs.

*OCW notice: © source unknown (the bird photographs). All rights reserved — excluded from the CC license.*

## Slide 59 — Masked Autoencoder (MAE)

![Slide 59 — Masked Autoencoder (MAE)](../images/11-representation-learning-reconstruction-based/slide-59.jpg)

Title: "Masked Autoencoder (MAE)". A figure from He, Chen, Xie et al. reading left to right, with sans-serif labels:

- "input": a 5 × 5 grid of image patches, most of them masked (flat grey-brown); eight patches are visible (row 1 column 3; row 2 columns 2 and 5; row 3 columns 1 and 3; row 4 columns 1 and 5; row 5 column 3), showing pieces of a flamingo (orange-pink feathers and a black background).
- A grey arrow to a vertical column of eight stacked visible patches.
- A tall rounded grey box labelled "encoder", and a column of eight light-blue (cyan) rectangles (the encoded tokens) to its right.
- A grey arrow to a long column of tokens: light-blue encoded tokens interleaved with grey masked tokens, 15 drawn (five light blue and ten grey) with a vertical ellipsis "⋮" in the gap between the twelfth and thirteenth; a rounded grey box labelled "decoder" beside it; and to its right a column of salmon-red tokens of the same length, also with an ellipsis.
- A grey arrow to a 5 × 5 grid of patches showing the complete flamingo photograph, labelled "target".

Printed at the lower right, in small blue type: "Courtesy of He, et al. Used under CC BY." Above the large citation "[He, Chen, Xie, et al. 2021]". This slide carries no "All rights reserved" notice; the line sits under the target image.

## Slide 60 — Bidirectional Transformers (BERT)

![Slide 60 — Bidirectional Transformers (BERT)](../images/11-representation-learning-reconstruction-based/slide-60.jpg)

Title: "Bidirectional Transformers (BERT)". Left: a figure of BERT pre-training, in a rounded light-grey panel. At the top, three red up arrows labelled "NSP", "Mask LM" and "Mask LM". Under them a row of green rounded cells: "C", $\mathrm{T}_ 1$, "…", $\mathrm{T}_ N$, $\mathrm{T}_ {[\text{SEP}]}$, $\mathrm{T}_ 1'$, "…", $\mathrm{T}_ M'$. Below, a large light-blue box labelled "BERT" holding two layers of faint ellipses with criss-cross connections, and at its right edge a small grey dotted fragment. A row of yellow cells: $\mathrm{E}_ {[\text{CLS}]}$, $\mathrm{E}_ 1$, "…", $\mathrm{E}_ N$, $\mathrm{E}_ {[\text{SEP}]}$, $\mathrm{E}_ 1'$, "…", $\mathrm{E}_ M'$, each fed by a small grey up arrow from a pink cell: "[CLS]", "Tok 1", "…", "Tok N", "[SEP]", "Tok 1", "…", "TokM". Under them, grouping brackets labelled "Masked Sentence A" and "Masked Sentence B", a blue up arrow, and the caption "Unlabeled Sentence A and B Pair".

Right: the typewriter sentence "Colorless green ideas sleep furiously" at the top, with nine upward arrows below it, a row of nine tall empty rectangles, nine more upward arrows, a second row of nine tall empty rectangles, then two crossing straight lines on each side that squeeze toward a small box holding a red serif letter "A" at the centre and fan out again below to a third row of nine tall rectangles that alternate: white, black, black, white, black, white, black, white, black. A curved black arrow runs from the right of the last (black) rectangle up and to the left, ending with a left-pointing arrowhead in the open gap between the ends of the two right-hand lines, pointing toward the "A" box and touching neither line. Below, nine upward arrows and the same sentence, "Colorless green ideas sleep furiously". At the lower right: "[He, Chen, Xie, et al. 2021]" (as printed, the same citation as slide 59).

## Slide 61 — Masked prediction often works better than autoencoding

![Slide 61 — Masked prediction often works better than autoencoding](../images/11-representation-learning-reconstruction-based/slide-61.png)

Title: "Masked prediction often works better than autoencoding". Left, two rows of schematics, each a small framed square, an arrow, five grey vertical bars of decreasing height (a bottleneck), an arrow, another framed square:

- Top row: a square holding a bold $\mathbf{X}$, labelled "Raw Data" (two lines); the bars; a square holding a bold $\hat{\mathbf{X}}$, labelled "Reconstructed Data" (two lines).
- Bottom row: a square with a blue upper-left triangle holding a bold $\mathbf{X}_ 1$ (dotted diagonal, lower-right triangle white), labelled "Raw Grayscale Channel" (three lines); the bars; a square with a pale yellow lower-right triangle holding a bold $\hat{\mathbf{X}}_ 2$, labelled "Predicted Color Channels" (three lines).

Right, a line chart headed "Classification performance" with the subtitle "ImageNet Task [Russakovsky et al. 2015]". The vertical axis is labelled "Accuracy", with ticks 10, 15, 20, 25, 30, 35, 40; the horizontal axis is labelled "Layer", with eight rotated tick labels, left to right: conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool5. A legend in the top left of the plot names two series. Two series, each with a marker at every one of the eight positions:

- "autoencoder": black line with white-filled black-outlined circle markers: about 13 at conv1, 15 at pool1, 20 at conv2, 21 at pool2, 19.5 at conv3, 16 at conv4, 12 at conv5, 14 at pool5 (rises to pool2, then falls).
- "colorization": blue line with yellow-filled blue-outlined circle markers: about 13 at conv1, 14.5 at pool1, 25 at conv2, 26 at pool2, 31.5 at conv3, 33 at conv4, 32 at conv5, 32 at pool5 (rises to conv4, then stays level). It starts just below the autoencoder at conv1 and pool1 and is well above it from conv2 onward.

At the lower right: "[Zhang, Isola, Efros, ECCV 2016]". The slide prints no OCW notice.

## Slide 62 — Masked prediction often works better than autoencoding

Title: "Masked prediction often works better than autoencoding". Text:

**Why?**

- Hypothesis 1: It’s hard to control compression via a dimensional bottleneck. Requires fiddling with the architecture. Low-dimensional embeddings have bad properties in terms of optimization, etc.
- Hypothesis 2: Autoencoders have shortcuts where they can copy part of the input and get a decent loss. They fall into these traps (local minima) even if global minimizer is in fact good.
- Hypothesis 3: Masked prediction is closer to the downstream problems we care about, which are mainly about prediction.
- Still an open question!

At the upper right, rotated about 50 degrees clockwise (reading from upper left down to the lower right), the words "Ongoing science!" in bold red condensed capital-and-lowercase type.

## Slide 63 — How Much Information is the Machine Given during Learning?

This page is a slide by Yann LeCun, pasted in whole (its own blue title banner, footer and page number are part of the picture). The banner is blue, with the title "How Much Information is the Machine Given during Learning?" in yellow and "Y. LeCun" in white at its top right. Left, a bulleted outline with blue triangle bullets:

- “Pure” Reinforcement Learning (cherry) (the "(cherry)" in red)
  - The machine predicts a scalar reward given once in a while.
  - A few bits for some samples (bold red)
- Supervised Learning (icing) (the "(icing)" in tan)
  - The machine predicts a category or a few numbers for each input
  - Predicting human-supplied data
  - 10→10,000 bits per sample (bold tan)
- Self-Supervised Learning (cake génoise) (the "(cake génoise)" in dark brown)
  - The machine predicts any part of its input for any observed part.
  - Predicts future frames in videos
  - Millions of bits per sample (bold dark brown)

Right, a photograph of a chocolate layer cake on a green pedestal stand, with a slice cut out, a cherry on top and chocolate curls. Three thick arrows point into it: a red arrow from "A few bits for some samples" to the chocolate curls just left of the cherry on top; a tan arrow from "10→10,000 bits per sample" to the cake's frosting at its upper left side; and a dark brown arrow from "Millions of bits per sample" to the body of the cake, ending on the rim of the stand below the cut slice. At the bottom of the pasted slide, the footer text "© 2019 IEEE International Solid-State Circuits Conference" (left), "1.1: Deep Learning Hardware: Past, Present, & Future" (centre) and the pasted slide's own page number "59" (right); the lecture's printed slide number "63" overlaps the centre footer text. At the lower right: "[Slide Credit: Yann LeCun]".

The notice sits under the cake photograph, at its lower left: "© Yann LeCun, IEEE. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It sits under the cake photograph; it names the holder of the pasted slide as a whole.

*OCW notice: © Yann LeCun, IEEE (the pasted slide, including the cake photograph). All rights reserved — excluded from the CC license.*

## Slide 64 — Summary

Title: "Summary" (centred, slightly right of the slide's centre). A numbered list:

1. Deep nets learn *representations*, just like our brains do
2. This is useful because representations transfer — they act as prior knowledge that enables quick learning on new tasks
3. Representations can also be learned without labels, which is great since labels are expensive and limiting
4. Without labels there are many ways to learn representations. We saw:
   1. representations as compressed codes
   2. representations as predictions of missing data

## Slide 65 — (OCW end page)

OCW's appended end page (a smaller, 4:3-style page with a white background). Text, left-aligned: "MIT OpenCourseWare", "https://ocw.mit.edu", "6.7960 Deep Learning", "Fall 2024", "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms". At the bottom centre, the number "65" is printed on the page itself (the text layer did not read it as a slide number, but it is on the page). Not lecture content.
