---
title: Lecture 8 — Transformers (slide deck)
lecture: 8
slides: 55
source_pdf: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec8.pdf
note: Printed slide numbers 1–54 (bottom centre) equal the PDF page numbers exactly. Page 55 is OCW's appended end page (a smaller page), which prints 55, though the number-map script does not read it.
figure_audit: Transcribed by Sonnet from page images; 33 diagram-, equation-, code- and photo-heavy pages (4, 5, 11, 13–17, 19, 21–24, 26, 28–31, 34–36, 38, 39, 41–44, 46, 47, 49, 51, 52, 54) were then checked by Opus, a different model, from 150–600 dpi crops, the PDF's vector data (circles, boxes and arrows counted and located from page.get_drawings()) and the embedded rasters. Every equation agreed, and slide 39's code character for character. Corrections applied on 19 pages: slide 4 (ten birds), 13 (ten circles), 23 (where the bottom arrows start), 24 (eleven nodes, one pair unjoined), 28 (where the labels sit), 29 (no multiplication sign; two dotted arrows, not three), 30 (the bottom patch is the impala's body), 31 (no attention overlay exists in the PDF), 34 (the directions of two arrows), 35 (the conv graph has seven and six circles and a 6 × 7 matrix; the attn graph's pink and orange arrows swapped), 38 (a font), 42 (a shade; the w is lowercase), 44 (p has ten cells), 46 (where the titles sit, and an unsupported gloss removed), 47 (the right molecule has five rings), 49 (rules under the headings), 51 and 52 (where the arrows and the = sit) and 54 (the decoder's columns are shifted one right, so its first causal layer goes strictly forward). The renders of slides 35 and 54 were then checked against the corrected text.
---

# Lecture 8 — Transformers: slide-by-slide

Text and figures of all 55 pages of
[`mit6_7960_f24_lec8.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec8.pdf),
transcribed from the deck (speaker: Phillip Isola; the deck's title is "Lecture 8: Transformers" and the recorded title is "Architectures: Transformers"). Cite these as "slide N" — the printed number equals the PDF page number for slides 1–54; page 55 is OCW's appended end page. Diagrams, plots and photographs are described in prose since the KB is read as text.

**Images.** 34 slides carry a whole-slide render under their heading: 5, 11–17, 19, 21–24, 26, 28–30, 33–36, 38, 41–44, 46–52 and 54. Not rendered: the 7 slides with an OCW "All rights reserved" notice (4, 6, 7, 8, 31, 45, 53); build steps superseded by a rendered slide (20 by 21, 27 by 28, 32 by 33); text, equations and code this file reproduces exactly (9, 18, 37, 39); and the title, outline, divider and end pages (1, 2, 3, 10, 25, 40, 55). Use an image path only if it appears in this file.

Companion pages: [wiki page for this lecture](../../wiki/08-architectures-transformers.md) ·
[transcript](../transcripts/08-architectures-transformers.md)

**Signposting slides you can skip.** Slide 2 is the outline (its header prints "9. Transformers"); slides 10, 25 and 40 are section dividers for the three new ideas; slide 9 lists the three ideas; slide 55 is the OCW end page.

Some slides are **build steps** — the same slide re-shown with more revealed or one element changed: slides 6, 7 and 8 (the bird photograph with different attended crops for three questions), 11, 12 and 13 (tokens as a vector, an array and a set), 14 and 15 (tokenizing an image, then text and audio too), 18 and 19 (token-wise nonlinearity, with the diagram added), 20, 21 and 22 (the neural net, then the token net beside it, then a GNN beside the token net), 26, 27 and 28 (fc and attn layers; the savannah photograph with a question; the answers by attention), 32 and 33 (fc layer, then the self-attention layer), 42 and 43 (positional encoding for a signal, then for tokens), 50, 51 and 52 (GPT's attention, its training and mask, and a two-layer version). Slides 22, 26, 27, 28, 32, 33, 36, 49, 52 and 53 print no title; they are headed here with a description. They are transcribed individually, each with a note of what it adds.

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and outline |
| 3–8 | Motivation: a limitation of CNNs (long-distance relationships among far-apart patches) and the idea of attention |
| 9 | The three key architectural innovations: tokens, attention, positional codes |
| 10–24 | New idea 1, tokens: tokens as vectors of neurons; arrays and sets of tokens; tokenizing images, text and audio; notation; linear combination and token-wise nonlinearity; token nets; GNNs unrolled; transformers as GNNs over fully-connected graphs |
| 25–39 | New idea 2, attention: the attention layer, query-key-value attention, self-attention, attention maps (DINO), the expanded layer, a family of linear layers, MLP versus transformer, multihead self-attention, the ViT block, and its code |
| 40–47 | New idea 3, positional encoding: permutation equivariance; adding location information; Fourier positional codes; ScaleMAE, spherical-harmonic and Laplacian encodings |
| 48–55 | Examples: autoregressive models, GPT and its causal mask, the original transformer figure, an image-to-text architecture; OCW end page |

---

## Slide 1 — Lecture 8: Transformers

Title: "Lecture 8: Transformers". Subtitle: "Speaker: Phillip Isola".

Right side, a diagram of two transformer-style blocks stacked, with three tokens (three tall narrow empty rectangles side by side) at each level, drawn bottom to top. Four rows of three rectangles in all.

- Bottom row of three rectangles. Above it, labelled "self attn" (monospaced, to the left), a layer of arrows: every bottom rectangle has an arrow to every rectangle of the row above it (nine arrows, a full 3 × 3 crossing pattern, including the three straight-up ones). A curly brace beside the bottom row, with a curved arrow from the brace sweeping up and over to the letter $f$, marks this self-attention step as a function $f$ applied to the row.
- Second row of three rectangles. Above it, labelled "MLP (token-wise)" (the label "MLP" in monospace, "(token-wise)" in serif), three arrows go straight up, one per token, each only to the rectangle directly above it.
- Third row of three rectangles. Above it, a second "self attn" layer with the same nine crossing arrows. Beside this third row, a second brace and curved arrow labelled $f$.
- Top row of three rectangles.

Footer bar (grey): MIT logo, "6.7960 Deep Learning" (red), "https://phillipi.github.io/6.7960", right side "Fall 2024". The printed slide number "1" sits just right of the URL.

## Slide 2 — 9. Transformers

Title as printed: "9. Transformers" (the agenda header reads 9 although this is the recorded lecture 8).

- Three key ideas
  - Tokens
  - Attention
  - Positional encoding
- Examples of architectures and applications

## Slide 3 — Don Quixote by Pierre Menard

Alone on the slide, left of centre, in the lower middle: "*Don Quixote* by Pierre Menard" (the title *Don Quixote* in italics).

## Slide 4 — A Limitation of CNNs

Left: a photograph of a flock of long-winged white-and-black birds (storks) flying in a pale blue-grey sky, overlaid with a dashed black grid of 8 columns by 5 rows (nine vertical and six horizontal lines counting the frame; the cells are the image patches). A black scissors icon sits at the left edge on the second grid line, as if cutting the image along the grid. One bird is alone at the top right corner (in the top-right cell). Left of centre, lower down, three birds fly close together. Right of them are two more birds, one at about the middle of the frame and one a little further right; a seventh bird is cut off by the bottom edge near the centre; and a cluster of three birds sits at the far left edge, lower: one cut off by the frame's left edge and two overlapping just right of it. The rest is empty sky. Ten birds in all.

Under the photo, small print: "© Fredo Durand. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

Right: "How many birds are in this image?" Then: "Is the top right bird the same species as the bottom left bird?"

Bottom, large: "CNNs are built around the idea of locality, and are not well-suited to modeling long distance relationships"

*OCW notice: © Fredo Durand (flock of birds photograph). All rights reserved — excluded from the CC license.*

## Slide 5 — A Limitation of CNNs

![Slide 5 — A Limitation of CNNs](../images/08-architectures-transformers/slide-5.jpg)

Centre: a diagram of a three-layer, 1-D convolutional network with seven circles per layer; the bottom layer is labelled $x_1$ (under the leftmost circle) to $x_7$ (under the rightmost). Between adjacent layers each unit is connected to its nearest neighbours (itself and the neighbours either side, i.e. a width-3 filter), drawn as arrows pointing up. Most arrows are light grey; the black arrows trace the receptive field of the two ends.

- Bottom layer: circles 1 and 7 (the ends, $x_1$ and $x_7$) are hatched (diagonal stripes, one hatching direction on the left and the other on the right); circles 2 to 6 are plain white.
- Middle layer: circles 1 and 2 are hatched like the left one; circles 6 and 7 are hatched like the right one; circles 3, 4, 5 are white. Black arrows go from bottom circle 1 to middle circles 1 and 2, and from bottom circle 7 to middle circles 6 and 7.
- Top layer: circles 1, 2, 3 are hatched with the left hatching, circles 5, 6, 7 with the right hatching, and circle 4 (the centre) is white. Black arrows go from middle circle 1 to top 1 and 2, from middle 2 to top 1, 2 and 3, from middle 6 to top 5, 6 and 7 and from middle 7 to top 6 and 7.

The point of the picture: after two layers the influence of $x_1$ and the influence of $x_7$ have not met, because the top-layer centre unit is white.

Caption below: "Far apart image patches do not interact".

## Slide 6 — The Idea of Attention

Left: the bird photograph of slide 4 washed out to near-white, with five cyan-outlined rectangular crops shown at full colour (grey-blue sky behind the birds) and the rest of the image faded. The crops: (1) the top-right bird alone in a small square; (2) a wide crop in the lower left-centre holding three birds; (3) a crop at the left edge, lower, holding the overlapping cluster of birds; (4) a crop right of the first two, lower, holding two birds; (5) a small crop at the bottom edge near the centre holding the head and wing of one more bird. Every bird lies inside one of the crops; the empty sky is faded out.

Under the photo, the small-print notice: "© Fredo Durand. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

Right: "How many birds are in this image?"

*OCW notice: © Fredo Durand (flock of birds photograph with attended crops). All rights reserved — excluded from the CC license.*

## Slide 7 — The Idea of Attention

Left: the same washed-out bird photograph, now with only two cyan-outlined crops at full colour: the top-right bird (a small square at the top right), and the cluster of overlapping birds at the lower left (a small square at the left edge, lower). The remaining birds are faded.

Under the photo, the same small-print notice: "© Fredo Durand. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

Right: "Is the top right bird the same species as the bottom left bird?"

Build step of slide 6: the same image with the attended set changed to just the two birds the question asks about.

*OCW notice: © Fredo Durand (flock of birds photograph with attended crops). All rights reserved — excluded from the CC license.*

## Slide 8 — The Idea of Attention

Left: the washed-out bird photograph again, with a single large cyan-outlined rectangle at full colour in the upper middle of the frame, covering plain grey-blue sky with no bird in it. The birds, including the top-right one, are all faded.

Under the photo, the same small-print notice: "© Fredo Durand. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

Right: "What's the color of the sky?"

Build step of slides 6–7: the attended region is now sky only.

*OCW notice: © Fredo Durand (flock of birds photograph with an attended sky region). All rights reserved — excluded from the CC license.*

## Slide 9 — Three Key Architectural Innovations

1. Tokens
2. Attention
3. Positional Codes

(Slide 2's agenda calls the third "Positional encoding"; here it is "Positional Codes".)

## Slide 10 — New idea #1: tokens

(Section divider; the title alone, centred: "New idea #1: tokens".)

## Slide 11 — A New Data Type: Tokens

![Slide 11 — A New Data Type: Tokens](../images/08-architectures-transformers/slide-11.png)

- A **token** is just a vector of neurons. (note: GNNs also operate over tokens, but over there we called them "node attributes" or node "feature descriptors")
- But the connotation is that a token is an encapsulated bundle of information; with transformers we will operate over tokens rather than over neurons.

Two diagrams below, side by side.

- Left, "array of **neurons**": a column of five empty circles, with a vertical bar and the bold letter $\mathbf{x}$ to its left.
- Right, "array of **tokens**": a column of three tokens, each drawn as a small rectangle enclosing four small circles stacked vertically, with a vertical bar and the bold letter $\mathbf{T}$ to its left.

A grey-bordered, light-grey box at the right: "Note: sometimes the word "token" is instead used to refer to the atomic units of the data sequence we will model. In this usage tokens are the representation of the data only at the input and output layers. We use a more general definition where tokens are the representation of the data at *any* layer." (the word "any" in italics).

## Slide 12 — A new data structure: Tokens

![Slide 12 — A new data structure: Tokens](../images/08-architectures-transformers/slide-12.png)

Same two bullets as slide 11. The title now says "data structure" where slide 11 says "data type", and the grey note box is gone.

- Left, "array of **neurons**": a grid of empty circles, five rows by four columns (20 circles), labelled $\mathbf{x}$.
- Right, "array of **tokens**": a grid of tokens, three rows by four columns (12 tokens), each a small rectangle enclosing four stacked circles, labelled $\mathbf{T}$.

Build step: slide 11's column becomes a two-dimensional array.

## Slide 13 — A new data structure: Tokens

![Slide 13 — A new data structure: Tokens](../images/08-architectures-transformers/slide-13.png)

Same title and two bullets as slide 12.

- Left, "set of **neurons**": ten empty circles scattered with no grid order, labelled $\mathbf{x}$.
- Right, "set of **tokens**": eight tokens (four-circle rectangles) scattered with no grid order at different heights, labelled $\mathbf{T}$.

Build step: slide 12's grid arrangement becomes an unordered set, so the heading changes from "array of" to "set of".

## Slide 14 — Tokenizing the input data

![Slide 14 — Tokenizing the input data](../images/08-architectures-transformers/slide-14.jpg)

Left: a diagram in three horizontal bands, labelled in serif at the left, from bottom to top: "input", "patches", "tokens". The input is a photograph of a grassy savannah with two tall giraffes at the upper right, a zebra grazing at the left, and an impala (antelope with long ringed horns) in the foreground at the lower left. A black arrow labelled "crop" (monospace) points up from the photo to the band of patches: four small square crops (sky and hills with trees; a giraffe's head and neck; an "..." gap; the impala's head with horns; plain grass). Above each of the four shown patches a black arrow points up to an empty tall rectangle (a token). The fourth arrow is labelled $\mathbf{W}_ {\text{tokenize}}$ (bold W, monospace subscript "tokenize"). A brace to the right of the last token is labelled $\mathbf{t} \in \mathbb{R}^d$. At the top right "e.g., linear projection" with a dotted arrow pointing back to $\mathbf{W}_ {\text{tokenize}}$.

Right, two bullets:

- When operating over *neurons*, we represent the input as an array of scalar-valued measurements (e.g., pixels)
- When operating over *tokens*, we represent the input as an array of vector-valued measurements

## Slide 15 — Tokenizing the input data

![Slide 15 — Tokenizing the input data](../images/08-architectures-transformers/slide-15.jpg)

Text: "You can tokenize anything." and "General strategy: chop the input up into chunks, project each chunk to a vector."

Three diagrams side by side, separated by dotted vertical lines, each with labels "tokens", (middle band), "input" at the left:

- Left: slide 14's image diagram, smaller: the savannah photograph, the arrow "crop", four patches with "..." between the second and third, arrows up to four tokens, $\mathbf{W}_ {\text{tokenize}}$ on the last arrow and the brace $\mathbf{t} \in \mathbb{R}^d$. Middle band "patches".
- Middle: text. The input is the string "Three guineafowl." (monospace). Middle band "byte pairs": the chunks "[Th][re]" ... "[wl][.]" in brackets (monospace). Four black arrows go from the string to the chunks (two at each end of the string) and four more from the chunks up to four empty token rectangles (two at the left, two at the right).
- Right: audio. The input is a brown waveform (an audio signal, with loud and quiet stretches). Middle band "sound snippets": four short brown waveform segments (two at the left, two at the right). Arrows go from the waveform up to the snippets and from the snippets up to four token rectangles.

Each of the three has four tokens drawn (two at each end, with a gap implying more).

## Slide 16 — Notation

![Slide 16 — Notation](../images/08-architectures-transformers/slide-16.png)

Diagram. At left, three tall empty rectangles labelled $\mathbf{t}_ 1$, $\mathbf{t}_ 2$, $\mathbf{t}_ 3$ (column vectors). A dashed arrow curving up and over labelled "transpose" points to the right, to three horizontal rows of four cells each, labelled $\mathbf{t}_ 1^{\mathsf{T}}$, $\mathbf{t}_ 2^{\mathsf{T}}$, $\mathbf{t}_ 3^{\mathsf{T}}$ (row vectors, four cells each). Three solid arrows lead from the three rows into the three rows of a three-by-four grid of cells labelled $\mathbf{T}$ above it. A brace on the right side of the grid is labelled $N$ tokens (rotated), and a brace under it is labelled $d$ channels. So $\mathbf{T}$ stacks the transposed tokens as rows: $N$ rows (here 3), $d$ columns (here 4).

## Slide 17 — Linear combination of tokens

![Slide 17 — Linear combination of tokens](../images/08-architectures-transformers/slide-17.png)

Two columns of equal structure. Left heading (bold serif): "Linear combination of neurons". Right heading: "Linear combination of tokens".

Left: a diagram with three input circles at the bottom, labelled $x_1$, $x_2$, $x_3$ (light, dark grey, mid grey fill) and the label $\mathbf{x}_ {\text{in}}$ at the left; arrows labelled $w_1$, $w_2$, $w_3$ go up to a single output circle labelled $x_{\text{out}}$. Then the equations:

$$x_{\text{out}} = w_1 x_1 + w_2 x_2 + w_3 x_3$$

$$x_{\text{out}}[i] = \sum_ {j=1}^{N} w_{ij} x_{\text{in}}[j]$$

$$\mathbf{x}_ {\text{out}} = \mathbf{W} \mathbf{x}_ {\text{in}}$$

Right: a diagram with three tokens at the bottom (each a rectangle of four circles of varying grey), labelled $\mathbf{t}_ 1$, $\mathbf{t}_ 2$, $\mathbf{t}_ 3$, together labelled $\mathbf{T}_ {\text{in}}$ at the left; arrows labelled $w_1$, $w_2$, $w_3$ go up to a single output token labelled $\mathbf{t}_ {\text{out}}$. Then the equations:

$$\mathbf{t}_ {\text{out}} = w_1 \mathbf{t}_ 1 + w_2 \mathbf{t}_ 2 + w_3 \mathbf{t}_ 3$$

$$\mathbf{T}_ {\text{out}}[i, :] = \sum_ {j=1}^{N} w_{ij} \mathbf{T}_ {\text{in}}[j, :]$$

$$\mathbf{T}_ {\text{out}} = \mathbf{W} \mathbf{T}_ {\text{in}}$$

(On the page the subscripts "out" and "in" are in a monospace face.)

## Slide 18 — Token-wise nonlinearity

Left, two equations, each a column vector in square brackets with a vertical ellipsis:

$$\mathbf{x}_ {\text{out}} = \begin{bmatrix} \text{relu}(x_{\text{in}}[0]) \cr \vdots \cr \text{relu}(x_{\text{in}}[N-1]) \end{bmatrix}$$

$$\mathbf{T}_ {\text{out}} = \begin{bmatrix} F_\theta(\mathbf{T}_ {\text{in}}[0, :]) \cr \vdots \cr F_\theta(\mathbf{T}_ {\text{in}}[N-1, :]) \end{bmatrix}$$

Right: "F is typically an MLP" and "Equivalent to a CNN with 1x1 kernels run over token sequence".

A few clipped fragments of an animation frame show at the very bottom edge, below the second equation (a row of tiny dots and tick marks); they carry no readable content.

## Slide 19 — Token-wise nonlinearity

![Slide 19 — Token-wise nonlinearity](../images/08-architectures-transformers/slide-19.jpg)

The two equations of slide 18 at left, unchanged (including the clipped fragments at the bottom edge).

Right: a diagram. A horizontal axis at the bottom with an arrow to the right, labelled "tokens". Along it a row of eight tokens, each a small rectangle of four circles filled in shades of white, light grey, dark grey and black (the shading differs from token to token). A vertical line goes up from the first (leftmost) token into a light-blue square labelled $F_\theta$, and from the top of the square an arrow continues up to a first token in an upper row. The upper row has eight tokens too, each with a different shading from the one beneath it. Only the arrow for the leftmost token is drawn; the point is that the same $F_\theta$ is applied to each token separately.

Build step of slide 18: adds the diagram.

## Slide 20 — Token nets

Heading in bold serif: "Neural net". A diagram, bottom to top, of three circles per row in four rows. Labels at the left, each followed by a small right-pointing triangle:

- Bottom layer of arrows, "linear comb of neurons ▷": from the bottom row of three circles to the second row, every circle has an arrow to every circle above it (nine arrows, a full crossing pattern).
- Middle, "neuron-wise nonlinearity ▷": three straight vertical arrows, one per circle, from the second row to the third.
- Top layer of arrows, "linear comb of neurons ▷": nine crossing arrows from the third row to the top row of circles.

The right half of the slide is blank (a later build step adds the token version).

## Slide 21 — Token nets

![Slide 21 — Token nets](../images/08-architectures-transformers/slide-21.jpg)

Build step of slide 20: the right half is filled in. Two diagrams side by side.

- Left, heading "**Neural net**": slide 20's diagram unchanged. Three circles per row, four rows; "linear comb of neurons ▷" on the lowest and the highest arrow layer (nine crossing arrows each) and "neuron-wise nonlinearity ▷" between them (three straight arrows).
- Right, heading "**Token net**": the same wiring with tokens in place of neurons. Three tall empty rectangles per row, four rows. Lowest arrow layer labelled "linear comb of tokens ▷" (nine crossing arrows, every token to every token above), then "token-wise nonlinearity ▷" (three straight arrows, token to token above), then a second "linear comb of tokens ▷" (nine crossing arrows).

## Slide 22 — (no title; GNN beside Token net)

![Slide 22 — (no title; GNN beside Token net)](../images/08-architectures-transformers/slide-22.png)

No title is printed. Two diagrams side by side.

- Left, heading "**GNN**": three tall empty rectangles per row, four rows, wired as in the token net. The lowest arrow layer (nine crossing arrows) is labelled "AGGREGATE"; the three straight arrows between the middle rows are labelled "COMBINE"; the top arrow layer (nine crossing arrows) is labelled "AGGREGATE". A phrase in sans-serif at the right, "may be shared weights", has two dotted curved arrows, one pointing to the lower AGGREGATE layer's arrows and one to the upper AGGREGATE layer's arrows, meaning the two layers may share weights.
- Right, heading "**Token net**": slide 21's token net, unchanged ("linear comb of tokens ▷", "token-wise nonlinearity ▷", "linear comb of tokens ▷").

The neural-net diagram of slide 21 is gone; the GNN replaces it.

## Slide 23 — GNNs unrolled

![Slide 23 — GNNs unrolled](../images/08-architectures-transformers/slide-23.png)

Left, a small graph: four filled circles, each with a small vertical rectangle (a token) beside it in the same colour: teal (upper left), yellow (upper right), red-orange (centre) and blue (lower). Arrows (directed edges): teal to red, yellow to red, teal to blue, blue to red. So the red node receives from teal, yellow and blue, and the blue node receives from teal.

Bullets:

- Like an MLP, but nodes are vectors rather than scalars, edges are potentially complex functions (e.g., an edge can be an MLP)
- Each iteration of GNN message passing is a layer
  - AGGREGATE is akin to a linear layer
  - UPDATE is akin to a pointwise layer

Right, the same graph unrolled over two rounds of message passing, drawn bottom to top with four tokens per row (teal, red, blue, yellow, left to right). Labels at the right, bottom to top: $\text{AGGREGATE}^{(1)}$, $\text{UPDATE}^{(1)}$, $\text{AGGREGATE}^{(2)}$, $\text{UPDATE}^{(2)}$. At the bottom, the arrows start in blank space, about four-fifths of the way down the page, with no input row drawn beneath them: solid arrows from the teal, blue and yellow positions converge on the red token, and from the teal one also onto the blue; dashed arrows carry each node's own token up. After $\text{AGGREGATE}^{(1)}$ each node has its own coloured token plus a black token beside it (the aggregated message). Straight arrows ($\text{UPDATE}^{(1)}$) take each pair to a single coloured token in the middle row. The same pattern repeats for round (2), with black message tokens again beside each coloured token, and then four arrows leave the top, with a vertical ellipsis above (more rounds follow).

## Slide 24 — A view from the graph perspective

![Slide 24 — A view from the graph perspective](../images/08-architectures-transformers/slide-24.jpg)

Centre: a drawing of a graph on eleven light-grey circles (nodes) with dense black lines between them: $i$ and two more along the top (middle and right), one at the far left, one at the far right, four inside and two at the bottom corners. One node, at the top left, is darker grey and labelled $i$ in italics; the other ten are plain light grey. The lines join almost every pair (a pixel check finds 54 of the 55, with only the inner node left of centre and the far-right node not joined); the picture is meant to read as a fully connected graph.

Bottom: "Transformers may be viewed as Graph Neural Networks over fully-connected graphs"

## Slide 25 — New idea #2: attention

(Section divider; the title alone, centred: "New idea #2: attention".)

## Slide 26 — (no title; fc layer beside attn layer)

![Slide 26 — (no title; fc layer beside attn layer)](../images/08-architectures-transformers/slide-26.png)

No title is printed. Two small diagrams under the headings "`fc` layer" (monospace "fc") and "`attn` layer" (monospace "attn"), then a formula block and two sentences.

- Left, "fc layer": three empty tall rectangles at the bottom, labelled $\mathbf{t}_ {\text{in}}$, three lines from them converge on a square box containing a blue bold $\mathbf{W}$, and three arrows lead from the box up into a single empty rectangle labelled $t_{\text{out}}$ (as printed, with a plain italic $t$ here against the bold $\mathbf{t}_ {\text{in}}$).
- Right, "attn layer": the same drawing, but the box contains a red bold $\mathbf{A}$. A dotted curved arrow from the bottom right, labelled $f$, points at the $\mathbf{A}$ box: $\mathbf{A}$ is computed by a function $f$.

Right of the diagrams:

$$\mathbf{A} = f(\ldots) \qquad \triangleleft \text{ attention}$$

$$\mathbf{T}_ {\text{out}} = \mathbf{A} \mathbf{T}_ {\text{in}}$$

Bottom, two sentences. "W" is in blue: "W is free *parameters*." "A" is in red: "A is a function of some input *data*. The data tells us which tokens to attend to (assign high weight in weighted sum)".

## Slide 27 — (no title; the photo and the question)

No title is printed. Left, two vertical bars with labels: $t_{\text{out}}$ (top, a short bar with nothing beside it) and $\mathbf{t}_ {\text{in}}$ (a taller bar at the left of the photograph). The photograph is the savannah scene of slide 14: two giraffes at the upper right, a zebra grazing at the left, and an impala in the foreground at the lower left. Beneath it, in a box, in monospace: "How many animals are in the photo?" The rest of the slide is empty (later builds fill in the figure).

## Slide 28 — (no title; attention to answer two different questions)

![Slide 28 — (no title; attention to answer two different questions)](../images/08-architectures-transformers/slide-28.jpg)

No title is printed. The photograph of slide 27 appears twice, faded, with a few patches shown at full colour. As on slide 27, the labels $t_{\text{out}}$ (top) and $\mathbf{t}_ {\text{in}}$ (beside the photo), each with a vertical bar, sit at the far left beside the left panel only; the right panel carries no labels.

**Left panel, query "How many animals are in the photo?"** (boxed question under the photo, in monospace). Four patches are at full colour, each carrying a small white box with "1" in its bottom-left corner: the zebra (a square at the left), the impala's head and front (a square just below it), and, side by side at the top of the photo, a patch of trees with part of the shorter giraffe and a patch with the taller giraffe's head. Four dashed arrows from the question box point up to the four patches (the question attends to them), and four solid arrows go from the patches up to a single output token drawn at the top as a small tall rectangle in a box. The output token is a short three-cell vertical bar whose bottom cell holds "4": the four 1s have been summed to the count 4.

**Right panel, query "What is the color of the impala?"** (boxed question under the photo). Three adjacent patches along the impala (head, body, and rear/legs) are at full colour. Each carries a small orange-tan square whose shade differs slightly (light orange, brown, orange). Three dashed arrows from the question point up to the patches and three solid arrows lead up from the patches to the output token, a three-cell vertical bar whose middle cell is filled the same orange-brown: the colour read out.

## Slide 29 — query-key-value attention

![Slide 29 — query-key-value attention](../images/08-architectures-transformers/slide-29.jpg)

Title (left, two lines): "query-key-value attention". Legend box at the top right: three small coloured bars labelled "query" (yellow), "key" (magenta) and "value" (teal), and below it the formulas

$$\mathbf{q} = \mathbf{W}_ q \mathbf{t}$$

$$\mathbf{k} = \mathbf{W}_ k \mathbf{t}$$

$$\mathbf{v} = \mathbf{W}_ v \mathbf{t}$$

Main diagram, bottom to top. At the bottom a boxed text, in monospace, "What color is the impala's head", with a small empty rectangle (its token) inside at the left; from it, four dotted arrows run up to four yellow bars (the query copied to each of four positions). Each of the four positions reads like an equation: a number, "=", a horizontal magenta bar (the key, as a row) and, just right of it, a vertical yellow bar hanging down from the same height (the query, as a column); no multiplication sign is printed. The four numbers are 1, 0.2, 0.9 and 0.1, left to right, under the sky, giraffe, impala and grass patches. A brace labelled "query" marks the yellow bars, and one labelled "key" the magenta bars.

Above the keys, four photo patches labelled with $\mathbf{T}_ {\text{in}}$ at the left: sky and trees; a giraffe's head and neck; the impala's head with horns; plain grass. Each patch has a small empty white rectangle on its left (its token). Dotted arrows go down from each patch to the key bar and up from each patch to a teal bar (the value) above it. Brace "value" marks the teal bars.

Solid arrows run from the four score numbers (1, 0.2, 0.9, 0.1) straight up to four weights, printed above each teal bar: 0.1, 0.2, 1, 0.1 (left to right). Over each weight-and-value pair is a small curly brace, and curved arrows from the four go up to a summation sign $\Sigma$, with an arrow up from it to the output token $\mathbf{T}_ {\text{out}}$, an empty rectangle in a box at the top.

Left of the diagram, three formulas; the first two have dotted arrows pointing at the corresponding parts of the figure:

$$\mathbf{s} = [\mathbf{q}_ {\text{question}}^{T} \mathbf{k}_ 1, \ldots, \mathbf{q}_ {\text{question}}^{T} \mathbf{k}_ N]$$

(a dotted arrow points from this to the row of scores), then

$$\mathbf{A} = \text{softmax}(\mathbf{s})$$

(a dotted arrow points to the weights above the teal bars), and

$$\mathbf{T}_ {\text{out}} = \begin{bmatrix} a_1 \mathbf{v}_ 1^{\mathsf{T}} \cr \vdots \cr a_N \mathbf{v}_ N^{\mathsf{T}} \end{bmatrix}$$

(printed with "softmax" in monospace.) The superscript on $\mathbf{q}^{T}$ in $\mathbf{s}$ is an italic $T$ where the matrix transposes on the right print an upright $\mathsf{T}$.

Note on the printed numbers: the scores (1, 0.2, 0.9, 0.1) and the weights above the values (0.1, 0.2, 1, 0.1) are not the same set. The score row gives the sky patch the highest score and the impala 0.9, while the weights give the impala 1 and the sky 0.1, and the weights sum to 1.4, not 1. In the recording a student asks about the first score's 1, and the lecturer calls it a typo for 0.1 (≈39:27). The diagram is illustrative.

## Slide 30 — Self-attention

![Slide 30 — Self-attention](../images/08-architectures-transformers/slide-30.jpg)

Title: "Self-attention". A diagram built on the faded savannah photograph. At the bottom, below the faded photograph, one patch (the impala's body, the same crop as the middle $\mathbf{t}_ 2$ patch) is labelled $\mathbf{t}_ 2$ in a small white box at its upper edge. Three dashed arrows go from this patch up to three patches in the middle of the photo, and word "attention" in monospace sits at the lower right of the faded photo. The three middle patches are adjacent, left to right: the impala's head and front, labelled $\mathbf{t}_ 3$; its body, labelled $\mathbf{t}_ 2$; and grass with its rear legs, labelled $\mathbf{t}_ 1$. Three solid arrows go up from these three patches to a box at the top containing a black impala silhouette, with a small box labelled $\mathbf{t}_ 2$ on its upper left; the word "sum" in monospace sits beside the solid arrows. So the same token $\mathbf{t}_ 2$ is both the query (bottom) and one of the attended tokens, and the output is a new $\mathbf{t}_ 2$.

## Slide 31 — Attention maps in a trained transformer

Title: "Attention maps in a trained transformer". A 2 × 2 grid of four photographs separated by black bars: top left, a ship on a rough dark sea under a grey sky; top right, a mountain biker in a helmet and backpack riding down a steep grassy ridge; bottom left, a dark bulldog running toward the camera with mouth open across a field; bottom right, a brown horse trotting across a flat plain under a cloudy sky. No attention overlay or highlight appears: the four frames are a single embedded image with nothing drawn over it, so the PDF shows the plain frames only, despite the title. In the lecture this was a video of each token's attention (≈41:48).

Under the grid, small print: "© AI at Meta. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". At the lower right: ["DINO", Caron et all. 2021] (as printed, "et all").

*OCW notice: © AI at Meta (four video frames from the DINO attention-map figure). All rights reserved — excluded from the CC license.*

## Slide 32 — (no title; fc layer)

No title is printed. Heading "`fc` layer" (monospace "fc"). Three tall empty rectangles at the bottom labelled $\mathbf{T}_ {\text{in}}$ and three at the top labelled $\mathbf{T}_ {\text{out}}$. A square box with a blue bold $\mathbf{W}$ sits in the middle. Between the two rows are nine arrows from every bottom token to every top token (a full crossing pattern); the box is drawn over the crossing. The rest of the slide is empty.

## Slide 33 — (no title; self attn layer)

![Slide 33 — (no title; self attn layer)](../images/08-architectures-transformers/slide-33.png)

No title is printed. Heading "`self attn` layer" (monospace "self attn"). The same drawing as slide 32 with the box now holding a red bold $\mathbf{A}$ in place of the blue $\mathbf{W}$, three tokens in $\mathbf{T}_ {\text{in}}$ below and three in $\mathbf{T}_ {\text{out}}$ above, with nine crossing arrows. A brace beside the input row and a curved arrow from it, labelled $f$, point up to the $\mathbf{A}$ box: $f$ computes $\mathbf{A}$ from the input tokens themselves.

Build step relative to slide 32: the weights $\mathbf{W}$ are replaced by $\mathbf{A} = f(\mathbf{T}_ {\text{in}})$.

## Slide 34 — self attn layer (expanded)

![Slide 34 — self attn layer (expanded)](../images/08-architectures-transformers/slide-34.jpg)

Heading: "`self attn` layer (expanded)" (monospace "self attn"). Legend box (top right): query (yellow), key (magenta), value (teal).

Left diagram, bottom to top. Three input tokens $\mathbf{T}_ {\text{in}}$ (empty rectangles). Each has three short arrows up into a stack of three side-by-side coloured bars, yellow, magenta, teal (its query, key and value); the three stacks are labelled together $\mathbf{Q}_ {\text{in}}, \mathbf{K}_ {\text{in}}, \mathbf{V}_ {\text{in}}$. The first stack (left) has a dashed black outline around its yellow and magenta bars. Above them is a box with a red $\mathbf{A}$ and, from each of the three stacks, straight and diagonal arrows (nine, fully crossing) up to the three output tokens $\mathbf{T}_ {\text{out}}$ (empty rectangles). A small dashed grey square sits on the left vertical arrow, at the first token's path.

Upper right: a matrix picture: an $N \times N$ square labelled $\mathbf{A}$ (red), with a small dashed grey square in its top-left corner; to its left, past the $N$ label, an arrow points left toward the $\mathbf{A}$ box of the left diagram. Then "=" and a product of an $N \times M$ stack of yellow horizontal bars (three shown, the first dashed-outlined, $N$ to the left and $M$ above) and an $M \times N$ set of magenta vertical bars (three shown, the first dashed-outlined; $N$ above, $M$ to the right). A big curved arrow starts just right of the left diagram, between its $\mathbf{A}$ box and the stacks, sweeps down and around, and ends at the right of the magenta bars; the matching dashed outlines say that the dashed entry of $\mathbf{A}$ comes from the dashed query and key.

Lower right, equations, each with a "◁" label:

$$\mathbf{Q}_ {\text{in}} = \begin{bmatrix} \mathbf{q}_ 1^{\mathsf{T}} \cr \vdots \cr \mathbf{q}_ N^{\mathsf{T}} \end{bmatrix} = \begin{bmatrix} (\mathbf{W}_ q \mathbf{t}_ 1)^{\mathsf{T}} \cr \vdots \cr (\mathbf{W}_ q \mathbf{t}_ N)^{\mathsf{T}} \end{bmatrix} = \mathbf{T}_ {\text{in}} \mathbf{W}_ q^{\mathsf{T}} \qquad \triangleleft \text{ query matrix}$$

$$\mathbf{K}_ {\text{in}} = \begin{bmatrix} \mathbf{k}_ 1^{\mathsf{T}} \cr \vdots \cr \mathbf{k}_ N^{\mathsf{T}} \end{bmatrix} = \begin{bmatrix} (\mathbf{W}_ k \mathbf{t}_ 1)^{\mathsf{T}} \cr \vdots \cr (\mathbf{W}_ k \mathbf{t}_ N)^{\mathsf{T}} \end{bmatrix} = \mathbf{T}_ {\text{in}} \mathbf{W}_ k^{\mathsf{T}} \qquad \triangleleft \text{ key matrix}$$

$$\mathbf{V}_ {\text{in}} = \begin{bmatrix} \mathbf{v}_ 1^{\mathsf{T}} \cr \vdots \cr \mathbf{v}_ N^{\mathsf{T}} \end{bmatrix} = \begin{bmatrix} (\mathbf{W}_ v \mathbf{t}_ 1)^{\mathsf{T}} \cr \vdots \cr (\mathbf{W}_ v \mathbf{t}_ N)^{\mathsf{T}} \end{bmatrix} = \mathbf{T}_ {\text{in}} \mathbf{W}_ v^{\mathsf{T}} \qquad \triangleleft \text{ value matrix}$$

$$\mathbf{A} = f(\mathbf{T}_ {\text{in}}) = \text{softmax}\left( \frac{\mathbf{Q}_ {\text{in}} \mathbf{K}_ {\text{in}}^{\mathsf{T}}}{\sqrt{m}} \right) \qquad \triangleleft \text{ attention matrix}$$

$$\mathbf{T}_ {\text{out}} = \mathbf{A} \mathbf{V}_ {\text{in}}$$

(The scaling is printed $\sqrt{m}$, with a lowercase $m$; the diagram's dimension is labelled $M$ in capitals and the code of slide 39 uses $d$.) "softmax" is in monospace.

## Slide 35 — A family of linear layers

![Slide 35 — A family of linear layers](../images/08-architectures-transformers/slide-35.jpg)

Title: "A family of linear layers". A table of three rows (fc, conv, attn) with columns "Wiring graph", "Matrix", "Properties". Each row's name is on a rounded tag rotated about 45 degrees at the left, in monospace.

**fc.** Wiring graph: a column of seven circles at the left, labelled $x_1$, $x_2$ at the top two and "1" at the bottom (the bias input), and a column of six circles at the right; every left circle has a coloured arrow to every right circle (a dense tangle of multicoloured arrows). Matrix: a $6 \times 7$ grid of cells, every one a bright colour (no zeros), the colours varying from cell to cell, next to an input column vector of seven cells whose last cell is "1", an arrow "→", and an output column of six cells. Properties: "Fixed input dimensionality"; $N^2$ learnable parameters.

**conv.** Wiring graph: a column of seven circles at the left ($x_1$, $x_2$ at the top, "1" at the bottom) and six at the right; each right circle receives a blue arrow from the left circle level with it, a green one from the circle above and a pink one from the circle below (the first and last right circles get only two of these), the same three colours at every position, i.e. shared weights, and the bias "1" has yellow arrows to every right circle. Matrix: a black $6 \times 7$ grid with a diagonal band of green, light-blue and pink cells (the same order, left to right, along every row, shifted one step per row) and a column of yellow cells on the right edge; the input vector has a final "1". Properties: "Variable input dimensionality"; $k$+1 learnable parameters ( $k$ = kernel size); conv(translate($\mathbf{x}$)) = translate(conv($\mathbf{x}$)) (conv and translate in monospace, $\mathbf{x}$ bold, with a serif "=" between).

**attn.** Wiring graph: two tokens at each side, $\mathbf{t}_ 1$ and $\mathbf{t}_ 2$ (each a rectangle of three circles, braced and labelled at the left), and two output tokens. Coloured arrows (blue, orange, pink, green) connect the circles: each of the three channels of $\mathbf{t}_ 1$ goes straight across in blue to the matching output channel, and also in orange to the corresponding channel of $\mathbf{t}_ 2$'s output; $\mathbf{t}_ 2$'s channels go straight across in green and cross up to $\mathbf{t}_ 1$'s outputs in pink. Matrix: a black $6 \times 6$ grid with four coloured diagonals, each three cells long: light blue on the main diagonal of the top-left $3 \times 3$ block, pink on the diagonal of the top-right block, orange on the diagonal of the bottom-left block and green on the diagonal of the bottom-right block; input and output vectors have six cells (no "1"). Properties: "Variable input dimensionality"; $|\mathbf{W}_ q| + |\mathbf{W}_ k| + |\mathbf{W}_ v|$ learnable parameters; "attn(permute($\mathbf{T}$)) = permute(attn($\mathbf{T}$))" (with attn, permute in monospace).

## Slide 36 — (no title; MLP beside Transformer (vanilla))

![Slide 36 — (no title; MLP beside Transformer (vanilla))](../images/08-architectures-transformers/slide-36.jpg)

No title is printed. Two diagrams side by side.

- Left, heading "**MLP**": three circles per row, four rows. Lowest arrow layer labelled "linear" (monospace; nine crossing arrows), middle layer labelled "relu (neuron-wise)" ("relu" in monospace, "(neuron-wise)" in serif; three straight arrows), top layer labelled "linear" (nine crossing arrows).
- Right, heading "**Transformer (vanilla)**": the diagram of slide 1: three tokens per row, four rows. "self attn" (monospace; nine crossing arrows) then "MLP (token-wise)" (three straight arrows) then "self attn" (nine crossing arrows), with a brace and a curved arrow labelled $f$ beside each self attn layer's input row.

## Slide 37 — Multihead self-attention (MSA)

Title: "Multihead self-attention (MSA)".

Rather than having just one way of attending, why not have k?

Each gets its own parameterized query(), key(), value() functions.

Run them all in parallel, then (weighted) sum the output token code vectors

$$\mathbf{T}_ {\text{out}}^{i} = \text{attn}^{i}(\mathbf{T}_ {\text{in}}) \quad \text{for } i \in \lbrace 1, \ldots, k \rbrace$$

$$\bar{\mathbf{T}}_ {\text{out}} = \begin{bmatrix} \mathbf{T}_ {\text{out}}^{1}[0, :] & \ldots & \mathbf{T}_ {\text{out}}^{k}[0, :] \cr \vdots & \vdots & \vdots \cr \mathbf{T}_ {\text{out}}^{1}[N-1, :] & \ldots & \mathbf{T}_ {\text{out}}^{k}[N-1, :] \end{bmatrix} \qquad \triangleleft \quad \bar{\mathbf{T}}_ {\text{out}} \in \mathbb{R}^{N \times kv}$$

$$\mathbf{T}_ {\text{out}} = \bar{\mathbf{T}}_ {\text{out}} \mathbf{W}_ {\text{MSA}} \qquad \triangleleft \quad \mathbf{W}_ {\text{MSA}} \in \mathbb{R}^{kv \times d}$$

(The first line prints $k$ as a lowercase italic letter; in the sentence it is a plain "k". "attn" and the subscript "MSA" are in monospace, and the subscript "out" is monospace too.)

## Slide 38 — Transformer (ViT)

![Slide 38 — Transformer (ViT)](../images/08-architectures-transformers/slide-38.jpg)

Title (bold serif): "Transformer (ViT)". A diagram of one transformer block inside a light-grey rectangle, with the heading "repeat $\times L$" above it. Three tokens flow bottom to top through the block (three tall rectangles per row). From bottom to top:

- Three input tokens below the grey box, with arrows up into the box.
- Inside the box, "token norm" (monospace), then a row of three tokens, then "MSA" (monospace) with nine crossing arrows, a brace on the row below it and a blue curved arrow labelled $f$ beside the brace; then a row of three tokens.
- "token norm" again, then a row of three tokens.
- Three blue arrows labelled "MLP (tokenwise)" ("MLP" in monospace) up to the top row of three tokens, which exit the box with arrows pointing up.

Two residual (skip) connections on the right, each a curved arrow with a circled plus: the first starts below the box at the input tokens (right side) and runs up to a ⊕ beside the MSA layer, then on to the row after MSA (the arrowhead lands at the row just above the MSA, where the token norm sits); the second starts there and runs through a second ⊕ beside the MLP layer, landing at the top row of tokens. A further curved arrow leaves the top row up and out of the box, to the next repeat.

At the lower left, a dotted arc labelled "==" (two equals signs, in sans-serif) joins the word "layernorm" (monospace) to the "token norm" label at the bottom of the box, and below it:

$$x_{\text{out}}[k] = \frac{x_{\text{in}}[k] - \mathbb{E}[x_{\text{in}}[k]]}{\sqrt{\text{Var}[x_{\text{in}}[k]]}}$$

(with "Var" in monospace and the subscripts "out" and "in" in monospace).

## Slide 39 — (no title; code)

No title is printed. A boxed Python-style code listing, comments in blue-grey italics, `for`, `range` in green and `in` in purple:

```
# x : input data (RGB image)
# K : tokenization patch size
# d : token/query/key/value dimensionality (setting these all as the same)
# L : number of layers
# W_q_T, W_k_T, W_v_T : transposed query/key/value projection matrices
# mlp: tokenwise mlps

# tokenize input image
T = tokenize(x,K) # 3 x H x W image --> N x d array of token code vectors

# run tokens through all L layers
for l in range(L):

    # attention layer
    Q, K, V = nn.matmul(nn.layernorm(T),[W_q_T[l], W_k_T[l], W_v_T[l]])
    # nn.matmul does matrix multiplication
    A = nn.softmax(nn.matmul(Q,K.transpose()), dim=0)/sqrt(d)
    T = nn.matmul(A,V) + T # note residual connection

    # tokenwise mlp
    T = mlp[l](nn.layernorm(T)) + T # note residual connection

# T now contains the output token representation computed by the transformer
```

(As printed, `K` is used both as the patch size in `tokenize(x,K)` and as the key matrix in `Q, K, V`, and the softmax is divided by `sqrt(d)` outside the softmax call, which is taken with `dim=0`.)

## Slide 40 — New idea #3: positional encoding

(Section divider; the title alone, centred: "New idea #3: positional encoding".)

## Slide 41 — Permutation equivariance

![Slide 41 — Permutation equivariance](../images/08-architectures-transformers/slide-41.png)

Title: "Permutation equivariance". Two copies of a three-token self-attention-plus-pointwise network side by side, joined by "—— permute ——>" (the word "permute" in monospace on an arrow pointing right).

- Left network, three columns of tokens (tall rectangles), each token labelled to its left: bottom row $\mathbf{t}_ 1$, $\mathbf{t}_ 2$, $\mathbf{t}_ 3$; nine crossing arrows (self-attention) up to the middle row, also labelled $\mathbf{t}_ 1$, $\mathbf{t}_ 2$, $\mathbf{t}_ 3$; three straight arrows (token-wise) up to the top row, again $\mathbf{t}_ 1$, $\mathbf{t}_ 2$, $\mathbf{t}_ 3$.
- Right network, identical wiring with every row labelled $\mathbf{t}_ 2$, $\mathbf{t}_ 3$, $\mathbf{t}_ 1$, the permuted order.

Below, three equations; the first two are joined to the third by a down arrow:

$$F_\theta(\text{permute}(\mathbf{T}_ {\text{in}})) = \text{permute}(F_\theta(\mathbf{T}_ {\text{in}}))$$

$$\text{attn}(\text{permute}(\mathbf{T}_ {\text{in}})) = \text{permute}(\text{attn}(\mathbf{T}_ {\text{in}}))$$

$$\text{transformer}(\text{permute}(\mathbf{T}_ {\text{in}})) = \text{permute}(\text{transformer}(\mathbf{T}_ {\text{in}}))$$

("permute", "attn" and "transformer" are in monospace; the subscripts "in" too.) At the lower right, in large sans-serif: "Set2Set".

## Slide 42 — What if you don't want to be shift invariant?

![Slide 42 — What if you don't want to be shift invariant?](../images/08-architectures-transformers/slide-42.png)

Title: "What if you *don't* want to be shift invariant?" (the word "don't" in italics). Body text in a different, sans-serif face from the other slides.

1. Use an architecture that is not shift invariant (e.g., MLP)
2. Add location information to the *input* to the convolutional filters — this is called **positional encoding**

Diagram at the bottom: two columns of eight circles, labelled "pos" (left) and "signal" (right). The "pos" column's circles run in a gradient from light pink at the top to dark maroon at the bottom (eight distinct shades). The "signal" column is grey-scale: from the top, mid grey, light grey, white, black, light grey, mid grey, light grey, white. A light-blue box containing a bold lowercase $\mathbf{w}$ sits to the right. Lines run from the top three signal circles and from the top pos circle into the box, and lines from the box converge on the second circle of an output column of eight circles at the right (white, light grey, dark grey, dark grey, light grey, mid grey, black, white). So a filter sees a short window of the signal together with the position code of its first element.

## Slide 43 — What if you don't want to be permutation invariant?

![Slide 43 — What if you don't want to be permutation invariant?](../images/08-architectures-transformers/slide-43.png)

Title: "What if you *don't* want to be permutation invariant?" (the word "don't" in italics). Sans-serif body.

1. Use an architecture that is not permutation invariant (e.g., MLP)
2. Add location information to the token code vectors — this is called **positional encoding**

Diagram: the self-attention layer of slide 33. Three input tokens labelled $\mathbf{t}_ {\text{in}}$ at the bottom, each with its lowest cell coloured as a small position code (dark maroon, mid pink, light pink, left to right), the rest white; a box with a red $\mathbf{A}$ in the middle; nine crossing arrows up to three output tokens labelled $\mathbf{t}_ {\text{out}}$; a brace and a curved arrow labelled $f$ beside the inputs.

Build step of slide 42 for tokens: the same two options, with the position code appended to each token.

## Slide 44 — Fourier positional codes

![Slide 44 — Fourier positional codes](../images/08-architectures-transformers/slide-44.png)

Title: "Fourier positional codes". Subtitle: "Represent coordinates on Fourier basis".

A grid of small panels. At the far left, the heading "input" over the savannah photograph (giraffes at the top right, zebra, impala) with a small red square outlined at the upper right near the giraffes (the pixel whose coordinates are encoded). Right of it, two rows of five panels each, each panel the same size as the photo with the same small red square at the same position:

- Top row, green stripes (varying with $x$ only, so vertical stripes), titled left to right: $\sin(x)$, $\sin(x/B)$, $\sin(x/B^2)$, $\sin(x/B^3)$, $\sin(x/B^4)$. The first has many narrow stripes, and each next one has wider stripes, down to the fifth which is a slow gradient from light to dark green.
- Bottom row, blue stripes (varying with $y$ only, so horizontal stripes), titled $\sin(y)$, $\sin(y/B)$, $\sin(y/B^2)$, $\sin(y/B^3)$, $\sin(y/B^4)$, with the same progression from narrow to wide bands.

At the right, a tall thin vector labelled $\mathbf{p}$ outlined in red, of ten cells: the top five are shades of green (two of them nearly white) and the bottom five shades of blue. Two curly braces mark the green and blue parts, and two dotted arrows from the last green panel and the last blue panel point to the braces. So $\mathbf{p}$ collects the values of the ten sinusoids at the red-boxed location.

## Slide 45 — Other positional encodings

Title: "Other positional encodings". Text: "ScaleMAE uses ground sample distance positional encoding to train an MAE across spatial scales of remote sensing data". URL at lower right: https://arxiv.org/abs/2212.14532

Figure: at the left, two satellite images stacked. The top image, labelled "10m GSD" in white at its upper right, is a blurry dark aerial view; a red dashed rectangle marks a region of it, and red lines run from that rectangle's corners down to the bottom image, labelled "0.3m GSD" at its lower right, a sharp aerial photograph of buildings, trees, roads and a green sports field (the zoomed-in region, outlined in red). At the right, two pairs of narrow vertical colour strips (viridis colours: purple, blue, green, yellow) each with a white sine-like curve drawn over it, under the headings "GSDPE" and "PE". In the GSDPE column the top strip (for the 10m image) shows a curve with about one period over its height and the bottom strip (for the 0.3m image) shows a much slower curve covering only a part of one period; in the PE column both strips show the same faster curve with about four periods. Rotated captions beside them: "GSDPE varies with absolute scale" (beside the GSDPE strips) and "PE varies only with pixel resolution" (beside the PE strips).

Small print at the lower right: "© Reed, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". The notice sits under the ScaleMAE figure (the satellite images and positional-encoding strips).

*OCW notice: © Reed, et al. (ScaleMAE figure: satellite images at 10m and 0.3m ground sample distance, with GSDPE and PE strips). All rights reserved — excluded from the CC license.*

## Slide 46 — Other positional encodings

![Slide 46 — Other positional encodings](../images/08-architectures-transformers/slide-46.jpg)

Title: "Other positional encodings". Text: "Geographic location encoding with spherical harmonics and sinusoidal representation networks". URL at lower right: https://arxiv.org/abs/2310.06743. Credit: "Courtesy of Rußwurm, et al. Used under CC BY."

Left, a diagram. A grey-blue box, with "LINEAR "Neural Network"" (LINEAR in small caps) above it, shows a pyramid-shaped bar of weights $w_l^m$: coloured cells in a red-to-blue diverging scale (colour bar labelled "weight", ticks −3, −2, −1, 0, 1, 2, 3): one cell on the top row, three in the middle, five at the bottom. A pink box, with "SPHERICAL HARMONICS" (small caps) below it, shows a pyramid of small globes $Y_l^m$ for $l = 0$ (one globe), $l = 1$ (three) and $l = 2$ (five), over an axis labelled $m$ = −2, −1, 0, 1, 2. A circled $\Sigma$ joins the two with bracket lines and an arrow to a globe labelled "output", a map of the Earth shaded red over the northern region and blue towards the south.

Right, a grid of six globes. Rows are labelled "Linear(SH)" and "Siren(SH)", columns $L = 5$, $L = 10$, $L = 20$. The Linear(SH) row shows smooth red-and-blue patterns on Earth maps that become finer with larger $L$; the Siren(SH) row shows maps of Africa, Europe and Asia in red against a blue ocean, whose coastlines get sharper with larger $L$.

## Slide 47 — Other positional encodings

![Slide 47 — Other positional encodings](../images/08-architectures-transformers/slide-47.jpg)

Title: "Other positional encodings". Text: "Laplacian positional encodings to encode node positions in a graph". URL at lower right: https://arxiv.org/abs/2106.03893. Credit: "Courtesy of Kreuzer, et al. Used under CC BY-NC-SA."

Figure: two molecule graphs (atoms as dots, bonds as lines), one left of a thin grey vertical divider and one right, each shown four times, coloured by one eigenvector of the graph Laplacian. Each panel is labelled with its eigenvalue.

- Left molecule (a longer chain of rings): $\lambda_1 = 0.015$, $\lambda_2 = 0.045$ in the top row; $\lambda_{14} = 1.0$, $\lambda_{15} = 1.0$ in the bottom row.
- Right molecule (five rings: two fused pairs joined by a bond, and a fifth ring attached at the upper right): $\lambda_1 = 0.037$, $\lambda_2 = 0.1$ in the top row; $\lambda_{10} = 1.0$, $\lambda_{11} = 1.0$ in the bottom row.

The low-eigenvalue panels colour large parts of the graph smoothly (one region green, the other magenta); the $\lambda = 1.0$ panels are almost entirely white with just a few green or magenta atoms. A colour bar at the right, titled "Eigenvector $\phi$ colormap" (the $\phi$ bold italic), runs from dark green at the top, labelled "max", through white at "0", to magenta at the bottom, labelled "-max".

## Slide 48 — Autoregressive models

![Slide 48 — Autoregressive models](../images/08-architectures-transformers/slide-48.png)

Title: "Autoregressive models". Two rows, each an input phrase, an arrow, a grey box "Predictor", an arrow and an output.

- Top: "Once upon ___" (monospace, with a blank) → Predictor → "time" (the printed word shows what looks like an extra overstruck letter in the middle, "ta̲ime"; reading uncertain: the intended word is "time").
- Bottom: "Once ___ a time" → Predictor → "Upon".

So the predictor can fill the blank at the end (top) or in the middle (bottom).

## Slide 49 — (no title; Training and Sampling)

![Slide 49 — (no title; Training and Sampling)](../images/08-architectures-transformers/slide-49.png)

No title is printed. The slide is split by a horizontal line into two halves, each tagged by a grey rotated label at the left.

**Training** (top). A set in curly braces of (input, target) pairs, with the headings $\mathbf{x}_ 1, \ldots, \mathbf{x}_ {n-1}$ over the inputs and $\mathbf{x}_ n$ over the targets (each separated from its column by a horizontal rule):

| $\mathbf{x}_ 1, \ldots, \mathbf{x}_ {n-1}$ | $\mathbf{x}_ n$ |
| --- | --- |
| Once upon a | time |
| There and back | again |
| The slow brown | fox |
| To be or not to | be |
| ⋮ | ⋮ |

(monospace text; each row a pair separated by a comma). An arrow leads from the set to a box "Learner" and another arrow to the word "Predictor". Two dotted lines run from "Predictor" down to the larger "Predictor" box in the sampling half, linking the trained predictor to its use.

**Sampling** (bottom). The input "Colorless green ideas sleep" (monospace, with heading $\mathbf{x}_ 1, \ldots, \mathbf{x}_ {n-1}$ over it) → a tall box "Predictor" → the output "furiously" with heading $\hat{\mathbf{x}}_ n$ over it. A circular arrow under the Predictor box, curling back on itself, indicates that the output is fed back in as the next input.

## Slide 50 — GPT (and many other related models)

![Slide 50 — GPT (and many other related models)](../images/08-architectures-transformers/slide-50.png)

Title: "GPT (and many other related models)", with "(and many other related models)" in a smaller size.

A diagram, bottom to top. The input text under the diagram: "Colorless green ideas sleep ___" (monospace, with a blank). Nine arrows go up to a row of nine tokens (empty tall rectangles; the drawing has nine tokens although the text has only four words and a blank, so the tokens are not aligned one-to-one with the words). Two long lines meet at a small box containing a red $\mathbf{A}$ in the middle (a bow tie: the self-attention layer drawn compactly, everything attending across the row). Above it, a second row of nine tokens, then nine arrows to a third row of nine tokens, and from the rightmost token a single arrow up to the output word "furiously". A curved arrow at the lower right points up at the right end of the bow tie, marking the last token, whose output is read out.

## Slide 51 — GPT training (and many other related models)

![Slide 51 — GPT training (and many other related models)](../images/08-architectures-transformers/slide-51.png)

Title: "GPT training (and many other related models)", with the parenthesis in a smaller size.

Left: the diagram of slide 50, now with an output at every position. The input text under it: "Colorless green ideas sleep furiously" and the text above: "Colorless green ideas sleep furiously" (monospace), with three sets of nine arrows: from the input text up into the bottom row, from the middle row to the top row, and from the top row up to the output text; the bottom and middle rows are joined only by the bow-tie $\mathbf{A}$. The bow-tie $\mathbf{A}$ box and the curved arrow at the lower right are as in slide 50.

Upper right: a small diagram labelled "time index:" with 1, 2, 3, 4 under four dotted vertical lines. A bottom row has three tokens (at times 1, 2, 3) and a top row has three tokens (at times 2, 3, 4). Arrows through a black box $\mathbf{A}$: from the token at time 1 up to the top tokens at times 2, 3 and 4; from time 2 to 3 and 4; from time 3 to 4 (six arrows). Each output sees only earlier inputs.

Lower right: grids labelled $\mathbf{A}$, $\mathbf{T}_ {\text{in}}$ and $\mathbf{T}_ {\text{out}}$, with "=" between the $\mathbf{T}_ {\text{in}}$ and $\mathbf{T}_ {\text{out}}$ columns ($\mathbf{A} \mathbf{T}_ {\text{in}} = \mathbf{T}_ {\text{out}}$): a $4 \times 4$ grid for $\mathbf{A}$, a column of four cells for $\mathbf{T}_ {\text{in}}$ and a column of four cells for $\mathbf{T}_ {\text{out}}$. In $\mathbf{A}$, row 1 is all black; row 2 is white then three black; row 3 is two white then two black; row 4 is three white then one black. So only the cells strictly below the diagonal are white; the diagonal and everything above it are black. The last cell of $\mathbf{T}_ {\text{in}}$ is black and the first cell of $\mathbf{T}_ {\text{out}}$ is black; the rest of the cells are white. The black cells mark entries that are not used: the last input does not feed anything, and the first output has nothing to read.

## Slide 52 — (no title; two-layer causal attention over time)

![Slide 52 — (no title; two-layer causal attention over time)](../images/08-architectures-transformers/slide-52.png)

No title is printed. A diagram, drawn like slide 51's time-index diagram but with two attention layers. At the bottom, "time index:" and 1, 2, 3, 4 under four dotted vertical lines.

- Bottom row: three tokens, at times 1, 2, 3. Six arrows from them through a box $\mathbf{A}_ 1$: time 1 to times 2, 3 and 4; time 2 to 3 and 4; time 3 to 4. This gives a row of three tokens at times 2, 3, 4.
- Three straight arrows up (token-wise) to a third row (times 2, 3, 4).
- A second layer through a box $\mathbf{A}_ 2$: arrows from time 2 to times 2, 3 and 4 (including a straight one), from time 3 to 3 and 4, and from time 4 straight up to 4. Top row: three tokens, at times 2, 3, 4.

At the right, two matrix pictures:

- Upper: grids labelled $\mathbf{A}_ 2$, $\mathbf{T}_ {\text{in}}$ and $\mathbf{T}_ {\text{out}}$, with "=" between the last two ($\mathbf{A}_ 2 \mathbf{T}_ {\text{in}} = \mathbf{T}_ {\text{out}}$): a $3 \times 3$ grid for $\mathbf{A}_ 2$ with the upper triangle above the diagonal black (row 1: white, black, black; row 2: white, white, black; row 3: white, white, white) and columns of three white cells for $\mathbf{T}_ {\text{in}}$ and $\mathbf{T}_ {\text{out}}$.
- Lower: the same layout for $\mathbf{A}_ 1 \mathbf{T}_ {\text{in}} = \mathbf{T}_ {\text{out}}$, with the $4 \times 4$ grid of slide 51 (row 1 black; then 1, 2, 3 white cells from the left, the rest black), $\mathbf{T}_ {\text{in}}$ with its last cell black and $\mathbf{T}_ {\text{out}}$ with its first cell black.

Build step relating to slide 51: the same causal mask, now stacked over two layers, so after the first layer the sequence has one fewer position and the second layer's mask is $3 \times 3$ with the diagonal allowed.

## Slide 53 — (no title; "Attention Is All You Need")

No title is printed. Left, a picture of the first page of a paper, with a drop shadow: title "Attention Is All You Need"; authors "Ashish Vaswani*, Google Brain, avaswani@google.com", "Noam Shazeer*, Google Brain, noam@google.com", "Niki Parmar*, Google Research, nikip@google.com", "Jakob Uszkoreit*, Google Research, usz@google.com", "Llion Jones*, Google Research, llion@google.com", "Aidan N. Gomez* †, University of Toronto, aidan@cs.toronto.edu", "Łukasz Kaiser*, Google Brain, lukaszkaiser@google.com", "Illia Polosukhin* ‡, illia.polosukhin@gmail.com". Then "Abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."

Right, the paper's architecture figure (the encoder–decoder Transformer), with the slide's red annotations. Encoder stack (left, box labelled "N×"): "Inputs" at the bottom, "Input Embedding", a circled plus joined to a sine-wave symbol labelled "Positional Encoding", then "Multi-Head Attention" and "Add & Norm" (with a skip connection), then "Feed Forward" and "Add & Norm" (with a skip connection). Decoder stack (right, box labelled "N×"), top to bottom: "Output Probabilities", "Softmax", "Linear", "Add & Norm", "Feed Forward", "Add & Norm", "Multi-Head Attention" (taking input from the encoder's output), "Add & Norm", "Masked Multi-Head Attention", an output embedding box, and "Outputs (shifted right)" at the bottom. The slide has darkened the lower part of the decoder (from Add & Norm above the Masked Multi-Head Attention down to the Outputs label) with a grey overlay, and a white box in red text over it: "specific to autoregressive modeling". At the upper left of the diagram, red text "token-wise MLP (a.k.a. 1x1 conv)" with a red dotted arrow pointing at the encoder's "Feed Forward" box.

Notice at the bottom right: "© Vaswani, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". The notice sits under the architecture figure; the first-page screenshot at the left carries no separate notice.

*OCW notice: © Vaswani, et al. (the Transformer architecture figure, and the first page of "Attention Is All You Need" beside it). All rights reserved — excluded from the CC license.*

## Slide 54 — Image-to-text architecture (autoregressive)

![Slide 54 — Image-to-text architecture (autoregressive)](../images/08-architectures-transformers/slide-54.jpg)

Title: "Image-to-text architecture (autoregressive)" over two lines. A diagram, bottom to top.

**Image Encoder** (vertical label at the left, with lines above and below it). Three small image patches at the bottom: a dark crop of branches, a crop with an orange-yellow bird, and a dark crop of foliage. Arrows go up from them to three tokens; a layer labelled "self-attn" (monospace; nine crossing arrows, with a brace and curved arrow) takes them to three more tokens, and three arrows go up to a third row of three tokens.

**Text Decoder** (vertical label at the right, with lines). At the bottom right, the input words "A", "yellow", "bird" (monospace, rotated about 45 degrees), each with an arrow up to a token. These three text tokens sit in a row beside the three image tokens, so one row has six tokens (three image, three text).

Arrows from that row to the next row of three text-position tokens, which stand one column to the right of the input words (above "yellow", above "bird", and above an empty column right of "bird"). Four dotted vertical lines mark the columns: one rises from the top of the "A" token to the top of the decoder (no token sits above "A"), two run up the "yellow" and "bird" columns, and one runs down the empty fourth column to the level of the bottom row. Grey arrows from each of the three image tokens to each of the three text-position tokens (nine grey arrows, labelled "cross-attn" with a brace and a grey curved arrow on the left), and black arrows from the text tokens to strictly later positions (causal: "A" to all three, "yellow" to the second and third, "bird" to the third; six arrows, none straight up; labelled "causal self-attn" with a brace and curved arrow on the right). Straight arrows go up through a middle row, then a second "causal self-attn" layer (black arrows: first to second and third, second to third, plus straight ones) leads to a top row of three tokens with arrows up to the output words "yellow", "bird", "sitting" (monospace, rotated).

So the decoder, given the image tokens and the words so far ("A yellow bird"), predicts the next word at each position (yellow, bird, sitting).

## Slide 55 — MIT OpenCourseWare end page

OCW's appended end page, a smaller page with a white background. Text: "MIT OpenCourseWare", "https://ocw.mit.edu", "6.7960 Deep Learning", "Fall 2024", "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms". A small number "55" is printed at the bottom centre. Not lecture content.
