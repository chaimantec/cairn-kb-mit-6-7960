---
title: Lecture 10 — Memory and Sequence Modeling (slide deck)
lecture: 10
slides: 69
source_pdf: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec10.pdf
note: Printed slide numbers 1–68 (bottom centre) equal the PDF page numbers exactly. Page 69 is OCW's appended end page (a smaller page), which prints 69.
figure_audit: Transcribed by Sonnet from page images; 44 figure-, diagram-, chart-, table- and equation-heavy pages (3, 5, 9–14, 16, 18, 19, 23–28, 31, 33, 34, 36–39, 41–45, 47, 49–53, 56, 58–62, 64, 65 and 69) were then checked by Opus, a different model, from 50–600 dpi renders, the embedded rasters at native resolution, the PDF's vector data (circle shades, arrows and boxes from page.get_drawings()) and the text layer. Every equation agreed symbol for symbol (slides 23, 25, 26, 28, 36–39, 43, 49 and 50), as did all 16 cells of slide 59's table, and every flagged reading was settled (slide 43's bold subscripts, slide 24's arrows into and out of empty space, slide 28's ten arrows, slide 53's tree, slide 56's seven arrows). Corrections applied on 10 pages: slide 26 (no red arrows beside the U arrows; the first reading had put them there), 33 (only the two outer cells carry an "A"; the middle one shows its wiring, and its two inputs merge before the tanh), 34 (the "A" on the outer cells), 41 (the "a" over "time" is full-size, at the output position), 60 (only grid (a) is labelled q1–q6 against k1–k6; the reordered labels and dot positions of (b)–(d); the bucket counts), 61 (three global rows and columns in panel (d)), 64 (eight video frames, not seven; LVU at about 17, not 8), 11 (one walker band slopes the other way), 12 (the red line runs along the man's edge) and 16 (the cat faces the camera; the house is plainly visible), plus the toddler's position on slides 3 and 5 and the cut-off ∂ on slide 25. The renders of slides 26 and 33 were then checked against the corrected text.
---

# Lecture 10 — Memory and Sequence Modeling: slide-by-slide

Text and figures of all 69 pages of
[`mit6_7960_f24_lec10.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec10.pdf),
transcribed from the deck (speaker: Sara Beery; the deck's title slide reads "Lecture 10: Memory and sequence modeling", while its outline slide 2 and its closing slide 67 are headed "11. Memory and sequence modeling" and slide 68's is headed "9. Memory and sequence modeling"). Cite these as "slide N" — the printed number equals the PDF page number for slides 1–68; page 69 is OCW's appended end page. Diagrams, plots, screenshots and photographs are described in prose since the KB is read as text.

**Images.** 32 slides carry a whole-slide render under their heading: 13, 20, 22–28, 33–39, 41, 42, 44–53, 58, 62, 64 and 65. Not rendered: the 20 slides with an OCW "All rights reserved" notice (3–12, 14–19, 55, 56, 60 and 61); build steps superseded by a rendered slide (21, which slide 23 completes, and 30, which repeats slide 25's diagram); text, equations and a table this file reproduces exactly (29, 32, 43, 54, 57, 59, 63 and 66); slide 31, whose picture is a decorative cartoon beside a reading pointer; and the title, outline, divider and end pages (1, 2, 40, 67, 68 and 69). Use an image path only if it appears in this file.

Companion pages: [wiki page for this lecture](../../wiki/10-architectures-memory.md) · [transcript](../transcripts/10-architectures-memory.md)

**Signposting slides you can skip.** Slide 1 is the title; slide 2 is the outline (CNNs for sequences, RNNs, LSTMs, Sequence models and long memory); slide 67 repeats the outline and slide 68 repeats it again with a shorter last bullet; slide 69 is the OCW end page.

Some slides are **build steps** — the same slide re-shown with one element added — and are transcribed individually, each with a note of what it adds: slides 3 to 8 (the classroom photograph: bare, captioned "kindergarden classroom", boxed as television, person and chair, then asked "What color is the chair?", answered "red", and asked "What will the girl do next?"); slides 10 to 12 (the space–time block, then a row slice, then a column slice); slides 15 to 18 and 55 to 56 (the filter at "Frank" and at "Tiger", then a memory unit, then the long link from "Frank" to "Frank!", then many memory units); slides 20 to 23 (the recurrent grid, one column's arrows and equations, the single-column loop, the weight labels); slides 25, 26 and 28 (backpropagation through time, the summed loss, and the shared $\mathbf{W}$); slides 33 to 39 (the standard cell, the LSTM cell, then its parts one by one: cell state, forget gate, input gate and candidate values, state update, output gate); and slides 47 to 53 (the molecule captioner: circles, LSTM boxes, training with a max-likelihood objective, cross-entropy, teacher forcing, testing by sampling, beam search). Slides 29 and 54 repeat the same text, as do 2, 67 and 68. Slides with no printed title are headed here with a description: 3 to 8, 10 to 12, 14 to 18, 26, 28, 33 to 39, 42, 47 to 53, 55, 56 and 59.

Three printed slips run through the deck and are kept as printed: the outline is numbered "11." on slides 2 and 67 and "9." on slide 68 while the title slide says Lecture 10 (the recorded lecture is 10); "depedences" (slide 29) and "depedencies" (slide 54) for "dependences"; and "enchances" for "enhances" on slides 47 to 51, spelled correctly on slide 52.

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and outline |
| 3–8 | Motivation: one classroom photograph, asked what is in it, what colour the chair is, and what the girl will do next |
| 9–12 | Sequences: video, text and audio as sequences; a video as a space–time block and its row and column slices |
| 13–14 | CNNs for sequences: convolutions in time; a 3D filter on the space–time block |
| 15–18 | Why convolutions forget: a filter that sees "Frank" and later "Tiger" or "Frank!", and a memory unit that links them |
| 19–24 | RNNs: a hidden state over time, the recurrence, the weight matrices, deep RNNs |
| 25–28 | Backprop through time, summing the loss, parameter sharing and summed gradients |
| 29–31 | The problem of long-range dependences: why not remember everything; vanishing and exploding gradients; optional reading |
| 32–39 | LSTMs: the cell state, forget gate, input gate and candidate values, state update, output gate |
| 40–46 | Sequence models: autoregressive models, training and sampling, the autoregressive probability model, next-word classification, words as numbers |
| 47–53 | A molecule-to-text captioner: GNN and LSTM, training with max likelihood, teacher forcing, sampling and beam search |
| 54–58 | Long-range dependences again: other methods (temporal convolutions, attention, memory networks) and weight sharing in recurrence, convolution and attention |
| 59–63 | Long context with attention: path lengths and complexity, sparse attention (Reformer, Performers, Linformers), local plus global attention (Transformer XL, Longformer, Big Bird), retrieval (RETRO), context windows of named models |
| 64 | When long-term context is actually needed (certificate lengths of video datasets) |
| 65–66 | Fast and slow memory: parameters against activations, hypernets and code books |
| 67–68 | Outline again |
| 69 | OCW end page |

---

## Slide 1 — Lecture 10: Memory and sequence modeling

Title: "Lecture 10: Memory and sequence modeling". Subtitle: "Speaker: Sara Beery". The rest of the slide is blank.

Footer bar (grey): MIT logo, "6.7960 Deep Learning" (red), "https://phillipi.github.io/6.7960" (underlined), right side "Fall 2024". The printed slide number "1" sits just right of the URL.

## Slide 2 — 11. Memory and sequence modeling

Title as printed: "11. Memory and sequence modeling" (the number 11 is the deck's own; the title slide says Lecture 10).

- CNNs for sequences
- RNNs
- LSTMs
- Sequence models and long memory

## Slide 3 — (no title; a blurry photograph of toddlers in a classroom)

No text except the notice. A full-slide photograph, 4:3 and centred with white margins left and right, of a low-resolution, blurry video frame of a classroom. At the left an old beige computer monitor or television shows a bluish picture. In front of it a child in a cream top and blue skirt stands with their back to the camera, and beside them a second child in lavender holds a thin pole or handle that slopes down to the floor. At the centre a toddler in a cream top and a nappy or white shorts (too blurry to tell) bends over a small red-and-orange plastic chair, hands on it. A blurred child in pink shows through the window. Shelves, a window onto greenery and a wooden table fill the background at the right.

Small print at the lower left of the photograph: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It sits on the photograph's lower-left corner, so it covers the photograph.

*OCW notice: © source unknown (classroom photograph). All rights reserved — excluded from the CC license.*

## Slide 4 — (no title; the classroom photograph labelled "kindergarden classroom")

The same photograph as slide 3, with one added element: a caption in large bright-green sans-serif bold type across the bottom, "kindergarden classroom" (spelled as printed). Added over slide 3: the caption only.

The notice, same wording, sits at the lower left, partly on the photograph's left margin and overlapping the start of the caption: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (classroom photograph). All rights reserved — excluded from the CC license.*

## Slide 5 — (no title; the classroom photograph with three labelled boxes)

The same photograph, with three thick-outlined rectangles and a label for each, as in object detection. Added over slide 4: the three boxes in place of the green caption.

- An orange square around the television at the top left, labelled "television" in orange bold type just below it.
- A yellow rectangle around the toddler at the centre (from just above the head to below the feet), labelled "person" in yellow bold type above its top edge.
- A red-salmon rectangle around the chair at the lower centre, overlapping the lower left of the yellow box, labelled "chair" in the same salmon colour beside its lower left, to the left of the yellow box's bottom.

The notice sits at the lower left of the photograph, below the "chair" label: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (classroom photograph). All rights reserved — excluded from the CC license.*

## Slide 6 — (no title; the photograph with the question "What color is the chair?")

The same photograph with the question in large white bold type with a shadow across its top, in quotation marks: "“What color is the chair?”". No boxes.

The notice sits at the photograph's lower left, as on slide 5.

*OCW notice: © source unknown (classroom photograph). All rights reserved — excluded from the CC license.*

## Slide 7 — (no title; the question and the answer "red")

As slide 6, with a second line added below the question, centred, in the same white bold type: "red". Added over slide 6: the answer.

The notice sits at the photograph's lower left.

*OCW notice: © source unknown (classroom photograph). All rights reserved — excluded from the CC license.*

## Slide 8 — (no title; the photograph with the question "What will the girl do next?")

The same photograph with a different question in white bold type with a shadow across its top: "“What will the girl do next?”". No answer is shown.

The notice sits at the photograph's lower left.

*OCW notice: © source unknown (classroom photograph). All rights reserved — excluded from the CC license.*

## Slide 9 — Sequences

Title: "Sequences". Three horizontal rows, each with a long black arrow underneath pointing right and labelled "time" at the arrow's right end.

1. A strip of eight video frames side by side, each showing a stone building with a dark arched doorway at the top and a flight of steps, with people walking across the foreground in daylight. In the fourth frame a man's head and striped-shirt shoulder fills the left of the frame in close-up. Arrow and "time" below.
2. The text, in large quotation-marked pieces separated by commas: "An", "evening", "stroll", "through", "a", "city", "square". Arrow and "time" below.
3. A tan (brown-beige) audio waveform, a single symmetrical band that swells and narrows irregularly from about 13% to 84% of the slide's width, with a taller burst near its left third. Arrow and "time" below.

The notice sits at the lower left, below the waveform's arrow: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It is taken to cover the video frames and, possibly, the waveform; it does not sit beside either.

*OCW notice: © source unknown (video frames and audio waveform). All rights reserved — excluded from the CC license.*

## Slide 10 — (no title; the video as a space–time block)

No title is printed. At the top, the same strip of eight video frames as slide 9 (stone building, steps, walkers; the close-up head in the fourth), across the full width with no arrow.

Below it, centred, a block drawn as a 3D cube. Its front face is the first video frame (the stone building, a man in a white shirt holding a cup at the front right, a dark-clad couple at the left). Its top face is a smeared, streaky brown-and-cream texture slanting back, and its right face is a smeared strip in which the walkers appear stretched. Three black axis arrows frame the front face: a vertical arrow at the left pointing up labelled $m$ (italic), a horizontal arrow along the bottom pointing right labelled $n$, and a diagonal arrow along the right edge pointing up and to the right labelled $t$. (The arrowheads and the label $n$ are partly overlapped by the cube's corner.)

The notice sits at the lower left, beside neither the strip nor the cube: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (video frames and the space–time cube). All rights reserved — excluded from the CC license.*

## Slide 11 — (no title; a horizontal slice through the block)

No title is printed. Left: the cube of slide 10 enlarged, with axes $m$ (vertical, up), $n$ (horizontal, right) and $t$ (diagonal, up and right). A thick red line is drawn across it: horizontal along the front face at about the height of the people's waists, then, at the front face's right edge, turning to run diagonally up and to the right across the right face to its far edge. So the red line marks one row of pixels ($m$ fixed) through all the time steps.

Right: a flat rectangular image with a vertical arrow at its left labelled $t$ (pointing up) and a horizontal arrow beneath labelled $n$ (pointing right). It shows that row slice: a beige-grey background with a pale stone pattern at the left, crossed by several thick diagonal rope-like bands in dark purple, brown and white-grey (the walkers' tracks over time), most sloping up to the right but one dark purple band at the lower left sloping down to the right across the others (a walker going the other way), and one nearly horizontal orange-brown band slightly below the middle, sloping slightly down to the right across the full width, with a blue checkered fringe above it.

The notice sits at the lower left, below the cube: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the cube and its row slice). All rights reserved — excluded from the CC license.*

## Slide 12 — (no title; a vertical slice through the block)

No title is printed. Left: the cube again, with a thick red line drawn this time vertically up the front face (at about 63% of the face's width, just along the left edge of the man in white) and continuing straight along the top face, slanting back to its far edge. So it marks one column of pixels ($n$ fixed) through all the time steps. Axes as on slide 11: $m$ up at the left, $n$ along the bottom, $t$ diagonal.

Right: a flat rectangle with a vertical arrow at its left labelled $m$ and a horizontal arrow beneath it labelled $t$. It shows the column slice: the top third is dark brown, below it horizontal streaky bands of cream, grey and brown (the static background smeared over time), and in the lower middle three walkers' figures seen as vertical shapes, a woman in a dark patterned top at the left, a person in a green headscarf in the middle and a man in a patterned shirt at the right, with a tall thin orange vertical band at the left of the middle figure and a blue checkered vertical band between that orange band and the headscarf figure.

The notice sits at the lower left: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the cube and its column slice). All rights reserved — excluded from the CC license.*

## Slide 13 — Convolutions in time

![Slide 13 — Convolutions in time](../images/10-architectures-memory/slide-13.png)

Title: "Convolutions in time". Two rows of 17 circles each, filled in greyscale, one above the other, with a long black arrow beneath pointing right, labelled "time" at its right end. The circles are vector drawings; the slide holds no photograph, so it reuses no picture from the excluded slides before it (the video strip, the space–time block and its slices are not on it), and it prints no OCW notice.

- Bottom row (the input), left to right: light grey, white, dark grey, black, mid grey, mid grey, light grey, white, black, white, white, light grey, white, dark grey, light grey, light grey, dark grey.
- Top row (the output), left to right: white, light grey, dark grey, white, light grey, black, black, mid grey, mid grey, dark grey, light grey, dark grey, black, light grey, white, light grey, white.
- Between the rows, at the far left, a white square labelled with a bold lowercase $\mathbf{w}$. Three lines run from its bottom edge down to the first three circles of the bottom row, and three lines run from its top edge up to the second circle of the top row. So one output circle is computed from three neighbouring input circles by the filter $\mathbf{w}$, here at the leftmost position.

## Slide 14 — (no title; a 3D filter on the space–time block)

No title is printed. Left: the space–time cube of slides 10–12, here faded (washed out to pale tones), with axes in grey: $m$ up, $n$ along the bottom, $t$ diagonal up and right. A small black wireframe cube (solid front and top edges, dotted hidden edges) is placed over the cube's top-left-front corner. A black arrow in the middle points right. Right: a larger light-blue wireframe cube (solid and dotted edges) with a much smaller darker-blue cube at its top-left-front corner, the output volume with one entry marked.

The notice sits at the lower left, beneath the faded cube: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the faded space–time cube). All rights reserved — excluded from the CC license.*

## Slide 15 — (no title; the filter at "Frank")

No title is printed. The two rows of 17 circles, the filter box $\mathbf{w}$ and the "time" arrow of slide 13, with the same shades in both rows, but redrawn higher on the slide with the filter at the same far-left position. Added over slide 13: a label "Frank" above the top row's second circle, and a photograph under the time arrow at the far left (below the first two input circles): a ginger-and-white cat with a white chest and yellow-green eyes, looking at the camera, sitting on striped grey-and-white cushions. A thin black frame surrounds the photograph.

The notice sits at the lower left, overlapping the photograph's left edge: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the cat photograph). All rights reserved — excluded from the CC license.*

## Slide 16 — (no title; the filter at "Tiger")

No title is printed. The same two rows of circles as slide 15, but the filter box $\mathbf{w}$ now sits far to the right: three lines run from its bottom edge to the bottom row's twelfth to fourteenth circles (light grey, white, dark grey), and three lines run from its top edge to the top row's thirteenth circle (black). The label "Tiger" sits above that top circle. The photograph beneath the time arrow, under the same inputs, shows a ginger-and-white cat seen from below in three-quarter view, its face toward the camera, looking up among the leaves of a green plant; the cat is half hidden by foliage, and a house (blue-grey siding, white-framed windows) shows as a strip behind it, because the slide crops the photograph to its middle band. A thin black frame surrounds it. The Frank photograph is gone.

The notice sits at the lower left, far from the photograph: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the cat-in-foliage photograph). All rights reserved — excluded from the CC license.*

## Slide 17 — (no title; a memory unit is added at "Frank")

No title is printed. As slide 15 (filter at the far left, label "Frank", cat photograph at the lower left), with every element shifted down a little, and added: a label "Memory unit" in two lines at the top left and a gold (ochre) circle beside it, above the "Frank" label. The gold circle has no arrows or lines yet.

The notice sits at the lower right, away from the photograph: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the cat photograph). All rights reserved — excluded from the CC license.*

## Slide 18 — (no title; the memory is read out at "Frank!")

No title is printed. Both filters of slides 15–17 are shown at once on the same two rows of 17 circles: one at the far left (top circle two, labelled "Frank", fed by the first three input circles) and one at the right (top circle thirteen, a light grey one, labelled "Frank!" with an exclamation mark, fed by the twelfth to fourteenth input circles). The gold memory-unit circle at the top left, labelled "Memory unit" as on slide 17, now has a long curved black arrow from its right side arching across the slide and ending with an arrowhead at the top-row thirteenth circle, the one labelled "Frank!". Under the time arrow, both cat photographs: the cushion cat at the lower left and the cat-in-foliage photograph under the right filter. Added over slide 17: the second filter, the "Frank!" label, the curved arrow and the second photograph.

The notice sits at the lower right, beside the second photograph: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the two cat photographs). All rights reserved — excluded from the CC license.*

Slides 15, 16, 17 and 18 re-show slide 13's circle rows (an unnoticed vector drawing, not a picture) and the two cat photographs; slide 13 itself carries neither photograph.

## Slide 19 — Recurrent Neural Networks (RNNs)

Title: "Recurrent Neural Networks (RNNs)". Three labelled rows on the left, "Outputs" (top), "Hidden" (middle) and "Inputs" (bottom), over eight columns.

- Inputs: 17 greyscale circles in a row (the same shades as slide 13's bottom row: light grey, white, dark grey, black, mid grey, mid grey, light grey, white, black, white, white, light grey, white, dark grey, light grey, light grey, dark grey), with a long black arrow beneath pointing right (no label). Beneath that, a strip of eight video frames (the stone building and walkers of slide 9, with the close-up head in the fourth frame), the same image object as slide 9's strip. The printed slide number "19" is partly hidden behind the strip.
- Filter: a white box with bold $\mathbf{w}$; three lines run from its bottom to the first three input circles and three lines from its top to the first hidden circle.
- Hidden: eight coloured circles in a row, in order gold (ochre), yellow, turquoise, bright green, teal, bright blue, medium blue, dark navy, with a black arrow pointing right from each to the next. The colour changes along the row show the hidden state changing.
- Outputs: eight greyscale circles above, in order light grey, dark grey, black, mid grey, dark grey, dark grey, light grey, light grey, each with a short black arrow pointing up from the hidden circle below it.

The notice sits at the top right, above the last two output circles and far from the film strip: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It is taken to cover the video frames.

*OCW notice: © source unknown (the video frames). All rights reserved — excluded from the CC license.*

## Slide 20 — Recurrent Neural Networks (RNNs)

![Slide 20 — Recurrent Neural Networks (RNNs)](../images/10-architectures-memory/slide-20.png)

Title: "Recurrent Neural Networks (RNNs)". The abstract version of slide 19. This slide has no photograph, only vector drawing, and prints no notice, so it shows no picture from slides 3–19.

Three rows of eight empty (white, black-outlined) circles, with left-hand labels "Outputs" with bold typewriter $\mathbf{x}_ {\texttt{out}}$ (top), "Hidden" with bold $\mathbf{h}$ (middle) and "Inputs" with bold typewriter $\mathbf{x}_ {\texttt{in}}$ (bottom). Short black arrows point up from each input circle to the hidden circle above it, and from each hidden circle to the output circle above it. A black arrow points right from each hidden circle to the next one. A long black arrow beneath the inputs points right, labelled "time" at its right end. (The first hidden circle's left side has no incoming arrow.)

## Slide 21 — Recurrent Neural Networks (RNNs)

Title: "Recurrent Neural Networks (RNNs)". The same eight-column grid of empty circles as slide 20 (rows "Outputs" with bold typewriter $\mathbf{x}_ {\texttt{out}}$, "Hidden" with bold $\mathbf{h}$, "Inputs" with bold typewriter $\mathbf{x}_ {\texttt{in}}$, and a "time" arrow beneath), but only the arrows into the second column are drawn: one arrow from the first hidden circle right to the second, one up from the second input circle to the second hidden circle, and one up from the second hidden circle to the second output circle. Added over slide 20: the arrows are reduced to this one column, and the two equations below.

$$\mathbf{h}_ t = f(\mathbf{h}_ {t-1}, \mathbf{x}_ {\texttt{in}}[t])$$

$$\mathbf{x}_ {\texttt{out}}[t] = g(\mathbf{h}_ t)$$

No picture is on this slide, so it reuses none from the excluded slides.

## Slide 22 — Recurrent Neural Networks (RNNs)

![Slide 22 — Recurrent Neural Networks (RNNs)](../images/10-architectures-memory/slide-22.jpg)

Title: "**Recurrent** Neural Networks (RNNs)", with the word "Recurrent" in bold. A single column of three circles drawn small and centred, labelled at the left "Outputs" (top), "Hidden" (middle) and "Inputs" (bottom). An arrow points up from the input to the hidden circle and another from the hidden to the output circle. A small curved arrow leaves the top right of the hidden circle and loops back to its right side, a self-loop. To the right of the column the word "Recurrent!" is printed. Below, the two equations of slide 21 again:

$$\mathbf{h}_ t = f(\mathbf{h}_ {t-1}, \mathbf{x}_ {\texttt{in}}[t])$$

$$\mathbf{x}_ {\texttt{out}}[t] = g(\mathbf{h}_ t)$$

No picture is on this slide, so it reuses none from the excluded slides. Its equations carry no $\mathbf{x}_ {\texttt{out}}$ or $\mathbf{x}_ {\texttt{in}}$ labels in the diagram, which names the rows only in words.

## Slide 23 — Recurrent Neural Networks (RNNs)

![Slide 23 — Recurrent Neural Networks (RNNs)](../images/10-architectures-memory/slide-23.jpg)

Title: "Recurrent Neural Networks (RNNs)". The eight-column grid of slide 21 (empty circles, "time" arrow), with row labels "Outputs" with $\mathbf{x}_ {\texttt{out}}$, "Hidden" with bold $\mathbf{h}$, and "Inputs" with $\mathbf{x}_ {\texttt{in}}$. As on slide 21, only the second column's arrows are drawn, now each labelled with a bold matrix: $\mathbf{W}$ above the arrow from the first hidden circle to the second, $\mathbf{U}$ beside the arrow from the input up to the hidden circle, and $\mathbf{V}$ beside the arrow from the hidden circle up to the output. Added over slide 21: the weight labels and the weights in the equations.

$$\mathbf{h}_ t = \sigma_1(\mathbf{W}\mathbf{h}_ {t-1} + \mathbf{U}\mathbf{x}_ {\texttt{in}}[t] + \mathbf{b})$$

$$\mathbf{x}_ {\texttt{out}}[t] = \sigma_2(\mathbf{V}\mathbf{h}_ t + \mathbf{c})$$

## Slide 24 — Deep Recurrent Neural Networks (RNNs)

![Slide 24 — Deep Recurrent Neural Networks (RNNs)](../images/10-architectures-memory/slide-24.jpg)

Title: "*Deep* Recurrent Neural Networks (RNNs)", with the word "Deep" in italics. Four rows of eight empty circles, with a "time" arrow beneath. Row labels at the left: "Outputs" with $\mathbf{x}_ {\texttt{out}}$ (top row), "Inputs" with $\mathbf{x}_ {\texttt{in}}$ (bottom row), and between them two hidden rows, the upper labelled $\mathbf{h}_ L$ and the lower $\mathbf{h}_ 1$, joined by a long thin vertical bar at the left of the circles with the word "Hidden" beside its middle. Between the two hidden rows, under the second column, a vertical ellipsis (three dots) stands for the layers 2 to $L-1$ that are not drawn.

Arrows are drawn in the second column only. In the lower hidden row ($\mathbf{h}_ 1$): an arrow from the first circle to the second labelled $\mathbf{W}_ 1$, an arrow up from the input labelled $\mathbf{U}_ 1$, and an arrow leaving the circle upward labelled $\mathbf{U}_ 2$. In the upper hidden row ($\mathbf{h}_ L$): an arrow from the first circle to the second labelled $\mathbf{W}_ L$, an arrow arriving from below labelled $\mathbf{U}_ L$ (it starts in the empty middle of the slide, not at a circle), and an arrow up to the output circle labelled $\mathbf{V}$.

## Slide 25 — Backprop through time

![Slide 25 — Backprop through time](../images/10-architectures-memory/slide-25.png)

Title: "Backprop through time". The grid of slide 23 with three rows of eight circles ("Outputs" with $\mathbf{x}_ {\texttt{out}}$, "Hidden" with $\mathbf{h}$, "Inputs" with $\mathbf{x}_ {\texttt{in}}$) and a "time" arrow. This time the forward arrows are drawn in the first six columns: black arrows up from each of the six inputs to the hidden circle above it (each labelled $\mathbf{U}$ to its right), black arrows pointing right between the six hidden circles (five arrows, each labelled $\mathbf{W}$ above it), and one black arrow up from the sixth hidden circle to the sixth output circle, labelled $\mathbf{V}$. Columns seven and eight have circles only.

Two circles are drawn with a thick outline: the first input circle, which contains a small $\mathbf{x}_ 0$, and the sixth output circle, which has the label $\mathbf{x}_ {\texttt{out}}[t]$ above it (what looks like a tick mark just before it is the right edge of a $\partial$ that the label's crop cuts off; the text layer holds the label as $\partial \mathbf{x}_ {\texttt{out}}[t]$). Red arrows run backwards along the same path: a red arrow pointing down beside $\mathbf{V}$ from the sixth output to the sixth hidden circle, five red arrows pointing left under the $\mathbf{W}$ arrows, and a red arrow pointing down beside the first $\mathbf{U}$ from the first hidden circle to $\mathbf{x}_ 0$. At the bottom, the chain rule:

$$\frac{\partial \mathbf{x}_ {\texttt{out}}[t]}{\partial \mathbf{x}_ {\texttt{in}}[0]} = \frac{\partial \mathbf{x}_ {\texttt{out}}[t]}{\partial \mathbf{h}_ T} \frac{\partial \mathbf{h}_ T}{\partial \mathbf{h}_ {T-1}} \cdots \frac{\partial \mathbf{h}_ 1}{\partial \mathbf{h}_ 0} \frac{\partial \mathbf{h}_ 0}{\partial \mathbf{x}_ {\texttt{in}}[0]}$$

## Slide 26 — (no title; the loss summed over time)

![Slide 26 — (no title; the loss summed over time)](../images/10-architectures-memory/slide-26.png)

No title is printed. Six columns of circles (Outputs, Hidden, Inputs, labelled "Outputs" with $\mathbf{x}_ {\texttt{out}}$, "Hidden" with $\mathbf{h}$, "Inputs" with $\mathbf{x}_ {\texttt{in}}$; a "time" arrow beneath), each wired as on slide 25: black $\mathbf{U}$ arrows up from input to hidden, black $\mathbf{W}$ arrows right between hidden circles, black $\mathbf{V}$ arrows up from hidden to output, and red backward arrows (down beside each $\mathbf{V}$ and left under each $\mathbf{W}$; the $\mathbf{U}$ arrows have none). Above each output circle, a black arrow up and a red arrow down connect it to a loss label, from left to right $\mathcal{L}_ 0$, $\mathcal{L}_ 1$, $\mathcal{L}_ 2$, $\mathcal{L}_ 3$, $\mathcal{L}_ 4$, $\mathcal{L}_ T$. From each loss label a black arrow runs up and inward to a large $\Sigma$ at the top centre, with a red arrow running back down beside it from $\Sigma$ to the loss. Above $\Sigma$ there is a short black up arrow with a red down arrow beside it, ending at an italic $J$. At the top right:

$$\frac{\partial J}{\partial [\mathbf{W}, \mathbf{U}, \mathbf{V}]} = \sum_{t=0}^{T} \frac{\partial \mathcal{L}(\mathbf{x}_ {\texttt{out}}[t], \mathbf{y}_ t)}{\partial [\mathbf{W}, \mathbf{U}, \mathbf{V}]}$$

(The upper limit $T$ is printed upright in a sans-serif face.)

## Slide 27 — Parameter sharing

![Slide 27 — Parameter sharing](../images/10-architectures-memory/slide-27.png)

Title: "Parameter sharing". Two diagrams, both drawn with tan (pale orange) squares.

Left: three tan squares stacked vertically with black arrows pointing up: a bold $\mathbf{x}$ at the bottom with an arrow up into the lowest square, an arrow from it to the middle square, an arrow from that to the top square and one out of the top. A symbol $\theta$ sits to the right of the middle square, and three black arrows lead from it to the three squares: one curved arrow up to the top square's right side, one straight arrow left to the middle square, and one curved arrow down to the bottom square. So one parameter $\theta$ is shared by the three squares.

Right: two tan squares side by side, each with a black arrow leaving its top. Beneath them the word "branch". Two green curved arrows run up from "branch", one to each square, labelled $\theta^{a} = \theta$ (left) and $\theta^{b} = \theta$ (right). Two red curved arrows run from each square's outer side down toward "branch", labelled $\frac{\partial \mathcal{L}}{\partial \theta^{a}}$ (at the left) and $\frac{\partial \mathcal{L}}{\partial \theta^{b}}$ (at the right). Under "branch", a dashed green arrow pointing up labelled $\theta$ and a red arrow pointing down labelled $\sum_i \frac{\partial \mathcal{L}}{\partial \theta^{i}}$.

Caption at the bottom right: "Parameter sharing —> sum gradients".

## Slide 28 — (no title; the shared $\mathbf{W}$ and its gradient)

![Slide 28 — (no title; the shared W and its gradient)](../images/10-architectures-memory/slide-28.jpg)

No title is printed. The diagram of slide 26 drawn faded (grey and pale pink), with the row labels changed to "Outputs" with a bold $\hat{\mathbf{y}}$, "Hidden" with bold $\mathbf{h}$ and "Inputs" with bold $\mathbf{x}$, and over it, in full colour, a yellow square labelled $\mathbf{W}$ placed above the middle of the hidden row. From that yellow square, black curved arrows run down to each of the five $\mathbf{W}$ positions between the hidden circles (two curving to the left, one straight down to the middle one, two curving to the right), and red arrows run from each of those positions back up to the square. Added over slide 26: the yellow $\mathbf{W}$ square, its ten arrows, and, at the right of the diagram, in full colour:

$$\frac{\partial \mathcal{L}_ t}{\partial \mathbf{W}} = \sum_i \frac{\partial \mathcal{L}_ t}{\partial \mathbf{W}^{i}}$$

The faded formula at the top right is that of slide 26.

## Slide 29 — The problem of long-range dependences

Title: "The problem of long-range dependences" (as printed). Text: "Why not remember everything?"

- Memory size grows with t
- This kind of memory is **nonparametric**: there is no finite set of parameters we can use to model it
- RNNs make a Markov assumption — the future hidden state only depends on the immediately preceding hidden state
- By putting the right info in to the hidden state, RNNs can model depedences that are arbitrarily far apart

(The last bullet prints "depedences", as printed.)

## Slide 30 — The problem of long-range dependences

Title: "The problem of long-range dependences". The diagram and equation of slide 25 (eight columns, the six-column forward and red backward arrows, $\mathbf{x}_ 0$ and $\mathbf{x}_ {\texttt{out}}[t]$ in thick circles, the "time" arrow), moved slightly so the chain-rule equation sits under the arrow:

$$\frac{\partial \mathbf{x}_ {\texttt{out}}[t]}{\partial \mathbf{x}_ {\texttt{in}}[0]} = \frac{\partial \mathbf{x}_ {\texttt{out}}[t]}{\partial \mathbf{h}_ T} \frac{\partial \mathbf{h}_ T}{\partial \mathbf{h}_ {T-1}} \cdots \frac{\partial \mathbf{h}_ 1}{\partial \mathbf{h}_ 0} \frac{\partial \mathbf{h}_ 0}{\partial \mathbf{x}_ {\texttt{in}}[0]}$$

Below it, three bullets:

- Capturing long-range dependences requires propagating information through a long chain of dependences.
- Old observations are forgotten
- Stochastic gradients become high variance (noisy), and gradients may **vanish** or **explode**

## Slide 31 — Optional reading: more detailed discussion of stability analysis in recursion from a control theory perspective from Bhiksha Raj @ CMU

Title as printed (two lines): "Optional reading: more detailed discussion of stability analysis in recursion from a control theory perspective from Bhiksha Raj @ CMU".

A greyscale cartoon illustration, centre. A police officer in a cap, hands behind his back, stands scowling at the left under a street lamp. At his feet a man on his hands and knees looks at a smartphone he holds up, in the lamp's pool of light on the pavement. In the dim background at the right a car is parked in front of a building with a sign reading "BAR". A grey band across the lower part of the picture holds the printed text "The streetlight effect is a type of observational bias where people only look for whatever they are searching by looking where it is easiest", and under the picture the caption "“I'm searching for my keys.”" (A faint cartoonist's signature lies across the band.)

Credit in small type at the right of the picture: "“1-27. Drunk under the lamp post” by Peter Morville, CC BY-NC 2.0". Below, an underlined URL in two lines: "https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Spring.2019/archive-f19/www-bak11-22-2019/document/lecture/lec13.recurrent2.pdf". This slide prints no "All rights reserved" notice.

## Slide 32 — LSTMs

Title: "**LSTMs**" (bold). Subtitle: "Long Short Term Memory". Then "[Hochreiter & Schmidhuber, 1997]". Text:

"A special kind of RNN designed to avoid forgetting."

"This way the default behavior is not to forget an old state. Instead of forgetting by default, the network has to *learn to forget*." (the words "learn to forget" in italics).

## Slide 33 — (no title; the standard RNN cell, after Olah)

![Slide 33 — (no title; the standard RNN cell, after Olah)](../images/10-architectures-memory/slide-33.jpg)

No title is printed. A diagram after Chris Olah: three light-green rounded boxes in a row, for the cell at times $t-1$, $t$ and $t+1$. The two outer boxes each carry a large "A" over faint ghost wiring; the middle box has no "A" and shows its wiring instead. Below, three blue circles with the inputs $x_{t-1}$, $x_t$, $x_{t+1}$ (subscripts in small type), and above, three purple circles with the outputs $h_{t-1}$, $h_t$, $h_{t+1}$. In the middle box a black line comes in from the left box's right edge, bends down and merges with the line coming up from $x_t$, and a single arrow then enters a yellow rectangle labelled "tanh" from below; the rectangle's output goes up, bends right and leaves the box on the right (towards the next box), with a branch going up to $h_t$. Arrows run left to right between boxes.

Credit at the bottom right: "[Slide derived from Chris Olah: http://colah.github.io/posts/2015-08-Understanding-LSTMs/]".

## Slide 34 — (no title; the LSTM cell, after Olah)

![Slide 34 — (no title; the LSTM cell, after Olah)](../images/10-architectures-memory/slide-34.jpg)

No title is printed. The same three green boxes with the same inputs $x_{t-1}$, $x_t$, $x_{t+1}$ below and outputs $h_{t-1}$, $h_t$, $h_{t+1}$ above, but the middle box is drawn out in full as an LSTM cell and the outer two show faded copies of its wiring under a large black "A" each, as on slide 33. Two black lines run through the middle box left to right and out the right side with arrowheads. The upper one has a pink circle with a multiplication sign and then a pink circle with a plus sign on it. The lower one carries the input from $x_t$ and the previous output. Four yellow rectangles sit on the lower line, labelled left to right $\sigma$, $\sigma$, "tanh", $\sigma$. The first $\sigma$'s output goes up to the multiplication circle on the upper line. The second $\sigma$ and the tanh go up to a second pink multiplication circle which feeds the plus circle. The last $\sigma$ goes to a third pink multiplication circle at the right, which takes a pink oval "tanh" fed down from the upper line; its output goes to $h_t$ and on along the lower line.

Legend along the bottom, left to right: a yellow rectangle, "Neural Network Layer"; a pink circle, "Pointwise Operation"; a plain arrow, "Vector Transfer"; two lines merging into one arrow, "Concatenate"; one line forking into two arrows, "Copy".

Credit at the bottom: "[Slide derived from Chris Olah: http://colah.github.io/posts/2015-08-Understanding-LSTMs/]".

## Slide 35 — (no title; the cell state)

![Slide 35 — (no title; the cell state)](../images/10-architectures-memory/slide-35.jpg)

No title is printed. A single LSTM cell drawn faded in pale green, with the top line, running from the left edge to the right edge, highlighted in thick black: it carries a label $C_{t-1}$ above its start, a pink circle with a multiplication sign, a pink circle with a plus sign, and an arrowhead at the right labelled $C_t$. The faded parts show the four gates ($\sigma$, $\sigma$, tanh, $\sigma$), the tanh oval and the lower line with $h_{t-1}$ and $x_t$, and an arrow up to $h_t$ at the right. Caption below, in bold: Cell state, preceded by the symbol $C_t$ and an equals sign (printed as "Ct = Cell state" with a subscript t).

Credit: "[Slide derived from Chris Olah: http://colah.github.io/posts/2015-08-Understanding-LSTMs/]".

## Slide 36 — (no title; the forget gate)

![Slide 36 — (no title; the forget gate)](../images/10-architectures-memory/slide-36.jpg)

No title is printed. Left: the cell of slide 35, faded, with the first gate highlighted in black: the $\sigma$ box at the left of the lower row (yellow), fed by the thick black line carrying $h_{t-1}$ from the left and $x_t$ from below, with a thick black arrow up from it, labelled $f_t$, into the first multiplication circle on the cell-state line. Right, at the top: a small plot of the sigmoid, titled by the y-axis label $\sigma(x)$, with y ticks 0.0, 0.2, 0.4, 0.6, 0.8 and 1.0, x ticks $-4$, $-2$, 0, 2 and 4, and the x-axis label $x$. One blue S-shaped curve rises from near 0 at the left edge (about $x = -5$) through 0.5 at $x = 0$ to nearly 1 at the right (about $x = 5$). Beneath it:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Text at the bottom: "Decide what information to throw away from the cell state." and "Each element of cell state is multiplied by ~1 (remember) or ~0 (forget)."

Credit: "[Slide derived from Chris Olah: http://colah.github.io/posts/2015-08-Understanding-LSTMs/]".

## Slide 37 — (no title; the input gate and candidate values)

![Slide 37 — (no title; the input gate and candidate values)](../images/10-architectures-memory/slide-37.jpg)

No title is printed. Left: the faded cell with the second and third gates highlighted in black: the yellow $\sigma$ box and the yellow tanh box on the lower line (fed by $h_{t-1}$ and $x_t$), a thick arrow from the $\sigma$ labelled $i_t$ bending right into the second multiplication circle, and a line from the tanh labelled $\tilde{C}_ t$ going up into the same circle. Right, two equations with two annotations:

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_ t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

Above the first equation: "which indices to write to", with a black arrow pointing down-left at $i_t$. Below the second: "what to write to those indices", with a black arrow pointing up-left at $\tilde{C}_ t$. Text at the bottom: "Decide what new information to add to the cell state."

Credit: "[Slide derived from Chris Olah: http://colah.github.io/posts/2015-08-Understanding-LSTMs/]".

## Slide 38 — (no title; the cell-state update)

![Slide 38 — (no title; the cell-state update)](../images/10-architectures-memory/slide-38.jpg)

No title is printed. Left: the cell with the whole top line highlighted in black (the cell-state line from $C_{t-1}$ to $C_t$, with the two pink circles $\times$ and $+$) and the arrows into it from the forget gate ($f_t$ up into the $\times$) and from the input gate and candidate ($i_t$ and $\tilde{C}_ t$ meeting at a second pink $\times$ and then up into the $+$). The gate boxes themselves are faded. Right:

$$C_t = f_t \ast C_{t-1} + i_t \ast \tilde{C}_ t$$

Text at the bottom: "Forget selected old information, write selected new information."

Credit: "[Slide derived from Chris Olah: http://colah.github.io/posts/2015-08-Understanding-LSTMs/]".

## Slide 39 — (no title; the output gate)

![Slide 39 — (no title; the output gate)](../images/10-architectures-memory/slide-39.jpg)

No title is printed. Left: the cell with the output path highlighted in black: the last yellow $\sigma$ box, fed by $h_{t-1}$ and $x_t$, with an arrow labelled $o_t$ into a pink $\times$ circle that also takes the line from a pink "tanh" oval above it (fed from the cell-state line), and the output line from there to $h_t$ at the right and up to a second $h_t$ at the top. The rest is faded. Right:

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \ast \tanh(C_t)$$

(The first equation prints no dot between $W_o$ and the bracket, unlike slides 36 and 37.) Text at the bottom: "After having updated the cell state's information, decide what to output."

Credit: "[Slide derived from Chris Olah: http://colah.github.io/posts/2015-08-Understanding-LSTMs/]".

## Slide 40 — Sequence models

Title: "Sequence models", centred, on an otherwise blank slide. (A divider.)

## Slide 41 — Autoregressive models

![Slide 41 — Autoregressive models](../images/10-architectures-memory/slide-41.png)

Title: "Autoregressive models". Two rows, each a monospaced prompt with a blank, an arrow to a grey box labelled "Predictor", and an arrow to the predicted word.

- Top row: "Once upon ___" → Predictor → "time". (A full-size "a", in the same type as "time", is printed over its "t" and "i", so the output reads like "taime". The text layer holds the "a" as its own span at the output position. It is most likely the correct next word, "a", and "time" drawn at the same spot, two build steps flattened into one page; it is recorded as printed.)
- Bottom row: "Once ___ a time" → Predictor → "Upon".

## Slide 42 — (no title; training and sampling a predictor)

![Slide 42 — (no title; training and sampling a predictor)](../images/10-architectures-memory/slide-42.png)

No title is printed. Two panels separated by a thin horizontal rule, each with a grey rotated tab at the left: "Training" (top) and "Sampling" (bottom).

Training: a large curly-braced set of four (input, target) pairs, with two overbraced column headings, $\mathbf{x}_ 1, \ldots, \mathbf{x}_ {n-1}$ over the left column and $\mathbf{x}_ n$ over the right, followed by a vertical ellipsis in each column. In typewriter type, left column then right column: "Once upon a" and "time"; "There and back" and "again"; "The slow brown" and "fox"; "To be or not to" and "be". An arrow leads from the set to a tall box labelled "Learner", and an arrow from the Learner to the word "Predictor". Two dotted lines run from "Predictor" down across the rule to the corners of the larger box in the lower panel, so the lower box is that predictor.

Sampling: the text "Colorless green ideas sleep" in typewriter type under the heading $\mathbf{x}_ 1, \ldots, \mathbf{x}_ {n-1}$, an arrow to a tall box labelled "Predictor" with a small circular arrow beneath it (a loop), and an arrow to "furiously" under the heading $\hat{\mathbf{x}}_ n$.

## Slide 43 — Autoregressive probability model

Title: "Autoregressive probability model". Equations:

$$p(\mathbf{X}) = p(\mathbf{x}_ {\mathbf{n}} \mid \mathbf{x}_ {\mathbf{1}}, \ldots, \mathbf{x}_ {n-1})\thinspace p(\mathbf{x}_ {n-1} \mid \mathbf{x}_ 1, \ldots, \mathbf{x}_ {n-2}) \quad \ldots \quad p(\mathbf{x}_ 2 \mid \mathbf{x}_ 1)\thinspace p(\mathbf{x}_ 1)$$

$$p(\mathbf{X}) = \prod_{i=1}^{n} p(\mathbf{x}_ i \mid \mathbf{x}_ 1, \ldots, \mathbf{x}_ {i-1})$$

(In the first line, the subscripts of the first factor's $\mathbf{x}_ {\mathbf{n}}$ and $\mathbf{x}_ {\mathbf{1}}$ are printed in bold, unlike the italic subscripts elsewhere.)

Below, an example worked on the phrase $p(\texttt{Once upon a time})$ in typewriter type, with four braces and labels: a long brace over the whole phrase labelled $p(\texttt{time} \mid \texttt{Once, upon, a})$; a shorter brace over "Once upon a" labelled $p(\texttt{a} \mid \texttt{Once, upon})$; a brace under "Once" labelled $p(\texttt{Once})$; and a brace under "Once upon" labelled $p(\texttt{upon} \mid \texttt{Once})$.

## Slide 44 — Modeling a sequence of words

![Slide 44 — Modeling a sequence of words](../images/10-architectures-memory/slide-44.png)

Title: "Modeling a sequence of words". Text: "How to model $p(\texttt{time} \mid \texttt{Once, upon, a})$ ?" then "Just treat it as a next word classifier!"

Diagram, lower centre: the typewriter text "Once upon a", a hollow right-pointing block arrow labelled $f$, and a horizontal bar chart with its vertical axis at the left and a horizontal axis labelled 0 at its left end and 1 at its right end. Bars, top to bottom, labelled in typewriter type: "year" (a short bar, about 0.07), "time" (bold label, the longest bar, about 0.5), "day" (about 0.1), "elephant" (about 0.06), then a vertical ellipsis.

## Slide 45 — How to represent words as numbers?

![Slide 45 — How to represent words as numbers?](../images/10-architectures-memory/slide-45.png)

Title: "How to represent words as numbers?". Left: the typewriter text "Once upon a" and a block arrow labelled $f$. Centre: a heading "Prediction" (underlined) with bold $\hat{\mathbf{y}}$, then $f_\theta : X \rightarrow \mathbb{R}^K$, above a horizontal bar chart whose rows are labelled "a", "aardvark", "absolve", "accurate", "adapt", "aether", "after", "aghast" and a vertical ellipsis. All eight bars are very short and of similar length, "aether" the longest of them; the axis is marked 0 at its left and 1 at its right.

Right text: "We can represent words as 1-hot-vectors of size K, where K is the size the vocabulary (e.g., K=100,000)." (The words "of" is missing before "the vocabulary", as printed.)

## Slide 46 — How to represent words as numbers?

![Slide 46 — How to represent words as numbers?](../images/10-architectures-memory/slide-46.png)

Title: "How to represent words as numbers?". The same layout as slide 45 ("Once upon a", block arrow $f$, "Prediction" with $\hat{\mathbf{y}}$, $f_\theta : X \rightarrow \mathbb{R}^K$) with a bar chart whose rows are the letters "a", "b", "c", "d", "e", "f", "g", "h" and a vertical ellipsis; all bars short, "f" the longest; axis 0 to 1.

Right text: "Or, represent each character as a class (e.g., K=26 for English letters), and represent words as a sequence of characters."

## Slide 47 — (no title; "Molecule-2-text")

![Slide 47 — (no title; "Molecule-2-text")](../images/10-architectures-memory/slide-47.jpg)

No title is printed. A recurrent diagram with greyscale circles. Row labels at the left: "Outputs", "Hidden", "Inputs". Seven columns:

- Outputs (greyscale circles, left to right): light grey, dark grey, black, mid grey, dark grey, dark grey, light grey, each with a word printed above it, rotated about 45 degrees, in typewriter type: "A", "mild", "stimulant", "that", "enchances", "cognitive", "ability" ("enchances" as printed).
- Hidden (seven circles, each with an arrow up to the output above it): dark grey, mid grey, light grey, black, dark grey, black, light grey, joined by black arrows pointing right.
- Inputs: three circles (light grey, white, dark grey) at the lower left, joined by three lines to the first hidden circle (as in the filter diagrams above), and under them a thin-framed picture of a chemical structure drawing: a two-ring skeletal formula with atom labels O, H3C, N, CH3, N, N, O, N and CH3 (labels as printed).

At the right, in quotation marks: "Molecule-2-text".

## Slide 48 — (no title; the molecule captioner with LSTM boxes)

![Slide 48 — (no title; the molecule captioner with LSTM boxes)](../images/10-architectures-memory/slide-48.png)

No title is printed. The same task drawn with boxes. Row labels: "Outputs", "Hidden", "Input". Eight blue squares labelled "LSTM" in a row, joined by black arrows pointing right, each with an arrow up to a rotated typewriter word: "A", "mild", "stimulant", "that", "enchances", "cognitive", "ability", "END". Under the first LSTM an arrow comes up from a pink square labelled "GNN", and an arrow comes up into the GNN from the chemical structure drawing of slide 47 in a thin frame. Added over slide 47: the boxes in place of circles, the GNN, the eighth word "END".

## Slide 49 — (no title; training with a max-likelihood objective)

![Slide 49 — (no title; training with a max-likelihood objective)](../images/10-architectures-memory/slide-49.png)

No title is printed. A grey tab at the top left reads "Training". Row labels: "Targets" with italic $y$, "Outputs" with $p_\theta(\cdot)$, "Hidden", "Input". The eight LSTM boxes, the GNN and the chemical structure of slide 48 are drawn again. The target words are printed rotated above ("A", "mild", "stimulant", "that", "enchances", "cognitive", "ability", "END"). Above each LSTM, after the arrow up, is a small bar chart on a baseline of three bars, one per word of a tiny vocabulary: the first has a tall, a medium and a very small bar; the second three short bars; the third a tiny, a very tall and a tiny bar; the fourth two short bars and a tiny one; the fifth a short, a tall and a medium bar; the sixth a tiny, a short and a tall bar; the seventh two medium bars and a tiny one; the eighth a short, a medium and a tall bar. Added over slide 48: the bar charts, the Targets row and the text at the lower right:

"**Max-likelihood objective**: maximize probability the model assigns to each target word: $\arg\max_\theta \log p_\theta(y)$"

## Slide 50 — (no title; training with cross-entropy)

![Slide 50 — (no title; training with cross-entropy)](../images/10-architectures-memory/slide-50.png)

No title is printed. As slide 49 ("Training" tab, LSTM boxes, GNN, structure, the same eight output bar charts) with row labels "Targets" with bold $\mathbf{y}$, "Outputs" with bold $\hat{\mathbf{y}}$, "Hidden", "Input". Added over slide 49: a row of eight one-hot target charts above the output charts, each a single tall bar on a baseline, at the left position for words one, four and seven, at the middle position for words two, three and five, and at the right position for words six and eight, and no target words printed rotated. Text at the lower right:

"**Max-likelihood objective**: minimize cross-entropy between model outputs and one-hot encoded targets."

$$f^{\ast} = \arg\min_{f \in \mathcal{F}} \sum_{i=1}^{N} H(\mathbf{y}_ i, \hat{\mathbf{y}}_ i)$$

## Slide 51 — (no title; teacher forcing)

![Slide 51 — (no title; teacher forcing)](../images/10-architectures-memory/slide-51.png)

No title is printed. The "Training" tab, and beside it a bold heading "Teacher forcing". Row labels "Targets" with italic $y$, "Outputs" with $p_\theta(\cdot)$, "Hidden", "Input", as in slide 49, with the target words printed rotated above ("A", "mild", "stimulant", "that", "enchances", "cognitive", "ability", "END"), the output bar charts, eight LSTM boxes, the GNN and the structure. Added over slide 49: one short black arrow pointing up into each of the LSTM boxes two to eight from empty space beneath them, with no source drawn. Text at the lower right: "Condition each next word prediction on the **ground-truth** preceding word." (the words "ground-truth" in bold green).

## Slide 52 — (no title; testing by sampling)

![Slide 52 — (no title; testing by sampling)](../images/10-architectures-memory/slide-52.png)

No title is printed. A grey tab at the top left reads "Testing". Row labels: "Samples", "Outputs" with $p_\theta(\cdot)$, "Hidden", "Input". The eight LSTM boxes with their output bar charts, the GNN and the chemical structure are drawn as on slide 49. An arrow runs up from each bar chart to a rotated typewriter word, the samples: "A", "strong", "stimulant", "that", "enhances", "cognitive", "ability", "END". Each sample is also copied by a curved dotted arrow down and to the right to a typewriter word beneath the next LSTM, the inputs of the next step: "A", "strong", "stimulant", "that", "enhances", "cognitive", "ability", each with a short arrow up into the LSTM. (This slide spells "enhances" correctly, where slides 47–51 print "enchances".) Text at the lower right: "Sample from predicted distribution over words." and "Alternatively, sample most likely word."

## Slide 53 — (no title; beam search)

![Slide 53 — (no title; beam search)](../images/10-architectures-memory/slide-53.png)

No title is printed. The "Testing" tab and a bold heading "Beam search". At the left, "Tree of samples". A tree of words drawn with black arrows from left to right: "A" (orange) splits into "strong" (black) and "mild" (orange). "strong" splits into "stimulant" and "neural", each of which has two unlabelled arrows leaving it. "mild" splits into "stimulant" (orange) and "psychostimulant", each with two unlabelled arrows leaving it. The orange nodes, "A", "mild" and "stimulant", mark one path through the tree.

Text: "Sample multiple sequences (top-k greedy completions on each step), then pick the sequence with highest score." and "Score could be model's confidence: $p_\theta(\mathbf{y}_ 1, \ldots, \mathbf{y}_ T \mid \mathbf{x})$".

## Slide 54 — The problem of long-range dependences

Title: "The problem of long-range dependences". Text: "Why not remember everything?"

- Memory size grows with t
- This kind of memory is **nonparametric**: there is no finite set of parameters we can use to model it
- RNNs make a Markov assumption — the future hidden state only depends on the immediately preceding hidden state
- By putting the right info in to the hidden state, RNNs can model depedencies that are arbitrarily far apart

(This repeats slide 29 with a different misspelling: slide 29's last bullet prints "depedences", this one prints "depedencies".)

## Slide 55 — (no title; the memory unit again)

No title is printed. The diagram of slide 18: the label "Memory unit" with the gold circle at the top left, the two rows of 17 greyscale circles, both filter boxes $\mathbf{w}$ (left one over the second top circle labelled "Frank", right one over the thirteenth labelled "Frank!"), the long curved arrow from the gold circle to the thirteenth top circle, the "time" arrow and both cat photographs below. One change from slide 18: the thirteenth top circle, which was light grey there, is now black.

The notice sits at the lower right, beside the second photograph: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the two cat photographs). All rights reserved — excluded from the CC license.*

## Slide 56 — (no title; many memory units)

No title is printed. As slide 55, but the single memory unit is replaced by a row of seven coloured circles at the top, labelled "Memory units" (plural) at the left, then three dots: gold, yellow, turquoise, white, teal, bright blue, medium blue. A vertical line joins each of them to the top-row circles two to eight directly below. A fan of about seven thin black curved arrows runs from this row across the slide to the thirteenth top circle (black, the "Frank!" position), all ending in arrowheads at that circle. The labels "Frank" and "Frank!" are gone, but both filter boxes $\mathbf{w}$, both photographs and the "time" arrow remain.

The notice sits at the lower right, beside the second photograph: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © source unknown (the two cat photographs). All rights reserved — excluded from the CC license.*

## Slide 57 — The problem of long-range dependences

Title: "The problem of long-range dependences". Text: "Other methods exist that do directly link old “memories” (observations or hidden states) to future predictions:"

- Temporal convolutions
- Attention / Transformers (see https://arxiv.org/abs/1706.03762)
- Memory networks (see https://arxiv.org/abs/1410.3916)

## Slide 58 — Modeling arbitrarily long sequences

![Slide 58 — Modeling arbitrarily long sequences](../images/10-architectures-memory/slide-58.png)

Title: "Modeling arbitrarily long sequences". Three bullets, each with a small diagram of empty circles at the right.

- "**Recurrence** — recurrent weights are shared across time". Diagram: three rows of circles (the top row of seven, the other two of eight) with three blue arrows: one pointing right from a middle-row circle to the next, and one pointing up from the bottom row to the middle row and one from the middle row to the top row, all at the second column.
- "**Convolution** — conv weights are shared across time". Diagram: a row of eight circles with a row of six above; three blue arrows run from the first three circles of the lower row to the first circle of the upper row.
- "**Attention** — weights are dynamically determined as a function of the data (conv kernel with attention weights is shown on the right)" (the parenthesis in smaller type). Diagram: the same as the convolution diagram, with the three arrows drawn in red.

## Slide 59 — (no title; a table from "Attention Is All You Need")

No title is printed. A screenshot of a table, with the credit below: "[“Attention is All you Need”, https://arxiv.org/abs/1706.03762]". The slide prints no OCW notice. The screenshot is a single raster image not used on any other slide, so it reuses no picture from an excluded slide.

Caption: "Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolutions and $r$ the size of the neighborhood in restricted self-attention."

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
| --- | --- | --- | --- |
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

## Slide 60 — Even-larger-context transformers

Title: "Even-larger-context transformers". Subheading in bold italics: "Efficiency from sparsification". Bullets (the model names in bold):

- **Reformer** replaces quadratic dot product attention with a mechanism that uses local hashing to get to O(n log n) (https://arxiv.org/abs/2001.04451)
- **Performers** introduce the use of positive orthogonal random features within attention to get to O(n) (https://arxiv.org/pdf/2009.14794)
- **Linformers** use low-rank matrix approximation to get O(n) in time and space (https://arxiv.org/pdf/2006.04768)

Below, a two-part figure. Left: a vertical pipeline of rows of 16 small squares with labels at the left. "Sequence of queries=keys" is a row of 16 white squares. "LSH bucketing" is the same 16 squares coloured by bucket: 5 blue, 4 yellow, 3 maroon and 4 white. "Sort by LSH bucket" shows grey crossing arrows from that row to a sorted row in which the colours are grouped (blues, then yellows, then maroons, then whites). "Chunk sorted sequence to parallelize" shows the sorted row cut by four grey brackets into chunks of four squares, drawn as four separate groups; the chunks straddle buckets (four blue; one blue and three yellow; one yellow and three maroon; four white). "Attend within same bucket in own chunk and previous chunk" shows the four groups again with grey curved arrows joining squares within a chunk and back to the chunk before.

Right: four small $6 \times 6$ grids, each cell marked with a dot where attention is allowed. (a) "Normal" has columns $q_1$ to $q_6$ and rows $k_1$ to $k_6$ and ten scattered dots ($k_1$: $q_1, q_2, q_4$; $k_2$: $q_3, q_6$; $k_3$ to $k_5$: $q_5$; $k_6$: $q_3, q_6$). (b) "Bucketed" reorders them, columns $q_1, q_2, q_4, q_3, q_6, q_5$ against rows $k_1, k_2, k_6, k_3, k_4, k_5$, and colours the cells by bucket: a blue $1 \times 3$ strip ($k_1$ against $q_1, q_2, q_4$), a maroon $2 \times 2$ block ($k_2, k_6$ against $q_3, q_6$) and a yellow $3 \times 1$ strip ($k_3$ to $k_5$ against $q_5$), stepping down to the right. (c) "Q = K" labels both axes with the queries in that order, $q_1, q_2, q_4, q_3, q_6, q_5$, and shows a coloured triangular staircase of blue, maroon and yellow blocks along the diagonal, with hollow circles on the diagonal cells and filled dots off it. (d) "Chunked" is the staircase of (c), with the same axes, and three black rectangular frames around chunks.

The notice sits at the lower right of the slide, beside the lower-right grid (d): "© Kitaev, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Kitaev, et al. (the Reformer figure: LSH bucketing pipeline and attention matrices). All rights reserved — excluded from the CC license.*

## Slide 61 — Even-larger-context transformers

Title: "Even-larger-context transformers". Subheading in bold italics: "Local + global". Bullets (names in bold):

- **Transformer XL** uses segment-level recurrence and fancy positional encoding to increase context (https://arxiv.org/abs/1901.02860)
- **Longformer** scales self-attention linearly with sequence length as opposed to quadratically, using deconstructed local + global attention (https://arxiv.org/abs/2004.05150)
- **Big Bird** uses a combo of random, dense sliding window, and global token attention to get sparsity, also O(n) (https://arxiv.org/pdf/2007.14062)

Below, a row of four square attention-pattern grids, each a fine grid of cells shaded in teal on a black frame, with captions in a serif face: (a) "Full $n^2$ attention", every cell shaded, with a darker diagonal; (b) "Sliding window attention", only a diagonal band of cells shaded, running from the top left to the bottom right; (c) "Dilated sliding window", a wider diagonal band in which only alternate cells are shaded (a checkered band); (d) "Global+sliding window", the diagonal band of (b) plus three full-width shaded global rows and three matching full-height columns crossing it: a two-cell band at rows 1–2 and single rows 6 and 16, with columns 1–2, 6 and 16. Each panel is a $24 \times 24$ grid. Under them, a caption: "Figure 2: Comparing the full self-attention pattern and the configuration of attention patterns in our Longformer."

The notice sits at the right, above panel (d) and below the third bullet: "© Beltagy, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". It is taken to cover the whole four-panel figure.

*OCW notice: © Beltagy, et al. (the Longformer attention-pattern figure). All rights reserved — excluded from the CC license.*

## Slide 62 — Even-larger-context transformers

![Slide 62 — Even-larger-context transformers](../images/10-architectures-memory/slide-62.jpg)

Title: "Even-larger-context transformers". Subheading in bold italics: "Retrieval-enhanced". Bullet:

- **RETRO** enables retrieval from trillion-token databases based on local similarity, swapping model parameters for direct lookup (helps separate language modeling from fact lookup) (https://arxiv.org/abs/2112.04426)

Below, a two-part diagram in blue, pink and mint green. Left: a pink rounded box with the large title "LARGE GPT", holding two blue rounded buttons side by side with white text, "Language Information" (with an italic letter A icon) and "World Knowledge Information" (with a globe icon). Right: a mint-green rounded box with the title "RETRO" in blue, holding one blue button "Language Information" (with the A icon). A thick blue line leaves the RETRO box with a ring at its left end and runs right into a mint-green cylinder labelled "Database" in blue, joining a server-stack icon at the top of the cylinder; at the cylinder's bottom, a blue button "World Knowledge Information" (with the globe icon). So the language information stays inside the RETRO box and the world knowledge sits in the database outside it.

Credit at the lower left: "Image courtesy of J. Alammar. Used under CC BY-NC-SA." The slide prints no "All rights reserved" notice. Its figure is a single raster image used nowhere else in the deck, so it reuses no picture from an excluded slide.

## Slide 63 — How far back can we go with attention?

Title: "How far back can we go with attention?". Bullets:

- BERT: 512 tokens
- GPT-2: 1024 tokens
- GPT-3: 2048 tokens
- GPT-4: 8,000 tokens, with a souped up 32K token version available
- Anthropic apparently has a model with a 100K token window (about 75K words)

## Slide 64 — When do we actually need long-term context?

![Slide 64 — When do we actually need long-term context?](../images/10-architectures-memory/slide-64.jpg)

Title: "When do we actually need long-term context?". Credit at the lower left: "Courtesy of Mangalam, et al. Used under CC BY." and an underlined URL "https://arxiv.org/abs/2308.09126". The slide prints no "All rights reserved" notice, and its figures are rasters used on no other slide.

Left: a bubble chart with its y-axis labelled "Certificate Length" (ticks 0, 20, 40, 60, 80, 100) and its x-axis labelled "Video Clip Length" (ticks 0, 50, 100, 150, 200). Each bubble is a video-understanding dataset, labelled by name, with the bubble's area apparently varying by dataset. Large bubbles: "EgoSchema" (green, at the top, near x = 180 and y = 100), "ActivityNet-QA" (pink, large, near x = 125 and y = 2), "AGQA" (tan) and "NextQA" (olive) near the origin; "LVU" is a small dark-red bubble near x = 210 and y = 17. A thick orange-brown arrow runs from "LVU" up to "EgoSchema" with the red label "5.7x" beside its head. Near the origin, a small red square surrounds a cluster of black triangles, and from it faint lines run to a large inset that zooms into that corner, with its own axes (x ticks 2.5, 10.0 and 17.5; y ticks 0.25, 1.00 and 2.00). The inset's bubbles are labelled "UCF101" (small pink), "Kinetics" (cyan, cut off at the top), "Something-Something" (blue), "HVU-Action" (pink-violet, large), "HOW2QA" (small green), "HVU-Concept" (violet, large), "MSRVTT" (small brown), "AVA" (orange), "Youtube-8m" (a small grey triangle) and "IVQA" (small tan).

Right, upper part: a vertical column of eight video frames of an indoor scene, the first, sixth and seventh in colour and the others (second to fifth, and eighth) greyed. A curly brace at the first frame points to an orange certificate icon, and a brace across the sixth and seventh frames points to another. Beside them, a flow diagram: a yellow label "Marked Ground Truth" on a blue box holding a hand-drawn squiggle (a waveform), with a blue arrow bending down and left into a salmon rounded square holding a police-officer icon; from the officer a salmon arrow points right to the bold word "Certified!". Under the words "Minimum Certificate Set", two certificate icons separated by a comma sit inside curly braces, and a green arrow runs from the left end of that braced pair up into the officer.

Right, lower part: a bar chart of six blue bars with their values printed above them and no axis title. Bars left to right with x-axis labels: 30 (15), 50 (17), 70 (11), 85 (14), 100 (22) and 150-180 (21); the bar heights are proportional to the values. No title, y axis or axis label says what the bars measure.

## Slide 65 — Memory fast and slow

![Slide 65 — Memory fast and slow](../images/10-architectures-memory/slide-65.png)

Title: "Memory fast and slow". A diagram in serif type. At the top, centred, the text: parameters are “slow memory”. Upper row: "Data" over $\lbrace \mathbf{x}^{(i)}, \mathbf{y}^{(i)} \rbrace_{i=1}^{N}$, an arrow to a grey box labelled "Learner", an arrow to "Parameters" over $\theta$, with a dotted curved leader from $\theta$ to the words "Statistic of the dataset". A dotted vertical arrow points down from $\theta$ to the lower box. Lower row: "Data" over $\mathbf{x}^{(i)}$, an arrow to a grey box labelled "Neural Net", an arrow to "Activations" over $\mathbf{h}^{(i)}$, with a dotted curved leader from $\mathbf{h}^{(i)}$ to the words "Statistic of a datapoint". At the bottom, centred: activations are “fast memory”.

This slide has no raster picture and prints no notice, so it reuses none from an excluded slide.

## Slide 66 — Fast weights? Slow activations?

Title: "Fast weights? Slow activations?". Bullets:

- **Hypernets** are nets that output weights of another net — these weights are a “fast memory” of the input to the hypernet.
- **Code books** use tensors of activations that are learned (backprop to activations). These activations are “slow memory” of the dataset you are learning.

## Slide 67 — 11. Memory and sequence modeling

Title as printed: "11. Memory and sequence modeling". The outline of slide 2 again, with the same four bullets: "CNNs for sequences", "RNNs", "LSTMs", "Sequence models and long memory".

## Slide 68 — 9. Memory and sequence modeling

Title as printed: "9. Memory and sequence modeling". The outline again with a shorter last bullet:

- CNNs for sequences
- RNNs
- LSTMs
- Sequence models

(Here the outline number is 9; slides 2 and 67 print 11 and the title slide says Lecture 10.)

## Slide 69 — MIT OpenCourseWare end page

OCW's appended end page (792×612, smaller than the slides). It prints the number "69" at the bottom centre, in larger type than the slides' numbers. Text: "MIT OpenCourseWare", "https://ocw.mit.edu", "6.7960 Deep Learning", "Fall 2024", "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms" (the URLs are blue and underlined). Not lecture content.
