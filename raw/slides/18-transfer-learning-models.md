---
title: Lecture 18 — Transfer Learning I (slide deck)
lecture: 18
slides: 77
source_pdf: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec18.pdf
note: Printed slide numbers 1–76 (bottom centre) equal the PDF page numbers exactly. Page 77 is OCW's appended end page (a smaller 792×612 page), which also prints 77, although slide_number_map.py reports it as printing no number.
figure_audit: Transcribed by Sonnet from page images, in two halves; 62 chart-, diagram-, table-, equation- and photo-heavy pages (3–6, 8–10, 12–17, 19–43, 45, 46, 48, 49, 52–54, 56–63, 65–70 and 73–75) were then checked by Opus, a different model, split between four agents (all finished), from 100–600 dpi crops, the embedded rasters at native resolution, the vector data and the text layer. No equation was wrong. 19 pages were corrected and 37 given precisions only; 6 were right as written. The corrections were counts and positions (slide 5's eight pictures, not nine; slide 20's arrows and cells; slide 41's 15 dots; slide 57's cells; slide 60's grid; slide 69's token rows; slide 75's columns and circles), chart readings (slide 37's KD map, slide 41's values, slide 45's red box and end points, slide 61's linear axis), directions (slide 52's two braces, slide 34's connectors), pictures misdescribed (slide 21's network drawing, slide 33's false-colour image, slide 70's edge maps), names from outside the page (slide 39's shark, slide 45's parameter count, slide 75's "autoregressive"), and printed details (slide 10's unflagged slip, slide 53's bold numerals, slide 74's URL and quotes). The renders of slides 45 and 75 were then checked against the corrected text.
---

# Lecture 18 — Transfer Learning I: slide-by-slide

Text and figures of all 77 pages of
[`mit6_7960_f24_lec18.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec18.pdf),
transcribed from the deck (speaker: Sara Beery; the title slide reads "Lecture 18: Transfer Learning I", and the recording's
title is "Transfer Learning: Models"). Cite these as "slide N": the printed number equals the PDF page number for slides 1–76;
page 77 is OCW's appended end page and prints 77. Diagrams, plots and pictures are described in prose since the KB is read as
text. The equations are LaTeXiT images whose text layer is base64 junk; all were read from the page, never from the text layer.

**Images.** Twenty-six slides have a rendered image under their heading: 3, 8, 10, 21, 26, 27, 28, 30, 31, 35, 37–43, 45, 46, 59, 66, 67, 68, 70, 73 and 75. The 21 slides with an OCW notice are never rendered, and slide 74 is withheld: its "Text Encoder" and "Image Encoder" graphics are the images slide 63 draws under its "© Radford, et al." notice. The others were not rendered because they are text, tables or screenshots of text that this file reproduces (2, 7, 11, 18, 32, 44, 47–51, 54, 56, 58, 71 and 76), build steps a rendered slide supersedes (9 and 29 by 8, 22–25 by 26, 36 by 37, 65 by 66), section slides (55, 64 and 72), the title (1) or the end page (77).

Companion pages: [wiki page for this lecture](../../wiki/18-transfer-learning-models.md) · [transcript](../transcripts/18-transfer-learning-models.md)

**Signposting slides you can skip.** Slide 1 is the title; slide 2 the outline (learning with little data; transferring
knowledge about the mapping: finetuning, domain adaptation; about the outputs: knowledge distillation; foundation models:
prompting), which slide 76 repeats without "Prompting"; slides 8, 9 and 29 are the "Transfer Learning" banner with its three
kinds of knowledge (the mapping, the outputs, the inputs), bolding the one that follows; slides 55, 64 and 72 are section
slides; slide 71 lists two links; slide 77 is the OCW end page. The third kind of knowledge, about the inputs, is not taught in
this deck: the recording leaves it to the next lecture (≈1:31). The deck prints no lecture number but its own.

Some slides are **build steps** or repeats, transcribed individually with a note of what they add: 12 and 13 (testing, then
finetuning, after contrastive pretraining); 14 to 17 (mismatched input and output dimensions, then the network split into
$f_1$, $f_2$, $f_3$); 22 to 26 (the unsupervised domain adaptation recipe built up box by box), which 27 redraws as ALDI and 28
sets beside five other methods; 30 and 31 (the same student bars); 36 and 37; 46 and 73; 48 and 49; 50 and 51; 60 to 62 (CLIP as
learner, then two adaptors); and 65 and 66. Slides with no printed title are headed here with a description: 6, 10, 19, 22 to
26, 28, 41, 42, 45 and 77.

**Printed slips and oddities**, kept as printed: slide 10's "Neural" between "Like" and "Dislike", in both panels; slide 21's slide number printed twice; slides 30 and 31's "giraffee"; slide 36's "students view" beside "teacher’s view"; slide 43's "231gCO2eq/kWh" with no space; "SimCLR" twice in slides 46 and 73's grid of foundation models; slide 49's "1." for the second step, repeating slide 48's; slide 61's "Linear classifer"; and slide 74's quotation, opened with a straight quote and closed with a curly one. Slide 44 prints "Dalle-2" and slide 75 "DALL-E 2". Some pages hide things: slides 4 and 5 cover the pasted figure's panel label with a white rectangle, slide 3's number is black on black, and the text layer holds white, invisible letters on slides 8, 9, 10, 29, 31, 33 and 36; several pasted figures are cropped by the page (slides 21, 27, 42 and 65). Each entry describes only what the page shows.

**OCW licence notices** are on slides 4, 5, 6, 12 to 17, 19, 20, 33, 34, 52, 53, 57, 60 to 63 and 69, 21 in all, all typed and
in the text layer; the slide file's set and the text layer's are the same. A scan for notices drawn as glyph outlines found none.
Each is transcribed as an `*OCW notice: …*` line with the picture it covers. Slides 27, 28, 38, 39, 43, 58, 59, 65 to 68, 70 and
75 carry "Courtesy of … Used under CC BY" credits (slide 43's "CC BY-NC-SA"), not notices, and slide 40's painting is "in the
public domain". Slides 3 ("[Slide credit: Andrew Ng]"), 32 (a table from Hinton, Vinyals and Dean), 41 (OpenAI's chart), 42
(Ahmed and Thompson's chart), 45, 54, 56 and 71 reproduce or credit third-party material with no notice.

## Contents

| Slides | Section |
| ------ | ------- |
| 1–7 | Title; outline; why deep learning (Andrew Ng's chart); few-shot learning (the "dax" and the vehicles); deep against human learning; pretraining as prior knowledge |
| 8–18 | Knowledge about the mapping: pretraining, adapting and testing on music; finetuning; contrastive pre-training; mismatched input and output dimensions; avoiding forgetting the general representation |
| 19–28 | Domain adaptation: source and target domains; DANN; unsupervised domain adaptation built up step by step; Align and Distill (ALDI) against five other methods |
| 29–39 | Knowledge about the outputs: cross-entropy recalled; knowledge distillation and soft targets; cross-modal distillation and SoundNet; distilling an ensemble; contrastive representation distillation; compression against distillation; specialization |
| 40–54 | Foundation models: standing on the shoulders of giants; compute and parameter growth; climate cost; what training takes; performance on 58 tasks; learner and adaptor; "just ask"; GPT-3; new capabilities; few-shot learning and extrapolation |
| 55–63 | Prompting: hand-crafted templates; soft prompts; chain-of-thought; CLIP as learner, with a linear probe, and "just ask"; sketch against photo |
| 64–71 | Prompting in other modalities: visual prompting and adversarial reprogramming; the single pink pixel; adaptation on inputs, mapping or outputs; visual prompt tuning; visual prompting via inpainting; resources |
| 72–77 | Combining foundation models: VQGAN with CLIP; CLIP+GAN; DALL-E 2; summary; OCW end page |

## Slide 1 — Lecture 18: Transfer Learning I

Title: "Lecture 18: Transfer Learning I". Below it: "Speaker: Sara Beery". The rest of the slide is empty white. Grey footer: MIT logo, "6.7960 Deep Learning", the link "https://phillipi.github.io/6.7960" (underlined), and "Fall 2024" at the right. The printed slide number "1" sits just after the link.

## Slide 2 — Transfer Learning I

Title: "Transfer Learning I". An outline in bullets:

- "Learning with little data"
- "Transferring knowledge about the mapping"
  - "Finetuning"
  - "Domain adaptation"
- "Transferring knowledge about the outputs"
  - "Knowledge distillation"
- "Foundation models"
  - "Prompting"

Printed slide number "2".

## Slide 3 — Why deep learning

![Slide 3 — Why deep learning](../images/18-transfer-learning-models/slide-3.jpg)

White-on-black slide. Title: "Why deep learning". A line chart with white axes: a vertical axis with an arrowhead labelled "Performance" (rotated) and a horizontal axis with an arrowhead labelled "Amount of data". No tick marks or numbers. Two curves, both starting near the origin:

- Blue (cyan) curve, labelled "Deep learning" at its right end: rises steeply at first, slightly below the red curve at the very start, crosses the red curve about a third of the way along the x-axis, and then keeps rising roughly linearly to the top right, ending highest of the two.
- Red curve, labelled "Older learning algorithms" at its right end: rises quickly, slightly above the blue at first, then flattens into a horizontal plateau at mid height from about the crossing point to the right end.

Caption below the chart: "How do data science techniques scale with amount of data?" The title, the chart and the caption are one pasted picture (a screenshot of the credited slide); only the credit at the lower right, "[Slide credit: Andrew Ng]", is live text. The printed slide number "3" is in the text layer at bottom centre but is black on black and so not visible.

## Slide 4 — Few-shot Learning

*OCW notice: © AAAS. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the handwritten-character pictures).*

Title: "Few-shot Learning". At the upper left of the slide, centred above the grid, a single handwritten character, in black ink on white, inside a red square outline. Below it is a grid of 20 handwritten characters in 4 rows by 5 columns, each in its own black-outlined cell, all drawn in a similar script of loops and curls. Nothing in the grid is marked; the character in row 2, column 4 has the same shape as the boxed one. (The pasted figure's panel label at its top left is covered by a white rectangle.)

Text at the right: "This is a “dax”." then "Which of the below symbols are also daxes?"

Citation at the lower right: "[Lake, Salakhutdinov, Tenenbaum, 2015]". Printed slide number "4".

## Slide 5 — Few-shot Learning

*OCW notice: © AAAS. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the vehicle pictures).*

Title: "Few-shot Learning". At the upper left of the slide, centred above the other pictures, a picture of a black two-wheeled upright standing vehicle with a tall handlebar post, seen at three-quarter view, inside a red rectangular outline. Below it are eight pictures of wheeled vehicles in three columns. Left column, top to bottom: a one-wheeled vehicle with a saddle on a tall post; a red and chrome motorcycle; a green and white bicycle. Middle column: a scooter with a black frame and deck, a tall handlebar and dark blue tyres; below it, one tall picture spanning the lower two rows, a black saddle on a post above a single wheel in a fork. Right column: a grey car with dark stripes; an upright two-wheeled standing vehicle with a tall handlebar post, seen from the front, similar to the one in the red box; a blue and black motorcycle. (As on slide 4, the pasted figure's panel label is covered by a white rectangle.)

Text at the right: "Which of these is an example of the same concept as the item in the box?"

Citation at the lower right: "[Lake, Salakhutdinov, Tenenbaum, 2015]". Printed slide number "5".

## Slide 6 — (no title; “Deep learning” against Human learning)

*OCW notice: © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right, above the bottom question; it covers the blackboard and head pictures).*

The slide is split by a vertical dotted line. The left heading is "“Deep learning”" (in quotation marks) and the right heading is "Human learning".

Left: a picture of an empty blackboard (dark slate surface) in a wooden frame, standing against a pale blue-green wooden wall.

Right: a side-view head silhouette outlined in dark blue, facing left, filled with many gears in light cyan, mid blue and dark navy, which fill the cranium and run down into the neck. Three black leader lines start just clear of the back of the head and fan out to three labels stacked at the right: the top one rises to "Representations", the middle one runs nearly level to "Models", the bottom one curves down to "Skills".

Bottom, large text spanning the width: "How can we give deep nets prior knowledge?" Printed slide number "6".

## Slide 7 — How can we give deep nets prior knowledge?

Title: "How can we give deep nets prior knowledge?" (left-aligned). Three text blocks:

- "Just “pretrain” them on prior tasks!"
- "Then transfer knowledge from the pretrained nets to solve novel tasks."
- "One view: the **point** of deep learning is to enable problem solving with little data" (the word "point" is bold italic).

Printed slide number "7".

## Slide 8 — Transfer Learning

![Slide 8 — Transfer Learning](../images/18-transfer-learning-models/slide-8.jpg)

Title: the grey banner across the lower part of the slide reads "Transfer Learning" (large). At the left, a numbered list in plain weight: "1. Knowledge about the mapping", "2. Knowledge about the outputs", "3. Knowledge about the inputs".

At the right is a line drawing of a tall box (a rectangular prism in isometric view). The flat rhombus at the bottom is labelled "Inputs" (label at its left corner) and the rhombus at the top is labelled "Outputs" (label at its left corner). Five thin dotted curved arrows run from five black dots on the bottom face up to arrowheads on the top face; three start at the left and centre of the bottom face and end at the left part of the top face, and two start at the right of the bottom face and end at the right part of the top face. The arrows bow slightly inward toward the middle of the box as they rise. The picture shows a mapping from inputs to outputs.

Printed slide number "8".

## Slide 9 — Transfer Learning

Build step of slide 8: the same box diagram and list, but item "1. Knowledge about the mapping" is now in bold, and all five dotted arrows between the "Inputs" face and the "Outputs" face, with their start dots and arrowheads, are drawn thicker and bolder. Items 2 and 3 ("Knowledge about the outputs", "Knowledge about the inputs") stay in plain weight. The grey banner reads "Transfer Learning". Printed slide number "9".

## Slide 10 — (no title; pretraining, adapting and testing on music)

![Slide 10 — (no title; pretraining, adapting and testing on music)](../images/18-transfer-learning-models/slide-10.png)

Three panels separated by two vertical dotted lines. Each panel has a grey-shaded box heading in a serif font, a line of sans-serif text below it, and a diagram: a pair of beamed musical notes at the left, an arrow-less trapezoid (wide at the left, narrow at the right) containing a function letter, an arrow to the right, and a vertical column of grey-shaded circles with monospaced labels.

- Left panel. Heading "Pretraining", subheading "Genre recognition". The trapezoid contains $f$. The column has six circles labelled, from top to bottom, "classical" (mid grey), "hip hop" (light grey), "rock" (white), "metal" (dark grey), "alternative" (mid grey), "rap" (black). Italic caption below: "A lot of data".
- Middle panel. Heading "Adapting", subheading "Preference prediction". The trapezoid contains $f'$ (f prime). The column has three circles labelled "Like" (light grey), "Neural" (dark grey), "Dislike" (white). ("Neural" is printed so in both the Adapting and the Testing panel; between "Like" and "Dislike" it presumably stands for "Neutral", a reading from outside the course material.) Italic caption below: "A little data".
- Right panel. Heading "Testing", subheading "Preference prediction". The trapezoid contains $f'$. The column has three circles labelled "Like" (black), "Neural" (light grey), "Dislike" (dark grey). No caption below.

Each column of circles sits in a black-outlined box, with a short arrow into it from the trapezoid. The page does not say what the shading means. Printed slide number "10".

## Slide 11 — Finetuning

Title: "Finetuning". Three bullets (the symbols are bold in the deck):

- "Pretrain a network on task A, resulting in parameters **W** and **b**"
- "Initialize a second network with some or all of **W** and **b**"
- "Train the second network on task B, resulting in parameters **W’** and **b’**"

Centred text near the bottom: "The “learned representation” is the *encoder* (its weights and biases) so that’s what we transfer" ("encoder" is italic). Printed slide number "11".

## Slide 12 — Contrastive pre-training

*OCW notice: Tigers © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the tiger photographs).*

Title: "Contrastive pre-training". Two panels separated by a vertical dotted line.

Left panel: grey heading box "Pretraining", subheading "Self-supervised contrastive learning". Two small square photographs at the left: the upper one a tiger seen from the side in profile against greenery and rock, the lower one a tiger's legs and body walking by water on a rock. Each photograph has a short black arrow to its own copy of a network drawn as three white vertical rectangles of decreasing height (tall, medium, short), joined by short arrows, so there are two identical stacked networks. From the right end of each network's short rectangle a curved black arrow runs to a single black dot inside a large coral-red circle at the right (the dot sits near the circle's left edge, not at its centre); the upper arrow curves down to the dot and the lower arrow curves up to it, so both images land at the same point.

Right panel: grey heading box "Testing", subheading "New recognition task". A square photograph of a tiger's face looking at the camera, an arrow to a three-rectangle network of the same shape (tall, medium, short), and an arrow from it to a column of eight grey-shaded circles in a box, with labels from top to bottom: "dolphin" (mid grey), "cat" (light grey), "grizzly bear" (white), "angel fish" (dark grey), "chameleon" (mid grey), "tiger" (black), "iguana" (mid grey), "elephant" (light grey).

Printed slide number "12".

## Slide 13 — Contrastive pre-training

*OCW notice: Tigers © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the tiger photographs).*

Build of slide 12. The left panel ("Pretraining", "Self-supervised contrastive learning") is unchanged. The right heading box now reads "Finetuning" (not "Testing"), with the subheading "New recognition task" still. The tiger-face photograph, its arrow, the arrow into the output column and the same eight-circle output column (dolphin, cat, grizzly bear, angel fish, chameleon, tiger, iguana, elephant, with the same shading, "tiger" black) remain, but the three-rectangle network, with the arrows between its rectangles, is gone, leaving a blank gap there. Printed slide number "13".

## Slide 14 — What if the input/output dimensions don’t match?

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the two field photographs).*

Title: "What if the input/output dimensions don’t match?". Two rows, each with a grey heading box at the left.

Top row, heading "Pretraining": above a colour photograph of a green crop field is the script letter $\mathcal{X}$; an arrow leads to a light-blue square labelled $f$; an arrow leads to a box of two circles labelled in monospace "wheat" (light grey circle) and "corn" (dark grey circle), with the script letter $\mathcal{Y}$ above it.

Bottom row, heading "Finetuning": above a greyscale photograph of rows of crop plants is $\mathcal{X}'$; an arrow to the same light-blue square $f$; an arrow to a box of three circles labelled "wheat" (dark grey), "corn" (light grey), "soybeans" (white), with $\mathcal{Y}'$ above it.

Printed slide number "14".

## Slide 15 — What if the input/output dimensions don’t match?

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the two field photographs).*

Title: "What if the input/output dimensions don’t match?". Same as slide 14, with one addition: at the right, between the two rows, a red-outlined box holding two lines of math:

$$\mathcal{X}' \neq \mathcal{X}$$

$$\mathcal{Y}' \neq \mathcal{Y}$$

Printed slide number "15".

## Slide 16 — What if the input/output dimensions don’t match?

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the upper right, under the title; it covers the two field photographs).*

Title: "What if the input/output dimensions don’t match?". Same two rows, now with the network opened into three parts.

Top row ("Pretraining"): the colour field photograph ($\mathcal{X}$), an arrow, a narrow tall blue rectangle $f_1$, an arrow, the label $\mathcal{Z}_ 1$, an arrow, a large blue square $f_2$, an arrow, the label $\mathcal{Z}_ 2$, an arrow, a narrow tall blue rectangle $f_3$, an arrow, the output box (wheat light grey, corn dark grey) with $\mathcal{Y}$ above it.

Bottom row ("Finetuning"): the greyscale field photograph ($\mathcal{X}'$), an arrow, a narrow tall rectangle $f_1'$ in white (unfilled), an arrow, $\mathcal{Z}_ 1$, an arrow, the same large blue square $f_2$, an arrow, $\mathcal{Z}_ 2$, an arrow, a narrow tall white rectangle $f_3'$, an arrow, the output box (wheat dark grey, corn light grey, soybeans white) with $\mathcal{Y}'$ above it.

Two tall green outlined rectangles each run down from the top row to the bottom row, one around both $\mathcal{Z}_ 1$ labels and one around both $\mathcal{Z}_ 2$ labels, one per intermediate representation. The blue $f_2$ appears unprimed in both rows, while $f_1'$ and $f_3'$ in the bottom row are white; the recording explains the picture: the $\mathcal{Z}$'s are points of the same dimensionality in both networks, and "f1 goes to f1 prime … f3 goes to f3 prime … the f2 can stay the same" (≈18:38–19:23). The narrow $f_1$ and $f_3$ rectangles are as tall as the $f_2$ square and about a third as wide. Printed slide number "16".

## Slide 17 — What if the input/output dimensions don’t match?

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the upper right, under the title; it covers the two field photographs).*

Title: "What if the input/output dimensions don’t match?". Same layout as slide 16 with the green outlines removed and the finetuning row's blocks now all filled light turquoise rather than white and blue: $f_1'$ (narrow tall), $f_2'$ (large square, now primed and turquoise) and $f_3'$ (narrow tall), with $\mathcal{Z}_ 1$ and $\mathcal{Z}_ 2$ between them. The top row is unchanged (blue $f_1$, $f_2$, $f_3$). All three blocks of the bottom row are now primed and recoloured. Printed slide number "17".

## Slide 18 — How to avoid forgetting the general representation?

Title: "How to avoid forgetting the general representation?". Text: "Avoiding overfitting on a small amount of data for a new task while still enabling specialization can be a balancing act. A few methods:"

- "Freeze most of the network, train only the final layers"
- "Use a small learning rate"
- "Early stopping"
- "Continue to train on the original data as well"

Closing text: "All assume that you have a good, representative validation set for your task (not always possible when you may see distribution shift)". Printed slide number "18".

## Slide 19 — (no title; domain adaptation between a source and a target domain)

*OCW notice: Images © He, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the figure).*

No slide title. A figure with a blue circle at the upper left and a green circle at the lower left. The blue circle holds 16 solid blue shapes: four stars, four circles, four squares and four triangles, scattered. The green circle holds 16 unfilled green-outlined shapes of the same four kinds (triangles, circles, stars, squares). Between the left and right parts is a hollow right-pointing arrow with the word "Domain" above it and "Adaptation" below it. At the right, a blue circle and a green circle, the green offset to the right, overlap over most of their area. The merged picture holds 24 shapes, three of each kind from each domain: squares at the left, stars along the top, circles at the right and triangles along the bottom, each kind's blue and green shapes side by side. The whole figure, legend included, is one pasted picture.

Legend at the bottom: "Source domain:" followed by a solid blue circle, star, triangle and square; "Target domain:" followed by an unfilled green square, triangle, circle and star.

At the bottom right: the URL "https://link.springer.com/article/10.1007/s10489-022-03709-8". Printed slide number "19".

## Slide 20 — Domain Adversarial Neural Networks (DANN)

*OCW notice: © Ganin, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the diagram).*

Title: "Domain Adversarial Neural Networks (DANN)". A colour-coded network diagram.

- Green part (feature extractor): a grey slab labelled "input $\mathbf{x}$" (text running diagonally below it) at the left, followed by three green blocks (two slabs and a cube) joined by green block arrows, ending in a thin green vertical bar labelled "features $\mathbf{f}$" (rotated text). A bracket under the green blocks reads "feature extractor $G_f(\cdot; \theta_f)$".
- Blue part (label predictor): from the features bar, three blue block arrows and two tall thin blue bars between them lead to a small stacked blue block of five cells labelled "class label $y$". A bracket under it reads "label predictor $G_y(\cdot; \theta_y)$". A blue curved arrow runs left from a blue oval "loss $L_y$" (above and right of the class label, not joined to it) through a box with $\frac{\partial L_y}{\partial \theta_y}$ to the top of the features bar, where a green arrow takes over, runs left through a box with $\frac{\partial L_y}{\partial \theta_f}$, and ends with its arrowhead pointing down-left above the first green block, next to the input.
- Pink part (domain classifier): from the features bar, a hollow black arrow labelled "gradient reversal layer" (rotated) points down-right to a tall pink bar, then two pink arrows and a thin pink bar lead to a small pink cube labelled "domain label $d$". A bracket above it reads "domain classifier $G_d(\cdot; \theta_d)$". A pink curved arrow runs back from a pink oval "loss $L_d$" through a box $\lambda \frac{\partial L_d}{\partial \theta_d}$ to the gradient reversal layer, then continues as a green arrow with the box $-\lambda \frac{\partial L_d}{\partial \theta_f}$ back to the input side (to the left, below the green blocks).
- Legend in a rounded box at the lower left: a grey block arrow labelled "forwardprop" and a grey boxed $\frac{\partial L_y}{\partial \theta_f}$ with a left-pointing grey arrow labelled "backprop (and produced derivatives)".

At the bottom right: the URL "https://arxiv.org/abs/1505.07818". Printed slide number "20".

## Slide 21 — Unsupervised Domain Adaptation (UDA)

![Slide 21 — Unsupervised Domain Adaptation (UDA)](../images/18-transfer-learning-models/slide-21.jpg)

Title: "Unsupervised Domain Adaptation (UDA)". The page has no OCW notice and no credit line in view. Two rows of pictures feed one model.

Upper left, the label "*Unlabeled* target" (asterisks printed). Next to it, a pair of tall images side by side, each a dark navy rectangle holding a pale triangular wedge (broad at the top, narrowing to a point at the bottom) speckled with yellow dots, brighter yellow near the bottom point; each has one small white rectangular outline inside the wedge (in the left image at the right edge of the wedge about a third of the way down, in the right image near the top, right of centre). A thick black arrow points right to a small line drawing of a detection network: a thick slab whose face carries three outlined rectangles and a grid patch, an arrow labelled "RoIAlign" from the grid patch to the first of two gridded blocks, the word "conv" on the arrow between the blocks and on the line leaving the second, and a line from the first block up to a small flat box with an arrow to the label "class box" (the text is very small).

Lower left, the label "Labeled source". Next to it, a pair of similar tall wedge images, darker and greyer with fewer yellow specks, cut off at the bottom while still wide; the left one has a larger white rectangular outline at the upper right holding a pale diagonal streak, the right one a small white rectangle at the top left. Both pairs are crops of one larger pasted figure; nothing on the page says what the pictures show. A thick black arrow points right to a second small copy of the same detection-network drawing.

At the centre right, a larger copy of the same network drawing, with two thick black arrows pointing into it, one coming from the unlabeled target images (down-right) and one from the labeled source images (up-right). To its right, the large words "UDA model".

Printed slide number "21", printed twice in two fonts, the second overlapping the lower half of the first (a doubled number).

## Slide 22 — (no title; the target dataset through a model and a threshold)

No slide title. A diagram built from plain shapes: a circle labelled "Img" at the lower middle, with the words "Target dataset" to its left. A black arrow runs from the "Img" circle up and to the right and then up, into a grey rectangle labelled "Model". A short line connects the "Model" rectangle upward to a second grey rectangle labelled "Threshold", and a line from it runs up to a circle labelled "Pred" at the top. At the lower right corner, partly cut off by the slide edge, is a tall dark navy image holding a pale triangular wedge with yellow specks and a small white rectangular outline at the right edge of the wedge about a third of the way down (the same kind of picture as on slide 21). Printed slide number "22".

## Slide 23 — (no title; a student trained on a thresholded model's predictions)

No slide title. Build of slide 22. The "Target dataset" "Img" circle now has two arrows going up: the right one into "Model", as on slide 22 (with "Threshold" and a "Pred" circle above it), and a new left one into a larger grey rectangle labelled "Student", with a grey dashed vertical line inside it, and a line up to a second "Pred" circle at the upper left. A thin arrow runs from the right "Pred" circle to the left "Pred" circle, labelled "Supervision". The same wedge image sits at the lower right. Printed slide number "23".

## Slide 24 — (no title; student and teacher with an exponential moving average)

No slide title. Build of slide 23, with the "Model" rectangle renamed "Teacher" and a new arrow between the boxes: a black arrow from the "Student" rectangle pointing right to the "Teacher" rectangle, labelled "EMA". Everything else is as on slide 23: "Target dataset" "Img" circle with arrows to "Student" and "Teacher"; "Threshold" above "Teacher"; two "Pred" circles joined by an arrow labelled "Supervision" pointing left; the dashed line in "Student"; the same wedge image at the lower right. Printed slide number "24".

## Slide 25 — (no title; adding the source dataset with ground-truth supervision)

No slide title. Build of slide 24. The diagram is split by a thin vertical grey line into a left "Source dataset" side and a right "Target dataset" side. The "Student" rectangle is now wider and spans the dividing line, with two dashed vertical lines inside it, one in the left (source) part and one in the right (target) part. New on the left: a circle labelled "Img" with the words "Source dataset" to its left, and an arrow from it up into the Student's left part; a line from the Student's left part up to a "Pred" circle; and, to the left of that "Pred", a circle labelled "GT" with an arrow labelled "Supervision" pointing right from "GT" to "Pred". The right side is as on slide 24 ("Pred", "Supervision" arrow pointing left, "Threshold", "Teacher", "EMA" arrow from "Student", "Target dataset" "Img" with two arrows). Two wedge images appear at the bottom corners: at the lower left, on the source side, a greyer, fainter wedge with a larger white rectangular outline at its upper right, and at the lower right, on the target side, the yellow-speckled wedge with a small white rectangle (the page does not label either picture). Printed slide number "25".

## Slide 26 — (no title; alignment, distillation and softmax added in red)

![Slide 26 — (no title; alignment, distillation and softmax added in red)](../images/18-transfer-learning-models/slide-26.jpg)

No slide title. Build of slide 25 with new red labels: "Distillation" replaces the word "Supervision" on the right-hand arrow (the left-hand "Supervision" arrow from "GT" stays black); the box above "Teacher" now reads "Softmax" in red (it read "Threshold"); and inside the "Student" rectangle a red word "Alignment" with a black double-headed horizontal arrow beneath it joins the two dashed lines. The remaining labels ("GT", "Pred", "Student", "Teacher", "EMA", "Source dataset", "Target dataset", the "Img" circles) and the two corner wedge images are as on slide 25. Printed slide number "26".

## Slide 27 — Align and Distill (ALDI)

![Slide 27 — Align and Distill (ALDI)](../images/18-transfer-learning-models/slide-27.jpg)

Title (at the upper right): "Align and Distill (ALDI)". The left half is a small copy of the slide 26 diagram (with "Alignment", "Distillation" and "Softmax" in red, and no wedge images). The right half is the same diagram redrawn with symbols and coloured boxes:

- Top row: a circle $y_{src}$ joined by a double-headed arrow labelled $L_{sup}$ to a circle $p_{src}$; a circle $p_{tgt}$ joined by a long double-headed arrow, labelled in an orange box $L_{distill}$, to a circle $\hat{p}_ {tgt}$; a line from $\hat{p}_ {tgt}$ down through a break marked "SG" (a double slash) to a yellow box "Postprocess".
- Middle: a wide grey box "Student $\theta_{stu}$" with two dashed vertical lines and, between them, a red box $L_{align}$ above a double-headed arrow; an arrow labelled "EMA" pointing right to a grey box "Teacher $\theta_{tch}$" under the Postprocess box.
- Lower row: circles $x_{src}^{t}$, $x_{tgt}^{t}$ and $x_{tgt}^{\hat{t}}$ below the student and teacher. A purple box $t \sim T_{src}$ with an arrow up to $x_{src}^{t}$, under which is a circle $x_{src}$ with the label $\lvert x_{src} \rvert = B_{src}$ ($B_{src}$ in a green box). A blue box $t \sim T_{tgt}$ with an arrow up-left to $x_{tgt}^{t}$; the text $\hat{t} \sim T_{weak}$ beside an arrow up-right to $x_{tgt}^{\hat{t}}$; both come from a circle $x_{tgt}$ with $\lvert x_{tgt} \rvert = B_{tgt}$ ($B_{tgt}$ in a green box).
- The words "Source" and "Target" label the left and right sides, split by a thin vertical line.

*Courtesy credit (not an OCW notice):* "Courtesy of Kay, et al. Used under CC BY." (small print at the lower right). Also at the lower right: the text "aldi-daod.github.io" and a QR code. Printed slide number "27".

## Slide 28 — (no title; ALDI compared with five other methods)

![Slide 28 — (no title; ALDI compared with five other methods)](../images/18-transfer-learning-models/slide-28.jpg)

No slide title. Six small diagrams of the same form as the right half of slide 27 (the symbols $y_{src}$, $p_{src}$, $p_{tgt}$, $x_{src}$, $x_{tgt}$ and so on, a student box and, in most, a teacher box), each with a title. The ALDI diagram sits at the upper left inside a framed card with a drop shadow; the other five are at the upper middle, upper right, lower left, lower middle and lower right.

- "ALDI": as on slide 27. Orange $L_{distill}$, red $L_{align}$, yellow "Postprocess", "Teacher $\theta_{tch}$", purple $t \sim T_{src}$, blue $t \sim T_{tgt}$, $\hat{t} \sim T_{weak}$, $\lvert x_{src} \rvert = B_{src}$, $\lvert x_{tgt} \rvert = B_{tgt}$.
- "SADA" (upper middle): student only, no teacher; red box $L_{align}$ with the word "SADA" under it; a purple box "Flip" on the source side and a blue box "Flip" on the target side; $\lvert x_{src} \rvert = M_{src} = 24$ and $\lvert x_{tgt} \rvert = M_{tgt} = 24$ (the $M$ terms in green boxes).
- "PT" (upper right): student with no $L_{align}$ box; orange $L_{distill}$ with the word "Soft"; a yellow box "Softmax, Sum"; "Teacher $\theta_{tch}$"; purple "Flip" on the source side; a blue box "Flip, Jitter, Cutout" on the target side and the text "Flip, Scale" beside the arrow to the teacher side; $M_{src} = 16$ and $M_{tgt} = 32$.
- "AT" (lower left): red $L_{align}$ with the word "Image"; orange $L_{distill}$ with "Hard"; a yellow box "Thresh, NMS"; a purple box "Flip, Jitter (1/2), Cutout (1/2)" on the source side; a blue box "Flip, Jitter, Cutout" and the text "Flip, Scale" on the target side; $M_{src} = 16$ and $M_{tgt} = 32$.
- "UMT" (lower middle): red $L_{align}$ with "Img2Img"; orange $L_{distill}$ with "Hard"; yellow "Thresh, NMS"; no box on the source arrow; a blue box "CropPad, Jitter" and the text "Flip, Scale" on the target side; $M_{src} = 24$ and $M_{tgt} = 24$.
- "MIC" (lower right): red $L_{align}$ with "SADA"; orange $L_{distill}$ with "Hard"; yellow "Thresh, NMS"; a purple box "Flip" on the source side; a blue box "Flip, Jitter, MIC" and the text "Flip, Scale" on the target side; $M_{src} = 24$ and $M_{tgt} = 24$.

*Courtesy credit (not an OCW notice):* "Courtesy of Kay, et al. Used under CC BY." (small print at the lower right, under the MIC diagram). Printed slide number "28".

## Slide 29 — Transfer Learning

Title: the grey banner reads "Transfer Learning". Same numbered list and box diagram as slides 8 and 9 (items "1. Knowledge about the mapping", "2. Knowledge about the outputs", "3. Knowledge about the inputs"; the prism with "Inputs" and "Outputs" labels and five thin dotted arrows from the bottom face to the top face). Here item "2. Knowledge about the outputs" is bold, the other two plain, and the top ("Outputs") face of the box is shaded light grey. Printed slide number "29".

## Slide 30 — Recall: training a classifier with cross-entropy

![Slide 30 — Recall: training a classifier with cross-entropy](../images/18-transfer-learning-models/slide-30.png)

Title: "Recall: training a classifier with cross-entropy".

Left: a small black horizontal bar labelled $\mathbf{y}$ above it, and to its right a dotted double-headed arrow to a second black bar labelled $\hat{\mathbf{y}}$ above it. Below the second bar is a trapezoid (narrow at the top, wide at the bottom) containing $f$, with $\mathbf{x}$ underneath. Under the first bar, the word “Teacher” in quotation marks; under the second, “Student” in quotation marks. Below them the equation:

$$L(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_i \mathbf{y}_ i \log \hat{\mathbf{y}}_ i$$

Right: two bar charts, one above the other, each with the same nine category slots on the x-axis labelled (rotated text) "elephant", "cat" (bold), "giraffee", "monkey", "dog", "seal", "tiger", "duck" and "..." (the printed spelling "giraffee" is kept). Both y-axes run from 0 to 1.0.

- Upper chart, labelled $\mathbf{y}$: one data series of black bars. "cat" is 1.0; every other category has no bar (0).
- Lower chart, labelled $\hat{\mathbf{y}}$: one data series of black bars, approximately "elephant" 0.13, "cat" 0.42, "giraffee" 0.29, "monkey" 0.08, "dog" 0.18, "seal" 0.08, "tiger" 0.32, "duck" 0.16; no bar at "...".

Printed slide number "30".

## Slide 31 — Knowledge Distillation

![Slide 31 — Knowledge Distillation](../images/18-transfer-learning-models/slide-31.jpg)

Title: "Knowledge Distillation", followed by the citation "[Hinton, Vinyals, Dean, 2015]".

Left: a large trapezoid (narrow at the top, wide at the bottom) containing $f^T$ and a grey padlock icon at its lower right, with $\mathbf{x}$ beneath it and the word "Teacher" below that; above it a wide black bar labelled $\mathbf{y}^T$. At its right, a smaller trapezoid containing $f^S$, with $\mathbf{x}$ beneath it and the word "Student" below; above it a shorter black bar labelled $\mathbf{y}^S$. A dotted double-headed arrow joins the two bars. Below, the equation:

$$L(\mathbf{y}^S, \mathbf{y}^T) = - \sum_i \mathbf{y}^T_ i \log \mathbf{y}^S_ i$$

Right: two bar charts with the same category slots on the x-axis (rotated labels): "elephant", "cat" (bold), "giraffee", "monkey", "dog", "seal", "tiger" (bold), "duck", "..." (the printed spelling "giraffee" is kept). Both y-axes run from 0 to 1.0.

- Upper chart, labelled $\mathbf{y}^T$: one series of black bars, approximately "elephant" 0.04, "cat" 0.61, "giraffee" 0.15, "monkey" 0.04, "dog" 0.04, "seal" 0.04, "tiger" 0.55, "duck" 0.08; no bar at "...".
- Lower chart, labelled $\mathbf{y}^S$: one series of black bars, approximately "elephant" 0.13, "cat" 0.42, "giraffee" 0.29, "monkey" 0.08, "dog" 0.18, "seal" 0.08, "tiger" 0.32, "duck" 0.16; no bar at "..." (the same bars as the $\hat{\mathbf{y}}$ chart on slide 30).

Printed slide number "31".

## Slide 32 — Knowledge Distillation

Title: "Knowledge Distillation", followed by "[Hinton, Vinyals, Dean, 2015]". Left bullets:

- "The “label” you get for each input is more informative than just the ground truth class"
- "The “label” (**soft target**) says not just “this is a dog” but “this is a dog that looks a bit like a cat and very different from a bear” (so maybe it’s a small dog)."

Right, centred text: "Hinton paper shows lower train accuracy but higher test accuracy on speech recognition with soft targets". Below it a table reproduced from that paper, typeset as text (no OCW notice or credit is printed):

| System & training set | Train Frame Accuracy | Test Frame Accuracy |
| --- | --- | --- |
| Baseline (100% of training set) | 63.4% | 58.9% |
| Baseline (3% of training set) | 67.3% | 44.5% |
| Soft Targets (3% of training set) | 65.4% | 57.0% |

Caption beneath: "Table 5: Soft targets allow a new model to generalize well from only 3% of the training set. The soft targets are obtained by training on the full training set." Printed slide number "32".

## Slide 33 — Cross-Modal Distillation

*OCW notice: © Couprie, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the two pictures of the room).*

Title: "Cross-Modal Distillation", followed by "[Gupta, Hoffman, Malik, 2015]". Two trapezoids side by side, both the same size: left, $f^T$ with a grey padlock at its lower right, a wide black bar above labelled $\mathbf{y}^T$; right, $f^S$ with a black bar above labelled $\mathbf{y}^S$. A dotted double-headed arrow joins the two bars. Below the left trapezoid is a photograph of a classroom interior (desks, a whiteboard, bulletin boards) with $\mathbf{x} \in \mathcal{X}$ and the word "Teacher" underneath. Below the right trapezoid is a false-colour image of the same view: an orange-to-red field in the upper half, the desks and chairs as flat green and cyan shapes with yellow bands around them in the middle, light blue at the lower left and right, and a large dark-blue shape (the near desk) at the bottom centre, with $\mathbf{y} \in \mathcal{Y}$ and the word "Student" underneath. The page does not say what the colours encode.

Text at the right: "Useful if you have labeled data for one domain but not the other." Printed slide number "33".

## Slide 34 — Cross-Modal Distillation: SoundNet

*OCW notice: © Aytar, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the architecture diagram).*

Title: "Cross-Modal Distillation: SoundNet", followed by "[Aytar*, Vondrick*, Torralba, 2016]" (asterisks printed). A diagram:

- At the upper left a stack of grey squares labelled "Unlabeled Video". A horizontal line labelled "RGB Frames" runs right and splits, with arrows, into two chains of grey cubes of decreasing size ("Visual Recognition Networks"): the upper chain is labelled "ImageNet CNN" and the lower "Places CNN". From the upper chain a long arrow labelled "Object Distribution" points right to "KL"; from the lower chain a long arrow labelled "Scene Distribution" points right to a second "KL". Each KL has an arrow pointing up into it from below (a vertical line at the right).
- From the video a line goes down and right, labelled "Raw Waveform", into a row of blue vertical bars of decreasing height, labelled left to right "Input" (a vertical line), "conv1", "pool1", "conv2", "pool2", "conv3", "conv4", "conv5", "pool5", "conv6", "conv7", "conv8". Pool layers are drawn in a darker blue ("pool1", "pool2", "pool5"); conv layers in lighter blue; the last layer "conv8" is two stacked blocks: a line from the upper block runs right and up into the lower "KL" (Scene), and one from the lower block runs further right and up into the upper "KL" (Object); "KL" is unboxed text. The bars get shorter and wider from left to right, and each grey chain has five cubes. The block is captioned "SoundNet Architecture" and "Deep 1D Convolutional Network".

Printed slide number "34".

## Slide 35 — Distilling an ensemble

![Slide 35 — Distilling an ensemble](../images/18-transfer-learning-models/slide-35.jpg)

Title: "Distilling an ensemble". Left bullets:

- "Ensembling model outputs creates more accurate estimates."
- "Why? Each model is imperfect in different random ways. Errors cancel out, truth is shared."
- "Distill ensemble into a single model for fast inference."

Right: three trapezoids stacked at the left, each with a black bar above it, a grey padlock, and $x$ underneath: $f^{T_3}$ (top), $f^{T_2}$ (middle), $f^{T_1}$ (bottom). The words "Teachers" sit beneath the stack. At the right, a single trapezoid with $f^S$, a black bar above it and $x$ underneath, with the word "Student" below. A dotted curved double-headed arc runs from just right of the $T_3$ bar's end around to just right of the $T_1$ bar's end, bowing to the right; a straight dotted double-headed horizontal arrow runs from the $T_2$ bar to the student's bar, crossing the arc. Printed slide number "35".

## Slide 36 — Contrastive Representation Distillation

Title: "Contrastive Representation Distillation", followed by "[Tian, Krishnan, Isola, 2020]".

Left: a large circle with a dotted outline and a grey gradient fill (lighter at the upper left, darker at the lower right), standing for an embedding space. Three black dots inside it with curved arrows from their labels: $f^T(x_i)$ and $f^S(x_i)$, written outside the circle at its upper left, point to two dots close together near its upper left ($f^T(x_i)$ to the upper-right one, $f^S(x_i)$ to the lower-left one); $f^S(x_j)$, written outside the circle at its right, points to a dot at the lower right. A dotted line runs down from the circle to the middle of a dotted double-headed arrow between the teacher's and the student's outputs. Below, a large trapezoid with $f^T$, a grey padlock, $x$ underneath and "Teacher" below; and a smaller trapezoid with $f^S$, $x$ underneath and "Student" below. Above each trapezoid, two stacked bars: a white (unfilled) bar over a black bar.

Right bullets:

- "Contrastive learning supervises embeddings to be invariant to some viewing transformation."
- "The “viewing transformation” can be “teacher’s view” (big net) vs “students view” (small net)." (the printed text has “students view” with no apostrophe)
- "Then the student will output embeddings that are the same as the teacher’s."

Printed slide number "36".

## Slide 37 — Contrastive Representation Distillation

![Slide 37 — Contrastive Representation Distillation](../images/18-transfer-learning-models/slide-37.jpg)

Title: "Contrastive Representation Distillation", followed by "[Tian, Krishnan, Isola, 2020]". The left half repeats slide 36's diagram (embedding-space circle with $f^T(x_i)$, $f^S(x_i)$, $f^S(x_j)$; teacher with $f^T$ and padlock; student with $f^S$; "Teacher", "Student").

New on the right: two square heat maps side by side, titled beneath "KD" (left) and "CRD" (right), each with a vertical colour bar at its right running from $0.30$ (dark purple) through $0.15$, $0.00$ (white) and $-0.15$ to $-0.30$ (dark orange-brown). Both maps have row and column indices 0 to 96 in steps of 4 (the printed ticks are 0, 4, 8, ..., 96).

- KD map: a dense speckle of orange and purple cells of many intensities, from pale to dark, spread evenly across the whole square; orange cells are somewhat more common than purple ones throughout.
- CRD map: almost uniformly white to very pale orange, with only scattered pale lavender cells and no saturated cells.

Caption: "Correlation coefficients between class output logits — Student’s minus Teacher’s (CIFAR100: plots are 100 classes by 100 classes)". Printed slide number "37".

## Slide 38 — Compression vs. Distillation

![Slide 38 — Compression vs. Distillation](../images/18-transfer-learning-models/slide-38.png)

Title: "Compression vs. Distillation". Bullets:

- "Both give us smaller models with similar accuracy to large ones"
- "Distillation uses SGD to train a smaller model to match characteristics of a larger one"
- "Compression uses ***pruning*** or ***quantization*** to reduce the computational cost of a large model" (the words "pruning" and "quantization" are bold italic)

Lower left, an algorithm from a paper, set between horizontal rules ("Algorithm 1", "for", "do" and "end for" in bold):

Algorithm 1: "Prune $k \leq d_{\text{col}}$ weights from row $\mathbf{w}$ with inverse Hessian $\mathbf{H}^{-1} = (2\mathbf{X}\mathbf{X}^{\top})^{-1}$ according to OBS in $O(k \cdot d_{\text{col}}^{2})$ time."

$$M = \lbrace 1, \ldots, d_{\text{col}} \rbrace$$

for $i = 1, \ldots, k$ do

$$p \leftarrow \operatorname{argmin}_ {p \in M} \frac{1}{[\mathbf{H}^{-1}]_ {pp}} \cdot w_p^2$$

$$\mathbf{w} \leftarrow \mathbf{w} - \mathbf{H}^{-1}_ {:,p} \frac{1}{[\mathbf{H}^{-1}]_ {pp}} \cdot w_p$$

$$\mathbf{H}^{-1} \leftarrow \mathbf{H}^{-1} - \frac{1}{[\mathbf{H}^{-1}]_ {pp}} \mathbf{H}^{-1}_ {:,p} \mathbf{H}^{-1}_ {p,:}$$

$$M \leftarrow M - \lbrace p \rbrace$$

end for

Lower right, a figure of blue-shaded small grids left to right: "Dense Weight" (a 3 by 3 grid in shades of blue); three arrows from it to three rows of three cells labelled "Weight Rows"; next to each row a 3 by 3 grid labelled (column heading) "Pruning Traces" with a few blue cells; then three more rows of three cells under "Loss Changes"; a large brace; then a 3 by 3 grid "Global Mask" with three black cells; an arrow to a 3 by 3 grid "Pruned Weight" with a few blue cells. A curved arrow over the top from the Pruning Traces to the Pruned Weight is labelled "Less compute: load stored trace elements"; a curved arrow along the bottom from the Dense Weight to the Pruned Weight is labelled "Less memory: resolve not pruned weights". Caption: "Figure 1: Efficient global OBS using the row-wise results."

*Courtesy credit (not an OCW notice):* "Courtesy of Frantar, et al. Used under CC BY." (small print at the lower right of the figure). Citation across the bottom: "Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning, Frantar et al., 2023" (the name "Frantar" is underlined). Printed slide number "38", printed overlapping the citation line.

## Slide 39 — We almost never need a truly generalist model in practice. Specialization can enable much higher rates of compression with the same accuracy.

![Slide 39 — specialization enables higher rates of compression](../images/18-transfer-learning-models/slide-39.jpg)

Title (two lines): "We almost never need a truly generalist model in practice. Specialization can enable much higher rates of compression with the same accuracy."

A hand-drawn-style diagram in three rounded boxes, read left to right. At the left, a rounded box holds a fully connected neural network drawn as three columns of circles: a left column of four green circles (a vertical dotted line between the second and third marks omitted nodes), a middle column of three green circles, and a right column of four pale-orange circles, with black lines joining every circle to every circle in the next column. The text "IMAGENET" (with a small coloured logo mark in place of the "A") is printed across the bottom of this box. Under it the label "Large Generalist Model". A thick black arrow runs from this box to a box in the upper centre, with a double line at each end, labelled "TACO". A pair of short thick vertical lines joins "TACO" down to a drum-shaped container (a cylinder) labelled underneath "Calibration Set". The cylinder holds four small photographs: two photographs of sharks in the upper row (a grey shark in blue water, and a pinkish shark seen from below, its head a wide, flat crosspiece, among small dark fish in blue water) and two in the lower row (a pale, flat, elongated fish lying on sand, with blurred blue-grey rocks behind, and a pale brown fish's head with a large eye and an open, toothed mouth, against a dark grey background). A second thick black arrow runs from "TACO" to a rounded box at the right holding a smaller, sparser network (a left column of four green circles with a vertical dotted line, a middle column of three green circles, a right column of two pale-orange circles, with fewer connecting lines than the left network). The text "Sharks" is printed at the bottom of this box; underneath, the label "Sparse Specialist Model" (two lines). The figure shows a large ImageNet-trained model plus a small calibration set of shark images being turned into a small model specialised to sharks.

Credit (small print, lower left): "Courtesy of Kuznedelev, et al. Used under CC BY." (a credit, not an OCW notice).

Citation line at the foot: "Vision Models Can Be Efficiently Specialized via Few-Shot Task-Aware Compression, Kuznedelev et al., 2023" ("Kuznedelev" underlined). The printed slide number "39" overlaps the citation.

## Slide 40 — Foundation models

![Slide 40 — Foundation models](../images/18-transfer-learning-models/slide-40.jpg)

Title: "Foundation models". At the top right: "[Bommasani et al. 2021]" and the URL "https://arxiv.org/pdf/2108.07258.pdf".

At the left, a quotation in large text: "“If I have seen further it is by standing on the shoulders of Giants”" followed by "— Newton".

At the right, a framed painting of a landscape: tall dark-green trees fill the left and right, a pale blue sky with grey-blue clouds in the upper centre, a distant rocky peak in the middle with a strip of water on the horizon at the centre right, and in the foreground a very large man in a loincloth on the right, striding to the left, seen from behind, with one arm outstretched ahead of him, a bow in his trailing hand and a quiver hanging at his hip from a shoulder strap; a smaller figure in yellow stands on his shoulders and leans forward, pointing, and a further figure stands on a cloud above. In the lower middle, two small figures show from the waist up behind a bank, and a third, in a pale blue robe, stands nearer the large man looking up at him; tree stumps lie in the lower foreground.

Caption below the painting: "[Blind Orion Searching for the Rising Sun by Nicolas Poussin, 1658]" (the title in italics). Small print beneath it, overlapping the slide number: "Image is in the public domain." (a credit, not an OCW notice). Printed slide number "40".

## Slide 41 — (no title; chart "AlexNet to AlphaGo Zero: A 300,000x Increase in Compute")

![Slide 41 — (no title; chart "AlexNet to AlphaGo Zero: A 300,000x Increase in Compute")](../images/18-transfer-learning-models/slide-41.jpg)

A screenshot of a web chart, with no slide-level title. Chart heading, bold: "AlexNet to AlphaGo Zero: A 300,000x Increase in Compute". Below it two toggle buttons, "Log Scale" (selected, grey fill) and "Linear Scale" (unselected).

Axes: the y-axis is labelled "Petaflop/s-days" at its top, log scale, with gridlines labelled 1e+4 (top), 1e+3, 1e+2, 1e+1, 1e+0, 1e−1, 1e−2, 1e−3, 1e−4, 1e−5 (bottom). The x-axis has years 2012, 2013, 2014, 2015, 2016, 2017, 2018 at the bottom. There is one series: 15 blue dots, each with a name label, and a single grey dashed straight trend line rising from the lower left (about 2e−5 at the left edge, about 1e−4 at 2012) to the upper right (about 8e+2 at 2018.5), labelled "3.4-month doubling" (grey text, near 2013.7, below the line at about 1e−3). Dots, approximately (year, value):

- AlexNet: about 2012.4, about 6e−3
- Dropout: about 2012.5, about 2e−3
- Visualizing and Understanding Conv Nets (three-line label): about 2013.9, about 6e−3
- DQN: about 2014, about 3e−5 (far below the line)
- VGG: about 2014.7, about 1e−1
- Seq2Seq: about 2014.7, about 1e−1 (beside VGG, labelled to its right)
- GoogleNet: about 2014.7, about 2e−2
- DeepSpeech2: about 2015.9, about 2e−1
- ResNets: about 2015.9, about 1.2e−1
- Xception: about 2016.8, about 5e+0 (label below its dot)
- Neural Machine Translation: about 2016.8, about 8e+1
- Neural Architecture Search: about 2016.9, about 3e+1
- TI7 Dota 1v1: about 2017.6, about 7e+0 (label below its dot)
- AlphaGoZero: about 2017.8, about 2e+3 (a little left of AlphaZero)
- AlphaZero: about 2017.9, about 4e+2

Footnote under the chart (grey): "The total amount of compute, in petaflop/s-days,[2] used to train selected results that are relatively well known, used a lot of compute for their time, and gave enough information to estimate the compute used." At the lower right: "https://openai.com/blog/ai-and-compute/". Printed slide number "41".

## Slide 42 — (no title; chart "D. Computing Power Usage Over Time by:")

![Slide 42 — (no title; chart "D. Computing Power Usage Over Time by:")](../images/18-transfer-learning-models/slide-42.jpg)

Heading, no slide-level title: "D. Computing Power Usage Over Time by:". Legend at the upper left: a thick black line labelled "Regression". Three side-by-side scatter panels with a shared y-axis, each headed in blue text: "Academia" (pale blue), "Academia-Industry Collaboration" (medium blue), "Industry" (darker blue).

Shared y-axis label (rotated, bold capitals): "NUMBER OF PARAMETERS", log scale, gridlines labelled 10^10.5 (printed "10" with superscript "10.5"), 10^10.0, 10^9.5, 10^9.0, 10^8.5, 10^8.0, 10^7.5, 10^7.0, 10^6.5, 10^6.0 (the printed form is "10" with the exponent as a small superscript, i.e. 10^{10.5} down to 10^{6.0}). Each panel has its own x-axis of years (rotated labels): Academia 2012 to 2021; Academia-Industry Collaboration 2014 to 2021; Industry 2015 to 2021. Each panel has dots (one colour per panel, matching the heading), a pale grey shaded wedge enclosing the dots (widening to the right), and one thick black regression line. Three series, one per panel:

- Academia (pale blue dots): sparse single dots at 2012 (about 10^7.8), 2016 (about 10^7.9) and 2018 (about 10^6.6); columns of dots at 2019 (about 10^6.6 to 10^7.8), 2020 (about 10^6.3 to 10^8.0) and 2021 (about 10^6.7 to 10^8.0). The regression line is nearly flat, sloping slightly down from about 10^7.45 in 2012 to about 10^7.3 in 2021. Shaded wedge: roughly 10^8.0 at the top, extending down to about 10^6.3 at 2020.
- Academia-Industry Collaboration (medium blue dots): one dot at 2014 (about 10^8.1), dots at 2016 and 2017 (about 10^7.9), a dot at 2018 (about 10^6.9), columns at 2019 (about 10^6.8 to 10^8.0), 2020 (about 10^6.7 to 10^8.7) and a dense column at 2021 (about 10^6.5 to 10^8.8). The regression line is essentially flat at about 10^7.5.
- Industry (dark blue dots): dots from 2015 (about 10^7.1 to 10^7.6), 2016 (about 10^7.3 to 10^7.9), 2017 (about 10^8.0), 2018 (about 10^6.7 to 10^9.0), 2019 (about 10^6.5 to 10^9.0), 2020 (about 10^6.8 to 10^8.8) and a dense 2021 column (about 10^6.9 to 10^10.2, with the top dot near 10^10.2). The regression line rises clearly from about 10^6.9 in 2015 to about 10^8.0 in 2021. Shaded wedge widens from narrow at 2015 to the largest spread at 2021.

Credit at lower right: "Ahmed and Thompson (2022)". Printed slide number "42".

## Slide 43 — Climate cost

![Slide 43 — Climate cost](../images/18-transfer-learning-models/slide-43.jpg)

Title: "Climate cost". 

Figure at the top: a horizontal chevron arrow of six stages, left to right, with a bracket above the middle three labelled "Focus of our study". Stages: "Raw material extraction" (light grey), "Materials manufacturing" (mid grey), "Equipment manufacturing" (light green), "Model training" (mid green), "Model deployment" (darker green), "Disposal/ end-of-life" (dark grey/black, ending in an arrowhead). The bracket spans the three green stages. Credit under the right end: "Courtesy of Luccioni et al. Used under CC BY-NC-SA." (a credit, not an OCW notice). Caption: "Figure 1: While the LCA approach encompasses all the stages of the product life cycle (from raw material extraction to disposal), we focus on those in green, which range from equipment manufacturing to model deployment."

Table (from the paper):

| Model name | Number of parameters | Datacenter PUE | Carbon intensity of grid used | Power consumption | CO2eq emissions | CO2eq emissions × PUE |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-3 | 175B | 1.1 | 429 gCO2eq/kWh | 1,287 MWh | *502 tonnes* | 552 tonnes |
| Gopher | 280B | 1.08 | 330 gCO2eq/kWh | *1,066 MWh* | *352 tonnes* | 380 tonnes |
| OPT | 175B | *1.09* (superscript 2) | *231gCO2eq/kWh* | *324 MWh* | 70 tonnes | *76.3 tonnes* (superscript 3) |
| BLOOM | 176B | 1.2 | 57 gCO2eq/kWh | 433 MWh | 25 tonnes | 30 tonnes |

(OPT's "231gCO2eq/kWh" has no space after the number, as printed. Italics as printed: GPT-3's "502 tonnes"; Gopher's "1,066 MWh", "352 tonnes"; OPT's "1.09", "231gCO2eq/kWh", "324 MWh", "76.3 tonnes". The "2" in "CO2eq" is a subscript.) Caption: "Table 4: Comparison of carbon emissions between BLOOM and similar LLMs. Numbers in *italics* have been inferred based on data provided in the papers describing the models."

Two citation lines at the foot: "Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model, Luccioni et al., 2023" and "Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models, Li et al., 2023". Printed slide number "43" (overlapping the second citation).

## Slide 44 — What does it take to train one of these models?

Title: "What does it take to train one of these models?". Three bullets:

- "Dalle-2: **650m training examples** (semi-curated image-text pairs)"
- "GPT-3: ~**$4.6m to train *once*** (https://lambdalabs.com/blog/demystifying-gpt-3/)" (the URL underlined, in smaller type)
- "PaLM: **67 authors**"

Then two unbulleted lines: "And these numbers are increasing." and, as a separate paragraph, "But these numbers are small if we think of the result as foundational infrastructure; compare to the cost of building the interstate highway system, or the cost of developing the Mac OS X operating system." Printed slide number "44".

## Slide 45 — (no title; chart "Performance on 58 Tasks" with red annotations)

![Slide 45 — (no title; chart "Performance on 58 Tasks" with red annotations)](../images/18-transfer-learning-models/slide-45.jpg)

A line chart titled "Performance on 58 Tasks", with red hand-drawn boxes and red text added on the slide. 

Axes: x-axis "Model Parameters (Non-Embedding)", log scale, ticks 10^8, 10^9, 10^10, 10^11 (the chart's data run from about 4×10^7 to about 5×10^11). y-axis "Normalized Preferred Metric (Avg.)", ticks 0, 20, 40, 60, 80 (the axis runs from about −2 to about 95). Two horizontal reference lines: "Human (Best)" black dashed at about 91, and "Human (Avg.)" grey dashed at about 48. Legend (upper left, in the plot) with nine entries: "Gopher 5-shot" (green line with dots, dash-dot), "Chinchilla 5-shot" (brown diamond marker), "GPT-3 0-shot" (light orange dashed, dots), "GPT-3 1-shot" (darker orange dashed, dots), "PaLM 0-shot" (light blue, left-pointing triangles), "PaLM 1-shot" (mid blue), "PaLM 5-shot" (dark blue), "Human (Avg.)", "Human (Best)". Seven data series plus the two reference lines:

- Gopher 5-shot (green, dash-dot, round markers): from about 2 at 4×10^7 parameters, rising slowly to about 11 at 7×10^9, then to about 30 at 2.8×10^11.
- Chinchilla 5-shot (brown): a single diamond at about 45 at about 7×10^10.
- GPT-3 0-shot (light orange, dashed): about 3 at 10^8 rising slowly to about 6 at 10^10, then about 8 at about 2×10^11.
- GPT-3 1-shot (orange, dashed): about 3 at 10^8, about 10 at 10^10, about 15 at about 2×10^11.
- PaLM 0-shot (pale blue): about 7 at 8×10^9, about 15 at 6×10^10, about 15 at 5×10^11 (flat).
- PaLM 1-shot (mid blue): about 8 at 8×10^9, about 25 at 6×10^10, about 48 at 5×10^11.
- PaLM 5-shot (dark blue): about 12 at 8×10^9, about 33 at 6×10^10, about 54 at 5×10^11.

Red annotations: a red box around the lower-left region (every point below about 1.7×10^10, the first point of each PaLM series included, up to a value of about 23) with the red text "Not working at all" at the lower left outside the chart; a red box around the top-right ends of the PaLM 1-shot and 5-shot lines (about 5×10^11, values about 48 and 54), spanning about 36 to 65, with the Human (Avg.) line running through it with the red text (four lines) "Working as well as an average human" at the right. Printed slide number "45".

## Slide 46 — Learn foundation models / Use/adapt foundations to solve new problems

![Slide 46 — Learn foundation models / Use/adapt foundations to solve new problems](../images/18-transfer-learning-models/slide-46.png)

Two diagrams side by side, each with a heading.

Left, heading "Learn foundation models": three staggered rows of yellow-orange boxes with white labels. Top row (4 boxes): "BERT", "SimCLR", "CLIP", "GPT". Middle row (4 boxes): "AlexNet", "SimCLR", "DALL-E", "WaveNet". Bottom row (3 boxes): "BigGAN", "StyleGAN", "VQGAN". ("SimCLR" appears twice, as printed.) Below the boxes, an upward black arrow points from a grey box labelled "Learner" up toward the boxes; below "Learner" another upward arrow comes from the text "Tons of data".

Right, heading (two lines) "Use/adapt foundations to solve new problems": a pink box "App" at the top; an upward arrow from a grey box labelled "Adaptor" to "App"; two upward arrows enter "Adaptor" from below, from the text "Little or no data". Printed slide number "46".

## Slide 47 — New paradigm: “Deep Learning” with No Data

Title: "New paradigm: “Deep Learning” with *No* Data" (the word "No" in italics). Centre of the slide, large: "“Just ask”". Below, smaller: "(of course, this isn’t really machine *learning* anymore)" (the word "learning" in italics). At the lower right the URL "https://evjang.com/2021/10/23/generalization.html". Printed slide number "47" (overlapping the URL).

## Slide 48 — GPT-3

Title: "GPT-3", followed by "[Brown et al., 2020]" and the URL "https://arxiv.org/pdf/2005.14165.pdf".

At the left, large text: "1. **Learner**: Generative Pretraining" (three lines; "Learner" bold). At the centre-right a left-to-right diagram: monospace text in quotation marks "“Colorless green ideas sleep ____”" (two lines, the blank an underscore line) with an arrow to a grey box "Language model", then an arrow to monospace "“furiously”". Under the grey box: "(Predict next characters)". Printed slide number "48".

## Slide 49 — GPT-3

Title: "GPT-3", followed by "[Brown et al., 2020]" and the URL "https://arxiv.org/pdf/2005.14165.pdf". This repeats slide 48's layout with the adaptor step instead of the learner.

At the left, large text: "1. **Adaptor**: Prompt engineering" (three lines; "Adaptor" bold; the numeral is printed "1.", as on slide 48, although this is the second step: a printed slip, kept as printed). At the centre-right, a left-to-right diagram: monospace prompt text "[Review] + “The sentiment in this review is ____”" (three lines, ending in a blank underscore line before the closing quote), an arrow into a grey box "Language model", an arrow out to monospace "“positive”". Under the box: "(Predict next characters)". Printed slide number "49".

## Slide 50 — New capabilities by just asking

Title: "**New capabilities** by just asking" ("New capabilities" bold). Below it, at the upper left, one line of bold grey sans-serif text (a screenshot of a prompt): "Top ten dishes at Mulan, Cambridge, MA:". The rest of the slide is empty (a build step: slide 51 shows another prompt with an output). Printed slide number "50".

## Slide 51 — New capabilities by just asking

Title: "**New capabilities** by just asking". Below it, in bold grey sans-serif text, a heading and a numbered list:

"List of prime numbers:"

1. 2
2. 3
3. 5
4. 7
5. 11
6. 13

Printed slide number "51". (A different example from slide 50; slide 50's text is not shown here.)

## Slide 52 — Few-shot learning ability

*OCW notice: © Brown, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (upper right of the slide; it covers the diagram of nested loops and in-context sequences).*

Title: "Few-shot learning ability". A diagram with grey italic labels and blue, purple and grey elements.

At the top, a large grey curly brace opening downward, its point at the top under the label "outer loop" (grey italic), its arms reaching down at both ends, spanning the full width. Below it a long purple arrow pointing right, with the purple bold text above its left end: "Learning via SGD during unsupervised pre-training".

Beneath the arrow, three light-blue panels side by side, each a code-like list of six numbered lines (line numbers 1 to 6 in pale grey at the left of each panel; text in blue monospace), each with a tall blue downward arrow to its right labelled in rotated blue bold text "In-context learning", and a grey upward arrow with a grey italic label beneath the panel. At the left of the first panel is a grey curly brace opening right, toward the panel, its point facing left toward the label "inner loop" (grey italic).

- Panel 1 (label below: "sequence #1"): "5 + 8 = 13", "7 + 2 = 9", "1 + 0 = 1", "3 + 4 = 7", "5 + 9 = 14", "9 + 8 = 17".
- Panel 2 (label below: "sequence #2"): "gaot => goat", "sakne => snake", "brid => bird", "fsih => fish", "dcuk => duck", "cmihp => chimp".
- Panel 3 (label below: "sequence #3"): "thanks => merci", "hello => bonjour", "mint => menthe", "wall => mur", "otter => loutre", "bread => pain".

Printed slide number "52".

## Slide 53 — Few-shot learning ability: extrapolation

*OCW notice: Icons © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers the emoji-style icons).*

Title: "Few-shot learning ability: extrapolation". Two numbered lists of small grey elephant icons and red apple icons.

Left list, eleven lines numbered 1. to 11. (the first three numerals bold, the rest regular), where line $n$ holds $n$ elephant icons: 1 elephant on line 1, 2 on line 2, and so on up to 11 elephants on line 11, forming a triangle.

Right list, seven lines (numerals 1.–4. bold, 5.–7. regular), alternating elephant and apple icons starting with an elephant:

- 1. one elephant
- 2. elephant, apple
- 3. elephant, apple, elephant (with a wider gap before the last elephant)
- 4. elephant, apple, elephant, apple
- 5. elephant, apple, elephant, apple, elephant
- 6. elephant, apple, elephant, apple, elephant, apple
- 7. elephant, apple, elephant, apple, elephant, apple, elephant

Printed slide number "53".

## Slide 54 — GPT-3

Title: "GPT-3", followed by "[Brown et al., 2020]" and "https://arxiv.org/pdf/2005.14165.pdf". In the centre, a screenshot of a few-shot translation prompt in dark grey sans-serif text, three English/Spanish pairs separated by blank lines:

```
English: hello
Spanish: hola

English: this tastes really good
Spanish: esto sabe muy bien

English: where is the library?
Spanish: dónde está la biblioteca?
```

In the last pair, the text "dónde está la biblioteca?" (after "Spanish: ") is highlighted with a pale green background (marking it, presumably, as the model's output; the slide does not say). Printed slide number "54".

## Slide 55 — Prompting

A section slide: the single word "Prompting" in large text at the centre of an otherwise empty white slide. Printed slide number "55".

## Slide 56 — Hand-crafting good prompts is tricky…

Upper right: "Slide credit: Hyojin Bahng" (a slide credit). Title: "Hand-crafting good prompts is tricky…" (with the ellipsis character). A table reproduced from a paper (columns "Template", "Label words", "Accuracy"); the subscript in $S_1$ and $S_2$ is printed as a subscript, the angle brackets around them are printed "<" and ">", and "[MASK]" is in monospace. In each section the first row's accuracy is bold.

| Template | Label words | Accuracy |
| --- | --- | --- |
| SST-2 (positive/negative) | | mean (std) |
| `<S1>` It was `[MASK]` . | great/terrible | **92.7 (0.9)** |
| `<S1>` It was `[MASK]` . | good/bad | 92.5 (1.0) |
| `<S1>` It was `[MASK]` . | cat/dog | 91.5 (1.4) |
| `<S1>` It was `[MASK]` . | dog/cat | 86.2 (5.4) |
| `<S1>` It was `[MASK]` . | terrible/great | 83.2 (6.9) |
| Fine-tuning | - | 81.4 (3.8) |
| SNLI (entailment/neutral/contradiction) | | mean (std) |
| `<S1>` ? `[MASK]` , `<S2>` | Yes/Maybe/No | **77.2 (3.7)** |
| `<S1>` . `[MASK]` , `<S2>` | Yes/Maybe/No | 76.2 (3.3) |
| `<S1>` ? `[MASK]` `<S2>` | Yes/Maybe/No | 74.9 (3.0) |
| `<S1>` `<S2>` `[MASK]` | Yes/Maybe/No | 65.8 (2.4) |
| `<S2>` ? `[MASK]` , `<S1>` | Yes/Maybe/No | 62.9 (4.1) |
| `<S1>` ? `[MASK]` , `<S2>` | Maybe/No/Yes | 60.6 (4.8) |
| Fine-tuning | - | 48.4 (4.8) |

The table has horizontal rules above and below the header row, under each section's heading row, between the two sections, and at the bottom. Printed slide number "56".

## Slide 57 — Soft Prompts (Prompt Tuning)

Upper right: "Slide credit: Hyojin Bahng" (a slide credit). Title: "Soft Prompts (Prompt Tuning)". One bullet: "Optimizes a small continuous task-specific vector while having language model parameters frozen".

*OCW notice: Left © Li and Liang. Right © IEEE. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right; it covers both pasted figures, the left one and the right one).*

Left figure (two stacked diagrams, headed "Fine-tuning" at the top right of the figure and "Prefix-tuning" at its middle right). Fine-tuning: three overlapping pink rectangles, offset diagonally like a stack, labelled "Transformer (Translation)" (back), "Transformer (Summarization)" (middle) and "Transformer (Table-to-text)" (front); the front one shows a row of nine white vertical slots. Below it, a line of coloured text "name Starbucks type coffee shop [SEP] Starbucks serves coffee" (the first five words in blue, "[SEP] Starbucks serves coffee" in green), with the small labels "Input (table-to-text)" (blue "Input") under the left half and "Output (table-to-text)" (green "Output") under the right half. Prefix-tuning: a grey rectangle labelled "Transformer (Pretrained)" with a row of nine white vertical slots; at its left, three overlapping rounded boxes labelled "Prefix (Translation)", "Prefix (Summarization)" and "Prefix (Table-to-text)", the front one holding two pink vertical slots. Below it the same coloured line "name Starbucks type coffee shop [SEP] Starbucks serves coffee" with "Input (table-to-text)" and "Output (table-to-text)".

Right figure: three columns under two curly braces, "Strong Task Performance" (above the left two columns) and "Efficient Multitask Serving" (above the right two columns, the braces overlapping over the middle column). Column 1, "Model Tuning (a.k.a. “Fine-Tuning”)": an orange box "Pre-trained Model" with "Tunable" flanked by flame icons; an arrow up from a row of seven empty light-grey cells labelled "Input Text". Column 2, in a pale-yellow rounded frame, "Prompt Tuning (Ours)": a blue box "Pre-trained Model" with "Frozen" flanked by snowflake icons; two arrows up into it, one from a short row of three cells with flame icons labelled "Tunable Soft Prompt" and one from a row of six empty cells labelled "Input Text". Column 3, "Prompt Design (e.g. GPT-3)": a blue box "Pre-trained Model", "Frozen" with snowflakes; arrows up from three cells with snowflake icons labelled "Engineered Prompt" and from six empty cells labelled "Input Text".

Citations at the foot, right: "Prefix-tuning: Optimizing continuous prompts for generation. ACL 2021." and "The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP 2021." Printed slide number "57".

## Slide 58 — Chain-of-thought prompting

Title: "Chain-of-thought prompting". At the upper right: "[Wei, Wang, Schuurmans et al. 2022]" and under it, in small print, "Courtesy of Wei, et al. Used under CC BY." (a credit, not an OCW notice).

Two columns, each with a heading, a rounded box tagged "Input" (blue tab) and a rounded box tagged "Model Output" (green tab).

Left column, heading "Standard Prompting". Input box text:

"Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"

Model Output box: "A: The answer is 27." followed by a red cross mark.

Right column, heading "Chain of Thought Prompting". Input box text:

"Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"

(In the first answer, the text from "Roger started with 5 balls." to "5 + 6 = 11." is highlighted blue; "The answer is 11." is not.) Model Output box: "A: The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9." followed by a green check mark; the text from "The cafeteria had 23 apples originally." through "3 + 6 = 9." is highlighted green; "The answer is 9." is not. Printed slide number "58".

## Slide 59 — Chain-of-thought prompting

![Slide 59 — Chain-of-thought prompting](../images/18-transfer-learning-models/slide-59.jpg)

Title: "Chain-of-thought prompting". At the upper right: "[Wei, Wang, Schuurmans et al. 2022]" and, in small print, "Courtesy of Wei, et al. Used under CC BY." (a credit, not an OCW notice).

A bar chart. Legend (above the plot, four entries stacked in a column, each with a colour swatch): "Finetuned GPT-3 175B" (cream with diagonal hatching), "Prior best" (light blue with diagonal hatching), "PaLM 540B: standard prompting" (solid yellow), "PaLM 540B: chain-of-thought prompting" (solid orange). The y-axis is labelled "Solve rate (%)", ticks 0, 20, 40, 60, 80, 100. The x-axis has one group label, "Math Word Problems (GSM8K)". Four bars side by side, one series each, with the value printed above each bar:

- Finetuned GPT-3 175B (cream, hatched): 33
- Prior best (light blue, hatched): 55
- PaLM 540B: standard prompting (yellow): 18
- PaLM 540B: chain-of-thought prompting (orange): 57

Printed slide number "59".

## Slide 60 — CLIP

Title: "CLIP", followed by "[Radford et al., 2021]" and "https://arxiv.org/pdf/2103.00020.pdf". At the left, large text: "1. **Learner**: Contrastive Pretraining" (three lines; "Learner" bold). At the lower left: "[https://openai.com/blog/clip/]" (the URL underlined).

*OCW notice: © Radford, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the bottom right of the slide; it covers the contrastive-pretraining diagram).*

The diagram, right of centre. Top: a stack of four offset white cards, the front one with monospace text "Pepper the aussie pup", with an arrow to a purple trapezoid labelled "Text Encoder" (the wide side on the left). From the encoder's right edge a long horizontal line runs right and four arrows drop down from it into a horizontal row of five purple cells labelled "T1", "T2", "T3", "…", "TN" (subscripts as printed; an "…" also sits above the gap between the third and last arrows). Bottom left: a stack of four offset photographs, the front one a puppy with black, white and tan fur on grass, tongue out, with an arrow to a green trapezoid labelled "Image Encoder" (wide side on the left). From it, a bracket-shaped line fans out to a vertical column of five green cells labelled "I1", "I2", "I3", "⋮", "IN", each with an arrow into it (except the "⋮" cell). To the right of the column is a 5 × 5 grid of cells: row $i$, column $j$ reads "Ii·Tj" (I1·T1, I1·T2, I1·T3, "…", I1·TN; I2·T1 ... I2·TN; I3·T1 ... I3·TN; a row of "⋮" cells; IN·T1, IN·T2, IN·T3, "…", IN·TN), with the fourth column and fourth row of the grid holding "…" and "⋮" (the "⋮" row reads "⋮", "⋮", "⋮", "⋱", "⋮"). The diagonal cells (I1·T1, I2·T2, I3·T3, the "⋱" cell, IN·TN) are shaded light blue; the rest are white. The figure shows each image embedding being compared with each text embedding, with the matching pairs on the diagonal highlighted. Printed slide number "60".

## Slide 61 — CLIP

Title: "CLIP", followed by "[Radford et al., 2021]" and "https://arxiv.org/pdf/2103.00020.pdf". At the left, large text: "2. **Adaptor**: Linear classifer on top of image encodings" (printed "classifer", missing an "i"; four lines; "Adaptor" bold). At the lower left: "[https://openai.com/blog/clip/]" (underlined).

*OCW notice: © Radford, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the bottom, under the chart; it covers the chart; the slide number overlaps it).*

A line chart. y-axis: "Average Score (%)", ticks every 5 from 30 to 75. x-axis: "# of labeled training examples per class", ticks 0, 1, 2, 4, 8, 16 on a linear scale (so 0, 1 and 2 crowd together at the left and 16 sits far to the right of 8), with vertical grey gridlines at 0, 1, 2, 4 and 8 (16 is the right edge of the plot frame). Behind the four coloured curves are many faint grey curves (other models), most between the green and blue curves, and two lower faint curves reaching about 46 and 43 at 16. A purple star at x = 0, y about 62.3, labelled "Zero-Shot CLIP" (purple), and a thin horizontal grey line at about 62.3 from the star across the plot. Four coloured series, each with a dot at x = 1, 2, 4, 8, 16, with a text label near the right end:

- "Linear Probe CLIP" (purple): about 44.5, 54, 62.3, 66.3, 71.5.
- "BiT-M (ImageNet-21K)" (blue): about 39.7, 46.7, 53.8, 58.5, 62.8.
- "SimCLRv2" (orange): about 32, 39.7, 48.4, 54.3, 59.5.
- "ResNet50" (green): about 31.8, 38.9, 44.6, 49.6, 54.2.

The star (zero labelled examples) is level with the purple curve at 4 labelled examples per class. Printed slide number "61".

## Slide 62 — CLIP

Title: "CLIP", followed by "[Radford et al., 2021]" and "https://arxiv.org/pdf/2103.00020.pdf". At the left, large text: "2. **Adaptor**: Just ask" (two lines; "Adaptor" bold). At the lower left: "[https://openai.com/blog/clip/]" (underlined).

*OCW notice: © Radford, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the bottom centre; it covers the diagram; the slide number overlaps it).*

The diagram has two labelled parts. Part "(2) Create dataset classifier from label text": a vertical stack of four boxes with monospace pink text "plane", "car", "dog", then a gap with "⋮" marks, then "bird"; a bracket line joins the right side of the stack and sends an arrow into a black-bordered box "A photo of a {object}." (monospace; "{object}" in pink), then an arrow to a purple trapezoid "Text Encoder". From it a horizontal line runs right with arrows down to a row of five purple cells "T1", "T2", "T3", "…", "TN" (with "…" above the gap). Part "(3) Use for zero-shot prediction": a photograph of a dark dog on grass with leaves, an arrow to a green trapezoid "Image Encoder", an arrow to a green cell "I1", and to its right a row of five white cells "I1·T1", "I1·T2", "I1·T3", "…", "I1·TN", directly under the T-row; the third cell, "I1·T3", is shaded light blue with a blue border, and an arrow from it points down to a box "A photo of a dog." (monospace; "dog" in pink). Printed slide number "62".

## Slide 63 — New capabilities by just asking

Title: "**New capabilities** by just asking" ("New capabilities" bold).

*OCW notice: © Radford, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the bottom right, starting over the lower right corner of the Image Encoder trapezoid; it covers the diagram).*

A diagram of three rows. Row 1: monospace text "“A **sketch** of a banana”" (the word "sketch" in bold) with an arrow into a lavender trapezoid "Text Encoder", which outputs a small empty rectangle labelled "e1" (bold, subscript 1). Row 2: "“A **photo** of a banana”" ("photo" bold), an arrow into a second lavender "Text Encoder", outputting a rectangle "e2". Row 3: a pencil-style grey drawing of a bunch of bananas, an arrow into a pale green trapezoid "Image Encoder", outputting a rectangle "e3". At the right, two square boxes: "e1 · e3" with a thick green border (the first), and "e2 · e3" with a thin black border. A line from the e1 rectangle runs right then down with an arrow into the "e1 · e3" box; a line from the e2 rectangle runs right (crossing the first line) then down with an arrow into the "e2 · e3" box. The e3 rectangle has no line to the boxes. The picture suggests the sketch prompt matches the sketch image more strongly (the green border marks "e1 · e3"). Printed slide number "63".

## Slide 64 — Prompting in other modalities

A section slide: "Prompting in other modalities" in large text at the centre of an empty white slide. Printed slide number "64".

## Slide 65 — New capabilities by “visual prompting”

Title: "**New capabilities** by “visual prompting”" ("New capabilities" bold). At the lower right: "[Bahng, Jahanian, Sankaranarayanan, Isola 2022]". Credit at the lower left under the figure: "Courtesy of Bahng, et al. Used under CC BY." (a credit, not an OCW notice).

A figure panel headed "(a) Prompting with CLIP". Top row, left to right: a photograph of a dog with brown, white and black fur looking up at the camera, seated on pavement with a lead at the lower left; a "+" sign; a square frame labelled "Visual Prompt" (orange) whose border is multicoloured noise-like texture around an empty white centre; an arrow to the same photograph with that noisy border added around it; an arrow to a trapezoid "Image Encoder" with a padlock icon at its upper right; an arrow to a cell "I1", followed by a row of cells "I1 · T1" (shaded grey), "I1 · T2", "…", "I1 · TN", with an arrow from "I1 · T1" up to the text "“This is a photo of a dog.”". Directly under that row, a row of cells "T1", "T2", "…", "TN", each with an arrow up from a long horizontal line from a trapezoid "Text Encoder" (also with a padlock icon). Bottom row: a stack of four boxes "dog", "cat", "⋮", "bird" at the left, an arrow to a box "“This is a photo of a [object].”" labelled "Text Prompt" (green) above it, and an arrow to the "Text Encoder". Printed slide number "65".

## Slide 66 — New capabilities by “visual prompting”

![Slide 66 — New capabilities by “visual prompting”](../images/18-transfer-learning-models/slide-66.jpg)

Title: "**New capabilities** by “visual prompting”". At the lower right: "[Bahng, Jahanian, Sankaranarayanan, Isola 2022]". Credit at the lower left: "Courtesy of Bahng, et al. Used under CC BY." (a credit, not an OCW notice). This repeats slide 65's panel (a) in full and adds a second panel below.

Panel (a) is as on slide 65: "(a) Prompting with CLIP", the dog photograph plus the "Visual Prompt" border giving the bordered photograph, a locked "Image Encoder" giving "I1" and the row "I1 · T1" (grey) ... "I1 · TN", the output "“This is a photo of a dog.”", the "T1" ... "TN" row from a locked "Text Encoder", and the "Text Prompt" "“This is a photo of a [object].”" fed from the list "dog", "cat", "⋮", "bird".

New: panel "(b) Prompting (adversarial reprogramming) with vision models". Left to right: the same dog photograph, "+", a "Visual Prompt" (orange) noisy border frame, an arrow to the bordered photograph, an arrow to a locked "Image Encoder", an arrow to a green-headed list "Hard-coded Mapping" with three lines "ImageNet Index 0 → “Dog”", "ImageNet Index 1 → “Cat”", a "⋮", and "ImageNet Index 9 → “Bird”". Printed slide number "66".

## Slide 67 — New capabilities by “visual prompting”

![Slide 67 — New capabilities by “visual prompting”](../images/18-transfer-learning-models/slide-67.jpg)

Title: "**New capabilities** by “visual prompting”". At the lower right: "[Bahng, Jahanian, Sankaranarayanan, Isola 2022]". Credit at the lower left: "Courtesy of Bahng, et al. Used under CC BY." (a credit, not an OCW notice).

At the left, a square, blurry overhead-looking picture of a patchwork of dark green, brown, pink-beige and purple-brown rectangular fields divided by a pale diagonal strip; a single tiny red dot is visible a little right of centre, about a fifth of the way down from the top, on a pink-beige field. Caption below: "Adding a single pink pixel gives +3% acc boost".

At the right, a horizontal bar chart (one series, light-blue bars, with the value printed to the right of each bar in blue; the two negative bars are red with red values printed to their left). The x-axis is "Δ Score (%)" with ticks −20, 0, 20, 40, 60, 80; each bar's dataset name is printed at the left of the zero line (for the two negative bars, to the right of the bar). Bars from top to bottom:

| Dataset | Δ Score (%) |
| --- | --- |
| SVHN | +83.2 |
| CLEVR | +61.3 |
| EuroSAT | +56.4 |
| Resisc45 | +42.1 |
| iWildCAM | +37.6 |
| Camelyon17 | +37.1 |
| FMoW | +19.4 |
| DTD | +14.1 |
| CIFAR100 | +12.2 |
| Flowers102 | +8.4 |
| UCF101 | +6.2 |
| CIFAR10 | +5.2 |
| SUN397 | +0.6 |
| Food101 | -0.9 |
| OxfordPets | -0.9 |

Printed slide number "67".

## Slide 68 — Adaptation can occur on inputs, mapping, or outputs

![Slide 68 — Adaptation can occur on inputs, mapping, or outputs](../images/18-transfer-learning-models/slide-68.jpg)

Title: "Adaptation can occur on inputs, mapping, or outputs". At the lower right: "[Bahng, Jahanian, Sankaranarayanan, Isola 2022]". Credit at the lower left: "Courtesy of Bahng, et al. Used under CC BY." (a credit, not an OCW notice).

A figure with four columns and three horizontal bands (dashed grey horizontal lines separate them), band labels at the right in italics: "Inputs" (top), "Model parameters" (middle), "Outputs" (bottom). Column headers: a grey box "Pre-training" (left, separated from the others by a vertical dotted line), a grey box "Adapting to New Tasks" over three columns "(a) Fine-tuning", "(b) Linear Probe", "(c) Prompting", and a legend box "Adapted" (a yellow swatch in a dashed outline) at the top.

- Pre-training column: input a photograph of a grey and white cat looking up; arrow down to a light-grey trapezoid "f" (no fill colour for adaptation); arrow down to a bar chart with labels and black bars from longest to shortest: "Cat", "Dog", "Mug", "Hat".
- (a) Fine-tuning: input a photograph of a yellow and white daffodil-like flower; arrow to a yellow-filled trapezoid "Pre-trained f" (adapted, no padlock); arrow down to bars "Daffodil" (longest), "Rose", "Tulip", "Iris".
- (b) Linear Probe: the same flower photograph; arrow to a white trapezoid "Pre-trained f" with a padlock icon; arrow down to a yellow box "Linear Layer"; arrow to bars "Daffodil", "Rose", "Tulip", "Iris".
- (c) Prompting: a rounded frame labelled "Prompted Image" holding the flower photograph, a "+", and a small 3×3 yellow-toned pixel grid labelled "Visual Prompt"; arrow down to a white trapezoid "Pre-trained f" with a padlock icon; arrow down to bars "Daffodil", "Rose", "Tulip", "Iris".

Yellow marks the adapted part: the model in (a), the linear layer in (b), the input (the visual prompt) in (c). Printed slide number "68".

## Slide 69 — Visual Prompt Tuning (Input Encodings)

Upper right: "Slide credit: Hyojin Bahng" (a slide credit). Title: "Visual Prompt Tuning (Input Encodings)". Citation at the lower right: "Visual Prompt Tuning."

*OCW notice: © Jia, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (small print at the lower right of the figure; it covers the whole figure).*

The figure has a small input-encoding diagram at the left and two architecture diagrams (a) and (b) at the right, with row labels in grey bold: "Head" (top), "Backbone" (middle), "Input" (bottom).

Left diagram: a 2 × 2 grid of four image patches (a dog photograph cut into four tiles: fence and leaves, dog's face, ground and a lead, dog's body) with an arrow up to a row of small photographs labelled "I1", "…", "Im" (four thumbnails shown, with "I1" under the first and "Im" under the last); above them a blue bar with a snowflake icon labelled "Embed"; above that four grey vertical bars joined by a bracket labelled "E0".

A single dashed grey rounded outline encloses the backbones of both panels. Panel "(a) Visual-Prompt Tuning: Deep": three blue bars with snowflake icons labelled "Transformer Encoder Layer" with "L1" (bottom), "L2" (middle), "LN" (top), with "…" between L2 and LN. One token row is drawn between L1 and L2: a grey bar, a pink-backed group of two orange-red bars with flame icons and "…" between them, and four grey bars. The orange-red bars there have no line coming up from L1 (the grey bars do); they only feed arrows up into L2. A second such group is in the Input row, labelled P0. At the Input row: a small blue vertical bar labelled "CLS" with a snowflake, labelled "x0" below; the orange-red group labelled "P0" below; a bracket under the grey bars labelled "E0". Arrows point up through the layers; at the top an arrow goes up from the backbone to an orange-red box "Head" with a flame icon.

Panel "(b) Visual-Prompt Tuning: Shallow": the same stack of three frozen encoder layers (L1, L2, "…", LN) with the same Input row ("CLS" with snowflake labelled "x0"; the orange-red "P0" group with flames; grey bars with bracket "E0") but the orange-red prompt group appears only at the Input row; its token row between L1 and L2 is seven grey bars, with "…" between the second and third, each fed from L1 and feeding L2. An arrow goes up to a "Head" with a flame icon.

A legend at the upper right: an orange-red box with a flame labelled "Tuned" and a blue box with a snowflake labelled "Frozen". Printed slide number "69".

## Slide 70 — Visual Prompting via Image Inpainting

![Slide 70 — Visual Prompting via Image Inpainting](../images/18-transfer-learning-models/slide-70.jpg)

Upper right: "Slide credit: Hyojin Bahng" (a slide credit). Title: "Visual Prompting via Image Inpainting". Citation at the lower right: "Visual Prompting via Image Inpainting." Credit at the lower left under the figure: "Courtesy of Bar et al. Used under CC BY." (a credit, not an OCW notice).

Top half, a diagram. At the left three small pictures with labels: "x1" above a photograph of a rider in red and white on a brown horse on a green field (labelled below "Task Input Example"); "y1" above a black picture with a white silhouette of the horse and rider (labelled "Task Output Example"); "xq" above a photograph of a brown horse grazing in a field (labelled "Query"). Curved arrows run from "x1" and "y1" over to a 2 × 2 grid labelled below "Visual prompt image" (an L-shaped green outline around its three filled cells, with "xvp" outside its upper-right corner), and an arrow from the query to the grid; the text "Concatenate into single image" sits between. The grid holds, top row: the rider photograph and the black-and-white silhouette; bottom row: the grazing-horse photograph and a grey-and-white checkerboard (the empty, to-be-filled cell). An arrow leads to a grey rounded box "Inpainting Model f", then an arrow to the same 2 × 2 grid labelled "Output", in which the bottom-right cell is outlined in red and now shows a rough white silhouette, in the place of the grazing horse, on black.

A horizontal grey line separates the bottom half: five small 2 × 2 example grids side by side, each captioned below, each with the bottom-right cell outlined in red (the model's output):

- "Edge detection": top row a photograph of two people holding a large fish and its edge map (light lines on black); bottom row a four-legged animal with its head down at the water's edge and, outlined in red, its edge map (light lines on black).
- "Colorization": top row a grey-scale then colour photograph of an open-mouthed big cat; bottom row a grey-scale then colour photograph of a bird on a branch.
- "Inpainting": top row a pink flower with a butterfly, with a black square missing, then the completed image; bottom row a decorated cupcake with a black square missing, then the completed image.
- "Segmentation": top row a red sports car and its white-on-black silhouette; bottom row a dark SUV on a street and its white-on-black silhouette.
- "Style transfer": top row a brick building by a lawn with trees, then the same scene with snow; bottom row a building scene, then a snowy/icy version.

Printed slide number "70".

## Slide 71 — Good Resources

Upper right: "Slide credit: Hyojin Bahng" (a slide credit). Title: "Good Resources". Two bullets, both hyperlinks (teal, underlined):

- "https://github.com/thunlp/PromptPapers"
- "https://thegradient.pub/prompting/"

Printed slide number "71".

## Slide 72 — Combining Foundation Models

A section slide: "Combining Foundation Models" in large text at the centre of an empty white slide. Printed slide number "72".

## Slide 73 — Learn foundation models / Use/adapt foundations to solve new problems

![Slide 73 — Learn foundation models / Use/adapt foundations to solve new problems](../images/18-transfer-learning-models/slide-73.png)

Two diagrams side by side. This repeats slide 46's layout and adds a combined application.

Left, heading "Learn foundation models": the same three staggered rows of yellow-orange boxes as slide 46, top row "BERT", "SimCLR", "CLIP", "GPT"; middle row "AlexNet", "SimCLR", "DALL-E", "WaveNet"; bottom row "BigGAN", "StyleGAN", "VQGAN"; an upward arrow from a grey box "Learner", and an upward arrow into "Learner" from the text "Tons of data".

Right, heading (two lines) "Use/adapt foundations to solve new problems": a large pink box labelled "Text-to-image" at the top, containing two yellow-orange boxes, "VQGAN" at the left and "CLIP" at the right, joined by a dark chain-link icon between them; an upward arrow from a grey box "Adaptor" into the pink box; two upward arrows into "Adaptor" from "Little or no data". (What it adds relative to slide 46: the pink "App" box is replaced by the pink "Text-to-image" box holding "VQGAN" and "CLIP" linked.) Printed slide number "73".

## Slide 74 — New capabilities by plugging pretrained models together: CLIP+GAN

Title: "**New capabilities** by plugging pretrained models together: CLIP+GAN" ("New capabilities" bold). Code link at the foot: "Code: https://colab.research.google.com/drive/1_4PQqzM_0KKytCzWtn-ZPi4cCa5bwK2F?usp=sharing" (the slide number overlaps the link).

A diagram. Upper left, the label "INPUT:" (bold) and, in monospace, the quoted text: "What is the answer to the ultimate question of life, the universe, and everything?" (three lines; the opening quote is straight, the closing quote curly, as printed), with an arrow into a lavender trapezoid "Text Encoder", which outputs a small empty rectangle labelled "e1" (bold, subscript 1). A line from "e1" runs right then down with an arrow into a square box "e1 · e2" at the right, with the label "To maximize this" (two lines) and a dotted curved arrow pointing at the box. Lower part: a small square labelled "z" with a thick yellow border, with the label "Optimize this" and a dotted curved arrow pointing at it; an arrow from "z" into a blue trapezoid "Image Generator"; to its right under the label "OUTPUT:" (bold) a square picture that is a nearly uniform, muted purple-brown with faint blotches; then a pale green trapezoid "Image Encoder" outputting a rectangle labelled "e2" (bold, subscript 2). A line from "e2" runs right then up with an arrow into the "e1 · e2" box. The figure shows $z$ being optimised so that the generated image's embedding matches the text embedding. Printed slide number "74".

## Slide 75 — DALL-E 2

![Slide 75 — DALL-E 2](../images/18-transfer-learning-models/slide-75.jpg)

Title: "DALL-E 2". At the lower right: "[Ramesh et al. 2022]", and above it, in small print, "Courtesy of Ramesh et al. Used under CC BY." (a credit, not an OCW notice).

A diagram, left to right. At the far left, blue text in quotation marks, five lines: "“a corgi playing a flame throwing trumpet”" (printed as written). An arrow-less blue trapezoid labelled "text encoder" (the wide side on the left) leads to a short vertical blue column of eight cells (the text embedding). A bracket splits from the column into two branches.

Upper branch: a vertical column of eight blue cells (text embedding) with a double-headed arrow labelled "CLIP objective" to a vertical column of eight tan/lilac cells (image embedding), which sits to the left of a tan trapezoid labelled "img encoder" (the wide side on the right), whose input is a colourful illustration of a dog playing a trumpet that gives off yellow light, on a teal background.

A dashed line separates the upper and lower branches. Lower branch: a column of eight blue cells with two arrows, each into a labelled grey box: the upper box holds a row of five black circles with blue, lilac and tan arcs above them, each arc running from a circle to an earlier (left-hand) one, arrowhead on the earlier circle, and the lower box holds three columns of three circles (blue, lilac, tan), with arrows only along the middle row, blue → lilac → tan; the two boxes are together labelled "prior" beneath them. Their outputs lead to a column of eight tan cells and a three-column circle diagram (five circles per column, blue, lilac, tan, with arrows only along the middle row) labelled "decoder", which produces an illustration of a dog playing a trumpet in front of an orange-red circle on a dark background. Printed slide number "75".

## Slide 76 — Transfer Learning I

Title: "Transfer Learning I". A bulleted list with two indent levels:

- "Learning with little data"
- "Transferring knowledge about the mapping"
  - "Finetuning"
  - "Domain adaptation"
- "Transferring knowledge about the outputs"
  - "Knowledge distillation"
- "Foundation models"

Printed slide number "76".

## Slide 77 — (OCW end page)

OCW's appended end page (a smaller 792×612 page), not lecture content. Text, top left, in four blocks: "MIT OpenCourseWare" and "https://ocw.mit.edu" (the URL a blue underlined link); "6.7960 Deep Learning" and "Fall 2024"; and "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms" (the URL a blue underlined link). It prints the slide number "77" at the bottom centre.
