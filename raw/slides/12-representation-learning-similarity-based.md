---
title: Lecture 12 — Similarity-based Representation Learning (slide deck)
lecture: 12
slides: 70
source_pdf: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec12.pdf
note: Printed slide numbers 1–69 (bottom centre) equal the PDF page numbers exactly. Page 70 is OCW's appended end page (a smaller page), which also prints 70.
figure_audit: Transcribed by Sonnet from page images; 46 figure-, diagram-, chart-, equation- and photo-heavy pages (1, 6–8, 11–14, 17–25, 28–37, 39, 41–43, 48–55, 58–60, 64 and 66–68) were then checked by Opus, a different model, from 100–600 dpi renders, the embedded rasters at native resolution, the vector data, the fonts and the text layer. Every equation agreed but slide 39's bound, which prints no subscript on the loss; corrections were applied on 26 pages, and slide 37's garbled labels were decoded.
---

# Lecture 12 — Similarity-based Representation Learning: slide-by-slide

Text and figures of all 70 pages of
[`mit6_7960_f24_lec12.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec12.pdf),
transcribed from the deck (speaker: Sara Beery; the title slide reads "Lecture 12: Similarity-based Representation Learning", and slide 2's roadmap is headed "Roadmap: similarity-based representation learning", with no lecture number). Cite these as "slide N" — the printed number equals the PDF page number for slides 1–69; page 70 is OCW's appended end page and prints 70. Diagrams, plots, screenshots and photographs are described in prose since the KB is read as text. Several figures were rendered with missing glyphs (a transpose sign, a temperature, a set-membership sign, a script loss letter, and the labels of slides 24 and 37); where that happens the slide's text says so and gives the reading, if any, that the printed layout supports.

**Images.** 8 slides carry a whole-slide render under their heading: 8, 22, 23, 35, 43, 49, 50 and 51. Not rendered: the 37 slides with an OCW exclusion notice (1, 11, 14, 17, 19, 20, 24, 25, 28–34, 36, 37, 41, 42, 48 and 52–68; all but two of those notices are drawn as glyph outlines, so the PDF's text layer shows only slides 14's and 59's); slide 7, a build step that slide 8 completes; text, equation and screenshot slides this file reproduces (6, 10, 12, 13, 16, 18, 21 and 39); and the roadmap, text slides, summary and end page (2–5, 9, 15, 26, 27, 38, 40, 44–47, 69 and 70). See `AGENTS.md`.

Companion pages: [wiki page for this lecture](../../wiki/12-representation-learning-similarity-based.md) · [transcript](../transcripts/12-representation-learning-similarity-based.md)

**Signposting slides you can skip.** Slide 1 is the title; slide 2 is the roadmap (Representation learning — why?; What is a "good" representation?; Metric learning; Contrastive representation learning (self-supervised), with sub-items "What does it do?" and "Models"), and slide 26 repeats it with the fourth item in red; slide 27 is the section opener for self-supervised contrastive learning; slide 38 ("What is this method doing?") lists two ingredients; slide 69 is the summary; slide 70 is the OCW end page.

Some slides are **build steps** or near-repeats, transcribed individually with a note of what they add: slides 4 to 6 and 9 (the "good representation" quotation, its first two properties, the competition paper, and the full list of five); slides 7 and 8 (the t-SNE panels, then labelled "generalizes" and "cannot generalize" with a text block); slides 18 and 19 (the triplet loss, then its before-and-after picture of three photographs); slides 28 to 30 and 32 (the common contrastive setup, with a different lower-left text each time); slides 35 to 37 (the two-encoder "views" diagram with cross-channel, video and image-text examples); slides 44 and 46 (the properties list, first with a closing question, then with a third item); slides 49 and 50 (the SimCLR diagram, then with the projection-head bar chart); slides 55 to 59 (the iNaturalist taxonomy tree, then with "Coarse-grained", an arrow and "Fine-grained", two moths, and a chart frame); slides 60 to 66 (the iNat21 line chart, then with callouts, a "Big gap!" arrow, a red box on the coarse levels, and two red arrows); and slides 67 and 68 (a row of bird retrievals, then with a second row). Slides with no printed title are headed here with a description: 55 to 59, 67 and 68. Slide 60's title "iNat21" is printed inside the chart, and so are slides 61 to 66's. Slides 55 to 68 are slides borrowed from Elijah Cole (credited "Slide: Elijah Cole" at the lower right of each, with the citation of Cole et al., CVPR 2022 on slides 55 and 68).

Printed slips and oddities, kept as printed: slide 14's right plot is titled "Porjected 2–class data", and the figure is credited "Ng et al 2003" while slide 13's paper is by Xing, Ng, Jordan and Russell (2003); slide 58's notice prints "Duagran" for "Diagram"; slide 13 prints empty box glyphs where the set-membership sign should be, and plain upright S and D for the set letters; slide 12's Mahalanobis norm bars and slides 28 to 30, 32 and 39's transpose sign, temperature, expectation subscripts, summation sign and arrows print as stray marks (slide 39's script $\mathcal{L}$ prints as a breve "˘", and its mutual-information bound drops the "cont" subscript); slide 12's first distance equation is typed in a plain font with $\mathrm{W}$ and $\mathrm{x}$ not bold, where the lines around it use bold; slide 19's label of the monkey photograph is clipped to "x ⁻"; the labels of slide 24's photo groups and legend and the anchor and right-hand text of slide 37 are garbled (slide 37's decode by a consistent glyph substitution and are given there); slide 33's "Negative examples" are "randomly uniformly drawn from data"; slide 44 asks "What do the selection of positive and negative pairs encourage?"; slide 48's notice skips the words between "Chen, et al." and "content is excluded"; slide 53's notice is overprinted by the slide number; slide 64 prints "Imagenet" with a lowercase n; and slide 45 points to a "geometric DL lecture" without a number (not a lecture of this KB so far). The deck prints no other lecture number.

**OCW licence notices** are on slides 1, 11, 14, 17, 19, 20, 24, 25, 28 to 34, 36, 37, 41, 42, 48, 52 to 54 and 55 to 68. Slides 55 to 57 and 59 to 66 carry the same Elijah Cole notice, slide 58's differs ("Duagran", "Other images © source unknown"), and slides 67 and 68 read "Slide © Elijah Cole. Image © source unknown. …". The PDF's text layer carries only slides 14's and 59's; the rest were read from the page images. Each is transcribed as an `*OCW notice: …*` line with the figure it sits beside. Slides 7, 8, 22, 23 and 35 print "Courtesy of … Used under CC BY" or "CC BY-NC-SA" instead, which are not exclusions: slides 7 and 8 (Chuang et al.), 22 and 23 (Song et al.) and 35 (Tian et al.); slide 25's "Fig 4. Courtesy of Kaya and Bilge. Used under CC BY" covers the negative-mining figure only, and slide 33's "Original image courtesy of Von.grzanka. Used under CC-BY" its original dog photograph.

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and roadmap |
| 3–9 | Why learn representations?; what is a "good" representation (compact, explanatory, concentrated, separated, robust); the generalization competition and t-SNE of true and random labels |
| 10–16 | Similarity-based representation learning: unsupervised or supervised; metric learning, the linear Mahalanobis form, upper and lower bound constraints (Xing et al. 2003), deep metric learning |
| 17–25 | Contrastive losses: the moth intuition, the triplet loss and network, the lifted structured loss, example embeddings and neighbours, what makes an image "similar", hard-negative mining |
| 26–31 | Self-supervised contrastive representation learning: the common setup on a hypersphere, NCE and InfoNCE, why a hypersphere |
| 32–37 | Positives and negatives: augmentations, SimCLR, and variations (cross-channel, video, image–text) |
| 38–46 | What the method is doing: mutual information, alignment and uniformity, the feature distributions and their link to quality, what the choice of pairs teaches |
| 47–52 | Ingredients: data augmentation, the projection head, batch size, improving negative samples |
| 53 | Supervised and semi-supervised contrastive learning |
| 54–68 | Case study, iNaturalist 2021 (Cole et al.): the taxonomy, supervised against SimCLR and MoCo across the label hierarchy |
| 69 | Summary |
| 70 | OCW end page |

---

## Slide 1 — Lecture 12: Similarity-based Representation Learning

Title: "Lecture 12: Similarity-based Representation Learning" (wrapping onto two lines). Subtitle: "Speaker: Sara Beery".

Lower right, a figure in the style of a contrastive-learning diagram. At the left, a square photograph of a golden retriever's head and shoulders, seen at a three-quarter angle against green grass. Two curved black arrows leave it, one up-right and one down-right. The upper arrow points to a second square photograph of the same dog, recoloured in strong yellow-green tones (an augmented view); the lower arrow points to a third photograph of the same dog, recoloured in strong red-pink tones. A black curved line runs from the right edge of each of the two recoloured photographs (no arrowheads) into a large grey circle (a vertical gradient, lighter at the top and darker at the bottom, black outline) at the right. Inside the circle sit two green dots close together in the upper left, and one red dot lower right of centre. A further black curved line runs from the circle's lower right edge down to a fourth square photograph, at the bottom right, of a monkey (a patas-monkey-like primate with a pale grey-brown body and reddish-brown head) sitting in green grass with a hand at its mouth.

The notice sits beneath the dog and monkey photographs, left of the monkey: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits beside the dog and monkey photographs).*

Footer bar (grey): MIT logo at left, "6.7960 Deep Learning" (red), "https://phillipi.github.io/6.7960" (underlined), right side "Fall 2024". The printed slide number "1" sits just below the end of the URL.

## Slide 2 — Roadmap: similarity-based representation learning

Title: "Roadmap: similarity-based representation learning".

- Representation learning — why?
- What is a "good" representation?
- Metric learning
- Contrastive representation learning (self-supervised)
  - What does it do?
  - Models

## Slide 3 — Why learn representations?

Title: "Why learn representations?".

- To improve generalization
- To do more learning (transfer learning)
- To exploit geometric similarity for new data or queries:
  - Have we seen the face of this person before or is it new?
  - Retrieval: which items are similar to the query?
- To improve clustering with side information (similar/dissimilar pairs)
- Dimensionality reduction (often unsupervised)

At the bottom centre, in bold blue sans-serif type: "What do we expect from such representations?"

## Slide 4 — What is a "good" representation?

Title: "What is a "good" representation?". In a light-grey box spanning the slide's width: "Generally speaking, a good representation is one that makes a subsequent learning task easier." — *Deep Learning*, Goodfellow et al. 2016 (the book title in italics). Below, plain text: "What could this mean?".

## Slide 5 — What is a "good" representation?

Title: "What is a "good" representation?".

1. Compact (*minimal*)
2. Explanatory (*sufficient*)

(the words minimal and sufficient are in italics.)

## Slide 6 — What is a "good" representation?

Title: "What is a "good" representation?". Centre, a screenshot (with a soft drop shadow) of the title block of a paper: "NeurIPS 2020 Competition: Predicting Generalization in Deep Learning (Version 1.1)" with "Version 1.1" in red; authors in three rows, with their affiliation marks as printed: Yiding Jiang \*†, Pierre Foret†, Scott Yak†, Daniel M. Roy‡§ / Hossein Mobahi†§, Gintare Karolina Dziugaite¶§, Samy Bengio†§ / Suriya Gunasekar‖§, Isabelle Guyon \*§, Behnam Neyshabur†§; then an email address in pink typewriter type, "pgdl.neurips@gmail.com", and the date "December 16, 2020". (The affiliation marks are small and partly overprinted; they are given here as best read.)

Below the screenshot:

3 winning strategies look at:

- Geometry of representation: consistency, separation
- Robustness to perturbations

## Slide 7 — What helps generalization?

Title: "What helps generalization?". One bullet: "Representations of CIFAR-10 data with true and random labels".

Below, a figure of two square scatter-plot panels side by side, each in a thin black frame, with captions "(a) Clean Labels" and "(b) Random Labels" beneath them and, below that, "Figure 4: t-SNE visualization of representations. Classes are indicated by colors." Both panels use the same ten colours, one per class, no axes.

- Panel (a), Clean Labels: ten small, tight, well-separated round clusters with blank space between them. Roughly: pink-magenta at the top; a green-teal one and a lime-yellow one at the upper left and upper middle-right; a green one at the right; a red one at the centre; violet-magenta at the left; an orange-gold one below the centre; a purple one right of it; a cyan one at the lower left and a blue one at the bottom.
- Panel (b), Random Labels: the same ten colours, but each is a larger, fuzzier cluster and all ten are packed together into a single roughly circular mass in the middle of the panel, touching or overlapping one another, with scattered single stray dots around the edge (for example a pink dot at far left and a teal dot at far right).

At the lower right, in small type: "Courtesy of Chuang, et al. Used under CC BY." Beneath it, in italics: "image: Chuang et al., Measuring generalization with optimal transport, 2021". The printed slide number "7" overlaps the "a" of "image".

## Slide 8 — What helps generalization?

![Slide 8 — What helps generalization?](../images/12-representation-learning-similarity-based/slide-8.jpg)

Title: "What helps generalization?". Build step: as slide 7, plus two bold labels inside the top of the panels and a text block at the right.

One bullet: "Representations of CIFAR-10 data with true and random labels". The same two-panel figure, caption and credits as slide 7 ("Courtesy of Chuang, et al. Used under CC BY."; "image: Chuang et al., Measuring generalization with optimal transport, 2021"), with a bold sans-serif label "generalizes" across the top of panel (a) and "cannot generalize" across the top of panel (b).

At the right, text with bold run-in headings:

**Concentration/consistency**: Data from the same class is close together
**Separation**: classes are well separated
**Robustness**

(the first two are a bold heading followed by plain text; "Robustness" stands alone in bold.)

## Slide 9 — What is a "good" representation?

Title: "What is a "good" representation?".

1. Compact (*minimal*)
2. Explanatory (*sufficient*)
3. Concentration: Data from the same class is close together
4. Separation: classes are well separated
5. Robustness to irrelevant perturbations

At the bottom centre, in bold red sans-serif type: "How could we encourage a model during training to achieve this?"

Build step: slide 5's list, completed with items 3 to 5 (item 5 now reads "Robustness to irrelevant perturbations").

## Slide 10 — Similarity-based representation learning

Title: "Similarity-based representation learning". One bullet: "Encourage good representations via feedback in terms of similarity: pairs of similar/dissimilar inputs".

Below, two thick black arrows leave a common point under the bullet: one points down and to the left, ending above the bold word "Unsupervised" at the left; the other points down and to the right, ending above the bold word "Supervised" at the right.

## Slide 11 — Metric Learning

Title: "Metric Learning".

- Euclidean distance in input space may be not ideal
- Instead: learn a metric that respects desired properties
- Goal: learn a metric where:
  - data points that "belong together" are *similar* (close together) (the words "similar (close together)" in green, "similar" italic)
  - data points that are "different" are dissimilar (far apart) (the words "dissimilar (far apart)" in red)
- "Supervision": similarity information.

Along the bottom, three square-ish photographs of faces in a row. Left: a smiling man with short light-brown hair in a blue T-shirt against a blue water background. Middle: a man who looks the same (the slide's green arrow marks them as similar), seen from the side at a lectern with a headset microphone against a black background. Right, separated by a gap: a smiling woman with short brown hair in a denim shirt against a light-wood background. A green double-headed arrow joins the left and middle photographs; a red double-headed arrow joins the middle and right photographs.

The notice sits at the lower right, beneath the right photograph: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". The printed slide number "11" sits near the bottom centre.

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits under the three face photographs).*

## Slide 12 — Metric learning (linear)

Title: "Metric learning (linear)".

- Data points $\mathbf{x}_ 1, \ldots, \mathbf{x}_ n$
- Weak supervision, with two set definitions (italic bold $\boldsymbol{x}$ in the printed math):

$$\mathcal{S} := \lbrace (\boldsymbol{x}_ i, \boldsymbol{x}_ j) \mid \boldsymbol{x}_ i \text{ and } \boldsymbol{x}_ j \text{ are in the same class} \rbrace$$

$$\mathcal{D} := \lbrace (\boldsymbol{x}_ i, \boldsymbol{x}_ j) \mid \boldsymbol{x}_ i \text{ and } \boldsymbol{x}_ j \text{ are in different classes} \rbrace$$

At the right of these two lines, small bold-italic sans-serif labels: "similar" beside the first and "dissimilar" beside the second.

- **Goal:** learn a linear transformation $\mathbf{z} = \mathbf{W}\mathbf{x}$ that respects similarity (the word "Goal:" in bold)
- Use Euclidean distance in representation space:

$$\Vert \mathrm{z}_ i - \mathrm{z}_ j \Vert^2 = (\mathrm{x}_ i - \mathrm{x}_ j)^\top \mathrm{W}^\top \mathrm{W} (\mathrm{x}_ i - \mathrm{x}_ j)$$

This line is set in plain sans-serif type with no bold; its $\mathrm{W}^\top \mathrm{W}$ (both W's and the ⊤ between them) is printed in blue, while the ⊤ after the first bracket is black. At the right, also in blue and in the same plain type: $A = W^\top W$ (printed "A = W ⊤W").

Last line: "*Mahalanobis distance* with positive semidefinite matrix $\mathbf{A}$," followed by $d_{\mathbf{A}}(\mathbf{x}_ i, \mathbf{x}_ j) = \Vert \mathbf{x}_ i - \mathbf{x}_ j \Vert_ {\mathbf{A}}$. In the render the two norm bars are drawn as small tilde-like marks ("˜ x_i − x_j ˜ A") rather than vertical bars; read here as the norm with subscript $\mathbf{A}$ (the slide does not otherwise define it).

At the bottom centre, in bold red sans-serif type: "How can we phrase this as an optimization problem?"

## Slide 13 — "Losses": upper/lower bound constraints

Title: ""Losses": upper/lower bound constraints".

- first approach (*Xing et al 2003*):

The optimization problem, set in plain sans-serif type, with two coloured captions at its right:

$$\min_{\mathrm{A} \succeq 0} \sum_{(i,j) \in \mathcal{S}} d_{\mathrm{A}}(\mathrm{x}_ i, \mathrm{x}_ j)^2$$

$$\text{s.t.} \sum_{(k,\ell) \in \mathcal{D}} d_{\mathrm{A}}(\mathrm{x}_ k, \mathrm{x}_ \ell)^2 \geq 1$$

Caption beside the first line, in green: "min distance of similar points". Caption beside the constraint, in red: "keep distance of dissimilar points". The printed under-sum indices "(i,j) ∈ S" and "(k, ℓ) ∈ D" show an empty box glyph where $\in$ is and after S and D (font glyphs missing in the export), so what is printed is "(i,j ) ▯ S▯" and "(k, ℓ) ▯ D▯". The set letters themselves print as plain upright S and D, not calligraphic; they are read here as $\mathcal{S}$ and $\mathcal{D}$, as on slide 12. The points print as upright sans-serif x's, as in slide 12's expansion line.

At the upper right, a screenshot with a soft shadow of a paper's title block: "Distance metric learning, with application to clustering with side-information" by "Eric P. Xing, Andrew Y. Ng, Michael I. Jordan and Stuart Russell", "University of California, Berkeley", "Berkeley, CA 94720", and the email "{epxing,ang,jordan,russell}@cs.berkeley.edu". Beneath it, in italics: "introduced the term and problem in 2003".

- can swap objective and constraint (upper bound for similar pairs)
- many related ideas & follow-ups, e.g.
  *information-theoretic metric learning (Davis et al 2007)*:
  preserve distribution information (relative entropy between Gaussians) while observing upper/lower bounds as constraints

## Slide 14 — Simple example

Title: "Simple example". Two 3D scatter plots side by side, each with three labelled axes: a vertical z axis with ticks −10, 0, 10; a depth axis y running from the bottom corner (where it meets the x axis) up to the left, labelled 20, 0, −20 from its far (upper-left) end to the corner; and an x axis running to the right with ticks −20, 0, 20. Two data classes are drawn, red "+" markers and blue "×" markers.

- Left plot, titled "Original 2–class data": four small tight clusters, two red and two blue. At the upper left a red cluster (z about 5) with a blue cluster just to its right and slightly higher; at the lower right a second red cluster (z about −9) with a blue cluster just to its right. Each red cluster is next to a blue one, so the clusters of the two classes alternate.
- Right plot, titled "Porjected 2–class data": the same classes after projection, as two elongated thin clusters: a red strip at the left (z about −3) and, to its upper right, a blue strip (z about 0), both nearly horizontal and clearly separated, with one cluster per class.

The notice sits at the lower right: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". Beneath it, in italics: "figure: Ng et al 2003".

*OCW notice: © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits under the two scatter plots).*

## Slide 15 — Improvements / developments

Title: "Improvements / developments".

- Nonlinear transformations (kernels, deep metric learning)
- Contrastive losses
- Normalization of representations: angle instead of distance

## Slide 16 — Deep metric learning

Title: "Deep metric learning".

- Linear metric learning: learn a linear transformation $\mathbf{z} = \mathbf{W}\mathbf{x}$
- Deep metric learning: learn a nonlinear transformation $\mathbf{z} = f(\mathbf{x})$

In the second line the $f$ is highlighted with a pale-blue rounded box; a thin blue arrow from the blue bold text "neural network" (lower right) points up to the highlighted $f$. Below, centred, in italics: "optimize not over psd matrices but weights of a neural network".

## Slide 17 — Contrastive losses: intuition

Title: "Contrastive losses: intuition". Three photographs of moths on green leaves, two in an upper row and one centred below:

- Upper left: a white moth with pale blue-grey shading and orange-brown markings along the tips and edges of its wings, wings spread, on a green leaf.
- Upper right (the two upper photographs are far apart, with a wide blank gap between them): a pale cream moth with fine wavy lines on its wings, wings spread flat, resting on a green leaf among other leaves.
- Bottom centre: a white moth with grey-brown markings on the edges of its wings, on a green leaf with a reddish-brown stem.

The notice sits at the lower left: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". The printed slide number "17" overlaps the bottom photograph's lower edge.

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits beside the three moth photographs).*

## Slide 18 — Contrastive losses

Title: "Contrastive losses". Across the top, two bold sans-serif captions: "distance of dissimilar pair(s)" in dark red at the left and "distance of similar pair(s)" in green to its right. One bullet: "Triplet loss (*Schroff et al 2015*):". Then the displayed equation:

$$\mathcal{L}_ {\text{triplet}}(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-) = \sum_{\mathbf{x} \in \mathcal{X}} \max\left(0, \Vert f(\mathbf{x}) - f(\mathbf{x}^+) \Vert_ 2^2 - \Vert f(\mathbf{x}) - f(\mathbf{x}^-) \Vert_ 2^2 + \epsilon\right)$$

A green line underlines the first squared distance, $\Vert f(\mathbf{x}) - f(\mathbf{x}^+) \Vert_ 2^2$, and a red line underlines the second, $\Vert f(\mathbf{x}) - f(\mathbf{x}^-) \Vert_ 2^2$. (The two captions form a one-line header well above the equation: the red one sits at the left, above the start of the equation and left of both underlined terms; the green one sits to its right, above the green-underlined term and running on over the start of the red-underlined one. A tiny stray mark, printed as a dot, sits between them at their baseline.) A thin black line (its arrowhead is hidden under the equation) runs from the bold word "margin" at the right up-left towards the final $\epsilon$.

At the lower right, in regular type: "related: Large-margin Nearest Neighbor metric learning (LMNN) (*Weinberger et al 2009*)".

## Slide 19 — Contrastive losses

Title: "Contrastive losses". Build step: as slide 18, plus a picture of three photographs before and after, a thick grey arrow, and a notice. The arrow from "margin" now ends in a head at the $\epsilon$.

Same captions, bullet "Triplet loss (*Schroff et al 2015*):", the same equation $\mathcal{L}_ {\text{triplet}}(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-)$ with the green line under $\Vert f(\mathbf{x}) - f(\mathbf{x}^+) \Vert_ 2^2$ and the red line under $\Vert f(\mathbf{x}) - f(\mathbf{x}^-) \Vert_ 2^2$, and the label "margin" with a thin arrow pointing to $\epsilon$.

Below, two three-photo arrangements joined by a big grey right-pointing block arrow in the middle.

- Left arrangement: a photograph labelled x (a person in dark clothes riding a white horse on grass, a crowd behind) at the left; at the upper right a photograph labelled $x^+$ (a pale-coloured horse with a pale mane standing on grass); at the lower right a photograph labelled $x^-$ (a grey-blue monkey sitting, holding food, in front of green foliage). A green double-headed arrow joins x and $x^+$; a red double-headed arrow joins x and $x^-$. The two arrows are about the same length.
- Right arrangement: the same three photographs, but now x and $x^+$ are closer (a short green double-headed arrow) and $x^-$ sits far away at the lower right (a long red double-headed arrow from x down to $x^-$). In the render the label of the monkey reads "x −" with a small clipped mark beside it ("x⁻" cut off at the right edge).

The notice sits at the bottom centre-right: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". The printed slide number "19" overlaps the second line of the notice.

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits under the horse and monkey photographs).*

## Slide 20 — Triplet network

Title: "Triplet network". A block diagram with three horizontal rows, one each for the labels "anchor", "positive" and "negative" (small bold labels at the far left of each row).

- Each row: a face photograph at the left, a thin arrow into a white rectangle labelled "CNN", a thin arrow out to a tall rounded capsule (a vertical pill) holding four circles of different grey shades, which is that row's embedding. The heading "Embeddings" sits above the top capsule.
- Photographs: the anchor row shows a smiling man in a dark suit and white shirt, facing the camera; the positive row shows a man who looks the same, in profile or three-quarter view, in a suit; the negative row shows a different, smiling man with short brown hair in a dark suit.
- Between the CNN boxes, two vertical double-headed (hollow) arrows, one between anchor and positive and one between positive and negative, each labelled "Shared weights".
- Capsule circles from top to bottom: anchor, mid grey, light grey, black, very light grey; positive, mid grey, light grey, dark grey, very light grey; negative, black, light grey, light grey, dark grey. (Shades approximate.)
- At the right a hexagon labelled "Triplet Loss". Three arrows end on it: a curved one from the anchor capsule, a straight horizontal one from the positive capsule and a curved one from the negative capsule.

The notice sits at the lower right: "Image © Olivier Moindrot. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". Beneath it, in italics: "figure: https://omoindrot.github.io/triplet-loss".

*OCW notice: Image © Olivier Moindrot. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits under the whole triplet-network diagram, face photographs included).*

## Slide 21 — Contrastive losses

Title: "Contrastive losses". The same two bold captions as slide 18 sit across the top: "distance of dissimilar pair(s)" in dark red at the left and "distance of similar pair(s)" in green to its right (here no underlines accompany them).

- Improvements: compare to multiple negatives per positive pair, e.g. Lifted structured loss (*Song et al 2015*): compare to all negatives in a batch

$$\mathcal{L}_ {\text{struct}} = \frac{1}{2\vert \mathcal{P} \vert} \sum_{(i,j) \in \mathcal{P}} \max\left(0, \mathcal{L}_ {\text{struct}}^{(ij)}\right)^2$$

$$\text{where } \mathcal{L}_ {\text{struct}}^{(ij)} = D_{ij} + \max\left( \max_{(i,k) \in \mathcal{N}} \epsilon - D_{ik}, \max_{(j,l) \in \mathcal{N}} \epsilon - D_{jl} \right)$$

In the second line everything after $D_{ij} +$ (the outer $\max$ and its contents, over the negatives) is printed in red. A thin black arrow from the text $\Vert f(x_i) - f(x_j) \Vert_ 2$ (plain sans-serif, below left) points up-right to $D_{ij}$, defining it. At the lower right, in italics: "or smooth relaxation of the max".

## Slide 22 — Example embedding

![Slide 22 — Example embedding](../images/12-representation-learning-similarity-based/slide-22.jpg)

Title: "Example embedding". A figure from Song et al. Centre: a roughly oval cloud of tiny bird-photo thumbnails (a t-SNE map). Around it, ten enlarged square-grid panels, each outlined in a thin red/pink frame and joined by thin red lines to a small red box on the cloud where those thumbnails sit. In the panels white cells are empty grid cells. Counts are of panels, not of photos in them. Positions:

- Top row, four panels left to right: (1) yellow and olive-green small birds (a 6-column by 7-row grid; its top two rows hold photos only in columns 3 and 5, and the first cell of row 6 is blank); (2) small blue-backed, white-bellied birds (swallows and similar) on branches and posts; (3) small brown-grey birds with a crest, some among red berries; (4) black birds (crows or ravens), one tile showing a car wheel.
- Middle left: pale gulls and terns in flight against blue sky. Middle right: small brown streaky birds on grass and branches.
- Bottom row, four panels left to right: (1) white pelicans and large white water birds on water; (2) black-and-white puffins and dark seabirds; (3) red birds on green foliage; (4) woodpeckers with red heads and barred black-and-white backs on tree trunks, with one brown sparrow-like bird among them.

Caption, in serif type: "Figure 9: Barnes-Hut t-SNE visualization [36] of our embedding on the test split (class 101 to 200; 5,924 images) of CUB-200-2011. Best viewed on a monitor when zoomed in." (the "[36]" in green). At the lower right in small type: "Courtesy of Song, et al. Used under CC BY-NC-SA." and, in italics, "figure: Song et al 2015".

## Slide 23 — Example query results (neighbors)

![Slide 23 — Example query results (neighbors)](../images/12-representation-learning-similarity-based/slide-23.jpg)

Title: "Example query results (neighbors)". A grid of bird photographs, six rows by six columns (36 photos). A vertical dotted line separates the first column (the query image) from the five columns of retrieved neighbours to its right. Each row is one kind of bird, and the query and its five neighbours show that kind in different poses and backgrounds:

- Row 1: a black-and-white puffin with a large coloured bill (query: close-up head, and the neighbours puffins in various poses, one in flight against blue sky).
- Row 2: a small black bird with orange patches on wings and tail, perched among twigs or grass.
- Row 3: a small grey-and-white bird with a striped head, on rocks or a branch.
- Row 4: a dark glossy blue-black bird perched on a post, branch or ground.
- Row 5: a bird with a dark blue back and rust-orange throat and cream belly, on a branch or wire.
- Row 6: a bright scarlet-red bird with black wings, on a branch.

At the lower right: "Courtesy of Song, et al. Used under CC BY-NC-SA." and, in italics, "figure: Song et al 2015".

## Slide 24 — What makes an image "similar"?

Title: "What makes an image "similar"?". Left and centre, a figure of nine "triplet" groups arranged as three columns of groups by three rows; each group is three photographs in a row. Right, a bulleted block.

Each group has a small row of glyph text above its three photos which is not legible in the render (printed as `(`, `!"#"$"%&"` and `'`, garbled font characters), and beneath the left and the right photograph a row of five small circular badges: three filled colour dots (pink, orange, green) and two icons (a cyan circle holding a crescent moon, two small stars and a cloud, and a taupe person-in-circle icon). The left and right badge rows are always complementary: whichever colour dots are full under the left photo are faint under the right, and the reverse; the two icons are faint under every left photo and full under every right photo. Full colour dots, left photo then right: butterflies, orange and green, then pink; gallery, pink and green, then orange; pantry, pink and orange, then green; lilies, orange and green, then pink; trees, pink and green, then orange; stockings, pink and orange, then green; amphitheatre, standing stones and treehouses, all three dots, then none (only the two icons are full). The middle photo has no badges. The badges' meaning is given by a legend line at the bottom: a pink circle, an orange circle, a green circle, the blue icon and the grey icon, each followed by a text label that is garbled in the render (printed `!"#"$`, `%#&'`, `(!#"`, `%/0,+$1+` and `)*+,-.`) and so not legible.

The nine groups, in reading order (each of three photos):
1. Monarch butterflies seen from above with wings open (left: wings spread and drooping at the sides; middle: wings spread, tilted; right: wings spread flat).
2. An art gallery room with paintings on the walls (three different rooms, increasing depth of view).
3. Pantry shelves with jars, baskets and boxes.
4. Water lilies: a white lily on green pads, a pink lotus among green pads, a pink lotus on a deep-blue background.
5. Rows of green conifer-like trees (left: close-up wall of dark-green trees; middle: tall trees receding along a path; right: planted rows from above).
6. Christmas stockings with ornaments (white-and-red, blue-and-red, green-and-white stripes).
7. An amphitheatre or arena seating (red seats; grey tiered stone seats on a hillside; blue seats under a roof).
8. Standing stones (a dolmen of stacked rocks; a single tall stone; a rounded standing stone in grass).
9. Treehouses in trees (three different treehouses).

At the right: "Similar in:" followed by bullets "Pose", "Perspective", "Foreground color", "Number of items", "Object shape".

The notice sits at the bottom centre: "Images © Fu, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". The printed slide number "24" overprints the middle of its second line. At the lower right, in italics: "figure: Fu\*, Tamir\*, Sundaram\* et al 2023".

*OCW notice: Images © Fu, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits under the nine photo groups).*

## Slide 25 — Which pairs should we present?

Title: "Which pairs should we present?". Text: ""hard" negatives:" then

- currently "misplaced", i.e., closer to anchor than a positive example
- accelerate learning, needed for triplet loss

Centre left, a figure: a pale-blue square panel holding two nested circles, a large light-blue disc and, inside it, a smaller darker-blue disc. Small labelled squares sit in it: a white square "a" near the centre of the inner disc; a white square "p" to its right and slightly above; an orange square "n1" above "a" inside the inner disc; an orange square "n2" in the outer ring, above and to the right of the inner disc's top (above and right of n1); and an orange square "n3" outside the large disc at the upper right of the panel. Text labels: in the inner disc "Hard Negatives (a, p, n1)"; in the ring "Semi-Hard Negatives (a, p, n2)"; in the outer panel "Easy Negatives (a, p, n3)". A white double-headed block arrow labelled "Margin" lies across the ring at the upper left, between the inner disc's edge and the larger disc's edge. Caption beneath: "Figure 4. Negative Mining." (serif).

Centre right, three formulas in serif type, one above the other:

- "Hard Negative Mining" $d(a, n) \lt d(a, p)$
- "Semi-Hard Negative Mining" $d(a, p) \lt d(a, n) \lt d(a, p) + \mathit{margin}$
- "Easy Negative Mining" $d(a, p) + \mathit{margin} \lt d(a, n)$

(The word margin is printed in math italic, like $d$, $a$, $n$ and $p$. The figure, formulas and caption are a single raster.)

At the far right, three moth photographs stacked vertically: a pale moth with wings spread on a brown wall or bark (top); a white moth with brown-edged wings on grey stone (middle); a white-and-blue moth on a green leaf (bottom; the same photograph as slide 17's upper-left one).

At the lower left, small text: "Fig 4. Courtesy of Kaya and Bilge. Used under CC BY Other images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/." At the lower right, in italics: "figure: Kaya & Bilge: Deep Metric Learning: A Survey, 2019" (the tops of its first words, "figure: Kaya & Bilge:", are hidden under the edge of the figure image).

*OCW notice: Other images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower left, beside the negative-mining figure; "Fig 4. Courtesy of Kaya and Bilge. Used under CC BY" covers the figure, so the notice covers the three moth photographs).*

## Slide 26 — Roadmap: similarity-based representation learning

Title: "Roadmap: similarity-based representation learning". Build step: as slide 2, with the fourth item highlighted.

- Representation learning — why?
- What is a "good" representation?
- Metric learning
- Contrastive representation learning (self-supervised) (this item and its bullet in red)
  - What does it do?
  - Models

## Slide 27 — Self-supervised contrastive representation learning

Title: "Self-supervised contrastive representation learning". One bullet: "Ideas from metric learning and self-supervision".

## Slide 28 — Common setup

Title: "Common setup". Two bullets:

- Encoder maps data onto a hypersphere: $f : \mathcal{X} \rightarrow \mathbb{S}^{d-1}$ (in the render the arrow is a small double tick, "˝", and not legible as an arrow)
- Cross-entropy for softmax "classifier" to discriminate "classes" defined by similarities

Then an equation in serif type. The render drops three symbols (the transpose, the temperature and the "distributed as" sign are printed as the marks "ˆ", "˜" and "˙", and the summation sign as "ˇ"), so the equation is given with those slots marked and with the reading that the printed layout implies:

$$\min_f \mathbb{E}_ {(\mathbf{x}, \mathbf{x}^+) \sim p_{pos}, \lbrace \mathbf{x}_ i^- \rbrace_ {i=1}^{N} \sim p_{data}} \left[ -\log \frac{e^{f(\mathbf{x})^{\top} f(\mathbf{x}^+)/\tau}}{e^{f(\mathbf{x})^{\top} f(\mathbf{x}^+)/\tau} + \sum_{i=1}^{N} e^{f(\mathbf{x})^{\top} f(\mathbf{x}_ i^-)/\tau}} \right]$$

The subscripts pos and data are printed in italic. As printed, the exponents read "f(x)ˆ f(x⁺)/˜" and "f(x)ˆ f(x\_i⁻)/˜"; the $\top$ and $\tau$ shown above are the reading of those marks and are not legible in the render. (The slide does not name the temperature in a legible glyph.) The numerator $e^{f(\mathbf{x})^{\top} f(\mathbf{x}^+)/\tau}$ is in a pale-green rounded box with a thick green up arrow at its right, labelled in green italics "pull positive pair together". The sum over negatives in the denominator is in a pale-red rounded box with a thick dark-red down arrow below it, labelled in red italics "push negative pairs apart".

Below the equation, left:

$$\text{Symmetry: } \forall \mathbf{x}, \mathbf{x}^+, \mskip{5mu} p_{\mathrm{pos}}(\mathbf{x}, \mathbf{x}^+) = p_{\mathrm{pos}}(\mathbf{x}^+, \mathbf{x})$$

$$\text{Matching marginal: } \forall \mathbf{x}, \mskip{5mu} \int p_{\mathrm{pos}}(\mathbf{x}, \mathbf{x}^+) d\mathbf{x}^+ = p_{\mathtt{data}}(\mathbf{x})$$

(These two lines are a separate raster set in Computer Modern, where "pos" is roman and "data" typewriter; in the main equation both subscripts are italic. The five stray marks of the equation are real glyphs of its embedded STIX fonts, with no drawing behind them; their size and placement support the readings given: superscript ⊤, the italic temperature τ, "distributed as" ∼, a large Σ with N above and i = 1 below, and the arrow of $f : \mathcal{X} \rightarrow \mathbb{S}^{d-1}$.)

At the lower right, a figure: a grey gradient sphere (lighter at the top left), with two blue dots close together inside it near the bottom, just left of centre, and one red dot on the right of its lower half. Three photographs sit below it: a yellow Labrador-type dog standing in profile on grass (left), a second yellow Labrador standing on grass with its head turned to the right (middle) and a monkey sitting in grass with a hand to its mouth (right). Dotted curved arrows with arrowheads run from the left and middle dogs up to the two blue dots and from the monkey up to the red dot.

The notice sits at the lower left: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower left, beside the three dog and monkey photographs).*

## Slide 29 — Common setup

Title: "Common setup". Build step: as slide 28, without the Symmetry and Matching-marginal lines and without the dots or arrowheads on the sphere, and with a new bullet at the lower left.

- Encoder maps data onto a hypersphere: $f : \mathcal{X} \rightarrow \mathbb{S}^{d-1}$ (same garbled arrow as slide 28)
- Cross-entropy for softmax "classifier"

The same equation and green and red boxes, arrows and captions ("pull positive pair together", "push negative pairs apart") as slide 28 (same illegible marks "ˆ", "˜", "˙", "ˇ"). The sphere is plain grey with no dots, and three dotted curved lines (no arrowheads) run from it down to the same three photographs (two dogs and the monkey).

New bullet, in italics: "Noise-contrastive estimation (NCE) (Gutmann & Hyvärinen 2010), InfoNCE loss (van den Oord et al 2018), … similar losses also in metric learning".

The notice at the lower left repeats slide 28's: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower left, beside the three photographs).*

## Slide 30 — Common setup

Title: "Common setup". Build step: as slide 29, with a different lower-left text.

The same two bullets as slide 29 ("Encoder maps data onto a hypersphere: $f : \mathcal{X} \rightarrow \mathbb{S}^{d-1}$", "Cross-entropy for softmax "classifier""), the same equation, green and red boxes, arrows and captions, the same plain sphere with three dotted lines to the same three photographs. Lower-left text, no bullet: "As self-supervised learning, can outperform supervised pre-training (for some tasks)" and, in italics, "(He et al 2020, Misra & van der Maaten 2020)".

The same notice as slide 29: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower left, beside the three photographs).*

## Slide 31 — Why map to a hypersphere?

Title: "Why map to a hypersphere?".

- more stable training (logistic regression needs regularization)
- well-clustered classes on hypersphere are linearly separable (cut off caps)

Figure (Wang and Isola), lower centre. A grey gradient sphere, lighter at its upper left and sliced flat on its right, sits in the middle of a large circle outline that has a wedge-shaped sector cut out of its right side (two straight edges run from the sphere's flat face out to the circle at the upper and lower right). A tall pale-grey vertical slab (a plane seen in perspective, labelled with rotated two-line serif text "Linear / classifier") stands in that notch, to the right of the sphere, not through it. The ring around the sphere carries nine small photographs placed on it: three of dogs at the top (a dark dog's face, a standing tan dog, a small tan-and-white dog's face), three of cars down the left side (a grey sports car, a white saloon with red and blue stripes along its side, a small green city car) and three of aircraft at the bottom (a seaplane, a white airliner, an orange-and-white jet). Right of the slab is the cut-off wedge (two straight edges and an arc), holding the sliced cap (flat face to the left, dome bulging right) and three photographs of cats stacked vertically (a grey long-haired cat's face, a grey-tabby face, an orange-tabby face).

The notice sits at the lower right: "Image © Wang and Isola. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". Beneath it, in italics: "figure: Wang & Isola 2020".

*OCW notice: Image © Wang and Isola. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower right, beside the hypersphere figure with its animal, car and aircraft photographs).*

## Slide 32 — How can we make this "self-supervised"?

Title: "How can we make this "self-supervised"?". Build step: the loss figure of slides 28 to 30, moved up, with a new question at the lower left.

The same equation as slide 28, with the same pale-green box and green up arrow ("pull positive pair together"), the same pale-red box and dark-red down arrow ("push negative pairs apart"), the same illegible marks in place of the transpose ("ˆ"), the temperature ("˜"), the two "distributed as" signs in the subscript of $\mathbb{E}$ ("˙") and the summation sign ("ˇ"). Below it at the right, the same plain grey sphere with three dotted curved lines running down to three photographs (the two dogs and the monkey of slide 28), now smaller. Lower left, one bullet: "What are the similar (positive) and dissimilar (negative) pairs?".

The notice at the lower right: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower right, under the dog and monkey photographs).*

## Slide 33 — What are positive and negative examples?

Title: "What are positive and negative examples?".

At the upper left, in dark-red type, "Negative examples:" followed in black by "randomly uniformly drawn from data". To its right, a strip of eight photographs in a row: a person riding a white horse (a crowd behind); a white sports car seen from the side; the rear of a white car; a cargo ship at sunset against an orange sky; a dark ship on water; a pale-coated horse on grass; the deck of a dark vessel seen from above, with sea beyond; a red Coca-Cola delivery truck on a grey road. Some are letterboxed with black bars.

At the lower left, in green type, "Positive examples:" followed in black by "perturbations that keep semantic meaning, data augmentation". At the right, the augmentation figure of Chen et al.: ten photographs of the same pale-yellow dog standing on grass, in two rows of five, each captioned in serif type beneath it:

- Row 1: (a) Original (the full dog, facing left, tail curled up); (b) Crop and resize (a zoomed crop of its hindquarters and tail); (c) Crop, resize (and flip) (a zoomed crop of the head and shoulders, mirrored); (d) Color distort. (drop) (the dog in greyscale); (e) Color distort. (jitter) (the dog in a dark-green tint).
- Row 2: (f) Rotate {90°, 180°, 270°} (the dog rotated onto its side); (g) Cutout (the dog with a grey square covering its middle); (h) Gaussian noise; (i) Gaussian blur; (j) Sobel filtering (a grey relief-like edge image of the dog).

At the lower right, in green italics: "(Chen, Kornblith, Norouzi, Hinton 2020)".

The notice sits at the lower left: "Original image courtesy of Von.grzanka. Used under CC-BY. Manipulated images © Chen, et al. Other images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: Original image courtesy of Von.grzanka. Used under CC-BY. Manipulated images © Chen, et al. Other images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower left; the "Other images" are the strip of eight negative-example photographs).*

## Slide 34 — Positive and negative samples

Title: "Positive and negative samples". Text: "e.g. SimCLR:" then

- for each data point in the batch, generate 2 random augmentations as positive pair
- all other 2(B-1) augmented samples in the batch (of size B) are used as negatives

Lower right, a figure. At the left a photograph of a golden-coloured dog's head and shoulders on grass. Two curved black arrows leave it: the upper one to a copy of the dog tinted strong yellow-green, the lower one to a copy tinted strong red-pink; the blue italic text "positive pair" sits between the two copies. A curved black arrow labelled with an italic $f$ runs from each tinted copy to a grey gradient circle at the right; the upper arrow ends at a green dot near the top left of the circle's interior, the lower arrow at a second green dot just below it (two close green dots). At the lower right, a photograph of a monkey sitting in grass with a hand at its mouth, with the blue italic text "negative sample" to its left, and a curved black arrow labelled $f$ from it to a red dot in the lower right of the circle.

The notice sits at the lower left: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower left, beside the dog and monkey photographs).*

## Slide 35 — Variations

![Slide 35 — Variations](../images/12-representation-learning-similarity-based/slide-35.jpg)

Title: "Variations". A diagram on the left and text on the right.

Left: two trapezoids (narrow at the top, wide at the bottom), the left one filled pale green and labelled $f^x$, the right one filled pale blue and labelled $f^y$, each with a thick black bar just above its top edge and a photograph under its base. Under the left trapezoid, with a blue italic $x$ at its left and the label "Anchor" below, a rainbow-coloured image (red at the top, through orange and yellow to blue at the bottom, with dark-blue blobs at the bottom edge; a depth-map-like image). Under the right trapezoid, with a blue italic $y$ at its left and the label "Positive" below, a photograph of a classroom with desks and a whiteboard. From the top of each bar a blue dashed curved arrow runs up to a large dashed-outline grey gradient circle in the centre (the arrowheads point inwards, towards each other, at the circle's lower rim). At the upper left, a photograph of an office desk with a computer monitor, labelled "Negative" below, with a black dashed curved arrow running from it to the inside of the circle.

Right, in black with blue italic math: $(x, y)$ are two "views" of the same scene (the pair in blue). Below, "Cross-Channel" in orange followed by "Representation Learning" in black. Below, "[CMC, Tian, Krishnan, Isola 2020]" and a vertical ellipsis.

At the lower right of centre, under the citation: "Courtesy of Tian, et al. Used under CC BY-NC-SA." (not an exclusion notice).

## Slide 36 — Variations

Title: "Variations". Build step: as slide 35, with different photographs and text at the right. The circle is the same dashed one, but no arrowheads are visible: the circle's fill is painted over them, so all three dashed lines stop at its rim.

Left: the same diagram of the two trapezoids ($f^x$ in green, $f^y$ in blue), the two blue dashed lines to the central circle and the black dashed line from the negative photo, all ending at the rim with no arrowheads visible. Photographs: Negative (upper left, a blurred crop of a leg in blue jeans against a green background, with three dark vertical bars, at its left edge, right of centre and at its right edge), Anchor ($x$, a person in a dark jacket riding a BMX bicycle on grass), Positive ($y$, a person in dark clothing on a BMX bicycle on a paved or concrete-slatted surface).

Right: $(x, y)$ are two "views" of the same scene (set here in a sans-serif, not slide 35's serif). Below, "Video" in orange followed by "Representation Learning" in black. Then a list of references in a column:

- ["Slow Feature Learning", Wiskott & Sejnowski 2002]
- [Mobahi, Collobert, Weston 2009]
- [Wang & Gupta 2015]
- [Isola, Zoran, Krishnan, Adelson 2016]
- [Sermanet, Lynch, Chebotar et al. 2018]
- [van den Oord, Li, Vinyals 2018]

The notice sits at the lower right: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/" (a vertical ellipsis overprints both lines, at "source" and "license").

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower right, under the reference list; it covers the three bicycle photographs).*

## Slide 37 — Variations

Title: "Variations". Build step: as slides 35 and 36, with a text anchor and different right-hand text. As on slide 36, no arrowheads are visible: the blue lines end at the circle's rim, and the black dashed line from the negative runs a short way into the circle and stops.

Left: the same diagram. Negative (upper left): a photograph of two basketball players in mid-air, one in a white jersey with purple and gold trim, numbered 24 (white shorts with a gold stripe), shooting and one in a light-blue jersey numbered 10 reaching up to block. Anchor ($x$): not a photograph but two lines of brown text beside the $x$. The render shows them as garbled characters (printed `1,2"#,)*,/)3)#$,` and `4+/*&,)#,",3&*&/0`), but the garbling is a consistent one-to-one glyph substitution (the font numbers its glyphs in order of first use, and every repeated letter maps consistently, with lengths matching), and decoded it reads "A man is riding a" / "horse in a desert", a caption that matches the positive photograph. Positive ($y$): a photograph of a man in a checked shirt and hat riding a brown horse across a red-brown plain with hills beyond.

Right: after the blue $(x, y)$ the text is garbled in the render (printed `!"#$!%&'!()*$&+,!'-!%.$!+"/$!+0$1$`); decoded by the same consistent glyph substitution, it reads "are two "views" of the same scene", as on slides 35 and 36. Below, a line printed as `!"#$%"$&'()*)+# ,-&./&*&#0"0)+#,!&"/#)#$` decodes to "Language-Vision" in orange followed by " Representation Learning" in black (the joining character is a single glyph, most likely a hyphen). Then, legible: "[Karpathy, Joulin, Fei-Fei 2014]", a vertical ellipsis, and "[CLIP, Radford, Kim et al. 2021]".

The notice sits at the lower right: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower right; it covers the basketball and horse photographs).*

## Slide 38 — What is this method doing?

Title: "What is this method doing?". Text: "2 ingredients:" then

- Contrastive loss (which specific form)
- Data (which positive/negative pairs)

## Slide 39 — What is the contrastive loss doing?

Title: "What is the contrastive loss doing?". The loss of slide 28 is written again, this time with a name on the left: an equation in serif type whose left-hand side is a script letter (not rendered; a breve "˘", a different mark from the caron "ˇ" that stands for the summation sign, sits where the script $\mathcal{L}$ should be) with the italic subscript "cont", so $\mathcal{L}_ {cont}(f) =$ is the reading, followed by the same expectation, $-\log$ fraction and bracket as on slide 28, with the same illegible marks in the exponents, in the subscript of $\mathbb{E}$ and in the sum. In the fraction, the numerator $e^{f(\mathbf{x})^{\top} f(\mathbf{x}^+)/\tau}$ is in a pale-green box, the first denominator term (the same expression) is in a second pale-green box, and the sum over $N$ negatives is in a pale-red box.

- cross-entropy loss to distinguish data points
- maximizes a lower bound on mutual information between "views" $f(\mathbf{x}), f(\mathbf{x}^+)$ (*Poole et al, 2019*):

$$\text{MI}(f(\mathbf{x}), f(\mathbf{x}^+)) \geq \log(N) - \mathcal{L}(f)$$

(In the render the last term reads "˘( f )", the breve standing for the script $\mathcal{L}$. This line prints no "cont" subscript, though the left-hand side's $\mathcal{L}_ {cont}(f)$ suggests the same loss is meant. The boxed text is tinted dark green or dark red, the "+" overlaps the right end of the first green box, and the red box overprints the closing bracket.)

## Slide 40 — What (else) is the contrastive loss doing?

Title: "What (else) is the contrastive loss doing?". One bullet: "Recall: properties of "good" representations:" then a numbered list whose first words are in red:

1. Concentration/Alignment: Data from the same class is close together, remove irrelevant information
2. Separation: classes are well separated, do not lose information
3. Robustness to irrelevant perturbations

(The red words are "Concentration/Alignment", "Separation" and "Robustness".)

## Slide 41 — Alignment and separation

Title: "Alignment and separation". Two figures side by side, no body text. The slide is a pasted figure from Wang and Isola: everything but the notice, the title included, is a single low-resolution raster stretched over the page.

- Left: a grey gradient sphere (lighter at the upper left) above two unfilled outline trapezoids (narrow at the top, wide at the bottom), side by side, with no arrows drawn to the sphere. Under the left trapezoid, a small photograph of an aircraft with a pink-and-white fuselage and a pink tail on a red-brown apron under a green-grey sky, with an italic $x$ beneath it. Under the right trapezoid, a similar photograph of a pale airliner on a runway under a pale sky, with no label. A tiny dark speck appears at the top of each photograph.
- Right: a larger circular ring outline (thin black circle) with a grey gradient sphere in its centre, the ring carrying twelve small photographs evenly around it: three dogs along the top (a dark dog, a standing tan dog, a small white-and-tan dog's face), three animals down the right (a grey long-haired cat's face, a grey tabby on blue, an orange tabby), three cars on the left (a grey sports car, a white hatchback, a small green city car) and three aircraft along the bottom (a seaplane, a white airliner, an orange-and-white jet). The words "Feature Density" in serif type sit at the ring's bottom edge. Beneath the ring, in serif type, "**Uniformity:** Preserve maximal information" (the word "Uniformity:" in bold).

The notice sits at the lower right: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower right, under the ring of photographs).*

## Slide 42 — Feature distribution from Contrastive Learning

Title: "Feature distribution from Contrastive Learning". Text, in sans-serif: "**Toy example:**" and then "Train `CIFAR-10` encoders with $\mathcal{S}^1$ feature space (circle). Visualize feature distributions on the validation set." (the dataset name in typewriter type; the printed symbol is a script S with superscript 1).

Three heat-map panels in a row, each titled "Feature Distribution", each square with ticks 1.0, 0.5, 0.0, −0.5, −1.0 on the vertical axis and −1, 0, 1 on the horizontal axis. Colours run from white (no density) through pink and tan to green and dark teal or black (highest density). Captions beneath.

- Left, "Unsupervised Contrastive Learning": a nearly uniform ring of density around the unit circle, tan-green with a pink outer glow and slightly darker green stretches at the upper left, the lower right and the bottom; no gaps.
- Middle, "Supervised Predictive (NLL) Learning": a ring that is clearly uneven. A dense dark-teal arc runs from the left (about 9 o'clock) up to the top (about 12 o'clock); a small dense blob sits at the upper right (about 1 o'clock); a fainter tan blob at the right (about 3 o'clock); two dense blobs at the bottom (about 6 and 5 o'clock); between the 1 o'clock blob and the right blob, and in the lower right, the ring fades almost to nothing, and there is a clear white gap at the top between the dark arc's end (just left of 12 o'clock) and the 1 o'clock blob.
- Right, "Random Network Initialization": almost empty; only a thin black arc at the bottom, a shallow "U" running from about (−0.5, −0.9) through (0, −1.0) to about (0.75, −0.67), with faint pink tails reaching about (−0.65, −0.75) on the left and (0.95, −0.3) on the right.

Everything on the slide but the notice, the title included, is one raster image. The notice sits at the lower right: "Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: Images © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower right, under the three heat maps).*

## Slide 43 — Relation Between Representation Quality and Alignment & Uniformity

![Slide 43 — Relation Between Representation Quality and Alignment & Uniformity](../images/12-representation-learning-similarity-based/slide-43.jpg)

Title: "Relation Between Representation Quality and Alignment & Uniformity". Two scatter plots side by side, from Wang and Isola. Each has a legend (three marker types) and a colour bar for validation accuracy (red = low, grey-white = middle, blue = high).

Legend in each: "+" for $\mathcal{L}_ {\text{contrastive}}$ only; a filled circle for $\mathcal{L}_ {\text{align}}, \mathcal{L}_ {\text{uniform}}$ only; a triangle for "All three mixed". So there are three marker series in each plot, each point coloured by validation accuracy.

- Left plot, titled "Linear Classification on Outputs"; label beneath: "306 `STL-10` Encoders". Horizontal axis $\mathcal{L}_ {\text{uniform}}(t = 2)$, ticks −4 to 0; vertical axis $\mathcal{L}_ {\text{align}}(\alpha = 2)$, ticks 0.00 to 2.00 in steps of 0.25; colour bar "Val Accuracy" from 50 (dark red) to 85 (dark blue), ticks 50, 55, … 85. Counted from the vector markers: 211 circles, 55 "+" and 40 triangles, 306 in all, matching the label. The points lie along a hyperbola-like curve: a vertical stack of red-to-pink circles at the far left (x ≈ −3.9) from y ≈ 2.0 down to 0.6; a dense mass of blue markers of all three kinds at the lower left (x from −3.9 to −2.95, y from 0.55 down to 0.19), holding 132 circles, 43 of the 55 "+" (37 of them blue) and all 40 triangles (all blue), with one marker ringed in black near (−3.83, 0.35); then a sparse tail running to the right along y ≈ 0.1 to 0.3, of salmon and pale-pink circles with dark-red circles near (−2.4, 0.2), (−2.06, 0.2) and (−1.56, 0.25–0.29) and a dozen "+" (six dark red, five pale, one salmon), ending at (0, 0), where 25 dark-red circles are stacked on one spot. A few isolated red circles lie off the curve (near (−2.95, 1.0), (−2.85, 0.8), (−1.15, 0.95), (−2.2, 0.5)).
- Right plot, titled "Customer Review Classification on Outputs"; label beneath: "108 `BookCorpus` Encoders". Same axes. Colour bar "Val Accuracy" from 72 (dark red) to 80 (dark blue), ticks 72 to 80. Counted from the vector markers: 67 circles, 19 "+" and 22 triangles, 108 in all, matching the label. The circles fall along a descending curve from the upper left (a cluster of dark-red circles at (−3.9, 2.0)) through blue and light-blue points around (−3.8 to −3.2, 1.6 to 1.1), then pale orange and pink points running down to the right (near (−2.5, 0.8), (−1.9, 0.6), (−1.4, 0.4)), ending in a dark-red circle at (0, 0). Seventeen of the triangles are in the upper-left cluster (dark red at y ≈ 2.0, blue from y ≈ 1.8 down to 1.46); the others are one blue at (−3.3, 1.17), one pale blue at (−1.55, 0.5), one salmon-red at (−0.94, 0.27) and two under the circle at (0, 0). A second, much flatter row of "+" markers runs above the circles' curve from about (−3.1, 1.12) to (−1.6, 0.94), salmon, pale and red (one dark red at (−2.16, 1.0)), and four blue "+" sit well above the curve, at about (−3.43, 1.66), (−2.64, 1.80), (−1.22, 1.68) and (−0.94, 1.58). The legend box lies in the lower left of the plot area.

At the lower right, in italics: "figures: Wang & Isola, 2020". (Values are read from marker positions and are approximate.)

## Slide 44 — What is the contrastive loss doing?

Title: "What is the contrastive loss doing?".

- Loss function encourages:
  1. Concentration/Alignment: Data from the same class is close together, remove irrelevant information
  2. Separation: classes are well separated, do not lose information
- What do the selection of positive and negative pairs encourage?

(The words "Concentration/Alignment" and "Separation" are in red.)

## Slide 45 — What are we "teaching" the model via choice of pairs?

Title: "What are we "teaching" the model via choice of pairs?".

- positive pairs = augmentations of the same data point should be close
- => learned representation is invariant to perturbations induced by data augmentations: learned invariance (the words "learned invariance" in red)
- Finding the "right" invariances can be challenging for different types of data
- Learned versus hard-coded invariances (geometric DL lecture): when would we use which?

## Slide 46 — What is the contrastive loss doing?

Title: "What is the contrastive loss doing?". Build step: as slide 44, with the closing question replaced by a third item.

- Loss function encourages:
  1. Concentration/Alignment: Data from the same class is close together, remove irrelevant information
  2. Separation: classes are well separated, do not lose information
- Data encourages:
  3. Robustness to **irrelevant** perturbations

("Concentration/Alignment", "Separation" and "Robustness" are in red; "irrelevant" is in bold.)

## Slide 47 — Ingredients to make self-supervised CL work (better)

Title: "Ingredients to make self-supervised CL work (better)".

- heavy data augmentation
- projection heads
- large batch size (many negative examples)
- choice of data pairs / hard negative examples

A large right curly brace spans the four bullets, labelled "SimCLR model".

## Slide 48 — Effect of data augmentation

Title: "Effect of data augmentation". Left, a line chart. Right, the augmentation grid.

Chart: vertical axis "Decrease of accuracy from baseline", ticks 0, −5, −10, −15, −20, −25 (axis extends to about −28); horizontal axis "Transformations set", with five categories: "Baseline", "Remove grayscale", "Remove color", "Crop + blur only", "Crop only". Two series, each a line with a large round marker at every category, with a legend box at the upper right:

- BYOL (red line and markers): Baseline 0; Remove grayscale about −2; Remove color about −9; Crop + blur only about −11.5; Crop only about −13.
- SimCLR (repro) (blue line and markers): Baseline 0; Remove grayscale about −6; Remove color about −22; Crop + blur only about −26; Crop only about −27.5.

Both start at the same point at 0 on "Baseline" (only the blue marker shows there; the red one is under it). Caption beneath in serif type: "Impact of progressively removing transformations", and beneath it in italics "(figure: Grill et al 2020)". (Values are read from marker positions and are approximate.)

Right: the ten-panel dog-augmentation figure of slide 33 at a smaller size, in two rows of five, with the same captions in tiny type: (a) Original, (b) Crop and resize, (c) Crop, resize (and flip), (d) Color distort. (drop), (e) Color distort. (jitter); (f) Rotate {90°, 180°, 270°}, (g) Cutout, (h) Gaussian noise, (i) Gaussian blur, (j) Sobel filtering. Here the rotated dog (f) is turned 90° clockwise, legs pointing left and head at the upper right.

The notice at the lower left, as printed: "Original dog image courtesy of Von.grzanka. Used under CC-" / "BY. Manipulated images © Chen, et al." / "content is excluded from our Creative Commons license. For" / "more information, see" / "https://ocw.mit.edu/help/faq-fair-use/", with a stray full stop at the far right of the second line. Nothing is cut off: the words between "Chen, et al." and "content is excluded" (presumably "All rights reserved. This"; the slide carries no "other images") are simply not printed.

*OCW notice: Original dog image courtesy of Von.grzanka. Used under CC-BY. Manipulated images © Chen, et al. … content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower left, under the chart; the words between "Chen, et al." and "content" are not printed).*

## Slide 49 — Projection head

![Slide 49 — Projection head](../images/12-representation-learning-similarity-based/slide-49.jpg)

Title: "Projection head".

- contrastive loss is applied to a transformed version $g(\mathbf{h})$ of the representation $\mathbf{h}$
- $g$ is linear or small MLP
- use $\mathbf{h}$ for downstream task
- Projection head improves performance!

Right, a diagram (the SimCLR framework figure). At the bottom centre a circle containing a bold italic $\boldsymbol{x}$. Two arrows leave it, one up-left to a circle containing $\tilde{\boldsymbol{x}}_ i$ (labelled along the arrow $t \sim \mathcal{T}$), and one up-right to a circle containing $\tilde{\boldsymbol{x}}_ j$ (labelled $t' \sim \mathcal{T}$). From each of those circles an upward arrow labelled $f(\cdot)$ goes to $\boldsymbol{h}_ i$ (left) and $\boldsymbol{h}_ j$ (right). Between $\boldsymbol{h}_ i$ and $\boldsymbol{h}_ j$ is the word "Representation", flanked by two outward-pointing arrows ("⟵ Representation ⟶"). Above, a light-blue horizontal band spans the width and holds, at each end, an upward arrow labelled $g(\cdot)$ from $\boldsymbol{h}$ to $\boldsymbol{z}_ i$ (left) and $\boldsymbol{z}_ j$ (right). Across the top, a double-headed arrow between $\boldsymbol{z}_ i$ and $\boldsymbol{z}_ j$ is labelled "Maximize agreement".

## Slide 50 — Projection head

![Slide 50 — Projection head](../images/12-representation-learning-similarity-based/slide-50.jpg)

Title: "Projection head". Build step: slide 49's diagram, shrunk into the upper right, plus a bar chart below it.

- Projection head improves performance.
- Why?
  Possibly because representation $\mathbf{h}$ then need not be completely invariant to augmentations, can retain some information

Upper right: the same SimCLR diagram as slide 49 at smaller size. Below it, a bar chart: vertical axis "Top 1", ticks 30, 40, 50, 60, 70 (starting at 30); horizontal axis "Projection output dimensionality", seven groups labelled (rotated) 32, 64, 128, 256, 512, 1024, 2048. Legend "Projection" (semi-transparent box overlying the first three groups): Linear (blue), Non-linear (gold), None (green). Three series:

- Linear (blue): about 60 at 32, 61 at 64, 61 at 128, then about 60.5 at 256, 512, 1024 and 2048.
- Non-linear (gold): about 63.5 at 32, 64 at 64, 64 at 128, 64.5 at 256, 64.5 at 512, 64.5 at 1024, 64.5 at 2048.
- None (green): a single bar, at 2048 only, about 50.

(Values are read from bar heights and are approximate.)

## Slide 51 — Effect of batch size

![Slide 51 — Effect of batch size](../images/12-representation-learning-similarity-based/slide-51.jpg)

Title: "Effect of batch size". Left, a grouped bar chart (from Chen et al.); right, three bullets.

Chart: vertical axis "Top 1", ticks 50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0 (starts at 50.0); horizontal axis "Training epochs", ten groups labelled 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000. Each group has six bars, one per batch size, always in the same left-to-right order, with a legend titled "Batch size" at the lower right (overlying the 900 and 1000 groups):

- 256 (blue)
- 512 (gold)
- 1024 (green)
- 2048 (dark orange)
- 4096 (pink-mauve)
- 8192 (tan)

Bars rise with epochs, and with batch size up to about 1024 or 2048; from about 400 epochs the three largest sizes level off, and 2048 is the tallest at 600 and 1000 epochs. The gaps between batch sizes shrink as epochs grow. Read approximately from bar tops: at 100 epochs the six bars are about 57.5, 60.7, 62.8, 64, 64.4 and 64.7; at 500 epochs about 65.2, 66.7, 67.9, 68.1, 68.2 and 68.2; at 1000 epochs about 67, 68, 68.8, 69.3, 69.1 and 68.9. At every epoch count the 256 bar is lowest. Caption in serif type: "Figure 9. Linear evaluation models (ResNet-50) trained with different batch size and epochs. Each bar is a single run from scratch." with a small superscript "10". Beneath, in italics: "(Figure from Chen et al. 2020)".

Right, bullets:

- SimCLR uses all points in a batch as negative examples for a positive pair
- needs large number of negative pairs = large batch sizes
- Expensive. Newer methods make this more efficient (like MoCo, *He et al. 2020*)

## Slide 52 — Improving negative samples

Title: "Improving negative samples". One bullet: "We are pushing apart negative pairs. Negative pairs are random pairs from the data."

Left, a diagram. In a rounded grey box at the left, a photograph of a golden-coloured dog on grass with sky behind it, with italic $x$ beneath. Two thin arrows leave the box. The upper arrow leads to a rounded grey box holding an italic $x^+$ and a photograph of the head of a golden dog with its mouth open, outdoors, followed by $\sim p_x^+$. The lower arrow leads to a longer rounded grey box holding $x_i^-$ and three photographs in a row: a grey tabby cat on green, a passenger aircraft in a blue sky, and (in a red frame) an orange dog seated and looking up; then $\sim p$. In red serif type, directly above the red-framed dog at the right end of the lower box, the two-line label "false negative" / "sample" (no arrow).

Right, a line chart (from Chuang et al.): vertical axis "Top-1 Accuracy", ticks 80, 85, 90, 95; horizontal axis "Negative Sample Size (N)", five points at 30, 62, 126, 254, 510 (the labels "30" and "62" print close together as "3062"). Two series, each a line with small circle markers, legend at the lower right in the order Biased, then Unbiased:

- Unbiased (blue line), labelled in blue bold "using only true negatives": about 88.3 at 30, 92.3 at 62, 93.2 at 126, 94.0 at 254, 94.3 at 510.
- Biased (green line), labelled in green bold "random pairs": about 80.4 at 30, 84.7 at 62, 87.6 at 126, 89.8 at 254, 91.2 at 510.

(Values read from marker positions, approximate.)

The notice sits at the lower right: "© Chuang, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". Beneath it, in italics: "figure: Chuang et al, Debiased contrastive learning".

*OCW notice: © Chuang, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the lower right, under the chart and the diagram with its photographs).*

## Slide 53 — Supervised or semi-supervised contrastive learning

Title: "Supervised or semi-supervised contrastive learning".

- Contrastive learning provides more geometric and robustness feedback than cross-entropy loss
- Idea: in addition to data augmentation, use images from same class as positive pairs (multiple positive pairs)

Below, two diagrams side by side, separated by a thick black vertical line. In each, a grey gradient circle with a dotted outline in the middle, with short lines from points on its rim to photographs: a dark grey line to the anchor, orange lines to the positives, red lines to the negatives. Serif labels "Anchor", "Positive(s)" and "Negatives".

- Left, captioned "Self Supervised Contrastive": the anchor, a fluffy cream-and-orange pomeranian puppy photograph in a black frame at the upper left; one positive (labelled "Positive"), a second, smaller crop of the same puppy, below it, with one orange line; and, under "Negatives" at the upper right, three photographs stacked vertically (a baby elephant in grass, a kitten's face, a kitten yawning) each with a red line, and a fourth, a black, white and tan spaniel lying on a couch in a red frame, below the circle at the bottom centre of the panel, also with a red line. Four red lines in all.
- Right, captioned "Supervised Contrastive": the same anchor, the same three negatives with three red lines, and under "Positives" two photographs, the small puppy crop and, in a green frame, the same spaniel photo, each with an orange line (two orange lines). The spaniel, a negative on the left, is a positive here.

At the lower right, "Khosla et al 2020" in italics. The notice sits at the bottom, under the right-hand diagram: "© Khosla, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/" (the printed slide number "53" overprints "license." in its second line).

*OCW notice: © Khosla, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the bottom centre, beneath the two diagrams and their photographs).*

## Slide 54 — Case study: iNaturalist 2021

Title: "Case study: iNaturalist 2021" (bold sans-serif). At the upper right, four bullets in large sans-serif type:

- 10,000 Species
- 2.7M Training Images
- 50k Validation Images
- 500k Test Images

Under the title at the left, the notice: "© source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

The lower two thirds of the slide is a full-width mosaic of photographs from the iNaturalist dataset, in five rows of varying-width tiles: 13, 11, 11, 9 and 13 visible tiles (57 in all, counted from the tile borders; the end tiles are cut by the slide's edges, as the image overhangs the left, right and bottom). They are mostly organisms in the wild, with dark borders between tiles. Row 1 includes a red-and-black spider on rock, a green caterpillar on a leaf, a squirrel on bark, an orange fish, a toad, two lizards, an owl and a jellyfish on black. Row 2 includes fungus, a mountain landscape with red flowers, an orange mushroom, a dark fish or frog face, a smiling man in a teal jacket (a selfie-like portrait), tall grey mushrooms, a squirrel on a post and a black-and-white bird of prey on a branch. Row 3 includes a fingertip with a tiny black beetle, a curled brown leaf, a bird held in hands with spread red-and-brown wings, a green insect, a long-legged bird on a road and a pale brown beetle-like insect. Row 4 includes brown birds on wet ground, a snake coiled on leaves, a pale bird in bare branches, a black-and-white striped snake, a swallowtail butterfly on flowers and a brown lizard on red rock. Row 5 includes a rocky mountainside, a dark-background orange crustacean with blue legs, a white worm-like animal, white flowers, a wading bird in reeds, an orchid, a lizard on grey rock and ocean waves with dolphins. (Row contents are approximate descriptions of the larger tiles.)

*OCW notice: © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits under the title, above the photograph mosaic).*

## Slide 55 — (no title; radial taxonomy tree)

No title is printed. A large circular tree diagram fills the slide's centre: a radial hierarchy with a single red dot near the centre (the root), thin black lines branching outwards in many levels, and small blue dots at the nodes, so that very many blue leaf dots form a dense outer ring. Inner rings of blue nodes show the intermediate levels. There is no text on the tree itself.

At the upper right, the notice: "Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". At the lower left: "Cole et al., *When Does Contrastive Visual Representation Learning Work?*, CVPR 2022" (the title in italics). At the lower right, in italics: "Slide: Elijah Cole".

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the radial tree).*

## Slide 56 — (no title; radial taxonomy tree, "Coarse-grained")

No title is printed. Build step: as slide 55, plus a gold rectangle with white sans-serif text "Coarse-grained" just above the red root dot, centred on it. Same tree, same credits and the same notice, "Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/" (at the upper right), and "Slide: Elijah Cole" at the lower right (the "Cole et al." citation of slide 55 is absent).

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the radial tree).*

## Slide 57 — (no title; radial taxonomy tree, coarse to fine)

No title is printed. Build step: as slide 56, plus a thick bright-blue arrow from the red root dot down and to the right, out to the outer ring at the lower right, and a gold rectangle labelled "Fine-grained" at the lower right, beyond the arrow's head. Same notice and "Slide: Elijah Cole" credit.

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the radial tree).*

## Slide 58 — (no title; radial taxonomy tree with two moths)

No title is printed. Build step: as slide 57, plus two photographs at the right with Latin names above them. Upper: "S. umbilicata", a white moth with pale-blue shading and orange-brown, dark-edged patches along its wing margins, on a green leaf (the same photograph as slide 17's upper-left one). Lower: "S. ornata", a pale cream moth with fine wavy lines among green leaves (the same photograph as slide 17's upper-right one).

The notice here is worded differently, and prints "Duagran" where "Diagram" is meant: "Slide © Elijah Cole. Duagran © Cole, et al. Other images © source unknown. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". Credit "Slide: Elijah Cole" at the lower right.

*OCW notice: Slide © Elijah Cole. Duagran © Cole, et al. Other images © source unknown. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, above the two moth photographs).*

## Slide 59 — (no title; radial tree over a hierarchy-depth chart)

No title is printed. A chart frame with a smaller copy of the radial tree (with its blue arrow from root to rim, in the same direction) placed over its plot area. Vertical axis "Top-1 Accuracy", ticks 0.5, 0.6, 0.7, 0.8, 0.9, 1.0; horizontal axis "Label Hierarchy Depth" with seven categories (rotated labels, with the number of labels at each level in brackets): "Kingdom (3)", "Phylum (13)", "Class (51)", "Order (273)", "Family (1103)", "Genus (4884)", "Species (10000)". No data series is visible: the chart beneath is slide 60's whole chart, as a raster, with its plot area, title and legend masked by a white panel and the tree's white background (only two faint specks of the Kingdom and Phylum gridlines peek out above the mask). At the bottom of the plot area, a gold rectangle "Coarse-grained" at the left, a long thick blue horizontal arrow pointing right, and a gold rectangle "Fine-grained" at the right.

The notice at the upper right: "Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". Credit "Slide: Elijah Cole" at the lower right.

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the tree and chart).*

## Slide 60 — iNat21

Title (printed at the top centre of the chart): "iNat21". A line chart; the axes are those of slide 59: vertical axis "Top-1 Accuracy", ticks 0.5 to 1.0; horizontal axis "Label Hierarchy Depth", categories "Kingdom (3)", "Phylum (13)", "Class (51)", "Order (273)", "Family (1103)", "Genus (4884)", "Species (10000)". Three series, each with large round markers joined by lines; legend at the lower left:

- Supervised (black): about 0.99 at Kingdom, 0.98 at Phylum, 0.97 at Class, 0.93 at Order, 0.90 at Family, 0.86 at Genus, 0.80 at Species. A smooth gentle decline.
- iNat21 SimCLR (salmon-orange): about 0.975 at Kingdom, 0.96 at Phylum, 0.925 at Class, then 0.79 at Order, 0.69 at Family, 0.59 at Genus, 0.50 at Species.
- iNat21 MoCo (tan): about 0.97 at Kingdom, 0.955 at Phylum, 0.92 at Class, then 0.79 at Order, 0.695 at Family, 0.59 at Genus, 0.505 at Species. From Order on, the SimCLR and MoCo lines overlap nearly exactly.

(Values read from marker positions, approximate. The two self-supervised curves match supervised closely down to Class, then fall away steeply.)

The notice at the upper right: "Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". Credit "Slide: Elijah Cole" at the lower right.

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the chart).*

## Slide 61 — iNat21

Title (printed at the top centre of the chart): "iNat21". Build step: as slide 60, plus a gold callout at the upper right, "Supervised training from scratch (full dataset).", with a thick black arrow from it pointing down-left at the black Supervised line (its head ends near the Genus to Species segment, to the right of the Species point).

The chart, legend, series and notice are those of slide 60: axes "Top-1 Accuracy" (0.5 to 1.0) and "Label Hierarchy Depth" (Kingdom (3), Phylum (13), Class (51), Order (273), Family (1103), Genus (4884), Species (10000)); series Supervised (black), iNat21 SimCLR (salmon) and iNat21 MoCo (tan); notice "Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/" at the upper right; "Slide: Elijah Cole" at the lower right.

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the chart).*

## Slide 62 — iNat21

Title (printed above the chart): "iNat21". Build step: as slide 61, plus a second gold callout, "Self-supervised with linear probe (full dataset).", at the right of the middle of the chart, with a thick black arrow from it pointing down-left at the SimCLR and MoCo lines (its head ends to the right of the Genus to Species segment of those lines). Same chart, series, notice and credit as slide 60.

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the chart).*

## Slide 63 — iNat21

Title (printed above the chart): "iNat21". Build step: as slide 62, plus a third gold callout in the middle left of the plot, between the Class and Order points at about 0.7 on the vertical axis: "Train on finest level, eval at coarser levels". Same chart, both earlier callouts and arrows, notice and credit.

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the chart).*

## Slide 64 — iNat21

Title (printed above the chart): "iNat21". Same chart, series, legend, notice and credit as slide 60. The callouts of slides 61 to 63 are gone, replaced by two new marks: at the left of the plot, a gold box reading "Not a small difference:" then "iNat21 ~30% gap" then "Imagenet ~7% gap"; and, at the Species position at the right, a thick red double-headed vertical arrow running from just below the black Supervised point (about 0.80) down to just above the SimCLR and MoCo points (about 0.50), with a gold label "Big gap!" over the middle of the arrow.

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the chart).*

## Slide 65 — iNat21

Title (printed above the chart): "iNat21". Same chart, series, legend, notice and credit as slide 60. New marks: a thick red rectangular outline around the Kingdom, Phylum and Class points of all three series (the top left of the plot, from about 0.90 to 1.0 on the vertical axis), and below it a gold box reading "Gap is small for coarse groupings".

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the chart).*

## Slide 66 — iNat21

Title (printed above the chart): "iNat21". Same chart, series, legend, notice and credit as slide 60. New marks: two thick red arrows both starting near the Class point and pointing right and down, one running just below the Supervised (black) line and ending below the Species point at about 0.77, the other running much more steeply, alongside the SimCLR and MoCo lines, and ending at about 0.54, just above the SimCLR and MoCo Species points; and, at the right, a gold box reading "Gap grows as" then "evaluation is made" then "more fine-grained".

*OCW notice: Slide © Elijah Cole. Image © Cole, et al. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, beside the chart).*

## Slide 67 — (no title; same-species and different-species retrievals, supervised on iNat21)

No title is printed. At the left, in large sans-serif type, "Supervised on iNat21". Below it, a row of eleven square photographs of small olive-green birds (the first, at the left, is a bird held in a person's hand, seen in profile with a striped head; the others show similar small olive or brown birds in hands, on branches or on the ground). The first photograph (the query) has only a thin black outline, no coloured frame; the other ten each have a coloured frame: three blue frames (the second, fourth and eleventh photographs) and seven red frames (the third and the fifth to tenth). A gold box labelled "Different species" sits above the row at centre-right, with seven thick black arrows fanning out from it to the seven red-framed photographs. A gold box labelled "Same species" sits below the row at centre-left, with three thick black arrows fanning out from it to the three blue-framed photographs.

The notice sits at the upper right: "Slide © Elijah Cole. Image © source unknown. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". Credit "Slide: Elijah Cole" at the lower right.

*OCW notice: Slide © Elijah Cole. Image © source unknown. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, above the row of bird photographs).*

## Slide 68 — (no title; supervised and SimCLR retrievals on iNat21)

No title is printed. Build step: slide 67's row, kept at the top with the same label, with a second row added. At the upper right, a large gold box with two lines of white text separated by a blank line: "On ImageNet, contrastive SSL matches supervised." then "On iNat21, contrastive SSL lags far behind".

- Top row, labelled "Supervised on iNat21": the same eleven photographs as slide 67, query first (a thin black outline only), then blue frames on the second, fourth and eleventh photographs and red frames on the rest. (The "Same species" and "Different species" boxes and arrows are gone.)
- Bottom row, labelled "SimCLR on iNat21": eleven photographs, the first being the same query photograph (a thin black outline only), followed by ten photographs all in red frames (the ninth is the same photograph as the top row's third): small birds held in hands (a grey flycatcher-like bird, a white-and-grey bird, a kingfisher with a blue back and orange front, a black-capped chickadee-like bird, a yellow-and-black bird, a bird held up to a person's face, a bird with a yellow throat on a hillside, and others). No blue frames appear in this row.

The notice at the upper right: "Slide © Elijah Cole. Image © source unknown. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". At the lower left, "Cole et al., *When Does Contrastive Visual Representation Learning Work?*, CVPR 2022" (the printed slide number "68" overprints its end), and "Slide: Elijah Cole" at the lower right.

*OCW notice: Slide © Elijah Cole. Image © source unknown. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (sits at the upper right, above the two rows of bird photographs).*

## Slide 69 — Summary

Title: "Summary".

- Good representations capture relevant similarity/dissimilarity information
- well-clustered, compact and separated/spread out classes:
  - preserves relevant information
  - teaches relevant invariances ("forget" irrelevant information)
- supervised or self-supervised

## Slide 70 — OCW end page

A smaller (792×612) page, not lecture content. Text: "MIT OpenCourseWare", "https://ocw.mit.edu", "6.7960 Deep Learning", "Fall 2024", and "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms". At the bottom centre it prints "70" (a small "70" overprints a larger one).
