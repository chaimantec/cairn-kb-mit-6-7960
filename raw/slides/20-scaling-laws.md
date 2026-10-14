---
title: Lecture 20 — Scaling Laws (slide deck)
lecture: 20
slides: 52
source_pdf: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec20.pdf
note: Printed slide numbers 1–51 (bottom centre) equal the PDF page numbers exactly. Page 52 is OCW's appended end page (a smaller 792×612 page), which also prints 52, although slide_number_map.py reports it as printing no number.
figure_audit: Transcribed by Sonnet from page images, in two halves; 43 chart-, diagram-, table-, equation- and photo-heavy pages (1–3, 5–20, 23–30, 32–35, 37, 39–45 and 47–50) were then checked by Opus, a different model, split between four agents (all finished), from 60–600 dpi crops, the embedded rasters at native resolution, the vector data and the text layer. No equation was wrong. 22 pages were corrected and 10 given precisions only; 11 were right as written. The corrections were positions and counts in charts (slide 1's envelope; slide 5's dots and arrow; slide 11's first dot; slide 17's dot range; slide 28's stars against the Kaplan line; slide 35's GPT arrow; slide 40's ring of 12 dots and its borders; slide 49's grid lines and arrowheads), one value off by a factor of ten (slide 45's yellow curve), stacking and placement (slide 24's gopher over the dinosaur; slide 44's label; slide 45's arrow; slide 41's chained SGD steps; slide 47's elephant behind the orange), misread type (slide 43's typed note called handwritten; slide 48's typeface), the crossing dashed lines of slides 26 and 29, slide 37's chart not identified as slide 10's, slide 23's hidden Chinchilla row, and printed slips left unflagged (the citation's "of" on slides 10–14, 18 and 19; slide 16's exponent; slide 42's sign; the caption of slides 26 and 29). The renders of slides 13 and 42 were then checked against the corrected text.
---

# Lecture 20 — Scaling Laws: slide-by-slide

Text and figures of all 52 pages of
[`mit6_7960_f24_lec20.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec20.pdf),
transcribed from the deck (speaker: Phillip Isola; the title slide reads "Lecture 20: Scaling Laws", matching the recording).
Cite these as "slide N": the printed number equals the PDF page number for slides 1–51; page 52 is OCW's appended end page and
prints 52. Diagrams, plots and pictures are described in prose since the KB is read as text. The equations are partly typed math
whose text layer loses its structure and partly pasted images; all were read from the page, never from the text layer.

**The recording covers only slides 1–17.** OCW's video and its transcript both end at ≈38:01, mid-sentence, as the lecturer
recaps "Finding two" after slide 17. Slides 18–51 (model shape, sample efficiency, Chinchilla, the data-manifold hypothesis,
breaking power-law scaling, the critical batch size and representational convergence) are known here only from the deck, with no
spoken commentary.

**Images.** Twenty slides have a rendered image under their heading: 2, 5, 6, 12, 13, 14, 16, 17, 18, 19, 20, 26, 28, 33, 35, 41, 42, 44, 49 and 50. The six slides with an OCW notice (3, 24, 40, 45, 47 and 48) are never rendered; no other slide reuses an excluded image. The others were not rendered because they are text, equations or tables that this file reproduces (4, 7, 8, 9, 15, 21, 22, 23, 25, 27, 30, 32, 34, 39, 43 and 51), charts a rendered slide already shows (10 and 11, both in 12; 29, which is 26's figure; 37, which is 10's chart), section dividers (31, 36, 38 and 46), the title (1) or the end page (52).

Companion pages: [wiki page for this lecture](../../wiki/20-scaling-laws.md) · [transcript](../transcripts/20-scaling-laws.md)

**Signposting slides you can skip.** Slide 1 is the title (with two of the Kaplan et al. charts that slides 12 and 14 show); slides
31, 36, 38 and 46 are section dividers ("Can we explain scaling laws? One hypothesis", "Breaking power law scaling", "Critical batch
size", "Representational convergence"); slide 52 is the OCW end page. The deck prints no lecture number but its own.

Some slides are **build steps** or repeats, transcribed individually with a note of what they add: 13 to 15 (the joint law in $N$
and $D$, then in $N$ and $S$, then the compute allocation); slide 11 repeats slide 10's chart beside a new one, slide 12 both of
them beside a third, and slide 37 slide 10's chart again; slides 26 and 29 show the same Chinchilla figure. Slides with no printed
title are headed here with a description: 2, 4, 5, 6, 24, 40, 41, 44, 45, 48 and 52.

**Printed slips and oddities**, kept as printed and flagged in each entry: the citation of Kaplan, McCandlish et al. on slides 10–14, 18 and 19 reads "Scaling Laws of Neural Language Models", while the paper's own title block on slide 8 reads "Scaling Laws for Neural Language Models"; slide 8's "shape (depth, with, attention heads, …)", probably width; slide 9's missing closing parenthesis; slide 12's compute constant, $2.3 \cdot 10^{8}$ in the chart legend and $3.1 \times 10^{8}$ in the equation below it; slide 13's legend order (3M before 25M); slide 16's fit function, whose second term has a minus sign in a denominator's exponent; slide 24's "Wikipedia Commons" (the repository is called Wikimedia Commons, which is outside the course material); slide 25's "4× more more data"; the caption of slides 26 and 29, whose colours and thirds do not match the plotted lines; slide 37, which shows slide 10's Kaplan et al. chart while citing only Sorscher et al.; slide 39's "Which is yields"; slide 41's "as a function given"; slide 42's label $\nabla\theta - \nabla_{\text{sgd}}\theta$, the opposite sign of its arrow and of the equation's noise term; and slide 43's "Plug the equation learning rate back into the equation". Some pages hide things: slides 23 and 25 crop their pasted images (slide 23's hidden row is Chinchilla's, 70 Billion parameters and 1.4 Trillion tokens); slide 33's 3D picture holds its two summands out of view; slide 20's raster holds a "RoBERTa Quantization" chart off the page; slide 44 holds a contour plot outside its clip; slide 45 crops its right chart's title ("Atari Breakout - Pareto Fronts"); slide 47 covers the typed arguments $f(x_i)$ and $f(x_j)$ with icons; and slide 50 holds another version of its figure out of view. Each entry describes what the page shows.

**OCW licence notices** are on slides 3, 24, 40, 45, 47 and 48, six in all, all typed and in the text layer; the slide file's set and the text layer's are the same. A scan for notices drawn as glyph outlines found one filled path of many segments, on slide 42: the red brush stroke under the gradient-noise label, not small print. Each notice is transcribed as an `*OCW notice: …*` line with the picture it covers; slides 47 and 48 exclude their apple, orange, elephant and giraffe icons ("Icons © source unknown"). Slide 24's notice block opens with a CC BY credit for the gopher photograph, and slide 50 carries "Courtesy of Huh, et al. Used under CC BY." Many slides reproduce charts from papers and from OpenAI with a citation or a URL and no notice; [`LICENSE.md`](../../LICENSE.md) lists them.

## Contents

| Slides | Section |
| ------ | ------- |
| 1–6 | Title; compute used in training, 2012–2018; "The Bitter Lesson"; the budget question; GPT-4's predicted loss and coding capability |
| 7–15 | Scaling laws as an optimization under a compute budget; Kaplan et al.'s setup; Finding #1, power laws in data, parameters and compute; Finding #2, the joint dependence on $N$ and $D$ and on $N$ and $S$, and the compute allocation |
| 16–17 | Similar findings: Rosenfeld et al. for vision and language, and extrapolation |
| 18–23 | Finding #3, model shape; Finding #4, larger models need fewer samples; Li et al.'s train large, then compress; Kaplan et al.'s take-aways, caveats and implications |
| 24–30 | Chinchilla (Hoffmann et al.): two years later, differences, three approaches, predictions, limitations; many more works on scaling laws |
| 31–35 | Can we explain scaling laws? The data-manifold hypothesis, a toy model, its relation to neural networks, empirical evidence |
| 36–37 | Breaking power-law scaling |
| 38–45 | Critical batch size: what the batch size should be; OpenAI Five's batch size; GD against SGD; gradient noise; progress per step; the critical batch size; compute cost against training time |
| 46–50 | Representational convergence: kernels; comparing a vision and a text kernel; scaling laws for cross-modal alignment; the Platonic representation hypothesis |
| 51–52 | Summary; OCW end page |

## Slide 1 — Lecture 20: Scaling Laws

Title: "Lecture 20: Scaling Laws". Below it: "Speaker: Phillip Isola". The footer band (light grey, across the bottom) carries the MIT logo at the left, "6.7960 Deep Learning" in red, the underlined URL "https://phillipi.github.io/6.7960" and, at the right, "Fall 2024". The printed slide number "1" is a tiny digit just under the end of the URL.

Two pasted charts sit under the speaker line, no caption or citation printed. The left one is the Compute panel of the three-chart image that slides 10–12 and 37 also draw (here clipped to its left chart), and the right one is slide 14's lower-left chart.

Left chart (a log-log plot with no title). Y axis: "Test Loss", log scale, ticks 2, 3, 4, 5, 6, 7 (2 at the bottom, 7 at the top). X axis: "Compute" with the sub-label "PF-days, non-embedding", log scale, ticks 10^-9, 10^-7, 10^-5, 10^-3, 10^-1, 10^1. Series:

- A bundle of many thin pale-blue curves (dozens, one per model size), each starting high at the upper left and falling steeply, then flattening as it meets the envelope.
- A thick black line along the lower edge of the blue bundle (the best loss achievable at each compute), running from about (10^-8, 6.4) down to about (0.3, 2.8).
- An orange dashed straight line (a power-law fit) running from the upper left (loss about 7 at about 10^-9) to the lower right (loss about 2.3 at 10^1), lying under the black line from about 10^-7 to 10^-4, touching it from about 10^-3 to 10^-1, and continuing alone past its end. Legend box at the lower left of the plot: orange dashes with the label $L = (C_{\min}/2.3 \cdot 10^{8})^{-0.050}$.

Right chart, titled "Loss vs Model Size and Training Steps". Y axis: "Loss", log-like scale, ticks 2.4, 2.8, 3.2, 3.6, 4.0, 4.4. X axis: "Estimated $S_{\min}$", log scale, ticks 10^4 and 10^5 (curves run from the left edge to about 2·10^5). A colour bar at the right, labelled "Parameters (non-embed)", log scale with ticks 10^6, 10^7, 10^8, running from purple (fewer parameters) at the bottom through blue and green to yellow (more parameters) at the top. About 13 solid curves, one per model size, coloured by the bar, each with a dotted fit running alongside; all fall as steps increase and each ends at a different step count (the top curves, dark indigo, highest loss at about 3.9 to 4.0, run longest to the right, to about $1.7 \cdot 10^{5}$; the yellow curve, lowest loss, about 3.0 at the left falling to about 2.4, ends earliest at about 4·10^4). Higher-parameter curves lie lower on the plot.

## Slide 2 — (no title; chart "AlexNet to AlphaGo Zero: A 300,000x Increase in Compute (Log Scale)")

![Slide 2 — (no title; chart "AlexNet to AlphaGo Zero: A 300,000x Increase in Compute (Log Scale)")](../images/20-scaling-laws/slide-2.jpg)

A pasted scatter chart. Chart title (bold, top left): "AlexNet to AlphaGo Zero: A 300,000x Increase in Compute (Log Scale)". Y axis label "Petaflop/s-days", log scale, ticks 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3, 1e+4. X axis: years 2012 to 2018, one tick per year. One series of light-blue dots, each labelled with a name (approximate year, approximate petaflop/s-days read off the grid):

- AlexNet (mid 2012, about 6e-3) and Dropout (mid 2012, about 2e-3).
- DQN (end of 2013, about 2.5e-5).
- Visualizing and Understanding Conv Nets (end of 2013, about 6e-3).
- VGG and Seq2Seq (about late 2014, both about 1e-1, nearly overlapping dots).
- GoogleNet (late 2014, about 2e-2).
- DeepSpeech2 (end of 2015, about 2e-1) and ResNets (end of 2015, about 1e-1).
- Xception (late 2016, about 5e+0).
- Neural Machine Translation (late 2016, about 8e+1) and Neural Architecture Search (end of 2016, about 3e+1).
- TI7 Dota 1v1 (mid 2017, about 7e+0).
- AlphaGoZero (late 2017, about 2e+3) and AlphaZero (end of 2017, about 4e+2).

A pale-grey dashed straight line rises from the lower left (about 2e-5 at the left edge, about 2011.4, crossing 1e-4 at 2012) to the upper right (about 7e+2 at the right edge, about 2018.5), with the annotation "3.4-month doubling" under it, spanning about 2013 to 2014. Bottom right: the URL "https://openai.com/blog/ai-and-compute/". Printed slide number "2".

## Slide 3 — The Bitter Lesson

*OCW notice: © Rich Sutton. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (top right of the slide, small serif print, the URL blue and underlined; it covers the pasted text of Sutton's essay).*

A pasted page of text. Heading "The Bitter Lesson", then "Rich Sutton", then "March 13, 2019", then the paragraph, transcribed as printed:

"The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation. There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent."

Printed slide number "3".

## Slide 4 — (no title; question about a compute budget)

Large text only: "If you had a given budget of compute, what model would you train on how much data?" Printed slide number "4".

## Slide 5 — (no title; chart "OpenAI codebase next word prediction")

![Slide 5 — (no title; chart "OpenAI codebase next word prediction")](../images/20-scaling-laws/slide-5.jpg)

A pasted chart with two added annotations. Chart title (bold): "OpenAI codebase next word prediction". Y axis label "Bits per word", ticks 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 (linear). X axis label "Compute", log scale, ticks 100p, 10n, 1μ, 100μ, 0.01, 1. Legend (top right box): grey dot "Observed", grey dashes "Prediction", green dot "gpt-4". Series:

- Observed: 9 grey dots on a falling curve, roughly (about 100p, 5.6), (about 2n, 4.15), (about 20n, 3.45), (about 100n, 2.95), (about 400n, 2.65), (about 1μ, 2.45), (about 6μ, 2.2), (about 25μ, 2.0), (about 70μ, 1.9).
- Prediction: a light-grey dashed curve through those dots, flattening as compute grows and continuing right to about 1.3 bits per word at compute 1.
- gpt-4: one green dot at compute 1, about 1.25 bits per word, on the end of the dashed curve.

Added annotations in large text: "Design model here" above a black down arrow at the last grey dot (about 70μ, just left of the 100μ tick), and "Rather than here" above a black down arrow at the green dot (compute 1); both arrows stop above their dots, at about 2.6 bits per word. Between the two, a blue horizontal bracket with end ticks, at about the 2.2 level just below the arrowheads, spans from the first arrow to the second, labelled above it in bold blue "10,000x savings". Bottom right: "https://openai.com/research/gpt-4". Printed slide number "5".

## Slide 6 — (no title; chart "Capability prediction on 23 coding problems")

![Slide 6 — (no title; chart "Capability prediction on 23 coding problems")](../images/20-scaling-laws/slide-6.jpg)

A pasted chart. Chart title (bold): "Capability prediction on 23 coding problems". Y axis label "– Mean Log Pass Rate" (printed with a leading dash), ticks 0, 1, 2, 3, 4, 5 (linear). X axis label "Compute", log scale, ticks 1μ, 10μ, 100μ, 0.001, 0.01, 0.1, 1. Legend (top right): grey dot "Observed", grey dashes "Prediction", green dot "gpt-4". Series:

- Observed: 5 grey dots, roughly (about 1μ, 4.05), (about 5μ, 3.05), (about 20μ, 2.2), (about 50μ, 1.75), (about 0.0005, 1.15).
- Prediction: a light-grey dashed curve through them, falling and flattening toward the right, ending near 0.3 at compute 1.
- gpt-4: one green dot at compute 1, about 0.3.

Bottom right: "https://openai.com/research/gpt-4". Printed slide number "6".

## Slide 7 — Scaling laws

Title: "Scaling laws". Three centred lines of bold sans text, the first with a red italic serif phrase:

- "minimize" then in red italics "test-loss(model size, data, batch size, steps,…)" then "such that" then "compute" in italics, then "within budget".
- "for a given architecture, how much data / what model size would be needed for a target performance?"
- "can we extrapolate from smaller to larger models / experiments??"

Below, a displayed equation (pasted image):

$$N_{opt}(C), D_{opt}(C) = \underset{N,D \text{ s.t. } \mathrm{FLOPs}(N,D)=C}{\arg\min} L(N, D)$$

Printed slide number "7".

## Slide 8 — How does the test loss scale as a function of data, #parameters etc?

Title: "How does the test loss scale as a function of data, #parameters etc?" Bullets:

- autoregressive Transformer
- data: WebText2, 96GB of text, $2.29 \cdot 10^{10}$ tokens
  1024 token context (a continuation line; the page prints no semicolons)
- vary:
  - model size: 768 - 1.5 billion non-embedding parameters
  - data: 22M - 23B tokens
  - shape (depth, with, attention heads, …)
  - batch size
- compute = 6 x #parameters x batchsize x #steps
- computation: in PF-days (petaflop/s-days)
  1 PF-day = $8.64 \cdot 10^{19}$ FLOPs (a continuation line, in smaller type)

At the lower right, a pasted image of a paper's title block: "Scaling Laws for Neural Language Models" between two horizontal rules, then the author block: Jared Kaplan * (Johns Hopkins University, OpenAI; jaredk@jhu.edu), Sam McCandlish* (OpenAI; sam@openai.com); then Tom Henighan (OpenAI; henighan@openai.com), Tom B. Brown (OpenAI; tom@openai.com), Benjamin Chess (OpenAI; bchess@openai.com), Rewon Child (OpenAI; rewon@openai.com); then Scott Gray (OpenAI; scott@openai.com), Alec Radford (OpenAI; alec@openai.com), Jeffrey Wu (OpenAI; jeffwu@openai.com), Dario Amodei (OpenAI; damodei@openai.com). Printed as "with" in "shape (depth, with, attention heads, …)" (the course page does not say, but the context suggests width was meant; transcribed as printed). Printed slide number "8".

## Slide 9 — Finding #1: power law relationships

Title: "Finding #1: power law relationships". Centred equation:

$$L(X) = (1/X)^{\alpha} = X^{-\alpha}$$

A blue arrow points up to $L(X)$ from the blue bold label "test  loss"; a green arrow points up to $X^{-\alpha}$ from the green bold label "resource (data/parameters/ compute" (the closing parenthesis is missing as printed, and the text wraps over three lines). Bullet at the bottom: "scale X -> 2X  =>  Loss -> $2^{-\alpha}$ Loss". Printed slide number "9".

## Slide 10 — How does the test loss scale as a function of data?

Title: "How does the test loss scale as a function of data?" First bullet: "#parameters, compute: “infinite”".

A pasted log-log chart (centre). Y axis "Test loss", ticks 2.7, 3.0, 3.3, 3.6, 3.9, 4.2. X axis "Dataset Size" with the sub-label "tokens", log scale, ticks 10^8 and 10^9. Series: a blue line with 7 blue dots falling from about (2·10^7, 4.12) to about (1.4·10^9, 2.75), nearly straight; and a grey straight fit line through them, with the legend $L = (D/5.4 \cdot 10^{13})^{-0.095}$. A blue arrow labelled in bold blue "log scale" points at the x axis.

Second bullet: "power law relationship:", then

$$L(D) = (D_c / D)^{\alpha_D} ; \quad \alpha_D \sim 0.095, \quad D_c \sim 5.4 \times 10^{13} \text{ (tokens)}$$

Citation, bottom right, italic: "Kaplan*, McCandlish*, et al. “Scaling Laws of Neural Language Models,” 2020" (printed "of"; the paper's own title block on slide 8 reads "Scaling Laws for Neural Language Models"; the printed slide number "10" overlaps the comma after "Kaplan*"). The chart is the middle panel of the three-chart image that slide 12 shows whole. In the pasted equation the subscript c of $D_c$ is upright. Printed slide number "10".

## Slide 11 — How does the test loss scale as a function of #parameters?

Title: "How does the test loss scale as a function of #parameters?" First bullet: "#tokens, compute: “infinite”".

Two pasted log-log charts side by side, sharing a y-axis label "Test loss" at the far left of the left one.

- Left chart (dataset size; same as slide 10's): y ticks 2.7, 3.0, 3.3, 3.6, 3.9, 4.2; x axis "Dataset Size" with sub-label "tokens", ticks 10^8 and 10^9; a blue line with 7 dots falling from about 4.12 to about 2.75 and a grey fit line, legend $L = (D/5.4 \cdot 10^{13})^{-0.095}$.
- Right chart (parameters): y ticks 2.4, 3.2, 4.0, 4.8, 5.6 (log scale, so unevenly spaced); x axis "Parameters" with sub-label "non-embedding", ticks 10^5, 10^7, 10^9. A blue line with about 22 dots falls almost straight from about (6·10^3, 5.9) (the axis starts near 3·10^3) to about (1.5·10^9, 2.3), with a grey straight fit line through it; legend: $L = (N/8.8 \cdot 10^{13})^{-0.076}$.

Below, displayed:

$$L(N) = (N_c / N)^{\alpha_N} ; \quad \alpha_N \sim 0.076, \quad N_c \sim 8.8 \times 10^{13} \text{ (non-embedding parameters)}$$

Citation: "Kaplan*, McCandlish*, et al. “Scaling Laws of Neural Language Models,” 2020" (printed "of", as on slide 10; printed slide number "11" overlaps the comma). Printed slide number "11".

## Slide 12 — How does the test loss scale with available compute?

![Slide 12 — How does the test loss scale with available compute?](../images/20-scaling-laws/slide-12.jpg)

Title: "How does the test loss scale with available compute?" Bullet in red: "power law for all of them".

Three pasted charts in a row, with a y-axis label "Test Loss" at the far left:

- Left (compute): y axis log, ticks 2, 3, 4, 5, 6, 7; x axis "Compute" with sub-label "PF-days, non-embedding", log scale, ticks 10^-9, 10^-7, 10^-5, 10^-3, 10^-1, 10^1. A bundle of many thin pale-blue curves falling steeply from the top, a thick black envelope along their lower edge (from about (10^-8, 6.4) to about (0.25, 2.8)), and an orange dashed straight fit line (from the upper left to the lower right, reaching about 2.3 at 10^1). Legend (lower left): orange dashes, $L = (C_{\min}/2.3 \cdot 10^{8})^{-0.050}$.
- Middle (dataset size): the same chart as slide 11's left one, legend $L = (D/5.4 \cdot 10^{13})^{-0.095}$.
- Right (parameters): the same chart as slide 11's right one, legend $L = (N/8.8 \cdot 10^{13})^{-0.076}$.

Below, displayed (with a thin grey underline drawn under $L(C_{\min}) = (C_c^{\min}$, stopping before the slash; the c subscripts and "min" are upright):

$$L(C_{\min}) = (C_c^{\min} / C_{\min})^{\alpha_C^{\min}} ; \quad \alpha_C^{\min} \sim 0.050, \quad C_c^{\min} \sim 3.1 \times 10^{8} \text{ (PF-days)}$$

Citation: "Kaplan*, McCandlish*, et al. “Scaling Laws of Neural Language Models,” 2020" (printed "of", as on slide 10; slide number "12" overlaps the comma). The constant is printed as $2.3 \cdot 10^{8}$ in the left chart's legend and as $3.1 \times 10^{8}$ in the equation below; both are as printed, and the deck does not reconcile them. Printed slide number "12".

## Slide 13 — Finding #2: simultaneous dependence on multiple factors

![Slide 13 — Finding #2: simultaneous dependence on multiple factors](../images/20-scaling-laws/slide-13.jpg)

Title: "Finding #2: simultaneous dependence on multiple factors". Bullet: "“Performance improves predictably as long as we scale up $N$ and $D$ in tandem, but enters a regime of diminishing returns if either $N$ or $D$ is held fixed while the other increases.”" (N and D in italics). Second bullet: "Loss as function of #parameters N & data D:", with the equation at the right:

$$L(N, D) = \left[ \left( \frac{N_c}{N} \right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D} \right]^{\alpha_D}$$

$N_c$ and $D_c$ are highlighted with translucent blue rounded boxes, and two blue arrows from the blue bold word "constants" (top right) run down and to the left to them.

Lower left, a pasted chart titled "Loss vs Model and Dataset Size". Y axis "Loss", ticks 2.5, 3.0, 3.5, 4.0, 4.5 (log-like); x axis "Tokens in Dataset", log scale, ticks 10^7, 10^8, 10^9, 10^10. Legend "Params" with six entries, colour from yellow to dark purple: 708M (yellow), 302M (yellow-green), 85M (green), 3M (blue), 25M (teal), 393.2K (dark purple) (printed in this order, 3M before 25M). Series: six dotted-fit curves with dots, one per model size, each falling with more tokens and flattening; the dark purple 393.2K curve is highest (about 4.7 at 2·10^7 down to about 4.2 flat), the slate-blue 3M curve next (about 4.5 to about 3.65), the teal 25M next (about 4.2 to about 3.1), the green 85M next (to about 2.85), then the yellow-green 302M (to about 2.6) and the yellow 708M lowest (to about 2.4 at 2·10^10), with the lower curves still dropping at 10^9. The dots are at about 2·10^7, 4·10^7, 9·10^7, 2·10^8, 4·10^8, 7·10^8, 1.4·10^9 and 2.4·10^10 tokens where present (the 393.2K curve has dots only at the first three, the sixth and the last); the dotted fits begin at 10^7, before the first data.

At the right below the equation, a small table:

| Parameter | $\alpha_N$ | $\alpha_D$ | $N_c$ | $D_c$ |
| --- | --- | --- | --- | --- |
| Value | 0.076 | 0.103 | $6.4 \times 10^{13}$ | $1.8 \times 10^{13}$ |

Citation: "Kaplan*, McCandlish*, et al. “Scaling Laws of Neural Language Models,” 2020" (printed "of", as on slide 10; slide number "13" overlaps the comma). Printed slide number "13".

## Slide 14 — Finding #2: simultaneous dependence on multiple factors

![Slide 14 — Finding #2: simultaneous dependence on multiple factors](../images/20-scaling-laws/slide-14.jpg)

Same title and first bullet as slide 13. Second bullet: "Loss as function of #parameters N & data D:" with the same $L(N, D)$ equation as slide 13 (now without highlights):

$$L(N, D) = \left[ \left( \frac{N_c}{N} \right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D} \right]^{\alpha_D}$$

Third bullet: "Loss as function of #parameters N & #steps S:", with the equation at the right, $N_c$ and $S_c$ highlighted with translucent light-blue rounded boxes and two blue arrows from the blue bold word "constants" (at the right edge, between the two equations) pointing to them:

$$L(N, S_{\min}) = \left( \frac{N_c}{N} \right)^{\alpha_N} + \left( \frac{S_c}{S_{\min}} \right)^{\alpha_S}$$

Lower left, the pasted chart "Loss vs Model Size and Training Steps" (the same as the right chart of slide 1): y axis "Loss", ticks 2.4, 2.8, 3.2, 3.6, 4.0, 4.4; x axis "Estimated $S_{\min}$", log scale, ticks 10^4 and 10^5; colour bar "Parameters (non-embed)" with ticks 10^6, 10^7, 10^8, purple at the bottom to yellow at the top; about 13 solid curves with dotted fits, purple ones highest, yellow lowest, each ending at a different step count (from about $4 \cdot 10^{4}$ for the yellow curve to about $1.8 \cdot 10^{5}$ for the top purple one). The chart is the same embedded image as slide 1's right chart.

Citation: "Kaplan*, McCandlish*, et al. “Scaling Laws of Neural Language Models,” 2020" (printed "of", as on slide 10; slide number "14" overlaps the comma). Printed slide number "14".

## Slide 15 — Finding #2: simultaneous dependence on multiple factors

Same title and first bullet as slide 13. Then the two loss bullets and equations as on slide 14, with no highlights or "constants" arrows:

- "Loss as function of #parameters N & data D:" with $L(N, D) = \left[ \left( \frac{N_c}{N} \right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D} \right]^{\alpha_D}$ (displayed at the right).
- "Loss as function of #parameters N & #steps S:" with $L(N, S_{\min}) = \left( \frac{N_c}{N} \right)^{\alpha_N} + \left( \frac{S_c}{S_{\min}} \right)^{\alpha_S}$ (displayed at the right, below).

Third bullet: "from this, they conclude to scale, with compute $C$:" followed in red by "#params $\propto C^{0.73}$, data = batch-size $\cdot$ #steps $\propto C^{0.27}$" (C in italics). At the lower right, in bold blue: "Take these exact equations/numbers with a grain of salt!". Printed slide number "15".

## Slide 16 — Similar findings in other works

![Slide 16 — Similar findings in other works](../images/20-scaling-laws/slide-16.jpg)

Title: "Similar findings in other works". Right of centre, two pasted 3D surface plots, each with a coloured surface (red-orange at one end to blue at the other), contour lines projected on the floor and on the back walls, and small blue dots on the surface:

- "(a) Wiki103 error (cross entropy) landscape.": z axis "log10(err)", ticks 0.55, 0.60, 0.65, 0.70, 0.75, 0.80; x axis "log2(data fraction)", ticks -5 to 0; y axis "log2(model fraction)", ticks 0, -2, -4, -6, -8, -10, -12.
- "(b) CIFAR10 error (top1) landscape.": z axis "log10(err)", ticks -0.4, -0.6, -0.8, -1.0, -1.2; x axis "log2(data fraction)", ticks -5 to 0; y axis "log2(model fraction)", ticks 6, 4, 2, 0, -2, -4, -6, -8.

At the lower left, a bullet: "Rosenfeld et al:" (italic), then "for vision, language, fit function" and the equation

$$L(N, D) \approx \frac{a(N)}{D^{\alpha_1(N)}} + \frac{b(N)}{N^{-\alpha_2(D)}} + c_{\infty}$$

Printed so, with a minus sign in the second denominator's exponent (the text layer agrees): read literally, the second term grows with $N$, unlike the first, which falls as a power of $D$. Probably a slip on the slide; the recording does not read the formula out (≈36:30 says only "the sum of these two power laws").

Citation, bottom right, italic: "Rosenfeld et al, 2020. A constructive prediction of the generalization error across scales" (the slide number "16" overlaps the "2" of "2020"). Printed slide number "16".

## Slide 17 — Extrapolation is possible!

![Slide 17 — Extrapolation is possible!](../images/20-scaling-laws/slide-17.jpg)

Title: "Extrapolation is possible!" Bullet: "from Rosenfeld et al" (name in italics). Three pasted panels:

- "(a) Illustration.": a grid of circles, 7 rows by 7 columns. Y axis "model fraction: log2(m/M)", ticks 0, -2, -4, -6, -8, -10, -12 (rows); x axis "data fraction: log2(n/N)", ticks -6, -5, -4, -3, -2, -1, 0 (columns). Row 0 and row -2: three empty (white outlined) circles then four red filled circles. Row -4: four green filled circles then three red filled circles. Rows -6, -8, -10, -12: four green filled circles then three empty circles. (Green = fit, red = extrapolated, as the legends of the next two panels show.)
- "(b) Extrapolation on ImageNet": a scatter of "estimated top1 error" (y, 0.0 to 1.0) against "measured top1 error" (x, 0.0 to 1.0) with a blue diagonal line y = x. Legend: green dot "fit", red dot "extrapolated". Annotations: "μ:-4.5%" and "σ:4.681%" (upper left), "model fraction 1/16, data fraction 1/8" (lower right). Green dots lie along the diagonal from about 0.53 to 1.0, red dots from about 0.27 to 0.47, all close to the line.
- "(c) Extrapolation on WikiText-103.": "estimated test loss" (y, 3.0 to 7.0) against "measured test loss" (x, 3 to 7), blue diagonal. Same legend. Annotations: "μ:0.5%" and "σ:1.689%", "model fraction 1/16, data fraction 1/8". Red dots from about 3.4 to 4.4, green from about 4.5 to 6.4, close to the line. In (b) and (c) the dots are joined in sequence by a thin grey line.

Citation: "Rosenfeld et al, 2020. A constructive prediction of the generalization error across scales" (slide number "17" overlaps the "2"). Printed slide number "17".

## Slide 18 — Finding #3: dependence on model shape

![Slide 18 — Finding #3: dependence on model shape](../images/20-scaling-laws/slide-18.jpg)

Title: "Finding #3: dependence on model shape". Bullet: "“Performance depends strongly on scale, weakly on model shape”". Two pasted charts side by side sharing a y-axis "Loss Increase", ticks 0%, 2%, 4%, 6%, 8%, 10%.

- Left: x axis "Aspect Ratio ($d_{\text{model}}$ / $n_{\text{layer}}$)", log scale, ticks 10^1, 10^2, 10^3. Legend: blue dots "50M Params", orange crosses "274M Params", green stars "1.5B Params". Three lines, each roughly U-shaped with minimum near 0% at an aspect ratio of about 30 to 100: blue starts at about 2.7% (aspect about 4), falls to 0% at about 50 and rises to about 4.6% at about 700; orange starts at about 1% (aspect about 3), reaches 0% at about 25, and climbs steeply to about 8% at about 1600; green starts at about 2.1%, falls to about 0% and rises to about 2.8% at about 700. Annotation: "A wide range of architectures achieve similar performance" with two black vertical lines hanging under it from about 4% down to about 1%, at aspect ratios of about 9 and about 300, bracketing the flat region.
- Right: x axis "Attention Head Dimension ($d_{\text{model}}$ / $n_{\text{head}}$)" with the sub-label "25M Parameters", log scale, ticks 10^1 and 10^2. Legend: blue dots ($d_{\text{model}} = 256$), orange crosses ($d_{\text{model}} = 512$), green downward-pointing triangles ($d_{\text{model}} = 1024$). Markers at head dimensions of about 16, 32, 64, 128 and 256. Three nearly flat lines near 0%: blue about 0% at 16 and 32, rising to about 1.1% at 128 and 1.8% at 256; orange about 0.5% at 16, near 0% in the middle, about 1.6% at 256; green about 0.55% at 16, 0% at 64 and 128, about 0.8% at 256. Annotation: a black error-bar mark spanning 1% (between 3% and 4%) with the text "22% additional compute compensates for 1% loss increase" to its right.

Citation: "Kaplan*, McCandlish*, et al. “Scaling Laws of Neural Language Models,” 2020" (printed "of", as on slide 10; slide number "18" overlaps the comma). Printed slide number "18".

## Slide 19 — Finding #4: larger models require fewer samples to reach the same performance

![Slide 19 — Finding #4: larger models require fewer samples to reach the same performance](../images/20-scaling-laws/slide-19.jpg)

Title (two lines): "Finding #4: larger models require fewer samples to reach the same performance". A pasted chart with the printed heading "Larger models require **fewer samples** to reach the same performance" (bold "fewer samples"). Y axis "Test Loss", ticks 4, 6, 8, 10; x axis "Tokens Processed", log scale, ticks 10^7, 10^9, 10^11. About 18 curves coloured from dark purple to yellow, all starting at about 10.8 at the left. The purple and blue curves stay flat at about 10.8 until about 10^8 tokens; the larger (green to yellow) curves start declining at once and fall steeply earlier, from about 10^7. Each flattens at a different level: the purple (smallest) curves fall last (around 10^9) and level out highest (about 6.5 and 6.0), while the yellow (largest) curves fall earliest and level out lowest (about 2.5 to 3, still declining slowly). Two arrows label the ends: "10^3 Params" (arrow pointing left at the top dark-purple curve, upper right, at a loss of about 7.4) and "10^9 Params" (arrow pointing right at the yellow curves in their steep fall).

Citation: "Kaplan*, McCandlish*, et al. “Scaling Laws of Neural Language Models,” 2020" (printed "of", as on slide 10; slide number "19" overlaps the comma). Printed slide number "19".

## Slide 20 — Similar findings…

![Slide 20 — Similar findings…](../images/20-scaling-laws/slide-20.jpg)

Title: "Similar findings…" (the ellipsis is the single character). Bullet: "Li et al, 2020: larger models train faster, not much loss via compression" ("Li et al, 2020:" in italics).

A flow diagram of two rows of three grey boxes joined by short black arrows:

- Row 1, labelled in bold red "Common Practice": "Train Small Model" → "Stop Training When Converged" → "Lightly Compress".
- Row 2, labelled in bold green "Optimal": "Train Large Model" → "Stop Training Early" → "Heavily Compress".

At the right, a pasted chart titled "RoBERTa Pruning" (printed "RoBERTa"). Y axis "MNLI Validation Accuracy", ticks 0.75, 0.80, 0.85; x axis "Number of Parameters (Millions)", ticks 0, 50, 100, 150, 200. Legend "Original Size": purple squares "3 Layers", pink diamonds "6 Layers", teal circles "12 Layers", yellow-green downward-pointing triangles "18 Layers", orange upward-pointing triangles "24 Layers". Five series, each a line of markers labelled with small percentages (90%, 75%, 60%, 45%, 30%, 15%, 0%) giving how much was pruned; accuracy rises as the parameter count rises, then flattens:

- 3 Layers (purple): from about 0.72 at about 5M (90%) to about 0.757 at about 48M (0%).
- 6 Layers (pink): from about 0.76 at about 7M (90%) to about 0.81 at about 70M (0%), roughly level over its last three markers.
- 12 Layers (teal): from about 0.80 at about 10M (90%) to about 0.843 at about 110M (0%).
- 18 Layers (green): from about 0.825 at about 15M (90%) to about 0.848 at about 150M (0%).
- 24 Layers (orange): from about 0.835 at about 20M (90%) to about 0.855 at about 200M (0%).

The percent labels are tiny and partly overlapping, so their exact positions are not all legible. Bottom right, in bold: the URL "https://bair.berkeley.edu/blog/2020/03/05/compress/". Printed slide number "20".

## Slide 21 — Take-aways from Kaplan et al.

Title: "Take-aways from Kaplan et al." ("Kaplan et al." in italics). Four bullets, with the phrases in red as marked:

- "test loss scales as" "power law" (red) "as a function of data, parameter count, compute"
- "test loss is more sensitive to parameter count than data: with more compute," "increase model by a larger factor than dataset size" (red)
- "don’t train to completion, rather make your model larger"
- "in general, scaling laws can help to “optimally” allocate computation"

Printed slide number "21".

## Slide 22 — “Caveats” for Kaplan et al

Title: "“Caveats” for Kaplan et al" (the quotation marks are around "Caveats"; "Kaplan et al" in italics). Three bullets:

- "“… the scaling with D at very large model size still remains mysterious. Without a theory or systematic understanding of the corrections to our scaling laws, it’s difficult to determine in what circumstances they can be trusted”"
- "didn’t thoroughly investigate small data regime, poor fits of L(N,D) in that regime. Didn’t investigate regularization or data augmentation."
- "didn’t tune all hyperparameters; possibly better learning rates for short training runs"

Printed slide number "22".

## Slide 23 — Implications

Title: "Implications". Bullet: "subsequently:" then in red "larger models, scale data only moderately". Below, a pasted table (three columns, rules above the header, under it and at the bottom; the citations inside the parentheses, author, "et al.," and year, are in blue, as links; the parentheses are black):

| Model | Size (# Parameters) | Training Tokens |
| --- | --- | --- |
| LaMDA (Thoppilan et al., 2022) | 137 Billion | 168 Billion |
| GPT-3 (Brown et al., 2020) | 175 Billion | 300 Billion |
| Jurassic (Lieber et al., 2021) | 178 Billion | 300 Billion |
| Gopher (Rae et al., 2021) | 280 Billion | 300 Billion |
| MT-NLG 530B (Smith et al., 2022) | 530 Billion | 270 Billion |

"Gopher" is in italics in the model column. The pasted image is cropped: below the visible table it holds a further row, hidden on the page, "*Chinchilla* | 70 Billion | 1.4 Trillion", and its closing rule; the rule under the MT-NLG row is the raster's thinner mid-rule. Printed slide number "23".

## Slide 24 — (no title; "Finding the right predictions is not easy…", gopher and dinosaur against chinchilla, with the Chinchilla paper heading)

*OCW notice: Dinosaur © Universal Pictures. Chinchilla © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/ (top right of the slide, small serif print, the URL blue and underlined; it covers the dinosaur still and the chinchilla photograph).* The same block begins with a credit: "Pocket gopher courtesy of LeonardoWeiss on Wikipedia Commons. Used under CC BY." (a CC BY credit for the gopher photograph, not a notice).

Layout. Upper left, two overlapping photographs: a close-up photograph of a small brown-grey furry rodent emerging from grass (the credit names it a pocket gopher), and, behind it and offset below and to the right, so that the gopher photograph covers its upper-left corner, a larger film-still-like photograph of a large dark dinosaur with open jaws on a wet road, with a yellow car at its right and the lower part of another vehicle at the left. Next to them, bold sans text: "spend compute on model size …". Middle, large title-style text: "Finding the right predictions is not easy…". At the right, bold sans text "… or on data?" above a photograph of a small grey furry animal with rounded ears, sitting on straw (the notice calls it a chinchilla).

Lower right, a pasted image of a paper's title block: the "DeepMind" logo (a blue circular swirl mark and the word) above a grey rule, the title "Training Compute-Optimal Large Language Models", and the author list: "Jordan Hoffmann*, Sebastian Borgeaud*, Arthur Mensch*, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals and Laurent Sifre*" then "*Equal contributions". Printed slide number "24".

## Slide 25 — Two years later…

Title: "Two years later…". A pasted paragraph of the paper (text partly highlighted in orange, the highlighted part marked here with bold):

"under a given compute budget. We find that current large language models are significantly under-trained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. **By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled.** We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4× more more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly"

(Printed as "4× more more data", with "more" twice; "Chinchilla" and "Gopher" in italics in the pasted text. The whole pasted paragraph is set in bold serif, so bold here marks only the amber highlight. The slide crops the pasted image at top and bottom: above the first visible line it holds "We investigate the optimal model size and number of tokens for training a transformer language model", and after "greatly" it continues "facilitating downstream usage. As a highlight, *Chinchilla* reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over *Gopher*." Neither shows on the page.) Below, two bullets:

- "still assume a power law, but find somewhat different equations / scaling"
- "still do not train to minimum loss, but train for many more tokens (& steps), increase #parameters and data equally: “smaller” models, train longer"

Printed slide number "25".

## Slide 26 — Differences

![Slide 26 — Differences](../images/20-scaling-laws/slide-26.jpg)

Title: "Differences". Bullets:

- "different learning rate schedules (not explored in Kaplan et al), adapted to dataset size"
- "larger models (> 500M parameters)" (printed on two lines)

At the lower right, a pasted chart with its caption. Y axis "Training loss", log-like scale, ticks 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 (6.0 at the top); x axis "FLOPS", log scale, ticks 10^17, 10^18, 10^19, 10^20, 10^21, 10^22. A colour bar "Million Parameters" on the right, log scale with ticks 75 (black at the bottom), 250, 500, 1000, 2500, 5000, 10000 (yellow at the top), running black to purple to red-orange to yellow. A fan of many curves coloured by that bar, each falling from the top of the plot; the dark and purple (small) curves are leftmost and fall earliest, the orange and yellow (large) curves lie to the right and fall steeply later, and all curves converge onto a common lower frontier of black dots running from about (10^18, 3.4) to about (10^22, 2.1); left of 10^18 the edge is the solid black curve of the smallest model. Three straight dashed lines cross one another, each running along a different stretch of the frontier: blue (steepest) starts highest at the left (about 4.05 at 10^17), lies along the frontier's left part (about 10^18 to 10^19) and falls below the plot just before 10^22; green (middle slope, about 3.7 at 10^17) lies along the middle part; and a salmon-red line (shallowest, lowest at the left, about 3.35 at 10^17) lies along the right part, from about 10^20.5, and is highest at the right edge. The lines are not labelled on the plot; the caption names them, and does not match the plot exactly: it names the lines orange, green and blue for the first, middle and last thirds of the frontier points, but no line is orange (the shallowest is salmon red), and the blue line follows the low-compute end of the frontier while the red one follows the high-compute end. Caption (printed under the chart): "Figure A5 | **Training curve envelopes.** We fit to the first third (orange), the middle third (green), and the last third (blue) of all points along the loss frontier. We plot only a subset of the points." (the words "Training curve envelopes." in bold). Printed slide number "26".

## Slide 27 — Approaches to estimate loss as a function of data and parameter count

Title: "Approaches to estimate loss as a function of data and parameter count". Bullets:

- "minimize $L(N,D)$ with constraint $FLOPs(N,D) = budget$" (the expressions L(N,D) and FLOPs(N,D) = budget are in italics)
- "Approach 1: for each fixed model size, record entire training curve (=different #training tokens, as they use 1 epoch only)"
- "Approach 2: fixed #FLOPs, use final training loss for different N,D"
- "Approach 3: fit parametric loss function", with, at the right of that bullet, a pasted equation image:

$$\hat{L}(N, D) \triangleq E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}.$$

A red bullet: "for all 3: optimal scaling of #parameters N / data D with compute C is roughly". Below it, centred, the typed equation $N \propto C^{0.49}, D \propto C^{0.51}$, and to its right in bold sans-serif "vs Kaplan et al: #params" $\propto C^{0.73}$, "data" $\propto C^{0.27}$ ("Kaplan et al" in bold italics).

Last lines, indented: "=> existing large models would need much more training data." and, in blue, "Better allocation: **smaller model, more data/steps**" (the words "smaller model, more data/steps" bold). Printed slide number "27" (it overlaps the word "model" in the last line).

## Slide 28 — Predictions: extrapolation

![Slide 28 — Predictions: extrapolation](../images/20-scaling-laws/slide-28.jpg)

Title: "Predictions: extrapolation". At the right, bold text: "For a given #FLOPs, how many parameters should we allocate? (compute = 6 x data x params)".

A chart reproduced from a paper, at the left. Axes: x-axis "FLOPs", log scale, ticks $10^{17}$, $10^{19}$, $10^{21}$, $10^{23}$, $10^{25}$; y-axis "Parameters", log scale, ticks 10M, 100M, 1.0B, 10B, 100B, 1T (printed "10M" overlapping the $10^{17}$ tick label at the lower left corner). Light grey grid lines. Data and curves, per the legend at the right:

- Many small blue dots (a cloud of individual runs) rising from about 100M parameters at about $10^{18}$ FLOPs to about 10B at about $10^{22}$ FLOPs, then ending there; nothing is plotted in the first decade.
- "Approach 1": a blue straight line (on the log-log axes) rising from about 30M parameters at $10^{17}$ FLOPs to about 250B at $10^{25}$ FLOPs.
- "Approach 2": an orange straight line almost on top of the blue one.
- "Approach 3": a green straight line, slightly flatter, ending lower, at about 150B at $10^{25}$ FLOPs.
- "Kaplan et al (2020)": a black dashed straight line, steeper than the three, entering at the bottom edge (10M) at about $2 \times 10^{17}$ FLOPs, crossing 1T at about $10^{24}$ FLOPs and leaving the top of the plot at about $4 \times 10^{24}$.

Four stars (legend entries under the lines, in this order), plotted between about $3 \times 10^{23}$ and $10^{24}$ FLOPs at the upper right:

- "Chinchilla (70B)": a turquoise star on the Approach 1/2 lines, at about 70B parameters and about $6 \times 10^{23}$ FLOPs.
- "Gopher (280B)": an orange star just right of (below) the black dashed line, with a clear gap, at about 280B, directly above the Chinchilla star (the same FLOPs).
- "GPT-3 (175B)": a red star just right of (below) the dashed line, at about 175B; the leftmost of the four, at about $3 \times 10^{23}$ FLOPs.
- "Megatron-Turing NLG (530B)": a purple star at the top, touching the dashed line on its right side, at about 530B and about $10^{24}$ FLOPs.

Printed slide number "28".

## Slide 29 — Discussion / limitations

Title: "Discussion / limitations". Bullets:

- "use (smoothed) training loss as a proxy for test loss"
- "expensive experiments: only two large-scale training runs (Gopher, Chinchilla), no additional tests at intermediate scales"
- "assume a power law for L(N) — but some concavity at high compute budgets, so it may not hold universally"
- "only one epoch of training (each data point seen once)"

At the lower right, a pasted chart with a caption: slide 26's figure again (the same embedded image, placed a little further right). Axes: x-axis "FLOPS", log scale, ticks $10^{17}$, $10^{18}$, $10^{19}$, $10^{20}$, $10^{21}$, $10^{22}$; y-axis "Training loss", log scale, ticks 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0. Many closely spaced curves (training-loss curves) run from the upper left down to the lower right, each descending steeply and then flattening; they are coloured by model size with a colour bar at the right labelled "Million Parameters" with ticks 75 (bottom, black), 250, 500, 1000, 2500, 5000 and 10000 (top, yellow), running black, purple, red-orange, orange, yellow from small to large. The small-model curves (black and purple) are at the left and lie lowest at small FLOPs; the larger (orange and yellow) curves start further right; each curve comes down onto a common lower envelope, the small models' at the left and the large models' further right, and black dots mark that envelope from about $10^{18}$ FLOPs at a loss of about 3.4 to about $10^{22}$ FLOPs at about 2.1. Three thin dashed straight lines cross the chart from the upper left to the lower right, all starting at the $10^{17}$ grid line: blue (steepest, highest at the left at about 4.05, leaving the bottom of the plot at about $10^{21.9}$), green (middle, about 3.7 at the left) and a salmon red (shallowest, lowest at the left at about 3.35, ending at the right edge at about 1.95); these are the three fits named in the caption.

Caption under the chart: "Figure A5 | **Training curve envelopes.** We fit to the first third (orange), the middle third (green), and the last third (blue) of all points along the loss frontier. We plot only a subset of the points." (Transcribed as printed. The shallowest dashed line is drawn in a salmon red, the colour the caption calls orange. The caption's assignment does not match the plot: the blue line runs along the black dots at the left, the first third of the frontier, and the salmon line along those at the right, the last third, whereas the caption gives first third orange and last third blue. Slide 26 shows the same figure.) Printed slide number "29".

## Slide 30 — Many more on scaling laws…

Title: "Many more on scaling laws…". Bullets:

- "many earlier works, too" followed in italics by "(Tan & Le 2019, Cho et al 2015, Miceli Barone et al 2017, Johnson et al 2018, Hestness et al 2017)"
- "vision models tend to have better scaling in model size than transformers" and a second line, $L(N) \propto 1/N^{\alpha}$ "for vision models (CNNs), with larger" $\alpha \approx 0.5$ followed in italics by "(Rosenfeld et al 2019, Sharma et al)"
- "power law for L(D) across vision, language, speech tasks" followed in italics by "(Hestness et al 2017)"
- "power laws in theoretical bounds: model" followed in italics by "(Yarotsky, 2018)", then ", data size" and in italics "(Liang et al, 2019)"
- "…."

Printed slide number "30".

## Slide 31 — Can we explain scaling laws? One hypothesis

Section-divider slide. Only the title, "Can we explain scaling laws? One hypothesis", set in the vertical middle of the page. Printed slide number "31".

## Slide 32 — A hypothesis: data manifold

Title: "A hypothesis: data manifold". In the middle, a screenshot of a paper's title block with a drop shadow: the title "Scaling Laws from the Data Manifold Dimension"; "Utkarsh Sharma", "Department of Physics and Astronomy", "Johns Hopkins University", "Baltimore, MD 21218, USA", with the address "USHARMA7@JHU.EDU" at the right; "Jared Kaplan", "Department of Physics and Astronomy", "Johns Hopkins University", "Baltimore, MD 21218, USA", with "JAREDK@JHU.EDU" at the right. Below: a bullet "Idea: neural networks are doing regression on an embedded data manifold". Printed slide number "32".

## Slide 33 — A toy model

![Slide 33 — A toy model](../images/20-scaling-laws/slide-33.jpg)

Title: "A toy model". Left column, bullets:

- "Toy model: approximate Lipschitz function $f(x)$ on $[0,1]^{d}$ by a piecewise constant function $c(x)$."
- "Use hypercubes with side length $s$: $N = s^{-d}$ regions (values)."
- "MSE scales as" $L(s) = \int_0^1 \lvert f(x) - c(x) \rvert^2 d^d x \lesssim \lambda^2 (s^2 d)$, where the factor $\lambda^2$ is highlighted by a pale-green rounded box with a green arrow from the green bold label "Lipschitz constant" pointing to it.
- "So, up to constant factors," $L(N) \lesssim \frac{1}{N^{2/d}}$, with this formula inside a pale-red rounded box.
- "For piecewise linear approximations (ReLU), MSE and cross entropy scale as $s^4$, so" and, on the next line inside a pale-red rounded box, $L(N) \lesssim N^{-4/d}$.

Upper right, a one-dimensional picture: a pair of axes (vertical arrow and horizontal arrow) with a black wavy curve (a shallow dip over the first bar, a peak over the fourth, a trough over the sixth, rising again past the last) over seven adjacent light-blue bars of different heights approximating it (a step function); the third bar from the left is a darker blue and is labelled $R_j$ beneath the horizontal axis. Below it at the right, a 3D bar-style plot in blue: a plateau-like stepped surface on a grid with axes "x1" (ticks −2 to 5) and "x2" (ticks −2 to 5) and a vertical axis with ticks 0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00; it shows two perpendicular ridges of height 1, one along each axis, on a floor at 0, crossing in a central block of height 2: a piecewise constant function of two variables. (The embedded image also holds, cropped out of view, the two single ridges as summands, with "+" and "=" signs; the page shows only the sum.) Printed slide number "33".

## Slide 34 — Relation to neural networks: hypothesis

Title: "Relation to neural networks: hypothesis". Bullets:

- "For piecewise linear approximations (ReLU), MSE and cross entropy scale as $s^4$, so $L(N) \lesssim N^{-4/d}$"
- "Common belief: NN map data into a low-dimensional “manifold” that depends on data, loss/task"
- "If we take $d$ to be intrinsic dimension of data manifold, then we get a scaling law"; below it, in a pale-red rounded box, $L(N) \propto \frac{1}{N^{\alpha}}$ with $\alpha \approx 4/d$
- "Suggests that scaling exponent is strongly related to data and task: different models will scale similarly on the same data"

Printed slide number "34".

## Slide 35 — Empirical evidence

![Slide 35 — Empirical evidence](../images/20-scaling-laws/slide-35.jpg)

Title: "Empirical evidence". Bullets:

- "Hypothesis:" $L(N) \propto \frac{1}{N^{\alpha}}$ "with" $\alpha \approx 4/d$
- "Experiment: measure intrinsic dimension of embedding and relate to scaling exponent $\alpha$"
- "Good fit for vision models, only upper bound for transformer: $\alpha \gtrsim 4/d$ from upper bound on L(N)"

At the right, a scatter plot titled `Intrinsic Dimension vs 4/α`. Axes: x-axis `4/α`, log scale, labelled ticks $10^1$ and $10^2$ (with minor ticks); y-axis "Intrinsic Dimension", log scale, labelled ticks $10^1$ and $10^2$. Legend (top left): a light-blue line `ID = 4/α for reference`; a teal circle "Teacher Student"; a pink circle "CIFAR10"; a pink square "MNIST"; a pink star "Fashion MNIST"; a pink diamond "SVHN"; a dark-blue short horizontal dash "GPT". The reference line is a straight light-blue diagonal running from the lower left corner to about 100 at the right edge. Points: about 17 teal circles (Teacher Student) running from about (2.2, 3.0) at the lower left, above the line, up to about (14, 14), most on or near the line; four sit above it between $4/\alpha$ of about 4.6 and 8, and two below it, at about (3.9, 1.9) and (7.7, 6.8); a pink circle (CIFAR10) at about (18, 17) on the line; a pink square (MNIST) above the line at about (6, 9.5); a pink diamond (SVHN) and a pink star (Fashion MNIST) near (11, 11.5) and (11, 10) on the line (the "pink" markers are a violet-pink). In the upper right, a dark-blue upward arrow with a short horizontal bar at its base (no label), at a $4/\alpha$ of about 53, its base bar at an intrinsic dimension of about 90 and its arrowhead reaching about 160 (the legend's "GPT" marker is a short dark-blue dash, the same as the bar at the arrow's base and in the same colour, so it is the GPT entry, plotted as a lower bound on intrinsic dimension). Printed slide number "35".

## Slide 36 — Breaking power law scaling

Section-divider slide. Only the title, "Breaking power law scaling". Printed slide number "36".

## Slide 37 — Breaking power law scaling

Title: "Breaking power law scaling". Bullets:

- "Power laws with small exponents: a small improvement in loss may need an order of magnitude more data/parameters"
- "Diminishing returns: suggests that many training examples are highly redundant"
- "Pruning intelligently can lead to better scaling (e.g." followed in italics by "Sorscher et al, 2022" and ")"

At the upper right, a line chart. Axes: x-axis "Dataset Size" with the grey sub-label "tokens", log scale, labelled ticks $10^8$ and $10^9$ (with minor ticks); y-axis "Test loss", log scale, ticks 2.7, 3.0, 3.3, 3.6, 3.9, 4.2. A blue line joining 7 blue dots descends from about 4.1 at the left (about $2 \times 10^7$ tokens) to about 2.75 at the right (about $1.5 \times 10^9$ tokens): the dots are at about 4.1, 3.8, 3.55, 3.3, 3.15, 2.9 and 2.75. A thin grey straight line lies almost on top of it, labelled by a legend box at the upper right of the plot: a grey line entry with $L = (D/5.4 \cdot 10^{13})^{-0.095}$. A blue arrow points from the blue words "log scale" (at the lower right, bold blue) to the x-axis. This is slide 10's chart again, the Dataset Size panel of the figure slide 10 credits to Kaplan, McCandlish et al. (the same embedded image, cropped the same way); this slide's footer cites only Sorscher et al.

Footer citation, italic, at the bottom right: "Sorscher et al, Beyond neural scaling laws: beating power law scaling via data pruning" (the printed slide number "37" is overprinted by the words "Beyond" and is barely legible).

## Slide 38 — Critical batch size

Section-divider slide. Only the title, "Critical batch size". Printed slide number "38".

## Slide 39 — What should the batch size be?

Title: "What should the batch size be?". Bullets:

- "Bigger batches are more costly but provide better estimates of the full-batch gradient"
- "FLOPS(16 SGD steps with batch size 16) = FLOPS(1 SGD step with batch size 256). Which is yields better performance?" (printed as "Which is yields", a slip)
- "FLOPs aren’t everything: one compute node can only handle some max batch size at once (in parallel). If you use bigger batches then you have to split up across nodes or do sequential computation. Splitting incurs communication costs. Sequential computation incurs time cost."

Printed slide number "39".

## Slide 40 — (no title; OpenAI's Dota 2 system and the large-batch training paper)

*OCW notice: "Left © OpenAI. Right © OpenAI, McCandlish, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/" (upper right of the slide, in small serif type; it covers the left image, the graphic of the Dota 2 system, and the right image, the paper's title block and abstract).*

At the left, a dark-background graphic. Six thumbnail screenshots of a game map stacked in a column at the left, each with a coloured border (light blue, light blue, blue-violet, purple, pink-purple, orange, top to bottom), and six similar thumbnails stacked at the right (teal-green, green, green, then three orange borders); thin coloured curves, each the colour of the thumbnail it leaves, run from each left thumbnail into a ring of 12 coloured dots (teal at the top, then clockwise green, yellow, orange, pink, purple and light blue) at the centre, which encloses a five-lobed looping knot-like logo, blue at the upper left shading to pink; curves run on from the ring to each right thumbnail. Under it, two URLs: "https://openai.com/research/openai-five-defeats-dota-2-world-champions" and "https://cdn.openai.com/dota-2.pdf", and, larger, "Batch size: ~2 million".

At the right, a screenshot of a paper's first page: a thick black rule, the title "An Empirical Model of Large-Batch Training", a thin rule, the authors "Sam McCandlish*" (OpenAI, sam@openai.com), "Jared Kaplan" (Johns Hopkins University, OpenAI, jaredk@jhu.edu) and "Dario Amodei" (OpenAI, damodei@openai.com), then "and the OpenAI Dota Team†", the heading "Abstract" and its text: "In an increasing number of domains it has been demonstrated that deep learning models can be trained using relatively large batch sizes without sacrificing data efficiency. However the limits of this massive data parallelism seem to differ from domain to domain, ranging from batches of tens of thousands in ImageNet to batches of millions in RL agents that play the game Dota 2. To our knowledge there is limited conceptual understanding of why these limits to batch size differ or how we might choose the correct batch size in a new domain. In this paper, we demonstrate that a simple and easy-to-measure statistic called the *gradient noise scale* predicts the largest useful batch size across many domains and applications, including a number of supervised learning datasets (MNIST, SVHN, CIFAR-10, ImageNet, Billion Word), reinforcement learning domains (Atari and Dota), and even generative model training (autoencoders on SVHN). We find that the noise scale increases as the loss decreases over a training run and depends on the model size primarily through improved model performance. Our empirically-motivated theory also describes the tradeoff between compute-efficiency and time-efficiency, and provides a rough model of the benefits of adaptive batch-size training." A thick black curved arrow runs from the right edge of the left graphic to the "and the OpenAI Dota Team" line. Below the paper: "https://arxiv.org/abs/1812.06162". Printed slide number "40".

## Slide 41 — (no title; GD, SGD and the variability of the SGD solution)

![Slide 41 — (no title; GD, SGD and the variability of the SGD solution)](../images/20-scaling-laws/slide-41.jpg)

Three square panels side by side, each a blue gradient (dark blue at the edges, pale at an off-centre bright patch at the upper right), with a white-filled black circle at the lower left and a star at the upper right (the page does not say what they stand for; the recording does not reach this slide). Everything on the slide is vector drawing: the gradients are vector shadings with drop shadows, which is why a raster test reports nearly the whole page covered; at the upper right of the slide, a rounded cream box with a black border: "Variability of solution as a function given your mini-batch size" (printed so, "as a function given").

- Left panel: the label $J(\theta; \mathbf{X}_ {\text{train}})$ at its top; a yellow (gold) star; a white rounded label "GD" with a black border; one long black arrow from the circle toward the upper right, aimed at the star.
- Middle panel: the label $J(\theta; \mathbf{x}_ {\text{train}}^{(1)})$; a red star at the upper middle; a white rounded label "SGD" with a red border; one short red arrow from the circle pointing up and slightly right.
- Right panel: the label $J(\theta; X_{\text{train}}^{RV})$ (italic, not bold, X); seven pale-red stars scattered at the upper right and one orange star with a black border among them, partly overlapping the pale-red star to its right; a horizontal two-headed green arrow between the leftmost pale-red star and the orange star; from the circle, the long black arrow of the left panel and three short red arrows chained head to tail (a zigzag of SGD steps): the first up and slightly right, the second right and slightly down across the black arrow, the third up and to the right, ending beside the black arrow's tip.

Credits: bottom left "[slide credit: adapted from Minyoung Huh]"; bottom right, italic, "McCandlish, Kaplan, Amodei, “An Empirical Model of Large-Batch Training”, 2018" (the printed slide number "41" sits as a subscript between "McCandlish" and "Kaplan").

## Slide 42 — Gradient noise

![Slide 42 — Gradient noise](../images/20-scaling-laws/slide-42.jpg)

Title: "Gradient noise". Left, a diagram: a black-outlined circle containing $\theta$; from it a long black arrow to the right labelled $\nabla\theta$ above; a shorter grey arrow down and to the right labelled $\nabla_{\text{sgd}}\theta$ (the label sits below the arrow); a red arrow from the tip of the black arrow to the tip of the grey arrow, with the label $\nabla\theta - \nabla_{\text{sgd}}\theta$ beside it, underlined with a red brush stroke, and under that, in red bold, "Gradient Noise". (The label is printed as $\nabla\theta - \nabla_{\text{sgd}}\theta$, the opposite sign of the arrow's direction, from the tip of $\nabla\theta$ to the tip of $\nabla_{\text{sgd}}\theta$, and of the equation's Noise term below; transcribed as printed.) Below, the equation, with the first term on the right, $\nabla\theta$, labelled in blue "Signal" and the bracketed term labelled in red "Noise":

$$\nabla_{\text{sgd}}\theta = \nabla\theta + \left(\nabla_{\text{sgd}}\theta - \nabla\theta\right)$$

Right, a trajectory plot with no axes: a small circle marked "Start" at the lower left and a black dot marked "Minimum" at the upper right. Two paths join them: an orange path of about 30 widely spaced points (large straight segments rising steeply to a high point about a third of the way across, falling to a low point at the lower right with a small zigzag on the way down, and climbing back to end just above the minimum), and a blue path of many closely spaced points (a jagged, wandering path that starts inside the Start circle, dips just below it, also rises to the right but passes above and below the orange one, and overshoots the minimum above, to the right and below before ending inside the dot). A legend box at the lower right: orange dot "Less noise, larger steps"; blue dot "More noise, smaller steps".

Credits: bottom left "[slide credit: adapted from Minyoung Huh]"; bottom right, italic, "McCandlish, Kaplan, Amodei, “An Empirical Model of Large-Batch Training”, 2018" (the printed slide number "42" is a subscript between "McCandlish" and "Kaplan").

## Slide 43 — How much progress can one step of SGD make?

Title: "How much progress can one step of SGD make?". Underlined: "Optimal learning rate for SGD?". At the upper right a typed note in plain sans-serif, "covariance of G_sgd, with B=1", with a curved black arrow from below it pointing down at the $\Sigma$ in the first equation. Equations (pasted images):

$$\mathbb{E}\left[\mathcal{L}(\theta + \eta G_{\text{sgd}})\right] = \mathcal{L}(\theta) - \eta \lvert G \rvert^2 + \frac{\eta^2}{2}\left(G^T H G + \frac{\operatorname{trace}(H\Sigma)}{B}\right)$$

Bold italic caption: "Take the derivative and solve for the learning rate". Then

$$\eta_{\text{opt}} = \frac{1}{1 + \mathcal{B}_ {\text{noise}} / B}\thinspace  \eta_{\text{max}} \qquad \text{where} \qquad \mathcal{B}_ {\text{noise}} = \frac{\operatorname{trace}(H\Sigma)}{G^T H G}$$

Bold italic caption: "Plug the equation learning rate back into the equation" (printed so). Then

$$\Delta\mathcal{L}_ {\text{opt}} = \frac{1}{1 + \mathcal{B}_ {\text{noise}} / B}\thinspace  \Delta\mathcal{L}_ {\text{max}} \qquad \text{where} \qquad \Delta\mathcal{L}_ {\text{max}} = \frac{1}{2}\thinspace  \frac{\lvert G \rvert^4}{G^T H G}$$

(In the pasted images $\mathcal{B}$ is a script B and $\mathcal{L}$ a script L, while the $B$ under the trace term is a plain italic B; "trace" is in italics, and the subscripts "sgd", "opt", "max" and "noise" are sans-serif.) Credits: bottom left "[slide credit: adapted from Minyoung Huh]"; bottom right, italic, "McCandlish, Kaplan, Amodei, “An Empirical Model of Large-Batch Training”, 2018" (printed slide number "43" as a subscript between the names).

## Slide 44 — (no title; progress per step and the critical batch size)

![Slide 44 — (no title; progress per step and the critical batch size)](../images/20-scaling-laws/slide-44.jpg)

Left, bullets:

- "One step of SGD can make progress proportional to:" and, centred beneath, $\dfrac{1}{1 + \mathcal{B}_ {\text{noise}} / B}$
- "For small B, second term of denominator dominates, so increasing B linearly increases progress"
- "For large B, first term of denominator dominates, so increasing B has little effect"

Right, a chart titled "Predicted Training Speed". Axes: x-axis "Batch Size / Noise Scale ($B/\mathcal{B}$)", log scale, ticks $10^{-2}$, $10^{-1}$, $10^{0}$, $10^{1}$, $10^{2}$; y-axis $\epsilon_{\text{opt}}(B) / \epsilon_{\text{max}}$, log scale, ticks $10^{-2}$, $10^{-1}$, $10^{0}$. One blue curve (1 series) rises from $10^{-2}$ at x = $10^{-2}$ along a slope-one line to about 0.5 at x = $10^{0}$, then bends over and flattens to about 1 ($10^{0}$) for x from about $10^{1}$ to $10^{2}$. A vertical grey dashed line at x = $10^{0}$. Annotations (with short grey leader lines, not series): "Perfect scaling" under the rising part, left of the dashed line; "Ineffective scaling" below the flat part, right of it, its leader line rising to the curve near $10^{1}$ (both labels in bold serif). Under the plot, a black arrow pointing up at $10^{0}$ on the x-axis from the bold words "Critical batch size". Credit at the bottom right, italic: "McCandlish, Kaplan, Amodei, “An Empirical Model of Large-Batch Training”, 2018" (printed slide number "44" as a subscript between the names).

## Slide 45 — (no title; compute cost against training time)

*OCW notice: "© OpenAI, McCandlish, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/" (upper right of the slide, small serif type; it covers the two charts on the slide).*

Two charts from the paper. Left, a schematic: axes "Compute Cost" (vertical) and "Training Time" (horizontal), no ticks. A thick black curve falls steeply from the top near the left, then flattens to the right (a hyperbola-like frontier). The region above and to the right of the curve is orange, labelled "Technically Feasible" (at the top right). A dotted horizontal line labelled "Budget" at the left, and a dotted vertical line labelled "Deadline" (text rotated, at the bottom) bound a blue rectangle at the lower left, labelled "Economically Feasible"; where the blue region overlaps the orange, between the curve, the budget line and the deadline line, it is a blue-grey. A two-headed black arrow in the orange region just above the curve and parallel to it, with "More Hardware (Larger Batches)" above its upper left end and "Less Hardware (Smaller Batches)" above and right of its lower right end.

Right, a chart. Axes: x-axis "Training Time (Hours)", log scale, ticks $10^{-1}$, $10^{0}$, $10^{1}$; left y-axis `Compute Cost ($)`, log scale, ticks $10^{-1}$, $10^{0}$, $10^{1}$; right y-axis "Days of Gameplay", log scale, ticks $10^{0}$, $10^{1}$, $10^{2}$. Two series, each a solid curve with dotted lines joining dots, both falling steeply and then flattening, as in the schematic: a yellow one, legend "Score = 10", falling from above $10^{1}$ dollars at about 0.08 hours to flatten at about 0.08 dollars, just below the $10^{-1}$ tick, ending at about 1 hour; and a dark-purple one, legend "Score = 500", from about 28 dollars at about 0.2 hours down to about 1.8 dollars at about 12 hours, its last dot (at about 8 hours) rising slightly to about 2.3 dollars. Two labels with leader lines on the purple series: "4096 Parallel Players" at its topmost dot (about 15 dollars, about 0.2 hours) and "16 Parallel Players" at that last dot. (A faint row of marks above the right chart's frame is the cut-off lower edge of the chart's title; the embedded image, which the slide crops, reads "Atari Breakout - Pareto Fronts".) Credit at the bottom right, italic: "McCandlish, Kaplan, Amodei, “An Empirical Model of Large-Batch Training”, 2018" (printed slide number "45" as a subscript between the names).

## Slide 46 — Representational convergence

Section-divider slide. Only the title, "Representational convergence". Printed slide number "46".

## Slide 47 — Characterizing representations using kernels

*OCW notice: "Icons © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/" (lower right of the slide, small serif type; it covers the apple, orange, elephant and giraffe emoji-style icons on the slide).*

Title: "Characterizing representations using kernels". At the left, a heat-map with the heading $\mathbf{K}_ {\text{vision}}$ ("vision" in sans-serif subscript): a $4 \times 4$ grid of rounded coloured squares in a rounded white frame. The rows are labelled at the left, and the columns at the bottom, by four icons in the same order: a red apple, an orange (the fruit, with a green leaf), a grey elephant, a spotted giraffe. A vertical colour bar at the right, running from pale yellow at the top, through orange, red and purple, to black at the bottom, is labelled "similar" at the top and "dissimilar" at the bottom. The cells: the diagonal is pale yellow (most similar) in all four rows; apple-orange (both off-diagonal cells) is a deeper yellow-orange; elephant-giraffe (both cells) is orange; apple-elephant and orange-elephant are dark purple (apple-elephant darkest, orange-elephant slightly lighter); apple-giraffe and orange-giraffe are the same magenta-purple. The grid is symmetric, with dark grey lines between the cells inside a light-grey-bordered white frame.

At the right, in prose and equations: "Restrict our attention to **vector embeddings**" (the words "vector embeddings" bold), then

$$f : \mathcal{X} \rightarrow \mathbb{R}^{n}$$

then "Characterize a representation in terms of its **kernel**" (the word "kernel" bold), then

$$K : \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$$

and, at the bottom,

$$K(x_i, x_j) = \langle f(\text{apple}), f(\text{orange}) \rangle$$

where the arguments of $f$ are printed as icons: the first is a red apple; the second is an orange, with a small grey elephant icon behind it whose head and trunk peek out at its upper left (printed so; the page stacks one build of the icons over another). Under the icons the text layer still holds the typed form, $K(x_i, x_j) = \langle f(x_i), f(x_j) \rangle$. Printed slide number "47" at bottom centre, and a second, larger "47" at the bottom left corner.

## Slide 48 — (no title; sim of the vision kernel and the text kernel)

*OCW notice: "Icons © source unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/" (lower right of the slide, small serif type; it covers the apple, orange, elephant and giraffe icons in the left panel).*

A large expression across the slide: the word "sim", a large opening parenthesis, two panels separated by a large comma, and a large closing parenthesis. The left panel (pale-blue rounded background) is the heat-map of slide 47, headed $\mathbf{K}_ {\text{vision}}$, with the same four icons (apple, orange, elephant, giraffe) down the left side and along the bottom and the same cell colours (pale yellow diagonal; apple-orange deeper yellow; elephant-giraffe orange; the rest dark purple to magenta). The right panel (pale-cream rounded background) is a second $4 \times 4$ heat-map headed $\mathbf{K}_ {\text{text}}$ ("text" in sans-serif, like "vision"), with the rows labelled at the left by the quoted words "apple", "orange", "elephant", "giraffe" and the columns labelled at the bottom, rotated about 45°, by the same four quoted words (the quoted labels are in a typewriter face). Its cells: pale-yellow diagonal; apple-orange orange; elephant-giraffe amber (yellow-orange); apple-elephant and orange-elephant in purple and crimson (apple-elephant dark purple, orange-elephant crimson); apple-giraffe nearly black and orange-giraffe magenta. Printed slide number "48" at bottom centre, and a second "48" at the bottom left corner.

## Slide 49 — Scaling laws for cross-modal alignment

![Slide 49 — Scaling laws for cross-modal alignment](../images/20-scaling-laws/slide-49.jpg)

Title: "Scaling laws for cross-modal alignment". A scatter plot with trend lines. The y-axis is a black arrow pointing up, labelled "Alignment to VISION (DINOv2)" ("VISION" in slate blue-grey); the x-axis is a black line labelled "LANGUAGE performance" ("LANGUAGE" in orange). No numeric tick labels; light grey grid lines (4 horizontal, and one vertical line per language model). Above the plot, rotated labels name the language models, each over its vertical grid line, from left to right: "bloom0.56b", "bloom1.1b", "bloom1.7b", "bloom3b", "bloom7b", "openllama3b", "openllama7b", "openllama13b", "llama13b", "llama33b", "llama65b" (11 models, so 11 dots per series; the x positions are not evenly spaced: bloom0.56b at the far left, bloom1.1b, bloom1.7b and bloom3b close together, bloom7b mid-way, and the last six clustered at the right).

Four series, per the legend box at the lower right (a white rounded box over the plot's lower right):

- "DINO Small": yellow dots and a yellow straight trend line, the lowest line, rising from near the bottom left to about two-thirds of the way up at the right.
- "DINO Base": green dots and a green line, above the yellow one.
- "DINO Large": blue dots and a blue line, above the green one.
- "DINO Giant": dark-purple dots and a dark-purple line, the highest, ending at the top right.

Every trend line ends in a filled arrowhead of its colour at the upper right, and starts at the left in a round cap that looks like an extra dot beside the first data point. Each line rises from the lower left to the upper right, and the gaps between the lines widen slightly at the right (Giant and Large nearly coincide, with Giant slightly above). At the right end of the lines, three black dashed arrows leave from the end of the green line's arrowhead: one rising up and to the right at a moderate angle (about 25°), to the slide's right edge above the plot, one pointing right and slightly up (nearly flat), and one curving to the right and then down to the bottom of the slide.

Citation, italic, at the bottom: "Huh*, Cheung*, Wang*, Isola*, “The Platonic Representation Hypothesis,” ICML 2024" (the printed slide number "49" overlaps the text after "Cheung*," and is barely legible).

## Slide 50 — (no title; the Platonic representation hypothesis diagram)

![Slide 50 — (no title; the Platonic representation hypothesis diagram)](../images/20-scaling-laws/slide-50.jpg)

A diagram. At the top, $Z$ above a drawing of a blue cone with a red sphere in front of it at its lower right. Two arrows leave this scene: a solid black arrow down and to the left, toward a parallelogram labelled $X$ (an image plane) containing a purple cone and a red disc at its lower right, seen from a small camera drawn at the lower left of the plane (dotted lines from the camera to the plane's corners); and a dashed black arrow down and to the right, to $Y$ above a rounded speech-bubble box (its pointer at the left) with the typewriter text "A red sphere next to a blue cone." A horizontal solid black arrow runs from the image plane to the text box. Below the image plane, a downward arrow labelled $f_{\text{img}}$ ("img" in typewriter face) ends at a tall empty rectangle (a vector); below the text box, a downward arrow labelled $f_{\text{text}}$ ("text" in typewriter face) ends at a tall empty rectangle.

Credit, small sans-serif, lower right: "Courtesy of Huh, et al. Used under CC BY." (a credit, not an OCW notice). Citation, italic, at the bottom: "Huh*, Cheung*, Wang*, Isola*, “The Platonic Representation Hypothesis,” ICML 2024" (the printed slide number "50" overlaps the text after "Cheung*," and is barely legible).

## Slide 51 — Summary

Title: "Summary". Bullets:

- "Neural scaling laws: predict test loss as a function of resources and model hyperparameters"
- "Allows to “optimally” allocate compute resources"
- "Power law scaling in model size, data size across variety of models and tasks"
- "Actual parameters hard to measure, large extrapolation nontrivial"

Printed slide number "51".

## Slide 52 — (no title; MIT OpenCourseWare end page)

OCW's appended end page (smaller, landscape, plain white). Text: "MIT OpenCourseWare" and the link "https://ocw.mit.edu"; "6.7960 Deep Learning" and "Fall 2024"; "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms" (the link underlined). At the bottom centre it prints the number "52".
