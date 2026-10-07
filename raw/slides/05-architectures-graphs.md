---
title: Lecture 5 — Graph Neural Networks (slide deck)
lecture: 5
slides: 47
source_pdf: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec5.pdf
note: Printed slide numbers 1–46 (bottom centre) equal the PDF page numbers exactly. Page 47 is OCW's appended end page (a 4:3 page), not part of the lecture deck; it prints 47, which the number-map script does not read.
figure_audit: Transcribed by Sonnet from page images; 27 graph-, chart- and equation-heavy pages (3, 5, 8–13, 15–20, 22, 23, 27, 28, 30, 35–37, 39–41, 43, 44) were then checked by Opus, a different model, from 250–600 dpi crops and the PDF's vector data. Every equation agreed, and the printed slips on slides 30, 39, 40 and 41 were confirmed; the audit found one more, slide 18's missing closing brace. Corrections applied: counts on slides 8 (nine red dots), 9 (16 edges) and 41 (eight curves, two of them green, and the callout tips); arrow ends and directions on slides 16 and 43; blob membership on slides 12 and 15; dot positions on 10; the boxed region on 30; and fuller descriptions of slide 36's second equivalence class and slide 43's bars. It also confirmed that slides 3, 12, 15, 21 and 29 share one 19-node, 24-edge graph drawing, and that slide 3's Leskovec credit and notice sit under its Pinterest figure.
---

# Lecture 5 — Graph Neural Networks: slide-by-slide

Text and figures of all 47 pages of
[`mit6_7960_f24_lec5.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec5.pdf),
transcribed from the deck (speaker: Phillip Isola; the recorded title is "Architectures: Graphs"). Cite these as "slide N" — the printed number equals the PDF page number for slides 1–46; page 47 is OCW's appended end page. Diagrams, plots and photographs are described in prose since the KB is read as text.

**Images.** 19 slides carry a whole-slide render under their heading: 9–13, 15, 16, 19–23, 29, 35, 36, 39, 41, 42 and 44. Not rendered: the 14 slides with an OCW "All rights reserved" notice (3–8, 24–28, 30, 31, 43); build steps and repeats superseded by a rendered slide (34 by 35, 37 and 38 by 39); slides whose equations this file reproduces exactly (17, 18, 40, 46); and the title, roadmap, connections, summary and end pages (1, 2, 14, 32, 33, 45, 47). Slides 12, 15, 21 and 29 reuse slide 3's node-classification graph; they print no notice, and slide 3's Leskovec credit and notice sit under its Pinterest figure rather than that graph, so they are rendered. Use an image path only if it appears in this file.

Companion pages: [wiki page for this lecture](../../wiki/05-architectures-graphs.md) ·
[transcript](../transcripts/05-architectures-graphs.md)

**Signposting slides you can skip.** Slides 2, 14 and 33 are the roadmap, repeated with the current section in red; slide 45 is a text summary; and slide 47 is the OCW end page.

Some slides are **build steps** — the same slide re-shown with more revealed: slides 7–8 (the learned-simulator figure), 14 and 33 (roadmap), 25–27 (the tree view of node A), 34–35, 37–39 (Weisfeiler-Leman) and 21 (slide 15 with a read-out added). They are transcribed individually, each with a note of what it adds.

## Contents

| Slides | Section |
| ------ | ------- |
| 1 | Title |
| 2 | Roadmap |
| 3–9 | Learning tasks with graphs: node classification, link prediction (Pinterest), molecule property prediction, polypharmacy side effects, traffic times, learning to simulate physics, combinatorial optimization |
| 10 | Two goals: node embeddings and graph embedding |
| 11 | Idea 1: a fully-connected network on the adjacency matrix; permutation invariance and equivariance |
| 12–13 | Idea 2: images are like graphs; a CNN is a GNN over a grid graph |
| 14 | Roadmap (message passing GNNs) |
| 15–16 | Graph neural networks: encode nodes by message passing, then aggregate; the general AGGREGATE and UPDATE form |
| 17–20 | Aggregation functions: sum, average, min/max, learned MLP form; shortest path and Bellman-Ford |
| 21 | Graph embeddings and READOUT |
| 22–23 | GNNs unrolled as layers; an MLP as a GNN over a single node |
| 24 | Generalizations: edge attributes, multi-relational, attention, Janossy pooling |
| 25–28 | Node embeddings as a tree; shared weights; weight sharing and unseen graphs |
| 29 | Training a GNN |
| 30–31 | Example architectures: polypharmacy, Google Maps |
| 32 | Many connections |
| 33 | Roadmap (approximation power) |
| 34–35 | Which functions can GNNs approximate; distinguishing graphs; equivalence classes and a Stone-Weierstrass style theorem |
| 36–39 | Discriminative power; color refinement / Weisfeiler-Leman; GNNs versus WL |
| 40–41 | Injective aggregation and its effect on training accuracy |
| 42–43 | Structural graph properties GNNs cannot compute; positional encodings |
| 44 | Call back to CNNs: positional encoding |
| 45 | Summary |
| 46 | Appendix: graph Laplacian |
| 47 | MIT OpenCourseWare end page |

---

## Slide 1 — Lecture 5: Graph Neural Networks

Title: "Lecture 5: Graph Neural Networks". Subtitle: "Speaker: Phillip Isola".

Decorative background across the lower two thirds: a large grey network drawing of small pale-grey circles (roughly 30, some cut off at the left, right and bottom edges) joined by thin grey lines, with several hub nodes of high degree. One node, lower middle, is drawn darker than the rest. It carries no labels.

Footer bar (grey): MIT logo, "6.S898 Deep Learning" (red; the course's earlier number, transcribed as printed), "https://phillipi.github.io/6.7960", right side "Fall 2024".

## Slide 2 — Roadmap

(Agenda / signpost slide.)

- Learning tasks with graphs
- Message passing GNNs
- Approximation Power

## Slide 3 — Prediction with graphs: examples

Left: a graph drawing of 19 nodes (circle outlines with black edges). Eleven nodes are drawn as thick blue rings with white centres, seven as small plain white circles with black outlines and a drop shadow, and one, right of centre, as a thick red ring with a red "?" above it. The red node has seven edges: to a plain white node at upper left (which continues to the top-hub blue node), to a blue node at upper right, to a small white node at right, to a blue node at the lower left of it (which connects on to the lower-middle hub), to a blue node below left, to a small white node below that leads to a blue node at the bottom, and to a blue node at lower right (which is also joined to that bottom blue node). The upper-left hub is a blue node with seven edges. Caption below, bold, on two lines: "Node classification".

Right: an illustration of Pinterest pins and boards. Along the top, three pin thumbnails (a blue jacket captioned "Very ape blue structured coat", a blue chair captioned "Hans Wegner chair", and a plant-like sculpture captioned "This is just a beautiful image for thoughts. Yay or nay, your choice."). To their right, in blue and black text: "Pins: Visual bookmarks someone has saved from the internet to a board they've created. Pin features: Image, text, link" ("Pins" and "Pin features" in blue). Below, a row of eight board thumbnails (collages of small photos, with small captions such as "mid century modern ...", "Man Style", "men + style l", "Plants", "Men's Style", "Mid century modern ...", "Plants", "Mid century modern ..."), labelled "Boards" in blue underneath. Black, blue and red lines run down from the pins to some of the boards (the links). Caption below, bold: "Link Prediction". Below that, in small green italics: "(e.g. Ying et al, 2018; illustration: J. Leskovec)".

At the lower right, a framed text excerpt (the inline quotation is as printed): "concerned recommender systems, which are very naturally representable as a graph-structured task: with Pinterest being one of the most early adopters [18, 33]. GNNs have also been deployed for product recommendation at Amazon [12], E-commerce applications at Alibaba [32], engagement forecasting and friend ranking in Snapchat [22, 24], and most relevantly, they are powering traffic predictions within Baidu Maps [6]."

The caption sits directly under the Pinterest figure. Notice at bottom centre, just left of the framed excerpt (the page does not say which figure "Illustration" refers to): "Illustration © J. Leskovec. Text © Derrow-Pinion, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © J. Leskovec (Pinterest pins-and-boards illustration) and © Derrow-Pinion, et al. (quoted text excerpt). All rights reserved — excluded from the CC license.*

## Slide 4 — Example: molecule property prediction

At top centre, a ball-and-stick style picture of a molecule: black balls (carbon-like) joined into rings, with bright green, yellow and teal balls as substituents; two ring clusters joined through a yellow and green chain. Below it, a small white box "Graph / Molecule" ("Graph" bold) with a thick dark-blue arrow to a larger white box: "Property" (bold), "Solubility", "Toxicity", "Drug efficacy", "...". Below, in green italics: "(Duvenaud et al, 2015, Stokes et al 2020,...)".

At the lower right, the cover of the journal Cell: dark-blue background with microbe shapes, the word "Cell" in cyan, and a red glowing shield shape. Beside it, a blue banner with white text: "On the cover: Antibiotic resistance is a pervasive public health problem, requiring the adoption of creative approaches to drug discovery. In this issue, ... Show more".

Notice at the bottom left: "Above © Enzymlogic on Flickr. Right © Cell Press. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Enzymlogic on Flickr (molecule image) and © Cell Press (Cell cover and banner). All rights reserved — excluded from the CC license.*

## Slide 5 — Example: Polypharmacy side effects

Top: a legend of three rows with small emoji faces. A blue pill labelled "Drug 1" with an arrow to a smiling face; an orange pill labelled "Drug 2" with an arrow to a smiling face; both pills together labelled "combined" with an arrow to a grimacing, distressed face.

Centre: a network drawing of 28 nodes in two colour families. Nine are teal-green "drug" nodes: four are dark-outlined and labelled with a letter inside and a name beside them — "D" Doxycycline, "S" Simvastatin, "C" Ciprofloxacin, "M" Mupirocin — and five are pale and unlabelled. Nineteen are orange "protein" nodes: four dark-outlined (lower left, linked to each other and to C by thick black edges) and fifteen pale. Pale nodes and edges are grey-faded; the highlighted edges are black. Black edges join D to C (labelled "r" with subscript 2), S to C (also labelled "r" with subscript 2) and C to M (labelled "r" with subscript 1). C is also joined by four black edges to the four dark orange protein nodes. Each of D, S, C, M and the four dark orange nodes has a small stacked-bars icon (a rectangle divided into horizontal rows, an attribute vector) beside it. A thick red ellipse surrounds the C–M edge and its label. At the right, "Drugs" in teal bold and "Proteins" in orange bold.

Bottom left: a white box "Pair of Nodes / Drugs" ("Pair of Nodes" bold) with a thick blue arrow to a box "Edge / Interaction type" ("Edge" bold). Bottom right in green italics: "(Zitnik et al, 2018)".

Notice: "© Zitnik, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Zitnik, et al. (drug–protein network figure). All rights reserved — excluded from the CC license.*

## Slide 6 — Example: Predicting traffic times

Upper right: a flow diagram of coloured circles and arrows. Top row, left to right: a teal circle "Anonymised travel data", an arrow labelled "Analysed", a teal circle "Supersegments", an arrow labelled "Training data", a teal circle "Graph neural network", then a bent arrow labelled "Predictions" down into a pink-red circle "Routes ranked by ETA" (centre right). A dark-blue circle "Google Maps API" is top right; a dark-blue circle "Google Maps app" is bottom right; from the pink circle a bracket-shaped arrow labelled "Surfaced" goes up to the API circle and down to the app circle. A purple circle "Google Maps routing system" at bottom centre sends an arrow, labelled "Candidate user routes A→B", up into the pink circle.

Lower left: a framed figure, a dotted world map with percentages at city labels (for example 34% near Orlando and Washington DC, 37% Osaka, 51% Taichung City, 43% Sydney, 31% Singapore, 29% Denver, 27% Chicago, 26% Toronto, 21% New York, 22% San Jose and Las Vegas, 16% London and Copenhagen, 21% Berlin and Bangkok, 20% Chennai, 22% Jakarta, 23% Sao Paulo). Its caption is cut off by the slide's bottom edge: "Figure 1: Google Maps estimated time-of-arrival (ETA) prediction improvements for several world regions, when using our deployed graph neural network-based estimator. Numbers represent relative reduction in negative ETA outcomes compared to the prior approach used in production. A negative ETA outcome occurs when the ETA error from the ob- served travel duration is over some threshold and acts as a ..." (the last lines are clipped).

Notice at the bottom right: "Figures © Paulo Estriga & Adam Cain. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". Below it, a URL line, overlapping the slide number: "https://deepmind.com/blog/article/traffic-prediction-with-advanced-graph-neural-networks".

*OCW notice: © Paulo Estriga & Adam Cain (flow diagram and figures). All rights reserved — excluded from the CC license.*

## Slide 7 — Example: learning to simulate physics

A strip figure labelled "(a)". Four photo-like renderings of a glass box on a table. Left to right: a blue block of tiny particles standing in the box, labelled "X" with superscript "t" subscript 0 ($X^{t_0}$); the block collapsing and spreading; a splashing wave of particles; and a settled, sloshing pool, labelled $\tilde{X}^{t_K}$. Dashed arrows labelled $s_\theta$ join the first to the second and the third to the fourth. Between the second and third images, a block diagram labelled "Learned simulator, $s_\theta$": an arrow enters from the left and splits, one branch going straight to an oval "Update" and one through a box $d_\theta$ into the oval; an arrow leaves the oval to the right. Two thin grey lines open downward from the box $d_\theta$ (the start of a zoom-in that slide 8 completes). The rest of the slide is blank.

Notice at the bottom right: "© Sanchez-Gonzalez, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/". Citation in green italics: "(Sanchez-Gonzalez et al, 2020)".

*OCW notice: © Sanchez-Gonzalez, et al. (learned-simulator figure). All rights reserved — excluded from the CC license.*

## Slide 8 — Example: learning to simulate physics

Build step: slide 7's strip (a) again, with the same two short grey lines below $d_\theta$ as on slide 7; below a blank gap a lower strip of three panels appears, and a box is added at the bottom. The lower panels:

- "(c) Construct graph": at left, nine blue dots labelled $\mathbf{x}_ i$ for particles (one dot, middle, darker); an arrow to a small graph of grey nodes with one dark central node labelled $\mathbf{v}_ i^0$ whose edges are dark, one neighbour labelled $\mathbf{v}_ j^0$ and one edge labelled $\mathbf{e}_ {i,j}^0$, inside a light grey disc (the neighbourhood radius).
- "(d)" with a grey highlight label "Compute representation" (the label is the lecturer's addition, in a grey box overlapping the panel title): on a light grey background, two copies of a small graph with a dark central node, with grey double-headed arrows along the edges, the first labelled $\mathbf{v}_ i^m$ and $\mathbf{e}_ {i,j}^m$, an arrow to the second labelled $\mathbf{v}_ i^{m+1}$ and $\mathbf{e}_ {i,j}^{m+1}$.
- "(e) Extract dynamics info": a graph with a dark central node labelled $\mathbf{v}_ i^M$, an arrow to a set of nine red dots (matching the nine blue dots of (c)), one darker and labelled $\mathbf{y}_ i$.

(The superscripts and subscripts on these small labels are read from the page at slide scale and are small; treat them as approximate.) Bottom left: a white box "Collection of Nodes / Particles" ("Collection of Nodes" bold) with a thick blue arrow to a box "New state" (bold). Citation in green italics: "(Sanchez-Gonzalez et al, 2020)".

Notice: "© Sanchez-Gonzalez, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Sanchez-Gonzalez, et al. (learned-simulator figure). All rights reserved — excluded from the CC license.*

## Slide 9 — Example: (Combinatorial) Optimization

![Slide 9 — Example: (Combinatorial) Optimization](../images/05-architectures-graphs/slide-9.jpg)

Left bullet: "replace full algorithm or learn steps (e.g. branching decision)".

Left diagram: a weighted graph of 11 nodes. One node at the left is blue and labelled "source"; one at upper right is red and labelled "target"; the other nine are white. Edges carry integer weights, printed beside them: reading around the drawing, 2 (top edge, between the two upper-left white nodes), 1 (edge from the top-middle white node to the target), 3 (edge from the source up to the top-left white node), 2 (edge from the top-middle node to the second node down), 1 (edge from the source to that second node), 4 (edge from the top-middle node down to the middle node), 3 (edge from the second node down to a lower white node), 4 (edge from the source to the bottom-left white node), 2 (edge from the bottom-left white node to the lower white node), 1 and 1 (the two upper edges of a small triangle of three white nodes in the lower middle), 2 (the triangle's base), 5 (edge from the triangle's right node to the far-right white node) and 2 (edge from that far-right node up to the target). An unlabelled diagonal edge also joins the top-left white node to the second node down, and a second unlabelled edge joins the lower white node to the left node of the triangle: 16 edges, 14 of them weighted. (The weight-to-edge assignment was read by position and confirmed in the figure audit.) Below it, in italics: "“Neural Algorithmic Reasoning”", and in green italics to its right: "(e.g. Velickovic et al 2020)". Bottom left: a white box "Graph / Problem instance" ("Graph" bold) with a thick blue arrow to a box "Solution / decision" (bold) with "Path", "Branching variable", "...".

Right: a framed optimisation problem, a thick grey downward arrow, and a bipartite graph. The framed problem is

$$\min_{x} \thinspace c^\top x \quad Ax \leq b \quad l \leq x \leq u \quad x \in \mathbb{Z}^p \times \mathbb{R}^{n-p}$$

printed as four stacked lines: $\min_x c^\top x$, then $Ax \leq b$, then $l \leq x \leq u$, then $x \in \mathbb{Z}^p \times \mathbb{R}^{n-p}$. The bipartite graph has six circle nodes: three red on the left labelled $c_1, c_2, c_3$ and three blue on the right labelled $x_1, x_2, x_3$, with seven edges, joining $c_1$ to $x_1$ and $x_2$, $c_2$ to $x_1$, $x_2$ and $x_3$, and $c_3$ to $x_1$ and $x_3$. Labels beneath the columns: left "constraints" and "clauses", right "variables" and "variables". Green italics at the right: "(Gasse et al 2019)" beside the first row of labels and "(Selsam et al 2018)" beside the second.

## Slide 10 — Two goals

![Slide 10 — Two goals](../images/05-architectures-graphs/slide-10.png)

Text: "Input:" (blue bold) "Graph + attribute vector for each node (adjacency matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, feature matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$)".

Two diagrams. Left, headed "1. Node embeddings": a graph of four nodes (large coloured discs): red (upper left), green (upper right), green (lower left), blue (lower right). Edges: red–green(upper), red–blue (diagonal), green(upper)–blue, green(lower)–blue, so four edges. Each node has a small grey vertical bar beside it (an attribute vector; four bars). Four dashed grey curved arrows go from the nodes to four dots in a 2D coordinate plane (two axes with arrowheads): a red dot at upper right, two green dots close together just left of the vertical axis and above the horizontal axis, and a blue dot below the horizontal axis, also left of the vertical axis. Right, headed "2. Graph embedding": the same four-node graph inside a dashed rounded box, with one dashed arrow to a single dark-purple dot in a second coordinate plane.

Bottom, in red: "GNNs:" (bold) "learn a" "function" (bold) "from graph/neighborhood + node/edge attributes to vector".

## Slide 11 — Idea 1: fully-connected NN?

![Slide 11 — Idea 1: fully-connected NN?](../images/05-architectures-graphs/slide-11.png)

At the top right, a small copy of the slide 10 graph-embedding diagram (four-node graph in a dashed box, arrow to a purple dot in a coordinate plane).

Text: "Idea 1: Use the adjacency matrix as input to a neural network" ("Idea 1:" bold).

Two matrices side by side:

$$\mathbf{A} = \begin{pmatrix} 0 & 1 & 0 & 1 \cr 1 & 0 & 0 & 1 \cr 0 & 0 & 0 & 1 \cr 1 & 1 & 1 & 0 \end{pmatrix} \qquad \mathbf{P}\mathbf{A}\mathbf{P}^\top = \begin{pmatrix} 0 & 1 & 1 & 0 \cr 1 & 0 & 1 & 0 \cr 1 & 1 & 0 & 1 \cr 0 & 0 & 1 & 0 \end{pmatrix}$$

Then "We want:" and two blue bullets, each with an equation at the right:

- "Permutation invariance (graph embedding):" "(output: single vector) and" — $f(\mathbf{P}\mathbf{A}\mathbf{P}^\top, \mathbf{P}\mathbf{X}) = f(\mathbf{A}, \mathbf{X})$
- "Permutation equivariance (node embeddings):" "(output: one vector for each node)" — $f(\mathbf{P}\mathbf{A}\mathbf{P}^\top, \mathbf{P}\mathbf{X}) = \mathbf{P} f(\mathbf{A}, \mathbf{X})$

At the bottom right, a small picture of a coloured-rows column of four stripes (from top: blue, light green, red, dark green) with a grey arrow to a column of the same four stripes in a different order (dark green, red, blue, light green): a permutation of rows. ($\mathbf{P}$ is not defined on the slide; the reading as a permutation matrix follows from the labels.)

## Slide 12 — Idea 2: Images are like graphs...

![Slide 12 — Idea 2: Images are like graphs...](../images/05-architectures-graphs/slide-12.jpg)

Bullets: "Convolution? (local operator “encodes” local neighborhoods)"; then, below the figures, "What is different in a graph?" and "Commonalities:" followed in blue by "local operations, globalize through depth, weight sharing, input can have varying size".

Left figure: a grid graph of 16 nodes (4 by 4) joined by horizontal and vertical lines. Node colours, by row from the top: black, white, white, black; white, red (centre-ish), white, white; black, white, black, white; white, white, black, white. A green square outlines the top-left 3 by 3 block of nodes and has a pale fill. Inside it are the numerals "1", "2", "3", "4" near the red node's upper left, upper right, lower left and lower right ("1" in black bold, the others grey). A small vertical tick sits to the right of the grid, a remnant of the original figure.

Right figure: a graph of 19 nodes (small white ovals, black edges; on this slide a small, low-resolution picture) — the same layout and 24 edges as the slide 3 node-classification graph, drawn in plain white. A green outline with pale green fill encloses five nodes at the left: the left-hand node of degree four and its four neighbours (two white nodes at the far left, the node below it, and the top hub, which the outline reaches with a narrow arm). Whether the slide 3 and slide 12 graphs are the same drawing: the edge structure and node positions agree (see slide 15's note).

## Slide 13 — A CNN is a GNN over a grid graph

![Slide 13 — A CNN is a GNN over a grid graph](../images/05-architectures-graphs/slide-13.jpg)

Title (centred): "A CNN is a GNN over a grid graph".

Left diagram: a 6 by 6 grid of 36 circle nodes, each with a grey vertical rectangle (a small attribute bar) just to its left. The grid is split into four 3 by 3 blocks (two blocks across, two down); in each block the centre node is joined by black lines to its eight surrounding nodes (horizontal, vertical and diagonal). The four stars of eight lines are not joined to each other.

Right: "Review questions:" followed by bullets "What’s the kernel size?" and "What’s the stride?".

Bottom, centred: "GNN’s attribute vector per node == CNN’s column of channels at each index in a feature map".

## Slide 14 — Roadmap

(Agenda / signpost slide, same as slide 2 with the second item, "Message passing GNNs", in red.)

- Learning tasks with graphs
- Message passing GNNs (red)
- Approximation Power

## Slide 15 — Graph neural networks

![Slide 15 — Graph neural networks](../images/05-architectures-graphs/slide-15.png)

Diagram: a 19-node graph with a grey vertical bar beside each of 13 nodes (none beside the six plain white nodes). It is the same graph drawing (same node layout and edges) as slide 3's left graph and slide 12's right graph: the node counts match (19), the 24 edges are identical, the top hub with seven edges is at the top, the hub with seven edges at centre-right is where slide 3 has its red node, and the edges that join them agree. On this slide the nodes are coloured by region instead. A pale-green blob encloses five nodes at the left (the left-hand node of degree four and its four neighbours, including the top hub, which the blob reaches with a narrow arm), drawn with green outlines; a pale-blue blob encloses eight nodes at the right (the centre-right hub and its seven neighbours that lie inside the blob) with dark-blue outlines; six nodes stay plain white outside both blobs. A green arrow curves from the green blob to the first of four tall vertical bars at the right, a blue arrow from the blue blob to the second bar. The four bars are, left to right, green, blue, yellow, orange-yellow, followed by "...".

Blue box at the bottom:

"Idea:" (bold)
1. Encode each node (based on message passing between nodes)
2. Aggregate *set* of node embeddings into a graph embedding

## Slide 16 — Encoding neighborhoods: general form

![Slide 16 — Encoding neighborhoods: general form](../images/05-architectures-graphs/slide-16.png)

Left diagram: six large discs. A red disc in the middle has two blue discs above it (upper left, upper right), each with a black arrow pointing into the red disc, and a blue disc below it. Between the red disc and the lower blue disc there are two arrows side by side: a black one pointing up into red and a thin blue one pointing down towards the lower blue disc. Two dark-green discs sit below the lower blue disc, each sending a blue arrow up into the lower blue disc. A grey vertical bar sits by each of the two upper blue discs and by the lower blue disc, and a pink bar by the red disc. Rotated text beside the upper right: "node embedding".

Right text: "In each round $k$:" "Aggregate" (red bold) "information from neighbors":

$$\mathbf{m}_ {\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}^{(k)} \left( \lbrace \mathbf{h}_ u^{(k-1)} : u \in \mathcal{N}(v) \rbrace \right)$$

with a green arrow from the green italic label "feature description of node u in round k-1" up to $\mathbf{h}_ u^{(k-1)}$. Then "Update" (red bold) "current node representation by incorporating messages from neighbors":

$$\mathbf{h}_ v^{(k)} = \text{UPDATE}^{(k)} \left( \mathbf{h}_ v^{(k-1)}, \mathbf{m}_ {\mathcal{N}(v)}^{(k)} \right)$$

Footer citations in green italics: "(Merkwirth & Lengauer 2005; Scarselli et al 2009; Bruna et al 2014; Dai et al 2016; Battaglia et al., 2016; Defferrard et al., 2016; Duvenaud et al., 2015; Hamilton et al., 2017; Kearnes et al., 2016; Kipf & Welling, 2017; Gilmer et al 2017; Li et al., 2016; Velickovic et al., 2018; Verma & Zhang, 2018; Ying et al., 2018; Zhang et al., 2018; ...)" (the slide number overlaps the text "et al." in the middle).

## Slide 17 — What are the aggregation functions?

Left diagram: the three-neighbour message-passing picture of slide 16 without the lower layer: three discs, a red one in the middle and two blue ones above it (upper left and upper right) with black arrows pointing into red, and a third blue disc below red with a black arrow pointing up into red. Each blue disc has a grey vertical bar; the red disc has a pink bar.

Equations at the right:

$$\mathbf{m}_ {\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}^{(k)} \left( \lbrace \mathbf{h}_ u^{(k-1)} : u \in \mathcal{N}(v) \rbrace \right)$$

$$= \bigoplus_{u \in \mathcal{N}(v)} \psi^{(k)} \left( \mathbf{h}_ u^{(k-1)}, \mathbf{h}_ v^{(k-1)} \right)$$

where the symbol $\bigoplus_{u \in \mathcal{N}(v)}$ (a circled plus with $u \in \mathcal{N}(v)$ beneath) sits in a pale-blue rounded box and $\psi^{(k)}(\mathbf{h}_ u^{(k-1)}, \mathbf{h}_ v^{(k-1)})$ sits in a pale-green rounded box. Below:

$$\mathbf{h}_ v^{(k)} = \text{UPDATE}^{(k)} \left( \mathbf{h}_ v^{(k-1)}, \mathbf{m}_ {\mathcal{N}(v)}^{(k)} \right)$$

Bottom left: $\mathcal{N}(v) = \lbrace u \mid \exists (u, v) \in E(\mathcal{G}) \rbrace$ "node’s neighborhood".

## Slide 18 — What are the aggregation functions?

At the top right, the same three-neighbour diagram as slide 17, small.

"Aggregate" (red bold) ": permutation invariant, multi-set function". Then:

- "Sum, average, ..." with two formulas. The first, with its fraction $\frac{1}{|\mathcal{N}(v)|}$ drawn in faded light grey:

$$\mathbf{m}_ {\mathcal{N}(v)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_ u$$

  cited in green italics "(Merkwirth & Lengauer 2005, Scarselli et al 2009)". The second:

$$\mathbf{m}_ {\mathcal{N}(v)} = \sum_{u \in \mathcal{N}(v)} \frac{\mathbf{h}_ u}{\sqrt{|\mathcal{N}(v)||\mathcal{N}(u)|}}$$

  cited "(Kipf & Welling 2016, Hamilton et al 2017)".
- "Min / max" with "(coordinate-wise)" in italics, and

$$\mathbf{m}_ {\mathcal{N}(v)} = \max \left\lbrace \mathbf{h}_ u^{(k-1)} : u \in \mathcal{N}(v) \right.$$

  (As printed, the closing brace is missing.)

Then, in italics, "Can implement e.g. shortest path:"

$$d_v^{(k)} = \min_{u \in \mathcal{N}(v)} d_u^{(k-1)} + \text{cost}(u, v)$$

## Slide 19 — Learning a shortest path algorithm

![Slide 19 — Learning a shortest path algorithm](../images/05-architectures-graphs/slide-19.jpg)

Two code-like blocks drawn as stacked, staggered pink-to-purple rectangles with white bold text, headed "Bellman-Ford" (left) and "GNN" (right). Left: "for k = 1 ... |S| - 1:" (pink), indented "for u in S:" (magenta), further indented "d[k][u] = min_v d[k-1][v] + cost (v, u)" (purple; the $v$ is a subscript on "min"). Right: "for k = 1 ... GNN iter:" (pink), "for u in S:" (magenta), "h_u^(k) = Σ_v MLP(h_v^(k-1), h_u^(k-1))" (purple; $h_u^{(k)}$ with the $(k)$ as superscript, $\Sigma_v$ with $v$ subscript, as typeset on the slide). A dashed double-headed arrow joins the two purple lines. A thin arrow points up from the words "sum or max pooling" (bold) to the right-hand purple line.

## Slide 20 — Aggregation functions and updates

![Slide 20 — Aggregation functions and updates](../images/05-architectures-graphs/slide-20.jpg)

At the top right, the three-neighbour diagram again, with small hatched texture patches drawn on each of the three arrows (the arrows are now edge functions).

"Aggregate" (red bold) ": General form" "(universal approximation of multi-set functions):" (italic), with green citation "(Zaheer et al 2017, Qi et al 2017, Xu et al 2019)".

$$\mathbf{m}_ {\mathcal{N}(v)} = \text{MLP}_ 2 \left( \sum_{u \in \mathcal{N}(v)} \text{MLP}_ 1 (\mathbf{h}_ u, \mathbf{h}_ v) \right)$$

with $\text{MLP}_ 2$ and $\text{MLP}_ 1$ each in a pale-green rounded box. A pale-green framed note at the right: "Learned aggregation function" (italic).

"Update" (red bold) ": e.g."

$$\mathbf{h}_ v^{(k)} = \sigma \left( \mathbf{W}_ {\text{self}} \mathbf{h}_ v^{(k-1)} + \mathbf{W}_ {\text{neigh}} \mathbf{m}_ {\mathcal{N}(v)}^{(k)} + b \right)$$

with $\mathbf{W}_ {\text{self}}$ and $\mathbf{W}_ {\text{neigh}}$ each in a pale-green box, and two green arrows from the word "learned" (green) pointing up to the two boxes. (The slide writes the bias as a plain $b$, not bold.)

## Slide 21 — Graph embeddings

![Slide 21 — Graph embeddings](../images/05-architectures-graphs/slide-21.jpg)

At the top, next to the title, the small graph-embedding diagram of slide 10 (four-node graph in a dashed box, dashed arrow to a purple dot in a coordinate plane).

Main diagram: the same 19-node graph drawing as slide 15 (same layout: see slide 15), with the pale-green blob around five nodes at the left, the pale-blue blob around eight nodes at the right, a grey bar by each of those 13 nodes, and six plain white nodes outside. A green arrow goes from the green blob to the first of four tall bars (green, blue, yellow, orange-yellow), a blue arrow from the blue blob to the second bar. After the bars, "..." and a long black arrow to $\mathbf{h}_ {\mathcal{G}}$.

Below the arrow:

$$\mathbf{h}_ {\mathcal{G}} = \text{READOUT} \left( \lbrace \mathbf{h}_ v^{(K)} : v \in \mathcal{G} \rbrace \right)$$

and in italics "pooling operation (just like AGGREGATE)".

Blue box at the bottom, as on slide 15 but with item 2 now in bold: "Idea:" (bold); "1. Encode each node (based on message passing between nodes)"; "2. Aggregate *set* of node embeddings into a graph embedding" (bold).

Build note: relative to slide 15 this adds the title "Graph embeddings", the top diagram, $\mathbf{h}_ {\mathcal{G}}$, the READOUT equation, and the bolding of item 2.

## Slide 22 — GNNs unrolled

![Slide 22 — GNNs unrolled](../images/05-architectures-graphs/slide-22.png)

Left: a four-node directed graph: a teal disc (upper left), a yellow disc (upper right), a red-orange disc (middle) and a blue disc (below the red). Arrows: teal to red, yellow to red, blue to red, and teal to blue. Each disc has a small vertical bar of its own colour beside it.

Right: the unrolled computation, drawn bottom to top, columns for the four nodes (teal, red-orange, blue, yellow bars). From the bottom: dotted and solid arrows converging from below ("AGGREGATE" with superscript (1)) into four pairs of bars (each a coloured bar overlapped by a black bar); a vertical arrow from each pair up to a row of four plain coloured bars ("UPDATE" with superscript (1)); then solid and dotted arrows between the columns (the teal bar sends solid arrows to the red-orange and blue columns, the yellow bar to the red-orange column, and so on; read as "AGGREGATE" with superscript (2)) into a second row of four coloured-plus-black bar pairs; then four arrows upward ("UPDATE" with superscript (2)) and a vertical ellipsis above. Right-hand labels, top to bottom: $\text{UPDATE}^{(2)}$, $\text{AGGREGATE}^{(2)}$, $\text{UPDATE}^{(1)}$, $\text{AGGREGATE}^{(1)}$.

Bullets:

- "Like an MLP, but nodes are vectors rather than scalars, edges are potentially complex functions (e.g., an edge can be an MLP)"
- "Each iteration of GNN message passing is a layer"
  - "AGGREGATE is akin to a linear layer"
  - "UPDATE is akin to a pointwise layer"

## Slide 23 — What is the graph for an MLP?

![Slide 23 — What is the graph for an MLP?](../images/05-architectures-graphs/slide-23.png)

Text: "An MLP is a GNN over single node".

Left: a single grey vertical bar beside one circle (a one-node graph with its attribute vector). A large block arrow labelled "unroll" points right. Centre: a vertical stack, bottom to top, of three small bars each containing three stacked circles (the neurons of a layer as a vector), joined by two upward arrows labelled $\text{UPDATE}^{(1)}$ (between the bottom and middle bars) and $\text{UPDATE}^{(2)}$ (between the middle and top bars), then an arrow up to a vertical ellipsis.

Right: "Update:" (red bold) and

$$\mathbf{h}_ v^{(k)} = \sigma \left( \mathbf{W}_ {\text{self}} \mathbf{h}_ v^{(k-1)} + \mathbf{W}_ {\text{neigh}} \mathbf{m}_ {\mathcal{N}(v)}^{(k)} + b \right)$$

with $\mathbf{W}_ {\text{self}}$ and $\mathbf{W}_ {\text{neigh}}$ each in a pale-green box.

## Slide 24 — Generalizations

Bullets, with equations:

- "Use **edge attributes** / features in aggregation", with equation at the upper right:

$$\mathbf{m}_ {\mathcal{N}(v)} = \sum_{u \in \mathcal{N}(v)} \text{MLP}^{(k)} \left( \mathbf{h}_ u^{(k-1)}, \mathbf{h}_ v^{(k-1)}, \mathbf{w}_ {uv} \right)$$

- "**Multi-relational**: multiple “channels” *different aggregations for different types of edges*" (the second part in italics).
- "**Attention** (Velickovic et al 2018):" (citation in green italics) and below it

$$\mathbf{m}_ {\mathcal{N}(v)} = \sum_{u \in \mathcal{N}(v)} \alpha_{v,u} \mathbf{h}_ u$$

with $\alpha_{v,u}$ in blue.
- "Janossy pooling (Murphy et al 2018)" (citation in green italics) then "**permutation-sensitive** function averaged over permutations" (italic, with "permutation-sensitive" bold).

At the right, a copy of the slide 5 drug and protein network (same 28 nodes, Doxycycline, Simvastatin, Ciprofloxacin and Mupirocin, red ellipse around the C–M edge, "Drugs" and "Proteins" labels), smaller, with the green italic citation "(Zitnik et al, 2018)".

Notice: "© Zitnik, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Zitnik, et al. (drug–protein network figure). All rights reserved — excluded from the CC license.*

## Slide 25 — Node embeddings: tree view

Left, headed by "TARGET NODE" with a down arrow onto node A: the "INPUT GRAPH" (label below it, bold), a graph of six lettered, coloured discs: A (gold), B (red), C (green), D (blue), E (purple), F (pink). Seven grey edges: A–B, A–C, A–D, B–C, C–F, C–E and E–F. Three dotted black arrows point into A: one along the A–B edge (from B), one along the A–C edge (from C), one along the A–D edge (from D).

Right: the same message flows drawn as a tree: a gold disc A at the left, a thick black arrow into it from a large light-grey square; three dotted grey arrows enter the square from three discs at the right, B (red, top), C (green, middle) and D (blue, bottom).

Text at the bottom left: "grey boxes: aggregation functions that we learn".

Notice at the bottom right: "Illustration © J. Leskovec. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/", with the credit "(illustrations: J. Leskovec)" in italics beneath.

*OCW notice: © J. Leskovec (input graph and tree-view illustrations). All rights reserved — excluded from the CC license.*

## Slide 26 — Node embeddings: tree view

Build step of slide 25. Left: the same input graph and "TARGET NODE" label; the dotted arrows into A are gone and two new ones are shown, one along A–B pointing towards B and one along B–C pointing towards B (from C). Right: slide 25's tree, plus a second, small dark-grey square at the upper right that feeds B (a thick black arrow from the square to B), with two dotted arrows into that square from a gold A disc and a green C disc at the far right. (B's own neighbours are A and C.)

Same text "grey boxes: aggregation functions that we learn", same notice and credit as slide 25.

*OCW notice: © J. Leskovec (input graph and tree-view illustrations). All rights reserved — excluded from the CC license.*

## Slide 27 — Node embeddings: tree view

Build step of slide 26. The input graph at the left is as on slide 25 but with no dotted arrows. The tree at the right now has all three first-level boxes: the B box fed by A and C (as on slide 26); a C box (small dark square) feeding C, fed by four discs A (gold), B (red), E (purple) and F (pink); and a D box feeding D, fed by a single gold A disc. Labels: "layer 2" (bold, beside the large grey box), "layer 1" (beside the B, C and D discs), "layer 0 (input)" (above the input discs), and "shared weights" in green with two green arrows from it, one pointing to the B box and one pointing down to the C box.

Same "grey boxes: aggregation functions that we learn", notice and credit.

*OCW notice: © J. Leskovec (input graph and tree-view illustrations). All rights reserved — excluded from the CC license.*

## Slide 28 — Weight sharing

Bullets: "We use the same aggregation functions for all nodes." and "So we can generate encodings for previously unseen nodes & graphs too!" (the words "previously unseen nodes & graphs" in red) "(dynamic graphs, different molecules, ...)".

Top right: two computation trees, captioned "Compute graph for node A" (bold, left) and "Compute graph for node B" (right). Tree for A: a gold disc at the top fed by a light-grey box, which is fed by three discs (blue, green, red, in that order from left to right); under each is a dark-grey box with its inputs: under the blue disc, one gold disc; under the green disc, four discs (gold, red, purple, pink); under the red disc, two discs (gold, green). Tree for B: a red disc at the top fed by a light-grey box, fed by two discs (gold, green); under the gold disc a dark box with three inputs (blue, green, red); under the green disc a dark box with four inputs (gold, red, purple, pink). Dotted red rounded boxes labelled "shared parameters" surround the top (light-grey) boxes of the two trees (one box in a long dotted frame that spans both) and, separately, the dark first-level boxes of both trees.

Lower right: a small graph of nine nodes, eight in grey and one in purple, with black edges between the grey nodes; the purple node is joined by two dotted purple edges to two grey nodes (a new node added to the graph).

Notice at the bottom right: "Illustration © J. Leskovec. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/", with "Illustrations: J. Leskovec" in italics beneath.

*OCW notice: © J. Leskovec (compute-graph and small-graph illustrations). All rights reserved — excluded from the CC license.*

## Slide 29 — Training a GNN

![Slide 29 — Training a GNN](../images/05-architectures-graphs/slide-29.jpg)

Bullets: "**What is a data point?**"; "**What to specify?**" with sub-bullets "Aggregate, updates and readout functions" and "Loss function on prediction"; "**Train** with SGD".

Two small graph drawings, each of 19 nodes. Left, captioned "{node, label} pairs" (italic): the same graph drawing as slides 3, 12, 15 and 21, with a pale-blue blob around the centre-right hub and its neighbours (eight nodes inside) and a grey bar by each of those eight nodes. Right, captioned "{graph, label} pairs": the same 19-node graph, plain, with no blob and no bars. (Layouts are the same drawing; both are small, and each was counted as 19 nodes.)

## Slide 30 — Example architecture 1: polypharmacy

Bullet: "Different types of edges: drug-drug, drug-protein, protein-protein".

Top right: a small copy of the slide 5 drug and protein network (labels Doxycycline, Simvastatin, Ciprofloxacin, Mupirocin; red ellipse; "Drugs", "Proteins").

Left diagram: three black-framed boxes stacked, each with a small dark-grey square (a weight matrix) and a black vertical bar (its output), all feeding a light-grey vertical bar labelled $\phi$, which has an arrow to a teal triangle "C" labelled $\mathbf{h}_ c^{(k+1)}$.

- Top box, caption "r" with subscript 1, "Gastrointestinal bleed effect": a teal triangle "M" with a small hatched edge icon and an arrow into a grey square labelled $\mathbf{W}_ {r_1}^{(k)}$; the square's output arrow goes to the black bar and is labelled $\mathbf{h}_ {\mathcal{N}_ {r_1}^c}^{(k)}$; a teal triangle "C" labelled $\mathbf{h}_ c^{(k)}$ also sends an arrow to the black bar.
- Middle box, caption "r" with subscript 2, "Bradycardia effect": teal triangles "S" and "D", each with a hatched edge icon, send arrows into a grey square labelled $\mathbf{W}_ {r_2}^{(k)}$, whose output (labelled $\mathbf{h}_ {\mathcal{N}_ {r_2}^c}^{(k)}$) goes to a black bar; the triangle "C" labelled $\mathbf{h}_ c^{(k)}$ also feeds the bar.
- Bottom box, "Drug target relation": four orange discs (proteins), each with a hatched edge icon, send arrows into a grey square labelled $\mathbf{W}_ t^{(k)}$, whose output (labelled $\mathbf{h}_ {\mathcal{N}_ t^c}^{(k)}$) goes to a black bar.

The three black bars each have an arrow to the $\phi$ bar.

Equation, with everything after $\sum_r$ (the inner sum and both terms) in a pale-blue box:

$$\mathbf{h}_ v^{(k+1)} = \text{ReLU} \left( \sum_r \sum_{u \in \mathcal{N}_ r(v)} c^{vu} \mathbf{W}_ r^{(k)} \mathbf{h}_ u^{(k)} + c_r^v \mathbf{h}_ v^{(k)} \right)$$

(As printed, the first coefficient is written $c^{vu}$ with no subscript $r$, while the second is $c_r^v$; this may be a slip or may be deliberate.) Below it, in italics: "separate aggregation per edge type r, then sum them up".

Lower right, a framed paper title: "Modeling Polypharmacy Side Effects with Graph Convolutional Networks" / "Marinka Zitnik 1, Monica Agrawal 1 and Jure Leskovec 1,2,*" (the digits as superscripts).

Notice: "© Zitnik, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Zitnik, et al. (architecture diagram, network figure and paper title). All rights reserved — excluded from the CC license.*

## Slide 31 — Example architecture 2: Google Maps

Bullets:

- "segments $v$ (50-100m), supersegments $u$ (ca 20 segments);" then "input features: historical travel times, road type, traffic pattern at prediction time"
- "3 aggregation/update operations: $\mathbf{h}_ v$’s (using adjacent edges $\mathbf{h}_ e$, $\mathbf{h}_ u$), edges $\mathbf{h}_ e$ (using adjacent segments $\mathbf{h}_ v$ and $\mathbf{h}_ u$), $\mathbf{h}_ u$ (using segments $\mathbf{h}_ v$ and edges $\mathbf{h}_ e$ in supersegment)"
- "combination of pooling operations for each aggregation"
- "linear combination of losses per time horizon (segment travel time, super segment time, cumulative segment time, self-supervised / generative loss)"

Bottom: three pictures. Left and centre: two stylised road-junction maps in pale blue with blue road outlines and red dashed lines cutting the road into short segments; the centre one adds several pale circles (vehicles) on the segments. Right: a graph of 8 dark-blue discs joined by blue lines: a chain A – disc – disc, then a branching hub; specifically A, a second disc and a third disc in a row at the top, a fourth disc at upper right, a hub disc below them connected to the third and fourth discs, a disc to its right connected on to a disc labelled B, and one disc hanging below the hub. Each disc has a small four-cell shaded bar (feature vector) below it with an arrow pointing up to the disc.

Citation (green italics, overlapped by the slide number): "(Derrow-Pinion et al., ETA Prediction with Graph Neural Networks in Google Maps, CIKM 2021)".

Notice at the left, in small text: "© Derrow-Pinion, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Derrow-Pinion, et al. (road-segment and supersegment graph illustrations). All rights reserved — excluded from the CC license.*

## Slide 32 — Many connections

At the top right, the three-neighbour message-passing diagram (red disc, three blue discs).

Bullets (headings in blue bold):

- "**Graph signal processing and convolutions**"
- "**Inference in graphical models** (Dai et al 2016)" — sub-bullets: "Node embeddings = latent variables"; "Given node features and graph, infer latent variables"; "“Neural message passing”"
- "**Distributed / Local algorithms** (Sato et al 2019, Loukas 2020)" — sub-bullet: "Bounds for detection, verification, computation with GNNs"
- "**Random walks** (Xu et al 2018)" — sub-bullets: "Oversmoothing, graph structure and depth"; "(Adaptive) skip connections"
- "**Graph isomorphism testing** (Morris et al 2019, Xu et al 2019)"

(The citations are in smaller italic type after each heading.)

## Slide 33 — Roadmap

(Agenda / signpost slide, same as slide 2 with the third item, "Approximation Power", in red.)

- Learning tasks with graphs
- Message passing GNNs
- Approximation Power (red)

## Slide 34 — Which functions can GNNs approximate?

Top left, a small diagram. Two small graphs with shaded, gradient-coloured nodes: a star of five nodes (a teal centre joined to a pink node above, a green node to the left, a pink node to the right and a green node below) and, beneath it, a path of three nodes (green, teal, pink). To the right, a coordinate plane (two axes) containing two blue dots, one upper (near the top of the vertical axis) and one lower right. A blue dashed arrow goes from the star to the upper dot; a blue dashed arrow goes from the path to the lower dot; a grey, fainter dashed arrow goes from the path to the upper dot.

Right of the diagram, in red: "Which graphs can GNNs distinguish?" and below it $f(G) \neq f(G')$? (printed as $f(G) \neq f(G')?$).

The rest of the slide is blank.

## Slide 35 — Which functions can GNNs approximate?

![Slide 35 — Which functions can GNNs approximate?](../images/05-architectures-graphs/slide-35.jpg)

Build step of slide 34: the same top-left diagram and question, plus:

Left text: "**Distinction implies function approximation**" (bold) "for node and graph predictions"; "(Symmetric Stone-Weierstrass theorem)"; green citation "(Azizian & Lelarge 21, Chen-Villar-Chen-Bruna 19, Keriven & Peyré 19, Maron-Fetaya-Segol-Lipman 19)".

Right: a figure of seven small coloured-node graphs grouped into five blue-outlined blobs. The node colours are green, grey, red and yellow. From top left: one blob holds two four-node graphs (the first: a green node attached to a grey node, which is joined to a red node and a yellow node, with the yellow also joined to the red; the second: a green, a grey, a red and a yellow node with green–grey, green–red, grey–red and grey–yellow edges); a blob at the top right holds one five-node graph (yellow, grey, green, green, yellow, with eight edges); a small blob at the left holds one four-node graph like the first; a blob at the bottom centre holds a four-cycle (grey, green, red, grey); and a large blob at the lower right holds two five-node graphs. (Edge detail for these small graphs was read at slide scale; the point is that the graphs inside a blob are not distinguishable by the functions in the class, and each blob is an equivalence class.)

Below the figure:

$$(G, G') \in \rho(\mathcal{F}) \iff \forall F \in \mathcal{F}, \thinspace F(G) = F(G')$$

Box at the bottom, headed "Theorem." (blue): "If function $H$ on a compact domain does not assign different labels to graphs in one equivalence class, then it can be approximated by message passing GNNs:"

$$\forall \epsilon \gt 0, \thinspace \exists F \in \mathcal{F}^{\text{GNN}} : \sup_{G \in K} \lVert H(G) - F(G) \rVert \leq \epsilon$$

(The slide number overlaps the lower edge of the box.)

## Slide 36 — Discriminative Power

![Slide 36 — Discriminative Power](../images/05-architectures-graphs/slide-36.png)

Top row: a four-node graph (red top left, green top right, green bottom left, blue bottom right; edges red–green(top), red–blue (diagonal), green(top right)–blue, green(bottom left)–blue), a grey arrow, and three small trees followed by "...". Tree 1 has a red root with two children, green and blue; under the green child, a red and a blue leaf; under the blue child, green, green and red leaves. Tree 2 has a green root with children red and blue; under the red child, green and blue leaves; under the blue child, green, green and red leaves. Tree 3 has a green root, one blue child, and three leaves under the blue: green, green, red.

Middle and bottom, left: a blue oval labelled "Equivalence class" (blue, upper left) containing two six-node graphs. The upper graph is a two-by-three ladder: red (bold outline, top left), yellow, red along the top; red, yellow, red along the bottom; edges join neighbours along the rows and the yellow nodes are joined vertically, and the two red nodes at the left and the two at the right are each joined vertically. The lower graph is a "bow tie": two triangles, each of two reds and a yellow, with the two yellows joined (the top-left red has a bold outline). Each graph has a grey arrow to a tree rooted at the outlined red: root (red, outlined), children red and yellow; under red a red and a yellow leaf; under yellow a red, a yellow and a red leaf; "..." follows. The two trees are identical, and each matches its graph. (Each graph in this oval has 6 nodes and 7 edges.)

Right: a second blue oval, labelled "Equivalence class" (blue, lower right), containing two six-node red graphs, each with 9 edges and every node of degree 3: at left a triangular prism, an outer triangle (top, bottom-left and bottom-right nodes) and an inner triangle joined by three spokes; at right $K_{3,3}$, three nodes in each of two columns with all 9 edges between the columns.

Box at the bottom: "Theorem" (blue bold) "(Morris-Ritzert-Fey-Hamilton-Lenssen-Rattan-Grohe 19, Xu-Hu-Leskovec-Jegelka 19)" (green italics) and "Any GNN can at best distinguish the same graphs as the 1-dim WL algorithm."

## Slide 37 — Color refinement / Weisfeiler-Leman algorithm

Top: the four-node graph and three trees of slide 36 (small), with "..." and, at the upper right, green italics "(Morgan 65, Weisfeiler & Leman 68)".

Text: "coloring" followed by $c^{(t)} : V(G) \to \Sigma$. Then

$$c^{(t)}(v) = \text{Hash} \left( c^{(t-1)}(v), c^{(t-1)}(u) \mid u \in \mathcal{N}(v) \right)$$

with "Hash" in a pale-green box. Then "isomorphism test:"

$$\lbrace c^{(t_\infty)}(v) \vert v \in V(G) \rbrace \neq \lbrace c^{(t_\infty)}(v') \vert v' \in V(G') \rbrace ?$$

## Slide 38 — Color refinement / Weisfeiler-Leman algorithm

Build step of slide 37: same graph, trees, citation, "coloring" line and Hash equation. Added: "vs GNN:" and

$$\mathbf{h}_ v^{(t)} = f_{\text{Update}} \left( \mathbf{h}_ v^{(t-1)}, f_{\text{Agg}} \left( \lbrace \mathbf{h}_ u^{(t-1)} \mid u \in \mathcal{N}(v) \rbrace \right) \right)$$

with $f_{\text{Update}}$ in a blue box and $f_{\text{Agg}}(\lbrace \mathbf{h}_ u^{(t-1)} \mid u \in \mathcal{N}(v) \rbrace)$ in a pink box.

## Slide 39 — Color refinement / Weisfeiler-Leman algorithm

![Slide 39 — Color refinement / Weisfeiler-Leman algorithm](../images/05-architectures-graphs/slide-39.png)

Build step of slide 38: everything on slide 38, plus a framed box at the bottom: "Theorem" (blue bold) "(Morris-Ritzert-Fey-Hamilton-Lenssen-Rattan-Grohe 19, Xu-Hu-Leskovec-J 19)" (green italics; "-J" abbreviates "Jegelka" as on slide 36). "Any GNN can at best distinguish the same graphs as the 1-dim WL algorithm. For any $n$, there exists a GNN such that for any $t$, $c^{(t)} \equiv h^{(t)}$." (As printed, $h^{(t)}$ is not bold and has no node subscript.)

## Slide 40 — How could we ensure that the aggregation is injective?

- "Any (multi-)set function can be represented with nonlinear functions $g_1, g_2$ as:"

$$f_{\text{Agg}}(S) = g_1 \left( \sum_{\mathbf{h} \in S} g_2(\mathbf{h}) \right)$$

(As printed, only the closing parenthesis is visible; the opening one after $g_1$ is missing from the render.)

- "We can universally approximate $g_1$ and $g_2$ by MLPs! (see a few slides ago)"

$$\mathbf{m}_ {\mathcal{N}(v)}^{(k)} = \text{MLP}_ 2 \sum_{u \in \mathcal{N}(v)} \text{MLP}_ 1 \left( \mathbf{h}_ u^{(k-1)} \right)$$

## Slide 41 — Does this make a difference in practice?

![Slide 41 — Does this make a difference in practice?](../images/05-architectures-graphs/slide-41.jpg)

Bullet: "vary $g$ and pooling operation". Upper right, the equation of slide 40:

$$f_{\text{Agg}}(S) = g_1 \left( \sum_{\mathbf{h} \in S} g_2(\mathbf{h}) \right)$$

(same missing opening parenthesis).

Chart titled "PROTEINS". x-axis "Epoch", ticks 0, 50, 100, 150, 200, 250, 300, 350. y-axis "Training accuracy", ticks 0, 0.4, 0.6, 0.8, 1.0, with a wavy break mark between 0 and 0.4 (the axis is truncated). **Eight** overlapping noisy series, none individually legended on the slide (the embedded figure's own legend is clipped off the page):

- a magenta flat line at exactly 1.0 across the whole range;
- red solid: starts near 0.75 at epoch 0, climbs with jitter to about 0.97 by epoch 100 and about 1.0 by epoch 200, and stays there to 350;
- orange solid: almost coincides with the red;
- orange dash-dot: about 0.75 at epoch 0, 0.8 at 100, 0.87 at 200, 0.9 at 300, ending near 0.91 at 350;
- light-blue solid: rises from about 0.75 to about 0.9 at 350, close to the orange dash-dot, with deep dips (to about 0.4 around epoch 30–40) in the first 150 epochs;
- light-blue dash-dot: ends about 0.81, also with deep early dips;
- green solid (lighter): about 0.75–0.78 from epoch 50 to 350, with a dip to about 0.60 near epoch 3;
- dark-green dash-dot: runs almost on top of the green solid, ending about 0.78, with deep downward spikes to about 0.43 at epoch 70, 0.48 near 86, 0.48 near 122 and 0.56 near 134.

The slide labels them with three coloured text callouts and arrows:

- Red "Sum — MLP (injective)": the arrow's tip ends at about 0.95 near epoch 336, in the gap just below the top bundle (red, orange solid and magenta), aimed at it.
- Orange "Sum — linear+ReLu": the tip (epoch about 284, about 0.89) touches both the orange dash-dot and the light-blue solid, which coincide there.
- "Mean/Max — MLP/linear+ReLu" ("Mean" in blue, "/Max" in green, the rest black): a green arrow whose tip (about 0.79) falls between the green pair (about 0.78) and the light-blue dash-dot (about 0.81).

(Values are read approximately from the plot; the series count, the callout tips and the spike values were measured in the figure audit.)

## Slide 42 — Learning structural graph properties

![Slide 42 — Learning structural graph properties](../images/05-architectures-graphs/slide-42.jpg)

"Can GNNs compute:" (bold) with bullets "the length of the shortest / longest cycle?", "diameter of the graph?", "the number of occurrences of a motif?".

At the right, a black-and-grey chemical structure drawing: a cluster of fused five- and six-membered rings at left (atoms labelled S, H, N within a grey rounded rectangle), a single ring in the middle with an N (in a small grey rounded rectangle), and a fused ring system at right with N and O labels, joined by single bonds and dotted vertical lines.

Box at the bottom: "Lemma" (blue bold) "(Garg et al 2020, Chen et al 2020)" (blue italics). "No! Message Passing GNNs (as discussed here) cannot compute these in general."

## Slide 43 — Improving discriminative power

Bullets: "**Positional encodings**"; sub-bullet "Add node input features that encode “position” in the graph"; "For instance: eigenvectors of the graph Laplacian (or its normalized versions)" beside

$$\mathbf{L} = \mathbf{D} - \mathbf{A}$$

(printed in bold italic, $\mathbf{L}$, $\mathbf{D}$ and $\mathbf{A}$ all bold), with italic labels "Diagonal matrix with node degrees" and "Adjacency matrix" below it. Two arrows point up into empty space above the formula, one rising through the $A$ glyph from just above the "Diagonal matrix…" label and one from the "Adjacency matrix" label; as printed, neither ends at $D$ or $A$. Then "adds global structural information"; "Challenge: ambiguities (sign flips, eigenvalue multiplicities)".

Top right: four small panels of the same molecule graph (hexagonal rings joined together), with nodes coloured by an eigenvector, titled with eigenvalues: $\lambda_1 = 0.037$ (top left), $\lambda_2 = 0.1$ (top right), $\lambda_{10} = 1.0$ (bottom left), $\lambda_{11} = 1.0$ (bottom right). A vertical colour bar at the right is labelled "Eigenvector $\phi$ colormap" with "max" (dark green) at the top, "0" (white) in the middle and "-max" (magenta) at the bottom. Citation (small green italics): "(Kreuzer, Beaini, Hamilton, Létourneau, Tossou 2021)".

Bottom right: a bar chart. y-axis "Test Error", ticks 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30; no x-axis labels. Three bars with error bars (values measured from the vector data in the figure audit): pale green 0.252 ± 0.007, pale blue 0.198 ± 0.011, pale red 0.121 ± 0.005. (The PDF holds further bars and a legend under white boxes; only these three bars are visible.) A green label "“standard”" with an arrow to the green bar; a black label "Laplacian PE (basic and improved)" with a blue arrow ending at the blue bar's level, to its right, and a red arrow ending at the red bar's top right. Text: "Task: Molecule regression (ZINC)" and "from: Lim-Robinson-Zhao-Smidt-Sra-Maron-J 22" (italics).

Notice at the bottom: "Graphs © Kreuzer, et al and Lim, et al. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/".

*OCW notice: © Kreuzer, et al and Lim, et al. (eigenvector figure and bar chart). All rights reserved — excluded from the CC license.*

## Slide 44 — Call back to CNNs: What if you don't want to be shift invariant?

![Slide 44 — Call back to CNNs: What if you don't want to be shift invariant?](../images/05-architectures-graphs/slide-44.png)

Title on two lines: "Call back to CNNs:" and "What if you *don’t* want to be shift invariant?" ("don’t" in italics).

1. "Use an architecture that is not shift invariant (e.g., MLP)"
2. "Add location information to the *input* to the convolutional filters — this is called **positional encoding**"

Diagram: three columns of eight circles each. Left column, labelled "pos": circles shading from light pink at the top to dark maroon at the bottom. Middle column, labelled "signal": grey values from top to bottom: mid grey, light grey, white, black, light grey, mid grey, light grey, white. Right column (no label), the output: white, light grey, dark grey, dark grey, light grey, mid grey, black, white. A blue square labelled $\mathbf{w}$ sits between the left two columns and the right column; lines run from the top pos circle (a curve) and from the top three signal circles into the square, and three lines fan out from the square to the second circle of the output column.

## Slide 45 — Summary

- Encodes graph structure and node/edge attributes
- Important: permutation invariance/equivariance
- Main idea: **message passing** and **aggregations**
- **Can take graphs of varying size and structure** (similar to CNNs)
- Connections: graph signal processing, graphical models, distributed computing, isomorphism testing, ...
- Representational enhancements: higher-order, node IDs/augmentation

## Slide 46 — Appendix

- "Graph Laplacian (unnormalized): degree matrix D - adjacency matrix A"

$$\mathbf{L} = \mathbf{D} - \mathbf{A}$$

$$\mathbf{D}_ {ii} = \deg(v_i) \text{ or } \sum_{(v_i, v_j) \in E} w_{ij} \quad \text{and} \quad \mathbf{D}_ {ij} = 0 \text{ for } i \neq j$$

- "normalized:" $\mathbf{I} - \mathbf{D}^{-1} \mathbf{A}$ "or" $\mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$

## Slide 47 — MIT OpenCourseWare end page

Not lecture content: OCW's appended end page. Text: "MIT OpenCourseWare", "https://ocw.mit.edu", "6.7960 Deep Learning", "Fall 2024", "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms". The page is a different size (4:3) from the slides and prints the number "47" at the bottom centre.
