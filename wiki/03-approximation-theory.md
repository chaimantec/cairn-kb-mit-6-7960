# Lecture 3 — Approximation Theory

**Lecturer:** Jeremy Bernstein ·
**Video:** [youtube.com/watch?v=ySaoWrv3T_Q](https://www.youtube.com/watch?v=ySaoWrv3T_Q) (83 min) ·
**Slides:** [`mit6_7960_f24_lec3.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec3.pdf)
(43 pages, handwritten; transcribed slide by slide in [`raw/slides/03-approximation-theory.md`](../raw/slides/03-approximation-theory.md)) ·
**Transcript:** [`raw/transcripts/03-approximation-theory.md`](../raw/transcripts/03-approximation-theory.md)

## What this lecture establishes

The lecture asks what class of functions a neural network can express, and whether the answer
helps decide between a wide network and a deep one. It proves one universal function
approximation theorem in full: a three-layer ReLU network can approximate any Lipschitz function
on the unit hypercube to any accuracy, but the number of neurons it needs grows exponentially with
the input dimension. It then proves a **depth separation**: a function that a deep, narrow ReLU
network computes with a few thousand neurons would need around $10^{50}$ neurons in a three-layer
network. Both results speak only to approximation. Neither says whether the network can be
trained or will generalize, and the lecture closes on how hard the width-versus-depth question is
to settle in practice, using the scaling-law experiments of Kaplan et al. and the Chinchilla
paper as examples.

The lecturer is candid about the status of the first result: "I'm not claiming that this is an
important result. I'm just saying it's something you can prove" (≈18:36). The point is to show a
complete example of such a theorem and its proof, and then to ask whether it matters.

The deck is handwritten, in several ink colours on a dark background, and parts of it were
written live; the lecturer also switched to a graphing website mid-lecture to build a rectangle
out of ReLUs by hand (≈32:41–35:01). He mentions "an annotated version of the slides on the
website where it's a bit more spelled out" (≈42:07); the OCW deck is the one transcribed here.

**Notation on this page.** The handwritten slides write vectors and matrices unbolded ($x$,
$w^T x$, $W_L$); this page bolds them, following the [course notation](notation.md), and keeps
plain letters for scalars and one-dimensional inputs. The course notation reserves $L$ for the
loss, but this lecture uses $L$ for two other things, as the slides do: the **Lipschitz
constant** in the first half, and, in the depth-separation section, a **layer number**, which
for the last layer is the network's **depth**. Neither use is a loss.

## The question: wider or deeper?

The lecture opens on slide 2, "Would you rather…": a wide, flat network labelled "scale width"
beside a tall, narrow one labelled "scale depth". A student answers "depth", for "no reason",
and the lecturer's own answer is "it's not really clear. How do you even answer a question like
that? I don't know" (≈0:00–0:46).

![Slide 2: two hand-drawn networks of green units and pink connections, a wide flat one labelled "scale width" and a tall narrow one labelled "scale depth"](../raw/images/03-approximation-theory/slide-2.jpg)

*Slide 2 — the opening question, repeated as slide 23 when the lecture returns to it.*

Slide 3 frames the lecture around the slogans the question provokes: "neural nets are universal
function approximators" — so "what architecture should I use?" — answered either by "three layers
are enough" or by "stack more layers!!!", and finally "then what should I do in practice?" The
lecturer does not "claim to answer all of these questions", only to "provide some beginning of a
framework" (≈0:46–1:32).

## The machine learning puzzle

Slide 4 splits machine learning into three pieces of a puzzle, each its own question (≈1:32–3:04):

1. **Approximation.** "Does there exist a neural net in my model family that fits the training
   data?" Can the architecture even represent the function we want?
2. **Optimization.** "If it does exist, can I find it?"
3. **Generalization.** "Does it work well on unseen data?"

Solving a machine learning problem needs all three, but "this lecture will focus mainly on the
first question". The puzzle returns twice: slide 34 places the depth-separation result in it, and
slide 37 uses it to explain why practical questions are hard.

## Two motivating problems

**A staircase that one ReLU cannot fit.** Slide 5 plots two classes in the plane: crosses fill the
top two rows and the left five columns of a grid, and circles fill the bottom-right $3 \times 3$
block, so the boundary between them is a staircase. Can it be classified by a one-layer model,

$$f(\mathbf{x}) = \text{relu}(\mathbf{w}^T \mathbf{x} + b),$$

where $\mathbf{x} = (x_1, x_2)$ is the input, $\mathbf{w}$ a weight vector and $b$ a bias? A
student, Matt, says no: "Don't you need two linear separation planes?" The lecturer agrees. There
is only one hyperplane, $\mathbf{w}^T \mathbf{x} + b = 0$, "and ReLU does not distort the shape of
the hyperplane. It just applies nonlinearity on the output. This is basically a linear separator.
And this data is not linearly separable", a recap of lecture 1 (≈3:50). The slide's second
candidate is a sum of ReLUs, $f(\mathbf{x}) = \sum_ i \alpha_ i \thinspace \text{relu}(\mathbf{w}_ i^T \mathbf{x} + b_i) + \beta_ i$
(the final bias carries an index $i$ as written; a single output bias may be meant). "I think we
can do it with two layers, but you may need to think about it" (≈4:36). The point is the shape of
the question: there is a family of models and a function to fit, and we ask whether the family
contains one that fits.

![Slide 5: a grid of pink crosses with a 3-by-3 block of green circles in the lower right, above the one-layer and two-layer ReLU formulas](../raw/images/03-approximation-theory/slide-5.png)

*Slide 5 — the boundary between crosses and circles is a staircase, not a line.*

**A fractal.** Slide 6 shows Weierstrass's function, "everywhere continuous" but "nowhere
differentiable" — "a classic example of a pathological function" (≈5:22). It is a fractal: "It's
defined recursively. It's self-similar. If you zoom into any piece of the function, it resembles
the whole function" (≈6:08). Can a neural network fit it, and if so, how big must the network be?
A student suggests "you would have to go to infinity", asymptotically. The lecturer's answer is
honest: "I don't know what the answer is to this", and "I suppose the answer is, maybe". He offers
it as a final-project idea: take Weierstrass's pathological examples "and see if you can fit them
with a neural net" (≈5:22–6:08).

![Slide 6: a plot titled "A pathological function of Weierstrass", a jagged self-similar black curve on x from −1 to 1, credited "Hrothgar, Chebfun"](../raw/images/03-approximation-theory/slide-6.jpg)

*Slide 6 — Weierstrass's function. The plot is credited on the slide to "Hrothgar, Chebfun".*

## Formalizing approximation

Slide 7 sets up the problem in general (≈6:08–8:30). Choose a **family of curves** $G$, the
functions we would like to approximate. We are free to choose it, for instance so as to
"exclude pathological functions" like Weierstrass's, or to require that its members are
differentiable. Choose a **family of neural networks** $F$: "a neural architecture could specify a
family of functions", such as all five-layer ReLU MLPs. Then ask: for any curve $g \in G$, does
there exist a network $f \in F$ with

$$\text{error}(f, g) \lt \epsilon$$

for a small number $\epsilon$ that we choose? The error itself has to be chosen too. Two examples
on the slide:

$$\text{error}(f, g) \triangleq \max_x |f(x) - g(x)| \qquad \text{or} \qquad \text{error}(f, g) \triangleq \int dx \thinspace |f(x) - g(x)|.$$

The first is the **$L_\infty$ error**, "the max discrepancy between the two functions anywhere on
the input space"; the second is the **$L_1$ error**, which integrates the absolute difference
"along the whole axis" (≈7:44–8:30). The theorem below uses the $L_1$ error.

## Lipschitz functions

The family $G$ chosen for this lecture is the Lipschitz-continuous functions, which the lecturer
thinks "it's nice to teach you about … regardless of neural net approximation" (≈9:16). See also
[Lipschitz continuity](lipschitz-continuity.md).

**In one dimension** (slide 8). A function $g: \mathbb{R} \to \mathbb{R}$ is **$L$-Lipschitz** if

$$|g(x + \Delta x) - g(x)| \le L |\Delta x| \quad \text{for all } x \in \mathbb{R} \text{ and all } \Delta x \in \mathbb{R}.$$

$L$ is a number, "like, 10 or 5", that "measures how Lipschitz the function is": changing the input
by $\Delta x$ changes the output by at most $L$ times $|\Delta x|$ (≈9:16–10:04). Asked what this
looks like as $\Delta x$ becomes small, a student answered that "the slope is always less than some
constant $L$". That is the slide's intuition, "the slope of $g$ cannot exceed $L$": "you should
see a definition of the derivative hiding in here", so Lipschitz continuity is "a kind of
generalization of the notion of having a bounded derivative. Well, it may be equivalent actually,
but you need to think about that" (≈10:04–10:49).

The picture on the slide: if $g$ passes through the origin, it "can never stray outside $\pm Lx$".
Draw the lines $y = Lx$ and $y = -Lx$, and the function must stay between them. The same holds at
every point of the curve, so a cone, "or a kind of bow tie", slides along the function and the
function always lies inside it (≈10:49–11:34).

![Slide 8: the definition of an L-Lipschitz function, with a green curve through the origin staying inside the wedge between the dashed lines Lx and −Lx](../raw/images/03-approximation-theory/slide-8.png)

*Slide 8 — the Lipschitz "bow tie": through any point, the curve stays inside the wedge between the lines Lx and −Lx.*

**In $d$ dimensions** (slide 9). For $g: \mathbb{R}^d \to \mathbb{R}$, the output is still a
number, but the input change $\Delta \mathbf{x}$ is now a vector, so its size needs a norm. The
lecturer picks one "that I really like", the **RMS norm**:

$$|g(\mathbf{x} + \Delta \mathbf{x}) - g(\mathbf{x})| \le L \Vert \Delta \mathbf{x} \Vert_{\text{RMS}}, \qquad \Vert \mathbf{x} \Vert_{\text{RMS}} \triangleq \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}.$$

The RMS norm is the root-mean-square size of the entries, equivalently $1/\sqrt{d}$ times the
Euclidean norm (≈13:09). He calls the Euclidean norm "kind of a dimensional object" and the RMS
norm "a kind of non-dimensional analog": a vector of all ones has RMS norm 1 but Euclidean norm
$\sqrt{d}$. The point of the slide is to generalize Lipschitzness to many inputs "and to point out
that there's many different possible ways to measure the size of something" (≈13:57).

## The theorem

Slide 10 states what the first half of the lecture proves (≈14:43–15:30):

> **Theorem.** Let $g: [0,1]^d \to \mathbb{R}$ be any $L$-Lipschitz function. Then for any error
> $\epsilon \gt 0$ there exists a 3-layer ReLU network $f$ with $N = 4d(L/\epsilon)^d$ units such
> that
>
> $$\int_{[0,1]^d} |f(\mathbf{x}) - g(\mathbf{x})| \thinspace d\mathbf{x} \lt 2\epsilon.$$

$[0,1]^d$ is the **$d$-dimensional unit hypercube**: "raising it to the power $d$ means that in
all the dimensions of our $d$-dimensional space, we pick out that interval" (≈16:17). The error is
the $L_1$ error, "summing up all the little pieces and adding up all the absolute values of the
difference between our neural net and the function $g$", and it comes out "less than apparently
2 epsilon" (≈15:30).

Students' questions pinned the statement down (≈16:17–21:39):

- **$N$ is the total number of neurons**, not neurons per layer. "Because it's three layers, for
  the purpose of understanding the lecture, you can just pretend that the number 3 is the number
  1."
- **"Three-layer ReLU network"** means "the thing with three weight matrices and two ReLU
  functions that follow the first weight matrix and the second weight matrix. And then the third
  weight matrix doesn't have a ReLU." (The captions say "value functions" here; see the
  transcript's header.)
- **Why this theorem?** "Because the way to prove this theorem is not super involved", so the
  whole proof fits in a lecture. A student recalled lecture 1's statement that one hidden layer is
  enough; that "is a different result", and "there are probably more interesting versions of this
  result". This one is about three-layer networks.
- **The hypercube can be rescaled** to any box, which would change constants such as the Lipschitz
  constant. And it is a realistic assumption: in practice "you usually normalize your inputs to be
  coordinate-wise, about 1 in magnitude", so the data lives in a hypercube "for many problems, not
  for all problems".
- **The cost is exponential in dimension.** $N$ grows like $(L/\epsilon)^d$, so the number of
  neurons depends "exponentially on the dimension. So that's really bad dimension dependence. And
  we don't, in practice, really want to do that ever." Whether one can do better "probably depends
  on structure in the problem".

## The proof

### Strategy

Slide 11 lays out three steps (≈21:39–23:13). **Step one:** prove the result for one-dimensional
inputs by approximating the function with rectangular strips, ignoring ReLU networks for now.
**Step two:** generalize to higher input dimension with hyperrectangles. **Step three:** show that
ReLU networks can approximate rectangular strips, so that "a two-layer ReLU network can
approximate a hyperrectangle", and the third layer linearly combines the hyperrectangles.

### Step one: rectangles

Approximate $g$ on $[0, 1]$ by $N$ strips of equal width, each a flat rectangle centred on a grid
point (slide 12, ≈23:13–23:58):

$$f(x) = \sum_ i \alpha_ i \thinspace \mathbb{I}[x \in i^{\text{th}} \text{ interval}],$$

where $\alpha_ i$ is the height of the $i$-th rectangle and the indicator $\mathbb{I}[\cdot]$ is 1
when $x$ lies in the $i$-th interval and 0 otherwise. A student was confused by this, reading the
$i$ as an interval rather than an indicator; the lecturer clarified that for any input "only this
indicator is triggered. And then it gets scaled by its height. So there's a sum of terms, but only
one of them will ever be active" (≈41:22–42:53). As the strips narrow, the staircase hugs the
curve more closely.

![Slide 12: a smooth blue curve g(x) approximated by a staircase of thin pink-topped strips, with the indicator-sum formula and the claim error ≤ L/2N](../raw/images/03-approximation-theory/slide-12.jpg)

*Slide 12 — approximation with rectangles. The claim at the bottom: error grows with the Lipschitz constant and shrinks with the number of strips.*

The claim on slide 12 is

$$\text{error} \le \frac{L}{2N},$$

which "increases with Lipschitz constant" and "decreases with number of strips" (≈25:30).

**Why** (slide 13, ≈24:44–27:49). Put each rectangle's top at the height of $g$ at the strip's
left edge. The gap between rectangle and curve is bounded by Lipschitzness: a student saw that it
is "just … the area of the triangle". With $N$ strips each strip has width $1/N$, and because the
slope of $g$ cannot exceed $L$, the curve can rise or fall by at most $L/N$ across one strip. So
the gap fits inside a triangle of base $1/N$ and height $L/N$, with area
$\frac{1}{2} \frac{L}{N^2}$ ("The 1/2 base times height", ≈26:16). Summing over the $N$ strips,

$$\int dx \thinspace |f(x) - g(x)| \le N \times \frac{1}{2} \frac{L}{N^2} = \frac{1}{2} \frac{L}{N}.$$

![Slide 13: one strip close up: the rectangle top starts at the curve, and the curve stays under a dashed line of slope L, forming a triangle of width 1/N and height L/N](../raw/images/03-approximation-theory/slide-13.png)

*Slide 13 — by Lipschitzness the curve cannot leave the triangle, so each strip's error is at most its area.*

Then "flip things around": to guarantee error $\epsilon$, set $\epsilon = \frac{1}{2} L / N$ and
solve, giving the circled $N = \frac{1}{2} L/\epsilon$. Since the factor of $\frac{1}{2}$ "doesn't
really matter", the lecturer drops it, "being careful about the inequalities" (≈27:02–27:49).

### Step two: hyperrectangles

In $d$ dimensions, $g$ is a surface (for $d = 2$) or its higher-dimensional analogue, and the
strips become **hyperrectangles**: "Think: approximating a surface with cuboids" (slide 14,
≈28:36–31:05). "Whenever you need to think about high dimensions, you just think about things in
three dimensions." With $N$ hyperrectangles tiling the unit cube, each has side
$1/N^{1/d}$ and top face of area $1/N$, since the areas of all $N$ tops sum to 1. The error above
each one is an "error cap" whose height is at most $L / N^{1/d}$, by Lipschitzness again. So

$$\int d\mathbf{x} \thinspace |f(\mathbf{x}) - g(\mathbf{x})| \le N \times \frac{L}{N^{1/d}} \times \frac{1}{N} = \frac{L}{N^{1/d}},$$

the three factors being the number of hyperrectangles, the height of the error cap and the area of
the error cap. Setting this equal to $\epsilon$ and rearranging gives the circled

$$N = \left( \frac{L}{\epsilon} \right)^d \text{ hyperrectangles},$$

"which should remind you a lot of the thing which appeared in the theorem statement" (≈31:51). The
lecturer left the details of the cap "to think … through by looking at the slides".

![Slide 14: a blue wavy surface g(x) with a single pink cuboid column beneath one cell, its side labelled 1/N^(1/d), above the total-error calculation](../raw/images/03-approximation-theory/slide-14.png)

*Slide 14 — in d dimensions each strip becomes a column whose top face has area 1/N.*

### Step three: ReLU networks make rectangles

Now replace each rectangle with a small ReLU network (slide 15, ≈31:51–35:01):

$$f_c(x) = \begin{bmatrix} +1 \cr -1 \cr -1 \cr +1 \end{bmatrix}^T \text{relu} \begin{bmatrix} cx \cr cx - 1 \cr c(x-1) - 2 \cr c(x-1) - 3 \end{bmatrix}$$

This is "a two-layer ReLU network with a parameter $c$, a kind of weight $c$": four ReLU units
whose outputs are added with signs $+1, -1, -1, +1$. As $c \to \infty$, $f_c$ "converges to …
exactly this single rectangle in one dimension": $f_\infty$ is 0 for $x \lt 0$, jumps to 1 at
$x = 0$, stays at 1 until $x = 1$, and drops back to 0. The rectangle "can also translate
horizontally by adjusting weights and biases".

![Slide 15: the four-ReLU formula for f_c and, below it, the purple rectangular pulse f_∞ that is 1 between x = 0 and x = 1 and 0 elsewhere](../raw/images/03-approximation-theory/slide-15.png)

*Slide 15 — four ReLUs, with slopes steepened by the constant c, make a rectangle of width 1 and height 1.*

**The live demo.** The lecturer built this on a graphing website, which he recommends for the
problem sets (≈32:41–35:01). Start with $\text{relu}(x)$; subtract $\text{relu}(x - 1)$ and "it
flattens it out"; subtract another ReLU to bend it down; add a fourth to flatten the end, "each
time … translating them one along". That gives a trapezoid, not a rectangle: "The slopes are too
non-slope-- I want them to be slopier". So insert a constant $c$ multiplying the inputs, which
makes the slopes steeper "but it's also squeezing them together", and fix that by translating the
pieces back. Then let $c$ grow to 1,000: "And you see, OK, I did it. That's just combining ReLUs.
That's technically a two-layer neural net."

A student asked how this relates to a Riemann sum. One copy of this two-layer network
approximates each rectangle of the sum. "You translate it, and then you multiply it by a little
scalar to get it to be the size that you want. And that scalar would correspond to the third
layer in the network" (≈35:01–35:47). Asked why the theorem restricts to Lipschitz functions, the
lecturer said it is "to get a sense of the error when things are finite", for a finite number of
strips rather than the limit, and that he thinks it is necessary but is "not 100% sure"
(≈35:47).

### From rectangles to hyperrectangles: add, then threshold

A $d$-dimensional box is made from one-dimensional rectangles by a second trick (slide 16,
≈36:33–38:06): "Solve by adding 1-dimensional rectangles and thresholding appropriately". Take a
rectangle that varies along one axis (a wall running along the other axis) and another that varies
along the second. Add them, and the result is a plus-shaped surface: height 2 in the middle where
both are on, height 1 on the arms where only one is, and 0 elsewhere. In $d$ dimensions the sum
"only exceeds $d - 1$ when all rectangles are on", so subtract $d - 1$ and apply a ReLU: "it's just
going to pick out the place where they're all on", slicing off the top to leave a single box.

![Slide 16: three 3D surface plots, two perpendicular walls of height 1 and their plus-shaped sum of height 2 at the crossing, credited "Hongzhou Lin"](../raw/images/03-approximation-theory/slide-16.jpg)

*Slide 16 — two perpendicular walls add to a cross that reaches height 2 only where they overlap; thresholding at d − 1 keeps just that square. Plots credited to "Hongzhou Lin" on the slide.*

### Assembling the pieces

Slide 17 puts the three layers together (≈38:06–39:45):

$$\text{rectangle:} \quad f_c(x) = \begin{bmatrix} +1 \cr -1 \cr -1 \cr +1 \end{bmatrix}^T \text{relu} \begin{bmatrix} cx \cr cx - 1 \cr c(x-1) - 2 \cr c(x-1) - 3 \end{bmatrix}$$

$$\text{hyperrectangle:} \quad h_c(\mathbf{x}) = \text{relu}\left[ \sum_{i=1}^{d} f_c(x_i) - (d-1) \right]$$

$$\text{linear combination of hyperrectangles:} \quad f(\mathbf{x}) = \sum_ i \alpha_ i \thinspace h_c(\mathbf{x} - \mathbf{u}_ i)$$

where $x_i$ is the $i$-th coordinate of $\mathbf{x}$, $\mathbf{u}_ i$ is the position of the
$i$-th grid point, and $\alpha_ i$ sets the height of the $i$-th hyperrectangle. "There's a ReLU
here and there's a ReLU here. So it really is a three-layer network, by the definition that we
gave." Then "let this constant $c$ grow really large so that we really do approximate the
hyperrectangles. And we can approximate our arbitrary surface."

![Slide 17: the rectangle, hyperrectangle and linear-combination formulas in purple, pink and blue, ending with a small sketch of a wavy surface g(x)](../raw/images/03-approximation-theory/slide-17.png)

*Slide 17 — the whole construction: purple rectangles, pink hyperrectangles, blue weighted sum.*

The two ReLU layers do different jobs. "One of them is about approximating the rectangle to get
the slopes of the sides. And the other ReLU is more about slicing off the top of the
hyperrectangle" (≈40:36).

**Counting neurons** (≈42:53). Each rectangle uses four neurons ("the top purple bit"). A
hyperrectangle sums $d$ of them, so it needs $4d$. And $(L/\epsilon)^d$ hyperrectangles are
needed, which gives $N = 4d(L/\epsilon)^d$. "I hope it's right because I just tried to work out
the constant yesterday."

The lecture does not derive the factor of 2 in the theorem's bound of $2\epsilon$ from the steps
above; the lecturer reads it off the slide as "apparently 2 epsilon" (≈15:30).

## How seriously to take it

Slide 18 restates the theorem beside four comments (≈39:45–41:22):

- "General idea: approximate 'bumps' then linearly combine".
- "Needs exponentially many neurons in dimension".
- "Taking $c \to \infty$ feels unrealistic". "Usually you don't want the weights to blow up. That
  would be a sign that something's going wrong in your neural network."
- "Approximating rectangles feels like a trick". If the construction only uses ReLUs to make
  rectangles, "why wouldn't we just use a rectangle as our basis function to begin with?"

**It would not generalize.** Slide 19 imagines training a network built from these rectangles
(≈43:38–45:13). Each training point pulls its own rectangle toward it, "but there's all these other
ones which are never actually going to move", and "regularisation (weight decay) would suppress
the others". The fit is a row of isolated pulses, one per training point, when "visually, the
obvious way to approximate this data is just to draw a line through it". So "the training
performance is going to be really good, but the generalization performance is going to be really
bad. Because you're just fitting the data on tiny little strips". See
[generalization and double descent](generalization-and-double-descent.md).

![Slide 19: four green training crosses, each sitting on its own narrow pink rectangular pulse, with the curve flat at zero everywhere else](../raw/images/03-approximation-theory/slide-19.png)

*Slide 19 — trained as rectangles, the network memorizes each point and is zero in between: "would not generalise!"*

## Other approximation results

Slide 20 points to the wider literature (≈45:13–46:48). **Barron's theorem** says "smooth
functions can be approximated with fewer neurons" and "leverages Fourier representation". There is
also a classic result that **two layers are enough**, with the slide citing Hornik, Stinchcombe and
White (1989), which uses the **Stone–Weierstrass theorem**: in the lecturer's understanding "a
more powerful result" than the three-layer one just proved. Weierstrass, he notes, is the same
mathematician as the fractal of slide 6. He also had a theorem that, roughly, "you can use
polynomials to approximate any continuous function". "It's just kind of interesting that this
branch of math is related to a very classical branch of math."

A student asked how this relates to fitting a Taylor expansion. That "would be another way of just
fitting the Taylor expansion up to some degree, which is a bit like that Weierstrass thing of using
polynomials", "a valid mathematical way to approximate a function … But it's just not what we're
doing in deep learning" (≈46:48).

## Is universal approximation important?

Slide 21 asks two questions (≈46:48–49:59). Abbreviating universal function approximation as UFA:

- **Is UFA sufficient for learning to work?** "No, there are many UFAs that we usually don't do ML
  with": Fourier series, polynomials, "the space of Python programs". A student offered linear
  combinations of the lecture's own hyperrectangle bumps, which the lecturer accepted as valid. "Just
  because you've installed Python, you can't start doing machine learning. You need something
  else." Universal approximation is "just a piece of the puzzle".
- **Is UFA necessary?** If a model is not a universal approximator, should you worry? "My guess is
  probably we don't need to have a universal function approximator to do machine learning. But I
  think it's a little unclear how to answer that."

## Width versus depth

The second half returns to the opening question (slides 22–23, ≈49:59). If three layers can fit
anything, "why would we ever want to have a deep network?"

**Arguments for width** (slide 24, ≈50:47–53:53):

- "3 layer (or even 2 layer) NNs are universal function approximators."
- "Width is inherently parallelisable, depth is sequential." This came from a student: "width can
  be parallelized. It makes better use of parallel hardware", so "if you can get away with a shallow
  network that's very wide, you would really love that". The lecturer drew the general lesson that
  there are two kinds of constraint: what a network can represent, and "the computational constraint
  of, how efficient is this to run?"
- "Width is easier to train, depth leads to 'compound problems'." "Whenever you do compound
  anything, it is very liable to get out of control … if you just take a vanilla multilayer
  perceptron and make it 50 layers deep and just try to train it, you'll see it's very difficult to
  get it to train."

Students also suggested that wide networks have "feature representations that are nicer and more
easily interpretable". On the other side, one student argued for depth: stacking layers composes
more nonlinearities, which "can give us a richer space of functions through compositionality"
(≈50:47).

The slide ends "So, scaling width is obviously better!", and the next slide asks "Or is it?"

## Depth separations

**The idea** (slide 25, ≈53:53–54:40). Universal approximation results "suggest needing
exponentially many hidden units at small width" (here, small depth). A **depth separation** result
goes the other way: it constructs "deep networks that require exponentially more units to fit with
a shallow network".

**The shape of such a proof** (slide 26, ≈54:40–56:13) has three steps:

1. Pick a property of a function, for example "number of linear regions".
2. Construct a deep network that has a lot of this property without having many neurons.
3. Prove that a shallow network would need exponentially more units to have the same property.

"There's a literature which just proves different varieties of this kind of result … We're just
going to show you one particular example."

### Kinks in piecewise linear functions

A **piecewise linear** function is linear on each of a series of segments that stitch together.
A **kink** is "a place where the gradient changes" (slide 27, ≈56:13–57:00). The slide's example
function has four. The property used for the separation is the number of kinks.

![Slide 27: a green piecewise-linear curve with four pink dots marking where its slope changes, and the list of reasons ReLU networks are piecewise linear](../raw/images/03-approximation-theory/slide-27.png)

*Slide 27 — a kink is where the slope changes; this function has four.*

**ReLU networks are piecewise linear** (slide 27, ≈57:00–58:35). The argument breaks the network
into operations and shows each preserves the property: relu is piecewise linear (PWL); if $f$ and
$g$ are PWL then so are $f + g$ and $f \circ g$; and if $f$ is PWL then so is $\alpha \cdot f$
for any real $\alpha$. A ReLU network only composes, adds and scales such pieces, "so you're always
going to get something that's piecewise linear". A student's version of the same point: ReLU is "a
linear function, but with a threshold", and the linear layers only distort it linearly.

### Two intuitions: adding and applying ReLU

Think of a single neuron whose inputs are one-dimensional functions $f_1, \ldots, f_n$ of the input
$x$ (≈58:35–59:21).

**Adding at most adds the kinks** (slide 28). The neuron's pre-activation is
$y(x) = \sum_{i=1}^{n} \alpha_ i f_i(x)$, and "when we add functions, at most we add the number of
kinks". The slide's example adds a green $f_1$ with one kink to a blue $f_2$ with two, and the
purple sum has three. There can be no more, "because you only can have kinks at the places where
the functions that you're adding had them", and fewer if two kinks coincide (≈59:21–1:00:07).

![Slide 28: a neuron y summing inputs f_1 to f_n; below, a green curve with 1 kink plus a blue curve with 2 kinks gives a purple curve with 3](../raw/images/03-approximation-theory/slide-28.png)

*Slide 28 — summing functions: the kinks of the sum are at most the kinks of the parts.*

**Applying ReLU at most doubles them** (slide 29). For $y(x) = \text{relu}(f(x))$, "when we apply
relu, at most we double the number of kinks", "because linear pieces can be split in two": each
linear piece that crosses zero gets a new kink where ReLU flattens the negative part. In the
slide's example, the lecturer counts five kinks in $f$ and nine in $\text{relu}(f)$, "so you can
see it did nearly double in this case" (≈1:00:07–1:00:53).

![Slide 29: a pink piecewise-linear f with 5 kinks and, overlaid in green, relu(f), which follows f above the axis and is flat below it, with 9 kinks](../raw/images/03-approximation-theory/slide-29.png)

*Slide 29 — ReLU zeroes the negative parts, adding a kink at every crossing of the axis.*

### The bound: polynomial in width, exponential in depth

Slide 30 makes this formal (≈1:01:43–1:04:54). Take a deep ReLU network of width $n$ and look at
its $L$-th layer:

$$\mathbf{f}_ L(x) = \text{relu}(\mathbf{W}_ L \thinspace \mathbf{f}_ {L-1}(x) + \mathbf{b}_ L),$$

where $\mathbf{f}_ L(x) \in \mathbb{R}^n$ is the layer's output as a function of the scalar
input $x$, $\mathbf{W}_ L$ is an $n \times n$ matrix and $\mathbf{b}_ L \in \mathbb{R}^n$
is a bias. Let $\text{KINKS}_ L$ be the maximum number of kinks over the $n$ coordinates of
$\mathbf{f}_ L(x)$, each coordinate being a one-dimensional function. Then

$$\text{KINKS}_ L \le 2n \cdot \text{KINKS}_ {L-1}.$$

"The 2 comes from the doubling effect of the ReLU. And the $n$ comes from the fact that you're
adding $n$ things" (≈1:03:17). The slide continues "Since $\text{KINKS}_ 0 = 1$, this implies"

$$\text{KINKS}_ L \le (2n)^L.$$

Slide 31 defines the symbols: $\text{KINKS}_ L$ is the number of kinks "in the function at layer
$L$", $n$ is the width and $L$ is the depth. (The handwritten layer index is a capital $L$
throughout, the same letter as the Lipschitz constant of the first half; the slide-by-slide
transcription records the check.)

![Slide 30: a four-layer fully connected network with one layer circled, the layer recursion f_L = relu(W_L f_L−1 + b_L), and the boxed bound KINKS_L ≤ (2n)^L](../raw/images/03-approximation-theory/slide-30.png)

*Slide 30 — one layer at a time, the kink count can grow by at most a factor of 2n.*

**The base case was disputed in the room.** The lecturer had said the input has "basically 1"
kink, "because there's no kinks"; a student pointed out "It should be 0", and he agreed: "It
should be 0, you're right." But if $\text{KINKS}_ 0 = 0$ the recursion gives a bound of 0, which
cannot be right, since the ReLU itself adds kinks. Students suggested starting the induction from
one layer, and one noted that "the ReLU adds a kink where there was none". The lecturer left it as
"some accounting problem", "you just need to think about it a bit to work out what the base case
should be", and asked that the intuition carry (≈1:03:17–1:04:54). The slide prints
$\text{KINKS}_ 0 = 1$.

**Reading the bound** (slide 31, ≈1:04:54–1:05:41). With $n$ the width and $L$ the depth, the
upper bound "grows at best polynomially in width but exponentially in depth". "So if our goal is to
be able to approximate functions with many kinks, then it's better to make the network deeper
rather than wider." But it is only an upper bound: "Maybe it's never attained." The slide asks: "is
the bound ever attained?"

### The construction: a triangle composed with itself

"Short answer: Yes!" (slide 32, ≈1:05:41–1:07:14). Define

$$g(x) = \text{relu}\left[ 2 \cdot \text{relu}(x) - 4 \cdot \text{relu}\left(x - \tfrac{1}{2}\right) \right].$$

This is a small ReLU network whose graph is a triangle: 0 for $x \lt 0$, rising to 1 at
$x = \frac{1}{2}$, falling back to 0 at $x = 1$, and 0 beyond. Composing it with itself doubles the
number of linear regions each time: $g \circ g$ has two triangles, $g \circ g \circ g$ has four.
"You need to think a little bit to see why that is." So repeated composition really does keep
doubling the kinks: "it can really happen, at least with this one example."

![Slide 32: the triangle function g(x) peaking at 1 at x = ½, and sketches of g, g∘g and g∘g∘g with one, two and four peaks](../raw/images/03-approximation-theory/slide-32.png)

*Slide 32 — each self-composition of the triangle map doubles the number of teeth.*

**The numbers** (slide 33, ≈1:07:14–1:08:51). Compose $g$ with itself 500 times. The result has
$2^{500} - 1$ kinks. Each $g$ has two layers, so this is a network of 1000 layers of width 2. How
wide would a three-layer network have to be to have as many kinks? Inverting the bound
$\text{KINKS} \le (2n)^L$ with $L = 3$,

$$n \ge \frac{1}{2} \cdot \text{KINKS}^{1/L} = \frac{1}{2} \cdot (2^{500} - 1)^{1/3} \approx 7 \times 10^{49} \text{ units},$$

"almost 10 to the power 50 width". "This is the depth separation. There's a function, which is
not even that difficult to write down … And if you wanted to get a shallow network to exactly
approximate that function, it would need to be very, very, very wide." The lecturer adds that it is
"also on the first problem set": problem set 1's approximation section asks how adding a layer
affects the number of linear regions of a ReLU network, and whether deeper networks can be more
efficient
([`mit6_7960_f24_hw1.pdf`](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_hw1.pdf)).

### What it does not say

Slide 34 (≈1:08:51–1:09:37) is careful about the limits. The result does **not** mean that very
deep networks are easy to train (optimization), nor that they would generalize well
(generalization). "In our machine learning puzzle, it only tells us something about approximation!"

**Further reading** (slide 35, ≈1:09:37–1:10:24): Telgarsky (2015, 2016), "our depth separation";
Safran and Shamir (2017), "a different depth separation"; and Lu, Pu, Wang, Hu and Wang (2017),
with results on the "minimum width" needed to be a universal approximator even at large depth,
relating to the rank of the weight matrices. The lecturer singled out the last: if depth is so
powerful, why not use width 3 and go very deep? Because "if the input space has dimension $n$, you
need a width of at least $n$ to approximate any function basically. So there can be a minimum width
that you really need."

## Practical considerations

**In practice the pieces are conflated** (slide 37, ≈1:10:24–1:12:41). The puzzle's three pieces
are neat in theory, "but out there in the real world, it's not like that". When training goes
badly, you cannot tell whether the network cannot approximate the target or whether optimization
is failing. Generalization is the exception: "you could just compute the training error and the
test error, and you can see if they're different." The slide's scenario: "pretend you work at an LLM
startup" whose job is to train the most efficient LLM possible. Then you really care about the
optimal width versus depth, for training cost, inference cost and quality. "In a sense, it's the
most basic question … But it's, to a large extent, an unsolved problem."

**Scaling laws** (slide 38, ≈1:12:41–1:15:01). The lecturer showed two figures from Kaplan,
McCandlish et al. (2020), the first of which "people got T-shirts" of. It plots test loss on
log-log axes against compute, dataset size and parameters, and "the test loss … is always going
down as you scale any of these axes" (each panel prints a power-law fit). The loss cannot fall
forever, since "you can't have negative loss", but "back in 2020, they were saying, hey, the models
keep getting better. We should pay attention to this."

The point for this lecture is that they plot against **parameters**, not width or depth. Their
claim is that "within several orders of magnitude, the allocation of your compute budget between
width and depth doesn't matter. All that matters is the number of parameters and number of flops."
The second figure shows test loss against parameters for networks of 0, 1, 2, 3, 6 and more than
6 layers. Counting parameters without the embedding layers (the right-hand panel), "beyond depth 6,
it doesn't matter what the width is or what the depth is. The curves converge to each other." In the
figure the 2-, 3-, 6- and more-than-6-layer curves nearly collapse onto one line, and the 1-layer
curve, after crossing them at small sizes, ends clearly above. The lecturer called this "a very, very practical perspective, but from
some very careful experimentalists". See [scaling laws](scaling-laws.md).

![Slide 38: Kaplan et al.'s three power-law panels of test loss against compute, dataset size and parameters, and below them test loss against parameters for networks of 0 to more than 6 layers](../raw/images/03-approximation-theory/slide-38.jpg)

*Slide 38 — from Kaplan, McCandlish et al. (2020): with embedding parameters excluded (bottom right), networks of 2 or more layers fall on nearly one curve.*

**Confounders** (slide 39, ≈1:15:01–1:16:34). The "Chinchilla scaling rules" paper (the slide's
citation reads "Hoffmann, Borgeau, Mensch et al (2020)") asks the same questions and "question[s]
some results in Kaplan et al". In particular, "if you use a different learning rate schedule for
the training, you can get … qualitatively different conclusions". The general lesson: "precisely
answering any of these questions is very difficult, because there are so many parts of the training
pipeline that we don't understand." One setup says width versus depth doesn't matter; fix some
detail and "now it's like oh, no, scaling width is a lot better". The slide's conclusion: to really
answer "what is the optimal width versus depth", you "need to obsess over 'minor details' of the
training pipeline". Questions like this are hard to resolve experimentally because of confounding
variables, and hard to resolve theoretically too.

## Summary

Slide 41 (≈1:17:19–1:18:12):

- "Very wide shallow neural nets are universal function approximators": a wide enough three-layer
  MLP can fit any function in a broad class.
- "Deeper networks can fit certain kinds of function with many fewer neurons", because of their
  compositionality: "depth separations".
- "Unclear how these results interact with training and generalisation." It is still worth having
  them in your toolkit, because "a basic question is, can the network architecture that I have
  actually approximate the function that I'm asking it to approximate?"

## Preview: inductive biases

Slide 42 closes with a thought experiment (≈1:18:12–1:22:07). Take two problems: transcribing an
audio waveform as the word "hello", and classifying a photograph of a person as "human". In both
cases "we could flatten the inputs into vectors and just apply an MLP… Hey, it's a universal
function approximator. But, is this a good idea?"

![Slide 42: an audio waveform mapped to "hello" and a framed stick figure mapped to "human", above the question of whether to flatten both and apply an MLP](../raw/images/03-approximation-theory/slide-42.png)

*Slide 42 — two problems with differently structured inputs, previewing inductive biases.*

The class said no, with three objections. The approximator may exist but "you may not find it".
An MLP could be inefficient. And "an MLP doesn't take advantage of the structure in the input data.
So with an image, there's an inherent two-dimensional structure. And with a waveform, there's an
inherent sequential structure." The lecturer agreed with the last, then raised a complication of
his own: "now we're just applying transformers to everything". He concluded it does not undo the
point, because "even if you apply a transformer to audio or you apply it to images, actually the
preprocessing layer at the beginning of the network is very different", such as a 2D patch
representation for images. "I think really, you still need to model the data little bit." The
final thought: "perhaps we want to match the architecture to the problem that we're trying to
solve, or the structure of the data", which could be more efficient computationally, less hungry
for data, and make the approximator easier to find. The slide names the topic, inductive biases,
but gives no lecture number. The [course map](course-map.md) lists the architecture lectures that
follow: grids (lecture 4), graphs (5), transformers (8) and memory (10).

## See also

- [Representational power](representational-power.md) — what networks can approximate, across
  lectures 1 and 3.
- [Lipschitz continuity](lipschitz-continuity.md) — the definition, the bow-tie picture and the RMS
  norm.
- [Scaling laws](scaling-laws.md) — Kaplan et al. and Chinchilla as this lecture presents them.
- [Multilayer perceptrons](multilayer-perceptron.md) and
  [activation functions](activation-functions.md) — the layers and ReLUs the constructions are
  built from.
