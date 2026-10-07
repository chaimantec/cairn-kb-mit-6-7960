# Lipschitz continuity

A way of saying how fast a function can change. Lecture 3 introduces it to choose the family of
functions its approximation theorem covers, and the lecturer thinks it is "nice to teach you about
Lipschitz functions regardless of neural net approximation" (≈9:16). Covered so far:
[lecture 3](03-approximation-theory.md), slides 8–10 and 12–14; [lecture 7](07-scaling-rules-for-optimization.md),
slide 24, ≈59:18–1:02:24, which reuses the RMS norm.

On this page, as in lecture 3, $L$ is the **Lipschitz constant**, not the loss of the
[course notation](notation.md).

## The definition in one dimension

A function $g: \mathbb{R} \to \mathbb{R}$ is **$L$-Lipschitz** if, for all inputs
$x \in \mathbb{R}$ and all changes $\Delta x \in \mathbb{R}$,

$$|g(x + \Delta x) - g(x)| \le L |\Delta x|.$$

In words: if you change the input by $\Delta x$, the output changes by at most $L$ times as much.
$L$ is just a number, "like, 10 or 5", and it "measures how Lipschitz the function is" (slide 8,
≈9:16–10:04).

**It bounds the slope.** Let $\Delta x$ shrink and the left-hand side divided by $|\Delta x|$
becomes the size of the derivative, so "the slope of $g$ cannot exceed $L$" (slide 8). The lecturer
calls Lipschitz continuity "a kind of generalization of the notion of having a bounded derivative.
Well, it may be equivalent actually, but you need to think about that" (≈10:49). The lecture
leaves that question open.

**The bow-tie picture.** If $g$ passes through the origin, it "can never stray outside $\pm Lx$":
it must lie between the lines $y = Lx$ and $y = -Lx$. The same holds at every point of the curve, so
picture a cone, "or a kind of bow tie", centred on any point of the function, and the function has
to stay inside it (slide 8, ≈10:49–11:34).

## Many inputs, and the RMS norm

For $g: \mathbb{R}^d \to \mathbb{R}$ the output is still a number, so its change is still measured
by an absolute value. But the change in input, $\Delta \mathbf{x} \in \mathbb{R}^d$, is a vector,
"and you can't take the absolute value of a vector" (≈12:20). So the definition replaces
$|\Delta x|$ with a norm (slide 9):

$$|g(\mathbf{x} + \Delta \mathbf{x}) - g(\mathbf{x})| \le L \Vert \Delta \mathbf{x} \Vert_{\text{RMS}} \quad \text{for all } \mathbf{x}, \Delta \mathbf{x} \in \mathbb{R}^d.$$

The norm the lecturer picks, "a particular norm that I really like", is the **RMS norm**, the
root-mean-square size of a vector's entries:

$$\Vert \mathbf{x} \Vert_{\text{RMS}} \triangleq \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2} = \frac{1}{\sqrt{d}} \Vert \mathbf{x} \Vert_ 2,$$

where $x_i$ are the entries of $\mathbf{x}$ and $\Vert \mathbf{x} \Vert_ 2$ is its Euclidean norm
(≈13:09). The difference between the two is scale. "The Euclidean norm is kind of a dimensional
object. You can think about the RMS norm as a kind of non-dimensional analog of the Euclidean norm.
Because if all the entries of the vector are 1, then the RMS norm is 1, whereas the Euclidean norm
would be like, square root $d$." The broader point of the slide: "there's many different possible
ways to measure the size of something" (≈13:57).

## What it buys in lecture 3

Lipschitzness is what makes the approximation error **finite and countable**. Asked why the
theorem restricts itself to Lipschitz functions, the lecturer said it is "to get a sense of the
error when things are finite", for a finite number of approximating strips rather than the
limit (≈35:47). Concretely:

- **One dimension** (slides 12–13). Approximate $g$ on $[0,1]$ by $N$ flat strips of width $1/N$,
  each starting at the curve. Because the slope cannot exceed $L$, the curve moves by at most
  $L/N$ across a strip, so the gap fits in a triangle of area $\frac{1}{2} L/N^2$, and the total
  $L_1$ error is at most $\frac{1}{2} L/N$.
- **$d$ dimensions** (slide 14). With $N$ hyperrectangles of side $1/N^{1/d}$, the "error cap"
  above each is at most $L/N^{1/d}$ high and has area $1/N$, so the total error is at most
  $L/N^{1/d}$, and error $\epsilon$ needs $N = (L/\epsilon)^d$ boxes.

In both, a larger Lipschitz constant (a function allowed to change faster) means more pieces for
the same error. That is where the $(L/\epsilon)^d$ in the theorem of slide 10 comes from; see
[representational power](representational-power.md#a-universal-approximation-theorem-lecture-3).

The family is also a way of excluding pathological functions. Slide 7 says the family of curves
should "exclude pathological functions" such as Weierstrass's function, which is continuous
everywhere and differentiable nowhere (slide 6).

## The RMS norm again (lecture 7)

Lecture 7 picks the RMS norm back up, "from my other lecture" (≈1:00:05), as the natural way to measure a
layer's activations: a unit RMS norm means each coordinate is around 1, and transformers enforce it under
the name layer norm. Using it on a matrix's input and output gives the **RMS-RMS operator norm**, the most
a matrix can change the RMS norm of a vector passing through it (slide 24), which lecture 7 uses to scale
a network's initialization and updates with width. See [norms](norms.md) and [scaling
rules](scaling-rules.md).
