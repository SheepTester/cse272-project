#let dark_theme = true
#let title = "CSE 272 Project Proposal"
#let authors = "Sean Yen"

#let page-color = if dark_theme { rgb("#0f172a")} else {color.white}
#let text-color = if dark_theme{ rgb("#cbd5e1")} else {color.black}
#let faded = rgb("#64748b")
#let indent = h(1cm)
#let blue = rgb("#0ea5e9")


// formatting options: https://typst.app/docs/tutorial/formatting/#page-setup
// https://piazza.com/class/m5bnm9b0buy1qv/post/81
#set page(paper: "a4", margin: 0.75in, fill: page-color)
#set text(font: "Inter", fill: text-color, size: 12pt)
#set document(title: title, author: authors)
#set par(spacing: 1em, 
// justify: true
)
// #set heading(numbering: "1.1.")
#set footnote(numbering: "[1]")
#set footnote.entry(separator: line(length: 30% + 0pt, stroke: 0.5pt + text-color))
#show figure.caption: content => block(emph(content), width: 90%)
#show quote: set align(center)

#let indent(content) = block(inset: (left: 0.5cm), outset: (y: 0.05em), stroke: (left: (paint: faded, thickness: 1pt, dash: "dotted")), content)

#let TODO = box(fill: color.red, outset: 3pt)[#text(fill: color.white, weight: "bold", size: 1.5em)[#emoji.warning TODO #emoji.warning]]

#let page_break = pagebreak(weak: true)

#text(size: 2em, weight: "bold")[#title]

#text(fill: faded)[#authors]

For the final project, I intend on reimplementing volume rendering from `lajolla` (i.e. homework 2) in WebGPU.
Ideally, it should be fast enough that one can fly around a scene with a first person camera in the web browser.

= Why Homework 2?

One benefit of homework 2 is that the assignment involved implementing a function that returns the color of a given pixel coordinate, which is effectively what a fragment shader already does. Therefore, the most naive implementation that I could start with is to just translate my CPU implementation from homework 2 to the shader language. Of course, this probably wouldn't be very efficient.

Also, it helps that in homework 2, I wrote most of the rendering code, so I'm more familiar with the code that I will be translating.

= Why WebGPU?

Creating a web-based renderer has several benefits:

+ Web-based projects are a lot easier to share with other people because it doesn't require installing an executable. For me, it would make my project more meaningful if people could actually interact with it on their own computers.

+ Both WebGL and WebGPU are hardware agnostic, meaning that the same code can work across different hardware and operating systems, such as smartphones and the UCSD wayfinding kiosks. Admittedly, one downside of this is that it is harder to make hardware-specific optimizations.

In addition, I'm more familiar with writing code in JavaScript/TypeScript, so it would be easier to develop in the short amount of time we have left in the quarter.

I'm opting for WebGPU over WebGL. Because WebGPU is newer, its API is newer too, so it doesn't have some of the quirks and pain points of WebGL. I also think WebGPU's shading language, WGSL, is more interesting to use than WebGL's GLSL. WebGPU is also designed for parallelizing computational programs rather than just rendering graphics on the user's screen, so it offers features like workgroups and compute shaders that are more ergonomic to use than with WebGL. However, because WebGPU is newer, its device and browser support is weaker than WebGL.

= Goals

The purpose of this project is to try to make an _efficient_ parallelized renderer. This means going beyond just a naive port of the CPU renderer.

But because I'm using WebGPU, another goal is to make the renderer user friendly somehow. This will be in the form of an interactive scene where you can, at the very least, move the camera around. A moving camera would be a good demonstration of the performance boost from parallelization.

= Potential Challenges

I wasn't able to get all of homework 2 working, so hopefully solutions are posted somewhere, and if not, I'll only implement up to part 4 or 5 of homework 2, where my code seems to still work.
