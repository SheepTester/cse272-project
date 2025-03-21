#let dark_theme = true
#let release = true
#let title = "CSE 272 Final Project: Final Report"
#let authors = "Sean Yen"

#let page-color = if dark_theme { rgb("#0f172a")} else {color.white}
#let text-color = if dark_theme{ rgb("#cbd5e1")} else {color.black}
#let faded = rgb("#64748b")
#let russell-lore = gradient.linear(rgb("#0f5"), rgb("#88F"), ) //("#d1bb3d")
#let indent = h(1cm)
#let blue = rgb("#0ea5e9")


// formatting options: https://typst.app/docs/tutorial/formatting/#page-setup
// https://piazza.com/class/m5bnm9b0buy1qv/post/81
#set page(paper: "a4", margin: 0.75in, fill: page-color)
#set text(font: "Inter", fill: text-color, size: 10pt)
#set document(title: title, author: authors)
#set par(spacing: 1em, 
justify: release
)
#set page(header: context {
  set text(fill: faded)
  if counter(page).get().first() > 1 [
    #title
  ] else [
    #text(fill: text-color)[#authors]
  ]
  // // get heading on current page
  // let real_page = here().page()
  // let page_headings = query(heading).filter(h => h.location().page() == real_page)
  // let heading_num = if page_headings.len() > 0 {
  //   counter(heading).at(page_headings.first().location()).at(0)
  // } else {
  //   // if there are no headings on this page, get the heading level from the end of the previous page
  //   counter(heading).get().at(0)
  // }
  // if heading_num > 0 {
  //   [Question #heading_num]
  // }
  [
    #h(1fr)
    #counter(page).display(
      "1 of 1",
      both: true,
    )
    // #emoji.egg #emoji.drops #emoji.egg
  ]
})
// #set heading(numbering: "1.1.")
#set footnote(numbering: "[1]")
#set footnote.entry(separator: line(length: 30% + 0pt, stroke: 0.5pt + text-color))
#show figure.caption: content => block(emph(content), width: 90%)
#show quote: set align(center)
#set math.lr(size: 120%)

#let indent(content) = block(inset: (left: 0.5cm), outset: (y: 0.05em), stroke: (left: (paint: faded, thickness: 1pt, dash: "dotted")), content)

#let TODO = box(fill: color.red, outset: 3pt)[#text(fill: color.white, weight: "bold", size: 1.5em)[#emoji.warning TODO #emoji.warning]]

#let page_break = pagebreak(weak: true)
#let russell_lore(content) = if release {[]} else { text(fill: russell-lore, /* style: "italic", */ content)} // []

#text(size: 2em, weight: "bold")[#title]

#link("https://github.com/SheepTester/cse272-project")[#text(fill: blue)[GitHub repository]]

// Please submit your report and your code here. Your code can be either a GitHub link or a zip file. If you worked with another person on this project, please also put his/her name in the comment section.

// There should be a report. Tell us about your journey and what you have learned. Document what was in your mind when designing whatever you show in the end.
// https://piazza.com/class/m5ke562n9z969v/post/118

My project is porting homework 2 to WebGPU. In my proposal, I made three goals:

- Port homework 2 to WebGPU
- Make it efficient
- Make the scene interactive

Let's see how this played out.

= Porting to WebGPU

Even with the solution code,
#footnote[https://piazza.com/class/m5ke562n9z969v/post/110]
it was a struggle porting `lajolla` to WebGPU's shader language. This is because there were a lot of library functions that I also had to implement, and C++ variants made it a little bit difficult to find function implementations.

One decision was how to represent scenes. `lajolla` has support for two different types of many things, each defined as a variant: shapes, media, lights, filters, phase functions, etc. To keep things simple, I chose one of each of these variants to implement, which tended to be the case for most of the scenes I was testing on.

One feature I did implement was sending the scene information from the CPU to the GPU, so I was able to animate the balls moving around, which also gave a visual indication of how performant the GPU implementation was.

#figure(
  grid(
    columns: 4,
    column-gutter: 10pt,
    image("part1.png"),
    image("part2.png"),
    image("part3.png"),
    image("127.0.0.1_8000_.png"),
  ),
  caption: [
    Scene renderings of parts 1--4. Part 1 only performs $1$ sample and is rendered to a $360 times 360$ canvas. The others do $1024$ samples and are $512 times 512$.
  ]
) <screens>

I was able to implement up to part 4, where despite seemingly matching the solution, it didn't seem to work properly (see @screens). The GPU was also disconnecting/crashing sometimes in part 4, probably because it casts more shadow rays than part 3.

Because I wasn't able to fully re-implement homework 2, I didn't have much of a chance at making GPU-specific optimizations. However, a naive 1-to-1 port of homework 2 to the GPU already came with a massive speedboost. For example, rendering part 3 takes me 25.6677 seconds with `lajolla`. With the same resolution and sample count, I was able to move around in `volpath_test3` at more than 1 FPS, so that's a 25x speedup! Unfortunately, I wasn't able to get GPU time measurements working in time.

= Interactive scenes

I added basic FPS camera controls. To improve the framerate (since I didn't make any optimizations), I had to decrease the resolution to $128 times 128$. At this level, the canvas looked pretty pixelated, but you're still able to tell what's going on.

#figure(image("pixelated.png", height: 3in), caption: [`volpath_test2` at a $128 times 128$ resolution, which is the experience of flying around the scene. The pink ball has been set up to revolve around the blue sphere.])

= What I've learned

It's pretty difficult to get a volumetric path renderer set up, in general! I unfortunately wasn't able to make any clever GPU-specific optimizations, but I'm also surprised that with this much branch divergence (from each pixel's rays ending earlier than others), it's still so performant.

= Next steps

If I continue working on this, I'd like to at least get part 4 working, then try designing my own scene that better showcases what the renderer can do. In `lajolla`, because it takes so long to render a frame, it can be hard to see what the scene looks like from different angles or how different properties affect the final result. Being able to fly around in the scene and animate different parts helps to overcome this.

I think this also has some potential as a video game (as long as the entire scene is constructed with spheres); even the low resolution mode can be part of a game's aesthetic.
