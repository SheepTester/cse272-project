// https://canvas.ucsd.edu/courses/62911/assignments/907430

#let dark_theme = true
#let title = "CSE 272 Project Checkpoint"
#let authors = "Sean Yen"

#let page-color = if dark_theme { rgb("#0f172a")} else {color.white}
#let text-color = if dark_theme{ rgb("#cbd5e1")} else {color.black}
#let faded = rgb("#64748b")
#let indent = h(1cm)
#let blue = rgb("#0ea5e9")


// formatting options: https://typst.app/docs/tutorial/formatting/#page-setup
// https://piazza.com/class/m5bnm9b0buy1qv/post/81
#set page(paper: "a4", margin: 0.5in, fill: page-color)
#set text(font: "Inter", fill: text-color, size: 10pt)
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

There's not much progress to report yet because of other classes. I'll probably focus on assignments due earlier before locking into this final project.

That said, I have copied some boilerplate for WebGPU from a previous project, and implemented a PRNG on the GPU
#footnote[https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf], shown in @noise.

#figure(image("noise.png"), caption: [Each pixel sets a random color.]) <noise>

After that, I began working on reimplementing part 1 of homework 2.
@frag shows the shader code for sampling a random ray. Currently, the scene is hard-coded, so there's no way to change the filter from a scene file. It currently displays `ray.dir` as output to ensure that it's the same as the solution (@comp).

// https://stackoverflow.com/a/76334857/28188730
#figure(
    grid(
        columns: 2,     // 2 means 2 auto-sized columns
        gutter: 2mm,    // space between columns
        image("ours.png"),
        image("theirs.png"),
    ),
    caption: [A comparison of `ray.dir` between my WebGPU implementation (left) and the homework 2 solution in the EXR viewer (right). Exposure has been increased tremendously.]
) <comp>

The largest challenge has been and will be reimplementing all the library functions. I took a look at the next step, raycasting, and it appears that `lajolla` uses Embree to handle raycasting. I also may push back implementing scene loading and instead continue to hardcode the scene until I get the rest of rendering working. I'll start focusing on this project more during finals week.

The project is currently live on GitHub Pages: #text(fill: blue)[#link("https://sheeptester.github.io/cse272-project/")[sheeptester.github.io/cse272-project/]]. It's currently just a static render since I haven't implemented much else yet.

#figure(
  ```wgsl
  @fragment
  fn fragment_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
      var seed = seed_per_thread(
          u32((vertex.uv.x * canvas_size.x + vertex.uv.y) * canvas_size.y) + 69
      );
      seed = next_seed(seed);
  
      let dx = seed_to_float(seed);
      seed = next_seed(seed);
      let dy = seed_to_float(seed);
      seed = next_seed(seed);
      let offset = box_filter(vec2(dx, dy));
      let remapped_pos = vertex.uv + (vec2(0.5) + offset) / canvas_size;
      let dir = normalize(sample_to_cam * vec4(remapped_pos, 0.0, 1.0)).xyz;
      let ray = Ray((cam_to_world * vec4(vec3(0.0), 1.0)).xyz,
                    normalize(cam_to_world * vec4(dir, 0.0)).xyz);
  
      return vec4(ray.dir * 50.0, 1.0);
  }
  ```,
  caption: [Fragment shader so far.]
) <frag>
