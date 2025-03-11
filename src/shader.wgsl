struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> canvas_size: vec2<f32>;

@vertex
fn vertex_main(
    @builtin(vertex_index) index: u32,
) -> VertexOutput {
    const square_vertices = array(
        vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0),
        vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0),
    );

    var result: VertexOutput;
    result.position = vec4(square_vertices[index] * 2.0 - vec2(1.0), 0.0, 1.0);
    result.uv = square_vertices[index];
    return result;
}

@fragment
fn fragment_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    var seed = seed_per_thread(u32(vertex.uv.x * canvas_size.x) + u32(vertex.uv.y * canvas_size.y));
    seed = next_seed(seed);

    let r = seed_to_float(seed);
    seed = next_seed(seed);
    let g = seed_to_float(seed);
    seed = next_seed(seed);
    // let b = seed_to_float(seed);
    // seed = next_seed(seed);

    return vec4(r, g, canvas_size.y / 300.0, 1.0);
}

// PRNG for GPU
// https://stackoverflow.com/a/70110668/28188730
// https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf

fn seed_per_thread(id: u32) -> u32 {
    return id * 1099087573;
}

fn tau_step(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {
    let b = (((z << s1) ^ z) >> s2);
    return (((z & m) << s3) ^ b);
}

fn next_seed(seed: u32) -> u32 {
    let z1= tau_step(seed, 13, 19, 12, 429496729);
    let z2= tau_step(seed, 2, 25, 4, 4294967288);
    let z3= tau_step(seed, 3, 11, 17, 429496280);
    let z4= 1664525 * seed + 1013904223;
    return z1 ^ z2 ^ z3 ^ z4;
}

// between 0.0 and 1.0
fn seed_to_float(seed: u32) -> f32 {
    return f32(seed) * 2.3283064365387e-10;
}
