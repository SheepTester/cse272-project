const PI = radians(180.0);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> canvas_size: vec2<f32>;
@group(0) @binding(1) var<uniform> sample_to_cam: mat4x4<f32>;
@group(0) @binding(2) var<uniform> cam_to_world: mat4x4<f32>;

@vertex
fn vertex_main(
    @builtin(vertex_index) index: u32,
) -> VertexOutput {
    const square_vertices = array(
        vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0),
        vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0),
    );

    var result: VertexOutput;
    result.position = vec4(vec2(square_vertices[index].x * 2.0 - 1.0, 1.0 - 2.0 * square_vertices[index].y), 0.0, 1.0);
    result.uv = square_vertices[index];
    return result;
}

struct Medium {
    // homogeneous mediums for now
    sigma_a: f32,
    sigma_s: f32,
}

struct Sphere {
    // material_id: i32,
    light_id: i32,
    interior_medium_id: i32,
    exterior_medium_id: i32,

    center: vec3<f32>,
    radius: f32,
}

struct Light {
    shape_id: i32,
    // aka radiance
    intensity: vec3<f32>,
}

// volpath_test1
const scene_media: array<Medium, 1> = array(Medium(0.1 * 1, 0.7 * 1));
const scene_shape_count: i32 = 2;
const scene_shapes: array<Sphere, scene_shape_count> = array(
    Sphere(0, -1, 0, vec3(0), 1), // shape 0
    Sphere(1, -1, 0, vec3(-3, 0, -1.5), 1), // shape 1
);
const scene_light: array<Light, 2> = array(
    Light(0, vec3(0.4, 2.32, 3.2)), // light 0
    Light(1, vec3(24, 10, 24)), // light 1
);
const camera_medium_id: i32 = 0;

struct IntersectResult {
    shape_id: i32,
    distance: f32,
}

/// returns index of shape, or -1 if none
fn intersect_scene(ray: Ray) -> IntersectResult {
    var best = IntersectResult(-1, 1.0e5); // infinity not supported
    for (var i = 0; i < scene_shape_count; i++) {
        let sphere = scene_shapes[i];

        // thanks ChatGPT
        let origin_to_center = ray.origin - sphere.center;

        let direction_dot = dot(ray.dir, ray.dir);
        let b_term = 2.0 * dot(origin_to_center, ray.dir);
        let c_term = dot(origin_to_center, origin_to_center) - sphere.radius * sphere.radius;

        let discriminant = b_term * b_term - 4.0 * direction_dot * c_term;

        if discriminant < 0.0 {
            // No intersection
            continue;
        }

        let sqrt_discriminant = sqrt(discriminant);
        let t_near = (-b_term - sqrt_discriminant) / (2.0 * direction_dot);
        let t_far = (-b_term + sqrt_discriminant) / (2.0 * direction_dot);

        if t_near > 0.0 {
            // Closest valid intersection
            if (t_near < best.distance) {
                best.shape_id = i;
                best.distance = t_near;
            }
            continue;
        }
        if t_far > 0.0 {
            // Ray starts inside the sphere
            if (t_far < best.distance) {
                best.shape_id = i;
                best.distance = t_far;
            }
            continue;
        }
    }
    return best;
}

@fragment
fn fragment_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    var seed = seed_per_thread(u32((vertex.uv.x * canvas_size.x + vertex.uv.y) * canvas_size.y) + 69);
    seed = next_seed(seed);

    let dx = seed_to_float(seed);
    seed = next_seed(seed);
    let dy = seed_to_float(seed);
    seed = next_seed(seed);
    let offset = gaussian_filter(vec2(dx, dy));
    let remapped_pos = vertex.uv + (vec2(0.5) + offset) / canvas_size;
    let dir = normalize(sample_to_cam * vec4(remapped_pos, 0.0, 1.0)).xyz;
    let ray = Ray((cam_to_world * vec4(vec3(0.0), 1.0)).xyz, normalize(cam_to_world * vec4(dir, 0.0)).xyz);

    let current_medium_id = camera_medium_id;

    let result = intersect_scene(ray);

    if (result.shape_id == -1) {
        return vec4(vec3(0), 1);
    }

    let shape = scene_shapes[result.shape_id];
    let vertex_position = ray.origin + result.distance * ray.dir;
    let vertex_normal = normalize(vertex_position - shape.center);

    if (shape.light_id == -1) {
        return vec4(vec3(0), 1);
    }

    var transmittance = vec3(1.0);
    if (current_medium_id >= 0) {
        let medium = scene_media[current_medium_id];
        let sigma_t = medium.sigma_a + medium.sigma_s;
        let t = result.distance;
        transmittance = vec3(exp(-sigma_t * t));
    }
    let emission = select(scene_light[shape.light_id].intensity, vec3(0.0), dot(vertex_normal, -ray.dir) <= 0);
    return vec4(transmittance * emission, 1);
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

// Filters: default box filter for now

const FILTER_WIDTH: f32 = 1.0;
fn box_filter(rand: vec2<f32>) -> vec2<f32> {
    // Warp [0, 1]^2 to [-width/2, width/2]^2
    return (2.0 * rand - 1.0) * (FILTER_WIDTH / 2);
}

// defaults to 0.5
const FILTER_STDDEV: f32 = 0.5;
fn gaussian_filter(rand: vec2<f32>) -> vec2<f32> {
    let r = FILTER_STDDEV * sqrt(-2 * log(max(rand.x, 1e-8)));
    return vec2(
        r * cos(2 * PI * rand.y),
        r * sin(2 * PI * rand.y),
    );
}

// Rays

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
};
