const PI = radians(180.0);
const INFINITY = 1.0e5;

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
const scene_lights: array<Light, 2> = array(
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
    var best = IntersectResult(-1, INFINITY); // infinity not supported
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

    var radiance = vec3(0.0);
    let current_medium_id = camera_medium_id;
    if (current_medium_id < 0) {
        return vec4(vec3(0.0), 1.0);
    }

    let result = intersect_scene(ray);
    let vertex_position = ray.origin + result.distance * ray.dir;

    let medium = scene_media[current_medium_id];
    var max_t = INFINITY;
    if (result.shape_id != -1) {
        max_t = result.distance;
    }
    let sigma_t = medium.sigma_a + medium.sigma_s;

    let t = -log(1 - seed_to_float(seed)) / sigma_t;
    seed = next_seed(seed);

    if (t >= max_t) {
        if (result.shape_id != -1) {
            let shape = scene_shapes[result.shape_id];
            if (shape.light_id != -1) {
                let vertex_normal = normalize(vertex_position - shape.center);
                radiance += select(scene_lights[shape.light_id].intensity, vec3(0.0), dot(vertex_normal, -ray.dir) <= 0);
            }
        }
        return vec4(radiance, 1);
    }

    if (result.shape_id == -1) {
        return vec4(vec3(0), 1);
    }

    let shape = scene_shapes[result.shape_id];
    let vertex_interior_medium_id = current_medium_id;
    let vertex_exterior_medium_id = current_medium_id;

    let transmittance = exp(-sigma_t * t);

    var c1 = vec3(0.0);
    {
        let du = seed_to_float(seed);
        seed = next_seed(seed);
        let dv = seed_to_float(seed);
        seed = next_seed(seed);
        let light_w = seed_to_float(seed);
        seed = next_seed(seed);
        let shape_w = seed_to_float(seed);
        seed = next_seed(seed);
        let light_id = 1; // TODO: sample light
        let light = scene_lights[light_id];
        let point_on_light = sample_point_on_sphere(scene_shapes[light.shape_id], vertex_position, vec2(du, dv));
        var g = 0.0;
        let dir_light = normalize(point_on_light.position - vertex_position);
        let shadow_ray = Ray(vertex_position, dir_light); // doesn't use get_shadow_epsilon(scene)
        if (true) { // TODO: !occluded(scene, shadow_ray)
            g = max(-dot(dir_light, point_on_light.normal), 0.0) / distance_squared(point_on_light.position, vertex_position);
        }
        // TODO: p1
        let p1 = 1.0;
        if (g > 0 && p1 > 0) {
            let dir_view = -ray.dir;
            let f = eval_phase_function(dir_view, dir_light);
            let l = select(scene_lights[shape.light_id].intensity, vec3(0.0), dot(point_on_light.normal, -dir_light) <= 0); // TODO: sus
            let t = exp(-sigma_t * distance(vertex_position, point_on_light.position));
            c1 = medium.sigma_s * transmittance * t * g * f * l / p1;
        }
    }
    radiance += c1;
    return vec4(radiance, 1);
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

// Phase functions (HenyeyGreenstein)

const HENYEY_G: f32 = -0.5;

fn eval_phase_function(dir_in: vec3<f32>, dir_out: vec3<f32>) -> vec3<f32> {
    return vec3(
        1 / (4 * PI) * (1 - HENYEY_G * HENYEY_G) /
            pow((1 + HENYEY_G * HENYEY_G + 2 * HENYEY_G * dot(dir_in, dir_out)), 1.5)
    );
}

// Rays

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
};

// Sampling

struct PointAndNormal {
    position: vec3<f32>,
    normal: vec3<f32>,
}
fn sample_point_on_sphere(
    sphere: Sphere,
    ref_point: vec3<f32>,
    uv: vec2<f32>,
) -> PointAndNormal {
    if (distance_squared(ref_point, sphere.center) < sphere.radius * sphere.radius) {
        let z = 1.0 - 2.0 * uv.x;
        let r = sqrt(max(0.0, 1.0 - z * z));
        let phi = 2.0 * PI * uv.y;
        let offset = vec3(r * cos(phi), r * sin(phi), z);
        let position = sphere.center + sphere.radius * offset;
        let normal = offset;
        return PointAndNormal(position, normal);
    }

    let dir_to_center = normalize(sphere.center - ref_point);

    let sin_elevation_max_sq = sphere.radius * sphere.radius / distance_squared(ref_point, sphere.center);
    let cos_elevation_max = sqrt(max(0.0, 1.0 - sin_elevation_max_sq));
    let cos_elevation = (1.0 - uv.x) + uv.x * cos_elevation_max;
    let sin_elevation = sqrt(max(0.0, 1.0 - cos_elevation * cos_elevation));
    let azimuth = uv.y * 2 * PI;

    let dc = distance(ref_point, sphere.center);
    let ds = dc * cos_elevation - sqrt(max(0.0, sphere.radius * sphere.radius - dc * dc * sin_elevation * sin_elevation));
    let cos_alpha = (dc * dc + sphere.radius * sphere.radius - ds * ds) / (2 * dc * sphere.radius);
    let sin_alpha = sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha));
    let n_on_sphere = -to_world_frame(dir_to_center, vec3(sin_alpha * cos(azimuth), sin_alpha * sin(azimuth), cos_alpha));
    let p_on_sphere = sphere.radius * n_on_sphere + sphere.center;
    return PointAndNormal(p_on_sphere, n_on_sphere);
}

fn to_world_frame(n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let a = 1 / (1 + n.z);
    let b = -n.x * n.y * a;
    let x = select(
        vec3(1 - n.x * n.x * a, b, -n.x),
        vec3(0, -1, 0),
        n.z < -1 + 1e-6,
    );
    return x * v;
}

fn distance_squared(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let diff = a - b;
    return dot(diff, diff);
}
