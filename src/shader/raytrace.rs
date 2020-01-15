use crate::IMAGE_LENGTH;

pub struct Image {
    pub pixels: [[f32; 4]; IMAGE_LENGTH],
}

impl Default for Image {
    fn default() -> Self {
        Self {
            pixels: [[0.0, 0.0, 0.0, 1.0]; IMAGE_LENGTH],
        }
    }
}

pub struct SizedRandoms {
    pub randoms: [f32; crate::NUM_RANDOMS as usize],
}

vulkano_shaders::shader! {
    ty: "compute",
    include: [ "src/glsl" ],
    src: r#"
#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Constants {
    uvec2 image_size;
    float EPS;
    uint reflection_count_limit;
    uint num_randoms;
    float ray_length;
    float source_radius;
    uint num_model_vertices;
} constants;

layout(set = 0, binding = 1) buffer Randoms {
    float randoms[];
};

layout(set = 0, binding = 2) buffer ImageBuffer {
    vec4 pixels[][6];
} image;

layout(set = 0, binding = 3) buffer DistImageBuffer {
    vec4 pixels[][6];
} dist_image;

layout(set = 0, binding = 4) buffer ModelVertices {
    vec3 vertices[];
} model;

layout(set = 0, binding = 5) buffer ModelMaterialIndices {
    uint indices[];
} material_indices;

layout(set = 0, binding = 6) buffer ModelMaterials {
    vec3 materials[];
} materials;

struct Ray {
    vec3 origin;
    vec3 direction;
};

/*
const float dump_ratio = 0.000000;

// num perticles / m
const float particle_density = 1.0;
const float particle_try_dist = 0.0;
const float particle_probability = 0.0000;
const float particle_reflection_ratio = 1.00;
*/

struct Sphere {
    float radius;
    vec3  position;
    float reflection_ratio;
    // boolean
    int emission;
};

struct Plane {
    vec3 position;
    vec3 normal;
    float reflection_ratio;
};

struct Polygon {
    vec3 v0;
    vec3 v1;
    vec3 v2;
    float reflection_ratio;
    float diffusion;
    int emission;
};

struct Intersection {
    int   hit;
    int hit_emission;
    vec3  hitPoint;
    vec3  normal;
    float distance;
    vec3  rayDir;
    float intensity;
    float intensity_dump_ratio;
    float diffusion;
    float distance_to_live;
};

const float rand_ratio = 0.47;

// num perticles / m
const float particle_density = 1.0;
const float particle_try_dist = 0.0;
const float particle_probability = 0.0000;
const float particle_reflection_ratio = 1.00;

Sphere audio_source;

uint invocation_id() {
    return gl_GlobalInvocationID.y * constants.image_size.x + gl_GlobalInvocationID.x;
}

uint image_id() {
    return gl_GlobalInvocationID.z;
}

float rand2(vec2 co) {
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}
vec3 random_in_unit_sphere2(float seed) {
    vec3 p;
    float squared_length;
    uint count = 0;
    float x;
    float y;
    float z;
    do {
        count += 1;
        vec2 seed1 = gl_GlobalInvocationID.xy + sin(float(count * 0.1) + seed);
        vec2 seed2 = gl_GlobalInvocationID.yz + sin(float(count * 0.23) + seed + 1.0);
        vec2 seed3 = gl_GlobalInvocationID.xz + sin(float(count * 0.12) + seed + 2.0);
        x = 2.0 * rand2(seed1) - 1.0;
        y = 2.0 * rand2(seed2) - 1.0;
        z = 2.0 * rand2(seed3) - 1.0;
        squared_length = x * x + y * y + z * z;
    } while (squared_length >= 1.0);
    return vec3(x, y, z);
}

float seed_from_ray_intersection(Ray ray, Intersection intersection) {
    float from_ray = 
        ray.direction.z * 1.2345 +
        ray.direction.y * 1.2345 +
        ray.direction.x;

    float from_intersection =
        float(intersection.hit) * 1.2345 +
        intersection.normal.x + intersection.normal.y + 0.123 + intersection.normal.z +
        intersection.hitPoint.x + 0.321 + intersection.hitPoint.z + intersection.hitPoint.y +
        intersection.intensity +
        intersection.diffusion;

    float from_invocation_id = invocation_id();

    return fract(sin(from_ray + from_intersection * 1.23) + invocation_id() * 1.314);
}

float random_hash(int seed) {
    uint index = gl_GlobalInvocationID.z * constants.image_size.y * constants.image_size.x +
        gl_GlobalInvocationID.y * constants.image_size.x +
        gl_GlobalInvocationID.x;
    return randoms[uint(float(index + seed * 12389)) % constants.num_randoms];
}

float rand(float seed) {
    float r1 = fract(sin(dot(vec2(seed, 23.1395861 * random_hash(int(seed*2.3))), vec2(12.9898,78.233))) * 43758.5453);
    float r2 = fract(sin(dot(vec2(r1, 17.234565 * random_hash(int(r1*3.88))), vec2(12.9898,78.233))) * 43758.5453);
    return fract(sin(r1 - r2));
}

vec3 random_in_unit_sphere(float seed) {
    vec3 p;
    float squared_length;
    float count = seed;

    float x;
    float y;
    float z;

    float seed1 = floor(seed + 1.2) * fract(seed + 3.2);
    float seed2 = floor(seed + 3.2) * fract(seed + 1.2);
    float seed3 = floor(seed + 2.2) * fract(seed + 3.3);

    seed1 += gl_GlobalInvocationID.x * gl_GlobalInvocationID.y * 0.123;
    seed2 += gl_GlobalInvocationID.y * gl_GlobalInvocationID.z * 0.321;
    seed3 += gl_GlobalInvocationID.z * gl_GlobalInvocationID.x * 0.211;

    do {
        count += 0.111222333 * seed;

        x = 2.0 * rand((count + seed1 * 1.234)) - 1.0;
        y = 2.0 * rand((count + seed2 * 4.567)) - 1.0;
        z = 2.0 * rand((count + seed3 * 8.901)) - 1.0;

        squared_length = x * x + y * y + z * z;
    } while (squared_length >= 1.0);
    return vec3(x, y, z);
}

float determinant( vec3 a, vec3 b, vec3 c ) {
    return (a.x * b.y * c.z)
            + (a.y * b.z * c.x)
            + (a.z * b.x * c.y)
            - (a.x * b.z * c.y)
            - (a.y * b.x * c.z)
            - (a.z * b.y * c.x);
}

void initialize_intersection(inout Intersection intersection) {
    intersection.hit      = 0;
    intersection.hitPoint = vec3(0.0);
    intersection.normal   = vec3(0.0);
    intersection.intensity = 1.0;
    intersection.distance = constants.ray_length;
    intersection.rayDir   = vec3(0.0);
    intersection.hit_emission = 0;
    intersection.intensity_dump_ratio = 1.0;
    intersection.diffusion = 1.0;
    intersection.distance_to_live = constants.ray_length;
}

void intersect_particle(Ray ray, inout Intersection intersection) {
    /*
    int hit = 0;
    for(float dist = 0.0; dist < particle_try_dist; dist += 1.0/particle_density) {
        if (dist < intersection.distance && rand(dist) < particle_probability) {
            intersection.hitPoint = ray.origin + ray.direction * dist;
            intersection.normal = random_in_unit_sphere(dist);

            intersection.distance = dist;
            intersection.hit++;
            intersection.rayDir = ray.direction;
            intersection.intensity_dump_ratio *= particle_reflection_ratio;
            break;
        }
    }
    */
}

// have to be executed at last
void intersect_sphere(Ray ray, Sphere sphere, inout Intersection intersection){
    vec3  a = ray.origin - sphere.position;
    float b = dot(a, ray.direction);
    float c = dot(a, a) - (sphere.radius * sphere.radius);
    float d = b * b - c;
    float t = -b - sqrt(d);
    if(0.0 < d && constants.EPS < t && t < intersection.distance){
        intersection.hitPoint = ray.origin + ray.direction * t;
        intersection.normal = normalize(intersection.hitPoint - sphere.position);
        intersection.distance = t;
        intersection.hit++;
        intersection.rayDir = ray.direction;
        intersection.diffusion = rand_ratio;
    
        if(sphere.emission == 1) {
            intersection.hit_emission = 1;
            intersection.intensity_dump_ratio = 1.0;
        }
        else
        {
            intersection.hit_emission = 0;
            intersection.intensity_dump_ratio = sphere.reflection_ratio;
        }
    }
}

void intersect_plane(Ray ray, Plane plane, inout Intersection intersection){
    float d = -dot(plane.position, plane.normal);
    float v = dot(ray.direction, plane.normal);
    float t = -(dot(ray.origin, plane.normal) + d) / v;
    if(constants.EPS < t && t < intersection.distance){
        intersection.hitPoint = ray.origin + ray.direction * t;
        intersection.normal = normalize(plane.normal);
        intersection.distance = t;
        intersection.hit++;
        intersection.rayDir = ray.direction;
        intersection.diffusion = rand_ratio;
        intersection.intensity_dump_ratio = plane.reflection_ratio;
    }
}

void intersect_polygon(Ray ray, Polygon polygon, inout Intersection intersection) {
    vec3 invRay = -ray.direction;
    vec3 edge1 = polygon.v1 - polygon.v0;
    vec3 edge2 = polygon.v2 - polygon.v0;

    float denominator =  determinant( edge1, edge2, invRay );
    if ( denominator == 0.0 ) return;

    float invDenominator = 1.0 / denominator;
    vec3 d = ray.origin - polygon.v0;

    float u = determinant( d, edge2, invRay ) * invDenominator;
    if ( u < 0.0 || u > 1.0 ) return;

    float v = determinant( edge1, d, invRay ) * invDenominator;
    if ( v < 0.0 || u + v > 1.0 ) return;

    float t = determinant( edge1, edge2, d ) * invDenominator;
    if ( t < 0.0 || t > intersection.distance ) return;

    if(constants.EPS < t && t < intersection.distance) {
        intersection.hitPoint = ray.origin + ray.direction * t;
        intersection.normal   = normalize( cross( edge1, edge2 ) ) * sign( invDenominator );
        intersection.distance = t;
        intersection.hit++;
        intersection.rayDir = ray.direction;
        intersection.diffusion = polygon.diffusion;
    
        if(polygon.emission == 1) {
            intersection.hit_emission = 1;
            intersection.intensity_dump_ratio = 1.0;
        }
        else
        {
            intersection.hit_emission = 0;
            intersection.intensity_dump_ratio = polygon.reflection_ratio;
        }
    }
}

void intersect_all(Ray ray, inout Intersection intersection){
    //intersect_particle(ray, intersection);

    for(uint index = 0; index < constants.num_model_vertices; index += 3) {
        uint i = index;

        uint material_index = material_indices.indices[i/3];
        vec3 material = materials.materials[material_index];
        float reflection = material.x;
        float diffusion = material.y;
        int emission = 0;
        if ( 1.0 < material.z) {
            emission = 1;
        };

        Polygon p = Polygon(
            model.vertices[i],
            model.vertices[i + 1],
            model.vertices[i + 2],
            reflection,
            diffusion,
            emission
        );

        intersect_polygon(ray, p, intersection);
    }

    //intersect_sphere(ray, audio_source, intersection);
}

void compute() {

    float intensity = 0.0;
    float distance_ray = 0.0;

    // -1.0 ..= 1.0
    vec2 p = vec2(gl_GlobalInvocationID.xy) / constants.image_size * vec2(2.0) - vec2(1.0);
    p.y = -p.y;
    
    Ray ray;
    ray.origin = vec3(0.0, 0.0, 0.0);
    switch(image_id()) {
        case 0:
            ray.direction = normalize(vec3(p.x, p.y, -0.5)); // front
            break;

        case 1:
            ray.direction = normalize(vec3(-p.x, p.y, 0.5)); // back
            break;

        case 2:
            ray.direction = normalize(vec3(-0.5, p.y, -p.x)); // left
            break;

        case 3:
            ray.direction = normalize(vec3(0.5, p.y, p.x)); // right
            break;

        case 4:
            ray.direction = normalize(vec3(p.x, 0.5, p.y)); // top
            break;

        case 5:
            ray.direction = normalize(vec3(p.x, -0.5, -p.y)); // bottom
            break;
    }
    
    audio_source.radius = constants.source_radius;
    audio_source.position = vec3(0.0, 0.0, -2.0);
    audio_source.reflection_ratio = 1.0;
    audio_source.emission = 1;
    
    // intersection init
    Intersection its;
    initialize_intersection(its);
    
    // hit check
    Ray q;
    intersect_all(ray, its);
    if(0 < its.hit && its.hit_emission <= 0) {
        its.hit = 0;
        its.intensity *= its.intensity_dump_ratio;

        for(int j = 1; j < constants.reflection_count_limit; j++){
            q.origin = its.hitPoint + its.normal * constants.EPS;
        
            q.direction = reflect(its.rayDir, its.normal);
            q.direction = normalize(q.direction);
            float seed = seed_from_ray_intersection(q, its) - float(j) * 1.34;
            q.direction += its.diffusion * random_in_unit_sphere(seed);
            q.direction = normalize(q.direction);

            distance_ray += its.distance;
            its.distance_to_live -= its.distance;

            its.distance = constants.ray_length;

            if (its.distance_to_live < 0.0) {
                break;
            }
        
            intersect_all(q, its);

            if (0 < its.hit_emission) {
                break;
            }

            if (its.hit <= 0) {
                break;
            }

            its.hit = 0;

            its.intensity *= its.intensity_dump_ratio;
        }
    }
    
    if (its.hit_emission == 0) {
        its.intensity = 0;
    }

    intensity = its.intensity;

    distance_ray += constants.source_radius;

    image.pixels[invocation_id()][image_id()] = vec4(vec3(intensity), 1.0);
    dist_image.pixels[invocation_id()][image_id()] = vec4(vec3(distance_ray), 1.0);
    //dist_image.pixels[invocation_id()][image_id()] = vec4(vec3(its.distance), 1.0);
}


void main() {
    compute();
}
"#,
}
