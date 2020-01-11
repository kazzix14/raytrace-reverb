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
    src: "
#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Constants {
    uvec2 image_size;
    float EPS;
    uint reflection_count_limit;
    uint rays_per_pixel;
    uint num_randoms;
    float ray_length;
    float source_radius;
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

const float dump_ratio = 0.000000;
const float rand_ratio = 0.07;

// num perticles / m
const float particle_density = 1.0;
const float particle_try_dist = 0.0;
const float particle_probability = 0.0000;
const float particle_reflection_ratio = 1.00;

Sphere sphere[3];
Plane plane[6];
Polygon polygon;
Polygon polygon2;

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
        vec2 seed1 = gl_GlobalInvocationID.xy + (float(count) + seed);
        vec2 seed2 = gl_GlobalInvocationID.yz + (float(count) + seed + 1.0);
        vec2 seed3 = gl_GlobalInvocationID.xz + (float(count) + seed + 2.0);
        x = 2.0 * rand2(seed1) - 1.0;
        y = 2.0 * rand2(seed2) - 1.0;
        z = 2.0 * rand2(seed3) - 1.0;
        squared_length = x * x + y * y + z * z;
    } while (squared_length >= 1.0);
    return vec3(x, y, z);
}

float seed_from_ray_intersection(Ray ray, Intersection intersection) {
    float from_ray = 
        ray.direction.z * 1.2345 * 6.789 +
        ray.direction.y * 1.2345 * ray.direction.x;

    float from_intersection =
        float(intersection.hit) * 1.2345 +
        intersection.normal.x + intersection.normal.y + 0.123 +
        intersection.hitPoint.x + 0.321 + intersection.hitPoint.z +
        intersection.intensity +
        intersection.diffusion;

    float from_invocation_id = invocation_id();

    return fract(sin(from_ray + from_intersection * 1.23) * invocation_id());
}

float random_hash(int seed) {
    uint index = gl_GlobalInvocationID.z * constants.image_size.y * constants.image_size.x +
        gl_GlobalInvocationID.y * constants.image_size.x +
        gl_GlobalInvocationID.x;
    return randoms[uint(float(index + seed * 12389)) % constants.num_randoms];
}

float rand(float seed) {
    float r1 = fract(sin(dot(vec2(seed, random_hash(int(seed*2.3))), vec2(12.9898,78.233))) * 43758.5453);
    float r2 = fract(sin(dot(vec2(r1, random_hash(int(r1*3.88))), vec2(12.9898,78.233))) * 43758.5453);
    return fract(sin(seed + r1 - r2));
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

    seed1 += gl_GlobalInvocationID.x + gl_GlobalInvocationID.y;
    seed2 += gl_GlobalInvocationID.y + gl_GlobalInvocationID.z;
    seed3 += gl_GlobalInvocationID.z + gl_GlobalInvocationID.x;

    do {
        count += 0.111222333 * seed;

        x = 2.0 * rand(count + seed1 * 12.34) - 1.0;
        y = 2.0 * rand(count + seed2 * 45.67) - 1.0;
        z = 2.0 * rand(count + seed3 * 89.01) - 1.0;

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
            intersection.intensity_dump_ratio = max(1.0 - intersection.distance * dump_ratio, 0.0);
            intersection.intensity_dump_ratio *= particle_reflection_ratio;
            break;
        }
    }
    */
}

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
            intersection.intensity *= max(1.0 - intersection.distance * dump_ratio, 0.0);
            intersection.hit_emission += 1;
        }
        else
        {
            //intersection.intensity_dump_ratio = max(1.0 - intersection.distance * dump_ratio, 0.0);
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
        //intersection.intensity_dump_ratio = max(1.0 - intersection.distance * dump_ratio, 0.0);
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
    
        //intersection.intensity_dump_ratio = max(1.0 - intersection.distance * dump_ratio, 0.0);
        intersection.intensity_dump_ratio = polygon.reflection_ratio;
    }
}

void intersectExec(Ray ray, inout Intersection intersection){
    intersect_sphere(ray, sphere[0], intersection);
    intersect_sphere(ray, sphere[1], intersection);
    intersect_sphere(ray, sphere[2], intersection);
    intersect_plane(ray, plane[0], intersection);
    intersect_plane(ray, plane[1], intersection);
    intersect_plane(ray, plane[2], intersection);
    intersect_plane(ray, plane[3], intersection);
    intersect_plane(ray, plane[4], intersection);
    intersect_plane(ray, plane[5], intersection);
    intersect_polygon(ray, polygon, intersection);
    intersect_polygon(ray, polygon2, intersection);
    intersect_particle(ray, intersection);
}

void compute() {

    float intensity = 0.0;
    float distance = 0.0;
    int hit_count = 0;
    for (int count = 0; count < constants.rays_per_pixel; count++)
    {
        // -1.0 ..= 1.0
        vec2 p = vec2(gl_GlobalInvocationID.xy) / constants.image_size * vec2(2.0) - vec2(1.0);
        p.y = -p.y;
    
        Ray ray;
        ray.origin = vec3(0.0, 2.0, 4.0);
        switch(image_id()) {
            case 0:
                ray.direction = normalize(vec3(p.x, p.y, -0.5)); // front
                break;

            case 1:
                ray.direction = normalize(vec3(p.x, p.y, 0.5)); // back
                break;

            case 2:
                ray.direction = normalize(vec3(-0.5, p.y, p.x)); // left
                break;

            case 3:
                ray.direction = normalize(vec3(0.5, p.y, p.x)); // right
                break;

            case 4:
                ray.direction = normalize(vec3(p.x, 0.5, p.y)); // top
                break;

            case 5:
                ray.direction = normalize(vec3(p.x, -0.5, p.y)); // bottom
                break;
        }
    
        // sphere init
        sphere[0].radius = 0.5;
        sphere[0].position = vec3(0.0, -0.5, sin(1.0));
        sphere[0].reflection_ratio = 1.0;
        sphere[0].emission = 0;
    
        sphere[1].radius = 1.0;
        sphere[1].position = vec3(2.0, 0.0, 0.0);
        sphere[1].reflection_ratio = 1.0;
        sphere[1].emission = 0;
    
        sphere[2].radius = constants.source_radius;
        sphere[2].position = vec3(-2.0, 0.5, -5.0);
        sphere[2].reflection_ratio = 1.0;
        sphere[2].emission = 1;
    
        // plane init
        plane[0].position = vec3(0.0, -5.0, 0.0);
        plane[0].normal = vec3(0.0, 1.0, 0.0);
        plane[0].reflection_ratio = 1.0;
    
        plane[1].position = vec3(0.0, 5.0, 0.0);
        plane[1].normal = vec3(0.0, -1.0, 0.0);
        plane[1].reflection_ratio = 1.0;
    
        plane[2].position = vec3(-5.0, 0.0, 0.0);
        plane[2].normal = vec3(1.0, 0.0, 0.0);
        plane[2].reflection_ratio = 1.0;
    
        plane[3].position = vec3(5.0, 0.0, 0.0);
        plane[3].normal = vec3(-1.0, 0.0, 0.0);
        plane[3].reflection_ratio = 1.0;
    
        plane[4].position = vec3(0.0, 0.0, -10.0);
        plane[4].normal = vec3(0.0, 0.0, 1.0);
        plane[4].reflection_ratio = 1.0;
    
        plane[5].position = vec3(0.0, 0.0, 5.0);
        plane[5].normal = vec3(0.0, 0.0, -1.0);
        plane[5].reflection_ratio = 1.0;

        // init polygon
        polygon.v0 = vec3(-4.0, 4.0, 1.0);
        polygon.v1 = vec3(-4.0, 4.0, -4.0);
        polygon.v2 = vec3(-4.0, -4.0, 1.0);
        polygon.reflection_ratio = 1.0;
        polygon.diffusion = 0.01;

        // init polygon
        polygon2.v0 = vec3(4.0, 4.0, 1.0);
        polygon2.v1 = vec3(4.0, 4.0, -4.0);
        polygon2.v2 = vec3(4.0, -4.0, 1.0);
        polygon2.reflection_ratio = 1.0;
        polygon2.diffusion = 1.3;
    
        // intersection init
        Intersection its;
        initialize_intersection(its);
    
        // hit check
        Ray q;
        intersectExec(ray, its);
        if(0 < its.hit){
            for(int j = 1; j < constants.reflection_count_limit; j++){
                q.origin = its.hitPoint + its.normal * constants.EPS;
            
                q.direction = reflect(its.rayDir, its.normal);
                q.direction = normalize(q.direction);
                float seed = seed_from_ray_intersection(q, its) + float(j) * 23.4 - float(count) * 2.23;
                q.direction += its.diffusion * random_in_unit_sphere(seed);
                q.direction = normalize(q.direction);

                distance += its.distance;
                its.distance_to_live -= its.distance;

                its.distance = constants.ray_length;

                if (its.distance_to_live < 0.0) {
                    break;
                }
            
                intersectExec(q, its);
                its.intensity *= its.intensity_dump_ratio;
                if(its.hit > j){
                }

                if (0 < its.hit_emission) {
                    break;
                }
            }
        }
    
        if (its.hit_emission == 0) {
            its.intensity = 0;
        }
        else{
            hit_count += 1;
        }
        intensity += its.intensity;
    }

    intensity = intensity / float(constants.rays_per_pixel);
    distance = distance / float(constants.rays_per_pixel);

    image.pixels[invocation_id()][image_id()] = vec4(vec3(intensity), 1.0);
    dist_image.pixels[invocation_id()][image_id()] = vec4(vec3(distance), 1.0);
}


void main() {
    compute();
}
        ",
}
