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

#define PI 3.1415926535897932384626433832795

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Constants {
    float radius;
    uint image_length;
} constants;

layout(set = 0, binding = 1) buffer Intensities {
    vec4 intensities[];
};

layout(set = 0, binding = 2) buffer Distancies {
    vec4 distancies[];
};

uint invocation_id() {
    return gl_GlobalInvocationID.x;
}

void compute() {
    uint idx = invocation_id();
    if (constants.image_length * 6 <= idx) return;
    vec3 dist_decay = distancies[idx].xyz * distancies[idx].xyz;
    //vec3 radius_decay = vec3(4.0 * PI * constants.radius * constants.radius);
    //vec3 image_size_decay = vec3(float(constants.image_length * 6));
    //intensities[idx] /= vec4(dist_decay * radius_decay * image_size_decay, 1.0);
    intensities[idx] /= vec4(dist_decay, 1.0);
}

void main() {
    compute();
}
        ",
}
