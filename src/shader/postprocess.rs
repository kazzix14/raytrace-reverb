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
    uint rays_per_pixel;
} constants;

layout(set = 0, binding = 1) buffer Intensities {
    vec4 intensities[];
};

layout(set = 0, binding = 2) buffer Distancies {
    vec4 distancies[];
};

layout(set = 0, binding = 3) buffer Reflections {
    vec4 reflections[];
};

uint invocation_id() {
    return gl_GlobalInvocationID.x;
}

void compute() {

    // out of bounds
    uint idx = invocation_id();
    if (constants.image_length * 6 <= idx) return;

    // alpha = 1.0
    //float m = 0.230259;

    // alpha = 0.7
    float m = 0.1611813 / 2.0;

    float distance = distancies[idx].x * 0.001;
    float dist_decay = exp(-m * distance); 

    float rays_per_pixel_decay = 1.0 / float(constants.rays_per_pixel);
    float num_channels_decay = 1.0 / 4.0;

    float reflection = reflections[idx].x;

    float decay = dist_decay * num_channels_decay * reflection;

    if (intensities[idx].xyz == vec3(0.0)) 
        ;
    else
        intensities[idx] *= vec4(decay);
}

void main() {
    compute();
}
        ",
}
