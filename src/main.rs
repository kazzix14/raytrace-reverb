use crate::color::Color;
use rand::prelude::*;
use std::sync::Arc;
use vulkano;
use vulkano::command_buffer::CommandBuffer;
use vulkano::sync::GpuFuture;

// m/s
const SPEED_OF_SOUND: f32 = 340.0;

const SAMPLE_RATE: u32 = 44100;

const IMAGE_SIZE: [u32; 2] = [1024, 1024];
const IMAGE_LENGTH: usize = (IMAGE_SIZE[0] * IMAGE_SIZE[1]) as usize;

const WORKGROUP_SIZE: [u32; 3] = [32, 32, 1];
const NUM_DISPATCH: [u32; 3] = [
    IMAGE_SIZE[0] / WORKGROUP_SIZE[0],
    IMAGE_SIZE[1] / WORKGROUP_SIZE[1],
    1,
];

const NUM_RANDOMS: u32 = 128;

fn main() {
    let mut rng = rand::thread_rng();

    let instance = vulkano::instance::Instance::new(
        None,
        &vulkano::instance::InstanceExtensions::none(),
        None,
    )
    .expect("failed to create instance");
    let physical_devices = vulkano::instance::PhysicalDevice::enumerate(&instance);
    println!("{} device(s) found", physical_devices.len());
    physical_devices.for_each(|pd| println!("{} {:?}: {}", pd.index(), pd.ty(), pd.name()));

    let physical_device =
        vulkano::instance::PhysicalDevice::from_index(&instance, 2).expect("failed to find device");

    println!(
        "using {} {:?}: {}",
        physical_device.index(),
        physical_device.ty(),
        physical_device.name()
    );

    let queue_family = physical_device
        .queue_families()
        .filter(|qf| qf.supports_compute())
        .max_by_key(|qf| qf.queues_count())
        .expect("failed to find queue family");

    let (device, mut quques) = vulkano::device::Device::new(
        physical_device,
        physical_device.supported_features(),
        &vulkano::device::DeviceExtensions::supported_by_device(physical_device),
        [(queue_family, 1.0)].iter().cloned(),
    )
    .expect("failed to create device");

    let queue = quques.next().unwrap();

    let compute_shader =
        compute_shader::Shader::load(device.clone()).expect("failed to load compute shader");

    let local_image_buffer = unsafe {
        vulkano::buffer::DeviceLocalBuffer::raw(
            device.clone(),
            IMAGE_LENGTH * 4 * 4,
            vulkano::buffer::BufferUsage::all(),
            device.clone().physical_device().queue_families(),
        )
        .unwrap()
    };

    let local_dist_image_buffer = unsafe {
        vulkano::buffer::DeviceLocalBuffer::raw(
            device.clone(),
            IMAGE_LENGTH * 4 * 4,
            vulkano::buffer::BufferUsage::all(),
            device.clone().physical_device().queue_families(),
        )
        .unwrap()
    };

    let shared_image_buffer = vulkano::buffer::CpuAccessibleBuffer::from_iter(
        device.clone(),
        vulkano::buffer::BufferUsage::all(),
        (0..IMAGE_LENGTH * 4).map(|_| 0.0 as f32),
    )
    .unwrap();

    let shared_dist_image_buffer = vulkano::buffer::CpuAccessibleBuffer::from_iter(
        device.clone(),
        vulkano::buffer::BufferUsage::all(),
        (0..IMAGE_LENGTH * 4).map(|_| 0.0 as f32),
    )
    .unwrap();

    let (local_randoms_buffer, local_randoms_buffer_submit_command) =
        vulkano::buffer::ImmutableBuffer::from_data(
            compute_shader::SizedRandoms {
                randoms: {
                    let mut randoms = [0.0; NUM_RANDOMS as usize];
                    randoms.iter_mut().for_each(|v| *v = rng.gen());
                    randoms
                },
            },
            vulkano::buffer::BufferUsage::all(),
            queue.clone(),
        )
        .unwrap();

    local_randoms_buffer_submit_command
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let (local_atomic_counter_buffer, local_atomic_counter_buffer_submit_command) =
        vulkano::buffer::ImmutableBuffer::from_data(
            compute_shader::ty::AtomicCounter {
                atomic_seed: rng.next_u32(),
                atomic_randoms_index: rng.next_u32(),
            },
            vulkano::buffer::BufferUsage::all(),
            queue.clone(),
        )
        .unwrap();

    local_atomic_counter_buffer_submit_command
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let (local_constants_buffer, local_constants_buffer_submit_command) =
        vulkano::buffer::ImmutableBuffer::from_data(
            compute_shader::ty::Constants {
                image_size: IMAGE_SIZE,
                EPS: 0.00001,
                reflection_count_limit: 8,
                rays_per_pixel: 2,
                num_randoms: NUM_RANDOMS,
            },
            vulkano::buffer::BufferUsage::all(),
            queue.clone(),
        )
        .unwrap();

    local_constants_buffer_submit_command
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let compute_pipeline = Arc::new(
        vulkano::pipeline::ComputePipeline::new(
            device.clone(),
            //&cep,
            &compute_shader.main_entry_point(),
            &(),
        )
        .expect("failed to create compute pipeline"),
    );

    let descriptor_set = Arc::new(
        vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(
            compute_pipeline.clone(),
            0,
        )
        .add_buffer(local_constants_buffer.clone())
        .expect("failed to add local constants buffer")
        .add_buffer(local_atomic_counter_buffer.clone())
        .expect("failed to add local atomic counter buffer")
        .add_buffer(local_randoms_buffer.clone())
        .expect("failed to add local randoms buffer")
        .add_buffer(local_image_buffer.clone())
        .expect("failed to add local image buffer")
        .add_buffer(local_dist_image_buffer.clone())
        .expect("failed to add local image buffer")
        .build()
        .expect("failed to create descriptor set"),
    );

    vulkano::command_buffer::AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .expect("failed to create command buffer builder")
        .dispatch(
            NUM_DISPATCH,
            compute_pipeline.clone(),
            descriptor_set.clone(),
            (),
        )
        .expect("failed to dispatch")
        .copy_buffer(local_image_buffer.clone(), shared_image_buffer.clone())
        .expect("failed to copy buffer")
        .copy_buffer(
            local_dist_image_buffer.clone(),
            shared_dist_image_buffer.clone(),
        )
        .expect("failed to copy buffer")
        .build()
        .expect("failed to build command buffer")
        .execute(queue.clone())
        .expect("failed to execute command buffer")
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    println!("gpu done!!!");

    let mut intensities = shared_image_buffer
        .read()
        .expect("failed to read content")
        .iter()
        .copied()
        .collect::<Vec<f32>>();

    save_as_image(
        &intensities,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        "intensities.png".to_string(),
    );

    let distancies = shared_dist_image_buffer
        .read()
        .expect("failed to read content")
        .iter()
        .copied()
        .collect::<Vec<f32>>();

    save_as_image(
        &distancies,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        "distancies.png".to_string(),
    );

    let radius: f32 = 0.3;

    scale_intensities(&mut intensities, radius, IMAGE_LENGTH, &distancies);

    let mut impulse_response =
        build_intensity_vec(&distancies, &intensities, SPEED_OF_SOUND, SAMPLE_RATE);

    plot(&impulse_response, "ir.pdf");

    const FILTER_WIDTH: usize = 100;

    filter(&mut impulse_response, FILTER_WIDTH);
    plot(&impulse_response, "ir-filtered.pdf");

    ceil_at(&mut impulse_response, 100.0);
    normalize(&mut impulse_response);

    let mut signal = white_noise(impulse_response.len());
    amplitude(&mut signal, &impulse_response);

    write_wav(signal, "ir.wav".to_string());
}

fn scale_intensities(
    intensities: &mut Vec<f32>,
    audio_source_radius: f32,
    image_length: usize,
    distancies: &Vec<f32>,
) {
    intensities
        .iter_mut()
        .for_each(|v| *v /= 4.0 * std::f32::consts::PI * audio_source_radius.powi(2));

    intensities
        .iter_mut()
        .for_each(|v| *v /= image_length as f32);

    decay_intensities_by_distancies(intensities, &distancies);
}

fn save_as_image(source: &Vec<f32>, image_width: u32, image_height: u32, path: String) {
    let normalized = {
        let mut source = source.clone();
        normalize(&mut source);
        set_alpha_to_1(&mut source);
        source
    };

    let normalized_u8 = vec_f32_to_vec_u8(&normalized);

    let image = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
        image_width as u32,
        image_height as u32,
        normalized_u8,
    )
    .expect("failed to create image");

    image.save(path).expect("failed to save image");
}

fn set_alpha_to_1(target: &mut Vec<f32>) {
    assert_eq!(target.len() % 4, 0);
    target.iter_mut().skip(3).step_by(4).for_each(|v| *v = 1.0)
}

fn decay_intensities_by_distancies(intensities: &mut Vec<f32>, distancies: &Vec<f32>) {
    intensities
        .iter_mut()
        .zip(distancies.iter())
        .for_each(|(i, &d)| *i /= d.powi(2));
}

fn vec_f32_to_vec_u8(source: &Vec<f32>) -> Vec<u8> {
    source
        .iter()
        .map(|&v| {
            assert!(0.0 <= v && v <= 1.0, "v: {}", v);
            (v * std::u8::MAX as f32) as u8
        })
        .collect::<Vec<u8>>()
}

fn build_intensity_vec(
    distancies: &Vec<f32>,
    intensities: &Vec<f32>,
    speed_of_sound: f32,
    sample_rate: u32,
) -> Vec<f32> {
    assert_eq!(distancies.len(), intensities.len());

    // convert distance to time
    let times = distancies
        .iter()
        .map(|v| v / speed_of_sound)
        .collect::<Vec<f32>>();

    // convert time to index
    let indicies = times
        .iter()
        .map(|v| v * sample_rate as f32)
        .map(|v| v as usize)
        .collect::<Vec<usize>>();

    let length = indicies.iter().cloned().max().unwrap() + 1;
    let mut intensity_vec = vec![0.0; length];

    indicies
        .iter()
        .cloned()
        .zip(intensities.iter().cloned())
        .for_each(|(dist, value)| {
            intensity_vec[dist] += value;
        });
    intensity_vec
}

fn normalize(target: &mut Vec<f32>) {
    let max = max_of(target);
    target.iter_mut().for_each(|v| *v /= max);
}

fn max_of(target: &Vec<f32>) -> f32 {
    target.iter().fold(std::f32::NAN, |m, v| v.max(m))
}

fn ceil_at(target: &mut Vec<f32>, ceil: f32) {
    target.iter_mut().for_each(|v| *v = v.min(ceil));
}

fn amplitude(lhs: &mut Vec<[f32; 2]>, rhs: &Vec<f32>) {
    assert_eq!(lhs.len(), rhs.len());
    for index in 0..lhs.len() {
        lhs[index][0] *= rhs[index];
        lhs[index][1] *= rhs[index];
    }
}

fn filter(target: &mut Vec<f32>, filter_width: usize) {
    for index in 0..target.len() - filter_width {
        let sum: f32 = target.iter().skip(index).take(filter_width).sum();
        let ave = sum / filter_width as f32;
        target[index] = ave;
    }
}

fn white_noise(length: usize) -> Vec<[f32; 2]> {
    use rand::prelude::*;

    let mut rng = rand::thread_rng();
    let mut signal = vec![[0.0; 2]; length];
    for index in 0..length {
        signal[index] = [rng.gen(), rng.gen()];
    }

    signal
}

fn write_wav(content: Vec<[f32; 2]>, name: String) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(name, spec).unwrap();

    for [sample_left, sample_right] in content {
        let amp = std::i16::MAX as f32;
        writer.write_sample((sample_left * amp) as i16).unwrap();
        writer.write_sample((sample_right * amp) as i16).unwrap();
    }

    writer.finalize().unwrap();
}

fn plot(content: &Vec<f32>, name: &str) {
    use gnuplot::AxesCommon;
    let mut impulse_response_image = gnuplot::Figure::new();
    impulse_response_image
        .axes2d()
        .lines_points(
            0..content.len(),
            content,
            &[
                gnuplot::Caption(""),
                gnuplot::Color("black"),
                //gnuplot::LineWidth(1.0),
                gnuplot::PointSize(1.0),
                gnuplot::PointSymbol('.'),
            ],
        )
        /*
        .lines(
            0..content.len(),
            content,
            &[
                gnuplot::Caption(""),
                gnuplot::Color("black"),
                gnuplot::LineWidth(0.5),
            ],
        )
        */
        .set_y_log(Some(10.0));
    impulse_response_image.save_to_pdf(name, 6, 4).unwrap();
}

mod color {
    use num::Num;

    pub trait Color<T> {
        fn r(&self) -> &T;
        fn g(&self) -> &T;
        fn b(&self) -> &T;
        fn a(&self) -> &T;
        fn r_mut(&mut self) -> &mut T;
        fn g_mut(&mut self) -> &mut T;
        fn b_mut(&mut self) -> &mut T;
        fn a_mut(&mut self) -> &mut T;
    }

    impl<T> Color<T> for [T; 4]
    where
        T: Num,
    {
        fn r(&self) -> &T {
            &self[0]
        }

        fn g(&self) -> &T {
            &self[1]
        }

        fn b(&self) -> &T {
            &self[2]
        }

        fn a(&self) -> &T {
            &self[3]
        }

        fn r_mut(&mut self) -> &mut T {
            &mut self[0]
        }

        fn g_mut(&mut self) -> &mut T {
            &mut self[1]
        }

        fn b_mut(&mut self) -> &mut T {
            &mut self[2]
        }

        fn a_mut(&mut self) -> &mut T {
            &mut self[3]
        }
    }
}

mod compute_shader {
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
            } constants;

            layout(set = 0, binding = 1) buffer AtomicCounter {
                uint atomic_seed;
                uint atomic_randoms_index;
            };

            layout(set = 0, binding = 2) buffer Randoms {
                float randoms[];
            };

            layout(set = 0, binding = 3) buffer ImageBuffer {
                vec4 pixels[];
            } image;

            layout(set = 0, binding = 4) buffer DistImageBuffer {
                vec4 pixels[];
            } dist_image;

            struct Ray {
                vec3 origin;
                vec3 direction;
            };

            /*
            const float dump_ratio = 0.000000;
            const float rand_ratio = 0.2;

            // num perticles / m
            const float particle_density = 1.0;
            const float particle_try_dist = 0.0;
            const float particle_probability = 0.0000;
            const float particle_reflection_ratio = 1.00;
            */

            struct Sphere {
                float radius;
                vec3  position;
                vec3 color;
                float reflection_ratio;
                // boolean
                int emission;
            };

            struct Plane {
                vec3 position;
                vec3 normal;
                vec3 color;
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
                vec3  color;
                float distance;
                vec3  rayDir;
                float intensity;
                float intensity_dump_ratio;
                float diffusion;
            };

            const float dump_ratio = 0.000000;
            const float rand_ratio = 0.3;

            // num perticles / m
            const float particle_density = 1.0;
            const float particle_try_dist = 0.0;
            const float particle_probability = 0.0000;
            const float particle_reflection_ratio = 1.00;

            float global_seed = float(gl_GlobalInvocationID.y * constants.image_size.x + gl_GlobalInvocationID.x);

            Sphere sphere[3];
            Plane plane[6];
            Polygon polygon;
            Polygon polygon2;

            void update_global_seed(float f) {
                float fl = floor(f * 98765.4321);
                float fr = fract(f * 1234.56789);
                float fs = sin(fl + fr);
                global_seed += (fl + fr + fs + f) * 0.135792468;
                //global_seed = sin(global_seed);
            }

            float rand() {
                float local_seed = randoms[atomic_randoms_index % constants.num_randoms] + global_seed;
                update_global_seed(local_seed);
                return fract(sin(dot(vec2(local_seed, global_seed), vec2(12.9898,78.233))) * 43758.5453);
            }

            vec3 random_in_unit_sphere() {
                vec3 p;
                float squared_length;
                uint count = 0;

                float x;
                float y;
                float z;

                do {
                    x = 2.0 * rand() - 1.0;
                    y = 2.0 * rand() - 1.0;
                    z = 2.0 * rand() - 1.0;

                    squared_length = x * x + y * y + z * z;
                } while (squared_length >= 1.0);
                return vec3(x, y, z);
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
                intersection.color    = vec3(0.0);
                intersection.intensity = 1.0;
                intersection.distance = 1.0e+30;
                intersection.rayDir   = vec3(0.0);
                intersection.hit_emission = 0;
                intersection.intensity_dump_ratio = 1.0;
                intersection.diffusion = 0.1;
            }

            void intersect_particle(Ray ray, inout Intersection intersection) {
                /*
                int hit = 0;
                for(float dist = 0.0; dist < particle_try_dist; dist += 1.0/particle_density) {
                    if (dist < intersection.distance && rand() < particle_probability) {
                        intersection.hitPoint = ray.origin + ray.direction * dist;
                        intersection.normal = random_in_unit_sphere();

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
                    //intersection.color = sphere.color * d;
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
                        intersection.intensity_dump_ratio = max(1.0 - intersection.distance * dump_ratio, 0.0);
                        intersection.intensity_dump_ratio *= sphere.reflection_ratio;
                    }
                }
            }

            void intersect_plane(Ray ray, Plane plane, inout Intersection intersection){
                float d = -dot(plane.position, plane.normal);
                float v = dot(ray.direction, plane.normal);
                float t = -(dot(ray.origin, plane.normal) + d) / v;
                if(constants.EPS < t && t < intersection.distance){
                    intersection.hitPoint = ray.origin + ray.direction * t;
                    intersection.normal = plane.normal;
                    intersection.distance = t;
                    intersection.hit++;
                    intersection.rayDir = ray.direction;
                    intersection.diffusion = rand_ratio;
                    intersection.intensity_dump_ratio = max(1.0 - intersection.distance * dump_ratio, 0.0);
                    intersection.intensity_dump_ratio *= plane.reflection_ratio;
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
                
                    intersection.intensity_dump_ratio = max(1.0 - intersection.distance * dump_ratio, 0.0);
                    intersection.intensity_dump_ratio *= polygon.reflection_ratio;
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
                //intersect_polygon(ray, polygon, intersection);
                //intersect_polygon(ray, polygon2, intersection);
                intersect_particle(ray, intersection);
            }

            void compute() {
                atomicAdd(atomic_randoms_index, atomicAdd(atomic_randoms_index, 1));

                float intensity = 0.0;
                float distance = 0.0;
                int hit_count = 0;
                for (int count = 0; count < constants.rays_per_pixel; count++)
                {
                    // -1.0 ..= 1.0
                    vec2 p = vec2(gl_GlobalInvocationID) / constants.image_size * vec2(2.0) - vec2(1.0);
                    p.y = -p.y;
                
                    Ray ray;
                    ray.origin = vec3(0.0, 2.0, 4.0);
                    ray.direction = normalize(vec3(p.x, p.y, -0.5));
                    //ray.direction = normalize(vec3(-0.5, p.y, p.x));
                    //ray.direction = normalize(vec3(p.x, p.y, 0.5));
                
                    // sphere init
                    sphere[0].radius = 0.5;
                    sphere[0].position = vec3(0.0, -0.5, sin(1.0));
                    sphere[0].color = vec3(1.0, 0.0, 0.0);
                    sphere[0].reflection_ratio = 1.0;
                    sphere[0].emission = 0;
                
                    sphere[1].radius = 1.0;
                    sphere[1].position = vec3(2.0, 0.0, cos(1.0 * 0.666));
                    sphere[1].color = vec3(0.0, 1.0, 0.0);
                    sphere[1].reflection_ratio = 1.0;
                    sphere[1].emission = 0;
                
                    sphere[2].radius = 0.3;
                    sphere[2].position = vec3(-2.0, 0.5, -3.0);
                    sphere[2].color = vec3(0.0, 0.0, 1.0);
                    sphere[2].reflection_ratio = 1.0;
                    sphere[2].emission = 1;
                
                    // plane init
                    plane[0].position = vec3(0.0, -5.0, 0.0);
                    plane[0].normal = vec3(0.0, 1.0, 0.0);
                    plane[0].color = vec3(0.0, 1.0, 0.0);
                    plane[0].reflection_ratio = 1.0;
                
                    plane[1].position = vec3(0.0, 5.0, 0.0);
                    plane[1].normal = vec3(0.0, -1.0, 0.0);
                    plane[1].color = vec3(1.0, 0.0, 1.0);
                    plane[1].reflection_ratio = 1.0;
                
                    plane[2].position = vec3(-5.0, 0.0, 0.0);
                    plane[2].normal = vec3(1.0, 0.0, 0.0);
                    plane[2].color = vec3(1.0, 0.0, 0.0);
                    plane[2].reflection_ratio = 1.0;
                
                    plane[3].position = vec3(5.0, 0.0, 0.0);
                    plane[3].normal = vec3(-1.0, 0.0, 0.0);
                    plane[3].color = vec3(0.0, 1.0, 1.0);
                    plane[3].reflection_ratio = 1.0;
                
                    plane[4].position = vec3(0.0, 0.0, -5.0);
                    plane[4].normal = vec3(0.0, 0.0, 1.0);
                    plane[4].color = vec3(0.0, 0.0, 1.0);
                    plane[4].reflection_ratio = 1.0;
                
                    plane[5].position = vec3(0.0, 0.0, 5.0);
                    plane[5].normal = vec3(0.0, 0.0, -1.0);
                    plane[5].color = vec3(1.0, 1.0, 0.0);
                    plane[5].reflection_ratio = 1.0;

                    // init polygon
                    polygon.v0 = vec3(-4.0, 4.0, 1.0);
                    polygon.v1 = vec3(-4.0, 4.0, -4.0);
                    polygon.v2 = vec3(-4.0, -4.0, 1.0);
                    polygon.reflection_ratio = 1.0;
                    polygon.diffusion = 0.3;

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
                    vec3 destColor = vec3(ray.direction.y);
                    vec3 tempColor = vec3(0.0);
                    Ray q;
                    intersectExec(ray, its);
                    if(its.hit > 0){
                        destColor = its.color;
                        tempColor *= its.color;
                        for(int j = 1; j < constants.reflection_count_limit; j++){
                            q.origin = its.hitPoint + its.normal * constants.EPS;
                        
                            q.direction = reflect(its.rayDir, its.normal);
                            q.direction = normalize(q.direction);
                            //q.direction += its.diffusion * random_in_unit_sphere();
                            q.direction += its.diffusion * random_in_unit_sphere();
                            //q.direction = normalize(q.direction);

                            distance += its.distance;
                            //its.distance = 1.0e30;
                        
                            intersectExec(q, its);
                            its.intensity *= its.intensity_dump_ratio;
                            if(its.hit > j){
                                destColor += tempColor * its.color;
                                tempColor *= its.color;
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
                //distance = distance / float(hit_count);
                distance = distance / float(constants.rays_per_pixel);
            
                uint index =
                    gl_GlobalInvocationID.y * constants.image_size.x +
                    gl_GlobalInvocationID.x;                
            
                //image.pixels[index] = vec4(destColor, 1.0);
                image.pixels[index] = vec4(vec3(intensity), 1.0);
                //image.pixels[index] = vec4(vec3(rand()), 1.0);
                dist_image.pixels[index] = vec4(vec3(distance), 1.0);
            }


            void main() {
                compute();
            }
        ",
    }
}
