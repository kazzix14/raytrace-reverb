use crate::color::Color;
use std::sync::Arc;
use vulkano;
use vulkano::command_buffer::CommandBuffer;
use vulkano::sync::GpuFuture;

const IMAGE_SIZE: [u32; 2] = [1024, 1024];
const IMAGE_LENGTH: usize = (IMAGE_SIZE[0] * IMAGE_SIZE[1]) as usize;

const WORKGROUP_SIZE: [u32; 3] = [32, 32, 1];
const NUM_DISPATCH: [u32; 3] = [
    IMAGE_SIZE[0] / WORKGROUP_SIZE[0],
    IMAGE_SIZE[1] / WORKGROUP_SIZE[1],
    1,
];

fn main() {
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

    /*
    let cep = unsafe {
        compute_shader.module().compute_entry_point(
            std::ffi::CStr::from_ptr("initialize".as_ptr() as *const i8),
            vulkano::descriptor::pipeline_layout::EmptyPipelineDesc,
        )
    };
    */

    //let local_image_buffer = vulkano::buffer::DeviceLocalBuffer::<compute_shader::Image>::new(
    //    device.clone(),
    //    vulkano::buffer::BufferUsage::all(),
    //    device.clone().physical_device().queue_families(),
    //)
    //.unwrap();

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

    //let shared_image_buffer = vulkano::buffer::CpuAccessibleBuffer::from_data(
    //    device.clone(),
    //    vulkano::buffer::BufferUsage::all(),
    //    compute_shader::Image::default(),
    //)
    //.unwrap();

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

    let shared_constants_buffer = vulkano::buffer::CpuAccessibleBuffer::from_data(
        device.clone(),
        vulkano::buffer::BufferUsage::all(),
        compute_shader::Constants {
            image_size: IMAGE_SIZE,
        },
    )
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
        .add_buffer(shared_constants_buffer.clone())
        .expect("failed to add shared constants buffer")
        .add_buffer(local_image_buffer.clone())
        .expect("failed to add local image buffer")
        .add_buffer(local_dist_image_buffer.clone())
        .expect("failed to add local image buffer")
        .build()
        .expect("failed to create descriptor set"),
    );

    let image = vulkano::image::StorageImage::new(
        device.clone(),
        vulkano::image::Dimensions::Dim2d {
            width: IMAGE_SIZE[0],
            height: IMAGE_SIZE[1],
        },
        vulkano::format::R8G8B8A8Unorm,
        Some(queue.family()),
    )
    .unwrap();

    let command_buffer =
        vulkano::command_buffer::AutoCommandBufferBuilder::new(device.clone(), queue.family())
            .expect("failed to create command buffer builder")
            .dispatch(
                NUM_DISPATCH,
                compute_pipeline.clone(),
                descriptor_set.clone(),
                (),
            )
            .expect("failed to dispatch")
            .build()
            .expect("failed to build command buffer")
            .execute(queue.clone())
            .expect("failed to execute command buffer")
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    let command_buffer =
        vulkano::command_buffer::AutoCommandBufferBuilder::new(device.clone(), queue.family())
            .expect("failed to create command buffer builder")
            .copy_buffer(local_image_buffer.clone(), shared_image_buffer.clone())
            .expect("failed to copy buffer")
            .copy_buffer(
                local_dist_image_buffer.clone(),
                shared_dist_image_buffer.clone(),
            )
            .expect("failed to copy buffer")
            //.copy_buffer_to_image(local_image_buffer.clone(), image.clone())
            //.unwrap()
            .build()
            .expect("failed to build command buffer")
            .execute(queue.clone())
            .expect("failed to execute command buffer")
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

    /*
    let finished = command_buffer
        .execute(queue.clone())
        .expect("failed to execute command buffer");

    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
        */
    /*

    let content = shared_image_buffer.read().unwrap();
    //for pixel in content.pixels.iter().take(16) {
    //    println!("{:?}", pixel);
    //}
    let converted = content
        .pixels
        .iter()
        .flat_map(|pixel| {
            vec![
                (pixel.r().clone().min(1.0).max(0.0) * std::u8::MAX as f32) as u8,
                (pixel.g().clone().min(1.0).max(0.0) * std::u8::MAX as f32) as u8,
                (pixel.b().clone().min(1.0).max(0.0) * std::u8::MAX as f32) as u8,
                (pixel.a().clone().min(1.0).max(0.0) * std::u8::MAX as f32) as u8,
            ]
            .into_iter()
        })
        .collect::<Vec<u8>>();
        */

    //let image =
    //image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(IMAGE_SIZE[0], IMAGE_SIZE[1], converted)
    //.expect("failed to create image");

    let content = shared_image_buffer.read().expect("failed to read content");
    let content = content
        .iter()
        .map(|v| (v.min(1.0).max(0.0) * std::u8::MAX as f32) as u8)
        .collect::<Vec<u8>>();
    let image =
        image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(IMAGE_SIZE[0], IMAGE_SIZE[1], content)
            .expect("failed to create image");

    image.save("image.png").expect("failed to save image");

    let content = shared_dist_image_buffer
        .read()
        .expect("failed to read content");
    let content = content
        .iter()
        .map(|v| (v.min(1.0).max(0.0) * std::u8::MAX as f32) as u8)
        .collect::<Vec<u8>>();
    let image =
        image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(IMAGE_SIZE[0], IMAGE_SIZE[1], content)
            .expect("failed to create image");

    image.save("dist_image.png").expect("failed to save image");
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

    pub struct Constants {
        pub image_size: [u32; 2],
    }

    vulkano_shaders::shader! {
        ty: "compute",
        include: [ "src/glsl" ],
        src: "
            #version 450

            layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer ConstantsBuffer {
                uvec2 image_size;
            } constants;

            layout(set = 0, binding = 1) buffer ImageBuffer {
                vec4 pixels[];
            } image;

            layout(set = 0, binding = 2) buffer DistImageBuffer {
                vec4 pixels[];
            } dist_image;

            struct Ray {
                vec3 origin;
                vec3 direction;
            };

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
            };

            const vec3  LDR = vec3(0.577);
            const float EPS = 0.0001;
            const int   MAX_REF = 128;
            const float dump_ratio = 0.000000;
            const float rand_ratio = 1.0;
            const int EXEC_COUNT = 128;

            // num perticles / m
            const float particle_density = 5.0;
            const float particle_try_dist = 10.0;
            const float particle_probability = 0.0000000001;
            const float particle_reflection_ratio = 0.97;

            Sphere sphere[3];
            Plane plane[6];

            float rand(vec2 co) {
                return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
            }

            vec3 random_in_unit_sphere(float seed) {
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

                    x = 2.0 * rand(seed1) - 1.0;
                    y = 2.0 * rand(seed2) - 1.0;
                    z = 2.0 * rand(seed3) - 1.0;

                    squared_length = x * x + y * y + z * z;
                } while (squared_length >= 1.0);
                return vec3(x, y, z);
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
            }

            void intersect_particle(Ray ray, inout Intersection intersection) {
                int hit = 0;
                for(float dist = 0.0; dist < particle_try_dist; dist += 1.0/particle_density) {
                    vec2 seed = ray.direction.xy + ray.direction.yz + intersection.hitPoint.xy + intersection.hitPoint.yz - vec2(dist, intersection.normal.z) + gl_GlobalInvocationID.xy;
                    float seed2 = dist + ray.direction.x + ray.direction.y + intersection.hitPoint.x + intersection.normal.x + gl_GlobalInvocationID.x + gl_GlobalInvocationID.y;
                    if (dist < intersection.distance && rand(seed) < particle_probability) {
                        intersection.hitPoint = ray.origin + ray.direction * dist;
                        intersection.normal = 2.0 * random_in_unit_sphere(seed2) - 1.0;

                        intersection.distance = dist;
                        intersection.hit++;
                        intersection.rayDir = ray.direction;
                        intersection.intensity_dump_ratio = max(1.0 - intersection.distance * dump_ratio, 0.0);
                        intersection.intensity_dump_ratio *= particle_reflection_ratio;
                        break;
                    }
                }
            }

            void intersect_sphere(Ray ray, Sphere sphere, inout Intersection intersection){
                vec3  a = ray.origin - sphere.position;
                float b = dot(a, ray.direction);
                float c = dot(a, a) - (sphere.radius * sphere.radius);
                float d = b * b - c;
                float t = -b - sqrt(d);
                if(d > 0.0 && t > EPS && t < intersection.distance){
                    intersection.hitPoint = ray.origin + ray.direction * t;
                    intersection.normal = normalize(intersection.hitPoint - sphere.position);
                    d = clamp(dot(LDR, intersection.normal), 0.1, 1.0);
                    intersection.color = sphere.color * d;
                    intersection.distance = t;
                    intersection.hit++;
                    intersection.rayDir = ray.direction;
                
                    intersection.intensity *= max(1.0 - intersection.distance * dump_ratio, 0.0);
                
                    if(sphere.emission == 1) {
                        //intersection.hit--;
                        intersection.hit_emission += 1;
                    }
                    else
                    {
                        intersection.intensity_dump_ratio = sphere.reflection_ratio;
                    }
                }
            }

            void intersect_plane(Ray ray, Plane plane, inout Intersection intersection){
                float d = -dot(plane.position, plane.normal);
                float v = dot(ray.direction, plane.normal);
                float t = -(dot(ray.origin, plane.normal) + d) / v;
                if(t > EPS && t < intersection.distance){
                    intersection.hitPoint = ray.origin + ray.direction * t;
                    intersection.normal = plane.normal;
                    float d = clamp(dot(LDR, intersection.normal), 0.1, 1.0);
                    float m = mod(intersection.hitPoint.x, 2.0);
                    float n = mod(intersection.hitPoint.z, 2.0);
                    float f = 1.0 - min(abs(intersection.hitPoint.z), 25.0) * 0.04;
                    intersection.color = plane.color * d * f;
                    intersection.distance = t;
                    intersection.hit++;
                    intersection.rayDir = ray.direction;
                    intersection.intensity_dump_ratio = max(1.0 - intersection.distance * dump_ratio, 0.0);
                    intersection.intensity_dump_ratio *= plane.reflection_ratio;
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
                intersect_particle(ray, intersection);
            }

            void compute() {
                float intensity = 0.0;
                float distance = 0.0;
                int hit_count = 0;
                for (int count = 0; count < EXEC_COUNT; count++)
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
                    sphere[2].reflection_ratio = 0.0;
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
                        for(int j = 1; j < MAX_REF; j++){
                            q.origin = its.hitPoint + its.normal * EPS;
                        
                            q.direction = reflect(its.rayDir, its.normal);
                            q.direction = normalize(q.direction);
                            q.direction += rand_ratio * random_in_unit_sphere(length(its.hitPoint) + its.hitPoint.y + float(count) + its.intensity + float(j));
                            //q.direction = normalize(q.direction);

                            distance += its.distance;
                        
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
            
                intensity = intensity / float(EXEC_COUNT);
                distance = distance / float(hit_count);
            
                uint index =
                    gl_GlobalInvocationID.y * constants.image_size.x +
                    gl_GlobalInvocationID.x;                
            
                //image.pixels[index] = vec4(destColor, 1.0);
                image.pixels[index] = vec4(vec3(intensity), 1.0);
                dist_image.pixels[index] = vec4(vec3(distance / 200.0), 1.0);
            }


            void main() {
                compute();
            }
        ",
    }
}
