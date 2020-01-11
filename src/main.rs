mod shader;

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
    6,
];

const WORKGROUP_SIZE_POSTPROCESS: [u32; 3] = [1024, 1, 1];
const NUM_DISPATCH_POSTPROCESS: [u32; 3] = [
    (IMAGE_LENGTH as u32 * 6) / WORKGROUP_SIZE_POSTPROCESS[0] + 1,
    1,
    1,
];

const NUM_RANDOMS: u32 = 317;

fn main() {
    let radius: f32 = 0.3;
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

    let raytrace_shader =
        shader::raytrace::Shader::load(device.clone()).expect("failed to load compute shader");

    let local_image_buffer = unsafe {
        vulkano::buffer::DeviceLocalBuffer::raw(
            device.clone(),
            IMAGE_LENGTH * 4 * 4 * 6,
            vulkano::buffer::BufferUsage::all(),
            device.clone().physical_device().queue_families(),
        )
        .unwrap()
    };

    let local_dist_image_buffer = unsafe {
        vulkano::buffer::DeviceLocalBuffer::raw(
            device.clone(),
            IMAGE_LENGTH * 4 * 4 * 6,
            vulkano::buffer::BufferUsage::all(),
            device.clone().physical_device().queue_families(),
        )
        .unwrap()
    };

    let shared_image_buffer = vulkano::buffer::CpuAccessibleBuffer::from_iter(
        device.clone(),
        vulkano::buffer::BufferUsage::all(),
        (0..IMAGE_LENGTH * 4 * 6).map(|_| 0.0 as f32),
    )
    .unwrap();

    let shared_dist_image_buffer = vulkano::buffer::CpuAccessibleBuffer::from_iter(
        device.clone(),
        vulkano::buffer::BufferUsage::all(),
        (0..IMAGE_LENGTH * 4 * 6).map(|_| 0.0 as f32),
    )
    .unwrap();

    let (local_randoms_buffer, local_randoms_buffer_submit_command) =
        vulkano::buffer::ImmutableBuffer::from_data(
            shader::raytrace::SizedRandoms {
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

    let (local_constants_buffer, local_constants_buffer_submit_command) =
        vulkano::buffer::ImmutableBuffer::from_data(
            shader::raytrace::ty::Constants {
                image_size: IMAGE_SIZE,
                EPS: 0.000001,
                reflection_count_limit: 128,
                rays_per_pixel: 32,
                num_randoms: NUM_RANDOMS,
                ray_length: 1e10,
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
            &raytrace_shader.main_entry_point(),
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
        .build()
        .expect("failed to build command buffer")
        .execute(queue.clone())
        .expect("failed to execute command buffer")
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    vulkano::command_buffer::AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .expect("failed to create command buffer builder")
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

    println!("raytrace done!!!");

    let intensities = shared_image_buffer
        .read()
        .expect("failed to read content")
        .iter()
        .copied()
        .collect::<Vec<f32>>();

    let distancies = shared_dist_image_buffer
        .read()
        .expect("failed to read content")
        .iter()
        .copied()
        .collect::<Vec<f32>>();

    for offset in 0..6 {
        let mut intensities = intensities.clone();
        normalize(&mut intensities);
        set_alpha_to_1(&mut intensities);
        let intensities = intensities
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(i, v)| {
                if ((offset * 4) <= (i % 24)) && ((i % 24) < (offset * 4 + 4)) {
                    Some(v)
                } else {
                    None
                }
            })
            .collect::<Vec<f32>>();

        let mut distancies = distancies.clone();
        normalize(&mut distancies);
        set_alpha_to_1(&mut distancies);
        let distancies = distancies
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(i, v)| {
                if ((offset * 4) <= (i % 24)) && ((i % 24) < (offset * 4 + 4)) {
                    Some(v)
                } else {
                    None
                }
            })
            .collect::<Vec<f32>>();

        save_as_image_with_out_normalize(
            &intensities,
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            format!("intensities-{}.png", offset.to_string()),
        );

        save_as_image_with_out_normalize(
            &distancies,
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            format!("distancies-{}.png", offset.to_string()),
        );
    }

    let (local_constants_buffer, local_constants_buffer_submit_command) =
        vulkano::buffer::ImmutableBuffer::from_data(
            shader::postprocess::ty::Constants {
                radius: radius,
                image_length: IMAGE_LENGTH as u32,
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

    let postprocess_shader =
        shader::postprocess::Shader::load(device.clone()).expect("failed to load compute shader");

    let compute_pipeline = Arc::new(
        vulkano::pipeline::ComputePipeline::new(
            device.clone(),
            &postprocess_shader.main_entry_point(),
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
        .add_buffer(shared_image_buffer.clone())
        .expect("failed to add local image buffer")
        .add_buffer(shared_dist_image_buffer.clone())
        .expect("failed to add local image buffer")
        .build()
        .expect("failed to create descriptor set"),
    );

    vulkano::command_buffer::AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .expect("failed to create command buffer builder")
        .dispatch(
            NUM_DISPATCH_POSTPROCESS,
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

    println!("postprocess done!!!");

    let mut intensities = shared_image_buffer
        .read()
        .expect("failed to read content")
        .iter()
        .copied()
        .collect::<Vec<f32>>();

    for offset in 0..6 {
        let mut intensities = intensities.clone();
        normalize(&mut intensities);
        set_alpha_to_1(&mut intensities);
        let intensities = intensities
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(i, v)| {
                // 24 = num_images(fblrtb = 6) * num_channels(rgba = 4)
                if ((offset * 4) <= (i % 24)) && ((i % 24) < (offset * 4 + 4)) {
                    Some(v)
                } else {
                    None
                }
            })
            .collect::<Vec<f32>>();

        save_as_image_with_out_normalize(
            &intensities,
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            format!("intensities-postprocessed-{}.png", offset.to_string()),
        );
    }

    //scale_intensities(&mut intensities, radius, IMAGE_LENGTH, &distancies);
    let mut intensities_left = get_half_of_cube(&intensities, false);
    let mut intensities_right = get_half_of_cube(&intensities, true);
    let mut distancies_left = get_half_of_cube(&distancies, false);
    let mut distancies_right = get_half_of_cube(&distancies, true);
    let mut impulse_response_left = build_intensity_vec(
        &distancies_left,
        &intensities_left,
        SPEED_OF_SOUND,
        SAMPLE_RATE,
    );

    let mut impulse_response_right = build_intensity_vec(
        &distancies_right,
        &intensities_right,
        SPEED_OF_SOUND,
        SAMPLE_RATE,
    );

    //plot(&impulse_response, "ir.pdf");

    const FILTER_WIDTH: usize = 100;

    {
        let mut impulse_response_filtered = impulse_response_left.clone();
        filter(&mut impulse_response_filtered, FILTER_WIDTH);
        plot(&impulse_response_filtered, "ir-filtered-left.pdf");

        let mut impulse_response_filtered = impulse_response_right.clone();
        filter(&mut impulse_response_filtered, FILTER_WIDTH);
        plot(&impulse_response_filtered, "ir-filtered-right.pdf");
    }

    //ceil_at(&mut impulse_response_left, 1.0);
    //ceil_at(&mut impulse_response_right, 1.0);
    let max_left = max_of(&impulse_response_left);
    let max_right = max_of(&impulse_response_right);
    let max = max_left.max(max_right);

    impulse_response_left.iter_mut().for_each(|v| *v /= max);
    impulse_response_right.iter_mut().for_each(|v| *v /= max);

    plot(&impulse_response_left, "ir-normalized-left.pdf");
    plot(&impulse_response_right, "ir-normalized-right.pdf");

    let len_left = impulse_response_left.len();
    let len_right = impulse_response_right.len();
    let len = len_left.max(len_right);

    let impulse_response = {
        (0..len)
            .map(|index| {
                let sample_left = impulse_response_left.get(index).unwrap_or(&0.0);
                let sample_right = impulse_response_right.get(index).unwrap_or(&0.0);
                [*sample_left, *sample_right]
            })
            .collect::<Vec<[f32; 2]>>()
    };

    let mut signal = white_noise(len);

    amplitude(&mut signal, &impulse_response);

    write_wav(signal, "ir.wav".to_string());
}

fn get_half_of_cube(source: &Vec<f32>, is_right: bool) -> Vec<f32> {
    // f b l r t b
    const NUM_CHANNELS: usize = 4;
    const IMAGE_HALF_WIDTH: usize = IMAGE_SIZE[0] as usize;

    source
        .iter()
        .skip(IMAGE_LENGTH * 2 * NUM_CHANNELS)
        .take(IMAGE_LENGTH * NUM_CHANNELS) // l
        .chain(
            source
                .iter()
                .take(IMAGE_LENGTH * 2 * NUM_CHANNELS) // f b
                .chain(
                    source
                        .iter()
                        .skip(IMAGE_LENGTH * 4 * NUM_CHANNELS)
                        .take(IMAGE_LENGTH * 2 * NUM_CHANNELS), // t b
                )
                .enumerate()
                .filter_map(|(i, v)| {
                    // gather left/right half
                    match ((i / (IMAGE_HALF_WIDTH * NUM_CHANNELS)) % 2, is_right) {
                        (0, false) => Some(v),
                        (1, false) => None,
                        (0, true) => None,
                        (1, true) => Some(v),
                        _ => unreachable!(),
                    }
                }),
        )
        .cloned()
        .collect()
}

fn scale_intensities(
    intensities: &mut Vec<f32>,
    audio_source_radius: f32,
    image_length: usize,
    distancies: &Vec<f32>,
) {
    /*
    intensities
        .iter_mut()
        .for_each(|v| *v /= 4.0 * std::f32::consts::PI * audio_source_radius.powi(2));

    intensities
        .iter_mut()
        .for_each(|v| *v /= image_length as f32);

    decay_intensities_by_distancies(intensities, &distancies);
    */
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

fn save_as_image_with_out_normalize(
    source: &Vec<f32>,
    image_width: u32,
    image_height: u32,
    path: String,
) {
    let source_u8 = vec_f32_to_vec_u8(&source);

    let image = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
        image_width as u32,
        image_height as u32,
        source_u8,
    )
    .expect("failed to create image");

    image.save(path).expect("failed to save image");
}

fn set_alpha_to_1(target: &mut Vec<f32>) {
    assert_eq!(target.len() % 4, 0);
    target.iter_mut().skip(3).step_by(4).for_each(|v| *v = 1.0)
}

fn decay_intensities_by_distancies(intensities: &mut Vec<f32>, distancies: &Vec<f32>) {
    assert_eq!(intensities.len(), distancies.len());
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

fn amplitude(lhs: &mut Vec<[f32; 2]>, rhs: &Vec<[f32; 2]>) {
    assert_eq!(lhs.len(), rhs.len());
    for index in 0..lhs.len() {
        lhs[index][0] *= rhs[index][0];
        lhs[index][1] *= rhs[index][1];
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
