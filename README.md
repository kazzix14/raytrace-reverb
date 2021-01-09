# raytrace-reverb

Create IRs of any place you want.

This software is for creating IRs from 3d models using raytracing.

# How to use

This software uses Vulkan API to interact with GPU. So you need an environment that can run Vulkan API.  
Run

    cargo run --release -- -i path/to/input

Then the program ask you to choose gpu. you answer in index and the program start generating IR.
When the IR generation is done, you can find your IR at `ProjectRootDir/ir.wav`.

## Input 3d model

Only supports obj with mtl.
You should specify properties and polygons witch treated as sound source of the polygons with mtl.
Ns represents roughness of surface, and kd represents reflextion ratio.
Objects that named 'AudioSource' are treated as audio source.  
By the way, I use Blender to create models for this software.

## Image size

Default size of the images that intermediate between raytracing and IR is 512x512.
Depends on the gpu you use, this software could crash due to gpu buffer exhaustion.
If It crashes, change IMAGE_SIZE constant to be smaller value like 64x64, and Try again.
(I should improve the design not to compile everytime We change the settings).
But notice that bigger IMAGE_SIZE will make resulting IR more accurate.

# LISCENSE

This project is licensed under the Mozilla Public License, v. 2.0 - see the [LICENSE](LICENSE) file for details
