use obj::*;
use palette::*;
use std::path::Path;

pub fn load() -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    // ns -> roughness
    // ka ->
    let mut obj = Obj::<SimplePolygon>::load(Path::new("resources/models/cube.obj")).unwrap();
    obj.load_mtls();
    //dbg!(obj.objects.clone());

    let position = obj.position;

    let materials = obj
        .objects
        .iter()
        .flat_map(|object| {
            object
                .groups
                .iter()
                .filter_map(|group| group.material.clone())
                .map(|material| material.into_owned())
        })
        .collect::<Vec<obj::Material>>();

    /*
      kd 0..=1
      ns 0..=1000
    */

    let materials = materials
        .iter()
        .flat_map(|mtl| {
            let kd = mtl.kd.expect("'kd' does not appeared on .mtl");
            let [r, g, b] = kd;
            let rgb = Srgb::new(r, g, b);
            let hsv = Hsv::from(rgb);
            let reflection = hsv.value;

            let ns = mtl.ns.expect("'ns' does not appeared on .mtl");
            let diffusion = 1.0 - (ns / 900.0);

            let emission = match mtl.name.find("AudioSource") {
                Some(_) => 1.2,
                None => 0.0,
            };

            assert_eq!(reflection.min(1.0).max(0.0), reflection);
            assert_eq!(diffusion.min(1.0).max(0.0), diffusion);

            const PADDING: f32 = 0.0;

            vec![reflection, diffusion, emission, PADDING].into_iter()
        })
        .collect::<Vec<f32>>();

    let material_indices = obj
        .objects
        .iter()
        .flat_map(|object| {
            object
                .groups
                .iter()
                .flat_map(|group| vec![group.index as u32; group.polys.len()].into_iter())
        })
        .collect::<Vec<u32>>();

    //unsafe { std::mem::transmute::<u32, f32>(group_indices[index] as u32) },

    let vertices = obj
        .objects
        .iter()
        .flat_map(|object| {
            object.groups.iter().flat_map(|group| {
                group.polys.iter().flat_map(|polygon| {
                    polygon
                        .iter()
                        .map(|vert| {
                            let vert_index = vert.0;
                            const PADDING: f32 = 0.0;

                            // vertices
                            vec![
                                position[vert_index][0],
                                position[vert_index][1],
                                position[vert_index][2],
                                PADDING,
                            ]
                            .into_iter()
                            .clone()
                            //position_and_material_index.clone().into_iter().clone()
                        })
                        .flatten()
                })
            })
        })
        .collect::<Vec<f32>>();

    (vertices, material_indices, materials)
}
