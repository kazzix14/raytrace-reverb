use obj::*;
use std::fs::File;
use std::io::BufReader;

pub fn obj() -> Obj {
    let input =
        BufReader::new(File::open("resources/models/cube.obj").expect("failed to open obj"));
    let obj = load_obj(input).expect("failed to load obj");
    obj
}

pub fn vertices() -> Vec<f32> {
    let obj = obj();
    obj.vertices
        .iter()
        .flat_map(
            |Vertex {
                 position,
                 normal: _,
             }| { position.iter().cloned() },
        )
        .collect()
}
