
#[derive(serde_unstructured::SerdeView)]
struct Data {
    #[serde(with = "path")]
    field: u8,
}

fn main() {}