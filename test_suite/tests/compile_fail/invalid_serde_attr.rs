
#[derive(serde_unstructured::SerdeView)]
struct Data {
    #[serde(what)]
    field: u8,
}

fn main() {}