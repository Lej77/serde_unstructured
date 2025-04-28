
#[derive(serde_unstructured::SerdeView)]
struct Data {
    #[serde_view(with = "path")]
    field: u8,
}

fn main() {}