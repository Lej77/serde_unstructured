
#[derive(serde_unstructured::SerdeView)]
struct Data {
    #[serde(what)]
    a: u8,
    #[serde(are)]
    b: String,
    #[serde(you())]
    c: usize,
    #[serde(doing = to::this::attribute)]
    d: bool,
}

fn main() {}