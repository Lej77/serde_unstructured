
#[derive(serde_unstructured::SerdeView)]
#[serde_view(crate = "path_that_does_not_exists::where_is_this")]
struct Data {
    field: u8,
}

fn main() {}