use serde::Deserialize;
use serde_json::json;
use serde_unstructured::{self, ProjectionTarget, SerdeView};

#[derive(SerdeView)]
struct NoDeserialization {
    _count: usize,
}

#[derive(SerdeView, Deserialize)]
#[allow(dead_code)]
struct Test {
    #[serde_view(rename = "count")]
    #[serde(rename = "override_this")]
    _count: usize,
    #[serde(alias = "alt_text")]
    text: String,
}

#[cfg_attr(all(), derive(SerdeView))]
// SerdeView ignores serde crate path override:
#[serde(crate = "not_valid::path")]
struct GenericTest<T> {
    _count: T,
}

#[derive(SerdeView, Deserialize)]
struct TestTuple(
    #[serde(alias = "4", alias = "what")]
    String,
);

#[derive(SerdeView)]
struct GenericTestTuple<T>(
    #[cfg_attr(all(), serde_view(skip))]
    #[serde(with = "serde")]
    T,
);

type BoxedError = Box<dyn std::error::Error + Send + Sync + 'static>;
type Result<T = (), E = BoxedError> = ::core::result::Result<T, E>;

#[test]
fn aliases_work() -> Result {
    let data = json!({"text": "Normal names work!"});
    let alt_data = json!({"alt_text": "Aliases work!"});
    let empty = json!({});

    // Can get data via the normal field name:
    {
        let text = serde_unstructured::view(data)
            .cast::<Test>()
            .project(|p| p.text())?
            .deserialize()?;
        assert_eq!(text, "Normal names work!");
    }
    // Can get data via aliases:
    {
        let alt_text = serde_unstructured::view(alt_data)
            .cast::<Test>()
            .project(|p| p.text())?
            .deserialize()?;
        assert_eq!(alt_text, "Aliases work!");
    }

    // Error path uses original field name:
    {
        let error = serde_unstructured::view(&empty)
            .cast::<Test>()
            .project(|p| p.text())
            .err()
            .unwrap();
        assert_eq!(error.to_string(), "Invalid unstructured data at: text")
    }

    Ok(())
}

#[test]
fn it_works() -> Result {
    let data = json!({"count": 3_usize, "text": "It works!"});
    let mut wrapped = serde_unstructured::view(data).cast::<Test>();

    // Check that errors work:
    let error = wrapped
        .as_ref()
        .try_project(|_| ProjectionTarget::index(100))
        .err()
        .unwrap();
    assert_eq!(error.to_string(), "Invalid unstructured data at: [100]");

    // Check that we can mutate parts of it:
    {
        let text = wrapped.as_mut().project(|p| p.text())?;
        *text.data = serde_json::Value::String("It has changed!".into());
    }

    // Check that projections work:
    let (count, text) = wrapped.try_project(|p| (p._count(), p.text()))?;
    let count = count.deserialize()?;
    assert_eq!(count, 3);

    let text = text.deserialize()?;
    assert_eq!(text, "It has changed!");

    Ok(())
}
