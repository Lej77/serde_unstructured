# serde_unstructured

This crate provides a derive macro similar to serde to only verify that
certain parts of some data follows a certain structure. This allows handling
unstructured data ergonomically by assuming it is similar in structure to a
struct that has derived serde traits.

## Usage

You can easily traverse unstructured data for types that implement the
[`SerdeView`] trait by doing the following:

1. Use the [`view`] function to create a [`SerdeViewTyped`],
   [`SerdeViewTypedMut`] or [`SerdeViewTypedRef`] wrapper around some
   [`UnstructuredData`] such as `serde_json::Value`.
2. Use a [`cast`](SerdeViewTyped::cast) method to declare what type the
   unstructured data is assumed to have.
3. Use a [`project`](SerdeViewTyped::project) method to traverse the fields
   of the data.
   - You can also use the [`as_ref`](SerdeViewTyped::as_ref) and
     [`as_mut`](SerdeViewTyped::as_mut) methods to easily borrow from the
     created wrapper.
   - Note that the project methods allows projecting into multiple different
     fields of a type with the same call. Using this you can for example get
     mutable reference for multiple fields at the same time.

Once you have reached the data for the field you are interested in you can:

- Use a [`deserialize`](SerdeViewTyped::deserialize) method to get access to
  the value of the field.
- Access the [`data`](SerdeViewTyped::data) field of the wrapper to get
  direct access to the unstructured data for the field.

## License

This project is released under either:

- [MIT License](https://github.com/Lej77/cast_trait_object/blob/master/LICENSE-MIT)
- [Apache License (Version 2.0)](https://github.com/Lej77/cast_trait_object/blob/master/LICENSE-APACHE)

at your choosing.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
