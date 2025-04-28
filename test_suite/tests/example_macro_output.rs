struct Bar {
    _value: usize,
}

struct Foo {
    _bar: Bar,
}

#[allow(non_camel_case_types)]
const _: () = {
    pub struct __View<__Prev> {
        path: serde_unstructured::ZeroSizedProjectionPath<__Prev>,
        marker: ::core::marker::PhantomData<Foo>,
    }
    impl<__Prev> serde_unstructured::SerdeView<__Prev> for Foo
    where
        __Prev: serde_unstructured::ZeroSizedPath,
    {
        type View = __View<__Prev>;

        fn get_view<'__view>() -> &'__view Self::View {
            &__View {
                path: serde_unstructured::ZeroSizedProjectionPath::NEW,
                marker: ::core::marker::PhantomData,
            }
        }
    }

    /// Match visibility of field
    pub struct __View__Field_bar;
    impl serde_unstructured::GetZeroSizedProjectionPathSegment for __View__Field_bar {
        fn get_path(
            _data: &dyn serde_unstructured::DynUnstructuredData,
        ) -> serde_unstructured::ProjectionPathSegment<'static> {
            serde_unstructured::ProjectionPathSegment::Field(
                serde_unstructured::MaybeStatic::Static("_bar"),
            )
        }
    }

    impl<__Prev> __View<__Prev>
    where
        __Prev: serde_unstructured::ZeroSizedPath,
    {
        /// Match visibility of field
        fn _bar(
            &self,
        ) -> serde_unstructured::ProjectionTarget<
            Bar,
            (
                __Prev,
                serde_unstructured::ZeroSizedProjectionPath<__View__Field_bar>,
            ),
        > {
            ::core::default::Default::default()
        }
    }
};
