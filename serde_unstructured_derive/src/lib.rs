#![warn(clippy::all)]

use std::{iter, ops::Deref};

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use proc_macro_error::*;
use quote::{format_ident, quote, ToTokens};
use serde_derive_internals as serde;
use syn::{
    parse_macro_input, punctuated::Punctuated, Attribute, Data, DeriveInput, Error, Field,
    GenericParam, Ident, ImplGenerics, LitStr, Path, PathSegment, Token, TypeGenerics, TypeParam,
    WherePredicate,
};

type Result<T, E = Error> = core::result::Result<T, E>;

/// Used for error messages.
const MACRO_NAME: &str = "SerdeView";
/// Used for attribute parsing.
const SERDE_ATTR: &str = "serde";
/// Used for attribute parsing.
const CUSTOM_ATTR: &str = "serde_view";
/// Used when generating code.
const GENERATED_TYPE_NAME_PREFIX: &str = "__SerdeView__";
/// Used to refer to items from the main crate.
const PACKAGE_NAME: &str = "serde_unstructured";

////////////////////////////////////////////////////////////////////////////////
// Errors for `serde_derive_internals`
////////////////////////////////////////////////////////////////////////////////

struct CheckSerdeErrors(Option<serde::Ctxt>);
impl CheckSerdeErrors {
    pub fn new(ctx: serde::Ctxt) -> Self {
        Self(Some(ctx))
    }
    /// Consume the wrapped `serde_derive_internals` context and return `true`
    /// if any errors was emitted.
    fn finish(&mut self) -> bool {
        let errors = if let Some(errors) = self.0.take().and_then(|ctx| ctx.check().err()) {
            errors
        } else {
            return false;
        };
        let fixed_errors = errors.into_iter().map(|error| {
            Error::new(
                error.span(),
                format_args!("`{}` failed at `serde` attribute: {}", MACRO_NAME, error),
            )
        });

        for error in fixed_errors {
            emit_error!(
                error.span(), error;
                help = format_args!("skip a field using a `#[{custom}(skip)]` attribute or use \
                    an empty `#[{custom}()]` attribute to ignore any `#[{serde}]` attributes.",
                    custom = CUSTOM_ATTR,
                    serde = SERDE_ATTR,
                );
            );
        }

        true
    }
    pub fn assume_errors(mut self) -> ! {
        if self.finish() {
            abort_if_dirty();
        }
        panic!("assumed that a serde attribute parsing error occurred");
    }
}
impl Deref for CheckSerdeErrors {
    type Target = serde::Ctxt;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}
impl Drop for CheckSerdeErrors {
    fn drop(&mut self) {
        self.finish();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Derive Macro
////////////////////////////////////////////////////////////////////////////////

#[proc_macro_error]
#[proc_macro_derive(SerdeView, attributes(serde, serde_view))]
pub fn serde_view_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    my_derive(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn my_derive(mut input: DeriveInput) -> Result<TokenStream2> {
    match &input.data {
        Data::Struct(_) => {}
        Data::Enum(_) | Data::Union(_) => {
            return Err(Error::new(
                Span::call_site(),
                format_args!(
                    "The `{}` derive macro can only be applied to structs.",
                    MACRO_NAME
                ),
            ))
        }
    }

    let had_custom_attr_on_type = rewrite_custom_attrs(&mut input.attrs);

    let had_custom_attr_on_field: Vec<bool> = if let Data::Struct(data) = &mut input.data {
        data.fields
            .iter_mut()
            .map(|field| rewrite_custom_attrs(&mut field.attrs))
            .collect()
    } else {
        unreachable!()
    };

    let serde_ctx = CheckSerdeErrors::new(serde::Ctxt::new());
    let serde_info = match serde::ast::Container::from_ast(
        &serde_ctx,
        &input,
        // This macro implements Projection into unstructured data, which is
        // really about deserializing data, so it makes the most sense to choose
        // options related to that:
        serde::Derive::Deserialize,
    ) {
        Some(v) => v,
        None => serde_ctx.assume_errors(),
    };

    let data = if let Data::Struct(data) = &input.data {
        data
    } else {
        unreachable!();
    };

    let my_crate = match serde_info.attrs.custom_serde_path() {
        Some(path) if had_custom_attr_on_type => quote!(#path),
        _ => get_my_crate(),
    };
    let type_visibility = &input.vis;

    let type_name = &input.ident;
    let view_ident = format_ident!("{}Type_{}", GENERATED_TYPE_NAME_PREFIX, type_name);
    let prev_ident = Ident::new("__Prev", Span::call_site());

    let generics = input.generics.clone();
    let (_, ty_generics, _) = generics.split_for_impl();
    let mut view_generics = input.generics.clone();
    let (impl_generics, view_ty_generics, where_clause) = {
        let prev_param = GenericParam::from(TypeParam::from(prev_ident.clone()));
        let after_lifetime = view_generics
            .params
            .iter()
            .position(|item| !matches!(item, GenericParam::Lifetime(_)));
        match after_lifetime {
            Some(index) => view_generics.params.insert(index, prev_param),
            None => view_generics.params.push(prev_param),
        }
        view_generics.split_for_impl()
    };
    let where_predicates = where_clause.map(|clause| &clause.predicates);

    let field_specific: TokenStream2 = data
        .fields
        .iter()
        .enumerate()
        .zip(serde_info.data.all_fields())
        .zip(had_custom_attr_on_field)
        .map(|(((index, field), serde_field), had_custom_attr)| {
            generate_project_method_for_field(
                index,
                field,
                serde_field,
                had_custom_attr,
                StructDeriveInfo {
                    my_crate: &my_crate,
                    prev_ident: &prev_ident,
                    where_predicates,
                    view_ident: &view_ident,
                    view_ty_generics: &view_ty_generics,
                    impl_generics: &impl_generics,
                },
            )
        })
        .collect();

    Ok(quote!(
        #[allow(non_camel_case_types)]
        const _: () = {
            #type_visibility struct #view_ident #impl_generics #where_clause {
                #[doc(hidden)]
                __private: (
                    #my_crate::ZeroSizedProjectionPath<#prev_ident>,
                    ::core::marker::PhantomData<#type_name #ty_generics>,
                ),
            }
            impl #impl_generics #my_crate::SerdeView<#prev_ident> for #type_name #ty_generics
            where
                #prev_ident: #my_crate::ZeroSizedPath,
                #where_predicates
            {
                type View = #view_ident #view_ty_generics;

                #[inline]
                fn get_view<'__view>() -> &'__view Self::View {
                    &#view_ident {
                        __private: (
                            #my_crate::ZeroSizedProjectionPath::NEW,
                            ::core::marker::PhantomData,
                        ),
                    }
                }
            }

            #field_specific
        };
    ))
}

#[derive(Clone, Copy)]
struct StructDeriveInfo<'a> {
    /// Path to the main crate.
    my_crate: &'a TokenStream2,
    /// Type parameter for the previous `PathTracker` that should be continued in
    /// projection methods.
    prev_ident: &'a Ident,
    /// Name of the generated view type.
    view_ident: &'a Ident,
    /// Generic parameters for the generated view type.
    view_ty_generics: &'a TypeGenerics<'a>,
    /// Define any generic type parameters that will be used by
    /// `view_ty_generics`.
    impl_generics: &'a ImplGenerics<'a>,
    /// Extra predicates to add to any where clause.
    where_predicates: Option<&'a Punctuated<WherePredicate, Token![,]>>,
}

fn generate_project_method_for_field(
    index: usize,
    field: &Field,
    serde_field: &serde::ast::Field<'_>,
    had_custom_attr: bool,
    type_info: StructDeriveInfo<'_>,
) -> TokenStream2 {
    let StructDeriveInfo {
        my_crate,
        prev_ident,
        view_ty_generics,
        impl_generics,
        where_predicates,
        view_ident,
    } = type_info;
    let Field { ident, vis, ty, .. } = field;
    let serde_field = &serde_field.attrs;

    let field_span = || {
        ident
            .as_ref()
            .map(|ident| ident.span())
            .or_else(|| {
                ty.into_token_stream()
                    .into_iter()
                    .next()
                    .map(|tree| tree.span())
            })
            .unwrap_or_else(Span::call_site)
    };

    if serde_field.skip_deserializing() {
        return quote!();
    }
    if serde_field.deserialize_with().is_some() {
        if had_custom_attr {
            emit_error!(
                field_span(),
                format_args!(
                    "{} can't project into fields that are using \
                    custom deserialization, remove any `#[{custom}(with = \"module\")]` \
                    or #[{custom}(deserialize_with = \"path\")] attribute.",
                    MACRO_NAME,
                    custom = CUSTOM_ATTR
                )
            );
        } else {
            emit_error!(
                field_span(),
                format_args!(
                    "{} can't project into fields that are using \
                    custom deserialization, skip the field manually \
                    using a `#[{custom}(skip)]` attribute or use an \
                    empty `#[{custom}()]` attribute to ignore any \
                    `#[{serde}]` attributes.",
                    MACRO_NAME,
                    custom = CUSTOM_ATTR,
                    serde = SERDE_ATTR
                )
            );
        }
        return quote!();
    }

    let method_name = ident
        .as_ref()
        .map(|ident| format_ident!("{}", ident))
        .unwrap_or_else(|| format_ident!("get_{}", index));

    let (projection_def, projection_ty) = if serde_field.flatten() || serde_field.transparent() {
        // When flattening we just cast the unstructured data to a
        // different type, we don't project into the data.
        (quote!(), quote!(#prev_ident))
    } else {
        let type_name = ident
            .as_ref()
            .map(|ident| format_ident!("{}Field_{}", GENERATED_TYPE_NAME_PREFIX, ident))
            .unwrap_or_else(|| format_ident!("{}Field_{}", GENERATED_TYPE_NAME_PREFIX, index));

        let data_ident = Ident::new("_data", Span::call_site());

        let serde_name = serde_field.name().deserialize_name();

        let alias_projection_path: TokenStream2 = {
            let mut aliases = serde_field.aliases().clone();
            aliases.retain(|item| *item != serde_name);

            if aliases.is_empty() {
                quote!()
            } else {
                // Include a check for the default name first, since that path
                // should be preferred.
                iter::once(serde_name).chain(aliases.iter().map(String::as_str)).map(|alias| {
                    match serde_name_as_projection_path(alias, my_crate) {
                        ParsedSerdeName::Field { name, projection_path } => {
                            quote!(
                                if ::core::option::Option::is_some(
                                    &#my_crate::DynUnstructuredData::dyn_object_project(#data_ident, #name)
                                ) {
                                    return #projection_path;
                                }
                            )
                        },
                        ParsedSerdeName::Index { index, projection_path } => {
                            quote!(
                                if ::core::option::Option::is_some(
                                    &#my_crate::DynUnstructuredData::dyn_array_project(#data_ident, #index)
                                ) {
                                    return #projection_path;
                                }
                            )
                        },
                    }
                }).collect()
            }
        };

        // If no alias was found (or was defined) then use this path:
        let projection_path =
            serde_name_as_projection_path(serde_name, my_crate).into_projection_path();

        (
            quote!(
                #vis struct #type_name;
                impl #my_crate::GetZeroSizedProjectionPathSegment for #type_name {
                    #[inline]
                    fn get_path(#data_ident: &dyn #my_crate::DynUnstructuredData) -> #my_crate::ProjectionPathSegment<'static> {
                        #alias_projection_path
                        #projection_path
                    }
                }
            ),
            quote!((#prev_ident, #my_crate::ZeroSizedProjectionPath<#type_name>)),
        )
    };

    quote!(
        #projection_def

        impl #impl_generics #view_ident #view_ty_generics
        where
            #prev_ident: #my_crate::ZeroSizedPath,
            #where_predicates
        {
            #[inline]
            #vis fn #method_name(
                &self,
            ) -> #my_crate::ProjectionTarget<#ty, #projection_ty> {
                ::core::default::Default::default()
            }
        }
    )
}

////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

enum ParsedSerdeName {
    Field {
        name: LitStr,
        projection_path: TokenStream2,
    },
    Index {
        index: usize,
        projection_path: TokenStream2,
    },
}
impl ParsedSerdeName {
    fn into_projection_path(self) -> TokenStream2 {
        match self {
            ParsedSerdeName::Field {
                projection_path, ..
            } => projection_path,
            ParsedSerdeName::Index {
                projection_path, ..
            } => projection_path,
        }
    }
}

fn serde_name_as_projection_path(serde_name: &str, my_crate: &TokenStream2) -> ParsedSerdeName {
    // Serde just converts indexes to strings for tuples:
    // https://github.com/serde-rs/serde/blob/master/serde_derive/src/internals/attr.rs#L1168
    if serde_name.contains(|c: char| !c.is_ascii_digit()) {
        let field_name_as_literal = LitStr::new(serde_name, Span::call_site());
        let projection_path = quote!(#my_crate::ProjectionPathSegment::Field(#my_crate::MaybeStatic::Static(#field_name_as_literal)));
        ParsedSerdeName::Field {
            name: field_name_as_literal,
            projection_path,
        }
    } else {
        let index: usize = serde_name.parse().unwrap_or_else(|e| {
            emit_call_site_error!(
                "failed to interpret serialization name \"{}\" as index: {}",
                serde_name,
                e
            );
            0
        });
        let projection_path = quote!(#my_crate::ProjectionPathSegment::Index(#index));
        ParsedSerdeName::Index {
            index,
            projection_path,
        }
    }
}

/// Rewrite custom attributes to `serde` attributes.
fn rewrite_custom_attrs(attrs: &mut Vec<Attribute>) -> bool {
    if !attrs.iter().any(|attr| attr.path().is_ident(CUSTOM_ATTR)) {
        // No custom attributes:
        return false;
    }

    // Remove all original `serde` attributes:
    attrs.retain(|attr| !attr.path().is_ident(SERDE_ATTR));

    // Rename `serde_view` attributes to `serde`:
    for attr in attrs {
        if attr.path().is_ident(CUSTOM_ATTR) {
            let span = attr.path().segments.first().unwrap().ident.span();
            let path = match &mut attr.meta {
                syn::Meta::Path(path) => path,
                syn::Meta::List(meta_list) => &mut meta_list.path,
                syn::Meta::NameValue(meta_name_value) => &mut meta_name_value.path,
            };
            *path = Path::from(PathSegment::from(Ident::new(SERDE_ATTR, span)));
        }
    }

    true
}

/// Get an identifier that resolves to the current crate. Can be used where `$crate`
/// would be used in a declarative macro.
fn get_my_crate() -> TokenStream2 {
    let name = proc_macro_crate::crate_name(PACKAGE_NAME).unwrap_or_else(|e| {
        abort_call_site!(
                "The `{}` derive macro expected `{}` to be present in `Cargo.toml`: {}",
                MACRO_NAME, PACKAGE_NAME, e;
                help = "if the derive macro was imported from another crate then consider \
                adding a #[{}(crate = \"path\")] attribute to the type", CUSTOM_ATTR;
        );
    });
    match name {
        // When macro is expanded by `rust-analyzer` then the `Itself` arm will
        // be used for integration tests, which is incorrect. (We don't use
        // integration tests currently so this is fine.)
        proc_macro_crate::FoundCrate::Itself => quote!(crate),
        proc_macro_crate::FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote! { ::#ident }
        }
    }
}
