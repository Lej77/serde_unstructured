//! This crate provides a derive macro similar to serde to only verify that
//! certain parts of some data follows a certain structure. This allows handling
//! unstructured data ergonomically by assuming it is similar in structure to a
//! struct that has derived serde traits.
//!
//! # Usage
//!
//! You can easily traverse unstructured data for types that implement the
//! [`SerdeView`] trait by doing the following:
//!
//! 1. Use the [`view`] function to create a [`SerdeViewTyped`],
//!    [`SerdeViewTypedMut`] or [`SerdeViewTypedRef`] wrapper around some
//!    [`UnstructuredData`] such as `serde_json::Value`.
//! 2. Use a [`cast`](SerdeViewTyped::cast) method to declare what type the
//!    unstructured data is assumed to have.
//! 3. Use a [`project`](SerdeViewTyped::project) method to traverse the fields
//!    of the data.
//!    - You can also use the [`as_ref`](SerdeViewTyped::as_ref) and
//!      [`as_mut`](SerdeViewTyped::as_mut) methods to easily borrow from the
//!      created wrapper.
//!    - Note that the project methods allows projecting into multiple different
//!      fields of a type with the same call. Using this you can for example get
//!      mutable reference for multiple fields at the same time.
//!
//! Once you have reached the data for the field you are interested in you can:
//!
//! - Use a [`deserialize`](SerdeViewTyped::deserialize) method to get access to
//!   the value of the field.
//! - Access the [`data`](SerdeViewTyped::data) field of the wrapper to get
//!   direct access to the unstructured data for the field.
//!
//! # License
//!
//! This project is released under either:
//!
//! - [MIT License](https://github.com/Lej77/cast_trait_object/blob/master/LICENSE-MIT)
//! - [Apache License (Version 2.0)](https://github.com/Lej77/cast_trait_object/blob/master/LICENSE-APACHE)
//!
//! at your choosing.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
// Warnings and docs:
#![warn(clippy::all)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(feature = "docs", feature(doc_cfg))]
// TODO: lint for docs and idioms.
// #![warn(missing_debug_implementations, missing_docs, rust_2018_idioms)]
#![doc(test(
    no_crate_inject,
    attr(
        deny(warnings, rust_2018_idioms),
        allow(unused_extern_crates, unused_variables)
    )
))]

/// Activate some code only when a certain config is met. Show the required config in the documentation.
///
/// To ensure `rustfmt` works on the enclosed code be sure to invoke this macro in a functional style.
/// Also if the code contains references to `self` then use the `impl Self { /* code... */ }` invocation.
macro_rules! cfg_with_docs {
    ($feature:meta, { $(impl Self { $($code:tt)* })* }) => {
        $(
            #[cfg($feature)]
            #[cfg_attr(feature = "docs", doc(cfg($feature)))]
            $($code)*
        )*
    };
    ($feature:meta, $({$($code:tt)*}),* $(,)?) => {
        $(
            #[cfg($feature)]
            #[cfg_attr(feature = "docs", doc(cfg($feature)))]
            $($code)*
        )*
    };
}

#[cfg(feature = "alloc")]
extern crate alloc;

use core::{cell::Cell, fmt, iter, marker::PhantomData, ops::Deref};
#[cfg(feature = "std")]
use std::error::Error;

/// A derive macro for the [`SerdeView`] trait.
///
/// # Attributes
///
/// This derive macro accepts `serde` attributes and `serde_view` attributes.
///
/// - These attributes accepts the same content as what `serde`'s derive macro
///   accepts.
/// - If any `serde_view` attribute is present on a field then all `serde`
///   attributes on that field will be ignored.
/// - If any `serde_view` attribute is present on a type then all `serde`
///   attributes on the type itself will be ignored, but `serde` attributes on
///   any fields will still be kept.
/// - `#[serde(crate = "path")` attributes will be ignored while
///   `#[serde_view(crate = "path")` attributes will be followed.
///
/// [`SerdeView`]: trait@SerdeView
pub use ::serde_unstructured_derive::SerdeView;
use serde::de::DeserializeOwned;

pub mod view_path;

mod projection;
use projection::*;

////////////////////////////////////////////////////////////////////////////////
// Path tracker
////////////////////////////////////////////////////////////////////////////////

/// Used by the [`PathTracker::display`] method.
pub struct PathTrackerDisplay<'a, T: ?Sized>(&'a T);
impl<T> fmt::Display for PathTrackerDisplay<'_, T>
where
    T: PathTracker,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PathTracker::format_path(self.0, PathTracker::segment_iter(self.0), f)
    }
}

mod sealed_lifetime {
    //! For more info see:
    //! <https://sabrinajewson.org/blog/the-better-alternative-to-lifetime-gats>
    pub trait Sealed: Sized {}
    pub struct Bounds<T>(T);
    impl<T> Sealed for Bounds<T> {}
}
use sealed_lifetime::{Bounds, Sealed};

/// A wrapper around [`String`] that is used by
/// [`PathTracker::with_owned_field`]. This wrapper can be used even if the
/// `alloc` feature isn't enabled.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct OwnedField(
    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "docs", doc(cfg(feature = "alloc")))]
    pub alloc::string::String,
);
impl OwnedField {
    cfg_with_docs!(feature = "alloc", {
        impl Self {
            pub fn new(value: alloc::string::String) -> Self {
                Self(value)
            }
        }
    });
    pub fn empty() -> Self {
        Default::default()
    }
    /// This creates an empty string if the `alloc` feature is disabled.
    pub fn lossy_to_string(_text: &str) -> Self {
        #[cfg(feature = "alloc")]
        {
            Self::new(_text.into())
        }
        #[cfg(not(feature = "alloc"))]
        {
            Self::empty()
        }
    }
}
impl Deref for OwnedField {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        #[cfg(feature = "alloc")]
        {
            &self.0
        }
        #[cfg(not(feature = "alloc"))]
        {
            ""
        }
    }
}

/// Supertrait for [`PathTracker`] that emulates lifetime GATs.
pub trait PathTrackerLifetime<'borrow, ImplicitBounds: Sealed = Bounds<&'borrow Self>> {
    type LinkedTracker: PathTracker;
    type SegmentIter: Iterator<Item = ProjectionPathSegment<'borrow>>;
}
/// A type that tracks the path to a value.
pub trait PathTracker: Clone + for<'borrow> PathTrackerLifetime<'borrow> {
    fn linked(previous: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker;

    type OwnedTracker: PathTracker + 'static + Send + Sync;
    fn into_owned(previous: Self) -> Self::OwnedTracker;

    type StaticProjectionTracker: PathTracker;
    fn with_static_projection(
        previous: Self,
        projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker;

    type StaticStrFieldTracker: PathTracker;
    fn with_field(previous: Self, field: &'static str) -> Self::StaticStrFieldTracker;

    type OwnedFieldTracker: PathTracker;
    fn with_owned_field(previous: Self, field: OwnedField) -> Self::OwnedFieldTracker;

    type IndexTracker: PathTracker;
    fn with_index(previous: Self, index: usize) -> Self::IndexTracker;

    fn segment_iter(tracker: &Self) -> <Self as PathTrackerLifetime<'_>>::SegmentIter;

    /// Determines if the tracker path is empty. If this is `true` then
    /// [`segment_iter`] should return an empty iterator.
    ///
    /// [`segment_iter`]: PathTracker::segment_iter
    fn is_empty(tracker: &Self) -> bool;

    /// The number of path segments that is stored inside this tracker. The
    /// iterator returned by [`segment_iter`] should return this many segments.
    ///
    /// [`segment_iter`]: PathTracker::segment_iter
    fn len(tracker: &Self) -> usize;

    /// Customize how the path should be formatted.
    fn format_path<'segment, I>(
        tracker: &Self,
        iterator: I,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    where
        I: Iterator<Item = ProjectionPathSegment<'segment>>;

    fn display(tracker: &Self) -> Option<PathTrackerDisplay<'_, Self>> {
        if PathTracker::is_empty(tracker) {
            None
        } else {
            Some(PathTrackerDisplay(tracker))
        }
    }
}

impl<'borrow> PathTrackerLifetime<'borrow> for () {
    type LinkedTracker = ();
    type SegmentIter = iter::Empty<ProjectionPathSegment<'borrow>>;
}
/// A `PathTracker` implementation that does nothing.
impl PathTracker for () {
    fn linked(_previous: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker {}

    type OwnedTracker = ();
    fn into_owned(_previous: Self) -> Self::OwnedTracker {}

    type StaticProjectionTracker = ();
    fn with_static_projection(
        _previous: Self,
        _projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker {
    }

    type StaticStrFieldTracker = ();
    fn with_field(_previous: Self, _field: &'static str) -> Self::StaticStrFieldTracker {}

    type IndexTracker = ();
    fn with_index(_previous: Self, _index: usize) -> Self::IndexTracker {}

    type OwnedFieldTracker = ();
    fn with_owned_field(_previous: Self, _field: OwnedField) -> Self::OwnedFieldTracker {}

    fn segment_iter(_tracker: &Self) -> <Self as PathTrackerLifetime<'_>>::SegmentIter {
        iter::empty()
    }

    fn is_empty(_tracker: &Self) -> bool {
        true
    }

    fn len(_tracker: &Self) -> usize {
        0
    }

    fn format_path<'segment, I>(
        _tracker: &Self,
        _iterator: I,
        _f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    where
        I: Iterator<Item = ProjectionPathSegment<'segment>>,
    {
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////
// View API (used by generated code)
////////////////////////////////////////////////////////////////////////////////

/// Marker trait for types that deserialize from arrays.
///
/// It should be possible to project into [`UnstructuredData`] using indexes if
/// the data would deserialize into a type that implements this trait.
pub trait SerdeArrayLike {
    /// The type that an item inside the array is expected to deserialize into.
    type Item;
}
cfg_with_docs!(
    feature = "alloc",
    {
        impl<T> SerdeArrayLike for alloc::vec::Vec<T> {
            type Item = T;
        }
    },
    {
        impl<T> SerdeArrayLike for alloc::boxed::Box<T>
        where
            T: ?Sized + SerdeArrayLike,
        {
            type Item = T::Item;
        }
    },
    {
        impl<T> SerdeArrayLike for alloc::rc::Rc<T>
        where
            T: ?Sized + SerdeArrayLike,
        {
            type Item = T::Item;
        }
    },
    {
        impl<T> SerdeArrayLike for alloc::sync::Arc<T>
        where
            T: ?Sized + SerdeArrayLike,
        {
            type Item = T::Item;
        }
    }
);
impl<T> SerdeArrayLike for [T] {
    type Item = T;
}
impl<T, const N: usize> SerdeArrayLike for [T; N] {
    type Item = T;
}
impl<T> SerdeArrayLike for &'_ T
where
    T: ?Sized + SerdeArrayLike,
{
    type Item = T::Item;
}
impl<T> SerdeArrayLike for &'_ mut T
where
    T: ?Sized + SerdeArrayLike,
{
    type Item = T::Item;
}

/// Provide a structured view into unstructured data. Types that implement this
/// can be projected into using [`project`](SerdeViewTyped::project) methods.
///
/// # Type Parameter
///
/// The `TPrevPath` type parameter represents the path that will be continued by
/// the returned view type.
pub trait SerdeView<TPrevPath: ZeroSizedPath> {
    /// A type that has methods to conveniently project into unstructured data.
    type View;
    /// Create the view type.
    fn get_view<'a>() -> &'a Self::View;
}

/// A borrowed value that can be either `&'a T` or `&'static T`.
#[derive(Debug)]
pub enum MaybeStatic<'a, T: ?Sized + 'static> {
    Borrowed(&'a T),
    Static(&'static T),
}
impl<T> Clone for MaybeStatic<'_, T>
where
    T: ?Sized + 'static,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for MaybeStatic<'_, T> where T: ?Sized + 'static {}
impl<'a, T> Deref for MaybeStatic<'a, T>
where
    T: ?Sized + 'static,
{
    type Target = &'a T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            MaybeStatic::Borrowed(v) => v,
            MaybeStatic::Static(v) => v,
        }
    }
}
impl<T> fmt::Display for MaybeStatic<'_, T>
where
    T: fmt::Display,
    T: ?Sized + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaybeStatic::Borrowed(v) => fmt::Display::fmt(*v, f),
            MaybeStatic::Static(v) => fmt::Display::fmt(*v, f),
        }
    }
}

/// A segment of the path taken by a projection.
#[derive(Debug, Clone, Copy)]
pub enum ProjectionPathSegment<'a> {
    Index(usize),
    Field(MaybeStatic<'a, str>),
}
impl ProjectionPathSegment<'_> {
    /// Format an iterator of path segments in a sensible way.
    pub fn format_path<'segment, I>(segments: I, f: &mut fmt::Formatter<'_>) -> fmt::Result
    where
        I: Iterator<Item = ProjectionPathSegment<'segment>>,
    {
        // Note: we use the Display impl for `view_path::ViewPathSegmentRef` to
        // format segments.

        let mut segments = segments.map(view_path::ViewPathSegmentRef::from);
        if let Some(segment) = segments.next() {
            write!(f, "{}", segment)?;
        }
        for segment in segments {
            if segment.is_field() {
                write!(f, ".")?;
            }
            write!(f, "{}", segment)?;
        }
        Ok(())
    }
}

/// Used by macro to specify projection path segments.
pub trait GetZeroSizedProjectionPathSegment {
    /// Get the path that should be projected into.
    ///
    /// The path can depend on the data's state. For example if multiple field
    /// names are allowed (aliases) then the implementation could check which of
    /// those field names exists and return one of those as the projection path.
    fn get_path(data: &dyn DynUnstructuredData) -> ProjectionPathSegment<'static>;
}

/// Marker trait that indicates that a projection path is zero sized.
pub trait ZeroSizedPath: Default {}
impl<A, B> ZeroSizedPath for (A, B)
where
    A: ZeroSizedPath,
    B: ZeroSizedPath,
{
}
impl ZeroSizedPath for () {}

/// Provides trait implementations for identifiers that are defined by zero
/// sized types.
pub struct ZeroSizedProjectionPath<T> {
    ident: PhantomData<fn() -> T>,
}
impl<T> ZeroSizedProjectionPath<T> {
    pub const NEW: Self = Self::new();

    #[inline]
    pub const fn new() -> Self {
        Self { ident: PhantomData }
    }
}
impl<T> Clone for ZeroSizedProjectionPath<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for ZeroSizedProjectionPath<T> {}
impl<T> Default for ZeroSizedProjectionPath<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl<T> ZeroSizedPath for ZeroSizedProjectionPath<T> {}

/// Used by `project` methods to specify what field/index should be projected
/// into.
pub struct ProjectionTarget<TTarget, TPath> {
    /// The expected type of the unstructured data after the projection.
    target_type: PhantomData<fn() -> TTarget>,
    /// The path the projection should follow.
    path: TPath,
}
impl<TTarget, TPath> ProjectionTarget<TTarget, TPath> {
    #[inline]
    pub fn new(path: TPath) -> Self {
        Self {
            target_type: PhantomData,
            path,
        }
    }

    /// Project into the fields of the target type. This is usually done
    /// automatically via the [`Deref`] trait, but if a field name overlaps with
    /// a method name then this can be useful.
    pub fn fields(&self) -> &<TTarget as SerdeView<TPath>>::View
    where
        // Methods on the created view can't be used unless `Path` implements `ZeroSizedPath`.
        TPath: ZeroSizedPath,
        TTarget: SerdeView<TPath>,
    {
        self
    }

    /// Project into an array.
    #[inline]
    pub fn get(&self, index: usize) -> ProjectionTarget<TTarget::Item, (TPath, usize)>
    where
        TPath: Clone,
        TTarget: SerdeArrayLike,
    {
        ProjectionTarget::new((self.path.clone(), index))
    }
}
impl ProjectionTarget<(), ()> {
    /// Project into an index.
    pub fn index(index: usize) -> ProjectionTarget<(), usize> {
        ProjectionTarget::new(index)
    }
}
impl<TTarget, TPath> Default for ProjectionTarget<TTarget, TPath>
where
    TPath: Default,
{
    #[inline]
    fn default() -> Self {
        Self::new(Default::default())
    }
}
/// If the target type implements [`SerdeView`] then we can project further.
impl<TTarget, TPath> Deref for ProjectionTarget<TTarget, TPath>
where
    // Methods on the created view can't be used unless `Path` implements `ZeroSizedPath`.
    TPath: ZeroSizedPath,
    TTarget: SerdeView<TPath>,
{
    type Target = <TTarget as SerdeView<TPath>>::View;

    #[inline]
    fn deref(&self) -> &Self::Target {
        TTarget::get_view()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unstructured data wrapper (provides a type and helper methods)
////////////////////////////////////////////////////////////////////////////////

/// Wraps unstructured data and provides it a "type".
pub struct SerdeViewTyped<D, V, T> {
    /// Unstructured data.
    pub data: D,
    view_type: PhantomData<fn() -> V>,
    /// The path taken through the unstructured data to reach the current value.
    pub tracker: T,
}
impl<D, V, T> SerdeViewTyped<D, V, T>
where
    D: UnstructuredData,
    T: PathTracker,
{
    #[inline]
    pub fn new(data: D, tracker: T) -> Self {
        Self {
            data,
            view_type: PhantomData,
            tracker,
        }
    }

    /// Deserialize the "unstructured" data into the expected type.
    pub fn deserialize(
        self,
    ) -> Result<V, SerdeViewError<<D as UnstructuredData>::DeserializationError, T>>
    where
        V: DeserializeOwned,
    {
        let path = self.tracker;
        D::deserialize_into(self.data).map_err(|e| SerdeViewError::new(e, path))
    }

    /// Project into the wrapped data.
    pub fn project<F, R>(self, f: F) -> R::Output
    where
        F: FnOnce(ProjectionTarget<V, ()>) -> R,
        R: ProjectionRequestList<
            SerdeViewTyped<(), (), ()>,
            D,
            ProjectOwned,
            SerdeViewError<(), ()>,
            T,
        >,
    {
        let target_list = f(ProjectionTarget::new(()));

        let result = project_into_list(&target_list, self.data, |data, node| {
            D::multi_project_into(data, node)
        });
        R::into_output(&target_list, result.0, self.tracker, result.1)
    }
    /// Project into the wrapped data. Return a single error if any projection
    /// failed.
    pub fn try_project<F, R>(
        self,
        f: F,
    ) -> Result<R::UnwrappedOutput, SerdeViewError<ValueNotFoundError, R::ErrorPath>>
    where
        F: FnOnce(ProjectionTarget<V, ()>) -> R,
        R: ProjectionRequestList<
            SerdeViewTyped<(), (), ()>,
            D,
            ProjectOwned,
            SerdeViewError<(), ()>,
            T,
        >,
    {
        let target_list = f(ProjectionTarget::new(()));

        let result = project_into_list(&target_list, self.data, |data, node| {
            D::multi_project_into(data, node)
        });
        try_unwrap_projection_result(&target_list, result.0, result.1, self.tracker)
    }

    /// Create a new view wrapper where the "unstructured" data is immutably
    /// borrowed from this wrapper.
    #[inline]
    pub fn as_ref(
        &self,
    ) -> SerdeViewTypedRef<'_, D, V, <T as PathTrackerLifetime<'_>>::LinkedTracker> {
        SerdeViewTypedRef::new(&self.data, T::linked(&self.tracker))
    }

    /// Create a new view wrapper where the "unstructured" data is mutably borrowed
    /// from this wrapper.
    #[inline]
    pub fn as_mut(
        &mut self,
    ) -> SerdeViewTypedMut<'_, D, V, <T as PathTrackerLifetime<'_>>::LinkedTracker> {
        SerdeViewTypedMut::new(&mut self.data, T::linked(&self.tracker))
    }

    /// Use the default path tracker to get better error messages if a
    /// projection fails.
    #[inline]
    pub fn with_tracker(
        self,
    ) -> SerdeViewTyped<
        D,
        V,
        view_path::ViewPath<
            'static,
            view_path::StorageConfig<view_path::HasLink<false>, view_path::EmptyStorage>,
        >,
    > {
        self.with_custom_tracker(view_path::ViewPath::empty())
    }
    /// Don't use a path tracker to improve error messages for projection
    /// failures.
    ///
    /// This can be useful to decrease overhead or simplify lifetimes.
    #[inline]
    pub fn without_tracker(self) -> SerdeViewTyped<D, V, ()> {
        self.with_custom_tracker(())
    }
    /// Use a custom path tracker to improve error messages if a projection
    /// fails.
    #[inline]
    pub fn with_custom_tracker<T2>(self, tracker: T2) -> SerdeViewTyped<D, V, T2>
    where
        T2: PathTracker,
    {
        SerdeViewTyped {
            data: self.data,
            view_type: PhantomData,
            tracker,
        }
    }

    /// Ensure path tracker is owned and doesn't cause lifetime issues.
    #[inline]
    pub fn with_owned_tracker(self) -> SerdeViewTyped<D, V, <T as PathTracker>::OwnedTracker> {
        SerdeViewTyped::new(self.data, T::into_owned(self.tracker))
    }

    #[inline]
    pub fn cast<V2>(self) -> SerdeViewTyped<D, V2, T> {
        SerdeViewTyped {
            data: self.data,
            view_type: PhantomData,
            tracker: self.tracker,
        }
    }

    /// Check if the wrapped data is an array and if so iterate over its values.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn try_array_iter(
        self,
    ) -> Result<
        SerdeViewTypedIter<D, V::Item, T>,
        SerdeViewError<ValueNotArrayError, <T as PathTracker>::OwnedTracker>,
    >
    where
        V: SerdeArrayLike,
    {
        self.try_unchecked_array_iter()
    }
    /// Check if the wrapped data is an array and if so iterate over its values.
    /// Prefer [`try_array_iter`] since it specifies the type of the iterator's
    /// items.
    ///
    /// [`try_array_iter`]: Self::try_array_iter
    #[inline]
    pub fn try_unchecked_array_iter<V2>(
        self,
    ) -> Result<
        SerdeViewTypedIter<D, V2, T>,
        SerdeViewError<ValueNotArrayError, <T as PathTracker>::OwnedTracker>,
    > {
        let inner = match <D as UnstructuredData>::array_into_iter(self.data) {
            Some(inner) => inner,
            None => {
                return Err(SerdeViewError::new(
                    ValueNotArrayError,
                    PathTracker::into_owned(self.tracker),
                ))
            }
        };
        Ok(SerdeViewTypedIter {
            inner,
            marker: PhantomData,
            tracker: self.tracker,
            index: 0,
        })
    }
}
impl<D, V, T> Clone for SerdeViewTyped<D, V, T>
where
    D: Clone,
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            view_type: PhantomData,
            tracker: self.tracker.clone(),
        }
    }
}

/// Created by [`SerdeViewTyped::try_unchecked_array_iter`].
pub struct SerdeViewTypedIter<D, V, T>
where
    D: UnstructuredData,
{
    inner: <D as UnstructuredData>::ArrayIntoIter,
    marker: PhantomData<fn() -> V>,
    tracker: T,
    index: usize,
}
impl<D, V, T> Iterator for SerdeViewTypedIter<D, V, T>
where
    D: UnstructuredData,
    T: PathTracker,
{
    type Item = SerdeViewTyped<D, V, <T as PathTracker>::IndexTracker>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.inner.next()?;
        let index = self.index;
        self.index += 1;
        Some(SerdeViewTyped::new(
            next,
            PathTracker::with_index(self.tracker.clone(), index),
        ))
    }
}

/// Wraps unstructured data and provides it a "type".
pub struct SerdeViewTypedMut<'data, D, V, T> {
    /// Unstructured data.
    pub data: &'data mut D,
    view_type: PhantomData<fn() -> V>,
    /// The path taken through the unstructured data to reach the current value.
    pub tracker: T,
}
impl<'data, D, V, T> SerdeViewTypedMut<'data, D, V, T>
where
    D: UnstructuredData,
    T: PathTracker,
{
    #[inline]
    pub fn new(data: &'data mut D, tracker: T) -> Self {
        Self {
            data,
            view_type: PhantomData,
            tracker,
        }
    }

    /// Project into the wrapped data.
    pub fn project<F, R>(self, f: F) -> R::Output
    where
        F: FnOnce(ProjectionTarget<V, ()>) -> R,
        R: ProjectionRequestList<
            SerdeViewTypedMut<'data, (), (), ()>,
            &'data mut D,
            ProjectBorrowed,
            SerdeViewError<(), ()>,
            T,
        >,
    {
        let target_list = f(ProjectionTarget::new(()));

        let result = project_into_list(&target_list, self.data, |data, node| {
            D::multi_project_mut(data, node)
        });
        R::into_output(&target_list, result.0, self.tracker, result.1)
    }
    /// Project into the wrapped data. Return a single error if any projection
    /// failed.
    pub fn try_project<F, R>(
        self,
        f: F,
    ) -> Result<R::UnwrappedOutput, SerdeViewError<ValueNotFoundError, R::ErrorPath>>
    where
        F: FnOnce(ProjectionTarget<V, ()>) -> R,
        R: ProjectionRequestList<
            SerdeViewTypedMut<'data, (), (), ()>,
            &'data mut D,
            ProjectBorrowed,
            SerdeViewError<(), ()>,
            T,
        >,
    {
        let target_list = f(ProjectionTarget::new(()));

        let result = project_into_list(&target_list, self.data, |data, node| {
            D::multi_project_mut(data, node)
        });
        try_unwrap_projection_result(&target_list, result.0, result.1, self.tracker)
    }

    /// Create a new view wrapper where the "unstructured" data is immutably
    /// borrowed from this wrapper.
    #[inline]
    pub fn as_ref(
        &self,
    ) -> SerdeViewTypedRef<'_, D, V, <T as PathTrackerLifetime<'_>>::LinkedTracker> {
        SerdeViewTypedRef::new(self.data, T::linked(&self.tracker))
    }

    /// Create a new view wrapper where the "unstructured" data is mutably borrowed
    /// from this wrapper.
    #[inline]
    pub fn as_mut(
        &mut self,
    ) -> SerdeViewTypedMut<'_, D, V, <T as PathTrackerLifetime<'_>>::LinkedTracker> {
        SerdeViewTypedMut::new(self.data, T::linked(&self.tracker))
    }

    /// Use the default path tracker to get better error messages if a
    /// projection fails.
    #[inline]
    pub fn with_tracker(
        self,
    ) -> SerdeViewTypedMut<
        'data,
        D,
        V,
        view_path::ViewPath<
            'static,
            view_path::StorageConfig<view_path::HasLink<false>, view_path::EmptyStorage>,
        >,
    > {
        self.with_custom_tracker(view_path::ViewPath::empty())
    }
    /// Don't use a path tracker to improve error messages for projection
    /// failures.
    ///
    /// This can be useful to decrease overhead or simplify lifetimes.
    #[inline]
    pub fn without_tracker(self) -> SerdeViewTypedMut<'data, D, V, ()> {
        self.with_custom_tracker(())
    }
    /// Use a custom path tracker to improve error messages if a projection
    /// fails.
    #[inline]
    pub fn with_custom_tracker<T2>(self, tracker: T2) -> SerdeViewTypedMut<'data, D, V, T2>
    where
        T2: PathTracker,
    {
        SerdeViewTypedMut {
            data: self.data,
            view_type: PhantomData,
            tracker,
        }
    }

    /// Ensure path tracker is owned and doesn't cause lifetime issues.
    #[inline]
    pub fn with_owned_tracker(
        self,
    ) -> SerdeViewTypedMut<'data, D, V, <T as PathTracker>::OwnedTracker> {
        SerdeViewTypedMut::new(self.data, PathTracker::into_owned(self.tracker))
    }

    #[inline]
    pub fn cast<V2>(self) -> SerdeViewTypedMut<'data, D, V2, T> {
        SerdeViewTypedMut {
            data: self.data,
            view_type: PhantomData,
            tracker: self.tracker,
        }
    }

    #[inline]
    pub fn try_retain<F>(
        self,
        f: F,
    ) -> Result<(), SerdeViewError<ValueNotArrayError, <T as PathTracker>::OwnedTracker>>
    where
        F: FnMut(SerdeViewTypedMut<'_, D, V::Item, <T as PathTracker>::IndexTracker>) -> bool,
        V: SerdeArrayLike,
    {
        self.try_unchecked_retain(f)
    }
    #[inline]
    pub fn try_unchecked_retain<F, V2>(
        self,
        mut f: F,
    ) -> Result<(), SerdeViewError<ValueNotArrayError, <T as PathTracker>::OwnedTracker>>
    where
        F: FnMut(SerdeViewTypedMut<'_, D, V2, <T as PathTracker>::IndexTracker>) -> bool,
    {
        if !UnstructuredData::is_array(&*self.data) {
            return Err(SerdeViewError::new(
                ValueNotArrayError,
                PathTracker::into_owned(self.tracker),
            ));
        }

        let mut count = 0;
        let tracker = &self.tracker;
        UnstructuredData::array_retain(self.data, |item| {
            let index = count;
            count += 1;
            f(SerdeViewTypedMut::new(
                item,
                <T as PathTracker>::with_index(tracker.clone(), index),
            ))
        });

        Ok(())
    }

    /// Check if the wrapped data is an array and if so iterate over its values.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn try_array_iter(
        self,
    ) -> Result<
        SerdeViewTypedMutIter<'data, D, V::Item, T>,
        SerdeViewError<ValueNotArrayError, <T as PathTracker>::OwnedTracker>,
    >
    where
        V: SerdeArrayLike,
    {
        self.try_unchecked_array_iter()
    }
    /// Check if the wrapped data is an array and if so iterate over its values.
    /// Prefer [`try_array_iter`] since it specifies the type of the iterator's
    /// items.
    ///
    /// [`try_array_iter`]: Self::try_array_iter
    #[inline]
    pub fn try_unchecked_array_iter<V2>(
        self,
    ) -> Result<
        SerdeViewTypedMutIter<'data, D, V2, T>,
        SerdeViewError<ValueNotArrayError, <T as PathTracker>::OwnedTracker>,
    > {
        let inner = match <D as UnstructuredData>::array_iter_mut(self.data) {
            Some(inner) => inner,
            None => {
                return Err(SerdeViewError::new(
                    ValueNotArrayError,
                    PathTracker::into_owned(self.tracker),
                ))
            }
        };
        Ok(SerdeViewTypedMutIter {
            inner,
            marker: PhantomData,
            tracker: self.tracker,
            index: 0,
        })
    }
}

/// Created by [`SerdeViewTypedMut::try_unchecked_array_iter`].
pub struct SerdeViewTypedMutIter<'a, D, V, T>
where
    D: UnstructuredData,
{
    inner: <D as UnstructuredDataLifetime<'a>>::ArrayIterMut,
    marker: PhantomData<fn() -> V>,
    tracker: T,
    index: usize,
}
impl<'a, D, V, T> Iterator for SerdeViewTypedMutIter<'a, D, V, T>
where
    D: UnstructuredData,
    T: PathTracker,
{
    type Item = SerdeViewTypedMut<'a, D, V, <T as PathTracker>::IndexTracker>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.inner.next()?;
        let index = self.index;
        self.index += 1;
        Some(SerdeViewTypedMut::new(
            next,
            PathTracker::with_index(self.tracker.clone(), index),
        ))
    }
}

/// Wraps unstructured data and provides it a "type".
pub struct SerdeViewTypedRef<'data, D, V, T> {
    /// Unstructured data.
    pub data: &'data D,
    view_type: PhantomData<fn() -> V>,
    /// The path taken through the unstructured data to reach the current value.
    pub tracker: T,
}
impl<'data, D, V, T> SerdeViewTypedRef<'data, D, V, T>
where
    D: UnstructuredData,
    T: PathTracker,
{
    #[inline]
    pub fn new(data: &'data D, tracker: T) -> Self {
        Self {
            data,
            view_type: PhantomData,
            tracker,
        }
    }

    /// Deserialize the "unstructured" data into the expected type.
    pub fn deserialize(
        self,
    ) -> Result<V, SerdeViewError<<D as UnstructuredData>::DeserializationError, T>>
    where
        V: DeserializeOwned,
    {
        D::deserialize_borrowed(self.data).map_err(|e| SerdeViewError::new(e, self.tracker))
    }

    /// Project into the wrapped data.
    pub fn project<F, R>(self, f: F) -> R::Output
    where
        F: FnOnce(ProjectionTarget<V, ()>) -> R,
        R: ProjectionRequestList<
            SerdeViewTypedRef<'data, (), (), ()>,
            &'data D,
            ProjectBorrowed,
            SerdeViewError<(), ()>,
            T,
        >,
    {
        let target_list = f(ProjectionTarget::new(()));

        let result = project_into_list_via_clone(&target_list, self.data);
        R::into_output(&target_list, result.0, self.tracker, result.1)
    }
    /// Project into the wrapped data. Return a single error if any projection
    /// failed.
    pub fn try_project<F, R>(
        self,
        f: F,
    ) -> Result<R::UnwrappedOutput, SerdeViewError<ValueNotFoundError, R::ErrorPath>>
    where
        F: FnOnce(ProjectionTarget<V, ()>) -> R,
        R: ProjectionRequestList<
            SerdeViewTypedRef<'data, (), (), ()>,
            &'data D,
            ProjectBorrowed,
            SerdeViewError<(), ()>,
            T,
        >,
    {
        let target_list = f(ProjectionTarget::new(()));

        let result = project_into_list_via_clone(&target_list, self.data);
        try_unwrap_projection_result(&target_list, result.0, result.1, self.tracker)
    }

    /// Create a new view wrapper where the "unstructured" data is immutably
    /// borrowed from this wrapper.
    #[inline]
    pub fn as_ref(
        &self,
    ) -> SerdeViewTypedRef<'_, D, V, <T as PathTrackerLifetime<'_>>::LinkedTracker> {
        SerdeViewTypedRef::new(self.data, PathTracker::linked(&self.tracker))
    }

    /// Use the default path tracker to get better error messages if a
    /// projection fails.
    #[inline]
    pub fn with_tracker(
        self,
    ) -> SerdeViewTypedRef<
        'data,
        D,
        V,
        view_path::ViewPath<
            'static,
            view_path::StorageConfig<view_path::HasLink<false>, view_path::EmptyStorage>,
        >,
    > {
        self.with_custom_tracker(view_path::ViewPath::empty())
    }
    /// Don't use a path tracker to improve error messages for projection
    /// failures.
    ///
    /// This can be useful to decrease overhead or simplify lifetimes.
    #[inline]
    pub fn without_tracker(self) -> SerdeViewTypedRef<'data, D, V, ()> {
        self.with_custom_tracker(())
    }
    /// Use a custom path tracker to improve error messages if a projection
    /// fails.
    #[inline]
    pub fn with_custom_tracker<T2>(self, tracker: T2) -> SerdeViewTypedRef<'data, D, V, T2>
    where
        T2: PathTracker,
    {
        SerdeViewTypedRef {
            data: self.data,
            view_type: PhantomData,
            tracker,
        }
    }

    /// Ensure path tracker is owned and doesn't cause lifetime issues.
    #[inline]
    pub fn with_owned_tracker(
        self,
    ) -> SerdeViewTypedRef<'data, D, V, <T as PathTracker>::OwnedTracker> {
        SerdeViewTypedRef::new(self.data, PathTracker::into_owned(self.tracker))
    }

    #[inline]
    pub fn cast<V2>(self) -> SerdeViewTypedRef<'data, D, V2, T> {
        SerdeViewTypedRef {
            data: self.data,
            view_type: PhantomData,
            tracker: self.tracker,
        }
    }

    /// Check if the wrapped data is an array and if so iterate over its values.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn try_array_iter(
        self,
    ) -> Result<
        SerdeViewTypedRefIter<'data, D, V::Item, T>,
        SerdeViewError<ValueNotArrayError, <T as PathTracker>::OwnedTracker>,
    >
    where
        V: SerdeArrayLike,
    {
        self.try_unchecked_array_iter()
    }
    /// Check if the wrapped data is an array and if so iterate over its values.
    /// Prefer [`try_array_iter`] since it specifies the type of the iterator's
    /// items.
    ///
    /// [`try_array_iter`]: Self::try_array_iter
    #[inline]
    pub fn try_unchecked_array_iter<V2>(
        self,
    ) -> Result<
        SerdeViewTypedRefIter<'data, D, V2, T>,
        SerdeViewError<ValueNotArrayError, <T as PathTracker>::OwnedTracker>,
    > {
        let inner = match <D as UnstructuredData>::array_iter(self.data) {
            Some(inner) => inner,
            None => {
                return Err(SerdeViewError::new(
                    ValueNotArrayError,
                    PathTracker::into_owned(self.tracker),
                ))
            }
        };
        Ok(SerdeViewTypedRefIter {
            inner,
            marker: PhantomData,
            tracker: self.tracker,
            index: 0,
        })
    }
}
impl<D, V, T> Clone for SerdeViewTypedRef<'_, D, V, T>
where
    T: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            view_type: PhantomData,
            tracker: self.tracker.clone(),
        }
    }
}
impl<D, V, T> Copy for SerdeViewTypedRef<'_, D, V, T> where T: Copy {}

/// Created by [`SerdeViewTypedRef::try_unchecked_array_iter`].
pub struct SerdeViewTypedRefIter<'a, D, V, T>
where
    D: UnstructuredData,
{
    inner: <D as UnstructuredDataLifetime<'a>>::ArrayIter,
    marker: PhantomData<fn() -> V>,
    tracker: T,
    index: usize,
}
impl<'a, D, V, T> Iterator for SerdeViewTypedRefIter<'a, D, V, T>
where
    D: UnstructuredData,
    T: PathTracker,
{
    type Item = SerdeViewTypedRef<'a, D, V, <T as PathTracker>::IndexTracker>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.inner.next()?;
        let index = self.index;
        self.index += 1;
        Some(SerdeViewTypedRef::new(
            next,
            PathTracker::with_index(self.tracker.clone(), index),
        ))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Standalone "new" functions
////////////////////////////////////////////////////////////////////////////////

/// Used by [`view_with_tracker`] and similar standalone functions to be generic
/// over ownership for the [`UnstructuredData`] type.
pub trait WrapInSerdeView<View, Tracker> {
    /// The typed serde view wrapper such as [`SerdeViewTyped`] that will be
    /// created.
    type Wrapper;
    /// Create the wrapper.
    fn create(data: Self, tracker: Tracker) -> Self::Wrapper;
}

/// Will wrap the data inside [`SerdeViewTyped`], [`SerdeViewTypedMut`] or
/// [`SerdeViewTypedRef`] depending on the ownership of the provided data. Uses
/// the default path tracker for better error messages.
#[inline]
pub fn view<Data>(data: Data) -> Data::Wrapper
where
    Data: WrapInSerdeView<
        (),
        view_path::ViewPath<
            'static,
            view_path::StorageConfig<view_path::HasLink<false>, view_path::EmptyStorage>,
        >,
    >,
{
    WrapInSerdeView::create(data, view_path::ViewPath::empty())
}
/// Will wrap the data inside [`SerdeViewTyped`], [`SerdeViewTypedMut`] or
/// [`SerdeViewTypedRef`] depending on the ownership of the provided data.
#[inline]
pub fn view_without_tracker<Data>(data: Data) -> Data::Wrapper
where
    Data: WrapInSerdeView<(), ()>,
{
    WrapInSerdeView::create(data, ())
}
/// Will wrap the data inside [`SerdeViewTyped`], [`SerdeViewTypedMut`] or
/// [`SerdeViewTypedRef`] depending on the ownership of the provided data. Uses
/// a custom path tracker for better error messages.
#[inline]
pub fn view_with_tracker<Data, Tracker>(data: Data, tracker: Tracker) -> Data::Wrapper
where
    Data: WrapInSerdeView<(), Tracker>,
    Tracker: PathTracker,
{
    WrapInSerdeView::create(data, tracker)
}

////////////////////////////////////////////////////////////////////////////////
// Serde Value abstraction
////////////////////////////////////////////////////////////////////////////////

/// Used with [`UnstructuredData`] to project into multiple fields of some data
/// at the the same time.
///
/// `D` is the [`UnstructuredData`] that is the result of a projection.
#[allow(clippy::len_without_is_empty)]
pub trait MultiProjectionRequest<D> {
    /// Get the field/index that that a request wants to project into.
    ///
    /// # Panics
    ///
    /// If `index` is greater than or equal to `len`.
    fn get(&self, index: usize) -> ProjectionPathSegment;

    /// Set the result of a request's projection.
    ///
    /// # Panics
    ///
    /// If `index` is greater than or equal to `len`.
    fn set(&mut self, index: usize, value: D);

    /// The number of projections that are requested.
    fn len(&self) -> usize;
}
impl<D, R> MultiProjectionRequest<D> for &mut R
where
    R: MultiProjectionRequest<D> + ?Sized,
{
    #[inline]
    fn get(&self, index: usize) -> ProjectionPathSegment {
        R::get(self, index)
    }

    #[inline]
    fn set(&mut self, index: usize, value: D) {
        R::set(self, index, value)
    }

    #[inline]
    fn len(&self) -> usize {
        R::len(self)
    }
}

/// Supertrait for [`UnstructuredData`] that emulates lifetime GATs.
pub trait UnstructuredDataLifetime<'borrow, ImplicitBounds: Sealed = Bounds<&'borrow Self>>:
    'borrow
{
    type ArrayIter: Iterator<Item = &'borrow Self>;
    type ArrayIterMut: Iterator<Item = &'borrow mut Self>;
}
/// A type that stores unstructured data. For example `serde_json::Value`.
pub trait UnstructuredData: Sized + for<'borrow> UnstructuredDataLifetime<'borrow> {
    type ArrayIntoIter: Iterator<Item = Self>;
    type DeserializationError;

    fn deserialize_into<T>(this: Self) -> Result<T, Self::DeserializationError>
    where
        T: DeserializeOwned;

    fn deserialize_borrowed<T>(this: &Self) -> Result<T, Self::DeserializationError>
    where
        T: DeserializeOwned;

    fn multi_project<'a, R>(this: &'a Self, mut req: R)
    where
        R: MultiProjectionRequest<&'a Self>,
    {
        for i in 0..req.len() {
            let result = match req.get(i) {
                ProjectionPathSegment::Index(index) => Self::array_project(this, index),
                ProjectionPathSegment::Field(field) => Self::object_project(this, &field),
            };
            if let Some(value) = result {
                req.set(i, value);
            }
        }
    }
    /// Project into multiple fields.
    ///
    /// # Panics
    ///
    /// If the path segments specify indexes and those indexes aren't sorted
    /// from smallest to largest.
    fn multi_project_mut<'a, R>(this: &'a mut Self, req: R)
    where
        R: MultiProjectionRequest<&'a mut Self>;
    /// Project into multiple fields.
    ///
    /// # Panics
    ///
    /// If the path segments specify indexes and those indexes aren't sorted
    /// from smallest to largest.
    fn multi_project_into<R>(this: Self, req: R)
    where
        R: MultiProjectionRequest<Self>;

    fn is_object(this: &Self) -> bool;

    fn object_project<'a>(this: &'a Self, field: &str) -> Option<&'a Self>;
    fn object_project_mut<'a>(this: &'a mut Self, field: &str) -> Option<&'a mut Self>;
    fn object_project_into(this: Self, field: &str) -> Option<Self>;

    fn is_array(this: &Self) -> bool;
    fn array_retain<F>(this: &mut Self, f: F)
    where
        F: FnMut(&mut Self) -> bool;

    fn array_project(this: &Self, index: usize) -> Option<&Self>;
    fn array_project_mut(this: &mut Self, index: usize) -> Option<&mut Self>;
    fn array_project_into(this: Self, index: usize) -> Option<Self>;

    fn array_iter(this: &Self) -> Option<<Self as UnstructuredDataLifetime<'_>>::ArrayIter>;
    fn array_iter_mut(
        this: &mut Self,
    ) -> Option<<Self as UnstructuredDataLifetime<'_>>::ArrayIterMut>;
    fn array_into_iter(this: Self) -> Option<Self::ArrayIntoIter>;
}

/// Object safe version of [`UnstructuredData`].
pub trait DynUnstructuredData {
    fn dyn_object_project<'a>(&'a self, field: &str) -> Option<&'a dyn DynUnstructuredData>;

    fn dyn_array_project(&self, index: usize) -> Option<&dyn DynUnstructuredData>;
}
impl<D> DynUnstructuredData for D
where
    D: UnstructuredData,
{
    fn dyn_object_project<'a>(&'a self, field: &str) -> Option<&'a dyn DynUnstructuredData> {
        Some(D::object_project(self, field)?)
    }

    fn dyn_array_project(&self, index: usize) -> Option<&dyn DynUnstructuredData> {
        Some(D::array_project(self, index)?)
    }
}

cfg_with_docs!(
    feature = "json",
    {
        impl<'borrow> UnstructuredDataLifetime<'borrow> for serde_json::Value {
            type ArrayIter = core::slice::Iter<'borrow, Self>;
            type ArrayIterMut = core::slice::IterMut<'borrow, Self>;
        }
    },
    {
        impl UnstructuredData for serde_json::Value {
            type ArrayIntoIter = alloc::vec::IntoIter<Self>;
            type DeserializationError = serde_json::Error;

            #[inline]
            fn deserialize_into<T>(this: Self) -> Result<T, Self::DeserializationError>
            where
                T: DeserializeOwned,
            {
                serde_json::from_value(this)
            }
            #[inline]
            fn deserialize_borrowed<T>(this: &Self) -> Result<T, Self::DeserializationError>
            where
                T: DeserializeOwned,
            {
                Self::deserialize_into(this.clone())
            }

            fn multi_project_mut<'a, R>(this: &'a mut Self, mut req: R)
            where
                R: MultiProjectionRequest<&'a mut Self>,
            {
                match this {
                    serde_json::Value::Array(array) => {
                        let mut array = array.as_mut_slice();
                        let mut removed_len = 0;

                        for req_i in 0..req.len() {
                            if let ProjectionPathSegment::Index(index) = req.get(req_i) {
                                if index >= array.len() + removed_len {
                                    // Indexes in req is sorted so no remaining
                                    // requests will have valid indexes:
                                    break;
                                }
                                if index < removed_len {
                                    panic!(
                                        "`MultiProjectionRequest` specified indexes that weren't \
                                            sorted, the index {too_large} was sorted before index {next}.",
                                        next = index,
                                        too_large = removed_len - 1,
                                    );
                                }
                                let (first, rest) = array.split_at_mut(index + 1 - removed_len);
                                array = rest;
                                removed_len = index + 1;
                                req.set(req_i, first.last_mut().unwrap());
                            }
                        }
                    }
                    serde_json::Value::Object(object) => {
                        for (key, value) in object.iter_mut() {
                            for i in 0..req.len() {
                                if matches!(req.get(i),  ProjectionPathSegment::Field(field) if *field == key)
                                {
                                    req.set(i, value);
                                    break;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            fn multi_project_into<R>(this: Self, mut req: R)
            where
                R: MultiProjectionRequest<Self>,
            {
                match this {
                    serde_json::Value::Array(mut array) => {
                        let mut unchanged_len = array.len();
                        let original_len = array.len();

                        for req_i in (0..req.len()).rev() {
                            if let ProjectionPathSegment::Index(index) = req.get(req_i) {
                                if index >= original_len {
                                    // Index doesn't exist.
                                    continue;
                                }
                                if index >= unchanged_len {
                                    panic!(
                                        "`MultiProjectionRequest` specified indexes that weren't \
                                            sorted, the index {too_large} was sorted before index {next}.",
                                        too_large = index,
                                        next = unchanged_len,
                                    );
                                }
                                req.set(req_i, array.swap_remove(index));
                                unchanged_len = index;
                            }
                        }
                    }
                    serde_json::Value::Object(mut object) => {
                        for i in 0..req.len() {
                            if let ProjectionPathSegment::Field(field) = req.get(i) {
                                if let Some(value) = object.remove(*field) {
                                    req.set(i, value);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            #[inline]
            fn is_object(this: &Self) -> bool {
                this.is_object()
            }

            #[inline]
            fn object_project<'a>(this: &'a Self, field: &str) -> Option<&'a Self> {
                this.as_object()?.get(field)
            }
            #[inline]
            fn object_project_mut<'a>(this: &'a mut Self, field: &str) -> Option<&'a mut Self> {
                this.as_object_mut()?.get_mut(field)
            }
            #[inline]
            fn object_project_into(this: Self, field: &str) -> Option<Self> {
                if let serde_json::Value::Object(mut object) = this {
                    object.remove(field)
                } else {
                    None
                }
            }

            #[inline]
            fn is_array(this: &Self) -> bool {
                this.is_array()
            }
            #[inline]
            fn array_retain<F>(this: &mut Self, f: F)
            where
                F: FnMut(&mut Self) -> bool,
            {
                if let Some(array) = this.as_array_mut() {
                    array.retain_mut(f);
                }
            }

            #[inline]
            fn array_project(this: &Self, index: usize) -> Option<&Self> {
                this.as_array()?.get(index)
            }
            #[inline]
            fn array_project_mut(this: &mut Self, index: usize) -> Option<&mut Self> {
                this.as_array_mut()?.get_mut(index)
            }
            #[inline]
            fn array_project_into(this: Self, index: usize) -> Option<Self> {
                if let serde_json::Value::Array(mut array) = this {
                    if index >= array.len() {
                        return None;
                    }
                    Some(array.swap_remove(index))
                } else {
                    None
                }
            }

            fn array_iter(
                this: &Self,
            ) -> Option<<Self as UnstructuredDataLifetime<'_>>::ArrayIter> {
                this.as_array().map(|array| array.iter())
            }
            fn array_iter_mut(
                this: &mut Self,
            ) -> Option<<Self as UnstructuredDataLifetime<'_>>::ArrayIterMut> {
                this.as_array_mut().map(|array| array.iter_mut())
            }
            fn array_into_iter(this: Self) -> Option<Self::ArrayIntoIter> {
                if let serde_json::Value::Array(array) = this {
                    Some(array.into_iter())
                } else {
                    None
                }
            }
        }
    },
    {
        impl<View, Tracker> WrapInSerdeView<View, Tracker> for serde_json::Value
        where
            Tracker: PathTracker,
        {
            type Wrapper = SerdeViewTyped<Self, View, Tracker>;
            #[inline]
            fn create(data: Self, tracker: Tracker) -> Self::Wrapper {
                SerdeViewTyped::new(data, tracker)
            }
        }
    },
    {
        impl<'data, View, Tracker> WrapInSerdeView<View, Tracker> for &'data serde_json::Value
        where
            Tracker: PathTracker,
        {
            type Wrapper = SerdeViewTypedRef<'data, serde_json::Value, View, Tracker>;
            #[inline]
            fn create(data: Self, tracker: Tracker) -> Self::Wrapper {
                SerdeViewTypedRef::new(data, tracker)
            }
        }
    },
    {
        impl<'data, View, Tracker> WrapInSerdeView<View, Tracker> for &'data mut serde_json::Value
        where
            Tracker: PathTracker,
        {
            type Wrapper = SerdeViewTypedMut<'data, serde_json::Value, View, Tracker>;
            #[inline]
            fn create(data: Self, tracker: Tracker) -> Self::Wrapper {
                SerdeViewTypedMut::new(data, tracker)
            }
        }
    }
);

////////////////////////////////////////////////////////////////////////////////
// Error type
////////////////////////////////////////////////////////////////////////////////

/// This error indicates that a value was't an array.
#[derive(Debug, Clone, Copy, Default)]
pub struct ValueNotArrayError;
impl fmt::Display for ValueNotArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value not an array")
    }
}
cfg_with_docs!(feature = "std", {
    impl Error for ValueNotArrayError {}
});

/// This error indicates that a wanted value couldn't be found.
#[derive(Debug, Clone, Copy, Default)]
pub struct ValueNotFoundError;
impl fmt::Display for ValueNotFoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value not found")
    }
}
cfg_with_docs!(feature = "std", {
    impl Error for ValueNotFoundError {}
});

/// Combines an error with a [`PathTracker`] to give better error messages.
#[derive(Debug, Clone, Copy)]
pub struct SerdeViewError<E, T> {
    error: E,
    path: T,
}
impl<E, T> SerdeViewError<E, T> {
    pub fn new(error: E, path: T) -> Self {
        Self { error, path }
    }
    pub fn into_inner(self) -> (E, T) {
        (self.error, self.path)
    }
    pub fn error(&self) -> &E {
        &self.error
    }
    pub fn path(&self) -> &T {
        &self.path
    }

    /// Ensure path tracker is owned and doesn't cause lifetime issues.
    #[inline]
    pub fn with_owned_tracker(self) -> SerdeViewError<E, <T as PathTracker>::OwnedTracker>
    where
        T: PathTracker,
    {
        SerdeViewError::new(self.error, PathTracker::into_owned(self.path))
    }
}
impl<E, T> fmt::Display for SerdeViewError<E, T>
where
    E: fmt::Display,
    T: PathTracker,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid unstructured data")?;
        if let Some(display) = PathTracker::display(&self.path) {
            write!(f, " at: {}", display)?;
        }
        #[cfg(not(feature = "std"))]
        {
            // Error trait doesn't provide source so we add extra info manually:
            write!(f, " caused by: {}", self.error)?;
        }
        Ok(())
    }
}
cfg_with_docs!(feature = "std", {
    impl<E, T> Error for SerdeViewError<E, T>
    where
        E: Error + 'static,
        T: PathTracker + fmt::Debug,
    {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            Some(&self.error)
        }
    }
});
