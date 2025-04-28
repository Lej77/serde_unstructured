//! Hooks for projection API.
//!
//! This can be private, similar to how `std` keeps its `Pattern` trait
//! private. The drawback of this is that downstream code can't be generic
//! over projection types, but that isn't the common usage of the crate.
use super::*;

////////////////////////////////////////////////////////////////////////////////
// Custom Path Trackers
////////////////////////////////////////////////////////////////////////////////

macro_rules! either {
    ($this:ident, |$val:ident| $body:expr) => {
        match $this {
            EitherPathTracker::Left($val) => EitherPathTracker::Left($body),
            EitherPathTracker::Right($val) => EitherPathTracker::Right($body),
        }
    };
}
macro_rules! unwrap_either {
    ($this:ident, |$val:ident| $body:expr) => {
        match $this {
            EitherPathTracker::Left($val) => $body,
            EitherPathTracker::Right($val) => $body,
        }
    };
}

#[derive(Clone, Debug)]
pub enum EitherPathTracker<L, R> {
    Left(L),
    Right(R),
}
impl<'borrow, L, R> PathTrackerLifetime<'borrow> for EitherPathTracker<L, R>
where
    L: PathTrackerLifetime<'borrow>,
    R: PathTrackerLifetime<'borrow>,
{
    type LinkedTracker = EitherPathTracker<
        <L as PathTrackerLifetime<'borrow>>::LinkedTracker,
        <R as PathTrackerLifetime<'borrow>>::LinkedTracker,
    >;
    type SegmentIter = EitherPathTracker<
        <L as PathTrackerLifetime<'borrow>>::SegmentIter,
        <R as PathTrackerLifetime<'borrow>>::SegmentIter,
    >;
}
impl<L, R> PathTracker for EitherPathTracker<L, R>
where
    L: PathTracker,
    R: PathTracker,
{
    fn linked(previous: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker {
        either!(previous, |previous| PathTracker::linked(previous))
    }

    type OwnedTracker = EitherPathTracker<L::OwnedTracker, R::OwnedTracker>;
    fn into_owned(previous: Self) -> Self::OwnedTracker {
        either!(previous, |previous| PathTracker::into_owned(previous))
    }

    type StaticProjectionTracker =
        EitherPathTracker<L::StaticProjectionTracker, R::StaticProjectionTracker>;
    fn with_static_projection(
        previous: Self,
        projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker {
        either!(previous, |previous| PathTracker::with_static_projection(
            previous, projection
        ))
    }

    type StaticStrFieldTracker =
        EitherPathTracker<L::StaticStrFieldTracker, R::StaticStrFieldTracker>;
    fn with_field(previous: Self, field: &'static str) -> Self::StaticStrFieldTracker {
        either!(previous, |previous| PathTracker::with_field(
            previous, field
        ))
    }

    type IndexTracker = EitherPathTracker<L::IndexTracker, R::IndexTracker>;
    fn with_index(previous: Self, index: usize) -> Self::IndexTracker {
        either!(previous, |previous| PathTracker::with_index(
            previous, index
        ))
    }

    type OwnedFieldTracker = EitherPathTracker<L::OwnedFieldTracker, R::OwnedFieldTracker>;
    fn with_owned_field(previous: Self, field: OwnedField) -> Self::OwnedFieldTracker {
        either!(previous, |previous| PathTracker::with_owned_field(
            previous, field
        ))
    }

    fn segment_iter(tracker: &Self) -> <Self as PathTrackerLifetime<'_>>::SegmentIter {
        either!(tracker, |tracker| PathTracker::segment_iter(tracker))
    }

    fn is_empty(tracker: &Self) -> bool {
        unwrap_either!(tracker, |tracker| PathTracker::is_empty(tracker))
    }

    fn len(tracker: &Self) -> usize {
        unwrap_either!(tracker, |tracker| PathTracker::len(tracker))
    }

    fn format_path<'segment, I>(
        tracker: &Self,
        iterator: I,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    where
        I: Iterator<Item = ProjectionPathSegment<'segment>>,
    {
        unwrap_either!(tracker, |tracker| PathTracker::format_path(
            tracker, iterator, f
        ))
    }
}
impl<L, R, I> Iterator for EitherPathTracker<L, R>
where
    L: Iterator<Item = I>,
    R: Iterator<Item = I>,
{
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        unwrap_either!(self, |v| v.next())
    }
}

/// Used by [`PartialPathTracker`] to filter what paths should be displayed.
#[derive(Debug, Clone)]
pub struct PartialPathTrackerIter<T> {
    inner: T,
    /// Number of path segments from the start of an iterator before some
    /// segments should be skipped.
    skip_starts_after: usize,
    /// The number of path segments that should be skipped.
    skip_len: usize,
}
impl<T> Iterator for PartialPathTrackerIter<T>
where
    T: Iterator,
{
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.skip_len == 0 {
            return self.inner.next();
        }
        match self.skip_starts_after.checked_sub(1) {
            Some(updated) => {
                self.skip_starts_after = updated;
                self.inner.next()
            }
            None => {
                let n = self.skip_len;
                self.skip_len = 0;
                self.inner.nth(n)
            }
        }
    }
}

/// A [`PathTracker`] that uses some path segments from another tracker.
#[derive(Debug, Clone)]
pub struct PartialPathTracker<T> {
    tracker: T,
    /// Number of path segments from the start of an iterator before some
    /// segments should be skipped.
    skip_starts_after: usize,
    /// The number of path segments that should be skipped.
    skip_len: usize,
}
impl<T> PartialPathTracker<T> {
    pub fn new(tracker: T, skip_start_after: usize, skip_len: usize) -> Self {
        Self {
            tracker,
            skip_starts_after: skip_start_after,
            skip_len,
        }
    }
    fn map_tracker<F, R>(self, f: F) -> PartialPathTracker<R>
    where
        F: FnOnce(T) -> R,
    {
        let PartialPathTracker {
            tracker,
            skip_starts_after,
            skip_len,
        } = self;
        PartialPathTracker {
            tracker: f(tracker),
            skip_starts_after,
            skip_len,
        }
    }
}
impl<'borrow, T> PathTrackerLifetime<'borrow> for PartialPathTracker<T>
where
    T: PathTracker,
{
    type LinkedTracker = PartialPathTracker<<T as PathTrackerLifetime<'borrow>>::LinkedTracker>;
    type SegmentIter = PartialPathTrackerIter<<T as PathTrackerLifetime<'borrow>>::SegmentIter>;
}
impl<T> PathTracker for PartialPathTracker<T>
where
    T: PathTracker,
{
    fn linked(previous: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker {
        PartialPathTracker::new(
            PathTracker::linked(&previous.tracker),
            previous.skip_starts_after,
            previous.skip_len,
        )
    }

    type OwnedTracker = PartialPathTracker<T::OwnedTracker>;
    fn into_owned(previous: Self) -> Self::OwnedTracker {
        previous.map_tracker(PathTracker::into_owned)
    }

    type StaticProjectionTracker = PartialPathTracker<T::StaticProjectionTracker>;
    fn with_static_projection(
        previous: Self,
        projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker {
        previous.map_tracker(|previous| PathTracker::with_static_projection(previous, projection))
    }

    type StaticStrFieldTracker = PartialPathTracker<T::StaticStrFieldTracker>;
    fn with_field(previous: Self, field: &'static str) -> Self::StaticStrFieldTracker {
        previous.map_tracker(|previous| PathTracker::with_field(previous, field))
    }

    type IndexTracker = PartialPathTracker<T::IndexTracker>;
    fn with_index(previous: Self, index: usize) -> Self::IndexTracker {
        previous.map_tracker(|previous| PathTracker::with_index(previous, index))
    }

    type OwnedFieldTracker = PartialPathTracker<T::OwnedFieldTracker>;
    fn with_owned_field(previous: Self, field: OwnedField) -> Self::OwnedFieldTracker {
        previous.map_tracker(|previous| PathTracker::with_owned_field(previous, field))
    }

    fn segment_iter(tracker: &Self) -> <Self as PathTrackerLifetime<'_>>::SegmentIter {
        PartialPathTrackerIter {
            inner: PathTracker::segment_iter(&tracker.tracker),
            skip_starts_after: tracker.skip_starts_after,
            skip_len: tracker.skip_len,
        }
    }

    fn is_empty(tracker: &Self) -> bool {
        if tracker.skip_starts_after == 0 {
            PathTracker::len(&tracker.tracker) <= tracker.skip_len
        } else {
            // Will keep some segments from the start of the wrapped path:
            PathTracker::is_empty(&tracker.tracker)
        }
    }

    fn len(tracker: &Self) -> usize {
        // TODO: optimize the counting here:
        Self::segment_iter(tracker).count()
    }

    fn format_path<'segment, I>(
        tracker: &Self,
        iterator: I,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    where
        I: Iterator<Item = ProjectionPathSegment<'segment>>,
    {
        T::format_path(&tracker.tracker, iterator, f)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Acceptable paths
////////////////////////////////////////////////////////////////////////////////

/// A path for projection into unstructured data.
///
/// Types that implement this trait should be stored as paths inside
/// [`ProjectionTarget`].
///
/// # Type parameters
///
/// - `TExpectedType` represents the type that the unstructured data is
///   expected to be deserializable into.
/// - `TOriginalPathTracker` is the type of the unmapped path tracker.
pub trait ProjectionPath<TExpectedType, TOriginalPathTracker> {
    /// Path tracker after the projection path has been appended.
    type PathTracker;

    type PathStorage;

    /// Indicates that this path is empty. If `false` then `len` must return
    /// a non-zero value.
    const IS_EMPTY: bool;

    fn create_path_storage(&self) -> Self::PathStorage;

    /// The segment of the path at the specified index.
    fn get(
        &self,
        index: usize,
        data: &dyn DynUnstructuredData,
        storage: &mut Self::PathStorage,
    ) -> Option<ProjectionPathSegment<'_>>;
    /// The total number of path segments for this projection.
    fn len(&self) -> usize;

    fn try_fold<TAcc, TError, TF, TBorrow>(
        &self,
        init: TAcc,
        f: TF,
        borrow_data_f: TBorrow,
        storage: &mut Self::PathStorage,
    ) -> Result<TAcc, TError>
    where
        TF: FnMut(TAcc, ProjectionPathSegment<'_>) -> Result<TAcc, TError>,
        TBorrow: FnMut(&TAcc) -> &dyn DynUnstructuredData;

    fn changed_path_tracker(
        &self,
        original: TOriginalPathTracker,
        storage: &Self::PathStorage,
    ) -> Self::PathTracker;
}
/// Empty path, used when constructing initial [`ProjectionTarget`].
impl<TExpectedType, TOriginalPathTracker> ProjectionPath<TExpectedType, TOriginalPathTracker>
    for ()
{
    type PathTracker = TOriginalPathTracker;
    type PathStorage = ();

    const IS_EMPTY: bool = true;

    fn create_path_storage(&self) -> Self::PathStorage {}

    fn get(
        &self,
        _: usize,
        _data: &dyn DynUnstructuredData,
        _storage: &mut (),
    ) -> Option<ProjectionPathSegment<'_>> {
        None
    }

    fn len(&self) -> usize {
        0
    }

    fn try_fold<TAcc, TError, TF, TBorrow>(
        &self,
        init: TAcc,
        _f: TF,
        _borrow_data_f: TBorrow,
        _storage: &mut (),
    ) -> Result<TAcc, TError>
    where
        TF: FnMut(TAcc, ProjectionPathSegment<'_>) -> Result<TAcc, TError>,
        TBorrow: FnMut(&TAcc) -> &dyn DynUnstructuredData,
    {
        Ok(init)
    }

    fn changed_path_tracker(
        &self,
        original: TOriginalPathTracker,
        _storage: &(),
    ) -> Self::PathTracker {
        original
    }
}
/// Type list of paths, used for all projections (since the first projection is `()`).
impl<A, B, TExpectedType, TOriginalPathTracker> ProjectionPath<TExpectedType, TOriginalPathTracker>
    for (A, B)
where
    A: ProjectionPath<TExpectedType, TOriginalPathTracker>,
    B: ProjectionPath<TExpectedType, A::PathTracker>,
{
    type PathTracker = <B as ProjectionPath<
        TExpectedType,
        <A as ProjectionPath<TExpectedType, TOriginalPathTracker>>::PathTracker,
    >>::PathTracker;
    type PathStorage = (A::PathStorage, B::PathStorage);

    const IS_EMPTY: bool = A::IS_EMPTY && B::IS_EMPTY;

    fn create_path_storage(&self) -> Self::PathStorage {
        (self.0.create_path_storage(), self.1.create_path_storage())
    }

    fn get(
        &self,
        index: usize,
        data: &dyn DynUnstructuredData,
        storage: &mut Self::PathStorage,
    ) -> Option<ProjectionPathSegment<'_>> {
        let len = self.0.len();
        if index < len {
            self.0.get(index, data, &mut storage.0)
        } else {
            self.1.get(index - len, data, &mut storage.1)
        }
    }

    fn len(&self) -> usize {
        self.0.len() + self.1.len()
    }

    fn try_fold<TAcc, TError, TF, TBorrow>(
        &self,
        init: TAcc,
        mut f: TF,
        mut borrow_data_f: TBorrow,
        storage: &mut Self::PathStorage,
    ) -> Result<TAcc, TError>
    where
        TF: FnMut(TAcc, ProjectionPathSegment<'_>) -> Result<TAcc, TError>,
        TBorrow: FnMut(&TAcc) -> &dyn DynUnstructuredData,
    {
        let after_a = self
            .0
            .try_fold(init, &mut f, &mut borrow_data_f, &mut storage.0)?;
        self.1
            .try_fold(after_a, &mut f, &mut borrow_data_f, &mut storage.1)
    }

    fn changed_path_tracker(
        &self,
        original: TOriginalPathTracker,
        storage: &Self::PathStorage,
    ) -> Self::PathTracker {
        self.1.changed_path_tracker(
            self.0.changed_path_tracker(original, &storage.0),
            &storage.1,
        )
    }
}
/// Path specified using zero sized types, these are generated by the derive macro.
impl<T, TExpectedType, TOriginalPathTracker> ProjectionPath<TExpectedType, TOriginalPathTracker>
    for ZeroSizedProjectionPath<T>
where
    TOriginalPathTracker: PathTracker,
    T: GetZeroSizedProjectionPathSegment,
{
    type PathTracker = TOriginalPathTracker::StaticProjectionTracker;
    type PathStorage = ProjectionPathSegment<'static>;

    const IS_EMPTY: bool = false;

    fn create_path_storage(&self) -> Self::PathStorage {
        ProjectionPathSegment::Field(MaybeStatic::Static("<unknown>"))
    }

    fn get(
        &self,
        index: usize,
        data: &dyn DynUnstructuredData,
        storage: &mut Self::PathStorage,
    ) -> Option<ProjectionPathSegment<'_>> {
        if index == 0 {
            *storage = T::get_path(data);
            Some(*storage)
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        1
    }

    fn try_fold<TAcc, TError, TF, TBorrow>(
        &self,
        init: TAcc,
        mut f: TF,
        mut borrow_data_f: TBorrow,
        storage: &mut Self::PathStorage,
    ) -> Result<TAcc, TError>
    where
        TF: FnMut(TAcc, ProjectionPathSegment<'_>) -> Result<TAcc, TError>,
        TBorrow: FnMut(&TAcc) -> &dyn DynUnstructuredData,
    {
        let path = T::get_path(borrow_data_f(&init));
        *storage = path;
        f(init, path)
    }

    fn changed_path_tracker(
        &self,
        original: TOriginalPathTracker,
        storage: &Self::PathStorage,
    ) -> Self::PathTracker {
        PathTracker::with_static_projection(original, *storage)
    }
}
/// Use an index as a path segment.
impl<TExpectedType, TOriginalPathTracker> ProjectionPath<TExpectedType, TOriginalPathTracker>
    for usize
where
    TOriginalPathTracker: PathTracker,
{
    type PathTracker = TOriginalPathTracker::IndexTracker;
    type PathStorage = ();

    const IS_EMPTY: bool = false;

    fn create_path_storage(&self) -> Self::PathStorage {}

    fn get(
        &self,
        index: usize,
        _data: &dyn DynUnstructuredData,
        _storage: &mut (),
    ) -> Option<ProjectionPathSegment<'_>> {
        if index == 0 {
            Some(ProjectionPathSegment::Index(*self))
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        1
    }

    fn try_fold<TAcc, TError, TF, TBorrow>(
        &self,
        init: TAcc,
        mut f: TF,
        _borrow_data_f: TBorrow,
        _storage: &mut (),
    ) -> Result<TAcc, TError>
    where
        TF: FnMut(TAcc, ProjectionPathSegment<'_>) -> Result<TAcc, TError>,
        TBorrow: FnMut(&TAcc) -> &dyn DynUnstructuredData,
    {
        f(init, ProjectionPathSegment::Index(*self))
    }

    fn changed_path_tracker(
        &self,
        original: TOriginalPathTracker,
        _storage: &(),
    ) -> Self::PathTracker {
        PathTracker::with_index(original, *self)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Construct output types
////////////////////////////////////////////////////////////////////////////////

/// Get the type of an error when it contains a projection path.
pub trait ErrorFromPathTracker<TInnerError, TPath> {
    type WithPath;

    fn create(inner: TInnerError, path: TPath) -> Self::WithPath;
}
// TODO: offer alternative variants of `SerdeViewTyped::project` methods that
// return errors with path trackers that aren't owned.
impl<E, T> ErrorFromPathTracker<E, T> for SerdeViewError<(), ()>
where
    T: PathTracker,
{
    type WithPath = SerdeViewError<E, <T as PathTracker>::OwnedTracker>;

    fn create(inner: E, path: T) -> Self::WithPath {
        SerdeViewError::new(inner, PathTracker::into_owned(path))
    }
}

/// Get the type of a `SerdeViewTyped` wrapper.
pub trait WithWrapper<TData, TExpectedType, TPath> {
    type Wrapper;

    fn create(data: TData, path: TPath) -> Self::Wrapper;
}
impl<D, V, T> WithWrapper<D, V, T> for SerdeViewTyped<(), (), ()>
where
    D: UnstructuredData,
    T: PathTracker,
{
    type Wrapper = SerdeViewTyped<D, V, T>;

    fn create(data: D, path: T) -> Self::Wrapper {
        SerdeViewTyped::new(data, path)
    }
}
impl<'data, D: 'data, V, T> WithWrapper<&'data mut D, V, T> for SerdeViewTypedMut<'data, (), (), ()>
where
    D: UnstructuredData,
    T: PathTracker,
{
    type Wrapper = SerdeViewTypedMut<'data, D, V, T>;

    fn create(data: &'data mut D, path: T) -> Self::Wrapper {
        SerdeViewTypedMut::new(data, path)
    }
}
impl<'data, D: 'data, V, T> WithWrapper<&'data D, V, T> for SerdeViewTypedRef<'data, (), (), ()>
where
    D: UnstructuredData,
    T: PathTracker,
{
    type Wrapper = SerdeViewTypedRef<'data, D, V, T>;

    fn create(data: &'data D, path: T) -> Self::Wrapper {
        SerdeViewTypedRef::new(data, path)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Helpers for multi projection
////////////////////////////////////////////////////////////////////////////////

/// Data for a [`MultiProjectionLinkedList`] node.
struct MultiProjectionLinkedListSlot<'a, D> {
    path: ProjectionPathSegment<'a>,
    /// Uses interior mutability to allow setting this value.
    result: Cell<Option<D>>,
}

/// Implements [`MultiProjectionRequest`] using as few type parameters as
/// possible to prevent unnecessary monomorphization.
pub struct MultiProjectionLinkedList<'a, D> {
    slot: MultiProjectionLinkedListSlot<'a, D>,
    next: Option<&'a MultiProjectionLinkedList<'a, D>>,
}
impl<'a, D> MultiProjectionLinkedList<'a, D> {
    fn iter(&self) -> impl Iterator<Item = &'_ MultiProjectionLinkedListSlot<'a, D>> {
        let mut next = Some(self);
        iter::from_fn(move || {
            let this = next?;
            next = this.next;
            Some(&this.slot)
        })
    }
}
impl<D> MultiProjectionRequest<D> for &'_ MultiProjectionLinkedList<'_, D> {
    fn get(&self, index: usize) -> ProjectionPathSegment {
        self.iter().nth(index).unwrap().path
    }

    fn set(&mut self, index: usize, value: D) {
        self.iter().nth(index).unwrap().result.set(Some(value));
    }

    fn len(&self) -> usize {
        self.iter().count()
    }
}

/// Middle step for projection of [`ProjectionRequestList`].
///
/// - Allows handling errors in different ways.
///   - Error on first error.
///   - or project all errors.
/// - The main use case for this trait is to allow for multi projection
///   (projecting into multiple targets from the same source value).
pub trait MultiProjectionResult<T> {
    fn new() -> Self;
    fn fill(value: T) -> Self
    where
        T: Clone;

    fn set(&mut self, index: usize, value: T);
    fn len(&self) -> usize;

    fn has_unset(&self) -> bool;
}
impl<T> MultiProjectionResult<T> for Result<T, usize> {
    fn new() -> Self {
        Err(0)
    }
    fn fill(value: T) -> Self
    where
        T: Clone,
    {
        Ok(value)
    }
    fn set(&mut self, _index: usize, value: T) {
        *self = Ok(value);
    }

    fn len(&self) -> usize {
        1
    }

    fn has_unset(&self) -> bool {
        self.is_err()
    }
}
impl<A, B, T> MultiProjectionResult<T> for (A, B)
where
    A: MultiProjectionResult<T>,
    B: MultiProjectionResult<T>,
{
    fn new() -> Self {
        (A::new(), B::new())
    }
    fn fill(value: T) -> Self
    where
        T: Clone,
    {
        (A::fill(value.clone()), B::fill(value))
    }
    fn set(&mut self, index: usize, value: T) {
        let a_len = self.0.len();
        if index < a_len {
            self.0.set(index, value);
        } else {
            self.1.set(index - a_len, value);
        }
    }

    fn len(&self) -> usize {
        self.0.len() + self.1.len()
    }

    fn has_unset(&self) -> bool {
        self.0.has_unset() || self.1.has_unset()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Helper for single projection
////////////////////////////////////////////////////////////////////////////////

/// Implements [`DataProjector`].
pub struct ProjectOwned;
/// Implements [`DataProjector`].
pub struct ProjectBorrowed;
/// Implemented for marker types to allow projecting through
/// [`UnstructuredData`] when ownership differs.
trait DataProjector<D>: Sized {
    fn project_array(data: D, index: usize) -> Option<D>;
    fn project_object(data: D, field: &str) -> Option<D>;
    fn as_dyn(data: &D) -> &dyn DynUnstructuredData;
}
impl<D> DataProjector<D> for ProjectOwned
where
    D: UnstructuredData,
{
    fn project_array(data: D, index: usize) -> Option<D> {
        D::array_project_into(data, index)
    }

    fn project_object(data: D, field: &str) -> Option<D> {
        D::object_project_into(data, field)
    }

    fn as_dyn(data: &D) -> &dyn DynUnstructuredData {
        data
    }
}
impl<'data, D> DataProjector<&'data D> for ProjectBorrowed
where
    D: UnstructuredData,
{
    fn project_array(data: &'data D, index: usize) -> Option<&'data D> {
        D::array_project(data, index)
    }

    fn project_object(data: &'data D, field: &str) -> Option<&'data D> {
        D::object_project(data, field)
    }

    fn as_dyn<'a>(data: &'a &'data D) -> &'a dyn DynUnstructuredData {
        &**data
    }
}
impl<'data, D> DataProjector<&'data mut D> for ProjectBorrowed
where
    D: UnstructuredData,
{
    fn project_array(data: &'data mut D, index: usize) -> Option<&'data mut D> {
        D::array_project_mut(data, index)
    }

    fn project_object(data: &'data mut D, field: &str) -> Option<&'data mut D> {
        D::object_project_mut(data, field)
    }

    fn as_dyn<'a>(data: &'a &'data mut D) -> &'a dyn DynUnstructuredData {
        &**data
    }
}

////////////////////////////////////////////////////////////////////////////////
// Handle type lists of ProjectionTarget
////////////////////////////////////////////////////////////////////////////////

/// A type list of [`ProjectionTarget`] types. Several requests can be
/// specified to project into multiple fields.
///
/// This allows specifying multiple projections like `(p.a(), (p.b(),
/// p.foo()))`.
///
/// # Type parameters
///
/// - `TError` should implement [`ErrorFromPathTracker`] to specify the
///   error type.
/// - `TOriginalPathTracker` is the type of the unmapped path tracker.
#[allow(clippy::wrong_self_convention)]
pub trait ProjectionRequestList<TWrapper, TData, TDataProjector, TError, TOriginalPathTracker> {
    /// The number of projection requests in the list.
    const LENGTH: usize;
    /// The projection output where each projection might have failed.
    type Output;
    /// The projection output if all projections were successful.
    type UnwrappedOutput;
    /// Error with path tracker.
    type ErrorPath;

    /// Store result for multi field projection.
    type MultiProjectionResult: MultiProjectionResult<TData>;

    /// Store projection path as it is constructed (it might depend on the state
    /// of intermediate projection results).
    type PathStorage;

    fn create_path_storage(&self) -> Self::PathStorage;

    fn into_output(
        &self,
        result: Self::MultiProjectionResult,
        path: TOriginalPathTracker,
        path_storage: Self::PathStorage,
    ) -> Self::Output;
    /// Get output if all projections were successful.
    ///
    /// # Panics
    ///
    /// If an projection failed then this will panic.
    fn into_unwrapped_output(
        &self,
        result: Self::MultiProjectionResult,
        path: TOriginalPathTracker,
        path_storage: Self::PathStorage,
    ) -> Self::UnwrappedOutput;
    /// Get an error if an projection failed.
    fn into_output_error_path(
        &self,
        result: &Self::MultiProjectionResult,
        path: TOriginalPathTracker,
        path_storage: &Self::PathStorage,
    ) -> Result<TOriginalPathTracker, Self::ErrorPath>;

    /// Simpler projection that doesn't support projecting into multiple
    /// fields from the same source value.
    fn single_project(
        &self,
        result: &mut Self::MultiProjectionResult,
        path_storage: &mut Self::PathStorage,
    );

    /// Build a linked list with a node for each request in the list.
    fn request<F, R>(
        &self,
        next: Option<&MultiProjectionLinkedList<'_, TData>>,
        data: TData,
        path_storage: &mut Self::PathStorage,
        f: F,
    ) -> (R, Self::MultiProjectionResult)
    where
        F: FnOnce(Option<&'_ MultiProjectionLinkedList<'_, TData>>, TData) -> R;
}
impl<TWrapper, TData, TDataProjector, TError, TPath, TExpectedType, TIdent>
    ProjectionRequestList<TWrapper, TData, TDataProjector, TError, TPath>
    for ProjectionTarget<TExpectedType, TIdent>
where
    TDataProjector: DataProjector<TData>,
    TIdent: ProjectionPath<TExpectedType, TPath>,
    TPath: PathTracker,
    TError: ErrorFromPathTracker<ValueNotFoundError, PartialPathTracker<TIdent::PathTracker>>,
    TWrapper: WithWrapper<TData, TExpectedType, TIdent::PathTracker>,
{
    const LENGTH: usize = if TIdent::IS_EMPTY { 0 } else { 1 };
    type Output = Result<
        TWrapper::Wrapper,
        <TError as ErrorFromPathTracker<
            ValueNotFoundError,
            PartialPathTracker<TIdent::PathTracker>,
        >>::WithPath,
    >;
    type UnwrappedOutput = TWrapper::Wrapper;
    type ErrorPath = PartialPathTracker<TIdent::PathTracker>;

    type MultiProjectionResult = Result<TData, usize>;
    type PathStorage = TIdent::PathStorage;

    fn create_path_storage(&self) -> Self::PathStorage {
        self.path.create_path_storage()
    }

    fn into_output(
        &self,
        result: Self::MultiProjectionResult,
        path: TPath,
        path_storage: Self::PathStorage,
    ) -> Self::Output {
        match result {
            Ok(data) => Ok(TWrapper::create(
                data,
                TIdent::changed_path_tracker(&self.path, path, &path_storage),
            )),
            Err(index_where_projection_failed) => {
                let path = <Self as ProjectionRequestList<
                    TWrapper,
                    TData,
                    TDataProjector,
                    TError,
                    TPath,
                >>::into_output_error_path(
                    self,
                    &Err(index_where_projection_failed),
                    path,
                    &path_storage,
                )
                .err()
                .unwrap();

                Err(TError::create(ValueNotFoundError, path))
            }
        }
    }

    fn into_unwrapped_output(
        &self,
        result: Self::MultiProjectionResult,
        path: TPath,
        path_storage: Self::PathStorage,
    ) -> Self::UnwrappedOutput {
        if let Ok(data) = result {
            let final_path = TIdent::changed_path_tracker(&self.path, path, &path_storage);
            TWrapper::create(data, final_path)
        } else {
            panic!("unwrapped projection result when the projection failed");
        }
    }

    fn into_output_error_path(
        &self,
        result: &Self::MultiProjectionResult,
        path: TPath,
        path_storage: &Self::PathStorage,
    ) -> Result<TPath, Self::ErrorPath> {
        if let &Err(index_where_projection_failed) = result {
            let mut original_len = PathTracker::len(&path);
            let final_path = TIdent::changed_path_tracker(&self.path, path, path_storage);
            let mut extra_len = self.path.len();
            if extra_len > 0 {
                original_len += 1;
                extra_len -= 1;
            }
            Err(PartialPathTracker::new(
                final_path,
                original_len + index_where_projection_failed,
                extra_len.saturating_sub(index_where_projection_failed),
            ))
        } else {
            Ok(path)
        }
    }

    fn single_project(
        &self,
        result: &mut Self::MultiProjectionResult,
        path_storage: &mut Self::PathStorage,
    ) {
        if result.is_err() {
            return;
        }
        let data = core::mem::replace(result, Err(0)).ok().unwrap();

        let mut i = 0;
        *result = TIdent::try_fold(
            &self.path,
            data,
            |data, path| {
                let value = match path {
                    ProjectionPathSegment::Index(index) => {
                        TDataProjector::project_array(data, index)
                    }
                    ProjectionPathSegment::Field(field) => {
                        TDataProjector::project_object(data, &field)
                    }
                }
                .ok_or(i);

                i += 1;
                value
            },
            TDataProjector::as_dyn,
            path_storage,
        )
    }

    fn request<F, R>(
        &self,
        next: Option<&MultiProjectionLinkedList<'_, TData>>,
        data: TData,
        path_storage: &mut Self::PathStorage,
        f: F,
    ) -> (R, Self::MultiProjectionResult)
    where
        F: FnOnce(Option<&'_ MultiProjectionLinkedList<'_, TData>>, TData) -> R,
    {
        if let Some(path) = TIdent::get(&self.path, 0, TDataProjector::as_dyn(&data), path_storage)
        {
            let node = MultiProjectionLinkedList {
                slot: MultiProjectionLinkedListSlot {
                    path,
                    result: Cell::new(None),
                },
                next,
            };
            let value = f(Some(&node), data);

            let result = match node.slot.result.into_inner() {
                Some(data) => {
                    // For projections with only one segment this should
                    // hopefully inline to `Ok(data)`:
                    if TIdent::len(&self.path) > 1 {
                        let mut i = 0;
                        TIdent::try_fold(
                            &self.path,
                            data,
                            |data, path| {
                                let value = if i == 0 {
                                    Ok(data)
                                } else {
                                    match path {
                                        ProjectionPathSegment::Index(index) => {
                                            TDataProjector::project_array(data, index)
                                        }
                                        ProjectionPathSegment::Field(field) => {
                                            TDataProjector::project_object(data, &field)
                                        }
                                    }
                                    .ok_or(i)
                                };

                                i += 1;
                                value
                            },
                            TDataProjector::as_dyn,
                            path_storage,
                        )
                    } else {
                        Ok(data)
                    }
                }
                None => Err(0),
            };

            (value, result)
        } else {
            // Empty path (so no projection request should be made):
            let value = f(None, data);
            (value, Err(0))
        }
    }
}
/// Implementation for type lists.
impl<A, B, TWrapper, TData, TDataProjector, TError, TPath>
    ProjectionRequestList<TWrapper, TData, TDataProjector, TError, TPath> for (A, B)
where
    TPath: PathTracker,
    A: ProjectionRequestList<TWrapper, TData, TDataProjector, TError, TPath>,
    B: ProjectionRequestList<TWrapper, TData, TDataProjector, TError, TPath>,
{
    const LENGTH: usize = A::LENGTH + B::LENGTH;
    type Output = (A::Output, B::Output);
    type UnwrappedOutput = (A::UnwrappedOutput, B::UnwrappedOutput);
    type ErrorPath = EitherPathTracker<A::ErrorPath, B::ErrorPath>;

    type MultiProjectionResult = (A::MultiProjectionResult, B::MultiProjectionResult);
    type PathStorage = (A::PathStorage, B::PathStorage);

    fn create_path_storage(&self) -> Self::PathStorage {
        (self.0.create_path_storage(), self.1.create_path_storage())
    }

    fn into_output(
        &self,
        result: Self::MultiProjectionResult,
        path: TPath,
        path_storage: Self::PathStorage,
    ) -> Self::Output {
        (
            self.0.into_output(result.0, path.clone(), path_storage.0),
            self.1.into_output(result.1, path, path_storage.1),
        )
    }

    fn into_unwrapped_output(
        &self,
        result: Self::MultiProjectionResult,
        path: TPath,
        path_storage: Self::PathStorage,
    ) -> Self::UnwrappedOutput {
        (
            self.0
                .into_unwrapped_output(result.0, path.clone(), path_storage.0),
            self.1.into_unwrapped_output(result.1, path, path_storage.1),
        )
    }

    fn into_output_error_path(
        &self,
        result: &Self::MultiProjectionResult,
        path: TPath,
        path_storage: &Self::PathStorage,
    ) -> Result<TPath, Self::ErrorPath> {
        if result.0.has_unset() {
            self.0
                .into_output_error_path(&result.0, path, &path_storage.0)
                .map_err(EitherPathTracker::Left)
        } else if result.1.has_unset() {
            self.1
                .into_output_error_path(&result.1, path, &path_storage.1)
                .map_err(EitherPathTracker::Right)
        } else {
            Ok(path)
        }
    }

    fn single_project(
        &self,
        result: &mut Self::MultiProjectionResult,
        path_storage: &mut Self::PathStorage,
    ) {
        self.0.single_project(&mut result.0, &mut path_storage.0);
        self.1.single_project(&mut result.1, &mut path_storage.1);
    }

    fn request<F, R>(
        &self,
        next: Option<&MultiProjectionLinkedList<'_, TData>>,
        data: TData,
        path_storage: &mut Self::PathStorage,
        f: F,
    ) -> (R, Self::MultiProjectionResult)
    where
        F: FnOnce(Option<&'_ MultiProjectionLinkedList<'_, TData>>, TData) -> R,
    {
        let (a, b) = self;
        let (storage_a, storage_b) = path_storage;
        let ((value, a_output), b_output) =
            b.request(next, data, storage_b, move |node, data| {
                a.request(node, data, storage_a, f)
            });
        (value, (a_output, b_output))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Utility methods that do most of the projection work.
////////////////////////////////////////////////////////////////////////////////

/// Project into a [`ProjectionRequestList`].
pub(crate) fn project_into_list_via_clone<TWrapper, TData, TPath, TDataProjector, TReq>(
    target_list: &TReq,
    data: TData,
) -> (TReq::MultiProjectionResult, TReq::PathStorage)
where
    // Could take Clone but we Copy as a lint to make sure we don't accidentally use this.
    TData: Copy,
    TReq: ProjectionRequestList<TWrapper, TData, TDataProjector, SerdeViewError<(), ()>, TPath>,
{
    let mut storage = target_list.create_path_storage();
    let result = if TReq::LENGTH == 0 {
        TReq::MultiProjectionResult::new()
    } else {
        let mut result = TReq::MultiProjectionResult::fill(data);
        TReq::single_project(target_list, &mut result, &mut storage);
        result
    };
    (result, storage)
}

/// Project into a [`ProjectionRequestList`].
pub(crate) fn project_into_list<TWrapper, TData, TPath, TDataProjector, TMultiProjector, TReq>(
    target_list: &TReq,
    data: TData,
    multi_project: TMultiProjector,
) -> (TReq::MultiProjectionResult, TReq::PathStorage)
where
    TReq: ProjectionRequestList<TWrapper, TData, TDataProjector, SerdeViewError<(), ()>, TPath>,
    TMultiProjector: FnOnce(TData, &'_ MultiProjectionLinkedList<'_, TData>),
{
    let mut storage = target_list.create_path_storage();
    let result = if TReq::LENGTH == 0 {
        TReq::MultiProjectionResult::new()
    } else if TReq::LENGTH == 1 {
        let mut result = TReq::MultiProjectionResult::new();
        MultiProjectionResult::set(&mut result, 0, data);
        TReq::single_project(target_list, &mut result, &mut storage);
        result
    } else {
        let (_, projection_result) = TReq::request(target_list, None, data, &mut storage, move |node, data| {
            multi_project(data, node.unwrap());
        });
        projection_result
    };
    (result, storage)
}
/// Returns an error if not all projections in a [`ProjectionRequestList`]
/// succeeded.
pub(crate) fn try_unwrap_projection_result<TWrapper, TData, TPath, TDataProjector, TReq>(
    target_list: &TReq,
    result: TReq::MultiProjectionResult,
    storage: TReq::PathStorage,
    path: TPath,
) -> Result<TReq::UnwrappedOutput, SerdeViewError<ValueNotFoundError, TReq::ErrorPath>>
where
    TReq: ProjectionRequestList<TWrapper, TData, TDataProjector, SerdeViewError<(), ()>, TPath>,
{
    if result.has_unset() {
        let error_path = TReq::into_output_error_path(target_list, &result, path, &storage)
            .err()
            .unwrap();
        Err(SerdeViewError::new(ValueNotFoundError, error_path))
    } else {
        Ok(TReq::into_unwrapped_output(target_list, result, path, storage))
    }
}
