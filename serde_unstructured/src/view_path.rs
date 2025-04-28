//! Tracks the path to a certain serde view wrapper. This allows for good
//! error messages in case the unstructured data doesn't follow the expected
//! structure.

#[cfg(feature = "alloc")]
use alloc::{borrow::Cow, vec::Vec};
use core::{fmt, marker::PhantomData};

use crate::{MaybeStatic, OwnedField, PathTracker, PathTrackerLifetime, ProjectionPathSegment};

////////////////////////////////////////////////////////////////////////////////
// Implement Path Tracker
////////////////////////////////////////////////////////////////////////////////

fn format_path_tracker<'segment, I>(mut iterator: I, f: &mut fmt::Formatter<'_>) -> fmt::Result
where
    I: Iterator<Item = ProjectionPathSegment<'segment>>,
{
    // Use dynamic dispatch to minimize monomorphization (this seems like a
    // sensible default).
    ProjectionPathSegment::format_path::<&mut dyn Iterator<Item = ProjectionPathSegment<'segment>>>(
        &mut iterator,
        f,
    )
}

/// Shared implementation of [`PathTrackerLifetime`].
macro_rules! common_lifetime {
    () => {
        type SegmentIter = ProjectionPathSegmentIter<'borrow>;
    };
}
/// Shared implementation of [`PathTracker`].
macro_rules! common_path_tracker {
    () => {
        fn segment_iter(
            tracker: &Self,
        ) -> <Self as PathTrackerLifetime<'_>>::SegmentIter {
            ProjectionPathSegmentIter::new(tracker.iter())
        }

        fn format_path<'segment, I>(
            _tracker: &Self,
            iterator: I,
            f: &mut fmt::Formatter<'_>,
        ) -> fmt::Result
        where
            I: Iterator<Item = ProjectionPathSegment<'segment>>,
        {
            format_path_tracker(iterator, f)
        }
    };
}
/// Shared implementation of [`PathTracker`].
macro_rules! common_path_tracker_len {
    () => {
        fn is_empty(tracker: &Self) -> bool {
            tracker.is_empty()
        }

        fn len(tracker: &Self) -> usize {
            tracker.len()
        }
    };
}

pub struct ProjectionPathSegmentIter<'a> {
    inner: ViewPathIter<'a>,
}
impl<'a> ProjectionPathSegmentIter<'a> {
    pub fn new(inner: ViewPathIter<'a>) -> Self {
        Self { inner }
    }
}
impl<'a> Iterator for ProjectionPathSegmentIter<'a> {
    type Item = ProjectionPathSegment<'a>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(Into::into)
    }
}

impl<'borrow, L> PathTrackerLifetime<'borrow>
    for ViewPath<'_, StorageConfig<L, MaybeOwnedStorage>>
where
    L: LinkStorage,
{
    type LinkedTracker = ViewPath<'borrow, StorageConfig<HasLink<true>, EmptyStorage>>;
    common_lifetime!();
}
impl<'path, L> PathTracker for ViewPath<'path, StorageConfig<L, MaybeOwnedStorage>>
where
    L: LinkStorage,
{
    fn linked(previous: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker {
        previous.as_ref().as_linked_path()
    }

    type OwnedTracker = ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>>;
    fn into_owned(previous: Self) -> Self::OwnedTracker {
        previous.into_owned()
    }

    type StaticProjectionTracker = ViewPath<'path, StorageConfig<L, MaybeOwnedStorage>>;
    fn with_static_projection(
        previous: Self,
        projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker {
        let segment = match projection {
            ProjectionPathSegment::Field(field) => ViewPathSegment::field(*field),
            ProjectionPathSegment::Index(index) => ViewPathSegment::index(index),
        };
        previous.with_pushed_segment(segment)
    }

    type StaticStrFieldTracker = ViewPath<'path, StorageConfig<L, MaybeOwnedStorage>>;
    fn with_field(previous: Self, field: &'static str) -> Self::StaticStrFieldTracker {
        let segment = ViewPathSegment::field(field);
        previous.with_pushed_segment(segment)
    }

    type IndexTracker = ViewPath<'path, StorageConfig<L, MaybeOwnedStorage>>;
    fn with_index(previous: Self, index: usize) -> Self::IndexTracker {
        let segment = ViewPathSegment::index(index);
        previous.with_pushed_segment(segment)
    }

    type OwnedFieldTracker = ViewPath<'path, StorageConfig<L, MaybeOwnedStorage>>;
    fn with_owned_field(previous: Self, field: OwnedField) -> Self::OwnedFieldTracker {
        let segment = ViewPathSegment::dynamic_field(field);
        previous.with_pushed_segment(segment)
    }

    common_path_tracker!();
    common_path_tracker_len!();
}

impl<'borrow> PathTrackerLifetime<'borrow>
    for ViewPath<'_, StorageConfig<HasLink<false>, EmptyStorage>>
{
    type LinkedTracker = ViewPath<'static, StorageConfig<HasLink<false>, EmptyStorage>>;
    common_lifetime!();
}
impl PathTracker for ViewPath<'_, StorageConfig<HasLink<false>, EmptyStorage>> {
    fn linked(_: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker {
        ViewPath::empty()
    }

    type OwnedTracker = ViewPath<'static, StorageConfig<HasLink<false>, EmptyStorage>>;
    fn into_owned(_: Self) -> Self::OwnedTracker {
        ViewPath::empty()
    }

    type StaticProjectionTracker =
        ViewPath<'static, StorageConfig<HasLink<false>, CopyableStorage>>;
    fn with_static_projection(
        _: Self,
        projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker {
        let segment = match projection {
            ProjectionPathSegment::Field(field) => ViewPathSegment::field(*field),
            ProjectionPathSegment::Index(index) => ViewPathSegment::index(index),
        };
        ViewPath::single(segment)
    }

    type StaticStrFieldTracker = ViewPath<'static, StorageConfig<HasLink<false>, CopyableStorage>>;
    fn with_field(_: Self, field: &'static str) -> Self::StaticStrFieldTracker {
        let segment = ViewPathSegment::field(field);
        ViewPath::single(segment)
    }

    type OwnedFieldTracker = ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>>;
    fn with_owned_field(_: Self, field: OwnedField) -> Self::OwnedFieldTracker {
        let segment = ViewPathSegment::dynamic_field(field);
        ViewPath::empty().with_pushed_segment(segment)
    }

    type IndexTracker = ViewPath<'static, StorageConfig<HasLink<false>, CopyableStorage>>;
    fn with_index(_: Self, index: usize) -> Self::IndexTracker {
        let segment = ViewPathSegment::index(index);
        ViewPath::single(segment)
    }

    fn is_empty(_tracker: &Self) -> bool {
        true
    }

    fn len(_tracker: &Self) -> usize {
        0
    }

    common_path_tracker!();
}

impl<'borrow, 'path> PathTrackerLifetime<'borrow>
    for ViewPath<'path, StorageConfig<HasLink<true>, EmptyStorage>>
{
    type LinkedTracker = ViewPath<'path, StorageConfig<HasLink<true>, EmptyStorage>>;
    common_lifetime!();
}
impl<'path> PathTracker for ViewPath<'path, StorageConfig<HasLink<true>, EmptyStorage>> {
    fn linked(previous: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker {
        *previous
    }

    type OwnedTracker = ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>>;
    fn into_owned(previous: Self) -> Self::OwnedTracker {
        previous.into_owned()
    }

    type StaticProjectionTracker = ViewPath<'path, StorageConfig<HasLink<true>, CopyableStorage>>;
    fn with_static_projection(
        previous: Self,
        projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker {
        let segment = match projection {
            ProjectionPathSegment::Field(field) => ViewPathSegment::field(*field),
            ProjectionPathSegment::Index(index) => ViewPathSegment::index(index),
        };
        previous.fill_empty(segment)
    }

    type StaticStrFieldTracker = ViewPath<'path, StorageConfig<HasLink<true>, CopyableStorage>>;
    fn with_field(previous: Self, field: &'static str) -> Self::StaticStrFieldTracker {
        let segment = ViewPathSegment::field(field);
        previous.fill_empty(segment)
    }

    type OwnedFieldTracker = ViewPath<'path, StorageConfig<HasLink<true>, MaybeOwnedStorage>>;
    fn with_owned_field(previous: Self, field: OwnedField) -> Self::OwnedFieldTracker {
        let segment = ViewPathSegment::dynamic_field(field);
        previous.change_empty_storage().with_pushed_segment(segment)
    }

    type IndexTracker = ViewPath<'path, StorageConfig<HasLink<true>, CopyableStorage>>;
    fn with_index(previous: Self, index: usize) -> Self::IndexTracker {
        let segment = ViewPathSegment::index(index);
        previous.fill_empty(segment)
    }

    common_path_tracker!();
    common_path_tracker_len!();
}

impl<'borrow> PathTrackerLifetime<'borrow>
    for ViewPath<'_, StorageConfig<HasLink<false>, CopyableStorage>>
{
    type LinkedTracker = ViewPath<'static, StorageConfig<HasLink<false>, CopyableStorage>>;
    common_lifetime!();
}
impl PathTracker for ViewPath<'_, StorageConfig<HasLink<false>, CopyableStorage>> {
    fn linked(previous: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker {
        previous.as_static()
    }

    type OwnedTracker = ViewPath<'static, StorageConfig<HasLink<false>, CopyableStorage>>;
    fn into_owned(previous: Self) -> Self::OwnedTracker {
        previous.as_static()
    }

    type StaticProjectionTracker = ViewPath<
        'static,
        StorageConfig<
            HasLink<false>,
            TypeListStorage<(
                ViewPathSegment<CopyableStorage>,
                ViewPathSegment<CopyableStorage>,
            )>,
        >,
    >;
    fn with_static_projection(
        previous: Self,
        projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker {
        let segment = match projection {
            ProjectionPathSegment::Field(field) => ViewPathSegment::field(*field),
            ProjectionPathSegment::Index(index) => ViewPathSegment::index(index),
        };
        previous
            .as_static()
            .as_type_list()
            .with_segment_type(segment)
    }

    type StaticStrFieldTracker = ViewPath<
        'static,
        StorageConfig<
            HasLink<false>,
            TypeListStorage<(
                ViewPathSegment<CopyableStorage>,
                ViewPathSegment<CopyableStorage>,
            )>,
        >,
    >;
    fn with_field(previous: Self, field: &'static str) -> Self::StaticStrFieldTracker {
        let segment = ViewPathSegment::field(field);
        previous
            .as_static()
            .as_type_list()
            .with_segment_type(segment)
    }

    type IndexTracker = ViewPath<
        'static,
        StorageConfig<
            HasLink<false>,
            TypeListStorage<(
                ViewPathSegment<CopyableStorage>,
                ViewPathSegment<CopyableStorage>,
            )>,
        >,
    >;
    fn with_index(previous: Self, index: usize) -> Self::IndexTracker {
        let segment = ViewPathSegment::index(index);
        previous
            .as_static()
            .as_type_list()
            .with_segment_type(segment)
    }

    type OwnedFieldTracker = ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>>;
    fn with_owned_field(previous: Self, field: OwnedField) -> Self::OwnedFieldTracker {
        let segment = ViewPathSegment::dynamic_field(field);
        previous
            .as_static()
            .into_maybe_owned()
            .with_pushed_segment(segment)
    }

    fn is_empty(_tracker: &Self) -> bool {
        false
    }

    fn len(_tracker: &Self) -> usize {
        1
    }

    common_path_tracker!();
}

impl<'borrow, 'path> PathTrackerLifetime<'borrow>
    for ViewPath<'path, StorageConfig<HasLink<true>, CopyableStorage>>
{
    type LinkedTracker = ViewPath<'path, StorageConfig<HasLink<true>, CopyableStorage>>;
    common_lifetime!();
}
impl<'path> PathTracker for ViewPath<'path, StorageConfig<HasLink<true>, CopyableStorage>> {
    fn linked(previous: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker {
        *previous
    }

    type OwnedTracker = ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>>;
    fn into_owned(previous: Self) -> Self::OwnedTracker {
        previous.into_owned()
    }

    type StaticProjectionTracker = ViewPath<
        'path,
        StorageConfig<
            HasLink<true>,
            TypeListStorage<(
                ViewPathSegment<CopyableStorage>,
                ViewPathSegment<CopyableStorage>,
            )>,
        >,
    >;
    fn with_static_projection(
        previous: Self,
        projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker {
        let segment = match projection {
            ProjectionPathSegment::Field(field) => ViewPathSegment::field(*field),
            ProjectionPathSegment::Index(index) => ViewPathSegment::index(index),
        };
        previous.as_type_list().with_segment_type(segment)
    }

    type StaticStrFieldTracker = ViewPath<
        'path,
        StorageConfig<
            HasLink<true>,
            TypeListStorage<(
                ViewPathSegment<CopyableStorage>,
                ViewPathSegment<CopyableStorage>,
            )>,
        >,
    >;
    fn with_field(previous: Self, field: &'static str) -> Self::StaticStrFieldTracker {
        let segment = ViewPathSegment::field(field);
        previous.as_type_list().with_segment_type(segment)
    }

    type IndexTracker = ViewPath<
        'path,
        StorageConfig<
            HasLink<true>,
            TypeListStorage<(
                ViewPathSegment<CopyableStorage>,
                ViewPathSegment<CopyableStorage>,
            )>,
        >,
    >;
    fn with_index(previous: Self, index: usize) -> Self::IndexTracker {
        let segment = ViewPathSegment::index(index);
        previous.as_type_list().with_segment_type(segment)
    }

    type OwnedFieldTracker = ViewPath<'path, StorageConfig<HasLink<true>, MaybeOwnedStorage>>;
    fn with_owned_field(previous: Self, field: OwnedField) -> Self::OwnedFieldTracker {
        let segment = ViewPathSegment::dynamic_field(field);
        previous.into_maybe_owned().with_pushed_segment(segment)
    }

    common_path_tracker!();
    common_path_tracker_len!();
}

impl<'borrow, L> PathTrackerLifetime<'borrow>
    for ViewPath<'_, StorageConfig<HasLink<false>, TypeListStorage<L>>>
where
    L: ViewPathTypeList + Clone,
{
    type LinkedTracker = ViewPath<'static, StorageConfig<HasLink<false>, TypeListStorage<L>>>;
    common_lifetime!();
}
impl<L> PathTracker for ViewPath<'_, StorageConfig<HasLink<false>, TypeListStorage<L>>>
where
    L: ViewPathTypeList + Clone,
{
    fn linked(previous: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker {
        // Note: the list should only contain types that implement Copy, so
        // clone should be cheep.
        previous.clone().as_static()
    }

    type OwnedTracker = ViewPath<'static, StorageConfig<HasLink<false>, TypeListStorage<L>>>;
    fn into_owned(previous: Self) -> Self::OwnedTracker {
        previous.as_static()
    }

    type StaticProjectionTracker = ViewPath<
        'static,
        StorageConfig<HasLink<false>, TypeListStorage<(L, ViewPathSegment<CopyableStorage>)>>,
    >;
    fn with_static_projection(
        previous: Self,
        projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker {
        let segment = match projection {
            ProjectionPathSegment::Field(field) => ViewPathSegment::field(*field),
            ProjectionPathSegment::Index(index) => ViewPathSegment::index(index),
        };
        previous.as_static().with_segment_type(segment)
    }

    type StaticStrFieldTracker = ViewPath<
        'static,
        StorageConfig<HasLink<false>, TypeListStorage<(L, ViewPathSegment<CopyableStorage>)>>,
    >;
    fn with_field(previous: Self, field: &'static str) -> Self::StaticStrFieldTracker {
        let segment = ViewPathSegment::field(field);
        previous.as_static().with_segment_type(segment)
    }

    type IndexTracker = ViewPath<
        'static,
        StorageConfig<HasLink<false>, TypeListStorage<(L, ViewPathSegment<CopyableStorage>)>>,
    >;
    fn with_index(previous: Self, index: usize) -> Self::IndexTracker {
        let segment = ViewPathSegment::index(index);
        previous.as_static().with_segment_type(segment)
    }

    type OwnedFieldTracker = ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>>;
    fn with_owned_field(previous: Self, field: OwnedField) -> Self::OwnedFieldTracker {
        let segment = ViewPathSegment::dynamic_field(field);
        previous
            .as_static()
            .into_maybe_owned()
            .with_pushed_segment(segment)
    }

    fn is_empty(tracker: &Self) -> bool {
        tracker.0.local.0.len() == 0
    }

    fn len(tracker: &Self) -> usize {
        tracker.0.local.0.len()
    }
    common_path_tracker!();
}

impl<'borrow, 'path, L> PathTrackerLifetime<'borrow>
    for ViewPath<'path, StorageConfig<HasLink<true>, TypeListStorage<L>>>
where
    L: ViewPathTypeList + Clone,
{
    type LinkedTracker = ViewPath<'path, StorageConfig<HasLink<true>, TypeListStorage<L>>>;
    common_lifetime!();
}
impl<'path, L> PathTracker for ViewPath<'path, StorageConfig<HasLink<true>, TypeListStorage<L>>>
where
    L: ViewPathTypeList + Clone,
{
    fn linked(previous: &Self) -> <Self as PathTrackerLifetime<'_>>::LinkedTracker {
        // Note: the list should only contain types that implement Copy, so
        // clone should be cheep.
        previous.clone()
    }

    type OwnedTracker = ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>>;
    fn into_owned(previous: Self) -> Self::OwnedTracker {
        previous.into_owned()
    }

    type StaticProjectionTracker = ViewPath<
        'path,
        StorageConfig<HasLink<true>, TypeListStorage<(L, ViewPathSegment<CopyableStorage>)>>,
    >;
    fn with_static_projection(
        previous: Self,
        projection: ProjectionPathSegment<'static>,
    ) -> Self::StaticProjectionTracker {
        let segment = match projection {
            ProjectionPathSegment::Field(field) => ViewPathSegment::field(*field),
            ProjectionPathSegment::Index(index) => ViewPathSegment::index(index),
        };
        previous.with_segment_type(segment)
    }

    type StaticStrFieldTracker = ViewPath<
        'path,
        StorageConfig<HasLink<true>, TypeListStorage<(L, ViewPathSegment<CopyableStorage>)>>,
    >;
    fn with_field(previous: Self, field: &'static str) -> Self::StaticStrFieldTracker {
        let segment = ViewPathSegment::field(field);
        previous.with_segment_type(segment)
    }

    type IndexTracker = ViewPath<
        'path,
        StorageConfig<HasLink<true>, TypeListStorage<(L, ViewPathSegment<CopyableStorage>)>>,
    >;
    fn with_index(previous: Self, index: usize) -> Self::IndexTracker {
        let segment = ViewPathSegment::index(index);
        previous.with_segment_type(segment)
    }

    type OwnedFieldTracker = ViewPath<'path, StorageConfig<HasLink<true>, MaybeOwnedStorage>>;
    fn with_owned_field(previous: Self, field: OwnedField) -> Self::OwnedFieldTracker {
        let segment = ViewPathSegment::dynamic_field(field);
        previous.into_maybe_owned().with_pushed_segment(segment)
    }

    common_path_tracker!();
    common_path_tracker_len!();
}

////////////////////////////////////////////////////////////////////////////////
// Storage abstraction
////////////////////////////////////////////////////////////////////////////////

pub mod storage;
use storage::*;
pub use storage::{
    CopyableStorage, EmptyStorage, HasLink, MaybeOwnedStorage, StorageConfig, TypeListStorage,
};

////////////////////////////////////////////////////////////////////////////////
// Path
////////////////////////////////////////////////////////////////////////////////

/// Used internally to abstract over [`ViewPath`]s with different storage
/// formats.
trait DynViewPath: fmt::Debug + fmt::Display + Send + Sync {
    fn state(&self) -> ViewPathStateRef<'_>;
}
impl<L: LinkStorage> DynViewPath for ViewPath<'_, StorageConfig<L, MaybeOwnedStorage>> {
    fn state(&self) -> ViewPathStateRef<'_> {
        self.0.map_to_ref(|buffer| {
            #[cfg(feature = "alloc")]
            {
                Some(ViewPathLocalStateRef::Buffer(buffer))
            }
            #[cfg(not(feature = "alloc"))]
            {
                Some(ViewPathLocalStateRef::TypeList(&buffer.len))
            }
        })
    }
}
impl<L: LinkStorage> DynViewPath for ViewPath<'_, StorageConfig<L, EmptyStorage>> {
    fn state(&self) -> ViewPathStateRef<'_> {
        self.0.map_to_ref(|&()| None)
    }
}
impl<L: LinkStorage> DynViewPath for ViewPath<'_, StorageConfig<L, CopyableStorage>> {
    fn state(&self) -> ViewPathStateRef<'_> {
        self.0
            .map_to_ref(|segment| Some(ViewPathLocalStateRef::Single(segment.as_ref())))
    }
}
impl<L, LS> DynViewPath for ViewPath<'_, StorageConfig<LS, TypeListStorage<L>>>
where
    L: ViewPathTypeList + Clone,
    LS: LinkStorage,
{
    fn state(&self) -> ViewPathStateRef<'_> {
        self.0
            .map_to_ref(|list| Some(ViewPathLocalStateRef::TypeList(list)))
    }
}

/// Borrowed version of [`ViewPath`] that also hides the storage format.
#[derive(Debug, Clone, Copy)]
pub struct ViewPathRef<'a> {
    /// This is a [`ViewPath`] with one of the storage formats specified in this
    /// module.
    ///
    /// We use dynamic dispatch instead of an enum since
    /// [`TypeListStorage`] has a type parameter so we would need
    /// dynamic dispatch somewhere no matter what we did.
    inner: &'a dyn DynViewPath,
}
impl<'a> ViewPathRef<'a> {
    pub fn is_empty(&self) -> bool {
        self.reversed_segments().next().is_some()
    }
    pub fn len(&self) -> usize {
        self.reversed_segments().size_hint().0
    }

    pub fn reversed_segments(self) -> ViewPathRevIter<'a> {
        ViewPathRevIter::new(self.state())
    }
    pub fn iter(self) -> ViewPathIter<'a> {
        ViewPathIter::new(self.state())
    }

    /// Create a new path that links directly to this path reference.
    pub fn as_linked_path(self) -> ViewPath<'a, StorageConfig<HasLink<true>, EmptyStorage>> {
        ViewPath(ViewPathState {
            link: Some((self, ())),
            local: (),
        })
    }
    fn state(self) -> ViewPathStateRef<'a> {
        self.inner.state()
    }

    /// Clone the current path into an owned version in the most efficient way.
    pub fn to_owned(self) -> ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>> {
        ViewPath(ViewPathState {
            link: None,
            local: self.collect_buffer(),
        })
    }
    #[allow(unused_mut)]
    fn collect_buffer(self) -> MaybeOwnedBuffer {
        let mut v = MaybeOwnedBuffer::new();
        #[cfg(feature = "alloc")]
        {
            let mut buffer = self.reversed_segments().map(Into::into).collect::<Vec<_>>();
            buffer.reverse();
            v.buffer = buffer;
        }
        v
    }
}
impl fmt::Display for ViewPathRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.inner, f)
    }
}
impl<'a, S: ViewPathStorage> From<&'a ViewPath<'a, S>> for ViewPathRef<'a> {
    fn from(path: &'a ViewPath<'a, S>) -> Self {
        path.as_ref()
    }
}

/// A path used to access unstructured data.
#[derive(Debug, Clone)]
pub struct ViewPath<'a, S: ViewPathStorage>(ViewPathState<'a, S>);
impl<'a, S: ViewPathStorage> ViewPath<'a, S> {
    pub fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }
    pub fn len(&self) -> usize {
        self.as_ref().len()
    }

    /// Get a simpler reference type to the view path.
    pub fn as_ref(&self) -> ViewPathRef<'_> {
        S::as_path_ref(self)
    }

    pub fn reversed_segments(&self) -> ViewPathRevIter<'_> {
        self.as_ref().reversed_segments()
    }
    pub fn iter(&self) -> ViewPathIter<'_> {
        self.as_ref().iter()
    }

    /// Clone all segments except for those in linked parents.
    pub fn into_maybe_owned(
        self,
    ) -> ViewPath<'a, StorageConfig<S::LinkStorage, MaybeOwnedStorage>> {
        S::into_maybe_owned_path(self)
    }
    /// Clone all segments to ensure the path has a `'static` lifetime.
    pub fn into_owned(self) -> ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>> {
        S::into_owned_path(self)
    }
}
impl ViewPath<'static, StorageConfig<HasLink<false>, CopyableStorage>> {
    // A path with a single segment.
    pub fn single(segment: ViewPathSegment<CopyableStorage>) -> Self {
        ViewPath(ViewPathState {
            link: None,
            local: segment,
        })
    }
}
impl<'path, L> ViewPath<'path, StorageConfig<L, CopyableStorage>>
where
    L: LinkStorage,
{
    // Change to [`TypeListStorage`].
    pub fn as_type_list(
        self,
    ) -> ViewPath<'path, StorageConfig<L, TypeListStorage<ViewPathSegment<CopyableStorage>>>> {
        ViewPath(self.0.map_state(TypeListBuffer))
    }
}
impl<BS> ViewPath<'_, StorageConfig<HasLink<false>, BS>>
where
    BS: BufferStorage,
{
    // An empty path.
    pub fn empty() -> Self
    where
        BS::Buffer: Default,
    {
        ViewPath(ViewPathState {
            link: None,
            local: Default::default(),
        })
    }
    /// Change the link type on an unlinked path.
    pub fn change_list_storage<L>(self) -> ViewPath<'static, StorageConfig<L, BS>>
    where
        L: LinkStorage,
    {
        ViewPath(ViewPathState {
            link: self.0.link.map(|(_, never)| match never {}),
            local: self.0.local,
        })
    }
    /// Cast an unlinked path to the `'static` lifetime.
    pub fn as_static(self) -> ViewPath<'static, StorageConfig<HasLink<false>, BS>> {
        Self::change_list_storage(self)
    }
}
impl<L, BS> ViewPath<'_, StorageConfig<L, BS>>
where
    L: LinkStorage,
    BS: BufferStorage,
{
    /// Create a new path with an extra segment.
    pub fn join(
        &self,
        segment: ViewPathSegment<CopyableStorage>,
    ) -> ViewPath<'_, StorageConfig<HasLink<true>, CopyableStorage>> {
        ViewPath(ViewPathState {
            link: Some((self.as_ref(), ())),
            local: segment,
        })
    }
}
impl<'path, L> ViewPath<'path, StorageConfig<L, EmptyStorage>>
where
    L: LinkStorage,
{
    /// Store a path segment in a path that previously only contained a link to
    /// a parent path.
    pub fn fill_empty(
        self,
        segment: ViewPathSegment<CopyableStorage>,
    ) -> ViewPath<'path, StorageConfig<L, CopyableStorage>> {
        ViewPath(self.0.map_state(|()| segment))
    }
    /// Change the storage type on an empty storage.
    pub fn change_empty_storage<S>(self) -> ViewPath<'path, StorageConfig<L, S>>
    where
        S: BufferStorage,
        S::Buffer: Default,
    {
        ViewPath(self.0.map_state(|()| Default::default()))
    }
}
impl<'path, L, LS> ViewPath<'path, StorageConfig<LS, TypeListStorage<L>>>
where
    L: ViewPathTypeList + Clone,
    LS: LinkStorage,
{
    /// Add a type that provides a path segment to the type list.
    pub fn with_segment_type<I>(
        self,
        segment_type: I,
    ) -> ViewPath<'path, StorageConfig<LS, TypeListStorage<(L, I)>>>
    where
        I: ViewPathTypeListIdent + Clone,
    {
        ViewPath(
            self.0
                .map(|link| link, |list| TypeListBuffer((list.0, segment_type))),
        )
    }
}
/// Operations for owned paths.
impl<'path, L> ViewPath<'path, StorageConfig<L, MaybeOwnedStorage>>
where
    L: LinkStorage,
{
    /// Add a path segment.
    pub fn push(&mut self, segment: ViewPathSegment<MaybeOwnedStorage>) {
        self.0.local.push(segment);
    }
    /// Add a path segment.
    pub fn with_pushed_segment(
        mut self,
        segment: ViewPathSegment<MaybeOwnedStorage>,
    ) -> ViewPath<'path, StorageConfig<L, MaybeOwnedStorage>> {
        self.push(segment);
        self
    }
}
impl<S: ViewPathStorage> fmt::Display for ViewPath<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ProjectionPathSegment::format_path(self.iter().map(Into::into), f)
    }
}
impl<S> Copy for ViewPath<'_, S>
where
    S: ViewPathStorage,
    S::Buffer: Copy,
    S::LinkedState: Copy,
{
}

////////////////////////////////////////////////////////////////////////////////
// Path Iterators
////////////////////////////////////////////////////////////////////////////////

impl<'a, S: ViewPathStorage> IntoIterator for &'a ViewPath<'a, S> {
    type IntoIter = ViewPathIter<'a>;
    type Item = ViewPathSegmentRef<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
/// Note that if the `alloc` feature is disabled then the created path won't
/// actually store anything.
impl core::iter::FromIterator<ViewPathSegment<MaybeOwnedStorage>>
    for ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>>
{
    fn from_iter<T: IntoIterator<Item = ViewPathSegment<MaybeOwnedStorage>>>(iter: T) -> Self {
        ViewPath(ViewPathState {
            link: None,
            local: iter.into_iter().collect(),
        })
    }
}

/// Iterator over a [`ViewPath`]'s segments. Starts with the segments that was
/// appended first.
///
/// Note that this iterator is double ended so it can be reversed as well.
#[derive(Debug, Clone)]
pub struct ViewPathIter<'a> {
    reversed_iter: ViewPathRevIter<'a>,
    /// The index of the last item that was yielded. When this is `0` then all
    /// items have been yielded.
    last_index: Option<usize>,
}
impl<'a> ViewPathIter<'a> {
    fn new(root: ViewPathStateRef<'a>) -> Self {
        Self {
            reversed_iter: ViewPathRevIter::new(root),
            last_index: None,
        }
    }
}
impl<'a> Iterator for ViewPathIter<'a> {
    type Item = ViewPathSegmentRef<'a>;
    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(last) = self.last_index {
            (last, Some(last))
        } else {
            self.reversed_iter.size_hint()
        }
    }
    fn next(&mut self) -> Option<Self::Item> {
        match self.last_index {
            Some(0) => None,
            // Get the element before the last element:
            Some(ref mut last_index) => {
                *last_index -= 1;
                self.reversed_iter.clone().nth(*last_index)
            }
            // First run, determine total length:
            None => {
                let (index, segment) = self.reversed_iter.clone().enumerate().last()?;
                self.last_index = Some(index);
                Some(segment)
            }
        }
    }
}
impl DoubleEndedIterator for ViewPathIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let segment = self.reversed_iter.next();
        if let Some(last_index) = &mut self.last_index {
            // The index of the last seen value will now be one step closer in
            // the iterator (since we don't need to skip quite as many values):
            *last_index = last_index.saturating_sub(1);
        }
        segment
    }
}

/// This indicates the previous action of the [`ViewPathRevIter`] iterator.
#[derive(Debug, Clone)]
enum ViewPathRevIterState {
    /// Yield segments from the current path's buffer.
    ReadingBuffer {
        /// The number of segments inside the buffer that have been yielded.
        read: usize,
    },
    /// Yield the segment inside the current `ViewPath` and then move to the
    /// path's parent.
    TraversingLinks,
}
/// Iterator over a [`ViewPath`]'s segments in reverse order. Starts with the
/// segment that was appended last.
///
/// This is the fastest way to iterate over a path's segments.
#[derive(Debug, Clone)]
pub struct ViewPathRevIter<'a> {
    current: Option<ViewPathStateRef<'a>>,
    state: ViewPathRevIterState,
}
impl<'a> ViewPathRevIter<'a> {
    fn new(path_state: ViewPathStateRef<'a>) -> Self {
        Self {
            current: Some(path_state),
            state: ViewPathRevIterState::TraversingLinks,
        }
    }
}
impl<'a> Iterator for ViewPathRevIter<'a> {
    type Item = ViewPathSegmentRef<'a>;
    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut path = if let Some(v) = self.current {
            v
        } else {
            return (0, Some(0));
        };
        let mut count = 0;

        // Add count for half read buffer:
        if let ViewPathRevIterState::ReadingBuffer { read } = self.state {
            match path.local {
                #[cfg(feature = "alloc")]
                Some(ViewPathLocalStateRef::Buffer(buffer)) => {
                    count += buffer.len().saturating_sub(read);
                }
                Some(ViewPathLocalStateRef::TypeList(buffer)) => {
                    count += buffer.0.len().saturating_sub(read);
                }
                None | Some(ViewPathLocalStateRef::Single(_)) => {
                    unreachable!(
                        "We can only reach the ReadingBuffer state if the path stored a buffer"
                    );
                }
            }
            // Finished with the current path:
            path = match path.link {
                Some(link) => link.state(),
                None => return (count, Some(count)),
            }
        }

        loop {
            // Count segments in this path:
            if let Some(local) = path.local {
                match local {
                    ViewPathLocalStateRef::Single(_) => {
                        count += 1;
                    }
                    #[cfg(feature = "alloc")]
                    ViewPathLocalStateRef::Buffer(buffer) => {
                        count += buffer.len();
                    }
                    ViewPathLocalStateRef::TypeList(list) => {
                        count += list.0.len();
                    }
                }
            }
            // Continue with linked parent:
            match path.link {
                Some(link) => {
                    path = link.state();
                }
                None => break,
            }
        }
        (count, Some(count))
    }
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let path = self.current?;
            match self.state {
                ViewPathRevIterState::ReadingBuffer { ref mut read } => {
                    match path.local {
                        #[cfg(feature = "alloc")]
                        Some(ViewPathLocalStateRef::Buffer(buffer)) => {
                            if let Some(next) = buffer.iter().rev().nth(*read) {
                                *read += 1;
                                return Some(next.as_ref());
                            }
                        }
                        Some(ViewPathLocalStateRef::TypeList(buffer)) => {
                            let index = buffer.0.len().saturating_sub(1).checked_sub(*read);
                            if let Some(next) = index.and_then(|index| buffer.0.get(index)) {
                                *read += 1;
                                return Some(next.as_static_ref());
                            }
                        }
                        None | Some(ViewPathLocalStateRef::Single(_)) => {
                            unreachable!(
                                "We can only reach the ReadingBuffer state if the path stored a buffer"
                            );
                        }
                    }
                    // Read all items in buffer => Move to next path:
                    self.current = path.link.map(|link| link.state());
                    self.state = ViewPathRevIterState::TraversingLinks;
                }
                ViewPathRevIterState::TraversingLinks => {
                    // Count segments in this path:
                    if let Some(local) = path.local {
                        match local {
                            ViewPathLocalStateRef::Single(item) => {
                                // Move to next path:
                                self.current = path.link.map(|link| link.state());
                                // Then return this path's item:
                                return Some(item);
                            }
                            #[cfg(feature = "alloc")]
                            ViewPathLocalStateRef::Buffer(_) => {
                                self.state = ViewPathRevIterState::ReadingBuffer { read: 0 };
                            }
                            ViewPathLocalStateRef::TypeList(_) => {
                                self.state = ViewPathRevIterState::ReadingBuffer { read: 0 };
                            }
                        }
                        // Start reading buffer in next loop iteration...
                    } else {
                        // Move to next path:
                        self.current = path.link.map(|link| link.state());
                    }
                }
            };
        }
    }
    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        // Optimized to skip right into the wanted element of an array, instead
        // of iterating over the array multiple times. Note: this method is used
        // heavily by the `ViewPathIter` iterator.
        loop {
            if n == 0 {
                break self.next();
            }
            match self.state {
                // Specialize the reading buffer case to skip multiple segments
                // at the same time:
                ViewPathRevIterState::ReadingBuffer { ref mut read } => {
                    let path = self.current?;
                    match path.local {
                        #[cfg(feature = "alloc")]
                        Some(ViewPathLocalStateRef::Buffer(buffer)) => {
                            let remaining = buffer.len().saturating_sub(*read);
                            if n >= remaining {
                                n -= remaining;
                            } else {
                                // Would normally go to element at `read` but now skips to `n` elements
                                // after that. If `n` is `0` we would return the same element as `next`
                                let v = buffer.iter().rev().nth(*read + n);
                                *read += 1 + n;
                                break v.map(Into::into);
                            }
                        }
                        Some(ViewPathLocalStateRef::TypeList(buffer)) => {
                            let remaining = buffer.0.len().saturating_sub(*read);
                            if n >= remaining {
                                n -= remaining;
                            } else {
                                let index = buffer.0.len().saturating_sub(1).checked_sub(*read + n);
                                let v = index.and_then(|index| buffer.0.get(index));
                                *read += 1 + n;
                                break v.map(|segment| segment.as_static_ref());
                            }
                        }
                        None | Some(ViewPathLocalStateRef::Single(_)) => {
                            unreachable!(
                                "We can only reach the ReadingBuffer state if the path stored a buffer"
                            );
                        }
                    }
                    // Read all items in buffer => Move to next path:
                    self.current = path.link.map(|link| link.state());
                    self.state = ViewPathRevIterState::TraversingLinks;
                }
                ViewPathRevIterState::TraversingLinks => {
                    self.next();
                    n -= 1;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Parent Link
////////////////////////////////////////////////////////////////////////////////

/// Borrowed version of [`ViewPathState`] that also hides the storage format.
#[derive(Debug, Clone, Copy, Default)]
struct ViewPathStateRef<'a> {
    link: Option<ViewPathRef<'a>>,
    local: Option<ViewPathLocalStateRef<'a>>,
}
/// Part of the [`ViewPathState`] that is local to the current [`ViewPath`]
#[derive(Debug, Clone, Copy)]
enum ViewPathLocalStateRef<'a> {
    Single(ViewPathSegmentRef<'a>),
    /// Reference into a `Vec` of segments.
    ///
    /// The segments might as well use the owned storage since the `Vec` would
    /// block any `Copy` implementation anyway.
    #[cfg(feature = "alloc")]
    Buffer(&'a [ViewPathSegment<MaybeOwnedStorage>]),
    TypeList(&'a TypeListBuffer<dyn ViewPathTypeList>),
}

/// The state for a [`ViewPath`].
#[derive(Debug, Clone, Copy)]
struct ViewPathState<'a, S: ViewPathStorage> {
    /// If this is `Some` then the path is linked to a "parent" path. Any path
    /// segments in the parent comes before path segments stored in the local
    /// state.
    ///
    /// `S::LinkedState` could be [`core::convert::Infallible`] to prevent a
    /// link from ever existing.
    link: Option<(ViewPathRef<'a>, S::LinkedState)>,
    /// Local state that isn't dependant on a parent path.
    local: S::Buffer,
}
impl<'a, S: ViewPathStorage> ViewPathState<'a, S> {
    fn map_to_ref<'b, F>(&'b self, f: F) -> ViewPathStateRef<'b>
    where
        F: FnOnce(&'b S::Buffer) -> Option<ViewPathLocalStateRef<'b>>,
    {
        ViewPathStateRef {
            link: self.link.as_ref().map(|(link, _)| *link),
            local: f(&self.local),
        }
    }
    fn map<FL, FB, S2>(self, map_link_state: FL, map_buffer: FB) -> ViewPathState<'a, S2>
    where
        FL: FnOnce(S::LinkedState) -> S2::LinkedState,
        FB: FnOnce(S::Buffer) -> S2::Buffer,
        S2: ViewPathStorage,
    {
        ViewPathState {
            link: self.link.map(|(link, state)| (link, map_link_state(state))),
            local: map_buffer(self.local),
        }
    }
    fn map_state<F, S2>(self, f: F) -> ViewPathState<'a, S2>
    where
        F: FnOnce(S::Buffer) -> S2::Buffer,
        S2: ViewPathStorage<LinkedState = S::LinkedState>,
    {
        self.map(|link| link, f)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Single Path Segment
////////////////////////////////////////////////////////////////////////////////

/// Borrowed version of [`ViewPathSegment`] that also hides the storage format.
#[derive(Debug, Clone, Copy)]
pub enum ViewPathSegmentRef<'a> {
    /// Access a field.
    Field(MaybeStatic<'a, str>),
    /// Access into an array.
    Index(usize),
}
impl ViewPathSegmentRef<'_> {
    /// Returns `true` if the view path segment is a field.
    #[must_use]
    pub fn is_field(&self) -> bool {
        matches!(self, Self::Field(..))
    }
    /// Note that if the `alloc` feature is disabled then non static borrowed
    /// fields will be forgotten.
    pub fn into_owned(self) -> ViewPathSegment<MaybeOwnedStorage> {
        match self {
            ViewPathSegmentRef::Field(MaybeStatic::Borrowed(borrowed)) => {
                ViewPathSegment::dynamic_field(OwnedField::lossy_to_string(borrowed))
            }
            ViewPathSegmentRef::Field(MaybeStatic::Static(borrowed)) => {
                ViewPathSegment::field(borrowed)
            }
            ViewPathSegmentRef::Index(index) => ViewPathSegment::index(index),
        }
    }
}
impl fmt::Display for ViewPathSegmentRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Field(field) => {
                if field.starts_with(|c: char| c.is_alphabetic())
                    && field.contains(|c: char| c.is_alphanumeric())
                {
                    write!(f, "{}", field)
                } else {
                    #[cfg(feature = "alloc")]
                    let buffer;
                    #[cfg(feature = "alloc")]
                    let field = if field.contains(['"', '\\']) {
                        buffer = field.replace('\\', "\\\\").replace('"', r#"\""#);
                        buffer.as_str()
                    } else {
                        field
                    };
                    write!(f, r#"["{}"]"#, field)
                }
            }
            Self::Index(index) => write!(f, "[{}]", index),
        }
    }
}
impl<'a, S: SegmentStorage> From<&'a ViewPathSegment<S>> for ViewPathSegmentRef<'a> {
    fn from(path: &'a ViewPathSegment<S>) -> Self {
        path.as_ref()
    }
}
/// Note that if the `alloc` feature is disabled then the content of borrowed
/// fields will be lost.
impl<'a> From<ViewPathSegmentRef<'a>> for ViewPathSegment<MaybeOwnedStorage> {
    fn from(value: ViewPathSegmentRef<'a>) -> Self {
        value.into_owned()
    }
}
impl<'a> From<ViewPathSegmentRef<'a>> for ProjectionPathSegment<'a> {
    fn from(path: ViewPathSegmentRef<'a>) -> Self {
        match path {
            ViewPathSegmentRef::Field(field) => ProjectionPathSegment::Field(field),
            ViewPathSegmentRef::Index(index) => ProjectionPathSegment::Index(index),
        }
    }
}
impl<'a> From<ProjectionPathSegment<'a>> for ViewPathSegmentRef<'a> {
    fn from(path: ProjectionPathSegment<'a>) -> Self {
        match path {
            ProjectionPathSegment::Field(field) => ViewPathSegmentRef::Field(field),
            ProjectionPathSegment::Index(index) => ViewPathSegmentRef::Index(index),
        }
    }
}

/// Used by the [`ViewPathSegment`] type to hide its implementation details.
#[derive(Debug)]
enum ViewPathSegmentHidden<S: SegmentStorage> {
    /// Access to a field.
    ///
    /// Can be owned which can be useful if the field name is computed.
    Field(S::Ident),
    /// Access into an array.
    Index(S::Index),
}
impl<S: SegmentStorage> Copy for ViewPathSegmentHidden<S>
where
    S::Ident: Copy,
    S::Index: Copy,
{
}
impl<S: SegmentStorage> Clone for ViewPathSegmentHidden<S>
where
    S::Ident: Clone,
    S::Index: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Field(field) => Self::Field(field.clone()),
            Self::Index(index) => Self::Index(index.clone()),
        }
    }
}

/// Describes a single access made by a path.
#[derive(Debug)]
pub struct ViewPathSegment<S: SegmentStorage>(ViewPathSegmentHidden<S>);
impl<S: SegmentStorage> ViewPathSegment<S> {
    pub fn index(index: usize) -> Self
    where
        S::Index: From<usize>,
    {
        ViewPathSegmentHidden::Index(index.into()).into()
    }
    pub fn field(field: &'static str) -> Self
    where
        S::Ident: From<&'static str>,
    {
        ViewPathSegmentHidden::Field(field.into()).into()
    }

    pub fn into_owned(self) -> ViewPathSegment<MaybeOwnedStorage> {
        S::into_maybe_owned_segment(self)
    }

    pub fn as_ref(&self) -> ViewPathSegmentRef<'_> {
        S::as_segment_ref(self)
    }

    /// Returns `true` if the view path segment is a field.
    #[must_use]
    pub fn is_field(&self) -> bool {
        matches!(self.0, ViewPathSegmentHidden::Field(..))
    }
    /// Returns `true` if the view path segment is an index.
    #[must_use]
    pub fn is_index(&self) -> bool {
        matches!(self.0, ViewPathSegmentHidden::Index(..))
    }
}
impl ViewPathSegment<CopyableStorage> {
    /// A field that is only known at runtime.
    pub fn as_static_ref(&self) -> ViewPathSegmentRef<'static> {
        match self.0 {
            ViewPathSegmentHidden::Field(field) => {
                ViewPathSegmentRef::Field(MaybeStatic::Static(field))
            }
            ViewPathSegmentHidden::Index(index) => ViewPathSegmentRef::Index(index),
        }
    }
}
impl ViewPathSegment<MaybeOwnedStorage> {
    /// A field that is only known at runtime.
    pub fn dynamic_field(field: OwnedField) -> Self {
        ViewPathSegmentHidden::Field(MaybeOwnedIdent::owned_str(field)).into()
    }
}
impl From<ViewPathSegment<CopyableStorage>> for ViewPathSegment<MaybeOwnedStorage> {
    fn from(segment: ViewPathSegment<CopyableStorage>) -> Self {
        segment.into_owned()
    }
}
impl<S: SegmentStorage> Clone for ViewPathSegment<S>
where
    S::Ident: Clone,
    S::Index: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<S: SegmentStorage> Copy for ViewPathSegment<S>
where
    S::Ident: Copy,
    S::Index: Copy,
{
}
impl<S: SegmentStorage> fmt::Display for ViewPathSegment<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.as_ref(), f)
    }
}

impl<S> From<ViewPathSegmentHidden<S>> for ViewPathSegment<S>
where
    S: SegmentStorage,
{
    fn from(segment: ViewPathSegmentHidden<S>) -> Self {
        Self(segment)
    }
}
impl<S> From<ViewPathSegment<S>> for ViewPathSegmentHidden<S>
where
    S: SegmentStorage,
{
    fn from(segment: ViewPathSegment<S>) -> Self {
        segment.0
    }
}
