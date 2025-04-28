//! Abstracts over how a path is stored.

use core::{convert::Infallible, iter::FromIterator, ops::Deref};

use super::*;

/// Abstracts over how a path is stored.
pub trait ViewPathStorage: fmt::Debug + Clone + Copy + Send + Sync {
    /// The type of the buffer that stores path segments.
    type Buffer: fmt::Debug + Clone;
    /// Extra state when path is linked to a parent path.
    type LinkedState: fmt::Debug + Clone;

    // TODO: if we add a BufferStorage type as well then we might be able to get
    // rid of all methods from this trait.
    type LinkStorage: LinkStorage<LinkedState = Self::LinkedState>;

    /// Used by [`ViewPath::as_ref`].
    fn as_path_ref<'a>(path: &'a ViewPath<'a, Self>) -> ViewPathRef<'a>;

    /// Used by [`ViewPath::into_owned`].
    fn into_owned_path(
        path: ViewPath<'_, Self>,
    ) -> ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>>;
    fn into_maybe_owned_path(
        path: ViewPath<'_, Self>,
    ) -> ViewPath<'_, StorageConfig<Self::LinkStorage, MaybeOwnedStorage>>;
}

/// Configuration of storage options. Implements [`ViewPathStorage`] which
/// abstracts over how [`ViewPath`]'s are stored.
#[derive(Debug, Clone, Copy)]
pub struct StorageConfig<L, B>(L, B);

impl<L, B> ViewPathStorage for StorageConfig<L, B>
where
    L: LinkStorage,
    B: BufferStorage,
{
    type Buffer = B::Buffer;
    type LinkedState = L::LinkedState;
    type LinkStorage = L;

    fn as_path_ref<'a>(path: &'a ViewPath<'a, Self>) -> ViewPathRef<'a> {
        B::as_path_ref(path)
    }

    fn into_owned_path(
        path: ViewPath<'_, Self>,
    ) -> ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>> {
        B::into_owned_path(path)
    }
    fn into_maybe_owned_path(
        path: ViewPath<'_, Self>,
    ) -> ViewPath<'_, StorageConfig<L, MaybeOwnedStorage>> {
        B::into_maybe_owned_path(path)
    }
}

/// Determines if a [`ViewPath`] is allowed to link to another path.
pub trait LinkStorage: fmt::Debug + Clone + Copy + Send + Sync + 'static {
    /// Extra state when path is linked to a parent path.
    type LinkedState: fmt::Debug + Clone + Send + Sync;
}

#[derive(Debug, Clone, Copy)]
pub struct HasLink<const IS_ALLOWED: bool>(());
/// No parent link allowed.
impl LinkStorage for HasLink<false> {
    type LinkedState = Infallible;
}
/// Parent links are allowed.
impl LinkStorage for HasLink<true> {
    type LinkedState = ();
}

pub trait SegmentStorage: fmt::Debug + Clone + Copy + Send + Sync + 'static {
    /// The type that stores an identifier for a path segment.
    type Ident: fmt::Debug + Clone + Send + Sync;
    type Index: fmt::Debug + Clone + Send + Sync;

    /// Used by [`ViewPathSegment::into_owned`].
    fn into_maybe_owned_segment(
        segment: ViewPathSegment<Self>,
    ) -> ViewPathSegment<MaybeOwnedStorage>;

    /// Used by [`ViewPathSegment::as_ref`].
    fn as_segment_ref(segment: &ViewPathSegment<Self>) -> ViewPathSegmentRef<'_>;
}
impl SegmentStorage for CopyableStorage {
    type Ident = &'static str;
    type Index = usize;

    fn into_maybe_owned_segment(
        segment: ViewPathSegment<Self>,
    ) -> ViewPathSegment<MaybeOwnedStorage> {
        match segment.0 {
            ViewPathSegmentHidden::Field(field) => ViewPathSegment::field(field),
            ViewPathSegmentHidden::Index(index) => ViewPathSegment::index(index),
        }
    }

    fn as_segment_ref(segment: &ViewPathSegment<Self>) -> ViewPathSegmentRef<'_> {
        match segment.0 {
            ViewPathSegmentHidden::Field(field) => {
                ViewPathSegmentRef::Field(MaybeStatic::Static(field))
            }
            ViewPathSegmentHidden::Index(index) => ViewPathSegmentRef::Index(index),
        }
    }
}

/// Used by the [`SegmentStorage`] trait implementation for
/// [`MaybeOwnedStorage`].
#[derive(Debug, Clone, Default)]
pub struct MaybeOwnedIdent {
    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "docs", doc(cfg(feature = "alloc")))]
    pub ident: Cow<'static, str>,
    #[cfg(not(feature = "alloc"))]
    #[cfg_attr(feature = "docs", doc(cfg(not(feature = "alloc"))))]
    ident: &'static str,
}
impl MaybeOwnedIdent {
    pub const fn static_str(ident: &'static str) -> Self {
        Self {
            #[cfg(feature = "alloc")]
            ident: Cow::Borrowed(ident),
            #[cfg(not(feature = "alloc"))]
            ident,
        }
    }
    pub fn owned_str(_ident: OwnedField) -> Self {
        #[cfg(feature = "alloc")]
        {
            Self {
                ident: Cow::Owned(_ident.0),
            }
        }
        #[cfg(not(feature = "alloc"))]
        {
            Self::static_str("<dynamic field>")
        }
    }
    pub fn borrow(&self) -> MaybeStatic<'_, str> {
        #[cfg(feature = "alloc")]
        {
            match &self.ident {
                Cow::Borrowed(field) => MaybeStatic::Static(field),
                Cow::Owned(field) => MaybeStatic::Borrowed(field),
            }
        }
        #[cfg(not(feature = "alloc"))]
        {
            MaybeStatic::Static(self.ident)
        }
    }
}
impl Deref for MaybeOwnedIdent {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.ident
    }
}
impl From<&'static str> for MaybeOwnedIdent {
    fn from(ident: &'static str) -> Self {
        Self::static_str(ident)
    }
}

impl SegmentStorage for MaybeOwnedStorage {
    type Ident = MaybeOwnedIdent;
    type Index = usize;

    fn into_maybe_owned_segment(
        segment: ViewPathSegment<Self>,
    ) -> ViewPathSegment<MaybeOwnedStorage> {
        segment
    }
    fn as_segment_ref(segment: &ViewPathSegment<Self>) -> ViewPathSegmentRef<'_> {
        match &segment.0 {
            ViewPathSegmentHidden::Field(field) => ViewPathSegmentRef::Field(field.borrow()),
            ViewPathSegmentHidden::Index(index) => ViewPathSegmentRef::Index(*index),
        }
    }
}

/// Abstracts over how a path is stored.
pub trait BufferStorage: fmt::Debug + Clone + Copy + Send + Sync + 'static {
    /// The type of the buffer that stores path segments.
    type Buffer: fmt::Debug + Clone;

    /// Used by [`ViewPathStorage::into_owned_path`].
    fn into_owned_path<L: LinkStorage>(
        path: ViewPath<'_, StorageConfig<L, Self>>,
    ) -> ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>>;
    /// Used by [`ViewPathStorage::into_maybe_owned_path`].
    fn into_maybe_owned_path<L: LinkStorage>(
        path: ViewPath<'_, StorageConfig<L, Self>>,
    ) -> ViewPath<'_, StorageConfig<L, MaybeOwnedStorage>>;

    /// Used by [`ViewPathStorage::as_path_ref`].
    fn as_path_ref<'a, L: LinkStorage>(
        path: &'a ViewPath<'a, StorageConfig<L, Self>>,
    ) -> ViewPathRef<'a>;
}

/// Used by the [`BufferStorage`] trait implementation for
/// [`MaybeOwnedStorage`].
#[derive(Debug, Clone)]
pub struct MaybeOwnedBuffer {
    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "docs", doc(cfg(feature = "alloc")))]
    pub buffer: Vec<ViewPathSegment<MaybeOwnedStorage>>,
    #[cfg(not(feature = "alloc"))]
    #[cfg_attr(feature = "docs", doc(cfg(not(feature = "alloc"))))]
    pub(crate) len: TypeListBuffer<SkippedSegments>,
    marker: PhantomData<ViewPathSegment<MaybeOwnedStorage>>,
}
impl MaybeOwnedBuffer {
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "alloc")]
            buffer: Vec::new(),
            #[cfg(not(feature = "alloc"))]
            len: TypeListBuffer(SkippedSegments(0)),
            marker: PhantomData,
        }
    }
    #[allow(unused_mut)]
    pub fn single(_item: ViewPathSegment<MaybeOwnedStorage>) -> Self {
        let mut v = Self::new();
        #[cfg(feature = "alloc")]
        {
            v.buffer = alloc::vec![_item];
        }
        #[cfg(not(feature = "alloc"))]
        {
            v.len.0 .0 = 1;
        }
        v
    }
    pub fn push(&mut self, _item: ViewPathSegment<MaybeOwnedStorage>) {
        #[cfg(feature = "alloc")]
        {
            self.buffer.push(_item);
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.len.0 .0 += 1;
        }
    }
    pub fn len(&self) -> usize {
        #[cfg(feature = "alloc")]
        {
            self.buffer.len()
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.len.0 .0
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl Default for MaybeOwnedBuffer {
    fn default() -> Self {
        Self::new()
    }
}
impl Deref for MaybeOwnedBuffer {
    type Target = [ViewPathSegment<MaybeOwnedStorage>];

    fn deref(&self) -> &Self::Target {
        #[cfg(feature = "alloc")]
        {
            &self.buffer
        }
        #[cfg(not(feature = "alloc"))]
        {
            &[]
        }
    }
}
impl FromIterator<ViewPathSegment<MaybeOwnedStorage>> for MaybeOwnedBuffer {
    #[allow(unused_mut)]
    fn from_iter<T: IntoIterator<Item = ViewPathSegment<MaybeOwnedStorage>>>(_iter: T) -> Self {
        let mut v = Self::new();
        #[cfg(feature = "alloc")]
        {
            v.buffer = _iter.into_iter().collect();
        }
        #[cfg(not(feature = "alloc"))]
        {
            v.len.0 .0 += _iter.into_iter().count();
        }
        v
    }
}

/// Path storage that can allocate.
///
/// Stores nothing if the `alloc` feature isn't enabled.
#[derive(Debug, Clone, Copy)]
pub struct MaybeOwnedStorage(());
impl BufferStorage for MaybeOwnedStorage {
    type Buffer = MaybeOwnedBuffer;

    #[allow(unused_mut)]
    fn into_owned_path<L: LinkStorage>(
        path: ViewPath<'_, StorageConfig<L, Self>>,
    ) -> ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>> {
        let ViewPathState { link, mut local } = path.0;
        if let Some((link, _)) = link {
            #[cfg(feature = "alloc")]
            {
                // Here: |  Link  [1, 2, 3, 4]  |  Local [5, 6, 7, 8]  |
                local.buffer.reverse();
                // Here: |  Link  [1, 2, 3, 4]  |  Local [8, 7, 6, 5]  |
                local
                    .buffer
                    .extend(link.reversed_segments().map(|segment| segment.into_owned()));
                // Here: |  Local [8, 7, 6, 5, 4, 3, 2, 1]  |
                local.buffer.reverse();
                // Here: |  Local [1, 2, 3, 4, 5, 6, 7, 8]  |
            }
            #[cfg(not(feature = "alloc"))]
            {
                local.len.0 .0 += link.len();
            }
        }
        ViewPath(ViewPathState { link: None, local })
    }
    fn into_maybe_owned_path<L: LinkStorage>(
        path: ViewPath<'_, StorageConfig<L, Self>>,
    ) -> ViewPath<'_, StorageConfig<L, MaybeOwnedStorage>> {
        path
    }

    fn as_path_ref<'a, L: LinkStorage>(
        path: &'a ViewPath<'a, StorageConfig<L, Self>>,
    ) -> ViewPathRef<'a> {
        ViewPathRef { inner: path }
    }
}

/// Path storage that can't contain anything. A [`ViewPath`] with this storage
/// must be empty unless it links to another path.
#[derive(Debug, Clone, Copy)]
pub struct EmptyStorage(());
impl BufferStorage for EmptyStorage {
    type Buffer = ();

    fn into_owned_path<L: LinkStorage>(
        path: ViewPath<'_, StorageConfig<L, Self>>,
    ) -> ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>> {
        path.as_ref().to_owned()
    }
    fn into_maybe_owned_path<L: LinkStorage>(
        path: ViewPath<'_, StorageConfig<L, Self>>,
    ) -> ViewPath<'_, StorageConfig<L, MaybeOwnedStorage>> {
        ViewPath(path.0.map_state(|()| MaybeOwnedBuffer::new()))
    }

    fn as_path_ref<'a, L: LinkStorage>(
        path: &'a ViewPath<'a, StorageConfig<L, Self>>,
    ) -> ViewPathRef<'a> {
        ViewPathRef { inner: path }
    }
}

/// Path storage that is always [`Copy`].
#[derive(Debug, Clone, Copy)]
pub struct CopyableStorage(());
impl BufferStorage for CopyableStorage {
    type Buffer = ViewPathSegment<CopyableStorage>;

    fn into_owned_path<L: LinkStorage>(
        path: ViewPath<'_, StorageConfig<L, Self>>,
    ) -> ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>> {
        path.as_ref().to_owned()
    }
    fn into_maybe_owned_path<L: LinkStorage>(
        path: ViewPath<'_, StorageConfig<L, Self>>,
    ) -> ViewPath<'_, StorageConfig<L, MaybeOwnedStorage>> {
        ViewPath(
            path.0
                .map_state(|state| MaybeOwnedBuffer::single(state.into_owned())),
        )
    }

    fn as_path_ref<'a, L: LinkStorage>(
        path: &'a ViewPath<'a, StorageConfig<L, Self>>,
    ) -> ViewPathRef<'a> {
        ViewPathRef { inner: path }
    }
}

/// Path storage that is always [`Copy`] and uses heterogeneous type lists.
///
/// The `L` type parameter represents the type level list that is used to store
/// the path.
pub struct TypeListStorage<L> {
    marker: PhantomData<fn() -> L>,
}
impl<L> fmt::Debug for TypeListStorage<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopyableTypeListStorage").finish()
    }
}
impl<L> Clone for TypeListStorage<L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<L> Copy for TypeListStorage<L> {}
impl<L> BufferStorage for TypeListStorage<L>
where
    L: ViewPathTypeList + Clone,
{
    type Buffer = TypeListBuffer<L>;

    fn into_owned_path<LS: LinkStorage>(
        path: ViewPath<'_, StorageConfig<LS, Self>>,
    ) -> ViewPath<'static, StorageConfig<HasLink<false>, MaybeOwnedStorage>> {
        path.as_ref().to_owned()
    }
    fn into_maybe_owned_path<LS: LinkStorage>(
        path: ViewPath<'_, StorageConfig<LS, Self>>,
    ) -> ViewPath<'_, StorageConfig<LS, MaybeOwnedStorage>> {
        ViewPath(
            path.0
                .map_state(|list| list.iter().map(|segment| segment.into_owned()).collect()),
        )
    }

    fn as_path_ref<'a, LS: LinkStorage>(
        path: &'a ViewPath<'a, StorageConfig<LS, Self>>,
    ) -> ViewPathRef<'a> {
        ViewPathRef { inner: path }
    }
}

/// Provides trait implementations for a heterogeneous type list.
#[derive(Clone, Copy)]
pub struct TypeListBuffer<L: ?Sized>(pub(super) L);
impl<L> TypeListBuffer<L>
where
    L: ViewPathTypeList + ?Sized,
{
    fn iter(&self) -> impl Iterator<Item = ViewPathSegment<CopyableStorage>> + '_ {
        struct Iter<'a, L: ?Sized> {
            list: &'a L,
            /// Number of items from the start that has been read.
            read_start: usize,
        }
        impl<L> Iterator for Iter<'_, L>
        where
            L: ViewPathTypeList + ?Sized,
        {
            type Item = ViewPathSegment<CopyableStorage>;

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.list.len().saturating_sub(self.read_start);
                (len, Some(len))
            }

            fn next(&mut self) -> Option<Self::Item> {
                let item = self.list.get(self.read_start)?;
                self.read_start += 1;
                Some(item)
            }
        }
        Iter {
            list: &self.0,
            read_start: 0,
        }
    }
}
impl<L> fmt::Debug for TypeListBuffer<L>
where
    L: ViewPathTypeList + ?Sized,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// A type that can provide a path segment. Stored using [`TypeListStorage`].
pub trait ViewPathTypeListIdent: Send + Sync + 'static {
    fn get_path(&self) -> ViewPathSegment<CopyableStorage>;
}
impl ViewPathTypeListIdent for ViewPathSegment<CopyableStorage> {
    fn get_path(&self) -> ViewPathSegment<CopyableStorage> {
        *self
    }
}

/// Implemented for lists of [`ViewPathTypeListIdent`] types.
#[allow(clippy::len_without_is_empty)]
pub trait ViewPathTypeList: Send + Sync + 'static {
    fn get(&self, index: usize) -> Option<ViewPathSegment<CopyableStorage>>;
    fn len(&self) -> usize;
}
/// Empty list.
impl ViewPathTypeList for () {
    fn get(&self, _: usize) -> Option<ViewPathSegment<CopyableStorage>> {
        None
    }
    fn len(&self) -> usize {
        0
    }
}
impl<A, B> ViewPathTypeList for (A, B)
where
    A: ViewPathTypeList,
    B: ViewPathTypeList,
{
    fn get(&self, index: usize) -> Option<ViewPathSegment<CopyableStorage>> {
        let a_len = self.0.len();
        if index < a_len {
            self.0.get(index)
        } else {
            self.1.get(index - a_len)
        }
    }
    fn len(&self) -> usize {
        self.0.len() + self.1.len()
    }
}
impl<T> ViewPathTypeList for T
where
    T: ViewPathTypeListIdent,
{
    fn get(&self, index: usize) -> Option<ViewPathSegment<CopyableStorage>> {
        if index == 0 {
            Some(self.get_path())
        } else {
            None
        }
    }
    fn len(&self) -> usize {
        1
    }
}

/// Some segments have been skipped.
#[derive(Debug, Clone)]
pub(crate) struct SkippedSegments(usize);
impl ViewPathTypeList for SkippedSegments {
    fn get(&self, index: usize) -> Option<ViewPathSegment<CopyableStorage>> {
        if index < self.0 {
            Some(ViewPathSegment::field("<unknown>"))
        } else {
            None
        }
    }
    fn len(&self) -> usize {
        self.0
    }
}
