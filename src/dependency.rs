//! Dependency resolution contracts (WEFTCODEC-FORMAT §5) and recursive frame
//! DAG decoding.
//!
//! Resolvers are the only door through which external bytes enter a decode.
//! The decoder verifies the returned length and domain-separated hash at every
//! materialization boundary; a resolver that lies yields `DependencyMismatch`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::hash::{hash_domain, Hash32, DOMAIN_CHUNK};
use crate::limits::{check_len, Limits};
use crate::modes;

/// Resolution contract from §5.
pub trait Resolver {
    /// Return the dependency plaintext for `hash`, which must be exactly
    /// `decoded_len` bytes with `H("weft/chunk/v1", bytes) == hash`.
    ///
    /// Implementations should fail with `DependencyMissing` when the bytes are
    /// unknown and `DependencyDenied` when policy refuses them.
    fn resolve(&self, hash: &Hash32, decoded_len: u64, dep_domain: u8) -> Result<Vec<u8>>;

    /// Return the complete frame for `content_hash` when this resolver can
    /// supply frames as well as plaintext.
    ///
    /// Mode 11 uses this to verify the base frame's own dtype/shape
    /// descriptor. The default reports that frames are unavailable; the DAG
    /// resolver overrides it. Plaintext resolution still goes through
    /// [`Resolver::resolve`], which enforces depth, cycle, and hash checks.
    fn resolve_frame(&self, _content_hash: &Hash32) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }
}

/// In-memory resolver keyed by digest.
#[derive(Clone, Debug, Default)]
pub struct MapResolver {
    entries: HashMap<Hash32, Vec<u8>>,
}

impl MapResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert bytes under an explicit digest.
    pub fn insert(&mut self, hash: Hash32, bytes: Vec<u8>) {
        self.entries.insert(hash, bytes);
    }

    /// Insert plaintext, computing its chunk-domain digest.
    pub fn insert_bytes(&mut self, bytes: Vec<u8>) -> Hash32 {
        let hash = hash_domain(DOMAIN_CHUNK, &bytes);
        self.entries.insert(hash, bytes);
        hash
    }

    pub fn get(&self, hash: &Hash32) -> Option<&[u8]> {
        self.entries.get(hash).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Resolver for MapResolver {
    fn resolve(&self, hash: &Hash32, decoded_len: u64, _dep_domain: u8) -> Result<Vec<u8>> {
        let bytes = self
            .entries
            .get(hash)
            .ok_or_else(|| Error::dependency_missing(format!("dependency {hash} not in map")))?;
        verify_dependency(hash, decoded_len, bytes)?;
        Ok(bytes.clone())
    }
}

/// Resolver backed by a directory of files named with lowercase hex digests.
/// Each file contains exactly the dependency plaintext.
#[derive(Clone, Debug)]
pub struct DirectoryResolver {
    dir: PathBuf,
}

impl DirectoryResolver {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        DirectoryResolver { dir: dir.into() }
    }

    pub fn directory(&self) -> &Path {
        &self.dir
    }
}

impl Resolver for DirectoryResolver {
    fn resolve(&self, hash: &Hash32, decoded_len: u64, _dep_domain: u8) -> Result<Vec<u8>> {
        let path = self.dir.join(hash.to_hex());
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::dependency_missing(format!(
                    "dependency file {} not found",
                    path.display()
                )))
            }
            Err(err) => {
                return Err(Error::io(format!(
                    "cannot stat dependency {}: {err}",
                    path.display()
                )))
            }
        };
        if !metadata.is_file() {
            return Err(Error::dependency_missing(format!(
                "dependency path {} is not a regular file",
                path.display()
            )));
        }
        if metadata.len() != decoded_len {
            return Err(Error::dependency_mismatch(format!(
                "dependency file {} has length {} but declared {decoded_len}",
                path.display(),
                metadata.len()
            )));
        }
        let bytes = fs::read(&path).map_err(|err| {
            Error::io(format!("cannot read dependency {}: {err}", path.display()))
        })?;
        verify_dependency(hash, decoded_len, &bytes)?;
        Ok(bytes)
    }
}

/// Resolver over the decoded plaintexts of a self-contained WFA1 archive.
#[derive(Clone, Debug, Default)]
pub struct ArchiveResolver {
    inner: MapResolver,
}

impl ArchiveResolver {
    pub fn new(entries: impl IntoIterator<Item = (Hash32, Vec<u8>)>) -> Self {
        let mut inner = MapResolver::new();
        for (hash, bytes) in entries {
            inner.insert(hash, bytes);
        }
        ArchiveResolver { inner }
    }

    /// Decode every archive entry (in topological order) and index its
    /// plaintext by content hash.
    pub fn from_archive(bytes: &[u8], limits: &Limits) -> Result<Self> {
        let decoded = crate::archive::decode_archive(bytes, limits)?;
        Ok(ArchiveResolver::new(
            decoded
                .into_iter()
                .map(|entry| (entry.content_hash, entry.bytes)),
        ))
    }
}

impl Resolver for ArchiveResolver {
    fn resolve(&self, hash: &Hash32, decoded_len: u64, dep_domain: u8) -> Result<Vec<u8>> {
        self.inner.resolve(hash, decoded_len, dep_domain)
    }
}

/// Source of dependency **frames** for DAG reconstruction.
pub trait FrameSource {
    /// Return the complete frame whose `content_hash` is `content_hash`.
    fn get_frame(&self, content_hash: &Hash32) -> Result<Option<Vec<u8>>>;
}

/// Frame source backed by a directory of complete frames, each named by its
/// lowercase content hash. Files are read lazily so the DAG fetch bound counts
/// only frames actually traversed.
#[derive(Clone, Debug)]
pub struct DirectoryFrameSource {
    dir: PathBuf,
    max_frame_bytes: u64,
}

impl DirectoryFrameSource {
    pub fn new(dir: impl Into<PathBuf>, max_frame_bytes: u64) -> Self {
        DirectoryFrameSource {
            dir: dir.into(),
            max_frame_bytes,
        }
    }

    pub fn directory(&self) -> &Path {
        &self.dir
    }
}

impl FrameSource for DirectoryFrameSource {
    fn get_frame(&self, content_hash: &Hash32) -> Result<Option<Vec<u8>>> {
        let path = self.dir.join(content_hash.to_hex());
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(Error::io(format!(
                    "cannot stat frame {}: {err}",
                    path.display()
                )))
            }
        };
        if !metadata.is_file() {
            return Ok(None);
        }
        check_len(metadata.len(), self.max_frame_bytes, "dependency frame")?;
        let bytes = fs::read(&path)
            .map_err(|err| Error::io(format!("cannot read frame {}: {err}", path.display())))?;
        check_len(bytes.len() as u64, self.max_frame_bytes, "dependency frame")?;
        Ok(Some(bytes))
    }
}

/// In-memory frame source keyed by frame content hash.
#[derive(Clone, Debug, Default)]
pub struct MapFrameSource {
    frames: HashMap<Hash32, Vec<u8>>,
}

impl MapFrameSource {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a frame, parsing it to learn its content hash.
    pub fn insert_frame(&mut self, bytes: Vec<u8>) -> Result<Hash32> {
        self.insert_frame_with_limits(bytes, &Limits::default())
    }

    /// Insert a frame using explicit limits for the structural parse.
    pub fn insert_frame_with_limits(&mut self, bytes: Vec<u8>, limits: &Limits) -> Result<Hash32> {
        let frame = Frame::parse(&bytes, limits)?;
        let hash = frame.header.content_hash;
        if let Some(existing) = self.frames.get(&hash) {
            if existing != &bytes {
                return Err(Error::corrupt(
                    "frame source has two different frames with the same content hash",
                ));
            }
        }
        self.frames.insert(hash, bytes);
        Ok(hash)
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

impl FrameSource for MapFrameSource {
    fn get_frame(&self, content_hash: &Hash32) -> Result<Option<Vec<u8>>> {
        Ok(self.frames.get(content_hash).cloned())
    }
}

#[derive(Default)]
struct DagStats {
    fetched: u64,
    decoded: u64,
}

struct DagResolver<'a> {
    source: &'a dyn FrameSource,
    limits: &'a Limits,
    visiting: &'a RefCell<HashSet<Hash32>>,
    stats: &'a RefCell<DagStats>,
    depth: u64,
}

impl Resolver for DagResolver<'_> {
    fn resolve(&self, hash: &Hash32, decoded_len: u64, _dep_domain: u8) -> Result<Vec<u8>> {
        let depth = self
            .depth
            .checked_add(1)
            .ok_or_else(|| Error::limit_exceeded("dependency depth overflow"))?;
        if depth > self.limits.max_dependency_depth {
            return Err(Error::limit_exceeded(format!(
                "dependency depth {depth} exceeds limit {}",
                self.limits.max_dependency_depth
            )));
        }
        if self.visiting.borrow().contains(hash) {
            return Err(Error::dependency_cycle(format!(
                "dependency cycle re-entered frame {hash}"
            )));
        }
        let frame_bytes = self.source.get_frame(hash)?.ok_or_else(|| {
            Error::dependency_missing(format!("frame source has no frame for {hash}"))
        })?;
        {
            let mut stats = self.stats.borrow_mut();
            stats.fetched = stats
                .fetched
                .checked_add(frame_bytes.len() as u64)
                .ok_or_else(|| Error::limit_exceeded("fetched byte count overflow"))?;
            if stats.fetched > self.limits.max_total_fetched_bytes {
                return Err(Error::limit_exceeded(format!(
                    "total fetched bytes {} exceed limit {}",
                    stats.fetched, self.limits.max_total_fetched_bytes
                )));
            }
        }
        let frame = Frame::parse(&frame_bytes, self.limits)?;
        if frame.header.content_hash != *hash {
            return Err(Error::dependency_mismatch(format!(
                "frame at {hash} declares content hash {}",
                frame.header.content_hash
            )));
        }

        self.visiting.borrow_mut().insert(*hash);
        let child = DagResolver {
            source: self.source,
            limits: self.limits,
            visiting: self.visiting,
            stats: self.stats,
            depth,
        };
        let decoded = modes::decode_bytes(&frame_bytes, &child, self.limits);
        self.visiting.borrow_mut().remove(hash);
        let decoded = decoded?;

        if decoded.len() as u64 != decoded_len {
            return Err(Error::dependency_mismatch(format!(
                "dependency frame {hash} decodes to {} bytes, declared {decoded_len}",
                decoded.len()
            )));
        }
        if hash_domain(DOMAIN_CHUNK, &decoded) != *hash {
            return Err(Error::dependency_mismatch(
                "dependency frame plaintext hash mismatch",
            ));
        }
        {
            let mut stats = self.stats.borrow_mut();
            stats.decoded = stats
                .decoded
                .checked_add(decoded.len() as u64)
                .ok_or_else(|| Error::limit_exceeded("decoded byte count overflow"))?;
            if stats.decoded > self.limits.max_decoded_bytes {
                return Err(Error::limit_exceeded(format!(
                    "cumulative decoded dependency bytes {} exceed limit {}",
                    stats.decoded, self.limits.max_decoded_bytes
                )));
            }
        }
        Ok(decoded)
    }

    fn resolve_frame(&self, content_hash: &Hash32) -> Result<Option<Vec<u8>>> {
        let frame = self.source.get_frame(content_hash)?;
        if let Some(bytes) = &frame {
            check_len(
                bytes.len() as u64,
                self.limits.max_frame_bytes,
                "dependency frame",
            )?;
        }
        Ok(frame)
    }
}

/// Recursively decode a frame DAG.
///
/// Dependency frames are fetched by content hash through `source`, decoded
/// recursively with a visiting set (cycle detection), and verified at every
/// boundary. Depth, fetched bytes, and cumulative decoded bytes are enforced.
pub fn decode_frame_dag(
    source: &dyn FrameSource,
    root_frame: &[u8],
    limits: &Limits,
) -> Result<Vec<u8>> {
    let root = Frame::parse(root_frame, limits)?;
    let root_hash = root.header.content_hash;
    let visiting = RefCell::new(HashSet::new());
    let stats = RefCell::new(DagStats::default());
    {
        let mut stats = stats.borrow_mut();
        stats.fetched = root_frame.len() as u64;
        check_len(
            stats.fetched,
            limits.max_total_fetched_bytes,
            "total fetched bytes",
        )?;
    }
    visiting.borrow_mut().insert(root_hash);
    let resolver = DagResolver {
        source,
        limits,
        visiting: &visiting,
        stats: &stats,
        depth: 0,
    };
    let decoded = modes::decode_bytes(root_frame, &resolver, limits);
    visiting.borrow_mut().remove(&root_hash);
    decoded
}

fn verify_dependency(hash: &Hash32, decoded_len: u64, bytes: &[u8]) -> Result<()> {
    if bytes.len() as u64 != decoded_len {
        return Err(Error::dependency_mismatch(format!(
            "resolved dependency length {} differs from declared {decoded_len}",
            bytes.len()
        )));
    }
    if hash_domain(DOMAIN_CHUNK, bytes) != *hash {
        return Err(Error::dependency_mismatch(format!(
            "resolved dependency {hash} fails hash verification"
        )));
    }
    Ok(())
}
