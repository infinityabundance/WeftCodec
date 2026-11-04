//! Self-contained WFA1 transport archive (WEFTCODEC-FORMAT §7).
//!
//! ```text
//! "WFA1" || u32le version=1 || u32le entry_count ||
//!   entry_count * ( u64le frame_len || frame_bytes ) ||
//!   u32le archive_crc32c
//! ```
//!
//! Entries are topologically ordered; every dependency is an earlier entry.
//! Creation rejects missing dependencies and cycles. Decoding rejects bad
//! CRCs, truncation, duplicate hashes that are not byte-identical, and
//! non-topological order.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::hash::Hash32;
use crate::limits::{check_len, Limits};
use crate::modes;

/// Archive magic.
pub const ARCHIVE_MAGIC: [u8; 4] = *b"WFA1";
/// Archive version.
pub const ARCHIVE_VERSION: u32 = 1;
const ARCHIVE_HEADER_LEN: usize = 12;
const ARCHIVE_TRAILER_LEN: usize = 4;

/// One frame inside a parsed archive.
#[derive(Clone, Debug)]
pub struct ArchiveEntry<'a> {
    pub frame_bytes: &'a [u8],
    pub frame: Frame<'a>,
}

/// A parsed and structurally validated archive.
#[derive(Clone, Debug)]
pub struct Archive<'a> {
    pub entries: Vec<ArchiveEntry<'a>>,
}

/// A successfully decoded archive entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedEntry {
    pub content_hash: Hash32,
    pub bytes: Vec<u8>,
}

/// Create a self-contained archive from complete frame bytes.
///
/// Frames are deduplicated when their bytes are identical. Every dependency
/// hash must correspond to another supplied frame. The output order is a
/// deterministic topological order (Kahn's algorithm, smallest hash first).
pub fn create_archive(frames: &[Vec<u8>], limits: &Limits) -> Result<Vec<u8>> {
    check_len(
        frames.len() as u64,
        limits.max_archive_entries,
        "archive entry count",
    )?;
    if frames.len() > u32::MAX as usize {
        return Err(Error::limit_exceeded(
            "archive entry count does not fit u32",
        ));
    }

    struct Info {
        bytes: Vec<u8>,
        dependencies: Vec<Hash32>,
    }

    let mut by_hash: HashMap<Hash32, Info> = HashMap::new();
    for bytes in frames {
        let frame = Frame::parse(bytes, limits)?;
        let hash = frame.header.content_hash;
        let dependencies = frame
            .dependencies
            .iter()
            .map(|dep| dep.hash)
            .collect::<Vec<_>>();
        if let Some(existing) = by_hash.get(&hash) {
            if existing.bytes != *bytes {
                return Err(Error::malformed(format!(
                    "duplicate content hash {hash} with different frame bytes"
                )));
            }
            continue;
        }
        by_hash.insert(
            hash,
            Info {
                bytes: bytes.clone(),
                dependencies,
            },
        );
    }

    // Deterministic Kahn topological sort: dependency -> dependents.
    let mut indegree: HashMap<Hash32, usize> = HashMap::new();
    let mut dependents: HashMap<Hash32, Vec<Hash32>> = HashMap::new();
    for (hash, info) in &by_hash {
        indegree.entry(*hash).or_insert(0);
        for dep in &info.dependencies {
            if !by_hash.contains_key(dep) {
                return Err(Error::dependency_missing(format!(
                    "frame {hash} depends on {dep}, which is not in the archive"
                )));
            }
            *indegree.entry(*hash).or_insert(0) += 1;
            dependents.entry(*dep).or_default().push(*hash);
        }
    }
    let mut ready: BTreeSet<Hash32> = indegree
        .iter()
        .filter_map(|(hash, degree)| if *degree == 0 { Some(*hash) } else { None })
        .collect();
    let mut order: Vec<Hash32> = Vec::with_capacity(by_hash.len());
    while let Some(hash) = ready.pop_first() {
        order.push(hash);
        if let Some(children) = dependents.get(&hash) {
            for child in children {
                let degree = indegree
                    .get_mut(child)
                    .ok_or_else(|| Error::malformed("archive graph inconsistency"))?;
                *degree = degree
                    .checked_sub(1)
                    .ok_or_else(|| Error::malformed("archive indegree underflow"))?;
                if *degree == 0 {
                    ready.insert(*child);
                }
            }
        }
    }
    if order.len() != by_hash.len() {
        return Err(Error::dependency_cycle(
            "archive dependency graph contains a cycle",
        ));
    }

    let mut out = Vec::new();
    out.extend_from_slice(&ARCHIVE_MAGIC);
    out.extend_from_slice(&ARCHIVE_VERSION.to_le_bytes());
    out.extend_from_slice(&(order.len() as u32).to_le_bytes());
    for hash in &order {
        let info = by_hash
            .get(hash)
            .ok_or_else(|| Error::malformed("archive entry vanished"))?;
        let len = info.bytes.len() as u64;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&info.bytes);
    }
    check_len(
        out.len() as u64,
        limits.max_total_fetched_bytes,
        "archive size",
    )?;
    let crc = crc32c::crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    Ok(out)
}

/// Parse and structurally validate an archive. Does not decode frame bodies.
pub fn parse_archive<'a>(bytes: &'a [u8], limits: &Limits) -> Result<Archive<'a>> {
    if bytes.len() < ARCHIVE_HEADER_LEN + ARCHIVE_TRAILER_LEN {
        return Err(Error::malformed(
            "archive is shorter than its header and trailer",
        ));
    }
    if bytes[0..4] != ARCHIVE_MAGIC {
        return Err(Error::malformed("bad archive magic (expected WFA1)"));
    }
    let version = le_u32(bytes, 4);
    if version != ARCHIVE_VERSION {
        return Err(Error::unsupported_version(format!(
            "unsupported archive version {version}"
        )));
    }
    let entry_count = le_u32(bytes, 8);
    check_len(
        u64::from(entry_count),
        limits.max_archive_entries,
        "archive entry count",
    )?;
    let crc_at = bytes.len() - ARCHIVE_TRAILER_LEN;
    let stored_crc = le_u32(bytes, crc_at);
    let computed_crc = crc32c::crc32c(&bytes[..crc_at]);
    if stored_crc != computed_crc {
        return Err(Error::corrupt("archive CRC32C mismatch"));
    }

    // Bound the entry vector by the smallest possible per-entry size so we
    // never allocate based on an untrusted count.
    let minimum_entries_len = u64::from(entry_count)
        .checked_mul(8)
        .ok_or_else(|| Error::malformed("archive entry count overflow"))?;
    if minimum_entries_len > (crc_at - ARCHIVE_HEADER_LEN) as u64 {
        return Err(Error::malformed(
            "archive entry count cannot fit in the remaining bytes",
        ));
    }

    let mut pos = ARCHIVE_HEADER_LEN;
    let mut entries = Vec::with_capacity((entry_count as usize).min(1024));
    for index in 0..entry_count as usize {
        if pos.checked_add(8).map(|end| end > crc_at).unwrap_or(true) {
            return Err(Error::malformed(format!(
                "archive entry {index} is truncated"
            )));
        }
        let frame_len = le_u64(bytes, pos);
        pos += 8;
        if frame_len < 100 {
            return Err(Error::malformed(format!(
                "archive entry {index} frame_len {frame_len} is below 100"
            )));
        }
        let frame_len_usize = usize::try_from(frame_len)
            .map_err(|_| Error::malformed("archive frame length does not fit usize"))?;
        let end = pos
            .checked_add(frame_len_usize)
            .ok_or_else(|| Error::malformed("archive entry extent overflow"))?;
        if end > crc_at {
            return Err(Error::malformed(format!(
                "archive entry {index} extends past the archive body"
            )));
        }
        let frame_bytes = &bytes[pos..end];
        let frame = Frame::parse(frame_bytes, limits)?;
        entries.push(ArchiveEntry { frame_bytes, frame });
        pos = end;
    }
    if pos != crc_at {
        return Err(Error::malformed(
            "archive has trailing bytes or entry_count undercounts the entries",
        ));
    }

    // Duplicate hashes are allowed only when byte-identical.
    let mut seen_bytes: HashMap<Hash32, &[u8]> = HashMap::new();
    for entry in &entries {
        let hash = entry.frame.header.content_hash;
        if let Some(existing) = seen_bytes.get(&hash) {
            if *existing != entry.frame_bytes {
                return Err(Error::corrupt(format!(
                    "archive contains two different frames with content hash {hash}"
                )));
            }
        } else {
            seen_bytes.insert(hash, entry.frame_bytes);
        }
    }

    // Topological order: every dependency must appear in an earlier entry.
    let all_hashes: HashSet<Hash32> = entries
        .iter()
        .map(|entry| entry.frame.header.content_hash)
        .collect();
    let mut earlier: HashSet<Hash32> = HashSet::new();
    for (index, entry) in entries.iter().enumerate() {
        for dep in &entry.frame.dependencies {
            if !earlier.contains(&dep.hash) {
                if all_hashes.contains(&dep.hash) {
                    return Err(Error::malformed(format!(
                        "archive entry {index} depends on a frame that appears later (not topological)"
                    )));
                }
                return Err(Error::dependency_missing(format!(
                    "archive entry {index} depends on {0}, which is not present",
                    dep.hash
                )));
            }
        }
        earlier.insert(entry.frame.header.content_hash);
    }

    Ok(Archive { entries })
}

/// Decode every archive entry in order, using only in-archive plaintext.
///
/// Duplicate byte-identical entries are decoded once. The returned vector is
/// in topological order.
pub fn decode_archive(bytes: &[u8], limits: &Limits) -> Result<Vec<DecodedEntry>> {
    let archive = parse_archive(bytes, limits)?;
    let mut plaintext: HashMap<Hash32, Vec<u8>> = HashMap::new();
    let mut stored_frames: HashMap<Hash32, &[u8]> = HashMap::new();
    let mut decoded_entries: Vec<DecodedEntry> = Vec::new();
    let mut cumulative_decoded: u64 = 0;

    for entry in &archive.entries {
        let hash = entry.frame.header.content_hash;
        if let Some(previous_frame) = stored_frames.get(&hash) {
            if *previous_frame != entry.frame_bytes {
                return Err(Error::corrupt(format!(
                    "archive has conflicting frames for content hash {hash}"
                )));
            }
            continue;
        }
        let resolver = ArchiveMapResolver { map: &plaintext };
        let plain = modes::decode_bytes(entry.frame_bytes, &resolver, limits)?;
        cumulative_decoded = cumulative_decoded
            .checked_add(plain.len() as u64)
            .ok_or_else(|| Error::limit_exceeded("archive decoded byte count overflow"))?;
        check_len(
            cumulative_decoded,
            limits.max_decoded_bytes,
            "cumulative archive decoded data",
        )?;
        stored_frames.insert(hash, entry.frame_bytes);
        plaintext.insert(hash, plain.clone());
        decoded_entries.push(DecodedEntry {
            content_hash: hash,
            bytes: plain,
        });
    }
    Ok(decoded_entries)
}

struct ArchiveMapResolver<'a> {
    map: &'a HashMap<Hash32, Vec<u8>>,
}

impl crate::dependency::Resolver for ArchiveMapResolver<'_> {
    fn resolve(&self, hash: &Hash32, decoded_len: u64, _dep_domain: u8) -> Result<Vec<u8>> {
        let bytes = self.map.get(hash).ok_or_else(|| {
            Error::dependency_missing(format!(
                "archive dependency {hash} has not been decoded yet"
            ))
        })?;
        if bytes.len() as u64 != decoded_len {
            return Err(Error::dependency_mismatch(
                "archive dependency length mismatch",
            ));
        }
        if crate::hash::hash_domain(crate::hash::DOMAIN_CHUNK, bytes) != *hash {
            return Err(Error::dependency_mismatch(
                "archive dependency hash mismatch",
            ));
        }
        Ok(bytes.clone())
    }
}

fn le_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn le_u64(bytes: &[u8], off: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[off..off + 8]);
    u64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dependency::MapFrameSource;
    use crate::modes;

    #[test]
    fn roundtrip_single_frame() {
        let frame = modes::encode_raw(b"archive payload", None, &Limits::default()).unwrap();
        let archive = create_archive(&[frame.clone()], &Limits::default()).unwrap();
        let decoded = decode_archive(&archive, &Limits::default()).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].bytes, b"archive payload");
    }

    #[test]
    fn missing_dependency_rejected() {
        let base = modes::encode_raw(b"base", None, &Limits::default()).unwrap();
        let dep = modes::encode_delta(b"base plus", b"base", &Limits::default()).unwrap();
        assert_eq!(
            create_archive(&[dep.clone()], &Limits::default())
                .unwrap_err()
                .class(),
            crate::error::ErrorClass::DependencyMissing
        );
        // Including the base works.
        let archive = create_archive(&[base, dep], &Limits::default()).unwrap();
        assert!(decode_archive(&archive, &Limits::default()).is_ok());
    }

    #[test]
    fn cycle_rejected() {
        // Build two frames that reference each other's content hashes.
        let dep_for = |hash: Hash32| {
            crate::frame::Dependency::new(hash, 1, crate::frame::DEP_DOMAIN_PAYLOAD, 0)
        };
        let content_hash = Hash32::ZERO;
        let parts_a = crate::frame::FrameParts {
            format_major: 1,
            format_minor: 0,
            domain: 0,
            mode: crate::frame::Mode::Reference,
            flags: 0,
            level: 0,
            decoded_len: 1,
            content_hash,
            dependencies: &[dep_for(Hash32::from_bytes([2u8; 32]))],
            chunks: &[],
            body: &[],
        };
        let a = crate::frame::encode_frame(&parts_a, &Limits::default()).unwrap();
        let parts_b = crate::frame::FrameParts {
            format_major: 1,
            format_minor: 0,
            domain: 0,
            mode: crate::frame::Mode::Reference,
            flags: 0,
            level: 0,
            decoded_len: 1,
            content_hash: Hash32::from_bytes([2u8; 32]),
            dependencies: &[dep_for(Hash32::ZERO)],
            chunks: &[],
            body: &[],
        };
        let b = crate::frame::encode_frame(&parts_b, &Limits::default()).unwrap();
        let err = create_archive(&[a, b], &Limits::default()).unwrap_err();
        assert_eq!(err.class(), crate::error::ErrorClass::DependencyCycle);
        let _ = MapFrameSource::new();
    }
}
