//! `weft.transport.v1` — a transport-aligned envelope around existing frames.
//!
//! The normative specification is `spec/WEFT-TRANSPORT.md`. This module is a
//! **layout**, not a transport: it places complete, unmodified WFT1 frames at
//! offsets a caller can name with an alignment (512, 4096, or 65536 bytes) so
//! one section can be fetched with one aligned range read.
//!
//! ```text
//! 4096-byte header || 56-byte entry per section || zero padding || sections
//! ```
//!
//! Enclosed frames keep their bytes, mode numbers, and content roots. Padding
//! is zero and is outside every frame, so it is not hashed.
//!
//! # Non-claims
//!
//! No DMA, GPU memory, RDMA, kernel-bypass, deployed-3FS, throughput, or
//! latency claim follows from this module. Alignment is necessary but not
//! sufficient for a zero-copy decoder; the acceptance court measures parse
//! cost, range-read bytes, wall time, and padding overhead, and says so.

use std::fmt;

use crate::error::{Error, Result};
use crate::frame::{self, Frame, FrameHeader};
use crate::hash::{DomainHasher, Hash32};
use crate::limits::{check_len, Limits};

/// Container magic.
pub const MAGIC: [u8; 4] = *b"WFTT";
/// Format major version.
pub const FORMAT_MAJOR: u16 = 1;
/// Format minor version.
pub const FORMAT_MINOR: u16 = 0;
/// Fixed header length. The section table starts here in v1.
pub const HEADER_LEN: usize = 4096;
/// One section-table entry, in bytes.
pub const SECTION_ENTRY_LEN: usize = 56;
/// Maximum section count.
pub const MAX_SECTIONS: u32 = 4096;
/// Allowed section alignments.
pub const ALIGNMENTS: [u32; 3] = [512, 4096, 65536];
/// Hash domain for the container-level payload digest.
pub const DOMAIN_TRANSPORT: &str = "weft/transport/v1";

/// Why a section exists in the container.
///
/// A role is a hint. It is not an input to any content root, and it does not
/// change the enclosed frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionRole {
    /// The artifact the caller asked for.
    Payload,
    /// A frame the payload frame depends on.
    Dependency,
    /// A frame describing the payload (manifest, descriptor, index).
    Metadata,
    /// A provenance or receipt frame.
    Provenance,
}

impl SectionRole {
    pub const fn as_u8(self) -> u8 {
        match self {
            SectionRole::Payload => 0,
            SectionRole::Dependency => 1,
            SectionRole::Metadata => 2,
            SectionRole::Provenance => 3,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            SectionRole::Payload => "payload",
            SectionRole::Dependency => "dependency",
            SectionRole::Metadata => "metadata",
            SectionRole::Provenance => "provenance",
        }
    }

    pub fn from_u8(value: u8) -> Result<SectionRole> {
        match value {
            0 => Ok(SectionRole::Payload),
            1 => Ok(SectionRole::Dependency),
            2 => Ok(SectionRole::Metadata),
            3 => Ok(SectionRole::Provenance),
            other => Err(Error::malformed(format!(
                "transport section role {other} is not defined in v1"
            ))),
        }
    }

    /// Parse a CLI spelling (`payload`, `dependency`, `metadata`,
    /// `provenance`).
    pub fn parse(text: &str) -> Result<SectionRole> {
        match text {
            "payload" => Ok(SectionRole::Payload),
            "dependency" => Ok(SectionRole::Dependency),
            "metadata" => Ok(SectionRole::Metadata),
            "provenance" => Ok(SectionRole::Provenance),
            other => Err(Error::parse(format!(
                "unknown transport role `{other}` (expected payload|dependency|metadata|provenance)"
            ))),
        }
    }
}

impl fmt::Display for SectionRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One frame to place in a container.
#[derive(Clone, Copy, Debug)]
pub struct TransportFrame<'a> {
    pub bytes: &'a [u8],
    pub role: SectionRole,
}

impl<'a> TransportFrame<'a> {
    pub const fn new(bytes: &'a [u8], role: SectionRole) -> Self {
        TransportFrame { bytes, role }
    }
}

/// A validated section-table entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section {
    /// Absolute byte offset in the container.
    pub offset: u64,
    /// Exact enclosed frame length. Section padding is outside this range.
    pub length: u64,
    /// Declared alignment (one of [`ALIGNMENTS`]).
    pub alignment: u32,
    /// The enclosed frame's mode byte.
    pub mode: u8,
    /// Why the frame is present.
    pub role: SectionRole,
    /// The enclosed frame's content root.
    pub root: Hash32,
}

impl Section {
    /// End of the enclosed frame bytes.
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.length)
    }
}

/// Parsed fixed header. The table is parsed separately so a reader can fetch
/// exactly two ranges before touching any section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransportHeader {
    pub flags: u32,
    pub section_count: u32,
    pub section_table_offset: u64,
    pub total_len: u64,
    pub payload_hash: Hash32,
}

/// A structurally validated container borrowing its bytes.
#[derive(Clone, Debug)]
pub struct Transport<'a> {
    pub header: TransportHeader,
    pub sections: Vec<Section>,
    pub bytes: &'a [u8],
}

/// Byte-range source used for single-section fetches.
///
/// `weftcodec` does not depend on a storage crate; a caller wires this to the
/// backend's `BlobPlane::get_range` (or to `read_into` for a shared region).
/// The court's implementation issues exactly one `get_range` per section.
pub trait RangeReader {
    fn read_range(&self, offset: u64, len: u64) -> Result<Vec<u8>>;
}

/// Fetch exactly one section: one range read of `[offset, offset + length)`.
pub fn fetch_section<R: RangeReader + ?Sized>(reader: &R, section: &Section) -> Result<Vec<u8>> {
    let bytes = reader.read_range(section.offset, section.length)?;
    if bytes.len() as u64 != section.length {
        return Err(Error::corrupt(format!(
            "transport section at {} returned {} bytes but the table records {}",
            section.offset,
            bytes.len(),
            section.length
        )));
    }
    Ok(bytes)
}

/// Whether `alignment` is a v1 alignment.
pub fn allowed_alignment(alignment: u32) -> bool {
    ALIGNMENTS.contains(&alignment)
}

fn checked_align_up(value: u64, alignment: u32) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::malformed(format!(
            "transport alignment {alignment} is not a power of two"
        )));
    }
    let alignment = u64::from(alignment);
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| Error::limit_exceeded("transport alignment overflows u64"))
    }
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(buf)
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Pack `frames` into a `weft.transport.v1` container at `alignment`.
///
/// Sections appear in the supplied order. The output is deterministic: the
/// same ordered frames and alignment always produce the same bytes.
pub fn create(frames: &[TransportFrame<'_>], alignment: u32, limits: &Limits) -> Result<Vec<u8>> {
    if frames.is_empty() {
        return Err(Error::malformed(
            "a transport container needs at least one section",
        ));
    }
    if frames.len() > MAX_SECTIONS as usize {
        return Err(Error::limit_exceeded(format!(
            "transport section count {} exceeds the {MAX_SECTIONS} bound",
            frames.len()
        )));
    }
    if !allowed_alignment(alignment) {
        return Err(Error::malformed(format!(
            "transport alignment {alignment} is not one of 512, 4096, 65536"
        )));
    }
    let count = frames.len() as u32;
    let table_end = (HEADER_LEN as u64)
        .checked_add(
            u64::from(count)
                .checked_mul(SECTION_ENTRY_LEN as u64)
                .ok_or_else(|| Error::limit_exceeded("transport table length overflows u64"))?,
        )
        .ok_or_else(|| Error::limit_exceeded("transport table end overflows u64"))?;

    let mut cursor = table_end;
    let mut sections = Vec::with_capacity(frames.len());
    let mut lengths = Vec::with_capacity(frames.len());
    let mut hasher = DomainHasher::new(DOMAIN_TRANSPORT);
    for entry in frames {
        // The encoder accepts only complete frames. This is the same parse the
        // reader performs, so an invalid frame can never be sealed.
        let parsed = Frame::parse(entry.bytes, limits)?;
        cursor = checked_align_up(cursor, alignment)?;
        let length = entry.bytes.len() as u64;
        check_len(length, limits.max_frame_bytes, "transport section")?;
        sections.push(Section {
            offset: cursor,
            length,
            alignment,
            mode: parsed.header.mode.as_u8(),
            role: entry.role,
            root: parsed.header.content_hash,
        });
        lengths.push(length);
        hasher.update(entry.bytes);
        cursor = cursor
            .checked_add(length)
            .ok_or_else(|| Error::limit_exceeded("transport section end overflows u64"))?;
    }
    let total_len = checked_align_up(cursor, alignment)?;
    check_len(
        total_len,
        limits.max_total_fetched_bytes,
        "transport container",
    )?;
    let total_len_usize = usize::try_from(total_len)
        .map_err(|_| Error::limit_exceeded("transport container does not fit usize"))?;

    let mut out = vec![0u8; total_len_usize];
    out[0..4].copy_from_slice(&MAGIC);
    out[4..6].copy_from_slice(&FORMAT_MAJOR.to_le_bytes());
    out[6..8].copy_from_slice(&FORMAT_MINOR.to_le_bytes());
    out[8..12].copy_from_slice(&0u32.to_le_bytes());
    out[12..16].copy_from_slice(&count.to_le_bytes());
    out[16..24].copy_from_slice(&(HEADER_LEN as u64).to_le_bytes());
    out[24..32].copy_from_slice(&total_len.to_le_bytes());
    let digest = hasher.finalize();
    out[32..64].copy_from_slice(digest.as_bytes());
    // 64..HEADER_LEN stays zero.

    for (index, section) in sections.iter().enumerate() {
        let at = HEADER_LEN + index * SECTION_ENTRY_LEN;
        out[at..at + 8].copy_from_slice(&section.offset.to_le_bytes());
        out[at + 8..at + 16].copy_from_slice(&section.length.to_le_bytes());
        out[at + 16..at + 20].copy_from_slice(&section.alignment.to_le_bytes());
        out[at + 20] = section.mode;
        out[at + 21] = section.role.as_u8();
        // at+22..at+24 reserved zero.
        out[at + 24..at + 56].copy_from_slice(section.root.as_bytes());
        let start = usize::try_from(section.offset)
            .map_err(|_| Error::limit_exceeded("section offset does not fit usize"))?;
        let length = usize::try_from(lengths[index])
            .map_err(|_| Error::limit_exceeded("section length does not fit usize"))?;
        out[start..start + length].copy_from_slice(frames[index].bytes);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Parse and validate the fixed header.
///
/// `bytes` must contain the complete 4096-byte header. The physical length is
/// not compared to `total_len` here; a reader that has only the header and
/// table calls [`parse_metadata`] once it has the whole container, or performs
/// its own length check after the two range reads.
pub fn read_header(bytes: &[u8]) -> Result<TransportHeader> {
    if bytes.len() < HEADER_LEN {
        return Err(Error::malformed(format!(
            "transport header is {} bytes, shorter than the {HEADER_LEN}-byte minimum",
            bytes.len()
        )));
    }
    if bytes[0..4] != MAGIC {
        return Err(Error::malformed("bad transport magic (expected WFTT)"));
    }
    let major = u16::from_le_bytes([bytes[4], bytes[5]]);
    let minor = u16::from_le_bytes([bytes[6], bytes[7]]);
    if major != FORMAT_MAJOR || minor != FORMAT_MINOR {
        return Err(Error::unsupported_version(format!(
            "unsupported transport version {major}.{minor} (this reader implements 1.0)"
        )));
    }
    let flags = le_u32(bytes, 8);
    if flags != 0 {
        return Err(Error::unsupported_feature(format!(
            "transport header flags 0x{flags:08x} are reserved in v1"
        )));
    }
    let section_count = le_u32(bytes, 12);
    if section_count == 0 {
        return Err(Error::malformed(
            "a transport container needs at least one section",
        ));
    }
    check_len(
        u64::from(section_count),
        u64::from(MAX_SECTIONS),
        "transport section count",
    )?;
    let section_table_offset = le_u64(bytes, 16);
    if section_table_offset != HEADER_LEN as u64 {
        return Err(Error::malformed(format!(
            "transport section_table_offset {section_table_offset} must be {HEADER_LEN} in v1"
        )));
    }
    let total_len = le_u64(bytes, 24);
    let table_end = (HEADER_LEN as u64)
        .checked_add(
            u64::from(section_count)
                .checked_mul(SECTION_ENTRY_LEN as u64)
                .ok_or_else(|| Error::limit_exceeded("transport table length overflows u64"))?,
        )
        .ok_or_else(|| Error::limit_exceeded("transport table end overflows u64"))?;
    if total_len < table_end {
        return Err(Error::malformed(format!(
            "transport total_len {total_len} is below the table end {table_end}"
        )));
    }
    let payload_hash = Hash32::from_slice(&bytes[32..64])?;
    if bytes[64..HEADER_LEN].iter().any(|byte| *byte != 0) {
        return Err(Error::malformed(
            "transport header reserved bytes must be zero",
        ));
    }
    Ok(TransportHeader {
        flags,
        section_count,
        section_table_offset,
        total_len,
        payload_hash,
    })
}

/// Parse and structurally validate the section table.
///
/// `table` must be exactly `section_count * 56` bytes. Offsets and lengths are
/// validated against `header.total_len`; no section body is read.
pub fn read_section_table(header: &TransportHeader, table: &[u8]) -> Result<Vec<Section>> {
    let expected = usize::try_from(
        u64::from(header.section_count)
            .checked_mul(SECTION_ENTRY_LEN as u64)
            .ok_or_else(|| Error::limit_exceeded("transport table length overflows u64"))?,
    )
    .map_err(|_| Error::limit_exceeded("transport table length does not fit usize"))?;
    if table.len() != expected {
        return Err(Error::malformed(format!(
            "transport table is {} bytes but {} sections require {expected}",
            table.len(),
            header.section_count
        )));
    }
    let table_end = (HEADER_LEN as u64)
        .checked_add(expected as u64)
        .ok_or_else(|| Error::limit_exceeded("transport table end overflows u64"))?;
    let mut sections = Vec::with_capacity(header.section_count as usize);
    let mut previous_end: u64 = table_end;
    for index in 0..header.section_count as usize {
        let at = index * SECTION_ENTRY_LEN;
        let offset = le_u64(table, at);
        let length = le_u64(table, at + 8);
        let alignment = le_u32(table, at + 16);
        let mode = table[at + 20];
        let role = SectionRole::from_u8(table[at + 21])?;
        let reserved = u16::from_le_bytes([table[at + 22], table[at + 23]]);
        let root = Hash32::from_slice(&table[at + 24..at + 56])?;
        if reserved != 0 {
            return Err(Error::malformed(format!(
                "transport section {index} reserved field must be zero"
            )));
        }
        if !allowed_alignment(alignment) {
            return Err(Error::malformed(format!(
                "transport section {index} alignment {alignment} is not one of 512, 4096, 65536"
            )));
        }
        if offset % u64::from(alignment) != 0 {
            return Err(Error::malformed(format!(
                "transport section {index} offset {offset} is not a multiple of {alignment}"
            )));
        }
        if length == 0 {
            return Err(Error::malformed(format!(
                "transport section {index} length is zero"
            )));
        }
        let end = offset
            .checked_add(length)
            .ok_or_else(|| Error::malformed(format!("transport section {index} end overflows")))?;
        if offset < table_end {
            return Err(Error::malformed(format!(
                "transport section {index} offset {offset} lies inside the header or table"
            )));
        }
        if end > header.total_len {
            return Err(Error::malformed(format!(
                "transport section {index} ends at {end}, past total_len {}",
                header.total_len
            )));
        }
        if offset < previous_end {
            if offset
                == sections
                    .last()
                    .map(|s: &Section| s.offset)
                    .unwrap_or(u64::MAX)
            {
                return Err(Error::malformed(format!(
                    "transport section {index} duplicates offset {offset}"
                )));
            }
            return Err(Error::malformed(format!(
                "transport section {index} overlaps the previous section"
            )));
        }
        previous_end = end;
        sections.push(Section {
            offset,
            length,
            alignment,
            mode,
            role,
            root,
        });
    }
    Ok(sections)
}

/// Parse and validate the container's structure without decoding frame bodies.
///
/// Validates the header, the table, every section's frame prefix (which
/// carries the mode and content root), the zero-padding rule, and the final
/// aligned length. Does **not** compute the container-level `payload_hash`,
/// which requires reading every section body; call [`verify_payload`] or
/// [`verify`] for that.
pub fn parse_metadata<'a>(bytes: &'a [u8], limits: &Limits) -> Result<Transport<'a>> {
    let header = read_header(bytes)?;
    if header.total_len != bytes.len() as u64 {
        return Err(Error::malformed(format!(
            "transport total_len {} does not match the {} bytes supplied",
            header.total_len,
            bytes.len()
        )));
    }
    let table_end = (HEADER_LEN as u64)
        .checked_add(
            u64::from(header.section_count)
                .checked_mul(SECTION_ENTRY_LEN as u64)
                .ok_or_else(|| Error::limit_exceeded("transport table length overflows u64"))?,
        )
        .ok_or_else(|| Error::limit_exceeded("transport table end overflows u64"))?;
    let table_end_usize = usize::try_from(table_end)
        .map_err(|_| Error::limit_exceeded("transport table end does not fit usize"))?;
    let sections = read_section_table(&header, &bytes[HEADER_LEN..table_end_usize])?;

    // Per-section frame prefix: 96-byte header plus its own dependency/chunk
    // tables. parse_prefix deliberately does not read the frame body.
    for (index, section) in sections.iter().enumerate() {
        check_len(section.length, limits.max_frame_bytes, "transport section")?;
        let start = usize::try_from(section.offset)
            .map_err(|_| Error::limit_exceeded("section offset does not fit usize"))?;
        let end = usize::try_from(section.end())
            .map_err(|_| Error::limit_exceeded("section end does not fit usize"))?;
        let (frame_header, _body) = frame::parse_prefix(&bytes[start..end], limits)?;
        if frame_header.content_hash != section.root {
            return Err(Error::corrupt(format!(
                "transport section {index} root {} does not match the enclosed frame header {}",
                section.root, frame_header.content_hash
            )));
        }
        if frame_header.mode.as_u8() != section.mode {
            return Err(Error::corrupt(format!(
                "transport section {index} mode {} does not match the enclosed frame mode {}",
                section.mode,
                frame_header.mode.as_u8()
            )));
        }
    }

    // Every byte outside a section is padding and must be zero.
    let mut cursor = table_end;
    for (index, section) in sections.iter().enumerate() {
        let gap_end = section.offset;
        if gap_end < cursor {
            return Err(Error::malformed(format!(
                "transport section {index} starts before the previous section ends"
            )));
        }
        let start = usize::try_from(cursor)
            .map_err(|_| Error::limit_exceeded("padding offset does not fit usize"))?;
        let end = usize::try_from(gap_end)
            .map_err(|_| Error::limit_exceeded("padding end does not fit usize"))?;
        if bytes[start..end].iter().any(|byte| *byte != 0) {
            return Err(Error::malformed(format!(
                "transport padding before section {index} is not zero"
            )));
        }
        cursor = section.end();
    }
    let tail_start = usize::try_from(cursor)
        .map_err(|_| Error::limit_exceeded("padding offset does not fit usize"))?;
    if bytes[tail_start..].iter().any(|byte| *byte != 0) {
        return Err(Error::malformed("transport trailing padding is not zero"));
    }
    let last = sections
        .last()
        .ok_or_else(|| Error::malformed("transport container has no sections"))?;
    let aligned_end = checked_align_up(last.end(), last.alignment)?;
    if aligned_end != header.total_len {
        return Err(Error::malformed(format!(
            "transport total_len {} is not the aligned end {aligned_end} of the last section",
            header.total_len
        )));
    }

    Ok(Transport {
        header,
        sections,
        bytes,
    })
}

/// Compute and compare the container-level payload digest.
pub fn verify_payload(transport: &Transport<'_>) -> Result<()> {
    let mut hasher = DomainHasher::new(DOMAIN_TRANSPORT);
    for (index, section) in transport.sections.iter().enumerate() {
        let start = usize::try_from(section.offset)
            .map_err(|_| Error::limit_exceeded("section offset does not fit usize"))?;
        let end = usize::try_from(section.end())
            .map_err(|_| Error::limit_exceeded("section end does not fit usize"))?;
        let bytes = transport.bytes.get(start..end).ok_or_else(|| {
            Error::corrupt(format!(
                "transport section {index} is outside the container"
            ))
        })?;
        hasher.update(bytes);
    }
    let computed = hasher.finalize();
    if computed != transport.header.payload_hash {
        return Err(Error::corrupt(format!(
            "transport payload_hash {computed} does not match the header's {}",
            transport.header.payload_hash
        )));
    }
    Ok(())
}

/// Full verification: structure, every section's complete frame, and the
/// container payload digest.
pub fn verify<'a>(bytes: &'a [u8], limits: &Limits) -> Result<Transport<'a>> {
    let transport = parse_metadata(bytes, limits)?;
    for (index, section) in transport.sections.iter().enumerate() {
        let start = usize::try_from(section.offset)
            .map_err(|_| Error::limit_exceeded("section offset does not fit usize"))?;
        let end = usize::try_from(section.end())
            .map_err(|_| Error::limit_exceeded("section end does not fit usize"))?;
        let frame = Frame::parse(&bytes[start..end], limits)?;
        if frame.header.content_hash != section.root {
            return Err(Error::corrupt(format!(
                "transport section {index} root does not match the decoded frame"
            )));
        }
        if frame.header.mode.as_u8() != section.mode {
            return Err(Error::corrupt(format!(
                "transport section {index} mode does not match the decoded frame"
            )));
        }
    }
    verify_payload(&transport)?;
    Ok(transport)
}

/// Verify one fetched section against its table entry.
///
/// A single-section reader uses this after [`fetch_section`]. It parses the
/// complete frame (CRC, tables, trailer) and compares the mode and content
/// root. It cannot verify the container-level `payload_hash`, which is a
/// whole-container property.
pub fn verify_section_bytes<'a>(
    section: &Section,
    bytes: &'a [u8],
    limits: &Limits,
) -> Result<Frame<'a>> {
    if bytes.len() as u64 != section.length {
        return Err(Error::corrupt(format!(
            "transport section is {} bytes but the table records {}",
            bytes.len(),
            section.length
        )));
    }
    let frame = Frame::parse(bytes, limits)?;
    if frame.header.content_hash != section.root {
        return Err(Error::corrupt(format!(
            "transport section root {} does not match the fetched frame {}",
            section.root, frame.header.content_hash
        )));
    }
    if frame.header.mode.as_u8() != section.mode {
        return Err(Error::corrupt(format!(
            "transport section mode {} does not match the fetched frame {}",
            section.mode,
            frame.header.mode.as_u8()
        )));
    }
    Ok(frame)
}

/// Return the frame prefix header for a section without decoding it.
pub fn section_header<'a>(
    section: &Section,
    bytes: &'a [u8],
    limits: &Limits,
) -> Result<FrameHeader> {
    if bytes.len() as u64 != section.length {
        return Err(Error::corrupt(
            "transport section length does not match its table entry",
        ));
    }
    let (header, _) = frame::parse_prefix(bytes, limits)?;
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorClass;
    use crate::modes;

    fn raw(text: &[u8]) -> Vec<u8> {
        modes::encode_raw(text, None, &Limits::default()).unwrap()
    }

    #[test]
    fn roundtrip_single_section() {
        let frame = raw(b"transport payload");
        let container = create(
            &[TransportFrame::new(&frame, SectionRole::Payload)],
            4096,
            &Limits::default(),
        )
        .unwrap();
        let transport = verify(&container, &Limits::default()).unwrap();
        assert_eq!(transport.sections.len(), 1);
        let section = transport.sections[0];
        assert_eq!(section.offset % 4096, 0);
        assert_eq!(section.length, frame.len() as u64);
        let bytes = &transport.bytes[section.offset as usize..section.end() as usize];
        assert_eq!(bytes, frame.as_slice());
        verify_section_bytes(&section, bytes, &Limits::default()).unwrap();
    }

    #[test]
    fn deterministic_and_padded() {
        let a = raw(b"alpha");
        let b = raw(b"beta");
        let frames = [
            TransportFrame::new(&a, SectionRole::Payload),
            TransportFrame::new(&b, SectionRole::Dependency),
        ];
        let one = create(&frames, 512, &Limits::default()).unwrap();
        let two = create(&frames, 512, &Limits::default()).unwrap();
        assert_eq!(one, two);
        assert_eq!(one.len() % 512, 0);
        let transport = parse_metadata(&one, &Limits::default()).unwrap();
        assert_eq!(transport.sections[0].offset, 4608);
        assert_eq!(transport.sections[1].offset, 5120);
    }

    #[test]
    fn misaligned_offset_rejected() {
        let frame = raw(b"misalign");
        let mut container = create(
            &[TransportFrame::new(&frame, SectionRole::Payload)],
            4096,
            &Limits::default(),
        )
        .unwrap();
        container[HEADER_LEN..HEADER_LEN + 8]
            .copy_from_slice(&((HEADER_LEN as u64) + 1).to_le_bytes());
        let err = parse_metadata(&container, &Limits::default()).unwrap_err();
        assert_eq!(err.class(), ErrorClass::Malformed);
    }

    #[test]
    fn wrong_root_rejected() {
        let frame = raw(b"root");
        let mut container = create(
            &[TransportFrame::new(&frame, SectionRole::Payload)],
            4096,
            &Limits::default(),
        )
        .unwrap();
        container[HEADER_LEN + 24] ^= 0x01;
        let err = parse_metadata(&container, &Limits::default()).unwrap_err();
        assert_eq!(err.class(), ErrorClass::Corrupt);
    }

    #[test]
    fn payload_hash_covers_frames_only() {
        let frame = raw(b"hash me");
        let container = create(
            &[TransportFrame::new(&frame, SectionRole::Payload)],
            4096,
            &Limits::default(),
        )
        .unwrap();
        let transport = parse_metadata(&container, &Limits::default()).unwrap();
        verify_payload(&transport).unwrap();
        let expected = {
            let mut hasher = DomainHasher::new(DOMAIN_TRANSPORT);
            hasher.update(&frame);
            hasher.finalize()
        };
        assert_eq!(transport.header.payload_hash, expected);
    }
}
