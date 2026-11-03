//! Generation of the committed adversarial decoder-bomb vectors.
//!
//! These vectors live in `conformance/v1/adversarial/` and are copied
//! byte-identically into the independent Go codec. Each vector records the
//! input bytes (hex), any dependency/frame-source files, the CLI flags the
//! court must pass, and the normative error class both implementations must
//! return.
//!
//! The generator deliberately writes structurally invalid frames directly
//! rather than through [`weftcodec::frame::encode_frame`], because the encoder
//! rejects invalid layouts by design.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::{json, Map, Value as Json};

use crate::error::{Error, Result};
use crate::frame::{
    features, flags, ChunkEntry, Dependency, Frame, CHUNK_ENTRY_LEN, DEP_DOMAIN_PAYLOAD,
    DEP_ENTRY_LEN, DOMAIN_PAYLOAD, DOMAIN_STRUCTURED, FORMAT_MAJOR, FORMAT_MINOR, HEADER_LEN,
};
use crate::hash::{hash_domain, Hash32, DOMAIN_CHUNK};
use crate::limits::Limits;
use crate::modes;
use crate::varint;

struct RawFrameSpec<'a> {
    mode: u8,
    domain: u8,
    flags: u16,
    features: u16,
    level: u16,
    decoded_len: u64,
    body_len: Option<u64>,
    content_hash: Hash32,
    deps: &'a [Dependency],
    chunks: &'a [ChunkEntry],
    body: &'a [u8],
}

/// Build a complete WFT1 frame with valid header/trailer CRCs but without any
/// semantic validation. Used only to freeze adversarial inputs.
fn raw_frame(spec: &RawFrameSpec<'_>) -> Vec<u8> {
    let dep_count = spec.deps.len() as u32;
    let chunk_count = spec.chunks.len() as u32;
    let table_len = spec
        .deps
        .len()
        .checked_mul(DEP_ENTRY_LEN)
        .and_then(|v| {
            spec.chunks
                .len()
                .checked_mul(CHUNK_ENTRY_LEN)
                .and_then(|c| v.checked_add(c))
        })
        .expect("adversarial table length");
    let body_len = spec.body_len.unwrap_or(spec.body.len() as u64);

    let mut header = [0u8; HEADER_LEN];
    header[0..4].copy_from_slice(b"WFT1");
    header[4] = FORMAT_MAJOR;
    header[5] = FORMAT_MINOR;
    header[6] = spec.domain;
    header[7] = spec.mode;
    header[8..10].copy_from_slice(&spec.flags.to_le_bytes());
    header[10..12].copy_from_slice(&spec.level.to_le_bytes());
    header[12..20].copy_from_slice(&spec.decoded_len.to_le_bytes());
    header[20..28].copy_from_slice(&body_len.to_le_bytes());
    header[28..32].copy_from_slice(&dep_count.to_le_bytes());
    header[32..36].copy_from_slice(&chunk_count.to_le_bytes());
    header[36..38].copy_from_slice(&spec.features.to_le_bytes());
    header[40..72].copy_from_slice(spec.content_hash.as_bytes());
    header[76..84].copy_from_slice(&(table_len as u64).to_le_bytes());

    let mut out = Vec::new();
    out.extend_from_slice(&header);
    for dep in spec.deps {
        out.extend_from_slice(dep.hash.as_bytes());
        out.extend_from_slice(&dep.decoded_len.to_le_bytes());
        out.push(dep.dep_domain);
        out.push(dep.dep_flags);
        out.extend_from_slice(&[0u8, 0u8]);
        out.extend_from_slice(&dep.codec_hint.to_le_bytes());
    }
    for chunk in spec.chunks {
        out.extend_from_slice(&chunk.encoded_off.to_le_bytes());
        out.extend_from_slice(&chunk.encoded_len.to_le_bytes());
        out.extend_from_slice(&chunk.decoded_len.to_le_bytes());
        out.extend_from_slice(chunk.hash.as_bytes());
    }
    out.extend_from_slice(spec.body);

    let header_crc = crc32c::crc32c(&{
        let mut input = Vec::with_capacity(92);
        input.extend_from_slice(&out[0..72]);
        input.extend_from_slice(&out[76..96]);
        input
    });
    out[72..76].copy_from_slice(&header_crc.to_le_bytes());
    let trailer_crc = crc32c::crc32c(&out);
    out.extend_from_slice(&trailer_crc.to_le_bytes());
    out
}

fn adv_vector(name: &str, description: &str, verb: &str, expect_class: &str, input: &[u8]) -> Json {
    let mut object = Map::new();
    object.insert("name".to_owned(), json!(name));
    object.insert("description".to_owned(), json!(description));
    object.insert("verb".to_owned(), json!(verb));
    object.insert("expect_class".to_owned(), json!(expect_class));
    object.insert("input_hex".to_owned(), json!(hex::encode(input)));
    Json::Object(object)
}

fn insert_flags(vector: &mut Json, pairs: &[(&str, &str)]) {
    let mut flags_out = BTreeMap::new();
    for (key, value) in pairs {
        flags_out.insert((*key).to_owned(), (*value).to_owned());
    }
    vector
        .as_object_mut()
        .expect("adversarial vector")
        .insert("flags".to_owned(), json!(flags_out));
}

fn dep_entry(bytes: &[u8]) -> (Json, Hash32, u64) {
    let hash = hash_domain(DOMAIN_CHUNK, bytes);
    (
        json!({"hash": hash.to_hex(), "hex": hex::encode(bytes)}),
        hash,
        bytes.len() as u64,
    )
}

fn frame_source_entry(frame: &[u8], limits: &Limits) -> Result<Json> {
    let parsed = Frame::parse(frame, limits)?;
    Ok(json!({
        "hash": parsed.header.content_hash.to_hex(),
        "hex": hex::encode(frame),
    }))
}

fn archive_from_entries(entries: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"WFA1");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        out.extend_from_slice(&(entry.len() as u64).to_le_bytes());
        out.extend_from_slice(entry);
    }
    let crc = crc32c::crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

fn archive_header_only(entry_count: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"WFA1");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&entry_count.to_le_bytes());
    let crc = crc32c::crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

fn deep_arrays(depth: usize) -> Vec<u8> {
    let mut body = Vec::new();
    for _ in 0..depth {
        body.push(0x07); // TAG_ARRAY
        body.push(1); // one element
    }
    body.push(0x00); // TAG_NULL
    body
}

fn insert_op(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0x01];
    varint::encode(bytes.len() as u64, &mut out);
    out.extend_from_slice(bytes);
    out
}

/// Write every adversarial vector into `<dir>/adversarial/`.
pub fn generate(dir: &Path, limits: &Limits) -> Result<()> {
    let adv = dir.join("adversarial");
    fs::create_dir_all(&adv)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", adv.display())))?;

    let mut vectors: Vec<(String, Json)> = Vec::new();
    let mut push = |name: &str, vector: Json| {
        vectors.push((format!("{name}.json"), vector));
    };

    // 1. A declared length far beyond any court memory. The decoder must reject
    // before allocating or decompressing.
    let ten_gib = 10u64 << 30;
    let frame = raw_frame(&RawFrameSpec {
        mode: 1, // ZSTD
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: 0,
        level: 3,
        decoded_len: ten_gib,
        body_len: Some(0),
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &[],
    });
    push(
        "001_declared_10gib",
        adv_vector(
            "declared_10gib",
            "Declared decoded length is 10 GiB; default max_decoded_bytes (4 GiB) must reject before allocation.",
            "decode",
            "LimitExceeded",
            &frame,
        ),
    );

    // 2. A real zstd expansion bomb under a low configured bound.
    let bomb_data = vec![0x41u8; 4 << 20];
    let bomb_frame = modes::encode_zstd(&bomb_data, 3, None, limits)?;
    let mut vector = adv_vector(
        "zstd_expansion_low_limit",
        "4 MiB of repeating bytes compressed with zstd, decoded under --max-decoded-bytes 1 MiB.",
        "decode",
        "LimitExceeded",
        &bomb_frame,
    );
    insert_flags(&mut vector, &[("--max-decoded-bytes", "1048576")]);
    push("002_zstd_expansion_low_limit", vector);

    // 3. Output length mismatch: the zstd body expands to more than declared.
    let zstd_source = modes::encode_zstd(&vec![0u8; 4096], 3, None, limits)?;
    let zstd_body = Frame::parse(&zstd_source, limits)?.body.to_vec();
    let mismatch = raw_frame(&RawFrameSpec {
        mode: 1,
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: 0,
        level: 3,
        decoded_len: 1024,
        body_len: None,
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &zstd_body,
    });
    push(
        "003_zstd_output_length_mismatch",
        adv_vector(
            "zstd_output_length_mismatch",
            "Zstd body expands to 4096 bytes while decoded_len declares 1024.",
            "decode",
            "Corrupt",
            &mismatch,
        ),
    );

    // 4. Tiny zstd body that *claims* 4 GiB. This is the regression vector for
    // the old bulk-decompressor preallocation bug: decoding must stay small and
    // return Corrupt, never allocate the declared length.
    let one_byte = modes::encode_zstd(b"x", 3, None, limits)?;
    let one_byte_body = Frame::parse(&one_byte, limits)?.body.to_vec();
    let huge_claim = raw_frame(&RawFrameSpec {
        mode: 1,
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: 0,
        level: 3,
        decoded_len: 4 << 30,
        body_len: None,
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &one_byte_body,
    });
    push(
        "004_zstd_declared_4gib_body_one_byte",
        adv_vector(
            "zstd_declared_4gib_body_one_byte",
            "Tiny zstd body declares a 4 GiB output. Must fail Corrupt without preallocating the declared length.",
            "decode",
            "Corrupt",
            &huge_claim,
        ),
    );

    // 5. Delta COPY offset+length arithmetic overflow.
    let base = b"base";
    let (base_dep_json, base_hash, base_len) = dep_entry(base);
    let base_dep = Dependency::new(base_hash, base_len, DEP_DOMAIN_PAYLOAD, 0);
    let mut body = vec![0x00];
    varint::encode(u64::MAX, &mut body);
    varint::encode(2, &mut body);
    let frame = raw_frame(&RawFrameSpec {
        mode: 3, // DELTA
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: features::DEPENDENCIES,
        level: 0,
        decoded_len: 2,
        body_len: None,
        content_hash: Hash32::ZERO,
        deps: std::slice::from_ref(&base_dep),
        chunks: &[],
        body: &body,
    });
    let mut vector = adv_vector(
        "delta_copy_offset_overflow",
        "DELTA COPY uses offset u64::MAX and length 2; offset+length overflows.",
        "decode",
        "Malformed",
        &frame,
    );
    vector
        .as_object_mut()
        .unwrap()
        .insert("deps".to_owned(), json!([base_dep_json]));
    push("005_delta_copy_offset_overflow", vector);

    // 6. Delta INSERT length arithmetic overflow.
    let mut body = vec![0x01];
    varint::encode(u64::MAX, &mut body);
    let frame = raw_frame(&RawFrameSpec {
        mode: 3,
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: features::DEPENDENCIES,
        level: 0,
        decoded_len: 1,
        body_len: None,
        content_hash: Hash32::ZERO,
        deps: std::slice::from_ref(&base_dep),
        chunks: &[],
        body: &body,
    });
    let mut vector = adv_vector(
        "delta_insert_length_overflow",
        "DELTA INSERT length u64::MAX overflows the body extent.",
        "decode",
        "Malformed",
        &frame,
    );
    vector
        .as_object_mut()
        .unwrap()
        .insert("deps".to_owned(), json!([base_dep_json]));
    push("006_delta_insert_length_overflow", vector);

    // 7. Delta operation-count limit.
    let mut body = Vec::new();
    for _ in 0..5 {
        body.extend_from_slice(&insert_op(b"a"));
    }
    let frame = raw_frame(&RawFrameSpec {
        mode: 3,
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: features::DEPENDENCIES,
        level: 0,
        decoded_len: 5,
        body_len: None,
        content_hash: Hash32::ZERO,
        deps: std::slice::from_ref(&base_dep),
        chunks: &[],
        body: &body,
    });
    let mut vector = adv_vector(
        "delta_op_count_limit",
        "Five DELTA operations decoded under --max-delta-ops 2.",
        "decode",
        "LimitExceeded",
        &frame,
    );
    vector
        .as_object_mut()
        .unwrap()
        .insert("deps".to_owned(), json!([base_dep_json]));
    insert_flags(&mut vector, &[("--max-delta-ops", "2")]);
    push("007_delta_op_count_limit", vector);

    // 8. Dependency depth chain through a frame source.
    let mut chain = Vec::new();
    let mut prev = b"chain-base".to_vec();
    chain.push(modes::encode_raw(&prev, None, limits)?);
    for index in 0..5 {
        let mut target = prev.clone();
        target.extend_from_slice(format!("-{index}").as_bytes());
        chain.push(modes::encode_delta(&target, &prev, limits)?);
        prev = target;
    }
    let root = chain.last().unwrap().clone();
    let frame_source: Vec<Json> = chain
        .iter()
        .map(|frame| frame_source_entry(frame, limits))
        .collect::<Result<_>>()?;
    let mut vector = adv_vector(
        "dependency_depth_chain",
        "Root DELTA depends on a five-frame chain; --max-dependency-depth 2 must reject.",
        "decode",
        "LimitExceeded",
        &root,
    );
    vector
        .as_object_mut()
        .unwrap()
        .insert("frame_source".to_owned(), json!(frame_source));
    insert_flags(&mut vector, &[("--max-dependency-depth", "2")]);
    push("008_dependency_depth_chain", vector);

    // 9. Self-cycle through a frame source.
    let cycle_hash = hash_domain(DOMAIN_CHUNK, b"cycle");
    let cycle_dep = Dependency::new(cycle_hash, 5, DEP_DOMAIN_PAYLOAD, 0);
    let cycle_frame = raw_frame(&RawFrameSpec {
        mode: 2, // REFERENCE
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: features::DEPENDENCIES,
        level: 0,
        decoded_len: 5,
        body_len: Some(0),
        content_hash: cycle_hash,
        deps: std::slice::from_ref(&cycle_dep),
        chunks: &[],
        body: &[],
    });
    let mut vector = adv_vector(
        "dependency_cycle_frame_source",
        "REFERENCE frame whose dependency is its own content hash; DAG decode must report DependencyCycle.",
        "decode",
        "DependencyCycle",
        &cycle_frame,
    );
    vector.as_object_mut().unwrap().insert(
        "frame_source".to_owned(),
        json!([{"hash": cycle_hash.to_hex(), "hex": hex::encode(&cycle_frame)}]),
    );
    push("009_dependency_cycle_frame_source", vector);

    // 10. Cycle rejected when creating an archive from two mutually-dependent
    // frames.
    let hash_a = Hash32::from_bytes([0xa1u8; 32]);
    let hash_b = Hash32::from_bytes([0xb2u8; 32]);
    let a = raw_frame(&RawFrameSpec {
        mode: 2,
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: features::DEPENDENCIES,
        level: 0,
        decoded_len: 1,
        body_len: Some(0),
        content_hash: hash_a,
        deps: &[Dependency::new(hash_b, 1, DEP_DOMAIN_PAYLOAD, 0)],
        chunks: &[],
        body: &[],
    });
    let b = raw_frame(&RawFrameSpec {
        mode: 2,
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: features::DEPENDENCIES,
        level: 0,
        decoded_len: 1,
        body_len: Some(0),
        content_hash: hash_b,
        deps: &[Dependency::new(hash_a, 1, DEP_DOMAIN_PAYLOAD, 0)],
        chunks: &[],
        body: &[],
    });
    let mut vector = adv_vector(
        "dependency_cycle_archive",
        "archive create receives two REFERENCE frames that depend on each other; cycle must be rejected.",
        "archive-create",
        "DependencyCycle",
        &[],
    );
    vector.as_object_mut().unwrap().insert(
        "files".to_owned(),
        json!([
            {"name": "a.wft", "hex": hex::encode(&a)},
            {"name": "b.wft", "hex": hex::encode(&b)},
        ]),
    );
    push("010_dependency_cycle_archive", vector);

    // 11. Dependency count limit.
    let fake_deps: Vec<Dependency> = (0..3)
        .map(|index| {
            Dependency::new(
                Hash32::from_bytes([index as u8; 32]),
                1,
                DEP_DOMAIN_PAYLOAD,
                0,
            )
        })
        .collect();
    let frame = raw_frame(&RawFrameSpec {
        mode: 2,
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: features::DEPENDENCIES,
        level: 0,
        decoded_len: 1,
        body_len: Some(0),
        content_hash: Hash32::ZERO,
        deps: &fake_deps,
        chunks: &[],
        body: &[],
    });
    let mut vector = adv_vector(
        "dependency_count_limit",
        "Three dependency entries decoded under --max-dependency-count 2.",
        "decode",
        "LimitExceeded",
        &frame,
    );
    insert_flags(&mut vector, &[("--max-dependency-count", "2")]);
    push("011_dependency_count_limit", vector);

    // 12. Chunk extent past the body.
    let chunk = ChunkEntry {
        encoded_off: 0,
        encoded_len: 4,
        decoded_len: 4,
        hash: Hash32::ZERO,
    };
    let frame = raw_frame(&RawFrameSpec {
        mode: 0,
        domain: DOMAIN_PAYLOAD,
        flags: flags::CHUNKED,
        features: features::CHUNK_HASHES,
        level: 0,
        decoded_len: 4,
        body_len: Some(0),
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: std::slice::from_ref(&chunk),
        body: &[],
    });
    push(
        "012_chunk_extent_overflow",
        adv_vector(
            "chunk_extent_overflow",
            "One chunk claims four encoded bytes while the body is empty.",
            "decode",
            "Malformed",
            &frame,
        ),
    );

    // 13. Overlapping chunks.
    let chunks = [
        ChunkEntry {
            encoded_off: 0,
            encoded_len: 1,
            decoded_len: 1,
            hash: Hash32::ZERO,
        },
        ChunkEntry {
            encoded_off: 0,
            encoded_len: 1,
            decoded_len: 1,
            hash: Hash32::ZERO,
        },
    ];
    let frame = raw_frame(&RawFrameSpec {
        mode: 0,
        domain: DOMAIN_PAYLOAD,
        flags: flags::CHUNKED,
        features: features::CHUNK_HASHES,
        level: 0,
        decoded_len: 2,
        body_len: Some(2),
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &chunks,
        body: b"ab",
    });
    push(
        "013_chunk_overlap",
        adv_vector(
            "chunk_overlap",
            "Two chunks both start at offset zero.",
            "decode",
            "Malformed",
            &frame,
        ),
    );

    // 14. Gap between chunks.
    let chunks = [
        ChunkEntry {
            encoded_off: 0,
            encoded_len: 1,
            decoded_len: 1,
            hash: Hash32::ZERO,
        },
        ChunkEntry {
            encoded_off: 2,
            encoded_len: 1,
            decoded_len: 1,
            hash: Hash32::ZERO,
        },
    ];
    let frame = raw_frame(&RawFrameSpec {
        mode: 0,
        domain: DOMAIN_PAYLOAD,
        flags: flags::CHUNKED,
        features: features::CHUNK_HASHES,
        level: 0,
        decoded_len: 2,
        body_len: Some(3),
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &chunks,
        body: b"abc",
    });
    push(
        "014_chunk_gap",
        adv_vector(
            "chunk_gap",
            "Second chunk starts after a one-byte gap.",
            "decode",
            "Malformed",
            &frame,
        ),
    );

    // 15. Chunk count limit.
    let chunks: Vec<ChunkEntry> = (0..4)
        .map(|index| ChunkEntry {
            encoded_off: index,
            encoded_len: 1,
            decoded_len: 1,
            hash: Hash32::ZERO,
        })
        .collect();
    let frame = raw_frame(&RawFrameSpec {
        mode: 0,
        domain: DOMAIN_PAYLOAD,
        flags: flags::CHUNKED,
        features: features::CHUNK_HASHES,
        level: 0,
        decoded_len: 4,
        body_len: Some(4),
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &chunks,
        body: b"abcd",
    });
    let mut vector = adv_vector(
        "chunk_count_limit",
        "Four chunks decoded under --max-chunks 2.",
        "decode",
        "LimitExceeded",
        &frame,
    );
    insert_flags(&mut vector, &[("--max-chunks", "2")]);
    push("015_chunk_count_limit", vector);

    // 16. Header length arithmetic overflow.
    let frame = raw_frame(&RawFrameSpec {
        mode: 0,
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: 0,
        level: 0,
        decoded_len: 0,
        body_len: Some(u64::MAX),
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &[],
    });
    push(
        "016_length_arithmetic_overflow",
        adv_vector(
            "length_arithmetic_overflow",
            "body_len u64::MAX overflows the header-derived frame length.",
            "decode",
            "Malformed",
            &frame,
        ),
    );

    // 17. Unknown critical mode.
    let frame = raw_frame(&RawFrameSpec {
        mode: 0xff,
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: 0,
        level: 0,
        decoded_len: 0,
        body_len: Some(0),
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &[],
    });
    push(
        "017_unknown_mode",
        adv_vector(
            "unknown_mode",
            "Mode byte 0xff is not a v1 mode.",
            "decode",
            "UnsupportedMode",
            &frame,
        ),
    );

    // 18. Unknown critical flag.
    let frame = raw_frame(&RawFrameSpec {
        mode: 0,
        domain: DOMAIN_PAYLOAD,
        flags: 0x0010,
        features: 0,
        level: 0,
        decoded_len: 0,
        body_len: Some(0),
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &[],
    });
    push(
        "018_unknown_critical_flag",
        adv_vector(
            "unknown_critical_flag",
            "Undefined critical header flag bit 0x0010.",
            "decode",
            "UnsupportedFeature",
            &frame,
        ),
    );

    // 19. Unknown critical feature.
    let frame = raw_frame(&RawFrameSpec {
        mode: 0,
        domain: DOMAIN_PAYLOAD,
        flags: 0,
        features: 0x0004,
        level: 0,
        decoded_len: 0,
        body_len: Some(0),
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &[],
    });
    push(
        "019_unknown_critical_feature",
        adv_vector(
            "unknown_critical_feature",
            "Undefined critical feature bit 0x0004.",
            "decode",
            "UnsupportedFeature",
            &frame,
        ),
    );

    // 20. Non-minimal WCB varint.
    let body = vec![0x05, 0x80, 0x00];
    let frame = raw_frame(&RawFrameSpec {
        mode: 5, // STRUCTURED
        domain: DOMAIN_STRUCTURED,
        flags: 0,
        features: 0,
        level: 0,
        decoded_len: body.len() as u64,
        body_len: None,
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &body,
    });
    push(
        "020_non_minimal_varint_structured",
        adv_vector(
            "non_minimal_varint_structured",
            "WCB text length uses a non-minimal varint (0x80 0x00).",
            "decode",
            "NonCanonical",
            &frame,
        ),
    );

    // 21. Overflowing WCB varint.
    let mut body = vec![0x05];
    body.extend_from_slice(&[0xff; 9]);
    body.push(0x02);
    let frame = raw_frame(&RawFrameSpec {
        mode: 5,
        domain: DOMAIN_STRUCTURED,
        flags: 0,
        features: 0,
        level: 0,
        decoded_len: body.len() as u64,
        body_len: None,
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &body,
    });
    push(
        "021_overflow_varint_structured",
        adv_vector(
            "overflow_varint_structured",
            "WCB varint has ten bytes with a tenth byte above 0x01.",
            "decode",
            "Malformed",
            &frame,
        ),
    );

    // 22. WCB depth limit.
    let body = deep_arrays(8);
    let frame = raw_frame(&RawFrameSpec {
        mode: 5,
        domain: DOMAIN_STRUCTURED,
        flags: 0,
        features: 0,
        level: 0,
        decoded_len: body.len() as u64,
        body_len: None,
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &body,
    });
    let mut vector = adv_vector(
        "wcb_depth_limit",
        "Eight nested WCB arrays decoded under --max-structured-depth 4.",
        "decode",
        "LimitExceeded",
        &frame,
    );
    insert_flags(&mut vector, &[("--max-structured-depth", "4")]);
    push("022_wcb_depth_limit", vector);

    // 23. WCB item limit.
    let mut body = vec![0x07, 10];
    for _ in 0..10 {
        body.push(0x00);
    }
    let frame = raw_frame(&RawFrameSpec {
        mode: 5,
        domain: DOMAIN_STRUCTURED,
        flags: 0,
        features: 0,
        level: 0,
        decoded_len: body.len() as u64,
        body_len: None,
        content_hash: Hash32::ZERO,
        deps: &[],
        chunks: &[],
        body: &body,
    });
    let mut vector = adv_vector(
        "wcb_item_limit",
        "Ten WCB items decoded under --max-structured-items 2.",
        "decode",
        "LimitExceeded",
        &frame,
    );
    insert_flags(&mut vector, &[("--max-structured-items", "2")]);
    push("023_wcb_item_limit", vector);

    // 24. Selector nesting depth.
    let source = format!("select {}true", "not ".repeat(100));
    let mut vector = adv_vector(
        "selector_depth_limit",
        "One hundred nested NOT operators under --max-selector-depth 4.",
        "selector-id",
        "LimitExceeded",
        source.as_bytes(),
    );
    insert_flags(&mut vector, &[("--max-selector-depth", "4")]);
    push("024_selector_depth_limit", vector);

    // 25. Selector input byte limit.
    let mut source = String::from("select true");
    source.push_str(&" ".repeat(100 * 1024));
    let mut vector = adv_vector(
        "selector_input_limit",
        "Selector source above 100 KiB under --max-selector-bytes 1024.",
        "selector-id",
        "LimitExceeded",
        source.as_bytes(),
    );
    insert_flags(&mut vector, &[("--max-selector-bytes", "1024")]);
    push("025_selector_input_limit", vector);

    // 26. Archive entry-count limit.
    let frames = [
        modes::encode_raw(b"one", None, limits)?,
        modes::encode_raw(b"two", None, limits)?,
        modes::encode_raw(b"three", None, limits)?,
    ];
    let archive = crate::archive::create_archive(&frames, limits)?;
    let mut vector = adv_vector(
        "archive_entry_count_limit",
        "Three-entry archive decoded under --max-archive-entries 2.",
        "archive-decode",
        "LimitExceeded",
        &archive,
    );
    insert_flags(&mut vector, &[("--max-archive-entries", "2")]);
    push("026_archive_entry_count_limit", vector);

    // 27. Archive with non-topological entry order.
    let base_bytes = b"archive-topological-base".to_vec();
    let target_bytes = b"archive-topological-base-plus".to_vec();
    let base_frame = modes::encode_raw(&base_bytes, None, limits)?;
    let delta_frame = modes::encode_delta(&target_bytes, &base_bytes, limits)?;
    let reordered = archive_from_entries(&[&delta_frame, &base_frame]);
    push(
        "027_archive_non_topological",
        adv_vector(
            "archive_non_topological",
            "Archive lists a DELTA frame before the base frame it depends on.",
            "archive-decode",
            "Malformed",
            &reordered,
        ),
    );

    // 28. Archive with an absurd declared entry count.
    let archive = archive_header_only(u32::MAX);
    push(
        "028_archive_entry_count_absurd",
        adv_vector(
            "archive_entry_count_absurd",
            "Archive header declares 4,294,967,295 entries with no table.",
            "archive-decode",
            "LimitExceeded",
            &archive,
        ),
    );

    // 29. Dictionary dependency mismatch: the resolver finds a file under the
    // declared hash but its bytes do not hash to that value.
    let dict = b"dictionary-bytes-for-vector";
    let dict_frame = modes::encode_dictionary(b"payload-compressed-with-dict", dict, 3, limits)?;
    let wrong_dict = b"DIFFERENT-dictionary-vector";
    assert_eq!(wrong_dict.len(), dict.len(), "mismatch vector length");
    let dict_hash = hash_domain(DOMAIN_CHUNK, dict);
    let mut vector = adv_vector(
        "dictionary_mismatch",
        "DICTIONARY resolver returns bytes that do not hash to the declared dependency hash.",
        "decode",
        "DependencyMismatch",
        &dict_frame,
    );
    vector.as_object_mut().unwrap().insert(
        "deps".to_owned(),
        json!([{"hash": dict_hash.to_hex(), "hex": hex::encode(wrong_dict)}]),
    );
    push("029_dictionary_mismatch", vector);

    // 30. Dictionary size limit.
    let big_dict = vec![0x5au8; 1024];
    let dict_frame =
        modes::encode_dictionary(b"payload-for-dictionary-limit", &big_dict, 3, limits)?;
    let mut vector = adv_vector(
        "dictionary_size_limit",
        "Dependency declares a 1024-byte dictionary under --max-dictionary-bytes 16.",
        "decode",
        "LimitExceeded",
        &dict_frame,
    );
    insert_flags(&mut vector, &[("--max-dictionary-bytes", "16")]);
    push("030_dictionary_size_limit", vector);

    // 31. Total fetched-bytes limit during DAG traversal.
    let (root_frame, source_frames) = {
        let base_bytes = b"fetch-limit-base".to_vec();
        let target_bytes = b"fetch-limit-base-plus".to_vec();
        let base_frame = modes::encode_raw(&base_bytes, None, limits)?;
        let root_frame = modes::encode_delta(&target_bytes, &base_bytes, limits)?;
        (root_frame.clone(), vec![base_frame, root_frame])
    };
    let fetch_limit = root_frame.len() as u64;
    let frame_source: Vec<Json> = source_frames
        .iter()
        .map(|frame| frame_source_entry(frame, limits))
        .collect::<Result<_>>()?;
    let mut vector = adv_vector(
        "reference_total_fetch_limit",
        "Root frame fits the fetch bound but the dependency frame pushes the total over it.",
        "decode",
        "LimitExceeded",
        &root_frame,
    );
    vector
        .as_object_mut()
        .unwrap()
        .insert("frame_source".to_owned(), json!(frame_source));
    insert_flags(
        &mut vector,
        &[("--max-total-fetched-bytes", &fetch_limit.to_string())],
    );
    push("031_reference_total_fetch_limit", vector);

    // 32-34. Truncation boundaries.
    let valid = modes::encode_raw(b"truncation-boundary", None, limits)?;
    for (name, length) in [
        ("032_truncated_frame_99", 99usize),
        ("033_truncated_frame_50", 50usize),
        ("034_empty_frame", 0usize),
    ] {
        push(
            name,
            adv_vector(
                name,
                "Frame input truncated below the 100-byte minimum.",
                "decode",
                "Malformed",
                &valid[..length.min(valid.len())],
            ),
        );
    }

    // 35. Frame byte limit.
    let mut vector = adv_vector(
        "frame_size_limit",
        "A valid frame above --max-frame-bytes 50.",
        "decode",
        "LimitExceeded",
        &valid,
    );
    insert_flags(&mut vector, &[("--max-frame-bytes", "50")]);
    push("035_frame_size_limit", vector);

    // 36. Short archive truncation.
    let short_archive = b"WFA1\x01\x00\x00\x00\x00\x00\x00\x00".to_vec();
    push(
        "036_archive_truncated_short",
        adv_vector(
            "archive_truncated_short",
            "Archive shorter than its header and trailer.",
            "archive-decode",
            "Malformed",
            &short_archive,
        ),
    );

    // Sort for deterministic file names and write.
    vectors.sort_by(|a, b| a.0.cmp(&b.0));
    for (file, value) in &vectors {
        let text = serde_json::to_string_pretty(value)
            .map_err(|e| Error::io(format!("cannot serialize adversarial vector: {e}")))?;
        fs::write(adv.join(file), format!("{text}\n"))
            .map_err(|e| Error::io(format!("cannot write {file}: {e}")))?;
    }

    // Provenance: exact generator and per-file hashes.
    let mut hashes = Map::new();
    for (file, _) in &vectors {
        let bytes = fs::read(adv.join(file))
            .map_err(|e| Error::io(format!("cannot reread {file}: {e}")))?;
        hashes.insert(
            file.clone(),
            json!(hex::encode(blake3::hash(&bytes).as_bytes())),
        );
    }
    let provenance = json!({
        "schema": "weftfs.adversarial-vectors/1",
        "generator": "weftcodec conformance --adversarial",
        "generator_source": "WeftCodec/src/adversarial.rs",
        "normative_spec": "WEFTCODEC-FORMAT 1.0 / spec/WEFTCODEC-FORMAT.md",
        "vector_count": vectors.len(),
        "sha256_manifest": "sha256sum of each file is recorded by bench/scripts/run_decoder_bomb_court.sh",
        "blake3_hashes": hashes,
    });
    let text = serde_json::to_string_pretty(&provenance)
        .map_err(|e| Error::io(format!("cannot serialize provenance: {e}")))?;
    fs::write(adv.join("PROVENANCE.json"), format!("{text}\n"))
        .map_err(|e| Error::io(format!("cannot write PROVENANCE.json: {e}")))?;
    Ok(())
}
