//! Mode 13 `ACTIVATION_STREAM` exactness, assembly, append, and vectors.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;
use weftcodec::activation_stream::{
    assemble_stream, decode_activation_chunk, decode_activation_chunk_range,
    encode_activation_chunk, stream_root, ActivationChunkSet, ActivationStreamHead,
    ActivationStreamWriter, StreamDtype, StreamKind, ACTIVATION_STREAM_SCHEMA,
};
use weftcodec::frame::Frame;
use weftcodec::hash::hash_domain;
use weftcodec::limits::Limits;
use weftcodec::tensor::DOMAIN_TENSOR_VALUES;
use weftcodec::ErrorClass;

fn limits() -> Limits {
    Limits::default()
}

fn head(dtype: StreamDtype, chunk_count: u64) -> ActivationStreamHead {
    ActivationStreamHead {
        stream: StreamKind::Activation,
        step: 11,
        rank: 0,
        world: 2,
        chunk_count,
        dtype,
        shape: vec![2, 4],
    }
}

fn chunk_bytes(seed: u16) -> Vec<u8> {
    let words: [u16; 8] = [
        0x3F80, 0x8000, 0x7FC0, 0x0001, 0xC000, 0x0000, 0x7F81, 0x7F80,
    ];
    let mut bytes = Vec::with_capacity(16);
    for (index, word) in words.iter().enumerate() {
        let value = word.wrapping_add(seed.wrapping_add(index as u16));
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn roundtrip_preserves_every_bit_pattern() {
    let head = head(StreamDtype::Bf16, 1);
    let elements = chunk_bytes(0);
    let frame = encode_activation_chunk(&head, 0, &elements, &limits()).expect("encode");
    let parsed = Frame::parse(&frame, &limits()).expect("parse");
    let decoded = decode_activation_chunk(&parsed, &limits()).expect("decode");
    assert_eq!(decoded.bytes, elements);
    assert_eq!(
        decoded.meta.digest_root,
        hash_domain(DOMAIN_TENSOR_VALUES, &elements)
    );
    assert_eq!(decoded.meta.stream, StreamKind::Activation);
    assert_eq!(decoded.meta.element_count, 8);
    let again = encode_activation_chunk(&head, 0, &elements, &limits()).expect("again");
    assert_eq!(again, frame);
}

#[test]
fn assembly_orders_by_chunk_index_and_rejects_duplicates_or_missing() {
    let head = head(StreamDtype::Bf16, 3);
    let chunks: Vec<Vec<u8>> = (0..3).map(chunk_bytes).collect();
    let frames: Vec<Vec<u8>> = chunks
        .iter()
        .enumerate()
        .map(|(index, elements)| {
            encode_activation_chunk(&head, index as u64, elements, &limits()).expect("chunk")
        })
        .collect();
    // Shuffled input order.
    let refs = [
        frames[2].as_slice(),
        frames[0].as_slice(),
        frames[1].as_slice(),
    ];
    let assembly = assemble_stream(&refs, &limits()).expect("assemble");
    let mut expected = Vec::new();
    for chunk in &chunks {
        expected.extend_from_slice(chunk);
    }
    assert_eq!(assembly.bytes, expected);
    assert_eq!(assembly.chunk_count, 3);
    assert_eq!(assembly.chunks_decoded, 3);

    let digests: Vec<_> = chunks
        .iter()
        .map(|chunk| hash_domain(DOMAIN_TENSOR_VALUES, chunk))
        .collect();
    assert_eq!(
        assembly.stream_root,
        stream_root(StreamKind::Activation, 11, 0, 2, 3, &digests, &limits()).unwrap()
    );

    // Duplicate index.
    let duplicate = [
        frames[0].as_slice(),
        frames[0].as_slice(),
        frames[2].as_slice(),
    ];
    assert_eq!(
        assemble_stream(&duplicate, &limits())
            .expect_err("duplicate")
            .class(),
        ErrorClass::NonCanonical
    );

    // Missing index.
    let missing = [frames[0].as_slice(), frames[1].as_slice()];
    assert_eq!(
        assemble_stream(&missing, &limits())
            .expect_err("missing")
            .class(),
        ErrorClass::Malformed
    );
}

#[test]
fn append_never_decodes_a_previous_chunk() {
    let head = head(StreamDtype::Bf16, 2);
    let mut writer = ActivationStreamWriter::new(head, &limits()).expect("writer");
    let first = writer
        .append_chunk(&chunk_bytes(0), &limits())
        .expect("chunk 0");
    let second = writer
        .append_chunk(&chunk_bytes(1), &limits())
        .expect("chunk 1");
    assert_eq!(writer.encoded_chunks(), 2);
    assert_eq!(writer.decoded_chunks(), 0, "append must not decode");
    assert_eq!(writer.next_index(), 2);

    let assembly = writer
        .assemble(&[first.as_slice(), second.as_slice()], &limits())
        .expect("assemble");
    assert_eq!(assembly.chunks_decoded, 2);
    assert_eq!(writer.decoded_chunks(), 2);
}

#[test]
fn chunk_three_of_four_decodes_without_earlier_chunks() {
    let head = head(StreamDtype::Bf16, 4);
    let chunks: Vec<Vec<u8>> = (0..4).map(chunk_bytes).collect();
    let mut frames: Vec<Vec<u8>> = chunks
        .iter()
        .enumerate()
        .map(|(index, elements)| {
            encode_activation_chunk(&head, index as u64, elements, &limits()).expect("chunk")
        })
        .collect();
    for frame in frames.iter_mut().take(3) {
        frame.truncate(4);
    }
    let mut set = ActivationChunkSet::new(frames);
    let chunk = set.decode_chunk(3, &limits()).expect("chunk 3");
    assert_eq!(chunk.meta.chunk_index, 3);
    assert_eq!(chunk.bytes, chunks[3]);
    assert_eq!(set.decoded_chunks(), 1);

    let slice = set
        .read_chunk_range(3, 4, 12, &limits())
        .expect("range inside chunk 3");
    assert_eq!(slice, chunks[3][4..12].to_vec());
    assert_eq!(set.decoded_chunks(), 2);
}

#[test]
fn range_decode_of_one_chunk_touches_only_that_chunk() {
    let head = head(StreamDtype::Bf16, 2);
    let elements = chunk_bytes(5);
    let frame = encode_activation_chunk(&head, 1, &elements, &limits()).expect("chunk");
    let parsed = Frame::parse(&frame, &limits()).expect("parse");
    assert_eq!(
        decode_activation_chunk_range(&parsed, 2, 10, &limits()).expect("range"),
        elements[2..10].to_vec()
    );
    assert_eq!(
        decode_activation_chunk_range(&parsed, 0, 99, &limits())
            .expect_err("out of range")
            .class(),
        ErrorClass::Malformed
    );
}

fn vector_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1/activation-stream")
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|entry| entry.expect("entry").path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    files.sort();
    files
}

fn load(path: &Path) -> Json {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hex_field(value: &Json, key: &str) -> Vec<u8> {
    hex::decode(value[key].as_str().expect("hex field")).expect("hex")
}

#[test]
fn conformance_vectors_replay_and_reencode() {
    let dir = vector_dir();
    let files = json_files(&dir);
    assert_eq!(files.len(), 11, "3 valid + 8 rejects");
    let mut rejects = 0u64;
    for path in files {
        let vector = load(&path);
        let name = vector["name"].as_str().unwrap_or("?");
        if let Some(expected) = vector.get("expect_error").and_then(Json::as_str) {
            let frame = hex_field(&vector, "frame_hex");
            let parsed = Frame::parse(&frame, &limits()).unwrap_or_else(|e| panic!("{name}: {e}"));
            let error = decode_activation_chunk(&parsed, &limits()).expect_err(name);
            assert_eq!(error.class().as_str(), expected, "{name}: {error}");
            rejects += 1;
            continue;
        }
        let head = ActivationStreamHead {
            stream: StreamKind::from_name(vector["stream"].as_str().unwrap()).unwrap(),
            step: vector["step"].as_u64().unwrap(),
            rank: vector["rank"].as_u64().unwrap() as u32,
            world: vector["world"].as_u64().unwrap() as u32,
            chunk_count: vector["chunk_count"].as_u64().unwrap(),
            dtype: StreamDtype::from_name(vector["dtype"].as_str().unwrap()).unwrap(),
            shape: vector["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|dim| dim.as_u64().unwrap())
                .collect(),
        };
        let mut frames = Vec::new();
        for chunk in vector["chunks"].as_array().expect("chunks") {
            let chunk_frame = hex_field(chunk, "frame_hex");
            let elements = hex_field(chunk, "elements_hex");
            let parsed_chunk =
                Frame::parse(&chunk_frame, &limits()).unwrap_or_else(|e| panic!("{name}: {e}"));
            let decoded = decode_activation_chunk(&parsed_chunk, &limits())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(decoded.bytes, elements, "{name}");
            assert_eq!(
                decoded.meta.digest_root.to_hex(),
                chunk["digest_root"].as_str().unwrap(),
                "{name}"
            );
            let reencoded = encode_activation_chunk(
                &head,
                chunk["chunk_index"].as_u64().unwrap(),
                &elements,
                &limits(),
            )
            .expect("re-encode");
            assert_eq!(
                reencoded, chunk_frame,
                "{name}: Go re-encode is not identical"
            );
            frames.push(chunk_frame);
        }
        let refs: Vec<&[u8]> = frames.iter().map(|frame| frame.as_slice()).collect();
        let assembly = assemble_stream(&refs, &limits()).expect("assemble");
        assert_eq!(
            assembly.stream_root.to_hex(),
            vector["stream_root"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(vector["schema"].as_str().unwrap(), ACTIVATION_STREAM_SCHEMA);
    }
    assert_eq!(rejects, 8);
}
