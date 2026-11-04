//! Mode 10 `TENSOR_SHARDED` exactness and shard-set tests.

use weftcodec::huffman;
use weftcodec::tensor;
use weftcodec::tensor_sharded::{
    assemble_sharded_words, decode_shard_frame, encode_shard_frame, encode_shard_frame_with_stored,
    encode_sharded_tensor, logical_root, shard_set_root, DecodedTensorShard, ShardedDtype,
    ShardedTensorInput,
};
use weftcodec::{Frame, Limits, TensorDtype};

fn limits() -> Limits {
    Limits::default()
}

fn pattern_q8(count: usize) -> Vec<i8> {
    (0..count)
        .map(|i| (((i * 7 + 3) % 251) as i32 - 125) as i8)
        .collect()
}

fn pattern_q4(count: usize) -> Vec<i8> {
    (0..count).map(|i| (i % 16) as i8 - 8).collect()
}

fn q8_case() -> ShardedTensorInput {
    ShardedTensorInput {
        dtype: ShardedDtype::Q8_0,
        shape: vec![4, 8],
        shard_axis: 0,
        shard_count: 2,
        block_size: 16,
        scales: vec![0.5f32.to_bits(), 0.25f32.to_bits()],
        zero_points: vec![0, 3],
        words: pattern_q8(32),
    }
}

fn q4_case() -> ShardedTensorInput {
    ShardedTensorInput {
        dtype: ShardedDtype::Q4_0,
        shape: vec![2, 4, 8],
        shard_axis: 1,
        shard_count: 4,
        block_size: 16,
        scales: vec![
            0.5f32.to_bits(),
            0.25f32.to_bits(),
            0.125f32.to_bits(),
            0.0625f32.to_bits(),
        ],
        zero_points: vec![0, 1, -1, 2],
        words: pattern_q4(64),
    }
}

fn decode_all(frames: &[Vec<u8>], limits: &Limits) -> Vec<DecodedTensorShard> {
    frames
        .iter()
        .map(|frame| {
            let parsed = Frame::parse(frame, limits).expect("parse");
            decode_shard_frame(&parsed, limits).expect("decode")
        })
        .collect()
}

fn expected_shard_words(input: &ShardedTensorInput, index: u32) -> Vec<i8> {
    let bytes: Vec<u8> = input.words.iter().map(|word| *word as u8).collect();
    let shards = tensor::split_shards(
        TensorDtype::I8,
        &input.shape,
        input.shard_axis,
        input.shard_count,
        &bytes,
        &limits(),
    )
    .expect("split");
    shards[index as usize]
        .iter()
        .map(|byte| *byte as i8)
        .collect()
}

#[test]
fn q8_0_rank2_two_shards_roundtrip() {
    let limits = limits();
    let input = q8_case();
    let frames = encode_sharded_tensor(&input, &limits).expect("encode");
    assert_eq!(frames.len(), 2);

    let root = logical_root(
        input.dtype,
        &input.shape,
        &input.words,
        &input.scales,
        &input.zero_points,
        &limits,
    )
    .expect("root");

    let decoded = decode_all(&frames, &limits);
    for (index, shard) in decoded.iter().enumerate() {
        assert_eq!(shard.meta.shard_index, index as u32);
        assert_eq!(shard.meta.layout_root, root);
        assert_eq!(shard.words, expected_shard_words(&input, index as u32));
        assert_eq!(shard.planes.len() as u64, shard.meta.planes_len().unwrap());
        let reparsed = encode_shard_frame(&shard.meta, &shard.planes, &limits).expect("re-encode");
        assert_eq!(reparsed, frames[index], "frame bytes must be deterministic");
    }

    assert_eq!(
        assemble_sharded_words(&decoded, &limits).expect("assemble"),
        input.words
    );

    // The shard index is authoritative; a shuffled set is accepted.
    let mut shuffled = decoded.clone();
    shuffled.reverse();
    assert_eq!(shard_set_root(&shuffled, &limits).expect("root"), root);
}

#[test]
fn q4_0_rank3_four_shards_roundtrip_and_nibbles() {
    let limits = limits();
    let input = q4_case();
    let frames = encode_sharded_tensor(&input, &limits).expect("encode");
    assert_eq!(frames.len(), 4);

    let decoded = decode_all(&frames, &limits);
    for (index, shard) in decoded.iter().enumerate() {
        let expected = expected_shard_words(&input, index as u32);
        assert_eq!(shard.words, expected);
        // Low nibble is the even element, high nibble the odd element.
        let quant_plane_len = expected.len() / 2;
        for pair in 0..quant_plane_len {
            let low = (shard.words[2 * pair] as u8) & 0x0f;
            let high = (shard.words[2 * pair + 1] as u8) & 0x0f;
            assert_eq!(shard.planes[pair], low | (high << 4));
        }
    }

    let mut shuffled = decoded.clone();
    shuffled.rotate_left(2);
    let root = shard_set_root(&shuffled, &limits).expect("root");
    assert_eq!(
        root,
        logical_root(
            input.dtype,
            &input.shape,
            &input.words,
            &input.scales,
            &input.zero_points,
            &limits
        )
        .expect("logical root")
    );
}

#[test]
fn q8_0_rank1_single_shard_roundtrip() {
    let limits = limits();
    let input = ShardedTensorInput {
        dtype: ShardedDtype::Q8_0,
        shape: vec![16],
        shard_axis: 0,
        shard_count: 1,
        block_size: 16,
        scales: vec![1.0f32.to_bits()],
        zero_points: vec![0],
        words: pattern_q8(16),
    };
    let frames = encode_sharded_tensor(&input, &limits).expect("encode");
    assert_eq!(frames.len(), 1);
    let decoded = decode_all(&frames, &limits);
    assert_eq!(decoded[0].words, input.words);
    assert_eq!(
        shard_set_root(&decoded, &limits).expect("root"),
        decoded[0].meta.layout_root
    );
}

#[test]
fn logical_root_is_independent_of_shard_axis_and_count() {
    let limits = limits();
    let words = pattern_q8(32);
    let scales = [0.5f32.to_bits(), 0.25f32.to_bits()];
    let zero_points = [0_i64, 3];
    let root = logical_root(
        ShardedDtype::Q8_0,
        &[4, 8],
        &words,
        &scales,
        &zero_points,
        &limits,
    )
    .expect("logical root");

    let mut input = q8_case();
    input.shard_axis = 1;
    let frames = encode_sharded_tensor(&input, &limits).expect("encode axis 1");
    let decoded = decode_all(&frames, &limits);
    assert_eq!(shard_set_root(&decoded, &limits).expect("root"), root);
}

#[test]
fn duplicate_and_missing_shards_are_rejected() {
    let limits = limits();
    let frames = encode_sharded_tensor(&q8_case(), &limits).expect("encode");
    let decoded = decode_all(&frames, &limits);

    let mut duplicate = decoded.clone();
    duplicate[1] = decoded[0].clone();
    let error = shard_set_root(&duplicate, &limits).expect_err("duplicate");
    assert_eq!(error.class(), weftcodec::ErrorClass::NonCanonical);

    let missing = vec![decoded[0].clone()];
    let error = shard_set_root(&missing, &limits).expect_err("missing");
    assert_eq!(error.class(), weftcodec::ErrorClass::Malformed);
}

#[test]
fn element_count_and_block_size_and_divisibility_and_index_rejects() {
    let limits = limits();
    let input = q8_case();
    let frames = encode_sharded_tensor(&input, &limits).expect("encode");
    let decoded = decode_all(&frames, &limits);
    let shard = &decoded[0];
    let stored = huffman::encode(&shard.planes);

    let mut meta = shard.meta.clone();
    meta.element_count = 31;
    let mut with_stored = meta.clone();
    with_stored.payload_len = stored.len() as u64;
    let bad = encode_shard_frame_with_stored(&with_stored, &shard.planes, &stored, &limits)
        .expect("build");
    let parsed = Frame::parse(&bad, &limits).expect("parse");
    assert_eq!(
        decode_shard_frame(&parsed, &limits).unwrap_err().class(),
        weftcodec::ErrorClass::Malformed
    );

    let mut meta = shard.meta.clone();
    meta.block_size = 24;
    meta.payload_len = stored.len() as u64;
    let bad =
        encode_shard_frame_with_stored(&meta, &shard.planes, &stored, &limits).expect("build");
    let parsed = Frame::parse(&bad, &limits).expect("parse");
    assert_eq!(
        decode_shard_frame(&parsed, &limits).unwrap_err().class(),
        weftcodec::ErrorClass::Malformed
    );

    let mut meta = shard.meta.clone();
    meta.shard_axis = 1;
    meta.shard_count = 3;
    meta.payload_len = stored.len() as u64;
    let bad =
        encode_shard_frame_with_stored(&meta, &shard.planes, &stored, &limits).expect("build");
    let parsed = Frame::parse(&bad, &limits).expect("parse");
    assert_eq!(
        decode_shard_frame(&parsed, &limits).unwrap_err().class(),
        weftcodec::ErrorClass::Malformed
    );

    let mut meta = shard.meta.clone();
    meta.shard_index = 2;
    meta.payload_len = stored.len() as u64;
    let bad =
        encode_shard_frame_with_stored(&meta, &shard.planes, &stored, &limits).expect("build");
    let parsed = Frame::parse(&bad, &limits).expect("parse");
    assert_eq!(
        decode_shard_frame(&parsed, &limits).unwrap_err().class(),
        weftcodec::ErrorClass::Malformed
    );
}

#[test]
fn scale_plane_mismatch_and_malformed_huffman_are_rejected() {
    let limits = limits();
    let frames = encode_sharded_tensor(&q8_case(), &limits).expect("encode");
    let decoded = decode_all(&frames, &limits);
    let shard = &decoded[0];

    let mut planes = shard.planes.clone();
    let last = planes.len() - 1;
    planes[last] ^= 0x01;
    let bad = encode_shard_frame(&shard.meta, &planes, &limits).expect("build");
    let parsed = Frame::parse(&bad, &limits).expect("parse");
    assert_eq!(
        decode_shard_frame(&parsed, &limits).unwrap_err().class(),
        weftcodec::ErrorClass::Corrupt
    );

    let mut stored = huffman::encode(&shard.planes);
    stored.truncate(stored.len() - 1);
    let mut meta = shard.meta.clone();
    meta.payload_len = stored.len() as u64;
    let bad =
        encode_shard_frame_with_stored(&meta, &shard.planes, &stored, &limits).expect("build");
    let parsed = Frame::parse(&bad, &limits).expect("parse");
    let class = decode_shard_frame(&parsed, &limits).unwrap_err().class();
    assert!(
        class == weftcodec::ErrorClass::Corrupt || class == weftcodec::ErrorClass::NonCanonical,
        "unexpected class {class}"
    );
}

#[test]
fn limits_fail_closed_before_allocation() {
    let mut rank_limits = limits();
    rank_limits.max_tensor_rank = 2;
    let input = ShardedTensorInput {
        dtype: ShardedDtype::Q8_0,
        shape: vec![2, 2, 2],
        shard_axis: 0,
        shard_count: 1,
        block_size: 16,
        scales: vec![1.0f32.to_bits()],
        zero_points: vec![0],
        words: vec![0; 8],
    };
    let error = encode_sharded_tensor(&input, &rank_limits).expect_err("rank limit");
    assert_eq!(error.class(), weftcodec::ErrorClass::LimitExceeded);

    let mut element_limits = limits();
    element_limits.max_tensor_elements = 4;
    let input = ShardedTensorInput {
        dtype: ShardedDtype::Q8_0,
        shape: vec![8],
        shard_axis: 0,
        shard_count: 1,
        block_size: 16,
        scales: vec![1.0f32.to_bits()],
        zero_points: vec![0],
        words: vec![0; 8],
    };
    let error = encode_sharded_tensor(&input, &element_limits).expect_err("element limit");
    assert_eq!(error.class(), weftcodec::ErrorClass::LimitExceeded);
}
