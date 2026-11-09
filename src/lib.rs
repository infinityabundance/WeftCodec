//! WeftCodec v1 — deterministic, lossless, context-conditioned codec.
//!
//! This crate implements the `weftcodec-format 1.0` wire format from
//! `spec/WEFTCODEC-FORMAT.md` and the `weft-selectors 1.0` language from
//! `spec/WEFT-SELECTORS.md`:
//!
//! * [`hash`] — domain-separated BLAKE3-256 digests.
//! * [`varint`] — minimal LEB128 with strict canonical validation.
//! * [`wcb`] — Weft Canonical Bytes, a strict deterministic structured format.
//! * [`frame`] — the 96-byte WFT1 header, dependency and chunk tables, trailer.
//! * [`modes`] — RAW, ZSTD, REFERENCE, DELTA, DICTIONARY, STRUCTURED.
//! * [`dependency`] — resolvers, frame sources, recursive DAG decoding.
//! * [`archive`] — the self-contained WFA1 transport container.
//! * [`selector`] — WeftQL parser, normalization, canonical selector identity.
//! * [`context`] — Context Root, causal closure, capability scope, `weft://`.
//! * [`stream`] — chunked encoding and random-range reads.
//! * [`tensor`] — logical tensor identity, layouts, and region addressing.
//!
//! All parsing of untrusted input returns `Result`; the crate denies unsafe
//! code. The single exception is [`simd`], which contains the AVX2 intrinsics
//! and carries a local `allow(unsafe_code)`; `tests/no_unsafe_outside_simd.rs`
//! fails if `unsafe` appears in any other source file.

#![deny(unsafe_code)]

pub mod activation_stream;
pub mod adversarial;
pub mod archive;
pub mod capability;
pub mod conformance;
pub mod context;
pub mod dependency;
pub mod error;
/// **EXPERIMENTAL — not a wire-format module.** See [`experimental`].
pub mod experimental;
pub mod frame;
pub mod generated;
pub mod hash;
pub mod huffman;
pub mod kv_cache;
pub mod kv_prov;
pub mod limits;
pub mod modes;
pub mod selector;
pub mod semantic;
pub mod semantic_delta;
pub mod simd;
pub mod sparse_delta;
pub mod stream;
pub mod tensor;
pub mod tensor_sharded;
pub mod varint;
pub mod wcb;
pub mod weight_delta;

pub use activation_stream::{
    activation_chunk_metadata, assemble_stream, decode_activation_chunk,
    decode_activation_chunk_range, encode_activation_chunk, stream_root, ActivationChunkSet,
    ActivationStreamAssembly, ActivationStreamHead, ActivationStreamMeta, ActivationStreamWriter,
    DecodedActivationChunk, StreamDtype, StreamKind, ACTIVATION_STREAM_SCHEMA,
};
pub use capability::{
    public_key_for_seed, sign_bytes, sign_envelope, verify_bytes, verify_envelope,
    verify_envelope_for_tenant, CapabilityBlob, CapabilityEnvelope, EnvelopeError, EnvelopeResult,
    Rejection, VerifiedEnvelope,
};
pub use error::{Error, ErrorClass, Result};
pub use frame::{Dependency, Frame, FrameHeader, Mode};
pub use hash::{DomainHasher, Hash32};
pub use kv_cache::{
    decode_kv_page, decode_kv_page_range, encode_kv_page, kv_cache_metadata, DecodedKvPage,
    KvCacheInput, KvCacheMeta, KvDtype, KvPageSet, KV_CACHE_SCHEMA,
};
pub use limits::Limits;
pub use modes::{
    decode_float_tile, decode_float_tile_streams, decode_float_tiled, decode_float_tiled_streams,
    decode_float_tiled_with_threads, decode_tiled_tensor, decode_tiled_tile, encode_float_tiled,
    encode_float_tiled_streams, encode_tiled_tensor, float_tile_root, float_tile_streams_root,
    float_tiled_metadata, float_tiled_region_indices, float_tiled_streams_metadata,
    float_tiled_streams_region_indices, interleaved_stream_len, parse_float_tiled,
    parse_float_tiled_descriptor, parse_float_tiled_streams, parse_float_tiled_streams_descriptor,
    parse_tiled_tensor, place_float_tile, place_float_tile_region, place_float_tiled_streams_tile,
    place_float_tiled_streams_tile_region, place_tiled_tile, place_tiled_tile_region,
    read_float_region, read_float_region_with, read_float_tiled_streams_region,
    read_region as read_tiled_region, tensor_tile_root, tiled_region_indices,
    tiled_tensor_metadata, FloatTiledDescriptor, FloatTiledEntry, FloatTiledMeta,
    FloatTiledStreamsDescriptor, FloatTiledStreamsEntry, FloatTiledStreamsMeta, TensorDtype,
    TiledTensorDescriptor, TiledTensorEntry, TiledTensorMeta, FLOAT_TILED_LAYOUT,
    FLOAT_TILED_SCHEMA, FLOAT_TILED_STREAMS_DEFAULT, FLOAT_TILED_STREAMS_MAX,
    FLOAT_TILED_STREAMS_MIN, FLOAT_TILED_STREAMS_SCHEMA, TENSOR_TILED_SCHEMA, TENSOR_TILE_CODEC,
    TENSOR_TILE_LAYOUT,
};
pub use semantic::{
    select_semantic_base, semantic_matches, semantic_record, semantic_root,
    semantic_root_with_limits, SemanticPolicy, SemanticRootRecord, DOMAIN_TENSOR_SEMANTIC,
    DOMAIN_TENSOR_SEMANTIC_VALUES, SEMANTIC_SCHEMA,
};
pub use semantic_delta::{
    base_semantic_root, decode_frame as decode_semantic_delta,
    decode_with_base_frame as decode_semantic_delta_with_base, encode_semantic_delta,
    semantic_delta_metadata, SemanticDeltaMeta, SEMANTIC_DELTA_SCHEMA,
};
pub use sparse_delta::{
    decode_frame as decode_sparse_delta, decode_with_base_frame as decode_sparse_delta_with_base,
    encode_sparse_delta, sparse_delta_metadata, SparseDeltaMeta, SPARSE_DELTA_CODEC,
    SPARSE_DELTA_OP, SPARSE_DELTA_SCHEMA,
};
pub use tensor::{
    logical_root, logical_root_from_values_hash, shard_root, values_hasher, LayoutDescriptor,
    LogicalRecord, PhysicalTensor, Region, RegionRead, TensorLayout,
};
pub use tensor_sharded::{
    assemble_sharded_words, decode_shard_frame, encode_shard_frame, encode_sharded_tensor,
    shard_set_root, DecodedTensorShard, ShardedDtype, ShardedTensorInput, TensorShardedMeta,
};
pub use weight_delta::{
    base_frame_descriptor, decode_frame as decode_weight_delta, decode_with_base_frame,
    encode_weight_delta, encode_weight_delta_with_resolver, weight_delta_metadata, WeightDeltaMeta,
    WeightDtype,
};
