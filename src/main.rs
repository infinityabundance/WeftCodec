//! `weftcodec` command-line interface.
#![forbid(unsafe_code)]

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use serde_json::json;

use weftcodec::activation_stream;
use weftcodec::archive;
use weftcodec::capability::{self, CapabilityEnvelope};
use weftcodec::context::{self, Budget, ContextRoot, ContextUri, DEFAULT_RESOLVER};
use weftcodec::dependency::{
    decode_frame_dag, DirectoryFrameSource, DirectoryResolver, MapResolver, Resolver,
};
use weftcodec::engram;
use weftcodec::error::{Error, ErrorClass, Result};
use weftcodec::frame::{Frame, Mode};
use weftcodec::hash::hash_domain;
use weftcodec::kv_cache;
use weftcodec::kv_prov;
use weftcodec::limits::{self, Limits};
use weftcodec::modes;
use weftcodec::selector;
use weftcodec::semantic::SemanticPolicy;
use weftcodec::semantic_delta;
use weftcodec::sparse_delta;
use weftcodec::tensor;
use weftcodec::tensor_sharded;
use weftcodec::wcb;
use weftcodec::weight_delta;

#[derive(Parser, Debug)]
#[command(
    name = "weftcodec",
    version,
    about = "WeftCodec v1 encoder/decoder (WFT1 frames, WCB, WeftQL selectors)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Encode input bytes (or canonical WCB for structured mode).
    Encode(EncodeArgs),
    /// Decode a frame to plaintext.
    Decode(DecodeArgs),
    /// Inspect frame metadata.
    Inspect(InspectArgs),
    /// Validate a frame and verify all hashes (resolving dependencies if given).
    Verify(VerifyArgs),
    /// Self-contained WFA1 archive operations.
    Archive {
        #[command(subcommand)]
        command: ArchiveCommand,
    },
    /// WeftQL selector identity operations.
    Selector {
        #[command(subcommand)]
        command: SelectorCommand,
    },
    /// Context Root identity and `weft://` addressing.
    Context {
        #[command(subcommand)]
        command: ContextCommand,
    },
    /// Causal closure identity.
    Causal {
        #[command(subcommand)]
        command: CausalCommand,
    },
    /// Authenticated capability envelope operations.
    Capability {
        #[command(subcommand)]
        command: CapabilityCommand,
    },
    /// Logical tensor identity, sharding, and region addressing.
    Tensor {
        #[command(subcommand)]
        command: TensorCommand,
    },
    /// Conformance vector generation (maintainer tool).
    Conformance(ConformanceArgs),
}

#[derive(Args, Debug)]
struct EncodeArgs {
    /// Input path, or `-` for stdin.
    #[arg(long)]
    input: String,
    /// Output path, or `-` for stdout.
    #[arg(long)]
    output: String,
    /// Encoding mode.
    #[arg(
        long,
        default_value = "auto",
        value_parser = ["raw", "zstd", "reference", "delta", "dictionary", "structured", "tensor", "tokens", "quantized", "quantized_tensor", "float_tensor", "float-tensor", "tensor_sharded", "tensor-sharded", "tensor_tiled", "tensor-tiled", "float_tiled", "float-tiled", "float_tiled_streams", "float-tiled-streams", "weight_delta", "weight-delta", "semantic_delta", "semantic-delta", "sparse_delta", "sparse-delta", "kv_cache", "kv-cache", "kv_prov", "kv-prov", "activation_stream", "activation-stream", "engram_shard", "engram-shard", "auto"]
    )]
    mode: String,
    /// Base bytes for reference/delta/auto.
    #[arg(long)]
    base: Option<String>,
    /// Raw-content zstd dictionary for dictionary/auto.
    #[arg(long)]
    dict: Option<String>,
    /// zstd level (informational in the header; used by the encoder).
    #[arg(long, default_value_t = 3)]
    level: i32,
    /// Chunk size in bytes (RAW and ZSTD only).
    #[arg(long)]
    chunk_size: Option<usize>,
    /// TENSOR dtype (`f16|bf16|f32|f64|i8|u8|i16|i32|i64`).
    #[arg(long)]
    dtype: Option<String>,
    /// TENSOR shape as comma-separated u64 extents (empty means scalar).
    #[arg(long)]
    shape: Option<String>,
    /// TENSOR metadata JSON file: `{"dtype":"f32","shape":[2,3]}`.
    #[arg(long)]
    metadata: Option<String>,
    /// KV_CACHE_PROV (mode 19) provenance JSON. Required for that mode.
    #[arg(long)]
    provenance: Option<String>,
    /// QUANTIZED_TENSOR affine scale (finite, > 0). Required for that mode.
    #[arg(long)]
    scale: Option<f64>,
    /// QUANTIZED_TENSOR zero point (i64, within the dtype range).
    #[arg(long = "zero-point", default_value_t = 0)]
    zero_point: i64,
    /// TENSOR_SHARDED shard axis.
    #[arg(long)]
    axis: Option<u32>,
    /// TENSOR_SHARDED shard count.
    #[arg(long)]
    shards: Option<u32>,
    /// TENSOR_SHARDED quantization block size (16, 32, 64, 128, or 256).
    #[arg(long = "block-size")]
    block_size: Option<u64>,
    /// TENSOR_SHARDED per-block f32 scale bits, comma-separated u32 values.
    #[arg(long)]
    scales: Option<String>,
    /// TENSOR_SHARDED per-block zero points, comma-separated i64 values.
    #[arg(long = "zero-points")]
    zero_points: Option<String>,
    /// TENSOR_TILED tile extents as comma-separated u64 values (rank-equal and
    /// dividing every shape extent).
    #[arg(long = "tile-shape")]
    tile_shape: Option<String>,
    /// SEMANTIC_DELTA policy: `none`, `mantissa_truncate:N`, or
    /// `block_quantize:B:L`.
    #[arg(long = "semantic-policy")]
    semantic_policy: Option<String>,
    /// FLOAT_TILED_STREAMS per-tile interleaved stream count (1..=16, default 4).
    #[arg(long = "n-streams")]
    n_streams: Option<u64>,
    /// ENGRAM_SHARD row-group edges as comma-separated u64 values (first 0,
    /// last = rows, strictly increasing).
    #[arg(long = "group-edges")]
    group_edges: Option<String>,
    /// ENGRAM_SHARD key-hash registry id (`ngram-sha256-v1` or
    /// `ngram-blake3-v1`).
    #[arg(long = "hash-function", default_value = engram::ENGRAM_HASH_BLAKE3)]
    hash_function: String,
    /// ENGRAM_SHARD vocabulary size (1..=rows).
    #[arg(long = "vocab-size")]
    vocab_size: Option<u64>,
    /// ENGRAM_SHARD declared precision (defaults to `--dtype`).
    #[arg(long)]
    precision: Option<String>,
    /// ENGRAM_SHARD optional quantization scheme (only `none` is implemented).
    #[arg(long)]
    quantization: Option<String>,
}

#[derive(serde::Deserialize)]
struct TensorMetaInput {
    dtype: String,
    shape: Vec<u64>,
    #[serde(default)]
    layout: Option<String>,
    #[serde(default)]
    endianness: Option<String>,
    #[serde(default)]
    scale: Option<f64>,
    #[serde(default)]
    zero_point: Option<i64>,
}

/// TENSOR_SHARDED metadata JSON.
#[derive(serde::Deserialize)]
struct ShardedMetaInput {
    dtype: String,
    shape: Vec<u64>,
    axis: u32,
    shards: u32,
    block_size: u64,
    scales: Vec<u64>,
    zero_points: Vec<i64>,
}

/// KV_CACHE (mode 12) metadata JSON.
#[derive(serde::Deserialize)]
struct KvCacheMetaInput {
    model: String,
    layer: u32,
    session: String,
    page_index: u64,
    page_count: u64,
    seq_start: u64,
    seq_end: u64,
    dtype: String,
    head_count: u32,
    head_dim: u32,
}

/// ACTIVATION_STREAM (mode 13) metadata JSON.
#[derive(serde::Deserialize)]
struct ActivationStreamMetaInput {
    stream: String,
    step: u64,
    rank: u32,
    world: u32,
    chunk_index: u64,
    chunk_count: u64,
    dtype: String,
    shape: Vec<u64>,
}

#[derive(serde::Deserialize)]
struct TokensInput {
    vocab_schema: String,
    token_ids: Vec<u64>,
}

/// Decoder resource limits. Defaults are the normative v1 values; every flag
/// can only lower the corresponding bound.
#[derive(Args, Debug, Clone)]
struct LimitArgs {
    #[arg(long, default_value_t = limits::DEFAULT_MAX_FRAME_BYTES)]
    max_frame_bytes: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_DECODED_BYTES)]
    max_decoded_bytes: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_DEPENDENCY_DEPTH)]
    max_dependency_depth: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_DEPENDENCY_COUNT)]
    max_dependency_count: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_TOTAL_FETCHED_BYTES)]
    max_total_fetched_bytes: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_CHUNKS)]
    max_chunks: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_DICTIONARY_BYTES)]
    max_dictionary_bytes: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_DELTA_OPS)]
    max_delta_ops: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_TENSOR_ELEMENTS)]
    max_tensor_elements: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_TENSOR_RANK)]
    max_tensor_rank: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_TOKENS)]
    max_tokens: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_SPARSE_RUNS)]
    max_sparse_runs: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_STRUCTURED_DEPTH)]
    max_structured_depth: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_STRUCTURED_ITEMS)]
    max_structured_items: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_SELECTOR_BYTES)]
    max_selector_bytes: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_SELECTOR_DEPTH)]
    max_selector_depth: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_SELECTOR_OPERANDS)]
    max_selector_operands: u64,
    #[arg(long, default_value_t = limits::DEFAULT_MAX_ARCHIVE_ENTRIES)]
    max_archive_entries: u64,
}

impl LimitArgs {
    fn to_limits(&self) -> Limits {
        Limits {
            max_frame_bytes: self.max_frame_bytes,
            max_decoded_bytes: self.max_decoded_bytes,
            max_dependency_depth: self.max_dependency_depth,
            max_dependency_count: self.max_dependency_count,
            max_total_fetched_bytes: self.max_total_fetched_bytes,
            max_chunks: self.max_chunks,
            max_dictionary_bytes: self.max_dictionary_bytes,
            max_delta_ops: self.max_delta_ops,
            max_tensor_elements: self.max_tensor_elements,
            max_tensor_rank: self.max_tensor_rank,
            max_tokens: self.max_tokens,
            max_sparse_runs: self.max_sparse_runs,
            max_structured_depth: self.max_structured_depth,
            max_structured_items: self.max_structured_items,
            max_selector_bytes: self.max_selector_bytes,
            max_selector_depth: self.max_selector_depth,
            max_selector_operands: self.max_selector_operands,
            max_archive_entries: self.max_archive_entries,
            ..Limits::default()
        }
    }
}

#[derive(Args, Debug)]
struct DecodeArgs {
    #[arg(long)]
    input: String,
    #[arg(long)]
    output: String,
    /// Base frame for WEIGHT_DELTA (a tensor-carrying frame).
    #[arg(long)]
    base: Option<String>,
    /// Directory of dependency plaintext files named by lowercase hex digest.
    #[arg(long)]
    deps: Option<String>,
    /// Directory of dependency *frames* named by content hash. When present,
    /// decode walks the dependency DAG (depth, cycle, and fetch limits apply).
    #[arg(long)]
    frame_source: Option<String>,
    /// Resolve a WEIGHT_DELTA base through the frame source (requires
    /// `--frame-source`) instead of supplying it with `--base`.
    #[arg(long)]
    resolve: bool,
    #[command(flatten)]
    limits: LimitArgs,
}

#[derive(Args, Debug)]
struct InspectArgs {
    /// Frame file.
    file: String,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct VerifyArgs {
    /// Frame file.
    file: String,
    /// Directory of dependency files named by lowercase hex digest.
    #[arg(long)]
    deps: Option<String>,
}

#[derive(Subcommand, Debug)]
enum ArchiveCommand {
    /// Create an archive containing the given frames.
    Create {
        #[arg(long)]
        output: String,
        /// Frame files (any order; creation sorts topologically).
        #[arg(required = true)]
        frames: Vec<String>,
    },
    /// Decode an archive into a directory of plaintext files.
    Decode {
        #[arg(long)]
        input: String,
        #[arg(long)]
        out_dir: String,
        #[command(flatten)]
        limits: LimitArgs,
    },
}

#[derive(Subcommand, Debug)]
enum SelectorCommand {
    /// Print the selector id (hex, or JSON).
    Id {
        #[arg(long)]
        input: String,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        limits: LimitArgs,
    },
    /// Print canonical WCB bytes as lowercase hex.
    Canonicalize {
        #[arg(long)]
        input: String,
        #[command(flatten)]
        limits: LimitArgs,
    },
}

#[derive(Args, Debug)]
struct ContextArgs {
    /// Selector source file, or `-` for stdin.
    #[arg(long)]
    selector: String,
    /// Namespace for the selector and the Context Root.
    #[arg(long)]
    namespace: String,
    /// Pinned snapshot commit hash (64 lowercase hex digits).
    #[arg(long)]
    snapshot: String,
    /// Comma-separated causal closure entries (64 hex digits each).
    #[arg(long)]
    causal: Option<String>,
    /// Resolver/policy version.
    #[arg(long, default_value = DEFAULT_RESOLVER)]
    resolver: String,
    /// Optional capability scope hash (64 hex digits).
    #[arg(long)]
    capability_hash: Option<String>,
    /// Resource budget, e.g. `max_results=10,max_bytes=1000,max_depth=3`.
    #[arg(long)]
    budget: Option<String>,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct ContextUriArgs {
    #[command(flatten)]
    base: ContextArgs,
    /// Emit the descriptive URI instead of the authoritative URI.
    #[arg(long)]
    descriptor: bool,
}

#[derive(Subcommand, Debug)]
enum ContextCommand {
    /// Print the ContextRootId (hex, or JSON).
    Id(ContextArgs),
    /// Print the canonical `weft://` address.
    Uri(ContextUriArgs),
    /// Parse an authoritative or descriptive `weft://` URI.
    Parse {
        /// The URI to parse.
        uri: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum CapabilityCommand {
    /// Sign an envelope JSON document, writing the canonical blob.
    Sign {
        /// Envelope JSON file (`weft.capability.envelope.v1`).
        #[arg(long)]
        envelope: String,
        /// File containing a 32-byte Ed25519 seed (hex or raw).
        #[arg(long)]
        key: String,
        /// Output blob path, or `-` for stdout.
        #[arg(long, default_value = "-")]
        out: String,
        #[arg(long)]
        json: bool,
    },
    /// Verify a canonical `{envelope, signature}` blob offline.
    Verify {
        #[arg(long)]
        blob: String,
        /// Revocation ids: hex lines, comma-separated, or a JSON array.
        #[arg(long)]
        revocations: Option<String>,
        /// Override the verification clock (UTC ns).
        #[arg(long = "now-ns")]
        now_ns: Option<i64>,
        /// Require this grantee (64 hex digits).
        #[arg(long)]
        grantee: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Print an envelope identity from JSON or canonical hex.
    Id {
        #[arg(long)]
        envelope: String,
        #[arg(long)]
        json: bool,
    },
    /// Print the Ed25519 public key derived from a seed file.
    Key {
        #[arg(long)]
        key: String,
    },
}

#[derive(Subcommand, Debug)]
enum CausalCommand {
    /// Print the causal closure record hash (hex, or JSON).
    Id {
        /// Comma-separated explicit parents (64 hex digits each).
        #[arg(long)]
        parents: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum TensorCommand {
    /// Print the layout-independent logical root of a tensor.
    Root(TensorRootArgs),
    /// Split canonical row-major bytes into N shard containers.
    Split(TensorSplitArgs),
    /// Reassemble shard containers and verify the logical root.
    Reassemble(TensorReassembleArgs),
}

#[derive(Args, Debug)]
struct TensorRootArgs {
    /// Physical tensor file (canonical row-major unless --layout says otherwise).
    #[arg(long)]
    input: Option<String>,
    /// Directory of self-describing shard containers named `shard-<index>.bin`.
    #[arg(long)]
    shards_dir: Option<String>,
    /// Tensor dtype (`f16|bf16|f32|f64|i8|u8|i16|i32|i64`).
    #[arg(long)]
    dtype: Option<String>,
    /// Comma-separated u64 extents.
    #[arg(long)]
    shape: Option<String>,
    /// Read a directory of TENSOR_SHARDED mode-10 frames instead.
    #[arg(long)]
    frames: bool,
    /// Physical layout: row-major, column-major, blocked, or sharded.
    #[arg(long, default_value = "row-major")]
    layout: String,
    /// Comma-separated block extents (required for blocked).
    #[arg(long)]
    block_shape: Option<String>,
    /// Shard axis (required with --shards-dir).
    #[arg(long)]
    axis: Option<u32>,
    /// Shard count (required with --shards-dir).
    #[arg(long)]
    shards: Option<u32>,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct TensorSplitArgs {
    /// Canonical row-major tensor file.
    #[arg(long)]
    input: String,
    /// Tensor dtype.
    #[arg(long)]
    dtype: String,
    /// Comma-separated u64 extents.
    #[arg(long)]
    shape: String,
    /// Axis to shard along.
    #[arg(long)]
    axis: u32,
    /// Number of shards.
    #[arg(long)]
    shards: u32,
    /// Output directory for `shard-<index>.bin` containers.
    #[arg(long)]
    out_dir: String,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct TensorReassembleArgs {
    /// Directory of `shard-<index>.bin` containers.
    #[arg(long)]
    shards_dir: String,
    /// Tensor dtype.
    #[arg(long)]
    dtype: String,
    /// Comma-separated u64 extents.
    #[arg(long)]
    shape: String,
    /// Axis the tensor was sharded along.
    #[arg(long)]
    axis: u32,
    /// Number of shards.
    #[arg(long)]
    shards: u32,
    /// Write the reassembled canonical row-major bytes here.
    #[arg(long)]
    output: Option<String>,
    /// Fail unless the logical root equals this hex digest.
    #[arg(long)]
    expect_root: Option<String>,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug)]
struct ConformanceArgs {
    /// Directory to write vectors into (e.g. conformance/v1).
    #[arg(long)]
    dir: String,
    /// Write only the adversarial decoder-bomb vectors into `<dir>/adversarial`.
    #[arg(long)]
    adversarial: bool,
    /// Write only the WEFT-TENSOR vectors into `<dir>/tensor`.
    #[arg(long)]
    tensor_only: bool,
    /// Write only the TENSOR_SHARDED (mode 10) vectors into `<dir>/sharded`.
    #[arg(long)]
    sharded_only: bool,
    /// Write only the WEIGHT_DELTA (mode 11) vectors into `<dir>/weight-delta`.
    #[arg(long)]
    weight_delta_only: bool,
    /// Write only the SPARSE_DELTA (mode 16) vectors into `<dir>/sparse-delta`.
    #[arg(long)]
    sparse_delta_only: bool,
    /// Write only the SEMANTIC_DELTA (mode 14) and semantic-root vectors into
    /// `<dir>/semantic-delta` and `<dir>/semantic-root`.
    #[arg(long)]
    semantic_delta_only: bool,
    /// Write only the KV_CACHE (mode 12) vectors into `<dir>/kv-cache`.
    #[arg(long)]
    kv_cache_only: bool,
    /// Write only the KV_CACHE_PROV (mode 19) vectors into `<dir>/kv-prov`.
    #[arg(long)]
    kv_prov_only: bool,
    /// Write only the ACTIVATION_STREAM (mode 13) vectors into
    /// `<dir>/activation-stream`.
    #[arg(long)]
    activation_stream_only: bool,
    /// Write only the TENSOR_TILED (mode 15) vectors into `<dir>/tensor-tiled`.
    #[arg(long)]
    tensor_tiled_only: bool,
    /// Write only the FLOAT_TILED (mode 17) vectors into `<dir>/float-tiled`.
    #[arg(long)]
    float_tiled_only: bool,
    /// Write only the FLOAT_TILED_STREAMS (mode 18) vectors into
    /// `<dir>/float-tiled-streams`.
    #[arg(long)]
    float_tiled_streams_only: bool,
    /// Write only the authenticated capability-envelope vectors into
    /// `<dir>/capability`.
    #[arg(long)]
    capability_only: bool,
    /// Write only the ENGRAM_SHARD (mode 20) vectors into `<dir>/engram`.
    #[arg(long)]
    engram_only: bool,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Encode(args) => encode(args),
        Command::Decode(args) => decode(args),
        Command::Inspect(args) => inspect(args),
        Command::Verify(args) => verify(args),
        Command::Archive { command } => match command {
            ArchiveCommand::Create { output, frames } => archive_create(&output, &frames),
            ArchiveCommand::Decode {
                input,
                out_dir,
                limits,
            } => archive_decode(&input, &out_dir, &limits.to_limits()),
        },
        Command::Selector { command } => match command {
            SelectorCommand::Id {
                input,
                json,
                limits,
            } => selector_id(&input, json, &limits.to_limits()),
            SelectorCommand::Canonicalize { input, limits } => {
                selector_canonicalize(&input, &limits.to_limits())
            }
        },
        Command::Context { command } => match command {
            ContextCommand::Id(args) => context_id(&args),
            ContextCommand::Uri(args) => context_uri(&args),
            ContextCommand::Parse { uri, json } => context_parse(&uri, json),
        },
        Command::Causal { command } => match command {
            CausalCommand::Id { parents, json } => causal_id(&parents, json),
        },
        Command::Capability { command } => match command {
            CapabilityCommand::Sign {
                envelope,
                key,
                out,
                json,
            } => capability_sign(&envelope, &key, &out, json),
            CapabilityCommand::Verify {
                blob,
                revocations,
                now_ns,
                grantee,
                json,
            } => capability_verify(
                &blob,
                revocations.as_deref(),
                now_ns,
                grantee.as_deref(),
                json,
            ),
            CapabilityCommand::Id { envelope, json } => capability_id(&envelope, json),
            CapabilityCommand::Key { key } => capability_key(&key),
        },
        Command::Tensor { command } => match command {
            TensorCommand::Root(args) => tensor_root(&args),
            TensorCommand::Split(args) => tensor_split(&args),
            TensorCommand::Reassemble(args) => tensor_reassemble(&args),
        },
        Command::Conformance(args) => {
            let limits = Limits::default();
            if args.adversarial {
                weftcodec::adversarial::generate(Path::new(&args.dir), &limits)?;
                eprintln!("wrote adversarial vectors to {}/adversarial", args.dir);
            } else if args.tensor_only {
                weftcodec::conformance::generate_tensor_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote tensor vectors to {}/tensor", args.dir);
            } else if args.sharded_only {
                weftcodec::conformance::generate_sharded_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote sharded vectors to {}/sharded", args.dir);
            } else if args.weight_delta_only {
                weftcodec::conformance::generate_weight_delta_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote weight-delta vectors to {}/weight-delta", args.dir);
            } else if args.sparse_delta_only {
                weftcodec::conformance::generate_sparse_delta_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote sparse-delta vectors to {}/sparse-delta", args.dir);
            } else if args.semantic_delta_only {
                weftcodec::conformance::generate_semantic_delta_v1(Path::new(&args.dir), &limits)?;
                eprintln!(
                    "wrote semantic-root and semantic-delta vectors to {}/semantic-root and {}/semantic-delta",
                    args.dir, args.dir
                );
            } else if args.kv_cache_only {
                weftcodec::conformance::generate_kv_cache_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote kv-cache vectors to {}/kv-cache", args.dir);
            } else if args.kv_prov_only {
                weftcodec::conformance::generate_kv_prov_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote kv-prov vectors to {}/kv-prov", args.dir);
            } else if args.activation_stream_only {
                weftcodec::conformance::generate_activation_stream_v1(
                    Path::new(&args.dir),
                    &limits,
                )?;
                eprintln!(
                    "wrote activation-stream vectors to {}/activation-stream",
                    args.dir
                );
            } else if args.tensor_tiled_only {
                weftcodec::conformance::generate_tiled_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote tensor-tiled vectors to {}/tensor-tiled", args.dir);
            } else if args.float_tiled_only {
                weftcodec::conformance::generate_float_tiled_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote float-tiled vectors to {}/float-tiled", args.dir);
            } else if args.float_tiled_streams_only {
                weftcodec::conformance::generate_float_tiled_streams_v1(
                    Path::new(&args.dir),
                    &limits,
                )?;
                eprintln!(
                    "wrote float-tiled-streams vectors to {}/float-tiled-streams",
                    args.dir
                );
            } else if args.capability_only {
                weftcodec::conformance::generate_capability_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote capability vectors to {}/capability", args.dir);
            } else if args.engram_only {
                weftcodec::conformance::generate_engram_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote engram vectors to {}/engram", args.dir);
            } else {
                weftcodec::conformance::generate_v1(Path::new(&args.dir), &limits)?;
                eprintln!("wrote conformance vectors to {}", args.dir);
            }
            Ok(())
        }
    }
}

fn encode(args: EncodeArgs) -> Result<()> {
    let limits = Limits::default();
    let data = read_limited(&args.input, limits.max_decoded_bytes)?;
    let base = match &args.base {
        Some(path) => Some(read_limited(path, limits.max_decoded_bytes)?),
        None => None,
    };
    let dict = match &args.dict {
        Some(path) => Some(read_limited(path, limits.max_dictionary_bytes)?),
        None => None,
    };

    let frame = if args.mode == "tensor" {
        let (dtype, shape) = tensor_metadata_args(&args)?;
        modes::encode_tensor(&data, dtype, &shape, &limits)?
    } else if args.mode == "quantized" || args.mode == "quantized_tensor" {
        let (dtype, shape, scale, zero_point) = quantized_metadata_args(&args)?;
        modes::encode_quantized_tensor(&data, dtype, &shape, scale, zero_point, &limits)?
    } else if args.mode == "float_tensor" || args.mode == "float-tensor" {
        let (dtype, shape) = float_tensor_metadata_args(&args)?;
        modes::encode_float_tensor(&data, dtype, &shape, &limits)?
    } else if args.mode == "tensor_sharded" || args.mode == "tensor-sharded" {
        let input = sharded_encode_args(&args, &data, &limits)?;
        let frames = tensor_sharded::encode_sharded_tensor(&input, &limits)?;
        fs::create_dir_all(&args.output)
            .map_err(|e| Error::io(format!("cannot create {}: {e}", args.output)))?;
        for (index, frame) in frames.iter().enumerate() {
            let path = Path::new(&args.output).join(format!("shard-{index}.wft1"));
            fs::write(&path, frame)
                .map_err(|e| Error::io(format!("cannot write {}: {e}", path.display())))?;
        }
        return Ok(());
    } else if args.mode == "tensor_tiled" || args.mode == "tensor-tiled" {
        let (dtype, shape) = tensor_metadata_args(&args)?;
        let tile_shape = tile_shape_arg(&args)?;
        modes::encode_tiled_tensor(&data, dtype, &shape, &tile_shape, &limits)?
    } else if args.mode == "float_tiled" || args.mode == "float-tiled" {
        let (dtype, shape) = float_tensor_metadata_args(&args)?;
        let tile_shape = tile_shape_arg(&args)?;
        modes::encode_float_tiled(&data, dtype, &shape, &tile_shape, &limits)?
    } else if args.mode == "float_tiled_streams" || args.mode == "float-tiled-streams" {
        let (dtype, shape) = float_tensor_metadata_args(&args)?;
        let tile_shape = tile_shape_arg(&args)?;
        let n_streams = args.n_streams.unwrap_or(modes::FLOAT_TILED_STREAMS_DEFAULT);
        modes::encode_float_tiled_streams(&data, dtype, &shape, &tile_shape, n_streams, &limits)?
    } else if args.mode == "engram_shard" || args.mode == "engram-shard" {
        let (dtype, shape) = tensor_metadata_args(&args)?;
        let edges = engram_edges_arg(&args)?;
        let vocab_size = args.vocab_size.unwrap_or(shape[0]);
        let mut input = engram::EngramInput::inline(
            dtype,
            shape.clone(),
            edges,
            args.hash_function.clone(),
            vocab_size,
        );
        if let Some(precision) = &args.precision {
            input.precision.storage_dtype = modes::TensorDtype::from_name(precision)?;
        }
        if let Some(scheme) = &args.quantization {
            input.precision.quantization = Some(engram::EngramQuantization {
                scheme: scheme.clone(),
                lossless: true,
            });
        }
        engram::encode_engram(&data, &input, &limits)?
    } else if args.mode == "weight_delta" || args.mode == "weight-delta" {
        let (dtype, shape) = weight_delta_metadata_args(&args)?;
        let base_frame = base
            .as_deref()
            .ok_or_else(|| Error::malformed("WEIGHT_DELTA encoding requires --base FRAME"))?;
        weight_delta::encode_weight_delta(&data, dtype, &shape, base_frame, &limits)?
    } else if args.mode == "sparse_delta" || args.mode == "sparse-delta" {
        let (dtype, shape) = weight_delta_metadata_args(&args)?;
        let base_frame = base
            .as_deref()
            .ok_or_else(|| Error::malformed("SPARSE_DELTA encoding requires --base FRAME"))?;
        sparse_delta::encode_sparse_delta(&data, dtype, &shape, base_frame, &limits)?
    } else if args.mode == "semantic_delta" || args.mode == "semantic-delta" {
        let (dtype, shape) = weight_delta_metadata_args(&args)?;
        let policy = semantic_policy_arg(&args)?;
        let base_frame = base
            .as_deref()
            .ok_or_else(|| Error::malformed("SEMANTIC_DELTA encoding requires --base FRAME"))?;
        semantic_delta::encode_semantic_delta(&data, dtype, &shape, base_frame, policy, &limits)?
    } else if args.mode == "kv_cache" || args.mode == "kv-cache" {
        let input = kv_cache_input_args(&args)?;
        kv_cache::encode_kv_page(&input, &data, &limits)?
    } else if args.mode == "kv_prov" || args.mode == "kv-prov" {
        let input = kv_cache_input_args(&args)?;
        let provenance = kv_prov_provenance_args(&args)?;
        kv_prov::encode_kv_prov_page(&input, &data, &provenance, &limits)?
    } else if args.mode == "activation_stream" || args.mode == "activation-stream" {
        let (head, chunk_index) = activation_stream_input_args(&args, &limits)?;
        activation_stream::encode_activation_chunk(&head, chunk_index, &data, &limits)?
    } else if args.mode == "tokens" {
        let input: TokensInput = serde_json::from_slice(&data).map_err(|err| {
            Error::malformed(format!(
                "tokens input must be JSON {{\"vocab_schema\":string,\"token_ids\":[u64,...]}}: {err}"
            ))
        })?;
        modes::encode_token_sequence(&input.token_ids, &input.vocab_schema, &limits)?
    } else if args.mode == "auto" {
        encode_auto(
            &data,
            base.as_deref(),
            dict.as_deref(),
            args.level,
            args.chunk_size,
            &limits,
        )?
    } else {
        let mode = Mode::from_name(&args.mode)?;
        encode_mode(
            mode,
            &data,
            base.as_deref(),
            dict.as_deref(),
            args.level,
            args.chunk_size,
            &limits,
        )?
    };
    write_output(&args.output, &frame)
}

fn encode_mode(
    mode: Mode,
    data: &[u8],
    base: Option<&[u8]>,
    dict: Option<&[u8]>,
    level: i32,
    chunk_size: Option<usize>,
    limits: &Limits,
) -> Result<Vec<u8>> {
    match mode {
        Mode::Raw => modes::encode_raw(data, chunk_size, limits),
        Mode::Zstd => modes::encode_zstd(data, level, chunk_size, limits),
        Mode::Reference => {
            let base =
                base.ok_or_else(|| Error::malformed("REFERENCE encoding requires --base"))?;
            modes::encode_reference(data, base, limits)
        }
        Mode::Delta => {
            let base = base.ok_or_else(|| Error::malformed("DELTA encoding requires --base"))?;
            modes::encode_delta(data, base, limits)
        }
        Mode::Dictionary => {
            let dict =
                dict.ok_or_else(|| Error::malformed("DICTIONARY encoding requires --dict"))?;
            modes::encode_dictionary(data, dict, level, limits)
        }
        Mode::Structured => {
            let value = wcb::decode(data, limits)?;
            modes::encode_structured(&value, limits)
        }
        Mode::Tensor => Err(Error::malformed(
            "TENSOR encoding requires the dedicated CLI path with --dtype/--shape/--metadata",
        )),
        Mode::TokenSequence => Err(Error::malformed(
            "TOKEN_SEQUENCE encoding requires a JSON token input",
        )),
        Mode::QuantizedTensor => Err(Error::malformed(
            "QUANTIZED_TENSOR encoding requires the dedicated CLI path with --dtype/--shape/--scale/--zero-point",
        )),
        Mode::FloatTensor => Err(Error::malformed(
            "FLOAT_TENSOR encoding requires the dedicated CLI path with --dtype/--shape/--metadata",
        )),
        Mode::TensorSharded => Err(Error::malformed(
            "TENSOR_SHARDED encoding requires the dedicated CLI path with --dtype/--shape/--axis/--shards/--block-size/--scales/--zero-points",
        )),
        Mode::WeightDelta => Err(Error::malformed(
            "WEIGHT_DELTA encoding requires the dedicated CLI path with --base/--dtype/--shape",
        )),
        Mode::SparseDelta => Err(Error::malformed(
            "SPARSE_DELTA encoding requires the dedicated CLI path with --base/--dtype/--shape",
        )),
        Mode::SemanticDelta => Err(Error::malformed(
            "SEMANTIC_DELTA encoding requires the dedicated CLI path with --base/--dtype/--shape/--semantic-policy",
        )),
        Mode::KvCache => Err(Error::malformed(
            "KV_CACHE encoding requires the dedicated CLI path with --metadata",
        )),
        Mode::KvCacheProv => Err(Error::malformed(
            "KV_CACHE_PROV encoding requires the dedicated CLI path with --metadata and --provenance",
        )),
        Mode::ActivationStream => Err(Error::malformed(
            "ACTIVATION_STREAM encoding requires the dedicated CLI path with --metadata",
        )),
        Mode::TensorTiled => Err(Error::malformed(
            "TENSOR_TILED encoding requires the dedicated CLI path with --dtype/--shape/--tile-shape",
        )),
        Mode::FloatTiled => Err(Error::malformed(
            "FLOAT_TILED encoding requires the dedicated CLI path with --dtype/--shape/--tile-shape",
        )),
        Mode::FloatTiledStreams => Err(Error::malformed(
            "FLOAT_TILED_STREAMS encoding requires the dedicated CLI path with --dtype/--shape/--tile-shape/--n-streams",
        )),
        Mode::EngramShard => Err(Error::malformed(
            "ENGRAM_SHARD encoding requires the dedicated CLI path with --dtype/--shape/--group-edges/--hash-function/--vocab-size",
        )),
    }
}

/// Parse the TENSOR descriptor from `--metadata` or `--dtype`/`--shape`.
fn tensor_metadata_args(args: &EncodeArgs) -> Result<(modes::TensorDtype, Vec<u64>)> {
    if let Some(path) = &args.metadata {
        if args.dtype.is_some() || args.shape.is_some() {
            return Err(Error::malformed(
                "--metadata cannot be combined with --dtype or --shape",
            ));
        }
        let bytes = read_limited(path, Limits::default().max_string_bytes)?;
        let input: TensorMetaInput = serde_json::from_slice(&bytes).map_err(|err| {
            Error::malformed(format!(
                "tensor metadata must be JSON {{\"dtype\":string,\"shape\":[u64,...]}}: {err}"
            ))
        })?;
        if let Some(layout) = input.layout {
            if layout != "row_major" {
                return Err(Error::malformed(format!(
                    "tensor layout `{layout}` is specified but not implemented in v1 (only row_major)"
                )));
            }
        }
        if let Some(endianness) = input.endianness {
            if endianness != "little" {
                return Err(Error::malformed(format!(
                    "tensor endianness `{endianness}` is specified but not implemented in v1 (only little)"
                )));
            }
        }
        let dtype = modes::TensorDtype::from_name(&input.dtype)?;
        return Ok((dtype, input.shape));
    }
    let dtype = args
        .dtype
        .as_deref()
        .ok_or_else(|| Error::malformed("TENSOR encoding requires --dtype or --metadata"))?;
    let dtype = modes::TensorDtype::from_name(dtype)?;
    let shape = match args.shape.as_deref() {
        None | Some("") => Vec::new(),
        Some(text) => text
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<u64>()
                    .map_err(|err| Error::malformed(format!("bad --shape extent `{part}`: {err}")))
            })
            .collect::<Result<Vec<u64>>>()?,
    };
    Ok((dtype, shape))
}

/// Parse the QUANTIZED_TENSOR descriptor from `--metadata` or the scalar flags.
fn quantized_metadata_args(args: &EncodeArgs) -> Result<(modes::TensorDtype, Vec<u64>, f64, i64)> {
    if let Some(path) = &args.metadata {
        if args.dtype.is_some() || args.shape.is_some() || args.scale.is_some() {
            return Err(Error::malformed(
                "--metadata cannot be combined with --dtype/--shape/--scale",
            ));
        }
        let bytes = read_limited(path, Limits::default().max_string_bytes)?;
        let input: TensorMetaInput = serde_json::from_slice(&bytes).map_err(|err| {
            Error::malformed(format!(
                "quantized tensor metadata must be JSON {{\"dtype\":string,\"shape\":[u64,...],\"scale\":number,\"zero_point\":integer}}: {err}"
            ))
        })?;
        if let Some(layout) = input.layout {
            if layout != "row_major" {
                return Err(Error::malformed(format!(
                    "quantized tensor layout `{layout}` is specified but not implemented in v1 (only row_major)"
                )));
            }
        }
        let scale = input.scale.ok_or_else(|| {
            Error::malformed("quantized tensor metadata requires a `scale` number")
        })?;
        let dtype = modes::TensorDtype::from_name(&input.dtype)?;
        return Ok((dtype, input.shape, scale, input.zero_point.unwrap_or(0)));
    }
    let dtype = args.dtype.as_deref().ok_or_else(|| {
        Error::malformed("QUANTIZED_TENSOR encoding requires --dtype or --metadata")
    })?;
    let dtype = modes::TensorDtype::from_name(dtype)?;
    let scale = args
        .scale
        .ok_or_else(|| Error::malformed("QUANTIZED_TENSOR encoding requires --scale"))?;
    let shape = match args.shape.as_deref() {
        None | Some("") => Vec::new(),
        Some(text) => text
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<u64>()
                    .map_err(|err| Error::malformed(format!("bad --shape extent `{part}`: {err}")))
            })
            .collect::<Result<Vec<u64>>>()?,
    };
    Ok((dtype, shape, scale, args.zero_point))
}

/// Parse `--tile-shape` for TENSOR_TILED.
/// Parse `--group-edges` for ENGRAM_SHARD.
fn engram_edges_arg(args: &EncodeArgs) -> Result<Vec<u64>> {
    let text = args.group_edges.as_deref().ok_or_else(|| {
        Error::malformed("ENGRAM_SHARD encoding requires --group-edges 0,A,...,ROWS")
    })?;
    let mut out = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(Error::malformed("bad --group-edges: empty extent"));
        }
        out.push(part.parse::<u64>().map_err(|err| {
            Error::malformed(format!("bad --group-edges extent `{part}`: {err}"))
        })?);
    }
    Ok(out)
}

fn tile_shape_arg(args: &EncodeArgs) -> Result<Vec<u64>> {
    let text = args
        .tile_shape
        .as_deref()
        .ok_or_else(|| Error::malformed("TENSOR_TILED encoding requires --tile-shape A,B,..."))?;
    let mut shape = Vec::new();
    for part in text.split(',') {
        shape.push(
            part.trim().parse::<u64>().map_err(|err| {
                Error::malformed(format!("bad --tile-shape extent `{part}`: {err}"))
            })?,
        );
    }
    Ok(shape)
}

/// Parse the FLOAT_TENSOR descriptor from `--metadata` or `--dtype`/`--shape`.
fn float_tensor_metadata_args(args: &EncodeArgs) -> Result<(modes::FloatDtype, Vec<u64>)> {
    if let Some(path) = &args.metadata {
        if args.dtype.is_some() || args.shape.is_some() {
            return Err(Error::malformed(
                "--metadata cannot be combined with --dtype or --shape",
            ));
        }
        let bytes = read_limited(path, Limits::default().max_string_bytes)?;
        let input: TensorMetaInput = serde_json::from_slice(&bytes).map_err(|err| {
            Error::malformed(format!(
                "float tensor metadata must be JSON {{\"dtype\":string,\"shape\":[u64,...]}}: {err}"
            ))
        })?;
        let dtype = modes::FloatDtype::from_name(&input.dtype)?;
        return Ok((dtype, input.shape));
    }
    let dtype = args
        .dtype
        .as_deref()
        .ok_or_else(|| Error::malformed("FLOAT_TENSOR encoding requires --dtype or --metadata"))?;
    let dtype = modes::FloatDtype::from_name(dtype)?;
    let shape = match args.shape.as_deref() {
        None | Some("") => Vec::new(),
        Some(text) => text
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<u64>()
                    .map_err(|err| Error::malformed(format!("bad --shape extent `{part}`: {err}")))
            })
            .collect::<Result<Vec<u64>>>()?,
    };
    Ok((dtype, shape))
}

/// Parse the WEIGHT_DELTA descriptor from `--dtype`/`--shape`.
fn weight_delta_metadata_args(args: &EncodeArgs) -> Result<(weight_delta::WeightDtype, Vec<u64>)> {
    if args.metadata.is_some() {
        return Err(Error::malformed(
            "WEIGHT_DELTA encoding does not accept --metadata; use --dtype/--shape",
        ));
    }
    let dtype = args.dtype.as_deref().ok_or_else(|| {
        Error::malformed("WEIGHT_DELTA encoding requires --dtype (bf16|fp16|fp32|q8_0|q4_0)")
    })?;
    let dtype = weight_delta::WeightDtype::from_name(dtype)?;
    let shape = match args.shape.as_deref() {
        None | Some("") => {
            return Err(Error::malformed(
                "WEIGHT_DELTA encoding requires --shape (rank >= 1)",
            ))
        }
        Some(text) => text
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<u64>()
                    .map_err(|err| Error::malformed(format!("bad --shape extent `{part}`: {err}")))
            })
            .collect::<Result<Vec<u64>>>()?,
    };
    Ok((dtype, shape))
}

/// Parse `--semantic-policy` for SEMANTIC_DELTA.
fn semantic_policy_arg(args: &EncodeArgs) -> Result<SemanticPolicy> {
    let text = args.semantic_policy.as_deref().ok_or_else(|| {
        Error::malformed(
            "SEMANTIC_DELTA encoding requires --semantic-policy (none|mantissa_truncate:N|block_quantize:B:L)",
        )
    })?;
    let text = text.trim();
    if text == "none" {
        return Ok(SemanticPolicy::None);
    }
    if let Some(rest) = text
        .strip_prefix("mantissa_truncate:")
        .or_else(|| text.strip_prefix("mantissa-truncate:"))
    {
        let bits = rest.parse::<u32>().map_err(|err| {
            Error::malformed(format!("bad --semantic-policy bits `{rest}`: {err}"))
        })?;
        return Ok(SemanticPolicy::MantissaTruncate { bits });
    }
    if let Some(rest) = text
        .strip_prefix("block_quantize:")
        .or_else(|| text.strip_prefix("block-quantize:"))
    {
        let mut parts = rest.split(':');
        let block_size = parts
            .next()
            .ok_or_else(|| Error::malformed("block_quantize policy requires block_size"))?
            .parse::<u64>()
            .map_err(|err| Error::malformed(format!("bad block_quantize block_size: {err}")))?;
        let levels = parts
            .next()
            .ok_or_else(|| Error::malformed("block_quantize policy requires levels"))?
            .parse::<u64>()
            .map_err(|err| Error::malformed(format!("bad block_quantize levels: {err}")))?;
        if parts.next().is_some() {
            return Err(Error::malformed(
                "block_quantize policy has trailing fields",
            ));
        }
        return Ok(SemanticPolicy::BlockQuantize { block_size, levels });
    }
    Err(Error::malformed(format!(
        "unknown --semantic-policy `{text}` (none|mantissa_truncate:N|block_quantize:B:L)"
    )))
}

/// JSON form of a semantic policy (for `inspect`).
fn semantic_policy_json(policy: &SemanticPolicy) -> serde_json::Value {
    match policy {
        SemanticPolicy::None => json!({ "scheme": "none" }),
        SemanticPolicy::MantissaTruncate { bits } => {
            json!({ "scheme": "mantissa_truncate", "bits": bits })
        }
        SemanticPolicy::BlockQuantize { block_size, levels } => json!({
            "scheme": "block_quantize",
            "block_size": block_size,
            "levels": levels
        }),
    }
}

fn read_metadata_json<T: serde::de::DeserializeOwned>(args: &EncodeArgs, what: &str) -> Result<T> {
    let path = args
        .metadata
        .as_deref()
        .ok_or_else(|| Error::malformed(format!("{what} encoding requires --metadata FILE")))?;
    let bytes = read_limited(path, Limits::default().max_string_bytes)?;
    serde_json::from_slice(&bytes)
        .map_err(|err| Error::malformed(format!("{what} metadata is not valid JSON: {err}")))
}

/// Parse a KV_CACHE page descriptor from `--metadata`.
fn kv_cache_input_args(args: &EncodeArgs) -> Result<kv_cache::KvCacheInput> {
    let input: KvCacheMetaInput = read_metadata_json(args, "KV_CACHE")?;
    Ok(kv_cache::KvCacheInput {
        model: input.model,
        layer: input.layer,
        session: input.session,
        page_index: input.page_index,
        page_count: input.page_count,
        seq_start: input.seq_start,
        seq_end: input.seq_end,
        dtype: kv_cache::KvDtype::from_name(&input.dtype)?,
        head_count: input.head_count,
        head_dim: input.head_dim,
    })
}

/// Parse a KV_CACHE_PROV provenance document from `--provenance`.
fn kv_prov_provenance_args(args: &EncodeArgs) -> Result<kv_prov::KvProvenanceInput> {
    let path = args
        .provenance
        .as_deref()
        .ok_or_else(|| Error::malformed("KV_CACHE_PROV encoding requires --provenance FILE"))?;
    let bytes = read_limited(path, Limits::default().max_string_bytes)?;
    let document: serde_json::Value = serde_json::from_slice(&bytes).map_err(|err| {
        Error::malformed(format!("KV_CACHE_PROV provenance is not valid JSON: {err}"))
    })?;
    kv_prov_document(&document)
}

fn kv_prov_hex_root(value: &serde_json::Value, field: &str) -> Result<weftcodec::Hash32> {
    let text = value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::malformed(format!("provenance `{field}` must be a hex string")))?;
    weftcodec::Hash32::from_hex(text)
}

fn kv_prov_u64(value: &serde_json::Value, field: &str) -> Result<u64> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Error::malformed(format!("provenance `{field}` must be a u64")))
}

/// Map strict JSON onto a canonical WCB value. Floats are not representable in
/// WCB and are rejected rather than rounded.
fn kv_prov_wcb_value(value: &serde_json::Value) -> Result<weftcodec::wcb::Value> {
    use weftcodec::wcb::Value;
    Ok(match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(flag) => Value::Bool(*flag),
        serde_json::Value::String(text) => Value::Str(text.clone()),
        serde_json::Value::Number(number) => {
            if let Some(unsigned) = number.as_u64() {
                Value::U64(unsigned)
            } else if let Some(signed) = number.as_i64() {
                Value::I64(signed)
            } else {
                return Err(Error::malformed(
                    "provenance extension numbers must be exact integers (WCB has no float)",
                ));
            }
        }
        serde_json::Value::Array(items) => Value::Array(
            items
                .iter()
                .map(kv_prov_wcb_value)
                .collect::<Result<Vec<_>>>()?,
        ),
        serde_json::Value::Object(map) => {
            let mut out = std::collections::BTreeMap::new();
            for (key, item) in map {
                out.insert(key.clone(), kv_prov_wcb_value(item)?);
            }
            Value::Map(out)
        }
    })
}

fn kv_prov_document(document: &serde_json::Value) -> Result<kv_prov::KvProvenanceInput> {
    let object = document
        .as_object()
        .ok_or_else(|| Error::malformed("KV_CACHE_PROV provenance must be a JSON object"))?;
    let allowed = [
        "model_root",
        "adapter_roots",
        "adapter_set_root",
        "tokenizer",
        "tokenizer_root",
        "attention",
        "positional",
        "layout",
        "salt",
        "extensions",
    ];
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::malformed(format!(
                "unknown KV_CACHE_PROV provenance key `{key}`"
            )));
        }
    }
    let model_root = kv_prov_hex_root(document, "model_root")?;
    let mut adapter_roots = Vec::new();
    if let Some(items) = document.get("adapter_roots") {
        let items = items
            .as_array()
            .ok_or_else(|| Error::malformed("provenance `adapter_roots` must be an array"))?;
        for item in items {
            let text = item
                .as_str()
                .ok_or_else(|| Error::malformed("provenance adapter roots must be hex strings"))?;
            adapter_roots.push(weftcodec::Hash32::from_hex(text)?);
        }
    }
    let tokenizer = document
        .get("tokenizer")
        .ok_or_else(|| Error::malformed("provenance `tokenizer` is required"))?;
    let tokenizer_scheme = tokenizer
        .get("scheme")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::malformed("provenance `tokenizer.scheme` must be a string"))?
        .to_owned();
    let tokenizer_vocab_size = kv_prov_u64(tokenizer, "vocab_size")?;
    let attention = document
        .get("attention")
        .ok_or_else(|| Error::malformed("provenance `attention` is required"))?;
    let attention = kv_prov::KvAttention {
        head_count: u32::try_from(kv_prov_u64(attention, "head_count")?)
            .map_err(|_| Error::malformed("provenance `attention.head_count` exceeds u32"))?,
        head_dim: u32::try_from(kv_prov_u64(attention, "head_dim")?)
            .map_err(|_| Error::malformed("provenance `attention.head_dim` exceeds u32"))?,
        sliding_window: kv_prov_u64(attention, "sliding_window")?,
        rope_theta_bits: kv_prov_u64(attention, "rope_theta_bits")?,
        rope_scale_bits: kv_prov_u64(attention, "rope_scale_bits")?,
    };
    let positional = kv_prov::KvPositional::from_name(
        document
            .get("positional")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error::malformed("provenance `positional` must be text"))?,
    )?;
    let layout = document
        .get("layout")
        .ok_or_else(|| Error::malformed("provenance `layout` is required"))?;
    let layout = kv_prov::KvLayout {
        dtype: kv_cache::KvDtype::from_name(
            layout
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| Error::malformed("provenance `layout.dtype` must be text"))?,
        )?,
        page_rows: kv_prov_u64(layout, "page_rows")?,
        layer_count: kv_prov_u64(layout, "layer_count")?,
    };
    let salt = kv_prov_hex_root(document, "salt")?;
    let mut extensions = Vec::new();
    if let Some(items) = document.get("extensions") {
        let items = items
            .as_array()
            .ok_or_else(|| Error::malformed("provenance `extensions` must be an array"))?;
        for item in items {
            let name = item
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| Error::malformed("provenance extension `name` must be text"))?;
            let value = item
                .get("value")
                .ok_or_else(|| Error::malformed("provenance extension `value` is required"))?;
            extensions.push(kv_prov::KvExtension::new(name, kv_prov_wcb_value(value)?));
        }
    }
    let input = kv_prov::KvProvenanceInput {
        model_root,
        adapter_roots,
        tokenizer_scheme,
        tokenizer_vocab_size,
        attention,
        positional,
        layout,
        salt,
        extensions,
    };
    // If the document declares derived roots, they must agree with what is
    // recomputed. The codec never trusts a declared root.
    let canonical = input.canonicalize()?;
    if let Some(declared) = document.get("adapter_set_root") {
        let declared = declared
            .as_str()
            .ok_or_else(|| Error::malformed("provenance `adapter_set_root` must be hex"))?;
        if weftcodec::Hash32::from_hex(declared)? != canonical.adapter_set_root {
            return Err(Error::malformed(
                "provenance declared adapter_set_root disagrees with adapter_roots",
            ));
        }
    }
    if let Some(declared) = document.get("tokenizer_root") {
        let declared = declared
            .as_str()
            .ok_or_else(|| Error::malformed("provenance `tokenizer_root` must be hex"))?;
        if weftcodec::Hash32::from_hex(declared)? != canonical.tokenizer_root {
            return Err(Error::malformed(
                "provenance declared tokenizer_root disagrees with the tokenizer descriptor",
            ));
        }
    }
    Ok(input)
}

/// Parse an ACTIVATION_STREAM chunk descriptor from `--metadata`.
fn activation_stream_input_args(
    args: &EncodeArgs,
    limits: &Limits,
) -> Result<(activation_stream::ActivationStreamHead, u64)> {
    let input: ActivationStreamMetaInput = read_metadata_json(args, "ACTIVATION_STREAM")?;
    let head = activation_stream::ActivationStreamHead {
        stream: activation_stream::StreamKind::from_name(&input.stream)?,
        step: input.step,
        rank: input.rank,
        world: input.world,
        chunk_count: input.chunk_count,
        dtype: activation_stream::StreamDtype::from_name(&input.dtype)?,
        shape: input.shape,
    };
    // Validate the head early so CLI errors are descriptor errors.
    let probe = activation_stream::ActivationStreamMeta {
        stream: head.stream,
        step: head.step,
        rank: head.rank,
        world: head.world,
        chunk_index: input.chunk_index,
        chunk_count: head.chunk_count,
        dtype: head.dtype,
        shape: head.shape.clone(),
        element_count: {
            let mut product: u64 = 1;
            for dim in &head.shape {
                product = product
                    .checked_mul(*dim)
                    .ok_or_else(|| Error::limit_exceeded("shape product overflows u64"))?;
            }
            product
        },
        digest_root: weftcodec::Hash32::ZERO,
        plane_codec: activation_stream::ACTIVATION_STREAM_PLANE_CODEC.to_owned(),
        payload_len: 0,
    };
    activation_stream::validate_meta(&probe, limits)?;
    Ok((head, input.chunk_index))
}

fn parse_u32_list(text: &str, what: &str) -> Result<Vec<u32>> {
    text.split(',')
        .map(|part| {
            let part = part.trim();
            let value =
                if let Some(hex) = part.strip_prefix("0x").or_else(|| part.strip_prefix("0X")) {
                    u32::from_str_radix(hex, 16)
                } else {
                    part.parse::<u32>()
                };
            value.map_err(|err| Error::malformed(format!("bad {what} value `{part}`: {err}")))
        })
        .collect()
}

fn parse_i64_list(text: &str, what: &str) -> Result<Vec<i64>> {
    text.split(',')
        .map(|part| {
            let part = part.trim();
            part.parse::<i64>()
                .map_err(|err| Error::malformed(format!("bad {what} value `{part}`: {err}")))
        })
        .collect()
}

/// Parse a TENSOR_SHARDED input from `--metadata` or the dedicated flags.
fn sharded_encode_args(
    args: &EncodeArgs,
    data: &[u8],
    limits: &Limits,
) -> Result<tensor_sharded::ShardedTensorInput> {
    let (dtype, shape, axis, shards, block_size, scales, zero_points) = if let Some(path) =
        &args.metadata
    {
        if args.dtype.is_some()
            || args.shape.is_some()
            || args.axis.is_some()
            || args.shards.is_some()
            || args.block_size.is_some()
            || args.scales.is_some()
            || args.zero_points.is_some()
        {
            return Err(Error::malformed(
                "--metadata cannot be combined with TENSOR_SHARDED flags",
            ));
        }
        let bytes = read_limited(path, limits.max_string_bytes)?;
        let input: ShardedMetaInput = serde_json::from_slice(&bytes).map_err(|err| {
                Error::malformed(format!(
                    "TENSOR_SHARDED metadata must be JSON {{\"dtype\":string,\"shape\":[u64,...],\"axis\":u32,\"shards\":u32,\"block_size\":u64,\"scales\":[u32,...],\"zero_points\":[i64,...]}}: {err}"
                ))
            })?;
        let dtype = tensor_sharded::ShardedDtype::from_name(&input.dtype)?;
        let mut scales = Vec::with_capacity(input.scales.len());
        for bits in input.scales {
            scales.push(
                u32::try_from(bits)
                    .map_err(|_| Error::malformed("TENSOR_SHARDED scale bits exceed u32"))?,
            );
        }
        (
            dtype,
            input.shape,
            input.axis,
            input.shards,
            input.block_size,
            scales,
            input.zero_points,
        )
    } else {
        let dtype = args.dtype.as_deref().ok_or_else(|| {
            Error::malformed("TENSOR_SHARDED encoding requires --dtype or --metadata")
        })?;
        let dtype = tensor_sharded::ShardedDtype::from_name(dtype)?;
        let shape = try_parse_extents(
            args.shape
                .as_deref()
                .ok_or_else(|| Error::malformed("TENSOR_SHARDED encoding requires --shape"))?,
        )?;
        let axis = args
            .axis
            .ok_or_else(|| Error::malformed("TENSOR_SHARDED encoding requires --axis"))?;
        let shards = args
            .shards
            .ok_or_else(|| Error::malformed("TENSOR_SHARDED encoding requires --shards"))?;
        let block_size = args
            .block_size
            .ok_or_else(|| Error::malformed("TENSOR_SHARDED encoding requires --block-size"))?;
        let scales = parse_u32_list(
            args.scales
                .as_deref()
                .ok_or_else(|| Error::malformed("TENSOR_SHARDED encoding requires --scales"))?,
            "--scales",
        )?;
        let zero_points = parse_i64_list(
            args.zero_points.as_deref().ok_or_else(|| {
                Error::malformed("TENSOR_SHARDED encoding requires --zero-points")
            })?,
            "--zero-points",
        )?;
        (dtype, shape, axis, shards, block_size, scales, zero_points)
    };
    if data.len() as u64 > limits.max_decoded_bytes {
        return Err(Error::limit_exceeded(
            "TENSOR_SHARDED word input exceeds limit",
        ));
    }
    let words = data.iter().map(|byte| *byte as i8).collect();
    Ok(tensor_sharded::ShardedTensorInput {
        dtype,
        shape,
        shard_axis: axis,
        shard_count: shards,
        block_size,
        scales,
        zero_points,
        words,
    })
}

fn try_parse_extents(text: &str) -> Result<Vec<u64>> {
    text.split(',')
        .map(|part| {
            part.trim()
                .parse::<u64>()
                .map_err(|err| Error::malformed(format!("bad shape extent `{part}`: {err}")))
        })
        .collect()
}

fn encode_auto(
    data: &[u8],
    base: Option<&[u8]>,
    dict: Option<&[u8]>,
    level: i32,
    chunk_size: Option<usize>,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let mut candidates: Vec<(u8, Vec<u8>)> = Vec::new();
    candidates.push((
        Mode::Raw.as_u8(),
        modes::encode_raw(data, chunk_size, limits)?,
    ));
    candidates.push((
        Mode::Zstd.as_u8(),
        modes::encode_zstd(data, level, chunk_size, limits)?,
    ));
    if let Some(base) = base {
        if base == data {
            candidates.push((
                Mode::Reference.as_u8(),
                modes::encode_reference(data, base, limits)?,
            ));
        }
        candidates.push((
            Mode::Delta.as_u8(),
            modes::encode_delta(data, base, limits)?,
        ));
    }
    if let Some(dict) = dict {
        candidates.push((
            Mode::Dictionary.as_u8(),
            modes::encode_dictionary(data, dict, level, limits)?,
        ));
    }
    let best = candidates
        .into_iter()
        .min_by(|a, b| a.1.len().cmp(&b.1.len()).then(a.0.cmp(&b.0)));
    match best {
        Some((_, frame)) => Ok(frame),
        None => Err(Error::new(
            ErrorClass::Malformed,
            "auto encoding produced no candidates",
        )),
    }
}

fn decode(args: DecodeArgs) -> Result<()> {
    let limits = args.limits.to_limits();
    let frame_bytes = read_limited(&args.input, limits.max_frame_bytes)?;
    let root = Frame::parse(&frame_bytes, &limits)?;

    // `--base` supplies the base frame explicitly. `--resolve` is an explicit
    // opt-in to the DAG resolver, which also verifies the base frame's own
    // dtype/shape descriptor.
    if root.header.mode == Mode::WeightDelta
        || root.header.mode == Mode::SemanticDelta
        || root.header.mode == Mode::SparseDelta
    {
        if args.resolve && args.frame_source.is_none() {
            return Err(Error::malformed(
                "delta modes require --frame-source with --resolve",
            ));
        }
        if let Some(base_path) = args.base.as_deref() {
            let base_bytes = read_limited(base_path, limits.max_frame_bytes)?;
            let resolver = resolver_for(args.deps.as_deref());
            let plaintext = if root.header.mode == Mode::SemanticDelta {
                semantic_delta::decode_with_base_frame(
                    &root,
                    &base_bytes,
                    resolver.as_ref(),
                    &limits,
                )?
            } else if root.header.mode == Mode::SparseDelta {
                sparse_delta::decode_with_base_frame(
                    &root,
                    &base_bytes,
                    resolver.as_ref(),
                    &limits,
                )?
            } else {
                weight_delta::decode_with_base_frame(
                    &root,
                    &base_bytes,
                    resolver.as_ref(),
                    &limits,
                )?
            };
            return write_output(&args.output, &plaintext);
        }
    } else if args.resolve {
        return Err(Error::malformed(
            "--resolve applies only to WEIGHT_DELTA, SEMANTIC_DELTA, and SPARSE_DELTA frames",
        ));
    }

    let plaintext = match args.frame_source.as_deref() {
        Some(dir) => {
            let source = DirectoryFrameSource::new(dir, limits.max_frame_bytes);
            decode_frame_dag(&source, &frame_bytes, &limits)?
        }
        None => {
            let resolver = resolver_for(args.deps.as_deref());
            modes::decode_bytes(&frame_bytes, resolver.as_ref(), &limits)?
        }
    };
    write_output(&args.output, &plaintext)
}

fn verify(args: VerifyArgs) -> Result<()> {
    let limits = Limits::default();
    let frame_bytes = read_limited(&args.file, limits.max_frame_bytes)?;
    let resolver = resolver_for(args.deps.as_deref());
    let plaintext = modes::decode_bytes(&frame_bytes, resolver.as_ref(), &limits)?;
    let frame = Frame::parse(&frame_bytes, &limits)?;
    println!(
        "OK mode={} decoded_len={} content_hash={} chunks={} dependencies={}",
        frame.header.mode.as_str(),
        plaintext.len(),
        frame.header.content_hash,
        frame.header.chunk_count,
        frame.header.dep_count
    );
    Ok(())
}

fn inspect(args: InspectArgs) -> Result<()> {
    let limits = Limits::default();
    let frame_bytes = read_limited(&args.file, limits.max_frame_bytes)?;
    let frame = Frame::parse(&frame_bytes, &limits)?;
    let header = &frame.header;
    if args.json {
        let dependencies: Vec<_> = frame
            .dependencies
            .iter()
            .map(|dep| {
                json!({
                    "hash": dep.hash.to_hex(),
                    "decoded_len": dep.decoded_len,
                    "dep_domain": dep.dep_domain,
                    "dep_flags": dep.dep_flags,
                    "codec_hint": dep.codec_hint,
                })
            })
            .collect();
        let chunks: Vec<_> = frame
            .chunks
            .iter()
            .map(|chunk| {
                json!({
                    "encoded_off": chunk.encoded_off,
                    "encoded_len": chunk.encoded_len,
                    "decoded_len": chunk.decoded_len,
                    "hash": chunk.hash.to_hex(),
                })
            })
            .collect();
        let mut output = json!({
            "magic": "WFT1",
            "format_major": header.format_major,
            "format_minor": header.format_minor,
            "domain": header.domain,
            "domain_name": if header.domain == 0 { "payload" } else { "structured" },
            "mode": header.mode.as_str(),
            "mode_value": header.mode.as_u8(),
            "flags": header.flags,
            "flag_names": header.flag_names(),
            "features": header.features,
            "feature_names": header.feature_names(),
            "level": header.level,
            "decoded_len": header.decoded_len,
            "body_len": header.body_len,
            "table_len": header.table_len,
            "frame_len": frame_bytes.len(),
            "dep_count": header.dep_count,
            "chunk_count": header.chunk_count,
            "content_hash": header.content_hash.to_hex(),
            "dependencies": dependencies,
            "chunks": chunks,
        });
        if header.mode == Mode::FloatTensor {
            output["exponent_coder"] = json!(modes::FLOAT_TENSOR_EXPONENT_CODER);
        }
        if header.mode == Mode::WeightDelta {
            let meta = weight_delta::weight_delta_metadata(&frame, &limits)?;
            output["dtype"] = json!(meta.dtype.as_str());
            output["shape"] = json!(meta.shape);
            output["element_count"] = json!(meta.element_count);
            output["word_bits"] = json!(meta.word_bits);
            output["base"] = json!(meta.base.to_hex());
            output["base_dtype"] = json!(meta.base_dtype.as_str());
            output["base_shape"] = json!(meta.base_shape);
            output["op"] = json!(meta.op);
            output["planes"] = json!(meta.planes);
            output["plane_codec"] = json!(meta.plane_codec);
            output["delta_root"] = json!(meta.delta_root.to_hex());
            output["payload_len"] = json!(meta.payload_len);
        }
        if header.mode == Mode::SparseDelta {
            let meta = sparse_delta::sparse_delta_metadata(&frame, &limits)?;
            output["dtype"] = json!(meta.dtype.as_str());
            output["shape"] = json!(meta.shape);
            output["element_count"] = json!(meta.element_count);
            output["storage_bytes"] = json!(meta.storage_bytes);
            output["op"] = json!(meta.op);
            output["sparse_codec"] = json!(meta.sparse_codec);
            output["base_content_root"] = json!(meta.base.to_hex());
            output["delta_root"] = json!(meta.delta_root.to_hex());
        }
        if header.mode == Mode::SemanticDelta {
            let meta = semantic_delta::semantic_delta_metadata(&frame, &limits)?;
            output["dtype"] = json!(meta.dtype.as_str());
            output["shape"] = json!(meta.shape);
            output["element_count"] = json!(meta.element_count);
            output["word_bits"] = json!(meta.word_bits);
            output["base"] = json!(meta.base.to_hex());
            output["base_dtype"] = json!(meta.base_dtype.as_str());
            output["base_shape"] = json!(meta.base_shape);
            output["base_semantic_root"] = json!(meta.base_semantic_root.to_hex());
            output["policy"] = semantic_policy_json(&meta.policy);
            output["planes"] = json!(meta.dtype.planes());
            output["plane_codec"] = json!(meta.plane_codec);
            output["delta_root"] = json!(meta.delta_root.to_hex());
            output["payload_len"] = json!(meta.payload_len);
        }
        if header.mode == Mode::KvCache {
            let meta = kv_cache::kv_cache_metadata(&frame, &limits)?;
            output["model"] = json!(meta.model);
            output["layer"] = json!(meta.layer);
            output["session"] = json!(meta.session);
            output["page_index"] = json!(meta.page_index);
            output["page_count"] = json!(meta.page_count);
            output["seq_start"] = json!(meta.seq_start);
            output["seq_end"] = json!(meta.seq_end);
            output["dtype"] = json!(meta.dtype.as_str());
            output["head_count"] = json!(meta.head_count);
            output["head_dim"] = json!(meta.head_dim);
            output["shape"] = json!(meta.shape);
            output["page_bytes"] = json!(meta.page_bytes);
            output["plane_codec"] = json!(meta.plane_codec);
            output["payload_len"] = json!(meta.payload_len);
        }
        if header.mode == Mode::KvCacheProv {
            let meta = kv_prov::kv_prov_metadata(&frame, &limits)?;
            output["provenance_root"] = json!(meta.root(&limits)?.to_hex());
            output["model"] = json!(meta.page.model);
            output["layer"] = json!(meta.page.layer);
            output["session"] = json!(meta.page.session);
            output["page_index"] = json!(meta.page.page_index);
            output["page_count"] = json!(meta.page.page_count);
            output["seq_start"] = json!(meta.page.seq_start);
            output["seq_end"] = json!(meta.page.seq_end);
            output["dtype"] = json!(meta.page.dtype.as_str());
            output["head_count"] = json!(meta.page.head_count);
            output["head_dim"] = json!(meta.page.head_dim);
            output["shape"] = json!(meta.page.shape);
            output["page_bytes"] = json!(meta.page.page_bytes);
            output["plane_codec"] = json!(meta.page.plane_codec);
            output["payload_len"] = json!(meta.page.payload_len);
            output["positional"] = json!(meta.provenance.positional.as_str());
            output["adapter_set_root"] = json!(meta.provenance.adapter_set_root.to_hex());
            output["tokenizer_root"] = json!(meta.provenance.tokenizer_root.to_hex());
            output["salt"] = json!(meta.provenance.salt.to_hex());
        }
        if header.mode == Mode::ActivationStream {
            let meta = activation_stream::activation_chunk_metadata(&frame, &limits)?;
            output["stream"] = json!(meta.stream.as_str());
            output["step"] = json!(meta.step);
            output["rank"] = json!(meta.rank);
            output["world"] = json!(meta.world);
            output["chunk_index"] = json!(meta.chunk_index);
            output["chunk_count"] = json!(meta.chunk_count);
            output["dtype"] = json!(meta.dtype.as_str());
            output["shape"] = json!(meta.shape);
            output["element_count"] = json!(meta.element_count);
            output["digest_root"] = json!(meta.digest_root.to_hex());
            output["plane_codec"] = json!(meta.plane_codec);
            output["payload_len"] = json!(meta.payload_len);
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&output)
                .map_err(|e| Error::io(format!("cannot serialize inspection: {e}")))?
        );
    } else {
        println!(
            "format: WFT1 {}.{}",
            header.format_major, header.format_minor
        );
        println!(
            "domain: {} ({})",
            header.domain,
            if header.domain == 0 {
                "payload"
            } else {
                "structured"
            }
        );
        println!("mode: {} ({})", header.mode.as_str(), header.mode.as_u8());
        if header.mode == Mode::FloatTensor {
            println!("exponent_coder: {}", modes::FLOAT_TENSOR_EXPONENT_CODER);
        }
        if header.mode == Mode::WeightDelta {
            let meta = weight_delta::weight_delta_metadata(&frame, &limits)?;
            println!("dtype: {}", meta.dtype.as_str());
            println!("shape: {:?}", meta.shape);
            println!("element_count: {}", meta.element_count);
            println!("word_bits: {}", meta.word_bits);
            println!("base: {}", meta.base);
            println!("base_dtype: {}", meta.base_dtype.as_str());
            println!("base_shape: {:?}", meta.base_shape);
            println!("op: {}", meta.op);
            println!("planes: {}", meta.planes);
            println!("plane_codec: {}", meta.plane_codec);
            println!("delta_root: {}", meta.delta_root);
            println!("payload_len: {}", meta.payload_len);
        }
        if header.mode == Mode::SparseDelta {
            let meta = sparse_delta::sparse_delta_metadata(&frame, &limits)?;
            println!("dtype: {}", meta.dtype.as_str());
            println!("shape: {:?}", meta.shape);
            println!("element_count: {}", meta.element_count);
            println!("storage_bytes: {}", meta.storage_bytes);
            println!("op: {}", meta.op);
            println!("sparse_codec: {}", meta.sparse_codec);
            println!("base_content_root: {}", meta.base);
            println!("delta_root: {}", meta.delta_root);
        }
        if header.mode == Mode::SemanticDelta {
            let meta = semantic_delta::semantic_delta_metadata(&frame, &limits)?;
            println!("dtype: {}", meta.dtype.as_str());
            println!("shape: {:?}", meta.shape);
            println!("element_count: {}", meta.element_count);
            println!("word_bits: {}", meta.word_bits);
            println!("base: {}", meta.base);
            println!("base_dtype: {}", meta.base_dtype.as_str());
            println!("base_shape: {:?}", meta.base_shape);
            println!("base_semantic_root: {}", meta.base_semantic_root);
            println!("policy: {}", meta.policy.as_str());
            println!("planes: {}", meta.dtype.planes());
            println!("plane_codec: {}", meta.plane_codec);
            println!("delta_root: {}", meta.delta_root);
            println!("payload_len: {}", meta.payload_len);
        }
        if header.mode == Mode::KvCache {
            let meta = kv_cache::kv_cache_metadata(&frame, &limits)?;
            println!("model: {}", meta.model);
            println!("layer: {}", meta.layer);
            println!("session: {}", meta.session);
            println!("page_index: {}", meta.page_index);
            println!("page_count: {}", meta.page_count);
            println!("seq_start: {}", meta.seq_start);
            println!("seq_end: {}", meta.seq_end);
            println!("dtype: {}", meta.dtype.as_str());
            println!("head_count: {}", meta.head_count);
            println!("head_dim: {}", meta.head_dim);
            println!("shape: {:?}", meta.shape);
            println!("page_bytes: {}", meta.page_bytes);
            println!("plane_codec: {}", meta.plane_codec);
            println!("payload_len: {}", meta.payload_len);
        }
        if header.mode == Mode::ActivationStream {
            let meta = activation_stream::activation_chunk_metadata(&frame, &limits)?;
            println!("stream: {}", meta.stream.as_str());
            println!("step: {}", meta.step);
            println!("rank: {}", meta.rank);
            println!("world: {}", meta.world);
            println!("chunk_index: {}", meta.chunk_index);
            println!("chunk_count: {}", meta.chunk_count);
            println!("dtype: {}", meta.dtype.as_str());
            println!("shape: {:?}", meta.shape);
            println!("element_count: {}", meta.element_count);
            println!("digest_root: {}", meta.digest_root);
            println!("plane_codec: {}", meta.plane_codec);
            println!("payload_len: {}", meta.payload_len);
        }
        println!("flags: 0x{:04x} {:?}", header.flags, header.flag_names());
        println!(
            "features: 0x{:04x} {:?}",
            header.features,
            header.feature_names()
        );
        println!("level: {}", header.level);
        println!("decoded_len: {}", header.decoded_len);
        println!("body_len: {}", header.body_len);
        println!("table_len: {}", header.table_len);
        println!("frame_len: {}", frame_bytes.len());
        println!("content_hash: {}", header.content_hash);
        println!("dependencies: {}", header.dep_count);
        for (index, dep) in frame.dependencies.iter().enumerate() {
            println!(
                "  [{index}] hash={} decoded_len={} domain={} hint={}",
                dep.hash, dep.decoded_len, dep.dep_domain, dep.codec_hint
            );
        }
        println!("chunks: {}", header.chunk_count);
        for (index, chunk) in frame.chunks.iter().enumerate() {
            println!(
                "  [{index}] off={} encoded_len={} decoded_len={} hash={}",
                chunk.encoded_off, chunk.encoded_len, chunk.decoded_len, chunk.hash
            );
        }
    }
    Ok(())
}

fn archive_create(output: &str, frame_paths: &[String]) -> Result<()> {
    let limits = Limits::default();
    let mut frames = Vec::with_capacity(frame_paths.len());
    for path in frame_paths {
        frames.push(read_limited(path, limits.max_frame_bytes)?);
    }
    let archive = archive::create_archive(&frames, &limits)?;
    write_output(output, &archive)
}

fn archive_decode(input: &str, out_dir: &str, limits: &Limits) -> Result<()> {
    let bytes = read_limited(input, limits.max_total_fetched_bytes)?;
    let entries = archive::decode_archive(&bytes, limits)?;
    fs::create_dir_all(out_dir).map_err(|e| Error::io(format!("cannot create {out_dir}: {e}")))?;
    for entry in &entries {
        let path = Path::new(out_dir).join(entry.content_hash.to_hex());
        fs::write(&path, &entry.bytes)
            .map_err(|e| Error::io(format!("cannot write {}: {e}", path.display())))?;
        println!("{} {}", entry.content_hash, entry.bytes.len());
    }
    Ok(())
}

fn selector_id(input: &str, as_json: bool, limits: &Limits) -> Result<()> {
    let bytes = read_limited(input, limits.max_selector_bytes)?;
    let source = std::str::from_utf8(&bytes)
        .map_err(|_| Error::parse("selector input is not valid UTF-8"))?;
    let selector = selector::parse_with(source, selector::DEFAULT_NAMESPACE, &limits)?;
    let canonical = selector.canonical_bytes_with_limits(&limits)?;
    let id = hash_domain(weftcodec::hash::DOMAIN_SELECTOR, &canonical);
    if as_json {
        let output = json!({
            "schema": selector::SELECTOR_SCHEMA,
            "namespace": selector.namespace,
            "selector_id": id.to_hex(),
            "canonical_hex": hex::encode(&canonical),
            "canonical_len": canonical.len(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output)
                .map_err(|e| Error::io(format!("cannot serialize selector id: {e}")))?
        );
    } else {
        println!("{id}");
    }
    Ok(())
}

fn selector_canonicalize(input: &str, limits: &Limits) -> Result<()> {
    let bytes = read_limited(input, limits.max_selector_bytes)?;
    let source = std::str::from_utf8(&bytes)
        .map_err(|_| Error::parse("selector input is not valid UTF-8"))?;
    let selector = selector::parse_with(source, selector::DEFAULT_NAMESPACE, &limits)?;
    let canonical = selector.canonical_bytes_with_limits(&limits)?;
    println!("{}", hex::encode(canonical));
    Ok(())
}

fn hex_list(text: &str, what: &str) -> Result<Vec<weftcodec::Hash32>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut hashes = Vec::new();
    for entry in text.split(',') {
        if entry.is_empty() {
            return Err(Error::parse(format!("{what} contains an empty entry")));
        }
        let hash = weftcodec::Hash32::from_hex(entry)
            .map_err(|e| Error::parse(format!("{what} entry: {}", e.message())))?;
        if hash == weftcodec::Hash32::ZERO {
            return Err(Error::parse(format!("{what} entry must not be all zero")));
        }
        hashes.push(hash);
    }
    Ok(hashes)
}

fn optional_hash(text: Option<&str>, what: &str) -> Result<Option<weftcodec::Hash32>> {
    match text {
        Some(text) => {
            let hash = weftcodec::Hash32::from_hex(text)
                .map_err(|e| Error::parse(format!("{what}: {}", e.message())))?;
            if hash == weftcodec::Hash32::ZERO {
                return Err(Error::parse(format!("{what} must not be all zero")));
            }
            Ok(Some(hash))
        }
        None => Ok(None),
    }
}

fn load_context(args: &ContextArgs) -> Result<(ContextRoot, String)> {
    let limits = Limits::default();
    let bytes = read_limited(&args.selector, limits.max_selector_bytes)?;
    let source = std::str::from_utf8(&bytes)
        .map_err(|_| Error::parse("selector input is not valid UTF-8"))?
        .to_owned();
    let causal = match &args.causal {
        Some(text) => hex_list(text, "causal closure")?,
        None => Vec::new(),
    };
    let capability_hash = optional_hash(args.capability_hash.as_deref(), "capability hash")?;
    let budget = match &args.budget {
        Some(spec) => Some(Budget::parse_spec(spec)?),
        None => None,
    };
    let root = context::build_context_root(
        &source,
        &args.namespace,
        &args.snapshot,
        &args.resolver,
        &causal,
        capability_hash,
        budget,
    )?;
    Ok((root, source))
}

fn context_json(root: &ContextRoot) -> Result<serde_json::Value> {
    Ok(json!({
        "schema": context::CONTEXT_ROOT_SCHEMA,
        "namespace": root.namespace,
        "selector_id": root.selector_id.to_hex(),
        "selector_canonical_hex": hex::encode(&root.selector_bytes),
        "snapshot": root.snapshot.to_hex(),
        "resolver": root.resolver,
        "causal": root.causal.iter().map(|hash| hash.to_hex()).collect::<Vec<_>>(),
        "ordering": root.ordering,
        "capability_hash": root.capability_hash.map(|hash| hash.to_hex()),
        "budget": root.budget.as_ref().map(|budget| budget.to_json()),
        "context_root_id": root.id()?.to_hex(),
        "uri": root.authoritative_uri()?,
        "canonical_hex": root.canonical_hex()?,
    }))
}

fn context_id(args: &ContextArgs) -> Result<()> {
    let (root, _) = load_context(args)?;
    if args.json {
        let output = context_json(&root)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&output)
                .map_err(|e| Error::io(format!("cannot serialize context root: {e}")))?
        );
    } else {
        println!("{}", root.id()?);
    }
    Ok(())
}

fn context_uri(args: &ContextUriArgs) -> Result<()> {
    let (root, source) = load_context(&args.base)?;
    let uri = if args.descriptor {
        root.descriptor_uri(&source)?
    } else {
        root.authoritative_uri()?
    };
    if args.base.json {
        let output = json!({
            "uri": uri,
            "context_root_id": root.id()?.to_hex(),
            "descriptor": root.descriptor_uri(&source)?,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output)
                .map_err(|e| Error::io(format!("cannot serialize URI: {e}")))?
        );
    } else {
        println!("{uri}");
    }
    Ok(())
}

fn context_parse(input: &str, as_json: bool) -> Result<()> {
    let limits = Limits::default();
    let parsed = context::parse_context_uri(input, &limits)?;
    match &parsed {
        ContextUri::Authoritative { namespace, id } => {
            let uri = parsed.canonical_uri(&limits)?;
            if as_json {
                let output = json!({
                    "form": "authoritative",
                    "namespace": namespace,
                    "context_root_id": id.to_hex(),
                    "uri": uri,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&output)
                        .map_err(|e| Error::io(format!("cannot serialize URI: {e}")))?
                );
            } else {
                println!("{uri}");
            }
        }
        ContextUri::Descriptor {
            namespace,
            selector_source,
            ..
        } => {
            let root = parsed
                .to_context_root(&limits)?
                .ok_or_else(|| Error::non_canonical("descriptor produced no root"))?;
            let uri = root.authoritative_uri()?;
            if as_json {
                let mut output = context_json(&root)?;
                output["form"] = json!("descriptor");
                output["namespace"] = json!(namespace);
                output["selector_source"] = json!(selector_source);
                output["descriptor"] = json!(input);
                output["uri"] = json!(uri);
                println!(
                    "{}",
                    serde_json::to_string_pretty(&output)
                        .map_err(|e| Error::io(format!("cannot serialize URI: {e}")))?
                );
            } else {
                println!("{uri}");
            }
        }
    }
    Ok(())
}

fn causal_id(parents: &str, as_json: bool) -> Result<()> {
    let parents = hex_list(parents, "causal parents")?;
    let closure = context::CausalClosure::from_parents(&parents)?;
    let id = closure.id()?;
    if as_json {
        let output = json!({
            "schema": context::CAUSAL_SCHEMA,
            "parents": closure.parents.iter().map(|hash| hash.to_hex()).collect::<Vec<_>>(),
            "closure": closure.closure.iter().map(|hash| hash.to_hex()).collect::<Vec<_>>(),
            "closure_hash": id.to_hex(),
            "canonical_hex": hex::encode(closure.canonical_bytes()?),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output)
                .map_err(|e| Error::io(format!("cannot serialize causal closure: {e}")))?
        );
    } else {
        println!("{id}");
    }
    Ok(())
}

fn resolver_for(deps: Option<&str>) -> Box<dyn Resolver> {
    match deps {
        Some(dir) => Box::new(DirectoryResolver::new(dir)),
        None => Box::new(MapResolver::new()),
    }
}

fn parse_extents(text: &str, what: &str) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(Error::parse(format!("{what} contains an empty extent")));
        }
        out.push(
            part.parse::<u64>()
                .map_err(|err| Error::parse(format!("bad {what} extent `{part}`: {err}")))?,
        );
    }
    Ok(out)
}

fn tensor_record(
    dtype: tensor::TensorDtype,
    shape: &[u64],
    row_major: &[u8],
    limits: &Limits,
) -> Result<(tensor::LogicalRecord, weftcodec::Hash32)> {
    let record = tensor::LogicalRecord::from_row_major(dtype, shape, row_major, limits)?;
    let root = record.root_with_limits(limits)?;
    Ok((record, root))
}

fn tensor_root(args: &TensorRootArgs) -> Result<()> {
    let limits = Limits::default();
    if args.frames {
        let dir = args
            .shards_dir
            .as_deref()
            .ok_or_else(|| Error::parse("--frames requires --shards-dir"))?;
        if args.dtype.is_some() || args.shape.is_some() || args.input.is_some() {
            return Err(Error::parse(
                "--frames cannot be combined with --dtype, --shape, or --input",
            ));
        }
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| Error::io(format!("cannot read {dir}: {e}")))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("shard-") && name.ends_with(".wft1"))
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(Error::malformed(format!(
                "{dir} contains no shard-<index>.wft1 frames"
            )));
        }
        let mut shards = Vec::with_capacity(paths.len());
        for path in paths {
            let bytes = fs::read(&path)
                .map_err(|e| Error::io(format!("cannot read {}: {e}", path.display())))?;
            let parsed = Frame::parse(&bytes, &limits)?;
            shards.push(tensor_sharded::decode_shard_frame(&parsed, &limits)?);
        }
        let root = tensor_sharded::shard_set_root(&shards, &limits)?;
        let meta = &shards[0].meta;
        if args.json {
            let output = json!({
                "schema": tensor_sharded::TENSOR_SHARDED_SCHEMA,
                "dtype": meta.dtype.as_str(),
                "shape": meta.shape,
                "element_count": meta.element_count,
                "shard_axis": meta.shard_axis,
                "shard_count": meta.shard_count,
                "block_size": meta.block_size,
                "layout_root": root.to_hex(),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output)
                    .map_err(|e| Error::io(format!("cannot serialize tensor root: {e}")))?
            );
        } else {
            println!("{root}");
        }
        return Ok(());
    }
    let dtype_text = args
        .dtype
        .as_deref()
        .ok_or_else(|| Error::parse("tensor root requires --dtype (or --frames)"))?;
    let shape_text = args
        .shape
        .as_deref()
        .ok_or_else(|| Error::parse("tensor root requires --shape (or --frames)"))?;
    let dtype = tensor::dtype_from_name(dtype_text)?;
    let shape = parse_extents(shape_text, "shape")?;

    let (row_major, layout_name) = if let Some(dir) = &args.shards_dir {
        let axis = args
            .axis
            .ok_or_else(|| Error::parse("--shards-dir requires --axis"))?;
        let shards = args
            .shards
            .ok_or_else(|| Error::parse("--shards-dir requires --shards"))?;
        let mut decoded = Vec::new();
        for index in 0..shards {
            let path = Path::new(dir).join(format!("shard-{index}.bin"));
            let path = path
                .to_str()
                .ok_or_else(|| Error::parse("shard path is not valid UTF-8"))?;
            let bytes = read_limited(path, limits.max_decoded_bytes)?;
            let shard = tensor::decode_shard(&bytes, &limits)?;
            let expected = tensor::LayoutDescriptor::sharded(dtype, &shape, axis, shards, index);
            if shard.descriptor != expected {
                return Err(Error::malformed(format!(
                    "shard {index} descriptor does not match --dtype/--shape/--axis/--shards"
                )));
            }
            decoded.push(shard);
        }
        let assembled = tensor::assemble_shards(&decoded, &limits)?;
        (assembled, "sharded".to_owned())
    } else {
        let input = args
            .input
            .as_deref()
            .ok_or_else(|| Error::parse("tensor root requires --input or --shards-dir"))?;
        let data = read_limited(input, limits.max_decoded_bytes)?;
        match args.layout.as_str() {
            "row-major" => (data, "row-major".to_owned()),
            "column-major" => {
                let physical = tensor::PhysicalTensor::column_major(dtype, &shape, data, &limits)?;
                (
                    physical.assemble_with_limits(&limits)?,
                    "column-major".to_owned(),
                )
            }
            "blocked" => {
                let block_shape = parse_extents(
                    args.block_shape
                        .as_deref()
                        .ok_or_else(|| Error::parse("blocked layout requires --block-shape"))?,
                    "block shape",
                )?;
                let blocks =
                    tensor::split_blocked_payload(dtype, &shape, &block_shape, &data, &limits)?;
                let physical =
                    tensor::PhysicalTensor::blocked(dtype, &shape, &block_shape, blocks, &limits)?;
                (
                    physical.assemble_with_limits(&limits)?,
                    "blocked".to_owned(),
                )
            }
            "sharded" => {
                return Err(Error::parse(
                    "sharded layouts must be read with --shards-dir",
                ))
            }
            other => {
                return Err(Error::unsupported_feature(format!(
                    "unknown tensor layout `{other}`"
                )))
            }
        }
    };

    let (record, root) = tensor_record(dtype, &shape, &row_major, &limits)?;
    if args.json {
        let output = json!({
            "schema": tensor::LOGICAL_SCHEMA,
            "dtype": dtype.as_str(),
            "shape": shape,
            "layout": layout_name,
            "element_count": record.element_count,
            "values_hash": record.values_hash.to_hex(),
            "logical_root": root.to_hex(),
            "canonical_hex": hex::encode(record.canonical_bytes_with_limits(&limits)?),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output)
                .map_err(|e| Error::io(format!("cannot serialize tensor root: {e}")))?
        );
    } else {
        println!("{root}");
    }
    Ok(())
}

fn tensor_split(args: &TensorSplitArgs) -> Result<()> {
    let limits = Limits::default();
    let dtype = tensor::dtype_from_name(&args.dtype)?;
    let shape = parse_extents(&args.shape, "shape")?;
    let data = read_limited(&args.input, limits.max_decoded_bytes)?;
    let shards = tensor::split_shards(dtype, &shape, args.axis, args.shards, &data, &limits)?;
    fs::create_dir_all(&args.out_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", args.out_dir)))?;
    let mut entries = Vec::with_capacity(shards.len());
    for (index, payload) in shards.iter().enumerate() {
        let descriptor =
            tensor::LayoutDescriptor::sharded(dtype, &shape, args.axis, args.shards, index as u32);
        let container = tensor::encode_shard(&descriptor, payload, &limits)?;
        let path = Path::new(&args.out_dir).join(format!("shard-{index}.bin"));
        fs::write(&path, &container)
            .map_err(|e| Error::io(format!("cannot write {}: {e}", path.display())))?;
        entries.push(json!({
            "index": index,
            "payload_bytes": payload.len(),
            "payload_hash": tensor::shard_payload_hash(payload).to_hex(),
            "container_bytes": container.len(),
            "descriptor_hex": hex::encode(descriptor.canonical_bytes_with_limits(&limits)?),
        }));
    }
    let (record, root) = tensor_record(dtype, &shape, &data, &limits)?;
    if args.json {
        let output = json!({
            "schema": tensor::LOGICAL_SCHEMA,
            "dtype": dtype.as_str(),
            "shape": shape,
            "axis": args.axis,
            "shards": args.shards,
            "element_count": record.element_count,
            "values_hash": record.values_hash.to_hex(),
            "logical_root": root.to_hex(),
            "canonical_hex": hex::encode(record.canonical_bytes_with_limits(&limits)?),
            "out_dir": args.out_dir,
            "shard_files": entries,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output)
                .map_err(|e| Error::io(format!("cannot serialize tensor split: {e}")))?
        );
    } else {
        println!("{root}");
    }
    Ok(())
}

fn tensor_reassemble(args: &TensorReassembleArgs) -> Result<()> {
    let limits = Limits::default();
    let dtype = tensor::dtype_from_name(&args.dtype)?;
    let shape = parse_extents(&args.shape, "shape")?;
    let mut decoded = Vec::with_capacity(args.shards as usize);
    for index in 0..args.shards {
        let path = Path::new(&args.shards_dir).join(format!("shard-{index}.bin"));
        let path = path
            .to_str()
            .ok_or_else(|| Error::parse("shard path is not valid UTF-8"))?;
        let bytes = read_limited(path, limits.max_decoded_bytes)?;
        let shard = tensor::decode_shard(&bytes, &limits)?;
        let expected =
            tensor::LayoutDescriptor::sharded(dtype, &shape, args.axis, args.shards, index);
        if shard.descriptor != expected {
            return Err(Error::malformed(format!(
                "shard {index} descriptor does not match --dtype/--shape/--axis/--shards"
            )));
        }
        decoded.push(shard);
    }
    let root = tensor::shard_root(&decoded, &limits)?;
    let row_major = tensor::assemble_shards(&decoded, &limits)?;
    let (record, recomputed) = tensor_record(dtype, &shape, &row_major, &limits)?;
    if recomputed != root {
        return Err(Error::corrupt(
            "reassembled bytes do not reproduce the shard logical root",
        ));
    }
    if let Some(expected) = &args.expect_root {
        let expected = weftcodec::Hash32::from_hex(expected)
            .map_err(|err| Error::parse(format!("--expect-root: {}", err.message())))?;
        if expected != root {
            return Err(Error::corrupt(format!(
                "logical root {root} does not equal expected {expected}"
            )));
        }
    }
    if let Some(output) = &args.output {
        write_output(output, &row_major)?;
    }
    if args.json {
        let output = json!({
            "schema": tensor::LOGICAL_SCHEMA,
            "dtype": dtype.as_str(),
            "shape": shape,
            "axis": args.axis,
            "shards": args.shards,
            "element_count": record.element_count,
            "values_hash": record.values_hash.to_hex(),
            "logical_root": root.to_hex(),
            "canonical_hex": hex::encode(record.canonical_bytes_with_limits(&limits)?),
            "written": args.output,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output)
                .map_err(|e| Error::io(format!("cannot serialize tensor reassembly: {e}")))?
        );
    } else {
        println!("{root}");
    }
    Ok(())
}

fn read_envelope(path: &str) -> Result<CapabilityEnvelope> {
    let bytes = fs::read(path).map_err(|e| Error::io(format!("cannot read {path}: {e}")))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| Error::parse("capability envelope file is not UTF-8"))?;
    let trimmed = text.trim();
    if trimmed.starts_with('{') {
        CapabilityEnvelope::from_json_str(trimmed)
    } else {
        let raw = hex::decode(trimmed).map_err(|_| {
            Error::parse("capability envelope file is neither JSON nor canonical hex")
        })?;
        CapabilityEnvelope::from_canonical_bytes(&raw, &Limits::default())
    }
}

fn read_seed(path: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path).map_err(|e| Error::io(format!("cannot read {path}: {e}")))?;
    if bytes.len() == 32 {
        return Ok(bytes);
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| Error::parse("key file must be 32 raw bytes or 64 hex digits"))?;
    let trimmed = text.trim();
    let decoded = hex::decode(trimmed)
        .map_err(|_| Error::parse("key file must be 32 raw bytes or 64 hex digits"))?;
    if decoded.len() != 32 {
        return Err(Error::parse("Ed25519 seed must be 32 bytes"));
    }
    Ok(decoded)
}

fn read_revocations(spec: Option<&str>) -> Result<Vec<weftcodec::Hash32>> {
    let Some(path) = spec else {
        return Ok(Vec::new());
    };
    let text =
        fs::read_to_string(path).map_err(|e| Error::io(format!("cannot read {path}: {e}")))?;
    let trimmed = text.trim();
    let mut out = Vec::new();
    if trimmed.starts_with('[') {
        let items: Vec<String> = serde_json::from_str(trimmed)
            .map_err(|err| Error::parse(format!("bad revocation JSON: {err}")))?;
        for item in items {
            out.push(weftcodec::Hash32::from_hex(item.trim())?);
        }
    } else {
        for item in trimmed.split(|c: char| c == ',' || c.is_whitespace()) {
            if item.is_empty() {
                continue;
            }
            out.push(weftcodec::Hash32::from_hex(item)?);
        }
    }
    Ok(out)
}

fn capability_sign(envelope_path: &str, key_path: &str, out: &str, json: bool) -> Result<()> {
    let envelope = read_envelope(envelope_path)?;
    let seed = read_seed(key_path)?;
    let blob = capability::sign_envelope(&envelope, &seed)?;
    let bytes = blob.encode()?;
    let envelope_id = envelope.id()?;
    let blob_id = blob.id()?;
    write_output(out, &bytes)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "envelope_id": envelope_id.to_hex(),
                "blob_id": blob_id.to_hex(),
                "tenant": envelope.tenant,
                "grantor": envelope.grantor.to_hex(),
                "grantee": envelope.grantee.to_hex(),
                "expires_at_ns": envelope.expires_at_ns,
                "revocation_id": envelope.revocation_id.to_hex(),
                "signature": hex::encode(&blob.signature),
                "blob_bytes": bytes.len(),
            }))
            .map_err(|e| Error::io(format!("cannot serialize sign result: {e}")))?
        );
    } else {
        println!("envelope_id={envelope_id}");
        println!("blob_id={blob_id}");
    }
    Ok(())
}

fn capability_verify(
    blob_path: &str,
    revocations_path: Option<&str>,
    now_ns: Option<i64>,
    grantee: Option<&str>,
    json: bool,
) -> Result<()> {
    let bytes =
        fs::read(blob_path).map_err(|e| Error::io(format!("cannot read {blob_path}: {e}")))?;
    let revocations = read_revocations(revocations_path)?;
    let expected = match grantee {
        Some(text) => Some(weftcodec::Hash32::from_hex(text.trim())?),
        None => None,
    };
    let now = now_ns.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0)
    });
    let verified =
        capability::verify_envelope(&bytes, &revocations, now, expected).map_err(|err| {
            Error::new(
                err.class(),
                format!("{}: {}", err.rejection(), err.message()),
            )
        })?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "valid": true,
                "envelope_id": verified.envelope_id.to_hex(),
                "blob_id": verified.blob_id.to_hex(),
                "tenant": verified.envelope.tenant,
                "grantor": verified.envelope.grantor.to_hex(),
                "grantee": verified.envelope.grantee.to_hex(),
                "namespaces": verified.envelope.namespaces,
                "roots": verified.envelope.roots.iter().map(|h| h.to_hex()).collect::<Vec<_>>(),
                "predicates": verified.envelope.predicates,
                "expires_at_ns": verified.envelope.expires_at_ns,
                "revocation_id": verified.envelope.revocation_id.to_hex(),
            }))
            .map_err(|e| Error::io(format!("cannot serialize verify result: {e}")))?
        );
    } else {
        println!("valid=true");
        println!("envelope_id={}", verified.envelope_id);
        println!("blob_id={}", verified.blob_id);
        println!("tenant={}", verified.envelope.tenant);
        println!("grantee={}", verified.envelope.grantee);
    }
    Ok(())
}

fn capability_key(key_path: &str) -> Result<()> {
    let seed = read_seed(key_path)?;
    println!("{}", capability::public_key_for_seed(&seed)?);
    Ok(())
}

fn capability_id(envelope_path: &str, json: bool) -> Result<()> {
    let envelope = read_envelope(envelope_path)?;
    let id = envelope.id()?;
    let sign_input = envelope.sign_input()?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "schema": capability::ENVELOPE_SCHEMA,
                "envelope_id": id.to_hex(),
                "sign_input": sign_input.to_hex(),
                "canonical_hex": hex::encode(envelope.canonical_bytes()?),
                "tenant": envelope.tenant,
                "grantor": envelope.grantor.to_hex(),
                "grantee": envelope.grantee.to_hex(),
            }))
            .map_err(|e| Error::io(format!("cannot serialize envelope id: {e}")))?
        );
    } else {
        println!("{id}");
    }
    Ok(())
}

fn read_limited(path: &str, max: u64) -> Result<Vec<u8>> {
    if path == "-" {
        let read_cap = max
            .checked_add(1)
            .ok_or_else(|| Error::limit_exceeded("input limit overflow"))?;
        let mut buffer = Vec::new();
        std::io::stdin()
            .take(read_cap)
            .read_to_end(&mut buffer)
            .map_err(|e| Error::io(format!("cannot read stdin: {e}")))?;
        if buffer.len() as u64 > max {
            return Err(Error::limit_exceeded(format!(
                "stdin exceeds limit {max} bytes"
            )));
        }
        return Ok(buffer);
    }
    let metadata = fs::metadata(path).map_err(|e| Error::io(format!("cannot stat {path}: {e}")))?;
    if metadata.len() > max {
        return Err(Error::limit_exceeded(format!(
            "{} has {} bytes, limit {max}",
            path,
            metadata.len()
        )));
    }
    let bytes = fs::read(path).map_err(|e| Error::io(format!("cannot read {path}: {e}")))?;
    if bytes.len() as u64 > max {
        return Err(Error::limit_exceeded(format!(
            "{path} grew to {} bytes while reading, limit {max}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

fn write_output(path: &str, bytes: &[u8]) -> Result<()> {
    if path == "-" {
        let mut stdout = std::io::stdout();
        stdout
            .write_all(bytes)
            .map_err(|e| Error::io(format!("cannot write stdout: {e}")))?;
        stdout
            .flush()
            .map_err(|e| Error::io(format!("cannot flush stdout: {e}")))?;
        Ok(())
    } else {
        fs::write(path, bytes).map_err(|e| Error::io(format!("cannot write {path}: {e}")))
    }
}
