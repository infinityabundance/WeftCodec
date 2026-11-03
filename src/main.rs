//! `weftcodec` command-line interface.
#![forbid(unsafe_code)]

use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use serde_json::json;

use weftcodec::archive;
use weftcodec::context::{self, Budget, ContextRoot, ContextUri, DEFAULT_RESOLVER};
use weftcodec::dependency::{DirectoryResolver, MapResolver, Resolver};
use weftcodec::error::{Error, ErrorClass, Result};
use weftcodec::frame::{Frame, Mode};
use weftcodec::hash::hash_domain;
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::selector;
use weftcodec::wcb;

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
        value_parser = ["raw", "zstd", "reference", "delta", "dictionary", "structured", "auto"]
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
}

#[derive(Args, Debug)]
struct DecodeArgs {
    #[arg(long)]
    input: String,
    #[arg(long)]
    output: String,
    /// Directory of dependency files named by lowercase hex digest.
    #[arg(long)]
    deps: Option<String>,
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
    },
    /// Print canonical WCB bytes as lowercase hex.
    Canonicalize {
        #[arg(long)]
        input: String,
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

#[derive(Args, Debug)]
struct ConformanceArgs {
    /// Directory to write vectors into (e.g. conformance/v1).
    #[arg(long)]
    dir: String,
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
            ArchiveCommand::Decode { input, out_dir } => archive_decode(&input, &out_dir),
        },
        Command::Selector { command } => match command {
            SelectorCommand::Id { input, json } => selector_id(&input, json),
            SelectorCommand::Canonicalize { input } => selector_canonicalize(&input),
        },
        Command::Context { command } => match command {
            ContextCommand::Id(args) => context_id(&args),
            ContextCommand::Uri(args) => context_uri(&args),
            ContextCommand::Parse { uri, json } => context_parse(&uri, json),
        },
        Command::Causal { command } => match command {
            CausalCommand::Id { parents, json } => causal_id(&parents, json),
        },
        Command::Conformance(args) => {
            let limits = Limits::default();
            weftcodec::conformance::generate_v1(Path::new(&args.dir), &limits)?;
            eprintln!("wrote conformance vectors to {}", args.dir);
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

    let frame = if args.mode == "auto" {
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
    }
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
    let limits = Limits::default();
    let frame_bytes = read_limited(&args.input, limits.max_frame_bytes)?;
    let resolver = resolver_for(args.deps.as_deref());
    let plaintext = modes::decode_bytes(&frame_bytes, resolver.as_ref(), &limits)?;
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
        let output = json!({
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

fn archive_decode(input: &str, out_dir: &str) -> Result<()> {
    let limits = Limits::default();
    let bytes = read_limited(input, limits.max_total_fetched_bytes)?;
    let entries = archive::decode_archive(&bytes, &limits)?;
    fs::create_dir_all(out_dir).map_err(|e| Error::io(format!("cannot create {out_dir}: {e}")))?;
    for entry in &entries {
        let path = Path::new(out_dir).join(entry.content_hash.to_hex());
        fs::write(&path, &entry.bytes)
            .map_err(|e| Error::io(format!("cannot write {}: {e}", path.display())))?;
        println!("{} {}", entry.content_hash, entry.bytes.len());
    }
    Ok(())
}

fn selector_id(input: &str, as_json: bool) -> Result<()> {
    let limits = Limits::default();
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

fn selector_canonicalize(input: &str) -> Result<()> {
    let limits = Limits::default();
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
