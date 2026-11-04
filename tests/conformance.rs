//! Verify every committed v1 conformance vector.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;
use weftcodec::dependency::MapResolver;
use weftcodec::hash::hash_domain;
use weftcodec::hash::DOMAIN_CHUNK;
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::selector;
use weftcodec::wcb;

fn v1_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/v1")
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();
    files.sort();
    files
}

fn load(path: &Path) -> Json {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn limits_from(vector: &Json) -> Limits {
    let mut limits = Limits::default();
    if let Some(object) = vector.get("limits").and_then(Json::as_object) {
        for (key, value) in object {
            let Some(number) = value.as_u64() else {
                continue;
            };
            limits = match key.as_str() {
                "max_frame_bytes" => limits.with_max_frame_bytes(number),
                "max_decoded_bytes" => limits.with_max_decoded_bytes(number),
                "max_dependency_depth" => limits.with_max_dependency_depth(number),
                "max_dependency_count" => limits.with_max_dependency_count(number),
                "max_total_fetched_bytes" => limits.with_max_total_fetched_bytes(number),
                "max_chunks" => limits.with_max_chunks(number),
                "max_dictionary_bytes" => limits.with_max_dictionary_bytes(number),
                "max_structured_depth" => limits.with_max_structured_depth(number),
                "max_structured_items" => limits.with_max_structured_items(number),
                "max_string_bytes" => limits.with_max_string_bytes(number),
                "max_delta_ops" => limits.with_max_delta_ops(number),
                "max_tensor_elements" => limits.with_max_tensor_elements(number),
                "max_tensor_rank" => limits.with_max_tensor_rank(number),
                "max_tokens" => limits.with_max_tokens(number),
                _ => limits,
            };
        }
    }
    limits
}

fn hex_field(vector: &Json, key: &str) -> Vec<u8> {
    let text = vector
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("vector missing {key}"));
    hex::decode(text).unwrap_or_else(|e| panic!("bad hex in {key}: {e}"))
}

#[test]
fn codec_vectors() {
    let dir = v1_dir().join("codec");
    let files = json_files(&dir);
    assert!(files.len() >= 20, "expected at least 20 codec vectors");
    for path in files {
        let vector = load(&path);
        if vector.get("superseded_by").is_some() {
            continue;
        }
        let name = vector["name"].as_str().unwrap_or("?");
        let frame = hex_field(&vector, "frame_hex");
        let limits = limits_from(&vector);

        let mut resolver = MapResolver::new();
        if let Some(deps) = vector.get("dependencies").and_then(Json::as_array) {
            for dep in deps {
                let hash = weftcodec::Hash32::from_hex(dep["hash"].as_str().unwrap())
                    .expect("dependency hash");
                let bytes = hex::decode(dep["bytes_hex"].as_str().unwrap()).expect("dep hex");
                assert_eq!(bytes.len() as u64, dep["decoded_len"].as_u64().unwrap());
                resolver.insert(hash, bytes);
            }
        }

        if let Some(error_class) = vector["expect"].get("error").and_then(Json::as_str) {
            let error = modes::decode_bytes(&frame, &resolver, &limits)
                .expect_err(&format!("{name} should fail"));
            assert_eq!(error.class().as_str(), error_class, "{name}: {error}");
            continue;
        }

        let decoded = modes::decode_bytes(&frame, &resolver, &limits)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let plaintext = hex_field(&vector, "plaintext_hex");
        assert_eq!(decoded, plaintext, "{name} plaintext");
        assert_eq!(
            hash_domain(DOMAIN_CHUNK, &decoded).to_hex(),
            vector["plaintext_hash"].as_str().unwrap(),
            "{name} plaintext hash"
        );
        let parsed = weftcodec::frame::Frame::parse(&frame, &limits).unwrap();
        assert_eq!(
            parsed.header.mode.as_str(),
            vector["mode"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            parsed.header.content_hash.to_hex(),
            vector["content_hash"].as_str().unwrap(),
            "{name} header content hash"
        );
        if let Some(chunks) = vector.get("chunk_count").and_then(Json::as_u64) {
            assert_eq!(u64::from(parsed.header.chunk_count), chunks, "{name}");
        }
    }
}

#[test]
fn structured_vectors() {
    let dir = v1_dir().join("structured");
    let files = json_files(&dir);
    assert!(files.len() >= 10, "expected at least 10 structured vectors");
    for path in files {
        let vector = load(&path);
        let name = vector["name"].as_str().unwrap_or("?");
        let limits = limits_from(&vector);
        if let Some(error_class) = vector.get("expect_error").and_then(Json::as_str) {
            let bytes = hex_field(&vector, "hex");
            let error = wcb::decode(&bytes, &limits).expect_err(name);
            assert_eq!(error.class().as_str(), error_class, "{name}: {error}");
            continue;
        }
        let bytes = hex_field(&vector, "canonical_hex");
        let value = wcb::decode(&bytes, &limits).unwrap_or_else(|e| panic!("{name}: {e}"));
        let reencoded = wcb::encode(&value, &limits).unwrap();
        assert_eq!(reencoded, bytes, "{name} canonical bytes");
    }
}

#[test]
fn selector_vectors() {
    let dir = v1_dir().join("selectors");
    let files = json_files(&dir);
    assert!(files.len() >= 10, "expected at least 10 selector vectors");
    for path in files {
        let vector = load(&path);
        let name = vector["name"].as_str().unwrap_or("?");
        let source = vector["source"].as_str().unwrap();
        let namespace = vector["namespace"]
            .as_str()
            .unwrap_or(selector::DEFAULT_NAMESPACE);
        let limits = Limits::default();
        let parsed = selector::parse_with(source, namespace, &limits)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let canonical = parsed.canonical_bytes_with_limits(&limits).unwrap();
        assert_eq!(
            hex::encode(&canonical),
            vector["canonical_hex"].as_str().unwrap(),
            "{name} canonical bytes"
        );
        let id = hash_domain(weftcodec::hash::DOMAIN_SELECTOR, &canonical);
        assert_eq!(
            id.to_hex(),
            vector["selector_id"].as_str().unwrap(),
            "{name} selector id"
        );
        let decoded = selector::Selector::from_canonical_bytes(&canonical, &limits)
            .unwrap_or_else(|e| panic!("{name} decode: {e}"));
        assert_eq!(decoded, parsed, "{name} AST");
    }
}

#[test]
fn vectors_are_not_empty_and_cover_modes() {
    let mut modes_seen = std::collections::HashSet::new();
    let mut errors_seen = std::collections::HashSet::new();
    for path in json_files(&v1_dir().join("codec")) {
        let vector = load(&path);
        if let Some(mode) = vector.get("mode").and_then(Json::as_str) {
            modes_seen.insert(mode.to_owned());
        }
        if let Some(error) = vector["expect"].get("error").and_then(Json::as_str) {
            errors_seen.insert(error.to_owned());
        }
    }
    for mode in [
        "raw",
        "zstd",
        "reference",
        "delta",
        "dictionary",
        "structured",
        "tensor",
        "tokens",
        "quantized_tensor",
        "float_tensor",
    ] {
        assert!(modes_seen.contains(mode), "missing mode vector: {mode}");
    }
    for class in [
        "Malformed",
        "Corrupt",
        "UnsupportedVersion",
        "UnsupportedMode",
        "UnsupportedFeature",
        "LimitExceeded",
        "DependencyMismatch",
        "NonCanonical",
    ] {
        assert!(
            errors_seen.contains(class),
            "missing error class vector: {class}"
        );
    }
}
