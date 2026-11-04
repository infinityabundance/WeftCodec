//! Replay the committed `conformance/v1/context` and `.../causal` vectors.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value as Json;
use weftcodec::context::{self, Budget, CausalClosure, ContextRoot, ContextUri};
use weftcodec::hash::Hash32;
use weftcodec::limits::Limits;
use weftcodec::selector;

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

fn hex_field(vector: &Json, key: &str) -> Vec<u8> {
    let text = vector
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("vector missing {key}"));
    hex::decode(text).unwrap_or_else(|e| panic!("bad hex in {key}: {e}"))
}

fn text_field<'a>(vector: &'a Json, key: &str) -> &'a str {
    vector
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("vector missing {key}"))
}

fn hash_list(vector: &Json, key: &str) -> Vec<String> {
    vector
        .get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("vector missing {key}"))
        .iter()
        .map(|value| value.as_str().expect("hash text").to_owned())
        .collect()
}

#[test]
fn context_vectors() {
    let files = json_files(&v1_dir().join("context"));
    assert!(
        files.len() >= 15,
        "expected at least 15 context vectors, found {}",
        files.len()
    );
    let limits = Limits::default();
    for path in files {
        let vector = load(&path);
        let name = text_field(&vector, "name");
        if let Some(error_class) = vector.get("expect_error").and_then(Json::as_str) {
            let bytes = hex_field(&vector, "record_hex");
            let error = ContextRoot::from_canonical_bytes(&bytes, &limits)
                .expect_err(&format!("{name} should fail"));
            assert_eq!(error.class().as_str(), error_class, "{name}: {error}");
            continue;
        }

        let bytes = hex_field(&vector, "canonical_hex");
        let root = ContextRoot::from_canonical_bytes(&bytes, &limits)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            root.id().unwrap().to_hex(),
            text_field(&vector, "context_root_id"),
            "{name} id"
        );
        assert_eq!(
            root.authoritative_uri().unwrap(),
            text_field(&vector, "uri"),
            "{name} uri"
        );
        assert_eq!(
            root.selector_id.to_hex(),
            text_field(&vector, "selector_id"),
            "{name} selector id"
        );
        assert_eq!(
            hex::encode(&root.selector_bytes),
            text_field(&vector, "selector_canonical_hex"),
            "{name} selector bytes"
        );
        assert_eq!(root.namespace, text_field(&vector, "namespace"));
        assert_eq!(root.snapshot.to_hex(), text_field(&vector, "snapshot"));
        assert_eq!(root.resolver, text_field(&vector, "resolver"));
        assert_eq!(root.causal.len(), hash_list(&vector, "causal").len());
        for (hash, expected) in root.causal.iter().zip(hash_list(&vector, "causal")) {
            assert_eq!(hash.to_hex(), expected, "{name} causal");
        }
        assert_eq!(
            root.capability_hash.map(|h| h.to_hex()),
            vector
                .get("capability_hash")
                .and_then(Json::as_str)
                .map(str::to_owned),
            "{name} capability hash"
        );

        // The stored source must reproduce the canonical selector.
        let source = text_field(&vector, "selector_source");
        let parsed = selector::parse_with(source, &root.namespace, &limits)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            parsed.canonical_bytes().unwrap(),
            root.selector_bytes,
            "{name}"
        );

        // Descriptor round-trip.
        let descriptor = text_field(&vector, "descriptor");
        let parsed_uri = context::parse_context_uri(descriptor, &limits)
            .unwrap_or_else(|e| panic!("{name} descriptor: {e}"));
        assert_eq!(
            parsed_uri.canonical_uri(&limits).unwrap(),
            text_field(&vector, "uri"),
            "{name} descriptor canonicalization"
        );
        match parsed_uri {
            ContextUri::Descriptor {
                selector_source,
                closure,
                budget,
                ..
            } => {
                assert_eq!(selector_source, source, "{name} descriptor source");
                assert_eq!(closure, root.causal, "{name} descriptor closure");
                assert_eq!(budget, root.budget, "{name} descriptor budget");
            }
            ContextUri::Authoritative { .. } => panic!("{name}: expected descriptor"),
        }
    }
}

#[test]
fn causal_vectors() {
    let files = json_files(&v1_dir().join("causal"));
    assert!(
        files.len() >= 10,
        "expected at least 10 causal vectors, found {}",
        files.len()
    );
    let limits = Limits::default();
    for path in files {
        let vector = load(&path);
        let name = text_field(&vector, "name");
        if let Some(error_class) = vector.get("expect_error").and_then(Json::as_str) {
            let bytes = hex_field(&vector, "record_hex");
            let error = CausalClosure::from_canonical_bytes(&bytes, &limits)
                .expect_err(&format!("{name} should fail"));
            assert_eq!(error.class().as_str(), error_class, "{name}: {error}");
            continue;
        }
        let bytes = hex_field(&vector, "canonical_hex");
        let closure = CausalClosure::from_canonical_bytes(&bytes, &limits)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            closure.id().unwrap().to_hex(),
            text_field(&vector, "closure_hash"),
            "{name} id"
        );
        let expected_parents: Vec<Hash32> = hash_list(&vector, "parents")
            .iter()
            .map(|hash| Hash32::from_hex(hash).unwrap())
            .collect();
        let expected_closure: Vec<Hash32> = hash_list(&vector, "closure")
            .iter()
            .map(|hash| Hash32::from_hex(hash).unwrap())
            .collect();
        assert_eq!(closure.parents, expected_parents, "{name} parents");
        assert_eq!(closure.closure, expected_closure, "{name} closure");
    }
}

#[test]
fn vectors_cover_rejection_classes() {
    let mut classes = std::collections::HashSet::new();
    for path in json_files(&v1_dir().join("context")) {
        let vector = load(&path);
        if let Some(class) = vector.get("expect_error").and_then(Json::as_str) {
            classes.insert(class.to_owned());
        }
    }
    for class in ["NonCanonical", "UnknownSchema", "UnsupportedVersion"] {
        assert!(
            classes.contains(class),
            "missing context reject class {class}"
        );
    }
    let mut causal = std::collections::HashSet::new();
    for path in json_files(&v1_dir().join("causal")) {
        let vector = load(&path);
        if let Some(class) = vector.get("expect_error").and_then(Json::as_str) {
            causal.insert(class.to_owned());
        }
    }
    for class in ["NonCanonical", "UnknownSchema"] {
        assert!(
            causal.contains(class),
            "missing causal reject class {class}"
        );
    }
}

#[test]
fn budget_is_participates_in_identity() {
    let limits = Limits::default();
    let base = context::build_context_root(
        "select true",
        "default",
        &"11".repeat(32),
        context::DEFAULT_RESOLVER,
        &[],
        None,
        None,
    )
    .unwrap();
    let with_budget = context::build_context_root(
        "select true",
        "default",
        &"11".repeat(32),
        context::DEFAULT_RESOLVER,
        &[],
        None,
        Some(Budget::parse_spec("max_results=1").unwrap()),
    )
    .unwrap();
    assert_ne!(base.id().unwrap(), with_budget.id().unwrap());
    assert_eq!(
        with_budget.canonical_bytes_with_limits(&limits).unwrap(),
        ContextRoot::from_canonical_bytes(
            &with_budget.canonical_bytes_with_limits(&limits).unwrap(),
            &limits
        )
        .unwrap()
        .canonical_bytes()
        .unwrap()
    );
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_weftcodec")
}

fn run(args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .output()
        .expect("run weftcodec")
}

fn assert_ok(output: &Output) {
    assert!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn context_cli_id_uri_parse_and_causal() {
    let dir = std::env::temp_dir().join(format!("weftcodec-context-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let source = dir.join("query.weftql");
    fs::write(&source, "select attrs.project == \"demo\" limit 10").unwrap();
    let snapshot = "22".repeat(32);
    let causal = format!("{},{}", "bb".repeat(32), "aa".repeat(32));

    let output = run(&[
        "context",
        "id",
        "--selector",
        source.to_str().unwrap(),
        "--namespace",
        "default",
        "--snapshot",
        &snapshot,
        "--causal",
        &causal,
        "--budget",
        "max_results=10",
        "--json",
    ]);
    assert_ok(&output);
    let info: Json = serde_json::from_slice(&output.stdout).unwrap();
    let expected = context::build_context_root(
        "select attrs.project == \"demo\" limit 10",
        "default",
        &snapshot,
        context::DEFAULT_RESOLVER,
        &[
            Hash32::from_hex(&"bb".repeat(32)).unwrap(),
            Hash32::from_hex(&"aa".repeat(32)).unwrap(),
        ],
        None,
        Some(Budget::parse_spec("max_results=10").unwrap()),
    )
    .unwrap();
    assert_eq!(info["context_root_id"], expected.id().unwrap().to_hex());
    assert_eq!(info["uri"], expected.authoritative_uri().unwrap());

    let output = run(&[
        "context",
        "uri",
        "--selector",
        source.to_str().unwrap(),
        "--namespace",
        "default",
        "--snapshot",
        &snapshot,
        "--causal",
        &causal,
        "--budget",
        "max_results=10",
        "--descriptor",
    ]);
    assert_ok(&output);
    let descriptor = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    assert!(descriptor.starts_with("weft://default/c/select%20"));

    let output = run(&["context", "parse", &descriptor, "--json"]);
    assert_ok(&output);
    let parsed: Json = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(parsed["form"], "descriptor");
    assert_eq!(parsed["uri"], expected.authoritative_uri().unwrap());

    let output = run(&["causal", "id", "--parents", &causal, "--json"]);
    assert_ok(&output);
    let causal_info: Json = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(causal_info["parents"][0], "aa".repeat(32));
    assert_eq!(causal_info["parents"][1], "bb".repeat(32));
    assert_eq!(causal_info["closure_hash"].as_str().unwrap().len(), 64);

    // Bad hex must exit nonzero and name the class.
    let bad = run(&[
        "context",
        "id",
        "--selector",
        source.to_str().unwrap(),
        "--namespace",
        "default",
        "--snapshot",
        "zz",
    ]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("Parse"));
}
