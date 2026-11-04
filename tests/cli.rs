//! End-to-end CLI checks, including `auto` honesty and typed error exits.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use weftcodec::hash::{hash_domain, DOMAIN_CHUNK};
use weftcodec::limits::Limits;
use weftcodec::modes;
use weftcodec::selector;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_weftcodec")
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("weftcodec-cli-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
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

fn pattern(len: usize, mut state: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state & 0xff) as u8);
    }
    out
}

fn inspect_json(path: &Path) -> serde_json::Value {
    let output = run(&["inspect", path.to_str().unwrap(), "--json"]);
    assert_ok(&output);
    serde_json::from_slice(&output.stdout).expect("inspect JSON")
}

#[test]
fn auto_chooses_raw_when_compression_loses() {
    let dir = temp_dir("auto-raw");
    let input = dir.join("input.bin");
    let output = dir.join("frame.wft");
    let data = pattern(4096, 0x1234_5678);
    fs::write(&input, &data).unwrap();
    let result = run(&[
        "encode",
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--mode",
        "auto",
    ]);
    assert_ok(&result);
    let info = inspect_json(&output);
    assert_eq!(info["mode"], "raw", "auto must pick RAW for random bytes");
    assert_eq!(info["decoded_len"], 4096);

    // Decode round-trip.
    let decoded = dir.join("decoded.bin");
    let result = run(&[
        "decode",
        "--input",
        output.to_str().unwrap(),
        "--output",
        decoded.to_str().unwrap(),
    ]);
    assert_ok(&result);
    assert_eq!(fs::read(decoded).unwrap(), data);
}

#[test]
fn auto_chooses_zstd_for_repetitive_data() {
    let dir = temp_dir("auto-zstd");
    let input = dir.join("input.txt");
    let output = dir.join("frame.wft");
    let data: Vec<u8> = b"weft codec context window "
        .iter()
        .copied()
        .cycle()
        .take(256 << 10)
        .collect();
    fs::write(&input, &data).unwrap();
    let result = run(&[
        "encode",
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--mode",
        "auto",
    ]);
    assert_ok(&result);
    let info = inspect_json(&output);
    assert_eq!(info["mode"], "zstd");
    assert!(
        fs::metadata(&output).unwrap().len() < data.len() as u64,
        "zstd frame should be smaller than plaintext"
    );
}

#[test]
fn delta_with_dependency_directory() {
    let dir = temp_dir("delta");
    let base = pattern(4096, 99);
    let mut target = base.clone();
    target[10] ^= 0xff;
    target.extend_from_slice(b"cli-delta-tail");
    let base_path = dir.join("base.bin");
    let target_path = dir.join("target.bin");
    let frame_path = dir.join("delta.wft");
    fs::write(&base_path, &base).unwrap();
    fs::write(&target_path, &target).unwrap();

    let result = run(&[
        "encode",
        "--input",
        target_path.to_str().unwrap(),
        "--output",
        frame_path.to_str().unwrap(),
        "--mode",
        "delta",
        "--base",
        base_path.to_str().unwrap(),
    ]);
    assert_ok(&result);
    let info = inspect_json(&frame_path);
    assert_eq!(info["mode"], "delta");
    assert_eq!(info["dep_count"], 1);

    let deps = dir.join("deps");
    fs::create_dir_all(&deps).unwrap();
    fs::write(deps.join(hash_domain(DOMAIN_CHUNK, &base).to_hex()), &base).unwrap();
    let decoded = dir.join("decoded.bin");
    let result = run(&[
        "decode",
        "--input",
        frame_path.to_str().unwrap(),
        "--output",
        decoded.to_str().unwrap(),
        "--deps",
        deps.to_str().unwrap(),
    ]);
    assert_ok(&result);
    assert_eq!(fs::read(&decoded).unwrap(), target);

    let verified = run(&[
        "verify",
        frame_path.to_str().unwrap(),
        "--deps",
        deps.to_str().unwrap(),
    ]);
    assert_ok(&verified);
    assert!(String::from_utf8_lossy(&verified.stdout).contains("OK mode=delta"));
}

#[test]
fn inspect_reports_wire_fields() {
    let dir = temp_dir("inspect");
    let data = pattern(3000, 7);
    let frame = modes::encode_zstd(&data, 3, Some(1024), &Limits::default()).unwrap();
    let path = dir.join("frame.wft");
    fs::write(&path, &frame).unwrap();
    let info = inspect_json(&path);
    assert_eq!(info["magic"], "WFT1");
    assert_eq!(info["format_major"], 1);
    assert_eq!(info["format_minor"], 0);
    assert_eq!(info["mode"], "zstd");
    assert_eq!(info["decoded_len"], 3000);
    assert_eq!(info["chunk_count"], 3);
    assert_eq!(info["flag_names"][0], "CHUNKED");
    assert_eq!(info["content_hash"].as_str().unwrap().len(), 64);
    assert_eq!(info["frame_len"], frame.len());
}

#[test]
fn structured_cli_roundtrip() {
    let dir = temp_dir("structured");
    let value = weftcodec::wcb::Value::map([
        ("a", weftcodec::wcb::Value::U64(1)),
        ("b", weftcodec::wcb::Value::Str("two".to_owned())),
    ]);
    let bytes = weftcodec::wcb::encode_default(&value).unwrap();
    let input = dir.join("value.wcb");
    let frame_path = dir.join("value.wft");
    let output = dir.join("out.wcb");
    fs::write(&input, &bytes).unwrap();
    let result = run(&[
        "encode",
        "--input",
        input.to_str().unwrap(),
        "--output",
        frame_path.to_str().unwrap(),
        "--mode",
        "structured",
    ]);
    assert_ok(&result);
    let info = inspect_json(&frame_path);
    assert_eq!(info["mode"], "structured");
    assert_eq!(info["domain"], 1);
    let result = run(&[
        "decode",
        "--input",
        frame_path.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
    ]);
    assert_ok(&result);
    assert_eq!(fs::read(output).unwrap(), bytes);
}

#[test]
fn archive_cli_roundtrip() {
    let dir = temp_dir("archive");
    let base = pattern(2048, 11);
    let mut target = base.clone();
    target[3] ^= 0x55;
    target.extend_from_slice(b"archive");
    let base_frame = modes::encode_raw(&base, None, &Limits::default()).unwrap();
    let delta_frame = modes::encode_delta(&target, &base, &Limits::default()).unwrap();
    let base_path = dir.join("base.wft");
    let delta_path = dir.join("delta.wft");
    fs::write(&base_path, &base_frame).unwrap();
    fs::write(&delta_path, &delta_frame).unwrap();
    let archive_path = dir.join("bundle.wfa");
    // Pass the dependent frame first; creation must reorder.
    let result = run(&[
        "archive",
        "create",
        "--output",
        archive_path.to_str().unwrap(),
        delta_path.to_str().unwrap(),
        base_path.to_str().unwrap(),
    ]);
    assert_ok(&result);
    let out_dir = dir.join("out");
    let result = run(&[
        "archive",
        "decode",
        "--input",
        archive_path.to_str().unwrap(),
        "--out-dir",
        out_dir.to_str().unwrap(),
    ]);
    assert_ok(&result);
    let base_out = out_dir.join(hash_domain(DOMAIN_CHUNK, &base).to_hex());
    let target_out = out_dir.join(hash_domain(DOMAIN_CHUNK, &target).to_hex());
    assert_eq!(fs::read(base_out).unwrap(), base);
    assert_eq!(fs::read(target_out).unwrap(), target);
}

#[test]
fn selector_cli_id_and_canonicalize() {
    let dir = temp_dir("selector");
    let source =
        "select attrs.project == \"demo\" and attrs.size >= 10 order by attrs.seq desc limit 5";
    let path = dir.join("query.weftql");
    fs::write(&path, source).unwrap();
    let parsed = selector::parse(source).unwrap();
    let expected_id = parsed.selector_id().unwrap().to_hex();
    let expected_bytes = hex::encode(parsed.canonical_bytes().unwrap());

    let result = run(&[
        "selector",
        "id",
        "--input",
        path.to_str().unwrap(),
        "--json",
    ]);
    assert_ok(&result);
    let json: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(json["selector_id"], expected_id);
    assert_eq!(json["canonical_hex"], expected_bytes);

    let result = run(&[
        "selector",
        "canonicalize",
        "--input",
        path.to_str().unwrap(),
    ]);
    assert_ok(&result);
    assert_eq!(
        String::from_utf8_lossy(&result.stdout).trim(),
        expected_bytes
    );
}

#[test]
fn errors_exit_nonzero_and_print_class() {
    let dir = temp_dir("errors");
    let mut frame = modes::encode_raw(b"hello", None, &Limits::default()).unwrap();
    let last = frame.len() - 1;
    frame[last] ^= 0xff;
    let path = dir.join("bad.wft");
    fs::write(&path, &frame).unwrap();
    let result = run(&[
        "decode",
        "--input",
        path.to_str().unwrap(),
        "--output",
        dir.join("out").to_str().unwrap(),
    ]);
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("Corrupt"), "stderr was: {stderr}");

    let missing = dir.join("missing.wft");
    let result = run(&[
        "decode",
        "--input",
        missing.to_str().unwrap(),
        "--output",
        dir.join("out").to_str().unwrap(),
    ]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("Io"));
}
