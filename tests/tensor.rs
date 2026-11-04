//! Replay the committed `conformance/v1/tensor` vectors and exercise the
//! tensor CLI verbs.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value as Json;
use weftcodec::limits::Limits;
use weftcodec::tensor::{self, DecodedShard, LayoutDescriptor, LogicalRecord};

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

fn text_field<'a>(vector: &'a Json, key: &str) -> &'a str {
    vector
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("vector missing {key}"))
}

fn hex_field(vector: &Json, key: &str) -> Vec<u8> {
    let text = vector
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("vector missing {key}"));
    hex::decode(text).unwrap_or_else(|e| panic!("bad hex in {key}: {e}"))
}

fn hex_list(vector: &Json, key: &str) -> Vec<Vec<u8>> {
    vector
        .get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("vector missing {key}"))
        .iter()
        .map(|value| {
            hex::decode(value.as_str().expect("hex text")).unwrap_or_else(|e| panic!("{key}: {e}"))
        })
        .collect()
}

fn u64_field(vector: &Json, key: &str) -> u64 {
    vector
        .get(key)
        .and_then(Json::as_u64)
        .unwrap_or_else(|| panic!("vector missing {key}"))
}

fn u64_list(vector: &Json, key: &str) -> Vec<u64> {
    vector
        .get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("vector missing {key}"))
        .iter()
        .map(|value| value.as_u64().expect("u64 extent"))
        .collect()
}

#[test]
fn tensor_vectors() {
    let dir = v1_dir().join("tensor");
    let files = json_files(&dir);
    assert!(
        files.len() >= 28,
        "expected at least 28 tensor vectors, found {}",
        files.len()
    );
    let limits = Limits::default();
    let mut logical_ok = 0;
    let mut logical_rejects = 0;
    let mut layout_ok = 0;
    let mut layout_rejects = 0;
    let mut shard_sets = 0;

    for path in files {
        let vector = load(&path);
        let name = text_field(&vector, "name");
        let kind = text_field(&vector, "kind");
        match kind {
            "logical" => {
                if let Some(expected) = vector.get("expect_error").and_then(Json::as_str) {
                    let bytes = hex_field(&vector, "record_hex");
                    let error = LogicalRecord::from_canonical_bytes(&bytes, &limits)
                        .expect_err(&format!("{name} should fail"));
                    assert_eq!(error.class().as_str(), expected, "{name}: {error}");
                    logical_rejects += 1;
                    continue;
                }
                let dtype = tensor::dtype_from_name(text_field(&vector, "dtype")).unwrap();
                let shape = u64_list(&vector, "shape");
                let values = hex_field(&vector, "values_hex");
                let record = LogicalRecord::from_row_major(dtype, &shape, &values, &limits)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    hex::encode(record.canonical_bytes_with_limits(&limits).unwrap()),
                    text_field(&vector, "canonical_hex"),
                    "{name} canonical bytes"
                );
                assert_eq!(
                    record.values_hash.to_hex(),
                    text_field(&vector, "values_hash"),
                    "{name} values hash"
                );
                assert_eq!(
                    record.element_count,
                    u64_field(&vector, "element_count"),
                    "{name} element count"
                );
                assert_eq!(
                    record.root_with_limits(&limits).unwrap().to_hex(),
                    text_field(&vector, "logical_root"),
                    "{name} root"
                );
                assert_eq!(
                    tensor::logical_root(dtype, &shape, &values, &limits)
                        .unwrap()
                        .to_hex(),
                    text_field(&vector, "logical_root"),
                    "{name} direct root"
                );
                // The parsed record must round-trip.
                let bytes = hex_field(&vector, "canonical_hex");
                let parsed = LogicalRecord::from_canonical_bytes(&bytes, &limits)
                    .unwrap_or_else(|e| panic!("{name} parse: {e}"));
                assert_eq!(parsed, record, "{name} round-trip");
                logical_ok += 1;
            }
            "layout" => {
                if let Some(expected) = vector.get("expect_error").and_then(Json::as_str) {
                    let bytes = hex_field(&vector, "record_hex");
                    let error = LayoutDescriptor::from_canonical_bytes(&bytes, &limits)
                        .expect_err(&format!("{name} should fail"));
                    assert_eq!(error.class().as_str(), expected, "{name}: {error}");
                    layout_rejects += 1;
                    continue;
                }
                let bytes = hex_field(&vector, "canonical_hex");
                let descriptor = LayoutDescriptor::from_canonical_bytes(&bytes, &limits)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    descriptor.dtype.as_str(),
                    text_field(&vector, "dtype"),
                    "{name} dtype"
                );
                assert_eq!(descriptor.shape, u64_list(&vector, "shape"), "{name} shape");
                assert_eq!(
                    descriptor.layout.name(),
                    text_field(&vector, "layout"),
                    "{name} layout"
                );
                match &descriptor.layout {
                    tensor::TensorLayout::Blocked { block_shape } => {
                        assert_eq!(
                            block_shape,
                            &u64_list(&vector, "block_shape"),
                            "{name} block shape"
                        );
                    }
                    tensor::TensorLayout::Sharded {
                        axis,
                        shards,
                        index,
                    } => {
                        assert_eq!(u64::from(*axis), u64_field(&vector, "axis"), "{name} axis");
                        assert_eq!(
                            u64::from(*shards),
                            u64_field(&vector, "shards"),
                            "{name} shards"
                        );
                        assert_eq!(
                            u64::from(*index),
                            u64_field(&vector, "index"),
                            "{name} index"
                        );
                    }
                    _ => {}
                }
                assert_eq!(
                    hex::encode(descriptor.canonical_bytes_with_limits(&limits).unwrap()),
                    text_field(&vector, "canonical_hex"),
                    "{name} re-encode"
                );
                layout_ok += 1;
            }
            "shards" => {
                let dtype = tensor::dtype_from_name(text_field(&vector, "dtype")).unwrap();
                let shape = u64_list(&vector, "shape");
                let axis = u64_field(&vector, "axis") as u32;
                let shards = u64_field(&vector, "shards") as u32;
                let values = hex_field(&vector, "values_hex");
                let expected_root = text_field(&vector, "logical_root");
                assert_eq!(
                    tensor::logical_root(dtype, &shape, &values, &limits)
                        .unwrap()
                        .to_hex(),
                    expected_root,
                    "{name} logical root"
                );

                let shard_payloads = hex_list(&vector, "shard_hex");
                let shard_descriptors = hex_list(&vector, "shard_descriptor_hex");
                let shard_containers = hex_list(&vector, "shard_container_hex");
                assert_eq!(shard_payloads.len(), shards as usize, "{name}");
                assert_eq!(shard_descriptors.len(), shards as usize, "{name}");
                assert_eq!(shard_containers.len(), shards as usize, "{name}");

                let split =
                    tensor::split_shards(dtype, &shape, axis, shards, &values, &limits).unwrap();
                assert_eq!(split, shard_payloads, "{name} split bytes");

                let mut decoded: Vec<DecodedShard> = Vec::new();
                for index in 0..shards as usize {
                    let descriptor =
                        LayoutDescriptor::sharded(dtype, &shape, axis, shards, index as u32);
                    assert_eq!(
                        descriptor.canonical_bytes_with_limits(&limits).unwrap(),
                        shard_descriptors[index],
                        "{name} shard {index} descriptor"
                    );
                    let container =
                        tensor::encode_shard(&descriptor, &shard_payloads[index], &limits).unwrap();
                    assert_eq!(
                        container, shard_containers[index],
                        "{name} shard {index} container bytes"
                    );
                    decoded.push(tensor::decode_shard(&container, &limits).unwrap());
                }
                assert_eq!(
                    tensor::shard_root(&decoded, &limits).unwrap().to_hex(),
                    expected_root,
                    "{name} shard root"
                );
                assert_eq!(
                    tensor::assemble_shards(&decoded, &limits).unwrap(),
                    values,
                    "{name} reassembled bytes"
                );

                // Shuffled shard order is accepted and reordered by index.
                let mut shuffled = decoded.clone();
                shuffled.reverse();
                assert_eq!(
                    tensor::shard_root(&shuffled, &limits).unwrap().to_hex(),
                    expected_root,
                    "{name} shuffled shard root"
                );
                shard_sets += 1;
            }
            other => panic!("{name}: unknown kind `{other}`"),
        }
    }

    assert_eq!(logical_ok, 3, "valid logical vectors");
    assert!(logical_rejects >= 8, "logical reject vectors");
    assert_eq!(layout_ok, 4, "valid layout vectors");
    assert!(layout_rejects >= 8, "layout reject vectors");
    assert!(shard_sets >= 2, "shard vectors");
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

#[test]
fn tensor_cli_root_split_reassemble() {
    let dir = std::env::temp_dir().join(format!("weftcodec-tensor-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let limits = Limits::default();
    let dtype = tensor::dtype_from_name("bf16").unwrap();
    let shape = vec![4u64, 3, 2];
    let values = pattern(24 * 2, 0x5157);
    let expected = tensor::logical_root(dtype, &shape, &values, &limits)
        .unwrap()
        .to_hex();

    let logical = dir.join("logical.bin");
    fs::write(&logical, &values).unwrap();

    // row-major root
    let output = run(&[
        "tensor",
        "root",
        "--input",
        logical.to_str().unwrap(),
        "--dtype",
        "bf16",
        "--shape",
        "4,3,2",
        "--json",
    ]);
    assert_ok(&output);
    let info: Json = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(info["logical_root"], expected);
    assert_eq!(info["layout"], "row-major");

    // column-major root
    let column = tensor::column_major_bytes(dtype, &shape, &values, &limits).unwrap();
    let column_path = dir.join("column.bin");
    fs::write(&column_path, &column).unwrap();
    let output = run(&[
        "tensor",
        "root",
        "--input",
        column_path.to_str().unwrap(),
        "--dtype",
        "bf16",
        "--shape",
        "4,3,2",
        "--layout",
        "column-major",
        "--json",
    ]);
    assert_ok(&output);
    let info: Json = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(info["logical_root"], expected);
    assert_eq!(info["layout"], "column-major");

    // blocked root
    let block_shape = vec![2u64, 2, 2];
    let blocks = tensor::blocked_bytes(dtype, &shape, &block_shape, &values, &limits).unwrap();
    let blocked: Vec<u8> = blocks.iter().flat_map(|block| block.clone()).collect();
    let blocked_path = dir.join("blocked.bin");
    fs::write(&blocked_path, &blocked).unwrap();
    let output = run(&[
        "tensor",
        "root",
        "--input",
        blocked_path.to_str().unwrap(),
        "--dtype",
        "bf16",
        "--shape",
        "4,3,2",
        "--layout",
        "blocked",
        "--block-shape",
        "2,2,2",
        "--json",
    ]);
    assert_ok(&output);
    let info: Json = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(info["logical_root"], expected);

    // split + reassemble
    let shards_dir = dir.join("shards");
    let output = run(&[
        "tensor",
        "split",
        "--input",
        logical.to_str().unwrap(),
        "--dtype",
        "bf16",
        "--shape",
        "4,3,2",
        "--axis",
        "0",
        "--shards",
        "4",
        "--out-dir",
        shards_dir.to_str().unwrap(),
        "--json",
    ]);
    assert_ok(&output);
    let split: Json = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(split["logical_root"], expected);
    assert_eq!(split["shard_files"].as_array().unwrap().len(), 4);
    for index in 0..4 {
        assert!(shards_dir.join(format!("shard-{index}.bin")).is_file());
    }

    let reassembled = dir.join("reassembled.bin");
    let output = run(&[
        "tensor",
        "reassemble",
        "--shards-dir",
        shards_dir.to_str().unwrap(),
        "--dtype",
        "bf16",
        "--shape",
        "4,3,2",
        "--axis",
        "0",
        "--shards",
        "4",
        "--output",
        reassembled.to_str().unwrap(),
        "--expect-root",
        &expected,
        "--json",
    ]);
    assert_ok(&output);
    let info: Json = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(info["logical_root"], expected);
    assert_eq!(fs::read(&reassembled).unwrap(), values);

    // tensor root --shards-dir agrees.
    let output = run(&[
        "tensor",
        "root",
        "--shards-dir",
        shards_dir.to_str().unwrap(),
        "--dtype",
        "bf16",
        "--shape",
        "4,3,2",
        "--axis",
        "0",
        "--shards",
        "4",
        "--json",
    ]);
    assert_ok(&output);
    let info: Json = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(info["logical_root"], expected);
    assert_eq!(info["layout"], "sharded");

    // A wrong expected root fails closed.
    let bad = run(&[
        "tensor",
        "reassemble",
        "--shards-dir",
        shards_dir.to_str().unwrap(),
        "--dtype",
        "bf16",
        "--shape",
        "4,3,2",
        "--axis",
        "0",
        "--shards",
        "4",
        "--expect-root",
        &"00".repeat(32),
    ]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("Corrupt"));

    let _ = fs::remove_dir_all(&dir);
}
