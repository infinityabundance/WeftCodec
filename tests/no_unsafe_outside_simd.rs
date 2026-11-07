//! The crate has exactly one module that may contain `unsafe`: `src/simd.rs`.
//!
//! This test walks `src/` and fails if any non-comment line in another file
//! mentions `unsafe`. It is deliberately blunt: adding an intrinsic anywhere
//! else is a design change that must be made deliberately in `simd.rs`.

use std::path::Path;

fn visit(dir: &Path, findings: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).expect("read src dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            visit(&path, findings);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) == Some("simd.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read source file");
        for (line_number, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            // Doc and line comments may describe the policy.
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
                continue;
            }
            // `unsafe_code` appears in crate-level lint policy attributes; the
            // prohibition covers actual unsafe usage (`unsafe { ... }`,
            // `unsafe fn`, `unsafe impl`, `unsafe trait`).
            let code = trimmed.replace("unsafe_code", "");
            if code.contains("unsafe") {
                findings.push(format!(
                    "{}:{}: {}",
                    path.display(),
                    line_number + 1,
                    line.trim()
                ));
            }
        }
    }
}

#[test]
fn unsafe_is_confined_to_simd_module() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut findings = Vec::new();
    visit(&src, &mut findings);
    assert!(
        findings.is_empty(),
        "`unsafe` must appear only in src/simd.rs; found:\n{}",
        findings.join("\n")
    );
    // And the allowed module must actually be the one holding it.
    let simd = std::fs::read_to_string(src.join("simd.rs")).expect("simd.rs");
    assert!(simd.contains("unsafe"));
    assert!(simd.contains("allow(unsafe_code)"));
}
