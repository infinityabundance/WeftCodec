//! Mode 19 `KV_CACHE_PROV` (WEFT-AI-CODEC §20, `WEFT-KV-PROVENANCE.md`).
//!
//! Mode 12 `KV_CACHE` freezes a KV page's bytes and geometry but says nothing
//! about the model weights, adapters, tokenizer, attention configuration,
//! positional treatment, cache layout, or sharing domain that produced it.
//! A consumer that resolves a page by content hash alone can therefore reuse an
//! incompatible prefix silently.
//!
//! Mode 19 binds the page to those dimensions. The **payload is the mode-12
//! payload verbatim** — same byte planes, same per-plane `huffman-canonical-v1`
//! streams, same `decoded_len`, same `content_hash = H("weft/chunk/v1", bytes)`.
//! The only wire difference is the descriptor envelope, which carries the whole
//! frozen mode-12 page descriptor plus the provenance dimensions and the payload
//! hash. The page identity is
//!
//! ```text
//! descriptor_root = H("weft/kv/prov/v1", WCB(descriptor))
//! ```
//!
//! so two pages with identical token bytes but different provenance necessarily
//! have different roots. The decoder fails closed on every internal
//! inconsistency; a resolver that is asked for a page under a non-matching scope
//! must likewise fail closed and emit zero bytes (enforced by the store path in
//! `weftfs::kv_capability`).

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::frame::{encode_frame, Frame, FrameParts, Mode, DOMAIN_PAYLOAD};
use crate::hash::{hash_domain, Hash32, DOMAIN_CHUNK};
use crate::kv_cache::{self, KvCacheInput, KvCacheMeta, KvDtype};
use crate::limits::Limits;
use crate::wcb::{self, Value};

/// Canonical descriptor schema.
pub const KV_PROV_SCHEMA: &str = "weft.ai.kv_prov.v1";
/// Page identity domain.
pub const DOMAIN_KV_PROV: &str = "weft/kv/prov/v1";
/// Adapter-set identity domain.
pub const DOMAIN_KV_ADAPTER_SET: &str = "weft/kv/adapters/v1";
/// Tokenizer identity domain.
pub const DOMAIN_KV_TOKENIZER: &str = "weft/kv/tokenizer/v1";
/// The only positional treatments defined in v1.
pub const POSITIONAL_ROPE: &str = "rope";
pub const POSITIONAL_ALIBI: &str = "alibi";
pub const POSITIONAL_NONE: &str = "none";

const ROOT_LEN: usize = 32;

/// Positional-encoding treatment. v1 defines exactly three names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPositional {
    Rope,
    Alibi,
    None,
}

impl KvPositional {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            POSITIONAL_ROPE => Ok(KvPositional::Rope),
            POSITIONAL_ALIBI => Ok(KvPositional::Alibi),
            POSITIONAL_NONE => Ok(KvPositional::None),
            other => Err(Error::malformed(format!(
                "unknown KV provenance positional treatment `{other}` (v1 allows rope, alibi, none)"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            KvPositional::Rope => POSITIONAL_ROPE,
            KvPositional::Alibi => POSITIONAL_ALIBI,
            KvPositional::None => POSITIONAL_NONE,
        }
    }
}

/// Attention configuration. `rope_*_bits` are IEEE-754 binary64 bit patterns,
/// carried exactly so no floating-point round trip can change the identity.
/// Both MUST be finite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvAttention {
    pub head_count: u32,
    pub head_dim: u32,
    /// `0` means "no sliding window".
    pub sliding_window: u64,
    pub rope_theta_bits: u64,
    pub rope_scale_bits: u64,
}

impl KvAttention {
    fn validate(&self) -> Result<()> {
        if self.head_count == 0 {
            return Err(Error::malformed(
                "KV provenance attention head_count must be greater than zero",
            ));
        }
        if self.head_dim == 0 {
            return Err(Error::malformed(
                "KV provenance attention head_dim must be greater than zero",
            ));
        }
        if !f64::from_bits(self.rope_theta_bits).is_finite() {
            return Err(Error::malformed(
                "KV provenance attention rope_theta_bits must be a finite binary64",
            ));
        }
        if !f64::from_bits(self.rope_scale_bits).is_finite() {
            return Err(Error::malformed(
                "KV provenance attention rope_scale_bits must be a finite binary64",
            ));
        }
        Ok(())
    }

    fn to_value(self) -> Value {
        Value::map([
            ("head_count", Value::U64(u64::from(self.head_count))),
            ("head_dim", Value::U64(u64::from(self.head_dim))),
            ("sliding_window", Value::U64(self.sliding_window)),
            ("rope_theta_bits", Value::U64(self.rope_theta_bits)),
            ("rope_scale_bits", Value::U64(self.rope_scale_bits)),
        ])
    }
}

/// Cache layout dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayout {
    pub dtype: KvDtype,
    pub page_rows: u64,
    pub layer_count: u64,
}

impl KvLayout {
    fn validate(&self) -> Result<()> {
        if self.page_rows == 0 {
            return Err(Error::malformed(
                "KV provenance layout page_rows must be at least 1",
            ));
        }
        if self.layer_count == 0 {
            return Err(Error::malformed(
                "KV provenance layout layer_count must be at least 1",
            ));
        }
        Ok(())
    }

    fn to_value(self) -> Value {
        Value::map([
            ("dtype", Value::Str(self.dtype.as_str().to_owned())),
            ("page_rows", Value::U64(self.page_rows)),
            ("layer_count", Value::U64(self.layer_count)),
        ])
    }
}

/// One explicit extension dimension. The value is any canonical WCB value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvExtension {
    pub name: String,
    pub value: Value,
}

impl KvExtension {
    pub fn new(name: impl Into<String>, value: Value) -> Self {
        KvExtension {
            name: name.into(),
            value,
        }
    }

    fn to_value(&self) -> Value {
        Value::map([
            ("name", Value::Str(self.name.clone())),
            ("value", self.value.clone()),
        ])
    }
}

/// Caller-supplied provenance dimensions. Adapter roots and extensions may be
/// in any order; [`KvProvenanceInput::canonicalize`] sorts them and derives the
/// adapter-set and tokenizer roots.
#[derive(Clone, Debug)]
pub struct KvProvenanceInput {
    pub model_root: Hash32,
    pub adapter_roots: Vec<Hash32>,
    pub tokenizer_scheme: String,
    pub tokenizer_vocab_size: u64,
    pub attention: KvAttention,
    pub positional: KvPositional,
    pub layout: KvLayout,
    pub salt: Hash32,
    pub extensions: Vec<KvExtension>,
}

impl KvProvenanceInput {
    /// Build inputs with no adapters and no extensions.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_root: Hash32,
        tokenizer_scheme: impl Into<String>,
        tokenizer_vocab_size: u64,
        attention: KvAttention,
        positional: KvPositional,
        layout: KvLayout,
        salt: Hash32,
    ) -> Self {
        KvProvenanceInput {
            model_root,
            adapter_roots: Vec::new(),
            tokenizer_scheme: tokenizer_scheme.into(),
            tokenizer_vocab_size,
            attention,
            positional,
            layout,
            salt,
            extensions: Vec::new(),
        }
    }

    pub fn with_adapter(mut self, root: Hash32) -> Self {
        self.adapter_roots.push(root);
        self
    }

    pub fn with_extension(mut self, name: impl Into<String>, value: Value) -> Self {
        self.extensions.push(KvExtension::new(name, value));
        self
    }

    /// Sort and derive the canonical provenance dimensions.
    ///
    /// Duplicate adapter roots and duplicate extension names are rejected: the
    /// registry is a set, and silently collapsing a duplicate would hide a
    /// caller error.
    pub fn canonicalize(&self) -> Result<KvProvenance> {
        if self.model_root == Hash32::ZERO {
            return Err(Error::malformed(
                "KV provenance model_root must not be all zero",
            ));
        }
        let mut adapter_roots = self.adapter_roots.clone();
        adapter_roots.sort_unstable();
        for root in &adapter_roots {
            if *root == Hash32::ZERO {
                return Err(Error::malformed(
                    "KV provenance adapter roots must not be all zero",
                ));
            }
        }
        for pair in adapter_roots.windows(2) {
            if pair[0] == pair[1] {
                return Err(Error::malformed(
                    "KV provenance adapter roots contain a duplicate",
                ));
            }
        }
        if self.tokenizer_scheme.is_empty() {
            return Err(Error::malformed(
                "KV provenance tokenizer scheme must not be empty",
            ));
        }
        if self.tokenizer_vocab_size == 0 {
            return Err(Error::malformed(
                "KV provenance tokenizer vocab_size must be at least 1",
            ));
        }
        self.attention.validate()?;
        self.layout.validate()?;
        let mut extensions = self.extensions.clone();
        extensions.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
        for extension in &extensions {
            if extension.name.is_empty() {
                return Err(Error::malformed(
                    "KV provenance extension name must not be empty",
                ));
            }
        }
        for pair in extensions.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(Error::malformed(format!(
                    "KV provenance extension `{}` is duplicated",
                    pair[0].name
                )));
            }
        }
        let adapter_set_root = adapter_set_root(&adapter_roots)?;
        let tokenizer_root = tokenizer_root(&self.tokenizer_scheme, self.tokenizer_vocab_size)?;
        Ok(KvProvenance {
            model_root: self.model_root,
            adapter_roots,
            adapter_set_root,
            tokenizer_scheme: self.tokenizer_scheme.clone(),
            tokenizer_vocab_size: self.tokenizer_vocab_size,
            tokenizer_root,
            attention: self.attention,
            positional: self.positional,
            layout: self.layout,
            salt: self.salt,
            extensions,
        })
    }
}

/// Canonical (sorted, derived) provenance dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvProvenance {
    pub model_root: Hash32,
    pub adapter_roots: Vec<Hash32>,
    pub adapter_set_root: Hash32,
    pub tokenizer_scheme: String,
    pub tokenizer_vocab_size: u64,
    pub tokenizer_root: Hash32,
    pub attention: KvAttention,
    pub positional: KvPositional,
    pub layout: KvLayout,
    pub salt: Hash32,
    pub extensions: Vec<KvExtension>,
}

/// `H("weft/kv/adapters/v1", WCB(sorted adapter roots))`.
pub fn adapter_set_root(adapter_roots: &[Hash32]) -> Result<Hash32> {
    let array = Value::Array(
        adapter_roots
            .iter()
            .map(|root| Value::Bytes(root.as_bytes().to_vec()))
            .collect(),
    );
    let bytes = wcb::encode(&array, &Limits::default())?;
    Ok(hash_domain(DOMAIN_KV_ADAPTER_SET, &bytes))
}

/// `H("weft/kv/tokenizer/v1", WCB({scheme, vocab_size}))`.
pub fn tokenizer_root(scheme: &str, vocab_size: u64) -> Result<Hash32> {
    let value = Value::map([
        ("scheme", Value::Str(scheme.to_owned())),
        ("vocab_size", Value::U64(vocab_size)),
    ]);
    let bytes = wcb::encode(&value, &Limits::default())?;
    Ok(hash_domain(DOMAIN_KV_TOKENIZER, &bytes))
}

/// The complete mode-19 descriptor: the frozen mode-12 page descriptor plus the
/// provenance dimensions and the payload hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvProvDescriptor {
    pub page: KvCacheMeta,
    pub provenance: KvProvenance,
    pub payload_hash: Hash32,
}

impl KvProvDescriptor {
    /// Build a descriptor for an encoded page.
    pub fn from_page(page: KvCacheMeta, provenance: KvProvenance, elements: &[u8]) -> Result<Self> {
        let descriptor = KvProvDescriptor {
            page,
            provenance,
            payload_hash: hash_domain(DOMAIN_CHUNK, elements),
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    fn to_value(&self) -> Value {
        Value::map([
            ("schema", Value::Str(KV_PROV_SCHEMA.to_owned())),
            ("page", kv_cache::descriptor_value(&self.page)),
            (
                "model_root",
                Value::Bytes(self.provenance.model_root.as_bytes().to_vec()),
            ),
            (
                "adapter_roots",
                Value::Array(
                    self.provenance
                        .adapter_roots
                        .iter()
                        .map(|root| Value::Bytes(root.as_bytes().to_vec()))
                        .collect(),
                ),
            ),
            (
                "adapter_set_root",
                Value::Bytes(self.provenance.adapter_set_root.as_bytes().to_vec()),
            ),
            (
                "tokenizer",
                Value::map([
                    (
                        "scheme",
                        Value::Str(self.provenance.tokenizer_scheme.clone()),
                    ),
                    (
                        "vocab_size",
                        Value::U64(self.provenance.tokenizer_vocab_size),
                    ),
                ]),
            ),
            (
                "tokenizer_root",
                Value::Bytes(self.provenance.tokenizer_root.as_bytes().to_vec()),
            ),
            ("attention", self.provenance.attention.to_value()),
            (
                "positional",
                Value::Str(self.provenance.positional.as_str().to_owned()),
            ),
            ("layout", self.provenance.layout.to_value()),
            (
                "salt",
                Value::Bytes(self.provenance.salt.as_bytes().to_vec()),
            ),
            (
                "extensions",
                Value::Array(
                    self.provenance
                        .extensions
                        .iter()
                        .map(KvExtension::to_value)
                        .collect(),
                ),
            ),
            (
                "payload_hash",
                Value::Bytes(self.payload_hash.as_bytes().to_vec()),
            ),
        ])
    }

    /// The canonical descriptor value. Exposed for conformance tooling; the
    /// bytes to publish are [`Self::canonical_bytes`].
    pub fn descriptor_value(&self) -> Value {
        self.to_value()
    }

    /// Canonical WCB bytes.
    pub fn canonical_bytes(&self, limits: &Limits) -> Result<Vec<u8>> {
        wcb::encode(&self.to_value(), limits)
    }

    /// `H("weft/kv/prov/v1", canonical bytes)` — the KV provenance identity.
    pub fn root(&self, limits: &Limits) -> Result<Hash32> {
        Ok(hash_domain(DOMAIN_KV_PROV, &self.canonical_bytes(limits)?))
    }

    /// Strict decode of canonical descriptor bytes.
    pub fn from_canonical_bytes(bytes: &[u8], limits: &Limits) -> Result<Self> {
        let value = wcb::decode(bytes, limits)?;
        let Value::Map(map) = &value else {
            return Err(Error::malformed(
                "KV_CACHE_PROV descriptor must be a WCB map",
            ));
        };
        const EXPECTED: [&str; 13] = [
            "schema",
            "page",
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
            "payload_hash",
        ];
        exact_keys(map, &EXPECTED, "KV_CACHE_PROV")?;
        let schema = text(map, "schema", "KV_CACHE_PROV")?;
        if schema != KV_PROV_SCHEMA {
            return Err(Error::malformed(format!(
                "unsupported KV_CACHE_PROV schema `{schema}`"
            )));
        }
        let page_value = map
            .get("page")
            .ok_or_else(|| Error::malformed("KV_CACHE_PROV descriptor is missing `page`"))?;
        let page = kv_cache::parse_page_descriptor(page_value, limits)?;
        let model_root = root_value(map, "model_root", "KV_CACHE_PROV")?;
        if model_root == Hash32::ZERO {
            return Err(Error::malformed(
                "KV provenance model_root must not be all zero",
            ));
        }
        let adapter_roots = adapter_roots_value(map)?;
        let adapter_set = root_value(map, "adapter_set_root", "KV_CACHE_PROV")?;
        let tokenizer_value = map
            .get("tokenizer")
            .ok_or_else(|| Error::malformed("KV_CACHE_PROV descriptor is missing `tokenizer`"))?;
        let tokenizer_map = map_value(tokenizer_value, "tokenizer")?;
        exact_keys(
            tokenizer_map,
            &["scheme", "vocab_size"],
            "KV_CACHE_PROV tokenizer",
        )?;
        let tokenizer_scheme = text(tokenizer_map, "scheme", "KV_CACHE_PROV tokenizer")?;
        let tokenizer_vocab_size = number(tokenizer_map, "vocab_size", "KV_CACHE_PROV tokenizer")?;
        let tokenizer_root_recorded = root_value(map, "tokenizer_root", "KV_CACHE_PROV")?;
        let attention = attention_value(map)?;
        let positional = KvPositional::from_name(&text(map, "positional", "KV_CACHE_PROV")?)?;
        let layout = layout_value(map)?;
        let salt = root_value(map, "salt", "KV_CACHE_PROV")?;
        let extensions = extensions_value(map)?;
        let payload_hash = root_value(map, "payload_hash", "KV_CACHE_PROV")?;

        // Derived roots are recomputed, never trusted.
        if adapter_set_root(&adapter_roots)? != adapter_set {
            return Err(Error::malformed(
                "KV provenance adapter_set_root does not match the adapter roots",
            ));
        }
        if tokenizer_root(&tokenizer_scheme, tokenizer_vocab_size)? != tokenizer_root_recorded {
            return Err(Error::malformed(
                "KV provenance tokenizer_root does not match the tokenizer descriptor",
            ));
        }

        let descriptor = KvProvDescriptor {
            page,
            provenance: KvProvenance {
                model_root,
                adapter_roots,
                adapter_set_root: adapter_set,
                tokenizer_scheme,
                tokenizer_vocab_size,
                tokenizer_root: tokenizer_root_recorded,
                attention,
                positional,
                layout,
                salt,
                extensions,
            },
            payload_hash,
        };
        descriptor.validate()?;
        // Canonical form: the re-encoded bytes must be identical. This rejects
        // any non-canonical construction WCB itself would already refuse, and
        // any semantic drift introduced by the parser.
        if descriptor.canonical_bytes(limits)? != bytes {
            return Err(Error::malformed(
                "KV_CACHE_PROV descriptor is not canonical",
            ));
        }
        Ok(descriptor)
    }

    /// Validate every dimension and cross-check it against the frozen page.
    pub fn validate(&self) -> Result<()> {
        self.provenance.attention.validate()?;
        self.provenance.layout.validate()?;
        if self.provenance.model_root == Hash32::ZERO {
            return Err(Error::malformed(
                "KV provenance model_root must not be all zero",
            ));
        }
        // Adapter roots must be sorted, unique and non-zero.
        for root in &self.provenance.adapter_roots {
            if *root == Hash32::ZERO {
                return Err(Error::malformed(
                    "KV provenance adapter roots must not be all zero",
                ));
            }
        }
        for pair in self.provenance.adapter_roots.windows(2) {
            if pair[0] >= pair[1] {
                return Err(Error::malformed(
                    "KV provenance adapter roots are not strictly sorted and unique",
                ));
            }
        }
        if adapter_set_root(&self.provenance.adapter_roots)? != self.provenance.adapter_set_root {
            return Err(Error::malformed(
                "KV provenance adapter_set_root does not match the adapter roots",
            ));
        }
        if self.provenance.tokenizer_scheme.is_empty() || self.provenance.tokenizer_vocab_size == 0
        {
            return Err(Error::malformed(
                "KV provenance tokenizer descriptor must have a scheme and vocab_size >= 1",
            ));
        }
        if tokenizer_root(
            &self.provenance.tokenizer_scheme,
            self.provenance.tokenizer_vocab_size,
        )? != self.provenance.tokenizer_root
        {
            return Err(Error::malformed(
                "KV provenance tokenizer_root does not match the tokenizer descriptor",
            ));
        }
        // Extensions must be sorted by name and unique.
        for pair in self.provenance.extensions.windows(2) {
            if pair[0].name.as_bytes() >= pair[1].name.as_bytes() {
                return Err(Error::malformed(
                    "KV provenance extensions are not strictly sorted with unique names",
                ));
            }
        }
        for extension in &self.provenance.extensions {
            if extension.name.is_empty() {
                return Err(Error::malformed(
                    "KV provenance extension name must not be empty",
                ));
            }
        }
        // Cross-checks against the frozen mode-12 page descriptor.
        if self.provenance.layout.dtype != self.page.dtype {
            return Err(Error::malformed(
                "KV provenance layout dtype disagrees with the page dtype",
            ));
        }
        let rows = self
            .page
            .seq_end
            .checked_sub(self.page.seq_start)
            .ok_or_else(|| Error::malformed("KV_CACHE_PROV page sequence range underflows"))?;
        if self.provenance.layout.page_rows != rows {
            return Err(Error::malformed(
                "KV provenance layout page_rows disagrees with the page token range",
            ));
        }
        if self.provenance.attention.head_count != self.page.head_count {
            return Err(Error::malformed(
                "KV provenance attention head_count disagrees with the page",
            ));
        }
        if self.provenance.attention.head_dim != self.page.head_dim {
            return Err(Error::malformed(
                "KV provenance attention head_dim disagrees with the page",
            ));
        }
        if u64::from(self.page.layer) >= self.provenance.layout.layer_count {
            return Err(Error::malformed(
                "KV_CACHE_PROV page layer is not below the layout layer_count",
            ));
        }
        if self.payload_hash == Hash32::ZERO {
            return Err(Error::malformed(
                "KV provenance payload_hash must not be all zero",
            ));
        }
        Ok(())
    }
}

fn exact_keys(map: &BTreeMap<String, Value>, expected: &[&str], what: &str) -> Result<()> {
    for key in map.keys() {
        if !expected.contains(&key.as_str()) {
            return Err(Error::malformed(format!("{what} has unknown key `{key}`")));
        }
    }
    for key in expected {
        if !map.contains_key(*key) {
            return Err(Error::malformed(format!("{what} is missing key `{key}`")));
        }
    }
    Ok(())
}

fn map_value<'a>(value: &'a Value, what: &str) -> Result<&'a BTreeMap<String, Value>> {
    match value {
        Value::Map(map) => Ok(map),
        _ => Err(Error::malformed(format!("{what} must be a WCB map"))),
    }
}

fn text(map: &BTreeMap<String, Value>, key: &str, what: &str) -> Result<String> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text.clone()),
        _ => Err(Error::malformed(format!("{what} `{key}` must be text"))),
    }
}

fn number(map: &BTreeMap<String, Value>, key: &str, what: &str) -> Result<u64> {
    match map.get(key) {
        Some(Value::U64(value)) => Ok(*value),
        _ => Err(Error::malformed(format!("{what} `{key}` must be u64"))),
    }
}

fn root_bytes(value: &Value, what: &str) -> Result<Hash32> {
    match value {
        Value::Bytes(bytes) => {
            if bytes.len() != ROOT_LEN {
                return Err(Error::malformed(format!(
                    "{what} must be exactly {ROOT_LEN} bytes, got {}",
                    bytes.len()
                )));
            }
            let mut out = [0u8; ROOT_LEN];
            out.copy_from_slice(bytes);
            Ok(Hash32::from_bytes(out))
        }
        _ => Err(Error::malformed(format!("{what} must be bytes"))),
    }
}

fn root_value(map: &BTreeMap<String, Value>, key: &str, what: &str) -> Result<Hash32> {
    match map.get(key) {
        Some(value) => root_bytes(value, &format!("{what} `{key}`")),
        None => Err(Error::malformed(format!("{what} `{key}` is missing"))),
    }
}

fn adapter_roots_value(map: &BTreeMap<String, Value>) -> Result<Vec<Hash32>> {
    let Some(value) = map.get("adapter_roots") else {
        return Err(Error::malformed("KV_CACHE_PROV `adapter_roots` is missing"));
    };
    let Value::Array(items) = value else {
        return Err(Error::malformed(
            "KV_CACHE_PROV `adapter_roots` must be an array",
        ));
    };
    let mut roots = Vec::with_capacity(items.len());
    for item in items {
        roots.push(root_bytes(item, "KV provenance adapter root")?);
    }
    Ok(roots)
}

fn attention_value(map: &BTreeMap<String, Value>) -> Result<KvAttention> {
    let value = map
        .get("attention")
        .ok_or_else(|| Error::malformed("KV_CACHE_PROV `attention` is missing"))?;
    let attention = map_value(value, "KV_CACHE_PROV attention")?;
    exact_keys(
        attention,
        &[
            "head_count",
            "head_dim",
            "sliding_window",
            "rope_theta_bits",
            "rope_scale_bits",
        ],
        "KV_CACHE_PROV attention",
    )?;
    let head_count = u32::try_from(number(attention, "head_count", "KV_CACHE_PROV attention")?)
        .map_err(|_| Error::malformed("KV provenance attention head_count exceeds u32"))?;
    let head_dim = u32::try_from(number(attention, "head_dim", "KV_CACHE_PROV attention")?)
        .map_err(|_| Error::malformed("KV provenance attention head_dim exceeds u32"))?;
    Ok(KvAttention {
        head_count,
        head_dim,
        sliding_window: number(attention, "sliding_window", "KV_CACHE_PROV attention")?,
        rope_theta_bits: number(attention, "rope_theta_bits", "KV_CACHE_PROV attention")?,
        rope_scale_bits: number(attention, "rope_scale_bits", "KV_CACHE_PROV attention")?,
    })
}

fn layout_value(map: &BTreeMap<String, Value>) -> Result<KvLayout> {
    let value = map
        .get("layout")
        .ok_or_else(|| Error::malformed("KV_CACHE_PROV `layout` is missing"))?;
    let layout = map_value(value, "KV_CACHE_PROV layout")?;
    exact_keys(
        layout,
        &["dtype", "page_rows", "layer_count"],
        "KV_CACHE_PROV layout",
    )?;
    Ok(KvLayout {
        dtype: KvDtype::from_name(&text(layout, "dtype", "KV_CACHE_PROV layout")?)?,
        page_rows: number(layout, "page_rows", "KV_CACHE_PROV layout")?,
        layer_count: number(layout, "layer_count", "KV_CACHE_PROV layout")?,
    })
}

fn extensions_value(map: &BTreeMap<String, Value>) -> Result<Vec<KvExtension>> {
    let Some(value) = map.get("extensions") else {
        return Err(Error::malformed("KV_CACHE_PROV `extensions` is missing"));
    };
    let Value::Array(items) = value else {
        return Err(Error::malformed(
            "KV_CACHE_PROV `extensions` must be an array",
        ));
    };
    let mut extensions = Vec::with_capacity(items.len());
    for item in items {
        let extension = map_value(item, "KV_CACHE_PROV extension")?;
        exact_keys(extension, &["name", "value"], "KV_CACHE_PROV extension")?;
        let name = text(extension, "name", "KV_CACHE_PROV extension")?;
        let value = extension
            .get("value")
            .cloned()
            .ok_or_else(|| Error::malformed("KV_CACHE_PROV extension `value` is missing"))?;
        extensions.push(KvExtension { name, value });
    }
    Ok(extensions)
}

/// A decoded mode-19 page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedKvProvPage {
    pub descriptor: KvProvDescriptor,
    /// `H("weft/kv/prov/v1", WCB(descriptor))`.
    pub root: Hash32,
    pub bytes: Vec<u8>,
}

fn split_body(body: &[u8]) -> Result<(&[u8], &[u8])> {
    if body.len() < 4 {
        return Err(Error::malformed(
            "KV_CACHE_PROV body is shorter than its 4-byte descriptor length",
        ));
    }
    let descriptor_len = u64::from(u32::from_le_bytes([body[0], body[1], body[2], body[3]]));
    let end = 4u64
        .checked_add(descriptor_len)
        .ok_or_else(|| Error::malformed("KV_CACHE_PROV descriptor length overflows"))?;
    if end > body.len() as u64 {
        return Err(Error::malformed(
            "KV_CACHE_PROV descriptor extends past the frame body",
        ));
    }
    let end = end as usize;
    Ok((&body[4..end], &body[end..]))
}

/// Encode one provenance-bound KV page.
///
/// The payload is built by the frozen mode-12 encoder and then carried
/// verbatim; only the descriptor envelope differs.
pub fn encode_kv_prov_page(
    input: &KvCacheInput,
    elements: &[u8],
    provenance: &KvProvenanceInput,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let page_frame = kv_cache::encode_kv_page(input, elements, limits)?;
    let parsed = Frame::parse(&page_frame, limits)?;
    let page = kv_cache::kv_cache_metadata(&parsed, limits)?;
    let (_, payload) = split_body(parsed.body)?;
    let payload = payload.to_vec();
    let provenance = provenance.canonicalize()?;
    let descriptor = KvProvDescriptor::from_page(page, provenance, elements)?;
    let descriptor_bytes = descriptor.canonical_bytes(limits)?;
    let descriptor_len = u32::try_from(descriptor_bytes.len())
        .map_err(|_| Error::limit_exceeded("KV_CACHE_PROV descriptor does not fit u32"))?;
    let capacity = 4usize
        .checked_add(descriptor_bytes.len())
        .and_then(|value| value.checked_add(payload.len()))
        .ok_or_else(|| Error::limit_exceeded("KV_CACHE_PROV body length overflow"))?;
    let mut body = Vec::with_capacity(capacity);
    body.extend_from_slice(&descriptor_len.to_le_bytes());
    body.extend_from_slice(&descriptor_bytes);
    body.extend_from_slice(&payload);
    let parts = FrameParts {
        format_major: crate::frame::FORMAT_MAJOR,
        format_minor: crate::frame::FORMAT_MINOR,
        domain: DOMAIN_PAYLOAD,
        mode: Mode::KvCacheProv,
        flags: 0,
        level: 0,
        decoded_len: descriptor.page.page_bytes,
        content_hash: descriptor.payload_hash,
        dependencies: &[],
        chunks: &[],
        body: &body,
    };
    encode_frame(&parts, limits)
}

/// Decode and fully verify one provenance-bound KV page.
///
/// The descriptor is validated, the frozen page descriptor is validated as in
/// mode 12, every derived root is recomputed, `payload_hash` must equal the
/// frame `content_hash`, and the reconstructed bytes must hash to
/// `payload_hash`. Any inconsistency is an error and no bytes are returned.
pub fn decode_kv_prov_page(frame: &Frame<'_>, limits: &Limits) -> Result<DecodedKvProvPage> {
    if frame.header.mode != Mode::KvCacheProv {
        return Err(Error::malformed(format!(
            "expected a KV_CACHE_PROV frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor_bytes, payload) = split_body(frame.body)?;
    let descriptor = KvProvDescriptor::from_canonical_bytes(descriptor_bytes, limits)?;
    let page_bytes = descriptor.page.page_bytes;
    if descriptor.page.payload_len != payload.len() as u64 {
        return Err(Error::malformed(format!(
            "KV_CACHE_PROV payload is {} bytes but page.payload_len declares {}",
            payload.len(),
            descriptor.page.payload_len
        )));
    }
    if frame.header.decoded_len != page_bytes {
        return Err(Error::corrupt(format!(
            "KV_CACHE_PROV decoded_len is {} but the page descriptor requires {page_bytes}",
            frame.header.decoded_len
        )));
    }
    if frame.header.content_hash != descriptor.payload_hash {
        return Err(Error::corrupt(
            "KV_CACHE_PROV payload_hash does not match the frame content_hash",
        ));
    }
    let bytes = kv_cache::decode_planes(descriptor.page.dtype, payload, page_bytes)?;
    if bytes.len() as u64 != page_bytes {
        return Err(Error::corrupt(
            "KV_CACHE_PROV reconstructed page length is inconsistent",
        ));
    }
    if hash_domain(DOMAIN_CHUNK, &bytes) != descriptor.payload_hash {
        return Err(Error::corrupt(
            "KV_CACHE_PROV payload hash mismatch against the reconstructed page",
        ));
    }
    let root = descriptor.root(limits)?;
    Ok(DecodedKvProvPage {
        descriptor,
        root,
        bytes,
    })
}

/// Parse and validate a mode-19 descriptor without decoding the payload.
pub fn kv_prov_metadata(frame: &Frame<'_>, limits: &Limits) -> Result<KvProvDescriptor> {
    if frame.header.mode != Mode::KvCacheProv {
        return Err(Error::malformed(format!(
            "expected a KV_CACHE_PROV frame, got mode {}",
            frame.header.mode.as_str()
        )));
    }
    let (descriptor_bytes, _) = split_body(frame.body)?;
    KvProvDescriptor::from_canonical_bytes(descriptor_bytes, limits)
}
