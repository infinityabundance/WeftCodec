//! Authenticated capability envelopes (`WEFT-CAPABILITIES.md` §9).
//!
//! A capability envelope is a canonical WCB record (`weft.capability.envelope.v1`)
//! that is signed by its grantor with Ed25519. The envelope is **public**: it
//! contains no secret. Only the grantor's private key is secret.
//!
//! Identities:
//!
//! ```text
//! EnvelopeId = H("weft/capability/envelope/v1", WCB(envelope))
//! sign_input = H("weft/capability/sign/v1",     WCB(envelope))
//! BlobId     = H("weft/capability/blob/v1",     WCB({envelope, signature}))
//! ```
//!
//! `verify_envelope` is deliberately offline: the caller supplies the
//! revocation set and the wall clock. There is no network lookup and no
//! ambient trust. Revocation coordination is explicitly external
//! (`WEFT-CAPABILITIES.md` §9.4).

use std::collections::BTreeMap;

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};

use crate::context::{Budget, CapabilityScope};
use crate::error::{Error, ErrorClass, Result};
use crate::hash::{hash_domain, Hash32};
use crate::limits::Limits;
use crate::wcb::{self, Value};

/// Canonical schema string of an envelope.
pub const ENVELOPE_SCHEMA: &str = "weft.capability.envelope.v1";
/// Domain separation for `EnvelopeId`.
pub const DOMAIN_ENVELOPE: &str = "weft/capability/envelope/v1";
/// Domain separation for the value a grantor signs.
pub const DOMAIN_SIGN: &str = "weft/capability/sign/v1";
/// Domain separation for the stored `{envelope, signature}` blob.
pub const DOMAIN_BLOB: &str = "weft/capability/blob/v1";
/// Ed25519 signature length.
pub const SIGNATURE_LEN: usize = 64;
/// Maximum bytes of a tenant label.
pub const MAX_TENANT_BYTES: usize = 256;

fn envelope_limits() -> Limits {
    Limits {
        max_structured_items: 65_536,
        max_structured_depth: 16,
        max_string_bytes: 64 * 1024,
        ..Limits::default()
    }
}

fn encode_value(value: &Value) -> Result<Vec<u8>> {
    wcb::encode(value, &envelope_limits())
}

fn decode_value(bytes: &[u8]) -> Result<Value> {
    wcb::decode(bytes, &envelope_limits())
}

/// A typed envelope rejection.
///
/// These are the interoperable names. `class()` maps each one to a WeftCodec
/// [`ErrorClass`] so codec callers keep a single error vocabulary; WeftFS folds
/// every rejection into a non-disclosing authorization denial at enforcement
/// points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rejection {
    /// Byte-level structure is not a valid WCB document.
    Malformed,
    /// The document is well formed but not canonical (or violates a field rule).
    NonCanonical,
    /// The schema string is not `weft.capability.envelope.v1`.
    UnknownSchema,
    /// The Ed25519 signature does not verify under `grantor`.
    BadSignature,
    /// The envelope is not bound to the expected grantee.
    WrongGrantee,
    /// `now_ns` is past `expires_at_ns`.
    Expired,
    /// `revocation_id` is present in the caller-supplied revocation set.
    Revoked,
}

impl Rejection {
    pub fn as_str(self) -> &'static str {
        match self {
            Rejection::Malformed => "Malformed",
            Rejection::NonCanonical => "NonCanonical",
            Rejection::UnknownSchema => "UnknownSchema",
            Rejection::BadSignature => "BadSignature",
            Rejection::WrongGrantee => "WrongGrantee",
            Rejection::Expired => "Expired",
            Rejection::Revoked => "Revoked",
        }
    }

    pub fn class(self) -> ErrorClass {
        match self {
            Rejection::Malformed => ErrorClass::Malformed,
            Rejection::NonCanonical => ErrorClass::NonCanonical,
            Rejection::UnknownSchema => ErrorClass::UnknownSchema,
            Rejection::BadSignature => ErrorClass::Corrupt,
            Rejection::WrongGrantee => ErrorClass::DependencyDenied,
            Rejection::Expired => ErrorClass::DependencyDenied,
            Rejection::Revoked => ErrorClass::DependencyDenied,
        }
    }
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An envelope rejection with a human-readable message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvelopeError {
    rejection: Rejection,
    message: String,
}

impl EnvelopeError {
    pub fn new(rejection: Rejection, message: impl Into<String>) -> Self {
        EnvelopeError {
            rejection,
            message: message.into(),
        }
    }

    pub fn rejection(&self) -> Rejection {
        self.rejection
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn class(&self) -> ErrorClass {
        self.rejection.class()
    }
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.rejection, self.message)
    }
}

impl std::error::Error for EnvelopeError {}

impl From<Error> for EnvelopeError {
    fn from(err: Error) -> Self {
        let rejection = match err.class() {
            ErrorClass::NonCanonical => Rejection::NonCanonical,
            ErrorClass::UnknownSchema => Rejection::UnknownSchema,
            _ => Rejection::Malformed,
        };
        EnvelopeError::new(rejection, err.message())
    }
}

/// Result alias for envelope verification.
pub type EnvelopeResult<T> = std::result::Result<T, EnvelopeError>;

/// An authenticated capability grant.
///
/// Every field is public. The `grantor` key signs the canonical record; the
/// `grantee` is a public key or a stable principal hash. `tenant` names the
/// isolation domain the grant belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityEnvelope {
    /// Ed25519 public key (32 bytes) of the signer.
    pub grantor: Hash32,
    /// Grantee public key or stable principal hash (32 bytes).
    pub grantee: Hash32,
    /// Tenant/isolation-domain label.
    pub tenant: String,
    /// Sorted, unique granted content/context roots. May be empty.
    pub roots: Vec<Hash32>,
    /// Sorted, unique granted namespaces. May be empty.
    pub namespaces: Vec<String>,
    /// Sorted, unique granted relation predicates. May be empty.
    pub predicates: Vec<String>,
    /// Inclusive expiry in UTC nanoseconds since the Unix epoch.
    pub expires_at_ns: Option<i64>,
    /// Optional additional budget ceiling.
    pub budget: Option<Budget>,
    /// Revocation identity (checked against a caller-supplied set).
    pub revocation_id: Hash32,
    /// Issue time in UTC nanoseconds since the Unix epoch.
    pub issued_at_ns: i64,
    /// Caller-chosen nonce, making re-issues of the same grant distinct.
    pub nonce: Hash32,
}

impl CapabilityEnvelope {
    /// Build and validate a canonical envelope.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grantor: Hash32,
        grantee: Hash32,
        tenant: impl Into<String>,
        roots: Vec<Hash32>,
        namespaces: Vec<String>,
        predicates: Vec<String>,
        expires_at_ns: Option<i64>,
        budget: Option<Budget>,
        revocation_id: Hash32,
        issued_at_ns: i64,
        nonce: Hash32,
    ) -> Result<Self> {
        let envelope = CapabilityEnvelope {
            grantor,
            grantee,
            tenant: tenant.into(),
            roots,
            namespaces,
            predicates,
            expires_at_ns,
            budget,
            revocation_id,
            issued_at_ns,
            nonce,
        };
        envelope.validate(&envelope_limits())?;
        Ok(envelope)
    }

    /// Strict validation of every field and cross-field rule.
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        if self.grantor == Hash32::ZERO {
            return Err(Error::non_canonical(
                "capability envelope grantor must not be zero",
            ));
        }
        if self.grantee == Hash32::ZERO {
            return Err(Error::non_canonical(
                "capability envelope grantee must not be zero",
            ));
        }
        if self.revocation_id == Hash32::ZERO {
            return Err(Error::non_canonical(
                "capability envelope revocation_id must not be zero",
            ));
        }
        if self.nonce == Hash32::ZERO {
            return Err(Error::non_canonical(
                "capability envelope nonce must not be zero",
            ));
        }
        if self.tenant.is_empty() || self.tenant.len() > MAX_TENANT_BYTES {
            return Err(Error::non_canonical(format!(
                "capability envelope tenant must be 1..={MAX_TENANT_BYTES} bytes"
            )));
        }
        // Reuse the frozen capability-scope rules for grants, budgets, and
        // sorted/unique arrays. `new` normalizes, so a mismatch in any list
        // means the input was not already canonical.
        let scope = CapabilityScope::new(
            self.tenant.clone(),
            self.namespaces.clone(),
            self.roots.clone(),
            self.predicates.clone(),
            self.budget.clone(),
            self.expires_at_ns,
        )?;
        if scope.namespaces != self.namespaces
            || scope.roots != self.roots
            || scope.predicates != self.predicates
        {
            return Err(Error::non_canonical(
                "capability envelope grant lists must be sorted and unique",
            ));
        }
        if let Some(expires) = self.expires_at_ns {
            if expires < self.issued_at_ns {
                return Err(Error::non_canonical(
                    "capability envelope expires_at_ns precedes issued_at_ns",
                ));
            }
        }
        // Bound the encoded record itself.
        let bytes = self.encode_with_limits(limits)?;
        if bytes.len() as u64 > limits.max_string_bytes {
            return Err(Error::limit_exceeded(format!(
                "capability envelope is {} bytes, above the {}-byte bound",
                bytes.len(),
                limits.max_string_bytes
            )));
        }
        Ok(())
    }

    /// Canonical WCB value. Every key is always present.
    pub fn to_wcb_value(&self) -> Value {
        let roots = Value::Array(
            self.roots
                .iter()
                .map(|hash| Value::Bytes(hash.as_bytes().to_vec()))
                .collect(),
        );
        let namespaces = Value::Array(
            self.namespaces
                .iter()
                .map(|item| Value::Str(item.clone()))
                .collect(),
        );
        let predicates = Value::Array(
            self.predicates
                .iter()
                .map(|item| Value::Str(item.clone()))
                .collect(),
        );
        let expires = match self.expires_at_ns {
            Some(ns) => Value::Timestamp(ns),
            None => Value::Null,
        };
        let budget = match &self.budget {
            Some(budget) => budget.to_wcb_value(),
            None => Value::Null,
        };
        Value::map([
            ("schema", Value::Str(ENVELOPE_SCHEMA.to_owned())),
            ("grantor", Value::Bytes(self.grantor.as_bytes().to_vec())),
            ("grantee", Value::Bytes(self.grantee.as_bytes().to_vec())),
            ("tenant", Value::Str(self.tenant.clone())),
            ("roots", roots),
            ("namespaces", namespaces),
            ("predicates", predicates),
            ("expires_at_ns", expires),
            ("budget", budget),
            (
                "revocation_id",
                Value::Bytes(self.revocation_id.as_bytes().to_vec()),
            ),
            ("issued_at_ns", Value::Timestamp(self.issued_at_ns)),
            ("nonce", Value::Bytes(self.nonce.as_bytes().to_vec())),
        ])
    }

    /// Canonical bytes with default envelope limits.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.canonical_bytes_with_limits(&envelope_limits())
    }

    pub fn canonical_bytes_with_limits(&self, limits: &Limits) -> Result<Vec<u8>> {
        self.validate(limits)?;
        self.encode_with_limits(limits)
    }

    fn encode_with_limits(&self, limits: &Limits) -> Result<Vec<u8>> {
        wcb::encode(&self.to_wcb_value(), limits)
    }

    /// `EnvelopeId = H("weft/capability/envelope/v1", WCB(envelope))`.
    pub fn id(&self) -> Result<Hash32> {
        Ok(hash_domain(DOMAIN_ENVELOPE, &self.canonical_bytes()?))
    }

    /// The bytes a grantor signs.
    pub fn sign_input(&self) -> Result<Hash32> {
        Ok(hash_domain(DOMAIN_SIGN, &self.canonical_bytes()?))
    }

    /// Decode strict canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8], limits: &Limits) -> Result<Self> {
        let value = wcb::decode(bytes, limits)?;
        let map = match &value {
            Value::Map(map) => map,
            _ => {
                return Err(Error::non_canonical(
                    "capability envelope must be a WCB map",
                ))
            }
        };
        const KEYS: [&str; 12] = [
            "schema",
            "grantor",
            "grantee",
            "tenant",
            "roots",
            "namespaces",
            "predicates",
            "expires_at_ns",
            "budget",
            "revocation_id",
            "issued_at_ns",
            "nonce",
        ];
        for key in map.keys() {
            if !KEYS.contains(&key.as_str()) {
                return Err(Error::non_canonical(format!(
                    "unknown capability envelope key `{key}`"
                )));
            }
        }
        for key in KEYS {
            if !map.contains_key(key) {
                return Err(Error::malformed(format!(
                    "capability envelope is missing `{key}`"
                )));
            }
        }
        let schema = match map.get("schema") {
            Some(Value::Str(text)) => text,
            _ => {
                return Err(Error::non_canonical(
                    "capability envelope schema must be text",
                ))
            }
        };
        if schema != ENVELOPE_SCHEMA {
            return Err(Error::unknown_schema(format!(
                "unknown capability envelope schema `{schema}`"
            )));
        }
        let grantor = required_hash(map, "grantor")?;
        let grantee = required_hash(map, "grantee")?;
        let tenant = match map.get("tenant") {
            Some(Value::Str(text)) => text.clone(),
            _ => {
                return Err(Error::non_canonical(
                    "capability envelope tenant must be text",
                ))
            }
        };
        let roots = hash_array(map, "roots")?;
        let namespaces = text_array(map, "namespaces")?;
        let predicates = text_array(map, "predicates")?;
        let expires_at_ns = match map.get("expires_at_ns") {
            Some(Value::Null) => None,
            Some(Value::Timestamp(ns)) => Some(*ns),
            Some(Value::I64(ns)) => Some(*ns),
            _ => {
                return Err(Error::non_canonical(
                    "capability envelope expires_at_ns must be a timestamp or null",
                ))
            }
        };
        let budget = match map.get("budget") {
            Some(Value::Null) => None,
            Some(value) => Some(Budget::from_wcb_value(value)?),
            None => return Err(Error::malformed("capability envelope is missing `budget`")),
        };
        let revocation_id = required_hash(map, "revocation_id")?;
        let issued_at_ns = match map.get("issued_at_ns") {
            Some(Value::Timestamp(ns)) => *ns,
            Some(Value::I64(ns)) => *ns,
            _ => {
                return Err(Error::non_canonical(
                    "capability envelope issued_at_ns must be a timestamp",
                ))
            }
        };
        let nonce = required_hash(map, "nonce")?;
        let envelope = CapabilityEnvelope {
            grantor,
            grantee,
            tenant,
            roots,
            namespaces,
            predicates,
            expires_at_ns,
            budget,
            revocation_id,
            issued_at_ns,
            nonce,
        };
        envelope.validate(limits)?;
        let reencoded = envelope.encode_with_limits(limits)?;
        if reencoded != bytes {
            return Err(Error::non_canonical(
                "capability envelope bytes are not in canonical form",
            ));
        }
        Ok(envelope)
    }

    /// The envelope's grant as a frozen capability scope (used by enforcement).
    pub fn capability_scope(&self) -> Result<CapabilityScope> {
        CapabilityScope::new(
            self.tenant.clone(),
            self.namespaces.clone(),
            self.roots.clone(),
            self.predicates.clone(),
            self.budget.clone(),
            self.expires_at_ns,
        )
    }

    /// Parse a JSON envelope document (CLI convenience; never on the wire).
    pub fn from_json_str(text: &str) -> Result<Self> {
        let json: serde_json::Value = serde_json::from_str(text)
            .map_err(|err| Error::parse(format!("bad capability envelope JSON: {err}")))?;
        let object = json
            .as_object()
            .ok_or_else(|| Error::parse("capability envelope JSON must be an object"))?;
        for key in object.keys() {
            match key.as_str() {
                "schema" | "grantor" | "grantee" | "tenant" | "roots" | "namespaces"
                | "predicates" | "expires_at_ns" | "budget" | "revocation_id" | "issued_at_ns"
                | "nonce" => {}
                other => {
                    return Err(Error::non_canonical(format!(
                        "unknown capability envelope key `{other}`"
                    )))
                }
            }
        }
        let schema = object
            .get("schema")
            .and_then(|value| value.as_str())
            .ok_or_else(|| Error::parse("capability envelope `schema` must be text"))?;
        if schema != ENVELOPE_SCHEMA {
            return Err(Error::unknown_schema(format!(
                "unknown capability envelope schema `{schema}`"
            )));
        }
        let grantor = json_hash(object.get("grantor"), "grantor")?;
        let grantee = json_hash(object.get("grantee"), "grantee")?;
        let tenant = object
            .get("tenant")
            .and_then(|value| value.as_str())
            .ok_or_else(|| Error::parse("capability envelope `tenant` must be text"))?
            .to_owned();
        let roots = match object.get("roots") {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(value) => {
                let items = value
                    .as_array()
                    .ok_or_else(|| Error::parse("capability envelope `roots` must be an array"))?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(json_hash(Some(item), "roots[]")?);
                }
                out
            }
        };
        let namespaces = json_text_array(object.get("namespaces"), "namespaces")?;
        let predicates = json_text_array(object.get("predicates"), "predicates")?;
        let expires_at_ns = match object.get("expires_at_ns") {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => Some(value.as_i64().ok_or_else(|| {
                Error::parse("capability envelope `expires_at_ns` must be an integer")
            })?),
        };
        let budget = match object.get("budget") {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => Some(json_budget(value)?),
        };
        let revocation_id = json_hash(object.get("revocation_id"), "revocation_id")?;
        let issued_at_ns = object
            .get("issued_at_ns")
            .and_then(|value| value.as_i64())
            .ok_or_else(|| Error::parse("capability envelope `issued_at_ns` must be an integer"))?;
        let nonce = json_hash(object.get("nonce"), "nonce")?;
        CapabilityEnvelope::new(
            grantor,
            grantee,
            tenant,
            roots,
            namespaces,
            predicates,
            expires_at_ns,
            budget,
            revocation_id,
            issued_at_ns,
            nonce,
        )
    }

    /// JSON representation (CLI convenience).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": ENVELOPE_SCHEMA,
            "grantor": self.grantor.to_hex(),
            "grantee": self.grantee.to_hex(),
            "tenant": self.tenant,
            "roots": self.roots.iter().map(|hash| hash.to_hex()).collect::<Vec<_>>(),
            "namespaces": self.namespaces,
            "predicates": self.predicates,
            "expires_at_ns": self.expires_at_ns,
            "budget": self.budget.as_ref().map(|budget| budget.to_json()),
            "revocation_id": self.revocation_id.to_hex(),
            "issued_at_ns": self.issued_at_ns,
            "nonce": self.nonce.to_hex(),
        })
    }
}

fn required_hash(map: &BTreeMap<String, Value>, key: &str) -> Result<Hash32> {
    match map.get(key) {
        Some(Value::Bytes(bytes)) => Hash32::from_slice(bytes),
        _ => Err(Error::non_canonical(format!(
            "capability envelope `{key}` must be bytes(32)"
        ))),
    }
}

fn hash_array(map: &BTreeMap<String, Value>, key: &str) -> Result<Vec<Hash32>> {
    match map.get(key) {
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::Bytes(bytes) => out.push(Hash32::from_slice(bytes)?),
                    _ => {
                        return Err(Error::non_canonical(format!(
                            "capability envelope `{key}` entries must be bytes(32)"
                        )))
                    }
                }
            }
            Ok(out)
        }
        _ => Err(Error::non_canonical(format!(
            "capability envelope `{key}` must be an array"
        ))),
    }
}

fn text_array(map: &BTreeMap<String, Value>, key: &str) -> Result<Vec<String>> {
    match map.get(key) {
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::Str(text) => out.push(text.clone()),
                    _ => {
                        return Err(Error::non_canonical(format!(
                            "capability envelope `{key}` entries must be text"
                        )))
                    }
                }
            }
            Ok(out)
        }
        _ => Err(Error::non_canonical(format!(
            "capability envelope `{key}` must be an array"
        ))),
    }
}

fn json_hash(value: Option<&serde_json::Value>, what: &str) -> Result<Hash32> {
    let text = value
        .and_then(|value| value.as_str())
        .ok_or_else(|| Error::parse(format!("capability envelope `{what}` must be hex text")))?;
    Hash32::from_hex(text)
}

fn json_text_array(value: Option<&serde_json::Value>, what: &str) -> Result<Vec<String>> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(value) => {
            let items = value.as_array().ok_or_else(|| {
                Error::parse(format!("capability envelope `{what}` must be an array"))
            })?;
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(
                    item.as_str()
                        .ok_or_else(|| {
                            Error::parse(format!(
                                "capability envelope `{what}` entries must be text"
                            ))
                        })?
                        .to_owned(),
                );
            }
            Ok(out)
        }
    }
}

fn json_budget(value: &serde_json::Value) -> Result<Budget> {
    let map = value
        .as_object()
        .ok_or_else(|| Error::parse("capability envelope `budget` must be an object"))?;
    let mut budget = Budget::new();
    for (key, value) in map {
        let number = value
            .as_u64()
            .ok_or_else(|| Error::parse(format!("budget `{key}` must be u64")))?;
        match key.as_str() {
            "max_results" => budget.max_results = Some(number),
            "max_bytes" => budget.max_bytes = Some(number),
            "max_depth" => budget.max_depth = Some(number),
            "max_work" => budget.max_work = Some(number),
            other => {
                return Err(Error::non_canonical(format!(
                    "unknown budget key `{other}`"
                )))
            }
        }
    }
    budget.validate()?;
    Ok(budget)
}

/// A signed envelope as stored: canonical `{envelope, signature}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityBlob {
    pub envelope: CapabilityEnvelope,
    /// Raw Ed25519 signature over `envelope.sign_input()`.
    pub signature: Vec<u8>,
}

impl CapabilityBlob {
    pub fn new(envelope: CapabilityEnvelope, signature: Vec<u8>) -> Result<Self> {
        if signature.len() != SIGNATURE_LEN {
            return Err(Error::malformed(format!(
                "capability signature must be {SIGNATURE_LEN} bytes, got {}",
                signature.len()
            )));
        }
        Ok(CapabilityBlob {
            envelope,
            signature,
        })
    }

    /// Canonical WCB bytes `{envelope: bytes, signature: bytes}`.
    pub fn encode(&self) -> Result<Vec<u8>> {
        if self.signature.len() != SIGNATURE_LEN {
            return Err(Error::malformed(format!(
                "capability signature must be {SIGNATURE_LEN} bytes, got {}",
                self.signature.len()
            )));
        }
        let envelope = self.envelope.canonical_bytes()?;
        encode_value(&Value::map([
            ("envelope", Value::Bytes(envelope)),
            ("signature", Value::Bytes(self.signature.clone())),
        ]))
    }

    pub fn id(&self) -> Result<Hash32> {
        Ok(hash_domain(DOMAIN_BLOB, &self.encode()?))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let value = decode_value(bytes)?;
        let map = match &value {
            Value::Map(map) => map,
            _ => return Err(Error::non_canonical("capability blob must be a WCB map")),
        };
        for key in map.keys() {
            if key != "envelope" && key != "signature" {
                return Err(Error::non_canonical(format!(
                    "unknown capability blob key `{key}`"
                )));
            }
        }
        let envelope_bytes = match map.get("envelope") {
            Some(Value::Bytes(bytes)) => bytes,
            _ => {
                return Err(Error::non_canonical(
                    "capability blob `envelope` must be bytes",
                ))
            }
        };
        let signature = match map.get("signature") {
            Some(Value::Bytes(bytes)) => bytes.clone(),
            _ => {
                return Err(Error::non_canonical(
                    "capability blob `signature` must be bytes",
                ))
            }
        };
        if signature.len() != SIGNATURE_LEN {
            return Err(Error::non_canonical(format!(
                "capability signature must be {SIGNATURE_LEN} bytes"
            )));
        }
        let envelope =
            CapabilityEnvelope::from_canonical_bytes(envelope_bytes, &envelope_limits())?;
        let blob = CapabilityBlob {
            envelope,
            signature,
        };
        let reencoded = blob.encode()?;
        if reencoded != bytes {
            return Err(Error::non_canonical(
                "capability blob bytes are not in canonical form",
            ));
        }
        Ok(blob)
    }
}

/// An envelope that passed offline verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedEnvelope {
    pub envelope: CapabilityEnvelope,
    pub envelope_id: Hash32,
    pub blob_id: Hash32,
    pub signature: Vec<u8>,
}

impl VerifiedEnvelope {
    pub fn grantor(&self) -> Hash32 {
        self.envelope.grantor
    }

    pub fn grantee(&self) -> Hash32 {
        self.envelope.grantee
    }

    pub fn tenant(&self) -> &str {
        &self.envelope.tenant
    }

    pub fn capability_scope(&self) -> Result<CapabilityScope> {
        self.envelope.capability_scope()
    }
}

/// Derive the Ed25519 public key for a 32-byte seed.
pub fn public_key_for_seed(seed: &[u8]) -> Result<Hash32> {
    if seed.len() != 32 {
        return Err(Error::malformed(format!(
            "Ed25519 seed must be 32 bytes, got {}",
            seed.len()
        )));
    }
    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(seed);
    Ok(Hash32::from_bytes(
        SigningKey::from_bytes(&key_bytes)
            .verifying_key()
            .to_bytes(),
    ))
}

/// Sign a canonical envelope with a 32-byte Ed25519 seed.
pub fn sign_envelope(envelope: &CapabilityEnvelope, seed: &[u8]) -> Result<CapabilityBlob> {
    if seed.len() != 32 {
        return Err(Error::malformed(format!(
            "Ed25519 seed must be 32 bytes, got {}",
            seed.len()
        )));
    }
    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(seed);
    let signing_key = SigningKey::from_bytes(&key_bytes);
    // The grantor field must be the public key that actually signs. This is a
    // construction-time check, not part of verification.
    let public = signing_key.verifying_key();
    if envelope.grantor.as_bytes() != public.as_bytes() {
        return Err(Error::non_canonical(
            "capability envelope grantor does not match the signing key",
        ));
    }
    let signature = signing_key.sign(envelope.sign_input()?.as_bytes());
    CapabilityBlob::new(envelope.clone(), signature.to_bytes().to_vec())
}

/// Verify a stored blob offline.
///
/// Checks, in normative order: canonical form, Ed25519 signature under
/// `grantor`, optional grantee binding, expiry, and revocation-set membership.
/// It never performs I/O.
pub fn verify_envelope(
    blob: &[u8],
    revocations: &[Hash32],
    now_ns: i64,
    expected_grantee: Option<Hash32>,
) -> EnvelopeResult<VerifiedEnvelope> {
    let blob = CapabilityBlob::decode(blob)?;
    let envelope = &blob.envelope;
    let verifying_key = VerifyingKey::from_bytes(envelope.grantor.as_bytes()).map_err(|err| {
        EnvelopeError::new(
            Rejection::BadSignature,
            format!("invalid grantor key: {err}"),
        )
    })?;
    let signature = Signature::from_slice(&blob.signature).map_err(|err| {
        EnvelopeError::new(
            Rejection::BadSignature,
            format!("invalid signature encoding: {err}"),
        )
    })?;
    let input = envelope
        .sign_input()
        .map_err(|err| EnvelopeError::new(Rejection::NonCanonical, err.message()))?;
    verifying_key
        .verify_strict(&input.as_bytes()[..], &signature)
        .map_err(|_| {
            EnvelopeError::new(
                Rejection::BadSignature,
                "capability envelope signature does not verify under grantor",
            )
        })?;
    if let Some(expected) = expected_grantee {
        if expected != envelope.grantee {
            return Err(EnvelopeError::new(
                Rejection::WrongGrantee,
                "capability envelope is not bound to the expected grantee",
            ));
        }
    }
    if let Some(expires) = envelope.expires_at_ns {
        if now_ns > expires {
            return Err(EnvelopeError::new(
                Rejection::Expired,
                "capability envelope has expired",
            ));
        }
    }
    if revocations.iter().any(|id| *id == envelope.revocation_id) {
        return Err(EnvelopeError::new(
            Rejection::Revoked,
            "capability envelope revocation id is revoked",
        ));
    }
    let envelope_id = envelope
        .id()
        .map_err(|err| EnvelopeError::new(Rejection::NonCanonical, err.message()))?;
    let blob_id = blob
        .id()
        .map_err(|err| EnvelopeError::new(Rejection::NonCanonical, err.message()))?;
    Ok(VerifiedEnvelope {
        envelope: blob.envelope,
        envelope_id,
        blob_id,
        signature: blob.signature,
    })
}

/// Verify without an expected grantee.
pub fn verify_envelope_for_tenant(
    blob: &[u8],
    revocations: &[Hash32],
    now_ns: i64,
) -> EnvelopeResult<VerifiedEnvelope> {
    verify_envelope(blob, revocations, now_ns, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn key(byte: u8) -> Hash32 {
        let sk = SigningKey::from_bytes(&seed(byte));
        Hash32::from_bytes(sk.verifying_key().to_bytes())
    }

    fn sample() -> CapabilityEnvelope {
        CapabilityEnvelope::new(
            key(7),
            Hash32::from_bytes([9u8; 32]),
            "tenant-a",
            vec![Hash32::from_bytes([0x11; 32])],
            vec!["default".to_owned()],
            vec!["derived_from".to_owned()],
            Some(2_000_000_000_000_000_000),
            Some(Budget {
                max_results: Some(10),
                max_bytes: None,
                max_depth: Some(2),
                max_work: Some(100),
            }),
            Hash32::from_bytes([0x22; 32]),
            1_700_000_000_000_000_000,
            Hash32::from_bytes([0x33; 32]),
        )
        .unwrap()
    }

    #[test]
    fn canonical_round_trip() {
        let envelope = sample();
        let bytes = envelope.canonical_bytes().unwrap();
        let decoded = CapabilityEnvelope::from_canonical_bytes(&bytes, &envelope_limits()).unwrap();
        assert_eq!(decoded, envelope);
        assert_eq!(decoded.id().unwrap(), envelope.id().unwrap());
        assert_eq!(envelope.id().unwrap(), hash_domain(DOMAIN_ENVELOPE, &bytes));
        assert_eq!(
            envelope.sign_input().unwrap(),
            hash_domain(DOMAIN_SIGN, &bytes)
        );
    }

    #[test]
    fn sign_verify_round_trip() {
        let envelope = sample();
        let blob = sign_envelope(&envelope, &seed(7)).unwrap();
        let bytes = blob.encode().unwrap();
        let verified = verify_envelope(
            &bytes,
            &[],
            1_700_000_000_000_000_000,
            Some(envelope.grantee),
        )
        .unwrap();
        assert_eq!(verified.envelope, envelope);
        assert_eq!(verified.envelope_id, envelope.id().unwrap());
        assert_eq!(verified.blob_id, blob.id().unwrap());
    }

    #[test]
    fn reject_matrix() {
        let envelope = sample();
        let blob = sign_envelope(&envelope, &seed(7)).unwrap();
        let bytes = blob.encode().unwrap();

        // Bad signature: flip a signature byte.
        let mut bad = blob.clone();
        bad.signature[0] ^= 0xff;
        let err = verify_envelope(&bad.encode().unwrap(), &[], 0, None).unwrap_err();
        assert_eq!(err.rejection(), Rejection::BadSignature);

        // Wrong grantee.
        let err =
            verify_envelope(&bytes, &[], 0, Some(Hash32::from_bytes([0xAA; 32]))).unwrap_err();
        assert_eq!(err.rejection(), Rejection::WrongGrantee);

        // Expired.
        let err = verify_envelope(&bytes, &[], 2_000_000_000_000_000_001, None).unwrap_err();
        assert_eq!(err.rejection(), Rejection::Expired);

        // Revoked.
        let err = verify_envelope(&bytes, &[envelope.revocation_id], 0, None).unwrap_err();
        assert_eq!(err.rejection(), Rejection::Revoked);

        // Non-canonical: tamper with the outer WCB.
        let mut noncanonical = bytes.clone();
        noncanonical.push(0);
        let err = verify_envelope(&noncanonical, &[], 0, None).unwrap_err();
        assert_eq!(err.rejection(), Rejection::NonCanonical);
    }

    #[test]
    fn rejects_unsorted_grants() {
        let err = CapabilityEnvelope::new(
            key(7),
            Hash32::from_bytes([9u8; 32]),
            "tenant-a",
            Vec::new(),
            vec!["b".to_owned(), "a".to_owned()],
            Vec::new(),
            None,
            None,
            Hash32::from_bytes([0x22; 32]),
            0,
            Hash32::from_bytes([0x33; 32]),
        )
        .unwrap_err();
        assert_eq!(err.class(), ErrorClass::NonCanonical);
    }
}
