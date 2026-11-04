//! Context Root, causal closure, capability scope, and `weft://` addressing.
//!
//! Normative sources:
//! `spec/WEFT-CONTEXT-ROOT.md`, `spec/WEFT-CAUSAL.md`, and
//! `spec/WEFT-CAPABILITIES.md`.
//!
//! Canonical record bytes are WCB. Identity is domain-separated:
//!
//! ```text
//! ContextRootId     = H("weft/context-root/v1", WCB(context-root record))
//! CausalClosureId   = H("weft/causal/v1",       WCB(causal record))
//! CapabilityHash    = H("weft/capability/v1",   WCB(capability record))
//! ```
//!
//! This module is pure data: it does not resolve selectors, read snapshots, or
//! authorize principals. Resolution and enforcement belong to the store.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::hash::{hash_domain, Hash32, DOMAIN_CAPABILITY, DOMAIN_CAUSAL, DOMAIN_CONTEXT_ROOT};
use crate::limits::Limits;
use crate::selector::{self, Selector};
use crate::wcb::{self, Value};

// The schema strings, domains, policy names, and text limits below are
// generated from `spec/schema/identity-v1.schema` (see `scripts/gen-types.sh`).
// Editing these values by hand is a defect; the drift court compares the
// canonical source against both generated files.
pub use crate::generated::{
    CAPABILITY_SCHEMA, CAUSAL_SCHEMA, CONTEXT_ROOT_SCHEMA, DEFAULT_RESOLVER, MAX_NAMESPACE_BYTES,
    MAX_RESOLVER_BYTES, ORDERING_SELECTOR_THEN_OBJECT_ID, URI_PREFIX,
};

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

/// Resource budget (WEFT-CONTEXT-ROOT §4).
///
/// At least one bound must be present. Every stated bound is a ceiling; it is
/// never silently ignored and never exceeded.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Budget {
    pub max_results: Option<u64>,
    pub max_bytes: Option<u64>,
    pub max_depth: Option<u64>,
    pub max_work: Option<u64>,
}

impl Budget {
    /// Empty budget (invalid until at least one bound is set).
    pub fn new() -> Self {
        Budget::default()
    }

    /// Parse `max_results=..,max_bytes=..,max_depth=..,max_work=..`.
    ///
    /// Values are canonical decimal (`0` or `[1-9][0-9]*`); zero and duplicate
    /// keys are rejected.
    pub fn parse_spec(spec: &str) -> Result<Self> {
        if spec.is_empty() {
            return Err(Error::parse("budget specification must not be empty"));
        }
        let mut budget = Budget::new();
        for entry in spec.split(',') {
            if entry.is_empty() {
                return Err(Error::parse("empty budget entry"));
            }
            let (name, value) = entry
                .split_once('=')
                .ok_or_else(|| Error::parse(format!("budget entry `{entry}` is missing `=`")))?;
            if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return Err(Error::parse(format!(
                    "budget value for `{name}` must be decimal digits"
                )));
            }
            if value.len() > 1 && value.starts_with('0') {
                return Err(Error::parse(format!(
                    "budget value for `{name}` is not canonical decimal"
                )));
            }
            let parsed: u64 = value
                .parse()
                .map_err(|_| Error::parse(format!("budget value for `{name}` overflows u64")))?;
            let slot = match name {
                "max_results" => &mut budget.max_results,
                "max_bytes" => &mut budget.max_bytes,
                "max_depth" => &mut budget.max_depth,
                "max_work" => &mut budget.max_work,
                _ => {
                    return Err(Error::parse(format!(
                        "unknown budget key `{name}` (expected max_results, max_bytes, max_depth, max_work)"
                    )))
                }
            };
            if slot.is_some() {
                return Err(Error::parse(format!("duplicate budget key `{name}`")));
            }
            *slot = Some(parsed);
        }
        budget.validate().map_err(|err| err.context("budget"))?;
        Ok(budget)
    }

    /// True when no bound is present.
    pub fn is_empty(&self) -> bool {
        self.max_results.is_none()
            && self.max_bytes.is_none()
            && self.max_depth.is_none()
            && self.max_work.is_none()
    }

    /// A budget is valid when non-empty and every stated bound is ≥ 1.
    pub fn validate(&self) -> Result<()> {
        if self.is_empty() {
            return Err(Error::non_canonical("budget must state at least one bound"));
        }
        for (name, value) in [
            ("max_results", self.max_results),
            ("max_bytes", self.max_bytes),
            ("max_depth", self.max_depth),
            ("max_work", self.max_work),
        ] {
            if let Some(value) = value {
                if value == 0 {
                    return Err(Error::non_canonical(format!(
                        "budget `{name}` must be at least 1"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Canonical WCB map value.
    pub fn to_wcb_value(&self) -> Value {
        let mut map = BTreeMap::new();
        if let Some(value) = self.max_results {
            map.insert("max_results".to_owned(), Value::U64(value));
        }
        if let Some(value) = self.max_bytes {
            map.insert("max_bytes".to_owned(), Value::U64(value));
        }
        if let Some(value) = self.max_depth {
            map.insert("max_depth".to_owned(), Value::U64(value));
        }
        if let Some(value) = self.max_work {
            map.insert("max_work".to_owned(), Value::U64(value));
        }
        Value::Map(map)
    }

    /// Decode a budget map, rejecting unknown/duplicate/zero entries.
    pub fn from_wcb_value(value: &Value) -> Result<Self> {
        let map = as_map(value, "budget")?;
        let mut budget = Budget::new();
        for (key, value) in map {
            let number = match value {
                Value::U64(number) => *number,
                _ => {
                    return Err(Error::non_canonical(format!(
                        "budget `{key}` must be a u64"
                    )))
                }
            };
            let slot = match key.as_str() {
                "max_results" => &mut budget.max_results,
                "max_bytes" => &mut budget.max_bytes,
                "max_depth" => &mut budget.max_depth,
                "max_work" => &mut budget.max_work,
                other => {
                    return Err(Error::non_canonical(format!(
                        "unknown budget key `{other}`"
                    )))
                }
            };
            *slot = Some(number);
        }
        budget.validate()?;
        Ok(budget)
    }

    /// Canonical descriptor query fragment (no leading `budget=`).
    pub fn to_query_value(&self) -> String {
        let mut parts = Vec::new();
        if let Some(value) = self.max_results {
            parts.push(format!("max_results={value}"));
        }
        if let Some(value) = self.max_bytes {
            parts.push(format!("max_bytes={value}"));
        }
        if let Some(value) = self.max_depth {
            parts.push(format!("max_depth={value}"));
        }
        if let Some(value) = self.max_work {
            parts.push(format!("max_work={value}"));
        }
        parts.join(",")
    }

    /// JSON object for CLI output.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "max_results": self.max_results,
            "max_bytes": self.max_bytes,
            "max_depth": self.max_depth,
            "max_work": self.max_work,
        })
    }
}

// ---------------------------------------------------------------------------
// Identity mode
// ---------------------------------------------------------------------------

/// Content/causal identity mode (WEFT-CAUSAL §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityMode {
    /// v1 default: `VersionId` depends on content/version bytes only.
    BytesOnly,
    /// `VersionId` additionally includes the causal closure hash.
    CausalExact,
}

impl IdentityMode {
    /// Parse the stored text form.
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "bytes_only" => Ok(IdentityMode::BytesOnly),
            "causal_exact" => Ok(IdentityMode::CausalExact),
            other => Err(Error::unsupported_version(format!(
                "unknown identity mode `{other}`"
            ))),
        }
    }

    /// Stored text form.
    pub fn as_str(self) -> &'static str {
        match self {
            IdentityMode::BytesOnly => "bytes_only",
            IdentityMode::CausalExact => "causal_exact",
        }
    }
}

impl std::fmt::Display for IdentityMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Causal closure
// ---------------------------------------------------------------------------

/// Causal closure record (WEFT-CAUSAL §1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalClosure {
    /// Explicit parents, strictly ascending and unique.
    pub parents: Vec<Hash32>,
    /// Transitive ancestor closure, strictly ascending and unique; superset of
    /// `parents`.
    pub closure: Vec<Hash32>,
}

impl CausalClosure {
    /// Build a closure from explicit parents only (the CLI construction).
    ///
    /// Parents are sorted and deduplicated; zero hashes are rejected.
    pub fn from_parents(parents: &[Hash32]) -> Result<Self> {
        let mut parents = parents.to_vec();
        sort_unique_hashes(&mut parents, "parents")?;
        let closure = parents.clone();
        let record = CausalClosure { parents, closure };
        record.validate(&Limits::default())?;
        Ok(record)
    }

    /// Build a validated closure from explicit parents and a full closure.
    pub fn new(parents: Vec<Hash32>, closure: Vec<Hash32>) -> Result<Self> {
        let record = CausalClosure { parents, closure };
        record.validate(&Limits::default())?;
        Ok(record)
    }

    /// Strict validation of array rules.
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        check_entries(self.parents.len(), limits, "causal parents")?;
        check_entries(self.closure.len(), limits, "causal closure")?;
        check_ascending_unique(&self.parents, "causal parents")?;
        check_ascending_unique(&self.closure, "causal closure")?;
        for parent in &self.parents {
            if !contains_hash(&self.closure, parent) {
                return Err(Error::non_canonical(format!(
                    "causal closure is missing parent {parent}"
                )));
            }
        }
        Ok(())
    }

    /// Canonical WCB value.
    pub fn to_wcb_value(&self) -> Value {
        Value::map([
            ("schema", Value::Str(CAUSAL_SCHEMA.to_owned())),
            ("parents", hashes_to_value(&self.parents)),
            ("closure", hashes_to_value(&self.closure)),
        ])
    }

    /// Canonical WCB bytes with default limits.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.canonical_bytes_with_limits(&Limits::default())
    }

    /// Canonical WCB bytes with explicit limits.
    pub fn canonical_bytes_with_limits(&self, limits: &Limits) -> Result<Vec<u8>> {
        self.validate(limits)?;
        wcb::encode(&self.to_wcb_value(), limits)
    }

    /// `H("weft/causal/v1", canonical bytes)`.
    pub fn id(&self) -> Result<Hash32> {
        Ok(hash_domain(DOMAIN_CAUSAL, &self.canonical_bytes()?))
    }

    /// Decode canonical closure bytes, rejecting every non-canonical form.
    pub fn from_canonical_bytes(bytes: &[u8], limits: &Limits) -> Result<Self> {
        let value = wcb::decode(bytes, limits)?;
        let map = as_map(&value, "causal record")?;
        for key in map.keys() {
            match key.as_str() {
                "schema" | "parents" | "closure" => {}
                other => {
                    return Err(Error::non_canonical(format!(
                        "unknown causal record key `{other}`"
                    )))
                }
            }
        }
        let schema = required_text(map, "schema")?;
        if schema != CAUSAL_SCHEMA {
            return Err(Error::unknown_schema(format!(
                "unknown causal schema `{schema}`"
            )));
        }
        let parents = required_hashes(map, "parents", limits, "causal parents")?;
        let closure = required_hashes(map, "closure", limits, "causal closure")?;
        let record = CausalClosure { parents, closure };
        record.validate(limits)?;
        let reencoded = wcb::encode(&record.to_wcb_value(), limits)?;
        if reencoded != bytes {
            return Err(Error::non_canonical(
                "causal bytes are not in canonical form",
            ));
        }
        Ok(record)
    }
}

// ---------------------------------------------------------------------------
// Capability scope
// ---------------------------------------------------------------------------

/// Capability scope record (WEFT-CAPABILITIES §1).
///
/// The record is an identifier for a grant. It is **not** a signature and
/// carries no cryptographic authority by itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityScope {
    pub principal: String,
    pub namespaces: Vec<String>,
    pub roots: Vec<Hash32>,
    /// Relation predicate allowlist; empty means no relation traversal.
    pub predicates: Vec<String>,
    pub budget: Option<Budget>,
    /// Inclusive expiry, UTC nanoseconds since the Unix epoch.
    pub expires_at: Option<i64>,
}

impl CapabilityScope {
    /// Build a canonical scope: lists are sorted and deduplicated.
    pub fn new(
        principal: impl Into<String>,
        namespaces: Vec<String>,
        roots: Vec<Hash32>,
        predicates: Vec<String>,
        budget: Option<Budget>,
        expires_at: Option<i64>,
    ) -> Result<Self> {
        let mut namespaces = namespaces;
        namespaces.sort();
        namespaces.dedup();
        for namespace in &namespaces {
            validate_namespace(namespace)?;
        }
        let mut roots = roots;
        sort_unique_hashes(&mut roots, "capability roots")?;
        let mut predicates = predicates;
        predicates.sort();
        predicates.dedup();
        for predicate in &predicates {
            validate_predicate(predicate)?;
        }
        if let Some(budget) = &budget {
            budget.validate()?;
        }
        let scope = CapabilityScope {
            principal: principal.into(),
            namespaces,
            roots,
            predicates,
            budget,
            expires_at,
        };
        scope.validate(&Limits::default())?;
        Ok(scope)
    }

    /// Strict validation.
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        if self.principal.is_empty() || self.principal.len() > 256 {
            return Err(Error::non_canonical(
                "capability principal must be 1..=256 bytes",
            ));
        }
        if self.namespaces.is_empty() && self.roots.is_empty() {
            return Err(Error::non_canonical(
                "capability must grant at least one namespace or root",
            ));
        }
        check_capability_entries(self.namespaces.len(), limits, "capability namespaces")?;
        check_capability_entries(self.roots.len(), limits, "capability roots")?;
        check_capability_entries(self.predicates.len(), limits, "capability predicates")?;
        for namespace in &self.namespaces {
            validate_namespace(namespace)?;
        }
        if !strictly_sorted_strings(&self.namespaces) {
            return Err(Error::non_canonical(
                "capability namespaces must be sorted and unique",
            ));
        }
        check_ascending_unique(&self.roots, "capability roots")?;
        if !strictly_sorted_strings(&self.predicates) {
            return Err(Error::non_canonical(
                "capability predicates must be sorted and unique",
            ));
        }
        for predicate in &self.predicates {
            validate_predicate(predicate)?;
        }
        if let Some(budget) = &self.budget {
            budget.validate()?;
        }
        Ok(())
    }

    /// Canonical WCB value.
    pub fn to_wcb_value(&self) -> Value {
        let mut map = BTreeMap::new();
        map.insert(
            "schema".to_owned(),
            Value::Str(CAPABILITY_SCHEMA.to_owned()),
        );
        map.insert("principal".to_owned(), Value::Str(self.principal.clone()));
        map.insert(
            "namespaces".to_owned(),
            Value::Array(
                self.namespaces
                    .iter()
                    .map(|namespace| Value::Str(namespace.clone()))
                    .collect(),
            ),
        );
        map.insert("roots".to_owned(), hashes_to_value(&self.roots));
        if !self.predicates.is_empty() {
            map.insert(
                "predicates".to_owned(),
                Value::Array(
                    self.predicates
                        .iter()
                        .map(|predicate| Value::Str(predicate.clone()))
                        .collect(),
                ),
            );
        }
        if let Some(budget) = &self.budget {
            map.insert("budget".to_owned(), budget.to_wcb_value());
        }
        if let Some(expires_at) = self.expires_at {
            map.insert("expires_at".to_owned(), Value::Timestamp(expires_at));
        }
        Value::Map(map)
    }

    /// Canonical WCB bytes.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.canonical_bytes_with_limits(&Limits::default())
    }

    /// Canonical WCB bytes with explicit limits.
    pub fn canonical_bytes_with_limits(&self, limits: &Limits) -> Result<Vec<u8>> {
        self.validate(limits)?;
        wcb::encode(&self.to_wcb_value(), limits)
    }

    /// `H("weft/capability/v1", canonical bytes)`.
    pub fn id(&self) -> Result<Hash32> {
        Ok(hash_domain(DOMAIN_CAPABILITY, &self.canonical_bytes()?))
    }

    /// Decode canonical capability bytes, rejecting every non-canonical form.
    pub fn from_canonical_bytes(bytes: &[u8], limits: &Limits) -> Result<Self> {
        let value = wcb::decode(bytes, limits)?;
        let map = as_map(&value, "capability record")?;
        for key in map.keys() {
            match key.as_str() {
                "schema" | "principal" | "namespaces" | "roots" | "predicates" | "budget"
                | "expires_at" => {}
                other => {
                    return Err(Error::non_canonical(format!(
                        "unknown capability record key `{other}`"
                    )))
                }
            }
        }
        let schema = required_text(map, "schema")?;
        if schema != CAPABILITY_SCHEMA {
            return Err(Error::unknown_schema(format!(
                "unknown capability schema `{schema}`"
            )));
        }
        let principal = required_text(map, "principal")?.to_owned();
        let namespaces = required_text_array(map, "namespaces", "capability namespaces")?;
        let roots = required_hashes(map, "roots", limits, "capability roots")?;
        let predicates = match map.get("predicates") {
            Some(value) => text_array(value, "capability predicates")?,
            None => Vec::new(),
        };
        let budget = match map.get("budget") {
            Some(value) => Some(Budget::from_wcb_value(value)?),
            None => None,
        };
        let expires_at = match map.get("expires_at") {
            Some(Value::Timestamp(value)) => Some(*value),
            Some(_) => {
                return Err(Error::non_canonical(
                    "capability expires_at must be a timestamp",
                ))
            }
            None => None,
        };
        let scope = CapabilityScope {
            principal,
            namespaces,
            roots,
            predicates,
            budget,
            expires_at,
        };
        scope.validate(limits)?;
        let reencoded = wcb::encode(&scope.to_wcb_value(), limits)?;
        if reencoded != bytes {
            return Err(Error::non_canonical(
                "capability bytes are not in canonical form",
            ));
        }
        Ok(scope)
    }
}

// ---------------------------------------------------------------------------
// Context Root
// ---------------------------------------------------------------------------

/// A Context Root (WEFT-CONTEXT-ROOT §2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextRoot {
    pub namespace: String,
    pub selector_bytes: Vec<u8>,
    pub selector_id: Hash32,
    pub snapshot: Hash32,
    pub resolver: String,
    pub causal: Vec<Hash32>,
    pub ordering: String,
    pub capability_hash: Option<Hash32>,
    pub budget: Option<Budget>,
}

impl ContextRoot {
    /// Build a Context Root from a parsed selector and pinned view parameters.
    ///
    /// Causal entries are sorted and deduplicated. The selector namespace is the
    /// record namespace; the selector id is derived, never supplied.
    pub fn new(
        selector: &Selector,
        snapshot: Hash32,
        resolver: &str,
        causal: Vec<Hash32>,
        capability_hash: Option<Hash32>,
        budget: Option<Budget>,
    ) -> Result<Self> {
        validate_namespace(&selector.namespace)?;
        if snapshot == Hash32::ZERO {
            return Err(Error::non_canonical("snapshot must not be all zero"));
        }
        validate_resolver(resolver)?;
        if capability_hash == Some(Hash32::ZERO) {
            return Err(Error::non_canonical("capability hash must not be all zero"));
        }
        if let Some(budget) = &budget {
            budget.validate()?;
        }
        let mut causal = causal;
        sort_unique_hashes(&mut causal, "causal closure")?;

        let selector_bytes = selector.canonical_bytes()?;
        let selector_id = hash_domain(crate::hash::DOMAIN_SELECTOR, &selector_bytes);
        let root = ContextRoot {
            namespace: selector.namespace.clone(),
            selector_bytes,
            selector_id,
            snapshot,
            resolver: resolver.to_owned(),
            causal,
            ordering: ORDERING_SELECTOR_THEN_OBJECT_ID.to_owned(),
            capability_hash,
            budget,
        };
        root.validate(&Limits::default())?;
        Ok(root)
    }

    /// Decode the embedded selector.
    pub fn selector(&self) -> Result<Selector> {
        Selector::from_canonical_bytes(&self.selector_bytes, &Limits::default())
    }

    /// Strict validation of every record invariant.
    pub fn validate(&self, limits: &Limits) -> Result<()> {
        validate_namespace(&self.namespace)?;
        validate_resolver(&self.resolver)?;
        if self.snapshot == Hash32::ZERO {
            return Err(Error::non_canonical("snapshot must not be all zero"));
        }
        if self.ordering != ORDERING_SELECTOR_THEN_OBJECT_ID {
            return Err(Error::unsupported_version(format!(
                "unknown ordering policy `{}` (v1 supports `{ORDERING_SELECTOR_THEN_OBJECT_ID}`)",
                self.ordering
            )));
        }
        if self.capability_hash == Some(Hash32::ZERO) {
            return Err(Error::non_canonical("capability hash must not be all zero"));
        }
        check_entries(self.causal.len(), limits, "causal closure")?;
        check_ascending_unique(&self.causal, "causal closure")?;
        if let Some(budget) = &self.budget {
            budget.validate()?;
        }

        let selector = Selector::from_canonical_bytes(&self.selector_bytes, limits)?;
        if selector.namespace != self.namespace {
            return Err(Error::non_canonical(format!(
                "selector namespace `{}` does not match context namespace `{}`",
                selector.namespace, self.namespace
            )));
        }
        let reencoded = selector.canonical_bytes_with_limits(limits)?;
        if reencoded != self.selector_bytes {
            return Err(Error::non_canonical(
                "selector bytes are not in canonical form",
            ));
        }
        let expected = hash_domain(crate::hash::DOMAIN_SELECTOR, &self.selector_bytes);
        if expected != self.selector_id {
            return Err(Error::non_canonical(
                "selector_id does not match the embedded selector bytes",
            ));
        }
        Ok(())
    }

    /// Canonical WCB value.
    pub fn to_wcb_value(&self) -> Value {
        let mut map = BTreeMap::new();
        map.insert(
            "schema".to_owned(),
            Value::Str(CONTEXT_ROOT_SCHEMA.to_owned()),
        );
        map.insert("namespace".to_owned(), Value::Str(self.namespace.clone()));
        map.insert(
            "selector".to_owned(),
            Value::Bytes(self.selector_bytes.clone()),
        );
        map.insert(
            "selector_id".to_owned(),
            Value::Bytes(self.selector_id.as_bytes().to_vec()),
        );
        map.insert(
            "snapshot".to_owned(),
            Value::Bytes(self.snapshot.as_bytes().to_vec()),
        );
        map.insert("resolver".to_owned(), Value::Str(self.resolver.clone()));
        map.insert("causal".to_owned(), hashes_to_value(&self.causal));
        map.insert("ordering".to_owned(), Value::Str(self.ordering.clone()));
        if let Some(capability_hash) = self.capability_hash {
            map.insert(
                "capability_hash".to_owned(),
                Value::Bytes(capability_hash.as_bytes().to_vec()),
            );
        }
        if let Some(budget) = &self.budget {
            map.insert("budget".to_owned(), budget.to_wcb_value());
        }
        Value::Map(map)
    }

    /// Canonical WCB bytes with default limits.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.canonical_bytes_with_limits(&Limits::default())
    }

    /// Canonical WCB bytes with explicit limits.
    pub fn canonical_bytes_with_limits(&self, limits: &Limits) -> Result<Vec<u8>> {
        self.validate(limits)?;
        wcb::encode(&self.to_wcb_value(), limits)
    }

    /// Canonical record hex (lowercase).
    pub fn canonical_hex(&self) -> Result<String> {
        Ok(hex::encode(self.canonical_bytes()?))
    }

    /// `ContextRootId = H("weft/context-root/v1", canonical bytes)`.
    pub fn id(&self) -> Result<Hash32> {
        Ok(hash_domain(DOMAIN_CONTEXT_ROOT, &self.canonical_bytes()?))
    }

    /// Authoritative `weft://<namespace>/c/<id>` address.
    pub fn authoritative_uri(&self) -> Result<String> {
        Ok(format!(
            "{URI_PREFIX}{}/c/{}",
            self.namespace,
            self.id()?.to_hex()
        ))
    }

    /// Descriptor spelling for this root and its selector source.
    ///
    /// The source must parse (under this namespace) to exactly the record's
    /// canonical selector bytes; otherwise this is not a spelling of this root.
    pub fn descriptor_uri(&self, selector_source: &str) -> Result<String> {
        let parsed = selector::parse_with(selector_source, &self.namespace, &Limits::default())?;
        let canonical = parsed.canonical_bytes()?;
        if canonical != self.selector_bytes {
            return Err(Error::non_canonical(
                "selector source does not denote the record's canonical selector",
            ));
        }
        let mut uri = format!(
            "{URI_PREFIX}{}/c/{}@{}",
            self.namespace,
            percent_encode(selector_source),
            self.snapshot.to_hex()
        );
        let mut params: Vec<String> = Vec::new();
        if !self.causal.is_empty() {
            params.push(format!("closure={}", hashes_to_hex_list(&self.causal)));
        }
        if let Some(capability_hash) = self.capability_hash {
            params.push(format!("cap={}", capability_hash.to_hex()));
        }
        if let Some(budget) = &self.budget {
            params.push(format!("budget={}", budget.to_query_value()));
        }
        if self.resolver != DEFAULT_RESOLVER {
            params.push(format!("resolver={}", self.resolver));
        }
        if !params.is_empty() {
            uri.push('?');
            uri.push_str(&params.join("&"));
        }
        Ok(uri)
    }

    /// Decode canonical context-root bytes, rejecting every non-canonical form.
    pub fn from_canonical_bytes(bytes: &[u8], limits: &Limits) -> Result<Self> {
        let value = wcb::decode(bytes, limits)?;
        let map = as_map(&value, "context root")?;
        for key in map.keys() {
            match key.as_str() {
                "schema" | "namespace" | "selector" | "selector_id" | "snapshot" | "resolver"
                | "causal" | "ordering" | "capability_hash" | "budget" => {}
                other => {
                    return Err(Error::non_canonical(format!(
                        "unknown context-root key `{other}`"
                    )))
                }
            }
        }
        let schema = required_text(map, "schema")?;
        if schema != CONTEXT_ROOT_SCHEMA {
            return Err(Error::unknown_schema(format!(
                "unknown context-root schema `{schema}`"
            )));
        }
        let namespace = required_text(map, "namespace")?.to_owned();
        let selector_bytes = required_bytes(map, "selector")?.to_vec();
        let selector_id = required_hash(map, "selector_id", "selector_id")?;
        let snapshot = required_hash(map, "snapshot", "snapshot")?;
        let resolver = required_text(map, "resolver")?.to_owned();
        let causal = required_hashes(map, "causal", limits, "causal closure")?;
        let ordering = required_text(map, "ordering")?.to_owned();
        let capability_hash = match map.get("capability_hash") {
            Some(value) => Some(hash_from_value(value, "capability_hash")?),
            None => None,
        };
        let budget = match map.get("budget") {
            Some(value) => Some(Budget::from_wcb_value(value)?),
            None => None,
        };
        let root = ContextRoot {
            namespace,
            selector_bytes,
            selector_id,
            snapshot,
            resolver,
            causal,
            ordering,
            capability_hash,
            budget,
        };
        root.validate(limits)?;
        let reencoded = wcb::encode(&root.to_wcb_value(), limits)?;
        if reencoded != bytes {
            return Err(Error::non_canonical(
                "context-root bytes are not in canonical form",
            ));
        }
        Ok(root)
    }
}

// ---------------------------------------------------------------------------
// weft:// URI
// ---------------------------------------------------------------------------

/// A parsed `weft://` URI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContextUri {
    /// `weft://<namespace>/c/<64hex>`: carries only the address.
    Authoritative { namespace: String, id: Hash32 },
    /// Descriptive spelling with selector text plus view parameters.
    Descriptor {
        namespace: String,
        selector_source: String,
        snapshot: Hash32,
        closure: Vec<Hash32>,
        capability_hash: Option<Hash32>,
        budget: Option<Budget>,
        resolver: String,
    },
}

impl ContextUri {
    /// Build the Context Root for a descriptor; `None` for authoritative form
    /// (an address alone does not reveal its record).
    pub fn to_context_root(&self, limits: &Limits) -> Result<Option<ContextRoot>> {
        match self {
            ContextUri::Authoritative { .. } => Ok(None),
            ContextUri::Descriptor {
                namespace,
                selector_source,
                snapshot,
                closure,
                capability_hash,
                budget,
                resolver,
            } => {
                let selector = selector::parse_with(selector_source, namespace, limits)?;
                let mut causal = closure.clone();
                sort_unique_hashes(&mut causal, "causal closure")?;
                let root = ContextRoot::new(
                    &selector,
                    *snapshot,
                    resolver,
                    causal,
                    *capability_hash,
                    budget.clone(),
                )?;
                Ok(Some(root))
            }
        }
    }

    /// The canonical authoritative URI. Descriptors are canonicalized first.
    pub fn canonical_uri(&self, limits: &Limits) -> Result<String> {
        match self {
            ContextUri::Authoritative { namespace, id } => {
                Ok(format!("{URI_PREFIX}{namespace}/c/{}", id.to_hex()))
            }
            ContextUri::Descriptor { .. } => {
                let root = self
                    .to_context_root(limits)?
                    .ok_or_else(|| Error::non_canonical("descriptor produced no root"))?;
                root.authoritative_uri()
            }
        }
    }
}

/// Parse either `weft://` form.
pub fn parse_context_uri(input: &str, limits: &Limits) -> Result<ContextUri> {
    if input.len() as u64 > limits.max_uri_bytes {
        return Err(Error::limit_exceeded(format!(
            "URI length {} exceeds limit {}",
            input.len(),
            limits.max_uri_bytes
        )));
    }
    let rest = input
        .strip_prefix(URI_PREFIX)
        .ok_or_else(|| Error::parse("URI must start with `weft://`"))?;
    if rest.contains('#') {
        return Err(Error::parse("URI fragments are not permitted"));
    }
    let (authority, path_and_query) = rest
        .split_once('/')
        .ok_or_else(|| Error::parse("URI is missing the `/c/` path"))?;
    validate_namespace(authority).map_err(|err| Error::parse(err.message()))?;
    if !path_and_query.starts_with("c/") {
        return Err(Error::parse("URI path must start with `/c/`"));
    }
    let tail = &path_and_query[2..];
    if tail.is_empty() {
        return Err(Error::parse("URI is missing the address after `/c/`"));
    }
    let (path, query) = match tail.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (tail, None),
    };
    if path.is_empty() {
        return Err(Error::parse("URI is missing the address after `/c/`"));
    }

    if let Some(at) = path.find('@') {
        // Descriptor form.
        let (encoded_selector, snapshot_hex) = (&path[..at], &path[at + 1..]);
        if snapshot_hex.contains('@') {
            return Err(Error::parse("descriptor contains more than one raw `@`"));
        }
        if encoded_selector.is_empty() {
            return Err(Error::parse("descriptor selector text is empty"));
        }
        let selector_source = percent_decode(encoded_selector)?;
        let snapshot = parse_uri_hash(snapshot_hex, "snapshot")?;
        let mut closure: Vec<Hash32> = Vec::new();
        let mut capability_hash: Option<Hash32> = None;
        let mut budget: Option<Budget> = None;
        let mut resolver = DEFAULT_RESOLVER.to_owned();
        if let Some(query) = query {
            if query.is_empty() {
                return Err(Error::parse("empty URI query"));
            }
            let mut seen = std::collections::BTreeSet::new();
            for param in query.split('&') {
                if param.is_empty() {
                    return Err(Error::parse("empty URI query parameter"));
                }
                let (key, value) = param
                    .split_once('=')
                    .ok_or_else(|| Error::parse(format!("query parameter `{param}` has no `=`")))?;
                if !seen.insert(key.to_owned()) {
                    return Err(Error::parse(format!("duplicate query parameter `{key}`")));
                }
                match key {
                    "closure" => {
                        if value.is_empty() {
                            return Err(Error::parse("closure parameter must not be empty"));
                        }
                        for entry in value.split(',') {
                            closure.push(parse_uri_hash(entry, "closure entry")?);
                        }
                    }
                    "cap" => {
                        capability_hash = Some(parse_uri_hash(value, "cap")?);
                    }
                    "budget" => {
                        budget = Some(Budget::parse_spec(value)?);
                    }
                    "resolver" => {
                        validate_resolver(value).map_err(|err| Error::parse(err.message()))?;
                        resolver = value.to_owned();
                    }
                    other => {
                        return Err(Error::parse(format!(
                            "unknown URI query parameter `{other}`"
                        )))
                    }
                }
            }
        }
        Ok(ContextUri::Descriptor {
            namespace: authority.to_owned(),
            selector_source,
            snapshot,
            closure,
            capability_hash,
            budget,
            resolver,
        })
    } else {
        // Authoritative form.
        if query.is_some() {
            return Err(Error::parse(
                "the authoritative form does not accept a query",
            ));
        }
        let id = parse_uri_hash(path, "ContextRootId")?;
        Ok(ContextUri::Authoritative {
            namespace: authority.to_owned(),
            id,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn as_map<'a>(value: &'a Value, what: &str) -> Result<&'a BTreeMap<String, Value>> {
    match value {
        Value::Map(map) => Ok(map),
        _ => Err(Error::non_canonical(format!("{what} must be a WCB map"))),
    }
}

fn required_text<'a>(map: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a str> {
    match map.get(key) {
        Some(Value::Str(text)) => Ok(text),
        Some(_) => Err(Error::non_canonical(format!("`{key}` must be text"))),
        None => Err(Error::non_canonical(format!("missing `{key}`"))),
    }
}

fn required_bytes<'a>(map: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a [u8]> {
    match map.get(key) {
        Some(Value::Bytes(bytes)) => Ok(bytes),
        Some(_) => Err(Error::non_canonical(format!("`{key}` must be bytes"))),
        None => Err(Error::non_canonical(format!("missing `{key}`"))),
    }
}

fn hash_from_value(value: &Value, what: &str) -> Result<Hash32> {
    match value {
        Value::Bytes(bytes) => {
            if bytes.len() != crate::hash::HASH_LEN {
                return Err(Error::non_canonical(format!(
                    "`{what}` must be exactly 32 bytes"
                )));
            }
            Hash32::from_slice(bytes)
        }
        _ => Err(Error::non_canonical(format!("`{what}` must be bytes(32)"))),
    }
}

fn required_hash(map: &BTreeMap<String, Value>, key: &str, what: &str) -> Result<Hash32> {
    let value = map
        .get(key)
        .ok_or_else(|| Error::non_canonical(format!("missing `{key}`")))?;
    let hash = hash_from_value(value, what)?;
    if hash == Hash32::ZERO {
        return Err(Error::non_canonical(format!(
            "`{what}` must not be all zero"
        )));
    }
    Ok(hash)
}

fn required_hashes(
    map: &BTreeMap<String, Value>,
    key: &str,
    limits: &Limits,
    what: &str,
) -> Result<Vec<Hash32>> {
    let value = map
        .get(key)
        .ok_or_else(|| Error::non_canonical(format!("missing `{key}`")))?;
    let items = match value {
        Value::Array(items) => items,
        _ => {
            return Err(Error::non_canonical(format!(
                "`{key}` must be an array of bytes(32)"
            )))
        }
    };
    check_entries(items.len(), limits, what)?;
    let mut hashes = Vec::with_capacity(items.len());
    for item in items {
        let hash = hash_from_value(item, what)?;
        if hash == Hash32::ZERO {
            return Err(Error::non_canonical(format!(
                "{what} must not contain the all-zero hash"
            )));
        }
        hashes.push(hash);
    }
    Ok(hashes)
}

fn required_text_array(
    map: &BTreeMap<String, Value>,
    key: &str,
    what: &str,
) -> Result<Vec<String>> {
    let value = map
        .get(key)
        .ok_or_else(|| Error::non_canonical(format!("missing `{key}`")))?;
    text_array(value, what)
}

fn text_array(value: &Value, what: &str) -> Result<Vec<String>> {
    match value {
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::Str(text) => out.push(text.clone()),
                    _ => {
                        return Err(Error::non_canonical(format!(
                            "{what} must contain only text"
                        )))
                    }
                }
            }
            Ok(out)
        }
        _ => Err(Error::non_canonical(format!("{what} must be an array"))),
    }
}

fn hashes_to_value(hashes: &[Hash32]) -> Value {
    Value::Array(
        hashes
            .iter()
            .map(|hash| Value::Bytes(hash.as_bytes().to_vec()))
            .collect(),
    )
}

fn contains_hash(hashes: &[Hash32], needle: &Hash32) -> bool {
    hashes.binary_search(needle).is_ok()
}

fn check_entries(count: usize, limits: &Limits, what: &str) -> Result<()> {
    if count as u64 > limits.max_causal_entries {
        return Err(Error::limit_exceeded(format!(
            "{what} has {count} entries, limit {}",
            limits.max_causal_entries
        )));
    }
    Ok(())
}

fn check_capability_entries(count: usize, limits: &Limits, what: &str) -> Result<()> {
    if count as u64 > limits.max_capability_entries {
        return Err(Error::limit_exceeded(format!(
            "{what} has {count} entries, limit {}",
            limits.max_capability_entries
        )));
    }
    Ok(())
}

fn sort_unique_hashes(hashes: &mut Vec<Hash32>, what: &str) -> Result<()> {
    for hash in hashes.iter() {
        if *hash == Hash32::ZERO {
            return Err(Error::non_canonical(format!(
                "{what} must not contain the all-zero hash"
            )));
        }
    }
    hashes.sort();
    hashes.dedup();
    Ok(())
}

fn check_ascending_unique(hashes: &[Hash32], what: &str) -> Result<()> {
    for pair in hashes.windows(2) {
        if pair[0] >= pair[1] {
            return Err(Error::non_canonical(format!(
                "{what} must be strictly ascending and unique"
            )));
        }
    }
    for hash in hashes {
        if *hash == Hash32::ZERO {
            return Err(Error::non_canonical(format!(
                "{what} must not contain the all-zero hash"
            )));
        }
    }
    Ok(())
}

fn strictly_sorted_strings(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

/// Validate the namespace grammar (WEFT-CONTEXT-ROOT §3.1).
pub fn validate_namespace(namespace: &str) -> Result<()> {
    let bytes = namespace.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_NAMESPACE_BYTES {
        return Err(Error::non_canonical("namespace must be 1..=128 bytes"));
    }
    if namespace == "." || namespace == ".." {
        return Err(Error::non_canonical("namespace `.` or `..` is not allowed"));
    }
    let first = bytes[0];
    if !(first.is_ascii_alphanumeric() || first == b'_') {
        return Err(Error::non_canonical(
            "namespace must begin with an ASCII letter, digit, or `_`",
        ));
    }
    if !bytes
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || matches!(*b, b'_' | b'.' | b'-'))
    {
        return Err(Error::non_canonical(
            "namespace may contain only ASCII letters, digits, `_`, `.`, and `-`",
        ));
    }
    Ok(())
}

/// Validate resolver/policy version text (WEFT-CONTEXT-ROOT §3.2).
pub fn validate_resolver(resolver: &str) -> Result<()> {
    let bytes = resolver.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_RESOLVER_BYTES {
        return Err(Error::non_canonical("resolver must be 1..=64 bytes"));
    }
    if !bytes[0].is_ascii_alphabetic() {
        return Err(Error::non_canonical(
            "resolver must begin with an ASCII letter",
        ));
    }
    if !bytes
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || matches!(*b, b'.' | b'-' | b'_'))
    {
        return Err(Error::non_canonical(
            "resolver may contain only ASCII letters, digits, `.`, `-`, and `_`",
        ));
    }
    Ok(())
}

fn validate_predicate(predicate: &str) -> Result<()> {
    if predicate.is_empty() || predicate.len() > 256 {
        return Err(Error::non_canonical(
            "capability predicate must be 1..=256 bytes",
        ));
    }
    if !predicate
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'/'))
    {
        return Err(Error::non_canonical(
            "capability predicate contains a disallowed byte",
        ));
    }
    Ok(())
}

fn parse_uri_hash(text: &str, what: &str) -> Result<Hash32> {
    if text.len() != 64 {
        return Err(Error::parse(format!(
            "{what} must be exactly 64 lowercase hex digits"
        )));
    }
    if !text
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Error::parse(format!(
            "{what} must be lowercase hexadecimal"
        )));
    }
    let hash =
        Hash32::from_hex(text).map_err(|err| Error::parse(format!("{what}: {}", err.message())))?;
    if hash == Hash32::ZERO {
        return Err(Error::parse(format!("{what} must not be all zero")));
    }
    Ok(hash)
}

fn hashes_to_hex_list(hashes: &[Hash32]) -> String {
    hashes
        .iter()
        .map(|hash| hash.to_hex())
        .collect::<Vec<_>>()
        .join(",")
}

fn is_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
}

/// Strict percent-encoding of selector text (WEFT-CONTEXT-ROOT §5.3).
pub fn percent_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.as_bytes() {
        if is_unreserved(*byte) {
            out.push(*byte as char);
        } else {
            out.push('%');
            out.push_str(&format!("{byte:02X}"));
        }
    }
    out
}

/// Strict percent-decoding, rejecting non-canonical escapes.
pub fn percent_decode(text: &str) -> Result<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'%' {
            if index + 2 >= bytes.len() {
                return Err(Error::parse("truncated percent escape"));
            }
            let high = bytes[index + 1];
            let low = bytes[index + 2];
            let hex_digit = |b: u8| -> Option<u8> {
                match b {
                    b'0'..=b'9' => Some(b - b'0'),
                    b'A'..=b'F' => Some(b - b'A' + 10),
                    _ => None,
                }
            };
            let high = hex_digit(high)
                .ok_or_else(|| Error::parse("percent escape must use uppercase hex digits"))?;
            let low = hex_digit(low)
                .ok_or_else(|| Error::parse("percent escape must use uppercase hex digits"))?;
            let decoded = (high << 4) | low;
            if is_unreserved(decoded) {
                return Err(Error::parse(
                    "percent escape encodes an unreserved character (non-canonical)",
                ));
            }
            out.push(decoded);
            index += 3;
        } else if is_unreserved(byte) {
            out.push(byte);
            index += 1;
        } else {
            return Err(Error::parse(format!(
                "raw byte 0x{byte:02X} in selector text must be percent-encoded"
            )));
        }
    }
    String::from_utf8(out)
        .map_err(|_| Error::parse("percent-decoded selector text is not valid UTF-8"))
}

/// Convenience constructor used by the CLI and tests.
pub fn build_context_root(
    selector_source: &str,
    namespace: &str,
    snapshot_hex: &str,
    resolver: &str,
    causal: &[Hash32],
    capability_hash: Option<Hash32>,
    budget: Option<Budget>,
) -> Result<ContextRoot> {
    let limits = Limits::default();
    let selector = selector::parse_with(selector_source, namespace, &limits)?;
    let snapshot = parse_uri_hash(snapshot_hex, "snapshot")?;
    let mut causal = causal.to_vec();
    sort_unique_hashes(&mut causal, "causal closure")?;
    ContextRoot::new(
        &selector,
        snapshot,
        resolver,
        causal,
        capability_hash,
        budget,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> Hash32 {
        Hash32::from_hex(&"11".repeat(32)).unwrap()
    }

    fn root() -> ContextRoot {
        build_context_root(
            "select attrs.project == \"demo\" limit 10",
            "default",
            &"22".repeat(32),
            DEFAULT_RESOLVER,
            &[],
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn budget_roundtrip_and_rejects() {
        let budget = Budget::parse_spec("max_results=10,max_bytes=1000,max_depth=3").unwrap();
        assert_eq!(budget.max_results, Some(10));
        assert_eq!(budget.max_depth, Some(3));
        let bytes = wcb::encode(&budget.to_wcb_value(), &Limits::default()).unwrap();
        let value = wcb::decode(&bytes, &Limits::default()).unwrap();
        assert_eq!(Budget::from_wcb_value(&value).unwrap(), budget);
        for bad in [
            "",
            "max_results=0",
            "max_results=01",
            "max_results=1,max_results=2",
            "unknown=1",
            "max_results=",
        ] {
            assert!(Budget::parse_spec(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn context_root_roundtrip_and_uri() {
        let root = root();
        let bytes = root.canonical_bytes().unwrap();
        let decoded = ContextRoot::from_canonical_bytes(&bytes, &Limits::default()).unwrap();
        assert_eq!(decoded, root);
        assert_eq!(decoded.id().unwrap(), root.id().unwrap());
        let uri = root.authoritative_uri().unwrap();
        assert_eq!(
            uri,
            format!("weft://default/c/{}", root.id().unwrap().to_hex())
        );
        let parsed = parse_context_uri(&uri, &Limits::default()).unwrap();
        assert_eq!(parsed.canonical_uri(&Limits::default()).unwrap(), uri);
        assert_eq!(parsed.to_context_root(&Limits::default()).unwrap(), None);
    }

    #[test]
    fn descriptor_roundtrip() {
        let source = "select attrs.project == \"demo\" limit 10";
        let root = build_context_root(
            source,
            "default",
            &"22".repeat(32),
            DEFAULT_RESOLVER,
            &[Hash32::from_hex(&"aa".repeat(32)).unwrap()],
            None,
            Some(Budget::parse_spec("max_results=10").unwrap()),
        )
        .unwrap();
        let descriptor = root.descriptor_uri(source).unwrap();
        assert!(descriptor.contains("?closure="));
        let parsed = parse_context_uri(&descriptor, &Limits::default()).unwrap();
        assert_eq!(
            parsed.canonical_uri(&Limits::default()).unwrap(),
            root.authoritative_uri().unwrap()
        );
        let rebuilt = parsed.to_context_root(&Limits::default()).unwrap().unwrap();
        assert_eq!(rebuilt, root);
    }

    #[test]
    fn rejects_bad_uri_spellings() {
        let good = root();
        let id = good.id().unwrap().to_hex();
        let source_hex = percent_encode("select true");
        let snapshot_hex = "22".repeat(32);
        let cases = [
            format!("weft://default/c/{}", id.to_uppercase()),
            format!("weft://default/c/{}", "00".repeat(32)),
            format!("weft://default/c/{}?x=1", id),
            format!("weft://bad ns/c/{}", id),
            format!("weft://default/c/{}@{}", source_hex, "00".repeat(32)),
            format!(
                "weft://default/c/{}@{}?closure=bb",
                source_hex, snapshot_hex
            ),
            format!("weft://default/c/%41@{}", snapshot_hex),
            format!(
                "weft://default/c/select%20true@{}?budget=max_results=0",
                snapshot_hex
            ),
            format!(
                "weft://default/c/select%20true@{}?budget=max_results=2,max_results=3",
                snapshot_hex
            ),
            format!("weft://default/c/select%20true@{}#frag", snapshot_hex),
        ];
        for case in cases {
            assert!(
                parse_context_uri(&case, &Limits::default()).is_err(),
                "accepted {case}"
            );
        }
    }

    #[test]
    fn causal_closure_roundtrip_and_strictness() {
        let a = Hash32::from_hex(&"aa".repeat(32)).unwrap();
        let b = Hash32::from_hex(&"bb".repeat(32)).unwrap();
        let closure = CausalClosure::from_parents(&[b, a, b]).unwrap();
        assert_eq!(closure.parents, vec![a, b]);
        assert_eq!(closure.closure, vec![a, b]);
        let bytes = closure.canonical_bytes().unwrap();
        let decoded = CausalClosure::from_canonical_bytes(&bytes, &Limits::default()).unwrap();
        assert_eq!(decoded, closure);
        assert_eq!(decoded.id().unwrap(), closure.id().unwrap());

        let unsorted = CausalClosure {
            parents: vec![b, a],
            closure: vec![b, a],
        };
        let bytes = wcb::encode(&unsorted.to_wcb_value(), &Limits::default()).unwrap();
        assert!(CausalClosure::from_canonical_bytes(&bytes, &Limits::default()).is_err());
    }

    #[test]
    fn capability_scope_roundtrip() {
        let scope = CapabilityScope::new(
            "agent-1",
            vec!["default".to_owned()],
            vec![Hash32::from_hex(&"33".repeat(32)).unwrap()],
            vec!["derived_from".to_owned()],
            Some(Budget::parse_spec("max_results=4").unwrap()),
            Some(123),
        )
        .unwrap();
        let bytes = scope.canonical_bytes().unwrap();
        let decoded = CapabilityScope::from_canonical_bytes(&bytes, &Limits::default()).unwrap();
        assert_eq!(decoded, scope);
        assert_eq!(decoded.id().unwrap(), scope.id().unwrap());
        assert!(CapabilityScope::new("p", vec![], vec![], vec![], None, None).is_err());
    }

    #[test]
    fn identity_mode_parsing() {
        assert_eq!(
            IdentityMode::parse("bytes_only").unwrap(),
            IdentityMode::BytesOnly
        );
        assert_eq!(
            IdentityMode::parse("causal_exact").unwrap(),
            IdentityMode::CausalExact
        );
        assert_eq!(IdentityMode::CausalExact.to_string(), "causal_exact");
        let error = IdentityMode::parse("causalish").unwrap_err();
        assert_eq!(error.class(), crate::ErrorClass::UnsupportedVersion);
    }

    #[test]
    fn text_validators() {
        for good in ["default", "a", "a.b-c_9"] {
            assert!(validate_namespace(good).is_ok(), "{good}");
        }
        for bad in ["", ".", "..", "a/b", "a b", "-a", "é"] {
            assert!(validate_namespace(bad).is_err(), "{bad}");
        }
        assert!(validate_resolver("weft-resolver-v1").is_ok());
        assert!(validate_resolver("1bad").is_err());
        let _ = snapshot();
    }
}
