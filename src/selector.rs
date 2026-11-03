//! WeftQL selector parsing, normalization, and canonical identity
//! (WEFT-SELECTORS v1).
//!
//! The AST is public; evaluation belongs to the filesystem. Identity is
//! `H("weft/selector/v1", WCB(selector map))`.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result};
use crate::hash::{hash_domain, Hash32, DOMAIN_SELECTOR};
use crate::limits::Limits;
use crate::wcb::{self, Value};

/// Selector schema string.
pub const SELECTOR_SCHEMA: &str = "weft.selector.v1";
/// Default namespace used by the text grammar, which has no namespace clause.
pub const DEFAULT_NAMESPACE: &str = "default";

const RESERVED: &[&str] = &[
    "select", "and", "or", "not", "true", "false", "null", "has", "tag", "relation", "order", "by",
    "asc", "desc", "limit", "offset",
];

fn is_keyword(word: &str) -> bool {
    RESERVED.contains(&word)
}

/// A dotted attribute path.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Field {
    pub segments: Vec<String>,
}

impl Field {
    pub fn new(segments: Vec<String>) -> Self {
        Field { segments }
    }

    /// Dot-joined canonical text.
    pub fn as_string(&self) -> String {
        self.segments.join(".")
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_string())
    }
}

/// A scalar literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Literal {
    Null,
    Bool(bool),
    U64(u64),
    I64(i64),
    Str(String),
}

/// Comparison operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    fn discriminator(self) -> &'static str {
        match self {
            CmpOp::Eq => "attr_eq",
            CmpOp::Ne => "attr_ne",
            CmpOp::Lt => "attr_lt",
            CmpOp::Le => "attr_le",
            CmpOp::Gt => "attr_gt",
            CmpOp::Ge => "attr_ge",
        }
    }
}

/// Selector predicate AST. `And`/`Or` operands are in source order; order is
/// semantic for evidence and is never sorted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    True,
    False,
    Has(Field),
    Tag(String),
    Relation {
        predicate: String,
        target: Hash32,
    },
    Compare {
        field: Field,
        op: CmpOp,
        value: Literal,
    },
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Not(Box<Expr>),
}

/// One ordering term. `desc == false` means ascending (the default).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderItem {
    pub field: Field,
    pub desc: bool,
}

/// A parsed, normalized selector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selector {
    pub namespace: String,
    pub predicate: Expr,
    pub order: Vec<OrderItem>,
    pub limit: Option<u64>,
    /// `Some(0)` normalizes to `None`.
    pub offset: Option<u64>,
}

/// Parse with default limits and the default namespace.
pub fn parse(input: &str) -> Result<Selector> {
    parse_with(input, DEFAULT_NAMESPACE, &Limits::default())
}

/// Parse with an explicit namespace and limits.
pub fn parse_with(input: &str, namespace: &str, limits: &Limits) -> Result<Selector> {
    if input.len() as u64 > limits.max_selector_bytes {
        return Err(Error::limit_exceeded(format!(
            "selector source length {} exceeds limit {}",
            input.len(),
            limits.max_selector_bytes
        )));
    }
    if namespace.is_empty() {
        return Err(Error::parse("namespace must not be empty"));
    }
    if namespace.len() as u64 > limits.max_string_bytes {
        return Err(Error::limit_exceeded("namespace too long"));
    }
    let tokens = Lexer::new(input).tokenize()?;
    let mut parser = Parser {
        tokens,
        index: 0,
        limits: limits.clone(),
        operands: 0,
    };
    parser.parse_query(namespace)
}

/// Canonical bytes for a selector source string.
pub fn canonical_bytes(input: &str) -> Result<Vec<u8>> {
    parse(input)?.canonical_bytes()
}

/// Selector identity for a selector source string.
pub fn selector_id(input: &str) -> Result<Hash32> {
    parse(input)?.selector_id()
}

impl Selector {
    /// Canonical WCB value of the selector map.
    pub fn to_wcb_value(&self) -> Value {
        let mut map = BTreeMap::new();
        map.insert("schema".to_owned(), Value::Str(SELECTOR_SCHEMA.to_owned()));
        map.insert("namespace".to_owned(), Value::Str(self.namespace.clone()));
        map.insert("predicate".to_owned(), expr_to_value(&self.predicate));
        if !self.order.is_empty() {
            let items = self
                .order
                .iter()
                .map(|item| {
                    Value::map([
                        ("field", Value::Str(item.field.as_string())),
                        (
                            "dir",
                            Value::Str(if item.desc { "desc" } else { "asc" }.to_owned()),
                        ),
                    ])
                })
                .collect();
            map.insert("order".to_owned(), Value::Array(items));
        }
        if let Some(limit) = self.limit {
            map.insert("limit".to_owned(), Value::U64(limit));
        }
        if let Some(offset) = self.offset {
            if offset != 0 {
                map.insert("offset".to_owned(), Value::U64(offset));
            }
        }
        Value::Map(map)
    }

    /// Canonical WCB bytes with default limits.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.canonical_bytes_with_limits(&Limits::default())
    }

    /// Canonical WCB bytes with explicit limits.
    pub fn canonical_bytes_with_limits(&self, limits: &Limits) -> Result<Vec<u8>> {
        wcb::encode(&self.to_wcb_value(), limits)
    }

    /// `H("weft/selector/v1", canonical_bytes)`.
    pub fn selector_id(&self) -> Result<Hash32> {
        Ok(hash_domain(DOMAIN_SELECTOR, &self.canonical_bytes()?))
    }

    /// `H("weft/selector/v1", canonical_bytes)` with explicit limits.
    pub fn selector_id_with_limits(&self, limits: &Limits) -> Result<Hash32> {
        Ok(hash_domain(
            DOMAIN_SELECTOR,
            &self.canonical_bytes_with_limits(limits)?,
        ))
    }

    /// Decode canonical selector bytes back into an AST, rejecting every
    /// non-canonical form (wrong schema, unknown keys, nested same-op and/or,
    /// `not true`/`not false`, double negation, offset 0, non-text paths...).
    pub fn from_canonical_bytes(bytes: &[u8], limits: &Limits) -> Result<Selector> {
        let value = wcb::decode(bytes, limits)?;
        let map = match &value {
            Value::Map(map) => map,
            _ => return Err(Error::non_canonical("selector value must be a WCB map")),
        };
        for key in map.keys() {
            match key.as_str() {
                "schema" | "namespace" | "predicate" | "order" | "limit" | "offset" => {}
                other => {
                    return Err(Error::non_canonical(format!(
                        "unknown selector map key `{other}`"
                    )))
                }
            }
        }
        let schema = match map.get("schema") {
            Some(Value::Str(schema)) => schema,
            Some(_) => return Err(Error::non_canonical("schema must be text")),
            None => return Err(Error::non_canonical("selector map is missing schema")),
        };
        if schema != SELECTOR_SCHEMA {
            return Err(Error::unknown_schema(format!(
                "unknown selector schema `{schema}`"
            )));
        }
        let namespace = match map.get("namespace") {
            Some(Value::Str(namespace)) if !namespace.is_empty() => namespace.clone(),
            Some(Value::Str(_)) => return Err(Error::non_canonical("namespace must be non-empty")),
            Some(_) => return Err(Error::non_canonical("namespace must be text")),
            None => return Err(Error::non_canonical("selector map is missing namespace")),
        };
        let predicate = map
            .get("predicate")
            .ok_or_else(|| Error::non_canonical("selector map is missing predicate"))?;
        let mut operands = 0u64;
        let predicate = expr_from_value(predicate, limits, &mut operands, 0)?;

        let mut order = Vec::new();
        if let Some(value) = map.get("order") {
            let items = match value {
                Value::Array(items) => items,
                _ => return Err(Error::non_canonical("order must be an array")),
            };
            let mut seen = std::collections::HashSet::new();
            for item in items {
                let item_map = match item {
                    Value::Map(item_map) => item_map,
                    _ => return Err(Error::non_canonical("order item must be a map")),
                };
                for key in item_map.keys() {
                    if key != "field" && key != "dir" {
                        return Err(Error::non_canonical("unknown order item key"));
                    }
                }
                let field_text = match item_map.get("field") {
                    Some(Value::Str(text)) => text,
                    _ => return Err(Error::non_canonical("order field must be text")),
                };
                let field = field_from_canonical_text(field_text, limits)?;
                let desc = match item_map.get("dir") {
                    Some(Value::Str(dir)) if dir == "asc" => false,
                    Some(Value::Str(dir)) if dir == "desc" => true,
                    _ => {
                        return Err(Error::non_canonical(
                            "order dir must be text \"asc\" or \"desc\"",
                        ))
                    }
                };
                if !seen.insert(field.as_string()) {
                    return Err(Error::non_canonical("duplicate order field"));
                }
                order.push(OrderItem { field, desc });
            }
        }

        let limit = match map.get("limit") {
            Some(Value::U64(value)) => Some(*value),
            Some(_) => return Err(Error::non_canonical("limit must be u64")),
            None => None,
        };
        let offset = match map.get("offset") {
            Some(Value::U64(0)) => {
                return Err(Error::non_canonical(
                    "offset 0 must be omitted from canonical form",
                ))
            }
            Some(Value::U64(value)) => Some(*value),
            Some(_) => return Err(Error::non_canonical("offset must be u64")),
            None => None,
        };

        let selector = Selector {
            namespace,
            predicate,
            order,
            limit,
            offset,
        };
        // Canonical bytes are unique; re-encoding must reproduce the input.
        let reencoded = wcb::encode(&selector.to_wcb_value(), limits)?;
        if reencoded != bytes {
            return Err(Error::non_canonical(
                "selector bytes are not in canonical normalized form",
            ));
        }
        Ok(selector)
    }
}

fn expr_to_value(expr: &Expr) -> Value {
    match expr {
        Expr::True => Value::Array(vec![Value::Str("true".to_owned())]),
        Expr::False => Value::Array(vec![Value::Str("false".to_owned())]),
        Expr::Has(field) => Value::Array(vec![
            Value::Str("has".to_owned()),
            Value::Str(field.as_string()),
        ]),
        Expr::Tag(name) => {
            Value::Array(vec![Value::Str("tag".to_owned()), Value::Str(name.clone())])
        }
        Expr::Relation { predicate, target } => Value::Array(vec![
            Value::Str("relation".to_owned()),
            Value::Str(predicate.clone()),
            Value::Bytes(target.as_bytes().to_vec()),
        ]),
        Expr::Compare { field, op, value } => Value::Array(vec![
            Value::Str(op.discriminator().to_owned()),
            Value::Str(field.as_string()),
            literal_to_value(value),
        ]),
        Expr::And(items) => Value::Array(vec![
            Value::Str("and".to_owned()),
            Value::Array(items.iter().map(expr_to_value).collect()),
        ]),
        Expr::Or(items) => Value::Array(vec![
            Value::Str("or".to_owned()),
            Value::Array(items.iter().map(expr_to_value).collect()),
        ]),
        Expr::Not(inner) => Value::Array(vec![Value::Str("not".to_owned()), expr_to_value(inner)]),
    }
}

fn literal_to_value(literal: &Literal) -> Value {
    match literal {
        Literal::Null => Value::Null,
        Literal::Bool(value) => Value::Bool(*value),
        Literal::U64(value) => Value::U64(*value),
        Literal::I64(value) => Value::I64(*value),
        Literal::Str(value) => Value::Str(value.clone()),
    }
}

fn literal_from_value(value: &Value) -> Result<Literal> {
    match value {
        Value::Null => Ok(Literal::Null),
        Value::Bool(value) => Ok(Literal::Bool(*value)),
        Value::U64(value) => Ok(Literal::U64(*value)),
        Value::I64(value) => Ok(Literal::I64(*value)),
        Value::Str(value) => Ok(Literal::Str(value.clone())),
        _ => Err(Error::non_canonical(
            "selector literal must be text, integer, bool, or null",
        )),
    }
}

fn text_of(value: &Value, what: &str) -> Result<String> {
    match value {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(Error::non_canonical(format!("{what} must be text"))),
    }
}

fn expr_from_value(value: &Value, limits: &Limits, operands: &mut u64, depth: u64) -> Result<Expr> {
    if depth > limits.max_selector_depth {
        return Err(Error::limit_exceeded(
            "selector expression nesting too deep",
        ));
    }
    *operands = operands
        .checked_add(1)
        .ok_or_else(|| Error::limit_exceeded("selector operand count overflow"))?;
    if *operands > limits.max_selector_operands {
        return Err(Error::limit_exceeded(format!(
            "selector operand count {} exceeds limit {}",
            *operands, limits.max_selector_operands
        )));
    }
    let items = match value {
        Value::Array(items) => items,
        _ => return Err(Error::non_canonical("selector expression must be an array")),
    };
    let discriminator = items
        .first()
        .ok_or_else(|| Error::non_canonical("empty selector expression"))?;
    let discriminator = text_of(discriminator, "selector discriminator")?;
    match discriminator.as_str() {
        "true" => {
            if items.len() != 1 {
                return Err(Error::non_canonical("`true` expression takes no operands"));
            }
            Ok(Expr::True)
        }
        "false" => {
            if items.len() != 1 {
                return Err(Error::non_canonical("`false` expression takes no operands"));
            }
            Ok(Expr::False)
        }
        "has" => {
            require_arity(items, 2, "has")?;
            Ok(Expr::Has(field_from_canonical_text(
                &text_of(&items[1], "has field")?,
                limits,
            )?))
        }
        "tag" => {
            require_arity(items, 2, "tag")?;
            Ok(Expr::Tag(text_of(&items[1], "tag name")?))
        }
        "relation" => {
            require_arity(items, 3, "relation")?;
            let predicate = text_of(&items[1], "relation predicate")?;
            let target = match &items[2] {
                Value::Bytes(bytes) if bytes.len() == 32 => {
                    Hash32::from_slice(bytes).map_err(|e| e.context("relation target"))?
                }
                Value::Bytes(_) => {
                    return Err(Error::non_canonical(
                        "relation target must be exactly 32 bytes",
                    ))
                }
                _ => {
                    return Err(Error::non_canonical(
                        "relation target must be a WCB bytes value",
                    ))
                }
            };
            Ok(Expr::Relation { predicate, target })
        }
        "attr_eq" | "attr_ne" | "attr_lt" | "attr_le" | "attr_gt" | "attr_ge" => {
            require_arity(items, 3, "comparison")?;
            let op = match discriminator.as_str() {
                "attr_eq" => CmpOp::Eq,
                "attr_ne" => CmpOp::Ne,
                "attr_lt" => CmpOp::Lt,
                "attr_le" => CmpOp::Le,
                "attr_gt" => CmpOp::Gt,
                _ => CmpOp::Ge,
            };
            let field =
                field_from_canonical_text(&text_of(&items[1], "comparison field")?, limits)?;
            let value = literal_from_value(&items[2])?;
            Ok(Expr::Compare { field, op, value })
        }
        "and" | "or" => {
            require_arity(items, 2, "boolean operator")?;
            let nested = match &items[1] {
                Value::Array(nested) => nested,
                _ => return Err(Error::non_canonical("and/or operands must be a WCB array")),
            };
            if nested.len() < 2 {
                return Err(Error::non_canonical("and/or requires at least 2 operands"));
            }
            let mut parsed = Vec::with_capacity(nested.len());
            for item in nested {
                let child = expr_from_value(item, limits, operands, depth + 1)?;
                // Nested same-operator nodes must have been flattened.
                if (discriminator == "and" && matches!(child, Expr::And(_)))
                    || (discriminator == "or" && matches!(child, Expr::Or(_)))
                {
                    return Err(Error::non_canonical(
                        "nested same-operator and/or must be flattened",
                    ));
                }
                parsed.push(child);
            }
            if discriminator == "and" {
                Ok(Expr::And(parsed))
            } else {
                Ok(Expr::Or(parsed))
            }
        }
        "not" => {
            require_arity(items, 2, "not")?;
            let child = expr_from_value(&items[1], limits, operands, depth + 1)?;
            match child {
                Expr::True | Expr::False => Err(Error::non_canonical(
                    "not true/false must be normalized to false/true",
                )),
                Expr::Not(_) => Err(Error::non_canonical("double negation must be removed")),
                other => Ok(Expr::Not(Box::new(other))),
            }
        }
        other => Err(Error::non_canonical(format!(
            "unknown selector discriminator `{other}`"
        ))),
    }
}

fn require_arity(items: &[Value], expected: usize, what: &str) -> Result<()> {
    if items.len() != expected {
        return Err(Error::non_canonical(format!(
            "{what} expression has {} elements, expected {expected}",
            items.len()
        )));
    }
    Ok(())
}

fn field_from_canonical_text(text: &str, limits: &Limits) -> Result<Field> {
    let segments: Vec<String> = text.split('.').map(str::to_owned).collect();
    if segments.is_empty() || segments.iter().any(|segment| segment.is_empty()) {
        return Err(Error::non_canonical("empty field path segment"));
    }
    if segments.len() as u64 > limits.max_field_segments {
        return Err(Error::limit_exceeded("field path has too many segments"));
    }
    for segment in &segments {
        if !is_valid_ident(segment) || is_keyword(segment) {
            return Err(Error::non_canonical(format!(
                "invalid field path segment `{segment}`"
            )));
        }
    }
    Ok(Field::new(segments))
}

fn is_valid_ident(text: &str) -> bool {
    let mut bytes = text.bytes();
    match bytes.next() {
        Some(first) if first.is_ascii_alphabetic() || first == b'_' => {}
        _ => return false,
    }
    bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

// ---------------------------------------------------------------------------
// Lexer

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Ident(String),
    Number(String),
    Str(String),
    LParen,
    RParen,
    Comma,
    Dot,
    Colon,
    EqEq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
    Eof,
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Lexer { src, pos: 0 }
    }

    fn tokenize(mut self) -> Result<Vec<(Token, usize)>> {
        let mut tokens = Vec::new();
        loop {
            let (token, pos) = self.next_token()?;
            let end = matches!(token, Token::Eof);
            tokens.push((token, pos));
            if end {
                return Ok(tokens);
            }
        }
    }

    fn error(&self, pos: usize, message: impl Into<String>) -> Error {
        Error::parse_at(message, pos)
    }

    fn peek_byte(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn bump_byte(&mut self) -> Option<u8> {
        let byte = self.peek_byte()?;
        self.pos += 1;
        Some(byte)
    }

    fn take_while<F: Fn(u8) -> bool>(&mut self, predicate: F) -> &'a str {
        let start = self.pos;
        while let Some(byte) = self.peek_byte() {
            if predicate(byte) {
                self.pos += 1;
            } else {
                break;
            }
        }
        &self.src[start..self.pos]
    }

    fn next_token(&mut self) -> Result<(Token, usize)> {
        loop {
            match self.peek_byte() {
                Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') => {
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let start = self.pos;
        let byte = match self.peek_byte() {
            Some(byte) => byte,
            None => return Ok((Token::Eof, start)),
        };
        let token = match byte {
            b'(' => {
                self.pos += 1;
                Token::LParen
            }
            b')' => {
                self.pos += 1;
                Token::RParen
            }
            b',' => {
                self.pos += 1;
                Token::Comma
            }
            b'.' => {
                self.pos += 1;
                Token::Dot
            }
            b':' => {
                self.pos += 1;
                Token::Colon
            }
            b'=' => {
                self.pos += 1;
                if self.bump_byte() != Some(b'=') {
                    return Err(self.error(start, "expected `==`"));
                }
                Token::EqEq
            }
            b'!' => {
                self.pos += 1;
                if self.bump_byte() != Some(b'=') {
                    return Err(self.error(start, "expected `!=`"));
                }
                Token::Ne
            }
            b'<' => {
                self.pos += 1;
                if self.peek_byte() == Some(b'=') {
                    self.pos += 1;
                    Token::Le
                } else {
                    Token::Lt
                }
            }
            b'>' => {
                self.pos += 1;
                if self.peek_byte() == Some(b'=') {
                    self.pos += 1;
                    Token::Ge
                } else {
                    Token::Gt
                }
            }
            b'"' => self.lex_string()?,
            b'-' => {
                self.pos += 1;
                if !self
                    .peek_byte()
                    .map(|b| b.is_ascii_digit())
                    .unwrap_or(false)
                {
                    return Err(self.error(start, "expected digits after `-`"));
                }
                let digits = self.take_while(|b| b.is_ascii_digit());
                Token::Number(format!("-{digits}"))
            }
            b if b.is_ascii_digit() => {
                let text = self.take_while(|b| b.is_ascii_alphanumeric() || b == b'_');
                Token::Number(text.to_owned())
            }
            b if b.is_ascii_alphabetic() || b == b'_' => {
                let text = self.take_while(|b| b.is_ascii_alphanumeric() || b == b'_');
                Token::Ident(text.to_owned())
            }
            other => return Err(self.error(start, format!("unexpected character 0x{other:02x}"))),
        };
        Ok((token, start))
    }

    fn lex_string(&mut self) -> Result<Token> {
        let start = self.pos;
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let byte = match self.peek_byte() {
                Some(byte) => byte,
                None => return Err(self.error(start, "unterminated string")),
            };
            match byte {
                b'"' => {
                    self.pos += 1;
                    return Ok(Token::Str(out));
                }
                b'\\' => {
                    self.pos += 1;
                    self.lex_escape(&mut out)?;
                }
                0x00..=0x1f => return Err(self.error(self.pos, "raw control character in string")),
                _ => {
                    let ch = self.src[self.pos..]
                        .chars()
                        .next()
                        .ok_or_else(|| self.error(self.pos, "invalid UTF-8 in string"))?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn lex_escape(&mut self, out: &mut String) -> Result<()> {
        let pos = self.pos;
        let byte = self
            .bump_byte()
            .ok_or_else(|| self.error(pos, "unterminated escape"))?;
        match byte {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{0008}'),
            b'f' => out.push('\u{000c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let high = self.read_hex4()?;
                let code = if (0xD800..=0xDBFF).contains(&high) {
                    if self.peek_byte() != Some(b'\\') {
                        return Err(self.error(pos, "high surrogate without low surrogate"));
                    }
                    self.pos += 1;
                    if self.bump_byte() != Some(b'u') {
                        return Err(self.error(pos, "high surrogate without low surrogate"));
                    }
                    let low = self.read_hex4()?;
                    if !(0xDC00..=0xDFFF).contains(&low) {
                        return Err(self.error(pos, "invalid low surrogate"));
                    }
                    0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
                } else if (0xDC00..=0xDFFF).contains(&high) {
                    return Err(self.error(pos, "lone low surrogate"));
                } else {
                    high
                };
                let ch = char::from_u32(code)
                    .ok_or_else(|| self.error(pos, "invalid unicode scalar value"))?;
                out.push(ch);
            }
            other => return Err(self.error(pos, format!("invalid escape `\\{}`", other as char))),
        }
        Ok(())
    }

    fn read_hex4(&mut self) -> Result<u32> {
        let start = self.pos;
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = self
                .bump_byte()
                .ok_or_else(|| self.error(start, "truncated \\u escape"))?;
            let digit = match byte {
                b'0'..=b'9' => u32::from(byte - b'0'),
                b'a'..=b'f' => u32::from(byte - b'a' + 10),
                b'A'..=b'F' => u32::from(byte - b'A' + 10),
                _ => return Err(self.error(start, "invalid hex digit in \\u escape")),
            };
            value = (value << 4) | digit;
        }
        Ok(value)
    }
}

// ---------------------------------------------------------------------------
// Parser

struct Parser {
    tokens: Vec<(Token, usize)>,
    index: usize,
    limits: Limits,
    operands: u64,
}

impl Parser {
    fn peek(&self) -> &(Token, usize) {
        &self.tokens[self.index.min(self.tokens.len() - 1)]
    }

    fn peek_token(&self) -> &Token {
        &self.peek().0
    }

    fn peek_position(&self) -> usize {
        self.peek().1
    }

    fn bump(&mut self) -> (Token, usize) {
        let token = self.tokens[self.index.min(self.tokens.len() - 1)].clone();
        if self.index + 1 < self.tokens.len() {
            self.index += 1;
        }
        token
    }

    fn error(&self, pos: usize, message: impl Into<String>) -> Error {
        Error::parse_at(message, pos)
    }

    fn check_depth(&self, depth: u32) -> Result<()> {
        if u64::from(depth) > self.limits.max_selector_depth {
            return Err(Error::limit_exceeded(format!(
                "selector nesting depth {depth} exceeds limit {}",
                self.limits.max_selector_depth
            )));
        }
        Ok(())
    }

    fn bump_operand(&mut self) -> Result<()> {
        self.operands = self
            .operands
            .checked_add(1)
            .ok_or_else(|| Error::limit_exceeded("selector operand count overflow"))?;
        if self.operands > self.limits.max_selector_operands {
            return Err(Error::limit_exceeded(format!(
                "selector operand count {} exceeds limit {}",
                self.operands, self.limits.max_selector_operands
            )));
        }
        Ok(())
    }

    fn eat_keyword(&mut self, keyword: &str) -> bool {
        if let Token::Ident(word) = self.peek_token() {
            if word == keyword {
                self.bump();
                return true;
            }
        }
        false
    }

    fn expect_keyword(&mut self, keyword: &str) -> Result<usize> {
        let pos = self.peek_position();
        match self.bump() {
            (Token::Ident(word), _) if word == keyword => Ok(pos),
            (_, _) => Err(self.error(pos, format!("expected `{keyword}`"))),
        }
    }

    fn expect_token(&mut self, expected: Token) -> Result<usize> {
        let pos = self.peek_position();
        let (token, token_pos) = self.bump();
        if token == expected {
            Ok(pos)
        } else {
            Err(self.error(token_pos, format!("expected {expected:?}")))
        }
    }

    fn parse_query(&mut self, namespace: &str) -> Result<Selector> {
        self.expect_keyword("select")?;
        let predicate = self.parse_or(0)?;
        let predicate = normalize(predicate);

        let mut order = Vec::new();
        if self.eat_keyword("order") {
            self.expect_keyword("by")?;
            let mut seen = std::collections::HashSet::new();
            loop {
                let field = self.parse_field()?;
                let desc = if self.eat_keyword("desc") {
                    true
                } else {
                    self.eat_keyword("asc");
                    false
                };
                if !seen.insert(field.as_string()) {
                    return Err(self.error(self.peek_position(), "duplicate order field"));
                }
                order.push(OrderItem { field, desc });
                if matches!(self.peek_token(), Token::Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
        }

        let limit = if self.eat_keyword("limit") {
            Some(self.parse_u64("limit")?)
        } else {
            None
        };
        let offset = if self.eat_keyword("offset") {
            Some(self.parse_u64("offset")?)
        } else {
            None
        };
        if !matches!(self.peek_token(), Token::Eof) {
            return Err(self.error(
                self.peek_position(),
                "unexpected token after selector query",
            ));
        }

        Ok(Selector {
            namespace: namespace.to_owned(),
            predicate,
            order,
            limit,
            offset: offset.filter(|value| *value != 0),
        })
    }

    fn parse_u64(&mut self, clause: &str) -> Result<u64> {
        let (token, pos) = self.bump();
        match token {
            Token::Number(text) => {
                if text.starts_with('-') {
                    return Err(self.error(pos, format!("{clause} must not be negative")));
                }
                parse_digits_u64(&text)
                    .map_err(|message| self.error(pos, format!("invalid {clause}: {message}")))
            }
            _ => Err(self.error(pos, format!("expected integer after `{clause}`"))),
        }
    }

    fn parse_or(&mut self, depth: u32) -> Result<Expr> {
        self.check_depth(depth)?;
        let mut items = vec![self.parse_and(depth)?];
        while self.eat_keyword("or") {
            items.push(self.parse_and(depth)?);
        }
        if items.len() == 1 {
            match items.pop() {
                Some(single) => Ok(single),
                None => Err(self.error(self.peek_position(), "empty expression")),
            }
        } else {
            self.bump_operand()?;
            Ok(Expr::Or(items))
        }
    }

    fn parse_and(&mut self, depth: u32) -> Result<Expr> {
        self.check_depth(depth)?;
        let mut items = vec![self.parse_unary(depth)?];
        while self.eat_keyword("and") {
            items.push(self.parse_unary(depth)?);
        }
        if items.len() == 1 {
            match items.pop() {
                Some(single) => Ok(single),
                None => Err(self.error(self.peek_position(), "empty expression")),
            }
        } else {
            self.bump_operand()?;
            Ok(Expr::And(items))
        }
    }

    fn parse_unary(&mut self, depth: u32) -> Result<Expr> {
        self.check_depth(depth)?;
        if self.eat_keyword("not") {
            let inner = self.parse_unary(depth + 1)?;
            self.bump_operand()?;
            Ok(Expr::Not(Box::new(inner)))
        } else {
            self.parse_primary(depth)
        }
    }

    fn parse_primary(&mut self, depth: u32) -> Result<Expr> {
        self.check_depth(depth)?;
        if matches!(self.peek_token(), Token::LParen) {
            self.bump();
            let inner = self.parse_or(depth + 1)?;
            self.expect_token(Token::RParen)?;
            return Ok(inner);
        }
        if self.eat_keyword("true") {
            self.bump_operand()?;
            return Ok(Expr::True);
        }
        if self.eat_keyword("false") {
            self.bump_operand()?;
            return Ok(Expr::False);
        }
        if self.eat_keyword("has") {
            self.expect_token(Token::LParen)?;
            let field = self.parse_field()?;
            self.expect_token(Token::RParen)?;
            self.bump_operand()?;
            return Ok(Expr::Has(field));
        }
        if self.eat_keyword("tag") {
            self.expect_token(Token::LParen)?;
            let name = self.parse_name_argument("tag argument")?;
            self.expect_token(Token::RParen)?;
            self.bump_operand()?;
            return Ok(Expr::Tag(name));
        }
        if self.eat_keyword("relation") {
            self.expect_token(Token::LParen)?;
            let predicate = self.parse_name_argument("relation predicate")?;
            self.expect_token(Token::Comma)?;
            let target = self.parse_object_ref()?;
            self.expect_token(Token::RParen)?;
            self.bump_operand()?;
            return Ok(Expr::Relation { predicate, target });
        }
        let field = self.parse_field()?;
        let op = self.parse_cmp()?;
        let value = self.parse_literal()?;
        self.bump_operand()?;
        Ok(Expr::Compare { field, op, value })
    }

    fn parse_field(&mut self) -> Result<Field> {
        let (token, pos) = self.bump();
        let mut segments = Vec::new();
        match token {
            Token::Ident(segment) => {
                if is_keyword(&segment) {
                    return Err(self.error(
                        pos,
                        format!("reserved keyword `{segment}` cannot be a field"),
                    ));
                }
                segments.push(segment);
            }
            other => return Err(self.error(pos, format!("expected field name, found {other:?}"))),
        }
        while matches!(self.peek_token(), Token::Dot) {
            self.bump();
            let (token, pos) = self.bump();
            match token {
                Token::Ident(segment) => {
                    if is_keyword(&segment) {
                        return Err(self.error(
                            pos,
                            format!("reserved keyword `{segment}` cannot be a field"),
                        ));
                    }
                    segments.push(segment);
                }
                other => {
                    return Err(
                        self.error(pos, format!("expected field path segment, found {other:?}"))
                    )
                }
            }
        }
        if segments.len() as u64 > self.limits.max_field_segments {
            return Err(Error::limit_exceeded(format!(
                "field path has {} segments, limit {}",
                segments.len(),
                self.limits.max_field_segments
            )));
        }
        Ok(Field::new(segments))
    }

    /// The grammar spells these arguments as strings, but the worked example in
    /// WEFT-SELECTORS §7 writes `relation(derived_from, ...)` bare. Accept a
    /// non-keyword identifier as an equivalent text argument; the canonical
    /// encoding is identical. Recorded in NOTES.md.
    fn parse_name_argument(&mut self, what: &str) -> Result<String> {
        let (token, pos) = self.bump();
        match token {
            Token::Str(value) => Ok(value),
            Token::Ident(value) if !is_keyword(&value) => Ok(value),
            other => Err(self.error(
                pos,
                format!("expected string or identifier for {what}, found {other:?}"),
            )),
        }
    }

    fn parse_cmp(&mut self) -> Result<CmpOp> {
        let (token, pos) = self.bump();
        match token {
            Token::EqEq => Ok(CmpOp::Eq),
            Token::Ne => Ok(CmpOp::Ne),
            Token::Lt => Ok(CmpOp::Lt),
            Token::Le => Ok(CmpOp::Le),
            Token::Gt => Ok(CmpOp::Gt),
            Token::Ge => Ok(CmpOp::Ge),
            other => Err(self.error(
                pos,
                format!("expected comparison operator, found {other:?}"),
            )),
        }
    }

    fn parse_literal(&mut self) -> Result<Literal> {
        let (token, pos) = self.bump();
        match token {
            Token::Str(value) => Ok(Literal::Str(value)),
            Token::Number(text) => parse_integer_literal(&text)
                .map_err(|message| self.error(pos, format!("invalid integer literal: {message}"))),
            Token::Ident(word) if word == "true" => Ok(Literal::Bool(true)),
            Token::Ident(word) if word == "false" => Ok(Literal::Bool(false)),
            Token::Ident(word) if word == "null" => Ok(Literal::Null),
            other => Err(self.error(
                pos,
                format!("expected literal (string, int, bool, null), found {other:?}"),
            )),
        }
    }

    fn parse_object_ref(&mut self) -> Result<Hash32> {
        if let Token::Ident(word) = self.peek_token() {
            if word == "obj"
                && matches!(
                    self.tokens.get(self.index + 1).map(|(token, _)| token),
                    Some(Token::Colon)
                )
            {
                self.bump(); // obj
                self.bump(); // colon
                let (token, pos) = self.bump();
                return parse_hex_token(&token, pos);
            }
        }
        let (token, pos) = self.bump();
        parse_hex_token(&token, pos)
    }
}

fn parse_hex_token(token: &Token, pos: usize) -> Result<Hash32> {
    let text = match token {
        Token::Ident(text) | Token::Number(text) => text,
        _ => return Err(Error::parse_at("expected 64-digit object reference", pos)),
    };
    if text.len() != 64 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::parse_at(
            "object reference must be exactly 64 hex digits",
            pos,
        ));
    }
    Hash32::from_hex(text).map_err(|error| error.context("object reference"))
}

fn parse_digits_u64(text: &str) -> std::result::Result<u64, String> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err("expected decimal digits".to_owned());
    }
    let mut value: u64 = 0;
    for byte in text.bytes() {
        let digit = u64::from(byte - b'0');
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| "integer exceeds u64".to_owned())?;
    }
    Ok(value)
}

fn parse_integer_literal(text: &str) -> std::result::Result<Literal, String> {
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err("expected decimal digits".to_owned());
    }
    let mut magnitude: u128 = 0;
    for byte in digits.bytes() {
        magnitude = magnitude
            .checked_mul(10)
            .and_then(|v| v.checked_add(u128::from(byte - b'0')))
            .ok_or_else(|| "integer is too large".to_owned())?;
    }
    if negative {
        if magnitude == 0 {
            return Ok(Literal::U64(0));
        }
        if magnitude > (1u128 << 63) {
            return Err("negative integer is below -2^63".to_owned());
        }
        if magnitude == (1u128 << 63) {
            Ok(Literal::I64(i64::MIN))
        } else {
            Ok(Literal::I64(-(magnitude as i64)))
        }
    } else {
        if magnitude > u128::from(u64::MAX) {
            return Err("integer exceeds u64".to_owned());
        }
        Ok(Literal::U64(magnitude as u64))
    }
}

/// Apply the only normalizations allowed by WEFT-SELECTORS §2.
fn normalize(expr: Expr) -> Expr {
    match expr {
        Expr::Not(inner) => match normalize(*inner) {
            Expr::True => Expr::False,
            Expr::False => Expr::True,
            Expr::Not(double) => *double,
            other => Expr::Not(Box::new(other)),
        },
        Expr::And(items) => {
            let mut flattened = Vec::new();
            for item in items {
                match normalize(item) {
                    Expr::And(nested) => flattened.extend(nested),
                    other => flattened.push(other),
                }
            }
            Expr::And(flattened)
        }
        Expr::Or(items) => {
            let mut flattened = Vec::new();
            for item in items {
                match normalize(item) {
                    Expr::Or(nested) => flattened.extend(nested),
                    other => flattened.push(other),
                }
            }
            Expr::Or(flattened)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorClass;

    #[test]
    fn canonical_example() {
        let source = "select attrs.project == \"demo\" and relation(derived_from, \
                      obj:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff) \
                      order by attrs.seq desc limit 10";
        let selector = parse(source).unwrap();
        assert_eq!(selector.namespace, "default");
        assert_eq!(selector.limit, Some(10));
        assert_eq!(selector.order.len(), 1);
        assert!(selector.order[0].desc);
        let bytes = selector.canonical_bytes().unwrap();
        let decoded = Selector::from_canonical_bytes(&bytes, &Limits::default()).unwrap();
        assert_eq!(decoded, selector);
    }

    #[test]
    fn whitespace_and_hex_case_identity() {
        let a = "select attrs.x == 1 order by a";
        let b = "  SELECT is invalid";
        assert!(parse(b).is_err());
        let c = "select attrs.x==1 order by a asc";
        let d = "select\nattrs.x == 01\torder by a";
        assert_eq!(
            parse(a).unwrap().selector_id().unwrap(),
            parse(c).unwrap().selector_id().unwrap()
        );
        assert_eq!(
            parse(a).unwrap().selector_id().unwrap(),
            parse(d).unwrap().selector_id().unwrap()
        );
        assert_eq!(
            canonical_bytes("select attrs.x == 1 offset 0").unwrap(),
            canonical_bytes("select attrs.x == 1").unwrap()
        );
        let e =
            "select relation(p, AB112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)";
        let f = "select relation(p, obj:ab112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)";
        assert_eq!(
            parse(e).unwrap().selector_id().unwrap(),
            parse(f).unwrap().selector_id().unwrap()
        );
    }

    #[test]
    fn operand_order_changes_identity() {
        let a = parse("select a == 1 and b == 2").unwrap();
        let b = parse("select b == 2 and a == 1").unwrap();
        assert_ne!(a.selector_id().unwrap(), b.selector_id().unwrap());
    }

    #[test]
    fn normalization() {
        let selector = parse("select not not true").unwrap();
        assert_eq!(selector.predicate, Expr::True);
        let selector = parse("select not false").unwrap();
        assert_eq!(selector.predicate, Expr::True);
        let selector = parse("select a == 1 and (b == 2 and c == 3)").unwrap();
        match selector.predicate {
            Expr::And(items) => assert_eq!(items.len(), 3),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn grammar_forms() {
        for source in [
            "select true",
            "select false",
            "select has(attrs.x)",
            "select tag(\"demo\")",
            "select relation(p, 00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)",
            "select a == \"x\"",
            "select a != 1",
            "select a < -2",
            "select a <= 3 and b >= 4 or not c > 5",
            "select a == null order by a, b desc limit 5 offset 2",
        ] {
            parse(source).unwrap_or_else(|e| panic!("{source}: {e}"));
        }
    }

    #[test]
    fn error_positions_and_escapes() {
        let err = parse("select a = 1").unwrap_err();
        assert_eq!(err.class(), ErrorClass::Parse);
        assert!(err.position().is_some());
        let err = parse("select a == \"\\uD800\"").unwrap_err();
        assert_eq!(err.class(), ErrorClass::Parse);
        let ok = parse("select a == \"\\uD83D\\uDE00\"").unwrap();
        match ok.predicate {
            Expr::Compare {
                value: Literal::Str(value),
                ..
            } => assert_eq!(value, "😀"),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(
            parse("select a == \"\\q\"").unwrap_err().class(),
            ErrorClass::Parse
        );
        assert_eq!(
            parse("select a == \"raw\ncontrol\"").unwrap_err().class(),
            ErrorClass::Parse
        );
    }

    #[test]
    fn limits() {
        let limits = Limits::default().with_max_selector_bytes(4);
        assert_eq!(
            parse_with("select true", DEFAULT_NAMESPACE, &limits)
                .unwrap_err()
                .class(),
            ErrorClass::LimitExceeded
        );
        let limits = Limits::default().with_max_field_segments(1);
        assert_eq!(
            parse_with("select a.b == 1", DEFAULT_NAMESPACE, &limits)
                .unwrap_err()
                .class(),
            ErrorClass::LimitExceeded
        );
        let limits = Limits::default().with_max_selector_depth(1);
        assert_eq!(
            parse_with("select not not true", DEFAULT_NAMESPACE, &limits)
                .unwrap_err()
                .class(),
            ErrorClass::LimitExceeded
        );
    }

    #[test]
    fn non_canonical_rejected() {
        // Build a canonical selector then mutate the offset to zero.
        let selector = parse("select a == 1 offset 2").unwrap();
        let value = selector.to_wcb_value();
        let Value::Map(mut map) = value else { panic!() };
        map.insert("offset".to_owned(), Value::U64(0));
        let bytes = wcb::encode_default(&Value::Map(map)).unwrap();
        assert_eq!(
            Selector::from_canonical_bytes(&bytes, &Limits::default())
                .unwrap_err()
                .class(),
            ErrorClass::NonCanonical
        );
        // Wrong schema.
        let selector = parse("select true").unwrap();
        let Value::Map(mut map) = selector.to_wcb_value() else {
            panic!()
        };
        map.insert("schema".to_owned(), Value::Str("other".to_owned()));
        let bytes = wcb::encode_default(&Value::Map(map)).unwrap();
        assert_eq!(
            Selector::from_canonical_bytes(&bytes, &Limits::default())
                .unwrap_err()
                .class(),
            ErrorClass::UnknownSchema
        );
    }
}
