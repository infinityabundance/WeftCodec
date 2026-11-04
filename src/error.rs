//! Typed error classes for WeftCodec v1.
//!
//! The interoperable contract is the [`ErrorClass`], not the human-readable
//! message. Classes follow `spec/WEFTCODEC-FORMAT.md` §11. The selector grammar
//! adds `Parse` and `UnknownSchema` from `spec/WEFT-SELECTORS.md` §5; both are
//! carried in the same enum so callers have a single error type.

use std::fmt;

/// Normative error classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ErrorClass {
    /// Structural or byte error, including bad varints and bad tables.
    Malformed,
    /// Checksum, hash, or length mismatch.
    Corrupt,
    /// Major/minor not supported.
    UnsupportedVersion,
    /// Mode not implementable.
    UnsupportedMode,
    /// Unknown critical flag/feature.
    UnsupportedFeature,
    /// A configured resource bound would be exceeded.
    LimitExceeded,
    /// Dependency cannot be found.
    DependencyMissing,
    /// Authorization refuses the dependency.
    DependencyDenied,
    /// Resolved bytes fail hash or length verification.
    DependencyMismatch,
    /// Reconstruction dependency graph is cyclic.
    DependencyCycle,
    /// Structured bytes violate WCB canonical form.
    NonCanonical,
    /// Transport/read failure.
    Io,
    /// Selector grammar, escape, integer, or token error.
    Parse,
    /// Selector schema string is not `weft.selector.v1`.
    UnknownSchema,
}

impl ErrorClass {
    /// The normative class name, as used in conformance vectors and CLI output.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorClass::Malformed => "Malformed",
            ErrorClass::Corrupt => "Corrupt",
            ErrorClass::UnsupportedVersion => "UnsupportedVersion",
            ErrorClass::UnsupportedMode => "UnsupportedMode",
            ErrorClass::UnsupportedFeature => "UnsupportedFeature",
            ErrorClass::LimitExceeded => "LimitExceeded",
            ErrorClass::DependencyMissing => "DependencyMissing",
            ErrorClass::DependencyDenied => "DependencyDenied",
            ErrorClass::DependencyMismatch => "DependencyMismatch",
            ErrorClass::DependencyCycle => "DependencyCycle",
            ErrorClass::NonCanonical => "NonCanonical",
            ErrorClass::Io => "Io",
            ErrorClass::Parse => "Parse",
            ErrorClass::UnknownSchema => "UnknownSchema",
        }
    }
}

impl fmt::Display for ErrorClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A typed codec error. Carries a class, human text, and an optional byte
/// offset (used by the selector parser). Constructing or displaying an `Error`
/// never panics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    class: ErrorClass,
    message: String,
    position: Option<usize>,
}

impl Error {
    /// Construct an error with an explicit class and message.
    pub fn new(class: ErrorClass, message: impl Into<String>) -> Self {
        Error {
            class,
            message: message.into(),
            position: None,
        }
    }

    /// Construct an error with a byte position.
    pub fn at(class: ErrorClass, message: impl Into<String>, position: usize) -> Self {
        Error {
            class,
            message: message.into(),
            position: Some(position),
        }
    }

    /// The normative class.
    pub fn class(&self) -> ErrorClass {
        self.class
    }

    /// The human-readable message (without the class prefix).
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The byte position, when the error came from a parser.
    pub fn position(&self) -> Option<usize> {
        self.position
    }

    /// Attach or replace the byte position.
    pub fn with_position(mut self, position: usize) -> Self {
        self.position = Some(position);
        self
    }

    /// Attach context to the front of the message.
    pub fn context(mut self, context: impl AsRef<str>) -> Self {
        self.message = format!("{}: {}", context.as_ref(), self.message);
        self
    }

    pub fn malformed(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::Malformed, message)
    }
    pub fn corrupt(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::Corrupt, message)
    }
    pub fn unsupported_version(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::UnsupportedVersion, message)
    }
    pub fn unsupported_mode(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::UnsupportedMode, message)
    }
    pub fn unsupported_feature(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::UnsupportedFeature, message)
    }
    pub fn limit_exceeded(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::LimitExceeded, message)
    }
    pub fn dependency_missing(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::DependencyMissing, message)
    }
    pub fn dependency_denied(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::DependencyDenied, message)
    }
    pub fn dependency_mismatch(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::DependencyMismatch, message)
    }
    pub fn dependency_cycle(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::DependencyCycle, message)
    }
    pub fn non_canonical(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::NonCanonical, message)
    }
    pub fn io(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::Io, message)
    }
    pub fn parse(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::Parse, message)
    }
    pub fn parse_at(message: impl Into<String>, position: usize) -> Self {
        Self::at(ErrorClass::Parse, message, position)
    }
    pub fn unknown_schema(message: impl Into<String>) -> Self {
        Self::new(ErrorClass::UnknownSchema, message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.position {
            Some(pos) => write!(f, "{}: {} (at byte {})", self.class, self.message, pos),
            None => write!(f, "{}: {}", self.class, self.message),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::io(err.to_string())
    }
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
