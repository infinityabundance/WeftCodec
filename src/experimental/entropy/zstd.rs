//! `zstd` reference points at levels 3 and 19, via the crate's existing
//! dependency. No new dependency is introduced.

use crate::error::{Error, Result};

use super::{check_expected_len, Coder};

/// One zstd compression level.
#[derive(Clone, Copy, Debug)]
pub struct Zstd {
    level: i32,
}

impl Zstd {
    /// Construct a zstd coder at `level`.
    pub fn new(level: i32) -> Self {
        Zstd { level }
    }

    /// The level passed to the encoder.
    pub fn level(&self) -> i32 {
        self.level
    }
}

fn io_error(context: &str, error: std::io::Error) -> Error {
    Error::corrupt(format!("{context}: {error}"))
}

impl Coder for Zstd {
    fn id(&self) -> &'static str {
        match self.level {
            19 => "zstd-19",
            _ => "zstd-3",
        }
    }

    fn encode(&self, input: &[u8]) -> Vec<u8> {
        zstd::bulk::compress(input, self.level)
            .expect("zstd bulk compression is infallible on a slice")
    }

    fn decode(&self, encoded: &[u8], expected_len: usize) -> Result<Vec<u8>> {
        check_expected_len(expected_len)?;
        let mut out = vec![0u8; expected_len];
        self.decode_into(encoded, &mut out)?;
        Ok(out)
    }

    fn decode_into(&self, encoded: &[u8], out: &mut [u8]) -> Result<()> {
        let mut decompressor = zstd::bulk::Decompressor::new()
            .map_err(|error| io_error("zstd decompressor", error))?;
        let written = decompressor
            .decompress_to_buffer(encoded, out)
            .map_err(|error| io_error("zstd decode", error))?;
        if written != out.len() {
            return Err(Error::corrupt(format!(
                "zstd decoded {written} bytes but the caller expects {}",
                out.len()
            )));
        }
        Ok(())
    }
}
