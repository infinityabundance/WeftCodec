//! No coding: the identity coder, used as the byte-count floor.

use crate::error::{Error, Result};

use super::Coder;

/// The identity coder. `decode` requires the stream length to equal
/// `expected_len` exactly.
#[derive(Clone, Copy, Debug, Default)]
pub struct Raw;

impl Coder for Raw {
    fn id(&self) -> &'static str {
        "raw"
    }

    fn encode(&self, input: &[u8]) -> Vec<u8> {
        input.to_vec()
    }

    fn decode(&self, encoded: &[u8], expected_len: usize) -> Result<Vec<u8>> {
        if encoded.len() != expected_len {
            return Err(Error::corrupt(format!(
                "raw stream is {} bytes but expected_len is {expected_len}",
                encoded.len()
            )));
        }
        Ok(encoded.to_vec())
    }

    fn decode_into(&self, encoded: &[u8], out: &mut [u8]) -> Result<()> {
        if encoded.len() != out.len() {
            return Err(Error::corrupt(format!(
                "raw stream is {} bytes but the output buffer is {} bytes",
                encoded.len(),
                out.len()
            )));
        }
        out.copy_from_slice(encoded);
        Ok(())
    }
}
