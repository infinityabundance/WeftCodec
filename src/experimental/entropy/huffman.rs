//! `huffman-canonical-v1` baseline (the shipped coder, imported unchanged).

use crate::error::Result;
use crate::huffman as shipped;

use super::{check_expected_len, Coder};

/// The frozen `huffman-canonical-v1` coder used by modes 9–15.
#[derive(Clone, Copy, Debug, Default)]
pub struct Huffman;

impl Coder for Huffman {
    fn id(&self) -> &'static str {
        shipped::CODER_ID
    }

    fn encode(&self, input: &[u8]) -> Vec<u8> {
        shipped::encode(input)
    }

    fn decode(&self, encoded: &[u8], expected_len: usize) -> Result<Vec<u8>> {
        check_expected_len(expected_len)?;
        shipped::decode(encoded, expected_len)
    }

    fn decode_into(&self, encoded: &[u8], out: &mut [u8]) -> Result<()> {
        let decoded = self.decode(encoded, out.len())?;
        out.copy_from_slice(&decoded);
        Ok(())
    }
}
