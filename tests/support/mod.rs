//! Shared helpers for integration tests (deterministic PRNG, frame fixups).
#![allow(dead_code)]

use weftcodec::dependency::MapResolver;
use weftcodec::hash::{hash_domain, DOMAIN_CHUNK};

/// Deterministic xorshift64 byte stream.
pub fn pattern(len: usize, mut state: u64) -> Vec<u8> {
    if state == 0 {
        state = 0x9e37_79b9_7f4a_7c15;
    }
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state & 0xff) as u8);
    }
    out
}

pub fn repeat(text: &[u8], len: usize) -> Vec<u8> {
    text.iter().copied().cycle().take(len).collect()
}

pub fn resolver_with(entries: &[Vec<u8>]) -> MapResolver {
    let mut resolver = MapResolver::new();
    for bytes in entries {
        resolver.insert_bytes(bytes.clone());
    }
    resolver
}

pub fn dep_hash(bytes: &[u8]) -> weftcodec::Hash32 {
    hash_domain(DOMAIN_CHUNK, bytes)
}

pub fn fix_trailer(frame: &mut [u8]) {
    let at = frame.len() - 4;
    let crc = crc32c::crc32c(&frame[..at]);
    frame[at..].copy_from_slice(&crc.to_le_bytes());
}

pub fn fix_header_and_trailer(frame: &mut [u8]) {
    let mut input = Vec::with_capacity(92);
    input.extend_from_slice(&frame[0..72]);
    input.extend_from_slice(&frame[76..96]);
    let crc = crc32c::crc32c(&input);
    frame[72..76].copy_from_slice(&crc.to_le_bytes());
    fix_trailer(frame);
}
