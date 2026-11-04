//! Minimal unsigned LEB128 varints (WEFTCODEC-FORMAT §1).
//!
//! Encoding is unsigned LEB128 with mandatory minimality:
//! * at most 10 bytes;
//! * a multi-byte encoding MUST NOT end in `0x00`;
//! * the tenth byte MUST be `0x00` or `0x01`.
//!
//! Decoders return `NonCanonical` for non-minimal encodings and `Malformed`
//! for overlong or truncated ones.

use crate::error::{Error, Result};

/// Maximum encoded length of a `u64` varint.
pub const MAX_VARINT_LEN: usize = 10;

/// Number of bytes [`encode`] will write for `value`.
pub fn encoded_len(value: u64) -> usize {
    let mut v = value;
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

/// Append the minimal LEB128 encoding of `value` to `out`.
pub fn encode(value: u64, out: &mut Vec<u8>) {
    let mut v = value;
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Encode into a fixed buffer, returning the number of bytes written.
pub fn encode_to_buf(value: u64, buf: &mut [u8; MAX_VARINT_LEN]) -> usize {
    let mut v = value;
    let mut n = 0;
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf[n] = byte;
            return n + 1;
        }
        buf[n] = byte | 0x80;
        n += 1;
    }
}

/// Decode one varint at `*pos`, advancing the cursor on success.
///
/// On error the cursor is left unspecified; callers must abandon the parse.
pub fn decode(input: &[u8], pos: &mut usize) -> Result<u64> {
    let (value, next) = decode_at(input, *pos)?;
    *pos = next;
    Ok(value)
}

/// Decode one varint starting at `pos`, returning the value and next position.
pub fn decode_at(input: &[u8], pos: usize) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut offset = pos;
    let mut count = 0usize;
    loop {
        if offset >= input.len() {
            return Err(Error::malformed("truncated varint"));
        }
        let byte = input[offset];
        offset += 1;
        count += 1;
        if count > MAX_VARINT_LEN {
            return Err(Error::malformed("varint longer than 10 bytes"));
        }
        if count == MAX_VARINT_LEN {
            // Only the low bit can contribute to bit 63; anything else either
            // overflows u64 or is non-minimal.
            if byte > 0x01 {
                return Err(Error::malformed("varint overflows u64"));
            }
        }
        let low = u64::from(byte & 0x7f);
        if shift < 64 {
            value |= low << shift;
        } else if low != 0 {
            return Err(Error::malformed("varint overflows u64"));
        }
        if byte & 0x80 == 0 {
            if count > 1 && byte == 0x00 {
                return Err(Error::non_canonical("non-minimal varint"));
            }
            return Ok((value, offset));
        }
        shift += 7;
        if shift > 63 {
            return Err(Error::malformed("varint exceeds 10 bytes"));
        }
    }
}

/// Zigzag encoding for signed values: `(v << 1) ^ (v >> 63)`.
pub fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Inverse of [`zigzag_encode`].
pub fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorClass;

    #[test]
    fn roundtrip_boundaries() {
        for v in [
            0u64,
            1,
            127,
            128,
            129,
            255,
            256,
            16384,
            u32::MAX as u64,
            u64::MAX / 2,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let mut buf = Vec::new();
            encode(v, &mut buf);
            assert_eq!(buf.len(), encoded_len(v));
            let mut pos = 0;
            assert_eq!(decode(&buf, &mut pos).unwrap(), v);
            assert_eq!(pos, buf.len());
            let mut fixed = [0u8; MAX_VARINT_LEN];
            let n = encode_to_buf(v, &mut fixed);
            assert_eq!(&fixed[..n], &buf[..]);
        }
    }

    #[test]
    fn rejects_non_minimal() {
        // 0 encoded in two bytes.
        let mut pos = 0;
        let err = decode(&[0x80, 0x00], &mut pos).unwrap_err();
        assert_eq!(err.class(), ErrorClass::NonCanonical);

        // 1 encoded with a redundant zero group.
        let mut pos = 0;
        let err = decode(&[0x81, 0x00], &mut pos).unwrap_err();
        assert_eq!(err.class(), ErrorClass::NonCanonical);
    }

    #[test]
    fn rejects_overlong_and_truncated() {
        let mut pos = 0;
        assert_eq!(
            decode(&[0x80], &mut pos).unwrap_err().class(),
            ErrorClass::Malformed
        );
        let mut pos = 0;
        assert_eq!(
            decode(&[0xff; 11], &mut pos).unwrap_err().class(),
            ErrorClass::Malformed
        );
        let mut pos = 0;
        // 10th byte may only be 0x00/0x01.
        assert_eq!(
            decode(
                &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02],
                &mut pos
            )
            .unwrap_err()
            .class(),
            ErrorClass::Malformed
        );
        let mut pos = 0;
        assert_eq!(
            decode(
                &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01],
                &mut pos
            )
            .unwrap(),
            u64::MAX
        );
    }

    #[test]
    fn zigzag_roundtrip() {
        for v in [0i64, 1, -1, 2, -2, i64::MAX, i64::MIN, -123456789] {
            assert_eq!(zigzag_decode(zigzag_encode(v)), v);
        }
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
    }
}
