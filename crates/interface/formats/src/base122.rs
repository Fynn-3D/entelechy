//! Dense UTF-8 byte packing for `<script>` elements in an exported page:
//! seven bits per safe ASCII byte, and each byte unsafe in HTML or
//! script text paired with the seven bits after it in one two-byte UTF-8
//! scalar (`U+0100..=U+04FF`, or `U+0500..=U+0507` when none follows),
//! keeping the stream at eight UTF-8 bytes per seven input bytes.

/// Seven-bit values unsafe verbatim in an HTML `<script>` or string:
/// NUL, LF, CR, `"`, `&`, `<`, `>`, `\`.
const BAD: [u8; 8] = [0, 10, 13, 34, 38, 60, 62, 92];

const UNSAFE: u8 = 0xff;

const BAD_SLOT: [u8; 128] = {
    let mut table = [UNSAFE; 128];
    let mut i = 0;
    while i < BAD.len() {
        table[BAD[i] as usize] = i as u8;
        i += 1;
    }
    table
};

/// Packs `bytes` into UTF-8 base122 safe inside an HTML `<script>` element.
pub fn base122(bytes: &[u8]) -> String {
    let total_bits = bytes.len() * 8;
    let mut bit_at = 0;
    let mut next_seven = || -> Option<u8> {
        if bit_at >= total_bits {
            return None;
        }
        let byte_at = bit_at / 8;
        let bit_off = bit_at % 8;
        bit_at += 7;
        let hi = bytes[byte_at] as u16;
        let lo = bytes.get(byte_at + 1).copied().unwrap_or(0) as u16;
        let word = (hi << 8) | lo;
        Some(((word >> (9 - bit_off)) & 0x7f) as u8)
    };
    let mut out = String::with_capacity(bytes.len() * 8 / 7 + 2);
    while let Some(first) = next_seven() {
        let slot = BAD_SLOT[first as usize];
        if slot == UNSAFE {
            out.push(first as char);
        } else if let Some(second) = next_seven() {
            let point = 0x100 + ((slot as u32) << 7) + (second as u32);
            out.push(char::from_u32(point).expect("U+0100..=U+04FF"));
        } else {
            let point = 0x500 + (slot as u32);
            out.push(char::from_u32(point).expect("U+0500..=U+0507"));
        }
    }
    out
}

/// Unpacks UTF-8 [`base122`] back into the original bytes.
pub fn unbase122(text: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(text.len() * 7 / 8 + 2);
    let mut acc = 0u32;
    let mut bits = 0u32;
    let mut push = |seven: u8| {
        acc = (acc << 7) | (seven as u32);
        bits += 7;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    };
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        let code = ch as u32;
        if code < 128 {
            if BAD_SLOT[code as usize] != UNSAFE {
                return Err(format!("unescaped byte {code} in base122"));
            }
            push(code as u8);
        } else if (0x100..0x500).contains(&code) {
            let pair = code - 0x100;
            push(BAD[(pair >> 7) as usize]);
            push((pair & 0x7f) as u8);
        } else if (0x500..0x508).contains(&code) {
            if chars.peek().is_some() {
                return Err("trailing base122 escape before end of stream".into());
            }
            push(BAD[(code - 0x500) as usize]);
        } else {
            return Err(format!("`{ch}` is not base122"));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base122_round_trips_and_avoids_unsafe_bytes() {
        for len in 0..=64 {
            let sample: Vec<u8> = (0..len).map(|i| (i * 37 + 13) as u8).collect();
            let packed = base122(&sample);
            assert_eq!(unbase122(&packed).unwrap(), sample, "len {len}");
            for bad in BAD {
                assert!(!packed.as_bytes().contains(&bad), "contains {bad}");
            }
            assert!(
                packed.len() <= (sample.len() * 8).div_ceil(7) + 1,
                "len {} -> {} bytes",
                sample.len(),
                packed.len()
            );
        }
        // Runs of unsafe bytes, zeroes, ones, and every byte value.
        for pattern in [
            vec![0u8; 31],
            vec![0xffu8; 31],
            BAD.repeat(17),
            (0..=255u8).cycle().take(2048).collect(),
        ] {
            let packed = base122(&pattern);
            assert_eq!(unbase122(&packed).unwrap(), pattern);
            for bad in BAD {
                assert!(!packed.as_bytes().contains(&bad), "contains {bad}");
            }
        }
        assert!(unbase122("<").is_err());
        assert!(unbase122("\u{0600}").is_err());
        assert!(unbase122("\u{0500}a").is_err());
    }
}
