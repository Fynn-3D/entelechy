//! How a scalar is kept in memory. A Kernel computes in 32-bit words —
//! `Float`, `Int`, `Bool` — and a Field may keep a scalar narrower than
//! that: widened when it is read, narrowed when it is written. What a
//! Kernel sees is a value; an encoding is only storage.
//!
//! The rules here are the rules: the interpreter calls them, the JIT
//! calls them or emits the same operations, and the GPU emits the same
//! integer arithmetic, so a narrowed value is the same bits everywhere.
//!
//! - An integer keeps its low bits and reads back sign- or zero-extended,
//!   as a machine's byte store does. A boolean reads back as whether any
//!   bit is set.
//! - A normalized float (`unorm8`, `snorm8`, `unorm16`) is clamped into
//!   its range, NaN taken as zero, scaled, and rounded to nearest with
//!   ties to even; it reads back as the integer times the reciprocal of
//!   the scale, a multiplication every device rounds alike.
//! - A half float rounds to nearest, ties to even, overflows to infinity
//!   and keeps NaN a NaN; it reads back exactly.

use std::fmt;

use crate::Scalar;

/// How one scalar is kept. [`Encoding::Word`] is the Kernel's own
/// 32-bit word; the others are narrower.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// A whole 32-bit word: `f32`, `i32`, or a `0`/`1` boolean.
    #[default]
    Word,
    /// A float as IEEE half precision.
    F16,
    /// A float in `0..=1` as a byte.
    Unorm8,
    /// A float in `-1..=1` as a signed byte.
    Snorm8,
    /// A float in `0..=1` as 16 bits.
    Unorm16,
    /// An integer or boolean as an unsigned byte.
    U8,
    /// An integer or boolean as a signed byte.
    I8,
    /// An integer or boolean as unsigned 16 bits.
    U16,
    /// An integer or boolean as signed 16 bits.
    I16,
}

impl Encoding {
    /// Every narrow encoding, in the order they are documented.
    pub const NARROW: [Encoding; 8] = [
        Encoding::F16,
        Encoding::Unorm8,
        Encoding::Snorm8,
        Encoding::Unorm16,
        Encoding::U8,
        Encoding::I8,
        Encoding::U16,
        Encoding::I16,
    ];

    /// The spelling after `as` in a port type, and in a schema's text.
    pub fn name(self) -> &'static str {
        match self {
            Self::Word => "word",
            Self::F16 => "f16",
            Self::Unorm8 => "unorm8",
            Self::Snorm8 => "snorm8",
            Self::Unorm16 => "unorm16",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
        }
    }

    /// The encoding a name spells, the narrow ones only.
    pub fn parse(name: &str) -> Option<Self> {
        Self::NARROW.into_iter().find(|e| e.name() == name)
    }

    /// Bytes one scalar takes.
    pub fn bytes(self) -> usize {
        match self {
            Self::Word => 4,
            Self::F16 | Self::Unorm16 | Self::U16 | Self::I16 => 2,
            Self::Unorm8 | Self::Snorm8 | Self::U8 | Self::I8 => 1,
        }
    }

    /// Whether this encoding keeps floats (as against integers).
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::Unorm8 | Self::Snorm8 | Self::Unorm16
        )
    }

    /// Whether a scalar of this kind may be kept this way: floats by
    /// the float encodings, integers and booleans by the integer ones.
    pub fn fits(self, scalar: Scalar) -> bool {
        match self {
            Self::Word => true,
            _ if self.is_float() => scalar == Scalar::Float,
            _ => scalar != Scalar::Float,
        }
    }

    /// The Kernel's word for the stored bits (in the low `bytes()`),
    /// read as a scalar of `kind`.
    pub fn decode(self, kind: Scalar, bits: u32) -> u32 {
        let word = match self {
            Self::Word => bits,
            Self::F16 => f16_to_f32(bits & 0xFFFF),
            Self::Unorm8 => ((bits & 0xFF) as f32 * UNORM8).to_bits(),
            Self::Unorm16 => ((bits & 0xFFFF) as f32 * UNORM16).to_bits(),
            Self::Snorm8 => ((bits as u8 as i8) as f32 * SNORM8).max(-1.0).to_bits(),
            Self::U8 => bits & 0xFF,
            Self::I8 => bits as u8 as i8 as i32 as u32,
            Self::U16 => bits & 0xFFFF,
            Self::I16 => bits as u16 as i16 as i32 as u32,
        };
        if kind == Scalar::Bool && self != Self::Word {
            u32::from(word != 0)
        } else {
            word
        }
    }

    /// The stored bits (in the low `bytes()`) for the Kernel's word.
    pub fn encode(self, word: u32) -> u32 {
        let unit = |scale: f32, low: f32| {
            let x = f32::from_bits(word);
            let x = if x.is_nan() { 0.0 } else { x.clamp(low, 1.0) };
            (x * scale).round_ties_even() as i32 as u32
        };
        match self {
            Self::Word => word,
            Self::F16 => f32_to_f16(word),
            Self::Unorm8 => unit(255.0, 0.0),
            Self::Unorm16 => unit(65535.0, 0.0),
            Self::Snorm8 => unit(127.0, -1.0) & 0xFF,
            Self::U8 | Self::I8 => word & 0xFF,
            Self::U16 | Self::I16 => word & 0xFFFF,
        }
    }
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The `bytes` stored at byte `byte` of `storage`, in the low bits. A
/// scalar sits at a multiple of its size, so it is within one word.
pub fn load(storage: &[u32], byte: usize, bytes: usize) -> u32 {
    let word = storage[byte / 4];
    if bytes == 4 {
        return word;
    }
    (word >> ((byte % 4) * 8)) & ((1u32 << (bytes * 8)) - 1)
}

/// Stores the low `bytes` of `bits` at byte `byte` of `storage`, leaving
/// the other bytes of its word as they were.
pub fn store(storage: &mut [u32], byte: usize, bytes: usize, bits: u32) {
    let word = &mut storage[byte / 4];
    if bytes == 4 {
        *word = bits;
        return;
    }
    let shift = (byte % 4) * 8;
    let mask = ((1u32 << (bytes * 8)) - 1) << shift;
    *word = (*word & !mask) | ((bits << shift) & mask);
}

/// The reciprocals a normalized integer is read back through, as `f32`
/// constants every backend spells by their bits.
pub const UNORM8: f32 = 1.0 / 255.0;
/// See [`UNORM8`].
pub const UNORM16: f32 = 1.0 / 65535.0;
/// See [`UNORM8`].
pub const SNORM8: f32 = 1.0 / 127.0;

/// A half float's bits to a float's, exactly.
pub fn f16_to_f32(h: u32) -> u32 {
    let sign = (h & 0x8000) << 16;
    let exponent = (h >> 10) & 0x1F;
    let mantissa = h & 0x3FF;
    match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // A subnormal: shift the leading one up to the implicit bit.
            let shift = mantissa.leading_zeros() - 21;
            let mantissa = (mantissa << shift) & 0x3FF;
            sign | ((113 - shift) << 23) | (mantissa << 13)
        }
        (0x1F, _) => sign | 0x7F80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    }
}

/// A float's bits to a half float's: to nearest, ties to even; too large
/// is infinity, too small a signed zero, NaN stays NaN (quiet).
pub fn f32_to_f16(x: u32) -> u32 {
    let sign = (x >> 16) & 0x8000;
    let exponent = ((x >> 23) & 0xFF) as i32;
    let mantissa = x & 0x7F_FFFF;
    if exponent == 0xFF {
        let nan = if mantissa != 0 {
            0x200 | (mantissa >> 13)
        } else {
            0
        };
        return sign | 0x7C00 | nan;
    }
    let e = exponent - 112;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    let (value, dropped, shift) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal: the implicit one joins the mantissa and the whole
        // shifts right by what the exponent lacks.
        let full = mantissa | 0x80_0000;
        let shift = (14 - e) as u32;
        (full >> shift, full & ((1 << shift) - 1), shift)
    } else {
        (((e as u32) << 10) | (mantissa >> 13), mantissa & 0x1FFF, 13)
    };
    let half = 1u32 << (shift - 1);
    let up = dropped > half || (dropped == half && value & 1 == 1);
    // A carry out of the mantissa is the next exponent, and out of the
    // largest exponent infinity: both are what the bits add up to.
    sign | (value + u32::from(up))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_half_float_reads_back_every_value_it_holds() {
        for h in 0..=0xFFFFu32 {
            let f = f16_to_f32(h);
            let back = f32_to_f16(f);
            if f32::from_bits(f).is_nan() {
                assert_eq!(back & 0x7C00, 0x7C00, "{h:#06x} stays NaN");
                assert_ne!(back & 0x3FF, 0, "{h:#06x} stays NaN");
            } else {
                assert_eq!(back, h, "{h:#06x} round trip");
            }
        }
    }

    #[test]
    fn a_float_narrows_to_the_nearest_half_ties_to_even() {
        let half = |x: f32| f32_to_f16(x.to_bits());
        assert_eq!(half(1.0), 0x3C00);
        assert_eq!(half(-2.0), 0xC000);
        assert_eq!(half(65504.0), 0x7BFF);
        assert_eq!(half(65520.0), 0x7C00, "rounds up to infinity");
        assert_eq!(half(1e9), 0x7C00);
        assert_eq!(half(f32::NEG_INFINITY), 0xFC00);
        // 1 + 2^-11 is halfway between 1 and the next half: to even.
        assert_eq!(half(1.0 + 2f32.powi(-11)), 0x3C00);
        assert_eq!(half(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
        // The smallest subnormal, and half of it (ties to even: zero).
        assert_eq!(half(2f32.powi(-24)), 0x0001);
        assert_eq!(half(2f32.powi(-25)), 0x0000);
        assert_eq!(half(1.5 * 2f32.powi(-24)), 0x0002);
        assert_eq!(half(-1e-30), 0x8000);
    }

    #[test]
    fn normalized_and_integer_encodings_follow_their_rules() {
        let word = |x: f32| x.to_bits();
        let float = |e: Encoding, bits: u32| f32::from_bits(e.decode(Scalar::Float, bits));
        assert_eq!(Encoding::Unorm8.encode(word(1.0)), 255);
        assert_eq!(Encoding::Unorm8.encode(word(2.0)), 255);
        assert_eq!(Encoding::Unorm8.encode(word(-1.0)), 0);
        assert_eq!(Encoding::Unorm8.encode(word(f32::NAN)), 0);
        assert_eq!(Encoding::Unorm8.encode(word(0.5)), 128, "127.5 to even");
        assert_eq!(Encoding::Snorm8.encode(word(-1.0)), 0x81);
        assert_eq!(Encoding::Snorm8.encode(word(-2.0)), 0x81);
        assert_eq!(float(Encoding::Snorm8, 0x80), -1.0, "-128 reads as -1");
        assert_eq!(Encoding::Unorm16.encode(word(1.0)), 65535);
        for v in 0..=255 {
            let x = float(Encoding::Unorm8, v);
            assert_eq!(Encoding::Unorm8.encode(x.to_bits()), v, "unorm8 {v}");
            let s = Encoding::Snorm8.decode(Scalar::Float, v);
            if v != 0x80 {
                assert_eq!(Encoding::Snorm8.encode(s), v, "snorm8 {v}");
            }
        }
        for v in [0, 1, 32767, 65534, 65535] {
            let x = Encoding::Unorm16.decode(Scalar::Float, v);
            assert_eq!(Encoding::Unorm16.encode(x), v, "unorm16 {v}");
        }
        assert_eq!(Encoding::U8.encode(300), 44);
        assert_eq!(Encoding::I8.decode(Scalar::Int, 0xFF) as i32, -1);
        assert_eq!(Encoding::U8.decode(Scalar::Int, 0xFF), 255);
        assert_eq!(Encoding::I16.decode(Scalar::Int, 0x8000) as i32, -32768);
        assert_eq!(Encoding::U8.decode(Scalar::Bool, 7), 1);
        assert!(Encoding::U8.fits(Scalar::Bool));
        assert!(!Encoding::U8.fits(Scalar::Float));
        assert!(!Encoding::F16.fits(Scalar::Int));
    }
}
