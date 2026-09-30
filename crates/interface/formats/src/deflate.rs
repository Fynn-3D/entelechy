//! DEFLATE (RFC 1951) and zlib (RFC 1950) encoding: matches found in a
//! 32 KiB window through chains of three-byte hashes, one step of lazy
//! matching, and each block's own Huffman codes, their lengths held to
//! the format's limits by package-merge. A stream that would come out
//! longer than the bytes themselves is stored instead.

use crate::inflate::adler32;

const WINDOW: usize = 1 << 15;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
/// How many earlier places with the same three bytes a match is sought
/// among, and a length good enough to stop seeking.
const CHAIN: usize = 128;
const GOOD: usize = 128;
/// Symbols a block holds before its codes are written and new ones begun.
const BLOCK: usize = 1 << 16;

/// A literal byte, or a match: its length and how far back it is.
#[derive(Clone, Copy)]
enum Token {
    Literal(u8),
    Match { length: u16, distance: u16 },
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// The order the code-length code's lengths are written in.
const LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// The code of a length (3..=258): its symbol's index past 257.
fn length_code(length: u16) -> usize {
    LENGTH_BASE
        .iter()
        .rposition(|&base| base <= length)
        .unwrap_or(0)
}

/// The code of a distance (1..=32768).
fn distance_code(distance: u16) -> usize {
    DISTANCE_BASE
        .iter()
        .rposition(|&base| base <= distance)
        .unwrap_or(0)
}

/// The bytes as literals and matches.
fn tokens(bytes: &[u8]) -> Vec<Token> {
    let n = bytes.len();
    let mut out = Vec::with_capacity(n / 2 + 16);
    let mut head = vec![usize::MAX; 1 << 15];
    let mut prev = vec![usize::MAX; WINDOW];
    let hash = |i: usize| -> usize {
        let v = (bytes[i] as usize) << 10 ^ (bytes[i + 1] as usize) << 5 ^ bytes[i + 2] as usize;
        v & ((1 << 15) - 1)
    };
    let insert = |i: usize, head: &mut Vec<usize>, prev: &mut Vec<usize>| {
        if i + MIN_MATCH <= n {
            let h = hash(i);
            prev[i % WINDOW] = head[h];
            head[h] = i;
        }
    };
    // The longest match for the bytes at i among those already seen.
    let longest = |i: usize, head: &Vec<usize>, prev: &Vec<usize>| -> (usize, usize) {
        if i + MIN_MATCH > n {
            return (0, 0);
        }
        let most = (n - i).min(MAX_MATCH);
        let (mut best, mut at) = (0, 0);
        let mut candidate = head[hash(i)];
        let mut tries = CHAIN;
        while candidate != usize::MAX && i - candidate <= WINDOW && tries > 0 {
            if bytes[candidate + best.min(most - 1)] == bytes[i + best.min(most - 1)] {
                let mut length = 0;
                while length < most && bytes[candidate + length] == bytes[i + length] {
                    length += 1;
                }
                if length > best {
                    best = length;
                    at = i - candidate;
                    if length >= GOOD || length == most {
                        break;
                    }
                }
            }
            let next = prev[candidate % WINDOW];
            // A chain runs back only; one that turns forward is stale.
            if next == usize::MAX || next >= candidate {
                break;
            }
            candidate = next;
            tries -= 1;
        }
        if best >= MIN_MATCH {
            (best, at)
        } else {
            (0, 0)
        }
    };
    let mut i = 0;
    while i < n {
        let (length, distance) = longest(i, &head, &prev);
        insert(i, &mut head, &mut prev);
        if length == 0 {
            out.push(Token::Literal(bytes[i]));
            i += 1;
            continue;
        }
        // One step of lazy matching: a longer match one byte on wins.
        let (next, _) = if length < GOOD {
            longest(i + 1, &head, &prev)
        } else {
            (0, 0)
        };
        if next > length {
            out.push(Token::Literal(bytes[i]));
            i += 1;
            continue;
        }
        out.push(Token::Match {
            length: length as u16,
            distance: distance as u16,
        });
        for j in i + 1..i + length {
            insert(j, &mut head, &mut prev);
        }
        i += length;
    }
    out
}

/// Code lengths for symbols of these frequencies, none past `limit`
/// bits, by package-merge; an unused symbol has none.
fn code_lengths(frequencies: &[u32], limit: u32) -> Vec<u8> {
    let used: Vec<usize> = (0..frequencies.len())
        .filter(|&s| frequencies[s] > 0)
        .collect();
    let mut lengths = vec![0u8; frequencies.len()];
    match used.len() {
        0 => return lengths,
        1 => {
            lengths[used[0]] = 1;
            return lengths;
        }
        _ => {}
    }
    // An item of a level: a leaf (a symbol), or a package of two items
    // of the level before, by their places there.
    #[derive(Clone, Copy)]
    enum Item {
        Leaf(usize),
        Package(usize, usize),
    }
    let mut leaves: Vec<(u64, Item)> = used
        .iter()
        .map(|&s| (frequencies[s] as u64, Item::Leaf(s)))
        .collect();
    leaves.sort_by_key(|(weight, _)| *weight);
    let mut levels: Vec<Vec<(u64, Item)>> = vec![leaves.clone()];
    for _ in 1..limit {
        let below = levels.last().unwrap();
        let packages: Vec<(u64, Item)> = (0..below.len() / 2)
            .map(|k| {
                (
                    below[2 * k].0 + below[2 * k + 1].0,
                    Item::Package(2 * k, 2 * k + 1),
                )
            })
            .collect();
        let mut merged = Vec::with_capacity(leaves.len() + packages.len());
        let (mut a, mut b) = (0, 0);
        while a < leaves.len() || b < packages.len() {
            if b >= packages.len() || (a < leaves.len() && leaves[a].0 <= packages[b].0) {
                merged.push(leaves[a]);
                a += 1;
            } else {
                merged.push(packages[b]);
                b += 1;
            }
        }
        levels.push(merged);
    }
    // The first 2n - 2 items of the last level, unpacked: a symbol's
    // length is how many times it is reached.
    let mut open: Vec<(usize, usize)> = (0..2 * used.len() - 2)
        .map(|i| (levels.len() - 1, i))
        .collect();
    while let Some((level, index)) = open.pop() {
        match levels[level][index].1 {
            Item::Leaf(s) => lengths[s] += 1,
            Item::Package(a, b) => {
                open.push((level - 1, a));
                open.push((level - 1, b));
            }
        }
    }
    lengths
}

/// The canonical codes for these lengths, each bit-reversed as the
/// stream sends it.
fn codes(lengths: &[u8]) -> Vec<u16> {
    let mut count = [0u16; 16];
    for &l in lengths {
        count[l as usize] += 1;
    }
    count[0] = 0;
    let mut next = [0u16; 16];
    let mut code = 0u16;
    for bits in 1..16 {
        code = (code + count[bits - 1]) << 1;
        next[bits] = code;
    }
    lengths
        .iter()
        .map(|&l| {
            if l == 0 {
                return 0;
            }
            let c = next[l as usize];
            next[l as usize] += 1;
            c.reverse_bits() >> (16 - l)
        })
        .collect()
}

/// Bits out, least significant first.
struct Bits {
    out: Vec<u8>,
    acc: u64,
    count: u32,
}

impl Bits {
    fn put(&mut self, value: u32, bits: u32) {
        self.acc |= (value as u64) << self.count;
        self.count += bits;
        while self.count >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.count -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.count > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

/// The run-length form of the code lengths: (symbol, extra bits' value).
fn runs(lengths: &[u8]) -> Vec<(u8, u8)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lengths.len() {
        let l = lengths[i];
        let mut run = 1;
        while i + run < lengths.len() && lengths[i + run] == l {
            run += 1;
        }
        let mut left = run;
        if l == 0 {
            while left >= 11 {
                let take = left.min(138);
                out.push((18, (take - 11) as u8));
                left -= take;
            }
            if left >= 3 {
                out.push((17, (left - 3) as u8));
                left = 0;
            }
        } else {
            out.push((l, 0));
            left -= 1;
            while left >= 3 {
                let take = left.min(6);
                out.push((16, (take - 3) as u8));
                left -= take;
            }
        }
        for _ in 0..left {
            out.push((l, 0));
        }
        i += run;
    }
    out
}

/// One block of tokens with its own codes; `last` marks the stream's end.
fn block(bits: &mut Bits, tokens: &[Token], last: bool) {
    let mut literal_freq = [0u32; 286];
    let mut distance_freq = [0u32; 30];
    for token in tokens {
        match *token {
            Token::Literal(byte) => literal_freq[byte as usize] += 1,
            Token::Match { length, distance } => {
                literal_freq[257 + length_code(length)] += 1;
                distance_freq[distance_code(distance)] += 1;
            }
        }
    }
    literal_freq[256] = 1;
    // A block without matches still names one distance code.
    if distance_freq.iter().all(|&f| f == 0) {
        distance_freq[0] = 1;
    }
    let literal_lengths = code_lengths(&literal_freq, 15);
    let distance_lengths = code_lengths(&distance_freq, 15);
    let hlit = literal_lengths
        .iter()
        .rposition(|&l| l > 0)
        .map_or(257, |p| p + 1)
        .max(257);
    let hdist = distance_lengths
        .iter()
        .rposition(|&l| l > 0)
        .map_or(1, |p| p + 1)
        .max(1);
    let mut all = literal_lengths[..hlit].to_vec();
    all.extend_from_slice(&distance_lengths[..hdist]);
    let runs = runs(&all);
    let mut length_freq = [0u32; 19];
    for &(symbol, _) in &runs {
        length_freq[symbol as usize] += 1;
    }
    let length_lengths = code_lengths(&length_freq, 7);
    let hclen = LENGTH_ORDER
        .iter()
        .rposition(|&s| length_lengths[s] > 0)
        .map_or(4, |p| p + 1)
        .max(4);
    let length_codes = codes(&length_lengths);
    let literal_codes = codes(&literal_lengths);
    let distance_codes = codes(&distance_lengths);

    bits.put(u32::from(last), 1);
    bits.put(2, 2);
    bits.put((hlit - 257) as u32, 5);
    bits.put((hdist - 1) as u32, 5);
    bits.put((hclen - 4) as u32, 4);
    for &s in &LENGTH_ORDER[..hclen] {
        bits.put(length_lengths[s] as u32, 3);
    }
    for &(symbol, extra) in &runs {
        let s = symbol as usize;
        bits.put(length_codes[s] as u32, length_lengths[s] as u32);
        match symbol {
            16 => bits.put(extra as u32, 2),
            17 => bits.put(extra as u32, 3),
            18 => bits.put(extra as u32, 7),
            _ => {}
        }
    }
    for token in tokens {
        match *token {
            Token::Literal(byte) => {
                let s = byte as usize;
                bits.put(literal_codes[s] as u32, literal_lengths[s] as u32);
            }
            Token::Match { length, distance } => {
                let l = length_code(length);
                bits.put(
                    literal_codes[257 + l] as u32,
                    literal_lengths[257 + l] as u32,
                );
                bits.put((length - LENGTH_BASE[l]) as u32, LENGTH_EXTRA[l] as u32);
                let d = distance_code(distance);
                bits.put(distance_codes[d] as u32, distance_lengths[d] as u32);
                bits.put(
                    (distance - DISTANCE_BASE[d]) as u32,
                    DISTANCE_EXTRA[d] as u32,
                );
            }
        }
    }
    bits.put(literal_codes[256] as u32, literal_lengths[256] as u32);
}

/// Bytes as a DEFLATE stream of compressed blocks.
pub fn deflate(bytes: &[u8]) -> Vec<u8> {
    let tokens = tokens(bytes);
    let mut bits = Bits {
        out: Vec::with_capacity(bytes.len() / 4 + 64),
        acc: 0,
        count: 0,
    };
    let mut blocks = tokens.chunks(BLOCK).peekable();
    if blocks.peek().is_none() {
        block(&mut bits, &[], true);
    }
    while let Some(chunk) = blocks.next() {
        block(&mut bits, chunk, blocks.peek().is_none());
    }
    bits.finish()
}

/// Bytes as a zlib stream of stored (uncompressed) blocks.
fn stored(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() + bytes.len() / 65535 * 5 + 5);
    let mut chunks = bytes.chunks(65535).peekable();
    if chunks.peek().is_none() {
        out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    while let Some(chunk) = chunks.next() {
        out.push(if chunks.peek().is_none() { 1 } else { 0 });
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out
}

/// Bytes as a zlib stream: compressed, or stored where that is shorter.
pub fn zlib(bytes: &[u8]) -> Vec<u8> {
    let compressed = deflate(bytes);
    let body = if compressed.len() < bytes.len() + bytes.len() / 65535 * 5 + 5 {
        compressed
    } else {
        stored(bytes)
    };
    let mut out = Vec::with_capacity(body.len() + 6);
    out.extend_from_slice(&[0x78, 0x9c]);
    out.extend_from_slice(&body);
    out.extend_from_slice(&adler32(bytes).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inflate::unzlib;

    fn round_trip(bytes: &[u8]) -> usize {
        let packed = zlib(bytes);
        assert_eq!(
            unzlib(&packed, usize::MAX).unwrap(),
            bytes,
            "{} bytes",
            bytes.len()
        );
        packed.len()
    }

    /// Streams of every shape come back as they went: empty, one byte,
    /// text, runs longer than a match, noise, and more than a window and
    /// a block of symbols.
    #[test]
    fn streams_round_trip() {
        round_trip(b"");
        round_trip(b"a");
        round_trip(b"abcabcabcabcabcabcabc and so on, abcabc");
        round_trip(&[7u8; 100_000]);
        let mut seed = 0x2545_f491u32;
        let noise: Vec<u8> = (0..70_000)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as u8
            })
            .collect();
        round_trip(&noise);
        let long: Vec<u8> = (0..400_000u32)
            .map(|i| ((i / 7) % 251) as u8 ^ ((i % 3) as u8))
            .collect();
        round_trip(&long);
    }

    /// Repetitive bytes come out a small part of their size; noise no
    /// larger than stored.
    #[test]
    fn repetition_is_small_and_noise_is_no_larger() {
        let rows: Vec<u8> = (0..200_000u32)
            .map(|i| [10u8, 20, 30, 255][(i % 4) as usize])
            .collect();
        assert!(round_trip(&rows) < rows.len() / 100);
        let mut seed = 99u32;
        let noise: Vec<u8> = (0..50_000)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (seed >> 16) as u8
            })
            .collect();
        assert!(round_trip(&noise) <= noise.len() + noise.len() / 65535 * 5 + 11);
    }

    /// Package-merge holds every code to its limit and keeps the lengths
    /// a complete prefix code.
    #[test]
    fn code_lengths_are_limited_and_complete() {
        let frequencies: Vec<u32> = (0..286).map(|i| 1u32 << (i % 30).min(24)).collect();
        let lengths = code_lengths(&frequencies, 15);
        assert!(lengths.iter().all(|&l| (1..=15).contains(&l)));
        let kraft: f64 = lengths.iter().map(|&l| 0.5f64.powi(l as i32)).sum();
        assert!((kraft - 1.0).abs() < 1e-9, "{kraft}");
    }
}
