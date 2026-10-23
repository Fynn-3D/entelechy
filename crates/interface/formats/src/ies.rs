//! A light's photometry in IES LM-63 (`.ies`) as rows of candela: one a
//! degree from the light's aim (0, type C's nadir) to straight behind it
//! (180), each the mean over the file's horizontal planes, linearly
//! between the file's vertical angles and none past them. The candela
//! multiplier and the ballast factor are taken in; a tilt is not.

use entelechy_host::{Domain, FieldData, Scalar, Schema};

/// The rows a profile holds: one a degree, 0 to 180.
pub const ROWS: usize = 181;

/// A light's photometry, as a profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// Candela a degree from the aim, [`ROWS`] of them.
    pub candela: Vec<f32>,
    /// The most of them.
    pub peak: f32,
    /// The horizontal planes the file held.
    pub planes: usize,
}

/// Reads an LM-63 file of type C photometry (every light fixture's but
/// floodlights' and road lights').
pub fn read(bytes: &[u8]) -> Result<Profile, String> {
    let text = String::from_utf8_lossy(bytes);
    let mut lines = text.lines();
    // Keywords run to the TILT line; numbers follow it.
    let tilt = loop {
        let line = lines
            .next()
            .ok_or("an IES file names its TILT before its numbers; none is found")?;
        if let Some(rest) = line.trim().strip_prefix("TILT=") {
            break rest.trim().to_owned();
        }
    };
    let mut numbers = lines
        .flat_map(|line| line.split(|c: char| c.is_whitespace() || c == ','))
        .filter(|word| !word.is_empty())
        .map(|word| {
            word.parse::<f32>()
                .map_err(|_| format!("`{word}` is not a number"))
        });
    let mut next = |what: &str| {
        numbers
            .next()
            .unwrap_or_else(|| Err(format!("the file ends before its {what}")))
    };
    if tilt == "INCLUDE" {
        next("tilt geometry")?;
        let angles = next("tilt angle count")? as usize;
        for _ in 0..angles * 2 {
            next("tilt angles")?;
        }
    } else if tilt != "NONE" {
        return Err(format!(
            "TILT={tilt} names another file; only NONE or INCLUDE is read"
        ));
    }
    next("lamp count")?;
    next("lumens")?;
    let multiplier = next("candela multiplier")?;
    let vertical = next("vertical angle count")? as usize;
    let horizontal = next("horizontal angle count")? as usize;
    let kind = next("photometric type")? as i32;
    for what in ["units", "width", "length", "height"] {
        next(what)?;
    }
    let ballast = next("ballast factor")?;
    next("ballast-lamp factor")?;
    next("input watts")?;
    if kind != 1 {
        return Err(format!(
            "photometric type {kind} is not read; only type C (1) is"
        ));
    }
    if vertical < 2 || horizontal < 1 {
        return Err(format!(
            "{vertical} vertical and {horizontal} horizontal angle(s); a profile needs two and one"
        ));
    }
    let angles: Vec<f32> = (0..vertical)
        .map(|_| next("vertical angles"))
        .collect::<Result<_, _>>()?;
    for _ in 0..horizontal {
        next("horizontal angles")?;
    }
    let values: Vec<f32> = (0..vertical * horizontal)
        .map(|_| next("candela values"))
        .collect::<Result<_, _>>()?;
    if angles.windows(2).any(|w| w[1] <= w[0]) {
        return Err("the vertical angles do not rise".into());
    }
    let scale = multiplier * if ballast > 0.0 { ballast } else { 1.0 };
    let candela: Vec<f32> = (0..ROWS)
        .map(|degree| {
            let a = degree as f32;
            let sum: f32 = values
                .chunks_exact(vertical)
                .map(|plane| between(&angles, plane, a))
                .sum();
            sum / horizontal as f32 * scale
        })
        .collect();
    let peak = candela.iter().copied().fold(0.0f32, f32::max);
    Ok(Profile {
        candela,
        peak,
        planes: horizontal,
    })
}

/// A plane's candela at angle `a`, linearly between the angles, none
/// past them.
fn between(angles: &[f32], plane: &[f32], a: f32) -> f32 {
    let last = angles.len() - 1;
    if a < angles[0] || a > angles[last] {
        return 0.0;
    }
    let i = angles.partition_point(|x| *x <= a).clamp(1, last);
    let (a0, a1) = (angles[i - 1], angles[i]);
    let t = (a - a0) / (a1 - a0);
    plane[i - 1] + (plane[i] - plane[i - 1]) * t
}

/// A profile as rows of one float, candela a degree.
pub fn field(profile: &Profile) -> Result<FieldData, String> {
    let words = profile.candela.iter().map(|v| v.to_bits()).collect();
    FieldData::from_storage(
        Schema::new(vec![Scalar::Float]),
        Domain::Rows(ROWS as u32),
        words,
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOWNLIGHT: &str = "IESNA:LM-63-2002\n[TEST] a downlight\n[MANUFAC] none\nTILT=NONE\n\
        1 1000 2.0 4 2 1 2 0.1 0.1 0.0\n1.0 1.0 10\n0 30 60 90\n0 180\n\
        100, 80, 20, 0\n100 60 20 0\n";

    #[test]
    fn a_downlight_is_read_a_degree_at_a_time() {
        let profile = read(DOWNLIGHT.as_bytes()).unwrap();
        assert_eq!(profile.candela.len(), ROWS);
        assert_eq!(profile.planes, 2);
        // The multiplier doubles the planes' mean.
        assert_eq!(profile.candela[0], 200.0);
        assert_eq!(profile.candela[30], 140.0);
        assert!((profile.candela[15] - 170.0).abs() < 1e-4);
        assert_eq!(profile.candela[90], 0.0);
        assert_eq!(profile.candela[120], 0.0);
        assert_eq!(profile.peak, 200.0);
        let data = field(&profile).unwrap();
        assert_eq!(data.domain(), Domain::Rows(181));
    }

    #[test]
    fn a_tilt_included_is_passed_over_and_one_elsewhere_refused() {
        let included =
            DOWNLIGHT.replace("TILT=NONE\n", "TILT=INCLUDE\n1\n3\n0 45 90\n1.0 0.9 0.8\n");
        assert_eq!(read(included.as_bytes()).unwrap().candela[0], 200.0);
        let named = DOWNLIGHT.replace("TILT=NONE", "TILT=lamp.tlt");
        assert!(read(named.as_bytes()).unwrap_err().contains("another file"));
    }

    #[test]
    fn a_file_cut_short_or_of_another_type_is_refused() {
        let short = &DOWNLIGHT[..DOWNLIGHT.len() - 12];
        assert!(read(short.as_bytes()).unwrap_err().contains("ends before"));
        let floodlight = DOWNLIGHT.replace("4 2 1 2", "4 2 3 2");
        assert!(read(floodlight.as_bytes()).unwrap_err().contains("type C"));
        assert!(read(b"no tilt here\n1 2 3").unwrap_err().contains("TILT"));
    }
}
