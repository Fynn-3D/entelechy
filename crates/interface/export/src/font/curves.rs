//! A closed run of points as a TrueType contour: quadratic curves through
//! the points, corners kept sharp, straight runs as their ends.

/// How far, in font units, a curve may leave the points it passes.
const WITHIN: f64 = 1.0;

/// A turn sharper than this cosine (about 41 degrees) is a corner.
const CORNER: f64 = 0.75;

/// A turn sharper than this cosine (about 20 degrees) is a corner where
/// a neighbour turns a quarter as much (a mitre's inner side).
const KINK: f64 = 0.94;

/// How far off its chord, as a sine, a run's tangent may lie for one
/// step of it to be straight (about 3 degrees).
const ALONG: f64 = 0.05;

/// A point of a contour: x, y, and whether it lies on the curve (else a
/// quadratic's control).
pub type Point = (i32, i32, bool);

type P = (f64, f64);

fn unit(x: f64, y: f64) -> P {
    let length = x.hypot(y);
    if length > 0.0 {
        (x / length, y / length)
    } else {
        (0.0, 0.0)
    }
}

fn corner_free(a: P, b: P, c: P) -> bool {
    let (ux, uy, vx, vy) = (b.0 - a.0, b.1 - a.1, c.0 - b.0, c.1 - b.1);
    let (lu, lv) = (ux.hypot(uy), vx.hypot(vy));
    lu > 0.0 && lv > 0.0 && (ux * vx + uy * vy) / (lu * lv) >= CORNER
}

/// The tangent at `a` from it and the next two points (a parabola's), or
/// toward the next where the run turns there.
fn one_sided(a: P, b: P, c: P) -> P {
    if corner_free(a, b, c) {
        unit(4.0 * b.0 - 3.0 * a.0 - c.0, 4.0 * b.1 - 3.0 * a.1 - c.1)
    } else {
        unit(b.0 - a.0, b.1 - a.1)
    }
}

fn quad(a: P, c: P, b: P, t: f64) -> P {
    let u = 1.0 - t;
    (
        u * u * a.0 + 2.0 * u * t * c.0 + t * t * b.0,
        u * u * a.1 + 2.0 * u * t * c.1 + t * t * b.1,
    )
}

/// The squared distance from `p` to the polyline `line`.
fn distance2(p: P, line: &[P]) -> f64 {
    line.windows(2).fold(f64::MAX, |best, w| {
        let (a, b) = (w[0], w[1]);
        let (ux, uy) = (b.0 - a.0, b.1 - a.1);
        let ll = ux * ux + uy * uy;
        let t = if ll == 0.0 {
            0.0
        } else {
            (((p.0 - a.0) * ux + (p.1 - a.1) * uy) / ll).clamp(0.0, 1.0)
        };
        best.min((a.0 + t * ux - p.0).powi(2) + (a.1 + t * uy - p.1).powi(2))
    })
}

/// Whether every point lies within [`WITHIN`] of the quadratic from `a`
/// over `c` to `b`.
fn near_quad(points: &[P], a: P, c: P, b: P) -> bool {
    let line: Vec<P> = (0..=8).map(|j| quad(a, c, b, f64::from(j) / 8.0)).collect();
    points
        .iter()
        .all(|&p| distance2(p, &line) <= WITHIN * WITHIN)
}

/// Whether the quadratic stays near the polyline through `samples` (a
/// chord may itself sit a little inside the curve it cuts, so three times
/// [`WITHIN`]).
fn along(a: P, c: P, b: P, samples: &[P]) -> bool {
    (1..8).all(|j| distance2(quad(a, c, b, f64::from(j) / 8.0), samples) <= 9.0 * WITHIN * WITHIN)
}

/// The run of points (closed, in font units) as a contour. The points stay
/// on the curve, a corner always; between two kept points one quadratic,
/// its control where their tangents meet, reaches as far along as it stays
/// within [`WITHIN`] of the points it passes; a straight run is its chord;
/// a kept point midway between two controls is left implied.
pub fn curved(run: &[(i32, i32)]) -> Vec<Point> {
    let mut pts: Vec<P> = Vec::with_capacity(run.len());
    for (k, &(x, y)) in run.iter().enumerate() {
        if run[(k + run.len() - 1) % run.len()] != (x, y) {
            pts.push((f64::from(x), f64::from(y)));
        }
    }
    let n = pts.len();
    if n < 4 {
        return pts.iter().map(|p| (p.0 as i32, p.1 as i32, true)).collect();
    }
    let at = |i: usize| pts[i % n];
    let before = |i: usize| pts[(i + n - 1) % n];
    let turn: Vec<f64> = (0..n)
        .map(|i| {
            let (a, b, c) = (before(i), at(i), at(i + 1));
            let (ux, uy, vx, vy) = (b.0 - a.0, b.1 - a.1, c.0 - b.0, c.1 - b.1);
            let (lu, lv) = (ux.hypot(uy), vx.hypot(vy));
            if lu == 0.0 || lv == 0.0 {
                std::f64::consts::PI
            } else {
                ((ux * vx + uy * vy) / (lu * lv)).clamp(-1.0, 1.0).acos()
            }
        })
        .collect();
    // A corner turns sharply where a neighbour hardly turns, or less
    // sharply where a neighbour turns a quarter as much (a mitre's inner
    // side); a round end drawn with few points turns sharply at each, and
    // is a curve.
    let (sharp, kink) = (CORNER.acos(), KINK.acos());
    let corner: Vec<bool> = (0..n)
        .map(|i| {
            let (b, a) = (turn[(i + n - 1) % n], turn[(i + 1) % n]);
            (turn[i] > sharp && turn[i] > 1.8 * b.min(a))
                || (turn[i] > kink && turn[i] > 4.0 * b.min(a))
        })
        .collect();
    // A point beside a corner takes its tangent from its own side: the
    // chord across to the corner would skew it.
    let smooth: Vec<P> = (0..n)
        .map(|i| {
            let (after, behind) = (corner[(i + 1) % n], corner[(i + n - 1) % n]);
            if behind && !after {
                one_sided(at(i), at(i + 1), at(i + 2))
            } else if after && !behind {
                let t = one_sided(at(i), before(i), before(i + n - 1));
                (-t.0, -t.1)
            } else {
                unit(at(i + 1).0 - before(i).0, at(i + 1).1 - before(i).1)
            }
        })
        .collect();
    let start = corner.iter().position(|&c| c).unwrap_or(0);
    let mut out: Vec<(f64, f64, bool, bool)> = Vec::new();
    let mut s = start;
    loop {
        let leave = if corner[s] {
            one_sided(at(s), at(s + 1), at(s + 2))
        } else {
            smooth[s]
        };
        let mut best: (usize, Option<P>) = ((s + 1) % n, None);
        let reach = if corner[(s + 1) % n] || (s + 1) % n == start {
            1
        } else {
            n
        };
        for step in 1..=reach {
            let e = (s + step) % n;
            let inner: Vec<P> = (1..step).map(|k| at(s + k)).collect();
            let arrive = if corner[e] {
                let t = one_sided(at(e), before(e), before(e + n - 1));
                (-t.0, -t.1)
            } else {
                smooth[e]
            };
            let (a, b) = (at(s), at(e));
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let chord = dx.hypot(dy);
            // one step is straight only where the run's tangents lie along it
            let level = |t: P| (t.0 * dy - t.1 * dx).abs() <= ALONG * chord;
            if chord > 0.0
                && (step > 1 || (level(leave) && level(arrive)))
                && inner
                    .iter()
                    .all(|p| ((p.0 - a.0) * dy - (p.1 - a.1) * dx).abs() / chord <= WITHIN)
            {
                best = (e, None);
            } else {
                // a quadratic, its control where the tangents meet; one step
                // that will not take one stays its chord, and a longer run is tried
                let den = leave.0 * arrive.1 - leave.1 * arrive.0;
                let t = (dx * arrive.1 - dy * arrive.0) / den;
                let u = (leave.0 * dy - leave.1 * dx) / den;
                let c = (a.0 + leave.0 * t, a.1 + leave.1 * t);
                let mut samples = vec![a];
                samples.extend(&inner);
                samples.push(b);
                let fits = den.abs() >= 1e-6
                    && t > 0.0
                    && u > 0.0
                    && t <= chord
                    && u <= chord
                    && near_quad(&inner, a, c, b)
                    && along(a, c, b, &samples);
                if fits {
                    best = (e, Some(c));
                } else if step > 1 {
                    break;
                }
            }
            if corner[e] || e == start {
                break;
            }
        }
        let (e, c) = best;
        out.push((at(s).0, at(s).1, true, corner[s] || s == start));
        if let Some(c) = c {
            out.push((c.0.round(), c.1.round(), false, false));
        }
        s = e;
        if s == start {
            break;
        }
    }
    // An on-curve point between two controls, at their midpoint (within
    // half a unit), is implied.
    let m = out.len();
    let mut drop = vec![false; m];
    for i in 0..m {
        let (x, y, on, fixed) = out[i];
        let (a, b) = (out[(i + m - 1) % m], out[(i + 1) % m]);
        if on
            && !fixed
            && !a.2
            && !b.2
            && !drop[(i + m - 1) % m]
            && !drop[(i + 1) % m]
            && (a.0 + b.0 - 2.0 * x).abs() <= 1.0
            && (a.1 + b.1 - 2.0 * y).abs() <= 1.0
        {
            drop[i] = true;
        }
    }
    out.iter()
        .zip(drop)
        .filter(|(_, d)| !d)
        .map(|(p, _)| (p.0 as i32, p.1 as i32, p.2))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 撇's outline whose start is mitred against a level stroke's end
    /// (又's): its sides run on as curves up to the mitre, not as chords.
    #[test]
    fn a_curve_beside_a_mitre_stays_a_curve() {
        let run = [
            (883, 783),
            (815, 576),
            (741, 432),
            (651, 301),
            (545, 186),
            (423, 84),
            (285, -2),
            (134, -73),
            (98, 4),
            (244, 72),
            (373, 153),
            (486, 247),
            (584, 355),
            (668, 476),
            (737, 610),
            (781, 711),
        ];
        let out = curved(&run);
        let chords = out.windows(2).filter(|w| w[0].2 && w[1].2).count();
        // the tail's square end, and the run off the mitre's outer point
        assert!(chords <= 2, "{out:?}");
        assert!(out.iter().any(|p| p.2 && (p.0, p.1) == (781, 711)));
    }
}
