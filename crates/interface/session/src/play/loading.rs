//! The picture while a scene's Kernels prepare: on black, a hairline
//! across the window's middle with a seed on it for each Kernel. A seed
//! waiting is a faint point, one compiled a pale one; one building on its
//! device glows gold and breathes; one built is gold, and the line between
//! two built seeds is lit. Under the line, how many are built and the
//! Kernel taking longest. Before any Kernel is known, one seed breathes
//! alone in the middle.
//!
//! Drawn on the CPU as a band of `0x00RRGGBB` pixels the window's width,
//! for a presenter to lay over black at [`Band::top`].

use std::f32::consts::TAU;

use entelechy_kernel::{Preparing, Stage};

use super::hud;

/// The band: its pixels, and where its top lies in a window `height` tall.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Band {
    /// The pixels, `width` per row.
    pub panel: hud::Panel,
    /// Its top's row in the window.
    pub top: u32,
}

const LINE: [f32; 3] = [38.0, 34.0, 58.0];
const WAITING: [f32; 3] = [78.0, 72.0, 104.0];
const COMPILED: [f32; 3] = [158.0, 148.0, 196.0];
const GOLD: [f32; 3] = [236.0, 188.0, 92.0];
const GLOW: [f32; 3] = [255.0, 196.0, 104.0];
const LIT: [f32; 3] = [132.0, 98.0, 46.0];
const WORDS: [f32; 3] = [112.0, 106.0, 140.0];

/// The band for a window `width` by `height` device pixels at `scale`
/// device pixels to the logical one, `seconds` into the loading.
pub fn band(now: &Preparing, seconds: f64, width: u32, height: u32, scale: f64) -> Band {
    let s = scale.max(1.0) as f32;
    let tall = ((64.0 * s) as u32).min(height.max(1));
    let width = width.max(1);
    let mut picture = Picture {
        width,
        height: tall,
        rgb: vec![[0.0; 3]; (width * tall) as usize],
    };
    let mid = 20.0 * s;
    let t = seconds as f32;
    let n = now.seeds.len();
    let (left, right) = (width as f32 * 0.25, width as f32 * 0.75);
    let at = |i: usize| {
        if n < 2 {
            width as f32 * 0.5
        } else {
            left + (right - left) * i as f32 / (n - 1) as f32
        }
    };
    if n == 0 {
        // Nothing known yet: the seed before the laws.
        let breath = 0.5 + 0.5 * (t * TAU / 2.4).sin();
        picture.glow(
            width as f32 * 0.5,
            mid,
            (3.0 + 3.0 * breath) * s,
            GLOW,
            0.35 + 0.3 * breath,
        );
        picture.disc(width as f32 * 0.5, mid, 1.4 * s, GOLD, 0.8);
    } else {
        let thin = s.round().max(1.0);
        picture.bar(left, right, mid, thin, LINE, 1.0);
        for (i, pair) in now.seeds.windows(2).enumerate() {
            if pair.iter().all(|seed| seed.stage == Stage::Built) {
                picture.bar(at(i), at(i + 1), mid, thin, LIT, 1.0);
            }
        }
        for (i, seed) in now.seeds.iter().enumerate() {
            let x = at(i);
            match seed.stage {
                Stage::Waiting => picture.disc(x, mid, 1.2 * s, WAITING, 1.0),
                Stage::Compiled => picture.disc(x, mid, 1.6 * s, COMPILED, 1.0),
                Stage::Built => picture.disc(x, mid, 2.0 * s, GOLD, 1.0),
                Stage::Compiling | Stage::Building => {
                    let breath = 0.5 + 0.5 * (t * TAU / 1.6 + i as f32 * 0.7).sin();
                    let grown = (seed.took.as_secs_f32() / 8.0).min(1.0);
                    picture.glow(
                        x,
                        mid,
                        (4.0 + 3.0 * breath + 4.0 * grown) * s,
                        GLOW,
                        0.45 + 0.35 * breath,
                    );
                    picture.disc(x, mid, 1.8 * s, GOLD, 1.0);
                }
            }
        }
        let mut said = format!("{} / {n}", now.built());
        if let Some(seed) = now.longest().filter(|seed| seed.took.as_secs() >= 1) {
            said.push_str(&format!(
                "    {}  {} s",
                seed.id.name(),
                seed.took.as_secs()
            ));
        }
        let text = (s.round() as u32).max(1);
        picture.words(&hud::panel(&said, text), (mid + 12.0 * s) as u32, WORDS);
    }
    Band {
        panel: picture.pixels(),
        top: (height / 2).saturating_sub(mid as u32).min(height - tall),
    }
}

/// Light gathered on black, each channel 0 to 255.
struct Picture {
    width: u32,
    height: u32,
    rgb: Vec<[f32; 3]>,
}

impl Picture {
    fn add(&mut self, x: i64, y: i64, colour: [f32; 3], amount: f32) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 || amount <= 0.0 {
            return;
        }
        let pixel = &mut self.rgb[(y as u32 * self.width + x as u32) as usize];
        for (channel, light) in pixel.iter_mut().zip(colour) {
            *channel = (*channel + light * amount).min(255.0);
        }
    }

    /// A line `thick` pixels from `from` to `to` at `y`.
    fn bar(&mut self, from: f32, to: f32, y: f32, thick: f32, colour: [f32; 3], amount: f32) {
        let top = (y - thick * 0.5).round() as i64;
        for row in top..top + thick as i64 {
            for x in from.round() as i64..=to.round() as i64 {
                self.set(x, row, colour, amount);
            }
        }
    }

    /// A disc of radius `r`, its edge smoothed, over what is there.
    fn disc(&mut self, cx: f32, cy: f32, r: f32, colour: [f32; 3], amount: f32) {
        let reach = r.ceil() as i64 + 1;
        for y in cy as i64 - reach..=cy as i64 + reach {
            for x in cx as i64 - reach..=cx as i64 + reach {
                let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
                let cover = (r - d + 0.5).clamp(0.0, 1.0);
                self.set(x, y, colour, amount * cover);
            }
        }
    }

    /// Light falling off softly to radius `r`, added.
    fn glow(&mut self, cx: f32, cy: f32, r: f32, colour: [f32; 3], amount: f32) {
        let reach = r.ceil() as i64 * 2;
        for y in cy as i64 - reach..=cy as i64 + reach {
            for x in cx as i64 - reach..=cx as i64 + reach {
                let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt() / r;
                self.add(x, y, colour, amount * (-d * d * 1.6).exp());
            }
        }
    }

    /// `colour` laid over what is there by `amount`.
    fn set(&mut self, x: i64, y: i64, colour: [f32; 3], amount: f32) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 || amount <= 0.0 {
            return;
        }
        let pixel = &mut self.rgb[(y as u32 * self.width + x as u32) as usize];
        for (channel, paint) in pixel.iter_mut().zip(colour) {
            *channel += (paint - *channel) * amount.min(1.0);
        }
    }

    /// Text from the HUD's font, centred, its ink in `colour`.
    fn words(&mut self, text: &hud::Panel, top: u32, colour: [f32; 3]) {
        let left = (self.width.saturating_sub(text.width) / 2) as i64;
        for y in 0..text.height {
            for x in 0..text.width {
                if text.pixels[(y * text.width + x) as usize] != 0 {
                    self.set(left + x as i64, (top + y) as i64, colour, 1.0);
                }
            }
        }
    }

    fn pixels(self) -> hud::Panel {
        let pixels = self
            .rgb
            .iter()
            .map(|[r, g, b]| ((*r as u32) << 16) | ((*g as u32) << 8) | (*b as u32))
            .collect();
        hud::Panel {
            width: self.width,
            height: self.height,
            pixels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use entelechy_kernel::{KernelId, Seed};
    use std::time::Duration;

    fn seed(name: &str, stage: Stage, seconds: u64) -> Seed {
        Seed {
            id: KernelId::new(name, 1).expect("an id"),
            stage,
            took: Duration::from_secs(seconds),
        }
    }

    #[test]
    fn the_band_lights_what_is_built_and_names_the_longest() {
        let now = Preparing {
            seeds: vec![
                seed("t.a", Stage::Built, 0),
                seed("t.b", Stage::Built, 0),
                seed("t.c", Stage::Building, 3),
                seed("t.d", Stage::Waiting, 0),
            ],
        };
        let band = band(&now, 1.0, 400, 300, 1.0);
        let panel = &band.panel;
        assert_eq!((panel.width, panel.height), (400, 64));
        assert!(band.top + panel.height <= 300);
        let at = |x: u32, y: u32| panel.pixels[(y * panel.width + x) as usize];
        let red = |p: u32| (p >> 16) & 255;
        // Between the two built seeds the line is lit; past the last, not.
        let lit = at(150, 20);
        let unlit = at(290, 20);
        assert!(red(lit) > red(unlit), "{lit:06x} {unlit:06x}");
        // The words are drawn under the line.
        assert!((34..panel.height).any(|y| (0..400).any(|x| at(x, y) != 0)));
        // With nothing known, one seed in the middle.
        let alone = super::band(&Preparing::default(), 0.6, 400, 300, 1.0);
        assert!(alone.panel.pixels[(20 * 400 + 200) as usize] != 0);
    }
}
