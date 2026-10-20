//! The picture while a scene's Kernels prepare: a nebula. Each Kernel is a
//! star; the Kernels that feed each other in the scene pull together and
//! push the rest off, so the scene's structure gathers into clusters as it
//! becomes known, and the whole turns as a galaxy, the rim further than
//! the heart. A star waiting is dust, one compiled pale; one building
//! breathes gold and flares when built, and stays gold. Threads run between
//! the stars that feed each other, lit where both are built, and the gas
//! about them gathers where they stand densest, warming as they light.
//! Under it, how many are built and the Kernel taking longest. Before any
//! Kernel is known, one seed breathes alone in the middle.
//!
//! Drawn on the CPU as `0x00RRGGBB` pixels the window's size, by a [`Sky`]
//! kept from one picture to the next.

use std::collections::HashMap;
use std::f32::consts::TAU;

use entelechy_ir::KernelId;
use entelechy_kernel::{Preparing, Stage};

use super::hud;

const GROUND: [f32; 3] = [3.0, 4.0, 8.0];
const THREAD: [f32; 3] = [140.0, 173.0, 255.0];
const VIOLET: [f32; 3] = [212.0, 98.0, 236.0];
const DUST: [f32; 3] = [120.0, 130.0, 190.0];
const PALE: [f32; 3] = [196.0, 190.0, 240.0];
const GOLD: [f32; 3] = [236.0, 188.0, 92.0];
const GLOW: [f32; 3] = [255.0, 196.0, 104.0];
const WHITE: [f32; 3] = [230.0, 235.0, 255.0];
const WORDS: [f32; 3] = [112.0, 106.0, 140.0];

/// The nebula kept between pictures: where each star stands, and the sky
/// behind them for the window's size.
#[derive(Debug, Default)]
pub struct Sky {
    /// The Kernels the stars are, in the seeds' order.
    known: Vec<KernelId>,
    stars: Vec<Star>,
    links: Vec<(usize, usize)>,
    /// When the stars or their links last changed, in seconds of loading.
    changed: f32,
    /// The layout's radius as shown, eased toward the layout's own.
    far: f32,
    ground: Ground,
    picture: hud::Panel,
}

#[derive(Clone, Debug)]
struct Star {
    x: f32,
    y: f32,
    /// When it was first seen built.
    lit: Option<f32>,
}

/// What does not change while the window keeps its size: the clouds' noise
/// a cell of `cell` pixels each, and the far stars.
#[derive(Debug, Default)]
struct Ground {
    size: (u32, u32, u32),
    cell: u32,
    columns: u32,
    rows: u32,
    /// Per cell: how thick the cloud is, and its tint.
    clouds: Vec<[f32; 2]>,
    /// The far stars: pixel, brightness, and its twinkle's phase.
    far: Vec<(u32, f32, f32)>,
}

impl Sky {
    /// The picture for a window `width` by `height` device pixels at
    /// `scale` device pixels to the logical one, `seconds` into loading.
    pub fn draw(
        &mut self,
        now: &Preparing,
        seconds: f64,
        width: u32,
        height: u32,
        scale: f64,
    ) -> &hud::Panel {
        let (width, height) = (width.max(1), height.max(1));
        let s = scale.max(1.0) as f32;
        let t = seconds as f32;
        self.ground.fit(width, height, s);
        self.follow(now, t);
        let steps = if self.stars.len() > 300 { 1 } else { 2 };
        for _ in 0..steps {
            self.settle(t);
        }
        let mut picture = Picture {
            width,
            height,
            pixels: std::mem::take(&mut self.picture.pixels),
            fade: (t / 0.6).clamp(0.0, 1.0),
        };
        picture.pixels.clear();
        picture.pixels.resize((width * height) as usize, 0);
        let (cx, cy) = (width as f32 * 0.5, height as f32 * 0.46);
        let reach = 0.4 * (width as f32).min(height as f32);
        if now.seeds.is_empty() {
            self.ground.paint(&mut picture, &[], (cx, cy), reach, s, t);
            // Nothing known yet: the seed before the laws.
            let breath = 0.5 + 0.5 * (t * TAU / 2.4).sin();
            picture.glow(cx, cy, (3.0 + 3.0 * breath) * s, GLOW, 0.35 + 0.3 * breath);
            picture.disc(cx, cy, 1.4 * s, GOLD, 0.8);
        } else {
            let shown = self.shown(cx, cy, reach, t);
            let lights: Vec<(f32, f32, bool)> = shown
                .iter()
                .zip(&now.seeds)
                .map(|(&(x, y), seed)| (x, y, seed.stage == Stage::Built))
                .collect();
            self.ground
                .paint(&mut picture, &lights, (cx, cy), reach, s, t);
            let mut degree = vec![0u32; shown.len()];
            for &(a, b) in &self.links {
                degree[a] += 1;
                degree[b] += 1;
                let built = |i: usize| now.seeds[i].stage == Stage::Built;
                let (colour, amount) = match (built(a), built(b)) {
                    (true, true) => (GOLD, 0.34),
                    (false, false) => (THREAD, 0.12),
                    _ => (mix(THREAD, GOLD, 0.5), 0.18),
                };
                picture.line(shown[a], shown[b], s.round().max(1.0), colour, amount);
            }
            for (i, seed) in now.seeds.iter().enumerate() {
                let (x, y) = shown[i];
                let big = 1.0 + 0.18 * (1.0 + degree[i] as f32).ln();
                let breath = 0.5 + 0.5 * (t * TAU / 1.6 + i as f32 * 0.7).sin();
                match seed.stage {
                    Stage::Waiting => {
                        picture.glow(x, y, 2.5 * s * big, DUST, 0.12);
                        picture.disc(x, y, 0.8 * s * big, DUST, 0.55);
                    }
                    Stage::Compiling => {
                        picture.glow(
                            x,
                            y,
                            (3.0 + 2.0 * breath) * s * big,
                            PALE,
                            0.2 + 0.2 * breath,
                        );
                        picture.disc(x, y, 1.0 * s * big, PALE, 0.8);
                    }
                    Stage::Compiled => {
                        picture.glow(x, y, 3.0 * s * big, PALE, 0.18);
                        picture.disc(x, y, 1.2 * s * big, PALE, 1.0);
                    }
                    Stage::Building => {
                        let grown = (seed.took.as_secs_f32() / 8.0).min(1.0);
                        picture.glow(
                            x,
                            y,
                            (4.0 + 3.0 * breath + 4.0 * grown) * s * big,
                            GLOW,
                            0.45 + 0.35 * breath,
                        );
                        picture.disc(x, y, 1.5 * s * big, GOLD, 1.0);
                    }
                    Stage::Built => {
                        let flare = self.stars[i]
                            .lit
                            .map_or(0.0, |lit| (-(t - lit).max(0.0) / 0.6).exp());
                        if flare > 0.01 {
                            picture.glow(x, y, (6.0 + 10.0 * flare) * s * big, GLOW, 0.9 * flare);
                        }
                        picture.glow(x, y, 5.0 * s * big, GOLD, 0.22);
                        picture.disc(x, y, 1.6 * s * big, GOLD, 1.0);
                        picture.disc(x, y, 0.7 * s * big, WHITE, 0.5 + 0.5 * flare);
                    }
                }
            }
            let mut said = format!("{} / {}", now.built(), now.seeds.len());
            if let Some(seed) = now.longest().filter(|seed| seed.took.as_secs() >= 1) {
                said.push_str(&format!(
                    "    {}  {} s",
                    seed.id.name(),
                    seed.took.as_secs()
                ));
            }
            let text = (s.round() as u32).max(1);
            let top = (height as f32 * 0.9) as u32;
            picture.words(
                &hud::panel(&said, text),
                top.min(height.saturating_sub(8 * text)),
                WORDS,
            );
        }
        self.picture = hud::Panel {
            width,
            height,
            pixels: picture.pixels,
        };
        &self.picture
    }

    /// Keeps a star for each seed, by its Kernel: one new rises on a
    /// sunflower's spiral, the seeds' order outward.
    fn follow(&mut self, now: &Preparing, t: f32) {
        let ids: Vec<&KernelId> = now.seeds.iter().map(|seed| &seed.id).collect();
        let same =
            self.known.len() == ids.len() && self.known.iter().zip(&ids).all(|(a, b)| a == *b);
        if !same || self.links != now.links {
            let mut old: HashMap<KernelId, Star> =
                self.known.drain(..).zip(self.stars.drain(..)).collect();
            for (k, id) in ids.iter().enumerate() {
                let star = old.remove(*id).unwrap_or_else(|| {
                    let (r, a) = (0.6 * (k as f32 + 0.5).sqrt(), k as f32 * 2.399_963);
                    Star {
                        x: r * a.cos(),
                        y: r * a.sin(),
                        lit: None,
                    }
                });
                self.stars.push(star);
                self.known.push((*id).clone());
            }
            self.links = now.links.clone();
            self.changed = t;
        }
        for (star, seed) in self.stars.iter_mut().zip(&now.seeds) {
            match (seed.stage, star.lit) {
                (Stage::Built, None) => star.lit = Some(t),
                (Stage::Built, Some(_)) => {}
                _ => star.lit = None,
            }
        }
    }

    /// A step of the stars drawn together: linked ones pull, all push each
    /// other off, the middle holds them; calmer the longer nothing changed.
    fn settle(&mut self, t: f32) {
        let n = self.stars.len();
        if n == 0 {
            return;
        }
        let heat = 0.01 + 0.3 * (-(t - self.changed).max(0.0) / 2.5).exp();
        let mut force = vec![(0.0f32, 0.0f32); n];
        for i in 0..n {
            for j in i + 1..n {
                let (mut dx, mut dy) = (
                    self.stars[i].x - self.stars[j].x,
                    self.stars[i].y - self.stars[j].y,
                );
                let mut d2 = dx * dx + dy * dy;
                if d2 < 1e-4 {
                    let a = (i * 7 + j * 13) as f32;
                    (dx, dy, d2) = (a.cos() * 0.01, a.sin() * 0.01, 1e-4);
                }
                if d2 > 9.0 {
                    continue;
                }
                let f = 1.0 / d2;
                force[i].0 += dx * f;
                force[i].1 += dy * f;
                force[j].0 -= dx * f;
                force[j].1 -= dy * f;
            }
        }
        for &(a, b) in &self.links {
            let (dx, dy) = (
                self.stars[b].x - self.stars[a].x,
                self.stars[b].y - self.stars[a].y,
            );
            let d = (dx * dx + dy * dy).sqrt();
            force[a].0 += dx * d;
            force[a].1 += dy * d;
            force[b].0 -= dx * d;
            force[b].1 -= dy * d;
        }
        let (mut mx, mut my) = (0.0, 0.0);
        for (star, (fx, fy)) in self.stars.iter_mut().zip(force) {
            let (fx, fy) = (fx - star.x * 0.02, fy - star.y * 0.02);
            let m = (fx * fx + fy * fy).sqrt().max(1e-6);
            let k = m.min(heat) / m;
            star.x += fx * k;
            star.y += fy * k;
            mx += star.x;
            my += star.y;
        }
        let (mx, my) = (mx / n as f32, my / n as f32);
        for star in &mut self.stars {
            star.x -= mx;
            star.y -= my;
        }
    }

    /// Where each star shows: turned as a galaxy, the rim further than the
    /// heart, all slowly; the disc a little tilted; its radius `reach`.
    fn shown(&mut self, cx: f32, cy: f32, reach: f32, t: f32) -> Vec<(f32, f32)> {
        let far = self
            .stars
            .iter()
            .map(|star| (star.x * star.x + star.y * star.y).sqrt())
            .fold(0.5f32, f32::max);
        self.far = if self.far == 0.0 {
            far
        } else {
            self.far + (far - self.far) * 0.08
        };
        let per = reach / self.far;
        self.stars
            .iter()
            .map(|star| {
                let r = (star.x * star.x + star.y * star.y).sqrt();
                let a = star.y.atan2(star.x) + 2.1 * (r / self.far).min(1.0).powf(1.3) + t * 0.02;
                (cx + r * per * a.cos(), cy + r * per * a.sin() * 0.78)
            })
            .collect()
    }
}

impl Ground {
    /// Made again only when the window's size or scale changes.
    fn fit(&mut self, width: u32, height: u32, s: f32) {
        let size = (width, height, (s * 100.0) as u32);
        if self.size == size {
            return;
        }
        let cell = (4.0 * s).round().max(4.0) as u32;
        let (columns, rows) = (width.div_ceil(cell) + 1, height.div_ceil(cell) + 1);
        let unit = (width.min(height) as f32 * 0.45).max(1.0);
        let mut clouds = Vec::with_capacity((columns * rows) as usize);
        for row in 0..rows {
            for column in 0..columns {
                let p = (
                    ((column * cell) as f32 - width as f32 * 0.5) / unit,
                    ((row * cell) as f32 - height as f32 * 0.46) / unit,
                );
                clouds.push([
                    fbm(p.0 * 2.2, p.1 * 2.2),
                    fbm(p.0 * 1.3 + 31.0, p.1 * 1.3 + 7.0),
                ]);
            }
        }
        // A far star in about one of each eighty logical cells three wide.
        let step = (3.0 * s).round().max(1.0) as u32;
        let mut far = Vec::new();
        for y in (0..height).step_by(step as usize) {
            for x in (0..width).step_by(step as usize) {
                let (i, j) = (x / step + 101, y / step + 211);
                if hash(i, j) < 0.012 {
                    let bright = 0.25 + 0.6 * hash(i + 7, j + 3);
                    far.push((y * width + x, bright, hash(i + 1, j + 9) * TAU));
                }
            }
        }
        *self = Ground {
            size,
            cell,
            columns,
            rows,
            clouds,
            far,
        };
    }

    /// The gas and the far stars: the clouds thick where `lights` stand
    /// dense (warm where they are built) and at the heart, `reach` round.
    fn paint(
        &self,
        picture: &mut Picture,
        lights: &[(f32, f32, bool)],
        (cx, cy): (f32, f32),
        reach: f32,
        s: f32,
        t: f32,
    ) {
        let (width, height) = (picture.width, picture.height);
        // How dense the stars stand, in cells four of the clouds' wide.
        let coarse = self.cell * 4;
        let (gw, gh) = (width.div_ceil(coarse) + 2, height.div_ceil(coarse) + 2);
        let mut dense = vec![[0.0f32; 2]; (gw * gh) as usize];
        let spread = (reach * 0.16).max(30.0 * s) / coarse as f32;
        let span = (spread * 2.5).ceil() as i64;
        for &(x, y, built) in lights {
            let (gx, gy) = (x / coarse as f32, y / coarse as f32);
            for row in gy as i64 - span..=gy as i64 + span {
                for column in gx as i64 - span..=gx as i64 + span {
                    if row < 0 || column < 0 || row >= gh as i64 || column >= gw as i64 {
                        continue;
                    }
                    let d2 = (column as f32 - gx).powi(2) + (row as f32 - gy).powi(2);
                    let w = (-d2 / (spread * spread)).exp();
                    let cell = &mut dense[(row as u32 * gw + column as u32) as usize];
                    cell[0] += w;
                    if built {
                        cell[1] += w;
                    }
                }
            }
        }
        let dense_at = |x: f32, y: f32| {
            let (gx, gy) = ((x / coarse as f32).max(0.0), (y / coarse as f32).max(0.0));
            let (c, r) = ((gx as u32).min(gw - 2), (gy as u32).min(gh - 2));
            let (fx, fy) = (gx - c as f32, gy - r as f32);
            let at = |c: u32, r: u32| dense[(r * gw + c) as usize];
            let (a, b, cc, d) = (at(c, r), at(c + 1, r), at(c, r + 1), at(c + 1, r + 1));
            let lerp = |k: usize| {
                let top = a[k] + (b[k] - a[k]) * fx;
                let bottom = cc[k] + (d[k] - cc[k]) * fx;
                top + (bottom - top) * fy
            };
            (lerp(0), lerp(1))
        };
        // The gas, a colour each cell, laid between the cells smoothly.
        let mut gas = vec![[0.0f32; 3]; self.clouds.len()];
        for row in 0..self.rows {
            for column in 0..self.columns {
                let i = (row * self.columns + column) as usize;
                let (x, y) = ((column * self.cell) as f32, (row * self.cell) as f32);
                let [cloud, tint] = self.clouds[i];
                let (all, built) = dense_at(x, y);
                let heart =
                    (-((x - cx).powi(2) + ((y - cy) / 0.78).powi(2)).sqrt() / (reach * 0.55)).exp();
                let gather = 1.0 - (-all * 0.35).exp();
                let warm = 1.0 - (-built * 0.35).exp();
                let thick = (cloud * 1.3 - 0.25).clamp(0.0, 1.0).powf(1.8)
                    * (0.12 + 1.3 * gather + 0.35 * heart);
                let hue = mix(THREAD, VIOLET, smooth(0.35, 0.7, tint));
                let hue = mix(hue, GOLD, warm * 0.7);
                for k in 0..3 {
                    gas[i][k] = GROUND[k] + hue[k] * thick * 0.7 + THREAD[k] * heart * 0.05;
                }
            }
        }
        // A row of the cells blended for each row of pixels, then across.
        let (cell, columns) = (self.cell as usize, self.columns as usize);
        let mut line = vec![[0.0f32; 3]; columns];
        for y in 0..height as usize {
            let (r, fy) = (y / cell, (y % cell) as f32 / cell as f32);
            let (top, bottom) = (&gas[r * columns..], &gas[(r + 1) * columns..]);
            for c in 0..columns {
                for k in 0..3 {
                    line[c][k] = (top[c][k] + (bottom[c][k] - top[c][k]) * fy) * picture.fade;
                }
            }
            let row = &mut picture.pixels[y * width as usize..(y + 1) * width as usize];
            for (c, span) in row.chunks_mut(cell).enumerate() {
                let (a, b) = (line[c], line[c + 1]);
                let step = [
                    (b[0] - a[0]) / cell as f32,
                    (b[1] - a[1]) / cell as f32,
                    (b[2] - a[2]) / cell as f32,
                ];
                let mut v = a;
                for pixel in span {
                    *pixel = ((v[0].clamp(0.0, 255.0) as u32) << 16)
                        | ((v[1].clamp(0.0, 255.0) as u32) << 8)
                        | v[2].clamp(0.0, 255.0) as u32;
                    for k in 0..3 {
                        v[k] += step[k];
                    }
                }
            }
        }
        for &(at, bright, phase) in &self.far {
            let twinkle = 0.7 + 0.3 * (t * 1.3 + phase).sin();
            picture.add_at(at as usize, WHITE, bright * twinkle * 0.8);
        }
    }
}

/// The light gathered on the window, each channel 0 to 255.
struct Picture {
    width: u32,
    height: u32,
    pixels: Vec<u32>,
    /// How far in the picture has faded, 0 to 1.
    fade: f32,
}

impl Picture {
    fn add_at(&mut self, at: usize, colour: [f32; 3], amount: f32) {
        let amount = amount * self.fade;
        let Some(pixel) = self.pixels.get_mut(at) else {
            return;
        };
        if amount <= 0.0 {
            return;
        }
        let mut out = 0u32;
        for (k, light) in colour.iter().enumerate() {
            let was = ((*pixel >> (16 - 8 * k)) & 255) as f32;
            out = (out << 8) | (was + light * amount).min(255.0) as u32;
        }
        *pixel = out;
    }

    fn add(&mut self, x: i64, y: i64, colour: [f32; 3], amount: f32) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return;
        }
        self.add_at((y as u32 * self.width + x as u32) as usize, colour, amount);
    }

    /// `colour` laid over what is there by `amount`.
    fn set(&mut self, x: i64, y: i64, colour: [f32; 3], amount: f32) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return;
        }
        let amount = (amount * self.fade).min(1.0);
        if amount <= 0.0 {
            return;
        }
        let pixel = &mut self.pixels[(y as u32 * self.width + x as u32) as usize];
        let mut out = 0u32;
        for (k, paint) in colour.iter().enumerate() {
            let was = ((*pixel >> (16 - 8 * k)) & 255) as f32;
            out = (out << 8) | (was + (paint - was) * amount) as u32;
        }
        *pixel = out;
    }

    /// A thread `thick` pixels from `a` to `b`, its light added, its edges
    /// shared between the pixels it passes.
    fn line(&mut self, a: (f32, f32), b: (f32, f32), thick: f32, colour: [f32; 3], amount: f32) {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let length = (dx * dx + dy * dy).sqrt();
        let steps = length.ceil().max(1.0) as usize;
        for step in 0..=steps {
            let f = step as f32 / steps as f32;
            for k in 0..thick as usize {
                let off = k as f32 - (thick - 1.0) * 0.5;
                let (x, y) = (
                    a.0 + dx * f - dy / length.max(1e-3) * off - 0.5,
                    a.1 + dy * f + dx / length.max(1e-3) * off - 0.5,
                );
                let (x0, y0) = (x.floor(), y.floor());
                let (fx, fy) = (x - x0, y - y0);
                let (x0, y0) = (x0 as i64, y0 as i64);
                self.add(x0, y0, colour, amount * (1.0 - fx) * (1.0 - fy));
                self.add(x0 + 1, y0, colour, amount * fx * (1.0 - fy));
                self.add(x0, y0 + 1, colour, amount * (1.0 - fx) * fy);
                self.add(x0 + 1, y0 + 1, colour, amount * fx * fy);
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
}

fn mix(a: [f32; 3], b: [f32; 3], f: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}

fn smooth(from: f32, to: f32, x: f32) -> f32 {
    let f = ((x - from) / (to - from)).clamp(0.0, 1.0);
    f * f * (3.0 - 2.0 * f)
}

/// A number from 0 to 1 for a whole point.
fn hash(x: u32, y: u32) -> f32 {
    let h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263));
    let h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    ((h ^ (h >> 16)) & 65535) as f32 / 65535.0
}

/// Smooth noise from 0 to 1, a whole point apart.
fn noise(x: f32, y: f32) -> f32 {
    let (ix, iy) = (x.floor(), y.floor());
    let (fx, fy) = (x - ix, y - iy);
    let (ux, uy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let (i, j) = ((ix as i32) as u32, (iy as i32) as u32);
    let a = hash(i, j) + (hash(i.wrapping_add(1), j) - hash(i, j)) * ux;
    let b = hash(i, j.wrapping_add(1))
        + (hash(i.wrapping_add(1), j.wrapping_add(1)) - hash(i, j.wrapping_add(1))) * ux;
    a + (b - a) * uy
}

/// Clouds: the noise at five sizes, each half the last.
fn fbm(x: f32, y: f32) -> f32 {
    let (mut sum, mut weight, mut size) = (0.0, 0.5, 1.0);
    for k in 0..5 {
        sum += noise(x * size + k as f32 * 17.3, y * size + k as f32 * 9.1) * weight;
        weight *= 0.5;
        size *= 2.03;
    }
    sum / 0.97
}

#[cfg(test)]
mod tests {
    use super::*;
    use entelechy_kernel::Seed;
    use std::time::Duration;

    fn seed(name: &str, stage: Stage, seconds: u64) -> Seed {
        Seed {
            id: KernelId::new(name, 1).expect("an id"),
            stage,
            took: Duration::from_secs(seconds),
        }
    }

    #[test]
    fn the_nebula_gathers_what_is_linked_and_lights_what_is_built() {
        let now = Preparing {
            seeds: vec![
                seed("t.a", Stage::Built, 0),
                seed("t.b", Stage::Built, 0),
                seed("t.c", Stage::Building, 3),
                seed("t.d", Stage::Waiting, 0),
                seed("t.e", Stage::Waiting, 0),
            ],
            links: vec![(0, 1), (1, 2)],
        };
        let mut sky = Sky::default();
        for frame in 0..90 {
            sky.draw(&now, 1.0 + frame as f64 / 30.0, 320, 200, 1.0);
        }
        let near = |a: usize, b: usize| {
            let (p, q) = (&sky.stars[a], &sky.stars[b]);
            ((p.x - q.x).powi(2) + (p.y - q.y).powi(2)).sqrt()
        };
        assert!(
            near(0, 1) < near(3, 4),
            "linked {} unlinked {}",
            near(0, 1),
            near(3, 4)
        );
        let picture = sky.draw(&now, 4.0, 320, 200, 1.0);
        assert_eq!((picture.width, picture.height), (320, 200));
        let red = |p: u32| (p >> 16) & 255;
        let brightest = picture.pixels.iter().map(|&p| red(p)).max().unwrap_or(0);
        assert!(brightest > 200, "a built star is gold: {brightest}");
        // The words stand under the nebula.
        assert!((178..200).any(|y| (0..320).any(|x| red(picture.pixels[y * 320 + x]) > 60)));
        // With nothing known, one seed in the middle.
        let mut alone = Sky::default();
        let picture = alone.draw(&Preparing::default(), 1.0, 320, 200, 1.0);
        assert!(red(picture.pixels[92 * 320 + 160]) > 60);
    }
}
