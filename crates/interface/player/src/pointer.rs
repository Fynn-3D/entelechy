//! The pointer held in the window: the system's cursor locked and hidden,
//! the scene's pointer moved by the mouse's own motion and kept inside the
//! window, so it cannot leave it (nor reach its title bar) while the
//! window has the focus. The scene draws the pointer it reads.

/// Where a held pointer at `at` goes when the mouse moves `delta` (in the
/// system's logical units, `scale` physical pixels each): moved, and kept
/// on the window's `size` in physical pixels.
pub fn held_step(at: (f64, f64), delta: (f64, f64), scale: f64, size: (u32, u32)) -> (f64, f64) {
    let most = (
        f64::from(size.0.max(1)) - 1.0,
        f64::from(size.1.max(1)) - 1.0,
    );
    (
        (at.0 + delta.0 * scale).clamp(0.0, most.0),
        (at.1 + delta.1 * scale).clamp(0.0, most.1),
    )
}

#[cfg(test)]
mod tests {
    use super::held_step;

    #[test]
    fn a_held_pointer_moves_by_the_mouse_and_stays_in_the_window() {
        assert_eq!(
            held_step((100.0, 50.0), (3.0, -2.0), 2.0, (800, 600)),
            (106.0, 46.0)
        );
        // Pushed past an edge, it stays at that edge, the other way free.
        assert_eq!(
            held_step((790.0, 50.0), (40.0, 5.0), 1.0, (800, 600)),
            (799.0, 55.0)
        );
        assert_eq!(
            held_step((3.0, 2.0), (-10.0, -10.0), 1.0, (800, 600)),
            (0.0, 0.0)
        );
        // A window of no size holds it at its corner.
        assert_eq!(held_step((5.0, 5.0), (1.0, 1.0), 1.0, (0, 0)), (0.0, 0.0));
    }
}
