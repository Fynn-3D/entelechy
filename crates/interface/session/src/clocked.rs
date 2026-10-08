//! Each node's share of a clocked frame: the runs it gave the device,
//! between its marks, timed once the device has done them. Nothing waits:
//! a frame is told a few frames after it ran, by its number.

use std::collections::VecDeque;
use std::ops::Range;
use std::time::Duration;

use orla_graph::NodeId;
use orla_kernel::ClockedFrame;

/// The frames ended and not yet told, each its nodes' runs: as many as the
/// device's clock keeps waiting.
const FRAMES_KEPT: usize = 8;

/// A clocked frame's device time, node by node.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceTimes {
    /// The frame's number, counted from the first frame clocked.
    pub frame: u64,
    /// The device's whole busy time on it.
    pub busy: Duration,
    /// Each node that gave the device a run, its runs' busy time, in
    /// evaluation order; a fused group under the node it ran as.
    pub nodes: Vec<(NodeId, Duration)>,
    /// The runs no node gave between its marks: what came before the
    /// first, between two, or after the last (a frame's sound, say).
    pub between: Duration,
}

/// Each node's runs in a frame, between its marks.
type Owned = Vec<(NodeId, Range<usize>)>;

/// The frames ended, their nodes' runs, until the device tells them.
#[derive(Debug, Default)]
pub(crate) struct NodeClock {
    frames: VecDeque<(u64, Owned)>,
}

impl NodeClock {
    /// Frame `number` ended, its nodes having given the device `clocked`.
    pub(crate) fn ended(&mut self, number: u64, clocked: Owned) {
        self.frames.push_back((number, clocked));
        while self.frames.len() > FRAMES_KEPT {
            self.frames.pop_front();
        }
    }

    /// Forgets the frames kept, as clocking stops.
    pub(crate) fn clear(&mut self) {
        self.frames.clear();
    }

    /// The device's frame `done`, its runs shared out to the nodes that
    /// gave them; `None` for a frame this clock no longer keeps.
    pub(crate) fn times(&self, done: &ClockedFrame) -> Option<DeviceTimes> {
        let (_, clocked) = self
            .frames
            .iter()
            .find(|(number, _)| *number == done.frame)?;
        let mut owned = vec![false; done.runs.len()];
        let mut nodes = Vec::new();
        for (node, runs) in clocked {
            let runs = runs.start.min(done.runs.len())..runs.end.min(done.runs.len());
            if runs.is_empty() {
                continue;
            }
            owned[runs.clone()].iter_mut().for_each(|o| *o = true);
            nodes.push((node.clone(), done.runs[runs].iter().sum()));
        }
        let between = done
            .runs
            .iter()
            .zip(&owned)
            .filter(|(_, owned)| !**owned)
            .map(|(run, _)| *run)
            .sum();
        Some(DeviceTimes {
            frame: done.frame,
            busy: done.busy(),
            nodes,
            between,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn a_frames_runs_go_to_the_nodes_that_gave_them_and_the_rest_between() {
        let mut clock = NodeClock::default();
        let a = NodeId("a".into());
        let b = NodeId("b".into());
        let idle = NodeId("idle".into());
        clock.ended(3, vec![(a.clone(), 1..3), (idle, 3..3), (b.clone(), 3..4)]);
        let done = ClockedFrame {
            frame: 3,
            runs: vec![ms(1), ms(2), ms(3), ms(4), ms(5)],
        };
        let times = clock.times(&done).unwrap();
        assert_eq!(
            times.nodes,
            vec![(a, ms(5)), (b, ms(4))],
            "a node that gave no run is left out"
        );
        assert_eq!(
            times.between,
            ms(6),
            "the run before the first node and the one after the last"
        );
        assert_eq!(times.busy, ms(15));
        assert_eq!(times.frame, 3);
        // A frame told after the clock let it go is not shared out.
        for n in 4..4 + FRAMES_KEPT as u64 {
            clock.ended(n, Vec::new());
        }
        assert_eq!(clock.times(&done), None);
    }
}
