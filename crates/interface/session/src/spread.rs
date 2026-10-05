//! Work spread over the cores: each item mapped on whichever thread is
//! free, the results in the items' order. Where one core is all there is,
//! or no thread can be spawned (a page), each is mapped in turn here.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `f` of each item, in the items' order, on as many threads as there
/// are cores.
pub(crate) fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let workers = orla_rules::cores().min(items.len());
    let made: Mutex<Vec<Option<R>>> = Mutex::new(items.iter().map(|_| None).collect());
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            let at = next.fetch_add(1, Ordering::Relaxed);
            let Some(item) = items.get(at) else {
                break;
            };
            let result = f(item);
            made.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)[at] = Some(result);
        }
    };
    if workers >= 2 {
        std::thread::scope(|scope| {
            for _ in 0..workers {
                if std::thread::Builder::new()
                    .name("orla-spread".into())
                    .spawn_scoped(scope, work)
                    .is_err()
                {
                    break;
                }
            }
        });
    }
    // What no thread took (none spawned, or one core) is mapped here.
    work();
    made.into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .into_iter()
        .map(|result| result.expect("every item is mapped"))
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn each_item_is_mapped_once_in_its_order() {
        let items: Vec<u32> = (0..100).collect();
        assert_eq!(
            super::map(&items, |i| i * 2),
            (0..100).map(|i| i * 2).collect::<Vec<_>>()
        );
        assert!(super::map(&[] as &[u32], |i| *i).is_empty());
    }
}
