//! Independent frames, with reusable per-worker storage and ordered results.

pub(crate) fn map<W, O: Send>(
    n: usize,
    threads: usize,
    init: impl Fn() -> W + Sync,
    run: impl Fn(usize, &mut W) -> O + Sync,
) -> Vec<O> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<O>>> = Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            scope.spawn(|| {
                let mut workspace = init();
                loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    if t >= n {
                        break;
                    }
                    let result = run(t, &mut workspace);
                    out.lock().expect("frame result lock")[t] = Some(result);
                }
            });
        }
    });
    out.into_inner()
        .expect("workers joined")
        .into_iter()
        .map(|o| o.expect("every frame was taken"))
        .collect()
}
