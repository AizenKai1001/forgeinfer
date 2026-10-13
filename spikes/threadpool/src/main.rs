//! Feasibility spike (throwaway): fork-join dispatch latency of three thread-pool designs, with 1 caller + (T-1) workers.
//! A CPU decode step has about 200 parallel regions (30 layers x 7 matrix-vector products); at 5 ms per token each region has about 25 us, so the dispatch cost decides the design.
//!   condvar : workers park on a Condvar (cheap when idle, but a wake-up costs tens of microseconds)
//!   spin    : workers busy-wait on an atomic generation counter (fast, but fragile when the machine is oversubscribed: a descheduled worker stalls everybody)
//!   hybrid  : spin for a short while, then yield the time slice for a while, then park on the Condvar
use std::hint::{black_box, spin_loop};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Condvar,
    Spin,
    Hybrid,
}

struct Shared {
    mode: Mode,
    n: usize,
    gen: AtomicU64,
    done: AtomicUsize,
    stop: AtomicBool,
    job: AtomicUsize, // pointer to a `&dyn Fn(usize)` fat pointer stored in the caller's frame (spike code: unsafe on purpose)
    lock: Mutex<()>,
    cv: Condvar,
}

type Job<'a> = &'a (dyn Fn(usize) + Sync);

fn worker(sh: Arc<Shared>, idx: usize) {
    let mut seen = 0u64;
    loop {
        // wait for a new generation
        let mut spins = 0u32;
        loop {
            let g = sh.gen.load(Ordering::Acquire);
            if g != seen {
                seen = g;
                break;
            }
            if sh.stop.load(Ordering::Relaxed) {
                return;
            }
            match sh.mode {
                Mode::Spin => spin_loop(),
                Mode::Condvar => {
                    let guard = sh.lock.lock().unwrap();
                    if sh.gen.load(Ordering::Acquire) == seen && !sh.stop.load(Ordering::Relaxed) {
                        let _ = sh.cv.wait_timeout(guard, Duration::from_millis(50)).unwrap();
                    }
                }
                Mode::Hybrid => {
                    spins += 1;
                    if spins < 4000 {
                        spin_loop();
                    } else if spins < 4400 {
                        thread::yield_now();
                    } else {
                        let guard = sh.lock.lock().unwrap();
                        if sh.gen.load(Ordering::Acquire) == seen && !sh.stop.load(Ordering::Relaxed) {
                            let _ = sh.cv.wait_timeout(guard, Duration::from_millis(50)).unwrap();
                        }
                    }
                }
            }
        }
        if sh.stop.load(Ordering::Acquire) {
            return;
        }
        let job: Job = unsafe { *(sh.job.load(Ordering::Acquire) as *const Job) };
        job(idx);
        sh.done.fetch_add(1, Ordering::AcqRel);
    }
}

struct Pool {
    sh: Arc<Shared>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl Pool {
    fn new(n: usize, mode: Mode) -> Pool {
        let sh = Arc::new(Shared { mode, n, gen: AtomicU64::new(0), done: AtomicUsize::new(0), stop: AtomicBool::new(false), job: AtomicUsize::new(0), lock: Mutex::new(()), cv: Condvar::new() });
        let handles = (1..n)
            .map(|i| {
                let s = sh.clone();
                thread::spawn(move || worker(s, i))
            })
            .collect();
        Pool { sh, handles }
    }

    fn run(&self, f: Job) {
        let sh = &self.sh;
        let fat: Job = f;
        sh.done.store(0, Ordering::Relaxed);
        sh.job.store(&fat as *const Job as usize, Ordering::Release);
        sh.gen.fetch_add(1, Ordering::AcqRel);
        if sh.mode != Mode::Spin {
            let _g = sh.lock.lock().unwrap();
            sh.cv.notify_all();
        }
        f(0);
        let want = sh.n - 1;
        let mut spins = 0u32;
        while sh.done.load(Ordering::Acquire) < want {
            spins += 1;
            if sh.mode == Mode::Spin || spins < 4000 {
                spin_loop();
            } else {
                thread::yield_now();
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::Relaxed);
        self.sh.gen.fetch_add(1, Ordering::AcqRel);
        {
            let _g = self.sh.lock.lock().unwrap();
            self.sh.cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn bench(mode: Mode, n: usize, work_floats: usize, rounds: usize) -> f64 {
    let pool = Pool::new(n, mode);
    let data: Vec<f32> = vec![1.0; work_floats * n];
    let sink = AtomicUsize::new(0);
    let f = |i: usize| {
        let s: f32 = data[i * work_floats..(i + 1) * work_floats].iter().sum();
        sink.fetch_add(s as usize, Ordering::Relaxed);
    };
    for _ in 0..200 {
        pool.run(&f);
    }
    let t = Instant::now();
    for _ in 0..rounds {
        pool.run(&f);
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / rounds as f64;
    black_box(sink.load(Ordering::Relaxed));
    us
}

fn main() {
    println!("us per fork-join dispatch (each worker sums {} floats = a few microseconds of work); the machine is shared with running services\n", 4096);
    println!("{:>8} {:>10} {:>10} {:>10}", "threads", "condvar", "spin", "hybrid");
    for n in [2usize, 4, 6, 8] {
        let c = bench(Mode::Condvar, n, 4096, 20_000);
        let s = bench(Mode::Spin, n, 4096, 20_000);
        let h = bench(Mode::Hybrid, n, 4096, 20_000);
        println!("{n:>8} {c:>10.1} {s:>10.1} {h:>10.1}");
    }
    // the same with a pause between rounds longer than the spin window (the pool goes idle between tokens)
    println!("\nafter 2 ms of idleness (a token boundary), first dispatch latency in us:");
    for (name, mode) in [("condvar", Mode::Condvar), ("spin", Mode::Spin), ("hybrid", Mode::Hybrid)] {
        let pool = Pool::new(4, mode);
        let sink = AtomicUsize::new(0);
        let f = |_i: usize| {
            sink.fetch_add(1, Ordering::Relaxed);
        };
        let mut total = 0.0;
        let reps = 200;
        for _ in 0..reps {
            thread::sleep(Duration::from_millis(2));
            let t = Instant::now();
            pool.run(&f);
            total += t.elapsed().as_secs_f64() * 1e6;
        }
        println!("{name:>8}: {:.1} us", total / reps as f64);
    }
}
