use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use parking_lot::Mutex as ParkingLotMutex;

use std::{
    sync::{Arc, Mutex as StdMutex},
    thread,
    time::Duration,
};

use numa_mutex::QueueMutex;

//
// ==============================
// CONFIG
// ==============================
//

const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];

const ITERATIONS: usize = 100_000;

//
// ==============================
// BENCH HELPERS
// ==============================
//

fn bench_std_mutex(threads: usize) {
    let mutex = Arc::new(StdMutex::new(0usize));

    thread::scope(|s| {
        for _ in 0..threads {
            let mutex = mutex.clone();

            s.spawn(move || {
                for _ in 0..ITERATIONS {
                    let mut guard = mutex.lock().unwrap();

                    *guard += 1;

                    black_box(*guard);
                }
            });
        }
    });
}

fn bench_parking_lot(threads: usize) {
    let mutex = Arc::new(ParkingLotMutex::new(0usize));

    thread::scope(|s| {
        for _ in 0..threads {
            let mutex = mutex.clone();

            s.spawn(move || {
                for _ in 0..ITERATIONS {
                    let mut guard = mutex.lock();

                    *guard += 1;

                    black_box(*guard);
                }
            });
        }
    });
}

fn bench_queue_mutex(threads: usize) {
    let mutex = Arc::new(QueueMutex::new(0usize));

    thread::scope(|s| {
        for _ in 0..threads {
            let mutex = mutex.clone();

            s.spawn(move || {
                for _ in 0..ITERATIONS {
                    let mut guard = mutex.lock();

                    *guard += 1;

                    black_box(*guard);
                }
            });
        }
    });
}

//
// ==============================
// CONTENTION BENCH
// ==============================
//

fn contention_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutex_contention");

    for &threads in THREAD_COUNTS {
        group.throughput(Throughput::Elements((threads * ITERATIONS) as u64));

        //
        // std::sync::Mutex
        //
        group.bench_with_input(BenchmarkId::new("std", threads), &threads, |b, &threads| {
            b.iter(|| bench_std_mutex(threads));
        });

        //
        // parking_lot
        //
        group.bench_with_input(
            BenchmarkId::new("parking_lot", threads),
            &threads,
            |b, &threads| {
                b.iter(|| bench_parking_lot(threads));
            },
        );

        //
        // QueueMutex
        //
        group.bench_with_input(
            BenchmarkId::new("queue_mutex", threads),
            &threads,
            |b, &threads| {
                b.iter(|| bench_queue_mutex(threads));
            },
        );
    }

    group.finish();
}

//
// ==============================
// LONG CRITICAL SECTION BENCH
// ==============================
//

fn long_critical_section(c: &mut Criterion) {
    let mut group = c.benchmark_group("long_critical_section");

    for &threads in THREAD_COUNTS {
        group.bench_with_input(
            BenchmarkId::new("queue_mutex", threads),
            &threads,
            |b, &threads| {
                b.iter(|| {
                    let mutex = Arc::new(QueueMutex::new(0usize));

                    thread::scope(|s| {
                        for _ in 0..threads {
                            let mutex = mutex.clone();

                            s.spawn(move || {
                                for _ in 0..1000 {
                                    let mut g = mutex.lock();

                                    //
                                    // Simulate work.
                                    //
                                    std::thread::sleep(Duration::from_micros(5));

                                    *g += 1;

                                    black_box(*g);
                                }
                            });
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

//
// ==============================
// MAIN
// ==============================
//

criterion_group!(benches, contention_benchmark, long_critical_section,);

criterion_main!(benches);
