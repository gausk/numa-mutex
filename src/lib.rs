//#![cfg(target_os = "linux")]

use std::{
    cell::UnsafeCell,
    hint::spin_loop,
    ops::{Deref, DerefMut},
    sync::atomic::{
        AtomicU32,
        Ordering::{Acquire, Relaxed, Release},
    },
};

const UNLOCKED: u32 = 0;
const LOCKED: u32 = 1;
const CONTENDED: u32 = 2;

/// How long to spin before parking.
const SPIN_LIMIT: usize = 100;

/// NUMA-aware-ish adaptive futex mutex.
///
/// Current features:
/// - fast uncontended CAS path
/// - adaptive spinning
/// - futex parking
/// - cacheline aligned state
///
/// Future extensions:
/// - NUMA-local waiter queues
/// - topology-aware wakeups
/// - async integration
#[repr(align(64))]
pub struct NumaMutex<T> {
    state: AtomicU32,
    value: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for NumaMutex<T> {}
unsafe impl<T: Send> Sync for NumaMutex<T> {}

pub struct MutexGuard<'a, T> {
    mutex: &'a NumaMutex<T>,
}

impl<T> NumaMutex<T> {
    pub const fn new(value: T) -> Self {
        Self {
            state: AtomicU32::new(UNLOCKED),
            value: UnsafeCell::new(value),
        }
    }

    #[inline]
    pub fn lock(&self) -> MutexGuard<'_, T> {
        //
        // FAST PATH
        //
        if self
            .state
            .compare_exchange(UNLOCKED, LOCKED, Acquire, Relaxed)
            .is_ok()
        {
            return MutexGuard { mutex: self };
        }

        //
        // ADAPTIVE SPIN PHASE
        //
        for _ in 0..SPIN_LIMIT {
            let state = self.state.load(Relaxed);

            // Only attempt CAS if unlocked.
            if state == UNLOCKED {
                if self
                    .state
                    .compare_exchange(UNLOCKED, CONTENDED, Acquire, Relaxed)
                    .is_ok()
                {
                    return MutexGuard { mutex: self };
                }
            }

            spin_loop();
        }

        //
        // FUTEX SLOW PATH
        //
        loop {
            //
            // Mark mutex contended.
            //
            if self.state.swap(CONTENDED, Acquire) == UNLOCKED {
                break;
            }

            //
            // Sleep while contended.
            //
            wait(&self.state, CONTENDED);
        }

        MutexGuard { mutex: self }
    }

    #[inline]
    fn unlock(&self) {
        //
        // Release ownership.
        //
        if self.state.swap(UNLOCKED, Release) == CONTENDED {
            wake_one(&self.state);
        }
    }
}

impl<T> Drop for MutexGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.mutex.unlock();
    }
}

impl<T> Deref for MutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.value.get() }
    }
}

impl<T> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.value.get() }
    }
}

//
// ===== FUTEX =====
//

#[inline]
fn wait(a: &AtomicU32, expected: u32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected,
            std::ptr::null::<libc::timespec>(),
        );
    }
}

#[inline]
fn wake_one(a: &AtomicU32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicU32,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            1,
        );
    }
}

//
// ===== TEST =====
//

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, thread};

    #[test]
    fn test_mutex() {
        let mutex = Arc::new(NumaMutex::new(0usize));

        thread::scope(|s| {
            for _ in 0..8 {
                let mutex = mutex.clone();

                s.spawn(move || {
                    for _ in 0..100_000 {
                        let mut guard = mutex.lock();
                        *guard += 1;
                    }
                });
            }
        });

        assert_eq!(*mutex.lock(), 800_000);
    }
}
