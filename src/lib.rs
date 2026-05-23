#![cfg(target_os = "linux")]

use std::{
    cell::UnsafeCell,
    hint::spin_loop,
    marker::PhantomPinned,
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr,
    sync::atomic::{
        AtomicBool, AtomicPtr,
        Ordering::{Acquire, Relaxed, Release},
    },
};

//
// ==============================
// CONFIG
// ==============================
//

const SPIN_LIMIT: usize = 64;

//
// ==============================
// FUTEX HELPERS
// ==============================
//

#[inline]
fn futex_wait(a: &AtomicBool, expected: bool) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicBool,
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected as i32,
            ptr::null::<libc::timespec>(),
        );
    }
}

#[inline]
fn futex_wake_one(a: &AtomicBool) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            a as *const AtomicBool,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            1,
        );
    }
}

//
// ==============================
// WAIT NODE
// ==============================
//

struct WaitNode {
    next: AtomicPtr<WaitNode>,

    //
    // true  => waiting
    // false => lock granted
    //
    waiting: AtomicBool,

    //
    // Prevent move after linking into queue.
    //
    _pin: PhantomPinned,
}

impl WaitNode {
    fn new() -> Pin<Box<Self>> {
        Box::pin(Self {
            next: AtomicPtr::new(ptr::null_mut()),
            waiting: AtomicBool::new(true),
            _pin: PhantomPinned,
        })
    }
}

//
// ==============================
// QUEUED PARKING MUTEX
// ==============================
//

#[repr(align(64))]
pub struct QueueMutex<T> {
    //
    // Tail of MCS queue.
    //
    tail: AtomicPtr<WaitNode>,

    value: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for QueueMutex<T> {}
unsafe impl<T: Send> Sync for QueueMutex<T> {}

pub struct MutexGuard<'a, T> {
    mutex: &'a QueueMutex<T>,
}

impl<T> QueueMutex<T> {
    pub const fn new(value: T) -> Self {
        Self {
            tail: AtomicPtr::new(ptr::null_mut()),
            value: UnsafeCell::new(value),
        }
    }

    #[inline]
    pub fn lock(&self) -> MutexGuard<'_, T> {
        //
        // Allocate pinned wait node.
        //
        let node = WaitNode::new();

        //
        // SAFETY:
        // Node is pinned and stable in memory.
        //
        let node_ptr = unsafe { Pin::into_inner_unchecked(node) as *mut WaitNode };

        //
        // Join queue.
        //
        let prev = self.tail.swap(node_ptr, Acquire);

        //
        // Fast path:
        // queue was empty.
        //
        if prev.is_null() {
            unsafe {
                (*node_ptr).waiting.store(false, Relaxed);
            }

            //
            // Leak node ownership into guard.
            //
            std::mem::forget(unsafe { Box::from_raw(node_ptr) });

            return MutexGuard { mutex: self };
        }

        //
        // Link from predecessor.
        //
        unsafe {
            (*prev).next.store(node_ptr, Release);
        }

        //
        // Adaptive spin.
        //
        for _ in 0..SPIN_LIMIT {
            let waiting = unsafe { (*node_ptr).waiting.load(Acquire) };

            if !waiting {
                std::mem::forget(unsafe { Box::from_raw(node_ptr) });

                return MutexGuard { mutex: self };
            }

            spin_loop();
        }

        //
        // Park via futex.
        //
        loop {
            let waiting = unsafe { (*node_ptr).waiting.load(Acquire) };

            if !waiting {
                break;
            }

            futex_wait(unsafe { &(*node_ptr).waiting }, true);
        }

        //
        // Node consumed by queue.
        //
        std::mem::forget(unsafe { Box::from_raw(node_ptr) });

        MutexGuard { mutex: self }
    }

    #[inline]
    fn unlock(&self) {
        //
        // Current owner node is implicit.
        //
        // We reconstruct a temporary stack node
        // to detect whether queue becomes empty.
        //
        // Real production implementation should
        // store owner node in thread-local storage.
        //

        //
        // Simplified unlock:
        // walk queue tail carefully.
        //
        let tail = self.tail.load(Acquire);

        if tail.is_null() {
            return;
        }

        //
        // Wait for successor linkage.
        //
        let mut next = unsafe { (*tail).next.load(Acquire) };

        //
        // No successor.
        //
        if next.is_null() {
            if self
                .tail
                .compare_exchange(tail, ptr::null_mut(), Release, Relaxed)
                .is_ok()
            {
                return;
            }

            //
            // Someone joined concurrently.
            //
            while next.is_null() {
                spin_loop();

                next = unsafe { (*tail).next.load(Acquire) };
            }
        }

        //
        // Direct handoff.
        //
        unsafe {
            (*next).waiting.store(false, Release);
        }

        futex_wake_one(unsafe { &(*next).waiting });
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

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.value.get() }
    }
}

impl<T> DerefMut for MutexGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.value.get() }
    }
}

//
// ==============================
// TEST
// ==============================
//

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, thread};

    #[test]
    fn stress() {
        let mutex = Arc::new(QueueMutex::new(0usize));

        thread::scope(|s| {
            for _ in 0..8 {
                let mutex = mutex.clone();

                s.spawn(move || {
                    for _ in 0..100_000 {
                        let mut g = mutex.lock();
                        *g += 1;
                    }
                });
            }
        });

        assert_eq!(*mutex.lock(), 800_000);
    }
}
