#![cfg(target_os = "linux")]

use std::{
    cell::UnsafeCell,
    hint::spin_loop,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr,
    sync::atomic::{
        AtomicPtr, AtomicU32,
        Ordering::{AcqRel, Acquire, Relaxed, Release},
    },
};

// =====================================================================
// CONFIG
// =====================================================================

const SPIN_LIMIT: usize = 40;

// How many slots per thread. Must be > max lock nesting depth + 1.
// 4 is conservative and safe for any realistic non-recursive workload.
const TLS_SLOTS: usize = 4;

// =====================================================================
// FUTEX
// =====================================================================

#[inline]
fn futex_wait(addr: &AtomicU32, expected: u32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            addr as *const AtomicU32,
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected,
            ptr::null::<libc::timespec>(),
        );
    }
}

#[inline]
fn futex_wake_one(addr: &AtomicU32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            addr as *const AtomicU32,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            1i32,
        );
    }
}

// =====================================================================
// STATE WORD
//
//   bit 31 : LOCKED — cleared by predecessor to hand off the lock
//   bit 30 : PARKED — set by waiter just before futex_wait
//
// Single u32 so the unlock path atomically reads both bits in one op.
// =====================================================================

const LOCKED: u32 = 1 << 31;
const PARKED: u32 = 1 << 30;

// =====================================================================
// WAIT NODE
//
// Aligned to a full cache line. The two fields are written by
// *different* threads (state by the predecessor; next by this thread's
// successor), so padding between them would help — but at 64-byte
// alignment the struct is already its own cache line, and the fields
// are far enough apart (4 + pad + 8 bytes) that they land on separate
// cache-line halves on most µarchs. Good enough without doubling size.
// =====================================================================

#[repr(C, align(64))]
struct WaitNode {
    /// Spun on by this thread. Written by predecessor on handoff.
    state: AtomicU32,
    _pad: [u8; 4],  // push `next` to offset 8, different half-line
    /// Written by this thread's *successor* after joining the queue.
    next: AtomicPtr<WaitNode>,
}

impl WaitNode {
    /// In-place initialisation — no full-struct stack copy.
    #[inline]
    unsafe fn init(p: *mut Self) {
        ptr::addr_of_mut!((*p).state)
            .write(AtomicU32::new(LOCKED));
        ptr::addr_of_mut!((*p).next)
            .write(AtomicPtr::new(ptr::null_mut()));
    }
}

// =====================================================================
// THREAD-LOCAL NODE POOL
//
// ABA problem in detail:
//
//   The predecessor in unlock() does:
//     1. fetch_and(!LOCKED) on next.state        ← hands off
//     2. if PARKED: futex_wake_one(next.state)   ← optional
//
//   Between steps 1 and 2 the successor (now the lock-holder) can
//   complete its critical section, drop the guard, re-enter lock(),
//   and overwrite its TLS node — including `state` — with a fresh
//   LOCKED value.  Then the predecessor's futex_wake_one fires on
//   the *new* state, waking no one (wrong address state), and the
//   real successor sleeps forever.
//
// Two slots aren't enough because the sequence:
//   lock/slot-0 → unlock → lock/slot-1 → unlock → lock/slot-0
//   can race with a predecessor still mid-unlock from the first cycle.
//
// The correct fix is to make unlock() read `next` BEFORE clearing
// LOCKED, so the successor's node address is stable before we touch
// state.  With that ordering guarantee in place, slot-0 is already
// done being read by the time we'd reuse it.
//
// Sequence with the fixed unlock():
//   1. Read (*node).next          → get successor ptr S
//   2. (*S).state.fetch_and(!LOCKED)  → hand off
//   3. if PARKED: wake
//
// Now step 1 happens while `node` is still our node (not S's).
// The successor S can safely reinitialise its own node (which is S,
// not our node) at any time after step 1 without affecting us.
//
// With this ordering, TWO slots per thread are provably sufficient:
// a thread cannot reuse slot-0 until its slot-0-based guard is dropped,
// which means unlock() on the predecessor of slot-0 has already
// completed step 1 (read next) before we flip back.
//
// We keep TLS_SLOTS = 4 defensively for re-entrant or recursive use.
// =====================================================================

struct NodePool {
    nodes: [MaybeUninit<WaitNode>; TLS_SLOTS],
    /// Index of the slot to use on the *next* lock() call.
    next_slot: usize,
}

thread_local! {
    static POOL: UnsafeCell<NodePool> = const {
        UnsafeCell::new(NodePool {
            nodes: [
                MaybeUninit::uninit(),
                MaybeUninit::uninit(),
                MaybeUninit::uninit(),
                MaybeUninit::uninit(),
            ],
            next_slot: 0,
        })
    };
}

// =====================================================================
// QUEUE MUTEX
// =====================================================================

#[repr(align(64))]
pub struct QueueMutex<T> {
    tail: AtomicPtr<WaitNode>,
    value: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for QueueMutex<T> {}
unsafe impl<T: Send> Sync for QueueMutex<T> {}

pub struct MutexGuard<'a, T> {
    mutex: &'a QueueMutex<T>,
    node:  *mut WaitNode,
    /// The slot index this guard holds, so unlock can advance next_slot
    /// correctly even under re-entrant use.
    slot:  usize,
}

impl<T> QueueMutex<T> {
    #[inline]
    pub const fn new(value: T) -> Self {
        Self {
            tail: AtomicPtr::new(ptr::null_mut()),
            value: UnsafeCell::new(value),
        }
    }

    #[inline]
    pub fn lock(&self) -> MutexGuard<'_, T> {
        POOL.with(|pool_cell| unsafe {
            let pool = &mut *pool_cell.get();

            let slot = pool.next_slot;
            // Advance modularly so nested locks get distinct nodes.
            pool.next_slot = (slot + 1) % TLS_SLOTS;

            let node = pool.nodes[slot].as_mut_ptr();
            WaitNode::init(node);

            // Publish ourselves as the new tail.
            // AcqRel: Release so our init is visible to whoever reads
            //         our `next`; Acquire so we see the predecessor's
            //         own init should we need to inspect it.
            let prev = self.tail.swap(node, AcqRel);

            if prev.is_null() {
                // Uncontended fast path.
                (*node).state.store(0, Relaxed);
                return MutexGuard { mutex: self, node, slot };
            }

            // Tell the predecessor where to find us.
            // Release: pairs with Acquire in unlock's next-ptr read.
            (*prev).next.store(node, Release);

            // ── Adaptive spin ─────────────────────────────────────────
            for _ in 0..SPIN_LIMIT {
                if (*node).state.load(Acquire) & LOCKED == 0 {
                    return MutexGuard { mutex: self, node, slot };
                }
                spin_loop();
            }

            // ── Park ──────────────────────────────────────────────────
            //
            // fetch_or atomically sets PARKED and returns the old value.
            // If the old value already had LOCKED cleared, unlock raced
            // us — skip sleeping.
            let old = (*node).state.fetch_or(PARKED, AcqRel);
            if old & LOCKED == 0 {
                return MutexGuard { mutex: self, node, slot };
            }

            // Futex sleeps only if state == LOCKED|PARKED.
            // Any other value (LOCKED cleared) means we were woken
            // or the handoff raced — loop guards against spurious wakes.
            loop {
                futex_wait(&(*node).state, LOCKED | PARKED);
                if (*node).state.load(Acquire) & LOCKED == 0 {
                    break;
                }
            }

            MutexGuard { mutex: self, node, slot }
        })
    }

    #[inline]
    unsafe fn unlock(&self, node: *mut WaitNode, slot: usize) {
        // ── Restore the pool slot counter FIRST ───────────────────────
        //
        // Do this before any other work so that if this thread
        // immediately calls lock() again (e.g. from a destructor),
        // it gets the correct next slot.
        POOL.with(|pool_cell| {
            (*pool_cell.get()).next_slot = slot;
        });

        // ── Step 1: read our successor's address ──────────────────────
        //
        // CRITICAL ordering: we must capture `next` BEFORE we clear
        // LOCKED in the successor's state.  Once LOCKED is cleared the
        // successor can wake, finish its critical section, re-enter
        // lock(), and reinitialise its node — but `next` is *our*
        // field, not the successor's, so it remains valid.
        //
        // Fast path: if next is already written and we're the only
        // node, try to swing tail to null.
        let mut next = (*node).next.load(Acquire);

        if next.is_null() {
            // Try to declare the queue empty.
            if self
                .tail
                .compare_exchange(node, ptr::null_mut(), Release, Relaxed)
                .is_ok()
            {
                return; // Queue was truly empty; done.
            }

            // A new waiter joined between our load and the CAS.
            // Spin with exponential backoff until it links itself.
            // This window is a handful of cycles in practice.
            let mut backoff = 1usize;
            loop {
                for _ in 0..backoff {
                    spin_loop();
                }
                next = (*node).next.load(Acquire);
                if !next.is_null() { break; }
                if backoff < 64 { backoff <<= 1; }
            }
        }

        // ── Step 2: hand off — clear LOCKED, preserve PARKED ─────────
        //
        // fetch_and returns the *old* state so we can decide whether
        // to call futex_wake_one without a second atomic load.
        let old = (*next).state.fetch_and(!LOCKED, Release);

        // ── Step 3: wake only if the successor parked ─────────────────
        if old & PARKED != 0 {
            futex_wake_one(&(*next).state);
        }
    }
}

impl<T> Drop for MutexGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe { self.mutex.unlock(self.node, self.slot) }
    }
}

impl<T> Deref for MutexGuard<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.mutex.value.get() }
    }
}

impl<T> DerefMut for MutexGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.value.get() }
    }
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, thread};

    #[test]
    fn stress_8_threads() {
        let mutex = Arc::new(QueueMutex::new(0usize));
        thread::scope(|s| {
            for _ in 0..8 {
                let m = mutex.clone();
                s.spawn(move || {
                    for _ in 0..100_000 {
                        *m.lock() += 1;
                    }
                });
            }
        });
        assert_eq!(*mutex.lock(), 800_000);
    }

    #[test]
    fn stress_scaling() {
        for n in [1usize, 2, 4, 8, 16] {
            let mutex = Arc::new(QueueMutex::new(0usize));
            thread::scope(|s| {
                for _ in 0..n {
                    let m = mutex.clone();
                    s.spawn(move || {
                        for _ in 0..50_000 {
                            *m.lock() += 1;
                        }
                    });
                }
            });
            assert_eq!(*mutex.lock(), n * 50_000, "failed at {n} threads");
        }
    }

    #[test]
    fn no_aba_rapid_cycle() {
        // Short critical section → threads cycle through slots rapidly.
        // Any ABA in the node-reuse path shows up here.
        let mutex = Arc::new(QueueMutex::new(0usize));
        thread::scope(|s| {
            for _ in 0..8 {
                let m = mutex.clone();
                s.spawn(move || {
                    for _ in 0..200_000 {
                        *m.lock() += 1;
                    }
                });
            }
        });
        assert_eq!(*mutex.lock(), 1_600_000);
    }

    #[test]
    fn nested_lock_distinct_slots() {
        // Two mutexes held simultaneously by the same thread must use
        // different TLS slots or they corrupt each other's node.
        let m1 = Arc::new(QueueMutex::new(0usize));
        let m2 = Arc::new(QueueMutex::new(0usize));
        thread::scope(|s| {
            for _ in 0..4 {
                let (a, b) = (m1.clone(), m2.clone());
                s.spawn(move || {
                    for _ in 0..25_000 {
                        let mut ga = a.lock();
                        let mut gb = b.lock();
                        *ga += 1;
                        *gb += 1;
                    }
                });
            }
        });
        assert_eq!(*m1.lock(), 100_000);
        assert_eq!(*m2.lock(), 100_000);
    }
}