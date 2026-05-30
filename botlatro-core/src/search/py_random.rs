//! Bit-exact port of CPython's `random.Random` (MT19937) — just enough to
//! reproduce `Random(seed).sample(range(n), k)` for native-beam draw parity.
//!
//! The native beam's inter-node draws come from `DeckModel.sample_draws`, which
//! is `tuple(pool[i] for i in rng.sample(range(len(pool)), n))` with
//! `rng = random.Random(seed)`. The two prior native-beam attempts used xoshiro
//! and diverged from Python's MT19937 at every branch. This module reproduces
//! Python's RNG bit-for-bit so the Rust beam can compute the SAME draws.
//!
//! Verified against CPython via `py_sample_indices` (see the parity test in
//! Python). Covers: int seeding (init_by_array over the seed's 32-bit words),
//! genrand_uint32, getrandbits(k), _randbelow(n), and `sample` (both the
//! pool/selection algorithm for small n and the set-based algorithm for large n).

use pyo3::prelude::*;
use std::collections::HashSet;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct MtRandom {
    mt: [u32; N],
    mti: usize,
}

impl MtRandom {
    /// `init_genrand` — CPython `init_genrand`.
    fn init_genrand(seed: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = seed;
        for i in 1..N {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, mti: N }
    }

    /// `init_by_array` — CPython `init_by_array`. `random.seed(int)` builds the
    /// key from the absolute value's little-endian 32-bit words and calls this.
    fn init_by_array(key: &[u32]) -> Self {
        let mut r = Self::init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        let key_len = key.len().max(1);
        let mut k = N.max(key.len());
        while k > 0 {
            r.mt[i] = (r.mt[i]
                ^ ((r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1_664_525)))
                .wrapping_add(*key.get(j).unwrap_or(&0))
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                r.mt[0] = r.mt[N - 1];
                i = 1;
            }
            if j >= key_len {
                j = 0;
            }
            k -= 1;
        }
        let mut k = N - 1;
        while k > 0 {
            r.mt[i] = (r.mt[i]
                ^ ((r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                r.mt[0] = r.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        r.mt[0] = 0x8000_0000;
        r
    }

    /// Seed from a non-negative integer the way `random.seed(int)` does:
    /// the key is the value's 32-bit little-endian words (at least one word).
    pub fn seed_from_u128(seed: u128) -> Self {
        let mut key: Vec<u32> = Vec::new();
        let mut v = seed;
        if v == 0 {
            key.push(0);
        }
        while v > 0 {
            key.push((v & 0xffff_ffff) as u32);
            v >>= 32;
        }
        Self::init_by_array(&key)
    }

    /// CPython `genrand_uint32` — generate the next 32-bit word.
    fn genrand_uint32(&mut self) -> u32 {
        if self.mti >= N {
            let mag01 = [0u32, MATRIX_A];
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// CPython `random_getrandbits(k)` for k >= 1.
    fn getrandbits(&mut self, k: u32) -> u128 {
        if k <= 32 {
            return (self.genrand_uint32() >> (32 - k)) as u128;
        }
        let mut result: u128 = 0;
        let mut shift = 0u32;
        let mut remaining = k;
        while remaining > 0 {
            let take = remaining.min(32);
            let mut word = self.genrand_uint32();
            if take < 32 {
                word >>= 32 - take;
            }
            result |= (word as u128) << shift;
            shift += 32;
            remaining -= take;
        }
        result
    }

    /// CPython `Random._randbelow_with_getrandbits(n)` (n > 0).
    fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = 64 - (n.leading_zeros()); // bit_length
        let mut r = self.getrandbits(k) as u64;
        while r >= n {
            r = self.getrandbits(k) as u64;
        }
        r
    }

    /// CPython `random.sample(range(n), k)` returning the chosen indices in
    /// selection order. Mirrors the stdlib's setsize/pool/set branching.
    pub fn sample_indices(&mut self, n: u64, k: u64) -> Vec<u64> {
        let mut result = vec![0u64; k as usize];
        // setsize = 21; if k > 5: setsize += 4 ** ceil(log(k*3, 4))
        let mut setsize: u64 = 21;
        if k > 5 {
            // ceil(log(k*3, 4)) == number of base-4 digits to represent (k*3 - 1)
            let target = (k as f64) * 3.0;
            let exp = (target.ln() / 4f64.ln()).ceil() as u32;
            setsize += 4u64.saturating_pow(exp);
        }
        if n <= setsize {
            // pool / selection-sampling
            let mut pool: Vec<u64> = (0..n).collect();
            for i in 0..k {
                let j = self.randbelow(n - i);
                result[i as usize] = pool[j as usize];
                pool[j as usize] = pool[(n - i - 1) as usize];
            }
        } else {
            let mut selected: HashSet<u64> = HashSet::new();
            for i in 0..k {
                let mut j = self.randbelow(n);
                while selected.contains(&j) {
                    j = self.randbelow(n);
                }
                selected.insert(j);
                result[i as usize] = j;
            }
        }
        result
    }
}

/// Test/verification hook: `Random(seed).sample(range(n), k)` as indices.
#[pyfunction]
pub fn py_sample_indices(seed: u64, n: u64, k: u64) -> Vec<u64> {
    let mut r = MtRandom::seed_from_u128(seed as u128);
    r.sample_indices(n, k)
}

/// Test/verification hook: first `count` outputs of `Random(seed).getrandbits(32)`.
#[pyfunction]
pub fn py_getrandbits32(seed: u64, count: usize) -> Vec<u32> {
    let mut r = MtRandom::seed_from_u128(seed as u128);
    (0..count).map(|_| r.genrand_uint32()).collect()
}
