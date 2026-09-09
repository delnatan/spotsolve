//! The refactor guard PORTING_NOTES section 2 asks for, and did not have.
//!
//! Every refactor of this crate was accepted only if the whole-pipeline output
//! stayed bit-identical -- and section 2 says to keep a test that checks it,
//! because a tolerance-based fixture comparison does not. The layer 6 fixtures
//! match the *Python* to 1e-6 px (section 19: cross-language bit-identity is
//! not achievable and should not be chased). That is the right contract across
//! the seam and the wrong one within this crate, where the only honest
//! question about a refactor is whether it changed anything at all.
//!
//! This runs `detect`'s own pass order over every fixture case and hashes the
//! final `(positions, amplitudes, se)` of all of them together.
//!
//! # When this fails
//!
//! It means the arithmetic changed, nothing more. That is a defect **only if
//! you did not intend it**. A deliberate change -- a different summation order,
//! a reciprocal traded for a divide -- is a decision to make on purpose, with
//! the position shift measured against the ~1e-6 px per-fit contract of
//! section 19 and the reason written down. Re-baseline in that same commit,
//! and say in the message what moved and by how much. Never re-baseline to make
//! a red test green.
//!
//! It is machine-independent by construction: `libm` is a port of musl's, so
//! `erf` is bit-identical on macOS, Linux and Windows, and the release profile
//! forbids `target-cpu=native` precisely so LLVM cannot contract an FMA here.
//! A failure that appears only on one machine is a build-flag bug, not a
//! rounding difference to be tolerated.

mod common;

use common::*;
use spotsolve_core::evidence::Prior;
use spotsolve_core::passes::{self, Emitters, Frame, Solver};

/// FNV-1a over the raw f64 bit patterns.
///
/// Hand-rolled because the dev-dependency budget is one crate (`serde_json`,
/// to read the fixtures) and this needs to be a change detector, not a
/// cryptographic digest. `to_bits` and not `to_string`: the point is to catch
/// a one-ulp move, which any decimal formatting shorter than 17 digits hides.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn push(&mut self, v: f64) {
        // NaN has many bit patterns and REFINE reports it for a singular
        // Fisher block, so it is folded to one value: whether an SE is
        // unusable is the contract, which of the quiet NaNs it is is not.
        let bits = if v.is_nan() { 0x7ff8_0000_0000_0000 } else { v.to_bits() };
        for b in bits.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// Hash of every fixture case's final state, in fixture order.
///
/// Baselined 2026-09-09, against the blocked `lmcl::fisher` accumulators. That
/// change was accepted on exactly this value being unmoved from the scalar
/// loop that preceded it.
const EXPECTED: u64 = 0x5d92_8e8d_4df3_42b7;

#[test]
fn the_passes_are_bit_identical_to_their_baseline() {
    let fx = load("06_passes");
    let mut s = Solver::new();
    let mut hash = Fnv::new();

    for case in fx.cases() {
        let (h, w, d_e) = mat_at(case, "d_e");
        let (_, _, bmap) = mat_at(case, "bmap");
        let frame = Frame {
            d_e: &d_e,
            bmap: &bmap,
            h,
            w,
            sigma: f64_at(case, "sigma"),
            k_max: usize_at(case, "k_max"),
        };
        let prior = Prior { lam: f64_at(case, "lam"), a_s: f64_at(case, "A_s") };

        let (n, two, pos) = mat_at(case, "positions");
        assert_eq!(two, 2);
        let amp = vec_at(case, "amplitudes");
        assert_eq!(amp.len(), n);
        let mut em = Emitters::from_parts(pos, amp);

        let add = &case["add"];
        let (n_cand, two, cand) = mat_at(add, "cand");
        assert_eq!(two, 2);
        let camp = vec_at(add, "camp");
        assert_eq!(camp.len(), n_cand);
        passes::add_pass(&mut s, &frame, &mut em, &cand, &camp, prior);

        let (_, _, model) = mat_at(&case["split"], "model");
        passes::split_pass(&mut s, &frame, &mut em, &model, prior);

        let se = passes::refine(
            &mut s,
            &frame,
            &mut em,
            200,
            usize_at(&case["refine"], "max_sweeps"),
            passes::REFINE_TOL,
        );
        passes::prune(&mut s, &frame, &mut em, prior);

        // The emitter count is part of the fingerprint: a change that adds or
        // drops one must not be able to hash the same.
        hash.push(em.len() as f64);
        for i in 0..em.len() {
            hash.push(em.y(i));
            hash.push(em.x(i));
            hash.push(em.amp[i]);
        }
        // `se` is from REFINE, before PRUNE removed anything, so it is hashed
        // whole rather than indexed by the survivors.
        for &v in &se {
            hash.push(v);
        }
    }

    let got = hash.finish();
    assert_eq!(
        got,
        EXPECTED,
        "\nthe passes no longer reproduce their baseline bit for bit.\n\
         got 0x{got:016x}, expected 0x{EXPECTED:016x}.\n\
         If you did not mean to change the arithmetic, this is a bug.\n\
         If you did, measure what moved against the 1e-6 px contract, then\n\
         re-baseline EXPECTED in the same commit and say why in the message."
    );
}
