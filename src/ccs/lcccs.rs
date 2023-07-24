use super::util::{compute_sum_Mz, VirtualPolynomial};
use super::{CCCSInstance, CCSWitness, CCS};
use crate::ccs::util::compute_all_sum_Mz_evals;
use crate::hypercube::BooleanHypercube;
use crate::spartan::math::Math;
use crate::spartan::polynomial::MultilinearPolynomial;
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_FE_FOR_RO, NUM_HASH_BITS},
  errors::NovaError,
  gadgets::{
    nonnative::{bignat::nat_to_limbs, util::f_to_nat},
    utils::scalar_as_base,
  },
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, R1CS},
  traits::{
    commitment::CommitmentEngineTrait, commitment::CommitmentTrait, AbsorbInROTrait, Group, ROTrait,
  },
  utils::*,
  Commitment, CommitmentKey, CE,
};
use bitvec::vec;
use core::{cmp::max, marker::PhantomData};
use ff::{Field, PrimeField};
use flate2::{write::ZlibEncoder, Compression};
use itertools::concat;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::ops::{Add, Mul};
use std::sync::Arc;

/// A type that holds a LCCCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LCCCS<G: Group> {
  pub(crate) w_comm: Commitment<G>,
  pub(crate) x: Vec<G::Scalar>,
  pub(crate) u: G::Scalar,
  pub(crate) v: Vec<G::Scalar>,
  // Random evaluation point for the v_i
  pub r_x: Vec<G::Scalar>,
  pub(crate) z: Vec<G::Scalar>,
}

impl<G: Group> LCCCS<G> {
  // XXX: Double check that this is indeed correct.
  /// Samples public parameters for the specified number of constraints and variables in an CCS
  // pub fn commitment_key(&self) -> CommitmentKey<G> {
  //   let total_nz = self.ccs.M.iter().fold(0, |acc, m| acc + m.coeffs().len());

  //   G::CE::setup(b"ck", max(max(self.ccs.m, self.ccs.t), total_nz))
  // }

  /// Checks if the CCS instance is satisfiable given a witness and its shape
  pub fn is_sat(
    &self,
    ccs: &CCS<G>,
    ccs_m_mle: &[MultilinearPolynomial<G::Scalar>],
    ck: &CommitmentKey<G>,
  ) -> Result<(), NovaError> {
    let w = &self.z[(1 + ccs.l)..];
    // check that C is the commitment of w. Notice that this is not verifying a Pedersen
    // opening, but checking that the Commmitment comes from committing to the witness.
    let comm_eq = self.w_comm == CE::<G>::commit(ck, w);

    let computed_v = compute_all_sum_Mz_evals::<G>(ccs_m_mle, w, &self.r_x, ccs.s_prime);
    let vs_eq = computed_v == self.v;

    if vs_eq && comm_eq {
      Ok(())
    } else {
      Err(NovaError::UnSat)
    }
  }

  /// Compute all L_j(x) polynomials
  // Can we recieve the MLE of z directy?
  pub fn compute_Ls(
    &self,
    ccs: &CCS<G>,
    ccs_m_mle: &[MultilinearPolynomial<G::Scalar>],
    ck: &CommitmentKey<G>,
    z: &[G::Scalar],
  ) -> Vec<VirtualPolynomial<G::Scalar>> {
    let z_mle = dense_vec_to_mle(ccs.s_prime, z);

    let mut vec_L_j_x = Vec::with_capacity(ccs.t);
    for M_j in ccs_m_mle.iter() {
      // Sanity check
      assert_eq!(z_mle.get_num_vars(), ccs.s_prime);

      let sum_Mz = compute_sum_Mz::<G>(M_j, &z_mle);
      let sum_Mz_virtual = VirtualPolynomial::new_from_mle(&Arc::new(sum_Mz), G::Scalar::ONE);
      let L_j_x = sum_Mz_virtual.build_f_hat(&self.r_x).unwrap();
      vec_L_j_x.push(L_j_x);
    }

    vec_L_j_x
  }
}

#[cfg(test)]
mod tests {
  use pasta_curves::{Ep, Fq};
  use rand_core::OsRng;

  use super::*;

  fn satisfied_ccs_is_satisfied_lcccs_with<G: Group>() {
    // Gen test vectors & artifacts
    let z = CCS::<Ep>::get_test_z(3);
    let (ccs, witness, instance, mles) = CCS::<Ep>::gen_test_ccs(&z);
    let ck = ccs.commitment_key();
    assert!(ccs.is_sat(&ck, &instance, &witness).is_ok());

    // LCCCS with the correct z should pass
    let (lcccs, _) = ccs.to_lcccs(&mut OsRng, &ck, &z);
    assert!(lcccs.is_sat(&ck, &witness).is_ok());

    // Wrong z so that the relation does not hold
    let mut bad_z = z;
    bad_z[3] = G::Scalar::ZERO;

    // LCCCS with the wrong z should not pass `is_sat`.
    // LCCCS with the correct z should pass
    let (lcccs, _) = ccs.to_lcccs(&mut OsRng, &ck, &bad_z);
    assert!(lcccs.is_sat(&ck, &witness).is_err());
  }

  fn test_lcccs_v_j_with<G: Group>() {
    let mut rng = OsRng;

    // Gen test vectors & artifacts
    let z = CCS::<Ep>::get_test_z(3);
    let (ccs, _, _) = CCS::<Ep>::gen_test_ccs(&z);
    let ck = ccs.commitment_key();

    // Get LCCCS
    let (lcccs, _) = ccs.to_lcccs(&mut rng, &ck, &z);

    let vec_L_j_x = lcccs.compute_Ls(&z);
    assert_eq!(vec_L_j_x.len(), lcccs.v.len());

    for (v_i, L_j_x) in lcccs.v.into_iter().zip(vec_L_j_x) {
      let sum_L_j_x = BooleanHypercube::new(ccs.s)
        .map(|y| L_j_x.evaluate(&y).unwrap())
        .fold(G::Scalar::ZERO, |acc, result| acc + result);
      assert_eq!(v_i, sum_L_j_x);
    }
  }

  fn test_bad_v_j_with<G: Group>() {
    let mut rng = OsRng;

    // Gen test vectors & artifacts
    let z = CCS::<Ep>::get_test_z(3);
    let (ccs, witness, instance) = CCS::<Ep>::gen_test_ccs(&z);
    let ck = ccs.commitment_key();

    // Mutate z so that the relation does not hold
    let mut bad_z = z.clone();
    bad_z[3] = G::Scalar::ZERO;

    // Get LCCCS
    let (lcccs, _) = ccs.to_lcccs(&mut rng, &ck, &z);

    // Bad compute L_j(x) with the bad z
    let vec_L_j_x = lcccs.compute_Ls(&bad_z);
    assert_eq!(vec_L_j_x.len(), lcccs.v.len());
    // Assert LCCCS is not satisfied with the bad Z
    assert!(lcccs.is_sat(&ck, &CCSWitness { w: bad_z }).is_err());

    // Make sure that the LCCCS is not satisfied given these L_j(x)
    // i.e. summing L_j(x) over the hypercube should not give v_j for all j
    let mut satisfied = true;
    for (v_i, L_j_x) in lcccs.v.into_iter().zip(vec_L_j_x) {
      let sum_L_j_x = BooleanHypercube::new(ccs.s)
        .map(|y| L_j_x.evaluate(&y).unwrap())
        .fold(G::Scalar::ZERO, |acc, result| acc + result);
      if v_i != sum_L_j_x {
        satisfied = false;
      }
    }

    assert!(!satisfied);
  }

  #[test]
  fn satisfied_ccs_is_satisfied_lcccs() {
    satisfied_ccs_is_satisfied_lcccs_with::<Ep>();
  }

  #[test]
  /// Test linearized CCCS v_j against the L_j(x)
  fn test_lcccs_v_j() {
    test_lcccs_v_j_with::<Ep>();
  }

  /// Given a bad z, check that the v_j should not match with the L_j(x)
  #[test]
  fn test_bad_v_j() {
    test_bad_v_j_with::<Ep>();
  }
}
