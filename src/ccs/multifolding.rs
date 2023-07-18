use super::cccs::{self, CCCSInstance};
use super::lcccs::LCCCS;
use super::util::{compute_sum_Mz, VirtualPolynomial};
use super::{CCSWitness, CCS};
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
use rand_core::RngCore;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::ops::{Add, Mul};
use std::sync::Arc;

// XXX: THe idea is to have Multifolding as IVC instance in the future, holding the main CCS
// instances. Then the rest of CCS, CCCS, LCCCS hold references to it.
// Is our single source of data.
#[derive(Debug)]
pub struct Multifolding<G: Group> {
  ccs: CCS<G>,
  ccs_mle: Vec<MultilinearPolynomial<G::Scalar>>,
  ck: CommitmentKey<G>,
  lcccs: LCCCS<G>,
}

impl<G: Group> Multifolding<G> {
  /// Generates a new Multifolding instance based on the given CCS.
  pub fn new(
    ccs: CCS<G>,
    ccs_mle: Vec<MultilinearPolynomial<G::Scalar>>,
    lcccs: LCCCS<G>,
    ck: CommitmentKey<G>,
  ) -> Self {
    Self {
      ccs,
      ccs_mle,
      ck,
      lcccs,
    }
  }

  pub fn init<R: RngCore>(
    mut rng: &mut R,
    ccs: CCS<G>,
    ccs_mle: Vec<MultilinearPolynomial<G::Scalar>>,
    z: Vec<G::Scalar>,
  ) -> Self {
    let w: Vec<G::Scalar> = z[(1 + ccs.l)..].to_vec();
    let ck = ccs.commitment_key();
    let r_w = G::Scalar::random(rng);
    let w_comm = <G as Group>::CE::commit(&ck, &w);

    let r_x: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(rng)).collect();
    let v = ccs.compute_v_j(&z, &r_x, &ccs_mle);

    let lcccs = LCCCS::new(&ccs, &ccs_mle, &ck, z, rng);

    Self {
      ccs,
      ccs_mle,
      lcccs,
      ck,
    }
  }

  /// Compute sigma_i and theta_i from step 4
  pub fn compute_sigmas_and_thetas(
    &self,
    z2: &[G::Scalar],
    r_x_prime: &[G::Scalar],
  ) -> (Vec<G::Scalar>, Vec<G::Scalar>) {
    (
      // sigmas
      compute_all_sum_Mz_evals::<G>(
        &self.ccs_mle,
        self.lcccs.z.as_slice(),
        r_x_prime,
        self.ccs.s_prime,
      ),
      // thetas
      compute_all_sum_Mz_evals::<G>(&self.ccs_mle, z2, r_x_prime, self.ccs.s_prime),
    )
  }

  /// Compute the right-hand-side of step 5 of the multifolding scheme
  pub fn compute_c_from_sigmas_and_thetas(
    &self,
    sigmas: &[G::Scalar],
    thetas: &[G::Scalar],
    gamma: G::Scalar,
    beta: &[G::Scalar],
    r_x_prime: &[G::Scalar],
  ) -> G::Scalar {
    let mut c = G::Scalar::ZERO;

    let e1 = eq_eval(&self.lcccs.r_x, r_x_prime);
    let e2 = eq_eval(beta, r_x_prime);

    // (sum gamma^j * e1 * sigma_j)
    for (j, sigma_j) in sigmas.iter().enumerate() {
      let gamma_j = gamma.pow([j as u64]);
      c += gamma_j * e1 * sigma_j;
    }

    // + gamma^{t+1} * e2 * sum c_i * prod theta_j
    let mut lhs = G::Scalar::ZERO;
    for i in 0..self.ccs.q {
      let mut prod = G::Scalar::ONE;
      for j in self.ccs.S[i].clone() {
        prod *= thetas[j];
      }
      lhs += self.ccs.c[i] * prod;
    }
    let gamma_t1 = gamma.pow([(self.ccs.t + 1) as u64]);
    c += gamma_t1 * e2 * lhs;
    c
  }

  /// Compute g(x) polynomial for the given inputs.
  pub fn compute_g(
    &self,
    cccs_instance: &CCCSInstance<G>,
    gamma: G::Scalar,
    beta: &[G::Scalar],
  ) -> VirtualPolynomial<G::Scalar> {
    let mut vec_L = self.lcccs.compute_Ls(&self.ccs, &self.ccs_mle, &self.ck);

    let mut Q = cccs_instance
      .compute_Q(beta)
      .expect("Q comp should not fail");

    let mut g = vec_L[0].clone();
    // XXX: This can probably be done with Iter::reduce
    for (j, L_j) in vec_L.iter_mut().enumerate().skip(1) {
      let gamma_j = gamma.pow([j as u64]);
      L_j.scalar_mul(&gamma_j);
      g = g.add(L_j);
    }

    let gamma_t1 = gamma.pow([(self.ccs.t + 1) as u64]);
    Q.scalar_mul(&gamma_t1);
    g = g.add(&Q);
    g
  }

  // XXX: Add some docs
  pub fn fold(
    &mut self,
    cccs2: (&Commitment<G>, &[G::Scalar]),
    sigmas: &[G::Scalar],
    thetas: &[G::Scalar],
    r_x_prime: Vec<G::Scalar>,
    rho: G::Scalar,
  ) {
    let w_folded_comm = self.lcccs.w_comm + cccs2.0.mul(rho);
    let folded_u = self.lcccs.u + rho;
    let folded_v: Vec<G::Scalar> = sigmas
      .iter()
      .zip(
        thetas
          .iter()
          .map(|x_i| *x_i * rho)
          .collect::<Vec<G::Scalar>>(),
      )
      .map(|(a_i, b_i)| *a_i + b_i)
      .collect();

    // XXX: Update NIMFS LCCCS instance. (This should be done via a fn);
    self.lcccs.w_comm = w_folded_comm;
    self.lcccs.u = folded_u;
    self.lcccs.v = folded_v;
    self.fold_z(cccs2.1, rho);
  }

  // XXX: Add docs
  fn fold_z(&mut self, z2: &[G::Scalar], rho: G::Scalar) {
    self.lcccs.z[1..]
      .iter_mut()
      .zip(
        z2[1..]
          .iter()
          .map(|x_i| *x_i * rho)
          .collect::<Vec<G::Scalar>>(),
      )
      .for_each(|(a_i, b_i)| *a_i += b_i);

    // XXX: There's no handling of r_w atm. So we will ingore until all folding is implemented,
    // let r_w = w1.r_w + rho * w2.r_w;
  }
}

/// Evaluate eq polynomial.
pub fn eq_eval<F: PrimeField>(x: &[F], y: &[F]) -> F {
  assert_eq!(x.len(), y.len());

  let mut res = F::ONE;
  for (&xi, &yi) in x.iter().zip(y.iter()) {
    let xi_yi = xi * yi;
    res *= xi_yi + xi_yi - xi - yi + F::ONE;
  }
  res
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ccs::{test, util::virtual_poly::build_eq_x_r};
  use pasta_curves::{Ep, Fq};
  use rand_core::OsRng;
  // NIMFS: Non Interactive Multifolding Scheme
  type NIMFS<G> = Multifolding<G>;

  fn test_compute_g_with<G: Group>() {
    let z1 = CCS::<G>::get_test_z(3);
    let z2 = CCS::<G>::get_test_z(4);

    let (_, ccs_witness_1, ccs_instance_1) = CCS::<G>::gen_test_ccs(&z2);
    let (ccs, ccs_witness_2, ccs_instance_2) = CCS::<G>::gen_test_ccs(&z1);
    let ck = ccs.commitment_key();

    assert!(ccs.is_sat(&ck, &ccs_instance_1, &ccs_witness_1).is_ok());
    assert!(ccs.is_sat(&ck, &ccs_instance_2, &ccs_witness_2).is_ok());

    let mut rng = OsRng;
    let gamma: G::Scalar = G::Scalar::random(&mut rng);
    let beta: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut rng)).collect();

    let lcccs = LCCCS::new(&ccs, &mles, &ck, z1, &mut OsRng);
    let cccs_instance = CCCSInstance::new(&ccs, &mles, &z2, &ck);

    let mut sum_v_j_gamma = G::Scalar::ZERO;
    for j in 0..lcccs_instance.v.len() {
      let gamma_j = gamma.pow([j as u64]);
      sum_v_j_gamma += lcccs.v[j] * gamma_j;
    }

    let nimfs = NIMFS::new(ccs, mles, lcccs, ck);

    // Compute g(x) with that r_x
    let g = nimfs.compute_g(&cccs_instance, gamma, &beta);

    // evaluate g(x) over x \in {0,1}^s
    let mut g_on_bhc = G::Scalar::ZERO;
    for x in BooleanHypercube::new(ccs.s) {
      g_on_bhc += g.evaluate(&x).unwrap();
    }

    // evaluate sum_{j \in [t]} (gamma^j * Lj(x)) over x \in {0,1}^s
    let mut sum_Lj_on_bhc = G::Scalar::ZERO;
    let vec_L = lcccs.compute_Ls(&ccs, &mles, &ck);
    for x in BooleanHypercube::new(ccs.s) {
      for (j, coeff) in vec_L.iter().enumerate() {
        let gamma_j = gamma.pow([j as u64]);
        sum_Lj_on_bhc += coeff.evaluate(&x).unwrap() * gamma_j;
      }
    }

    // Q(x) over bhc is assumed to be zero, as checked in the test 'test_compute_Q'
    assert_ne!(g_on_bhc, G::Scalar::ZERO);

    // evaluating g(x) over the boolean hypercube should give the same result as evaluating the
    // sum of gamma^j * Lj(x) over the boolean hypercube
    assert_eq!(g_on_bhc, sum_Lj_on_bhc);

    // evaluating g(x) over the boolean hypercube should give the same result as evaluating the
    // sum of gamma^j * v_j over j \in [t]
    assert_eq!(g_on_bhc, sum_v_j_gamma);
  }

  fn test_compute_sigmas_and_thetas_with<G: Group>() {
    let z1 = CCS::<G>::get_test_z(3);
    let z2 = CCS::<G>::get_test_z(4);

    let (_, ccs_witness_1, ccs_instance_1) = CCS::<G>::gen_test_ccs(&z2);
    let (ccs, ccs_witness_2, ccs_instance_2) = CCS::<G>::gen_test_ccs(&z1);
    let ck: CommitmentKey<G> = ccs.commitment_key();

    assert!(ccs.is_sat(&ck, &ccs_instance_1, &ccs_witness_1).is_ok());
    assert!(ccs.is_sat(&ck, &ccs_instance_2, &ccs_witness_2).is_ok());

    let mut rng = OsRng;
    let gamma: G::Scalar = G::Scalar::random(&mut rng);
    let beta: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut rng)).collect();
    let r_x_prime: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut rng)).collect();

    let lcccs = LCCCS::new(&ccs, &mles, &ck, z1, &mut OsRng);
    let cccs_instance = CCCSInstance::new(&ccs, &mles, &z2, &ck);

    // Generate a new multifolding instance
    let nimfs = NIMFS::new(ccs, mles, lcccs, ck);

    // XXX: This needs to be properly thought?
    let (sigmas, thetas) = nimfs.compute_sigmas_and_thetas(&z2, &r_x_prime);

    let g = nimfs.compute_g(&cccs_instance, gamma, &beta);
    // Assert `g` is correctly computed here.
    {
      // evaluate g(x) over x \in {0,1}^s
      let mut g_on_bhc = G::Scalar::ZERO;
      for x in BooleanHypercube::new(ccs.s) {
        g_on_bhc += g.evaluate(&x).unwrap();
      }
      // evaluate sum_{j \in [t]} (gamma^j * Lj(x)) over x \in {0,1}^s
      let mut sum_Lj_on_bhc = G::Scalar::ZERO;
      let vec_L = lcccs.compute_Ls(&ccs, &mles, &ck);
      for x in BooleanHypercube::new(ccs.s) {
        for (j, coeff) in vec_L.iter().enumerate() {
          let gamma_j = gamma.pow([j as u64]);
          sum_Lj_on_bhc += coeff.evaluate(&x).unwrap() * gamma_j;
        }
      }

      // evaluating g(x) over the boolean hypercube should give the same result as evaluating the
      // sum of gamma^j * Lj(x) over the boolean hypercube
      assert_eq!(g_on_bhc, sum_Lj_on_bhc);
    };

    // XXX: We need a better way to do this. Sum_Mz has also the same issue.
    // reverse the `r` given to evaluate to match Spartan/Nova endianness.
    let mut revsersed = r_x_prime.clone();
    revsersed.reverse();

    // we expect g(r_x_prime) to be equal to:
    // c = (sum gamma^j * e1 * sigma_j) + gamma^{t+1} * e2 * sum c_i * prod theta_j
    // from `compute_c_from_sigmas_and_thetas`
    let expected_c = g.evaluate(&revsersed).unwrap();

    let c = nimfs.compute_c_from_sigmas_and_thetas(&sigmas, &thetas, gamma, &beta, &r_x_prime);
    assert_eq!(c, expected_c);
  }

  #[test]
  fn test_compute_g() {
    test_compute_g_with::<Ep>();
  }

  fn test_lccs_fold_with<G: Group>() {
    let z1 = CCS::<G>::get_test_z(3);
    let z2 = CCS::<G>::get_test_z(4);

    // ccs stays the same regardless of z1 or z2
    let (ccs, ccs_witness_1, ccs_instance_1) = CCS::<G>::gen_test_ccs(&z1);
    let (_, ccs_witness_2, ccs_instance_2) = CCS::<G>::gen_test_ccs(&z2);
    let ck: CommitmentKey<G> = ccs.commitment_key();

    assert!(ccs.is_sat(&ck, &ccs_instance_1, &ccs_witness_1).is_ok());
    assert!(ccs.is_sat(&ck, &ccs_instance_2, &ccs_witness_2).is_ok());

    let mut rng = OsRng;
    let r_x_prime: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut rng)).collect();

    let cccs = CCCSInstance::new(&ccs, &mles, &z2, &ck);
    assert!(cccs.is_sat().is_ok());

    // Generate a new multifolding instance
    let mut nimfs = Multifolding::init(&mut rng, ccs, mles, z1);
    assert!(nimfs.lcccs.is_sat(&ck, &lcccs_witness).is_ok());
    let (sigmas, thetas) = nimfs.compute_sigmas_and_thetas(&z1, &z2, &r_x_prime);

    let rho = Fq::random(&mut rng);
    let folded = nimfs.fold(
      &lcccs_instance,
      (&ccs_instance_2.comm_w, [z2[0]].as_slice()),
      &sigmas,
      &thetas,
      r_x_prime,
      rho,
    );

    let w_folded = NIMFS::fold_witness(&lcccs_witness, &ccs_witness_2, rho);

    // check lcccs relation
    assert!(folded.is_sat(&ck, &w_folded).is_ok());
  }

  #[test]
  fn test_compute_sigmas_and_thetas() {
    test_compute_sigmas_and_thetas_with::<Ep>()
  }

  #[test]
  fn test_lcccs_fold() {
    test_lccs_fold_with::<Ep>()
  }
}
