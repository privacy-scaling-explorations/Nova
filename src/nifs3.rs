/// Module that allows 3-to-1 NIFS

use crate::{
  constants::{NUM_CHALLENGE_BITS, NUM_FE_FOR_RO},
  errors::NovaError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  scalar_as_base,
  traits::{commitment::CommitmentTrait, AbsorbInROTrait, Group, ROTrait},
  Commitment, CommitmentKey, CompressedCommitment,
};
use core::marker::PhantomData;
use serde::{Deserialize, Serialize};


/// A SNARK that holds the proof of a step of an incremental computation
/// Because of 3-to-1 folding we need three commitments to crossterms
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS3<G: Group> {
  pub(crate) comm_T_1: CompressedCommitment<G>,
  pub(crate) comm_T_2: CompressedCommitment<G>,
  pub(crate) comm_T_3: CompressedCommitment<G>,
  _p: PhantomData<G>,
}

type ROConstants<G> =
  <<G as Group>::RO as ROTrait<<G as Group>::Base, <G as Group>::Scalar>>::Constants;

impl<G: Group> NIFS3<G> {
  #![allow(dead_code)]
  /// Take as input a Relaxed R1CS instance-witness tuple `(U1, W1)` and
  /// two R1CS instance-witness tuples `(U2, W2)` and `(U3, W3)` and fold them
  pub fn prove(
    ck: &CommitmentKey<G>,
    ro_consts: &ROConstants<G>,
    S: &R1CSShape<G>,
    U1: &RelaxedR1CSInstance<G>,
    W1: &RelaxedR1CSWitness<G>,
    U2: &R1CSInstance<G>,
    W2: &R1CSWitness<G>,
    U3: &R1CSInstance<G>,
    W3: &R1CSWitness<G>,
  ) -> Result<(NIFS3<G>, (RelaxedR1CSInstance<G>, RelaxedR1CSWitness<G>)), NovaError> {
    // initialize a new RO
    let mut ro = G::RO::new(ro_consts.clone(), NUM_FE_FOR_RO + 11); // XXX introduce a new constant

    // append S to the transcript
    S.absorb_in_ro(&mut ro);

    // append instances to transcript
    U1.absorb_in_ro(&mut ro);
    U2.absorb_in_ro(&mut ro);
    U3.absorb_in_ro(&mut ro);

    // compute commitments to the cross-terms
    let (T_1, comm_T_1, T_2, comm_T_2, T_3, comm_T_3) = S.commit_T_three(ck, U1, W1, U2, W2, U3, W3)?;

    // append commitments to the transcript and obtain a challenge
    comm_T_1.absorb_in_ro(&mut ro);
    comm_T_2.absorb_in_ro(&mut ro);
    comm_T_3.absorb_in_ro(&mut ro);

    // compute a challenge from the RO
    let r_1 = ro.squeeze(NUM_CHALLENGE_BITS);
    let r_2 = ro.squeeze(NUM_CHALLENGE_BITS);

    // fold the instance using the randomizers and crossterm commitments
    let U = U1.fold_three([&U2, &U3], [&comm_T_1, &comm_T_2, &comm_T_3], [&r_1, &r_2])?;

    // fold the witness using randomizers and crossterms
    let W = W1.fold_three([&W2, &W3], [&T_1, &T_2, &T_3], [&r_1, &r_2])?;

    // return the folded instance and witness
    Ok((
      Self {
        comm_T_1: comm_T_1.compress(),
        comm_T_2: comm_T_2.compress(),
        comm_T_3: comm_T_3.compress(),
        _p: Default::default(),
      },
      (U, W),
    ))
  }

  pub fn verify(
    &self,
    ro_consts: &ROConstants<G>,
    S_digest: &G::Scalar,
    U1: &RelaxedR1CSInstance<G>,
    U2: &R1CSInstance<G>,
    U3: &R1CSInstance<G>,
  ) -> Result<RelaxedR1CSInstance<G>, NovaError> {
    // initialize a new RO
    let mut ro = G::RO::new(ro_consts.clone(), NUM_FE_FOR_RO + 11); // XXX update constant

    // append the digest of S to the transcript
    ro.absorb(scalar_as_base::<G>(*S_digest));

    // append instances to transcript
    U1.absorb_in_ro(&mut ro);
    U2.absorb_in_ro(&mut ro);
    U3.absorb_in_ro(&mut ro);

    // append `comm_T` to the transcript and obtain a challenge
    let comm_T_1 = Commitment::<G>::decompress(&self.comm_T_1)?;
    let comm_T_2 = Commitment::<G>::decompress(&self.comm_T_2)?;
    let comm_T_3 = Commitment::<G>::decompress(&self.comm_T_3)?;
    comm_T_1.absorb_in_ro(&mut ro);
    comm_T_2.absorb_in_ro(&mut ro);
    comm_T_3.absorb_in_ro(&mut ro);

    // compute a challenge from the RO
    let r_1 = ro.squeeze(NUM_CHALLENGE_BITS);
    let r_2 = ro.squeeze(NUM_CHALLENGE_BITS);

    // fold the instance using the randomizers and crossterm commitments
    let U = U1.fold_three([&U2, &U3], [&comm_T_1, &comm_T_2, &comm_T_3], [&r_1, &r_2])?;

    // return the folded instance
    Ok(U)
  }
}
