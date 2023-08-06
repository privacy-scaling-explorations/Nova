#![allow(unused)]
#![allow(missing_docs)]
use crate::traits::Group;
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::{AllocatedNum, Num},
    Assignment,
  },
  ConstraintSystem, LinearCombination, SynthesisError,
};
use ff::{Field, PrimeField, PrimeFieldBits};

pub fn verify<Scalar, CS>(
  mut cs: CS,
  claim: AllocatedNum<Scalar>,
  num_rounds: usize,
  degree_bound: usize,
  polys: Vec<Vec<AllocatedNum<Scalar>>>,
  challenges: Vec<AllocatedNum<Scalar>>, // this is while Transcript circuit is not ready
) -> Result<(AllocatedNum<Scalar>, Vec<AllocatedNum<Scalar>>), SynthesisError>
where
  Scalar: PrimeField + PrimeFieldBits,
  CS: ConstraintSystem<Scalar>,
{
  let mut e = claim;
  for (i, poly) in polys.iter().enumerate() {
    // verify poly degree // TODO move check to constraints
    if poly.len() - 1 != degree_bound {
      panic!(
        "poly degree {:?} != degree_bound {:?}",
        poly.len() - 1,
        degree_bound
      );
    }

    // eval at zero
    let eval_0 = poly[0].clone();
    // eval at one
    let eval_1 = eval_at_one(cs.namespace(|| format!("sc_eval_at_one {}", i)), poly)?;

    cs.enforce(
      || format!("(s(0)+s(1)) * 1 = s(r) {}", i),
      |lc| lc + eval_0.get_variable() + eval_1.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + e.get_variable(),
    );

    let r_i = challenges[i].clone(); // this is while Transcript circuit is not ready

    e = uni_evaluate(
      cs.namespace(|| format!("sc_unieval {}", i)),
      poly.clone(),
      r_i,
    )?;
  }

  // note: challenges will be replaced by r vector once the Transcript circuit is ready
  Ok((e, challenges))
}

pub fn eval_at_one<Scalar, CS>(
  mut cs: CS,
  poly: &Vec<AllocatedNum<Scalar>>,
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
  Scalar: PrimeField + PrimeFieldBits,
  CS: ConstraintSystem<Scalar>,
{
  let mut eval: Num<Scalar> = Num::<Scalar>::from(poly[0].clone());
  let mut allocated_eval_aux: AllocatedNum<Scalar> =
    AllocatedNum::<Scalar>::alloc(&mut cs, || Ok(eval.get_value().unwrap()))?;
  for (i, coeff) in poly.iter().enumerate().skip(1) {
    // logic: eval_aux = eval + coef
    let eval_aux = eval.clone().add(&Num::<Scalar>::from(coeff.clone()));
    // r1cs constr: (eval + coeff) * 1 = eval_aux
    let allocated_eval: AllocatedNum<Scalar> =
      AllocatedNum::<Scalar>::alloc(&mut cs, || Ok(eval.get_value().unwrap()))?;
    allocated_eval_aux =
      AllocatedNum::<Scalar>::alloc(&mut cs, || Ok(eval_aux.get_value().unwrap()))?;
    cs.enforce(
      || format!("eval + coeff = eval_aux{}", i),
      |lc| lc + allocated_eval.get_variable() + coeff.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + allocated_eval_aux.get_variable(),
    );
    eval = eval_aux;
  }
  Ok(allocated_eval_aux)
}

// method compatible with src/spartan/sumcheck.rs > UniPoly::evaluate()
pub fn uni_evaluate<Scalar, CS>(
  mut cs: CS,
  coeffs: Vec<AllocatedNum<Scalar>>,
  r: AllocatedNum<Scalar>,
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
  Scalar: PrimeField + PrimeFieldBits,
  CS: ConstraintSystem<Scalar>,
{
  let mut eval = Num::<Scalar>::from(coeffs[0].clone());
  let mut power = r.clone();
  for (i, coeff) in coeffs.iter().enumerate().skip(1) {
    // logic: eval_aux = eval + power * coeff
    let eval_aux = eval.clone().add(&Num::<Scalar>::from(
      power.mul(cs.namespace(|| format!("mul {}", i)), &coeff)?,
    ));
    // r1cs constr: power * coeff = eval - eval_aux
    let allocated_eval: AllocatedNum<Scalar> =
      AllocatedNum::<Scalar>::alloc(&mut cs, || Ok(eval.get_value().unwrap()))?;
    let allocated_eval_aux: AllocatedNum<Scalar> =
      AllocatedNum::<Scalar>::alloc(&mut cs, || Ok(eval_aux.get_value().unwrap()))?;
    cs.enforce(
      || format!("power * coeff = eval_aux - eval{}", i),
      |lc| lc + power.get_variable(),
      |lc| lc + coeff.get_variable(),
      |lc| lc + allocated_eval_aux.get_variable() - allocated_eval.get_variable(),
    );
    eval = eval_aux;

    // logic: power_aux = power * r
    let power_aux = power.mul(cs.namespace(|| format!("mul2 {}", i)), &r)?;
    // r1cs constr: power_aux = power * r
    cs.enforce(
      || format!("power_aux = power * r {}", i),
      |lc| lc + power.get_variable(),
      |lc| lc + r.get_variable(),
      |lc| lc + power_aux.get_variable(),
    );
    power = power_aux;
  }
  let allocated_eval: AllocatedNum<Scalar> =
    AllocatedNum::<Scalar>::alloc(&mut cs, || Ok(eval.get_value().unwrap()))?;

  Ok(allocated_eval)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bellperson::{
    r1cs::{NovaShape, NovaWitness},
    {shape_cs::ShapeCS, solver::SatisfyingAssignment},
  };
  use crate::spartan::sumcheck::UniPoly;
  use pasta_curves::{arithmetic::CurveAffine, pallas, pallas::Scalar, vesta};

  use crate::traits::{Group, TranscriptEngineTrait, TranscriptReprTrait};
  type PastaG1 = pasta_curves::pallas::Point;
  use crate::spartan::polynomial::MultilinearPolynomial;
  use crate::spartan::sumcheck::SumcheckProof;

  fn synthesize_uni_evaluate<Scalar, CS>(
    mut cs: CS,
    coeffs: Vec<Scalar>,
    r: Scalar,
  ) -> AllocatedNum<Scalar>
  where
    Scalar: PrimeField + PrimeFieldBits,
    CS: ConstraintSystem<Scalar>,
  {
    // prepare inputs
    let r_var = AllocatedNum::<Scalar>::alloc(cs.namespace(|| "alloc r"), || Ok(r)).unwrap();
    let mut coeffs_var: Vec<AllocatedNum<Scalar>> = Vec::new();
    for (i, coeff) in coeffs.iter().enumerate() {
      coeffs_var.push(
        AllocatedNum::<Scalar>::alloc(cs.namespace(|| format!("alloc coeffs {}", i)), || {
          Ok(*coeff)
        })
        .unwrap(),
      );
    }

    // define inputs
    // let _ = r_var.inputize(cs.namespace(|| "Input r"));
    // for (i, coeff) in coeffs_var.iter().enumerate() {
    //   let _ = coeff.inputize(cs.namespace(|| format!("Input coeffs {}", i)));
    // }

    // evaluate in-circuit (synthesize)
    let res = uni_evaluate(cs.namespace(|| "uni_evaluate"), coeffs_var, r_var).unwrap();
    // let _ = res.inputize(cs.namespace(|| "Output res"));

    res
  }

  #[test]
  fn test_uni_evaluate() {
    let evals = vec![
      Scalar::from(1_u64),
      Scalar::from(2_u64),
      Scalar::from(3_u64),
    ];
    let p = UniPoly::<pallas::Point>::from_evals(&evals);
    let r = Scalar::from(5_u64);

    // let's test it against the CS
    let mut cs: ShapeCS<pallas::Point> = ShapeCS::new();
    let _ = synthesize_uni_evaluate(&mut cs, p.get_coeffs(), r);
    println!("Number of constraints: {}", cs.num_constraints());

    let (shape, ck) = cs.r1cs_shape();

    let mut cs: SatisfyingAssignment<pallas::Point> = SatisfyingAssignment::new();
    let res = synthesize_uni_evaluate(&mut cs, p.get_coeffs(), r);
    assert_eq!(res.get_value().unwrap(), p.evaluate(&r));

    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  fn synthesize_sumcheck_verify<Scalar, CS>(
    mut cs: CS,
    claim: &Scalar,
    num_rounds: usize,
    degree_bound: usize,
    polys: &Vec<Vec<Scalar>>,
    challenges: &Vec<Scalar>,
  ) -> (AllocatedNum<Scalar>, Vec<AllocatedNum<Scalar>>)
  where
    Scalar: PrimeField + PrimeFieldBits,
    CS: ConstraintSystem<Scalar>,
  {
    // prepare inputs
    let claim_var =
      AllocatedNum::<Scalar>::alloc(cs.namespace(|| "alloc claim"), || Ok(*claim)).unwrap();

    let mut polys_var: Vec<Vec<AllocatedNum<Scalar>>> = Vec::new();
    for (i, poly) in polys.iter().enumerate() {
      let mut poly_var: Vec<AllocatedNum<Scalar>> = Vec::new();
      for (j, coef) in poly.iter().enumerate() {
        poly_var.push(
          AllocatedNum::<Scalar>::alloc(cs.namespace(|| format!("alloc poly {},{}", i, j)), || {
            Ok(poly[j])
          })
          .unwrap(),
        );
      }
      polys_var.push(poly_var);
    }
    let mut challenges_var: Vec<AllocatedNum<Scalar>> = Vec::new();
    for (i, challenge) in challenges.iter().enumerate() {
      challenges_var.push(
        AllocatedNum::<Scalar>::alloc(cs.namespace(|| format!("alloc challenge {}", i)), || {
          Ok(*challenge)
        })
        .unwrap(),
      );
    }

    // verify in-circuit (synthesize)
    let res = verify(
      cs.namespace(|| "verify"),
      claim_var,
      num_rounds,
      degree_bound,
      polys_var,
      challenges_var,
    )
    .unwrap();
    // let _ = res.inputize(cs.namespace(|| "Output res"));

    res
  }

  fn test_sumcheck_verify(g: MultilinearPolynomial<Scalar>) {
    let mut transcript_p = <pasta_curves::Ep as Group>::TE::new(b"sumcheck");
    let (sc_proof, claim) = SumcheckProof::<PastaG1>::prove(&g, &mut transcript_p).unwrap();

    // generate the verifier challenges for the circuit (in the future this will be done
    // in-circuit). Also prepare the uncompressed polys.
    let mut transcript_v = <pasta_curves::Ep as Group>::TE::new(b"sumcheck");
    let mut r: Vec<Scalar> = Vec::new();
    let mut polys: Vec<Vec<Scalar>> = Vec::new();
    let mut e = claim;
    for i in 0..sc_proof.compressed_polys.len() {
      let poly = sc_proof.compressed_polys[i].decompress(&e);
      polys.push(poly.get_coeffs());
      transcript_v.absorb(b"p", &poly);
      let r_i = transcript_v.squeeze(b"c").unwrap();
      r.push(r_i);
      e = poly.evaluate(&r_i);
    }

    let num_rounds = g.get_num_vars();
    let degree_bound = 2;

    // verify sumcheck proof (no circuit)
    let mut transcript_v = <pasta_curves::Ep as Group>::TE::new(b"sumcheck");
    let (e, r) = sc_proof
      .verify(claim, num_rounds, degree_bound, &mut transcript_v)
      .unwrap();
    assert_eq!(e, g.evaluate(&r));

    // let's test it against the CS
    let mut cs: ShapeCS<pallas::Point> = ShapeCS::new();
    let _ = synthesize_sumcheck_verify(&mut cs, &claim, num_rounds, degree_bound, &polys, &r);
    println!("Number of constraints: {}", cs.num_constraints());

    let (shape, ck) = cs.r1cs_shape();

    let mut cs: SatisfyingAssignment<pallas::Point> = SatisfyingAssignment::new();
    let (verify_e, verify_r) =
      synthesize_sumcheck_verify(&mut cs, &claim, num_rounds, degree_bound, &polys, &r);
    assert_eq!(verify_e.get_value().unwrap(), g.evaluate(&r));

    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  #[test]
  fn test_sumcheck_verify_hardcoded_values() {
    // g(X_0, X_1, X_2) = 2 X_0^3 + X_0 X_2 + X_1 X_2
    let Z = vec![
      Scalar::zero(),
      Scalar::zero(),
      Scalar::zero(),
      Scalar::from(1),
      Scalar::from(2),
      Scalar::from(3),
      Scalar::from(2),
      Scalar::from(4),
    ];
    let g = MultilinearPolynomial::<Scalar>::new(Z.clone());
    test_sumcheck_verify(g);
  }

  #[test]
  fn test_sumcheck_verify_random_values() {
    let rng = &mut rand::rngs::OsRng;
    let Z: Vec<Scalar> = vec![Scalar::random(rng); 64];
    let g = MultilinearPolynomial::<Scalar>::new(Z.clone());
    test_sumcheck_verify(g);
  }
}
