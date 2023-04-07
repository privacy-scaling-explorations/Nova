#![allow(non_snake_case)]

use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use criterion::*;
use ff::PrimeField;
use nova_snark::{
  traits::{
    circuit::{StepCircuit, TrivialTestCircuit},
    Group,
  },
  parallel_prover::{
    PublicParams,
    NovaTreeNode
  }

};
use std::time::Duration;

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
type C1 = NonTrivialTestCircuit<<G1 as Group>::Scalar>;
type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;

criterion_group! {
name = parallel_merge_node;
config = Criterion::default().warm_up_time(Duration::from_millis(3000));
targets = bench_parallel_merge_node
}

criterion_main!(parallel_merge_node);

fn bench_parallel_merge_node(c: &mut Criterion) {
  let num_cons_verifier_circuit_primary = 9819;
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in
    [9819, 16384, 32768, 65536, 131072, 262144, 524288, 1048576].iter()
  {
    // number of constraints in the step circuit
    let num_cons = num_cons_in_augmented_circuit - num_cons_verifier_circuit_primary;

    let mut group = c.benchmark_group(format!("Par-StepCircuitSize-{num_cons}"));
    group.sample_size(10);
    
    let non_trivial = NonTrivialTestCircuit::new(num_cons);

    // Produce public parameters
    let pp = PublicParams::<G1, G2, C1, C2>::setup(
      non_trivial.clone(),
      TrivialTestCircuit::default(),
    );

    let left = NovaTreeNode::new(
        &pp,
        non_trivial.clone(),
        TrivialTestCircuit::default(),
        0,
        vec![<G1 as Group>::Scalar::from(2u64)],
        non_trivial.clone().output(&vec![<G1 as Group>::Scalar::from(2u64)][..]),
        vec![<G2 as Group>::Scalar::from(1u64)],
        vec![<G2 as Group>::Scalar::from(1u64)]
    ).unwrap();
    let in_second = non_trivial.output(&non_trivial.output(&vec![<G1 as Group>::Scalar::from(2u64)][..])[..]);
    let out_second = non_trivial.output(&in_second[..]);

    let right = NovaTreeNode::new(
        &pp,
        non_trivial.clone(),
        TrivialTestCircuit::default(),
        2,
        in_second,
        out_second,
        vec![<G2 as Group>::Scalar::from(2u64)],
        vec![<G2 as Group>::Scalar::from(2u64)]
    ).unwrap();

    group.bench_function("Prove Merge", |b| {
      b.iter(|| {
        // produce a recursive SNARK for a step of the recursion
        assert!(black_box(left.clone()).merge(
            black_box(right.clone()), 
            black_box(&pp), 
            black_box(&non_trivial),
            black_box(&TrivialTestCircuit::default())
        )
        .is_ok());
      })
    });

    group.finish();
  }
}


#[derive(Clone, Debug, Default)]
struct NonTrivialTestCircuit<F: PrimeField> {
  num_cons: usize,
  _p: PhantomData<F>,
}

impl<F> NonTrivialTestCircuit<F>
where
  F: PrimeField,
{
  pub fn new(num_cons: usize) -> Self {
    Self {
      num_cons,
      _p: Default::default(),
    }
  }
}
impl<F> StepCircuit<F> for NonTrivialTestCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    // Consider a an equation: `x^2 = y`, where `x` and `y` are respectively the input and output.
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..self.num_cons {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }
    Ok(vec![y])
  }

  fn output(&self, z: &[F]) -> Vec<F> {
    let mut x = z[0];
    let mut y = x;
    for _i in 0..self.num_cons {
      y = x * x;
      x = y;
    }
    vec![y]
  }
}
