use rand::Rng;
use serde::{Deserialize, Serialize};

/// Xavier (Glorot) weight initialization.
///
/// Returns a random value from Uniform(-1, 1) divided by √n where
/// `num_connections` is the number of inputs to the neuron.
///
/// Why this matters: if weights are too large, the sigmoid saturates
/// and gradients vanish. If they are too small, signals diminish
/// through the layers. Xavier init keeps the variance of activations
/// roughly constant across layers — a balance that helps gradient flow.
///
/// Takes a generic RNG so callers can control determinism (pass a
/// seeded `StdRng` in tests, thread-local `rand::rng()` in examples).
pub fn xavier_init(num_connections: usize, rng: &mut impl Rng) -> f64 {
    rng.random_range(-1.0..1.0) / (num_connections as f64).sqrt()
}

/// A single artificial neuron — the fundamental computational unit.
///
/// A neuron stores one learnable weight per incoming connection plus
/// one learnable bias. It does **not** store the last input or output —
/// the [`Layer`](crate::neural_layer::Layer) owns the activation function
/// and threads data explicitly during forward and backward passes.
///
/// This makes the neuron a pure container of learned parameters.
/// You can inspect `weights` and `bias` after training without
/// worrying about stale cached state.
#[derive(Serialize, Deserialize)]
pub struct Neuron {
    /// One learnable weight per incoming connection.
    ///
    /// Initialized with [`xavier_init`] and updated by gradient descent
    /// during training. A weight's magnitude reflects how much its
    /// corresponding input influences this neuron's output.
    pub weights: Vec<f64>,

    /// A learnable offset (bias) added to the weighted sum.
    ///
    /// The bias shifts the activation function left or right, acting
    /// as the neuron's "threshold" — it determines how large the
    /// weighted sum must be before the neuron "fires".
    pub bias: f64,
}

impl Neuron {
    /// Creates a new neuron with `num_connections` inputs.
    ///
    /// Weights and bias are initialized with [`xavier_init`] using the
    /// provided RNG. Pass a seeded RNG in tests for deterministic runs.
    pub fn new(num_connections: usize, rng: &mut impl Rng) -> Self {
        Self {
            weights: (0..num_connections)
                .map(|_| xavier_init(num_connections, rng))
                .collect(),
            bias: xavier_init(num_connections, rng),
        }
    }

    /// Compute the weighted sum Σ(input_i × weight_i) + bias.
    ///
    /// This is the **linear** part of the neuron's computation. The
    /// non-linear activation function is applied by the layer, which
    /// separates concerns: the neuron handles the linear transform,
    /// the layer handles the non-linear squashing.
    pub fn weighted_sum(&self, inputs: &[f64]) -> f64 {
        inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>()
            + self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn new_neuron_has_correct_number_of_weights() {
        let n = Neuron::new(5, &mut seeded_rng());
        assert_eq!(n.weights.len(), 5);
    }

    #[test]
    fn new_neuron_has_finite_bias() {
        let n = Neuron::new(3, &mut seeded_rng());
        assert!(n.bias.is_finite());
    }

    #[test]
    fn xavier_init_produces_finite_values() {
        let mut rng = seeded_rng();
        for n in [1usize, 5, 100] {
            let val = xavier_init(n, &mut rng);
            assert!(val.is_finite(), "xavier_init({n}) returned non-finite value");
        }
    }

    #[test]
    fn weighted_sum_is_finite() {
        let n = Neuron::new(3, &mut seeded_rng());
        let s = n.weighted_sum(&[0.5, 0.2, 0.1]);
        assert!(s.is_finite());
    }

    #[test]
    fn weighted_sum_zero_weights_equals_bias() {
        let n = Neuron {
            weights: vec![0.0, 0.0],
            bias: 1.5,
        };
        let s = n.weighted_sum(&[10.0, -5.0]);
        assert!((s - 1.5).abs() < 1e-15);
    }

    #[test]
    fn weighted_sum_known_value() {
        let n = Neuron {
            weights: vec![2.0, 3.0],
            bias: 1.0,
        };
        // 2×0.5 + 3×0.2 + 1.0 = 1.0 + 0.6 + 1.0 = 2.6
        let s = n.weighted_sum(&[0.5, 0.2]);
        assert!((s - 2.6).abs() < 1e-15);
    }

    #[test]
    fn xavier_init_deterministic_with_seed() {
        let mut rng1 = seeded_rng();
        let mut rng2 = seeded_rng();
        let a = xavier_init(10, &mut rng1);
        let b = xavier_init(10, &mut rng2);
        assert_eq!(a, b, "same seed should produce same value");
    }
}
