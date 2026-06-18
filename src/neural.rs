use crate::activation_functions::sigmoid;

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
pub fn xavier_init(num_connections: usize) -> f64 {
    let mut rng = rand::rng();
    rng.random_range(-1.0..1.0) / (num_connections as f64).sqrt()
}

/// A single artificial neuron — the fundamental computational unit.
///
/// A neuron computes a weighted sum of its inputs, adds a bias, and
/// passes the result through a non-linear activation function:
///
/// ```text
/// output = σ(weighted_sum + bias)
///       = σ(∑(input_i × weight_i) + bias)
/// ```
///
/// The fields are `pub` so you can inspect them after training —
/// try printing `neuron.weights` to see what the network learned.
#[derive(Serialize, Deserialize)]
pub struct Neuron {
    /// The most recent input vector passed to [`activate`].
    ///
    /// Stored so that [`output`] can recompute the neuron's output
    /// without requiring the input vector to be passed again. This
    /// is useful during backpropagation when weight updates reference
    /// both the stored input and the neuron's output.
    pub input: Vec<f64>,

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
    /// Weights and bias are initialized with [`xavier_init`] to promote
    /// stable gradient flow. The input vector starts empty; it will be
    /// filled on the first call to [`activate`].
    pub fn new(num_connections: usize) -> Self {
        Self {
            input: Vec::new(),
            weights: (0..num_connections)
                .map(|_| xavier_init(num_connections))
                .collect(),
            bias: xavier_init(num_connections),
        }
    }

    /// Compute and return the neuron's output for the given `inputs`.
    ///
    /// Stores a copy of `inputs` in `self.input` for later use by
    /// [`output`]. This is separate from the returned output value
    /// because backpropagation needs both the input vector (for weight
    /// updates) and the output value (for gradient computation), but
    /// they are needed at different points in the training loop.
    ///
    /// Returns σ(∑(input_i × weight_i) + bias).
    pub fn activate(&mut self, inputs: &[f64]) -> f64 {
        self.input = Vec::from(inputs);
        let weighted_sum: f64 = self
            .input
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>()
            + self.bias;
        sigmoid(weighted_sum)
    }

    /// Returns the neuron's output from the **stored** input vector.
    ///
    /// This recomputes σ(∑(input_i × weight_i) + bias) using the
    /// input that was saved by the most recent [`activate`] call.
    ///
    /// # Why recompute instead of caching?
    ///
    /// `activate` already computed this value and could have stored it.
    /// We recompute for **pedagogical clarity**: separating the forward
    /// pass (which *sets* the input) from the output query makes the
    /// data flow explicit. In a production implementation you would
    /// cache the result rather than recompute it.
    pub fn output(&self) -> f64 {
        let weighted_sum: f64 = self
            .input
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>()
            + self.bias;
        sigmoid(weighted_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_neuron_has_correct_number_of_weights() {
        let n = Neuron::new(5);
        assert_eq!(n.weights.len(), 5);
    }

    #[test]
    fn xavier_init_produces_finite_values() {
        for n in [1, 5, 100] {
            let val = xavier_init(n);
            assert!(
                val.is_finite(),
                "xavier_init({n}) returned non-finite value"
            );
        }
    }

    #[test]
    fn activate_returns_value_in_sigmoid_range() {
        let mut neuron = Neuron::new(3);
        let output = neuron.activate(&[0.5, 0.2, 0.1]);
        assert!(output.is_finite());
        assert!(
            (0.0..1.0).contains(&output),
            "sigmoid output should be in (0, 1), got {output}"
        );
    }

    #[test]
    fn activate_stores_input_vector() {
        let mut neuron = Neuron::new(3);
        let inputs = vec![1.0, 2.0, 3.0];
        neuron.activate(&inputs);
        assert_eq!(neuron.input, inputs);
    }

    #[test]
    fn output_matches_last_activate_return_value() {
        let mut neuron = Neuron::new(3);
        let a = neuron.activate(&[0.5, 0.2, 0.1]);
        let b = neuron.output();
        assert!(
            (a - b).abs() < 1e-15,
            "output() should match last activate(): {a} vs {b}"
        );
    }
}
