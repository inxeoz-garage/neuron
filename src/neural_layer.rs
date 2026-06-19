use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::neural::Neuron;

/// A single layer of neurons in a feed-forward network.
///
/// Each neuron in the layer receives the **same** input vector (the
/// output of the previous layer, or the raw data for the first hidden
/// layer). The layer's output is the vector of every neuron's
/// activation — one scalar per neuron.
///
/// The layer owns the [`Activation`] function that every neuron in this
/// layer shares. This lets you use different activations per layer
/// (e.g. ReLU for hidden layers, sigmoid for the output layer).
#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation: Activation,
}

impl Layer {
    /// Creates a layer with `num_neurons` neurons, each connected to
    /// `num_inputs` inputs from the previous layer.
    pub fn new(num_neurons: usize, num_inputs: usize, activation: Activation, rng: &mut impl Rng) -> Self {
        Self {
            neurons: (0..num_neurons)
                .map(|_| Neuron::new(num_inputs, rng))
                .collect(),
            activation,
        }
    }

    /// Compute the output of every neuron in the layer.
    ///
    /// Each neuron independently computes σ(∑(input_i × weight_i) + bias)
    /// where σ is this layer's activation function. The result is a vector
    /// of length `num_neurons` — the activations passed to the next layer.
    ///
    /// Takes `&self` (no mutation) because neurons no longer store their
    /// input vectors internally.
    pub fn feed_forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| {
                let z = neuron.weighted_sum(inputs);
                self.activation.activate(z)
            })
            .collect()
    }

    /// Update weights and biases using the backpropagated errors.
    ///
    /// For each neuron, the weight update follows gradient descent:
    ///
    /// ```text
    /// Δweight_i = learning_rate × error × σ′(output) × input_i
    /// Δbias     = learning_rate × error × σ′(output)
    /// ```
    ///
    /// `errors` must have one scalar per neuron — the partial derivative
    /// of the loss with respect to this layer's output.
    pub fn adjust_weights(&mut self, inputs: &[f64], learning_rate: f64, errors: &[f64]) {
        for (neuron, error) in self.neurons.iter_mut().zip(errors.iter()) {
            let z = neuron.weighted_sum(inputs);
            let output = self.activation.activate(z);
            // δ = error × σ′(output)  — shared factor for all weight updates
            let delta = error * self.activation.derivative(output);
            for (i, input) in inputs.iter().enumerate() {
                neuron.weights[i] += learning_rate * delta * input;
            }
            neuron.bias += learning_rate * delta;
        }
    }

    /// Propagate the error signal backward to the previous layer.
    ///
    /// For each neuron in the **previous** layer, the error is the
    /// weighted sum of this layer's errors, using the corresponding
    /// weight from each neuron:
    ///
    /// ```text
    /// error_prev[j] = Σ(this_layer.neurons[i].weights[j] × errors[i])
    /// ```
    ///
    /// This is the backward pass of backpropagation: each weight
    /// "distributes" its neuron's error back to the input that fed it.
    pub fn propagate_error(&self, errors: &[f64]) -> Vec<f64> {
        // Number of neurons in the previous layer (input dimension of this layer)
        let num_prev = self.neurons[0].weights.len();
        (0..num_prev)
            .map(|j| {
                self.neurons
                    .iter()
                    .zip(errors.iter())
                    .map(|(neuron, &err)| neuron.weights[j] * err)
                    .sum()
            })
            .collect()
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
    fn feed_forward_returns_correct_number_of_outputs() {
        let mut rng = seeded_rng();
        let layer = Layer::new(4, 3, Activation::Sigmoid, &mut rng);
        let outputs = layer.feed_forward(&[1.0, 0.5, 0.2]);
        assert_eq!(outputs.len(), 4);
    }

    #[test]
    fn feed_forward_outputs_are_in_sigmoid_range() {
        let mut rng = seeded_rng();
        let layer = Layer::new(4, 3, Activation::Sigmoid, &mut rng);
        let outputs = layer.feed_forward(&[-5.0, 0.0, 5.0]);
        for &o in &outputs {
            assert!((0.0..1.0).contains(&o), "output {o} should be in (0, 1)");
        }
    }

    #[test]
    fn adjust_weights_changes_bias() {
        let mut rng = seeded_rng();
        let mut layer = Layer::new(1, 2, Activation::Sigmoid, &mut rng);
        layer.feed_forward(&[1.0, 2.0]);
        let bias_before = layer.neurons[0].bias;
        layer.adjust_weights(&[1.0, 2.0], 0.1, &[0.5]);
        let bias_after = layer.neurons[0].bias;
        assert!(
            (bias_after - bias_before).abs() > 1e-15,
            "bias should change after adjust_weights"
        );
    }

    #[test]
    fn propagate_error_returns_correct_length() {
        let mut rng = seeded_rng();
        let layer = Layer::new(3, 2, Activation::Sigmoid, &mut rng);
        let errors = layer.propagate_error(&[0.1, 0.2, 0.3]);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn propagate_error_all_zero_weights() {
        let layer = Layer {
            neurons: vec![
                Neuron {
                    weights: vec![0.0, 0.0],
                    bias: 0.0,
                },
                Neuron {
                    weights: vec![0.0, 0.0],
                    bias: 0.0,
                },
            ],
            activation: Activation::Sigmoid,
        };
        let errors = layer.propagate_error(&[1.0, 2.0]);
        for e in errors {
            assert!((e - 0.0).abs() < 1e-15, "error should be 0 when weights are 0");
        }
    }

    #[test]
    fn propagate_error_known_value() {
        // Two neurons, each with weights [1.0, 2.0], errors [0.5, 1.5]
        // error_prev[0] = 1.0×0.5 + 1.0×1.5 = 2.0
        // error_prev[1] = 2.0×0.5 + 2.0×1.5 = 4.0
        let layer = Layer {
            neurons: vec![
                Neuron {
                    weights: vec![1.0, 2.0],
                    bias: 0.0,
                },
                Neuron {
                    weights: vec![1.0, 2.0],
                    bias: 0.0,
                },
            ],
            activation: Activation::Sigmoid,
        };
        let errors = layer.propagate_error(&[0.5, 1.5]);
        assert!((errors[0] - 2.0).abs() < 1e-14);
        assert!((errors[1] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn relu_layer_output_is_non_negative() {
        let mut rng = seeded_rng();
        let layer = Layer::new(4, 3, Activation::ReLU, &mut rng);
        let outputs = layer.feed_forward(&[-5.0, 0.0, 5.0]);
        for &o in &outputs {
            assert!(o >= 0.0, "ReLU output {o} should be >= 0");
        }
    }

    #[test]
    fn tanh_layer_output_is_in_minus_one_to_one() {
        let mut rng = seeded_rng();
        let layer = Layer::new(4, 3, Activation::Tanh, &mut rng);
        let outputs = layer.feed_forward(&[-5.0, 0.0, 5.0]);
        for &o in &outputs {
            assert!((-1.0..=1.0).contains(&o), "tanh output {o} should be in (-1, 1)");
        }
    }
}
