use serde::{Deserialize, Serialize};

use crate::activation_functions::sigmoid_derivative;
use crate::neural::Neuron;

/// A single layer of neurons in a feed-forward network.
///
/// Each neuron in the layer receives the **same** input vector (the
/// output of the previous layer, or the raw data for the first hidden
/// layer). The layer's output is the vector of every neuron's
/// activation — one scalar per neuron.
///
/// The `neurons` field is public so you can inspect individual neurons
/// after training (e.g. print their weights).
#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    /// Creates a layer with `num_neurons` neurons, each connected to
    /// `num_inputs` inputs from the previous layer.
    pub fn new(num_neurons: usize, num_inputs: usize) -> Self {
        Self {
            neurons: (0..num_neurons).map(|_| Neuron::new(num_inputs)).collect(),
        }
    }

    /// Compute the output of every neuron in the layer.
    ///
    /// Each neuron independently computes σ(∑(input_i × weight_i) + bias).
    /// The result is a vector of length `num_neurons` — the activations
    /// that will be passed to the next layer (or used as the network
    /// output for the final layer).
    pub fn feed_forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter_mut()
            .map(|neuron| neuron.activate(inputs))
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
            let output = neuron.output();
            // δ = error × σ′(output)  — shared factor for all weight updates
            let delta = error * sigmoid_derivative(output);
            for (i, input) in inputs.iter().enumerate() {
                neuron.weights[i] += learning_rate * delta * input;
            }
            neuron.bias += learning_rate * delta;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feed_forward_returns_correct_number_of_outputs() {
        let mut layer = Layer::new(4, 3); // 4 neurons, 3 inputs each
        let outputs = layer.feed_forward(&[1.0, 0.5, 0.2]);
        assert_eq!(outputs.len(), 4);
    }

    #[test]
    fn feed_forward_outputs_are_in_sigmoid_range() {
        let mut layer = Layer::new(4, 3);
        let outputs = layer.feed_forward(&[-5.0, 0.0, 5.0]);
        for &o in &outputs {
            assert!((0.0..1.0).contains(&o), "output {o} should be in (0, 1)");
        }
    }

    #[test]
    fn adjust_weights_changes_bias() {
        let mut layer = Layer::new(1, 2);
        layer.feed_forward(&[1.0, 2.0]);
        let bias_before = layer.neurons[0].bias;
        layer.adjust_weights(&[1.0, 2.0], 0.1, &[0.5]);
        let bias_after = layer.neurons[0].bias;
        assert!(
            (bias_after - bias_before).abs() > 1e-15,
            "bias should change after adjust_weights"
        );
    }
}
