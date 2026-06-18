use serde::{Deserialize, Serialize};

use crate::data::Sample;
use crate::neural_layer::Layer;

/// A feed-forward neural network consisting of sequential [`Layer`]s.
///
/// Data flows forward through the layers:
///
/// ```text
/// input → [Layer 0] → [Layer 1] → … → [Layer N] → output
/// ```
///
/// Every neuron in layer L receives the **entire output** of layer L-1
/// as its input vector. This is a *fully connected* (dense) architecture.
///
/// The `layers` field is public so you can inspect parameters after
/// training, e.g. `net.layers[0].neurons[0].weights`.
#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    /// Create a new network with the given topology.
    ///
    /// `layer_sizes` describes the number of neurons in each layer,
    /// **including** the input layer. For example, `&[2, 4, 1]` creates:
    ///
    /// - An **input** layer of size 2 (just specifies how many inputs
    ///   the network receives — no actual neurons are allocated for it).
    /// - A **hidden** layer with 4 neurons, each receiving 2 inputs.
    /// - An **output** layer with 1 neuron, receiving 4 inputs.
    ///
    /// The first element of `layer_sizes` defines the number of features
    /// in the input data, nothing more.
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            // Layer i receives `layer_sizes[i]` inputs (the previous
            // layer's output or the raw data) and has `layer_sizes[i+1]`
            // neurons.
            layers.push(Layer::new(layer_sizes[i + 1], layer_sizes[i]));
        }
        Self { layers }
    }

    /// Run a forward pass through the entire network.
    ///
    /// Each layer transforms the data in sequence. The return value is
    /// the output of the final layer — the network's prediction.
    ///
    /// This takes `&mut self` because [`Layer::feed_forward`] calls
    /// [`Neuron::activate`], which stores the input vector inside each
    /// neuron for later use during backpropagation.
    pub fn feed_forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut current_inputs = inputs.to_vec();
        for layer in &mut self.layers {
            current_inputs = layer.feed_forward(&current_inputs);
        }
        current_inputs
    }

    /// Backpropagate the error for a single training sample.
    ///
    /// This performs two passes:
    ///
    /// 1. **Forward pass**: store activations for every layer (needed
    ///    later as the "input" view for each layer's weight update).
    /// 2. **Backward pass**: starting from the output layer, compute
    ///    the error contribution of each neuron and update its weights.
    ///
    /// The math behind the backward pass:
    ///
    /// - Output layer error = expected − actual
    /// - Hidden layer error = Σ(weight_to_next_neuron × next_error)
    /// - Weight update     = learning_rate × error × σ′(output) × input
    pub fn backpropagate(&mut self, inputs: &[f64], expected_outputs: &[f64], learning_rate: f64) {
        // ── Forward pass: store every layer's activation ──
        let mut activations = vec![inputs.to_vec()];
        for layer in &mut self.layers {
            activations.push(layer.feed_forward(activations.last().unwrap()));
        }

        // ── Output layer error ──
        let mut errors: Vec<f64> = self
            .layers
            .last()
            .unwrap()
            .neurons
            .iter()
            .enumerate()
            .map(|(i, neuron)| expected_outputs[i] - neuron.output())
            .collect();

        // ── Backward pass: update layers from output to input ──
        for i in (0..self.layers.len()).rev() {
            let layer_inputs = &activations[i];
            self.layers[i].adjust_weights(layer_inputs, learning_rate, &errors);

            // Propagate error to the preceding layer (if any)
            if i > 0 {
                let prev_neurons = &self.layers[i].neurons;
                errors = prev_neurons
                    .iter()
                    .map(|neuron| {
                        neuron
                            .weights
                            .iter()
                            .zip(errors.iter())
                            .map(|(weight, &err)| weight * err)
                            .sum()
                    })
                    .collect();
            }
        }
    }

    /// Run one training epoch: backpropagate every sample once.
    pub fn train_epoch(&mut self, data: &[Sample], learning_rate: f64) {
        for (inputs, expected) in data {
            self.backpropagate(inputs, expected, learning_rate);
        }
    }

    /// Compute the mean squared error over a dataset.
    ///
    /// MSE = (1/N) × Σ(actual − expected)² for every output neuron
    /// across every sample. A decreasing MSE over epochs indicates
    /// the network is learning.
    pub fn mean_squared_error(&mut self, data: &[Sample]) -> f64 {
        let total: f64 = data
            .iter()
            .map(|(inputs, expected)| {
                let outputs = self.feed_forward(inputs);
                outputs
                    .iter()
                    .zip(expected.iter())
                    .map(|(o, e)| (e - o).powi(2))
                    .sum::<f64>()
            })
            .sum();
        total / data.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data;

    #[test]
    fn new_network_has_correct_number_of_layers() {
        let net = NeuralNetwork::new(&[2, 4, 1]);
        assert_eq!(net.layers.len(), 2); // input spec doesn't create a layer
    }

    #[test]
    fn feed_forward_returns_correct_output_size() {
        let mut net = NeuralNetwork::new(&[2, 4, 1]);
        let output = net.feed_forward(&[0.5, 0.3]);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn feed_forward_output_is_in_sigmoid_range() {
        let mut net = NeuralNetwork::new(&[2, 4, 1]);
        let output = net.feed_forward(&[-10.0, 10.0]);
        for &o in &output {
            assert!((0.0..=1.0).contains(&o), "output {o} out of range");
        }
    }

    #[test]
    fn mean_squared_error_is_non_negative() {
        let mut net = NeuralNetwork::new(&[2, 4, 1]);
        let samples = data::xor();
        let mse = net.mean_squared_error(&samples);
        assert!(mse.is_finite());
        assert!(mse >= 0.0);
    }

    #[test]
    fn backpropagate_does_not_panic() {
        let mut net = NeuralNetwork::new(&[2, 4, 1]);
        net.backpropagate(&[0.0, 0.0], &[0.0], 0.7);
    }

    #[test]
    fn mse_decreases_over_training() {
        let mut net = NeuralNetwork::new(&[2, 4, 1]);
        let samples = data::xor();
        let initial_mse = net.mean_squared_error(&samples);

        for _ in 0..1_000 {
            net.train_epoch(&samples, 0.7);
        }

        let after_mse = net.mean_squared_error(&samples);
        assert!(
            after_mse < initial_mse,
            "MSE did not decrease after 1000 epochs: {initial_mse:.6} → {after_mse:.6}"
        );
    }

    #[test]
    fn network_improves_on_xor_over_10k_epochs() {
        let mut net = NeuralNetwork::new(&[2, 4, 1]);
        let samples = data::xor();

        for _ in 0..10_000 {
            net.train_epoch(&samples, 0.7);
        }

        let final_mse = net.mean_squared_error(&samples);
        assert!(
            final_mse < 0.1,
            "MSE after 10k epochs should be below 0.1, got {final_mse:.6}"
        );
    }
}
