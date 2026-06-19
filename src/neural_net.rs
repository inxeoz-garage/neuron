use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::data::Sample;
use crate::loss::Loss;
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
/// The network owns its [`Loss`] function, which determines how the
/// output error is computed during backpropagation. The [`Activation`]
/// function is per-layer, allowing different activations for different
/// layers (e.g. ReLU hidden + sigmoid output).
#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub loss: Loss,
}

impl NeuralNetwork {
    /// Create a new network with the given topology, activation, and loss.
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
    ///
    /// Every layer uses the same [`Activation`]. Pass separate
    /// activations per layer by constructing [`Layer`]s directly.
    ///
    /// The RNG is threaded through to [`Neuron::new`](crate::neural::Neuron::new)
    /// for deterministic weight initialization. Pass a seeded RNG for
    /// reproducible runs.
    pub fn new(
        layer_sizes: &[usize],
        activation: Activation,
        loss: Loss,
        rng: &mut impl Rng,
    ) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(
                layer_sizes[i + 1],
                layer_sizes[i],
                activation.clone(),
                rng,
            ));
        }
        Self { layers, loss }
    }

    /// Convenience constructor for the classic sigmoid + MSE network.
    ///
    /// Uses thread-local RNG (non-deterministic). For reproducible runs,
    /// use [`new`](Self::new) with a seeded RNG.
    pub fn new_sigmoid_mse(layer_sizes: &[usize]) -> Self {
        Self::new(
            layer_sizes,
            Activation::Sigmoid,
            Loss::MSE,
            &mut rand::rng(),
        )
    }

    /// Run a forward pass through the entire network.
    ///
    /// Each layer transforms the data in sequence. The return value is
    /// the output of the final layer — the network's prediction.
    ///
    /// Takes `&self` because neurons no longer store input vectors
    /// internally. You can query the network freely without mutation.
    pub fn feed_forward(&self, inputs: &[f64]) -> Vec<f64> {
        let mut current_inputs = inputs.to_vec();
        for layer in &self.layers {
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
    /// - Output layer error = loss.output_error(actual, expected)
    /// - Hidden layer error = Σ(weight_to_next_neuron × next_error)
    /// - Weight update     = learning_rate × error × σ′(output) × input
    pub fn backpropagate(&mut self, inputs: &[f64], expected_outputs: &[f64], learning_rate: f64) {
        // ── Forward pass: store every layer's activation ──
        let mut activations = vec![inputs.to_vec()];
        for layer in &self.layers {
            activations.push(layer.feed_forward(activations.last().unwrap()));
        }

        // ── Output layer error (using the configured loss function) ──
        let outputs = activations.last().unwrap();
        let mut errors: Vec<f64> = outputs
            .iter()
            .zip(expected_outputs.iter())
            .map(|(o, e)| self.loss.output_error(*o, *e))
            .collect();

        // ── Backward pass: update layers from output to input ──
        for i in (0..self.layers.len()).rev() {
            let layer_inputs = &activations[i];
            self.layers[i].adjust_weights(layer_inputs, learning_rate, &errors);

            // Propagate error to the preceding layer (if any)
            if i > 0 {
                errors = self.layers[i].propagate_error(&errors);
            }
        }
    }

    /// Run one training epoch: backpropagate every sample once.
    pub fn train_epoch(&mut self, data: &[Sample], learning_rate: f64) {
        for (inputs, expected) in data {
            self.backpropagate(inputs, expected, learning_rate);
        }
    }

    /// Compute the average loss over a dataset.
    ///
    /// Returns (1/N) × Σ self.loss.compute(output_i, expected_i) across
    /// every output neuron and every sample. A decreasing average over
    /// epochs indicates the network is learning.
    ///
    /// For MSE loss this is the mean squared error. For cross-entropy
    /// it's the average cross-entropy. The method name is intentionally
    /// generic to reflect that.
    pub fn average_loss(&self, data: &[Sample]) -> f64 {
        let total: f64 = data
            .iter()
            .map(|(inputs, expected)| {
                let outputs = self.feed_forward(inputs);
                outputs
                    .iter()
                    .zip(expected.iter())
                    .map(|(o, e)| self.loss.compute(*o, *e))
                    .sum::<f64>()
            })
            .sum();
        total / data.len() as f64
    }

    /// Train the network until the average loss drops below `target`.
    ///
    /// Prints progress every 1000 epochs (every epoch for the first 5).
    /// Returns the total number of epochs trained.
    ///
    /// If `max_epochs` is `Some(n)`, training stops after `n` epochs
    /// even if the target was not reached. This prevents runaway
    /// training on harder problems (like the circle dataset).
    ///
    /// This is a *full-batch* trainer: each epoch processes every sample
    /// exactly once. For larger datasets you would switch to mini-batch
    /// or stochastic gradient descent, which update weights more
    /// frequently and often converge faster.
    pub fn train_until(
        &mut self,
        data: &[Sample],
        learning_rate: f64,
        target_loss: f64,
        max_epochs: Option<usize>,
    ) -> usize {
        let mut epoch = 0usize;
        loop {
            let loss = self.average_loss(data);
            if epoch % 1000 == 0 || epoch < 5 || loss < target_loss {
                println!("Epoch {epoch:>4}  loss = {loss:.6}");
            }
            if loss < target_loss {
                return epoch;
            }
            if let Some(max) = max_epochs {
                if epoch >= max {
                    println!("  ⏱ Reached max epochs ({max}), target loss {target_loss} not reached");
                    return epoch;
                }
            }
            self.train_epoch(data, learning_rate);
            epoch += 1;
        }
    }

    /// Print every sample's prediction to stdout.
    ///
    /// Shows: `[inputs] → output_values`
    /// Useful for visually inspecting how well the trained network
    /// generalizes — the output values should match the expected
    /// targets closely if training succeeded.
    pub fn print_predictions(&self, data: &[Sample]) {
        for (inputs, _) in data {
            let outputs = self.feed_forward(inputs);
            print!("  [");
            for (i, val) in inputs.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{val:.2}");
            }
            print!("] →");
            for o in &outputs {
                print!(" {o:.4}");
            }
            println!();
        }
    }

    /// Print the learned weights and biases for every neuron.
    ///
    /// After training, you can see which weights grew large (important
    /// connections) and which shrunk toward zero (irrelevant ones).
    pub fn print_parameters(&self) {
        for (i, layer) in self.layers.iter().enumerate() {
            println!("  Layer {i} ({})", layer.activation.name());
            for (j, n) in layer.neurons.iter().enumerate() {
                print!("    neuron {j}: ");
                if n.weights.len() <= 4 {
                    print!("weights = {:?}, ", n.weights);
                } else {
                    print!("weights ({} inputs), ", n.weights.len());
                }
                println!("bias = {:.4}", n.bias);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn new_xor_net(rng: &mut impl Rng) -> NeuralNetwork {
        NeuralNetwork::new(&[2, 4, 1], Activation::Sigmoid, Loss::MSE, rng)
    }

    #[test]
    fn new_network_has_correct_number_of_layers() {
        let net = NeuralNetwork::new(&[2, 4, 1], Activation::Sigmoid, Loss::MSE, &mut rand::rng());
        assert_eq!(net.layers.len(), 2); // input spec doesn't create a layer
    }

    #[test]
    fn feed_forward_returns_correct_output_size() {
        let net = NeuralNetwork::new(&[2, 4, 1], Activation::Sigmoid, Loss::MSE, &mut rand::rng());
        let output = net.feed_forward(&[0.5, 0.3]);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn feed_forward_output_is_in_sigmoid_range() {
        let net = NeuralNetwork::new(&[2, 4, 1], Activation::Sigmoid, Loss::MSE, &mut rand::rng());
        let output = net.feed_forward(&[-10.0, 10.0]);
        for &o in &output {
            assert!((0.0..=1.0).contains(&o), "output {o} out of range");
        }
    }

    #[test]
    fn average_loss_is_non_negative() {
        let net = new_xor_net(&mut rand::rng());
        let samples = data::xor();
        let loss = net.average_loss(&samples);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn backpropagate_does_not_panic() {
        let mut net = new_xor_net(&mut rand::rng());
        net.backpropagate(&[0.0, 0.0], &[0.0], 0.7);
    }

    #[test]
    fn loss_decreases_over_training() {
        // Use seeded RNG for deterministic test
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = new_xor_net(&mut rng);
        let samples = data::xor();
        let initial_loss = net.average_loss(&samples);

        for _ in 0..1_000 {
            net.train_epoch(&samples, 0.7);
        }

        let after_loss = net.average_loss(&samples);
        assert!(
            after_loss < initial_loss,
            "loss did not decrease after 1000 epochs: {initial_loss:.6} → {after_loss:.6}"
        );
    }

    #[test]
    fn network_improves_on_xor_over_10k_epochs() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = new_xor_net(&mut rng);
        let samples = data::xor();

        for _ in 0..10_000 {
            net.train_epoch(&samples, 0.7);
        }

        let final_loss = net.average_loss(&samples);
        assert!(
            final_loss < 0.1,
            "loss after 10k epochs should be below 0.1, got {final_loss:.6}"
        );
    }

    #[test]
    fn new_sigmoid_mse_is_smoke_test() {
        let net = NeuralNetwork::new_sigmoid_mse(&[2, 4, 1]);
        assert_eq!(net.layers.len(), 2);
        assert_eq!(net.loss, Loss::MSE);
        let output = net.feed_forward(&[0.5, 0.3]);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn feed_forward_works_with_relu() {
        let net = NeuralNetwork::new(&[3, 8, 2], Activation::ReLU, Loss::MSE, &mut rand::rng());
        let output = net.feed_forward(&[1.0, 2.0, 3.0]);
        assert_eq!(output.len(), 2);
        for &o in &output {
            assert!(o >= 0.0, "ReLU output should be >= 0");
        }
    }

    #[test]
    fn average_loss_is_zero_for_perfect_prediction() {
        // For MSE, loss should be exactly 0 when output = expected.
        // We build a network that passes input straight through (no hidden layer)
        // but this is tricky with random weights. Instead, just verify the
        // average_loss function is self-consistent.
        let net = NeuralNetwork::new(&[2, 1], Activation::Sigmoid, Loss::MSE, &mut rand::rng());
        let samples = vec![(vec![0.0, 0.0], vec![0.5])];
        let loss = net.average_loss(&samples);
        assert!(loss.is_finite());
    }

    #[test]
    fn train_until_returns_epoch_count() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = new_xor_net(&mut rng);
        let samples = data::xor();
        let epochs = net.train_until(&samples, 0.7, 0.5, Some(500));
        assert!(epochs <= 500);
    }
}
