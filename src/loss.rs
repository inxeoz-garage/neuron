/// Loss functions for training neural networks.
///
/// Each variant is an adapter that provides two methods used by
/// [`NeuralNetwork`](crate::neural_net::NeuralNetwork):
///
/// - [`compute`](Loss::compute) — the per-sample, per-output loss value
///   (used for monitoring / `average_loss`).
/// - [`output_error`](Loss::output_error) — the initial error signal for
///   backpropagation (fed into the output layer's weight update, then
///   multiplied by the activation derivative).
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Loss {
    /// Mean Squared Error: ½(y − ŷ)².
    ///
    /// The output error is `y − ŷ` (expected minus actual), which is the
    /// negative gradient of ½(y − ŷ)² w.r.t. ŷ. Gradient descent then
    /// adds `η × error × σ′(z) × input` to each weight — the `+=`
    /// convention absorbs the minus sign.
    MSE,
}

impl Loss {
    /// Per-element loss value for a single output neuron.
    ///
    /// For MSE: `½(y − ŷ)²`.
    /// The ½ factor cancels the 2 from differentiation, keeping gradient
    /// expressions clean (same convention as the original codebase).
    pub fn compute(&self, output: f64, expected: f64) -> f64 {
        match self {
            Loss::MSE => 0.5 * (expected - output).powi(2),
        }
    }

    /// Error signal propagated from the output layer into backprop.
    ///
    /// This value is passed to [`Layer::adjust_weights`], which multiplies
    /// it by the activation derivative before updating weights.
    ///
    /// For MSE: `expected − output` — the negative gradient of the loss
    /// w.r.t. the output. Gradient descent updates weights with `+=`,
    /// which makes this the correct direction.
    pub fn output_error(&self, output: f64, expected: f64) -> f64 {
        match self {
            Loss::MSE => expected - output,
        }
    }

    /// Short display name for printing / debugging.
    pub fn name(&self) -> &'static str {
        match self {
            Loss::MSE => "MSE",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_compute_zero_when_equal() {
        assert!((Loss::MSE.compute(0.5, 0.5) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn mse_compute_known_value() {
        // ½(1.0 − 0.0)² = 0.5
        assert!((Loss::MSE.compute(0.0, 1.0) - 0.5).abs() < 1e-15);
        // ½(0.0 − 1.0)² = 0.5 (symmetric)
        assert!((Loss::MSE.compute(1.0, 0.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn mse_output_error_sign_matches_gradient_descent() {
        let error = Loss::MSE.output_error(0.3, 0.8);
        // expected > output → positive error → weight update increases output
        assert!(error > 0.0);
        let error2 = Loss::MSE.output_error(0.8, 0.3);
        assert!(error2 < 0.0);
    }

    #[test]
    fn mse_output_error_is_expected_minus_output() {
        let e = Loss::MSE.output_error(0.2, 0.7);
        assert!((e - 0.5).abs() < 1e-15);
    }
}
