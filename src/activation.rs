/// Activation functions for neural network layers.
///
/// Each variant is an adapter that satisfies the two-method interface
/// (activate / derivative) used by [`Layer`](crate::neural_layer::Layer)
/// during forward and backward propagation.
///
/// # Educational note
///
/// An enum was chosen over a trait + dynamic dispatch because:
///
/// - **Discoverability** — all available activations live in one place.
/// - **Simplicity** — no heap allocation, no trait objects, no serialization
///   gymnastics.
/// - **Serialization** — `#[derive(Serialize, Deserialize)]` works out of the
///   box, which is important for the planned save/load demo.
///
/// If you add a new activation, add a variant here and implement both
/// methods in the `match` arms below.
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Activation {
    /// Logistic sigmoid: σ(x) = 1 / (1 + e⁻ˣ).
    ///
    /// Maps any real to (0, 1). Historically the first activation used
    /// with backpropagation. Derivative is σ(x)·(1−σ(x)), expressible
    /// in terms of the output itself.
    Sigmoid,

    /// Rectified Linear Unit: ReLU(x) = max(0, x).
    ///
    /// Derivative is 0 for x ≤ 0, 1 for x > 0. Cheap, no saturation
    /// in the positive domain. Can "die" if too many neurons get stuck at 0.
    ReLU,

    /// Hyperbolic tangent: tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ).
    ///
    /// Maps any real to (-1, 1). Zero-centered, which helps gradient
    /// flow compared to sigmoid. Derivative is 1 − tanh²(x).
    Tanh,
}

impl Activation {
    /// Apply the activation function to a weighted sum.
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
        }
    }

    /// Derivative of the activation function, computed from the
    /// **output** value (the result of [`activate`](Self::activate)).
    ///
    /// For sigmoid:     σ′(x) = σ(x) · (1 − σ(x))
    /// For ReLU:        ReLU′(x) = 0 if x ≤ 0, 1 if x > 0
    /// For tanh:        tanh′(x) = 1 − tanh²(x)
    ///
    /// The argument `y` is the *output* of the activation — for sigmoid
    /// this lets us avoid recomputing the weighted sum.
    pub fn derivative(&self, y: f64) -> f64 {
        match self {
            Activation::Sigmoid => y * (1.0 - y),
            Activation::ReLU => {
                if y > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Tanh => 1.0 - y * y,
        }
    }

    /// Short display name for printing / debugging.
    pub fn name(&self) -> &'static str {
        match self {
            Activation::Sigmoid => "sigmoid",
            Activation::ReLU => "ReLU",
            Activation::Tanh => "tanh",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    // ── Sigmoid ────────────────────────────────────────────────────

    #[test]
    fn sigmoid_at_zero_is_point_five() {
        let a = Activation::Sigmoid;
        assert!((a.activate(0.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn sigmoid_saturates_to_zero_for_large_negative() {
        let a = Activation::Sigmoid;
        assert!(a.activate(-100.0) < 1e-40);
    }

    #[test]
    fn sigmoid_saturates_to_one_for_large_positive() {
        let a = Activation::Sigmoid;
        assert!((a.activate(100.0) - 1.0).abs() < 1e-40);
    }

    #[test]
    fn sigmoid_derivative_known_value() {
        let a = Activation::Sigmoid;
        // σ′(0) = σ(0)·(1−σ(0)) = 0.5·0.5 = 0.25
        assert!((a.derivative(0.5) - 0.25).abs() < 1e-15);
    }

    #[test]
    fn sigmoid_derivative_symmetric_around_0_5() {
        let a = Activation::Sigmoid;
        assert!((a.derivative(0.3) - a.derivative(0.7)).abs() < 1e-15);
    }

    // ── ReLU ───────────────────────────────────────────────────────

    #[test]
    fn relu_negative_is_zero() {
        let a = Activation::ReLU;
        assert_eq!(a.activate(-5.0), 0.0);
        assert_eq!(a.activate(-0.001), 0.0);
    }

    #[test]
    fn relu_positive_is_identity() {
        let a = Activation::ReLU;
        assert_eq!(a.activate(3.0), 3.0);
        assert_eq!(a.activate(0.0), 0.0);
    }

    #[test]
    fn relu_derivative_is_zero_for_negative_or_zero() {
        let a = Activation::ReLU;
        assert_eq!(a.derivative(0.0), 0.0);
        assert_eq!(a.derivative(-1.0), 0.0);
    }

    #[test]
    fn relu_derivative_is_one_for_positive() {
        let a = Activation::ReLU;
        assert_eq!(a.derivative(3.0), 1.0);
        assert_eq!(a.derivative(0.001), 1.0);
    }

    // ── Tanh ───────────────────────────────────────────────────────

    #[test]
    fn tanh_at_zero_is_zero() {
        let a = Activation::Tanh;
        assert!((a.activate(0.0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn tanh_saturates_to_one_for_large_positive() {
        let a = Activation::Tanh;
        assert!((a.activate(100.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn tanh_saturates_to_minus_one_for_large_negative() {
        let a = Activation::Tanh;
        assert!((a.activate(-100.0) - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn tanh_derivative_known_value() {
        let a = Activation::Tanh;
        // tanh(0) = 0, so tanh′(0) = 1 − 0 = 1
        assert!((a.derivative(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn tanh_derivative_at_one() {
        let a = Activation::Tanh;
        // tanh(arctanh(0.5)) = 0.5, so derivative = 1 − 0.25 = 0.75
        // Actually, the argument to derivative() is the *output* y = tanh(x).
        // tanh′(x) = 1 − tanh²(x) = 1 − y²
        // When y = tanh(x) = 0.5, derivative = 1 − 0.25 = 0.75
        assert!((a.derivative(0.5) - 0.75).abs() < 1e-15);
    }

    #[test]
    fn derivative_sigmoid_matches_reference() {
        let a = Activation::Sigmoid;
        // numerical check: σ′(x) ≈ (σ(x+h) − σ(x−h)) / 2h
        let x = 0.7;
        let y = a.activate(x);
        let h = 1e-8;
        let dy = (a.activate(x + h) - a.activate(x - h)) / (2.0 * h);
        assert!((a.derivative(y) - dy).abs() < 1e-5);
    }

    #[test]
    fn derivative_relu_matches_reference() {
        let a = Activation::ReLU;
        let x = 2.0;
        let h = 1e-8;
        let dy = (a.activate(x + h) - a.activate(x - h)) / (2.0 * h);
        // ReLU output = x, derivative = 1
        assert!((a.derivative(a.activate(x)) - dy).abs() < 1e-5);
    }

    #[test]
    fn derivative_tanh_matches_reference() {
        let a = Activation::Tanh;
        let x = 0.5;
        let h = 1e-8;
        let dy = (a.activate(x + h) - a.activate(x - h)) / (2.0 * h);
        assert!((a.derivative(a.activate(x)) - dy).abs() < 1e-5);
    }
}
