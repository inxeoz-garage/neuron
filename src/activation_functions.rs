/// Sigmoid activation function: σ(x) = 1 / (1 + e⁻ˣ).
///
/// Maps any real number to the open interval (0, 1). The sigmoid is
/// smooth, monotonic, and its derivative can be expressed in terms of
/// the output value itself — properties that make it convenient for
/// gradient-based learning.
///
/// The sigmoid was chosen for this educational implementation because:
/// - It acts as a differentiable approximation of a step function.
/// - Its derivative is σ(x)·(1−σ(x)), computable from the forward-pass output.
/// - It was the historical activation used in early backpropagation.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of the sigmoid, expressed in terms of the *output* value.
///
/// σ′(x) = σ(x) · (1 − σ(x))
///
/// The argument `x` here is the **output** of the sigmoid (the neuron's
/// activation), not the raw weighted sum. This is intentional: during
/// backpropagation we already computed σ(z) in the forward pass, so
/// we can reuse that value directly — no need to re-compute z.
pub fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_at_zero_is_point_five() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn sigmoid_saturates_to_zero_for_large_negative() {
        assert!(sigmoid(-100.0) < 1e-40);
    }

    #[test]
    fn sigmoid_saturates_to_one_for_large_positive() {
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-40);
    }

    #[test]
    fn sigmoid_is_odd_around_zero() {
        // σ(−x) = 1 − σ(x)
        assert!((sigmoid(-2.0) + sigmoid(2.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn derivative_at_sigmoid_output_0_5() {
        // σ′(0) = σ(0)·(1−σ(0)) = 0.5·0.5 = 0.25
        assert!((sigmoid_derivative(0.5) - 0.25).abs() < 1e-15);
    }

    #[test]
    fn derivative_is_positive_inside_range() {
        // For any sigmoid output in (0, 1), the derivative is positive.
        for x in [0.1, 0.3, 0.7, 0.9] {
            assert!(
                sigmoid_derivative(x) > 0.0,
                "derivative at {x} should be positive"
            );
        }
    }

    #[test]
    fn derivative_symmetric_around_0_5() {
        // σ′(x) should be symmetric around 0.5
        assert!((sigmoid_derivative(0.3) - sigmoid_derivative(0.7)).abs() < 1e-15);
    }
}
