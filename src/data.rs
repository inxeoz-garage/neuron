/// A labeled training sample.
///
/// The first element is the input vector fed into the network.
/// The second element is the expected output vector (the target).
/// During training, the error between the network's prediction and
/// this target drives the weight updates via backpropagation.
pub type Sample = (Vec<f64>, Vec<f64>);

/// Returns the XOR truth table — the canonical "hello world" for
/// neural networks.
///
/// XOR is *not linearly separable*: no single straight line can
/// separate {(0,0), (1,1)} from {(0,1), (1,0)}. This is why a
/// network **without** a hidden layer cannot learn XOR, while a
/// network **with** one hidden layer can — the hidden neurons learn
/// to project the inputs into a space where the output neuron can
/// separate them linearly.
///
/// Each sample: two binary inputs → one binary output.
pub fn xor() -> Vec<Sample> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}
