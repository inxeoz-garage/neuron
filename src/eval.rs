use crate::data::Sample;
use crate::neural_net::NeuralNetwork;

/// Run the network on every sample in `data` and print each prediction.
///
/// This is useful for visually inspecting how well the trained network
/// generalizes — the output values should match the expected targets
/// closely if training succeeded.
pub fn inference(net: &mut NeuralNetwork, data: &[Sample]) {
    for (inputs, _) in data {
        let outputs = net.feed_forward(inputs);
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
pub fn parameters(net: &NeuralNetwork) {
    for (i, layer) in net.layers.iter().enumerate() {
        println!("  Layer {i}:");
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
