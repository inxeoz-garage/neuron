use neuron::data;
use neuron::neural_net::NeuralNetwork;

fn main() {
    let mut net = NeuralNetwork::new(&[2, 4, 1]);
    let data = data::xor();
    let lr = 0.7;

    let epochs = neuron::train::until_mse(&mut net, &data, lr, 0.001);
    println!("\nTrained in {epochs} epochs\n");

    neuron::eval::inference(&mut net, &data);
    println!("\nLearned parameters:");
    neuron::eval::parameters(&net);
}
