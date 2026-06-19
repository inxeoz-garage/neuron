use neuron::{data, NeuralNetwork};

fn train_xor() -> NeuralNetwork {
    let mut net = NeuralNetwork::new_sigmoid_mse(&[2, 4, 1]);
    let data = data::xor();
    net.train_until(&data, 0.7, 0.001, Some(200_000));
    net
}

fn main() -> Result<(), eframe::Error> {
    let net = train_xor();
    neuron::visualize::visualize(&net)
}
