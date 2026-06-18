use neuron::{data, train, NeuralNetwork};

fn train_xor() -> NeuralNetwork {
    let mut net = NeuralNetwork::new(&[2, 4, 1]);
    let data = data::xor();
    train::until_mse(&mut net, &data, 0.7, 0.001, Some(200_000));
    net
}

fn main() -> Result<(), eframe::Error> {
    let net = train_xor();
    neuron::visualize::visualize(&net)
}
