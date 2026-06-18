use neuron::{data, eval, train, NeuralNetwork};

fn main() {
    let mut net = NeuralNetwork::new(&[2, 4, 1]);
    let data = data::xor();
    let lr = 0.7;

    let epochs = train::until_mse(&mut net, &data, lr, 0.001, None);
    println!("\nTrained in {epochs} epochs\n");

    println!("Truth table:");
    eval::inference(&mut net, &data);
    println!("\nLearned parameters:");
    eval::parameters(&net);
}
