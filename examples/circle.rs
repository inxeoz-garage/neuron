use neuron::{data, eval, train, NeuralNetwork};

fn main() {
    let mut net = NeuralNetwork::new(&[2, 16, 1]);
    let data = data::circle();
    let lr = 0.7;

    let epochs = train::until_mse(&mut net, &data, lr, 0.15, Some(50_000));
    println!("\nTrained in {epochs} epochs\n");

    println!("Predictions (5×5 grid — output ≈ 1 inside circle):");
    eval::inference(&mut net, &data);
    println!("\nLearned parameters:");
    eval::parameters(&net);
}
