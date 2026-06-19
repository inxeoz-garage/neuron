use neuron::{data, NeuralNetwork};

fn main() {
    let mut net = NeuralNetwork::new_sigmoid_mse(&[2, 16, 1]);
    let data = data::circle();
    let lr = 0.7;

    let epochs = net.train_until(&data, lr, 0.15, Some(50_000));
    println!("\nTrained in {epochs} epochs\n");

    println!("Predictions (5×5 grid — output ≈ 1 inside circle):");
    net.print_predictions(&data);
    println!("\nLearned parameters:");
    net.print_parameters();
}
