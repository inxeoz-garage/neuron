use neuron::{data, NeuralNetwork};

fn main() {
    // Parity-3 is famously hard for sigmoid networks (more local minima
    // than XOR). More hidden neurons and a relaxed target help, but
    // convergence is not guaranteed on every run.
    let mut net = NeuralNetwork::new_sigmoid_mse(&[3, 16, 1]);
    let data = data::parity3();
    let lr = 0.7;

    let epochs = net.train_until(&data, lr, 0.01, Some(200_000));
    println!("\nTrained in {epochs} epochs\n");

    println!("Truth table (3-bit parity — odd number of 1s):");
    net.print_predictions(&data);
    println!("\nLearned parameters:");
    net.print_parameters();
}
