use neuron::{data, eval, train, NeuralNetwork};

fn main() {
    // Parity-3 is famously hard for sigmoid networks (more local minima
    // than XOR). More hidden neurons and a relaxed target help, but
    // convergence is not guaranteed on every run.
    let mut net = NeuralNetwork::new(&[3, 16, 1]);
    let data = data::parity3();
    let lr = 0.7;

    let epochs = train::until_mse(&mut net, &data, lr, 0.01, Some(200_000));
    println!("\nTrained in {epochs} epochs\n");

    println!("Truth table (3-bit parity — odd number of 1s):");
    eval::inference(&mut net, &data);
    println!("\nLearned parameters:");
    eval::parameters(&net);
}
