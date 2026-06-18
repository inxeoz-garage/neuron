use neuron::{data, eval, train, NeuralNetwork};

fn main() {
    let mut net = NeuralNetwork::new(&[3, 8, 2]);
    let data = data::full_adder();
    let lr = 0.7;

    let epochs = train::until_mse(&mut net, &data, lr, 0.001, None);
    println!("\nTrained in {epochs} epochs\n");

    println!("Truth table (a, b, carry_in) → (sum, carry_out):");
    eval::inference(&mut net, &data);
    println!("\nLearned parameters:");
    eval::parameters(&net);
}
