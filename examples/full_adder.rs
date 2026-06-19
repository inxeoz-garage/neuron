use neuron::{data, NeuralNetwork};

fn main() {
    let mut net = NeuralNetwork::new_sigmoid_mse(&[3, 8, 2]);
    let data = data::full_adder();
    let lr = 0.7;

    let epochs = net.train_until(&data, lr, 0.001, None);
    println!("\nTrained in {epochs} epochs\n");

    println!("Truth table (a, b, carry_in) → (sum, carry_out):");
    net.print_predictions(&data);
    println!("\nLearned parameters:");
    net.print_parameters();
}
