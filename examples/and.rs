use neuron::{data, NeuralNetwork};

fn main() {
    let mut net = NeuralNetwork::new_sigmoid_mse(&[2, 1]);
    let data = data::and();
    let lr = 0.7;

    let epochs = net.train_until(&data, lr, 0.001, None);
    println!("\nTrained in {epochs} epochs\n");

    println!("Truth table:");
    net.print_predictions(&data);
    println!("\nLearned parameters:");
    net.print_parameters();
}
