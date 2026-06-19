use neuron::{data, NeuralNetwork};

fn main() {
    let mut net = NeuralNetwork::new_sigmoid_mse(&[1, 16, 1]);
    let data = data::sine();
    let lr = 0.7;

    let epochs = net.train_until(&data, lr, 0.005, Some(200_000));
    println!("\nTrained in {epochs} epochs\n");

    println!("Sine approximation (x → predicted, expected):");
    for (inputs, expected) in &data {
        let outputs = net.feed_forward(inputs);
        let x = inputs[0];
        let pred = outputs[0];
        let exp = expected[0];
        let marker = if (pred - exp).abs() < 0.05 { "✓" } else { " " };
        println!("  {marker} x={x:<5.2}  pred={pred:.4}  exp={exp:.4}");
    }
    println!("\nLearned parameters:");
    net.print_parameters();
}
