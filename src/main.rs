use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "and" | "or" | "xor" | "half_adder" | "circle" => {
                println!("Run: cargo run --example {}", args[1]);
                return;
            }
            _ => {}
        }
    }

    println!("Neuron — Neural Network from Scratch");
    println!();
    println!("Available examples (run individually):");
    println!("  cargo run --example and        — AND gate");
    println!("  cargo run --example or         — OR gate");
    println!("  cargo run --example xor        — XOR (needs hidden layer)");
    println!("  cargo run --example half_adder — Half-adder (multi-output)");
    println!("  cargo run --example circle     — Circle classifier (continuous)");
    println!();
    println!("Or run all tests:");
    println!("  cargo test");
}
