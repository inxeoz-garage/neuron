# Neuron — Neural Network from Scratch in Rust

A minimal, pedagogical feed-forward neural network implementing backpropagation on XOR. Built from first principles — no frameworks, no automatic differentiation, just matrix-free scalar neurons.

The full derivation from linear regression through gradient descent is documented in [`neuron.md`](neuron.md).

## Quick Start

```bash
# Train and test on XOR
cargo run --release
```

Converges to MSE < 0.001 in ~50k epochs and prints the XOR truth table result.

## Project Structure

```
src/
├── main.rs          # Entry point: create net, load data, train loop, test
├── lib.rs           # Crate root — declares public modules
├── data.rs          # Sample type alias, xor() dataset function
├── neural.rs        # Sigmoid, sigmoid_derivative, xavier_init, Neuron
├── neural_layer.rs  # Layer — vector of neurons, forward + weight update
└── neural_net.rs    # NeuralNetwork — feed_forward, backpropagate, train_epoch, MSE
```

### Module responsibilities

| Module | Exports | Role |
|---|---|---|
| `data` | `Sample`, `xor()` | Dataset definitions — swap to train on another problem |
| `neural` | `Neuron`, `sigmoid`, `sigmoid_derivative`, `xavier_init` | Single neuron: weighted sum, activation, output computation |
| `neural_layer` | `Layer` | Collection of neurons, layer-level forward pass and weight adjustment |
| `neural_net` | `NeuralNetwork` | Network topology, forward/backward propagation, training helpers |

## How It Works

```
input [2] → hidden layer [4 neurons] → output layer [1 neuron] → prediction
```

- **Activation**: sigmoid on every neuron.
- **Init**: Xavier (uniform) weight initialization.
- **Loss**: mean squared error.
- **Training**: full-batch gradient descent with backpropagation.

### Key methods on `NeuralNetwork`

```rust
// Forward pass — returns output vector
net.feed_forward(&[0.0, 1.0]);

// Backpropagate one sample
net.backpropagate(&[0.0, 1.0], &[1.0], 0.7);

// One training epoch over the whole dataset
net.train_epoch(&data, 0.7);

// Mean squared error over the dataset
net.mean_squared_error(&data);
```

## Extending

### Add a new dataset

Create a function in `src/data.rs` returning `Vec<Sample>`:

```rust
/// AND gate: both inputs high → output high.
pub fn and() -> Vec<Sample> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ]
}
```

Then swap the call in `main.rs`:

```rust
let data = data::and();
```

### Change network topology

```rust
// 3 inputs → 8 hidden → 4 hidden → 2 outputs
let mut net = NeuralNetwork::new(&[3, 8, 4, 2]);
```

### Add a new activation function

1. Add the function and its derivative to `src/neural.rs`:

```rust
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}
```

2. Call it inside `Neuron::activate` and `Neuron::output` (or make activation configurable).

### Use the library from another crate

Add to your `Cargo.toml`:

```toml
neuron = { git = "…" }
```

Then:

```rust
use neuron::neural_net::NeuralNetwork;
use neuron::data;

let mut net = NeuralNetwork::new(&[2, 16, 1]);
net.train_epoch(&data::xor(), 0.5);
```

### Serialize/Deserialize

Networks derive `Serialize`/`Deserialize` via serde:

```rust
let json = serde_json::to_string(&net).unwrap();
let net: NeuralNetwork = serde_json::from_str(&json).unwrap();
```

## Dependencies

| Crate | Version | Why |
|---|---|---|
| `rand` | 0.9 | Xavier weight initialization |
| `serde` | 1 (with `derive`) | Optional serialization |

## See Also

- [`neuron.md`](neuron.md) — full mathematical derivation from linear regression to backpropagation
