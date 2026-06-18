# Neuron — Neural Network from Scratch in Rust

A minimal, pedagogical feed-forward neural network implementing backpropagation from first principles — no frameworks, no autodiff, just scalar neurons.

The full derivation from linear regression through gradient descent is documented in [`neuron.md`](neuron.md).

## Quick Start — Run the Examples

Each example trains a fresh network on a different problem, from simplest to hardest:

```bash
# Simple boolean gates (single neuron, linearly separable)
cargo run --release --example and          # AND gate
cargo run --release --example or           # OR gate

# Needs a hidden layer (not linearly separable)
cargo run --release --example xor          # XOR — the classic test

# Multi-output: two output neurons at once
cargo run --release --example half_adder   # (sum, carry)

# Continuous inputs, non-linear decision boundary
cargo run --release --example circle       # 2D circle classification
```

Run all unit tests:

```bash
cargo test
```

## Example Progression

| Example | Topology | Difficulty | What it teaches |
|---|---|---|---|
| `and` | `[2, 1]` | ★ | Single neuron can learn linearly separable problems |
| `or` | `[2, 1]` | ★ | Same — contrasts with XOR |
| `xor` | `[2, 4, 1]` | ★★ | Hidden layer needed for non-linearly-separable problems |
| `half_adder` | `[2, 4, 2]` | ★★★ | Multi-output: one network solving two problems at once |
| `circle` | `[2, 16, 1]` | ★★★★ | Continuous inputs, curved boundary — tests generalization |

## Project Structure

```
src/                          examples/
├── main.rs                   ├── and.rs
├── lib.rs                    ├── or.rs
├── activation_functions.rs   ├── xor.rs
├── data.rs                   ├── half_adder.rs
├── eval.rs                   └── circle.rs
├── neural.rs
├── neural_layer.rs
├── neural_net.rs
└── train.rs
```

### Module responsibilities

| Module | Exports | Role |
|---|---|---|
| `activation_functions` | `sigmoid`, `sigmoid_derivative` | Non-linear activation and its derivative |
| `data` | `Sample`, `xor`, `and`, `or`, `half_adder`, `circle` | Training datasets — swap or add your own |
| `neural` | `Neuron`, `xavier_init` | Single neuron: weighted sum, activation, output |
| `neural_layer` | `Layer` | Neuron collection: layer-level forward pass, weight adjustment |
| `neural_net` | `NeuralNetwork` | Network topology, forward/backward propagation, training helpers |
| `train` | `until_mse` | Training loop with convergence check and logging |
| `eval` | `inference`, `parameters` | Run predictions, inspect learned weights |
| `main` | — | Lists available examples |

## How It Works

```
input [2] → hidden layer [4 neurons] → output layer [1 neuron] → prediction
```

- **Activation**: sigmoid on every neuron.
- **Init**: Xavier (uniform) weight initialization.
- **Loss**: mean squared error.
- **Training**: full-batch gradient descent with backpropagation.

### Key API

```rust
use neuron::NeuralNetwork;
use neuron::data;

let mut net = NeuralNetwork::new(&[2, 4, 1]);

// Forward pass — returns output vector
net.feed_forward(&[0.0, 1.0]);

// Backpropagate one sample
net.backpropagate(&[0.0, 1.0], &[1.0], 0.7);

// One training epoch over all samples
net.train_epoch(&data::xor(), 0.7);

// Mean squared error over a dataset
net.mean_squared_error(&data::xor());

// Train until convergence
neuron::train::until_mse(&mut net, &data::xor(), 0.7, 0.001, None);
```

## Extending

### Add a problem

Add a dataset function to `src/data.rs`:

```rust
/// NOR gate: output is 1 only when both inputs are 0.
pub fn nor() -> Vec<Sample> {
    vec![
        (vec![0.0, 0.0], vec![1.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}
```

Create an example binary `examples/nor.rs`:

```rust
use neuron::{data, eval, train, NeuralNetwork};
fn main() {
    let mut net = NeuralNetwork::new(&[2, 1]);
    let epochs = train::until_mse(&mut net, &data::nor(), 0.7, 0.001, None);
    println!("Trained in {epochs} epochs");
    eval::inference(&mut net, &data::nor());
}
```

Then `cargo run --release --example nor`.

### Change network topology

```rust
// 3 inputs → 8 hidden → 4 hidden → 2 outputs
let mut net = NeuralNetwork::new(&[3, 8, 4, 2]);
```

### Add a new activation function

1. Add the function and its derivative to `src/activation_functions.rs`:

```rust
pub fn relu(x: f64) -> f64 { x.max(0.0) }
pub fn relu_derivative(x: f64) -> f64 { if x > 0.0 { 1.0 } else { 0.0 } }
```

2. Call it inside `Neuron::activate` and `Neuron::output`.

### Use from another crate

```toml
[dependencies]
neuron = { git = "https://github.com/inxeoz-garage/neuron" }
```

```rust
use neuron::NeuralNetwork;
use neuron::data;

let mut net = NeuralNetwork::new(&[2, 16, 1]);
net.train_epoch(&data::xor(), 0.5);
```

### Serialize / Deserialize

Networks derive `serde`:

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
