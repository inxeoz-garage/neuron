# Future Work

Ideas for extending the project, roughly ordered from small/mechanical to ambitious.

## Training & Optimization

- [ ] **Momentum** — Add momentum term to gradient descent to escape local minima (parity3 would become more reliable).
- [ ] **Learning rate schedule** — Decay LR over epochs instead of fixed 0.7. Helps convergence on sine and parity3.
- [ ] **Multiple restarts** — Train N networks from different seeds, keep the best. Trivial parallelism; makes parity3/circle demo runs more reliable.
- [ ] **Mini-batch / stochastic SGD** — Replace full-batch with mini-batch updates. Currently uses full-batch: average gradient across all samples before stepping.
- [ ] **Early stopping** — Stop when validation MSE stops improving; prevents overfitting on sine.
- [ ] **Regularisation** — L1 or L2 weight decay. Helps circle and sine generalise.

## Architectural

- [ ] **Configurable activation per layer** — Currently hardcoded sigmoid. Allow ReLU, tanh, etc. per layer (e.g. ReLU hidden + sigmoid output).
- [ ] **Dropout** — Add dropout layer type for regularisation.
- [ ] **Softmax output** — Multi-class classification. Currently sigmoid outputs treat each neuron independently.
- [ ] **Cross-entropy loss** — Better for classification than MSE. Add as alternative loss function.
- [ ] **Layer normalisation / batch normalisation** — Speeds up convergence on deeper nets.
- [ ] **Weight initialisation strategies** — Offer Xavier, He, random uniform, zero as options.
- [ ] **Deep network support** — Test with 3+ hidden layers. Current API supports arbitrary depth; never demonstrated.

## Examples & Demos

- [ ] **MNIST digit classifier** — 784→{64,32}→10, softmax + cross-entropy. The canonical "real" demo.
- [ ] **Iris dataset** — 4→{8,4}→3, small classic benchmark, easy to verify.
- [ ] **Binary addition (N-bit)** — Generalise half/full adder to arbitrary N-bit addition with a recurrent or deep net.
- [ ] **Differential equations** — Train a network to approximate a solution to y' = f(x, y). Educational.
- [ ] **Image classification (toy)** — 4×4 binary images → class. Illustrates pixel inputs without real image deps.

## Infrastructure

- [ ] **`cargo bench` benchmarks** — Benchmark forward/backward pass per layer size. Track regressions.
- [ ] **`--seed N` CLI flag** — Reproducible runs across examples.
- [ ] **Serialisation demos** — Save/load trained network with serde (already derived). Add example showing how.
- [ ] **Jupyter / evcxr integration** — Interactive exploration in a notebook via `evcxr_jupyter`.

## Documentation & Pedagogy

- [ ] **Interactive visualisation** — ASCII or gnuplot output of decision boundary for 2-input examples (and, xor, circle). Plot the sine approximation.
- [ ] **Backpropagation deep-dive** — Expand neuron.md with worked example: forward → MSE → gradient → update, step by step.
- [ ] **Exercise suggestions** — Add EXERCISES.md with prompts for students (e.g. "change the circle topology to [2,32,1] and compare", "add ReLU in the hidden layer").
- [ ] **Implementation in other languages** — Side-by-side comparison with Python/numpy version of the same architecture.

## Testing

- [ ] **Property-based tests** — Use `proptest` or `quickcheck` to verify gradient correctness (e.g. numerical gradient approximation matches analytic).
- [ ] **Gradient checking** — Finite-difference test for backprop implementation. Catches subtle bugs.
- [ ] **Convergence property** — On AND/OR/XOR, assert final MSE < 0.001 within N epochs.

## Cross-cutting

- [ ] **no_std support** — Replace `Vec` with fixed-capacity arrays for embedded targets. Minor curiosity; not practical.
- [ ] **SIMD / rayon** — Parallelise feed-forward and backprop over layer neurons. Mostly useless at toy scale.
- [ ] **GPU compute (wgpu / rust-gpu)** — Overkill for this codebase, but interesting as a learning exercise.
