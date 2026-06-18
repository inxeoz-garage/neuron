/// A labeled training sample.
///
/// The first element is the input vector fed into the network.
/// The second element is the expected output vector (the target).
/// During training, the error between the network's prediction and
/// this target drives the weight updates via backpropagation.
pub type Sample = (Vec<f64>, Vec<f64>);

// ── Boolean logic gates ──────────────────────────────────────────

/// AND gate: output is 1 only when **both** inputs are 1.
///
/// AND is *linearly separable* — a single neuron (no hidden layer)
/// can learn it. This makes it the simplest test: if the network
/// can't learn AND, something is fundamentally broken.
pub fn and() -> Vec<Sample> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ]
}

/// OR gate: output is 1 when **at least one** input is 1.
///
/// Like AND, this is linearly separable. A single neuron works.
/// Comparing AND, OR, and XOR teaches you what "linearly separable"
/// really means — draw them on a 2D grid and see which ones a
/// straight line can separate.
pub fn or() -> Vec<Sample> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ]
}

/// XOR gate: output is 1 when **exactly one** input is 1.
///
/// XOR is the classic counter-example: it is **not** linearly
/// separable. No single straight line can separate the four points.
/// This is why a network **must** have a hidden layer to learn XOR —
/// the hidden neurons re-project the inputs into a linearly separable
/// space for the output neuron.
///
/// This is the canonical "hello world" for neural networks.
pub fn xor() -> Vec<Sample> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}

// ── Multi-output ─────────────────────────────────────────────────

/// Half-adder: adds two bits, producing a **sum** and a **carry**.
///
/// ```text
/// (a, b) → (sum, carry)
/// sum   = a XOR b   (1 when inputs differ)
/// carry = a AND b   (1 when both inputs are 1)
/// ```
///
/// This is the first **multi-output** example. The network has two
/// output neurons. Notice that the half-adder **composes** XOR and
/// AND — two problems you already solved separately, now combined
/// into one network. If the network can learn the half-adder, it has
/// learned internal representations that serve both outputs at once.
pub fn half_adder() -> Vec<Sample> {
    vec![
        (vec![0.0, 0.0], vec![0.0, 0.0]),
        (vec![0.0, 1.0], vec![1.0, 0.0]),
        (vec![1.0, 0.0], vec![1.0, 0.0]),
        (vec![1.0, 1.0], vec![0.0, 1.0]),
    ]
}

// ── Continuous, non-linearly-separable ───────────────────────────

/// Circle classification: is the point (x, y) inside the circle of
/// radius 0.5 centered at (0.5, 0.5)?
///
/// This is a **continuous** problem (inputs are real numbers, not
/// booleans) with a **non-linear decision boundary**. A single neuron
/// can only draw straight lines, so it cannot learn a circular boundary.
/// A hidden layer of sigmoid neurons approximates the circle by
/// assembling several "bumps" — each neuron covers one region of the
/// input space.
///
/// A 5×5 grid gives 25 training samples. The network never sees the
/// test points between grid cells, so this also teaches generalization.
///
/// **Why the network plateaus:** A single hidden layer of sigmoid
/// neurons struggles to form a clean closed curve — gradient descent
/// gets stuck in local minima. This is a known limitation of shallow
/// networks. Solutions include: more neurons, more layers (e.g.
/// [2, 8, 8, 1]), or a different activation function (ReLU).
/// The example still converges to MSE < 0.15, proving the network
/// **is** learning — it just can't reach the same precision as the
/// boolean problems.
pub fn circle() -> Vec<Sample> {
    let mut samples = Vec::with_capacity(25);
    for i in 0..5 {
        for j in 0..5 {
            let x = i as f64 / 4.0;
            let y = j as f64 / 4.0;
            let dx = x - 0.5;
            let dy = y - 0.5;
            let inside = if dx * dx + dy * dy < 0.25 {
                1.0
            } else {
                0.0
            };
            samples.push((vec![x, y], vec![inside]));
        }
    }
    samples
}
