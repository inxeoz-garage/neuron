use crate::data::Sample;
use crate::neural_net::NeuralNetwork;

/// Train the network until MSE drops below `target_mse`.
///
/// Prints progress every 1000 epochs (every epoch for the first 5).
/// Returns the total number of epochs trained.
///
/// This is a *full-batch* trainer: each epoch processes every sample
/// exactly once. For larger datasets you would switch to mini-batch
/// or stochastic gradient descent, which update weights more frequently
/// and often converge faster.
pub fn until_mse(
    net: &mut NeuralNetwork,
    data: &[Sample],
    learning_rate: f64,
    target_mse: f64,
) -> usize {
    let mut epoch = 0usize;
    loop {
        let mse = net.mean_squared_error(data);
        if epoch % 1000 == 0 || epoch < 5 || mse < target_mse {
            println!("Epoch {epoch:>4}  MSE = {mse:.6}");
        }
        if mse < target_mse {
            return epoch;
        }
        net.train_epoch(data, learning_rate);
        epoch += 1;
    }
}
