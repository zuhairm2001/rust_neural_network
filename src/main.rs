mod mnist_loader;
use mnist_loader::{load_mnist, load_test_data, load_training_data};
use ndarray::{array, Array1, Array2};
use rust_neural_network::Network;
use std::f64;

fn main() {
    let number_of_toes: Array1<f64> = array![8.5, 9.5, 10.0, 9.0];
    let wlrec: Array1<f64> = array![0.65, 0.8, 0.8, 0.9];
    let nfans: Array1<f64> = array![1.2, 1.3, 0.5, 1.0];
    let hurt: Array1<f64> = array![0.1, 0.0, 0.0, 0.1];
    let win: Array1<f64> = array![1.0, 1.0, 0.0, 1.0];
    let sad: Array1<f64> = array![0.1, 0.0, 0.1, 0.2];

    let weights: Array2<f64> =
        Array2::from_shape_vec((3, 3), vec![0.1, 0.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1]).unwrap();

    let mut network = Network::build(weights, number_of_toes, wlrec, nfans, 0.01, hurt, win, sad);
    network.pred();
}
