use ndarray::{arr1, Array1, Array2};
use std::f64;

pub struct Network {
    weights: Array2<f64>,
    input1: Array1<f64>,
    input2: Array1<f64>,
    input3: Array1<f64>,
    alpha: f64,
    output_test1: Array1<f64>,
    output_test2: Array1<f64>,
    output_test3: Array1<f64>,
}

impl Network {
    pub fn build(
        build_weights: Array2<f64>,
        build_input1: Array1<f64>,
        build_input2: Array1<f64>,
        build_input3: Array1<f64>,
        build_alpha: f64,
        build_output1: Array1<f64>,
        build_output2: Array1<f64>,
        build_output3: Array1<f64>,
    ) -> Network {
        let network = Network {
            weights: (build_weights),
            input1: (build_input1),
            input2: (build_input2),
            input3: (build_input3),
            alpha: (build_alpha),
            output_test1: (build_output1),
            output_test2: (build_output2),
            output_test3: (build_output3),
        };

        network
    }

    fn w_sum(input: &Array1<f64>, weights: Array1<f64>) -> f64 {
        assert_eq!(input.len(), weights.len());
        let mut output: f64 = 0.0;

        for i in 0..input.len() {
            output += input[i] * weights[i];
        }
        output
    }

    fn vect_mat_mul(inputs_vec: &Array1<f64>, weights_matrix: Array2<f64>) -> Array1<f64> {
        assert_eq!(inputs_vec.len(), weights_matrix.nrows());
        let mut output: Array1<f64> = arr1(&[0.0, 0.0, 0.0]);

        for i in 0..inputs_vec.len() {
            let weight_vec = weights_matrix.row(i).to_owned();
            output[i] = Self::w_sum(&inputs_vec, weight_vec);
        }
        output
    }

    pub fn pred(&mut self) {
        let inputs_vec = arr1(&[self.input1[0], self.input2[0], self.input3[0]]);
        let prediction = Self::vect_mat_mul(&inputs_vec, self.weights.clone());
        println!("Prediction is ...");
        for i in &prediction {
            println!("{}", i);
        }
        self.learn(&inputs_vec, &prediction);

        let new_prediction = Self::vect_mat_mul(&inputs_vec, self.weights.clone());
        println!("After learning ...");
        for i in &new_prediction {
            println!("{}", i);
        }
    }

    fn outer_prod(vec_a: &Array1<f64>, vec_b: Array1<f64>) -> Array2<f64> {
        let mut out: Array2<f64> = Array2::zeros((vec_a.len(), vec_b.len()));

        for i in 0..vec_a.len() {
            for j in 0..vec_b.len() {
                out[[i, j]] = vec_a[i] * vec_b[j]
            }
        }
        out
    }

    fn learn(&mut self, inputs_vec: &Array1<f64>, pred: &Array1<f64>) {
        let mut error: Array1<f64> = Array1::zeros(3);
        let mut delta: Array1<f64> = Array1::zeros(3);
        let truth: Array1<f64> = arr1(&[
            self.output_test1[0],
            self.output_test2[0],
            self.output_test3[0],
        ]);
        println!("Performing gradient descent");

        for i in 0..truth.len() {
            error[i] = (pred[i] - truth[i]) * (pred[i] - truth[i]);
            delta[i] = pred[i] - truth[i];
        }
        let weight_deltas: Array2<f64> = Self::outer_prod(&inputs_vec, delta);

        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                self.weights[[i, j]] -= self.alpha * weight_deltas[[i, j]];
            }
        }
    }
}
