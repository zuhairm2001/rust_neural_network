use ndarray::{arr1, Array1, Array2};
use std::f64;

pub struct Network {
    weights: Array2<f64>,
    input1: Array1<f64>,
    input2: Array1<f64>,
    input3: Array1<f64>,
}

impl Network {
    pub fn build(
        build_weights: Array2<f64>,
        build_input1: Array1<f64>,
        build_input2: Array1<f64>,
        build_input3: Array1<f64>,
    ) -> Network {
        let network = Network {
            weights: (build_weights),
            input1: (build_input1),
            input2: (build_input2),
            input3: (build_input3),
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

    fn vect_mat_mul(inputs_vec: Array1<f64>, weights_matrix: Array2<f64>) -> Array1<f64> {
        assert_eq!(inputs_vec.len(), weights_matrix.nrows());
        let mut output: Array1<f64> = arr1(&[0.0, 0.0, 0.0]);

        for i in 0..inputs_vec.len() {
            let weight_vec = weights_matrix.row(i).to_owned();
            output[i] = Self::w_sum(&inputs_vec, weight_vec);
        }
        output
    }

    pub fn pred(&self) {
        let inputs_vec = arr1(&[self.input1[0], self.input2[0], self.input3[0]]);
        let prediction = Self::vect_mat_mul(inputs_vec.clone(), self.weights.clone());
        println!("Prediction is ...");
        for i in prediction {
            println!("{}", i);
        }
    }
}
