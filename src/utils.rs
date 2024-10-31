extern crate ndarray as nd;

use std::vec::Vec;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray::stack;
use ndarray_linalg::SVD;

pub fn offdiag_column(matrix: &Array2<f64>, j: usize) -> Array1<f64> {
    let mut column: Array1<f64> = matrix.column(j).to_owned();
    
    if j < column.len() {
        column.remove_index(Axis(0), j);
    } else {
        panic!("Index out of bounds");
    }
    
    column
}

pub fn set_column(matrix: &mut Array2<f64>, j: usize, column: &Array1<f64>) {
    for k in 0..matrix.nrows() {
        if k == j {
            matrix[[k, j]] = 0.0;
            continue;
        } else if k < j {
            matrix[[k, j]] = column[k];
        } else {
            matrix[[k, j]] = column[k - 1];
        }
    }
}

pub fn spectral_norm(matrix: &Array2<f64>) -> f64 {
    let svd = match matrix.svd(false, false) {
        Ok(svd) => svd,
        Err(_) => panic!("Failed to compute the singular value decomposition."),
    };
    let (_, s, _) = svd;
    s[0]
}

pub fn estimate_variance(y: &Array1<f64>, x: &Array2<f64>, theta: &Array1<f64>, intercept: f64, rho: f64) -> f64 {
    let mut residuals = y - intercept - theta.dot(&x.t());
    if rho > 0.0 {
        residuals = residuals.iter().cloned().filter(|&x| x.abs() < rho).collect();
    }
    residuals.iter().map(|x| x.powi(2)).sum::<f64>() / (residuals.len() as f64)
}

pub fn remove_column(matrix: &Array2<f64>, j: usize) -> Array2<f64> {
    let p = matrix.ncols();
    let mut columns = Vec::<ArrayView1<f64>>::new();
    
    for k in 0..p {
        if k == j {
            continue;
        }
        columns.push(matrix.column(k));
    }

    stack(Axis(1), &columns).unwrap()
}

pub fn get_column(matrix: &Array2<f64>, j: usize) -> Array1<f64> {
    let n = matrix.nrows();
    let mut vector = Array1::<f64>::zeros(n);
    for i in 0..n {
        vector[i] = matrix[[i, j]];
    }
    vector
}

pub fn split_data(data: &Array2<f64>, response: &Array1<f64>, fold: usize, cv_folds: usize) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let n = data.nrows();
    let fold_size = n / cv_folds;
    let start = fold * fold_size;
    let end = if fold == cv_folds - 1 { n } else { start + fold_size };
    let mut train_data = Vec::<ArrayView1<f64>>::new();
    let mut test_data = Vec::<ArrayView1<f64>>::new();
    let mut train_response = Vec::<f64>::new();
    let mut test_response = Vec::<f64>::new();

    for i in 0..n {
        if i >= start && i < end {
            test_data.push(data.row(i));
            test_response.push(response[i]);
        } else {
            train_data.push(data.row(i));
            train_response.push(response[i]);
        }
    }

    (stack(Axis(0), &train_data).unwrap(), stack(Axis(0), &test_data).unwrap(), Array1::from(train_response), Array1::from(test_response))
}