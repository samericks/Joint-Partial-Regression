extern crate ndarray as nd;
extern crate numpy as np;
mod solver;
mod utils;

use pyo3::prelude::*;
use pyo3::PyTypeInfo;
use pyo3::types::PyTuple;
use pyo3::exceptions::PyRuntimeWarning;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};

use std::vec::Vec;
use ndarray::{Array1, Array2};

#[pymodule]
fn graph_jpr(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_jpr, m)?)?;
    m.add_function(wrap_pyfunction!(fit_sparse_huber, m)?)?;
    Ok(())
}

#[pyfunction]
fn fit_jpr<'py>(
    py: Python<'py>,
    data_matrix: PyReadonlyArray2<'py, f64>,
    lambdas: Vec<f64>,
    rho: f64,
    fit_intercept: bool,
    n_lambdas: usize,
    cv_folds: usize,
    criterion: &str,
    max_iter: usize,
    tol: f64,
) -> PyResult<&'py PyTuple> {

    let x = data_matrix.as_array().to_owned();
    let (omega_start, tau2, intercepts, selected_lambdas, lambdas_grid, alpha, all_converged) = init_regression(&x, lambdas, rho, fit_intercept, n_lambdas, cv_folds, criterion, max_iter, tol);

    if !all_converged {
        let warning_msg = format!("Initial regression did not converge, consider increasing max_iter or decreasing tol.");
        PyErr::warn(py, PyRuntimeWarning::type_object(py), &warning_msg, 1)?;
    }
    
    let (omega, status) = solver::pd3o(&x, tau2, &selected_lambdas, rho, &intercepts, max_iter, tol, alpha, &omega_start);
    
    if !status.converged {
        let warning_msg = format!("Objective did not converge, relative error: {}", status.rel_eps);
        PyErr::warn(py, PyRuntimeWarning::type_object(py), &warning_msg, 1)?;
    }

    let output = PyTuple::new(py, &[omega.into_pyarray(py).to_object(py), intercepts.into_pyarray(py).to_object(py), selected_lambdas.into_pyarray(py).to_object(py), lambdas_grid.to_object(py)]);
    Ok(output)
}

#[pyfunction]
fn fit_sparse_huber<'py>(
    py: Python<'py>,
    data_matrix: PyReadonlyArray2<'py, f64>,
    response_vector: PyReadonlyArray1<'py, f64>,
    lambdas: Vec<f64>,
    rho: f64,
    fit_intercept: bool,
    n_lambdas: usize,
    cv_folds: usize,
    criterion: &str,
    alpha: f64,
    max_iter: usize,
    tol: f64,
    theta_start: PyReadonlyArray1<'py, f64>,
    intercept_start: f64,
) -> PyResult<&'py PyTuple> {

    let x = data_matrix.as_array().to_owned();
    let y = response_vector.as_array().to_owned();
    let theta;
    let intercept;
    let theta_start = theta_start.as_array().to_owned();
    let selected_lambda: f64;
    let lambda_grid = Vec::<f64>::new();
    let use_model_selection = cv_folds > 1 || criterion == "AIC" || criterion == "BIC";
    let status;

    if use_model_selection {
        let lambda_grid;

        if n_lambdas > 0 {
            lambda_grid = compute_grid(&x, &y, n_lambdas, 1e-3);
        } else {
            lambda_grid = lambdas.clone();
        }

        (theta, intercept, selected_lambda, status) = model_selection(&x, &y, &lambda_grid, rho, fit_intercept, cv_folds, criterion, alpha, max_iter, tol);
    } else {
        selected_lambda = lambdas[0];
        (theta, intercept, status) = solver::fista(&x, &y, selected_lambda, rho, fit_intercept, alpha, max_iter, tol, &theta_start, intercept_start); 
    }

    // if !use_model_selection {
    //     selected_lambda = lambdas[0];
    // } else {

    //     if n_lambdas > 0 {
    //         lambda_grid = compute_grid(&x, &y, n_lambdas, 1e-3);
    //     } else {
    //         lambda_grid = lambdas.clone();
    //     }

    //     if cv_folds > 1 {
    //         selected_lambda = cv(&x, &y, &lambda_grid, rho, fit_intercept, cv_folds, alpha, max_iter, tol, &theta_start);
    //     } else {
    //         selected_lambda = ic(&x, &y, &lambda_grid, rho, fit_intercept, criterion, alpha, max_iter, tol, &theta_start);
    //     }
    // }
    
    if !status.converged {
        let warning_msg = format!("Objective did not converge, relative error: {}", status.rel_eps);
        PyErr::warn(py, PyRuntimeWarning::type_object(py), &warning_msg, 1)?;
    }

    let output = PyTuple::new(py, &[theta.into_pyarray(py).to_object(py), intercept.to_object(py), selected_lambda.to_object(py), lambda_grid.to_object(py)]);

    Ok(output)
}

pub fn init_regression(
    x: &Array2<f64>, 
    lambdas: Vec<f64>,
    rho: f64, 
    fit_intercept: bool,
    n_lambdas: usize,
    cv_folds: usize,
    criterion: &str,
    max_iter: usize,
    tol: f64,
) -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<f64>, Vec<Vec<f64>>, f64, bool) {

    let p = x.ncols();
    let alpha = 1.0 / utils::spectral_norm(&x).powi(2);
    let mut tau2 = Array1::<f64>::zeros(p);
    let mut omega_start = Array2::<f64>::zeros((p, p));
    let mut tau2_max = 0.0; 
    let mut intercepts = Array1::<f64>::zeros(p);
    let mut all_converged = true;
    let mut selected_lambdas = Vec::<f64>::new();
    let use_model_selection = cv_folds > 1 || criterion == "AIC" || criterion == "BIC";
    let mut lambda_grid;
    let mut lambdas_grid = Vec::<Vec<f64>>::new();

    if !use_model_selection {
        selected_lambdas = lambdas.clone();
    }

    for j in 0..p {
        let x_j = x.column(j).to_owned();
        let x_not_j = utils::remove_column(&x, j);
        let theta_j: Array1<f64>; 
        let intercept_j: f64;
        let lambda_j: f64;
        let status: solver::Status;    

        if use_model_selection {
            if n_lambdas > 0 {
                lambda_grid = compute_grid(&x_not_j, &x_j, n_lambdas, 1e-3);
                lambdas_grid.push(lambda_grid.clone());
            } else {
                lambda_grid = lambdas.clone();
            }

            (theta_j, intercept_j, lambda_j, status) = model_selection(&x_not_j, &x_j, &lambda_grid, rho, fit_intercept, cv_folds, criterion, alpha, max_iter, tol);
            selected_lambdas.push(lambda_j);
        } else {
            (theta_j, intercept_j, status) = solver::fista(&x_not_j, &x_j, lambdas[j], rho, fit_intercept, alpha, max_iter, tol, &Array1::<f64>::zeros(p-1), 0.0); 
        }
        
        if !status.converged {
            all_converged = false;
        }

        intercepts[j] = intercept_j;
        tau2[j] = utils::estimate_variance(&x_j, &x_not_j, &theta_j, intercept_j, rho);

        if tau2[j] > tau2_max {
            tau2_max = tau2[j];
        }

        utils::set_column(&mut omega_start, j, &(-theta_j / tau2[j]));
        omega_start[[j, j]] = 1.0 / tau2[j];
    }
    
    (omega_start, tau2, intercepts, selected_lambdas, lambdas_grid, alpha / tau2_max.powi(2), all_converged)
}

fn model_selection(
    x: &Array2<f64>,
    y: &Array1<f64>,
    lambdas: &Vec<f64>, 
    rho: f64, 
    fit_intercept: bool, 
    cv_folds: usize,
    criterion: &str,
    alpha: f64, 
    max_iter: usize, 
    tol: f64,
) -> (Array1<f64>, f64, f64, solver::Status) {
    
    let mut selected_lambda = 0.0;
    let mut lowest_score = f64::INFINITY;
    let mut selected_theta = Array1::<f64>::zeros(x.ncols());
    let mut selected_intercept = 0.0;
    let mut theta_start = Array1::<f64>::zeros(x.ncols());
    let mut intercept_start = 0.0;
    let mut status = solver::Status { rel_eps: 0.0, converged: true };
    let mut _status;

    for lambda in lambdas.iter() {
        let score;
        let theta;
        let intercept;

        if criterion == "AIC" || criterion == "BIC" {
            (theta, intercept, _status) = solver::fista(&x, &y, *lambda, rho, fit_intercept, alpha, max_iter, tol, &theta_start, intercept_start);
            score = utils::ic(&y, &x, &theta, intercept, criterion);
        } else {
            (theta, intercept, score, _status) = cv(&x, &y, lambda, rho, fit_intercept, cv_folds, alpha, max_iter, tol, &theta_start, intercept_start);
        }

        if !(_status.converged) {
            status.converged = false;
            status.rel_eps = _status.rel_eps;
        }

        theta_start = theta.clone();
        intercept_start = intercept;

        if score < lowest_score {
            lowest_score = score;
            selected_lambda = *lambda;
            selected_theta = theta;
            selected_intercept = intercept;
        }
    }

    if !(criterion == "AIC" || criterion == "BIC") {
        (selected_theta, selected_intercept, _status) = solver::fista(&x, &y, selected_lambda, rho, fit_intercept, alpha, max_iter, tol, &selected_theta, selected_intercept);

        if !(_status.converged) {
            status.converged = false;
            status.rel_eps = _status.rel_eps;
        }
    }

    (selected_theta, selected_intercept, selected_lambda, status)
}


fn cv(
    x: &Array2<f64>,
    y: &Array1<f64>,
    lambda: &f64, 
    rho: f64, 
    fit_intercept: bool, 
    cv_folds: usize,
    alpha: f64, 
    max_iter: usize, 
    tol: f64,
    theta_start: &Array1<f64>,
    intercept_start: f64
) -> (Array1<f64>, f64, f64, solver::Status) {

    let mut theta = Array1::<f64>::zeros(x.ncols());
    let mut intercept= 0.0;
    let mut loss= 0.0;
    let mut status= solver::Status { rel_eps: 0.0, converged: true };

    for fold in 0..cv_folds {
        let _status;
        let (x_train, x_test, y_train, y_test) = utils::split_data(&x, &y, fold, cv_folds);
        (theta, intercept, _status) = solver::fista(&x_train, &y_train, *lambda, rho, fit_intercept, alpha, max_iter, tol, &theta_start, intercept_start);
        
        if !(_status.converged) {
            status.converged = false;
            status.rel_eps = _status.rel_eps;
        }

        loss += utils::mse(&y_test - intercept - &theta.dot(&x_test.t()));
    }

    (theta, intercept, loss, status)
}



// fn ic(
//     x: &Array2<f64>,
//     y: &Array1<f64>,
//     lambdas: &Vec<f64>, 
//     rho: f64, 
//     fit_intercept: bool, 
//     criterion: &str,
//     alpha: f64, 
//     max_iter: usize, 
//     tol: f64,
//     theta_start: &Array1<f64>
// ) -> f64 {
    
//     let mut selected_lambda = 0.0;
//     let mut lowest_score = f64::INFINITY;

//     for lambda in lambdas.iter() {
//         let score;

//         let (theta_j, intercept_j, _status) = solver::fista(&x, &y, *lambda, rho, fit_intercept, alpha, max_iter, tol, &theta_start);
        
//         if criterion == "AIC" {
//             score = utils::aic(&y, &x, &theta_j, intercept_j);
//         } else {
//             score = utils::bic(&y, &x, &theta_j, intercept_j);
//         }

//         if score < lowest_score {
//             lowest_score = score;
//             selected_lambda = *lambda;
//         }
//     }

//     selected_lambda
// }

fn compute_grid(x: &Array2<f64>, y: &Array1<f64>, n_lambdas: usize, eps: f64) -> Vec<f64> {
    let n = x.nrows() as f64;
    let lambda_max = ((y.dot(x)).iter().map(|&xy| xy * xy).sum::<f64>()).sqrt() / n;

    if lambda_max <= f64::EPSILON {
        return vec![f64::EPSILON; n_lambdas];
    }

    let lambda_min = lambda_max * eps;
    utils::geomspace(lambda_min, lambda_max, n_lambdas)
}


