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
    cv_folds: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<&'py PyTuple> {

    let x = data_matrix.as_array().to_owned();
    let (omega_start, tau2, intercepts, selected_lambdas, alpha, all_converged) = init_regression(&x, lambdas, rho, fit_intercept, cv_folds, max_iter, tol);

    if !all_converged {
        let warning_msg = format!("Initial regression did not converge, consider increasing max_iter or decreasing tol.");
        PyErr::warn(py, PyRuntimeWarning::type_object(py), &warning_msg, 1)?;
    }
    
    let (omega, status) = solver::pd3o(&x, tau2, &selected_lambdas, rho, &intercepts, max_iter, tol, alpha, &omega_start);
    
    if !status.converged {
        let warning_msg = format!("Objective did not converge, relative error: {}", status.rel_eps);
        PyErr::warn(py, PyRuntimeWarning::type_object(py), &warning_msg, 1)?;
    }

    let output = PyTuple::new(py, &[omega.into_pyarray(py).to_object(py), intercepts.into_pyarray(py).to_object(py)]);
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
    cv_folds: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
    theta_start: PyReadonlyArray1<'py, f64>,
) -> PyResult<&'py PyTuple> {

    let x = data_matrix.as_array().to_owned();
    let y = response_vector.as_array().to_owned();
    let theta_start = theta_start.as_array().to_owned();
    let mut selected_lambda = lambdas[0];

    if cv_folds > 1 {
        selected_lambda = cv(&x, &y, &lambdas, rho, fit_intercept, cv_folds, alpha, max_iter, tol, &theta_start);
    }

    let (theta, intercept, status) = solver::fista(&x, &y, selected_lambda, rho, fit_intercept, alpha, max_iter, tol, &theta_start);

    if !status.converged {
        let warning_msg = format!("Objective did not converge, relative error: {}", status.rel_eps);
        PyErr::warn(py, PyRuntimeWarning::type_object(py), &warning_msg, 1)?;
    }

    let output = PyTuple::new(py, &[theta.into_pyarray(py).to_object(py), intercept.to_object(py)]);

    Ok(output)
}

pub fn init_regression(
    x: &Array2<f64>, 
    lambdas: Vec<f64>,
    rho: f64, 
    fit_intercept: bool,
    cv_folds: usize,
    max_iter: usize,
    tol: f64,
) -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<f64>, f64, bool) {

    let p = x.ncols();
    let alpha = 1.0 / utils::spectral_norm(&x).powi(2);
    let mut tau2 = Array1::<f64>::zeros(p);
    let mut omega_start = Array2::<f64>::zeros((p, p));
    let mut tau2_max = 0.0; 
    let mut intercepts = Array1::<f64>::zeros(p);
    let theta_start = Array1::<f64>::zeros(p-1);
    let mut all_converged = true;
    let mut selected_lambdas = Vec::<f64>::new();

    if cv_folds <= 1 {
        selected_lambdas = lambdas.clone();
    }

    for j in 0..p {
        let x_j = x.column(j).to_owned();
        let x_not_j = utils::remove_column(&x, j);

        if cv_folds > 1 {
            selected_lambdas.push(cv(&x_not_j, &x_j, &lambdas, rho, fit_intercept, cv_folds, alpha, max_iter, tol, &theta_start));
        }
        
        let (theta_j, intercept_j, status) = solver::fista(&x_not_j, &x_j, selected_lambdas[j], rho, fit_intercept, alpha, max_iter, tol, &theta_start); 
        
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
    
    (omega_start, tau2, intercepts, selected_lambdas, alpha / tau2_max.powi(2), all_converged)
}

pub fn cv(
    x: &Array2<f64>,
    y: &Array1<f64>,
    lambdas: &Vec<f64>, 
    rho: f64, 
    fit_intercept: bool, 
    cv_folds: usize,
    alpha: f64, 
    max_iter: usize, 
    tol: f64,
    theta_start: &Array1<f64>
) -> f64 {

    let mut selected_lambda = 0.0;
    let mut lowest_loss = f64::INFINITY;

    for lambda in lambdas.iter() {
        let mut loss = 0.0;

        for fold in 0..cv_folds {
            let (x_train, x_test, y_train, y_test) = utils::split_data(&x, &y, fold, cv_folds);
            let (theta_j, intercept_j, _status) = solver::fista(&x_train, &y_train, *lambda, rho, fit_intercept, alpha, max_iter, tol, &theta_start);
            
            loss += utils::estimate_variance(&y_test, &x_test, &theta_j, intercept_j, rho);

            if loss < lowest_loss {
                lowest_loss = loss;
                selected_lambda = *lambda;
            }
        }
    }

    selected_lambda
}