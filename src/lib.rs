extern crate ndarray as nd;
extern crate numpy as np;
mod solver;

use pyo3::prelude::*;
use pyo3::PyTypeInfo;
use pyo3::types::PyTuple;
use pyo3::exceptions::PyRuntimeWarning;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};

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
    max_iter: usize,
    tol: f64,
) -> PyResult<&'py PyTuple> {

    let x = data_matrix.as_array().to_owned();
    let (omega_start, tau2, intercepts, alpha, all_converged) = solver::fit_pr(&x, &lambdas, rho, fit_intercept, max_iter, tol);
    if !all_converged {
        let warning_msg = format!("Initial regression did not converge, consider increasing max_iter or decreasing tol.");
        PyErr::warn(py, PyRuntimeWarning::type_object(py), &warning_msg, 1)?;
    }
    let (omega, status) = solver::pd3o(&x, tau2, &lambdas, rho, &intercepts, max_iter, tol, alpha, &omega_start);
    
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
    lambda: f64,
    rho: f64,
    fit_intercept: bool,
    alpha: f64,
    max_iter: usize,
    tol: f64,
    theta_start: PyReadonlyArray1<'py, f64>,
) -> PyResult<&'py PyTuple> {

    let x = data_matrix.as_array().to_owned();
    let y = response_vector.as_array().to_owned();
    let theta_start = theta_start.as_array().to_owned();

    let (theta, intercept, status) = solver::fista(&x, &y, lambda, rho, fit_intercept, alpha, max_iter, tol, &theta_start);

    if !status.converged {
        let warning_msg = format!("Objective did not converge, relative error: {}", status.rel_eps);
        PyErr::warn(py, PyRuntimeWarning::type_object(py), &warning_msg, 1)?;
    }

    let output = PyTuple::new(py, &[theta.into_pyarray(py).to_object(py), intercept.to_object(py)]);

    Ok(output)
}
