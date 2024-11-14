extern crate ndarray as nd;
use super::utils;

use std::vec::Vec;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Eigh, UPLO};
use ndarray_linalg::norm::Norm;
use ndarray_linalg::opnorm::OperationNorm;

pub struct Status {
    pub rel_eps: f64,
    pub converged: bool,
}

pub fn pd3o(
    x: &Array2<f64>, 
    tau2: Array1<f64>, 
    lambdas: &Vec<f64>,
    rho: f64, 
    intercepts: &Array1<f64>,
    max_iter: usize,
    tol: f64,
    alpha: f64,
    omega_start: &Array2<f64>,
) -> (Array2<f64>, Status) {

    let x = x - intercepts.to_owned().insert_axis(Axis(0));
    let mut omega_curr = omega_start.clone();
    let mut omega_next = omega_start.clone();
    let n = x.nrows() as f64;
    let lambdas = Array1::<f64>::from(lambdas.clone());
    let beta = 1.0 / alpha;
    let mut u_curr = Array2::<f64>::eye(x.ncols());
    let mut grad_curr = jpr_grad(&x, &tau2, &omega_curr, rho);
    let mut rel_eps = 0.0;
    let mut converged = false;

    for _iter in 0..max_iter {
        omega_next = proj_psd(&omega_curr - alpha * &u_curr - alpha * &grad_curr);
        
        let grad_next = jpr_grad(&x, &tau2, &omega_next, rho);
        let v_curr = &u_curr + beta * (2.0 * &omega_next - &omega_curr) + alpha * beta * (&grad_curr - &grad_next);
        u_curr = &v_curr - beta * prox_h(&v_curr / beta, &tau2, n * &lambdas / beta);

        rel_eps = rel_eps_pd3o(&omega_curr, &omega_next);

        if rel_eps < tol {
            converged = true;
            break
        }

        omega_curr = omega_next.clone();
        grad_curr = grad_next.clone();
    }

    let status = Status {
        rel_eps: rel_eps,
        converged: converged
    };
    (omega_next, status)
}

pub fn fista(
    x: &Array2<f64>, 
    y: &Array1<f64>, 
    lambda: f64,
    rho: f64, 
    fit_intercept: bool,
    alpha: f64,
    max_iter: usize,
    tol: f64,
    theta_start: &Array1<f64>,
    intercept_start: f64,
) -> (Array1<f64>, f64, Status) {
    
    let n = x.nrows() as f64;
    let mut theta_curr = theta_start.clone();
    let mut theta_old = theta_start.clone();
    let mut intercept_curr = intercept_start;
    let mut intercept_old = intercept_start;

    if fit_intercept {
        intercept_curr = y.iter().sum::<f64>() / n;
        intercept_old = intercept_curr;
    }
    
    let mut loss_grad_curr = loss_grad(&x, &y, &theta_curr, intercept_curr, rho);
    let mut converged = false;
    let mut rel_eps = 0.0;

    for iter in 0..max_iter {
        let k = 1.0 + iter as f64;
        
        theta_curr = &theta_curr - (k - 2.0) / (k + 1.0) * (&theta_curr - &theta_old);
        theta_curr = soft_threshold(&(theta_curr + alpha * loss_grad_curr.dot(x)), n * alpha * lambda);

        if fit_intercept {
            intercept_curr = &intercept_curr - (k - 2.0) / (k + 1.0) * (&intercept_curr - &intercept_old);
            intercept_curr += alpha * loss_grad_curr.iter().sum::<f64>();
        }
        
        loss_grad_curr = loss_grad(&x, &y, &theta_curr, intercept_curr, rho);
        rel_eps = eps_fista(&theta_curr, &theta_old);

        if rel_eps < tol {
            converged = true;
            break
        }

        theta_old = theta_curr.clone();
        intercept_old = intercept_curr;
    }
    
    let status = Status {
        rel_eps: rel_eps,
        converged: converged
    };

    (theta_curr, intercept_curr, status)
}

fn loss_grad(x: &Array2<f64>, y: &Array1<f64>, theta: &Array1<f64>, intercept: f64, rho: f64) -> Array1<f64> {
    let mut grad = y - intercept - &theta.dot(&x.t());

    if rho > 0.0 {
        grad = grad.map(|x| if x.abs() <= rho { *x } else { rho * x.signum() });
    }

    grad
}

fn soft_threshold(theta: &Array1<f64>, lambda: f64) -> Array1<f64> {
    let mut theta_new = Array1::<f64>::zeros(theta.len());

    for j in 0..theta.len() {
        theta_new[j] = if theta[j] > lambda {
            theta[j] - lambda
        } else if theta[j] < -lambda {
            theta[j] + lambda
        } else {
            0.0
        };
    }
    
    theta_new
}

fn huber_grad(x: &Array2<f64>, residuals: &Array1<f64>, rho: f64) -> Array1<f64> {  
    let huber = residuals.map(|x| if x.abs() <= rho { *x } else { rho * x.signum() });    
    huber.dot(x)
}

fn proj_psd(matrix: Array2<f64>) -> Array2<f64> {
    let sym_part = (&matrix + &matrix.t()) / 2.0;

    let eig = match sym_part.eigh(UPLO::Lower) {
        Ok(eig) => eig,
        Err(_) => panic!("Failed to compute eigenvalue decomposition."),
    };

    let (eigvals, eigvecs) = eig;
    let eigvals_pos = eigvals.map(|x| x.max(0.0));
    let proj = eigvecs.dot(&Array2::<f64>::from_diag(&eigvals_pos)).dot(&eigvecs.t());

    proj
}

fn jpr_grad(
    x: &Array2<f64>,
    tau2: &Array1<f64>,
    omega: &Array2<f64>,
    rho: f64,
) -> Array2<f64> {

    let p = x.ncols();
    let mut grad = Array2::<f64>::zeros((p, p));

    for j in 0..p {
        let omega_not_j_j = utils::offdiag_column(omega, j);
        let x_j = utils::get_column(&x, j);
        let x_not_j = utils::remove_column(&x, j);
        let residual = x_j + tau2[j] * &omega_not_j_j.dot(&x_not_j.t());

        if rho > 0.0 {
            utils::set_column(&mut grad, j, &huber_grad(&(tau2[j] * x_not_j), &residual, rho));
        } else {
            utils::set_column(&mut grad, j, &(tau2[j] * &residual.dot(&x_not_j)));
        }
    }
    
    grad
}

fn prox_h(v: Array2<f64>, tau2: &Array1<f64>, lambdas: Array1<f64>) -> Array2<f64> {
    let p = v.ncols();
    let mut prox = Array2::<f64>::zeros((p, p));

    for j in 0..p {
        let v_j = v.column(j).into_owned();
        for k in 0..p {
            if k == j {
                prox[[j, j]] = 1.0 / tau2[j];
                continue;
            };
            
            let threshold = lambdas[j] * tau2[j];
            prox[[k, j]] = if v_j[k] > threshold {
                v_j[k] - threshold
            } else if v_j[k] < -threshold {
                v_j[k] + threshold
            } else {
                0.0
            };
            
        }
    }

    prox
}

fn eps_fista(theta_curr: &Array1<f64>, theta_old: &Array1<f64>) -> f64 {
    let p = theta_curr.len() as f64;
    (theta_curr - theta_old).norm_l2() / p.sqrt()
}

fn rel_eps_pd3o(omega_curr: &Array2<f64>, omega_next: &Array2<f64>) -> f64 {
    let error = match (omega_next - omega_curr).opnorm_fro() {
        Ok(error) => error, 
        Err(_) => panic!("Failed to compute the Frobenius norm")
    };
    let norm = match omega_curr.opnorm_fro() {
        Ok(norm) => norm, 
        Err(_) => panic!("Failed to compute the Frobenius norm")
    };

    error / norm
}

