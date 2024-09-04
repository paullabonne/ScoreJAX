from jax import lax, grad, hessian, jit, random
import jax.numpy as jnp
from jax.scipy.special import gammaln
from functools import partial
import numpy as np
from scipy.optimize import minimize


def K_fun(eta):
    """Helper for the Skew t density."""

    A = gammaln((eta + 1) / eta * 0.5)
    B = -jnp.log(jnp.pi / eta) / 2
    C = -gammaln(1 / (2 * eta))
    K = A + B + C
    return K


def sign(x):
    """Custom sign function where 0 is treated as positive."""
    return jnp.where(x < 0, -1, 1)


def log_f(v, params):
    """Log likelihood of the skew t

    Args:
        v: Vector of prediction errors.
        params: Matrix of parameters.

    Returns:
        Vector of log likelihood
    """

    scale = params[:, 0]
    shape = params[:, 1]
    eta = 1 / params[:, 2]

    log_f = (
        K_fun(eta)
        - 0.5 * jnp.log(scale**2)
        - (1 + eta)
        / (2 * eta)
        * jnp.log(1 + eta * v**2 / (((1 - sign(v) * shape) * scale) ** 2))
    )

    # handle nan or inf values
    log_f = jnp.nan_to_num(log_f, nan=-10.0, posinf=-10.0, neginf=-10.0)

    return log_f


@partial(jit, static_argnums=(2,))
def score(v, params, partial):
    """Score function of the skew t likelihood.

    Args:
        v: Vector of prediction errors.
        params: Matrix of parameters.
        partial: String indicating the parameter with which to differentiate.

    Returns:
        Vector of scores.
    """

    scale = params[:, 0]
    shape = params[:, 1]
    eta = 1 / params[:, 2]

    zeta = v / scale
    w = (1 + eta) / ((1 - sign(v) * shape) ** 2 + eta * zeta**2)

    if partial == "loc":
        score = 1 / scale * w * zeta
    elif partial == "scale":
        score = 1 / (2 * scale**2) * (w * zeta**2 - 1)
    elif partial == "shape":
        score = -sign(v) / (1 - sign(v) * shape) * w * zeta**2

    score = jnp.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score


def build_model(y):
    """ Build the matrices used in the score driven recursion.

    Args:
        y: Matrix of observations.

    Returns:
        Tupple of model components.
    """
    
    N, m = y.shape
    p = m + 1

    identity_block = jnp.repeat(jnp.eye(m)[:, :, None], 3, axis=2)

    # array of observation/link matrices
    Z = jnp.zeros((m, p, 3))
    Z = Z.at[:, 0:m, :].set(identity_block)

    # array of transition matrices
    T = jnp.zeros((p, p, 3))
    T = T.at[0:m, 0:m, :].set(identity_block)

    # array of gains
    K = jnp.zeros((p, p, 3))
    K = K.at[0:m, 0:m, :].set(identity_block)

    # array of states
    a = jnp.zeros((p, N + 1, 3))
    
    # Where observations are missing 
    fin = jnp.isfinite(y).astype(jnp.float32)

    model = (y, fin, a, Z, T, K)

    return model


def initialise(pars, model):
    """ Initialise the model using a set of parameters.

    Args:
        pars: Vector of parameters to use of initialisation.
        model: Tupple of model components.
        
    Returns:
        List of initialised components.
    """
    
    _, _, _, Z, T, K = model

    m, p, _ = Z.shape

    n_pars = 0

    a_init = jnp.zeros((p, 3))

    for i in range(3):
        # initial state
        a_init = a_init.at[0:m, i].set(pars[n_pars : n_pars + m] / 1e1)
        n_pars += m

        # observation matrices
        Z = Z.at[0, m, i].set(1.0)  # factor is one for the first series
        Z = Z.at[1:, m, i].set(pars[n_pars : (n_pars + m - 1)])
        n_pars += m - 1

        # gain matrix
        K = K.at[0 : m + 1, 0 : m + 1, i].set(
            jnp.diag(jnp.exp(pars[n_pars : n_pars + m + 1]) / 1e2)
        )
        n_pars += m + 1

        # transition matrix
        T = T.at[m, m, i].set(jnp.tanh(pars[n_pars]) * 0.95)
        n_pars += 1

    # nu (tail parameter)
    nu = jnp.exp(pars[n_pars : n_pars + m]) + 2.0
    n_pars += m

    return a_init, Z, T, K, nu, n_pars


def filter_step(model_t):
    """ Score driven recursion from t and t+1

    Args:
        model_t: Tupple of model components at t.
        
    Returns:
        List of model components at t+1.
    """
    
    y_t, fin_t, a_loc_t, a_scale_t, a_shape_t, Z, T, K, nu = model_t

    m = y_t.shape[0]

    params = jnp.zeros((m, 3))
    params = params.at[:, 0].set(jnp.exp(Z[:, :, 1] @ a_scale_t))
    params = params.at[:, 1].set(jnp.tanh(Z[:, :, 2] @ a_shape_t))
    params = params.at[:, 2].set(nu)

    # prediction error
    v = y_t - Z[:, :, 0] @ a_loc_t
    v_no_nan = jnp.nan_to_num(v, nan=0)

    # computing the scaled score
    score_loc = score(v_no_nan, params, "loc")
    score_scale = score(v_no_nan, params, "scale")
    score_shape = score(v_no_nan, params, "shape")

    # filtering step
    a_loc_tt = a_loc_t + K[:, :, 0] @ Z[:, :, 0].T @ score_loc
    a_scale_tt = a_scale_t + K[:, :, 1] @ Z[:, :, 1].T @ score_scale
    a_shape_tt = a_shape_t + K[:, :, 2] @ Z[:, :, 2].T @ score_shape

    # prediction step
    a_loc_t1 = T[:, :, 0] @ a_loc_tt
    a_scale_t1 = T[:, :, 1] @ a_scale_tt
    a_shape_t1 = T[:, :, 2] @ a_shape_tt

    # compute loglik
    ll_t = log_f(v, params) * fin_t

    return a_loc_t1, a_scale_t1, a_shape_t1, ll_t


def step_wrapper(carry, xs):
    """ Helper for lax.scan. """
    
    a_loc_t, a_scale_t, a_shape_t, Z, T, K, nu, ll_sum = carry
    y_t, fin_t = xs

    model_t = (y_t, fin_t, a_loc_t, a_scale_t, a_shape_t, Z, T, K, nu)
    a_loc_t1, a_scale_t1, a_shape_t1, ll_t = filter_step(model_t)

    return (a_loc_t1, a_scale_t1, a_shape_t1, Z, T, K, nu, ll_sum), (
        ll_t,
        a_loc_t1,
        a_scale_t1,
        a_shape_t1,
    )


def recursion(y, fin, a_init, Z, T, K, nu):
    """ Runs the score driven recursion.

    Args:
        Observations and initialised model components.
        
    Returns:
        Log likelihood and states.
    """
    
    initial_carry = (a_init[:, 0], a_init[:, 1], a_init[:, 2], Z, T, K, nu, 0.0)

    # Inputs that are iterated over the first dimension
    inputs = y, fin

    # Execute scan
    (_, _, _, _, _, _, _, _), (loglik, a_loc, a_scale, a_shape) = lax.scan(
        step_wrapper, initial_carry, inputs
    )

    return (loglik, a_loc, a_scale, a_shape)


@partial(jit, static_argnums=(1, 2))
def filter(pars, model, mle):
    """ Initialise and run the score driven model.

    Args:
        pars: Parameters used for initialisation / parametrisation.
        model: model components.
        mle: Boolean indicating wether to return only the negative log likelihood or all model components.
        
    Returns:
        Model after recursion execution or negative log likelihood.
    """
    
    y, fin, _, Z, T, K = model

    # initialise the model given the parameters
    a_init, Z, T, K, nu, _ = initialise(pars, model)

    # run the score driven recursion
    filter_result = recursion(y, fin, a_init, Z, T, K, nu)

    if mle:
        result = -jnp.sum(filter_result[0])
    else:
        result = (
            filter_result[1],
            filter_result[2],
            filter_result[3],
            Z,
            T,
            K,
            nu,
            filter_result[0],
        )

    return result


@jit
def log_likelihood(pars, model):
    """ Helper for MLE, returns the loglikehood"""
    
    loglik = filter(pars, model, mle=True)
    return loglik


@jit
def sd_filter(pars, model):
    """ Helper, returns the model after intialisation and recursion"""

    filter_results = filter(pars, model, mle=False)
    return filter_results


# code for maximum likelihood estimation with BFGS
# gradient
ll_grad = jit(grad(log_likelihood))

# hessian
ll_hessian = jit(hessian(log_likelihood))


def ll_scipy(pars, *args):
    return log_likelihood(pars, args).item()


def ll_grad_scipy(pars, *args):
    return np.array(ll_grad(pars, args))


def mle(model, iter, pertu, key, init_par=None, printing=False):
    """ Maximum likelihood estimation

    Args:
        model: model components.
        iter: (int) number of iterations.
        pertu: (real scalar) noise to shock parameters during optimisation.
        init_par: (optional) initial vector of parameters.
        printing: String, show intermdiate results.
        
    Returns:
        Mainly function and parameters at the maximum.
    """
    
    # count the number of parameters to estimate
    n_pars = initialise(jnp.zeros(10000), model)[5]

    # initialisation
    pars_init = np.random.normal(n_pars, pertu, size=n_pars)

    if init_par is None:
        key, subkey = random.split(key)
        pars_init = jnp.zeros(n_pars) + random.normal(subkey, shape=(n_pars,)) * pertu
    else:
        pars_init = init_par[0:n_pars]

    result = minimize(
        fun=ll_scipy, x0=pars_init, jac=ll_grad_scipy, method="BFGS", args=model
    )

    min_mle = result.fun
    par_mle = result.x
    result_min = result

    for i in range(iter):
        key, subkey = random.split(key)
        pars_init = par_mle + random.normal(subkey, shape=(n_pars,)) * pertu
        
        # Use scipy.optimize.minimize with the BFGS method
        result = minimize(
            fun=ll_scipy, x0=pars_init, jac=ll_grad_scipy, method="BFGS", args=model
        )

        if result.fun < min_mle:
            result_min = result
            min_mle = result.fun
            par_mle = result.x

            if printing:
                print("New MLE at ", min_mle, "; iter", i)

    return result_min


import warnings


def simulation(nb_iter, model, mle, var):
    """ Simulate the model around the ML parameters.
    Runs the score driven recursion for each set drawn parameters.

    Args:
        nb_iter: (int) Number of draws.
        model: Estimated model.
        mle: Vector of estimated MLE parameters.
        init_par: (optional) initial vector of parameters.
        var: Estimated variance covariance matrix of the parameters.
        
    Returns: Array of simulated states.
        
    """
    
    y, _, _, _, _, _ = model

    N, m = y.shape

    fac = np.zeros((N, nb_iter, 3))

    for i in range(nb_iter):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pars = np.random.multivariate_normal(mle, var, size=1)[0]

        # run the filter
        filter_i = sd_filter(pars, model)
        
        # retreive states
        a_loc = filter_i[0]
        a_scale = filter_i[1]
        a_shape = filter_i[2]

        # store states
        fac[:, i, 0] = a_loc[:, m]
        fac[:, i, 1] = a_scale[:, m]
        fac[:, i, 2] = a_shape[:, m]

    return fac
