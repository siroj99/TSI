import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import gudhi.point_cloud.timedelay as td
from gudhi.weighted_rips_complex import WeightedRipsComplex
from tqdm.auto import tqdm
import seaborn as sns
import gudhi.representations as gr
import pandas as pd
from copy import deepcopy
import scipy
import scipy.integrate

from .general_functions import remove_inf, filtration_from_points , top_summary_functions

def persistence_timedelay(price, dim, delay, skip, filtration_type = "alpha", max_edge_length=None, add_zeros = False, max_dim=2, dtm=False, dtm_m=0.03):
    if add_zeros:
        price = np.append(price, np.zeros((dim - 1) * delay))
    if max_edge_length is None and filtration_type == "rips":
        max_edge_length = np.max(price) - np.min(price)
    
    points = td.TimeDelayEmbedding(dim=dim, delay=delay, skip=skip)(price)
    return filtration_from_points(points, filtration_type=filtration_type, max_edge_length=max_edge_length, max_dim=max_dim, dtm=dtm, dtm_m=dtm_m)

def persistence_timesublevel(price, max_dimension=0):
    D = np.tril(np.full((price.shape[0], price.shape[0]), np.inf), -2)
    return WeightedRipsComplex(distance_matrix=D+D.T, weights=price/2).create_simplex_tree(max_dimension=max_dimension).persistence()

def brownian_motion(mu, sigma, dt):
    # m = (mu - 0.5 * sigma**2)
    # s = sigma
    logret = np.random.normal(mu, sigma * np.sqrt(dt), size=int(1/dt))
    price = logret.cumsum()
    return price

def geometric_brownian_motion(mu, sigma, dt, n_steps=None):
    if n_steps is None:
        n_steps = int(1/dt)
    m = (mu - 0.5 * sigma**2)
    s = sigma
    logret = np.random.normal(m * dt, s * np.sqrt(dt), size=n_steps)
    price = np.exp(logret.cumsum())
    return price

def estimate_gbm_params(prices, dt):
    # Log returns
    log_returns = np.log(prices[1:] / prices[:-1])

    # Mean and std of returns
    mean_r = np.mean(log_returns)
    std_r = np.std(log_returns, ddof=1)

    # Volatility (annualized)
    sigma_annual = std_r / np.sqrt(dt)

    # Drift (annualized, GBM form)
    mu_annual = mean_r / dt + 0.5 * sigma_annual**2

    return mu_annual, sigma_annual

bm_summary_functions = deepcopy(top_summary_functions)
bm_summary_functions.bm_functions={
    "mu_estimate": lambda timeframe, dt: estimate_gbm_params(timeframe, dt)[0],
    "sigma_estimate": lambda timeframe, dt: estimate_gbm_params(timeframe, dt)[1]
}
bm_summary_functions.function_names = top_summary_functions.function_names + list(bm_summary_functions.bm_functions.keys())


def summary_curve(price, tf_size, dt,  overlapping_window = True, summary_functions = bm_summary_functions, persistence_func = lambda price: persistence_timedelay(price, 3, 1, 1)):
    """
    methods: dict of method name to tuple (use_persistence: Bool, function)
    """
    if not overlapping_window:
        tf_skip = tf_size
    else:        
        tf_skip = 1
    timeframes = td.TimeDelayEmbedding(dim=tf_size, delay=1, skip=tf_skip)(price)
    summary_curve = {method: [None] * (tf_size-tf_skip) for method in summary_functions.function_names}
    for timeframe in timeframes:
        persistence = persistence_func(timeframe)
        func_evals = summary_functions(persistence, price=timeframe, dt=dt)
        for method_name in func_evals.keys():
            # summary_curve[method_name].append(func_evals[method_name])
            summary_curve[method_name] += [func_evals[method_name]] * (tf_skip)
    return summary_curve

def plot_summary_curves(price, summary_curves, use_log=[], title="Price and summary curves"):
    n_curves = len(summary_curves)
    n_rows = n_curves + 1

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        sharex=True,
        figsize=(12, 2.5 * n_rows)
    )

    if n_rows == 1:
        axes = [axes]

    axes[0].plot(price, color="tab:orange")
    axes[0].set_ylabel("price")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    for i, (name, curve_values) in enumerate(summary_curves.items(), start=1):
        axes[i].plot(curve_values, color="tab:blue")
        axes[i].set_ylabel(name)
        if name.split("-")[0] in use_log:
            axes[i].set_yscale("log")
        axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

    return fig, axes