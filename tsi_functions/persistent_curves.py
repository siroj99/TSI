import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import gudhi.point_cloud.timedelay as td
from tqdm.auto import tqdm
import seaborn as sns
import gudhi.representations as gr
import pandas as pd
from copy import deepcopy
import scipy
import scipy.integrate     

from .general_functions import remove_inf

class PersistentCurve():
    def __init__(self, persistence, psi, bars_fun = lambda bars: bars, T = np.sum, remove_inf_fun = remove_inf()):
        self.psi = psi
        self.T = T
        self.bars_fun = bars_fun
        self.remove_inf_fun = remove_inf_fun

        self.persistence = {}
        for dim, bar in persistence:
            self.persistence.setdefault(dim, []).append(bar)
        
        for dim in self.persistence.keys():
            self.persistence[dim] = self.remove_inf_fun(self.persistence[dim])
            self.persistence[dim] = sorted(self.persistence[dim], key=lambda x: (x[0],x[1]))

        self.min_birth = min([bar[0] for bars in self.persistence.values() for bar in bars])
        self.max_death = max([bar[1] for bars in self.persistence.values() for bar in bars])
        # print("Min birth:", self.min_birth)
        # print("Max death:", self.max_death)

        self.curve_values = None
        self.use_normalized = False

    def compute_curve(self, n_values = None, t_values = None):
        # Not super optimized
        if t_values is None:
            self.t_values = np.linspace(self.min_birth, self.max_death, n_values)
        else:
            self.t_values = t_values
        self.curve_values = {dim: [] for dim in self.persistence.keys()}
        for dim, bars in self.persistence.items():
            bars_out = self.bars_fun(bars)
            for t in self.t_values:
                if self.use_normalized:
                    psi_evals = [self.normalized_psi(dim, bars_out, bar[0], bar[1], t) for bar in bars if bar[0] <= t and bar[1] > t]
                else:
                    psi_evals = [self.psi(bars_out, bar[0], bar[1], t) for bar in bars if bar[0] <= t and bar[1] > t]
                if len(psi_evals) > 0:
                    self.curve_values[dim].append(self.T(psi_evals))
                else:
                    self.curve_values[dim].append(0)
        return self.curve_values
    
    def plot_curve(self, curve_values=None, title="Persistent Curve", xlabel="Time", ylabel="Value", use_log=False, show=True, ax=None, max_dim=2, plot_args = {"linestyle": "-"}):
        if curve_values is None:
            curve_values = self.curve_values

        if ax is None:
            fig, ax = plt.subplots(min(len(curve_values.keys()), max_dim+1), 1, sharex=True, figsize=(15, 3*min(len(curve_values.keys()), max_dim+1)))
            if len(curve_values.keys()) == 1:
                ax = [ax]
        for dim, values in curve_values.items():
            if dim > max_dim:
                continue
            ax[dim].plot(self.t_values, values, color={0: "tab:blue", 1: "tab:orange", 2: "tab:green"}.get(dim, "tab:blue"), **plot_args)
            ax[dim].set_ylabel(f"Dim {dim}")
        ax[0].set_title(title)
        ax[-1].set_xlabel(xlabel)
        if use_log:
            for a in ax:
                a.set_yscale("log")
        
        if show:
            plt.show()
        return ax

    def normalize(self):
        """
        Psi should be independend of t.
        """
        self.norm_constants = {}
        for dim, bars in self.persistence.items():
            bars_out = self.bars_fun(bars)
            self.norm_constants[dim] = np.sum([np.abs(self.psi(bars_out, bar[0], bar[1], 0)) for bar in bars])
        
        # self.old_psi = self.psi
        self.normalized_psi = lambda dim, bars, b, d, t: self.psi(bars, b, d, t) / self.norm_constants[dim] if self.norm_constants[dim] != 0 else 0
        self.use_normalized = True

    def norm(self, p=1, n=1000):
        if self.curve_values is None:
            self.compute_curve(n_values=n)
        norms = {}
        for dim in self.curve_values.keys():
            if p != 1:
                norms[dim] = scipy.integrate.simpson(np.power(self.curve_values[dim], p), self.t_values)
            else:
                norms[dim] = scipy.integrate.simpson(np.abs(self.curve_values[dim]), self.t_values)
        return norms
        
    def distance(self, other, p=1, n=1000):
        if self.curve_values is None:
            self.compute_curve(n_values=n)
        if other.curve_values is None:
            other.compute_curve(n_values=n)
        dist = {}
        for dim in self.curve_values.keys():
            if dim in other.curve_values:
                if p != 1:
                    dist[dim] = scipy.integrate.simpson(np.power(np.abs(np.array(self.curve_values[dim]) - np.array(other.curve_values[dim])), p), self.t_values)
                else:
                    dist[dim] = scipy.integrate.simpson(np.abs(np.array(self.curve_values[dim]) - np.array(other.curve_values[dim])), self.t_values)
            else:
                dist[dim] = scipy.integrate.simpson(np.abs(self.curve_values[dim]), self.t_values)
        for dim in other.curve_values.keys():
            if dim not in self.curve_values:
                if p != 1:
                    dist[dim] = scipy.integrate.simpson(np.power(np.abs(np.array(other.curve_values[dim])), p), self.t_values)
                else:
                    dist[dim] = scipy.integrate.simpson(np.abs(np.array(other.curve_values[dim])), self.t_values)
        return dist