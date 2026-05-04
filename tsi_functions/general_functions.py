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
from .utils_dtm import DTMFiltration, AlphaDTMFiltration 
    
class remove_inf():
    """
    Remove infinite bars from a barcode by replacing them with finite values based on the specified type and lambda parameter. 
    Definition from: "On the stability of persistent entropy and new summary functions for topological data analysis"
    """
    def __init__(self, type_ = "mu", lambda_ = 0, p = 1):
        assert type_ in ["old", "mu", "nu", "tau"], "type_ must be one of 'old', 'mu', 'nu', or 'tau'"
        self.type_ = type_
        self.lambda_ = lambda_
        self.p = p
    
    def __call__(self, barcode, theoretical_max = None):
        if self.type_ == "old":
            return [bar if bar[1] != float('inf') else (bar[0], self.lambda_) for bar in barcode]
        finite_bars = []
        if self.lambda_ > 0 or self.type_ == "tau":
            infinite_bars = []

            if self.type_ == "mu":
                max_length = 0
            elif self.type_ == "nu":
                s = 0
            elif self.type_ == "tau":
                max_death = -float('inf')

        for bar in barcode:
            if bar[1] != float('inf'):
                finite_bars.append(bar)
                if self.type_ == "mu":
                    if self.lambda_ > 0:
                        max_length = max(max_length, bar[1] - bar[0])
                elif self.type_ == "nu":
                    if self.lambda_ > 0:
                        s += (bar[1] - bar[0])**self.p
                elif self.type_ == "tau":
                    max_death = max(max_death, bar[1])
            elif self.lambda_ > 0 or self.type_ == "tau":
                infinite_bars.append(bar)
        
        if self.lambda_ == 0 and self.type_ != "tau":
            return finite_bars
        
        if self.type_ == "mu":
            if max_length == 0 and theoretical_max is not None:
                max_length = theoretical_max
            infinite_bars = [(bar[0], bar[0] + max_length * self.lambda_) for bar in infinite_bars]
            return finite_bars + infinite_bars

        if self.type_ == "nu":
            if s == 0 and theoretical_max is not None:
                s = theoretical_max*len(barcode)
            L_ap = s**(1/self.p) if s > 0 else 0
            infinite_bars = [(bar[0], bar[0] + L_ap * self.lambda_) for bar in infinite_bars]
            return finite_bars + infinite_bars

        if self.type_ == "tau":
            # print("Max death:", max_death)
            if max_death == -float('inf') and theoretical_max is not None:
                max_death = theoretical_max
            infinite_bars = [(bar[0], max_death * (1 + self.lambda_)) for bar in infinite_bars]
            print("n infinite bars:", len(infinite_bars))
            return finite_bars + infinite_bars
        

class summary_functions():
    def __init__(self, persistence_functions = {"TSI": lambda bars, lengths: np.var(lengths)}, bm_functions = {},  remove_inf_fun = remove_inf(), dims = ["all", 0, 1, "sum"]):
        self.dims = dims
        self.persistence_functions = persistence_functions
        self.bm_functions = bm_functions
        # self.function_names = list(persistence_functions.keys()) + list(bm_functions.keys())
        self.function_names = [f"{name}-{dim}" for name in persistence_functions.keys() for dim in self.dims] + list(bm_functions.keys())
        self.remove_inf_fun = remove_inf_fun

    def set_dims(self, dims):
        self.dims = dims
        self.function_names = [f"{name}-{dim}" for name in self.persistence_functions.keys() for dim in self.dims] + list(self.bm_functions.keys())
    
    def __call__(self, persistence, price = None, dt = None, theoretical_max_r = None):
        bars_dict = {}
        for dim, bar in persistence:
            bars_dict.setdefault(dim, []).append(bar)
        bars_dict = {dim: self.remove_inf_fun(bars_dict[dim], theoretical_max = theoretical_max_r) for dim in bars_dict.keys()}
        out = {}

        for dim in bars_dict.keys():
            if dim not in self.dims and "sum" not in self.dims:
                continue
            bars = bars_dict[dim]
            lengths = np.array([bar[1] - bar[0] for bar in bars])
            for name, func in self.persistence_functions.items():
                f_eval = func(bars, lengths)
                if dim in self.dims:
                    # if len(bars) == 0 or np.sum(lengths) == 0:
                    #     out[f"{name}-{dim}"] = np.nan
                    # else:
                    out[f"{name}-{dim}"] = f_eval
                if "sum" in self.dims:
                    out.setdefault(f"{name}-sum", 0)
                    out[f"{name}-sum"] += f_eval
        
        for dim in self.dims:
            if dim not in bars_dict.keys() and dim != "sum":
                for name, func in self.persistence_functions.items():
                    out[f"{name}-{dim}"] = func([], [])
        
        if "all" in self.dims:
            bars = []
            for dim in bars_dict.keys():
                bars += bars_dict[dim]
            lengths = np.array([bar[1] - bar[0] for bar in bars])
            for name, func in self.persistence_functions.items():
                # if len(bars) == 0:
                #     out[f"{name}-all"] = np.nan
                # else:
                out[f"{name}-all"] = func(bars, lengths)    

        if price is not None and dt is not None:
            for name, func in self.bm_functions.items():
                out[f"{name}"] = func(price, dt)

        for name in self.function_names:
            if name not in out:
                out[name] = np.nan

        return out
    
def filtration_from_points(points, filtration_type = "rips", max_edge_length = 1.0, max_dim=3, dtm = False, dtm_m = 0.03):
    if dtm:
        if filtration_type == "alpha":
            persistence = AlphaDTMFiltration(points, dtm_m, p=2, dimension_max=max_dim, filtration_max=max_edge_length).persistence()
        elif filtration_type == "rips":
            persistence = DTMFiltration(points, dtm_m, p=2, dimension_max=max_dim, filtration_max=max_edge_length).persistence()
    elif filtration_type == "alpha":
        Alpha_complex = gd.AlphaComplex(points=points)
        Alpha_simplex_tree_W = Alpha_complex.create_simplex_tree()
        persistence = Alpha_simplex_tree_W.persistence()
        alpha_to_cech_fun = lambda t: 2*np.sqrt(t)
        persistence = [(dim, (alpha_to_cech_fun(birth), alpha_to_cech_fun(death))) for dim, (birth, death) in persistence]

    elif filtration_type == "rips":
        rips_complex = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
        rips_simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        persistence = rips_simplex_tree.persistence()
    
    return persistence
    
def pe_doesnt_work(bars, lengths):
    if len(bars) == 0:
        return 0
    try:
        return gr.vector_methods.Entropy()(np.array(bars))[0]
    except:
        print("bars:\n", bars)
        raise ValueError("Persistent entropy computation failed")

def Renyi_entropy(bars, lengths, alpha):
    if len(bars) == 0 or np.sum(lengths) == 0:
        return 0
    p = lengths / np.sum(lengths) if np.sum(lengths) > 0 else np.zeros_like(lengths)
    if alpha == 1:
        return -np.sum(p * np.log(p))
    else:
        return (1/(1-alpha)) * np.log(np.sum(p**alpha))
    
top_summary_functions = summary_functions(persistence_functions = {
    "persistent_entropy": pe_doesnt_work,# lambda bars, lengths: gr.vector_methods.Entropy()(np.array(bars))[0],
    "normalized_persistent_entropy": lambda bars, lengths: gr.vector_methods.Entropy()(np.array(bars))[0]/np.log(np.sum(lengths)) if np.sum(lengths) > 0 else 0,
    "Renyi_2": lambda bars, lengths: Renyi_entropy(bars, lengths, alpha=2),
    "TSI": lambda bars, lengths: np.var(lengths, ddof=1) if len(lengths) > 1 else 0,
    "logTSI": lambda bars, lengths: np.log(np.var(lengths, ddof=1)) if len(lengths) > 1 and np.var(lengths, ddof=1) > 0 else -np.inf,
    "nTSI": lambda bars, lengths: np.var(lengths, ddof=1) / np.sum(lengths) if np.sum(lengths) > 0 else 0,
    "mTSI": lambda bars, lengths: np.var(lengths, ddof=1) / np.mean(lengths) if np.sum(lengths) > 0 else 0,
    "TSigI": lambda bars, lengths: np.sum(lengths**2) / np.sum(lengths) if np.sum(lengths) > 0 else 0,
    "cvTSI": lambda bars, lengths: np.var(lengths, ddof=1) / (np.mean(lengths))**2 if np.sum(lengths) > 0 else 0,
    "normalized_cvTSI": lambda bars, lengths: np.var(lengths, ddof=1) / (np.mean(lengths))**2 / len(bars) if np.sum(lengths) > 0 and len(lengths) > 1 else 0,
    "stdTSI": lambda bars, lengths: np.std(lengths, ddof=1) if len(lengths) > 1 else 0,
    "test": lambda bars, lengths: np.var(np.log(np.array(lengths)), ddof=1) if len(lengths) > 1 else 0,
    "mean_length": lambda bars, lengths: np.mean(lengths) if len(lengths) > 0 else 0,
    "sum_length": lambda bars, lengths: np.sum(lengths),
    "max_length": lambda bars, lengths: np.max(lengths) if len(lengths) > 0 else 0,
    "n_bars": lambda bars, lengths: len(bars)
}, remove_inf_fun=remove_inf(type_="mu", lambda_= 1)
)
    
