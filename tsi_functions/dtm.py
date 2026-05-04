import numpy as np
import math
import random
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gudhi as gd
import gudhi.point_cloud.timedelay as td
from tqdm.auto import tqdm
import seaborn as sns
import gudhi.representations as gr
import pandas as pd
from copy import deepcopy
import scipy
import scipy.integrate     

from .utils_dtm import *

def estimate_conf_region(X, alpha=0.95, m=None, n=1000, size_of_sample = None, loading_bar = True, replace = True):
    if not size_of_sample:
        size_of_sample = int(X.shape[0]*0.9)
    
    DTM_values = DTM(X, X, m)

    differences = []
    subsample_bar = tqdm(range(n), leave=False)
    for subsample_i in subsample_bar:
        indices = np.random.choice(X.shape[0], size_of_sample, replace=replace)
        X_subsample = X[indices]
        DTM_subsample = DTM(X_subsample, X, m)
        diff = np.sqrt(n)*np.max(np.abs(DTM_values - DTM_subsample))
        differences.append(diff)

    return np.percentile(differences, alpha * 100)/np.sqrt(n)
