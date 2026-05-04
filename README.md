The repository is split into 3 different notebooks.
- synthetic_brownian_motion:      
  - For analysis of synthetic time series data.
- points_on_circle:               
  - For analysis of synthetic data of points sampled from circles.
- making_images:                      
  - For making the images in the paper.

Most of the functions are in the directory `tsi_functions`. In `general_functions.py`, the first thing is to use the `filtration_from_points` function, which outputs the persistence of a set of points using Gudhi.

Besides this, you can find `top_summary_functions` in the same script, which is a function that given persitence output of gudhi, computes the TSI and entropy and other statistics.

For time series, there is a separate script containing some functions. It contains `bm_summary_functions`, which adds two outputs to `top_summary_functions`, namely the estimation of Geometric Browinian Motion parameters. 

Then `summary_curve` uses one of the two `summary_functions` to evaluate TSI and entropy and the other methods on a time series. For this it also requires a `persistence_func`, which takes the price and outputs the persistence. Use either `persistence_timedelay` or `persistence_timesublevel`. 
