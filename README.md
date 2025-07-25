# Code for paper: Convergence Properties of Natural Gradient Descent for Minimizing KL Divergence

This repository contains the simulation code to supplement the TMLR paper "Convergence Properties of Natural Gradient Descent for Minimizing KL Divergence" by Adwait Datar and Nihat Ay.

Paper available at : https://openreview.net/forum?id=h6hjjAF5Bj

The code has the following files as entry points:

### `natgrad_and_grad_flow.m`

This script simulates and visualizes **natural gradient flow** and **standard gradient flow** for probability distributions on the simplex, comparing their dynamics in **primal (eta)** and **dual (theta)** coordinate systems.

It reproduces the experiments from our TMLR paper by:

1. Simulating the time evolution of a probability distribution `p` under:
   - Natural gradient flow,
   - Standard gradient flow in eta-coordinates,
   - Standard gradient flow in theta-coordinates.

2. Computing and plotting the **KL divergence decay** along each trajectory (in linear and log scales).

3. Visualizing **trajectories and contour plots** for the 2D case.

### `conotur_plot_of_L_star.m`

This script plots the contour plots of L* expressed in theta coordinates.

### `conotur_plot_of_L.m`

This script plots the contour plots of L expressed in eta coordinates.

### `empirical_conv_rate_vs_learning_rate_GD.m`

This script simulates **natural gradient flow** and **standard gradient flow** in **primal (eta)** and **dual (theta)** coordinate systems with various learning rates and plots empirical convergence times for each learning rate.

### `plot_sections_eta_theta.m`

This script visualizes sections of the KL divergence functions in eta and theta coordinates.Plots these sections alongside quadratic approximations for comparison.


## Prerequisites
The simulation code in this repository was tested in the following environment:
* *Windows 11 Pro* Version 23H2
* *Matlab* 2023a