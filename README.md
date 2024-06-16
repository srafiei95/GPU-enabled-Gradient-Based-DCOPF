# GPU-enabled Gradien-Based DCOPF
This repository contains code for  loading two standard IEEE systems of 10000 and 2000 buses, functions and script ti run all the simulations presented in the following preprint:
*GPU-Accelerated DCOPF using Gradient-Based Optimization*
This paper was submitted to the *the Hawaii International Conference on System Sciences (HICSS) 2025* for single-blind peer review.
The models are implemented in ```Julia-1.10``` Language, using the ```JuMP.jl``` library for mathematical programming. Optimization problems are solved with ```MOSEK``` and ```Gurobi``` for benchmarking against the gradient-based methods.
For any questions, please contact srafiei@uvm.edu.
