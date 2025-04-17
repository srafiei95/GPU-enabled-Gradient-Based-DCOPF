# GPU-enabled Gradient-Based DCOPF
This repository contains code for  loading three standard IEEE systems of 10000, 4601, and 2000 buses, functions and script to run all the simulations presented in the following work:
*GPU-Accelerated DCOPF using Gradient-Based Optimization*
The models are implemented in ```Julia-1.10``` Language, using the ```JuMP.jl``` library for mathematical programming. Optimization problems are solved with ```MOSEK``` and ```Gurobi``` for benchmarking against the gradient-based methods.
For any questions, please contact srafiei@uvm.edu.

**Academic Use Notice**:

If you use this code in your research, teaching, or any academic publication, please cite the following paper:

S. S. Rafiei and S. Chevalier. "GPU-Accelerated DCOPF using Gradient-Based Optimization." Proceedings of the 58th Hawaii International Conference on System Sciences (HICSS), 2025. DOI: 10.24251/HICSS.2025.378
