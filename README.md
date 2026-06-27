# GDU-HMDE
This repository provides the MATLAB implementation of the algorithm GDU-HMDE.

# Paper
If you use this code, please cite the following paper:

Differential evolution with gated dimension update and hierarchical memory for numerical optimization.

# Requirements
The code was tested with:
- Ubuntu 22.04 LTS
- MATLAB R2024a or later


# Folder Structure
```
GDU-HMDE/
│
├── input_data17/ # Supporting data files for the CEC 2017 benchmark function
├── cec17_func.cpp # CEC2017 benchmark function implementation
├── cec17_func.mexa64 # Precompiled MEX file for Linux/macOS (recompile required for Windows) 
├── GDUHM_DE.m # MATLAB implementation of the proposed algorithm
├── append_hist.m # Function for recording convergence history
└── README.md # Project documentation
```
# How to Run

Example:

```matlab
func_num=1;
fhd=str2func('cec17_func');
dim = 50;
xmin = -100;
xmax = 100;
[bestx, bestf, hist] = GDUHM_DE(func_num, fhd, dim, xmin, xmax);
disp(bestx)
disp(bestf)
y = hist(:);
x = 1:numel(y);
figure;
plot(x, y, '-', 'LineWidth', 1.5, 'MarkerSize', 2);

Where:
func_num : benchmark function index
fhd : objective function handle
dim : problem dimension
xmin/xmax : search range
```

# License
This project is licensed under the MIT License.
The implementation is provided to support reproducible research in evolutionary computation.
