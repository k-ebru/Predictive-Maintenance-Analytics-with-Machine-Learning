# Original MATLAB prototypes

The analysis was first prototyped in MATLAB. These scripts are kept here for
reference — the Python implementation in `src/` and `notebooks/` follows the
same preprocessing pipeline and modelling choices.

| File | What it does |
|------|--------------|
| `regression.m` | Random forest and stepwise quadratic regression for TTF prediction |
| `classification.m` | Logistic regression and KNN for healthy/faulty classification with 10-fold engine-level cross-validation |

To reproduce the results, set the MATLAB working directory to the project's
`data/` folder so `readtable('train_selected.csv')` etc. resolve correctly.
