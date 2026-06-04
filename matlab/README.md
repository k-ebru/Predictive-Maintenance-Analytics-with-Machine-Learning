# Original MATLAB prototypes

The analysis was first prototyped in MATLAB. These scripts are kept here
for reference. The Python implementation under `src/` and `notebooks/`
follows the same preprocessing pipeline and the same modelling choices,
and reaches almost identical numerical results.

| File | What it does |
|------|--------------|
| `regression.m`     | Random forest and stepwise quadratic regression for TTF prediction |
| `classification.m` | Logistic regression and KNN for healthy/faulty classification with 10-fold engine-level cross-validation |

To reproduce in MATLAB, set the working directory to the project's `data/`
folder so calls like `readtable('train_selected.csv')` resolve correctly.
