# Data

Pre-selected subset of the NASA C-MAPSS turbofan engine degradation
dataset. Only the feature-selected columns (cycle counter and four sensors)
are kept, plus the per-cycle TTF target for the training set and the
held-out test ground truth.

| File | Description |
|------|-------------|
| `train_selected.csv` | 20,631 cycles across 100 engines (`id`, `cycle`, `s1`–`s4`, `ttf`) |
| `test_selected.csv` | Cycle histories for 100 test engines (same schema, no `ttf`) |
| `PM_truth.txt` | True TTF at the final observed cycle for each of the 100 test engines |

Original C-MAPSS dataset:
[NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
