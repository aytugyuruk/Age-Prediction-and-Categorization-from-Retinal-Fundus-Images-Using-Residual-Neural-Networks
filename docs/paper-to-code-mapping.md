# Paper-to-code mapping

| Paper section | Code |
|---|---|
| III. Materials (dataset, train/val/test split) | `csvFiles/{train,validation,test}/*.csv` — as provided by the dataset authors, patient-level split |
| IV-A. Image Preprocessing | `src/preprocessing/crop_and_resize.py` (crop + pad + resize), `src/preprocessing/ben_graham_filter.py` (optional Graham filter) |
| IV-B. Deep Learning Model | `src/train.py`, `build_model()` — ResNet-18/34/50/101/152, ImageNet-pretrained, fully fine-tuned, custom regression head |
| IV-C. Label Distribution Smoothing | `src/train.py`, `get_lds_weights()` |
| IV-D. Age Prediction and Categorization | `src/evaluate.py`, `age_to_class()` — regression output is mapped to age categories only at evaluation time, not trained as a separate classifier |
| IV-E. Evaluation Metrics | `src/evaluate.py`, `compute_per_class_table()` (Table II style) and `compute_overall_table()` (Table III style) |
| IV-F. Environmental Settings (AdamW, Smooth L1, ReduceLROnPlateau, 80 epochs, batch 32) | `src/train.py`, `main()` |
| V. Results, Tables II–V | Reproduced by running `src/train.py` then `src/evaluate.py` for each of the 5 models, on both the filtered and non-filtered CSV splits |

## Metric definitions

- MAE: mean absolute error between predicted and true chronological age.
- Accuracy (overall, Table III/V): the test-set-support-weighted mean of the five per-class one-vs-rest accuracies (see `compute_overall_table()` in `evaluate.py`).
- Precision / Recall / F1 (overall): standard scikit-learn weighted multiclass averages over the five age categories.
- `--seed` in `train.py` produces a consistent, reproducible run; it does not guarantee bit-identical weights to the released checkpoint.
