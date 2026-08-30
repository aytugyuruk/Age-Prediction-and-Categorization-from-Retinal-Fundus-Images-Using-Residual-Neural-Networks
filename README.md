# Retinal Fundus Age Prediction (ResNet)

> [!NOTE]
> **Retinal Age Prediction research series · Study 01**
>
> [Publications overview](https://mehmetaytugyuruk.github.io/publications/) · [Study 02: Vision Transformers](https://github.com/mehmetaytugyuruk/retina-vit-age-prediction) · [Study 03: Color spaces](https://github.com/mehmetaytugyuruk/retina-color-spaces-age-prediction)

**Resources:** [Pretrained weights](https://huggingface.co/mehmetaytugyuruk/retina-resnet-age-prediction) · [Paper-to-code mapping](docs/paper-to-code-mapping.md) · [Citation](#citation) · [Companion ViT study](https://github.com/mehmetaytugyuruk/retina-vit-age-prediction)

## Publication

> M. A. Yürük and A. Memiş, "Age Prediction and Categorization from Retinal Fundus Images Using Residual Neural Networks," in *2026 5th International Informatics and Software Engineering Conference (IISEC)*, Ankara, Türkiye, Feb. 2026, pp. 628–633. doi: [10.1109/IISEC69317.2026.11418414](https://doi.org/10.1109/IISEC69317.2026.11418414)

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FIISEC69317.2026.11418414-blue)](https://doi.org/10.1109/IISEC69317.2026.11418414)
[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-11418414-00629B)](https://ieeexplore.ieee.org/document/11418414)

## Overview

Predicts chronological age from color retinal fundus images using five ResNet variants (ResNet-18/34/50/101/152), and derives an age-category classification (Pediatric / Young Adult / Middle Age / Senior / Elderly) from the regression output.

A companion study using Vision Transformers on the same dataset is available in the [Vision Transformer study repository](https://github.com/mehmetaytugyuruk/retina-vit-age-prediction).

## Results

Best result per preprocessing variant, on the held-out test set (1,462 images):

| Preprocessing | Best model | MAE (years) | Age-category F-measure |
|---|---|---|---|
| Non-filtered | ResNet-101 | 5.09 | 0.7157 (ResNet-18) |
| Graham-filtered | **ResNet-101** | **5.02** | **0.7165 (ResNet-34)** |

Full per-class and per-model tables are in the paper.

## Dataset

[Retina Age Analysis Dataset](https://huggingface.co/datasets/ramankamran/retina-age-analysis) (Kamran, 2025), MIT-licensed, 9,857 fundus images with chronological age and age-category labels, pre-split 6,902 / 1,493 / 1,462 (train/val/test) on a patient basis.

This repo does not redistribute the images. To reproduce:
1. Download the dataset from Hugging Face.
2. Run the preprocessing scripts below to produce the non-filtered and Graham-filtered variants.
3. Place the results under `ImageFolders/non_filtered_images/` and `ImageFolders/filtered_images/` so the paths in `csvFiles/*/*.csv` resolve correctly (or edit the CSVs to point elsewhere).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training and evaluation automatically select CUDA, Apple Metal (MPS), or CPU,
in that order.

## Preprocessing

```bash
# Step 1: crop the retina disk, pad, and resize to a square canvas.
python src/preprocessing/crop_and_resize.py <raw_images_dir> ImageFolders/non_filtered_images

# Step 2 (optional): Ben Graham filter, for the filtered-image experiments.
python src/preprocessing/ben_graham_filter.py ImageFolders/non_filtered_images ImageFolders/filtered_images
```

## Training

```bash
python src/train.py --model resnet101 \
    --train-csv csvFiles/train/trainFilteredImages.csv \
    --val-csv csvFiles/validation/validationFilteredImages.csv \
    --output-dir checkpoints
```

`--model` accepts `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`. Swap the `--train-csv`/`--val-csv` arguments for the `*NonFilteredImages.csv` files to reproduce the non-filtered results instead.

## Evaluation

```bash
python src/evaluate.py --model resnet101 \
    --checkpoint checkpoints/best_resnet101.pth \
    --test-csv csvFiles/test/testFilteredImages.csv
```

Prints per-age-category and overall metrics (MAE, accuracy, precision, recall, F1), and saves a summary table as a PNG next to the checkpoint.

## Pretrained checkpoints

Not distributed via this repository (checkpoint files range from ~130MB to ~680MB). All 10 checkpoints (5 architectures × filtered/non-filtered) are hosted on Hugging Face Hub:

**[Hugging Face checkpoint repository](https://huggingface.co/mehmetaytugyuruk/retina-resnet-age-prediction)**

```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download("mehmetaytugyuruk/retina-resnet-age-prediction", "resnet101-filtered.pth")
```

See the model card for the full file list, per-model results, and a loading example.

## Repository structure

```
src/
├── train.py                       # training, all 5 ResNet variants via --model
├── evaluate.py                    # evaluation + per-class/overall metric tables
└── preprocessing/
    ├── crop_and_resize.py         # fundus disk crop + pad + resize
    └── ben_graham_filter.py       # optional Ben Graham vessel-enhancement filter
csvFiles/                          # train/val/test splits (filtered and non-filtered)
docs/paper-to-code-mapping.md      # which script/table corresponds to which paper section
```

See [docs/paper-to-code-mapping.md](docs/paper-to-code-mapping.md) for exactly how each part of the paper maps to the code.

## Citation

```bibtex
@inproceedings{yuruk2026age,
  title     = {Age Prediction and Categorization from Retinal Fundus Images Using Residual Neural Networks},
  author    = {Yürük, Mehmet Aytuğ and Memiş, Abbas},
  booktitle = {2026 5th International Informatics and Software Engineering Conference (IISEC)},
  pages     = {628--633},
  year      = {2026},
  address   = {Ankara, Türkiye},
  publisher = {IEEE},
  doi       = {10.1109/IISEC69317.2026.11418414}
}
```

## License

Code released under the [MIT License](LICENSE). The dataset is separately licensed by its authors (MIT, see the [dataset card](https://huggingface.co/datasets/ramankamran/retina-age-analysis)).
