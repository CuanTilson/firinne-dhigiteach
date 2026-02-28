# Data Workflow

This project keeps raw datasets outside the repository and stores only manifests, validation outputs, and helper scripts in git.

## Storage Policy

Do not copy raw datasets into the repository.
Use an external or secondary drive instead.

Example layout:
- `D:/firinne_datasets/genimage/...`
- `D:/firinne_datasets/kaggle/...`

The repository keeps:
- `data/scripts/` - manifest and validation scripts
- `data/manifests/` - generated CSV manifests and validation JSON
- `data/README.md` - workflow reference

## Current Dataset Scope

Primary training dataset:
- GenImage subset
  - `Midjourney`
  - `stable_diffusion_v_1_5`

External evaluation dataset:
- Kaggle AI-vs-Human dataset
  - labeled `train.csv`
  - unlabeled `test.csv` is not used for supervised evaluation

## Manifest Schema

All manifests use:
- `filepath`
- `label`
- `source`
- `generator`
- `split`

Labels are binary:
- `real`
- `ai`

## Build Manifests

```powershell
python data/scripts/build_week2_manifests.py `
  --genimage-root "D:/firinne_datasets/genimage" `
  --kaggle-csv-dir "D:/firinne_datasets/kaggle" `
  --kaggle-train-images "D:/firinne_datasets/kaggle/train_data" `
  --kaggle-test-images "D:/firinne_datasets/kaggle/test_data_v2" `
  --generators Midjourney stable_diffusion_v_1_5 `
  --seed 2026 `
  --max-per-class 2000
```

## Validate Manifests

```powershell
python data/scripts/validate_manifest_images.py `
  --input data/manifests/genimage_train.csv `
  --output data/manifests/genimage_train.cleaned.csv `
  --report data/manifests/genimage_train.validation.json `
  --min-width 256 `
  --min-height 256
```

Repeat validation for the validation, test, and Kaggle manifests.

## Current Locked Rules

- Seed: `2026`
- Minimum resolution: `256x256`
- GenImage split ratio: `70/15/15`
- Kaggle is used as external evaluation, not primary training

## Outputs You Should Expect

- `genimage_train.cleaned.csv`
- `genimage_val.cleaned.csv`
- `genimage_test.cleaned.csv`
- `kaggle_external_eval.cleaned.csv`
- matching `*.validation.json` reports
- `dataset_stats.json`
