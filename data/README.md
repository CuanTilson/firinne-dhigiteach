# Data Workflow

Raw datasets are not stored in this repository. The project stores scripts and CSV/JSON manifests that describe how training and evaluation data was selected.

## Stored Here

```text
data/scripts/      manifest building, validation, and coverage-audit scripts
data/manifests/    generated CSV manifests and dataset summaries
data/README.md     this workflow note
```

## Raw Dataset Location

Use an external location for raw datasets, for example:

```text
D:/firinne_datasets/genimage/...
D:/firinne_datasets/kaggle/...
```

## Current Model A v2.1 Dataset Scope

Training generators:

```text
ADM
glide
Midjourney
stable_diffusion_v_1_5
```

External evaluation:

```text
Kaggle AI-vs-Human cleaned evaluation manifest
```

`BigGAN` was downloaded but excluded from the final v2.1 run because validation removed the intended AI samples from the cleaned split.

## Manifest Schema

Manifests use:

```text
filepath,label,source,generator,split
```

Labels:

```text
real
ai
```

## Build Model A v2.1 Manifests

```powershell
python data/scripts/build_week2_manifests.py `
  --genimage-root "D:/firinne_datasets/genimage" `
  --kaggle-csv-dir "D:/firinne_datasets/kaggle" `
  --kaggle-train-images "D:/firinne_datasets/kaggle/train_data" `
  --kaggle-test-images "D:/firinne_datasets/kaggle/test_data_v2" `
  --generators ADM glide Midjourney stable_diffusion_v_1_5 `
  --seed 2026 `
  --max-per-class 2000 `
  --max-per-generator-per-class 500 `
  --output-dir data/manifests/model_a_v2_1
```

## Validate Manifests

```powershell
python data/scripts/validate_manifest_images.py `
  --input data/manifests/model_a_v2_1/genimage_train.csv `
  --output data/manifests/model_a_v2_1/genimage_train.cleaned.csv `
  --report data/manifests/model_a_v2_1/genimage_train.validation.json `
  --min-width 256 `
  --min-height 256
```

Repeat for validation and test manifests.

## Audit Coverage

```powershell
python data/scripts/audit_manifest_coverage.py `
  --manifest data/manifests/model_a_v2_1/genimage_train.cleaned.csv `
  --manifest data/manifests/model_a_v2_1/genimage_val.cleaned.csv `
  --manifest data/manifests/model_a_v2_1/genimage_test.cleaned.csv `
  --expected-generator ADM `
  --expected-generator glide `
  --expected-generator Midjourney `
  --expected-generator stable_diffusion_v_1_5 `
  --output-json artifacts/model_a_v2_1_gpu/manifest_coverage_audit.json `
  --output-md artifacts/model_a_v2_1_gpu/manifest_coverage_audit.md
```

## Locked Rules

```text
seed: 2026
minimum image size: 256x256
split ratio: 70/15/15
external evaluation: Kaggle cleaned manifest
```
