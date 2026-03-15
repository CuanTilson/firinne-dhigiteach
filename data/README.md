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
  - baseline generators:
    - `Midjourney`
    - `stable_diffusion_v_1_5`
  - broader `Model A v2` generators:
    - `ADM`
    - `glide`
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

### Baseline manifest build

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

### Broader `Model A v2` manifest build

Use this when you want to test whether broader generator diversity improves external
generalization without changing the training code.

```powershell
python data/scripts/build_week2_manifests.py `
  --genimage-root "D:/firinne_datasets/genimage" `
  --kaggle-csv-dir "D:/firinne_datasets/kaggle" `
  --kaggle-train-images "D:/firinne_datasets/kaggle/train_data" `
  --kaggle-test-images "D:/firinne_datasets/kaggle/test_data_v2" `
  --generators ADM BigGAN glide Midjourney stable_diffusion_v_1_5 `
  --seed 2026 `
  --max-per-class 2000 `
  --max-per-generator-per-class 500 `
  --output-dir data/manifests/model_a_v2
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

For the broader `Model A v2` experiment, use the same validator against:
- `data/manifests/model_a_v2/genimage_train.csv`
- `data/manifests/model_a_v2/genimage_val.csv`
- `data/manifests/model_a_v2/genimage_test.csv`
- `data/manifests/model_a_v2/kaggle_external_eval.csv`

## Audit Generator Coverage After Validation

Do not assume the cleaned manifests still preserve the intended generator mix. Run a coverage
audit after validation and before training:

```powershell
python data/scripts/audit_manifest_coverage.py `
  --manifest data/manifests/model_a_v2/genimage_train.cleaned.csv `
  --manifest data/manifests/model_a_v2/genimage_val.cleaned.csv `
  --manifest data/manifests/model_a_v2/genimage_test.cleaned.csv `
  --expected-generator ADM `
  --expected-generator BigGAN `
  --expected-generator glide `
  --expected-generator Midjourney `
  --expected-generator stable_diffusion_v_1_5 `
  --output-json artifacts/model_a_v2_gpu/manifest_coverage_audit.json `
  --output-md artifacts/model_a_v2_gpu/manifest_coverage_audit.md
```

If any expected `generator:label` pair disappears after validation, do not treat the run as the
final broader-dataset experiment. Fix the dataset composition first and rerun.

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

For `Model A v2`, the same files should exist under:
- `data/manifests/model_a_v2/`
