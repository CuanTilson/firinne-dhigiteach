# Model A Training

This folder contains the first self-trained image classifier workflow for the project.

## Purpose

Model A is a binary classifier:
- `real` -> `0`
- `ai` -> `1`

It is trained from the cleaned CSV manifests generated during Week 2.

## Inputs

Required manifests:
- `data/manifests/genimage_train.cleaned.csv`
- `data/manifests/genimage_val.cleaned.csv`
- `data/manifests/genimage_test.cleaned.csv`

Optional external evaluation:
- `data/manifests/kaggle_external_eval.cleaned.csv`

## Training Environments

### Backend app environment
Use this for the backend service only.
It is not the preferred place for GPU training.

### Dedicated training environment
Use `.venv-train` for model training.
This environment currently supports CUDA on the local RTX 4070.

## Example GPU Run

```powershell
.\.venv-train\Scripts\python.exe -m backend.models.training.train_model_a `
  --train-manifest data/manifests/genimage_train.cleaned.csv `
  --val-manifest data/manifests/genimage_val.cleaned.csv `
  --test-manifest data/manifests/genimage_test.cleaned.csv `
  --external-manifest data/manifests/kaggle_external_eval.cleaned.csv `
  --output-dir artifacts/model_a_baseline_gpu `
  --epochs 5 `
  --batch-size 16 `
  --image-size 224 `
  --lr 1e-4 `
  --seed 2026
```

## Current Baseline Design

- architecture: `ResNet-18`
- initialization: no pretrained weights
- optimizer: `Adam`
- default image size: `224`
- default batch size:
  - CPU: start low
  - GPU: `16` is currently working on the local machine

## Outputs

Each run writes:
- `model_a_best.pt`
- `run_manifest.json`

The manifest captures:
- config
- seed
- device
- manifests used
- per-epoch validation metrics
- final test metrics
- external metrics
- weights SHA-256

## Current Interpretation

The first baseline run demonstrates:
- strong in-domain performance on the GenImage subset
- weak external generalization on Kaggle

That result is expected to feed the next phase:
- broader training diversity
- improved evaluation reporting
- Model B learned fusion with forensic features

## Exporting Report-Ready Metrics

Use the export helper to turn a completed `run_manifest.json` into report-friendly files:

```powershell
.\.venv-train\Scripts\python.exe -m backend.models.training.export_model_a_results `
  --run-manifest artifacts/model_a_baseline_gpu/run_manifest.json
```

This creates an `exports/` folder beside the run manifest containing:
- `metrics_summary.csv`
- `confusion_matrices.csv`
- `metrics_summary.json`
- `evaluation_summary.md`

## Preparing Model B Features

Model B is the planned learned fusion stage. It does not replace the forensic pipeline; it learns from its outputs.

Feature export utility:

```powershell
.\\backend\\.venv\\Scripts\\python.exe -m backend.models.training.export_model_b_features `
  --output artifacts/model_b_features.csv
```

Optional supervised label attachment:

```powershell
.\\backend\\.venv\\Scripts\\python.exe -m backend.models.training.export_model_b_features `
  --output artifacts/model_b_features_labeled.csv `
  --label-map artifacts/model_b_label_map.csv
```

Expected label map schema:
- `record_id`
- `label`

Current Model B feature rows include:
- `ml_probability`
- metadata anomaly score and finding count
- C2PA presence, signature validity, score, and AI-assertion count
- JPEG structure and quantization features
- noise residual features
- ELA features
- watermark detection and confidence
- current rule-based forensic score

Use the backend environment for this export, not `.venv-train`, because the exporter reads the application database through SQLAlchemy.

## Creating a Label Map for Model B

Model B needs labeled feature rows. The current exporter only writes feature values; the target label must be attached separately.

If your analysis records come from files that also appear in cleaned manifests, you can generate a label-map template and auto-fill any filename matches:

```powershell
.\\backend\\.venv\\Scripts\\python.exe -m backend.models.training.create_model_b_label_map `
  --features-csv artifacts/model_b_features.csv `
  --output artifacts/model_b_label_map.csv `
  --manifest data/manifests/genimage_train.cleaned.csv `
  --manifest data/manifests/genimage_val.cleaned.csv `
  --manifest data/manifests/genimage_test.cleaned.csv
```

The output columns are:
- `record_id`
- `filename`
- `inferred_label`
- `final_label`
- `label_source`
- `notes`

Workflow:
1. export Model B features from the database
2. generate the label-map template
3. review any blank or ambiguous labels
4. pass the completed label map back into `export_model_b_features`

Important limitation:
- your current local database only contains analysis records that have actually been run through the application
- if those records were not generated from benchmark files with known labels, you will not get a meaningful supervised Model B dataset yet

## Building a Small Labeled Model B Dataset

The quickest controlled workflow is:
1. sample a small balanced batch from cleaned manifests
2. analyze those images through the backend forensic pipeline
3. export feature rows with labels attached
4. train a first Model B baseline

Example:

```powershell
.\\backend\\.venv\\Scripts\\python.exe -m backend.models.training.build_model_b_dataset `
  --manifest data/manifests/genimage_val.cleaned.csv `
  --manifest data/manifests/genimage_test.cleaned.csv `
  --per-class 100 `
  --seed 2026 `
  --output-dir artifacts/model_b_dataset
```

Outputs:
- `artifacts/model_b_dataset/model_b_features_labeled.csv`
- `artifacts/model_b_dataset/model_b_label_map.csv`
- `artifacts/model_b_dataset/build_summary.json`

## Training the First Model B Baseline

Start with a simple interpretable linear model:

```powershell
.\\.venv-train\\Scripts\\python.exe -m backend.models.training.train_model_b `
  --features-csv artifacts/model_b_dataset/model_b_features_labeled.csv `
  --output-dir artifacts/model_b_baseline `
  --epochs 200 `
  --batch-size 32 `
  --lr 1e-2 `
  --seed 2026
```

This produces:
- `artifacts/model_b_baseline/model_b_best.pt`
- `artifacts/model_b_baseline/run_manifest.json`

This baseline is intended for comparison against:
1. rule-based fusion only
2. Model A only
3. Model A + Model B

## Auditing Model B for Leakage

The first Model B result should not be trusted automatically. Because the initial dataset is small
and sourced from a controlled benchmark batch, you should audit whether any single feature is acting
as a shortcut.

Run:

```powershell
.\.venv-train\Scripts\python.exe -m backend.models.training.audit_model_b_features `
  --features-csv artifacts/model_b_dataset/model_b_features_labeled.csv `
  --output-dir artifacts/model_b_audit
```

Outputs:
- `artifacts/model_b_audit/feature_leakage_report.csv`
- `artifacts/model_b_audit/feature_leakage_summary.json`
- `artifacts/model_b_audit/feature_leakage_report.md`

Use this before presenting Model B results in the report. If a single feature or file-format pattern
almost perfectly separates the classes, the current Model B experiment is only an internal check and
not yet a defensible generalization result.

## Harder Model B Dataset

If the audit shows format leakage in the GenImage-based batch, rebuild the Model B dataset from the
Kaggle external manifest where both classes use `.jpg`:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.models.training.build_model_b_dataset `
  --manifest data/manifests/kaggle_external_eval.cleaned.csv `
  --per-class 100 `
  --seed 2026 `
  --require-extension-overlap `
  --output-dir artifacts/model_b_dataset_kaggle `
  --analysis-copy-dir backend/storage/uploaded/model_b_batch_kaggle
```

Then rerun the audit:

```powershell
.\.venv-train\Scripts\python.exe -m backend.models.training.audit_model_b_features `
  --features-csv artifacts/model_b_dataset_kaggle/model_b_features_labeled.csv `
  --output-dir artifacts/model_b_audit_kaggle
```

## Model B Ablations

`train_model_b.py` now supports feature ablations:
- `--exclude-features`
- `--exclude-prefixes`

Example ablations on the harder Kaggle-based dataset:

```powershell
.\.venv-train\Scripts\python.exe -m backend.models.training.train_model_b `
  --features-csv artifacts/model_b_dataset_kaggle/model_b_features_labeled.csv `
  --output-dir artifacts/model_b_baseline_kaggle_all `
  --epochs 200 `
  --batch-size 32 `
  --lr 1e-2 `
  --seed 2026
```

```powershell
.\.venv-train\Scripts\python.exe -m backend.models.training.train_model_b `
  --features-csv artifacts/model_b_dataset_kaggle/model_b_features_labeled.csv `
  --output-dir artifacts/model_b_baseline_kaggle_no_jpeg `
  --epochs 200 `
  --batch-size 32 `
  --lr 1e-2 `
  --seed 2026 `
  --exclude-prefixes jpeg_
```

```powershell
.\.venv-train\Scripts\python.exe -m backend.models.training.train_model_b `
  --features-csv artifacts/model_b_dataset_kaggle/model_b_features_labeled.csv `
  --output-dir artifacts/model_b_baseline_kaggle_no_format `
  --epochs 200 `
  --batch-size 32 `
  --lr 1e-2 `
  --seed 2026 `
  --exclude-prefixes jpeg_ metadata_ `
  --exclude-features jpeg_structure_valid hashes_match
```
