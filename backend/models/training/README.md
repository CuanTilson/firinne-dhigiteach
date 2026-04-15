# Model Training Utilities

This folder contains training and evaluation utilities for the project models.

## Model A

Model A is the self-trained binary image classifier.

```text
real -> 0
ai   -> 1
```

Current preferred checkpoint:

```text
artifacts/model_a_v2_1_gpu/model_a_best.pt
artifacts/model_a_v2_1_gpu/run_manifest.json
```

Architecture and training setup:

```text
architecture: ResNet-18
optimizer: Adam
image size: 224
seed: 2026
external evaluation: Kaggle cleaned manifest
```

Train Model A v2.1:

```powershell
.\.venv-train\Scripts\python.exe -m backend.models.training.train_model_a `
  --train-manifest data/manifests/model_a_v2_1/genimage_train.cleaned.csv `
  --val-manifest data/manifests/model_a_v2_1/genimage_val.cleaned.csv `
  --test-manifest data/manifests/model_a_v2_1/genimage_test.cleaned.csv `
  --external-manifest data/manifests/kaggle_external_eval.cleaned.csv `
  --output-dir artifacts/model_a_v2_1_gpu `
  --epochs 5 `
  --batch-size 16 `
  --image-size 224 `
  --lr 1e-4 `
  --seed 2026
```

Export Model A metrics:

```powershell
.\.venv-train\Scripts\python.exe -m backend.models.training.export_model_a_results `
  --run-manifest artifacts/model_a_v2_1_gpu/run_manifest.json
```

Export ROC-AUC and threshold outputs:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.models.training.analyze_model_a_thresholds `
  --run-manifest artifacts/model_a_v2_1_gpu/run_manifest.json `
  --weights artifacts/model_a_v2_1_gpu/model_a_best.pt `
  --output-dir artifacts/model_a_v2_1_gpu/exports `
  --skip-external
```

## Model B

Model B is the offline learned-fusion experiment. It uses exported forensic features and is not the live inference path.

Main scripts:

```text
model_b_features.py
build_model_b_dataset.py
train_model_b.py
audit_model_b_features.py
export_model_b_comparison.py
```

The final dissertation evidence uses the leakage-audited Kaggle-250 comparison outputs under:

```text
artifacts/model_b_comparison_kaggle_250/
```

## Chapter 6 Evaluation Pack

Consolidate frozen metrics and confusion matrices:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.tools.export_chapter6_evaluation_pack
```

Outputs:

```text
artifacts/chapter6_evaluation/chapter6_metrics_summary.csv
artifacts/chapter6_evaluation/chapter6_confusion_summary.csv
artifacts/chapter6_evaluation/chapter6_evaluation_summary.md
```
