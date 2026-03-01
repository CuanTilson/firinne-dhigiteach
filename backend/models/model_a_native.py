import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms


class ModelADetector:
    def __init__(self, weights_path: Path, run_manifest_path: Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = Path(weights_path)
        self.run_manifest_path = Path(run_manifest_path) if run_manifest_path else None
        self.run_manifest = self._load_run_manifest()
        self.image_size = int(
            self.run_manifest.get("config", {}).get("image_size", 224)
        )

        base_model = models.resnet18(weights=None)
        base_model.fc = nn.Linear(base_model.fc.in_features, 2)
        state = torch.load(self.weights_path, map_location=self.device)
        base_model.load_state_dict(state)
        base_model.to(self.device)
        base_model.eval()

        self.base_model = base_model
        self.model = _GradCAMBinaryWrapper(base_model)
        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            with torch.no_grad():
                dummy = torch.randn(
                    1, 3, self.image_size, self.image_size, device=self.device
                )
                _ = self.base_model(dummy)

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.target_layer = self.base_model.layer4

    def _load_run_manifest(self) -> dict:
        if self.run_manifest_path and self.run_manifest_path.is_file():
            try:
                return json.loads(self.run_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def predict(self, image_path: Path):
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.base_model(tensor)
            probs = torch.softmax(logits, dim=1)
            ai_prob = probs[:, 1].item()

        label = "ai-generated" if ai_prob > 0.5 else "real"
        return {
            "probability": ai_prob,
            "label": label,
            "tensor": tensor,
            "gradcam_target_index": 1,
        }

    def get_model(self):
        return self.model

    def get_target_layer(self):
        return self.target_layer

    def get_metadata(self) -> dict:
        config = self.run_manifest.get("config", {})
        return {
            "detector": "model_a",
            "display_name": "Model A",
            "image_size": self.image_size,
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "lr": config.get("lr"),
            "seed": self.run_manifest.get("seed"),
            "weights_sha256": self.run_manifest.get("weights_sha256"),
        }


class _GradCAMBinaryWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.layer4 = base_model.layer4

    def forward(self, x):
        logits = self.base_model(x)
        return logits[:, 1:2]
