import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from vendor.CNNDetection.networks.resnet import resnet50


class CNNDetectionModel:
    def __init__(self, weights_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = resnet50(num_classes=1)
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state["model"])

        self.model.to(self.device)
        self.model.eval()

        # Warm-up GPU to initialise context
        if self.device.type == "cuda":
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224, device=self.device)
                _ = self.model(dummy)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # We attach GradCAM to model.layer4, the last conv block
        self.target_layer = self.model.layer4

    def predict(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.sigmoid(out).item()

        label = "ai-generated" if prob > 0.5 else "real"

        return {
            "probability": prob,
            "label": label,
            "tensor": tensor,  # needed for Grad-CAM
        }

    def get_model(self):
        return self.model

    def get_target_layer(self):
        return self.target_layer
