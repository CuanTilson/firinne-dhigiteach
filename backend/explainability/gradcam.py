import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageOps


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # ——— Proper, future-proof hooks ———
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

        # Model eval mode
        self.model.eval()

        # 224x224 Grad-CAM transform
        self.resize_transform = torch.nn.Sequential(
            torch.nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
        )

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple — index 0 is the outgoing gradient
        self.gradients = grad_output[0].detach()

    def generate(
        self, input_tensor, original_image_path: Path, save_path: Path
    ) -> Path:
        """
        - input_tensor: full-resolution tensor from your CNNDetectionModel
        - original_image_path: used only for final overlay
        """

        # ——— 1. Create a resized 224×224 tensor for Grad-CAM pass ———
        with torch.no_grad():
            small_tensor = (
                F.interpolate(
                    input_tensor,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
                .clone()
                .detach()
            )

        small_tensor.requires_grad = True

        # ——— 2. Forward + Backward on small tensor ———
        self.model.zero_grad()

        output = self.model(small_tensor)
        output.backward(torch.ones_like(output))

        # ——— 3. Compute weighted activations ———
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        activations = self.activations.squeeze(0)

        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        # heatmap: mean over channels
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)

        # ——— 4. Load original image (robust to non-ASCII paths) ———
        with Image.open(original_image_path) as img:
            img = ImageOps.exif_transpose(img)
            orig = np.array(img.convert("RGB"))

        # ——— 5. Upscale heatmap back to *original* dimensions ———
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

        # Colour-map
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # ——— 6. Overlay ———
        super_img = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

        # ——— 7. Save (Unicode-safe) ———
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(super_img).save(save_path)

        # ——— 8. Free CUDA memory ———
        torch.cuda.empty_cache()

        return save_path
