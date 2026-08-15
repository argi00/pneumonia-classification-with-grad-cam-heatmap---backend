from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import Args
from model import ModelPumonie


class PneumoniaPredictor:

    def __init__(self):

        # --------------------------------------------------
        # Paths
        # --------------------------------------------------

        self.project_root = Path(
            "."
        )

        self.checkpoint_path = (
             "best.ckpt"
        )

        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------

        self.args = Args()

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # --------------------------------------------------
        # Load model
        # --------------------------------------------------

        self.model = ModelPumonie.load_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            args=self.args,
        )

        self.model.to(self.device)
        self.model.eval()

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.args.size, self.args.size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                self.args.mean,
                self.args.std
            ),
        ])

        self.display_transform = transforms.Compose([
            transforms.Resize(
                (self.args.size, self.args.size)
            ),
            transforms.ToTensor(),
        ])

        # --------------------------------------------------
        # Grad-CAM target layer
        # --------------------------------------------------

        self.target_layer = (
            self.model
            .model
            .model
            .features
            .denseblock4
            .denselayer16
            .conv2
        )

        self.activations = None
        self.gradients = None

        self._register_hooks()

    # ======================================================
    # Grad-CAM hooks
    # ======================================================

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(
            forward_hook
        )

        self.target_layer.register_full_backward_hook(
            backward_hook
        )

    # ======================================================
    # Prediction
    # ======================================================

    def predict(self, image: Image.Image):

        # Preprocess
        input_tensor = self.transform(image)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # Gradients must be enabled for Grad-CAM
        input_tensor.requires_grad_(True)

        # Clear previous gradients
        self.model.zero_grad()

        # Forward
        output = self.model(input_tensor)

        # Binary classifier
        probability = torch.sigmoid(output).item()

        prediction = (
            "PNEUMONIA"
            if probability >= 0.5
            else "NORMAL"
        )

        # Backpropagate pneumonia score
        output.backward()

        # Generate Grad-CAM
        cam = self._generate_gradcam()

        # Prepare visualization image
        rgb_image = self.display_transform(
            image
        ).permute(1, 2, 0).numpy()

        overlay = self._create_overlay(
            rgb_image,
            cam
        )

        return {
            "prediction": prediction,
            "probability": probability,
            "cam": overlay,
        }

    # ======================================================
    # Generate Grad-CAM
    # ======================================================

    def _generate_gradcam(self):

        activation = self.activations
        gradient = self.gradients

        # Global average pooling
        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps
        cam = (
            weights * activation
        ).sum(dim=1)

        # ReLU
        cam = torch.relu(cam)

        # Remove batch dimension
        cam = cam[0]

        # CPU / NumPy
        cam = cam.detach().cpu().numpy()

        # Normalize
        cam -= cam.min()

        if cam.max() != 0:
            cam /= cam.max()

        # Resize
        cam = cv2.resize(
            cam,
            (
                self.args.size,
                self.args.size
            )
        )

        return cam

    # ======================================================
    # Create visualization
    # ======================================================

    def _create_overlay(
        self,
        rgb_image,
        cam
    ):

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        heatmap = (
            heatmap.astype(np.float32)
            / 255
        )

        overlay = (
            0.5 * rgb_image
            + 0.5 * heatmap
        )

        overlay = np.clip(
            overlay,
            0,
            1
        )

        return overlay