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

        self.checkpoint_path = "best.ckpt"
        self.args = Args()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = ModelPumonie.load_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            args=self.args,
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size)),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean, self.args.std),
        ])

        self.display_transform = transforms.Compose([
            transforms.Resize((self.args.size, self.args.size)),
            transforms.ToTensor(),
        ])

        self.target_layer = (
            self.model.model.model.features.denseblock4.denselayer16.conv2
        )
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):

        self.target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o)
        )
        self.target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0])
        )

    def predict(self, image: Image.Image):

        input_tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )
        input_tensor.requires_grad_(True)

        self.model.zero_grad()
        output = self.model(input_tensor)

        probability = torch.sigmoid(output).item()
        prediction = "PNEUMONIA" if probability >= 0.5 else "NORMAL"

        output.backward()

        cam = self._generate_gradcam()
        rgb_image = self.display_transform(image).permute(1, 2, 0).numpy()

        return {
            "prediction": prediction,
            "probability": probability,
            "cam": self._create_overlay(rgb_image, cam),
        }

    def _generate_gradcam(self):

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))
        cam = cam[0].detach().cpu().numpy()

        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cv2.resize(cam, (self.args.size, self.args.size))

    def _create_overlay(self, rgb_image, cam):

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

        return np.clip(0.5 * rgb_image + 0.5 * heatmap, 0, 1)