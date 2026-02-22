from __future__ import annotations

import base64
import io
from typing import List

import numpy as np
import open_clip
import torch
from PIL import Image


class CLIPEmbedder:
    """Wrapper around an OpenCLIP model for image embeddings."""

    def __init__(
        self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(self.device)
        model.eval()

        self.model = model
        self.preprocess = preprocess
        self.embedding_dim = (
            getattr(getattr(model, "visual", None), "output_dim", None)
            or getattr(model, "embed_dim", 512)
        )

    def _decode_base64_image(self, b64_str: str) -> Image.Image:
        if not b64_str:
            raise ValueError("Empty base64 image string")
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(raw))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _encode(self, image_tensor: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            if self.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    embeddings = self.model.encode_image(image_tensor)
            else:
                embeddings = self.model.encode_image(image_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype(np.float32)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL image into a 512-dim CLIP vector."""
        return self.embed_images([image])[0]

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """Embed a list of PIL images."""
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        image_tensor = torch.stack([self.preprocess(image) for image in images]).to(
            self.device
        )
        return self._encode(image_tensor)

    def embed_base64(self, b64_str: str) -> np.ndarray:
        """Decode a base64 PNG string and embed it."""
        image = self._decode_base64_image(b64_str)
        return self.embed_image(image)

    def embed_base64_batch(self, b64_strings: List[str]) -> np.ndarray:
        """Decode and embed a batch of base64 PNG strings."""
        images = [self._decode_base64_image(b64_str) for b64_str in b64_strings]
        return self.embed_images(images)