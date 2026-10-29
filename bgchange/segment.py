"""Subject segmentation with BiRefNet (the model chosen in E1, see EXPERIMENTS.md) -> soft alpha in [0, 1]."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ID = "ZhengPeng7/BiRefNet"
# trust_remote_code runs Python code from this repo, so pin the commit: the code that runs is the code
# benchmarked in E1 (main has not changed since 2026-02-04).
REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
INPUT_SIZE = 1024  # BiRefNet's training resolution

to_tensor = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
])


def load_birefnet() -> torch.nn.Module:
    """Download (once, then cached by Hugging Face) and build BiRefNet at the pinned commit."""
    from transformers import AutoModelForImageSegmentation  # 5 s import: only when the real model is needed
    return AutoModelForImageSegmentation.from_pretrained(REPO_ID, revision=REVISION, trust_remote_code=True)


class Segmenter:
    """BiRefNet loaded once; `segmenter(img)` returns the alpha of a PIL image.

    The alpha is float32 in [0, 1], at the image's own size. On a GPU the model runs in fp16 (same
    scores as fp32 to 3 decimals in E1, and faster); on CPU in fp32 (fp16 is slow or unsupported there).
    """

    def __init__(self, device: str | None = None, model: torch.nn.Module | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model = load_birefnet() if model is None else model  # tests pass a small stand-in model
        self.model = model.to(self.device, self.dtype).eval()  # cast explicitly: the hub weights are stored in fp16

    @torch.inference_mode()
    def __call__(self, img: Image.Image) -> np.ndarray:
        rgb = img if img.mode == "RGB" else img.convert("RGB")
        x = to_tensor(rgb).unsqueeze(0).to(self.device, self.dtype)
        logits = self.model(x)[-1]  # list of side outputs, the last one is the final prediction
        alpha = F.interpolate(logits.sigmoid(), size=(img.height, img.width), mode="bilinear", align_corners=False)
        return alpha[0, 0].float().cpu().numpy()
