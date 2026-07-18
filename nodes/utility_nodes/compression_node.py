import torch, numpy as np, io
from PIL import Image
class OmniSimulator_CompressionArtifacts:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "quality": ("INT", {"default": 75, "min": 1, "max": 100}),
                             "cycles": ("INT", {"default": 1, "min": 1, "max": 10})}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "apply"; CATEGORY = "OmniSimulator/Utils"
    def apply(self, image, quality, cycles):
        results = []
        for i in range(image.shape[0]):
            arr = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            for _ in range(cycles):
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=quality, subsampling=2)
                buf.seek(0); pil = Image.open(buf).convert("RGB")
            results.append(torch.from_numpy(np.array(pil).astype(np.float32)/255.0).unsqueeze(0))
        return (torch.cat(results, 0),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_CompressionArtifacts": OmniSimulator_CompressionArtifacts}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_CompressionArtifacts": "OmniSimulator Compression Artifacts"}