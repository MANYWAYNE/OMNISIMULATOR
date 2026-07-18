import torch
import torch.nn.functional as F
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import os

class TextureMatcher:
    def __init__(self, profile_path, device):
        self.device = device
        try:
            with np.load(profile_path) as stats:
                # Use log to compress the range of values for a more stable loss
                self.glcm_target_mean = torch.log1p(torch.tensor(stats['glcm_props'].mean(axis=0), dtype=torch.float32, device=self.device))
                self.lbp_target_mean = torch.tensor(stats['lbp_hist'].mean(axis=0), dtype=torch.float32, device=self.device)
                self.n_points, self.radius, self.lbp_bins = 24, 3, self.lbp_target_mean.shape[0]

            print(f"  [TextureMatcher] Loaded profile '{os.path.basename(profile_path)}'.")

        except Exception as e:
            raise ValueError(f"Failed to load or parse texture profile from '{profile_path}': {e}")

    def _tensor_to_gray_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        img_chw = tensor.squeeze(0)
        gray_chw = 0.299 * img_chw[0, :, :] + 0.587 * img_chw[1, :, :] + 0.114 * img_chw[2, :, :]
        return (gray_chw.detach().cpu().numpy() * 255).astype(np.uint8)

    def __call__(self, img_tensor: torch.Tensor, log_stats: bool = False):
        if img_tensor.shape[0] > 1:
            img_tensor = img_tensor[0:1]

        gray_np = self._tensor_to_gray_numpy(img_tensor)

        # 1. GLCM Loss
        try:
            glcm = graycomatrix(gray_np, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            glcm_vec = torch.tensor([contrast, homogeneity], dtype=torch.float32, device=self.device)
            
            # --- THE FIX: Calculate loss in log space ---
            loss_glcm = F.l1_loss(torch.log1p(glcm_vec), self.glcm_target_mean)
            
            if log_stats:
                print(f"    - Current GLCM: contrast={contrast:.2f}, homogeneity={homogeneity:.4f} -> Loss: {loss_glcm.item():.4f}")

        except Exception as e:
            print(f"Warning: GLCM calculation failed: {e}")
            loss_glcm = torch.tensor(0.0, device=self.device)

        # 2. LBP Loss (LBP is already a normalized histogram, MSE is fine here)
        try:
            lbp = local_binary_pattern(gray_np, self.n_points, self.radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_bins, range=(0, self.lbp_bins), density=True)
            lbp_vec = torch.tensor(lbp_hist, dtype=torch.float32, device=self.device)
            loss_lbp = F.mse_loss(lbp_vec, self.lbp_target_mean)
        except Exception as e:
            print(f"Warning: LBP calculation failed: {e}")
            loss_lbp = torch.tensor(0.0, device=self.device)

        # Weight GLCM more heavily as it's a more robust indicator of overall texture
        total_loss = (loss_glcm * 10.0) + loss_lbp
        return total_loss