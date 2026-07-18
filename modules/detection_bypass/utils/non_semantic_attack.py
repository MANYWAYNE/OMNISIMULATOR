import sys as _sys, os as _os
_OmniSimulator_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.realpath(__file__)), '..', '..', '..'))
if _OmniSimulator_ROOT not in _sys.path: _sys.path.insert(0, _OmniSimulator_ROOT)
del _sys, _os, _OmniSimulator_ROOT

import torch
import torch.optim as optim
import numpy as np
import threading

from modules.detection_bypass.utils.perceptual_loss import build_perceptual_model


def _optimization_worker(img_np: np.ndarray, config: dict, device: torch.device, result_container: list, exception_container: list):
    """
    Runs the core UnMarker-style optimization in a separate thread.
    This is CRITICAL to escape the global torch.no_grad() context of ComfyUI.
    """
    try:
        print_log_every_n = config.get('print_log_every_n', 0)

        img_tensor = torch.from_numpy(img_np).to(device)
        img_tensor = (img_tensor - 0.5) / 0.5

        batch_size, channels, height, width = img_tensor.shape
        delta = torch.randn(batch_size, 1, height, width, device=device, dtype=img_tensor.dtype) * 0.01
        delta.requires_grad = True

        # Real lpips if installed, otherwise a dependency-free built-in proxy —
        # either way this never raises, so the attack always runs.
        perceptual_model, backend = build_perceptual_model(device)

        optimizer = optim.Adam([delta], lr=config['learning_rate'])

        img_gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
        with torch.no_grad():
            img_fft = torch.fft.fft2(img_gray)

        if print_log_every_n > 0:
            print(f"  [Spectral Normalizer] ✓ Thread started. Optimizing for {config['iterations']} iterations "
                  f"(perceptual backend: {backend})...")
            print(f"  [Params] LR={config['learning_rate']:.1E}, Perceptual(t={config['t_lpips']:.3f}, c={config['c_lpips']:.3f}), L2(t={config['t_l2']:.1E}, c={config['c_l2']:.2f}), GradClip={config['grad_clip_value']:.3f}")

        for i in range(config['iterations']):
            optimizer.zero_grad()

            delta_rgb = delta.expand_as(img_tensor)
            x_nw = torch.clamp(img_tensor + delta_rgb, -1, 1)

            x_nw_gray = 0.299 * x_nw[:, 0:1] + 0.587 * x_nw[:, 1:2] + 0.114 * x_nw[:, 2:3]
            x_nw_fft = torch.fft.fft2(x_nw_gray)
            fft_diff = x_nw_fft - img_fft
            loss_dfl = -torch.sqrt(fft_diff.real**2 + fft_diff.imag**2 + 1e-8).mean()  # Use mean for stability

            loss_perceptual = perceptual_model(x_nw, img_tensor).mean()
            loss_l2 = delta.pow(2).mean().sqrt()

            perceptual_penalty = config['c_lpips'] * torch.relu(loss_perceptual - config['t_lpips'])
            l2_penalty = config['c_l2'] * torch.relu(loss_l2 - config['t_l2'])
            total_loss = loss_dfl + perceptual_penalty + l2_penalty

            total_loss.backward()

            if delta.grad is not None:
                torch.nn.utils.clip_grad_norm_(delta, config['grad_clip_value'])

            optimizer.step()

            if print_log_every_n > 0 and (i % print_log_every_n == 0 or i == config['iterations'] - 1):
                print(f"    Iter {i:04d}: Total Loss={total_loss.item():.4f} | FFT Loss={loss_dfl.item():.4f} | Perceptual={loss_perceptual.item():.4f} | L2={loss_l2.item():.4f}")

        if print_log_every_n > 0:
            final_perceptual = perceptual_model(torch.clamp(img_tensor + delta.expand_as(img_tensor), -1, 1), img_tensor).mean().item()
            print(f"  [Spectral Normalizer] ✓ Optimization complete. Final perceptual distance: {final_perceptual:.4f}")

        delta_rgb = delta.expand_as(img_tensor)
        final_x_nw = torch.clamp(img_tensor + delta_rgb, -1, 1).squeeze(0).cpu().detach()
        final_x_nw = (final_x_nw + 1) / 2
        final_x_nw = final_x_nw.permute(1, 2, 0)
        result_np = (final_x_nw.clamp(0, 1) * 255).numpy().astype(np.uint8)
        result_container.append(result_np)

    except Exception as e:
        exception_container.append(e)


@torch.enable_grad()
def non_semantic_attack(img_arr: np.ndarray, **kwargs) -> np.ndarray:
    config = {
        'iterations': kwargs.get('iterations', 500), 'learning_rate': kwargs.get('learning_rate', 3e-4),
        't_lpips': kwargs.get('t_lpips', 0.04), 'c_lpips': kwargs.get('c_lpips', 1e-2),
        't_l2': kwargs.get('t_l2', 3e-5), 'c_l2': kwargs.get('c_l2', 0.6),
        'grad_clip_value': kwargs.get('grad_clip_value', 0.05),
        'print_log_every_n': kwargs.get('print_log_every_n', 0)
    }

    seed = kwargs.get('seed')
    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_np = (img_arr.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

    result_container, exception_container = [], []

    thread = threading.Thread(target=_optimization_worker, args=(img_np, config, device, result_container, exception_container))
    thread.start()
    thread.join()

    if exception_container:
        raise exception_container[0]
    if not result_container:
        raise RuntimeError("Spectral Normalizer optimization thread failed to return a result.")

    return result_container[0]