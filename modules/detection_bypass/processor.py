import argparse
import os
from PIL import Image
import numpy as np
import piexif
from datetime import datetime

# Import all utility functions, including our new UnMarker attack
from .utils import (
    add_gaussian_noise,
    clahe_color_correction,
    randomized_perturbation,
    fourier_match_spectrum,
    auto_white_balance_ref,
    load_lut,
    apply_lut,
    glcm_normalize,
    lbp_normalize,
    attack_non_semantic, # <-- This is now imported from our new module
    blend_colors
)
from .camera_pipeline import simulate_camera_pipeline

def add_fake_exif():
    """Generates a plausible set of fake EXIF data."""
    now = datetime.now()
    datestamp = now.strftime("%Y:%m:%d %H:%M:%S")
    zeroth_ifd = {
        piexif.ImageIFD.Make: b"Apple",
        piexif.ImageIFD.Model: b"iPhone 15 Pro",
        piexif.ImageIFD.Software: b"17.4.1",
        piexif.ImageIFD.DateTime: datestamp.encode('utf-8'),
    }
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: datestamp.encode('utf-8'),
        piexif.ExifIFD.DateTimeDigitized: datestamp.encode('utf-8'),
        piexif.ExifIFD.ExposureTime: (1, 125),
        piexif.ExifIFD.FNumber: (28, 10),
        piexif.ExifIFD.ISOSpeedRatings: 200,
        piexif.ExifIFD.FocalLength: (50, 1),
    }
    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": {}, "1st": {}, "thumbnail": None}
    return piexif.dump(exif_dict)

def process_image(path_in, path_out, args):
    """The main processing function that applies a sequence of effects based on args."""
    img = Image.open(path_in).convert('RGB')
    arr = np.array(img)

    ref_arr_fft = None
    if args.fft_ref:
        try:
            ref_arr_fft = np.array(Image.open(args.fft_ref).convert('RGB'))
        except Exception as e:
            print(f"Warning: failed to load FFT/AWB reference '{args.fft_ref}': {e}.")

    # --- This is the core change: we now call the imported UnMarker function ---
    if args.non_semantic:
        print("Applying non-semantic UnMarker attack...")
        try:
            arr = attack_non_semantic(
                arr,
                iterations=args.ns_iterations, learning_rate=args.ns_learning_rate,
                t_lpips=args.ns_t_lpips, t_l2=args.ns_t_l2,
                c_lpips=args.ns_c_lpips, c_l2=args.ns_c_l2,
                grad_clip_value=args.ns_grad_clip
            )
        except Exception as e:
            print(f"Warning: Non-semantic attack failed: {e}. Skipping.")

    # ... (the rest of the function remains the same, applying other effects)
    if args.clahe:
        arr = clahe_color_correction(arr, clip_limit=args.clahe_clip, tile_grid_size=(args.tile, args.tile))
    if args.fft:
        arr = fourier_match_spectrum(arr, ref_img_arr=ref_arr_fft, mode=args.fft_mode, alpha=args.fft_alpha, cutoff=args.cutoff, strength=args.fstrength, randomness=args.randomness, phase_perturb=args.phase_perturb, radial_smooth=args.radial_smooth, seed=args.seed)
    if args.glcm:
        arr = glcm_normalize(arr, ref_img_arr=ref_arr_fft, distances=args.glcm_distances, angles=args.glcm_angles, levels=args.glcm_levels, strength=args.glcm_strength, seed=args.seed)
    if args.lbp:
        arr = lbp_normalize(arr, ref_img_arr=ref_arr_fft, radius=args.lbp_radius, n_points=args.lbp_n_points, method=args.lbp_method, strength=args.lbp_strength, seed=args.seed)
    if args.noise:
        arr = add_gaussian_noise(arr, std_frac=args.noise_std, seed=args.seed)
    if args.perturb:
        arr = randomized_perturbation(arr, magnitude_frac=args.perturb_magnitude, seed=args.seed)
    if args.sim_camera:
        arr = simulate_camera_pipeline(arr, bayer=args.no_no_bayer, jpeg_cycles=args.jpeg_cycles, jpeg_quality_range=(args.jpeg_qmin, args.jpeg_qmax), vignette_strength=args.vignette_strength, chroma_aberr_strength=args.chroma_strength, iso_scale=args.iso_scale, read_noise_std=args.read_noise, hot_pixel_prob=args.hot_pixel_prob, banding_strength=args.banding_strength, motion_blur_kernel=args.motion_blur_kernel, seed=args.seed)
    if args.awb:
        awb_ref = np.array(Image.open(args.ref).convert('RGB')) if args.ref else None
        arr = auto_white_balance_ref(arr, awb_ref)
    if args.lut:
        try:
            lut = load_lut(args.lut)
            arr = apply_lut(np.clip(arr, 0, 255).astype(np.uint8), lut, strength=args.lut_strength)
        except Exception as e:
            print(f"Warning: failed to apply LUT '{args.lut}': {e}. Skipping.")
    
    out_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    out_img.save(path_out, exif=add_fake_exif())

# The build_argparser and main functions are for standalone script execution and can remain unchanged.
# ... (rest of processor.py, i.e., build_argparser and if __name__ == "__main__":, remains the same)
def build_argparser():
    p = argparse.ArgumentParser(description="Image postprocessing pipeline with camera simulation, LUT support, GLCM, and LBP normalization")
    p.add_argument('input', help='Input image path')
    p.add_argument('output', help='Output image path')
    p.add_argument('--awb', action='store_true', default=False, help='Enable automatic white balancing. Uses grey-world if --ref is not provided.')
    p.add_argument('--ref', help='Optional reference image for auto white-balance (only used if --awb is enabled)', default=None)
    p.add_argument('--noise-std', type=float, default=0.02, help='Gaussian noise std fraction of 255 (0-0.1)')
    p.add_argument('--clahe-clip', type=float, default=2.0, help='CLAHE clip limit')
    p.add_argument('--tile', type=int, default=8, help='CLAHE tile grid size')
    p.add_argument('--cutoff', type=float, default=0.25, help='Fourier cutoff (0..1)')
    p.add_argument('--fstrength', type=float, default=0.9, help='Fourier blend strength (0..1)')
    p.add_argument('--randomness', type=float, default=0.05, help='Randomness for Fourier mask modulation')
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    p.add_argument('--fft-ref', help='Optional reference image for FFT spectral matching, GLCM, and LBP', default=None)
    p.add_argument('--fft-mode', choices=('auto','ref','model'), default='auto', help='FFT mode: auto picks ref if available')
    p.add_argument('--fft-alpha', type=float, default=1.0, help='Alpha for 1/f model (spectrum slope)')
    p.add_argument('--phase-perturb', type=float, default=0.08, help='Phase perturbation strength (radians)')
    p.add_argument('--radial-smooth', type=int, default=5, help='Radial smoothing (bins) for spectrum profiles')
    p.add_argument('--glcm', action='store_true', default=False, help='Enable GLCM normalization using FFT reference if available')
    p.add_argument('--glcm-distances', type=int, nargs='+', default=[1], help='Distances for GLCM computation')
    p.add_argument('--glcm-angles', type=float, nargs='+', default=[0, np.pi/4, np.pi/2, 3*np.pi/4], help='Angles for GLCM computation (in radians)')
    p.add_argument('--glcm-levels', type=int, default=256, help='Number of gray levels for GLCM')
    p.add_argument('--glcm-strength', type=float, default=0.9, help='Strength of GLCM feature matching (0..1)')
    p.add_argument('--lbp', action='store_true', default=False, help='Enable LBP normalization using FFT reference if available')
    p.add_argument('--lbp-radius', type=int, default=3, help='Radius of LBP operator')
    p.add_argument('--lbp-n-points', type=int, default=24, help='Number of circularly symmetric neighbor set points for LBP')
    p.add_argument('--lbp-method', choices=('default', 'ror', 'uniform', 'var'), default='uniform', help='LBP method')
    p.add_argument('--lbp-strength', type=float, default=0.9, help='Strength of LBP histogram matching (0..1)')
    p.add_argument('--non-semantic', action='store_true', default=False, help='Apply non-semantic attack on the image')
    p.add_argument('--ns-iterations', type=int, default=500, help='Iterations for non-semantic attack')
    p.add_argument('--ns-learning-rate', type=float, default=3e-4, help='Learning rate for non-semantic attack')
    p.add_argument('--ns-t-lpips', type=float, default=4e-2, help='LPIPS threshold for non-semantic attack')
    p.add_argument('--ns-t-l2', type=float, default=3e-5, help='L2 threshold for non-semantic attack')
    p.add_argument('--ns-c-lpips', type=float, default=1e-2, help='LPIPS constant for non-semantic attack')
    p.add_argument('--ns-c-l2', type=float, default=0.6, help='L2 constant for non-semantic attack')
    p.add_argument('--ns-grad-clip', type=float, default=0.05, help='Gradient clipping value for non-semantic attack')
    p.add_argument('--sim-camera', action='store_true', default=False, help='Enable camera-pipeline simulation (Bayer, CA, vignette, JPEG cycles)')
    p.add_argument('--no-no-bayer', dest='no_no_bayer', action='store_false', help='Disable Bayer/demosaic step (double negative kept for backward compat)')
    p.set_defaults(no_no_bayer=True)
    p.add_argument('--jpeg-cycles', type=int, default=1, help='Number of JPEG recompression cycles to apply')
    p.add_argument('--jpeg-qmin', type=int, default=88, help='Min JPEG quality for recompression')
    p.add_argument('--jpeg-qmax', type=int, default=96, help='Max JPEG quality for recompression')
    p.add_argument('--vignette-strength', type=float, default=0.35, help='Vignette strength (0..1)')
    p.add_argument('--chroma-strength', type=float, default=1.2, help='Chromatic aberration strength (pixels)')
    p.add_argument('--iso-scale', type=float, default=1.0, help='ISO/exposure scale for Poisson noise')
    p.add_argument('--read-noise', type=float, default=2.0, help='Read noise sigma for sensor noise')
    p.add_argument('--hot-pixel-prob', type=float, default=1e-6, help='Per-pixel probability of hot pixel')
    p.add_argument('--banding-strength', type=float, default=0.0, help='Horizontal banding amplitude (0..1)')
    p.add_argument('--motion-blur-kernel', type=int, default=1, help='Motion blur kernel size (1 = none)')
    p.add_argument('--lut', type=str, default=None, help='Path to a 1D PNG (256x1) or .npy LUT, or a .cube 3D LUT')
    p.add_argument('--lut-strength', type=float, default=0.1, help='Strength to blend LUT (0.0 = no effect, 1.0 = full LUT)')
    p.add_argument('--noise', action='store_true', default=False, help='Enable Gaussian noise addition')
    p.add_argument('--clahe', action='store_true', default=False, help='Enable CLAHE color correction')
    p.add_argument('--fft', action='store_true', default=False, help='Enable FFT spectral matching')
    p.add_argument('--perturb', action='store_true', default=False, help='Enable randomized perturbation')
    p.add_argument('--perturb-magnitude', type=float, default=0.008, help='Randomized perturb magnitude fraction (0..0.05)')
    p.add_argument('--blend', action='store_true', default=False, help='Enable color')
    p.add_argument('--blend-tolerance', type=float, default=10.0, help='Color tolerance for blending (smaller = more colors)')
    p.add_argument('--blend-min-region', type=int, default=50, help='Minimum region size to retain (in pixels)')
    p.add_argument('--blend-max-samples', type=int, default=100000, help='Maximum pixels to sample for k-means (for speed)')
    p.add_argument('--blend-n-jobs', type=int, default=None, help='Number of worker threads for blending (default: os.cpu_count())')
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not os.path.exists(args.input):
        print("Input not found:", args.input)
        raise SystemExit(2)
    process_image(args.input, args.output, args)
    print("Saved:", args.output)