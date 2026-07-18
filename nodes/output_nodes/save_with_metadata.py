import sys, os, traceback

_DIR  = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.normpath(os.path.join(_DIR, '..', '..'))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

import torch
import numpy as np
from PIL import Image
import json
import random
import folder_paths
import tempfile
import shutil
import uuid
import subprocess
import datetime
from pathlib import Path

_NODE_DIR       = Path(_ROOT)
_LOCATIONS_FILE = _NODE_DIR / "modules" / "location_data" / "locations.json"
_SRGB_ICC       = _NODE_DIR / "modules" / "color_profiles" / "sRGB_IEC61966-2-1_no_black_scaling.icc"

def _is_tool(name): return shutil.which(name) is not None
EXIFTOOL = _is_tool("exiftool") or _is_tool("exiftool.exe")

_locations_data   = None
_location_choices = None

def _load_locations():
    global _locations_data
    if _locations_data is not None:
        return _locations_data
    try:
        if not _LOCATIONS_FILE.exists():
            _LOCATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            default = {
                "North America": {"United States": [
                    {"city": "New York",    "lat": [40.4774, 40.9176], "lon": [-74.2591, -73.7002]},
                    {"city": "Los Angeles", "lat": [33.7037, 34.3373], "lon": [-118.6682, -118.1553]},
                    {"city": "Chicago",     "lat": [41.6440, 42.0230], "lon": [-87.9401, -87.5241]},
                    {"city": "Miami",       "lat": [25.7093, 25.9785], "lon": [-80.3197, -80.1397]},
                    {"city": "Houston",     "lat": [29.5237, 30.1107], "lon": [-95.7742, -95.0141]},
                ]},
                "Europe": {
                    "France":         [{"city": "Paris",    "lat": [48.8156, 48.9021], "lon": [2.2241,  2.4699]}],
                    "United Kingdom": [{"city": "London",   "lat": [51.2868, 51.6919], "lon": [-0.5103, 0.3340]}],
                    "Germany":        [{"city": "Berlin",   "lat": [52.3382, 52.6755], "lon": [13.0882, 13.7611]}],
                    "Italy":          [{"city": "Rome",     "lat": [41.7949, 42.0140], "lon": [12.3406, 12.6166]}],
                    "Spain":          [{"city": "Barcelona","lat": [41.3200, 41.4690], "lon": [2.0695,  2.2285]}],
                    "Romania":        [{"city": "Bucharest","lat": [44.3369, 44.5436], "lon": [25.9699, 26.2264]}],
                },
            }
            with open(_LOCATIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(default, f, indent=2)
            _locations_data = default
        else:
            with open(_LOCATIONS_FILE, 'r', encoding='utf-8') as f:
                _locations_data = json.load(f)
    except Exception as e:
        print(f"[OmniSimulator] Could not load locations.json: {e}")
        _locations_data = {}
    return _locations_data

def _get_location_choices():
    global _location_choices
    if _location_choices is not None:
        return _location_choices
    data = _load_locations()
    specials = ["Synthesize Random (US City)", "Synthesize Random (Any City)",
                "Synthesize Random", "From Profile"]
    try:
        countries = sorted({c for r in data.values() for c in r})
        random_country = [f"Synthesize Random ({c})" for c in countries if c != "United States"]
        cities = sorted(
            f"{city['city']}, {country}"
            for region in data.values()
            for country, city_list in region.items()
            for city in city_list
        )
    except Exception:
        random_country, cities = [], []
    _location_choices = specials + random_country + cities
    return _location_choices

def _city_bounds(location_choice, data, rng):
    lc = location_choice or "Synthesize Random (US City)"
    if lc == "From Profile":
        return None
    if lc == "Synthesize Random":
        lc = "Synthesize Random (Any City)"
    if lc == "Synthesize Random (Any City)":
        all_cities = [c for r in data.values() for cl in r.values() for c in cl]
        return rng.choice(all_cities) if all_cities else None
    if lc.startswith("Synthesize Random ("):
        country = lc[len("Synthesize Random ("):-1]
        if country == "US City":
            country = "United States"
        cities = [c for r in data.values() for cn, cl in r.items() if cn == country for c in cl]
        return rng.choice(cities) if cities else None
    parts = [p.strip() for p in lc.split(",", 1)]
    if len(parts) == 2:
        city_name, country_name = parts
        for r in data.values():
            if country_name in r:
                for c in r[country_name]:
                    if c["city"] == city_name:
                        return c
    return None

SCENE_TYPES = [
    "Synthesize Random",
    "From Profile",
    "Daylight (Bright)",
    "Daylight (Cloudy)",
    "Golden Hour",
    "Indoors",
    "Night",
]

DATETIME_MODES = ["Synthesize Random", "From Profile"]

def _scene_tags(scene, rng):
    if scene in ("From Profile", "", None):
        return {}
    if scene == "Synthesize Random":
        scene = rng.choice([s for s in SCENE_TYPES if s not in ("Synthesize Random", "From Profile")])
    presets = {
        "Daylight (Bright)": {"EXIF:ISO": rng.choice([25, 32, 50]),
                              "EXIF:ExposureTime": 1 / rng.randint(1000, 4000),
                              "EXIF:FNumber": 2.2},
        "Daylight (Cloudy)": {"EXIF:ISO": rng.choice([64, 100, 125]),
                              "EXIF:ExposureTime": 1 / rng.randint(250, 1000),
                              "EXIF:FNumber": 2.2},
        "Golden Hour":       {"EXIF:ISO": rng.choice([100, 200]),
                              "EXIF:ExposureTime": 1 / rng.randint(60, 500),
                              "EXIF:FNumber": 2.0},
        "Indoors":           {"EXIF:ISO": rng.choice([400, 800]),
                              "EXIF:ExposureTime": 1 / rng.randint(30, 120),
                              "EXIF:FNumber": 1.8},
        "Night":             {"EXIF:ISO": rng.choice([800, 1600, 2500]),
                              "EXIF:ExposureTime": 1 / rng.randint(15, 60),
                              "EXIF:FNumber": 1.8},
    }
    return presets.get(scene, {})

class OmniSimulator_SaveWithMetadata:
    OUTPUT_NODE = True
    CATEGORY    = "OmniSimulator/Authenticity"
    FUNCTION    = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        now             = datetime.datetime.now()
        default_end     = now.strftime("%Y-%m-%dT%H:%M:%S")
        default_start   = (now - datetime.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        loc_choices     = _get_location_choices()
        default_loc     = "Synthesize Random (US City)"
        if default_loc not in loc_choices:
            default_loc = loc_choices[0] if loc_choices else "From Profile"

        return {
            "required": {
                "image":      ("IMAGE",),
                "seed":       ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "profile_path":      ("STRING", {"forceInput": True,
                                                  "tooltip": "Connect OmniSimulator Authenticity Profile Selector here."}),
                "output_subfolder":  ("STRING", {"default": "OmniSimulator",
                                                  "tooltip": "Subfolder inside ComfyUI output/"}),
                "filename_prefix":   ("STRING", {"default": "IMG"}),
                "jpeg_quality":      ("INT",    {"default": 95, "min": 1, "max": 100,
                                                  "tooltip": "JPEG quality 1-100."}),
                "inject_icc_profile":("BOOLEAN",{"default": False,
                                                  "label_on": "Embed sRGB ICC",
                                                  "label_off": "Tag sRGB only"}),
                "location_synthesis":(loc_choices, {"default": default_loc}),
                "datetime_synthesis":(DATETIME_MODES, {"default": "Synthesize Random"}),
                "start_date":        ("STRING", {"default": default_start}),
                "end_date":          ("STRING", {"default": default_end}),
                "scene_type":        (SCENE_TYPES, {"default": "Synthesize Random"}),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("filepath",)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @staticmethod
    def _coerce(value, choices, default):
        return value if (value and value in choices) else default

    @staticmethod
    def _run_exiftool(temp_path, tags_dict, icc_path=None):
        cmd = shutil.which("exiftool") or shutil.which("exiftool.exe")
        if not cmd:
            raise RuntimeError("ExifTool not found in PATH.")

        arg_fd, arg_path = tempfile.mkstemp(suffix=".txt")
        try:
            with os.fdopen(arg_fd, 'w', encoding='utf-8') as af:
                for k, v in tags_dict.items():
                    af.write(f"-{k}={v}\n")
            command = [cmd, "-n", "-@", arg_path]
            if icc_path:
                command.append(f"-ICC_Profile<={icc_path}")
            command += ["-overwrite_original", "-ignoreMinorErrors", temp_path]
            result = subprocess.run(command, capture_output=True, text=True,
                                    encoding='utf-8', errors='ignore')
            if result.returncode not in (0, 1):
                print(f"[OmniSimulator ExifTool] stderr: {result.stderr.strip()[:400]}")
        finally:
            if os.path.exists(arg_path):
                os.remove(arg_path)

    def execute(self, image, seed,
                profile_path=None,
                output_subfolder="OmniSimulator",
                filename_prefix="IMG",
                jpeg_quality=95,
                inject_icc_profile=False,
                location_synthesis=None,
                datetime_synthesis=None,
                start_date=None,
                end_date=None,
                scene_type=None):

        loc_choices = _get_location_choices()
        default_loc = "Synthesize Random (US City)" if "Synthesize Random (US City)" in loc_choices \
                      else (loc_choices[0] if loc_choices else "From Profile")
        location_synthesis = self._coerce(location_synthesis, loc_choices,    default_loc)
        datetime_synthesis = self._coerce(datetime_synthesis, DATETIME_MODES, "Synthesize Random")
        scene_type         = self._coerce(scene_type,         SCENE_TYPES,    "Synthesize Random")
        start_date = start_date or (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        end_date   = end_date   or datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        try:
            jpeg_quality = int(jpeg_quality)
        except (TypeError, ValueError):
            jpeg_quality = 95
        jpeg_quality = max(1, min(100, jpeg_quality))

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = 0
        seed = seed % (2**32)

        exiftool_available = EXIFTOOL
        if not exiftool_available:
            print("[OmniSimulator] ExifTool not found — saving image(s) WITHOUT metadata injection. "
                  "Install from https://exiftool.org to enable EXIF/GPS/ICC synthesis.")

        output_dir = folder_paths.get_output_directory()
        subfolder_clean = (output_subfolder or "").strip()
        prefix = os.path.join(subfolder_clean, filename_prefix or "IMG") if subfolder_clean else (filename_prefix or "IMG")
        data = _load_locations()

        results        = []
        first_filepath = ""

        for i in range(image.shape[0]):
            img_np  = image[i].cpu().numpy()
            img_pil = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8), 'RGB')
            w, h    = img_pil.size

            full_folder, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
                prefix, output_dir, w, h)
            final_path = os.path.join(full_folder, f"{fname}_{counter:05}_.jpg")

            rng = random.Random(seed + counter)

            tags = {}
            json_path = None
            if profile_path and profile_path.strip():
                jp = profile_path if profile_path.endswith('.json') else f"{profile_path}.json"
                if os.path.exists(jp):
                    try:
                        with open(jp, 'r', encoding='utf-8') as f:
                            lib = json.load(f)
                        tags = dict(rng.choice(lib)) if lib else {}
                        json_path = jp
                    except Exception as e:
                        print(f"[OmniSimulator] Could not read profile '{jp}': {e}")

            _SKIP_PREFIXES = ("File:", "Composite:", "JFIF:", "_", "ExifTool:", "SourceFile")
            _SKIP_SUFFIXES = ("Orientation", "ColorSpace", "ThumbnailImage")
            tags = {k: v for k, v in tags.items()
                    if not k.startswith(_SKIP_PREFIXES)
                    and k.split(":", 1)[-1] not in _SKIP_SUFFIXES
                    and isinstance(v, (str, int, float, bool))}

            if json_path and not any(k.split(":", 1)[-1] in ("Make", "Model") for k in tags):
                device_slug = os.path.splitext(os.path.basename(json_path))[0]
                device_name = "iPhone " + device_slug.replace("iphone_", "").replace("_", " ").title()
                tags.setdefault("EXIF:Make", "Apple")
                tags.setdefault("EXIF:Model", device_name)
                tags.setdefault("EXIF:Software", rng.choice(["17.4.1", "17.5.1", "17.6.1", "18.0", "18.1.1"]))
                print(f"[OmniSimulator] Profile record had no Make/Model — using baseline '{device_name}' instead.")

            icc_file_to_inject = None
            if inject_icc_profile:
                if _SRGB_ICC.exists():
                    icc_file_to_inject = str(_SRGB_ICC)
                if json_path:
                    device_icc = os.path.splitext(json_path)[0] + ".icc"
                    if os.path.exists(device_icc):
                        icc_file_to_inject = device_icc

            tags.update({
                "EXIF:ExifImageWidth":  w, "EXIF:ExifImageHeight": h,
                "XMP:RegionAppliedToDimensionsW": w,
                "XMP:RegionAppliedToDimensionsH": h,
                "EXIF:Orientation": 1,
                "EXIF:ColorSpace":  1,
            })

            if "MakerNotes:PhotoIdentifier" in tags:
                tags["MakerNotes:PhotoIdentifier"] = str(uuid.uuid4()).upper()

            city = _city_bounds(location_synthesis, data, rng)
            if city:
                clat = (city["lat"][0] + city["lat"][1]) / 2
                clon = (city["lon"][0] + city["lon"][1]) / 2
                lat  = clat + rng.gauss(0, 0.01)
                lon  = clon + rng.gauss(0, 0.01)
                tags.update({
                    "EXIF:GPSLatitude":     abs(lat),
                    "EXIF:GPSLatitudeRef":  "N" if lat >= 0 else "S",
                    "EXIF:GPSLongitude":    abs(lon),
                    "EXIF:GPSLongitudeRef": "E" if lon >= 0 else "W",
                    "EXIF:GPSAltitude":     rng.uniform(10, 200),
                    "EXIF:GPSAltitudeRef":  0,
                })

            if datetime_synthesis == "Synthesize Random":
                try:
                    t0  = datetime.datetime.fromisoformat(start_date)
                    t1  = datetime.datetime.fromisoformat(end_date)
                    if t1 < t0:
                        t0, t1 = t1, t0
                    dt  = t0 + (t1 - t0) * rng.random()
                    ds  = dt.strftime("%Y:%m:%d %H:%M:%S")
                    sub = rng.randint(100, 999)
                    tags.update({
                        "EXIF:DateTimeOriginal":    ds,
                        "EXIF:CreateDate":          ds,
                        "EXIF:ModifyDate":          ds,
                        "EXIF:SubSecTimeOriginal":  sub,
                        "EXIF:SubSecTimeDigitized": sub,
                    })
                except Exception as e:
                    print(f"[OmniSimulator] datetime synthesis failed: {e}")

            tags.update(_scene_tags(scene_type, rng))

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(tmp_fd)
            saved_ok = False
            try:
                img_pil.save(tmp_path, format="JPEG", quality=jpeg_quality)
                if exiftool_available:
                    self._run_exiftool(tmp_path, tags, icc_path=icc_file_to_inject)
                shutil.move(tmp_path, final_path)
                saved_ok = True
            except Exception:
                traceback.print_exc()
                if os.path.exists(tmp_path) and not saved_ok:
                    try:
                        shutil.move(tmp_path, final_path)
                    except Exception:
                        pass
            finally:
                if os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except Exception: pass

            if i == 0:
                first_filepath = final_path

            results.append({
                "filename": os.path.basename(final_path),
                "subfolder": subfolder,
                "type": "output",
            })

            print(f"[OmniSimulator] Saved: {os.path.basename(final_path)}")

        return {"ui": {"images": results}, "result": (first_filepath,)}

NODE_CLASS_MAPPINGS = {
    "OmniSimulator_SaveWithMetadata": OmniSimulator_SaveWithMetadata,
    "OmniSimulator_SaveWithAuthenticMetadata":   OmniSimulator_SaveWithMetadata,
    "OmniSimulator_SynthesizeAuthenticMetadata": OmniSimulator_SaveWithMetadata,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniSimulator_SaveWithMetadata":             "OmniSimulator Save With Metadata",
    "OmniSimulator_SaveWithAuthenticMetadata":    "OmniSimulator Save With Metadata",
    "OmniSimulator_SynthesizeAuthenticMetadata":  "OmniSimulator Save With Metadata",
}