class OmniSimulator_Metadata_Inspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"profile_path": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("info",); FUNCTION = "inspect"; CATEGORY = "OmniSimulator/Authenticity"
    def inspect(self, profile_path):
        import os, json, numpy as np
        lines = []
        npz_path = f"{profile_path}.npz" if not profile_path.endswith(".npz") else profile_path
        json_path = f"{profile_path}.json" if not profile_path.endswith(".json") else profile_path
        if os.path.exists(npz_path):
            with np.load(npz_path) as data:
                lines.append(f"NPZ keys: {list(data.keys())}")
                for k in data.keys(): lines.append(f"  {k}: shape={data[k].shape}")
        if os.path.exists(json_path):
            with open(json_path) as f: meta = json.load(f)
            lines.append(f"JSON records: {len(meta) if isinstance(meta, list) else 1}")
            if isinstance(meta, list) and meta:
                lines.append(f"JSON keys: {list(meta[0].keys())[:10]}")
        return ("\n".join(lines),)

NODE_CLASS_MAPPINGS = {"OmniSimulator_Metadata_Inspector": OmniSimulator_Metadata_Inspector}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSimulator_Metadata_Inspector": "OmniSimulator Metadata Inspector"}