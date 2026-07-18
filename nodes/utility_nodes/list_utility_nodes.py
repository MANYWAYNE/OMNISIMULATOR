import torch
class OmniSimulator_BatchFromImageList:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"images": ("IMAGE",)}}
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",); FUNCTION = "batch"; CATEGORY = "OmniSimulator/Utils"
    def batch(self, images):
        import torch
        return (torch.cat(images, dim=0),)

class OmniSimulator_ImageListFromBatch:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",); OUTPUT_IS_LIST = (True,); FUNCTION = "split"; CATEGORY = "OmniSimulator/Utils"
    def split(self, image): return ([image[i:i+1] for i in range(image.shape[0])],)

class OmniSimulator_PickFromList:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"list": ("*",), "index": ("INT", {"default": 0, "min": 0, "max": 9999})}}
    INPUT_IS_LIST = True; RETURN_TYPES = ("*",); FUNCTION = "pick"; CATEGORY = "OmniSimulator/Utils"
    def pick(self, list, index): return (list[min(index[0], len(list)-1)],)

class OmniSimulator_StringListFromStrings:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"string_1": ("STRING", {"default": ""}), "string_2": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("STRING",); OUTPUT_IS_LIST = (True,); FUNCTION = "make"; CATEGORY = "OmniSimulator/Utils"
    def make(self, **kwargs): return ([v for v in kwargs.values() if v],)