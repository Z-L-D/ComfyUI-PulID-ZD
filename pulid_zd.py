import os
import folder_paths

from PIL import Image
import torch
import numpy as np
import comfy.utils
from comfy.ldm.modules.attention import optimized_attention

from insightface.app import FaceAnalysis
from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
# from facexlib.parsing import init_parsing_model
# from facexlib.utils.face_restoration_helper import FaceRestoreHelper

import sys
# Define the path to your custom nodes folder
custom_nodes_path = r"C:\Production\Applied Science\Software\PYTHON\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-pulid-zd"

# Append the path if it's not already in sys.path
if custom_nodes_path not in sys.path:
    sys.path.append(custom_nodes_path)

from .utils.models import MODEL_INSIGHTFACE_DIR, INSIGHTFACE_DIR, MODEL_PULID_DIR, PULID_DIR, MODEL_CLIP_DIR, CLIP_DIR, MODEL_FACEDETECT, MODEL_FACERESTORE
from .utils.pipeline_comfyflux import PulidModel



# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(
        np.array(image).astype(np.float32) / 255.0
    ).unsqueeze(0)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class PulidModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pulid_file": (folder_paths.get_filename_list(PULID_DIR), )}}

    RETURN_TYPES = ("PULID",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    def load_model(self, pulid_file):
        ckpt_path = folder_paths.get_full_path(MODEL_PULID_DIR, pulid_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model
        
        # Also initialize the model, takes longer to load but then it doesn't have to be done every time you change parameters in the apply node
        model = PulidModel(model)

        return (model,)
    
    
class PulidInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insightface"
    CATEGORY = "pulid"

    def load_insightface(self, provider):
        model = FaceAnalysis(name="antelopev2", root=MODEL_INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)
    

class PulidEvaClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load_eva_clip"
    CATEGORY = "pulid"

    def load_eva_clip(self):
        from .eva_clip.factory import create_model_and_transforms

        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)

        model = model.visual

        eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            model["image_mean"] = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            model["image_std"] = (eva_transform_std,) * 3

        return (model,)
    

class ImageGetWidthHeight:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_width_height"
    CATEGORY = "image"

    def get_width_height(self, image):
        pil_image = tensor2pil(image)
        width, height = pil_image.size
        return (width, height)



# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ImageGetWidthHeight": ImageGetWidthHeight,
    "PulidModelLoader": PulidModelLoader,
    "PulidInsightFaceLoader": PulidInsightFaceLoader,
    # "PulidEvaClipLoader": PulidEvaClipLoader,
    # "ApplyPulid": ApplyPulid,
    # "ApplyPulidAdvanced": ApplyPulidAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageGetWidthHeight": "Image Get Width and Height",
    "PulidModelLoader": "Load PuLID Model",
    "PulidInsightFaceLoader": "Load InsightFace (PuLID)",
    # "PulidEvaClipLoader": "Load Eva Clip (PuLID)",
    # "ApplyPulid": "Apply PuLID",
    # "ApplyPulidAdvanced": "Apply PuLID Advanced",
}
