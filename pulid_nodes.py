import folder_paths

from .comfymodels.path import INSIGHTFACE_DIR, INSIGHTFACE_PATH, PULID_DIR, PULID_PATH, CLIP_DIR, CLIP_PATH, FACEDETECT_DIR, FACEDETECT_PATH, FACERESTORE_DIR, FACERESTORE_PATH
from .comfynodes.load_model_pulid import load_model_pulid
from .comfynodes.load_model_insightface import load_model_insightface
from .comfynodes.load_model_evaclip import load_eva_clip
from .comfynodes.apply_pulid import apply_pulid



class LoadModelPulID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "pulid_file": (folder_paths.get_filename_list(PULID_DIR), )
            }
        }

    RETURN_TYPES = ("PULID",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    load_model = load_model_pulid
    
    
    
class LoadModelInsightFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            }
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    load_model = load_model_insightface
    


class LoadModelEvaClip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "required": {},
        }

    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    load_model = load_eva_clip



class ApplyPulID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "pulid": ("PULID", ),
                "eva_clip": ("EVA_CLIP", ),
                "face_analysis": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
                "method": (["fidelity", "style", "neutral"],),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            },
            "optional": {
                "attn_mask": ("MASK", ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_modifier"
    CATEGORY = "pulid"

    apply_modifier = apply_pulid 


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "LoadModelPulID": LoadModelPulID,
    "LoadModelInsightFace": LoadModelInsightFace,
    "LoadModelEvaClip": LoadModelEvaClip,
    "ApplyPulID": ApplyPulID,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadModelPulID": "Load Model PuLID",
    "LoadModelInsightFace": "Load Model InsightFace (PuLID)",
    "LoadModelEvaClip": "Load Model Eva Clip (PuLID)",
    "ApplyPulID": "Apply PuLID",
}
