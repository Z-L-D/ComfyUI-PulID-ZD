import folder_paths

import comfy.utils
from ..comfymodels.path import INSIGHTFACE_DIR, INSIGHTFACE_PATH, PULID_DIR, PULID_PATH, CLIP_DIR, CLIP_PATH, FACEDETECT_DIR, FACEDETECT_PATH, FACERESTORE_DIR, FACERESTORE_PATH
from ..utils.pulid_pipeline_sdxl import PulidModelSDXL
from ..utils.pulid_pipeline_flux import PulidModelFlux


def load_model_pulid(self, pulid_file):
    ckpt_path = folder_paths.get_full_path(PULID_DIR, pulid_file)

    model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

    if(pulid_file == "ip-adapter_pulid_sdxl_fp16.safetensors"):
        model = PulidModelSDXL(model)
        
    elif(pulid_file == "pulid_flux_v0.9.0.safetensors"):
        model = PulidModelFlux(model)
    

    return (model, pulid_file,)