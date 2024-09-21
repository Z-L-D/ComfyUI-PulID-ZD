import folder_paths
import comfy.utils
from ..models.path import INSIGHTFACE_DIR, INSIGHTFACE_PATH, PULID_DIR, PULID_PATH, CLIP_DIR, CLIP_PATH, FACEDETECT_DIR, FACEDETECT_PATH, FACERESTORE_DIR, FACERESTORE_PATH

def load_model_pulid(self, pulid_file):
        ckpt_path = folder_paths.get_full_path(PULID_DIR, pulid_file)

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