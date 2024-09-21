import folder_paths
import comfy.utils
from insightface.app import FaceAnalysis

from ..comfymodels.path import INSIGHTFACE_DIR, INSIGHTFACE_PATH, PULID_DIR, PULID_PATH, CLIP_DIR, CLIP_PATH, FACEDETECT_DIR, FACEDETECT_PATH, FACERESTORE_DIR, FACERESTORE_PATH

def load_model_insightface(self, provider):
    model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_PATH, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
    model.prepare(ctx_id=0, det_size=(640, 640))

    return (model,)