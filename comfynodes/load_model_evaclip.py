from safetensors.torch import load_file as load_safetensors
import folder_paths
import comfy.utils
import torch
from torch import nn

from ..eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from ..eva_clip.factory import create_model_and_transforms
from ..comfymodels.path import INSIGHTFACE_DIR, INSIGHTFACE_PATH, PULID_DIR, PULID_PATH, CLIP_DIR, CLIP_PATH, FACEDETECT_DIR, FACEDETECT_PATH, FACERESTORE_DIR, FACERESTORE_PATH



def load_eva_clip(self):
    model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
    model = model.visual
    eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
    eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)

    if not isinstance(eva_transform_mean, (list, tuple)):
        model["image_mean"] = (eva_transform_mean,) * 3
    if not isinstance(eva_transform_std, (list, tuple)):
        model["image_std"] = (eva_transform_std,) * 3
    return (model,)
    
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!! - TODO: MIGRATE TO ALLOW FOR LOADING FOLDERIZED VERSION AND IN SAFETENSORS
# !!!! - TODO: CONSIDER LOADING THROUGH CLIP VISION LOADER
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# def load_eva_clip(self, clip_file):
#     # Get the full path of the selected file
#     model_path = folder_paths.get_full_path(CLIP_DIR, clip_file)

#     # Check file extension to determine how to load it
#     if model_path.lower().endswith(".safetensors"):
#         print(f"Loading model from {model_path} (safetensors)")
#         model = load_safetensors(model_path)
#     elif model_path.lower().endswith(".pt") or model_path.lower().endswith(".pth"):
#         print(f"Loading model from {model_path} (torch)")
#         model = torch.load(model_path)
#     else:
#         raise ValueError(f"Unsupported file format: {model_path}")
    
#     # If the model is a dictionary (which is often the case), extract the 'visual' part
#     if isinstance(model, dict):
#         if 'visual' in model:
#             model = model['visual']  # Extract the 'visual' part, assuming this is what you need
#         else:
#             raise KeyError(f"Expected 'visual' key in the model dictionary, but it was not found.")
    
#     # Ensure 'model' is now a PyTorch module (i.e., nn.Module), not a dictionary
#     if not isinstance(model, torch.nn.Module):
#         raise TypeError(f"Loaded model is of type {type(model)}, expected a PyTorch model (nn.Module).")
    
#     # Adjust transformations if necessary
#     eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
#     eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)

#     if not isinstance(eva_transform_mean, (list, tuple)):
#         model.image_mean = (eva_transform_mean,) * 3
#     if not isinstance(eva_transform_std, (list, tuple)):
#         model.image_std = (eva_transform_std,) * 3
    
#     return (model,)