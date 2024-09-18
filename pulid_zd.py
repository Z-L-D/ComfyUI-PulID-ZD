import os
import folder_paths

from PIL import Image
import torch
import numpy as np
import comfy.utils

import sys
# Define the path to your custom nodes folder
# custom_nodes_path = r"C:\Production\Applied Science\Software\PYTHON\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-pulid-zd"

# Append the path if it's not already in sys.path
if custom_nodes_path not in sys.path:
    sys.path.append(custom_nodes_path)

from insightface.app import FaceAnalysis
# from facexlib.parsing import init_parsing_model
# from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .utils.models import MODEL_INSIGHTFACE_DIR, INSIGHTFACE_DIR, MODEL_PULID_DIR, PULID_DIR, MODEL_CLIP_DIR, CLIP_DIR, MODEL_FACEDETECT, MODEL_FACERESTORE
from .utils.pipeline_comfyflux import PulidModel, To_KV, tensor_to_image, image_to_tensor, tensor_to_size, set_model_patch_replace, Attn2Replace, pulid_attention, to_gray



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


class ApplyPulid:
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
    FUNCTION = "apply_pulid"
    CATEGORY = "pulid"

    def apply_pulid(self, model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
        work_model = model.clone()
        
        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        eva_clip.to(device, dtype=dtype)
        pulid_model = pulid.to(device, dtype=dtype)

        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.to(device, dtype=dtype)

        if method == "fidelity" or projection == "ortho_v2":
            num_zero = 8
            ortho = False
            ortho_v2 = True
        elif method == "style" or projection == "ortho":
            num_zero = 16
            ortho = True
            ortho_v2 = False
        else:
            num_zero = 0
            ortho = False
            ortho_v2 = False
        
        if fidelity is not None:
            num_zero = fidelity

        #face_analysis.det_model.input_size = (640,640)
        image = tensor_to_image(image)

        # face_helper = FaceRestoreHelper(
        #     upscale_factor=1,
        #     face_size=512,
        #     crop_ratio=(1, 1),
        #     det_model='retinaface_resnet50',
        #     save_ext='png',
        #     device=device,
        # )

        # face_helper.face_parse = None
        # face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond = []
        uncond = []

        for i in range(image.shape[0]):
            # get insightface embeddings
            iface_embeds = None
            for size in [(size, size) for size in range(640, 256, -64)]:
                face_analysis.det_model.input_size = size
                face = face_analysis.get(image[i])
                if face:
                    face = sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[-1]
                    iface_embeds = torch.from_numpy(face.embedding).unsqueeze(0).to(device, dtype=dtype)
                    break
            else:
                # No face detected, skip this image
                print('Warning: No face detected in image', i)
                continue

            # get eva_clip embeddings
            # face_helper.clean_all()
            # face_helper.read_image(image[i])
            # face_helper.get_face_landmarks_5(only_center_face=True)
            # face_helper.align_warp_face()

            # if len(face_helper.cropped_faces) == 0:
            #     # No face detected, skip this image
            #     continue
            
            # face = face_helper.cropped_faces[0]
            # face = image_to_tensor(face).unsqueeze(0).permute(0,3,1,2).to(device)
            # parsing_out = face_helper.face_parse(T.functional.normalize(face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            # parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            # bg = sum(parsing_out == i for i in bg_label).bool()
            # white_image = torch.ones_like(face)
            # face_features_image = torch.where(bg, white_image, to_gray(face))
            # # apparently MPS only supports NEAREST interpolation?
            # face_features_image = T.functional.resize(face_features_image, eva_clip.image_size, T.InterpolationMode.BICUBIC if 'cuda' in device.type else T.InterpolationMode.NEAREST).to(device, dtype=dtype)
            # face_features_image = T.functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)
            
            # id_cond_vit, id_vit_hidden = eva_clip(face_features_image, return_all_features=False, return_hidden=True, shuffle=False)
            id_cond_vit = id_cond_vit.to(device, dtype=dtype)
            # for idx in range(len(id_vit_hidden)):
            #     id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

            # combine embeddings
            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
            if noise == 0:
                id_uncond = torch.zeros_like(id_cond)
            else:
                id_uncond = torch.rand_like(id_cond) * noise
            id_vit_hidden_uncond = []
            # for idx in range(len(id_vit_hidden)):
                # if noise == 0:
                    # id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[idx]))
                # else:
                    # id_vit_hidden_uncond.append(torch.rand_like(id_vit_hidden[idx]) * noise)
            
            # cond.append(pulid_model.get_image_embeds(id_cond, id_vit_hidden))
            uncond.append(pulid_model.get_image_embeds(id_uncond, id_vit_hidden_uncond))

        if not cond:
            # No faces detected, return the original model
            print("pulid warning: No faces detected in any of the given images, returning unmodified model.")
            return (work_model,)
        
        # average embeddings
        cond = torch.cat(cond).to(device, dtype=dtype)
        uncond = torch.cat(uncond).to(device, dtype=dtype)
        if cond.shape[0] > 1:
            cond = torch.mean(cond, dim=0, keepdim=True)
            uncond = torch.mean(uncond, dim=0, keepdim=True)

        if num_zero > 0:
            if noise == 0:
                zero_tensor = torch.zeros((cond.size(0), num_zero, cond.size(-1)), dtype=dtype, device=device)
            else:
                zero_tensor = torch.rand((cond.size(0), num_zero, cond.size(-1)), dtype=dtype, device=device) * noise
            cond = torch.cat([cond, zero_tensor], dim=1)
            uncond = torch.cat([uncond, zero_tensor], dim=1)

        sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = work_model.get_model_object("model_sampling").percent_to_sigma(end_at)

        patch_kwargs = {
            "pulid": pulid_model,
            "weight": weight,
            "cond": cond,
            "uncond": uncond,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "ortho": ortho,
            "ortho_v2": ortho_v2,
            "mask": attn_mask,
        }

        number = 0
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
            number += 1

        return (work_model,)

class ApplyPulidAdvanced(ApplyPulid):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "pulid": ("PULID", ),
                "eva_clip": ("EVA_CLIP", ),
                "face_analysis": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "projection": (["ortho_v2", "ortho", "none"],),
                "fidelity": ("INT", {"default": 8, "min": 0, "max": 32, "step": 1 }),
                "noise": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            },
            "optional": {
                "attn_mask": ("MASK", ),
            },
        }
    

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
    "PulidEvaClipLoader": PulidEvaClipLoader,
    "ApplyPulid": ApplyPulid,
    "ApplyPulidAdvanced": ApplyPulidAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageGetWidthHeight": "Image Get Width and Height",
    "PulidModelLoader": "Load PuLID Model",
    "PulidInsightFaceLoader": "Load InsightFace (PuLID)",
    "PulidEvaClipLoader": "Load Eva Clip (PuLID)",
    "ApplyPulid": "Apply PuLID",
    "ApplyPulidAdvanced": "Apply PuLID Advanced",
}
