# import torch
# from torch import nn
# import torchvision.transforms as torchTransforms
# import torch.nn.functional as torchFunctional

# import math
# import comfy.utils
# from comfy.ldm.modules.attention import optimized_attention

# from ..facexlib.parsing import init_parsing_model
# from ..facexlib.utils.face_restoration_helper import FaceRestoreHelper
# from ..pulid.encoders_flux import IDFormer
# from ..utils.tensors import tensor_to_image, tensor_to_size, image_to_tensor, to_gray

# class PulidModelFlux(nn.Module):
#     def __init__(self, model):
#         super().__init__()

#         self.image_proj_model = self.init_id_adapter()
#         self.image_proj_model.load_state_dict(model["image_proj"])
        
#     def init_id_adapter(self):
#         image_proj_model = IDFormer()
#         return image_proj_model
    
#     def get_image_embeds(self, face_embed, clip_embeds):
#         embeds = self.image_proj_model(face_embed, clip_embeds)
#         return embeds
    
# def PulIDPipelineFLUX(self, work_model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
#     # comfy device and memory management   
#     device = comfy.model_management.get_torch_device()
#     dtype = comfy.model_management.unet_dtype()


#     # send eva_clip and pulid_model to vram
#     eva_clip.to(device, dtype=dtype)
#     pulid_model = pulid.to(device, dtype=dtype)

################################################

# import torch
# from torch import nn
# import torchvision.transforms as torchTransforms
# import torch.nn.functional as torchFunctional

# import math
# import comfy.utils
# from comfy.ldm.modules.attention import optimized_attention

# from ..facexlib.parsing import init_parsing_model
# from ..facexlib.utils.face_restoration_helper import FaceRestoreHelper
# from ..pulid.encoders_flux import IDFormer, PerceiverAttentionCA
# from ..utils.tensors import tensor_to_image, tensor_to_size, image_to_tensor, to_gray

# class PulidModelFlux(nn.Module):
#     def __init__(self, model):
#         super().__init__()

#         self.image_proj_model = self.init_id_adapter()
#         self.image_proj_model.load_state_dict(model["image_proj"])
#         self.pulid_ca = self.init_pulid_ca(model["pulid_ca"])

#     def init_id_adapter(self):
#         image_proj_model = IDFormer()
#         return image_proj_model

#     def init_pulid_ca(self, pulid_ca_state_dict_list):
#         num_ca = len(pulid_ca_state_dict_list)
#         pulid_ca = nn.ModuleList([PerceiverAttentionCA() for _ in range(num_ca)])
#         for i, ca in enumerate(pulid_ca):
#             ca.load_state_dict(pulid_ca_state_dict_list[i])
#         return pulid_ca

#     def get_image_embeds(self, id_cond, id_vit_hidden):
#         embeds = self.image_proj_model(id_cond, id_vit_hidden)
#         return embeds

# def PulIDPipelineFLUX(self, work_model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
#     # Device and dtype management
#     device = comfy.model_management.get_torch_device()
#     dtype = comfy.model_management.unet_dtype()
#     if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
#         dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

#     # Move models to device and dtype
#     eva_clip.to(device, dtype=dtype)
#     pulid_model = pulid.to(device, dtype=dtype)

#     # Process images to extract embeddings
#     face_analysis.det_model.input_size = (640, 640)
#     image = tensor_to_image(image)

#     face_helper = FaceRestoreHelper(
#         upscale_factor=1,
#         face_size=512,
#         crop_ratio=(1, 1),
#         det_model='retinaface_resnet50',
#         save_ext='png',
#         device=device,
#     )
#     face_helper.face_parse = None
#     face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)

#     bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
#     cond_embeddings = []

#     for i in range(image.shape[0]):
#         # Get insightface embeddings
#         iface_embeds = None
#         for size in [(size, size) for size in range(640, 256, -64)]:
#             face_analysis.det_model.input_size = size
#             face = face_analysis.get(image[i])
#             if face:
#                 face = sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[-1]
#                 iface_embeds = torch.from_numpy(face.embedding).unsqueeze(0).to(device, dtype=dtype)
#                 break
#         else:
#             # No face detected, skip this image
#             print('Warning: No face detected in image', i)
#             continue

#         # Get EVA-CLIP embeddings
#         face_helper.clean_all()
#         face_helper.read_image(image[i])
#         face_helper.get_face_landmarks_5(only_center_face=True)
#         face_helper.align_warp_face()

#         if len(face_helper.cropped_faces) == 0:
#             # No face detected, skip this image
#             continue

#         face = face_helper.cropped_faces[0]
#         face = image_to_tensor(face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
#         parsing_out = face_helper.face_parse(
#             torchTransforms.functional.normalize(face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         )[0]
#         parsing_out = parsing_out.argmax(dim=1, keepdim=True)
#         bg = sum(parsing_out == i for i in bg_label).bool()
#         white_image = torch.ones_like(face)
#         face_features_image = torch.where(bg, white_image, to_gray(face))

#         # Resize and normalize face_features_image
#         face_features_image = torchTransforms.functional.resize(
#             face_features_image,
#             eva_clip.image_size,
#             torchTransforms.InterpolationMode.BICUBIC
#             if 'cuda' in device.type
#             else torchTransforms.InterpolationMode.NEAREST,
#         ).to(device, dtype=dtype)
#         face_features_image = torchTransforms.functional.normalize(
#             face_features_image, eva_clip.image_mean, eva_clip.image_std
#         )

#         id_cond_vit, id_vit_hidden = eva_clip(
#             face_features_image, return_all_features=False, return_hidden=True, shuffle=False
#         )
#         id_cond_vit = id_cond_vit.to(device, dtype=dtype)
#         for idx in range(len(id_vit_hidden)):
#             id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

#         id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

#         # Combine embeddings
#         id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
#         cond_embedding = pulid_model.get_image_embeds(id_cond, id_vit_hidden)
#         cond_embeddings.append(cond_embedding)

#     if not cond_embeddings:
#         # No faces detected, return the original model
#         print("pulid warning: No faces detected in any of the given images, returning unmodified model.")
#         return (work_model,)

#     # Average embeddings
#     cond_embedding = torch.cat(cond_embeddings).to(device, dtype=dtype)
#     if cond_embedding.shape[0] > 1:
#         cond_embedding = torch.mean(cond_embedding, dim=0, keepdim=True)

#     # Integrate embeddings and pulid_ca modules into the model
#     unet = work_model.get_model_object("unet")

#     # Assuming the UNet can accept these modules
#     # Set the necessary attributes in the UNet
#     unet.pulid_ca = pulid_model.pulid_ca
#     unet.pulid_cond_embedding = cond_embedding
#     unet.pulid_weight = weight
#     unet.pulid_start_at = start_at
#     unet.pulid_end_at = end_at

#     # Note: The integration may require modifying the UNet's forward pass or using hooks
#     # This depends on the capabilities of your framework

#     # Return the modified model
#     return (work_model,)

################################################

import torch
from torch import nn
import torchvision.transforms as torchTransforms
import torch.nn.functional as torchFunctional

import math
import comfy.utils
from comfy.ldm.modules.attention import optimized_attention

from ..facexlib.parsing import init_parsing_model
from ..facexlib.utils.face_restoration_helper import FaceRestoreHelper
from ..pulid.encoders_flux import IDFormer, PerceiverAttentionCA
from ..utils.tensors import tensor_to_image, tensor_to_size, image_to_tensor, to_gray

class PulidModelFlux(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.image_proj_model = self.init_id_adapter()
        self.image_proj_model.load_state_dict(model["image_proj"])
        self.pulid_ca = self.init_pulid_ca(model["pulid_ca"])

    def init_id_adapter(self):
        image_proj_model = IDFormer()
        return image_proj_model

    def init_pulid_ca(self, pulid_ca_state_dict_list):
        num_ca = len(pulid_ca_state_dict_list)
        pulid_ca = nn.ModuleList([PerceiverAttentionCA() for _ in range(num_ca)])
        for i, ca in enumerate(pulid_ca):
            ca.load_state_dict(pulid_ca_state_dict_list[i])
        return pulid_ca

    def get_image_embeds(self, id_cond, id_vit_hidden):
        embeds = self.image_proj_model(id_cond, id_vit_hidden)
        return embeds

def PulIDPipelineFLUX(work_model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
    # Device and dtype management
    device = comfy.model_management.get_torch_device()
    dtype = comfy.model_management.unet_dtype()
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

    # Move models to device and dtype
    eva_clip.to(device, dtype=dtype)
    pulid_model = pulid.to(device, dtype=dtype)

    # Process images to extract embeddings
    face_analysis.det_model.input_size = (640, 640)
    image = tensor_to_image(image)

    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        device=device,
    )
    face_helper.face_parse = None
    face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)

    bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
    cond_embeddings = []

    for i in range(image.shape[0]):
        # Get insightface embeddings
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

        # Get EVA-CLIP embeddings
        face_helper.clean_all()
        face_helper.read_image(image[i])
        face_helper.get_face_landmarks_5(only_center_face=True)
        face_helper.align_warp_face()

        if len(face_helper.cropped_faces) == 0:
            # No face detected, skip this image
            continue

        face = face_helper.cropped_faces[0]
        face = image_to_tensor(face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        parsing_out = face_helper.face_parse(
            torchTransforms.functional.normalize(face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(face)
        face_features_image = torch.where(bg, white_image, to_gray(face))

        # Resize and normalize face_features_image
        face_features_image = torchTransforms.functional.resize(
            face_features_image,
            eva_clip.image_size,
            torchTransforms.InterpolationMode.BICUBIC
            if 'cuda' in device.type
            else torchTransforms.InterpolationMode.NEAREST,
        ).to(device, dtype=dtype)
        face_features_image = torchTransforms.functional.normalize(
            face_features_image, eva_clip.image_mean, eva_clip.image_std
        )

        id_cond_vit, id_vit_hidden = eva_clip(
            face_features_image, return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit = id_cond_vit.to(device, dtype=dtype)
        for idx in range(len(id_vit_hidden)):
            id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

        id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

        # Combine embeddings
        id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
        cond_embedding = pulid_model.get_image_embeds(id_cond, id_vit_hidden)
        cond_embeddings.append(cond_embedding)

    if not cond_embeddings:
        # No faces detected, return the original model
        print("pulid warning: No faces detected in any of the given images, returning unmodified model.")
        return (work_model,)

    # Average embeddings
    cond_embedding = torch.cat(cond_embeddings).to(device, dtype=dtype)
    if cond_embedding.shape[0] > 1:
        cond_embedding = torch.mean(cond_embedding, dim=0, keepdim=True)

    # Set up patching similar to SDXL pipeline
    sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(start_at)
    sigma_end = work_model.get_model_object("model_sampling").percent_to_sigma(end_at)

    patch_kwargs = {
        "pulid_ca": pulid_model.pulid_ca,
        "latents": cond_embedding,
        "weight": weight,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
    }

    # Apply patches to the model
    set_model_patch_replace(work_model, patch_kwargs)

    return (work_model,)

def set_model_patch_replace(model, patch_kwargs):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "cross_attention" not in to["patches_replace"]:
        to["patches_replace"]["cross_attention"] = {}
    else:
        to["patches_replace"]["cross_attention"] = to["patches_replace"]["cross_attention"].copy()

    # Define the keys where you want to replace the attention modules
    keys = get_attention_keys(model)

    for key in keys:
        if key not in to["patches_replace"]["cross_attention"]:
            to["patches_replace"]["cross_attention"][key] = CrossAttentionReplace(**patch_kwargs)
            model.model_options["transformer_options"] = to
        else:
            # If already exists, you can modify or add to it
            pass

def get_attention_keys(model):
    # Generate the keys for attention modules that you want to replace
    # This function needs to be adapted based on your model's structure
    keys = []
    # Example: Collect keys from input_blocks, middle_block, and output_blocks
    # You need to identify where the cross-attention modules are located
    # For demonstration purposes, let's assume we have keys like ("input", block_id, layer_id)
    for block_id in range(len(model.unet.input_blocks)):
        for layer_id in range(len(model.unet.input_blocks[block_id])):
            keys.append(("input", block_id, layer_id))
    for block_id in range(len(model.unet.output_blocks)):
        for layer_id in range(len(model.unet.output_blocks[block_id])):
            keys.append(("output", block_id, layer_id))
    # Middle block
    for layer_id in range(len(model.unet.middle_block)):
        keys.append(("middle", 0, layer_id))
    return keys

class CrossAttentionReplace:
    def __init__(self, pulid_ca, latents, weight, sigma_start, sigma_end):
        self.pulid_ca = pulid_ca
        self.latents = latents
        self.weight = weight
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.ca_index = 0  # To keep track of which pulid_ca module to use

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        # Only apply within the specified sigma range
        sigma = kwargs.get("sigma", 999999999.9)
        if sigma > self.sigma_start or sigma < self.sigma_end:
            return attn(hidden_states, encoder_hidden_states, attention_mask, temb, **kwargs)

        # Apply the custom cross-attention
        x = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        # Apply pulid_ca module
        pulid_ca_module = self.pulid_ca[self.ca_index % len(self.pulid_ca)]
        self.ca_index += 1

        # Prepare inputs for pulid_ca
        x = x + self.weight * pulid_ca_module(x, self.latents)

        # Continue with the rest of the attention mechanism
        attn_output = attn.to_out[0](x)
        attn_output = attn.to_out[1](attn_output)

        if attn.residual_connection:
            attn_output = attn_output + hidden_states

        attn_output = attn_output / attn.rescale_output_factor

        return attn_output
