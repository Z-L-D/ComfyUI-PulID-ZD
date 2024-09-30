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

def PulIDPipelineFLUX(work_model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
    # comfy device and memory management    
    device = comfy.model_management.get_torch_device()
    dtype = comfy.model_management.unet_dtype()
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32


    # send eva_clip and pulid_model to vram
    eva_clip.to(device, dtype=dtype)
    pulid_model = pulid.to(device, dtype=dtype)


    # check if a mask is being sent
    if attn_mask is not None:
        if attn_mask.dim() > 3:
            attn_mask = attn_mask.squeeze(-1)
        elif attn_mask.dim() < 3:
            attn_mask = attn_mask.unsqueeze(0)
        attn_mask = attn_mask.to(device, dtype=dtype)


    # applies weights closer to the face image reference
    if method == "fidelity" or projection == "ortho_v2":
        num_zero = 8
        ortho = False
        ortho_v2 = True
    # applies weights closer to the unet model preference
    elif method == "style" or projection == "ortho":
        num_zero = 16
        ortho = True
        ortho_v2 = False
    # neutral weighting
    else:
        num_zero = 0
        ortho = False
        ortho_v2 = False
        

    if fidelity is not None:
        num_zero = fidelity


    # define insightface face photo analysis input size
    face_analysis.det_model.input_size = (640,640)


    # convert input face photo to tensor
    image = tensor_to_image(image)


    # define face restore settings with retinaface_resnet50
    # !!!! TODO: build this out in the node to allow other models as in Reactor
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

        # apparently MPS only supports NEAREST interpolation?
        face_features_image = torchTransforms.functional.resize(
            face_features_image,
            eva_clip.image_size,
            torchTransforms.InterpolationMode.BICUBIC
            if 'cuda' in device.type
            else torchTransforms.InterpolationMode.NEAREST,
        ).to(device, dtype=dtype)
        face_features_image = torchTransforms.functional.normalize(
            face_features_image, 
            eva_clip.image_mean, 
            eva_clip.image_std
        )

        id_cond_vit, id_vit_hidden = eva_clip(
            face_features_image,
            return_all_features=False,
            return_hidden=True,
            shuffle=False
        )
        id_cond_vit = id_cond_vit.to(device, dtype=dtype)
        for idx in range(len(id_vit_hidden)):
            id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

        id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))