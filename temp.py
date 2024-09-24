please help me integrate this new machine learning pipeline for an updated model. the present pipeline is using SDXL. here is the current integrated pipeline: 

##################################
# pulid_pipeline_sdxl.py
##################################
import torch
from torch import nn
import torchvision.transforms as torchTransforms
import torch.nn.functional as torchFunctional

import math
import comfy.utils
from comfy.ldm.modules.attention import optimized_attention

from ..facexlib.parsing import init_parsing_model
from ..facexlib.utils.face_restoration_helper import FaceRestoreHelper
from ..pulid.encoders import IDEncoder
from ..utils.tensors import tensor_to_image, tensor_to_size, image_to_tensor, to_gray

class PulidModelSDXL(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.image_proj_model = self.init_id_adapter()
        self.image_proj_model.load_state_dict(model["image_proj"])
        self.ip_layers = To_KV(model["ip_adapter"])
    
    def init_id_adapter(self):
        image_proj_model = IDEncoder()
        return image_proj_model

    def get_image_embeds(self, face_embed, clip_embeds):
        embeds = self.image_proj_model(face_embed, clip_embeds)
        return embeds



class To_KV(nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value



def PulIDPipelineSDXL(self, work_model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
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
        face = image_to_tensor(face).unsqueeze(0).permute(0,3,1,2).to(device)
        parsing_out = face_helper.face_parse(torchTransforms.functional.normalize(face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(face)
        face_features_image = torch.where(bg, white_image, to_gray(face))

        # apparently MPS only supports NEAREST interpolation?
        face_features_image = torchTransforms.functional.resize(face_features_image, eva_clip.image_size, torchTransforms.InterpolationMode.BICUBIC if 'cuda' in device.type else torchTransforms.InterpolationMode.NEAREST).to(device, dtype=dtype)
        face_features_image = torchTransforms.functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)
            
        id_cond_vit, id_vit_hidden = eva_clip(face_features_image, return_all_features=False, return_hidden=True, shuffle=False)
        id_cond_vit = id_cond_vit.to(device, dtype=dtype)
        for idx in range(len(id_vit_hidden)):
            id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

        id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

        # combine embeddings
        id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
        if noise == 0:
            id_uncond = torch.zeros_like(id_cond)
        else:
            id_uncond = torch.rand_like(id_cond) * noise
        id_vit_hidden_uncond = []
        for idx in range(len(id_vit_hidden)):
            if noise == 0:
                id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[idx]))
            else:
                id_vit_hidden_uncond.append(torch.rand_like(id_vit_hidden[idx]) * noise)
            
        cond.append(pulid_model.get_image_embeds(id_cond, id_vit_hidden))
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



def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()
    
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(pulid_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(pulid_attention, **patch_kwargs)

class Attn2Replace:
    def __init__(self, callback=None, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]
    
    def add(self, callback, **kwargs):          
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9

        for i, callback in enumerate(self.callback):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                out = out + callback(out, q, k, v, extra_options, **self.kwargs[i])
        
        return out.to(dtype=dtype)



def pulid_attention(out, q, k, v, extra_options, module_key='', pulid=None, cond=None, uncond=None, weight=1.0, ortho=False, ortho_v2=False, mask=None, **kwargs):
    k_key = module_key + "_to_k_ip"
    v_key = module_key + "_to_v_ip"

    dtype = q.dtype
    seq_len = q.shape[1]
    cond_or_uncond = extra_options["cond_or_uncond"]
    b = q.shape[0]
    batch_prompt = b // len(cond_or_uncond)
    _, _, oh, ow = extra_options["original_shape"]
    
    k_cond = pulid.ip_layers.to_kvs[k_key](cond).repeat(batch_prompt, 1, 1)
    k_uncond = pulid.ip_layers.to_kvs[k_key](uncond).repeat(batch_prompt, 1, 1)
    v_cond = pulid.ip_layers.to_kvs[v_key](cond).repeat(batch_prompt, 1, 1)
    v_uncond = pulid.ip_layers.to_kvs[v_key](uncond).repeat(batch_prompt, 1, 1)
    ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
    ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

    out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        
    if ortho:
        out = out.to(dtype=torch.float32)
        out_ip = out_ip.to(dtype=torch.float32)
        projection = (torch.sum((out * out_ip), dim=-2, keepdim=True) / torch.sum((out * out), dim=-2, keepdim=True) * out)
        orthogonal = out_ip - projection
        out_ip = weight * orthogonal
    elif ortho_v2:
        out = out.to(dtype=torch.float32)
        out_ip = out_ip.to(dtype=torch.float32)
        attn_map = q @ ip_k.transpose(-2, -1)
        attn_mean = attn_map.softmax(dim=-1).mean(dim=1, keepdim=True)
        attn_mean = attn_mean[:, :, :5].sum(dim=-1, keepdim=True)
        projection = (torch.sum((out * out_ip), dim=-2, keepdim=True) / torch.sum((out * out), dim=-2, keepdim=True) * out)
        orthogonal = out_ip + (attn_mean - 1) * projection
        out_ip = weight * orthogonal
    else:
        out_ip = out_ip * weight

    if mask is not None:
        mask_h = oh / math.sqrt(oh * ow / seq_len)
        mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
        mask_w = seq_len // mask_h

        mask = torchFunctional.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
        mask = tensor_to_size(mask, batch_prompt)

        mask = mask.repeat(len(cond_or_uncond), 1, 1)
        mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

        # covers cases where extreme aspect ratios can cause the mask to have a wrong size
        mask_len = mask_h * mask_w
        if mask_len < seq_len:
            pad_len = seq_len - mask_len
            pad1 = pad_len // 2
            pad2 = pad_len - pad1
            mask = torchFunctional.pad(mask, (0, 0, pad1, pad2), value=0.0)
        elif mask_len > seq_len:
            crop_start = (mask_len - seq_len) // 2
            mask = mask[:, crop_start:crop_start+seq_len, :]

        out_ip = out_ip * mask

    return out_ip.to(dtype=dtype)

this pipeline is based on several library files from the PulID library. here are those files: 

Number 1:
##################################
# pipeline.py
##################################
import gc

import cv2
import insightface
import torch
import torch.nn as nn
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from pulid.encoders import IDEncoder
from pulid.utils import img2tensor, is_torch2_available, tensor2img

if is_torch2_available():
    from pulid.attention_processor import AttnProcessor2_0 as AttnProcessor
    from pulid.attention_processor import IDAttnProcessor2_0 as IDAttnProcessor
else:
    from pulid.attention_processor import AttnProcessor, IDAttnProcessor


class PuLIDPipeline:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = 'cuda'
        sdxl_base_repo = 'stabilityai/stable-diffusion-xl-base-1.0'
        sdxl_lightning_repo = 'ByteDance/SDXL-Lightning'
        self.sdxl_base_repo = sdxl_base_repo

        # load base model
        unet = UNet2DConditionModel.from_config(sdxl_base_repo, subfolder='unet').to(self.device, torch.float16)
        unet.load_state_dict(
            load_file(
                hf_hub_download(sdxl_lightning_repo, 'sdxl_lightning_4step_unet.safetensors'), device=self.device
            )
        )
        self.hack_unet_attn_layers(unet)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            sdxl_base_repo, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)
        self.pipe.watermark = None

        # scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )

        # ID adapters
        self.id_adapter = IDEncoder().to(self.device)

        ##############################
        # preprocessors
        ##############################

        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)


        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std


        # antelopev2
        snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        self.app = FaceAnalysis(
            name='antelopev2', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx')
        self.handler_ante.prepare(ctx_id=0)

        gc.collect()
        torch.cuda.empty_cache()

        self.load_pretrain()

        # other configs
        self.debug_img_list = []

    def hack_unet_attn_layers(self, unet):
        id_adapter_attn_procs = {}
        for name, _ in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is not None:
                id_adapter_attn_procs[name] = IDAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(unet.device)
            else:
                id_adapter_attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(id_adapter_attn_procs)
        self.id_adapter_attn_layers = nn.ModuleList(unet.attn_processors.values())

    def load_pretrain(self):
        hf_hub_download('guozinan/PuLID', 'pulid_v1.bin', local_dir='models')
        ckpt_path = 'models/pulid_v1.bin'
        state_dict = torch.load(ckpt_path, map_location='cpu')
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1 :]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    def get_id_embedding(self, image):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """
        self.face_helper.clean_all()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                image[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image, return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)
        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))

        id_embedding = self.id_adapter(id_cond, id_vit_hidden)
        uncond_id_embedding = self.id_adapter(id_uncond, id_vit_hidden_uncond)

        # return id_embedding
        return torch.cat((uncond_id_embedding, id_embedding), dim=0)

    def inference(self, prompt, size, prompt_n='', image_embedding=None, id_scale=1.0, guidance_scale=1.2, steps=4):
        images = self.pipe(
            prompt=prompt,
            negative_prompt=prompt_n,
            num_images_per_prompt=size[0],
            height=size[1],
            width=size[2],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            cross_attention_kwargs={'id_embedding': image_embedding, 'id_scale': id_scale},
        ).images

        return images


Number 2:

##################################
# attention_processor.py
##################################
# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ZERO = 0
ORTHO = False
ORTHO_v2 = False


class AttnProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IDAttnProcessor(nn.Module):
    r"""
    Attention processor for ID-Adapater.
    Args:
        hidden_size (int):
            The hidden size of the attention layer.
        cross_attention_dim (int):
            The number of channels in the encoder_hidden_states.
        scale (float, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None):
        super().__init__()
        self.id_to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.id_to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for id-adapter
        if id_embedding is not None:
            if NUM_ZERO == 0:
                id_key = self.id_to_k(id_embedding)
                id_value = self.id_to_v(id_embedding)
            else:
                zero_tensor = torch.zeros(
                    (id_embedding.size(0), NUM_ZERO, id_embedding.size(-1)),
                    dtype=id_embedding.dtype,
                    device=id_embedding.device,
                )
                id_key = self.id_to_k(torch.cat((id_embedding, zero_tensor), dim=1))
                id_value = self.id_to_v(torch.cat((id_embedding, zero_tensor), dim=1))

            id_key = attn.head_to_batch_dim(id_key).to(query.dtype)
            id_value = attn.head_to_batch_dim(id_value).to(query.dtype)

            id_attention_probs = attn.get_attention_scores(query, id_key, None)
            id_hidden_states = torch.bmm(id_attention_probs, id_value)
            id_hidden_states = attn.batch_to_head_dim(id_hidden_states)

            if not ORTHO:
                hidden_states = hidden_states + id_scale * id_hidden_states
            else:
                projection = (
                    torch.sum((hidden_states * id_hidden_states), dim=-2, keepdim=True)
                    / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                    * hidden_states
                )
                orthogonal = id_hidden_states - projection
                hidden_states = hidden_states + id_scale * orthogonal

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IDAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for ID-Adapater for PyTorch 2.0.
    Args:
        hidden_size (int):
            The hidden size of the attention layer.
        cross_attention_dim (int):
            The number of channels in the encoder_hidden_states.
    """

    def __init__(self, hidden_size, cross_attention_dim=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.id_to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.id_to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        id_embedding=None,
        id_scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for id embedding
        if id_embedding is not None:
            if NUM_ZERO == 0:
                id_key = self.id_to_k(id_embedding).to(query.dtype)
                id_value = self.id_to_v(id_embedding).to(query.dtype)
            else:
                zero_tensor = torch.zeros(
                    (id_embedding.size(0), NUM_ZERO, id_embedding.size(-1)),
                    dtype=id_embedding.dtype,
                    device=id_embedding.device,
                )
                id_key = self.id_to_k(torch.cat((id_embedding, zero_tensor), dim=1)).to(query.dtype)
                id_value = self.id_to_v(torch.cat((id_embedding, zero_tensor), dim=1)).to(query.dtype)

            id_key = id_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            id_value = id_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            id_hidden_states = F.scaled_dot_product_attention(
                query, id_key, id_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            id_hidden_states = id_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            id_hidden_states = id_hidden_states.to(query.dtype)

            if not ORTHO and not ORTHO_v2:
                hidden_states = hidden_states + id_scale * id_hidden_states
            elif ORTHO_v2:
                orig_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                id_hidden_states = id_hidden_states.to(torch.float32)
                attn_map = query @ id_key.transpose(-2, -1)
                attn_mean = attn_map.softmax(dim=-1).mean(dim=1)
                attn_mean = attn_mean[:, :, :5].sum(dim=-1, keepdim=True)
                projection = (
                    torch.sum((hidden_states * id_hidden_states), dim=-2, keepdim=True)
                    / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                    * hidden_states
                )
                orthogonal = id_hidden_states + (attn_mean - 1) * projection
                hidden_states = hidden_states + id_scale * orthogonal
                hidden_states = hidden_states.to(orig_dtype)
            else:
                orig_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                id_hidden_states = id_hidden_states.to(torch.float32)
                projection = (
                    torch.sum((hidden_states * id_hidden_states), dim=-2, keepdim=True)
                    / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                    * hidden_states
                )
                orthogonal = id_hidden_states - projection
                hidden_states = hidden_states + id_scale * orthogonal
                hidden_states = hidden_states.to(orig_dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

Number 3:

##################################
# encoders.py
##################################
import torch
import torch.nn as nn

class IDEncoder(nn.Module):
    def __init__(self, width=1280, context_dim=2048, num_token=5):
        super().__init__()
        self.num_token = num_token
        self.context_dim = context_dim
        h1 = min((context_dim * num_token) // 4, 1024)
        h2 = min((context_dim * num_token) // 2, 1024)
        self.body = nn.Sequential(
            nn.Linear(width, h1),
            nn.LayerNorm(h1),
            nn.LeakyReLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.LeakyReLU(),
            nn.Linear(h2, context_dim * num_token),
        )

        for i in range(5):
            setattr(
                self,
                f'mapping_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, context_dim),
                ),
            )

            setattr(
                self,
                f'mapping_patch_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, context_dim),
                ),
            )

    def forward(self, x, y):
        # x shape [N, C]
        x = self.body(x)
        x = x.reshape(-1, self.num_token, self.context_dim)

        hidden_states = ()
        for i, emb in enumerate(y):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(
                emb[:, 1:]
            ).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state,)
        hidden_states = torch.cat(hidden_states, dim=1)

        return torch.cat([x, hidden_states], dim=1)

the new pipeline in the pulid library is here:

Number 1:

##################################
# pipeline_flux.py
##################################
import gc

import cv2
import insightface
import torch
import torch.nn as nn
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from pulid.encoders_flux import IDFormer, PerceiverAttentionCA
from pulid.utils import img2tensor, tensor2img


class PuLIDPipeline(nn.Module):
    def __init__(self, dit, device, weight_dtype=torch.bfloat16, onnx_provider='gpu', *args, **kwargs):
        super().__init__()
        self.device = device
        self.weight_dtype = weight_dtype
        double_interval = 2
        single_interval = 4

        # init encoder
        self.pulid_encoder = IDFormer().to(self.device, self.weight_dtype)

        num_ca = 19 // double_interval + 38 // single_interval
        if 19 % double_interval != 0:
            num_ca += 1
        if 38 % single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA().to(self.device, self.weight_dtype) for _ in range(num_ca)
        ])

        dit.pulid_ca = self.pulid_ca
        dit.pulid_double_interval = double_interval
        dit.pulid_single_interval = single_interval

        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)
        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device, dtype=self.weight_dtype)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std
        # antelopev2
        snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        providers = ['CPUExecutionProvider'] if onnx_provider == 'cpu' \
            else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.app = FaceAnalysis(name='antelopev2', root='.', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx',
                                                            providers=providers)
        self.handler_ante.prepare(ctx_id=0)

        gc.collect()
        torch.cuda.empty_cache()

        # self.load_pretrain()

        # other configs
        self.debug_img_list = []

    def components_to_device(self, device):
        # everything but pulid_ca
        self.face_helper.face_det = self.face_helper.face_det.to(device)
        self.face_helper.face_parse = self.face_helper.face_parse.to(device)
        self.clip_vision_model = self.clip_vision_model.to(device)
        self.pulid_encoder = self.pulid_encoder.to(device)

    # def load_pretrain(self, pretrain_path=None):
    #     hf_hub_download('guozinan/PuLID', 'pulid_flux_v0.9.0.safetensors', local_dir='models')
    #     ckpt_path = 'models/pulid_flux_v0.9.0.safetensors'
    #     if pretrain_path is not None:
    #         ckpt_path = pretrain_path
    #     state_dict = load_file(ckpt_path)
    #     state_dict_dict = {}
    #     for k, v in state_dict.items():
    #         module = k.split('.')[0]
    #         state_dict_dict.setdefault(module, {})
    #         new_k = k[len(module) + 1:]
    #         state_dict_dict[module][new_k] = v

    #     for module in state_dict_dict:
    #         print(f'loading from {module}')
    #         getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

    #     del state_dict
    #     del state_dict_dict

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    @torch.no_grad()
    def get_id_embedding(self, image, cal_uncond=False):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """
        self.face_helper.clean_all()
        self.debug_img_list = []
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                image[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.weight_dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image.to(self.weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

        id_embedding = self.pulid_encoder(id_cond, id_vit_hidden)

        if not cal_uncond:
            return id_embedding, None

        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))
        uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)

        return id_embedding, uncond_id_embedding

Number 2: 

##################################
# encoders_flux.py
##################################
import math

import torch
import torch.nn as nn


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttentionCA(nn.Module):
    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, kv_dim=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)


class IDFormer(nn.Module):
    """
    - perceiver resampler like arch (compared with previous MLP-like arch)
    - we concat id embedding (generated by arcface) and query tokens as latents
    - latents will attend each other and interact with vit features through cross-attention
    - vit features are multi-scaled and inserted into IDFormer in order, currently, each scale corresponds to two
      IDFormer layers
    """
    def __init__(
            self,
            dim=1024,
            depth=10,
            dim_head=64,
            heads=16,
            num_id_token=5,
            num_queries=32,
            output_dim=2048,
            ff_mult=4,
    ):
        super().__init__()

        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth = depth // 5
        scale = dim ** -0.5

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * scale)
        self.proj_out = nn.Parameter(scale * torch.randn(dim, output_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        for i in range(5):
            setattr(
                self,
                f'mapping_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, dim),
                ),
            )

        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, dim * num_id_token),
        )

    def forward(self, x, y):

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.id_embedding_mapping(x)
        x = x.reshape(-1, self.num_id_token, self.dim)

        latents = torch.cat((latents, x), dim=1)

        for i in range(5):
            vit_feature = getattr(self, f'mapping_{i}')(y[i])
            ctx_feature = torch.cat((x, vit_feature), dim=1)
            for attn, ff in self.layers[i * self.depth: (i + 1) * self.depth]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents

        latents = latents[:, :self.num_queries]
        latents = latents @ self.proj_out
        return latents

both of these pipelines need to exist at the same time as options. they are called in two files: 

Number 1:

##################################
# encoders_flux.py
##################################
import folder_paths

import comfy.utils
from ..comfymodels.path import INSIGHTFACE_DIR, INSIGHTFACE_PATH, PULID_DIR, PULID_PATH, CLIP_DIR, CLIP_PATH, FACEDETECT_DIR, FACEDETECT_PATH, FACERESTORE_DIR, FACERESTORE_PATH
from ..utils.pulid_pipeline_sdxl import PulidModelSDXL


def load_model_pulid(self, pulid_file):
    ckpt_path = folder_paths.get_full_path(PULID_DIR, pulid_file)

    model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

    if(pulid_file == "ip-adapter_pulid_sdxl_fp16.safetensors"):
        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model
        
        # Also initialize the model, takes longer to load but then it doesn't have to be done every time you change parameters in the apply node
        model = PulidModelSDXL(model)
        
    elif(pulid_file == "pulid_flux_v0.9.0.safetensors"):
        model = ""
    

    return (model,)


Number 2: 

##################################
# apply_pulid.py
##################################
from ..utils.pulid_pipeline_sdxl import PulIDPipelineSDXL
from ..utils.pulid_pipeline_sdxl import PulIDPipelineSDXL

def apply_pulid(self, model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
    work_model = model.clone()

    work_model = PulIDPipelineSDXL(self, work_model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None)
        
    return work_model

finally, the outcome of this process should result in pulid_pipeline_flux.py, just as pulid_pipeline_sdxl.py exists. pulid_pipeline_flux.py should be modified correctly to allow the flux model pipeline to be integrated. pulid_pipeline_sdxl.py and the pulid library files should be used as a strict guide in reaching this goal. the current state of pulied_pipeline_flux.py is here:

import torch
from torch import nn
import torchvision.transforms as torchTransforms
import torch.nn.functional as torchFunctional

import math
import comfy.utils
from comfy.ldm.modules.attention import optimized_attention

from ..facexlib.parsing import init_parsing_model
from ..facexlib.utils.face_restoration_helper import FaceRestoreHelper
from ..pulid.encoders_flux import IDFormer
from ..utils.tensors import tensor_to_image, tensor_to_size, image_to_tensor, to_gray

class PulidModelFlux(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.image_proj_model = self.init_id_adapter()
        self.image_proj_model.load_state_dict(model["image_proj"])
        
    def init_id_adapter(self):
        image_proj_model = IDFormer()
        return image_proj_model
    
    def get_image_embeds(self, face_embed, clip_embeds):
        embeds = self.image_proj_model(face_embed, clip_embeds)
        return embeds
    
def PulIDPipelineFLUX(self, work_model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
    # comfy device and memory management   
    device = comfy.model_management.get_torch_device()
    dtype = comfy.model_management.unet_dtype()


    # send eva_clip and pulid_model to vram
    eva_clip.to(device, dtype=dtype)
    pulid_model = pulid.to(device, dtype=dtype)




