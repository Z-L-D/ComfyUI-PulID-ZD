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




