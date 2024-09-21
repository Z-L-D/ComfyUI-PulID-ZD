import torch
from torch import nn

from ..pulid.encoders_flux import IDFormer

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