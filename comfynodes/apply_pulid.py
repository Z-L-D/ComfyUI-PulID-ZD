from ..utils.pulid_pipeline_sdxl import PulIDPipelineSDXL
from ..utils.pulid_pipeline_flux import PulIDPipelineFLUX
from ..utils.pulid_pipeline_sdxl import PulidModelSDXL
from ..utils.pulid_pipeline_flux import PulidModelFlux



def apply_pulid(self, model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
    work_model = model.clone()

    # Check if the pulid model is SDXL or Flux
    if isinstance(pulid, PulidModelSDXL):
        print("Using SDXL model")
        work_model = PulIDPipelineSDXL(self, work_model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method, noise, fidelity, projection, attn_mask)
    elif isinstance(pulid, PulidModelFlux):
        print("Using Flux model")
        work_model = PulIDPipelineFLUX(self, work_model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method, noise, fidelity, projection, attn_mask)
    else:
        raise ValueError("Unknown model type")

    return work_model