from ..utils.pulid_pipeline_sdxl import PulIDPipelineSDXL



def apply_pulid(self, model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
    work_model = model.clone()

    work_model = PulIDPipelineSDXL(self, work_model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None)
        
    return work_model