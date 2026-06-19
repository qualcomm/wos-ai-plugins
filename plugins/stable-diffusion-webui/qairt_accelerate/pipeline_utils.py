# =============================================================================
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================

import qairt_constants as consts
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from qai_appbuilder import QNNContextProc, QNNShareMemory, QNNConfig, Runtime, LogLevel, ProfilingLevel
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDPMScheduler,
)

_qnn_initialized: bool = False

def ensure_qnn_config() -> None:
    """Call QNNConfig.Config exactly once for the lifetime of the process."""
    global _qnn_initialized
    if _qnn_initialized:
        return
    QNNConfig.Config(
        consts.QNN_LIBS_DIR, Runtime.HTP, LogLevel.ERROR, ProfilingLevel.BASIC
    )
    _qnn_initialized = True


class QPipeline:
    model_name = None

    def __init__(self, model_name):
        self.model_name = model_name


class StableDiffusionInput:
    is_txt2img = True
    user_prompt = ""
    uncond_prompt = ""
    user_seed = np.int64(0)
    user_step = 20
    user_text_guidance = 7.5
    is_high_resolution = False
    upscaler_model_name = None
    sampler_name = None
    model_name = None
    input_image: Image = None

    def __init__(
        self,
        is_txt2img,
        prompt,
        un_prompt,
        seed,
        step,
        text_guidance,
        sampler_name,
        model_name,
    ):
        self.is_txt2img = is_txt2img
        self.user_prompt = prompt
        self.uncond_prompt = un_prompt
        self.user_seed = seed
        self.user_step = step
        self.user_text_guidance = text_guidance
        self.sampler_name = sampler_name
        self.model_name = model_name

        assert (
            isinstance(self.user_seed, int) == True
        ), "user_seed should be of type int"
        assert (
            isinstance(self.user_step, int) == True
        ), "user_step should be of type int"
        assert (
            isinstance(self.user_text_guidance, float) == True
        ), "user_text_guidance should be of type float"

    def set_upscaler_model_name(self, upscaler_model_name):
        self.upscaler_model_name = upscaler_model_name
        self.upscaler_model_used, self.upscaler_model_path = get_upscaler_model(
            upscaler_model_name
        )

    def set_image(self, image: Image):
        self.input_image = image

    def set_high_res(self, is_high_resolution):
        self.is_high_resolution = is_high_resolution


def get_upscaler_model(model_name):
    # This is the default model, more upscale models will be added.
    return ("ESRGAN_4x", consts.ESRGAN_X4_MODEL_PATH)


def download_qualcomm_models_hf(model_path, hf_repo_id, files, revision):
    for file in files:
        hf_hub_download(
            repo_id=hf_repo_id,
            filename=file,
            local_dir=model_path,
            revision=revision,
        )


class UpscaleModel(QNNContextProc):
    # @timer
    def Inference(self, mem, input_data):
        input_datas = [input_data]
        output_data = super().Inference(mem, input_datas, perf_profile="burst")[0]
        return output_data


class UpscalerPipeline(QPipeline):
    model = None
    model_mem = None

    def __init__(self, model_name, model_path):
        super().__init__(model_name)
        name = "upscale"
        # process names
        model_proc = "~upscale"
        # share memory names.
        model_mem_name = name + "~memory"

        # Instance for RealESRGan which inherited from the class QNNContextProc, the model will be loaded into a separate process.
        self.model = UpscaleModel(name, model_proc, model_path)
        self.model_mem = QNNShareMemory(model_mem_name, 1024 * 1024 * 50)  # 50M

    # Release all the models.
    def __del__(self):
        del self.model
        del self.model_mem

    def execute(self, input):
        return self.model.Inference(self.model_mem, [input])

def set_scheduler(model_path, sampler_name):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler")
    if sampler_name == "Euler a":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler")
    elif sampler_name == "Euler":
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler")
    elif sampler_name == "LMS":
        scheduler = LMSDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler")
    elif sampler_name == "Heun":
        scheduler = HeunDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler")
    elif sampler_name == "DPM++ 2M":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler")
    elif sampler_name == "LMS Karras":
        scheduler = LMSDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler")
    elif sampler_name == "DPM++ 2M Karras":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler", 
            algorithm_type="dpmsolver++", 
            use_karras_sigmas=True,)
    elif sampler_name == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(
            model_path, subfolder="scheduler")
    elif sampler_name == "PLMS":
        scheduler = PNDMScheduler.from_pretrained(
            model_path, subfolder="scheduler")
    return scheduler
