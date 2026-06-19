# =============================================================================
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================

from qairt_sd_pipeline import QnnStableDiffusionPipeline
from qairt_sd_controlnet_pipeline import QnnControlNetPipeline
import qairt_constants as consts
from pipeline_utils import UpscalerPipeline, get_upscaler_model

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PipelineCache(metaclass=Singleton):
    pipeline = None

    def __init__(self):
        self.pipeline = None

    def reload_pipeline(self, qnn_model_name):
        if self.pipeline is not None and self.pipeline.model_name == qnn_model_name:
            return qnn_model_name

        print(f"### Loading model {qnn_model_name}")
        if self.pipeline is not None:
            del self.pipeline

        if qnn_model_name in consts.CONTROLNET_MODELS:
            self.pipeline = QnnControlNetPipeline(qnn_model_name)
        elif qnn_model_name in consts.STABLE_DIFFUSION_MODELS:
            self.pipeline = QnnStableDiffusionPipeline(qnn_model_name)
        else:
            raise Exception(f"Unrecognized model {qnn_model_name}!")
        return qnn_model_name
