# =============================================================================
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================


import os
from modules.paths_internal import script_path, extensions_dir, models_path
from pathlib import Path


QAI_APPBUILDER_WHEEL_URL = "https://github.com/quic/ai-engine-direct-helper/releases/download/v2.38.0/qai_appbuilder-2.38.0-cp310-cp310-win_amd64.whl"
QAIRT_VERSION = "2.46.0.260424"
QNN_SDK_DOWNLOAD_URL = f"https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/{QAIRT_VERSION}/v{QAIRT_VERSION}.zip"
DSP_ARCHS = ["73", "81"]

# BIOS SoC ID -> DSP architecture version mapping (mirrors get_arch_id.py)
BIOS_ARCH_MAP = {
    8380: 73,   # Snapdragon X Elite
    8480: 81,   # Snapdragon X2 Elite
}

# Architecture-specific binary ZIP URLs (QAI Hub S3, v0.53.1)
ARCH_BINARY_URLS = {
    73: {
        "controlnet": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/controlnet_canny/releases/v0.53.1/controlnet_canny-qnn_context_binary-w8a16-qualcomm_snapdragon_x_elite.zip",
        "sd1_5":      "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/stable_diffusion_v1_5/releases/v0.53.1/stable_diffusion_v1_5-qnn_context_binary-w8a16-qualcomm_snapdragon_x_elite.zip",
        "sd2_1":      "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/stable_diffusion_v2_1/releases/v0.53.1/stable_diffusion_v2_1-qnn_context_binary-w8a16-qualcomm_snapdragon_x_elite.zip",
    },
    81: {
        "controlnet": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/controlnet_canny/releases/v0.53.1/controlnet_canny-qnn_context_binary-w8a16-qualcomm_snapdragon_x2_elite.zip",
        "sd1_5":      "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/stable_diffusion_v1_5/releases/v0.53.1/stable_diffusion_v1_5-qnn_context_binary-w8a16-qualcomm_snapdragon_x2_elite.zip",
        "sd2_1":      "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/stable_diffusion_v2_1/releases/v0.53.1/stable_diffusion_v2_1-qnn_context_binary-w8a16-qualcomm_snapdragon_x2_elite.zip",
    },
}
EXTENSION_WS = os.path.join(extensions_dir, "qairt_accelerate")
QNN_LIBS_DIR = os.path.join(EXTENSION_WS, "qnn_assets", "qnn_libs")
CACHE_DIR = os.path.join(EXTENSION_WS, "qnn_assets", "cache")


SDK_SAVE_PATH = os.path.join(EXTENSION_WS, f"{QAIRT_VERSION}.zip")
QAIRT_DIR = r"C:\Qualcomm\AIStack\QAIRT"

env_path = os.environ.get("QNN_SDK_ROOT")
if env_path:
    QNN_SDK_ROOT = Path(env_path)
else:
    QNN_SDK_ROOT = Path(rf"C:\Qualcomm\AIStack\QAIRT\{QAIRT_VERSION}")

DEFAULT_TXT2IMG_MODEL = "Stable-Diffusion-1.5"
DEFAULT_IMG2IMG_MODEL = "ControlNet-v10-sd15-canny"

CONTROLNET_MODELS = [DEFAULT_IMG2IMG_MODEL]
STABLE_DIFFUSION_MODELS = [DEFAULT_TXT2IMG_MODEL, "Stable-Diffusion-2.1"]

CONTROLNET_DIR = os.path.abspath(os.path.join(models_path, "Stable-diffusion", f"qcom-{'ControlNet-v10-sd15-canny'}"))
SD1_5_DIR = os.path.abspath(os.path.join(models_path, "Stable-diffusion", f"qcom-{'Stable-Diffusion-v1.5'}"))
SD2_1_DIR = os.path.abspath(os.path.join(models_path, "Stable-diffusion", f"qcom-{'Stable-Diffusion-v2.1'}"))
