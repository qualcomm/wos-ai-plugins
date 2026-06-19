# =============================================================================
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================


import os
import pathlib

# Function to convert a given path to UNC path
def to_unc_path(path):
    return f"\\\\?\\{path}"

PLUGIN_DIR=pathlib.Path(__file__).parent.resolve()
LONG_PATH_PREFIX_PLUGIN_DIR=to_unc_path(PLUGIN_DIR)
VENV_PATH=os.path.join(PLUGIN_DIR, "venv")
VENV_PYTHON=os.path.join(VENV_PATH, "Scripts", "python.exe")

CONTROLNET_PIPELINE=os.path.join(PLUGIN_DIR, "model_pipelines", "ControlNetCanny.py")
OUTPUTS_DIR=os.path.join(PLUGIN_DIR, "Outputs")
LOGS_DIR=os.path.join(PLUGIN_DIR, "logs")
LOG_FILE=os.path.join(LOGS_DIR, "model_log.txt")


QAIRT_VERSION = "2.46.0.260424"
QNN_SDK_DOWNLOAD_URL = f"https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/{QAIRT_VERSION}/v{QAIRT_VERSION}.zip"

DSP_ARCH = ["73", "81"]  

QNN_LIBS_DIR = os.path.join(PLUGIN_DIR, "qnn_assets", "qnn_libs")
SDK_SAVE_PATH= os.path.join(LONG_PATH_PREFIX_PLUGIN_DIR, f"{QAIRT_VERSION}.zip")
QAIRT_DIR=f"C:\\Qualcomm\\AIStack\\QAIRT"
QNN_SDK_ROOT=f"C:\\Qualcomm\\AIStack\\QAIRT\\{QAIRT_VERSION}"

CONTROLNET_DIR=os.path.join(PLUGIN_DIR, "qnn_assets", "models", "controlnet")
CACHE_DIR = os.path.join(PLUGIN_DIR, "qnn_assets", "models", "cache")
CONVERTION_DIR = os.path.join(PLUGIN_DIR, "model_conversion")


CONTROLNET_BINARY_URLS = {
    73: "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/controlnet_canny/releases/v0.53.1/controlnet_canny-qnn_context_binary-w8a16-qualcomm_snapdragon_x_elite.zip",
    81: "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/controlnet_canny/releases/v0.53.1/controlnet_canny-qnn_context_binary-w8a16-qualcomm_snapdragon_x2_elite.zip",
}

BIOS_ARCH_MAP = {
    8380: 73,
    8480: 81,
}
