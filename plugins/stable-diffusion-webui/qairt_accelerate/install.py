# =============================================================================
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================

import launch
import os
import qairt_constants as consts
import platform


# check python version to be 3.10.6+
if (not platform.python_version().startswith("3.10.")) or (int(platform.python_version().split(".")[2])<6):
    raise Exception("Python version needs to be >=3.10.6 and <3.10.14")


launch.run_pip(
    f"install {consts.QAI_APPBUILDER_WHEEL_URL}",
    "Install python QNN",
)
if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers", "diffusers")
if not launch.is_installed("onnx"):
    launch.run_pip("install onnx", "onnx")

import common_utils as utils
import zipfile
import shutil
import requests
from detect_htp_arch import get_htp_architecture

os.makedirs(consts.EXTENSION_WS, exist_ok=True)

COMPONENT_KEYWORD_MAP = [
    ("text_encoder", "text_encoder.serialized.bin"),
    ("textencoder",  "text_encoder.serialized.bin"),
    ("text",         "text_encoder.serialized.bin"),
    ("vae_decoder",  "vae_decoder.serialized.bin"),
    ("vaedecoder",   "vae_decoder.serialized.bin"),
    ("vae",          "vae_decoder.serialized.bin"),
    ("unet",         "unet.serialized.bin"),
    ("controlnet",   "controlnet.serialized.bin"),
]


SD_EXPECTED_BINARIES = [
    "text_encoder.serialized.bin",
    "unet.serialized.bin",
    "vae_decoder.serialized.bin",
]

CONTROLNET_EXPECTED_BINARIES = [
    "controlnet.serialized.bin",
    "text_encoder.serialized.bin",
    "unet.serialized.bin",
    "vae_decoder.serialized.bin",
]


def _binaries_present(target_dir: str, expected: list) -> bool:
    """Return True only when every expected file already exists."""
    return all(os.path.isfile(os.path.join(target_dir, f)) for f in expected)


def extract_binaries_from_zip(zip_path: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        bin_entries = [name for name in zf.namelist() if name.lower().endswith(".bin")]
        if not bin_entries:
            raise RuntimeError(f"No .bin files found inside {zip_path}")

        for entry in bin_entries:
            basename = os.path.basename(entry).lower()
            target_name = None
            for keyword, mapped_name in COMPONENT_KEYWORD_MAP:
                if keyword in basename:
                    target_name = mapped_name
                    break

            if target_name is None:
                print(f"  [install] Warning: cannot map '{os.path.basename(entry)}' "
                      f"to a known component – skipping.")
                continue

            target_path = os.path.join(target_dir, target_name)
            if os.path.isfile(target_path):
                print(f"  [install] Already present, skipping: {target_name}")
                continue

            print(f"  [install] Extracting {os.path.basename(entry)} -> {target_name}")
            with zf.open(entry) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _download_and_extract(url: str, zip_save_path: str, target_dir: str,
                          expected_files: list, label: str) -> None:
    """Download *url* as a ZIP, extract binaries, then remove the ZIP."""
    if _binaries_present(target_dir, expected_files):
        print(f"[install] {label} binaries already present – skipping download.")
        return

    print(f"[install] Downloading {label} binaries...")
    utils.download_url(url, zip_save_path)

    print(f"[install] Extracting {label} binaries to {target_dir} ...")
    extract_binaries_from_zip(zip_save_path, target_dir)

    os.remove(zip_save_path)
    print(f"[install] {label} binaries ready.")


def download_qairt_sdk():
    # Setup QAIRT
    if not os.path.isdir(consts.QNN_SDK_ROOT):
        os.makedirs(consts.QAIRT_DIR, exist_ok=True)
        print(f"Downloading QAIRT SDK...")
        utils.download_url(consts.QNN_SDK_DOWNLOAD_URL, consts.SDK_SAVE_PATH)

        with zipfile.ZipFile(consts.SDK_SAVE_PATH, "r") as zip_ref:
            zip_ref.extractall(consts.EXTENSION_WS)
        shutil.move(
            os.path.join(consts.EXTENSION_WS, "qairt", consts.QAIRT_VERSION),
            os.path.join(consts.QNN_SDK_ROOT, ".."),
        )
        shutil.rmtree(os.path.join(consts.EXTENSION_WS, "qairt"))
        os.remove(consts.SDK_SAVE_PATH)


def setup_qairt_env(arch=None):
    SDK_lib_dir = os.path.join(consts.QNN_SDK_ROOT, "lib", "arm64x-windows-msvc")
    os.makedirs(consts.QNN_LIBS_DIR, exist_ok=True)

    # Copy common libraries
    common_libs = ["QnnHtp.dll", "QnnSystem.dll", "QnnHtpPrepare.dll"]
    for lib in common_libs:
        src = os.path.join(SDK_lib_dir, lib)
        dst = os.path.join(consts.QNN_LIBS_DIR, lib)
        if os.path.isfile(dst):
            os.remove(dst)
        shutil.copy(src, dst)

    if arch is not None:
        archs_to_copy = [str(arch)]
        print(f"[install] Copying stub/skel for detected arch v{arch}.")
    else:
        archs_to_copy = ["73", "81"]
        print(f"[install] Architecture unknown – copying stub/skel for default archs: {archs_to_copy}.")

    for arch_str in archs_to_copy:
        SDK_hexagon_dir = os.path.join(
            consts.QNN_SDK_ROOT, "lib", f"hexagon-v{arch_str}", "unsigned"
        )

        arch_specific_libs = [f"QnnHtpV{arch_str}Stub.dll"]
        hexagon_libs = [
            f"libQnnHtpV{arch_str}Skel.so",
            f"libqnnhtpv{arch_str}.cat",
        ]

        for lib in arch_specific_libs:
            src = os.path.join(SDK_lib_dir, lib)
            dst = os.path.join(consts.QNN_LIBS_DIR, lib)
            if os.path.isfile(dst):
                os.remove(dst)
            shutil.copy(src, dst)
            print(f"  [install] Copied {lib}")

        for lib in hexagon_libs:
            src = os.path.join(SDK_hexagon_dir, lib)
            dst = os.path.join(consts.QNN_LIBS_DIR, lib)
            if os.path.isfile(dst):
                os.remove(dst)
            shutil.copy(src, dst)
            print(f"  [install] Copied {lib}")


def main():
    arch = get_htp_architecture()
    if arch is None:
        print("[install] WARNING: Could not detect DSP architecture. "
              "Defaulting to arch 73 for binary download.")
        arch = 73
    else:
        print(f"[install] Detected DSP architecture: v{arch} "
              f"({'Snapdragon X Elite' if arch == 73 else 'Snapdragon X2 Elite'})")

    download_qairt_sdk()
    setup_qairt_env(arch)

    if arch not in consts.ARCH_BINARY_URLS:
        raise RuntimeError(
            f"[install] No binary URLs defined for architecture v{arch}. "
            f"Supported architectures: {list(consts.ARCH_BINARY_URLS.keys())}"
        )
    arch_urls = consts.ARCH_BINARY_URLS[arch]

    os.makedirs(consts.EXTENSION_WS, exist_ok=True)
    controlnet_zip = os.path.join(consts.EXTENSION_WS, "controlnet_binaries.zip")
    sd15_zip       = os.path.join(consts.EXTENSION_WS, "sd15_binaries.zip")
    sd21_zip       = os.path.join(consts.EXTENSION_WS, "sd21_binaries.zip")

    _download_and_extract(
        url=arch_urls["controlnet"],
        zip_save_path=controlnet_zip,
        target_dir=consts.CONTROLNET_DIR,
        expected_files=CONTROLNET_EXPECTED_BINARIES,
        label=f"ControlNet (arch v{arch})",
    )

    _download_and_extract(
        url=arch_urls["sd1_5"],
        zip_save_path=sd15_zip,
        target_dir=consts.SD1_5_DIR,
        expected_files=SD_EXPECTED_BINARIES,
        label=f"SD 1.5 (arch v{arch})",
    )

    _download_and_extract(
        url=arch_urls["sd2_1"],
        zip_save_path=sd21_zip,
        target_dir=consts.SD2_1_DIR,
        expected_files=SD_EXPECTED_BINARIES,
        label=f"SD 2.1 (arch v{arch})",
    )

if __name__ == "__main__":
    main()
