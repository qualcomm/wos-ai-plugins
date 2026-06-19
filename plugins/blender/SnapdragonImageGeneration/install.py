# =============================================================================
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================

import os
import qairt_constants as consts
import zipfile
import shutil
import requests
import subprocess
from detect_htp_arch import get_htp_architecture


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = True) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "errors": "ignore",
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return result.stdout or ""


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def download_qairt_sdk():
    # Setup QAIRT SDK
    if not os.path.isdir(consts.QNN_SDK_ROOT):
        os.makedirs(consts.QAIRT_DIR, exist_ok=True)
        print(f"Downloading QAIRT SDK...")
        download_url(consts.QNN_SDK_DOWNLOAD_URL, consts.SDK_SAVE_PATH)

        with zipfile.ZipFile(consts.SDK_SAVE_PATH, "r") as zip_ref:
            zip_ref.extractall(consts.LONG_PATH_PREFIX_PLUGIN_DIR)
        shutil.move(
            os.path.join(consts.PLUGIN_DIR, "qairt", consts.QAIRT_VERSION),
            os.path.join(consts.QNN_SDK_ROOT, ".."),
        )
        shutil.rmtree(os.path.join(consts.PLUGIN_DIR, "qairt"))
        os.remove(consts.SDK_SAVE_PATH)


def setup_qairt_env():
    """Copy QNN runtime libraries for the detected architecture into QNN_LIBS_DIR."""
    arch = get_htp_architecture()

    SDK_lib_dir = os.path.join(consts.QNN_SDK_ROOT, "lib", "arm64x-windows-msvc")

    # Common libraries required regardless of architecture
    common_libs = [
        "QnnHtp.dll",
        "QnnSystem.dll",
        "QnnHtpPrepare.dll",
    ]
    for lib in common_libs:
        src = os.path.join(SDK_lib_dir, lib)
        dst = os.path.join(consts.QNN_LIBS_DIR, lib)
        if not os.path.isfile(dst):
            shutil.copy(src, dst)

    if arch is not None:
        archs_to_copy = [str(arch)]
        print(f"Detected architecture: {arch}. Copying arch-specific libraries for v{arch}.")
    else:
        archs_to_copy = ["73", "81"] 
        print("WARNING: Could not detect architecture. Defaulting to archs 73 and 81.")

    for arch_str in archs_to_copy:
        stub_lib = f"QnnHtpV{arch_str}Stub.dll"
        src = os.path.join(SDK_lib_dir, stub_lib)
        dst = os.path.join(consts.QNN_LIBS_DIR, stub_lib)
        if not os.path.isfile(dst):
            shutil.copy(src, dst)

        SDK_hexagon_dir = os.path.join(
            consts.QNN_SDK_ROOT, "lib", f"hexagon-v{arch_str}", "unsigned"
        )
        hexagon_libs = [
            f"libQnnHtpV{arch_str}Skel.so",
            f"libqnnhtpv{arch_str}.cat",
        ]
        for lib in hexagon_libs:
            src = os.path.join(SDK_hexagon_dir, lib)
            dst = os.path.join(consts.QNN_LIBS_DIR, lib)
            if not os.path.isfile(dst):
                shutil.copy(src, dst)


def _map_bin_filename(filename):
    fname_lower = filename.lower()
    if "text_encoder" in fname_lower or "text-encoder" in fname_lower:
        return "text_encoder.bin"
    elif "unet" in fname_lower:
        return "unet.bin"
    elif "vae" in fname_lower:
        return "vae.bin"
    elif "controlnet" in fname_lower:
        return "controlnet.bin"
    return None  


def controlnet_download():
    os.makedirs(consts.CONTROLNET_DIR, exist_ok=True)
    # Detect architecture
    arch = get_htp_architecture()
    if arch is None:
        arch = 73
        print(f"WARNING: Could not detect architecture. Defaulting to arch {arch}.")

    # Skip download if all expected binaries are already present
    expected_files = ["controlnet.bin", "text_encoder.bin", "unet.bin", "vae.bin"]
    if all(os.path.isfile(os.path.join(consts.CONTROLNET_DIR, f)) for f in expected_files):
        print("ControlNet binaries already present, skipping download.")
        return

    url = consts.CONTROLNET_BINARY_URLS[arch]
    zip_filename = url.split("/")[-1]
    zip_save_path = os.path.join(consts.CONTROLNET_DIR, zip_filename)

    print(f"Detected architecture: {arch}")
    print(f"Downloading ControlNet-Canny binaries (arch {arch})...")
    print(f"  URL : {url}")
    download_url(url, zip_save_path)

    print("Extracting binaries...")
    with zipfile.ZipFile(zip_save_path, "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".bin"):
                continue
            src_name = os.path.basename(member)
            dest_name = _map_bin_filename(src_name)
            if dest_name is None:
                print(f"  Skipping unrecognised file: {src_name}")
                continue
            dest_path = os.path.join(consts.CONTROLNET_DIR, dest_name)
            with zf.open(member) as src_file, open(dest_path, "wb") as dst_file:
                dst_file.write(src_file.read())
            print(f"  Extracted: {src_name}  →  {dest_name}")

    os.remove(zip_save_path)
    print("ControlNet binaries ready.")


print("Downloading QAIRT model bin files...")

os.makedirs(consts.QNN_LIBS_DIR, exist_ok=True)
os.makedirs(consts.CONTROLNET_DIR, exist_ok=True)
os.makedirs(consts.LOGS_DIR, exist_ok=True)

download_qairt_sdk()
setup_qairt_env()
controlnet_download()
