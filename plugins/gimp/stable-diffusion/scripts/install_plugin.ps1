<#
=============================================================================

Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

=============================================================================
#>

function Get-DspArch {
    $detectScript = Join-Path $PSScriptRoot "detect_arch.ps1"
    if (Test-Path $detectScript) {
        try {
            $arch = & powershell -NoProfile -ExecutionPolicy Bypass -File $detectScript
            if ($arch -eq 73 -or $arch -eq 81) {
                return [int]$arch
            }
        } catch {}
    }
    return $null
}

function Download-Stable-Diffusion-Model-Data {
    echo "Downloading Stable Diffusion 1.5 Model data..."
    $wc = New-Object System.Net.WebClient
    If (-not (Test-Path "$data_path/betas.bin" -PathType Leaf)) {
        $wc.DownloadFile("$model_data_base_url/betas.bin", "$data_path/betas.bin")
    }
    If (-not (Test-Path "$data_path/lambdas.bin" -PathType Leaf)) {
        $wc.DownloadFile("$model_data_base_url/lambdas.bin", "$data_path/lambdas.bin")
    }
    If (-not (Test-Path "$data_path/openai-clip-vit-base-patch32" -PathType Leaf)) {
        $wc.DownloadFile("$model_data_base_url/openai-clip-vit-base-patch32", "$data_path/openai-clip-vit-base-patch32")
    }
    If (-not (Test-Path "$data_path/sd_precomute_data.tar" -PathType Leaf)) {
        $wc.DownloadFile("$model_data_base_url/sd_precomute_data.tar", "$data_path/sd_precomute_data.tar")
    }
}

function Download-Stable-Diffusion-Models {
    echo "Downloading Stable Diffusion 1.5 QNN context binaries..."
    $binary_zip = "$current_dir/sd_model_binaries.zip"
    $binary_extract_dir = "$current_dir/sd_model_binaries_extracted"
    (New-Object System.Net.WebClient).DownloadFile($binary_zip_url, "$binary_zip")
    echo "Extracting model binaries..."
    Expand-Archive -Path "$binary_zip" -DestinationPath "$binary_extract_dir" -Force
    echo "Copying model binaries (.bin files) to data directory..."
    Get-ChildItem -Path "$binary_extract_dir" -Filter "*.bin" -Recurse | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination "$data_path" -Force
        echo "  Copied: $($_.Name)"
    }
    Remove-Item -Recurse -Force "$binary_extract_dir"
    Remove-Item -Force "$binary_zip"
}

function Download-QNN-SDK-Libraries {
    echo "Downloading QNN SDK..."
    If (-not (Test-Path "$current_dir/qairt/$qnn_sdk_version")) {
        (New-Object System.Net.WebClient).DownloadFile($qnn_sdk_url, "$current_dir/qnn_sdk.zip")
        Expand-Archive "$current_dir/qnn_sdk.zip" -DestinationPath "$current_dir/"
        Remove-Item "$current_dir/qnn_sdk.zip"
    }
    $qnn_sdk_root = "$current_dir/qairt/$qnn_sdk_version"

    cp "$qnn_sdk_root/lib/arm64x-windows-msvc/QnnHtp.dll" "$current_dir/$plugin_name/"
    cp "$qnn_sdk_root/lib/arm64x-windows-msvc/QnnHtpPrepare.dll" "$current_dir/$plugin_name/"
    cp "$qnn_sdk_root/lib/arm64x-windows-msvc/QnnHtpNetRunExtensions.dll" "$current_dir/$plugin_name/"
    cp "$qnn_sdk_root/lib/arm64x-windows-msvc/QnnSystem.dll" "$current_dir/$plugin_name/"

    $skelStubArchs = if ($dsp_arch_known) { @($dsp_arch) } else { @(73, 81) }
    foreach ($v in $skelStubArchs) {
        cp "$qnn_sdk_root/lib/arm64x-windows-msvc/QnnHtpV${v}Stub.dll" "$current_dir/$plugin_name/"
        cp "$qnn_sdk_root/lib/hexagon-v${v}/unsigned/libQnnHtpV${v}Skel.so" "$current_dir/$plugin_name/"
		cp "$qnn_sdk_root/lib/hexagon-v${v}/unsigned/libqnnhtpv${v}.cat" "$current_dir/$plugin_name/"
    }
}

function Copy-Plugin-To-Gimp {
    Copy-Item -Path "$current_dir\$plugin_name" -Destination "$gimp_plugin_path" -Recurse -Force
}

try {
    $qnn_sdk_version = "2.46.0.260424"
    $plugin_name = "sd-snapdragon"
    $ai_hub_base_url = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com"
    $qnn_sdk_url = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$qnn_sdk_version/v$qnn_sdk_version.zip"
    $model_data_base_url = "$ai_hub_base_url/qai-hub-models/models/riffusion_quantized/v1"

    echo "Detecting DSP architecture..."
    $detected = Get-DspArch
    $dsp_arch_known = $null -ne $detected
    $dsp_arch = if ($dsp_arch_known) { $detected } else { 73 }
    if ($dsp_arch_known) {
        echo "Detected DSP architecture: v$dsp_arch"
    } else {
        echo "DSP architecture unknown. Defaulting to v73."
    }

    $binary_zip_urls = @{
        73 = "$ai_hub_base_url/qai-hub-models/models/stable_diffusion_v1_5/releases/v0.53.1/stable_diffusion_v1_5-qnn_context_binary-w8a16-qualcomm_snapdragon_x_elite.zip"
        81 = "$ai_hub_base_url/qai-hub-models/models/stable_diffusion_v1_5/releases/v0.53.1/stable_diffusion_v1_5-qnn_context_binary-w8a16-qualcomm_snapdragon_x2_elite.zip"
    }
    $binary_zip_url = $binary_zip_urls[$dsp_arch]
    $gimp_plugin_path = "C:\Users\$Env:UserName\AppData\Roaming\GIMP\2.99\plug-ins"
    $ErrorActionPreference = "Stop"
    $initial_dir = (Get-Item .).FullName
    $current_dir = $PSScriptRoot
    $data_path = "$current_dir/$plugin_name/StableDiffusionData"
    New-Item "$data_path" -ItemType directory -ea 0
    New-Item "$gimp_plugin_path" -ItemType directory -ea 0

    If (-not (Test-Path "$current_dir/$plugin_name")) {
        throw "Didn't find '$plugin_name' under $current_dir"
    }

    Download-QNN-SDK-Libraries
    Download-Stable-Diffusion-Model-Data
    Download-Stable-Diffusion-Models
    Copy-Plugin-To-Gimp
    echo "Plugin installation done!"
}
finally {
    Set-Location -Path "$initial_dir"
}
