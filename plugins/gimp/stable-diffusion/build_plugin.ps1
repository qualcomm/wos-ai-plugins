<#
=============================================================================

Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

=============================================================================
#>

function Prepare-Plugin-Release {
    New-Item "$current_dir/build/plugin-release/sd-snapdragon" -ItemType directory -ea 0
    cp "$current_dir/build/Release/*" "$current_dir/build/plugin-release/sd-snapdragon/"
    cp "$current_dir/ui/*.py" "$current_dir/build/plugin-release/sd-snapdragon/"
    cp "$current_dir/config/*" "$current_dir/build/plugin-release/sd-snapdragon/"
    cp "$current_dir/resources/*" "$current_dir/build/plugin-release/sd-snapdragon/"
    cp "$current_dir/src/tokenizer/target/aarch64-pc-windows-msvc/release/tokenizer.dll" `
        "$current_dir/build/plugin-release/sd-snapdragon/"
    cp "$current_dir/scripts/install_plugin.ps1" "$current_dir/build/plugin-release/"
    cp "$current_dir/scripts/detect_arch.ps1" "$current_dir/build/plugin-release/"
    cp "$current_dir/scripts/sd-snapdragon-cli.py" "$current_dir/build/plugin-release/sd-snapdragon/"
}

function Build-Tokenizer {
    Set-Location -Path "$current_dir/src/tokenizer"
    rustup target add aarch64-pc-windows-msvc
    cargo build --release --target aarch64-pc-windows-msvc
}

function Download-QNN-SDK {
    If (-not (Test-Path "$sdk_root_path")) {
        Invoke-WebRequest "$qnn_sdk_url" -OutFile "$current_dir/build/qnn_sdk.zip"
        Expand-Archive "$current_dir/build/qnn_sdk.zip" -DestinationPath "$current_dir/build/"
        Move-Item -Path "$current_dir/build/qairt/$qnn_sdk_version" -Destination "$sdk_root_path"
    }
    $env:QNN_SDK_ROOT = "$sdk_root_path"
    write-output "QNN SDK root : " $sdk_root_path
}

function Run-vcpkg {
    write-output "vcpkg root: " $vcpkg_root
    If (-not (Test-Path $vcpkg_root)){
        cd "$current_dir/build"
        git clone "$vcpkg_url"
        .\vcpkg\bootstrap-vcpkg.bat
        .\vcpkg\vcpkg integrate install
    }
}

function Build-Stable-Diffusion {
    cd "$current_dir/build"
    cmake -G "Visual Studio 18 2026" -A ARM64 `
        -DCMAKE_TOOLCHAIN_FILE="$vcpkg_root/scripts/buildsystems/vcpkg.cmake" ..
    cmake --build . --config Release
}

try {
    $qnn_sdk_version = "2.46.0.260424"
    $vcpkg_url = "https://github.com/microsoft/vcpkg"
    $qnn_sdk_url = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$qnn_sdk_version/v$qnn_sdk_version.zip"
    $sdk_qairt_path = "C:/Qualcomm/AIStack/QAIRT"
    $sdk_root_path = "$sdk_qairt_path/$qnn_sdk_version"
    $ErrorActionPreference = "Stop"
    $initial_dir = (Get-Item .).FullName
    $current_dir = $PSScriptRoot
    $vcpkg_root = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath("$current_dir\build\vcpkg")

    New-Item "$current_dir/build" -ItemType directory -ea 0
    cp "$current_dir/vcpkg.json" "$current_dir/build/"
    cp "$current_dir/vcpkg-configuration.json" "$current_dir/build/"

    Download-QNN-SDK
    Build-Tokenizer
    Run-vcpkg
    Build-Stable-Diffusion
    Prepare-Plugin-Release
}
finally {
    Set-Location -Path "$initial_dir"
}
