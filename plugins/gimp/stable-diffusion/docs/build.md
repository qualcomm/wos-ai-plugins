# How to Build

## 1. Pre-requisites
Before you begin, ensure you have the following software installed on your system:

### 1.1 Visual Studio 2026
Install Visual Studio 2026. To add the necessary Arm64 components, import the `components.vsconfig` configuration into the Visual Studio installer.

### 1.2 Rust Programming Language
Download and install Rust from the official Rust website: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

## 2. Building the Plugin
To build the plugin, execute the following command in your power shell with Admin rights.

```powershell
.\build_plugin.ps1
```

This command will compile the C++ project, build the Rust Tokenizer library. Upon successful completion, the plugin will be located in the `.\build\plugin-release\sd-snapdragon` directory.
