/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/


#include "UiHelper.h"


UiHelper::UiHelper(std::string config_path, std::string backend, std::string version) {
    outputModelImage = cv::Mat(512, 512, CV_8UC4);
    step_number = 0;
    imageUpdateStatus = false;
    config_file_path = config_path;
    backend_path = backend;
    model_version = version;
    std::ifstream isFilePath(config_file_path);
    if (!isFilePath) {
        std::cerr << "Config file is not a valid one. Input: " << config_file_path << std::endl;
    }
    std::ifstream isFilePathBackend(backend);
    if (!isFilePathBackend) {
        std::cerr << "Backend file is not a valid one. Input: " << backend_path << std::endl;
    }
}



UiHelper::~UiHelper() {
    delete app;

    if (sg_backendHandle) {
        dlclose(sg_backendHandle);
    }
    if (sg_modelHandle) {
        dlclose(sg_modelHandle);
    }
}

bool UiHelper::init() {
    // Load libraries method not required as custom helper is now statically linked
    /*if (0 != app->LoadLibraries("sd_custom_dll.dll")) {
        printf("LoadLibraries failure");
        return false;
    }*/
    int height=0, width=0;
    double bitWidth=0.0;
    if (model_version == VERSION_2_1) {
        height = 1024;
        width = 77;
        bitWidth = 1.0;
    }
    else {
        height = 768;
        width = 77;
        bitWidth = 1.0;
    }
    
    if (0 != app->Init(config_file_path, Helpers::stripPath(config_file_path),
        height, width, bitWidth,
        512, 512, 4.0,
        false, backend_path, model_version)) {
        printf("Initialization failure");
        return false;
    }
    std::cout << "UI Initialization complete" << std::endl;
    return true;
}

void UiHelper::reinit() {
    outputModelImage = cv::Mat(512, 512, CV_8UC4);
    step_number = 0;
    imageUpdateStatus = false;
}



bool UiHelper::executeStableDiffusion(int seed, int step, float scale, std::string input_text) {
    std::string text = input_text.substr(1);
    char buffer[1024]; sprintf(buffer, "%010d%010d%010f%s", seed, step, scale, text.c_str());

    std::string full_text(buffer);
    auto start = std::chrono::steady_clock::now();

    if (true != app->PreProcessInput((void*)full_text.c_str(), full_text.length(), false, false)) {
        printf("PreProcessInput failure");
        return false;
    }

    for (int mStepIdx = 0; mStepIdx < step; mStepIdx++) {
        
        bool runVAE = ((mStepIdx + 1) == step);
        printf("\n");
        if (true != app->RunInference(runVAE)) {
            printf("RunInference failure");
            return false;
        }

        if (true == runVAE) {
            printf("\n");
            if (true != app->PostProcessOutput(false, false, inferenceReturn)) {
                printf("PostProcessOutput failure");
                return false;
            }
            app->unlockPostProcessBufferAccess();
            convertOutputImageToCV();
        }
        step_number = mStepIdx;
    }
    auto stop = std::chrono::steady_clock::now();
    Helpers::logProfile("Overall Inference time: ", start, stop);
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time);

    auto epoch = now.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(epoch).count() % 1000000;

    std::cout << std::put_time(&tm, "%F %T.") << us << std::endl;
    return true;
}

void UiHelper::convertOutputImageToCV() {
    std::memcpy(outputModelImage.data, (unsigned char*)inferenceReturn.m_ImageData, 512 * 512 * 4);
    cv::cvtColor(outputModelImage, outputModelImage, cv::COLOR_BGRA2RGBA);
    imageUpdateStatus = true;
}

cv::Mat UiHelper::getOutputImageCV() {
    imageUpdateStatus = false;
    return outputModelImage;
}

bool UiHelper::getImageUpdateStatus() {
    return imageUpdateStatus;
}

Helpers::InferenceReturn UiHelper::getInferenceReturn() {
    return inferenceReturn;
}

int UiHelper::getStepNumber() {
    return step_number;
}