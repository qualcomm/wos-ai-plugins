/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/


#include "RuntimeApiHelpers.hpp"

RuntimeApiHelpers::RuntimeApiHelpers()
{
    m_preprocess_count = 0;
    m_inference_count = 0;
    m_postprocess_count = 0;
}

RuntimeApiHelpers::~RuntimeApiHelpers()
{
    DEMO_DEBUG("Tearing Down Original Data Pre Bank");
    tearDownOrgDataBuf(m_OrgDataPre);
    DEMO_DEBUG("Tearing Down Original Data Post Bank");
    tearDownOrgDataBuf(m_OrgDataPost);

    DEMO_DEBUG("Tearing Down Input Tensor Buffer Bank");
    tearDownTensorsBufBank(m_InputTensorsBufBank);
    DEMO_DEBUG("Tearing Down Output Tensor Buffer Bank");
    tearDownTensorsBufBank(m_OutputTensorsBufBank);

    if(m_CustomLibHandle)
    {
        DEMO_DEBUG("Closing custom Lib");
        dlclose(m_CustomLibHandle);
    }
}


int32_t RuntimeApiHelpers::LoadLibraries(std::string customLibName){
    // Lets open up our custom library and set it to global so that we can use it multiple times
    DEMO_DEBUG("customLibName %s", customLibName.c_str());
#ifdef UWP
    m_CustomLibHandle = LoadLibraryExW(std::wstring(customLibName.begin(), customLibName.end()).c_str(), NULL, 0);
#else
    m_CustomLibHandle = dlopen(customLibName.c_str(), RTLD_NOW);
#endif // UWP

        if(nullptr == m_CustomLibHandle){
        DEMO_ERROR("Problem loading CustomHelper Library %s:%s \n Cannot proceed.",
                            customLibName.c_str(), dlerror());
        return -1;
    }

    /** Now that we have a handle for the custom library we will populate the global variables for each helper function
    */

    // get a reference to the custom_Helpers Init function
#ifdef UWP
    m_PtrCustInit = (m_CustInitPtr)GetProcAddress((HMODULE)m_CustomLibHandle, "Init");
#else
    m_PtrCustInit = (m_CustInitPtr)dlsym(m_CustomLibHandle, "Init");
#endif
    if (nullptr == m_PtrCustInit) {
        DEMO_DEBUG("Did not find Init function: %s \n App will be only using built-in.",
                            dlerror());
    }


    // get a reference to the custom_Helpers PreprocessInput function
    // reset errors
#ifdef UWP
    m_PtrCustPre = (m_CustPrePtr)GetProcAddress((HMODULE)m_CustomLibHandle, "PreProcessInput");
#else
    m_PtrCustPre = (m_CustPrePtr) dlsym(m_CustomLibHandle, "PreProcessInput");
#endif
    if (nullptr == m_PtrCustPre) {
        DEMO_DEBUG("Did not find preprocess function: %s \n App will be using only built-in.",
                            dlerror());
    }

    // Get a reference to the custom Helpers user buffer post process function.
    // reset errors
#ifdef UWP
    m_PtrCustPost = (m_CustPostPtr)GetProcAddress((HMODULE)m_CustomLibHandle, "PostProcessOutput");
#else
    m_PtrCustPost = (m_CustPostPtr)dlsym(m_CustomLibHandle, "PostProcessOutput");
#endif
    if (nullptr == m_PtrCustPost) {
        DEMO_DEBUG("Can't find PostprocessOutput function. %s \n App will be using only built-in.",
                            dlerror());
    }

    return 0;
}

int32_t RuntimeApiHelpers::setEnvVariable(std::string envVariable, std::string value)
{
    int32_t ret = 0;//SetEnvironmentVariable(envVariable.c_str(), value.c_str());
    if (ret != 0)
    {
       DEMO_ERROR("Unable to set environment variable %s", envVariable.c_str());
    }

    return ret;
}

void RuntimeApiHelpers::tearDownTensorsBufBank(
        std::vector<std::unordered_map<std::string, std::unordered_map<std::string, void*>>>& tensorsBufBank) {
    for (auto& graphTensorBuf : tensorsBufBank) {
        for (auto& graphTensor : graphTensorBuf) {
            for (auto& tensor : graphTensor.second) {
                if (nullptr != tensor.second) {
                    if (m_FreeTensorsPointerSet.end() == m_FreeTensorsPointerSet.find(tensor.second)) {
                        free(tensor.second);
                        m_FreeTensorsPointerSet.insert(tensor.second);
                    }
                    tensor.second = nullptr;
                }
            }
        }
    }
}

void RuntimeApiHelpers::tearDownOrgDataBuf(std::vector<void*>& orgDataBuf) {
    for (size_t dataBufIdx = 0; dataBufIdx < orgDataBuf.size(); dataBufIdx++) {
        if (nullptr != orgDataBuf[dataBufIdx]) {
          free(orgDataBuf[dataBufIdx]);
          orgDataBuf[dataBufIdx] = nullptr;
        }
    }
}
