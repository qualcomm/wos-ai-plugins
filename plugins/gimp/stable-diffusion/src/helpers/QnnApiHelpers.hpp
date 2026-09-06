/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/

#ifndef _QNNAPIHELPERS_HPP_
#define _QNNAPIHELPERS_HPP_


//used for throwing errors back to Java
#include "Helpers.hpp"
#include "RuntimeApiHelpers.hpp"

#include "QnnApi.hpp"
#include "IOTensor.hpp"

#include "DataLoader.h"

#include "Scheduler.hpp"

#include "StableDiffusionHelper.hpp"

extern "C" void set_datapath(const char* text);
extern "C" size_t get_token_ids(const char* text, uint32_t* buffer, size_t buffer_length);
#define TOKEN_IDS_LEN 77

//*** Debugging knobs ***
// On-target folder to dump data
#define DEBUG_DUMP_TARGET_PATH  "output"

// Uncomment this to save images for debugging purposes
// #define DEBUG_DUMP
// #define OUTPUT_DUMP
//#define PRELOAD_DATA

// #define USE_PRELOADED_IMAGE

class QnnApiHelpers : public RuntimeApiHelpers
{
public:

    QnnApiHelpers();
    ~QnnApiHelpers();

    // Definition of virtual functions from RuntimeApiHelpers abstract class
    int32_t Init (std::string configFilePath, std::string nativeLibPath,
                  int32_t inputWidth, int32_t inputHeight, float inputChannel,
                  int32_t outputWidth, int32_t outputHeight, float outputChannel,
                  bool debugModeRequested, std::string runtimeNameOrFile, std::string model_version
                 );
    bool PreProcessInput(void* image,
                         uint32_t imageSize,
                         bool isFlipped,
                         bool overlayOnImage
                        );
    bool RunInference(bool runVAE,
                      bool dumpOutput,
                      std::string outputLocation
                     );
    bool PostProcessOutput(bool showLabels,
                           bool showConfScores,
                           Helpers::InferenceReturn &returnVal,
                           bool dumpPostOutput,
                           std::string outputLocation
                          );

    /**
    * @brief template function for executing HRNET. The input/output can be either user buffer or HRNET tensor.
    * @param input: the current frame to process
    * @param output: the result from model after running it using HRNET.

    * @return: true if no error, false otherwise
    */
    template<class T1, class T2>
    inline bool ExecuteModel ( T1& input, T2& output, std::string graphName )
    {
        // given that a dnn instance is created and we have input loaded with image data we can get our output
        // for our required app functionality Execute the network with the given single input.
        QNN_DEBUG("Now executing inference for graph %s", graphName.c_str());

        auto start = std::chrono::steady_clock::now();
        bool ret = m_qnnApi->graphExecute(input, output, graphName);
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("inference (cpp) took", start, stop);
        if(ret != true)
        {
            QNN_ERROR("ERROR executing inference: %d for graph %s", ret, graphName.c_str());
            return false;
        }
        QNN_DEBUG("Execute finished for graph %s", graphName.c_str());

        return true;
    }

#if defined(DEBUG_DUMP) || defined(OUTPUT_DUMP) || defined(PRELOAD_DATA)
    void setSampleNum(int sample_num) {
        char buffer[20]; sprintf(buffer, "%03d", sample_num);
        m_SampleNum = std::string(buffer);
    }
#endif
    //size_t get_token_ids(const char* text, uint32_t* buffer, size_t buffer_length);

private:
    // Function to append the debug dump path and then return the full path to the file name
#if defined(DEBUG_DUMP) || defined(OUTPUT_DUMP) || defined(PRELOAD_DATA)
    inline std::string getDebugFile(const std::string& filename) {
        return Helpers::joinPath(Helpers::joinPath(DEBUG_DUMP_TARGET_PATH, "sample_"+m_SampleNum), filename);
    }

    bool readTensorData(Qnn_Tensor_t* tensor, const std::string& filename) {
        return Helpers::readRawData(m_ioTensor->getBuffer(tensor), m_ioTensor->getBufferSize(tensor), filename);
    }

    bool writeTensorData(Qnn_Tensor_t* tensor, const std::string& filename) {
        return Helpers::writeRawData(m_ioTensor->getBuffer(tensor), m_ioTensor->getBufferSize(tensor), filename);
    }
#endif

    /**
    * @brief parses a key=value config value and sets the relevant class level variables
    * @param configFilePath: the path to the configuration file

    * @return: 0 if no errors, -1 otherwise
    */
    int32_t parseConfigPath(std::string configFilePath);

    // QNN specific variables
    std::unique_ptr<QnnApi> m_qnnApi;

    std::string m_backendExtensionsLibPath;
    std::string m_backendExtensionsConfigPath;
    bool m_LoadFromCachedBinary{false};

    bool m_sharedBuffer{false};
    std::unique_ptr<IOTensor> m_ioTensor;

    // Off-Target Data loader specific variable
    DataLoader* m_offTargetDataLoader;
    std::string m_dataLoaderInputTarfile;

    // Scheduler specific variables
    DPMSolverMultistepScheduler* m_schedulerSolver;
    // Memory variable to point the memory for 'batch noise pred.' - one of the inputs to Scheduler
    std::vector<float32_t> m_PredBatchNoise;
    // Memory variable to point the memory for 'scheduler latent' - one of the inputs to and output from Scheduler
    std::vector<float32_t> m_SchLatent;

    // Memory to hold quantized constant text embedding data 
    std::vector<uint16_t> m_ConstTextEmbeddingQuantized;

    // Variables to hold quantized parameters for Text Encoder
    Helpers::QuantParameters m_TeOutQuantParam;

    // Variables to hold quantized parameters for UNET
    Helpers::QuantParameters m_UnetInQuantParam;
    Helpers::QuantParameters m_LatentQuantParam;
    Helpers::QuantParameters m_TsEmbedQuantParam;
    Helpers::QuantParameters m_UnetOutQuantParam;

    // Variables to hold quantized parameters for VAE
    Helpers::QuantParameters m_VaeInQuantParam;

    // Tokenizer specific variables
    uint32_t m_TokenIds[TOKEN_IDS_LEN];

    // Variable to hold the path on-device where all data files reside
    std::string m_DemoDataFolder;

    // Variable to keep track of current step index value
    uint32_t m_StepIdx;

    // Variable to hold formatted text string of sample number
#if defined(DEBUG_DUMP) || defined(OUTPUT_DUMP) || defined(PRELOAD_DATA)
    std::string m_SampleNum;
#endif

    // Variable to save tensor name with their graph for Latent and Ts Embedding (UNET)
    std::pair<std::string, std::string> m_LatentTensorName, m_TsEmbedTensorName;

    // Vector to hold the list of tensors which should be connected with each-other
    std::vector<std::vector<std::pair<std::string, std::string>>> m_connectedIpOpTensorPairs;

    // Variable to map user provided graph name to their index
    std::unordered_map<std::string, int> m_givenNameToGraphNum;
    
    // Input and output tensors used for executing the model
    std::vector<std::unordered_map<std::string, Qnn_Tensor_t*>> m_InputTensorsBank, m_OutputTensorsBank;

    // Set to hold allocated QNN Buffer memories to be released in destructor
    std::unordered_set<void*> m_qnnTensorMemorySet;

    // Bank of Data's needed to be transferred between pre-processing and
    // execute, execute and Fint, Fint and post-processing stages
    std::vector<Helpers::StagesData> m_PreStagesData, m_PostStagesData;

    // Variable to keep track of the number of times the complete dataset has been
    // consumed
    int m_LoopNum;

    StableDiffusionHelper* sd_helper;
};

#endif
