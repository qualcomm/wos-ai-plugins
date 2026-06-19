/*
**************************************************************************************************
* Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause-Clear
**************************************************************************************************
*/


#include "QnnApiHelpers.hpp"

#define SEED_LENGTH 10
#define STEP_LENGTH 10
#define GUIDANCE_LENGTH 10

#define TEXT_ENCODER_MODEL_IDX 2
#define UNET_MODEL_IDX 0
#define VAE_MODEL_IDX 1

#define INJECTED_LIMIT_COUNT 0

QnnApiHelpers::QnnApiHelpers()
{
    sd_helper = new StableDiffusionHelper();
}
QnnApiHelpers::~QnnApiHelpers()
{
    // Free Qnn Tensor and their memory
    QNN_DEBUG("Tearing Down Input Tensors Bank");
    m_ioTensor->tearDownTensors(m_InputTensorsBank, m_numInputTensorsMap);
    QNN_DEBUG("Tearing Down Output Tensors Bank");
    m_ioTensor->tearDownTensors(m_OutputTensorsBank, m_numOutputTensorsMap);

    QNN_DEBUG("Making entry of tensors whose memory was released by QNN backend");
    // Make entry of tensors whose memory was released by QNN backend in m_FreeTensorsPointerSet
    for (const auto &freeTensorPointer : m_ioTensor->getFreeTensorsPointerSet())
        m_FreeTensorsPointerSet.insert(freeTensorPointer);

    // Destructor of OffTarget Data Loader
    delete m_offTargetDataLoader;

    // Destructor of Scheduler Solver
    delete m_schedulerSolver;
}

int32_t QnnApiHelpers::parseConfigPath(std::string configFilePath)
{

    std::map<std::string, std::string> kvpMap;
    std::vector<std::string> requiredKvp;

    Helpers::readKeyValueTxtFile(configFilePath, kvpMap);
    // populate the requiredKvp with the items we require from the model.txt file
    requiredKvp.push_back("input_tensor_name");
    requiredKvp.push_back("output_tensor_name");
    requiredKvp.push_back("latent_tensor_name");
    requiredKvp.push_back("tsembed_tensor_name");
    requiredKvp.push_back("data_loader_input_tarfile");
    requiredKvp.push_back("demo_data_folder");

    // lets check if our model.txt file had all the requirements
    if (!Helpers::checkKeysPresent(kvpMap, requiredKvp))
    {
        QNN_ERROR("configuration file is missing required key");
        return -1;
    }

    // Get basePath to change relative paths to absolute path later
    std::string basePath = Helpers::stripPath(configFilePath);

    // Now lets get our data from the key/value map we just created
    std::vector<std::string> modelsPathVec;
    if (kvpMap.find("model") != kvpMap.end())
    {
        modelsPathVec = Helpers::split(kvpMap["model"], ',');
        if (modelsPathVec.size() > 1)
        {
            QNN_ERROR("Multiple (total %zu) model files provided. Framework supports only one model!",
                      modelsPathVec.size());
            return -1;
        }
    }
    else if (kvpMap.find("model_context") != kvpMap.end())
    {
        modelsPathVec = Helpers::split(kvpMap["model_context"], ',');
        m_LoadFromCachedBinary = true;
    }
    else
    {
        QNN_ERROR("Either model or model_context key should be present in configuration file");
        return -1;
    }

    m_DemoDataFolder = kvpMap["demo_data_folder"];

    // Creating list of absolute model path and the mapping from user-provided-model-name to graph-num
    size_t graphNum = 0;
    for (const auto &modelPath : modelsPathVec)
    {
        const auto &modelNamePath = Helpers::split(modelPath, ':');
        const auto &absModelPath = Helpers::makePathAbsolute(m_DemoDataFolder, modelNamePath[1]);
        std::ifstream infile(absModelPath);
        if (infile.peek() == std::ifstream::traits_type::eof())
        {
            infile.close();
            QNN_ERROR("Model/Context file: %s is empty, something went wrong when copying!",
                      absModelPath.c_str());
            return -1;
        }
        infile.close();
        m_ModelsPathVec.push_back(absModelPath);
        m_givenNameToGraphNum[modelNamePath[0]] = graphNum++;
    }

    // Creating order of model execution
    // If there are more than one model, user must provide model_exec_order
    if (m_givenNameToGraphNum.size() > 1)
    {
        if (kvpMap.find("model_exec_order") == kvpMap.end())
        {
            QNN_ERROR("If there are more than 1 model provided, their order of execution must be provided");
            return -1;
        }
        auto modelsExecOrder = Helpers::split(kvpMap["model_exec_order"], ',');
        if (modelsExecOrder.size() != m_givenNameToGraphNum.size())
        {
            QNN_ERROR("The number of models provided in execution order doesn't match with the original models/contexts");
            QNN_ERROR("model_exec_order count is %zu while model/context files count is %zu",
                      modelsExecOrder.size(), m_givenNameToGraphNum.size());
            return -1;
        }
        for (const auto &execOrder : modelsExecOrder)
        {
            if (m_givenNameToGraphNum.count(execOrder) == 0)
            {
                QNN_ERROR("The graph name (%s) provided in model_exec_order is not found in provided models/contexts", execOrder.c_str());
                return -1;
            }
        }
        m_modelsExecOrder = modelsExecOrder;
    }
    // If there is only one model, the order cwould contain only that model name, So we don't check
    // if user has provided model_exec_order
    else
    {
        m_modelsExecOrder = {m_givenNameToGraphNum.begin()->first};
    }

    if (kvpMap.find("connected_tensor_pairs") != kvpMap.end())
    {
        for (const auto &equalSeparatedNamedPair : Helpers::split(kvpMap["connected_tensor_pairs"], ','))
        {
            std::vector<std::pair<std::string, std::string>> connectedIpOpTensorPairs;
            for (const auto &namedPair : Helpers::split(equalSeparatedNamedPair, '='))
            {
                const auto &graph_tensor = Helpers::split(namedPair, ':');
                if (m_givenNameToGraphNum.count(graph_tensor[0]) == 0)
                {
                    QNN_ERROR("The graph name (%s) associated to connected_tensor_pairs is not found in provided models/contexts",
                              graph_tensor[0].c_str());
                    return -1;
                }
                connectedIpOpTensorPairs.push_back({graph_tensor[0], graph_tensor[1]});
            }
            m_connectedIpOpTensorPairs.push_back(connectedIpOpTensorPairs);
        }
    }

    // Getting input tensor name for Qnn
    if (kvpMap.find("input_tensor_name") != kvpMap.end())
    {
        const auto &input_graph_tensor = Helpers::split(kvpMap["input_tensor_name"], ':');
        if ("-" != input_graph_tensor[1] && m_givenNameToGraphNum.count(input_graph_tensor[0]) == 0)
        {
            QNN_ERROR("The graph name associated to input_tensor_name is not found in provided models/contexts");
            QNN_ERROR("input tensor graph name is %s", input_graph_tensor[0].c_str());
            return -1;
        }
        m_InputTensorName = {input_graph_tensor[0], input_graph_tensor[1]};
    }

    // Getting output tensor name for Qnn
    if (kvpMap.find("output_tensor_name") != kvpMap.end())
    {
        const auto &output_graph_tensor = Helpers::split(kvpMap["output_tensor_name"], ':');
        if ("-" != output_graph_tensor[1] && m_givenNameToGraphNum.count(output_graph_tensor[0]) == 0)
        {
            QNN_ERROR("The graph name associated to output_tensor_name is not found in provided models/contexts");
            QNN_ERROR("output tensor graph name is %s", output_graph_tensor[0].c_str());
            return -1;
        }
        m_OutputTensorName = {output_graph_tensor[0], output_graph_tensor[1]};
    }

    // Getting latent tensor name for Qnn
    if (kvpMap.find("latent_tensor_name") != kvpMap.end())
    {
        const auto &input_graph_tensor = Helpers::split(kvpMap["latent_tensor_name"], ':');
        if ("-" != input_graph_tensor[1] && m_givenNameToGraphNum.count(input_graph_tensor[0]) == 0)
        {
            QNN_ERROR("The graph name associated to latent_tensor_name is not found in provided models/contexts");
            QNN_ERROR("input tensor graph name is %s", input_graph_tensor[0].c_str());
            return -1;
        }
        m_LatentTensorName = {input_graph_tensor[0], input_graph_tensor[1]};
    }

    // Getting Ts Embedding tensor name for Qnn
    if (kvpMap.find("tsembed_tensor_name") != kvpMap.end())
    {
        const auto &input_graph_tensor = Helpers::split(kvpMap["tsembed_tensor_name"], ':');
        if ("-" != input_graph_tensor[1] && m_givenNameToGraphNum.count(input_graph_tensor[0]) == 0)
        {
            QNN_ERROR("The graph name associated to tsembed_tensor_name is not found in provided models/contexts");
            QNN_ERROR("input tensor graph name is %s", input_graph_tensor[0].c_str());
            return -1;
        }
        m_TsEmbedTensorName = {input_graph_tensor[0], input_graph_tensor[1]};
    }

    m_backendExtensionsLibPath = "";
    if (kvpMap.find("backend_extensions_lib") != kvpMap.end())
    {
        m_backendExtensionsLibPath = kvpMap["backend_extensions_lib"];
    }

    m_backendExtensionsConfigPath = "";
    if (kvpMap.find("backend_extensions_config") != kvpMap.end())
    {
        m_backendExtensionsConfigPath = kvpMap["backend_extensions_config"];
    }

    m_dataLoaderInputTarfile = "";
    if (kvpMap.find("data_loader_input_tarfile") != kvpMap.end())
    {
        m_dataLoaderInputTarfile = Helpers::makePathAbsolute(m_DemoDataFolder, kvpMap["data_loader_input_tarfile"]);
    }

    return 0;
}

int32_t QnnApiHelpers::Init(
    std::string configFilePath, std::string nativeLibPath,
    int32_t inputWidth, int32_t inputHeight, float inputChannel,
    int32_t outputWidth, int32_t outputHeight, float outputChannel,
    bool debugModeRequested, std::string runtimeNameOrFile, std::string model_version)
{
#if defined(DEBUG_DUMP) || defined(OUTPUT_DUMP) || defined(PRELOAD_DATA)
    // Setting up 0 as sample number
    setSampleNum(0);
#endif

    // Let's first parse out config file to its a key/value pair and process
    if (0 != parseConfigPath(configFilePath))
    {
        QNN_ERROR("Error Parsing Config file.");
        return -1;
    }

    // Defining bank size of 1 for each process buffers
    m_preProcessBankSize = 1;
    m_postProcessBankSize = 1;

    // Set ADSP_LIBRARY_PATH for DSP runtime
    /*std::stringstream path;
    path << "c:\\Windows\\System32\\DriverStore\\FileRepository\\qcadsprpc8380.inf_arm64_e9a0dd63d7fa430e";
    std::cout << path.str() << std::endl;
    if (0 != setEnvVariable("ADSP_LIBRARY_PATH", path.str())) {
        QNN_ERROR("Error in setting up the ADSP_LIBRARY_PATH!");
        return -1;
    }*/

    // Setting up input and output images dimensions
    m_InputDims = Helpers::ImageDims(inputHeight, inputWidth, inputChannel, 2); // App model main input has bitwidth 16
    m_OutputDims = Helpers::ImageDims(outputHeight, outputWidth, outputChannel, 1);
    QNN_DEBUG("inputHeight = %d, inputWidth = %d, inputChannel = %f", inputHeight, inputWidth, inputChannel);
    QNN_DEBUG("outputHeight = %d, outputWidth = %d, outputChannel = %f", outputHeight, outputWidth, outputChannel);

    // Create and initialize Qnn APIs
    m_qnnApi = std::unique_ptr<QnnApi>(new QnnApi());

    // Initializing Qnn
    std::string systemLibraryPath = "QnnSystem.dll";
    std::string backendExtensionsLibPath;
    if (false == m_backendExtensionsLibPath.empty())
    {
        backendExtensionsLibPath = m_backendExtensionsLibPath;
    }

    if (true != m_qnnApi->initialize(runtimeNameOrFile, m_ModelsPathVec,
                                     BackendExtensionsConfigs(backendExtensionsLibPath, m_backendExtensionsConfigPath),
                                     PerfProfile::BURST,
                                     ContextConfigs(), {},
                                     m_LoadFromCachedBinary,
                                     systemLibraryPath,
                                     debugModeRequested))
    {
        QNN_ERROR("QNN initialization failed!");
        return -1;
    }

    // Getting graph info and its count needed for subsequent steps
    const auto &graphsInfo = m_qnnApi->getGraphsInfo();
    auto graphsCount = m_qnnApi->getGraphsCount();

    // Tracking number of input and output tensors
    for (size_t graphIdx = 0; graphIdx < graphsCount; graphIdx++)
    {
        const auto &graphInfo = graphsInfo[graphIdx];

        if (graphInfo->numInputTensors == 0)
        {
            QNN_ERROR("framework found 0 input tensors for graph index = %zu!", graphIdx);
            return -1;
        }

        if (graphInfo->numOutputTensors == 0)
        {
            QNN_ERROR("framework found 0 output tensors for graph index = %zu!", graphIdx);
            return -1;
        }

        m_numInputTensorsMap[graphInfo->graphName] = graphInfo->numInputTensors;
        m_numOutputTensorsMap[graphInfo->graphName] = graphInfo->numOutputTensors;
    }

    // Replacing user provided graph names to their actual names...
    std::unordered_map<std::string, std::string> givenNameToGraphName;
    for (const auto &givenNameGraphNum : m_givenNameToGraphNum)
    {
        givenNameToGraphName[givenNameGraphNum.first] = graphsInfo[givenNameGraphNum.second]->graphName;
    }

    // ... For input, output, latent-tenosr, tsembed-tensor, model execute sequence, and connect ops.
    {
        auto &inputTensorName = m_InputTensorName;
        if (false == inputTensorName.first.empty() && "-" != inputTensorName.second)
            inputTensorName.first = givenNameToGraphName[inputTensorName.first];
        auto &outputTensorName = m_OutputTensorName;
        if (false == outputTensorName.first.empty() && "-" != outputTensorName.second)
            outputTensorName.first = givenNameToGraphName[outputTensorName.first];
        auto &latentTensorName = m_LatentTensorName;
        if (false == latentTensorName.first.empty() && "-" != latentTensorName.second)
            latentTensorName.first = givenNameToGraphName[latentTensorName.first];
        auto &tsembedTensorName = m_TsEmbedTensorName;
        if (false == tsembedTensorName.first.empty() && "-" != tsembedTensorName.second)
            tsembedTensorName.first = givenNameToGraphName[tsembedTensorName.first];
        for (auto &modelExec : m_modelsExecOrder)
        {
            modelExec = givenNameToGraphName[modelExec];
        }
        for (auto &connectedIpOpTensorPair : m_connectedIpOpTensorPairs)
        {
            for (auto &ipOpTensorPair : connectedIpOpTensorPair)
            {
                ipOpTensorPair.first = givenNameToGraphName[ipOpTensorPair.first];
            }
        }
    }

    // Create and initialize ioTensor for DSP
    if (m_sharedBuffer == false)
    {
        m_ioTensor = std::unique_ptr<IOTensor>(new IOTensor());
    }
    else
    {
        m_ioTensor = std::unique_ptr<IOTensor>(new IOTensor(BufferAlloc::SHARED_BUFFER, m_qnnApi->getQnnInterfaceVer()));
    }

    // Ideally, we should create and initalize m_ioTensor for each context, but we want to
    // be able to see/use all the buffers in every contexts so that they can be connected
    // with each other. Hence, we are using only the first context to initialize the m_ioTensor
    // and use it for all graphs/contexts.
    if (true != m_ioTensor->initialize(m_qnnApi->getContexts()[0]))
    {
        QNN_ERROR("Failure to initialize IOTensor");
        return -1;
    }

    // Setup dimensions of each input and output tensors to be tracked later
    for (size_t graphIdx = 0; graphIdx < graphsCount; graphIdx++)
    {
        const auto &graphInfo = graphsInfo[graphIdx];

        // Setup dimensions of each input tensor to be tracked later
        std::unordered_map<std::string, Helpers::ImageDims> modelInputImageDims;
        for (size_t tensorIdx = 0; tensorIdx < m_numInputTensorsMap[graphInfo->graphName]; tensorIdx++)
        {
            auto &model_input = graphInfo->inputTensors[tensorIdx];

            std::vector<size_t> model_input_dims;
            std::string model_input_name;
            m_qnnApi->getTensorNameAndShape(model_input_name, model_input_dims, model_input);

            modelInputImageDims[model_input_name] =
                Helpers::ImageDims((int32_t)model_input_dims[1], (int32_t)model_input_dims[2],
                                   (float)model_input_dims[3], (int32_t)model_input_dims[4]);
            QNN_DEBUG("graphIdx = %lu, tensorIdx = %lu, tensorName = %s, model_input_dims[1] = %d, [2] = %d, [3] = %f, [4] = %d",
                      graphIdx, tensorIdx, model_input_name.c_str(), (int32_t)model_input_dims[1],
                      (int32_t)model_input_dims[2], (float)model_input_dims[3], (int32_t)model_input_dims[4]);
        }
        m_ModelInputImageDims[graphInfo->graphName] = modelInputImageDims;

        // Setup dimensions of each output tensor to be tracked later
        std::unordered_map<std::string, Helpers::ImageDims> modelOutputImageDims;
        for (size_t tensorIdx = 0; tensorIdx < m_numOutputTensorsMap[graphInfo->graphName]; tensorIdx++)
        {
            auto &model_output = graphInfo->outputTensors[tensorIdx];

            std::vector<size_t> model_output_dims;
            std::string model_output_name;
            m_qnnApi->getTensorNameAndShape(model_output_name, model_output_dims, model_output);

            modelOutputImageDims[model_output_name] =
                Helpers::ImageDims((int32_t)model_output_dims[1], (int32_t)model_output_dims[2],
                                   (float)model_output_dims[3], (int32_t)model_output_dims[4]);
            QNN_DEBUG("graphIdx = %lu, tensorIdx = %lu, tensorName = %s, model_output_dims[1] = %d, [2] = %d, [3] = %f, [4] = %d",
                      graphIdx, tensorIdx, model_output_name.c_str(), (int32_t)model_output_dims[1],
                      (int32_t)model_output_dims[2], (float)model_output_dims[3], (int32_t)model_output_dims[4]);
        }
        m_ModelOutputImageDims[graphInfo->graphName] = modelOutputImageDims;
    }

    // Checking if model input name provided are present in the graph
    {
        const auto &inputTensorName = m_InputTensorName;
        if (false == inputTensorName.first.empty() && "-" != inputTensorName.second)
        {
            if (m_ModelInputImageDims[inputTensorName.first].count(inputTensorName.second) == 0)
            {
                QNN_ERROR("model input %s is not found in the graph %s",
                          inputTensorName.second.c_str(), inputTensorName.first.c_str());
                QNN_DEBUG("Available model inputs in this graph are:");
                for (const auto &model_input : m_ModelInputImageDims[inputTensorName.first])
                {
                    QNN_DEBUG("%s", model_input.first.c_str());
                }
                return -1;
            }
        }
    }

    // Checking if model output tensors name provided are present in the graph
    {
        const auto &outputTensorName = m_OutputTensorName;
        if (false == outputTensorName.first.empty() && "-" != outputTensorName.second)
        {
            if (m_ModelOutputImageDims[outputTensorName.first].count(outputTensorName.second) == 0)
            {
                QNN_ERROR("model output %s is not found in the graph %s",
                          outputTensorName.second.c_str(), outputTensorName.first.c_str());
                QNN_DEBUG("Available model outputs in this graph are:");
                for (const auto &model_output : m_ModelOutputImageDims[outputTensorName.first])
                {
                    QNN_DEBUG("%s", model_output.first.c_str());
                }
                return -1;
            }
        }
    }

    // Checking if model latent tensors name provided are present in the graph
    {
        const auto &latentTensorName = m_LatentTensorName;
        if (false == latentTensorName.first.empty() && "-" != latentTensorName.second)
        {
            if (m_ModelInputImageDims[latentTensorName.first].count(latentTensorName.second) == 0)
            {
                QNN_ERROR("model input %s is not found in the graph %s",
                          latentTensorName.second.c_str(), latentTensorName.first.c_str());
                QNN_DEBUG("Available model inputs in this graph are:");
                for (const auto &model_output : m_ModelInputImageDims[latentTensorName.first])
                {
                    QNN_DEBUG("%s", model_output.first.c_str());
                }
                return -1;
            }
        }
    }

    // Checking if model tsembed tensors name provided are present in the graph
    {
        const auto &tsembedTensorName = m_TsEmbedTensorName;
        if (false == tsembedTensorName.first.empty() && "-" != tsembedTensorName.second)
        {
            if (m_ModelInputImageDims[tsembedTensorName.first].count(tsembedTensorName.second) == 0)
            {
                QNN_ERROR("model input %s is not found in the graph %s",
                          tsembedTensorName.second.c_str(), tsembedTensorName.first.c_str());
                QNN_DEBUG("Available model inputs in this graph are:");
                for (const auto &model_output : m_ModelInputImageDims[tsembedTensorName.first])
                {
                    QNN_DEBUG("%s", model_output.first.c_str());
                }
                return -1;
            }
        }
    }

    // Checking if the tensors name provided in m_connectedIpOpTensorPairs are present
    for (const auto &connectedIpOpTensor : m_connectedIpOpTensorPairs)
    {
        if (m_ModelInputImageDims[connectedIpOpTensor[0].first].count(connectedIpOpTensor[0].second) == 0)
        {
            QNN_ERROR("model input %s is not found in the graph %s",
                      connectedIpOpTensor[0].second.c_str(), connectedIpOpTensor[0].first.c_str());
            QNN_DEBUG("Available model inputs in this graph are:");
            for (const auto &model_input : m_ModelInputImageDims[connectedIpOpTensor[0].first])
            {
                QNN_DEBUG("%s", model_input.first.c_str());
            }
            return -1;
        }

        if (m_ModelOutputImageDims[connectedIpOpTensor[1].first].count(connectedIpOpTensor[1].second) == 0)
        {
            QNN_ERROR("model output %s is not found in the graph %s",
                      connectedIpOpTensor[1].second.c_str(), connectedIpOpTensor[1].first.c_str());
            QNN_DEBUG("Available model outputs in this graph are:");
            for (const auto &model_output : m_ModelOutputImageDims[connectedIpOpTensor[1].first])
            {
                QNN_DEBUG("%s", model_output.first.c_str());
            }
            return -1;
        }
    }

    // Check if the input image dimension matches with the input dimension of the model.
    // If it doesn't match, user must provide custom pre-processing function in order
    // for this functionality to continue.
    {
        const auto &inputTensorName = m_InputTensorName;
        if (false == inputTensorName.first.empty() && "-" != inputTensorName.second)
        {
            if (m_InputDims != m_ModelInputImageDims[inputTensorName.first][inputTensorName.second] &&
                nullptr == sd_helper)
            {
                QNN_ERROR("Model input dimensions and the dimension of input image don't match and no custom pre-processing function is found");
                QNN_DEBUG("Model: height=%d, width=%d, channel=%f, bitwidth=%d",
                          m_ModelInputImageDims[inputTensorName.first][inputTensorName.second].height,
                          m_ModelInputImageDims[inputTensorName.first][inputTensorName.second].width,
                          m_ModelInputImageDims[inputTensorName.first][inputTensorName.second].channel,
                          m_ModelInputImageDims[inputTensorName.first][inputTensorName.second].bitWidth);
                QNN_DEBUG("input: height=%d, width=%d, channel=%f, bitwidth=%d",
                          m_InputDims.height, m_InputDims.width, m_InputDims.channel, m_InputDims.bitWidth);
                return -1;
            }
        }
    }

    // Check if the output image dimension matches with the output dimension of the model.
    // If it doesn't match, user must provide custom post-processing function in order
    // for this functionality to continue.
    {
        const auto &outputTensorName = m_OutputTensorName;
        if (false == outputTensorName.first.empty() && "-" != outputTensorName.second)
        {
            if (m_OutputDims != m_ModelOutputImageDims[outputTensorName.first][outputTensorName.second] &&
                nullptr == sd_helper)
            {
                QNN_ERROR("Model output dimensions and the dimension of output image don't match and no custom post-processing function is found");
                QNN_DEBUG("Model: height=%d, width=%d, channel=%f, bitwidth=%d",
                          m_ModelOutputImageDims[outputTensorName.first][outputTensorName.second].height,
                          m_ModelOutputImageDims[outputTensorName.first][outputTensorName.second].width,
                          m_ModelOutputImageDims[outputTensorName.first][outputTensorName.second].channel,
                          m_ModelOutputImageDims[outputTensorName.first][outputTensorName.second].bitWidth);
                QNN_DEBUG("output: height=%d, width=%d, channel=%f, bitwidth=%d",
                          m_OutputDims.height, m_OutputDims.width, m_OutputDims.channel, m_OutputDims.bitWidth);
                return -1;
            }
        }
    }

    // Create input tensors bank, and track buffer of each memory
    for (uint8_t idx = 0; idx < m_preProcessBankSize; idx++)
    {
        std::unordered_map<std::string, Qnn_Tensor_t *> inputTensorsBank;
        std::unordered_map<std::string, std::unordered_map<std::string, void *>> inputTensorsBufBank;
        for (size_t graphIdx = 0; graphIdx < graphsCount; graphIdx++)
        {
            const auto &graphInfo = graphsInfo[graphIdx];

            std::unordered_map<std::string, size_t> inputTensorsSize;
            for (const auto &tensorNameShape : m_ModelInputImageDims[graphInfo->graphName])
            {
                inputTensorsSize[tensorNameShape.first] = tensorNameShape.second.getImageSize();
            }
            Qnn_Tensor_t *inputs = nullptr;
            std::unordered_map<std::string, void *> tensorNameToTensorPointer;
            if (true !=
                m_ioTensor->setupInputTensors(&inputs, tensorNameToTensorPointer,
                                              *graphInfo, inputTensorsSize))
            {
                QNN_ERROR("Error in setting up Input Tensors for bank idx: %d", idx);
                m_ioTensor->tearDownTensors(inputTensorsBank, m_numInputTensorsMap);
                m_ioTensor->tearDownTensors(m_InputTensorsBank, m_numInputTensorsMap);
                return -1;
            }
            inputTensorsBank[graphInfo->graphName] = inputs;
            inputTensorsBufBank[graphInfo->graphName] = tensorNameToTensorPointer;
            for (const auto &tensorNamePointer : tensorNameToTensorPointer)
            {
                m_qnnTensorMemorySet.insert(tensorNamePointer.second);
            }
        }
        m_InputTensorsBank.push_back(inputTensorsBank);
        m_InputTensorsBufBank.push_back(inputTensorsBufBank);
    }

    // Create output tensors bank, and track buffer of each memory
    for (uint8_t idx = 0; idx < m_postProcessBankSize; idx++)
    {
        std::unordered_map<std::string, Qnn_Tensor_t *> outputTensorsBank;
        std::unordered_map<std::string, std::unordered_map<std::string, void *>> outputTensorsBufBank;
        for (size_t graphIdx = 0; graphIdx < graphsCount; graphIdx++)
        {
            const auto &graphInfo = graphsInfo[graphIdx];

            std::unordered_map<std::string, size_t> outputTensorsSize;
            for (const auto &tensorNameShape : m_ModelOutputImageDims[graphInfo->graphName])
            {
                outputTensorsSize[tensorNameShape.first] = tensorNameShape.second.getImageSize();
            }
            Qnn_Tensor_t *outputs = nullptr;
            std::unordered_map<std::string, void *> tensorNameToTensorPointer;
            if (true !=
                m_ioTensor->setupOutputTensors(&outputs, tensorNameToTensorPointer,
                                               *graphInfo, outputTensorsSize))
            {
                QNN_ERROR("Error in setting up Output Tensors for bank idx: %d", idx);
                m_ioTensor->tearDownTensors(outputTensorsBank, m_numOutputTensorsMap);
                m_ioTensor->tearDownTensors(m_InputTensorsBank, m_numInputTensorsMap);
                m_ioTensor->tearDownTensors(m_OutputTensorsBank, m_numOutputTensorsMap);
                return -1;
            }
            outputTensorsBank[graphInfo->graphName] = outputs;
            outputTensorsBufBank[graphInfo->graphName] = tensorNameToTensorPointer;
            for (const auto &tensorNamePointer : tensorNameToTensorPointer)
            {
                m_qnnTensorMemorySet.insert(tensorNamePointer.second);
            }
        }
        m_OutputTensorsBank.push_back(outputTensorsBank);
        m_OutputTensorsBufBank.push_back(outputTensorsBufBank);
    }

    // Delete all but one memory of input tensor from m_connectedIpOpTensorPairs and all memory of
    // output tensor from m_connectedIpOpTensorPairs and connected them to non deleted tensor memory
    for (const auto &connectedIpOpTensor : m_connectedIpOpTensorPairs)
    {
        Qnn_Tensor_t *src = (Qnn_Tensor_t *)m_InputTensorsBufBank[0][connectedIpOpTensor[0].first][connectedIpOpTensor[0].second];

        // Delete all but one input tensor memory from m_connectedIpOpTensorPairs and let it use first tensor memory
        for (uint8_t idx = 1; idx < m_preProcessBankSize; idx++)
        {
            if (false == m_ioTensor->useSameMemory(
                             (Qnn_Tensor_t *)m_InputTensorsBufBank[idx][connectedIpOpTensor[0].first][connectedIpOpTensor[0].second], src))
            {
                QNN_ERROR("Error in setting up connection between ip tensors %s of graph %s",
                          connectedIpOpTensor[0].second.c_str(), connectedIpOpTensor[0].first.c_str());
                return -1;
            }
        }

        // Delete all output tensor memory from m_connectedIpOpTensorPairs and let it use first input tensor memory
        for (uint8_t idx = 0; idx < m_postProcessBankSize; idx++)
        {
            if (false == m_ioTensor->useSameMemory(
                             (Qnn_Tensor_t *)m_OutputTensorsBufBank[idx][connectedIpOpTensor[1].first][connectedIpOpTensor[1].second], src))
            {
                QNN_ERROR("Error in setting up connection between ip tensors %s of graph %s and output tensor %s of graph %s",
                          connectedIpOpTensor[0].second.c_str(), connectedIpOpTensor[0].first.c_str(),
                          connectedIpOpTensor[1].second.c_str(), connectedIpOpTensor[1].first.c_str());
                return -1;
            }
        }

        // Writing 0's in the src memory
        std::memset(m_ioTensor->getBuffer(src), 0, m_ioTensor->getBufferSize(src));
    }

    // Define memory for m_PredBatchNoise to hold Pred. batch noise data
    {
        // Pred. batch noise needs to hold the output of 2 Unet runs
        const auto &dim = m_ModelOutputImageDims[m_modelsExecOrder[UNET_MODEL_IDX]].begin()->second;
        m_PredBatchNoise = std::vector<float32_t>(2 * dim.height * dim.width * dim.channel);
    }

    // Define memory for m_SchLatent to hold scheduler output
    {
        // Scheduler output is the input to Unet Latent as well as input to scheduler for next iteration
        const auto &dim = m_ModelInputImageDims[m_LatentTensorName.first][m_LatentTensorName.second];
        m_SchLatent = std::vector<float32_t>(dim.height * dim.width * dim.channel);
    }

    // Define memory for m_ConstTextEmbeddingQuantized to hold quantized data of constant text embedding
    {
        // m_ConstTextEmbeddingQuantized holds quantized constant text embedding. Hence its size should be
        // equal to the size of input to UNET
        const auto &dim = m_ModelInputImageDims[m_InputTensorName.first][m_InputTensorName.second];
        m_ConstTextEmbeddingQuantized = std::vector<uint16_t>(dim.height * dim.width * dim.channel);
    }

    // Verification of Text Encoder Quantization
    // 1. input
    {
        const auto &tensor = (Qnn_Tensor_t *)m_InputTensorsBufBank[0][m_modelsExecOrder[TEXT_ENCODER_MODEL_IDX]].begin()->second;
        double scale;
        int32_t offset;
        if (false != m_qnnApi->getTensorQuantStatus(tensor, scale, offset))
        {
            QNN_ERROR("Input quantization of Text Encoder should be disabled! Found it enabled");
            return -1;
        }
        // Since the output from Tokenizer will be directly feed into Text Encoder, data length and size of m_TokenIds
        // must match with that of Text Encoder input tensor
        auto text_encoder_ip_dims = m_ModelInputImageDims[m_modelsExecOrder[TEXT_ENCODER_MODEL_IDX]].begin()->second;
        if (sizeof(m_TokenIds) != text_encoder_ip_dims.getImageSize())
        {
            QNN_ERROR("The total size of m_TokenIds %lu doesn't match with required size %lu of Text Encoder input",
                      sizeof(m_TokenIds), text_encoder_ip_dims.getImageSize());
            return -1;
        }
        else if (sizeof(m_TokenIds[0]) != text_encoder_ip_dims.bitWidth)
        {
            QNN_ERROR("The element size of m_TokenIds %lu doesn't match with required size %d of Text Encoder input element",
                      sizeof(m_TokenIds[0]), text_encoder_ip_dims.bitWidth);
            return -1;
        }
    }
    // 2. output
    {
        // Since the output from Text Encoder will be directly feed into Unet input, dims of output of Text
        // encoder must exactly match with the input dims of Unet.
        auto text_encoder_op_dims = m_ModelOutputImageDims[m_modelsExecOrder[TEXT_ENCODER_MODEL_IDX]].begin()->second;
        if (text_encoder_op_dims != m_ModelInputImageDims[m_InputTensorName.first][m_InputTensorName.second])
        {
            QNN_ERROR("Model input dimensions and the dimension of Text Encoder output don't match!");
            QNN_DEBUG("Model: height=%d, width=%d, channel=%f, bitwidth=%d",
                      m_ModelInputImageDims[m_InputTensorName.first][m_InputTensorName.second].height,
                      m_ModelInputImageDims[m_InputTensorName.first][m_InputTensorName.second].width,
                      m_ModelInputImageDims[m_InputTensorName.first][m_InputTensorName.second].channel,
                      m_ModelInputImageDims[m_InputTensorName.first][m_InputTensorName.second].bitWidth);
            QNN_DEBUG("Text Encoder output: height=%d, width=%d, channel=%f, bitwidth=%d",
                      text_encoder_op_dims.height, text_encoder_op_dims.width, text_encoder_op_dims.channel, text_encoder_op_dims.bitWidth);
            return -1;
        }
    }

    // Verification of VAE Shape
    // 1. input
    {
        // Since the output from Scheduler will be feed into VAE, data length of m_SchLatent
        // must match with that of VAE input tensor
        const auto &dim = m_ModelInputImageDims[m_modelsExecOrder[VAE_MODEL_IDX]].begin()->second;
        if (m_SchLatent.size() != (unsigned long)(dim.height * dim.width * dim.channel))
        {
            QNN_ERROR("The scheduler output data size %lu doesn't match with the input size %lu of VAE",
                      m_SchLatent.size(), (unsigned long)(dim.height * dim.width * dim.channel));
            return -1;
        }
    }

    // Reading input quantization parameters
    Helpers::QuantParameters inputQuantParam;
    const auto &input_tensor = (Qnn_Tensor_t *)m_InputTensorsBufBank[0][m_InputTensorName.first][m_InputTensorName.second];
    if (m_qnnApi->getTensorQuantStatus(input_tensor, inputQuantParam.scale, inputQuantParam.offset))
    {
        inputQuantParam.active = true;
        QNN_DEBUG("Quantization enabled for the input");
    }

    // Reading output de-quantization parameters
    Helpers::QuantParameters outputDequantParam;
    const auto &output_tensor = (Qnn_Tensor_t *)m_OutputTensorsBufBank[0][m_OutputTensorName.first][m_OutputTensorName.second];
    if (m_qnnApi->getTensorQuantStatus(output_tensor, outputDequantParam.scale, outputDequantParam.offset))
    {
        outputDequantParam.active = true;
        QNN_DEBUG("Dequantization enabled for the output");
    }

    // Reading output quantization parameters of Text Encoder
    {
        const auto &tensor = (Qnn_Tensor_t *)m_OutputTensorsBufBank[0][m_modelsExecOrder[TEXT_ENCODER_MODEL_IDX]].begin()->second;
        if (m_qnnApi->getTensorQuantStatus(tensor, m_TeOutQuantParam.scale, m_TeOutQuantParam.offset))
        {
            m_TeOutQuantParam.active = true;
        }
        if (false == m_TeOutQuantParam.active)
        {
            QNN_ERROR("The output qunatization of Text Encoder is supposed to be enable, found it disabled!");
            return -1;
        }
    }

    // Reading Embedding input quantization parameters of Unet
    {
        // Unet emebdding input is same as app main input
        m_UnetInQuantParam = inputQuantParam;
        if (false == m_UnetInQuantParam.active)
        {
            QNN_ERROR("The Embedding input qunatization of Unet is supposed to be enable, found it disabled!");
            return -1;
        }
    }
    // Reading Latent input quantization parameters of Unet
    {
        const auto &tensor = (Qnn_Tensor_t *)m_InputTensorsBufBank[0][m_LatentTensorName.first][m_LatentTensorName.second];
        if (m_qnnApi->getTensorQuantStatus(tensor, m_LatentQuantParam.scale, m_LatentQuantParam.offset))
        {
            m_LatentQuantParam.active = true;
        }
        if (false == m_LatentQuantParam.active)
        {
            QNN_ERROR("The Latent input qunatization of Unet is supposed to be enable, found it disabled!");
            return -1;
        }
    }
    // Reading Ts Embedding input quantization parameters of Unet
    {
        const auto &tensor = (Qnn_Tensor_t *)m_InputTensorsBufBank[0][m_TsEmbedTensorName.first][m_TsEmbedTensorName.second];
        if (m_qnnApi->getTensorQuantStatus(tensor, m_TsEmbedQuantParam.scale, m_TsEmbedQuantParam.offset))
        {
            m_TsEmbedQuantParam.active = true;
        }
        if (false == m_TsEmbedQuantParam.active)
        {
            QNN_ERROR("The Ts Embedding input qunatization of Unet is supposed to be enable, found it disabled!");
            return -1;
        }
    }
    // Reading output quantization parameters of Unet
    {
        const auto &tensor = (Qnn_Tensor_t *)m_OutputTensorsBufBank[0][m_modelsExecOrder[UNET_MODEL_IDX]].begin()->second;
        if (m_qnnApi->getTensorQuantStatus(tensor, m_UnetOutQuantParam.scale, m_UnetOutQuantParam.offset))
        {
            m_UnetOutQuantParam.active = true;
        }
        if (false == m_UnetOutQuantParam.active)
        {
            QNN_ERROR("The output qunatization of Unet is supposed to be enable, found it disabled!");
            return -1;
        }
    }

    // Reading input quantization parameters of VAE
    {
        const auto &tensor = (Qnn_Tensor_t *)m_InputTensorsBufBank[0][m_modelsExecOrder[VAE_MODEL_IDX]].begin()->second;
        if (m_qnnApi->getTensorQuantStatus(tensor, m_VaeInQuantParam.scale, m_VaeInQuantParam.offset))
        {
            m_VaeInQuantParam.active = true;
        }
        if (false == m_VaeInQuantParam.active)
        {
            QNN_ERROR("The input qunatization of VAE is supposed to be enable, found it disabled!");
            return -1;
        }
    }

    // Now, since all buffers and quantization/dequantization info is available, lets call
    // OffTarget Data loader setup processes
    m_offTargetDataLoader = new DataLoader;
    if (false == m_offTargetDataLoader->load((char *)m_dataLoaderInputTarfile.c_str(), model_version))
    {
        QNN_ERROR("Error in running load on m_offTargetDataLoader!");
        return -1;
    }

    // Now, since all buffers and quantization/dequantization info is available, lets call
    // Scheduler setup processes
    int64_t num_train_timsteps = 1000;
    std::vector<float32_t> betas(num_train_timsteps);
    std::string betas_file = Helpers::joinPath(m_DemoDataFolder, "betas.bin");
    if (true != Helpers::readRawData((void *)betas.data(), betas.size() * sizeof(betas[0]), betas_file))
    {
        QNN_ERROR("Error in loading file %s!", betas_file.c_str());
        return -1;
    }

    std::vector<float32_t> lambdas(num_train_timsteps);
    std::string lambdas_file = Helpers::joinPath(m_DemoDataFolder, "lambdas.bin");
    if (true != Helpers::readRawData((void *)lambdas.data(), lambdas.size() * sizeof(lambdas[0]), lambdas_file))
    {
        QNN_ERROR("Error in loading file %s!", lambdas_file.c_str());
        return -1;
    }

    m_schedulerSolver = new DPMSolverMultistepScheduler(/*num_train_timsteps=*/num_train_timsteps, /*beta_start=*/0.00085,
                                                        /*beta_end=*/0.012, /*beta_schedule=*/"scaled_linear",
                                                        /*trained_betas=*/betas, /*trained_lambdas=*/lambdas);

    // Now, since all buffers and quantization/dequantization info is available, lets call
    // Tokenizer setup processes
    set_datapath(m_DemoDataFolder.c_str());

    uint32_t imageReadSize = m_InputDims.getImageSize();

    // Define few storages to hold original images across stages for overlay, and other info
    for (size_t i = 0; i < m_preProcessBankSize; i++)
    {
#ifdef USE_PRELOADED_IMAGE
        m_OrgDataPre.push_back(nullptr);
#else
        m_OrgDataPre.push_back((void *)malloc(imageReadSize));
#endif
        m_PreStagesData.push_back(Helpers::StagesData());
    }

    for (size_t i = 0; i < m_postProcessBankSize; i++)
    {
#ifdef USE_PRELOADED_IMAGE
        m_OrgDataPost.push_back(nullptr);
#else
        m_OrgDataPost.push_back((void *)malloc(imageReadSize));
#endif
        m_PostStagesData.push_back(Helpers::StagesData());
    }

    // Speeds up the file reading time from IOs
    std::ios_base::sync_with_stdio(false);

    m_LoopNum = 0;

    m_pre_pingpong_index = 0;
    m_infer_in_pingpong_index = m_infer_out_pingpong_index = 0;
    m_post_pingpong_index = 0;

    // Let's also see if there is custom init that needs to be initialized
    if (nullptr != sd_helper)
    {
        QNN_DEBUG("Your custom Init function is being called....");
        int customInit = sd_helper->Init(configFilePath,
                                         m_InputDims,
                                         m_ModelInputImageDims[m_InputTensorName.first],
                                         inputQuantParam, outputDequantParam,
                                         m_ModelOutputImageDims[m_OutputTensorName.first],
                                         m_OutputDims);

        // check if the custom init initialized with no errors
        if (0 != customInit)
        {
            QNN_ERROR("Your custom Init function returned an error. Please check your logs.");
            return -1;
        }
    }

    return 0;
}

bool QnnApiHelpers::PreProcessInput(
    void *image,
    uint32_t imageSize,
    bool isFlipped,
    bool overlayOnImage)
{
    // Resetting Sub-processes counter at pre-processing since we know that pre-processing
    // will be called just once at the start of each new prompt
    {
        m_preprocess_count = m_inference_count = m_postprocess_count = 0;
    }
    auto &preprocess_count = m_preprocess_count;
    QNN_DEBUG("%s: START Iteration %d", __FUNCTION__, preprocess_count);

    // Read user provided text, seed, step and guidance scale values
    // image should encapsulate it as follows:
    //      [0:SEED_LENGTH-1] : seed,
    //      [SEED_LENGTH:SEED_LENGTH+STEP_LENGTH-1] : step,
    //      [SEED_LENGTH+STEP_LENGTH:SEED_LENGTH+STEP_LENGTH+GUIDANCE_LENGTH-1] : guidance scale,
    //      [SEED_LENGTH+STEP_LENGTH+GUIDANCE_LENGTH:imageSize-1] : text
    int32_t userSeed;
    uint32_t userSteps;
    float guidanceScale;
    std::string userText;
    {
        auto start = std::chrono::steady_clock::now();
        userSeed = std::stoi(std::string((char *)image, SEED_LENGTH));
        userSteps = std::stoi(std::string((char *)image + SEED_LENGTH, STEP_LENGTH));
        guidanceScale = std::stof(std::string((char *)image + SEED_LENGTH + STEP_LENGTH, GUIDANCE_LENGTH));
        userText = std::string((char *)image + SEED_LENGTH + STEP_LENGTH + GUIDANCE_LENGTH, imageSize - SEED_LENGTH - STEP_LENGTH - GUIDANCE_LENGTH);
        m_StepIdx = 0;
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("User data reading (cpp) took", start, stop);

#ifdef DEBUG_DUMP
        char buffer[20];
        sprintf(buffer, "%03d", preprocess_count);
        Helpers::writeTextFile(getDebugFile(Helpers::joinPath("user_data", std::string(buffer) + "_user_input.txt")),
                               {"userSeed = " + std::to_string(userSeed),
                                "userStep = " + std::to_string(userSteps),
                                "guidanceScale = " + std::to_string(guidanceScale),
                                "userText = " + userText});
#endif
    }

    // Checking the provided step value if it is available in data-loader
    std::vector<int32_t> available_step_seq;
    m_offTargetDataLoader->get_supported_num_steps(available_step_seq);
    bool valid_user_step = false;
    for (const auto &available_step : available_step_seq)
    {
        if (userSteps == available_step)
        {
            valid_user_step = true;
            break;
        }
    }
    if (false == valid_user_step)
    {
        QNN_ERROR("Provided user step %d is not available", userSteps);
        QNN_DEBUG("Available steps are:");
        for (const auto &available_step : available_step_seq)
        {
            QNN_DEBUG("%d", available_step);
        }
        return false;
    }
    m_offTargetDataLoader->set_num_steps(userSteps);

    // Checking the provided seed value if it is available in data-loader
    if (userSeed >= m_offTargetDataLoader->get_num_initial_latents())
    {
        QNN_ERROR("Provided user seed value %d is more than available # of seeds %d",
                  userSeed, m_offTargetDataLoader->get_num_initial_latents());
        return false;
    }

    // Setting up user provided time steps for Scheduler
    m_schedulerSolver->setTimesteps(userSteps);
    m_schedulerSolver->setGuidanceScale(guidanceScale);

    // Call Tokenizer
    {
        auto start = std::chrono::steady_clock::now();
        auto return_len = get_token_ids(userText.c_str(), m_TokenIds, TOKEN_IDS_LEN);
        if (TOKEN_IDS_LEN != return_len)
        {
            QNN_ERROR("The return token length is %lu is not equal to expected length %d",
                      return_len, TOKEN_IDS_LEN);
            return false;
        }

#ifdef DEBUG_DUMP
        char buffer[20];
        sprintf(buffer, "%03d", preprocess_count);
        Helpers::writeRawData((void *)userText.c_str(), userText.length(),
                              getDebugFile(Helpers::joinPath("tokenizer", std::string(buffer) + "_in.raw")));
        Helpers::writeRawData((void *)m_TokenIds, sizeof(m_TokenIds),
                              getDebugFile(Helpers::joinPath("tokenizer", std::string(buffer) + "_out.raw")));
#endif
#ifdef PRELOAD_DATA
        if (false == Helpers::readRawData((void *)m_TokenIds, sizeof(m_TokenIds),
                                          m_DemoDataFolder + "te/sample_" + m_SampleNum + "/inputs/000_text_input.bin"))
        {
            QNN_ERROR("There is an Error in reading the data from file");
            return false;
        }
#endif

        // Writing tokenizer output into Text Encoder input
        // Getting input tensor memory pointer of Text Encoder
        const auto &tensor = m_InputTensorsBufBank[m_pre_pingpong_index][m_modelsExecOrder[TEXT_ENCODER_MODEL_IDX]].begin()->second;
        void *text_encoder_mem_buf;
        if (0 != m_qnnTensorMemorySet.count(tensor))
        {
            text_encoder_mem_buf = m_ioTensor->getBuffer((Qnn_Tensor_t *)tensor);
        }
        else
        {
            text_encoder_mem_buf = tensor;
        }
        std::memcpy((char *)text_encoder_mem_buf, (char *)m_TokenIds, sizeof(m_TokenIds));
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("inference Tokenizer (cpp) took", start, stop);
    }

    // Run Text Encoder on HTP
    // Execute inference
    {
        auto start = std::chrono::steady_clock::now();
        const auto &graphName = m_modelsExecOrder[TEXT_ENCODER_MODEL_IDX];
        if (true != ExecuteModel(m_InputTensorsBank[m_pre_pingpong_index][graphName],
                                 m_OutputTensorsBank[m_pre_pingpong_index][graphName], graphName))
            return false;
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("inference Text Encoder (cpp) took", start, stop);

#ifdef DEBUG_DUMP
        char buffer[20];
        sprintf(buffer, "%03d", preprocess_count);
        for (const auto &tensorNameMemory : m_InputTensorsBufBank[m_pre_pingpong_index][graphName])
        {
            writeTensorData((Qnn_Tensor_t *)(tensorNameMemory.second),
                            getDebugFile(Helpers::joinPath("text_encoder", std::string(buffer) + "_" + tensorNameMemory.first + "_in.raw")));
        }
        for (const auto &tensorNameMemory : m_OutputTensorsBufBank[m_pre_pingpong_index][graphName])
        {
            writeTensorData((Qnn_Tensor_t *)(tensorNameMemory.second),
                            getDebugFile(Helpers::joinPath("text_encoder", std::string(buffer) + "_" + tensorNameMemory.first + "_out.raw")));
        }
#endif
    }
    // Now we will de-quantize the output of Text Encoder and then quantize it for Unet Text embedding input
    // We will use same buffer pointed by Text Encoder output for reading and writing as the input and output
    // data bitwidth is same
    {
        auto start = std::chrono::steady_clock::now();
        const auto &tensor = m_OutputTensorsBufBank[m_pre_pingpong_index][m_modelsExecOrder[TEXT_ENCODER_MODEL_IDX]].begin()->second;
        uint16_t *tensor_buf;
        size_t tensor_buf_len;
        if (0 != m_qnnTensorMemorySet.count(tensor))
        {
            tensor_buf = (uint16_t *)m_ioTensor->getBuffer((Qnn_Tensor_t *)tensor);
            tensor_buf_len = m_ioTensor->getBufferSize((Qnn_Tensor_t *)tensor) / 2;
        }
        else
        {
            tensor_buf = (uint16_t *)tensor;
            tensor_buf_len = 0;
        }

#ifdef PRELOAD_DATA
        std::vector<float32_t> float_text_embedding_T2(tensor_buf_len);
        if (false == Helpers::readRawData((void *)float_text_embedding_T2.data(),
                                          float_text_embedding_T2.size() * sizeof(float_text_embedding_T2[0]),
                                          m_DemoDataFolder + "unet/sample_" + m_SampleNum + "/inputs/000_t2_cond.raw"))
        {
            QNN_ERROR("There is an Error in reading the data from file");
            return false;
        }
#endif

        // Applying de-qunatization and then quantization on Text Encoder output data
        for (size_t idx = 0; idx < tensor_buf_len; idx++)
        {
            // De-quantization
            double value = ((double)(*tensor_buf) + m_TeOutQuantParam.offset) * m_TeOutQuantParam.scale;

#ifdef PRELOAD_DATA
            value = (double)float_text_embedding_T2[idx];
#endif

            // Quantization
            value = value / m_UnetInQuantParam.scale - m_UnetInQuantParam.offset;
            value = value < 0.0 ? 0.0 : value > 65535.0 ? 65535.0
                                                        : value;
            *tensor_buf = (uint16_t)value;
            tensor_buf++;
        }
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("dequantizing-quantizing of Text Encoder output (cpp) took", start, stop);
    }

    // Getting random initial latent data
    {
        auto start = std::chrono::steady_clock::now();
        const tensor_data_float32_t *latent_ptr = nullptr;
        int32_t seed;
        m_offTargetDataLoader->get_random_init_latent(userSeed, seed, latent_ptr);

        if (latent_ptr->size() != m_SchLatent.size())
        {
            QNN_ERROR("The random latent data size %lu doesn't match with the size %lu of latent input of Unet",
                      latent_ptr->size(), m_SchLatent.size());
            return false;
        }
        else if (sizeof((*latent_ptr)[0]) != sizeof(m_SchLatent[0]))
        {
            QNN_ERROR("The random latent data type size %lu doesn't match with the data type size %lu of latent input of Unet",
                      sizeof((*latent_ptr)[0]), sizeof(m_SchLatent[0]));
            return false;
        }

#ifdef DEBUG_DUMP
        char buffer[20];
        sprintf(buffer, "%03d", preprocess_count);
        Helpers::writeRawData((void *)(&userSeed), sizeof(userSeed),
                              getDebugFile(Helpers::joinPath("dataloader", std::string(buffer) + "_user_seed_in.raw")));
        Helpers::writeRawData((void *)(&seed), sizeof(seed),
                              getDebugFile(Helpers::joinPath("dataloader", std::string(buffer) + "_initial_seed_out.raw")));
        Helpers::writeRawData((void *)latent_ptr->data(), latent_ptr->size() * sizeof((*latent_ptr)[0]),
                              getDebugFile(Helpers::joinPath("dataloader", std::string(buffer) + "_initial_random_latent_out.raw")));
#endif
#ifdef PRELOAD_DATA
        if (false == Helpers::readRawData((void *)latent_ptr->data(), latent_ptr->size() * sizeof((*latent_ptr)[0]),
                                          m_DemoDataFolder + "unet/sample_" + m_SampleNum + "/inputs/000_t5_sample.bin"))
        {
            QNN_ERROR("There is an Error in reading the data from file");
            return false;
        }
#endif

        // Writing random initial data into scheduler latent output which will be feed to Unet and Scheduler
        // m_SchLatent will be filled by scheduler for subsequent runs
        std::memcpy((char *)m_SchLatent.data(), (char *)latent_ptr->data(), latent_ptr->size() * sizeof((*latent_ptr)[0]));
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("writing random latent (cpp) took", start, stop);
    }

    // Getting const embedding from data-loader for every 1st Unet run
    {
        auto start = std::chrono::steady_clock::now();
        const tensor_data_float32_t *const_text_embedding_ptr = nullptr;
        m_offTargetDataLoader->get_unconditional_text_embedding(const_text_embedding_ptr);

        if (const_text_embedding_ptr->size() != m_ConstTextEmbeddingQuantized.size())
        {
            QNN_ERROR("The const embedding data size %lu doesn't match with the size %lu of embedding input of Unet",
                      const_text_embedding_ptr->size(), m_ConstTextEmbeddingQuantized.size());
            return false;
        }

#ifdef DEBUG_DUMP
        char buffer[20];
        sprintf(buffer, "%03d", preprocess_count);
        Helpers::writeRawData((void *)const_text_embedding_ptr->data(), const_text_embedding_ptr->size() * sizeof((*const_text_embedding_ptr)[0]),
                              getDebugFile(Helpers::joinPath("dataloader", std::string(buffer) + "_const_text_embedding_out.raw")));
#endif
#ifdef PRELOAD_DATA
        if (false == Helpers::readRawData((void *)const_text_embedding_ptr->data(),
                                          const_text_embedding_ptr->size() * sizeof((*const_text_embedding_ptr)[0]),
                                          m_DemoDataFolder + "unet/sample_" + m_SampleNum + "/inputs/000_t3_uncond.bin"))
        {
            QNN_ERROR("There is an Error in reading the data from file");
            return false;
        }
#endif

        // Writing const embedding data into m_ConstTextEmbeddingQuantized
        // Applying qunatization on const embedding data
        for (size_t idx = 0; idx < const_text_embedding_ptr->size(); idx++)
        {
            double value = (*const_text_embedding_ptr)[idx] / m_UnetInQuantParam.scale - m_UnetInQuantParam.offset;
            value = value < 0.0 ? 0.0 : value > 65535.0 ? 65535.0
                                                        : value;
            m_ConstTextEmbeddingQuantized[idx] = (uint16_t)value;
        }
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("writing const-embedding input (cpp) took", start, stop);
    }

    // transfer data to shared variables for RunTime to process
    m_PreStagesData[m_pre_pingpong_index].overlayOnImage = overlayOnImage;

    // Increment pre-processing index
    QNN_DEBUG("%s: DONE Iteration %d", __FUNCTION__, preprocess_count);
    preprocess_count++;

    return true;
}

bool QnnApiHelpers::RunInference(
    bool runVAE,
    bool dumpOutput,
    std::string outputLocation)
{
    auto &inference_count = m_inference_count;
    QNN_DEBUG("%s: START QNN Iteration %d", __FUNCTION__, inference_count);

    // Getting text embedding buffer pointer of Unet (The app main input)
    void *unet_text_embedding_buf_ip;
    size_t unet_text_embedding_buf_ip_size;
    {
        const auto &tensor = m_InputTensorsBufBank[m_infer_in_pingpong_index][m_InputTensorName.first][m_InputTensorName.second];
        if (0 != m_qnnTensorMemorySet.count(tensor))
        {
            unet_text_embedding_buf_ip = m_ioTensor->getBuffer((Qnn_Tensor_t *)tensor);
            unet_text_embedding_buf_ip_size = m_ioTensor->getBufferSize((Qnn_Tensor_t *)tensor);
        }
        else
        {
            unet_text_embedding_buf_ip = tensor;
            unet_text_embedding_buf_ip_size = 0;
        }
    }

    // Write timestep into Unet Timestep tensor — handles INT32 and FLOAT32 scalar inputs
    {
        auto start = std::chrono::steady_clock::now();
        int32_t timeStep = 0;
        m_offTargetDataLoader->get_time_step(m_StepIdx, timeStep);

        const auto &tensor = m_InputTensorsBufBank[m_infer_in_pingpong_index][m_TsEmbedTensorName.first][m_TsEmbedTensorName.second];

        Qnn_Tensor_t *qnn_ts_tensor = nullptr;
        void *ts_buf = nullptr;

        if (0 != m_qnnTensorMemorySet.count(tensor))
        {
            qnn_ts_tensor = (Qnn_Tensor_t *)tensor;
            ts_buf = m_ioTensor->getBuffer(qnn_ts_tensor);
        }
        else
        {
            ts_buf = (void *)tensor;
        }

        Qnn_DataType_t ts_dtype = (Qnn_DataType_t)0;
        if (qnn_ts_tensor)
            ts_dtype = QNN_TENSOR_GET_DATA_TYPE(qnn_ts_tensor);

        if (ts_dtype == QNN_DATATYPE_FLOAT_32)
            *((float32_t *)ts_buf) = (float32_t)timeStep;
        else if (ts_dtype == QNN_DATATYPE_UFIXED_POINT_16)
        {
            double qval = (double)timeStep / m_TsEmbedQuantParam.scale - m_TsEmbedQuantParam.offset;
            qval = qval < 0.0 ? 0.0 : qval > 65535.0 ? 65535.0 : qval;
            *((uint16_t *)ts_buf) = (uint16_t)qval;
        }
        else
        {
            // Default: INT32 scalar
            *((int32_t *)ts_buf) = timeStep;
        }

        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("writing timestep input (cpp) took", start, stop);
    }

    // Writing Scheduler o/p into Unet Latent tensor
    {
        auto start = std::chrono::steady_clock::now();
        const auto &tensor = m_InputTensorsBufBank[m_infer_in_pingpong_index][m_LatentTensorName.first][m_LatentTensorName.second];
        uint16_t *latent_buf;
        if (0 != m_qnnTensorMemorySet.count(tensor))
        {
            latent_buf = (uint16_t *)m_ioTensor->getBuffer((Qnn_Tensor_t *)tensor);
        }
        else
        {
            latent_buf = (uint16_t *)tensor;
        }
        // Applying qunatization for Latent data
        for (size_t idx = 0; idx < m_SchLatent.size(); idx++)
        {
            double value = m_SchLatent[idx] / m_LatentQuantParam.scale - m_LatentQuantParam.offset;
            value = value < 0.0 ? 0.0 : value > 65535.0 ? 65535.0
                                                        : value;
            *latent_buf = (uint16_t)value;
            latent_buf++;
        }
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("writing scheduler into latent (cpp) took", start, stop);
    }

    // 1. Run for constant text embedding
    std::memcpy((char *)unet_text_embedding_buf_ip, (char *)m_ConstTextEmbeddingQuantized.data(), unet_text_embedding_buf_ip_size);
    // Execute inference
    {
        auto start = std::chrono::steady_clock::now();
        const auto &graphName = m_modelsExecOrder[UNET_MODEL_IDX];
        if (true != ExecuteModel(m_InputTensorsBank[m_infer_in_pingpong_index][graphName],
                                 m_OutputTensorsBank[m_infer_in_pingpong_index][graphName], graphName))
            return false;
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("inference Unet-0 (cpp) took", start, stop);

#ifdef DEBUG_DUMP
        char buffer[20];
        sprintf(buffer, "%03d", inference_count);
        for (const auto &tensorNameMemory : m_InputTensorsBufBank[m_infer_in_pingpong_index][graphName])
        {
            writeTensorData((Qnn_Tensor_t *)(tensorNameMemory.second),
                            getDebugFile(Helpers::joinPath("unet_0", std::string(buffer) + "_" + tensorNameMemory.first + "_in.raw")));
        }
        for (const auto &tensorNameMemory : m_OutputTensorsBufBank[m_infer_in_pingpong_index][graphName])
        {
            writeTensorData((Qnn_Tensor_t *)(tensorNameMemory.second),
                            getDebugFile(Helpers::joinPath("unet_0", std::string(buffer) + "_" + tensorNameMemory.first + "_out.raw")));
        }
#endif
    }
    // Copy the output into m_PredBatchNoise
    {
        auto start = std::chrono::steady_clock::now();
        const auto &tensor = m_OutputTensorsBufBank[m_infer_in_pingpong_index][m_modelsExecOrder[UNET_MODEL_IDX]].begin()->second;
        uint16_t *src;
        if (0 != m_qnnTensorMemorySet.count(tensor))
        {
            src = (uint16_t *)m_ioTensor->getBuffer((Qnn_Tensor_t *)tensor);
        }
        else
        {
            src = (uint16_t *)tensor;
        }
        // Applying de-qunatization on Unet output data
        for (size_t idx = 0; idx < m_PredBatchNoise.size() / 2; idx++)
        {
            double value = ((double)(*src) + m_UnetOutQuantParam.offset) * m_UnetOutQuantParam.scale;
            m_PredBatchNoise[idx] = (float32_t)value;
            src++;
        }
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("writing Unet-0 output (cpp) took", start, stop);

#ifdef PRELOAD_DATA
        if (inference_count < INJECTED_LIMIT_COUNT)
        {
            char buffer1[20];
            sprintf(buffer1, "%03d", inference_count);
            if (false == Helpers::readRawData((void *)m_PredBatchNoise.data(),
                                              m_PredBatchNoise.size() * sizeof(m_PredBatchNoise[0]) / 2,
                                              m_DemoDataFolder + "unet/sample_" + m_SampleNum + "/outputs/" + std::string(buffer1) + "_t6_pred_uncond.bin"))
            {
                QNN_ERROR("There is an Error in reading the data from file");
                return false;
            }
        }
#endif
    }

    // 2. Run for user text embedding
    // Getting text embedding buffer pointer from Text Encoder
    void *text_encoder_text_embedding_buf_op;
    {
        const auto &tensor = m_OutputTensorsBufBank[m_infer_in_pingpong_index][m_modelsExecOrder[TEXT_ENCODER_MODEL_IDX]].begin()->second;
        if (0 != m_qnnTensorMemorySet.count(tensor))
        {
            text_encoder_text_embedding_buf_op = m_ioTensor->getBuffer((Qnn_Tensor_t *)tensor);
        }
        else
        {
            text_encoder_text_embedding_buf_op = tensor;
        }
    }
    std::memcpy((char *)unet_text_embedding_buf_ip, (char *)text_encoder_text_embedding_buf_op, unet_text_embedding_buf_ip_size);
    // Execute inference
    {
        auto start = std::chrono::steady_clock::now();
        const auto &graphName = m_modelsExecOrder[UNET_MODEL_IDX];
        if (true != ExecuteModel(m_InputTensorsBank[m_infer_in_pingpong_index][graphName],
                                 m_OutputTensorsBank[m_infer_in_pingpong_index][graphName], graphName))
            return false;
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("inference Unet-1 (cpp) took", start, stop);

#ifdef DEBUG_DUMP
        char buffer[20];
        sprintf(buffer, "%03d", inference_count);
        for (const auto &tensorNameMemory : m_InputTensorsBufBank[m_infer_in_pingpong_index][graphName])
        {
            writeTensorData((Qnn_Tensor_t *)(tensorNameMemory.second),
                            getDebugFile(Helpers::joinPath("unet_1", std::string(buffer) + "_" + tensorNameMemory.first + "_in.raw")));
        }
        for (const auto &tensorNameMemory : m_OutputTensorsBufBank[m_infer_in_pingpong_index][graphName])
        {
            writeTensorData((Qnn_Tensor_t *)(tensorNameMemory.second),
                            getDebugFile(Helpers::joinPath("unet_1", std::string(buffer) + "_" + tensorNameMemory.first + "_out.raw")));
        }
#endif
    }
    // Copy the output into m_PredBatchNoise
    {
        auto start = std::chrono::steady_clock::now();
        const auto &tensor = m_OutputTensorsBufBank[m_infer_in_pingpong_index][m_modelsExecOrder[UNET_MODEL_IDX]].begin()->second;
        uint16_t *src;
        if (0 != m_qnnTensorMemorySet.count(tensor))
        {
            src = (uint16_t *)m_ioTensor->getBuffer((Qnn_Tensor_t *)tensor);
        }
        else
        {
            src = (uint16_t *)tensor;
        }
        // Applying de-qunatization on Unet output data
        for (size_t idx = m_PredBatchNoise.size() / 2; idx < m_PredBatchNoise.size(); idx++)
        {
            double value = ((double)(*src) + m_UnetOutQuantParam.offset) * m_UnetOutQuantParam.scale;
            m_PredBatchNoise[idx] = (float32_t)value;
            src++;
        }
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("writing Unet-1 output (cpp) took", start, stop);

#ifdef PRELOAD_DATA
        if (inference_count < INJECTED_LIMIT_COUNT)
        {
            char buffer1[20];
            sprintf(buffer1, "%03d", inference_count);
            if (false == Helpers::readRawData((void *)((char *)m_PredBatchNoise.data() + (m_PredBatchNoise.size() * sizeof(m_PredBatchNoise[0]) / 2)),
                                              m_PredBatchNoise.size() * sizeof(m_PredBatchNoise[0]) / 2,
                                              m_DemoDataFolder + "unet/sample_" + m_SampleNum + "/outputs/" + std::string(buffer1) + "_t7_pred_cond.bin"))
            {
                QNN_ERROR("There is an Error in reading the data from file");
                return false;
            }
        }
#endif
    }

    // Run Scheduler
    {
        auto start = std::chrono::steady_clock::now();
        int32_t timeStep;
        m_offTargetDataLoader->get_time_step(m_StepIdx, timeStep);

#ifdef DEBUG_DUMP
        char buffer[20];
        sprintf(buffer, "%03d", inference_count);
        Helpers::writeRawData((void *)(&timeStep), sizeof(timeStep),
                              getDebugFile(Helpers::joinPath("dataloader", std::string(buffer) + "_time_step_out.raw")));
#endif
#ifdef PRELOAD_DATA
        char buffer1[20];
        sprintf(buffer1, "%03d", inference_count);
        if (false == Helpers::readRawData((void *)&timeStep, sizeof(timeStep),
                                          m_DemoDataFolder + "unet/sample_" + m_SampleNum + "/inputs/" + std::string(buffer1) + "_timestep.bin"))
        {
            QNN_ERROR("There is an Error in reading the data from file");
            return false;
        }
#endif
#ifdef DEBUG_DUMP
        Helpers::writeRawData((void *)m_SchLatent.data(), m_SchLatent.size() * sizeof(m_SchLatent[0]),
                              getDebugFile(Helpers::joinPath("scheduler", std::string(buffer) + "_latent_in.raw")));
        Helpers::writeRawData((void *)m_PredBatchNoise.data(), m_PredBatchNoise.size() * sizeof(m_PredBatchNoise[0]),
                              getDebugFile(Helpers::joinPath("scheduler", std::string(buffer) + "_pred_batch_noise_in.raw")));
        Helpers::writeRawData((void *)&timeStep, sizeof(timeStep),
                              getDebugFile(Helpers::joinPath("scheduler", std::string(buffer) + "_time_step_in.raw")));
#endif

        if (true != m_schedulerSolver->step((void *)m_PredBatchNoise.data(), timeStep, (void *)m_SchLatent.data(), (void *)m_SchLatent.data()))
        {
            QNN_ERROR("There is an Error in running step-%d on Scheduler!", inference_count);
            return false;
        }
        auto stop = std::chrono::steady_clock::now();
        Helpers::logProfile("inference Scheduler (cpp) took", start, stop);

#ifdef DEBUG_DUMP
        Helpers::writeRawData((void *)m_SchLatent.data(), m_SchLatent.size() * sizeof(m_SchLatent[0]),
                              getDebugFile(Helpers::joinPath("scheduler", std::string(buffer) + "_latent_out.raw")));
#endif
#ifdef PRELOAD_DATA
        if (inference_count < INJECTED_LIMIT_COUNT)
        {
            if (false == Helpers::readRawData((void *)m_SchLatent.data(), m_SchLatent.size() * sizeof(m_SchLatent[0]),
                                              m_DemoDataFolder + "unet/sample_" + m_SampleNum + "/outputs/" + std::string(buffer1) + "_t5_latents.bin"))
            {
                QNN_ERROR("There is an Error in reading the data from file");
                return false;
            }
        }
#endif
    }

    // Increment the step index
    m_StepIdx++;

    // Running VAE
    if (true == runVAE)
    {
        // Writing Scheduler latent output data for VAE
        {
            auto start = std::chrono::steady_clock::now();
            uint16_t *dst;
            const auto &tensor = m_InputTensorsBufBank[m_infer_in_pingpong_index][m_modelsExecOrder[VAE_MODEL_IDX]].begin()->second;
            if (0 != m_qnnTensorMemorySet.count(tensor))
            {
                dst = (uint16_t *)m_ioTensor->getBuffer((Qnn_Tensor_t *)tensor);
            }
            else
            {
                dst = (uint16_t *)tensor;
            }
            // Applying qunatization on Scheduler output data for VAE
            for (size_t idx = 0; idx < m_SchLatent.size(); idx++)
            {
                double value = m_SchLatent[idx] / m_VaeInQuantParam.scale - m_VaeInQuantParam.offset;
                value = value < 0.0 ? 0.0 : value > 65535.0 ? 65535.0
                                                            : value;
                *dst = (uint16_t)value;
                dst++;
            }
            auto stop = std::chrono::steady_clock::now();
            Helpers::logProfile("Writing VAE input data (cpp) took", start, stop);
        }
        // Execute inference
        {
            auto start = std::chrono::steady_clock::now();
            const auto &graphName = m_modelsExecOrder[VAE_MODEL_IDX];
            if (true != ExecuteModel(m_InputTensorsBank[m_infer_in_pingpong_index][graphName],
                                     m_OutputTensorsBank[m_infer_in_pingpong_index][graphName], graphName))
                return false;
            auto stop = std::chrono::steady_clock::now();
            Helpers::logProfile("inference VAE (cpp) took", start, stop);

#ifdef DEBUG_DUMP
            char buffer[20];
            sprintf(buffer, "%03d", inference_count);
            for (const auto &tensorNameMemory : m_InputTensorsBufBank[m_infer_in_pingpong_index][graphName])
            {
                writeTensorData((Qnn_Tensor_t *)(tensorNameMemory.second),
                                getDebugFile(Helpers::joinPath("vae", std::string(buffer) + "_" + tensorNameMemory.first + "_in.raw")));
            }
            for (const auto &tensorNameMemory : m_OutputTensorsBufBank[m_infer_in_pingpong_index][graphName])
            {
                writeTensorData((Qnn_Tensor_t *)(tensorNameMemory.second),
                                getDebugFile(Helpers::joinPath("vae", std::string(buffer) + "_" + tensorNameMemory.first + "_out.raw")));
            }
#endif
        }

        // Transfer stages data which include original image for overlay
        if (m_PreStagesData[m_infer_in_pingpong_index].overlayOnImage)
        {
#ifdef USE_PRELOADED_IMAGE
            m_OrgDataPost[m_infer_in_pingpong_index] = m_OrgDataPre[m_infer_in_pingpong_index];
#else
            std::memcpy(m_OrgDataPost[m_infer_in_pingpong_index], m_OrgDataPre[m_infer_in_pingpong_index],
                        m_InputDims.getImageSize());
#endif
        }
        m_PostStagesData[m_infer_in_pingpong_index] = m_PreStagesData[m_infer_in_pingpong_index];

        lockPostProcessBufferAccess();

        // Increment inference ping-pong index
        m_infer_in_pingpong_index = ++m_infer_in_pingpong_index % m_preProcessBankSize;
    }

    QNN_DEBUG("%s: DONE QNN Iteration %d", __FUNCTION__, inference_count);
    inference_count++;

    return true;
}

bool QnnApiHelpers::PostProcessOutput(
    bool showLabels,
    bool showConfScores,
    Helpers::InferenceReturn &returnVal,
    bool dumpPostOutput,
    std::string outputLocation)
{
    auto &postprocess_count = m_postprocess_count;
    QNN_DEBUG("%s: START Iteration %d", __FUNCTION__, postprocess_count);

    bool ret = true;
    // Reading output tensor memory
    const auto &tensor = m_OutputTensorsBufBank[m_infer_out_pingpong_index][m_OutputTensorName.first][m_OutputTensorName.second];
    void *tensor_mem_buf;
    if (0 != m_qnnTensorMemorySet.count(tensor))
    {
        tensor_mem_buf = m_ioTensor->getBuffer((Qnn_Tensor_t *)tensor);
    }
    else
    {
        tensor_mem_buf = tensor;
    }

    auto start = std::chrono::steady_clock::now();
    if (nullptr != sd_helper)
    {
        std::unordered_map<std::string, void *> modelOutputData{
            {m_OutputTensorName.second, tensor_mem_buf}};
        ret = sd_helper->PostProcessOutput(modelOutputData,
                                           m_OrgDataPost[m_post_pingpong_index],
                                           m_PostStagesData[m_post_pingpong_index],
                                           showLabels, showConfScores, returnVal,
                                           dumpPostOutput, outputLocation);
        returnVal.m_loop = m_PostStagesData[m_post_pingpong_index].loopNum;
    }
    else
    {
        returnVal.m_Scenes = {};
        returnVal.m_loop = m_PostStagesData[m_post_pingpong_index].loopNum;
        returnVal.m_PostProcessedImageSize = m_OutputDims.getImageSize();
        returnVal.m_ImageData = (void *)malloc(returnVal.m_PostProcessedImageSize);
        std::memcpy((char *)returnVal.m_ImageData, (char *)tensor_mem_buf, returnVal.m_PostProcessedImageSize);
    }
    auto stop = std::chrono::steady_clock::now();
    Helpers::logProfile("PostProcessOutput (cpp) took", start, stop);

#ifdef DEBUG_DUMP
    char buffer[20];
    sprintf(buffer, "%03d", postprocess_count);
    writeTensorData((Qnn_Tensor_t *)(tensor),
                    getDebugFile(Helpers::joinPath("output", std::string(buffer) + "_" + m_OutputTensorName.second + "_in.raw")));
#endif
#ifdef OUTPUT_DUMP
    char buffer1[20];
    sprintf(buffer1, "%03d", postprocess_count);
    Helpers::writeRawData((void *)returnVal.m_ImageData, returnVal.m_PostProcessedImageSize,
                          getDebugFile(Helpers::joinPath("output", std::string(buffer1) + "_rgba_out.raw")));
#endif

    // Increment post ping-pong index
    m_post_pingpong_index = ++m_post_pingpong_index % m_postProcessBankSize;

    QNN_DEBUG("%s: DONE Iteration %d\n", __FUNCTION__, postprocess_count);
    postprocess_count++;

    return ret;
}

// size_t QnnApiHelpers::get_token_ids(const char* text, uint32_t* buffer, size_t buffer_length) {
//
// }
