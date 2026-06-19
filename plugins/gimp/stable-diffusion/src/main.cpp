// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------

#include <dlfcn.h>

#include <iostream>
#include <filesystem>
#include <memory>
#include <string>

#include "GetOpt.hpp"
#include "QnnApiHelpers.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define UIHELPER

#ifdef UIHELPER
#include "UiHelper.h"
#else

#define VAE_START_POINT 10
#define VAE_FREQ 5

static void *sg_backendHandle{nullptr};
static void *sg_modelHandle{nullptr};

#endif // UIHELPER

void showHelp()
{
    std::cout
        << "\nDESCRIPTION:\n"
        << "------------\n"
        << "Sample application demonstrating how to load and execute a neural network\n"
        << "using QNN APIs.\n"
        << "\n\n"
        << "REQUIRED ARGUMENTS:\n"
        << "-------------------\n"
        << "  --config_file_Path  <FILE>      Path to the model containing a QNN network.\n"
        << "\n"
        << "  --backend           <FILE>      Path to a QNN backend to execute the model.\n"
        << "\n"
        << "  --input_file        <FILE>      Path to a file listing the inputs for the network.\n"
        << "                                  If there are multiple graphs in model.so, this has\n"
        << "                                  to be comma separated list of input list files.\n"
        << "\n"
        << "  --retrieve_context  <VAL>       Path to cached binary from which to load a saved\n"
           "                                  context from and execute graphs. --retrieve_context "
           "and\n"
           "                                  --model are mutually exclusive. Only one of the "
           "options\n"
           "                                  can be specified at a time.\n"
        << "\n\n"

        << "OPTIONAL ARGUMENTS:\n"
        << "-------------------\n"

        << "  --debug                         Specifies that output from all layers of the network\n"
        << "                                  will be saved.\n"
        << "\n"
        << "  --output_dir        <DIR>       The directory to save output to. Defaults to "
           "./output.\n"
        << "\n"
        << "  --model_version      <VAL>      Version of StableDiffusion can be chosen here.\n "
        << "\n"
        << "  --output_data_type  <VAL>       Data type of the output. Values can be:\n\n"
           "                                    1. float_only:       dump outputs in float only.\n"
           "                                    2. native_only:      dump outputs in data type "
           "native\n"
           "                                                         to the model. For ex., "
           "uint8_t.\n"
           "                                    3. float_and_native: dump outputs in both float and\n"
           "                                                         native.\n\n"
           "                                    (This is N/A for a float model. In other cases,\n"
           "                                     if not specified, defaults to float_only.)\n"
        << "\n"
        << "  --input_data_type   <VAL>       Data type of the input. Values can be:\n\n"
           "                                    1. float:     reads inputs as floats and quantizes\n"
           "                                                  if necessary based on quantization\n"
           "                                                  parameters in the model.\n"
           "                                    2. native:    reads inputs assuming the data type to "
           "be\n"
           "                                                  native to the model. For ex., "
           "uint8_t.\n\n"
           "                                    (This is N/A for a float model. In other cases,\n"
           "                                     if not specified, defaults to float.)\n"
        << "\n"
        << "  --op_packages       <VAL>       Provide a comma separated list of op packages \n"
           "                                  and interface providers to register. The syntax is:\n"
           "                                  "
           "op_package_path:interface_provider[,op_package_path:interface_provider...]\n"
        << "\n"
        << "  --profiling_level   <VAL>       Enable profiling. Valid Values:\n"
           "                                    1. basic:    captures execution and init time.\n"
           "                                    2. detailed: in addition to basic, captures\n"
           "                                                 per Op timing for execution.\n"
        << "\n"
        << "  --save_context      <VAL>       Specifies that the backend context and metadata "
           "related \n"
           "                                  to graphs be saved to a binary file.\n"
           "                                  Value of this parameter is the name of the name\n"
           "                                  required to save the context binary to.\n"
           "                                  Saved in the same path as --output_dir option.\n"
           "                                  Note: --retrieve_context and --save_context are "
           "mutually\n"
           "                                  exclusive. Both options should not be specified at\n"
           "                                  the same time.\n"
        << "\n"
#ifdef QNN_ENABLE_DEBUG
        << "  --log_level                     Specifies max logging level to be set.  Valid "
           "settings: \n"
           "                                 \"error\", \"warn\", \"info\", \"verbose\" and "
           "\"debug\"."
           "\n"
#else
        << "  --log_level                     Specifies max logging level to be set.  Valid "
           "settings: \n"
           "                                 \"error\", \"warn\", \"info\" and \"verbose\"."
           "\n"
#endif
        << "\n"
        << "  --system_library     <FILE>     Path to QNN System library needed to exercise "
           "reflection APIs\n"
           "                                  when loading a context from a binary cache.\n"
           "\n"
        << "  --version                       Print the QNN SDK version.\n"
        << "\n"
        << "  --help                          Show this help message.\n"
        << std::endl;
}

void showHelpAndExit(std::string &&error)
{
    std::cerr << "ERROR: " << error << "\n";
    std::cerr << "Please check help below:\n";
    showHelp();
    std::exit(EXIT_FAILURE);
}

bool processCommandLine(int argc,
                        char **argv)
{
    enum OPTIONS
    {
        OPT_HELP = 0,
        OPT_CFG_FILE_PATH = 1,
        OPT_BACKEND = 2,
        OPT_INPUT_LIST = 3,
        OPT_OUTPUT_DIR = 4,
        OPT_OP_PACKAGES = 5,
        OPT_DEBUG_OUTPUTS = 6,
        OPT_OUTPUT_DATA_TYPE = 7,
        OPT_INPUT_DATA_TYPE = 8,
        OPT_LOG_LEVEL = 9,
        OPT_PROFILING_LEVEL = 10,
        OPT_RETRIEVE_CONTEXT = 11,
        OPT_SAVE_CONTEXT = 12,
        OPT_VERSION = 13,
        OPT_SYSTEM_LIBRARY = 14,
        OPT_PROMPT = 15,
        OPT_SEED = 16,
        OPT_STEP = 17,
        OPT_GUIDANCE_SCALE = 18,
        OPT_MODEL_VERSION = 19
        
    };

    const int noArgument = 0;
    const int requiredArgument = 1;

    // Create the command line options
    static struct WinOpt::option s_longOptions[] = {
        {"help", noArgument, NULL, OPT_HELP},
        {"config_file_Path", requiredArgument, NULL, OPT_CFG_FILE_PATH},
        {"backend", requiredArgument, NULL, OPT_BACKEND},
        {"system_library", requiredArgument, NULL, OPT_SYSTEM_LIBRARY},
        {"model_version", requiredArgument, NULL, OPT_MODEL_VERSION},
        {NULL, 0, NULL, 0}};

    // Command line parsing loop
    int longIndex = 0;
    int opt = 0;
    std::string config_file_Path;
    std::string backEndPath;
    std::string systemLibraryPath;
    std::string prompt;
    std::string model_version = VERSION_1_5;
    int seed = 0;
    float guidance_scale = 7.5;
    int step = 20;
    int num_images = 1;

    while ((opt = GetOptLongOnly(argc, argv, "", s_longOptions, &longIndex)) != -1)
    {
        switch (opt)
        {
        case OPT_HELP:
            showHelp();
            std::exit(EXIT_SUCCESS);
            break;

        case OPT_CFG_FILE_PATH:
            config_file_Path = WinOpt::optarg;
            break;

        case OPT_BACKEND:
            backEndPath = WinOpt::optarg;
            break;

        case OPT_SYSTEM_LIBRARY:
            systemLibraryPath = WinOpt::optarg;
            if (systemLibraryPath.empty())
            {
                showHelpAndExit("System library path not specified.");
            }
            break;
        case OPT_MODEL_VERSION: 
            if (WinOpt::optarg != NULL)
            {
                model_version = WinOpt::optarg;
                if (model_version != VERSION_1_5 && model_version != VERSION_2_1)
                {
                    showHelpAndExit("Invalid model version specified. Use SD_1_5 or SD_2_1.");
                }
            }
            break;
        default:
            std::cerr << "ERROR: Invalid argument passed: " << argv[WinOpt::optind - 1]
                      << "\nPlease check the Arguments section in the description below.\n";
            showHelp();
            std::exit(EXIT_FAILURE);
        }
    }

    if (config_file_Path.empty())
    {
        showHelpAndExit("Missing option: --config_file_Path\n");
    }

    if (backEndPath.empty())
    {
        showHelpAndExit("Missing option: --backend\n");
    }

    UiHelper *ui = new UiHelper(config_file_Path, backEndPath, model_version);

    // If init fails, exit cleanly before connecting to the socket server
    if (!ui->init()) {
        fprintf(stderr, "Initialization failure — aborting\n");
        delete ui;
        return false;
    }

    socket_communication::Client client("127.0.0.1", 5001);
    std::cout << "server started" << std::endl;
    client.Send("initialized");

    while (true)
    {
        std::string answer = client.Receive();
        std::cout << "[Server]: " << answer << std::endl;

        if (!answer.empty())
        {

            std::vector<std::string> inputs;
            // Returns first token
            char *token = strtok(&answer[0], ":");

            // Keep printing tokens while one of the
            // delimiters present in str[].
            while (token != NULL)
            {
                // std::cout << token << " ";
                inputs.push_back(token);
                token = strtok(NULL, ":");
            }

            if (inputs.size() < 5)
                break;

            seed = std::stoi(inputs[1]);
            step = std::stoi(inputs[2]);
            prompt = inputs[3];
            guidance_scale = std::stof(inputs[0]);
            num_images = std::stoi(inputs[4]);

            std::cout << "\nInputs are: \nPrompt : " << prompt << "\nSeed : " << seed
                      << "\nStep : " << step << "\nGuidance Scale: " << guidance_scale
                      << "\nNumber of Images: " << num_images << std::endl;
            break;
        }
    }
    std::string data;
    while (true)
    {
        if (num_images != 1) {
            seed = rand() % 50;
        }
        data = client.Receive();
        std::cout << "[Server]: " << data << std::endl;
        if (data == "execute_next")
        {
            bool execRet = ui->executeStableDiffusion(seed, step, guidance_scale, prompt);
            if (execRet)
                cv::imwrite("test.jpeg", ui->outputModelImage);
            client.Send("execution_complete");
        }
        else
            break;
    }
    delete ui;

    return true;
}

int main(int argc, char **argv)
{
    system("rmdir /s /q output");

    /*freopen("output.txt", "w", stdout);
    freopen("error.txt", "w", stderr);*/

    bool ret = processCommandLine(argc, argv);

    if (sg_backendHandle)
    {
        dlclose(sg_backendHandle);
    }
    if (sg_modelHandle)
    {
        dlclose(sg_modelHandle);
    }

    return ret ? 0 : -1;
}
