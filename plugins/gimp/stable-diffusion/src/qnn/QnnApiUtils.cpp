// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "QnnApiUtils.hpp"
#include "QnnTypeMacros.hpp"

char* strndup(const char* str, size_t len)
{
    size_t act = strnlen(str, len);
    char* dst = (char *)malloc(act + 1);
    if (dst != 0)
    {
        memmove(dst, str, act);
        dst[act] = '\0';
    }
    return dst;
}

bool freeQnnTensorWrapper(TensorWrapper &tensorWrapper) {
  // free all pointer allocations in struct
  if (nullptr != GET_TENSOR_WRAPPER_NAME(tensorWrapper)) {
    free((void *)GET_TENSOR_WRAPPER_NAME(tensorWrapper));
  }

  Qnn_Tensor_t &tensor = GET_TENSOR_WRAPPER_TENSOR(tensorWrapper);
#ifdef QNN_ENABLE_API_2x
  free(QNN_TENSOR_GET_DIMENSIONS(tensor));
#else
  free(tensor.maxDimensions);
  free(tensor.currentDimensions);
#endif
  return true;
}

bool freeQnnTensorWrappers(TensorWrapper *&tensorWrappers, uint32_t numTensors) {
  // free all pointer allocations in struct
  for (size_t i = 0; i < numTensors; i++) {
    freeQnnTensorWrapper(tensorWrappers[i]);
  }
  free(tensorWrappers);

  return true;
}

bool freeGraphsInfo(GraphInfoPtr_t **graphsInfo, uint32_t numGraphs) {
  if (graphsInfo == nullptr || *graphsInfo == nullptr) {
    return false;
  }
  for (uint32_t i = 0; i < numGraphs; i++) {
    if (nullptr != (*graphsInfo)[i]->graphName) {
      free((*graphsInfo)[i]->graphName);
    }
    freeQnnTensorWrappers((*graphsInfo)[i]->inputTensors, (*graphsInfo)[i]->numInputTensors);
    freeQnnTensorWrappers((*graphsInfo)[i]->outputTensors, (*graphsInfo)[i]->numOutputTensors);
  }
  free(**graphsInfo);
  free(*graphsInfo);
  *graphsInfo = nullptr;

  return true;
}

bool freeGraphInfo(GraphInfo_t *graphInfo) {
  if (graphInfo == nullptr) {
    return false;
  }
  if (nullptr != graphInfo->graphName) {
    free(graphInfo->graphName);
  }
  freeQnnTensorWrappers(graphInfo->inputTensors, graphInfo->numInputTensors);
  freeQnnTensorWrappers(graphInfo->outputTensors, graphInfo->numOutputTensors);
  free(graphInfo);
  return true;
}

// Not needed for API 2x, tensor names are already part of Qnn_Tensor_t
#ifndef QNN_ENABLE_API_2x
bool populateTensorNamesFromMetadata(
    std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>>
        &graphTensorIdToNamesMap,
    GraphInfo_t **&graphsInfo,
    const uint32_t graphsCount) {
  for (uint32_t gIdx = 0; gIdx < graphsCount; gIdx++) {
    std::string graphName = std::string((*graphsInfo)[gIdx].graphName);
    if (graphTensorIdToNamesMap.find(graphName) == graphTensorIdToNamesMap.end()) {
      QNN_ERROR("Graph [%s] not found in metadata.", graphName.c_str());
      return false;
    }
    for (uint32_t tIdx = 0; tIdx < (*graphsInfo)[gIdx].numInputTensors; tIdx++) {
      auto tensorId = QNN_TENSOR_GET_ID(&((*graphsInfo)[gIdx].inputTensors[tIdx].tensor));
      if (graphTensorIdToNamesMap[graphName].find(tensorId) ==
          graphTensorIdToNamesMap[graphName].end()) {
        QNN_ERROR("Input tensor name for [%u] in graph [%s] not found in metadata.",
                  tensorId, graphName.c_str());
        return false;
      }
      (*graphsInfo)[gIdx].inputTensors[tIdx].name =
          strndup(graphTensorIdToNamesMap[graphName][tensorId].c_str(),
                  strlen(graphTensorIdToNamesMap[graphName][tensorId].c_str()));
    }
    for (uint32_t tIdx = 0; tIdx < (*graphsInfo)[gIdx].numOutputTensors; tIdx++) {
      auto tensorId = QNN_TENSOR_GET_ID(&((*graphsInfo)[gIdx].outputTensors[tIdx].tensor));
      if (graphTensorIdToNamesMap[graphName].find(tensorId) ==
          graphTensorIdToNamesMap[graphName].end()) {
        QNN_ERROR("Output tensor name for [%u] in graph [%s] not found in metadata.",
                  tensorId, graphName.c_str());
        return false;
      }
      (*graphsInfo)[gIdx].outputTensors[tIdx].name =
          strndup(graphTensorIdToNamesMap[graphName][tensorId].c_str(),
                  strlen(graphTensorIdToNamesMap[graphName][tensorId].c_str()));
    }
  }
  return true;
}
#endif

bool copyTensorsInfo(const Qnn_Tensor_t *tensorsInfoSrc,
                     TensorWrapper *&tensorWrappers,
                     uint32_t tensorsCount) {

  auto returnStatus = true;
  tensorWrappers    = (TensorWrapper *)calloc(tensorsCount, sizeof(TensorWrapper));
  if (nullptr == tensorWrappers) {
    QNN_ERROR("Failed to allocate memory for tensorWrappers.");
    return false;
  }
  if (returnStatus) {
    for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
      QNN_DEBUG("Extracting tensorInfo for tensor Idx: %d", (int)tIdx);
      Qnn_Tensor_t &tensor = GET_TENSOR_WRAPPER_TENSOR(tensorWrappers[tIdx]);
      tensor               = QNN_TENSOR_INIT;

      const char *tensorName = QNN_TENSOR_GET_NAME(&tensorsInfoSrc[tIdx]);
      if (!tensorName) {
        QNN_TENSOR_SET_NAME(tensor, nullptr);
      } else {
        QNN_TENSOR_SET_NAME(tensor, strndup(tensorName, strlen(tensorName)));
      }

      QNN_TENSOR_SET_ID(tensor, QNN_TENSOR_GET_ID(&tensorsInfoSrc[tIdx]));
      QNN_TENSOR_SET_TYPE(tensor, QNN_TENSOR_GET_TYPE(&tensorsInfoSrc[tIdx]));
      QNN_TENSOR_SET_DATA_FORMAT(tensor, QNN_TENSOR_GET_DATA_FORMAT(&tensorsInfoSrc[tIdx]));
      QNN_TENSOR_SET_DATA_TYPE(tensor, QNN_TENSOR_GET_DATA_TYPE(&tensorsInfoSrc[tIdx]));
      Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
      qParams.encodingDefinition =
          QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).encodingDefinition;
      qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
      if (QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding ==
          QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        qParams.quantizationEncoding =
            QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding;
        qParams.scaleOffsetEncoding =
            QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).scaleOffsetEncoding;
      } else if (QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding ==
                 QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        qParams.quantizationEncoding =
            QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).quantizationEncoding;
        qParams.axisScaleOffsetEncoding.axis =
            QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx]).axisScaleOffsetEncoding.axis;
        qParams.axisScaleOffsetEncoding.numScaleOffsets =
            QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                .axisScaleOffsetEncoding.numScaleOffsets;
        if (QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                .axisScaleOffsetEncoding.numScaleOffsets > 0) {
          qParams.axisScaleOffsetEncoding.scaleOffset =
              (Qnn_ScaleOffset_t *)malloc(QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                                              .axisScaleOffsetEncoding.numScaleOffsets *
                                          sizeof(Qnn_ScaleOffset_t));
          if (qParams.axisScaleOffsetEncoding.scaleOffset) {
            for (size_t idx = 0; idx < QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                                           .axisScaleOffsetEncoding.numScaleOffsets;
                 idx++) {
              qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
                  QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                      .axisScaleOffsetEncoding.scaleOffset[idx]
                      .scale;
              qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
                  QNN_TENSOR_GET_QUANT_PARAMS(&tensorsInfoSrc[tIdx])
                      .axisScaleOffsetEncoding.scaleOffset[idx]
                      .offset;
            }
          }
        }
      }
      QNN_TENSOR_SET_QUANT_PARAMS(tensor, qParams);
      QNN_TENSOR_SET_RANK(tensor, QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]));
#ifdef QNN_ENABLE_API_2x
      QNN_TENSOR_SET_DIMENSIONS(tensor, nullptr);
      if (QNN_TENSOR_GET_RANK(tensorsInfoSrc[tIdx]) > 0) {
        QNN_TENSOR_SET_DIMENSIONS(
            tensor,
            (uint32_t *)malloc(QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]) * sizeof(uint32_t)));
        if (QNN_TENSOR_GET_DIMENSIONS(tensor)) {
          memcpy(QNN_TENSOR_GET_DIMENSIONS(tensor),
                 QNN_TENSOR_GET_DIMENSIONS(&tensorsInfoSrc[tIdx]),
                 QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]) * sizeof(uint32_t));
        }
      }
#else
      tensor.maxDimensions     = nullptr;
      tensor.currentDimensions = nullptr;
      if (QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]) > 0) {
        tensor.maxDimensions =
            (uint32_t *)malloc(QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]) * sizeof(uint32_t));
        if (tensor.maxDimensions) {
          memcpy(tensor.maxDimensions,
                 tensorsInfoSrc[tIdx].maxDimensions,
                 QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]) * sizeof(uint32_t));
        }
        tensor.currentDimensions =
            (uint32_t *)malloc(QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]) * sizeof(uint32_t));
        if (tensor.currentDimensions) {
          memcpy(tensor.currentDimensions,
                 tensorsInfoSrc[tIdx].currentDimensions,
                 QNN_TENSOR_GET_RANK(&tensorsInfoSrc[tIdx]) * sizeof(uint32_t));
        }
      }
#endif
    }
  }

  return returnStatus;
}

bool copyGraphsInfoV3(const QnnSystemContext_GraphInfoV3_t* graphInfoSrc,
    GraphInfo_t* graphInfoDst) {
    graphInfoDst->graphName = nullptr;
    if (graphInfoSrc->graphName) {
        graphInfoDst->graphName =
            strndup(graphInfoSrc->graphName, strlen(graphInfoSrc->graphName));
    }
    graphInfoDst->inputTensors = nullptr;
    graphInfoDst->numInputTensors = 0;
    if (graphInfoSrc->graphInputs) {
        if (!copyTensorsInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
            return false;
        }
        graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
    }
    graphInfoDst->outputTensors = nullptr;
    graphInfoDst->numOutputTensors = 0;
    if (graphInfoSrc->graphOutputs) {
        if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
            graphInfoDst->outputTensors,
            graphInfoSrc->numGraphOutputs)) {
            return false;
        }
        graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
    }
    return true;
}


bool copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t *graphInfoSrc,
                      GraphInfo_t *graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) {
    graphInfoDst->graphName = strndup(graphInfoSrc->graphName, strlen(graphInfoSrc->graphName));
  }
  graphInfoDst->inputTensors    = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors    = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                         graphInfoDst->outputTensors,
                         graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t *graphsInput,
                    const uint32_t numGraphs,
                    GraphInfo_t **&graphsInfo) {

  if (!graphsInput) {
    QNN_ERROR("Received nullptr for graphsInput.");
    return false;
  }
  auto returnStatus = true;
  graphsInfo =
      (GraphInfo_t **)calloc(numGraphs, sizeof(GraphInfo_t *));
  GraphInfo_t *graphInfoArr =
      (GraphInfo_t *)calloc(numGraphs, sizeof(GraphInfo_t));
  if (nullptr == graphsInfo || nullptr == graphInfoArr) {
    QNN_ERROR("Failure to allocate memory for *graphInfo");
    returnStatus = false;
  }
  if (true == returnStatus) {
    for (size_t gIdx = 0; gIdx < numGraphs; gIdx++) {
      QNN_DEBUG("Extracting graphsInfo for graph Idx: %d", (int)gIdx);
      if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        copyGraphsInfoV1(&graphsInput[gIdx].graphInfoV1, &graphInfoArr[gIdx]);
      }
      else if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
          copyGraphsInfoV3(&graphsInput[gIdx].graphInfoV3, &graphInfoArr[gIdx]);
      }
      graphsInfo[gIdx] = graphInfoArr + gIdx;
    }
  }
  if (true != returnStatus) {
    QNN_DEBUG("Received an ERROR during extractGraphsInfo. Freeing resources.");
    if (graphsInfo) {
      for (uint32_t gIdx = 0; gIdx < numGraphs; gIdx++) {
        if (graphsInfo[gIdx]) {
          if (nullptr != graphsInfo[gIdx]->graphName) {
            free(graphsInfo[gIdx]->graphName);
            graphsInfo[gIdx]->graphName = nullptr;
          }
          freeQnnTensorWrappers(graphsInfo[gIdx]->inputTensors, graphsInfo[gIdx]->numInputTensors);
          freeQnnTensorWrappers(graphsInfo[gIdx]->outputTensors, graphsInfo[gIdx]->numOutputTensors);
        }
      }
      free(*graphsInfo);
    }
    free(graphsInfo);
    graphsInfo = nullptr;
  }

  return true;
}

bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binaryInfo,
                              GraphInfo_t **&graphsInfo,
                              uint32_t &graphsCount) {
  if (nullptr == binaryInfo) {
    QNN_ERROR("binaryInfo is nullptr.");
    return false;
  }
  graphsCount = 0;
  if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    if (binaryInfo->contextBinaryInfoV1.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV1.graphs,
                          binaryInfo->contextBinaryInfoV1.numGraphs,
                          graphsInfo)) {
        QNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV1.numGraphs;
      return true;
    }
#ifdef QNN_ENABLE_API_2x_P2
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    if (binaryInfo->contextBinaryInfoV2.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV2.graphs,
                          binaryInfo->contextBinaryInfoV2.numGraphs,
                          graphsInfo)) {
        QNN_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV2.numGraphs;
      return true;
    }
#endif
  }
  else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
      if (binaryInfo->contextBinaryInfoV3.graphs) {
          if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV3.graphs,
              binaryInfo->contextBinaryInfoV3.numGraphs,
              graphsInfo)) {
              QNN_ERROR("Failed while copying graphs Info.");
              return false;
          }
          graphsCount = binaryInfo->contextBinaryInfoV3.numGraphs;
          return true;
      }
  }
  QNN_ERROR("Unrecognized system context binary info version.");
  return false;
}




size_t getFileSize(std::string filePath) {
  std::ifstream in(filePath, std::ifstream::binary);
  if (!in) {
    QNN_ERROR("Failed to open input file: %s", filePath.c_str());
    return 0;
  }
  in.seekg(0, in.end);
  const size_t length = in.tellg();
  in.seekg(0, in.beg);
  return length;
}

bool readBinaryFromFile(std::string filePath,
                        uint8_t* buffer,
                        size_t bufferSize) {
  if (nullptr == buffer) {
    QNN_ERROR("buffer is nullptr");
    return false;
  }
  std::ifstream in(filePath, std::ifstream::binary);
  if (!in) {
    QNN_ERROR("Failed to open input file: %s", filePath.c_str());
    return false;
  }
  if (!in.read(reinterpret_cast<char*>(buffer), bufferSize)) {
    QNN_ERROR("Failed to read the contents of: %s", filePath.c_str());
    return false;
  }
  return true;
}

#ifndef QNN_ENABLE_API_2x
// Function to help extract tensorsInfos from flatbuffers structures.
bool extractTensorsInfo(
    const flatbuffers::Vector<flatbuffers::Offset<QnnTensorInfo>> *fbTensorInfosVector,
    std::string graphName,
    std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>>
        &graphTensorIdToNamesMap,
    uint32_t tensorsCount) {

  for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
    QNN_DEBUG("Extracting tensorInfo for tensor Idx: %d", (int)tIdx);
    if (graphTensorIdToNamesMap.find(graphName) == graphTensorIdToNamesMap.end()) {
      graphTensorIdToNamesMap[graphName] = std::unordered_map<uint32_t, std::string>();
    }
    auto fbTensorInfo = fbTensorInfosVector->Get(tIdx);
    if (fbTensorInfo->name() != nullptr) {
      graphTensorIdToNamesMap[graphName][fbTensorInfo->id()] = fbTensorInfo->name()->str();
    } else {
      QNN_DEBUG("fbTensorInfo->name() is nullptr for graph [%s] and tensorId [%d].",
                graphName.c_str(), fbTensorInfo->id());
      graphTensorIdToNamesMap[graphName][fbTensorInfo->id()] = "";
    }
  }

  return true;
}


// Function to help extract graphs' metadata from loaded flatbuffers structure
// in deserializeData().
bool extractGraphsInfo(
    const ContextCache *contextCache,
    std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>>
        &graphTensorIdToNamesMap,
    uint32_t *graphsCount) {

  *graphsCount        = contextCache->graphsCount();
  auto fbGraphsVector = contextCache->graphsInfo();
  for (size_t gIdx = 0; gIdx < *graphsCount; gIdx++) {
    QNN_DEBUG("Extracting graphsInfo for graph Idx: %d", (int)gIdx);
    auto fbGraph = fbGraphsVector->Get(gIdx);
    if (true != extractTensorsInfo(fbGraph->inputTensorsInfo(),
                                   fbGraph->name()->str(),
                                   graphTensorIdToNamesMap,
                                   fbGraph->inputTensorsCount())) {
      return false;
    }
    if (true != extractTensorsInfo(fbGraph->outputTensorsInfo(),
                                   fbGraph->name()->str(),
                                   graphTensorIdToNamesMap,
                                   fbGraph->outputTensorsCount())) {
      return false;
    }
  }

  return true;
}

// Function to deserialize flatbuffers related to caching.
//  1. Flatbuffers are loaded from a binary buffer.
//  2. Metadata containing a map of tenor id to names is populated.
//  3. Binary blob is retrieved and copied into a uint8_t buffer.
bool deserializeData(
    std::string filePath,
    std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>>
        &graphTensorIdToNamesMap,
    uint32_t *graphsCount,
    std::shared_ptr<uint8_t> &binaryCache,
    uint64_t &binaryCacheSize) {

  size_t fileSize{0};
  fileSize = getFileSize(filePath);
  if (0 == fileSize) {
    QNN_ERROR("Received path to an empty file. Nothing to deserialize.");
    return false;
  }

  std::unique_ptr<char> buffer(new char[fileSize]);
  if (!buffer) {
    QNN_ERROR("Failed to allocate memory.");
    return false;
  }
  if (true != readBinaryFromFile(
          filePath, reinterpret_cast<uint8_t *>(buffer.get()), fileSize)) {
    QNN_ERROR("Failed to read binary data.");
    return false;
  }
  // Verify the buffer is well-formed
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t *>(buffer.get()), fileSize);
  if (!VerifyContextCacheBuffer(verifier)) {
    QNN_ERROR("Invalid flatbuffer binary: %s", filePath.c_str());
    return false;
  }
  auto contextCache = GetContextCache(buffer.get());
  binaryCacheSize   = contextCache->binaryCacheSize();
  binaryCache       = std::shared_ptr<uint8_t>(
      static_cast<uint8_t *>(malloc(binaryCacheSize * sizeof(uint8_t))), free);
  if (nullptr == binaryCache) {
    QNN_ERROR("Failed to allocate memory for binaryCache");
    return false;
  }
  memcpy(binaryCache.get(), contextCache->binaryCache()->Data(), binaryCacheSize);

  if (true != extractGraphsInfo(contextCache, graphTensorIdToNamesMap, graphsCount)) {
    QNN_ERROR("Failed to extract graphsInfo.");
    return false;
  }

  return true;
}
#endif

bool fillDims(std::vector<size_t>& dims, uint32_t* inDimensions, uint32_t rank) {
    if (nullptr == inDimensions) {
        QNN_ERROR("input dimensions is nullptr");
        return false;
    }

    // rank=0 is invalid; rank=1 is valid (e.g. scalar timestep tensor [1])
    if (0 == rank) {
        QNN_ERROR("Invalid rank : %d", rank);
        return false;
    }

    for (size_t r = 0; r < rank; r++) {
        dims.push_back(inDimensions[r]);
    }
    // In case, rank is less than 4, we are pushing 1s
    for (size_t r = rank; r < 4; r++) {
        dims.push_back(1);
    }

    return true;
}
