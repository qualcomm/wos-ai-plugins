// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------

#include "QnnMem.h"
#include "RpcMem.hpp"
#include "QnnTypeMacros.hpp"

#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS  1

RpcMem::RpcMem(Qnn_ContextHandle_t contextHandle, QNN_INTERFACE_VER_TYPE* qnnInterface)
    : m_libCdspRpc(nullptr),
      m_rpcMemAlloc(nullptr),
      m_rpcMemFree(nullptr),
      m_rpcMemToFd(nullptr),
      m_qnnInterface(qnnInterface),
      m_contextHandle(contextHandle) {
  (void)m_contextHandle;
}


bool RpcMem::initialize() {
  // On Android, 32-bit and 64-bit libcdsprpc.so can be found at /vendor/lib and /vendor/lib64
  //  respectively.
  m_libCdspRpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (nullptr == m_libCdspRpc) {
    QNN_ERROR("Unable to load backend. dlerror(): %s", dlerror());
    return false;
  }
  m_rpcMemAlloc = (RpcMemAllocFn_t)dlsym(m_libCdspRpc, "rpcmem_alloc");
  m_rpcMemFree  = (RpcMemFreeFn_t)dlsym(m_libCdspRpc, "rpcmem_free");
  m_rpcMemToFd  = (RpcMemToFdFn_t)dlsym(m_libCdspRpc, "rpcmem_to_fd");
  if (nullptr == m_rpcMemAlloc || nullptr == m_rpcMemFree || nullptr == m_rpcMemToFd) {
    QNN_ERROR("Unable to access symbols in libcdsprpc. dlerror(): %s", dlerror());
    return false;
  }
  return true;
}

RpcMem::~RpcMem() {
  if (m_libCdspRpc) {
    QNN_DEBUG("Closing libcdsprpc.so handle");
    dlclose(m_libCdspRpc);
  }
}

void* RpcMem::getBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_WARN("getBuffer: received a null pointer to a tensor");
    return nullptr;
  }
  if (m_tensorToRpcMem.find(tensor) == m_tensorToRpcMem.end()) {
    QNN_WARN("getBuffer: tensor not found");
    return nullptr;
  }
  return m_tensorToRpcMem[tensor].memPointer;
}

int RpcMem::getFd(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_WARN("getFd: received a null pointer to a tensor");
    return -1;
  }
  if (m_tensorToRpcMem.find(tensor) == m_tensorToRpcMem.end()) {
    QNN_WARN("getFd: tensor not found");
    return -1;
  }
  return m_tensorToRpcMem[tensor].fd;
}

size_t RpcMem::getBufferSize(Qnn_Tensor_t *tensor) {
  if (!tensor) {
    QNN_WARN("getBufferSize: received a null pointer to a tensor");
    return 0;
  }
  if (m_tensorToRpcMem.find(tensor) == m_tensorToRpcMem.end()) {
    QNN_WARN("getBufferSize: tensor not found");
    return 0;
  }
  return m_tensorToRpcMem[tensor].size;
};

bool RpcMem::allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) {
  if (m_libCdspRpc == nullptr) {
    QNN_ERROR("RpcMem not initialized");
    return false;
  }
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensor");
    return false;
  }
  if (m_tensorToRpcMem.find(tensor) != m_tensorToRpcMem.end()) {
    QNN_ERROR("Tensor already allocated");
    return false;
  }

  auto memPointer = m_rpcMemAlloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, tensorDataSize);
  auto status     = true;
  if (!memPointer) {
    QNN_ERROR("rpcmem_alloc failure");
    status = false;
  }
  int memfd = -1;
  if (status == true) {
    memfd = m_rpcMemToFd(memPointer);
    if (memfd == -1) {
      QNN_ERROR("rpcmem_to_fd failure");
      status = false;
    }
  }
  if (status == true) {
    Qnn_MemDescriptor_t memDescriptor = {
        {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
        QNN_TENSOR_GET_DATA_TYPE(tensor),
        QNN_MEM_TYPE_ION,
        {{-1}}};
    memDescriptor.ionInfo.fd = memfd;
    QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
    QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);

    Qnn_MemHandle_t memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
    if (QNN_SUCCESS != m_qnnInterface->memRegister(
#ifdef QNN_ENABLE_API_2x_P2
                            m_contextHandle,
#endif
                            &memDescriptor,
                            1,
                            &(memHandle))) {
      QNN_ERROR("Failure to register ion memory with the backend");
      status = false;
    }
    QNN_TENSOR_SET_MEM_HANDLE(tensor, memHandle);
  }
  if (status == true) {
    m_tensorToRpcMem.insert({tensor, RpcMemTensorData(memfd, memPointer, tensorDataSize)});
  }
  if (status == false) {
    if (m_rpcMemFree) {
      m_rpcMemFree(memPointer);
    }
  }
  return status;
}

bool RpcMem::freeTensorBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    QNN_ERROR("Received nullptr for tensor");
    return false;
  }

  if (m_sameMemoryFreeTensors.find(tensor) != m_sameMemoryFreeTensors.end()) {
    if (m_tensorToRpcMem.find(tensor) == m_tensorToRpcMem.end()) {
      QNN_ERROR("Tensor not found");
      return false;
    }
    m_tensorToRpcMem.erase(tensor);
  } else {
    auto memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
    if (QNN_SUCCESS != m_qnnInterface->memDeRegister(&memHandle, 1)) {
      QNN_ERROR("Failed to deregister ion memory with the backend");
      return false;
    }
    QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_UNDEFINED);
    if (m_tensorToRpcMem.find(tensor) == m_tensorToRpcMem.end()) {
      QNN_ERROR("Tensor not found");
      return false;
    }
    if (m_rpcMemFree) {
      m_rpcMemFree(m_tensorToRpcMem[tensor].memPointer);
    }
    m_tensorToRpcMem.erase(tensor);
  }

  return true;
}

bool RpcMem::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  if (nullptr == dest || nullptr == src) {
    QNN_ERROR("Received nullptr");
    return false;
  }
  if (m_tensorToRpcMem.find(src) == m_tensorToRpcMem.end()) {
    QNN_ERROR("Src Tensor not found");
    return false;
  }

  if (false == freeTensorBuffer(dest)) {
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
  QNN_TENSOR_SET_MEM_HANDLE(dest, QNN_TENSOR_GET_MEM_HANDLE(src));
  m_tensorToRpcMem.insert({dest, m_tensorToRpcMem[src]});
  m_sameMemoryFreeTensors.insert(dest);

  return true;
}
