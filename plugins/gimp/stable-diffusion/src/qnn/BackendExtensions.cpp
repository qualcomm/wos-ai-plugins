// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------

#include "BackendExtensions.hpp"
#include "NetRunBackend.hpp"

BackendExtensions::BackendExtensions(BackendExtensionsConfigs backendExtensionsConfig,
                                     void* backendLibHandle,
                                     PerfProfile perfProfile,
                                     std::shared_ptr<ICommandLineManager> clManager)
    : m_backendExtensionsLibPath(backendExtensionsConfig.sharedLibraryPath),
      m_backendExtensionsConfigPath(backendExtensionsConfig.configFilePath),
      m_backendInterface(nullptr),
      m_isNetRunBackendInterface(false),
      m_createBackendInterfaceFn(nullptr),
      m_destroyBackendInterfaceFn(nullptr),
      m_backendLibHandle(backendLibHandle),
      m_perfProfile(perfProfile),
      m_clManager(clManager) {
#ifdef QNN_ENABLE_API_2x_P2
  (void)m_perfProfile;
#endif
}

BackendExtensions::~BackendExtensions() {
  if (nullptr != m_backendInterface) {
    if (m_isNetRunBackendInterface) {
      QNN_DEBUG("Deleting NetRun Backend Interface");
      delete m_backendInterface;
    } else {
      if (nullptr != m_destroyBackendInterfaceFn) {
        QNN_DEBUG("Destroying Backend Interface");
        m_destroyBackendInterfaceFn(m_backendInterface);
      }
    }
  }
}

bool BackendExtensions::loadFunctionPointers() {
#ifdef UWP
    void* libHandle = LoadLibraryExW(std::wstring(m_backendExtensionsLibPath.begin(), m_backendExtensionsLibPath.end()).c_str(), NULL, 0);
#else
    void* libHandle = dlopen(m_backendExtensionsLibPath.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif // UWP
  if (nullptr == libHandle) {
    QNN_ERROR("Unable to load backend extensions lib: [%s]. dlerror(): [%s]",
              m_backendExtensionsLibPath.c_str(), dlerror());
    return false;
  }
#ifdef UWP
  m_createBackendInterfaceFn = (CreateBackendInterfaceFnType_t)GetProcAddress(
      (HMODULE)libHandle, "createBackendInterface");
  m_destroyBackendInterfaceFn = (DestroyBackendInterfaceFnType_t)GetProcAddress(
      (HMODULE)libHandle, "destroyBackendInterface");
#else
  m_createBackendInterfaceFn = (CreateBackendInterfaceFnType_t)dlsym(
      libHandle, "createBackendInterface");
  m_destroyBackendInterfaceFn = (DestroyBackendInterfaceFnType_t)dlsym(
      libHandle, "destroyBackendInterface");
#endif
  
  if (nullptr == m_createBackendInterfaceFn || nullptr == m_destroyBackendInterfaceFn) {
    QNN_ERROR("Unable to find symbols. dlerror(): [%s]", dlerror());
    return false;
  }

  return true;
}

void logStdoutCallback(const char* fmt,
    QnnLog_Level_t level,
    uint64_t timestamp,
    va_list argp) {
    const char* levelStr = "";
    switch (level) {
    case QNN_LOG_LEVEL_ERROR:
        levelStr = " ERROR ";
        break;
    case QNN_LOG_LEVEL_WARN:
        levelStr = "WARNING";
        break;
    case QNN_LOG_LEVEL_INFO:
        levelStr = "  INFO ";
        break;
    case QNN_LOG_LEVEL_DEBUG:
        levelStr = " DEBUG ";
        break;
    case QNN_LOG_LEVEL_VERBOSE:
        levelStr = "VERBOSE";
        break;
    case QNN_LOG_LEVEL_MAX:
        levelStr = "UNKNOWN";
        break;
    }
    fprintf(stdout, "[%-7s] ", levelStr);
    vfprintf(stdout, fmt, argp);
    fprintf(stdout, "\n");
}

bool BackendExtensions::initialize() {

  if (m_backendExtensionsLibPath.empty() && m_backendExtensionsConfigPath.empty()) {
    QNN_WARN("No BackendExtensions lib provided; initializing NetRunBackend Interface");
    m_isNetRunBackendInterface = true;
    m_backendInterface         = new NetRunBackend();
  } else {
    QNN_DEBUG("Loading supplied backend extensions lib.");
    QNN_DEBUG("Backend extensions lib path: %s", m_backendExtensionsLibPath.c_str());
    if (m_backendExtensionsConfigPath.empty()) {
      QNN_DEBUG("Backend extensions lib specified without a config file.");
    } else {
      QNN_DEBUG("Backend extensions config path: %s", m_backendExtensionsConfigPath.c_str());
    }
    if (!loadFunctionPointers()) {
      QNN_ERROR("Failed to load function pointers.");
      return false;
    }
    if (nullptr != m_createBackendInterfaceFn) {
      m_backendInterface = m_createBackendInterfaceFn();
    }
  }
  if (nullptr == m_backendInterface) {
    QNN_ERROR("Unable to load backend extensions interface.");
    return false;
  }
  if (!(m_backendInterface->setupLogging(logStdoutCallback, QNN_LOG_LEVEL_ERROR))) {
    QNN_WARN("Unable to initialize logging in backend extensions.");
  }
  if (!m_backendInterface->initialize(m_backendLibHandle)) {
    QNN_ERROR("Unable to initialize backend extensions interface.");
    return false;
  }
  if (!m_backendInterface->setPerfProfile(m_perfProfile)) {
    QNN_ERROR("Unable to set perf profile in backend extensions interface.");
    return false;
  }
  if (!m_backendInterface->loadConfig(m_backendExtensionsConfigPath)) {
    QNN_ERROR("Unable to load backend extensions interface config.");
    return false;
  }
  if ((m_clManager != nullptr) && !m_backendInterface->loadCommandLineArgs(m_clManager)) {
    QNN_ERROR("Unable to load backend extensions' command line arguments.");
    return false;
  }

  return true;
}

IBackend* BackendExtensions::interface1() { return m_backendInterface; }
