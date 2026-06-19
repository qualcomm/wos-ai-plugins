# =============================================================================
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================


from __future__ import annotations
import ctypes
import uuid
from typing import Optional

DXCORE_DEVICE_ID_TO_HTP = {
    0x41304430: 73,  # Hamoa
    0x30464630: 81,  # Glymur, Mahua
}

def _guid(s: str):
    return (ctypes.c_byte * 16)(*uuid.UUID(s).bytes_le)

IID_IDXCoreAdapterFactory = _guid("78ee5945-c36e-4b13-a669-005dd11c0f06")
IID_IDXCoreAdapterList    = _guid("526c7776-40e9-459b-b711-f32ad76dfc28")
IID_IDXCoreAdapter        = _guid("f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e")

ATTR_CORE_COMPUTE = _guid("248e2800-a793-4724-abaa-23a6de1be090")
ATTR_GENERIC_ML   = _guid("b71b0d41-1088-422f-a27c-0250b7d3a988")
ATTR_NPU          = _guid("d46140c4-add7-451b-9e56-06fe8c3b58ed")

PROP_IS_HARDWARE       = 11
PROP_HARDWARE_ID       = 3
PROP_HARDWARE_ID_PARTS = 14

class _HardwareID(ctypes.Structure):
    _fields_ = [("vendorID", ctypes.c_uint32), ("deviceID", ctypes.c_uint32),
                ("subSysID", ctypes.c_uint32), ("revision", ctypes.c_uint32)]

class _HardwareIDParts(ctypes.Structure):
    _fields_ = [("vendorID", ctypes.c_uint32), ("deviceID", ctypes.c_uint32),
                ("subSysID", ctypes.c_uint32), ("revision", ctypes.c_uint32),
                ("bus", ctypes.c_uint32), ("device", ctypes.c_uint32),
                ("function", ctypes.c_uint32)]

def _vtfn(ptr, idx, restype, *argtypes):
    vtbl = ctypes.cast(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0],
                       ctypes.POINTER(ctypes.c_void_p))
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[idx])

def _release(ptr):
    _vtfn(ptr, 2, ctypes.c_uint32)(ptr)

def get_htp_architecture() -> Optional[int]:
    try:
        lib = ctypes.WinDLL("dxcore.dll")
    except OSError:
        return None

    lib.DXCoreCreateAdapterFactory.restype = ctypes.c_long
    factory = ctypes.c_void_p()
    if lib.DXCoreCreateAdapterFactory(IID_IDXCoreAdapterFactory, ctypes.byref(factory)) < 0:
        return None

    result = None
    try:
        for attr in (ATTR_CORE_COMPUTE, ATTR_GENERIC_ML, ATTR_NPU):
            adapterlist = ctypes.c_void_p()
            CreateAdapterList = _vtfn(factory.value, 3, ctypes.c_long,
                                      ctypes.c_uint32, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
            if CreateAdapterList(factory.value, 1, attr,
                                 IID_IDXCoreAdapterList, ctypes.byref(adapterlist)) < 0:
                continue
            try:
                count = _vtfn(adapterlist.value, 4, ctypes.c_uint32)(adapterlist.value)
                for i in range(count):
                    adapter = ctypes.c_void_p()
                    GetAdapter = _vtfn(adapterlist.value, 3, ctypes.c_long,
                                       ctypes.c_uint32, ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_void_p))
                    if GetAdapter(adapterlist.value, i, IID_IDXCoreAdapter,
                                  ctypes.byref(adapter)) < 0:
                        continue
                    try:
                        IsPropertySupported = _vtfn(adapter.value, 5, ctypes.c_bool, ctypes.c_uint32)
                        GetProperty = _vtfn(adapter.value, 6, ctypes.c_long,
                                            ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p)

                        if IsPropertySupported(adapter.value, PROP_IS_HARDWARE):
                            is_hw = ctypes.c_bool(False)
                            if (GetProperty(adapter.value, PROP_IS_HARDWARE,
                                            ctypes.sizeof(is_hw), ctypes.byref(is_hw)) < 0
                                    or not is_hw.value):
                                continue

                        device_id = 0
                        if IsPropertySupported(adapter.value, PROP_HARDWARE_ID):
                            hw = _HardwareID()
                            if GetProperty(adapter.value, PROP_HARDWARE_ID,
                                           ctypes.sizeof(hw), ctypes.byref(hw)) == 0:
                                device_id = hw.deviceID
                        elif IsPropertySupported(adapter.value, PROP_HARDWARE_ID_PARTS):
                            hw = _HardwareIDParts()
                            if GetProperty(adapter.value, PROP_HARDWARE_ID_PARTS,
                                           ctypes.sizeof(hw), ctypes.byref(hw)) == 0:
                                device_id = hw.deviceID

                        result = DXCORE_DEVICE_ID_TO_HTP.get(device_id)
                        if result is not None:
                            break
                    finally:
                        _release(adapter.value)
            finally:
                _release(adapterlist.value)

            if result is not None:
                break
    finally:
        _release(factory.value)

    return result

if __name__ == "__main__":
    arch = get_htp_architecture()
    print(arch if arch is not None else "Unknown")
