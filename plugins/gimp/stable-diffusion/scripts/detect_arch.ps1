<#
=============================================================================

Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

=============================================================================
#>

$dxcoreSource = @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public static class DXCoreDetect
{
    private static readonly Dictionary<uint, int> DeviceIdToHtp =
        new Dictionary<uint, int>
        {
            { 0x41304430u, 73 },
            { 0x30464630u, 81 }
        };

    private static readonly Guid IID_IDXCoreAdapterFactory =
        new Guid("78ee5945-c36e-4b13-a669-005dd11c0f06");
    private static readonly Guid IID_IDXCoreAdapterList =
        new Guid("526c7776-40e9-459b-b711-f32ad76dfc28");
    private static readonly Guid IID_IDXCoreAdapter =
        new Guid("f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e");

    private static readonly Guid ATTR_CORE_COMPUTE =
        new Guid("248e2800-a793-4724-abaa-23a6de1be090");
    private static readonly Guid ATTR_GENERIC_ML =
        new Guid("b71b0d41-1088-422f-a27c-0250b7d3a988");
    private static readonly Guid ATTR_NPU =
        new Guid("d46140c4-add7-451b-9e56-06fe8c3b58ed");

    private const uint PROP_IS_HARDWARE       = 11;
    private const uint PROP_HARDWARE_ID       = 3;
    private const uint PROP_HARDWARE_ID_PARTS = 14;

    [DllImport("dxcore.dll", PreserveSig = true)]
    private static extern int DXCoreCreateAdapterFactory(
        ref Guid riid, out IntPtr ppvFactory);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int CreateAdapterListDelegate(
        IntPtr pThis, uint numAttributes, IntPtr filterAttributes,
        ref Guid riid, out IntPtr ppvAdapterList);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate uint GetAdapterCountDelegate(IntPtr pThis);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int GetAdapterDelegate(
        IntPtr pThis, uint index, ref Guid riid, out IntPtr ppvAdapter);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    [return: MarshalAs(UnmanagedType.I1)]
    private delegate bool IsPropertySupportedDelegate(IntPtr pThis, uint property);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int GetPropertyDelegate(
        IntPtr pThis, uint property, UIntPtr bufferSize, IntPtr outputBuffer);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate uint ReleaseDelegate(IntPtr pThis);

    private static T GetVtblFunc<T>(IntPtr pThis, int index) where T : class
    {
        IntPtr vtbl    = Marshal.ReadIntPtr(pThis);
        IntPtr funcPtr = Marshal.ReadIntPtr(vtbl, index * IntPtr.Size);
        return (T)(object)Marshal.GetDelegateForFunctionPointer(funcPtr, typeof(T));
    }

    private static void Release(IntPtr pThis)
    {
        if (pThis != IntPtr.Zero)
            try { GetVtblFunc<ReleaseDelegate>(pThis, 2)(pThis); } catch { }
    }

    public static int GetHtpArchitecture()
    {
        IntPtr factory = IntPtr.Zero;
        try
        {
            Guid fIid = IID_IDXCoreAdapterFactory;
            if (DXCoreCreateAdapterFactory(ref fIid, out factory) < 0
                    || factory == IntPtr.Zero)
                return -1;
        }
        catch { return -1; }

        int result = -1;
        try
        {
            var createAdapterList = GetVtblFunc<CreateAdapterListDelegate>(factory, 3);

            Guid[] attrs = new Guid[] { ATTR_CORE_COMPUTE, ATTR_GENERIC_ML, ATTR_NPU };

            foreach (Guid attr in attrs)
            {
                Guid   attrCopy = attr;
                IntPtr attrPtr  = Marshal.AllocHGlobal(16);
                try
                {
                    Marshal.StructureToPtr(attrCopy, attrPtr, false);

                    Guid   listIid     = IID_IDXCoreAdapterList;
                    IntPtr adapterList = IntPtr.Zero;
                    if (createAdapterList(factory, 1, attrPtr,
                                         ref listIid, out adapterList) < 0
                            || adapterList == IntPtr.Zero)
                        continue;

                    try
                    {
                        var  getCount   = GetVtblFunc<GetAdapterCountDelegate>(adapterList, 4);
                        var  getAdapter = GetVtblFunc<GetAdapterDelegate>(adapterList, 3);
                        uint count      = getCount(adapterList);

                        for (uint i = 0; i < count; i++)
                        {
                            Guid   adpIid  = IID_IDXCoreAdapter;
                            IntPtr adapter = IntPtr.Zero;
                            if (getAdapter(adapterList, i, ref adpIid, out adapter) < 0
                                    || adapter == IntPtr.Zero)
                                continue;

                            try
                            {
                                var isPropertySupported =
                                    GetVtblFunc<IsPropertySupportedDelegate>(adapter, 5);
                                var getProperty =
                                    GetVtblFunc<GetPropertyDelegate>(adapter, 6);

                                if (isPropertySupported(adapter, PROP_IS_HARDWARE))
                                {
                                    IntPtr isHwPtr = Marshal.AllocHGlobal(1);
                                    try
                                    {
                                        Marshal.WriteByte(isHwPtr, 0);
                                        int  hr   = getProperty(adapter, PROP_IS_HARDWARE,
                                                                (UIntPtr)1, isHwPtr);
                                        bool isHw = Marshal.ReadByte(isHwPtr) != 0;
                                        if (hr < 0 || !isHw) continue;
                                    }
                                    finally { Marshal.FreeHGlobal(isHwPtr); }
                                }

                                uint deviceId = 0;

                                if (isPropertySupported(adapter, PROP_HARDWARE_ID))
                                {
                                    IntPtr hwPtr = Marshal.AllocHGlobal(16);
                                    try
                                    {
                                        if (getProperty(adapter, PROP_HARDWARE_ID,
                                                        (UIntPtr)16, hwPtr) == 0)
                                            deviceId = (uint)Marshal.ReadInt32(hwPtr, 4);
                                    }
                                    finally { Marshal.FreeHGlobal(hwPtr); }
                                }
                                else if (isPropertySupported(adapter, PROP_HARDWARE_ID_PARTS))
                                {
                                    IntPtr hwPtr = Marshal.AllocHGlobal(28);
                                    try
                                    {
                                        if (getProperty(adapter, PROP_HARDWARE_ID_PARTS,
                                                        (UIntPtr)28, hwPtr) == 0)
                                            deviceId = (uint)Marshal.ReadInt32(hwPtr, 4);
                                    }
                                    finally { Marshal.FreeHGlobal(hwPtr); }
                                }

                                if (DeviceIdToHtp.ContainsKey(deviceId))
                                {
                                    result = DeviceIdToHtp[deviceId];
                                    break;
                                }
                            }
                            finally { Release(adapter); }
                        }
                    }
                    finally { Release(adapterList); }
                }
                finally { Marshal.FreeHGlobal(attrPtr); }

                if (result >= 0) break;
            }
        }
        finally { Release(factory); }

        return result;
    }
}
"@

if (-not ([System.Management.Automation.PSTypeName]'DXCoreDetect').Type) {
    Add-Type -TypeDefinition $dxcoreSource -Language CSharp
}

try {
    $arch = [DXCoreDetect]::GetHtpArchitecture()
    if ($arch -ge 0) {
        Write-Output $arch
        exit 0
    }
} catch {}

exit 1
