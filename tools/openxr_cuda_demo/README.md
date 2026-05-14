# OpenXR CUDA Demo

This is a small Windows-only validation app for the headset display path:

```text
CUDA kernel -> D3D11 texture -> OpenXR D3D11 swapchain -> SteamVR/OpenXR runtime -> headset
```

It does not load HorizonGS, PyTorch, or any scene model. The rendered image is a procedural per-eye CUDA pattern with a grid, motion, and pose-dependent color shift.

## Requirements

- Windows
- Visual Studio C++ toolchain
- CUDA Toolkit
- OpenXR SDK loader and headers
- SteamVR set as the active OpenXR runtime
- Quest 3 connected through Link or Air Link, with SteamVR able to see the headset

The OpenXR SDK can be provided by vcpkg or a normal SDK install. CMake first tries `find_package(OpenXR CONFIG)`, then falls back to `OPENXR_INCLUDE_DIR` and `OPENXR_LOADER_LIBRARY`.

With vcpkg, install the loader and configure with the vcpkg toolchain:

```powershell
vcpkg install openxr-loader:x64-windows
cmake -S tools/openxr_cuda_demo -B build/openxr_cuda_demo -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
```

## Build

From a Visual Studio developer PowerShell:

```powershell
cmake -S tools/openxr_cuda_demo -B build/openxr_cuda_demo -G "Visual Studio 17 2022" -A x64
cmake --build build/openxr_cuda_demo --config Release
```

If the Visual Studio generator cannot resolve CUDA but `nvcc` is on disk, use NMake from the same developer PowerShell:

```powershell
cmake -S tools/openxr_cuda_demo -B build/openxr_cuda_demo_nmake -G "NMake Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
cmake --build build/openxr_cuda_demo_nmake
```

If OpenXR is not found automatically:

```powershell
cmake -S tools/openxr_cuda_demo -B build/openxr_cuda_demo -G "Visual Studio 17 2022" -A x64 `
  -DOPENXR_INCLUDE_DIR=C:\path\to\OpenXR-SDK\include `
  -DOPENXR_LOADER_LIBRARY=C:\path\to\openxr_loader.lib
```

If Visual Studio reports that `CudaToolkitDir` is empty, point CMake/MSBuild at the CUDA Toolkit explicitly:

```powershell
cmake -S tools/openxr_cuda_demo -B build/openxr_cuda_demo -G "Visual Studio 17 2022" -A x64 `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```

You can also set the `CudaToolkitDir` environment variable before configuring.

## Run

Start SteamVR and connect the Quest 3 first, then run:

```powershell
.\build\openxr_cuda_demo\Release\openxr_cuda_demo.exe
```

Useful flags:

```powershell
.\openxr_cuda_demo.exe --frames 600
.\openxr_cuda_demo.exe --reference-space stage
.\openxr_cuda_demo.exe --quiet
```

`--frames -1` runs until the OpenXR runtime exits the session.

## What This Validates

- OpenXR runtime discovery
- `XR_KHR_D3D11_enable`
- correct D3D11 adapter selection from the OpenXR runtime LUID
- OpenXR stereo swapchain creation
- CUDA/D3D11 interop on the selected adapter
- copying CUDA-rendered D3D11 textures into OpenXR swapchain images
- `xrEndFrame` projection layer submission

Once this app shows animated stereo content in the headset, the remaining integration work is replacing the procedural CUDA renderer with a reusable frame source, such as HorizonGS or another renderer.

## Troubleshooting

- `xrCreateInstance failed: XR_ERROR_API_VERSION_UNSUPPORTED (-4)`: the active runtime rejected the requested OpenXR API version. The demo requests OpenXR 1.0 for compatibility; rebuild if this changed after an SDK/header update.
- `xrCreateInstance failed: XR_ERROR_RUNTIME_UNAVAILABLE`: no active OpenXR runtime is registered. Set SteamVR as the active OpenXR runtime.
- `Active OpenXR runtime does not advertise XR_KHR_D3D11_enable`: set SteamVR as the active OpenXR runtime.
- `OpenXR-selected D3D11 adapter is not visible to CUDA`: SteamVR selected an adapter that CUDA cannot use. Make sure SteamVR and the headset session are running on the NVIDIA GPU.
- `OpenXR runtime did not advertise a CUDA-compatible 8-bit RGBA/BGRA swapchain format`: the active runtime did not list any of the demo's supported color swapchain formats. The demo accepts `DXGI_FORMAT_R8G8B8A8_UNORM`, `DXGI_FORMAT_R8G8B8A8_UNORM_SRGB`, `DXGI_FORMAT_B8G8R8A8_UNORM`, and `DXGI_FORMAT_B8G8R8A8_UNORM_SRGB`.
