cmake_minimum_required(VERSION 3.20)

project(openxr_cuda_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenXR CONFIG QUIET)

add_executable(openxr_cuda_demo
    src/main.cpp
    src/cuda_renderer.cu
)

target_compile_definitions(openxr_cuda_demo PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    XR_USE_PLATFORM_WIN32
    XR_USE_GRAPHICS_API_D3D11
)

target_link_libraries(openxr_cuda_demo PRIVATE
    CUDA::cudart
    d3d11
    dxgi
    dxguid
    Ws2_32
)

if(TARGET OpenXR::openxr_loader)
    target_link_libraries(openxr_cuda_demo PRIVATE OpenXR::openxr_loader)
elseif(TARGET openxr_loader)
    target_link_libraries(openxr_cuda_demo PRIVATE openxr_loader)
else()
    find_path(OPENXR_INCLUDE_DIR NAMES openxr/openxr.h REQUIRED)
    find_library(OPENXR_LOADER_LIBRARY NAMES openxr_loader REQUIRED)
    target_include_directories(openxr_cuda_demo PRIVATE "${OPENXR_INCLUDE_DIR}")
    target_link_libraries(openxr_cuda_demo PRIVATE "${OPENXR_LOADER_LIBRARY}")
endif()

set_source_files_properties(src/cuda_renderer.cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
