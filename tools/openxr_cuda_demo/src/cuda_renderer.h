#pragma once

#include <array>
#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>
#include <d3d11.h>
#include <dxgiformat.h>
#include <wrl/client.h>

class CudaRenderer {
public:
    CudaRenderer() = default;
    ~CudaRenderer();

    CudaRenderer(const CudaRenderer&) = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;

    bool initialize(
        ID3D11Device* device,
        DXGI_FORMAT format,
        const std::array<uint32_t, 2>& widths,
        const std::array<uint32_t, 2>& heights,
        std::string* error);

    void shutdown();

    bool renderEye(
        int eye,
        float seconds,
        const std::array<float, 3>& position,
        const std::array<float, 4>& orientationXyzw,
        std::string* error);

    ID3D11Texture2D* texture(int eye) const;

private:
    struct EyeTarget {
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        cudaGraphicsResource_t cudaResource = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    std::array<EyeTarget, 2> eyes_;
    DXGI_FORMAT format_ = DXGI_FORMAT_UNKNOWN;
    bool writeBgra_ = false;
    bool initialized_ = false;
};
