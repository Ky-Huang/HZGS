#include "cuda_renderer.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <cuda.h>
#include <cuda_d3d11_interop.h>
#include <cuda_runtime.h>
#include <surface_functions.h>

namespace {

std::string cudaErrorString(cudaError_t result, const char* call)
{
    std::ostringstream oss;
    oss << call << " failed: " << cudaGetErrorString(result) << " (" << static_cast<int>(result) << ")";
    return oss.str();
}

bool checkCuda(cudaError_t result, const char* call, std::string* error)
{
    if (result == cudaSuccess) {
        return true;
    }
    if (error) {
        *error = cudaErrorString(result, call);
    }
    return false;
}

__device__ float frac(float v)
{
    return v - floorf(v);
}

__device__ float clamp01(float v)
{
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ unsigned char toByte(float v)
{
    return static_cast<unsigned char>(clamp01(v) * 255.0f + 0.5f);
}

DXGI_FORMAT cudaTextureFormatForSwapchain(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

bool formatUsesBgraOrder(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_B8G8R8A8_UNORM;
}

__global__ void renderDemoKernel(
    cudaSurfaceObject_t surface,
    int width,
    int height,
    int eye,
    float seconds,
    float px,
    float py,
    float pz,
    float qx,
    float qy,
    float qz,
    float qw,
    bool writeBgra)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
    const float cx = u * 2.0f - 1.0f;
    const float cy = v * 2.0f - 1.0f;
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float r = sqrtf(cx * cx * aspect * aspect + cy * cy);

    const float wave = 0.5f + 0.5f * sinf(18.0f * r - seconds * 3.0f);
    const float scan = 0.5f + 0.5f * sinf((u + seconds * 0.07f + eye * 0.11f) * 36.0f);
    const float gridX = fabsf(frac(u * 24.0f) - 0.5f);
    const float gridY = fabsf(frac(v * 24.0f) - 0.5f);
    const float grid = (gridX < 0.018f || gridY < 0.018f) ? 1.0f : 0.0f;
    const float ring = fabsf(r - (0.38f + 0.08f * sinf(seconds * 1.7f))) < 0.012f ? 1.0f : 0.0f;

    const float poseShift = 0.5f + 0.5f * sinf(px * 2.0f + py * 0.7f + pz * 1.3f + qx + qy + qz + qw);
    const float eyeTint = eye == 0 ? -0.18f : 0.18f;

    float red = 0.10f + 0.55f * u + 0.20f * wave + 0.25f * ring + eyeTint;
    float green = 0.12f + 0.45f * (1.0f - v) + 0.20f * scan + 0.18f * grid;
    float blue = 0.20f + 0.40f * poseShift + 0.20f * wave - eyeTint;

    if (eye == 0 && x < width / 18 && y < height / 6) {
        red = 0.95f;
        green = 0.18f;
        blue = 0.10f;
    }
    if (eye == 1 && x > width - width / 18 && y < height / 6) {
        red = 0.10f;
        green = 0.55f;
        blue = 0.95f;
    }

    const uchar4 color = writeBgra
        ? make_uchar4(toByte(blue), toByte(green), toByte(red), 255)
        : make_uchar4(toByte(red), toByte(green), toByte(blue), 255);
    surf2Dwrite(color, surface, x * static_cast<int>(sizeof(uchar4)), y);
}

} // namespace

CudaRenderer::~CudaRenderer()
{
    shutdown();
}

bool CudaRenderer::initialize(
    ID3D11Device* device,
    DXGI_FORMAT format,
    const std::array<uint32_t, 2>& widths,
    const std::array<uint32_t, 2>& heights,
    std::string* error)
{
    if (!device) {
        if (error) {
            *error = "CudaRenderer::initialize received a null D3D11 device.";
        }
        return false;
    }
    const DXGI_FORMAT textureFormat = cudaTextureFormatForSwapchain(format);
    if (textureFormat == DXGI_FORMAT_UNKNOWN) {
        if (error) {
            *error = "CUDA demo supports R8G8B8A8/B8G8R8A8 UNORM and SRGB swapchains.";
        }
        return false;
    }

    shutdown();
    format_ = textureFormat;
    writeBgra_ = formatUsesBgraOrder(format_);

    unsigned int cudaDeviceCount = 0;
    int cudaDevice = -1;
    if (!checkCuda(
            cudaD3D11GetDevices(&cudaDeviceCount, &cudaDevice, 1, device, cudaD3D11DeviceListAll),
            "cudaD3D11GetDevices",
            error)) {
        return false;
    }
    if (cudaDeviceCount == 0 || cudaDevice < 0) {
        if (error) {
            *error = "OpenXR-selected D3D11 adapter is not visible to CUDA.";
        }
        return false;
    }
    if (!checkCuda(cudaSetDevice(cudaDevice), "cudaSetDevice", error)) {
        return false;
    }

    for (int eye = 0; eye < 2; ++eye) {
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = widths[eye];
        desc.Height = heights[eye];
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = format_;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

        HRESULT hr = device->CreateTexture2D(&desc, nullptr, eyes_[eye].texture.GetAddressOf());
        if (FAILED(hr)) {
            if (error) {
                std::ostringstream oss;
                oss << "CreateTexture2D for CUDA eye " << eye << " failed with HRESULT 0x"
                    << std::hex << static_cast<unsigned long>(hr) << ".";
                *error = oss.str();
            }
            shutdown();
            return false;
        }

        eyes_[eye].width = widths[eye];
        eyes_[eye].height = heights[eye];

        cudaError_t registerResult = cudaGraphicsD3D11RegisterResource(
            &eyes_[eye].cudaResource,
            eyes_[eye].texture.Get(),
            cudaGraphicsRegisterFlagsSurfaceLoadStore);
        if (!checkCuda(registerResult, "cudaGraphicsD3D11RegisterResource", error)) {
            shutdown();
            return false;
        }
    }

    initialized_ = true;
    return true;
}

void CudaRenderer::shutdown()
{
    for (auto& eye : eyes_) {
        if (eye.cudaResource) {
            cudaGraphicsUnregisterResource(eye.cudaResource);
            eye.cudaResource = nullptr;
        }
        eye.texture.Reset();
        eye.width = 0;
        eye.height = 0;
    }
    initialized_ = false;
    format_ = DXGI_FORMAT_UNKNOWN;
    writeBgra_ = false;
}

bool CudaRenderer::renderEye(
    int eye,
    float seconds,
    const std::array<float, 3>& position,
    const std::array<float, 4>& orientationXyzw,
    std::string* error)
{
    if (!initialized_ || eye < 0 || eye >= 2 || !eyes_[eye].cudaResource) {
        if (error) {
            *error = "CudaRenderer::renderEye called before initialization or with an invalid eye index.";
        }
        return false;
    }

    cudaGraphicsResource_t resource = eyes_[eye].cudaResource;
    if (!checkCuda(cudaGraphicsMapResources(1, &resource, 0), "cudaGraphicsMapResources", error)) {
        return false;
    }

    cudaArray_t array = nullptr;
    cudaError_t mappedArrayResult = cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0);
    if (mappedArrayResult != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &resource, 0);
        return checkCuda(mappedArrayResult, "cudaGraphicsSubResourceGetMappedArray", error);
    }

    cudaResourceDesc resourceDesc = {};
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = array;

    cudaSurfaceObject_t surface = 0;
    cudaError_t surfaceResult = cudaCreateSurfaceObject(&surface, &resourceDesc);
    if (surfaceResult != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &resource, 0);
        return checkCuda(surfaceResult, "cudaCreateSurfaceObject", error);
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (eyes_[eye].width + block.x - 1) / block.x,
        (eyes_[eye].height + block.y - 1) / block.y);

    renderDemoKernel<<<grid, block>>>(
        surface,
        static_cast<int>(eyes_[eye].width),
        static_cast<int>(eyes_[eye].height),
        eye,
        seconds,
        position[0],
        position[1],
        position[2],
        orientationXyzw[0],
        orientationXyzw[1],
        orientationXyzw[2],
        orientationXyzw[3],
        writeBgra_);

    cudaError_t kernelResult = cudaGetLastError();
    cudaError_t syncResult = cudaDeviceSynchronize();

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &resource, 0);

    if (!checkCuda(kernelResult, "renderDemoKernel launch", error)) {
        return false;
    }
    if (!checkCuda(syncResult, "cudaDeviceSynchronize", error)) {
        return false;
    }
    return true;
}

ID3D11Texture2D* CudaRenderer::texture(int eye) const
{
    if (eye < 0 || eye >= 2) {
        return nullptr;
    }
    return eyes_[eye].texture.Get();
}
