#include "cuda_renderer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <winsock2.h>
#include <ws2tcpip.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace {

constexpr XrViewConfigurationType kViewConfig = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
constexpr std::array<DXGI_FORMAT, 4> kPreferredColorFormats = {
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
    DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_B8G8R8A8_UNORM,
};

struct Options {
    int frameLimit = -1;
    XrReferenceSpaceType referenceSpace = XR_REFERENCE_SPACE_TYPE_LOCAL;
    bool quiet = false;
    bool poseSocketEnabled = false;
    std::string poseSocketHost = "127.0.0.1";
    int poseSocketPort = 6110;
    int poseSocketRetrySeconds = 30;
    bool poseSocketOptional = false;
    int poseSocketStartDelaySeconds = 0;
    bool hgsStreamEnabled = false;
    float swapchainScale = 1.0f;
};

struct Swapchain {
    XrSwapchain handle = XR_NULL_HANDLE;
    int32_t width = 0;
    int32_t height = 0;
    std::vector<XrSwapchainImageD3D11KHR> images;
};

class DemoError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::string hrToString(HRESULT hr)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << static_cast<unsigned long>(hr);
    return oss.str();
}

void checkHr(HRESULT hr, const char* call)
{
    if (FAILED(hr)) {
        std::ostringstream oss;
        oss << call << " failed with HRESULT " << hrToString(hr);
        throw DemoError(oss.str());
    }
}

void checkXr(XrInstance instance, XrResult result, const char* call)
{
    if (XR_SUCCEEDED(result)) {
        return;
    }

    char buffer[XR_MAX_RESULT_STRING_SIZE] = {};
    if (instance != XR_NULL_HANDLE) {
        xrResultToString(instance, result, buffer);
    } else {
        switch (result) {
        case XR_ERROR_API_VERSION_UNSUPPORTED:
            std::snprintf(buffer, sizeof(buffer), "XR_ERROR_API_VERSION_UNSUPPORTED (%d)", static_cast<int>(result));
            break;
        case XR_ERROR_RUNTIME_UNAVAILABLE:
            std::snprintf(buffer, sizeof(buffer), "XR_ERROR_RUNTIME_UNAVAILABLE (%d)", static_cast<int>(result));
            break;
        case XR_ERROR_EXTENSION_NOT_PRESENT:
            std::snprintf(buffer, sizeof(buffer), "XR_ERROR_EXTENSION_NOT_PRESENT (%d)", static_cast<int>(result));
            break;
        default:
            std::snprintf(buffer, sizeof(buffer), "%d", static_cast<int>(result));
            break;
        }
    }

    std::ostringstream oss;
    oss << call << " failed: " << buffer;
    throw DemoError(oss.str());
}

bool sameLuid(const LUID& a, const LUID& b)
{
    return a.LowPart == b.LowPart && a.HighPart == b.HighPart;
}

std::string luidToString(const LUID& luid)
{
    std::ostringstream oss;
    oss << std::hex << luid.HighPart << ":" << luid.LowPart;
    return oss.str();
}

std::string dxgiFormatName(int64_t format)
{
    switch (static_cast<DXGI_FORMAT>(format)) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return "DXGI_FORMAT_R8G8B8A8_UNORM";
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB";
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        return "DXGI_FORMAT_B8G8R8A8_UNORM";
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return "DXGI_FORMAT_B8G8R8A8_UNORM_SRGB";
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return "DXGI_FORMAT_R10G10B10A2_UNORM";
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return "DXGI_FORMAT_R16G16B16A16_FLOAT";
    default: {
        std::ostringstream oss;
        oss << "DXGI_FORMAT(" << format << ")";
        return oss.str();
    }
    }
}

std::string dxgiFormatList(const std::vector<int64_t>& formats)
{
    std::ostringstream oss;
    for (size_t i = 0; i < formats.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << dxgiFormatName(formats[i]);
    }
    return oss.str();
}

std::string wsaErrorToString(int error)
{
    std::ostringstream oss;
    oss << "WSA error " << error;
    return oss.str();
}

void validateTcpPort(int port, const char* optionName)
{
    if (port <= 0 || port > 65535) {
        std::ostringstream oss;
        oss << optionName << " must be in range 1..65535.";
        throw DemoError(oss.str());
    }
}

DXGI_FORMAT chooseColorFormat(const std::vector<int64_t>& runtimeFormats)
{
    for (DXGI_FORMAT preferred : kPreferredColorFormats) {
        if (std::find(runtimeFormats.begin(), runtimeFormats.end(), static_cast<int64_t>(preferred)) != runtimeFormats.end()) {
            return preferred;
        }
    }
    return DXGI_FORMAT_UNKNOWN;
}

std::wstring referenceSpaceName(XrReferenceSpaceType type)
{
    switch (type) {
    case XR_REFERENCE_SPACE_TYPE_LOCAL:
        return L"local";
    case XR_REFERENCE_SPACE_TYPE_STAGE:
        return L"stage";
    case XR_REFERENCE_SPACE_TYPE_VIEW:
        return L"view";
    default:
        return L"unknown";
    }
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) {
            options.frameLimit = std::stoi(argv[++i]);
        } else if (arg == "--reference-space" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "local") {
                options.referenceSpace = XR_REFERENCE_SPACE_TYPE_LOCAL;
            } else if (value == "stage") {
                options.referenceSpace = XR_REFERENCE_SPACE_TYPE_STAGE;
            } else if (value == "view") {
                options.referenceSpace = XR_REFERENCE_SPACE_TYPE_VIEW;
            } else {
                throw DemoError("Unsupported --reference-space value. Use local, stage, or view.");
            }
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--pose-socket") {
            options.poseSocketEnabled = true;
        } else if (arg == "--pose-socket-host" && i + 1 < argc) {
            options.poseSocketHost = argv[++i];
            options.poseSocketEnabled = true;
        } else if (arg == "--pose-socket-port" && i + 1 < argc) {
            options.poseSocketPort = std::stoi(argv[++i]);
            validateTcpPort(options.poseSocketPort, "--pose-socket-port");
            options.poseSocketEnabled = true;
        } else if (arg == "--pose-socket-retry-seconds" && i + 1 < argc) {
            options.poseSocketRetrySeconds = std::max(0, std::stoi(argv[++i]));
            options.poseSocketEnabled = true;
        } else if (arg == "--pose-socket-optional") {
            options.poseSocketOptional = true;
            options.poseSocketEnabled = true;
        } else if (arg == "--pose-socket-start-delay-seconds" && i + 1 < argc) {
            options.poseSocketStartDelaySeconds = std::max(0, std::stoi(argv[++i]));
            options.poseSocketEnabled = true;
        } else if (arg == "--hgs-stream") {
            options.hgsStreamEnabled = true;
            options.poseSocketEnabled = true;
        } else if (arg == "--swapchain-scale" && i + 1 < argc) {
            options.swapchainScale = std::stof(argv[++i]);
            if (options.swapchainScale <= 0.0f || options.swapchainScale > 1.0f) {
                throw DemoError("--swapchain-scale must be in range (0, 1].");
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: openxr_cuda_demo.exe [--frames N] [--reference-space local|stage|view] [--quiet]\n"
                << "                            [--pose-socket] [--pose-socket-host HOST] [--pose-socket-port PORT]\n"
                << "                            [--pose-socket-retry-seconds N] [--pose-socket-optional]\n"
                << "                            [--pose-socket-start-delay-seconds N]\n"
                << "                            [--hgs-stream] [--swapchain-scale SCALE]\n";
            std::exit(0);
        } else {
            std::ostringstream oss;
            oss << "Unknown argument: " << arg;
            throw DemoError(oss.str());
        }
    }
    return options;
}

class PoseSocketClient {
public:
    ~PoseSocketClient()
    {
        close();
    }

    bool isConnected() const
    {
        return socket_ != INVALID_SOCKET;
    }

    void connectTo(const std::string& host, int port)
    {
        if (isConnected()) {
            return;
        }
        validateTcpPort(port, "--pose-socket-port");

        WSADATA wsaData = {};
        const int startupResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (startupResult != 0) {
            throw DemoError("WSAStartup failed: " + wsaErrorToString(startupResult));
        }
        wsaStarted_ = true;

        addrinfo hints = {};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        addrinfo* addresses = nullptr;
        const std::string portText = std::to_string(port);
        const int getAddrResult = getaddrinfo(host.c_str(), portText.c_str(), &hints, &addresses);
        if (getAddrResult != 0) {
            throw DemoError("getaddrinfo failed for pose socket endpoint: " + wsaErrorToString(getAddrResult));
        }

        SOCKET connectedSocket = INVALID_SOCKET;
        int lastError = 0;
        for (addrinfo* candidate = addresses; candidate != nullptr; candidate = candidate->ai_next) {
            SOCKET candidateSocket = socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
            if (candidateSocket == INVALID_SOCKET) {
                lastError = WSAGetLastError();
                continue;
            }

            if (::connect(candidateSocket, candidate->ai_addr, static_cast<int>(candidate->ai_addrlen)) == 0) {
                connectedSocket = candidateSocket;
                break;
            }

            lastError = WSAGetLastError();
            closesocket(candidateSocket);
        }
        freeaddrinfo(addresses);

        if (connectedSocket == INVALID_SOCKET) {
            throw DemoError("Could not connect pose socket: " + wsaErrorToString(lastError));
        }

        socket_ = connectedSocket;
    }

    void sendFrame(
        int frameId,
        XrTime predictedDisplayTime,
        const std::vector<XrView>& views,
        const std::vector<Swapchain>& swapchains)
    {
        if (!isConnected()) {
            return;
        }
        if (views.size() < 2 || swapchains.size() < 2) {
            throw DemoError("Pose socket frame requires two views and two swapchains.");
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(9);
        oss << "{\"frame_id\":" << frameId
            << ",\"timestamp_ns\":" << static_cast<long long>(predictedDisplayTime)
            << ",\"views\":{";
        appendEyeJson(oss, "left", views[0], swapchains[0]);
        oss << ",";
        appendEyeJson(oss, "right", views[1], swapchains[1]);
        oss << "}}\n";
        sendAll(oss.str());
    }

    void sendEos() noexcept
    {
        if (!isConnected() || eosSent_) {
            return;
        }
        try {
            sendAll("{\"type\":\"eos\"}\n");
            eosSent_ = true;
        } catch (...) {
        }
    }

    void close() noexcept
    {
        if (socket_ != INVALID_SOCKET) {
            closesocket(socket_);
            socket_ = INVALID_SOCKET;
        }
        if (wsaStarted_) {
            WSACleanup();
            wsaStarted_ = false;
        }
    }

private:
    static void appendEyeJson(std::ostringstream& oss, const char* name, const XrView& view, const Swapchain& swapchain)
    {
        oss << "\"" << name << "\":{"
            << "\"pose\":{"
            << "\"position\":[" << view.pose.position.x << "," << view.pose.position.y << "," << view.pose.position.z << "],"
            << "\"orientation_xyzw\":["
            << view.pose.orientation.x << ","
            << view.pose.orientation.y << ","
            << view.pose.orientation.z << ","
            << view.pose.orientation.w << "]"
            << "},"
            << "\"fov\":{"
            << "\"angle_left\":" << view.fov.angleLeft << ","
            << "\"angle_right\":" << view.fov.angleRight << ","
            << "\"angle_up\":" << view.fov.angleUp << ","
            << "\"angle_down\":" << view.fov.angleDown
            << "},"
            << "\"image_rect\":{"
            << "\"width\":" << swapchain.width << ","
            << "\"height\":" << swapchain.height
            << "}"
            << "}";
    }

    void sendAll(const std::string& payload)
    {
        size_t sentTotal = 0;
        while (sentTotal < payload.size()) {
            const size_t remaining = payload.size() - sentTotal;
            const int chunkSize = static_cast<int>(std::min<size_t>(remaining, 64 * 1024));
            const int sent = send(socket_, payload.data() + sentTotal, chunkSize, 0);
            if (sent == SOCKET_ERROR) {
                throw DemoError("Pose socket send failed: " + wsaErrorToString(WSAGetLastError()));
            }
            if (sent == 0) {
                throw DemoError("Pose socket closed while sending.");
            }
            sentTotal += static_cast<size_t>(sent);
        }
    }

    SOCKET socket_ = INVALID_SOCKET;
    bool wsaStarted_ = false;
    bool eosSent_ = false;
};

struct StreamImageFrame {
    int frameId = -1;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> left;
    std::vector<uint8_t> right;
};

class HorizonGsStreamClient {
public:
    HorizonGsStreamClient() = default;
    ~HorizonGsStreamClient()
    {
        stop();
    }

    HorizonGsStreamClient(const HorizonGsStreamClient&) = delete;
    HorizonGsStreamClient& operator=(const HorizonGsStreamClient&) = delete;

    void start(const std::string& host, int port, int retrySeconds, bool quiet)
    {
        if (running_) {
            return;
        }
        validateTcpPort(port, "--pose-socket-port");
        host_ = host;
        port_ = port;
        retrySeconds_ = std::max(0, retrySeconds);
        quiet_ = quiet;
        running_ = true;
        worker_ = std::thread(&HorizonGsStreamClient::workerLoop, this);
    }

    void stop() noexcept
    {
        if (!running_ && !worker_.joinable()) {
            return;
        }
        running_ = false;
        requestCv_.notify_all();
        closeSocket();
        if (worker_.joinable()) {
            worker_.join();
        }
        cleanupWinsock();
    }

    void submitFrame(
        int frameId,
        XrTime predictedDisplayTime,
        const std::vector<XrView>& views,
        const std::vector<Swapchain>& swapchains)
    {
        if (!running_ || views.size() < 2 || swapchains.size() < 2) {
            return;
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(9);
        oss << "{\"frame_id\":" << frameId
            << ",\"timestamp_ns\":" << static_cast<long long>(predictedDisplayTime)
            << ",\"views\":{";
        appendEyeJson(oss, "left", views[0], swapchains[0]);
        oss << ",";
        appendEyeJson(oss, "right", views[1], swapchains[1]);
        oss << "}}\n";

        {
            std::lock_guard<std::mutex> lock(requestMutex_);
            pendingRequest_ = oss.str();
            ++requestSequence_;
        }
        requestCv_.notify_one();
    }

    std::shared_ptr<const StreamImageFrame> latestFrame() const
    {
        std::lock_guard<std::mutex> lock(frameMutex_);
        return latestFrame_;
    }

    std::string statusMessage() const
    {
        std::lock_guard<std::mutex> lock(statusMutex_);
        return statusMessage_;
    }

private:
    static void appendEyeJson(std::ostringstream& oss, const char* name, const XrView& view, const Swapchain& swapchain)
    {
        oss << "\"" << name << "\":{"
            << "\"pose\":{"
            << "\"position\":[" << view.pose.position.x << "," << view.pose.position.y << "," << view.pose.position.z << "],"
            << "\"orientation_xyzw\":["
            << view.pose.orientation.x << ","
            << view.pose.orientation.y << ","
            << view.pose.orientation.z << ","
            << view.pose.orientation.w << "]"
            << "},"
            << "\"fov\":{"
            << "\"angle_left\":" << view.fov.angleLeft << ","
            << "\"angle_right\":" << view.fov.angleRight << ","
            << "\"angle_up\":" << view.fov.angleUp << ","
            << "\"angle_down\":" << view.fov.angleDown
            << "},"
            << "\"image_rect\":{"
            << "\"width\":" << swapchain.width << ","
            << "\"height\":" << swapchain.height
            << "}"
            << "}";
    }

    void log(const std::string& message) const
    {
        if (!quiet_) {
            std::cout << message << std::endl;
        }
    }

    void setStatus(const std::string& message)
    {
        std::lock_guard<std::mutex> lock(statusMutex_);
        statusMessage_ = message;
    }

    void startupWinsock()
    {
        if (wsaStarted_) {
            return;
        }
        WSADATA wsaData = {};
        const int startupResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (startupResult != 0) {
            throw DemoError("WSAStartup failed: " + wsaErrorToString(startupResult));
        }
        wsaStarted_ = true;
    }

    void cleanupWinsock() noexcept
    {
        if (wsaStarted_) {
            WSACleanup();
            wsaStarted_ = false;
        }
    }

    void connectToServer()
    {
        startupWinsock();

        addrinfo hints = {};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        addrinfo* addresses = nullptr;
        const std::string portText = std::to_string(port_);
        const int getAddrResult = getaddrinfo(host_.c_str(), portText.c_str(), &hints, &addresses);
        if (getAddrResult != 0) {
            throw DemoError("getaddrinfo failed for HorizonGS stream endpoint: " + wsaErrorToString(getAddrResult));
        }

        SOCKET connectedSocket = INVALID_SOCKET;
        int lastError = 0;
        for (addrinfo* candidate = addresses; candidate != nullptr; candidate = candidate->ai_next) {
            SOCKET candidateSocket = socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
            if (candidateSocket == INVALID_SOCKET) {
                lastError = WSAGetLastError();
                continue;
            }

            if (::connect(candidateSocket, candidate->ai_addr, static_cast<int>(candidate->ai_addrlen)) == 0) {
                connectedSocket = candidateSocket;
                break;
            }

            lastError = WSAGetLastError();
            closesocket(candidateSocket);
        }
        freeaddrinfo(addresses);

        if (connectedSocket == INVALID_SOCKET) {
            throw DemoError("Could not connect HorizonGS stream socket: " + wsaErrorToString(lastError));
        }

        {
            std::lock_guard<std::mutex> lock(socketMutex_);
            socket_ = connectedSocket;
        }
    }

    void closeSocket() noexcept
    {
        std::lock_guard<std::mutex> lock(socketMutex_);
        if (socket_ != INVALID_SOCKET) {
            shutdown(socket_, SD_BOTH);
            closesocket(socket_);
            socket_ = INVALID_SOCKET;
        }
    }

    SOCKET currentSocket() const
    {
        std::lock_guard<std::mutex> lock(socketMutex_);
        return socket_;
    }

    static void sendAll(SOCKET socketHandle, const std::string& payload)
    {
        size_t sentTotal = 0;
        while (sentTotal < payload.size()) {
            const size_t remaining = payload.size() - sentTotal;
            const int chunkSize = static_cast<int>(std::min<size_t>(remaining, 64 * 1024));
            const int sent = send(socketHandle, payload.data() + sentTotal, chunkSize, 0);
            if (sent == SOCKET_ERROR) {
                throw DemoError("HorizonGS stream send failed: " + wsaErrorToString(WSAGetLastError()));
            }
            if (sent == 0) {
                throw DemoError("HorizonGS stream socket closed while sending.");
            }
            sentTotal += static_cast<size_t>(sent);
        }
    }

    static std::string receiveLine(SOCKET socketHandle)
    {
        std::string line;
        char ch = 0;
        while (true) {
            const int received = recv(socketHandle, &ch, 1, 0);
            if (received == SOCKET_ERROR) {
                throw DemoError("HorizonGS stream receive failed: " + wsaErrorToString(WSAGetLastError()));
            }
            if (received == 0) {
                throw DemoError("HorizonGS stream socket closed.");
            }
            if (ch == '\n') {
                return line;
            }
            line.push_back(ch);
        }
    }

    static void receiveExact(SOCKET socketHandle, std::vector<uint8_t>& output, size_t byteCount)
    {
        output.resize(byteCount);
        size_t receivedTotal = 0;
        while (receivedTotal < byteCount) {
            const size_t remaining = byteCount - receivedTotal;
            const int chunkSize = static_cast<int>(std::min<size_t>(remaining, 64 * 1024));
            const int received = recv(
                socketHandle,
                reinterpret_cast<char*>(output.data() + receivedTotal),
                chunkSize,
                0);
            if (received == SOCKET_ERROR) {
                throw DemoError("HorizonGS stream image receive failed: " + wsaErrorToString(WSAGetLastError()));
            }
            if (received == 0) {
                throw DemoError("HorizonGS stream socket closed while receiving image.");
            }
            receivedTotal += static_cast<size_t>(received);
        }
    }

    static StreamImageFrame receiveFrame(SOCKET socketHandle)
    {
        const std::string header = receiveLine(socketHandle);
        std::istringstream iss(header);
        std::string tag;
        size_t leftBytes = 0;
        size_t rightBytes = 0;
        StreamImageFrame frame;
        iss >> tag >> frame.frameId >> frame.width >> frame.height >> frame.channels >> leftBytes >> rightBytes;
        if (!iss || tag != "HGSFRAME") {
            throw DemoError("Invalid HorizonGS stream header: " + header);
        }
        if (frame.width <= 0 || frame.height <= 0 || frame.channels != 4) {
            throw DemoError("Invalid HorizonGS stream image shape.");
        }
        const size_t expectedBytes = static_cast<size_t>(frame.width) * static_cast<size_t>(frame.height) * 4;
        if (leftBytes != expectedBytes || rightBytes != expectedBytes) {
            throw DemoError("HorizonGS stream image byte count does not match shape.");
        }

        receiveExact(socketHandle, frame.left, leftBytes);
        receiveExact(socketHandle, frame.right, rightBytes);
        return frame;
    }

    bool waitForRequest(std::string& request, uint64_t& sequence)
    {
        std::unique_lock<std::mutex> lock(requestMutex_);
        requestCv_.wait(lock, [&]() {
            return !running_ || requestSequence_ != consumedRequestSequence_;
        });
        if (!running_) {
            return false;
        }
        request = pendingRequest_;
        sequence = requestSequence_;
        consumedRequestSequence_ = requestSequence_;
        return true;
    }

    bool connectWithRetry()
    {
        std::string lastError;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(retrySeconds_);
        while (running_) {
            try {
                connectToServer();
                std::ostringstream oss;
                oss << "HorizonGS stream connected to " << host_ << ":" << port_;
                log(oss.str());
                setStatus(oss.str());
                return true;
            } catch (const std::exception& e) {
                lastError = e.what();
                closeSocket();
            }

            if (std::chrono::steady_clock::now() >= deadline) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        std::ostringstream oss;
        oss << "HorizonGS stream connection failed after " << retrySeconds_ << "s. Last error: " << lastError;
        log(oss.str());
        setStatus(oss.str());
        return false;
    }

    void workerLoop()
    {
        try {
            if (!connectWithRetry()) {
                running_ = false;
                return;
            }

            while (running_) {
                std::string request;
                uint64_t sequence = 0;
                if (!waitForRequest(request, sequence)) {
                    break;
                }

                SOCKET socketHandle = currentSocket();
                if (socketHandle == INVALID_SOCKET) {
                    break;
                }
                sendAll(socketHandle, request);
                StreamImageFrame frame = receiveFrame(socketHandle);
                auto framePtr = std::make_shared<StreamImageFrame>(std::move(frame));
                {
                    std::lock_guard<std::mutex> lock(frameMutex_);
                    latestFrame_ = framePtr;
                    latestFrameSequence_ = sequence;
                }
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "HorizonGS stream worker stopped: " << e.what();
            log(oss.str());
            setStatus(oss.str());
        }

        closeSocket();
        running_ = false;
    }

    std::string host_ = "127.0.0.1";
    int port_ = 6110;
    int retrySeconds_ = 30;
    bool quiet_ = false;
    std::atomic<bool> running_{false};
    std::thread worker_;

    mutable std::mutex socketMutex_;
    SOCKET socket_ = INVALID_SOCKET;
    bool wsaStarted_ = false;

    mutable std::mutex requestMutex_;
    std::condition_variable requestCv_;
    std::string pendingRequest_;
    uint64_t requestSequence_ = 0;
    uint64_t consumedRequestSequence_ = 0;

    mutable std::mutex frameMutex_;
    std::shared_ptr<const StreamImageFrame> latestFrame_;
    uint64_t latestFrameSequence_ = 0;

    mutable std::mutex statusMutex_;
    std::string statusMessage_;
};

const char* sessionStateName(XrSessionState state)
{
    switch (state) {
    case XR_SESSION_STATE_UNKNOWN:
        return "UNKNOWN";
    case XR_SESSION_STATE_IDLE:
        return "IDLE";
    case XR_SESSION_STATE_READY:
        return "READY";
    case XR_SESSION_STATE_SYNCHRONIZED:
        return "SYNCHRONIZED";
    case XR_SESSION_STATE_VISIBLE:
        return "VISIBLE";
    case XR_SESSION_STATE_FOCUSED:
        return "FOCUSED";
    case XR_SESSION_STATE_STOPPING:
        return "STOPPING";
    case XR_SESSION_STATE_LOSS_PENDING:
        return "LOSS_PENDING";
    case XR_SESSION_STATE_EXITING:
        return "EXITING";
    default:
        return "OTHER";
    }
}

bool runtimeSupportsD3D11()
{
    uint32_t extensionCount = 0;
    XrResult result = xrEnumerateInstanceExtensionProperties(nullptr, 0, &extensionCount, nullptr);
    if (XR_FAILED(result)) {
        return false;
    }

    std::vector<XrExtensionProperties> extensions(extensionCount);
    for (auto& extension : extensions) {
        extension.type = XR_TYPE_EXTENSION_PROPERTIES;
    }
    result = xrEnumerateInstanceExtensionProperties(nullptr, extensionCount, &extensionCount, extensions.data());
    if (XR_FAILED(result)) {
        return false;
    }

    return std::any_of(extensions.begin(), extensions.end(), [](const XrExtensionProperties& extension) {
        return std::string(extension.extensionName) == XR_KHR_D3D11_ENABLE_EXTENSION_NAME;
    });
}

class OpenXrCudaDemo {
public:
    explicit OpenXrCudaDemo(Options options)
        : options_(options)
    {
    }

    ~OpenXrCudaDemo()
    {
        cleanup();
    }

    void run()
    {
        initializeOpenXr();
        initializeD3D11();
        initializeSession();
        initializeSwapchains();
        initializeCuda();
        if (options_.hgsStreamEnabled) {
            initializeHorizonGsStream();
        } else {
            initializePoseSocket();
        }
        frameLoop();
    }

private:
    void log(const std::string& message) const
    {
        if (!options_.quiet) {
            std::cout << message << std::endl;
        }
    }

    void initializeOpenXr()
    {
        if (!runtimeSupportsD3D11()) {
            throw DemoError("Active OpenXR runtime does not advertise XR_KHR_D3D11_enable.");
        }

        const char* extensions[] = {XR_KHR_D3D11_ENABLE_EXTENSION_NAME};

        XrInstanceCreateInfo createInfo{XR_TYPE_INSTANCE_CREATE_INFO};
        std::snprintf(
            createInfo.applicationInfo.applicationName,
            sizeof(createInfo.applicationInfo.applicationName),
            "HorizonGS OpenXR CUDA Demo");
        createInfo.applicationInfo.applicationVersion = 1;
        std::snprintf(
            createInfo.applicationInfo.engineName,
            sizeof(createInfo.applicationInfo.engineName),
            "standalone-demo");
        createInfo.applicationInfo.engineVersion = 1;
        createInfo.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
        createInfo.enabledExtensionCount = static_cast<uint32_t>(std::size(extensions));
        createInfo.enabledExtensionNames = extensions;

        checkXr(XR_NULL_HANDLE, xrCreateInstance(&createInfo, &instance_), "xrCreateInstance");

        XrSystemGetInfo systemInfo{XR_TYPE_SYSTEM_GET_INFO};
        systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        checkXr(instance_, xrGetSystem(instance_, &systemInfo, &systemId_), "xrGetSystem");

        checkXr(
            instance_,
            xrGetInstanceProcAddr(
                instance_,
                "xrGetD3D11GraphicsRequirementsKHR",
                reinterpret_cast<PFN_xrVoidFunction*>(&xrGetD3D11GraphicsRequirementsKHR_)),
            "xrGetInstanceProcAddr(xrGetD3D11GraphicsRequirementsKHR)");
        if (!xrGetD3D11GraphicsRequirementsKHR_) {
            throw DemoError("xrGetD3D11GraphicsRequirementsKHR function pointer is null.");
        }

        graphicsRequirements_ = {XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR};
        checkXr(
            instance_,
            xrGetD3D11GraphicsRequirementsKHR_(instance_, systemId_, &graphicsRequirements_),
            "xrGetD3D11GraphicsRequirementsKHR");

        log("OpenXR runtime initialized. Required adapter LUID: " + luidToString(graphicsRequirements_.adapterLuid));
    }

    void initializeD3D11()
    {
        checkHr(CreateDXGIFactory1(IID_PPV_ARGS(dxgiFactory_.GetAddressOf())), "CreateDXGIFactory1");

        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> adapter;
            HRESULT hr = dxgiFactory_->EnumAdapters1(index, adapter.GetAddressOf());
            if (hr == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            checkHr(hr, "IDXGIFactory1::EnumAdapters1");

            DXGI_ADAPTER_DESC1 desc = {};
            checkHr(adapter->GetDesc1(&desc), "IDXGIAdapter1::GetDesc1");
            if (sameLuid(desc.AdapterLuid, graphicsRequirements_.adapterLuid)) {
                dxgiAdapter_ = adapter;
                adapterDesc_ = desc;
                break;
            }
        }

        if (!dxgiAdapter_) {
            throw DemoError("Could not find the D3D11 adapter requested by the OpenXR runtime.");
        }

        const D3D_FEATURE_LEVEL requestedLevels[] = {
            D3D_FEATURE_LEVEL_12_1,
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
        };

        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        checkHr(
            D3D11CreateDevice(
                dxgiAdapter_.Get(),
                D3D_DRIVER_TYPE_UNKNOWN,
                nullptr,
                flags,
                requestedLevels,
                static_cast<UINT>(std::size(requestedLevels)),
                D3D11_SDK_VERSION,
                d3dDevice_.GetAddressOf(),
                &featureLevel_,
                d3dContext_.GetAddressOf()),
            "D3D11CreateDevice");

        if (featureLevel_ < graphicsRequirements_.minFeatureLevel) {
            std::ostringstream oss;
            oss << "Created D3D11 feature level " << std::hex << featureLevel_
                << " is lower than OpenXR minimum " << graphicsRequirements_.minFeatureLevel;
            throw DemoError(oss.str());
        }

        std::wcout << L"D3D11 device created on adapter: " << adapterDesc_.Description << std::endl;
    }

    void initializeSession()
    {
        XrGraphicsBindingD3D11KHR graphicsBinding{XR_TYPE_GRAPHICS_BINDING_D3D11_KHR};
        graphicsBinding.device = d3dDevice_.Get();

        XrSessionCreateInfo sessionCreateInfo{XR_TYPE_SESSION_CREATE_INFO};
        sessionCreateInfo.next = &graphicsBinding;
        sessionCreateInfo.systemId = systemId_;
        checkXr(instance_, xrCreateSession(instance_, &sessionCreateInfo, &session_), "xrCreateSession");

        XrReferenceSpaceCreateInfo spaceCreateInfo{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        spaceCreateInfo.referenceSpaceType = options_.referenceSpace;
        spaceCreateInfo.poseInReferenceSpace.orientation.w = 1.0f;
        XrResult spaceResult = xrCreateReferenceSpace(session_, &spaceCreateInfo, &appSpace_);
        if (XR_FAILED(spaceResult) && options_.referenceSpace == XR_REFERENCE_SPACE_TYPE_STAGE) {
            log("Stage reference space unavailable; falling back to local.");
            spaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
            checkXr(instance_, xrCreateReferenceSpace(session_, &spaceCreateInfo, &appSpace_), "xrCreateReferenceSpace(local)");
        } else {
            checkXr(instance_, spaceResult, "xrCreateReferenceSpace");
        }

        std::wcout << L"OpenXR session created using " << referenceSpaceName(spaceCreateInfo.referenceSpaceType)
                   << L" reference space." << std::endl;
    }

    void initializeSwapchains()
    {
        uint32_t viewCount = 0;
        checkXr(
            instance_,
            xrEnumerateViewConfigurationViews(instance_, systemId_, kViewConfig, 0, &viewCount, nullptr),
            "xrEnumerateViewConfigurationViews(count)");
        if (viewCount != 2) {
            std::ostringstream oss;
            oss << "This demo expects exactly two stereo views, got " << viewCount << ".";
            throw DemoError(oss.str());
        }

        viewConfigViews_.resize(viewCount);
        for (auto& view : viewConfigViews_) {
            view.type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
        }
        checkXr(
            instance_,
            xrEnumerateViewConfigurationViews(
                instance_,
                systemId_,
                kViewConfig,
                viewCount,
                &viewCount,
                viewConfigViews_.data()),
            "xrEnumerateViewConfigurationViews");

        uint32_t formatCount = 0;
        checkXr(
            instance_,
            xrEnumerateSwapchainFormats(session_, 0, &formatCount, nullptr),
            "xrEnumerateSwapchainFormats(count)");
        std::vector<int64_t> formats(formatCount);
        checkXr(
            instance_,
            xrEnumerateSwapchainFormats(session_, formatCount, &formatCount, formats.data()),
            "xrEnumerateSwapchainFormats");

        colorFormat_ = chooseColorFormat(formats);
        if (colorFormat_ == DXGI_FORMAT_UNKNOWN) {
            throw DemoError(
                "OpenXR runtime did not advertise a CUDA-compatible 8-bit RGBA/BGRA swapchain format. Runtime formats: "
                + dxgiFormatList(formats));
        }
        log("Selected OpenXR swapchain format: " + dxgiFormatName(colorFormat_));

        swapchains_.resize(viewCount);
        for (uint32_t eye = 0; eye < viewCount; ++eye) {
            const XrViewConfigurationView& view = viewConfigViews_[eye];

            XrSwapchainCreateInfo swapchainCreateInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
            swapchainCreateInfo.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
            swapchainCreateInfo.format = static_cast<int64_t>(colorFormat_);
            swapchainCreateInfo.sampleCount = view.recommendedSwapchainSampleCount;
            swapchainCreateInfo.width = std::max(
                1u,
                static_cast<uint32_t>(static_cast<float>(view.recommendedImageRectWidth) * options_.swapchainScale + 0.5f));
            swapchainCreateInfo.height = std::max(
                1u,
                static_cast<uint32_t>(static_cast<float>(view.recommendedImageRectHeight) * options_.swapchainScale + 0.5f));
            swapchainCreateInfo.faceCount = 1;
            swapchainCreateInfo.arraySize = 1;
            swapchainCreateInfo.mipCount = 1;

            Swapchain& swapchain = swapchains_[eye];
            checkXr(
                instance_,
                xrCreateSwapchain(session_, &swapchainCreateInfo, &swapchain.handle),
                "xrCreateSwapchain");
            swapchain.width = static_cast<int32_t>(swapchainCreateInfo.width);
            swapchain.height = static_cast<int32_t>(swapchainCreateInfo.height);

            uint32_t imageCount = 0;
            checkXr(
                instance_,
                xrEnumerateSwapchainImages(swapchain.handle, 0, &imageCount, nullptr),
                "xrEnumerateSwapchainImages(count)");
            swapchain.images.resize(imageCount);
            for (auto& image : swapchain.images) {
                image.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR;
            }
            checkXr(
                instance_,
                xrEnumerateSwapchainImages(
                    swapchain.handle,
                    imageCount,
                    &imageCount,
                    reinterpret_cast<XrSwapchainImageBaseHeader*>(swapchain.images.data())),
                "xrEnumerateSwapchainImages");

            std::ostringstream oss;
            oss << "Eye " << eye << " swapchain: " << swapchain.width << "x" << swapchain.height
                << ", images=" << swapchain.images.size();
            log(oss.str());
        }
    }

    void initializeCuda()
    {
        std::array<uint32_t, 2> widths = {
            static_cast<uint32_t>(swapchains_[0].width),
            static_cast<uint32_t>(swapchains_[1].width),
        };
        std::array<uint32_t, 2> heights = {
            static_cast<uint32_t>(swapchains_[0].height),
            static_cast<uint32_t>(swapchains_[1].height),
        };

        std::string error;
        if (!cudaRenderer_.initialize(d3dDevice_.Get(), colorFormat_, widths, heights, &error)) {
            throw DemoError("CUDA renderer initialization failed: " + error);
        }
        log("CUDA renderer initialized.");
    }

    void initializePoseSocket()
    {
        if (!options_.poseSocketEnabled) {
            return;
        }

        if (options_.poseSocketStartDelaySeconds > 0) {
            std::ostringstream delayMessage;
            delayMessage << "Waiting " << options_.poseSocketStartDelaySeconds
                         << "s before connecting pose socket. Put on the headset and look at the anchor direction.";
            log(delayMessage.str());
            std::this_thread::sleep_for(std::chrono::seconds(options_.poseSocketStartDelaySeconds));
        }

        std::string lastError;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(options_.poseSocketRetrySeconds);
        while (true) {
            try {
                poseSocket_.connectTo(options_.poseSocketHost, options_.poseSocketPort);
                std::ostringstream oss;
                oss << "Pose socket connected to " << options_.poseSocketHost << ":" << options_.poseSocketPort;
                log(oss.str());
                return;
            } catch (const std::exception& e) {
                lastError = e.what();
                poseSocket_.close();
            }

            if (std::chrono::steady_clock::now() >= deadline) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        std::ostringstream error;
        error << "Could not connect pose socket at " << options_.poseSocketHost << ":" << options_.poseSocketPort
              << " after " << options_.poseSocketRetrySeconds << "s. Last error: " << lastError
              << ". Start render.py with --xr_mode openxr_socket first, or check that the port is listening.";
        if (options_.poseSocketOptional) {
            log(error.str());
            log("Continuing without sending OpenXR poses because --pose-socket-optional was set.");
            return;
        }
        throw DemoError(error.str());
    }

    void initializeHorizonGsStream()
    {
        if (options_.poseSocketStartDelaySeconds > 0) {
            std::ostringstream delayMessage;
            delayMessage << "Waiting " << options_.poseSocketStartDelaySeconds
                         << "s before starting HorizonGS stream. Put on the headset and look at the anchor direction.";
            log(delayMessage.str());
            std::this_thread::sleep_for(std::chrono::seconds(options_.poseSocketStartDelaySeconds));
        }

        hgsStreamClient_.start(
            options_.poseSocketHost,
            options_.poseSocketPort,
            options_.poseSocketRetrySeconds,
            options_.quiet);
        std::ostringstream oss;
        oss << "HorizonGS stream worker started for " << options_.poseSocketHost << ":" << options_.poseSocketPort;
        log(oss.str());
    }

    void frameLoop()
    {
        std::vector<XrView> views(swapchains_.size());
        for (auto& view : views) {
            view.type = XR_TYPE_VIEW;
        }

        std::vector<XrCompositionLayerProjectionView> projectionViews(swapchains_.size());
        for (auto& projectionView : projectionViews) {
            projectionView.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
        }

        const auto start = std::chrono::steady_clock::now();
        int renderedFrames = 0;

        while (!exitRequested_) {
            pollEvents();

            if (options_.frameLimit >= 0 && renderedFrames >= options_.frameLimit) {
                log("Frame limit reached; requesting exit.");
                xrRequestExitSession(session_);
                exitRequested_ = true;
                break;
            }

            if (!sessionRunning_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }

            XrFrameWaitInfo waitInfo{XR_TYPE_FRAME_WAIT_INFO};
            XrFrameState frameState{XR_TYPE_FRAME_STATE};
            checkXr(instance_, xrWaitFrame(session_, &waitInfo, &frameState), "xrWaitFrame");

            XrFrameBeginInfo beginInfo{XR_TYPE_FRAME_BEGIN_INFO};
            checkXr(instance_, xrBeginFrame(session_, &beginInfo), "xrBeginFrame");

            std::vector<XrCompositionLayerBaseHeader*> layers;
            XrCompositionLayerProjection projectionLayer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};

            if (frameState.shouldRender) {
                XrViewLocateInfo locateInfo{XR_TYPE_VIEW_LOCATE_INFO};
                locateInfo.viewConfigurationType = kViewConfig;
                locateInfo.displayTime = frameState.predictedDisplayTime;
                locateInfo.space = appSpace_;

                XrViewState viewState{XR_TYPE_VIEW_STATE};
                uint32_t viewCountOutput = 0;
                checkXr(
                    instance_,
                    xrLocateViews(
                        session_,
                        &locateInfo,
                        &viewState,
                        static_cast<uint32_t>(views.size()),
                        &viewCountOutput,
                        views.data()),
                    "xrLocateViews");

                if (viewCountOutput != views.size()) {
                    throw DemoError("xrLocateViews returned an unexpected view count.");
                }

                const auto now = std::chrono::steady_clock::now();
                const float seconds = std::chrono::duration<float>(now - start).count();
                if (options_.hgsStreamEnabled) {
                    hgsStreamClient_.submitFrame(renderedFrames, frameState.predictedDisplayTime, views, swapchains_);
                } else if (poseSocket_.isConnected()) {
                    poseSocket_.sendFrame(renderedFrames, frameState.predictedDisplayTime, views, swapchains_);
                }

                for (uint32_t eye = 0; eye < swapchains_.size(); ++eye) {
                    renderAndCopyEye(eye, views[eye], seconds);

                    projectionViews[eye].pose = views[eye].pose;
                    projectionViews[eye].fov = views[eye].fov;
                    projectionViews[eye].subImage.swapchain = swapchains_[eye].handle;
                    projectionViews[eye].subImage.imageRect.offset = {0, 0};
                    projectionViews[eye].subImage.imageRect.extent = {swapchains_[eye].width, swapchains_[eye].height};
                    projectionViews[eye].subImage.imageArrayIndex = 0;
                }

                projectionLayer.space = appSpace_;
                projectionLayer.viewCount = static_cast<uint32_t>(projectionViews.size());
                projectionLayer.views = projectionViews.data();
                layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&projectionLayer));
                ++renderedFrames;

                if (!options_.quiet && renderedFrames % 120 == 0) {
                    std::cout << "Rendered " << renderedFrames << " frames." << std::endl;
                }
            }

            XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
            endInfo.displayTime = frameState.predictedDisplayTime;
            endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
            endInfo.layerCount = static_cast<uint32_t>(layers.size());
            endInfo.layers = layers.empty() ? nullptr : layers.data();
            checkXr(instance_, xrEndFrame(session_, &endInfo), "xrEndFrame");
        }
    }

    void renderAndCopyEye(uint32_t eye, const XrView& view, float seconds)
    {
        if (options_.hgsStreamEnabled) {
            auto frame = hgsStreamClient_.latestFrame();
            if (frame && copyHorizonGsEye(eye, *frame)) {
                return;
            }
        }

        Swapchain& swapchain = swapchains_[eye];

        const std::array<float, 3> position = {
            view.pose.position.x,
            view.pose.position.y,
            view.pose.position.z,
        };
        const std::array<float, 4> orientation = {
            view.pose.orientation.x,
            view.pose.orientation.y,
            view.pose.orientation.z,
            view.pose.orientation.w,
        };

        std::string cudaError;
        if (!cudaRenderer_.renderEye(static_cast<int>(eye), seconds, position, orientation, &cudaError)) {
            throw DemoError("CUDA render failed: " + cudaError);
        }

        uint32_t imageIndex = 0;
        XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        checkXr(
            instance_,
            xrAcquireSwapchainImage(swapchain.handle, &acquireInfo, &imageIndex),
            "xrAcquireSwapchainImage");

        XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        waitInfo.timeout = XR_INFINITE_DURATION;
        checkXr(instance_, xrWaitSwapchainImage(swapchain.handle, &waitInfo), "xrWaitSwapchainImage");

        ID3D11Texture2D* destination = swapchain.images[imageIndex].texture;
        if (!destination) {
            throw DemoError("OpenXR returned a null D3D11 swapchain texture.");
        }

        d3dContext_->CopyResource(destination, cudaRenderer_.texture(static_cast<int>(eye)));
        d3dContext_->Flush();

        XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        checkXr(instance_, xrReleaseSwapchainImage(swapchain.handle, &releaseInfo), "xrReleaseSwapchainImage");
    }

    bool copyHorizonGsEye(uint32_t eye, const StreamImageFrame& frame)
    {
        if (eye >= swapchains_.size()) {
            return false;
        }
        Swapchain& swapchain = swapchains_[eye];
        if (frame.width != swapchain.width || frame.height != swapchain.height || frame.channels != 4) {
            return false;
        }

        const std::vector<uint8_t>& source = eye == 0 ? frame.left : frame.right;
        if (source.size() != static_cast<size_t>(swapchain.width) * static_cast<size_t>(swapchain.height) * 4) {
            return false;
        }

        const uint8_t* sourceData = source.data();
        std::vector<uint8_t> converted;
        if (colorFormat_ == DXGI_FORMAT_B8G8R8A8_UNORM || colorFormat_ == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) {
            converted.resize(source.size());
            for (size_t i = 0; i < source.size(); i += 4) {
                converted[i + 0] = source[i + 2];
                converted[i + 1] = source[i + 1];
                converted[i + 2] = source[i + 0];
                converted[i + 3] = source[i + 3];
            }
            sourceData = converted.data();
        }

        uint32_t imageIndex = 0;
        XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        checkXr(
            instance_,
            xrAcquireSwapchainImage(swapchain.handle, &acquireInfo, &imageIndex),
            "xrAcquireSwapchainImage");

        XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        waitInfo.timeout = XR_INFINITE_DURATION;
        checkXr(instance_, xrWaitSwapchainImage(swapchain.handle, &waitInfo), "xrWaitSwapchainImage");

        ID3D11Texture2D* destination = swapchain.images[imageIndex].texture;
        if (!destination) {
            throw DemoError("OpenXR returned a null D3D11 swapchain texture.");
        }

        const UINT rowPitch = static_cast<UINT>(static_cast<size_t>(swapchain.width) * 4);
        d3dContext_->UpdateSubresource(destination, 0, nullptr, sourceData, rowPitch, 0);
        d3dContext_->Flush();

        XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        checkXr(instance_, xrReleaseSwapchainImage(swapchain.handle, &releaseInfo), "xrReleaseSwapchainImage");
        return true;
    }

    void pollEvents()
    {
        while (true) {
            XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
            XrResult result = xrPollEvent(instance_, &event);
            if (result == XR_EVENT_UNAVAILABLE) {
                return;
            }
            checkXr(instance_, result, "xrPollEvent");

            switch (event.type) {
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                const auto* stateEvent = reinterpret_cast<const XrEventDataSessionStateChanged*>(&event);
                sessionState_ = stateEvent->state;
                log(std::string("OpenXR session state: ") + sessionStateName(sessionState_));

                if (sessionState_ == XR_SESSION_STATE_READY) {
                    XrSessionBeginInfo beginInfo{XR_TYPE_SESSION_BEGIN_INFO};
                    beginInfo.primaryViewConfigurationType = kViewConfig;
                    checkXr(instance_, xrBeginSession(session_, &beginInfo), "xrBeginSession");
                    sessionRunning_ = true;
                } else if (sessionState_ == XR_SESSION_STATE_STOPPING) {
                    sessionRunning_ = false;
                    checkXr(instance_, xrEndSession(session_), "xrEndSession");
                } else if (
                    sessionState_ == XR_SESSION_STATE_EXITING ||
                    sessionState_ == XR_SESSION_STATE_LOSS_PENDING) {
                    exitRequested_ = true;
                    sessionRunning_ = false;
                }
                break;
            }
            case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
                exitRequested_ = true;
                sessionRunning_ = false;
                break;
            default:
                break;
            }
        }
    }

    void cleanup()
    {
        hgsStreamClient_.stop();
        poseSocket_.sendEos();
        poseSocket_.close();

        cudaRenderer_.shutdown();

        for (auto& swapchain : swapchains_) {
            if (swapchain.handle != XR_NULL_HANDLE) {
                xrDestroySwapchain(swapchain.handle);
                swapchain.handle = XR_NULL_HANDLE;
            }
        }
        swapchains_.clear();

        if (appSpace_ != XR_NULL_HANDLE) {
            xrDestroySpace(appSpace_);
            appSpace_ = XR_NULL_HANDLE;
        }

        if (session_ != XR_NULL_HANDLE) {
            xrDestroySession(session_);
            session_ = XR_NULL_HANDLE;
        }

        d3dContext_.Reset();
        d3dDevice_.Reset();
        dxgiAdapter_.Reset();
        dxgiFactory_.Reset();

        if (instance_ != XR_NULL_HANDLE) {
            xrDestroyInstance(instance_);
            instance_ = XR_NULL_HANDLE;
        }
    }

    Options options_;

    XrInstance instance_ = XR_NULL_HANDLE;
    XrSystemId systemId_ = XR_NULL_SYSTEM_ID;
    XrSession session_ = XR_NULL_HANDLE;
    XrSpace appSpace_ = XR_NULL_HANDLE;
    XrSessionState sessionState_ = XR_SESSION_STATE_UNKNOWN;
    bool sessionRunning_ = false;
    bool exitRequested_ = false;

    PFN_xrGetD3D11GraphicsRequirementsKHR xrGetD3D11GraphicsRequirementsKHR_ = nullptr;
    XrGraphicsRequirementsD3D11KHR graphicsRequirements_{XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR};

    ComPtr<IDXGIFactory1> dxgiFactory_;
    ComPtr<IDXGIAdapter1> dxgiAdapter_;
    DXGI_ADAPTER_DESC1 adapterDesc_ = {};
    ComPtr<ID3D11Device> d3dDevice_;
    ComPtr<ID3D11DeviceContext> d3dContext_;
    D3D_FEATURE_LEVEL featureLevel_ = D3D_FEATURE_LEVEL_11_0;

    std::vector<XrViewConfigurationView> viewConfigViews_;
    std::vector<Swapchain> swapchains_;
    DXGI_FORMAT colorFormat_ = DXGI_FORMAT_UNKNOWN;
    CudaRenderer cudaRenderer_;
    PoseSocketClient poseSocket_;
    HorizonGsStreamClient hgsStreamClient_;
};

} // namespace

int main(int argc, char** argv)
{
    try {
        OpenXrCudaDemo demo(parseOptions(argc, argv));
        demo.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "openxr_cuda_demo error: " << e.what() << std::endl;
        return 1;
    }
}
