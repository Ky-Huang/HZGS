
# 启动 HorizonGS 图像流服务
cd D:\Code\horizonGS
conda activate horizon_gs_py_312_pt271_cu126
# 性能分析
# $env:HGS_XR_PROFILE=1
# $env:HGS_XR_SLOW_RENDER_MS=150


python render.py `
  -m outputs/horizongs/real/road_subset/fine `
  --xr_mode openxr_stream `
  --xr_config config/xr/openxr_road_anchor_frame100.yaml `
  --xr_match_swapchain_resolution_scale `
  --xr_socket_host 127.0.0.1 `
  --xr_socket_port 6110

  # 启动 SteamVR/OpenXR 投放端
  cd D:\Code\horizonGS

# 构建 cmake --build build\openxr_cuda_demo --config Release

.\build\openxr_cuda_demo\Release\openxr_cuda_demo.exe `
  --frames -1 `
  --hgs-stream `
  --pose-socket-host 127.0.0.1 `
  --pose-socket-port 6110 `
  --pose-socket-retry-seconds 60 `
  --swapchain-scale 0.25