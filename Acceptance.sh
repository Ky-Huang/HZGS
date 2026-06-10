# 指标1 分辨率
CUDA_VISIBLE_DEVICES=7 python render.py \
  -m outputs/horizongs/real/road_subset/fine \
  --render_only \
  --acceptance \
  --outputspath acceptance \
  --mixed_render \  # 混合渲染模式

# 指标2 帧率
CUDA_VISIBLE_DEVICES=7 python render.py \
  -m outputs/horizongs/real/road_subset/fine \
  --render_only \
  --acceptance \
  --outputspath acceptance \
  --mixed_render \
  --showFPS         # 统计fps

# 指标3 质量
CUDA_VISIBLE_DEVICES=7 python render.py \
  -m outputs/horizongs/real/road_subset/fine \
  --render_only \
  --acceptance \
  --outputspath acceptance/fix_lod \
  --mixed_render \
  --fix-lod 4       # 固定lod为最大

python tools/avg_psnr.py

# 指标4 图元支持
CUDA_VISIBLE_DEVICES=7 python render.py \
  -m outputs/horizongs/real/road_subset/fine \
  --render_only \
  --acceptance \
  --outputspath acceptance/orbit_mixed \
  --mixed_render \
  --orbit_render_car

CUDA_VISIBLE_DEVICES=7 python render.py \
  -m outputs/horizongs/real/road_subset/fine \
  --render_only \
  --acceptance \
  --outputspath acceptance/orbit_unmixed \
  --orbit_render_car