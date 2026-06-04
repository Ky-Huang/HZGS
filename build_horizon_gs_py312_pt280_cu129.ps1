$ErrorActionPreference = "Stop"

# Run on the build machine from Developer PowerShell for VS 2022.
# Output: horizon_gs_py312_pt280_cu129.zip

$EnvName = "horizon_gs_py312_pt280_cu129"
$PackOutput = Join-Path $PSScriptRoot "$EnvName.zip"
$TorchIndexUrl = "https://download.pytorch.org/whl/cu129"
$PygWheelUrl = "https://data.pyg.org/whl/torch-2.8.0+cu129.html"
$GsplatUrl = "git+https://github.com/tongji-rkr/gsplat-2dgs.git@ab4ce3921d54d1a781af00bd53dbb31b2d65a4c5"

$CondaExe = $env:CONDA_EXE
if ([string]::IsNullOrWhiteSpace($CondaExe) -or -not (Test-Path $CondaExe)) {
    $CondaExe = (Get-Command conda.exe -ErrorAction Stop).Source
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "cl.exe not found. Please run this script from Developer PowerShell for VS 2022."
}

$EnvList = & $CondaExe env list
if (-not (($EnvList | Select-String -SimpleMatch $EnvName) -ne $null)) {
    & $CondaExe create -y -n $EnvName `
        python=3.12 pip cmake ninja git glm `
        cuda-toolkit=12.9 cuda-nvcc=12.9 cuda-cudart-dev=12.9 `
        -c nvidia -c conda-forge -c defaults
    if ($LASTEXITCODE -ne 0) { throw "conda create failed" }
}

$PythonExeOutput = & $CondaExe run -n $EnvName python -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) {
    throw "Existing conda env is not usable. Run this once, then rerun the script: conda env remove -n $EnvName -y"
}
$PythonExe = ($PythonExeOutput | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Last 1).Trim()
if (-not (Test-Path $PythonExe)) {
    throw "python.exe was not found: $PythonExe"
}

$EnvPrefix = Split-Path $PythonExe -Parent
$CudaPrefix = Join-Path $EnvPrefix "Library"
$env:CUDA_HOME = $CudaPrefix
$env:CUDA_PATH = $CudaPrefix
$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:FORCE_CUDA = "1"
$env:PATH = "$CudaPrefix\bin;$EnvPrefix;$EnvPrefix\Scripts;$env:PATH"
$env:INCLUDE = "$CudaPrefix\include;$env:INCLUDE"
$env:LIB = "$CudaPrefix\lib;$env:LIB"

where.exe nvcc
nvcc --version

& $PythonExe -m pip install --upgrade pip setuptools wheel packaging
if ($LASTEXITCODE -ne 0) { throw "pip tools install failed" }

& $PythonExe -m pip install --upgrade torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url $TorchIndexUrl
if ($LASTEXITCODE -ne 0) { throw "pytorch install failed" }

& $PythonExe -m pip install --upgrade `
    absl-py addict ConfigArgParse dash einops flask gitpython imageio ipython ipywidgets jaxtyping `
    laspy lpips matplotlib nbformat numpy open3d opencv-python pandas plotly plyfile pydantic `
    pyquaternion pyyaml requests rich scikit-learn scipy shapely tensorboard tqdm wandb
if ($LASTEXITCODE -ne 0) { throw "python dependencies install failed" }

& $PythonExe -m pip install --upgrade torch-scatter==2.1.2+pt28cu129 -f $PygWheelUrl
if ($LASTEXITCODE -ne 0) { throw "torch-scatter install failed" }

& $PythonExe -m pip install -v --no-build-isolation --no-cache-dir --no-deps $GsplatUrl
if ($LASTEXITCODE -ne 0) { throw "gsplat install failed" }

& $PythonExe -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('arch_list', torch.cuda.get_arch_list())"
if ($LASTEXITCODE -ne 0) { throw "torch verification failed" }

& $PythonExe -c "import torch_scatter, gsplat; from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs; print('torch-scatter and gsplat ok')"
if ($LASTEXITCODE -ne 0) { throw "extension verification failed" }

& $PythonExe -m pip install --upgrade conda-pack
if ($LASTEXITCODE -ne 0) { throw "conda-pack install failed" }

$CondaPackExe = Join-Path $EnvPrefix "Scripts\conda-pack.exe"
& $CondaPackExe -n $EnvName -o $PackOutput --force
if ($LASTEXITCODE -ne 0) { throw "conda-pack failed" }

Write-Host "Done: $PackOutput" -ForegroundColor Green
