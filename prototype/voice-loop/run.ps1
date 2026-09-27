# PROTOTYPE launcher — one command:  powershell -File prototype\voice-loop\run.ps1
# First run creates the throwaway venv and downloads torch + model weights (several GB).
$ErrorActionPreference = "Stop"
$dir = $PSScriptRoot
$venvPy = Join-Path $dir ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
  # Python 3.12: kokoro pins numpy==1.26.4, which has no 3.13 wheel.
  Write-Host "[run] creating throwaway env (first run only)..."
  & C:\Users\Gustaf\anaconda3\Scripts\conda.exe create -p (Join-Path $dir ".venv") python=3.12 -y
  & $venvPy -m pip install --upgrade pip
  & $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  & $venvPy -m pip install transformers silero-vad onnxruntime soundfile librosa numpy fastapi "uvicorn[standard]" kokoro
}

& $venvPy (Join-Path $dir "server.py")
