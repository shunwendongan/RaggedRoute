[CmdletBinding()]
param(
  [string]$Venv = ".venv-triton"
)

$ErrorActionPreference = "Stop"

$py = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $py) {
  throw "Python launcher 'py' is required; install CPython 3.12 first."
}

& py -3.12 -m venv $Venv
$python = Join-Path $Venv "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 `
  torch==2.12.1+cu130 triton-windows==3.7.1.post27
& $python -c "import torch, triton; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability() == (8, 6); print(f'torch={torch.__version__}; triton={triton.__version__}; gpu={torch.cuda.get_device_name(0)}')"
