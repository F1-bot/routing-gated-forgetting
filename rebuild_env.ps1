$ErrorActionPreference = 'Continue'
$P = '.'; Set-Location $P
New-Item -ItemType Directory -Force logs, .tmp | Out-Null; "ENV BUILD START $(Get-Date -Format s)" | Out-File logs\env_check.log; $env:TEMP = "$P\.tmp"; $env:TMP = "$P\.tmp"   # sandbox default TEMP rejects chmod (ensurepip/pip WinError 5)
$conda = '<conda>\envs\nestor\python.exe'
if (-not (Test-Path "$P\.venv\nestor\Scripts\python.exe")) { & $conda -m venv --system-site-packages "$P\.venv\nestor" *> logs\env_venv.log }
$py = "$P\.venv\nestor\Scripts\python.exe"
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'; $env:PYTHONPATH = "$P\sitecustom"
& $py -m ensurepip --upgrade --default-pip *> logs\env_ensurepip.log
& $py -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 *> logs\env_torch.log
$tv = (& $py -c "import importlib.metadata as m; r=[x for x in (m.requires('torch') or []) if x.replace(' ','').startswith('triton')]; print(r[0] if r else 'none')") | Out-String
"torch triton requirement: $tv" | Out-File logs\env_triton.log
$mm = ([regex]::Match($tv, '(\d+)\.(\d+)')).Value
if ($mm) { & $py -m pip install "triton-windows==$mm.*" *>> logs\env_triton.log } else { & $py -m pip install "triton-windows" *>> logs\env_triton.log }
& $py -m pip install "tiktoken==0.14.*" "huggingface_hub==2.4.*" "pyarrow==25.*" jaxtyping zstandard python-dotenv wandb pytest psutil requests *> logs\env_extra.log
& $py -c "import torch, triton, tiktoken, pyarrow, huggingface_hub, jaxtyping, zstandard; a=torch.randn(512,512,device='cuda',dtype=torch.bfloat16); print('ENV_OK', torch.__version__, torch.version.cuda, triton.__version__, tiktoken.__version__, pyarrow.__version__, huggingface_hub.__version__, torch.cuda.get_device_name(0), hasattr(torch,'_grouped_mm'), float((a@a).float().abs().mean())>0)" *>> logs\env_check.log
"ENV DONE" | Out-File -Append logs\env_check.log
