$P = '.'; Set-Location $P; $env:TEMP = "$P\.tmp"; $env:TMP = "$P\.tmp"
$py = "$P\.venv\nestor\Scripts\python.exe"
& $py -c "import torch, triton, tiktoken, pyarrow, huggingface_hub, jaxtyping, zstandard, os; a=torch.randn(512,512,device='cuda',dtype=torch.bfloat16); print('ENV_OK', torch.__version__, torch.version.cuda, triton.__version__, tiktoken.__version__, pyarrow.__version__, huggingface_hub.__version__, torch.cuda.get_device_name(0), hasattr(torch,'_grouped_mm'), float((a@a).float().abs().mean())>0, os.devnull)" *>> logs\env_check.log
