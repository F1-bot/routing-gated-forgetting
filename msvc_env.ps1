# Dot-source before any torch.compile / triton run: sets up MSVC + Windows SDK like VsDevCmd.
$msvc = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519"
$sdk = "C:\Program Files (x86)\Windows Kits\10"
$sdkv = "10.0.22621.0"
$env:CC = "$msvc\bin\Hostx64\x64\cl.exe"
$env:CXX = "$msvc\bin\Hostx64\x64\cl.exe"
$env:INCLUDE = "$msvc\include;$sdk\Include\$sdkv\ucrt;$sdk\Include\$sdkv\shared;$sdk\Include\$sdkv\um;$sdk\Include\$sdkv\winrt"
$env:LIB = "$msvc\lib\x64;$sdk\Lib\$sdkv\ucrt\x64;$sdk\Lib\$sdkv\um\x64"
$env:PATH = "$msvc\bin\Hostx64\x64;$sdk\bin\$sdkv\x64;" + $env:PATH
$ws = (Get-Location).Path
$env:TRITON_CACHE_DIR = "$ws\tc"
$env:TORCHINDUCTOR_CACHE_DIR = "$ws\ic"
$env:TIKTOKEN_CACHE_DIR = "$ws\data\tiktoken"
$env:WANDB_MODE = "disabled"
$env:PYTHONUTF8 = "1"
