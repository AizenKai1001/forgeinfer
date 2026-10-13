<#
.SYNOPSIS
  Native Windows build of the ForgeInfer `forge` binary (trainer, engine, CLI) with MSVC + the CUDA toolkit. Builds only: nothing here starts the GPU.

.DESCRIPTION
  1. Finds Visual Studio 2022 (Build Tools or an IDE edition with the C++ tools) with vswhere and loads its vcvars64.bat environment into this
     PowerShell session (nvcc needs cl.exe as its host compiler even for -cubin/-ptx; CUDA 12.x rejects the MSVC 14.5x of Visual Studio 2026).
  2. Checks nvcc (CUDA_PATH or the PATH), cargo and the NVIDIA driver DLL (file version only; the driver is not loaded).
  3. cargo build --release -p forge-cli with a private target folder, temp files under it, and at most -Jobs parallel jobs.
     -CudaArchs adds native cubins (default 89 = Ada, the RTX 4060); sm_75 and sm_80 cubins and compute_75/80 PTX are always built.
  4. -Test also runs the CPU-only tests and only BUILDS the GPU test binaries (cargo test --no-run): no test that touches CUDA is started.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1
  powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 -Test
  powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 -CudaArchs none -Jobs 4
#>
[CmdletBinding()]
param(
    # extra native cubin architectures (FORGE_CUDA_ARCHS), comma separated; "none" = only the portable images (sm_75, sm_80 cubins + PTX)
    # (an empty string does not survive "powershell -File" when called from PowerShell 5.1, hence "none")
    [string]$CudaArchs = "89",
    # parallel cargo jobs (rustc processes); nvcc runs one at a time inside the forge-kernels build script
    [int]$Jobs = 6,
    # cargo target folder (default: <repo>\target); temp files go to <TargetDir>\tmp
    [string]$TargetDir = "",
    # also run the CPU-only tests and build (not run) the GPU test binaries
    [switch]$Test
)

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($TargetDir -eq "") { $TargetDir = Join-Path $repo "target" }
if ($CudaArchs -eq "none") { $CudaArchs = "" }
# temp files of vcvars, nvcc and cl go under the target folder, not into the user's temp folder
$tmp = Join-Path $TargetDir "tmp"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$env:TMP = $tmp
$env:TEMP = $tmp

function Fail([string]$msg) {
    Write-Host "build_windows: $msg" -ForegroundColor Red
    exit 1
}

# ---- 1. Visual Studio 2022 environment -------------------------------------------------------------------------------------------------
$pf86 = ${env:ProgramFiles(x86)}
if (-not $pf86) { $pf86 = "C:\Program Files (x86)" }
$vswhere = Join-Path $pf86 "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { Fail "vswhere.exe not found ($vswhere): install Visual Studio 2022 Build Tools with the 'Desktop development with C++' workload" }
$vcTools = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
$vsPath = & $vswhere -products * -version "[17.0,18.0)" -requires $vcTools -latest -property installationPath
if (-not $vsPath) {
    $vsPath = & $vswhere -products * -requires $vcTools -latest -property installationPath
    if (-not $vsPath) { Fail "no Visual Studio with the C++ x64 tools found: install Visual Studio 2022 Build Tools ('Desktop development with C++')" }
    Write-Host "warning: no Visual Studio 2022 found, using $vsPath; CUDA 12.x accepts only MSVC up to 14.4x (VS 2022) and will refuse a newer cl.exe" -ForegroundColor Yellow
}
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { Fail "vcvars64.bat not found under $vsPath" }
# run vcvars64.bat in cmd and copy the resulting environment into this session
$envDump = & cmd.exe /c "`"$vcvars`" >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) { Fail "vcvars64.bat failed ($vcvars)" }
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process") }
}
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $cl) { Fail "cl.exe is not on the PATH after vcvars64.bat" }
Write-Host "MSVC     : $($cl.Source)"

# ---- 2. CUDA, Rust, driver --------------------------------------------------------------------------------------------------------------
$nvcc = $null
if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) { $nvcc = Join-Path $env:CUDA_PATH "bin\nvcc.exe" }
if (-not $nvcc) {
    $c = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($c) { $nvcc = $c.Source }
}
if (-not $nvcc) { Fail "nvcc not found: install the CUDA toolkit (12.x or newer) or set CUDA_PATH; without it the binary would have NO GPU kernels" }
$nvccVer = (& $nvcc --version | Select-String "release") -join ""
Write-Host "nvcc     : $nvcc ($($nvccVer.Trim()))"
$cargo = Get-Command cargo.exe -ErrorAction SilentlyContinue
if (-not $cargo) { Fail "cargo not found: install Rust with rustup (default host x86_64-pc-windows-msvc)" }
$host_ = (& rustc -vV | Select-String "^host:") -join ""
Write-Host "Rust     : $((& rustc --version) -join '') ($($host_.Trim()))"
$drv = Join-Path $env:SystemRoot "System32\nvcuda.dll"
if (Test-Path $drv) {
    $vi = (Get-Item $drv).VersionInfo
    Write-Host "driver   : $($vi.FileDescription) / $($vi.ProductName) (file version only, the driver is not loaded)"
} else {
    Write-Host "warning: $drv not found: the build works, but forge.exe will not find a CUDA driver at run time" -ForegroundColor Yellow
}

# ---- 3. build ---------------------------------------------------------------------------------------------------------------------------
$env:CARGO_TARGET_DIR = $TargetDir
$env:TMP = $tmp
$env:TEMP = $tmp
$env:FORGE_CUDA_ARCHS = $CudaArchs
Write-Host "target   : $TargetDir (temp $tmp), jobs $Jobs, extra cubins: '$CudaArchs'"
$manifest = Join-Path $repo "Cargo.toml"
$t0 = Get-Date
& cargo build --release -j $Jobs --manifest-path $manifest -p forge-cli
if ($LASTEXITCODE -ne 0) { Fail "cargo build failed (exit $LASTEXITCODE)" }
$exe = Join-Path $TargetDir "release\forge.exe"
if (-not (Test-Path $exe)) { Fail "build finished but $exe is missing" }
$mb = [math]::Round((Get-Item $exe).Length / 1MB, 1)
Write-Host ("built    : {0} ({1} MiB) in {2:N0} s" -f $exe, $mb, ((Get-Date) - $t0).TotalSeconds) -ForegroundColor Green

# ---- 4. tests (CPU only; GPU test binaries are compiled, never run) --------------------------------------------------------------------
if ($Test) {
    # crates that cannot reach CUDA, plus the CPU-only unit tests of forge-cli (data tools) and forge-engine (text streaming)
    & cargo test --release -j $Jobs --manifest-path $manifest -p forge-core -p forge-kernels -p forge-gguf -p forge-quant -p forge-tok -p forge-chat -p forge-model
    if ($LASTEXITCODE -ne 0) { Fail "CPU tests failed (exit $LASTEXITCODE)" }
    & cargo test --release -j $Jobs --manifest-path $manifest -p forge-cli --bins
    if ($LASTEXITCODE -ne 0) { Fail "CPU unit tests of forge-cli failed (exit $LASTEXITCODE)" }
    & cargo test --release -j $Jobs --manifest-path $manifest -p forge-engine --lib
    if ($LASTEXITCODE -ne 0) { Fail "CPU unit tests of forge-engine failed (exit $LASTEXITCODE)" }
    # (forge-engine's integration test speculative.rs is left out: on main it calls tok.decode with one argument and does not compile on any platform)
    & cargo test --release -j $Jobs --manifest-path $manifest -p forge-cuda -p forge-gpu -p forge-train -p forge-serve --no-run
    if ($LASTEXITCODE -ne 0) { Fail "the GPU test binaries do not compile (exit $LASTEXITCODE)" }
    Write-Host "tests    : CPU tests passed; GPU test binaries built, not run" -ForegroundColor Green
}
Write-Host ""
Write-Host "binary: $exe"
