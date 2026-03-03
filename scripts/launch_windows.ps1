param(
    [ValidateSet("install", "start", "stop", "all")]
    [string]$Action = "all"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RuntimeDir = Join-Path $ProjectRoot "outputs\runtime"
$PidDir = Join-Path $RuntimeDir "pids"
$LogDir = Join-Path $RuntimeDir "logs"

function Ensure-Dirs {
    New-Item -ItemType Directory -Path $PidDir -Force | Out-Null
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Install-Deps {
    Push-Location $ProjectRoot
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv_cache"
    uv sync --group dev --group windows
    pnpm --dir (Join-Path $ProjectRoot "frontend") install
    Pop-Location
}

function Start-OneService {
    param(
        [string]$Name,
        [string]$ScriptFile
    )

    $outFile = Join-Path $LogDir "$Name.out.log"
    $errFile = Join-Path $LogDir "$Name.err.log"
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv_cache"
    $proc = Start-Process -FilePath "uv" -ArgumentList "run", "python", $ScriptFile -PassThru -WindowStyle Hidden -RedirectStandardOutput $outFile -RedirectStandardError $errFile -WorkingDirectory $ProjectRoot
    Set-Content -Path (Join-Path $PidDir "$Name.pid") -Value $proc.Id
}

function Start-Frontend {
    $outFile = Join-Path $LogDir "frontend.out.log"
    $errFile = Join-Path $LogDir "frontend.err.log"
    $frontendDir = Join-Path $ProjectRoot "frontend"
    $proc = Start-Process -FilePath "pnpm" -ArgumentList "--dir", $frontendDir, "dev", "--host", "127.0.0.1", "--port", "7860" -PassThru -WindowStyle Hidden -RedirectStandardOutput $outFile -RedirectStandardError $errFile -WorkingDirectory $ProjectRoot
    Set-Content -Path (Join-Path $PidDir "frontend.pid") -Value $proc.Id
}

function Start-Services {
    Ensure-Dirs
    Stop-Services
    Start-OneService -Name "backend_api" -ScriptFile "backend_api.py"
    Start-Frontend

    Write-Host "服务已启动"
    Write-Host "- Backend API:  http://127.0.0.1:8787"
    Write-Host "- 前端系统:      http://127.0.0.1:7860"
}

function Stop-Services {
    Ensure-Dirs
    foreach ($name in @("backend_api", "frontend")) {
        $pidFile = Join-Path $PidDir "$name.pid"
        if (Test-Path $pidFile) {
            $pid = Get-Content $pidFile
            $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($null -ne $proc) {
                Stop-Process -Id $pid -Force
                Write-Host "已停止 $name ($pid)"
            }
        }
    }
}

switch ($Action) {
    "install" { Install-Deps }
    "start" { Start-Services }
    "stop" { Stop-Services }
    "all" {
        Install-Deps
        Start-Services
    }
}
