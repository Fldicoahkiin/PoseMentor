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

function Start-Services {
    Ensure-Dirs
    Start-OneService -Name "backend_api" -ScriptFile "backend_api.py"
    Start-OneService -Name "admin_console" -ScriptFile "admin_console.py"
    Start-OneService -Name "app_demo" -ScriptFile "app_demo.py"

    Write-Host "服务已启动"
    Write-Host "- Backend API:  http://127.0.0.1:8787"
    Write-Host "- 管理后台:      http://127.0.0.1:7861"
    Write-Host "- 在线系统:      http://127.0.0.1:7860"
}

function Stop-Services {
    Ensure-Dirs
    foreach ($name in @("backend_api", "admin_console", "app_demo")) {
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
