# AAOS Production Startup Script (PowerShell)
# DevZen Enhanced: Full production stack startup with health checks

param(
    [switch]$SkipDocker,
    [switch]$LocalMode,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss.ffffff"
    Write-Host "[$timestamp] [$Level] $Message"
}

function Test-Port {
    param([string]$Host, [int]$Port)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect($Host, $Port)
        $tcp.Close()
        return $true
    } catch {
        return $false
    }
}

Write-Log "=========================================="
Write-Log "AAOS Production Startup"
Write-Log "=========================================="

# Step 1: Verify clean state (DEVZEN ENHANCEMENT #1)
Write-Log "Step 1: Verifying clean state..."

if (-not $SkipDocker) {
    Write-Log "Stopping existing containers..."
    docker-compose down -v --remove-orphans 2>$null

    # Start production stack
    Write-Log "Starting production stack..."
    docker-compose -f docker-compose.prod.yml up -d

    Write-Log "Waiting for services to start..."
    Start-Sleep -Seconds 10

    # Check container health
    docker-compose logs --tail=20
}

# Step 2: Redis flush safety (DEVZEN ENHANCEMENT #2)
Write-Log "Step 2: Redis flush safety check..."

$redisHost = if ($LocalMode) { "localhost" } else { "localhost" }
$redisPort = 6379

if (Test-Port -Host $redisHost -Port $redisPort) {
    Write-Log "Redis is accessible at ${redisHost}:${redisPort}"

    # Check for background operations
    $rdbSave = redis-cli -h $redisHost -p $redisPort INFO persistence 2>$null | Select-String "rdb_bgsave_in_progress:0"
    $aofRewrite = redis-cli -h $redisHost -p $redisPort INFO persistence 2>$null | Select-String "aof_rewrite_in_progress:0"

    if ($rdbSave -and $aofRewrite) {
        Write-Log "No background save operations in progress"
    } else {
        Write-Log "Background operations may be in progress - proceed with caution" "WARN"
    }
} else {
    Write-Log "Redis not accessible at ${redisHost}:${redisPort}" "ERROR"
    exit 1
}

# Step 3: Schema readiness (DEVZEN ENHANCEMENT #3)
Write-Log "Step 3: Verifying database schema..."

# Load environment
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
        }
    }
}

# Run alembic upgrade
Write-Log "Running Alembic migrations..."
python -m alembic upgrade head

# Step 4: Start orchestrator
Write-Log "Step 4: Starting orchestrator..."

$env:ENVIRONMENT = "production"
$env:LOG_LEVEL = "INFO"
$env:REDIS_HOST = $redisHost

if ($LocalMode) {
    # Start in local mode (foreground for debugging)
    Write-Log "Starting orchestrator in local mode..."
    python -m uvicorn src.orchestrator.core:app --host 0.0.0.0 --port 8000 --log-level info --workers $Workers
} else {
    # Start in background
    Write-Log "Starting orchestrator in background..."
    Start-Process -FilePath "python" -ArgumentList "-m uvicorn src.orchestrator.core:app --host 0.0.0.0 --port 8000 --log-level info --workers $Workers" -RedirectStandardOutput "aaos_prod.log" -RedirectStandardError "aaos_prod_error.log" -NoNewWindow

    Start-Sleep -Seconds 5

    # Verify startup
    Write-Log "Verifying orchestrator health..."
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 10
        if ($health.status -eq "healthy") {
            Write-Log "Orchestrator is healthy!"
        } else {
            Write-Log "Orchestrator health check returned unexpected status: $($health.status)" "WARN"
        }
    } catch {
        Write-Log "Health check failed: $_" "ERROR"
        exit 1
    }

    # Check for schema verification in logs
    $logContent = Get-Content "aaos_prod.log" -ErrorAction SilentlyContinue
    if ($logContent -match "Schema verification passed") {
        Write-Log "Schema verification passed"
    }

    # Check for correlation IDs (DEVZEN ENHANCEMENT #7)
    if ($logContent -match "corr-id:") {
        Write-Log "Correlation IDs detected in logs"
    } else {
        Write-Log "No correlation IDs detected - enable structured logging middleware" "WARN"
    }
}

Write-Log "=========================================="
Write-Log "AAOS Production startup complete!"
Write-Log "=========================================="
Write-Log ""
Write-Log "Next steps:"
Write-Log "  1. Start telemetry capture:"
Write-Log "     python scripts/capture_baseline_telemetry.py --duration 24h"
Write-Log ""
Write-Log "  2. Start load simulator:"
Write-Log "     python scripts/production_load_simulator.py --tasks-per-hour 100 --agent-count 5"
Write-Log ""
Write-Log "  3. Run checkpoint gates at T+6h, T+12h, T+18h, T+24h:"
Write-Log "     python scripts/checkpoint_gate.py --checkpoint T6h"
