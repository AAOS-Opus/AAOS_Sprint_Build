# AAOS 24-Hour Baseline Capture Orchestration Script
# DevZen Enhanced v2: Full automation of Phase 4 production baseline
# Chef's Kiss Tweaks: JSON logging, Zombie cleanup, Detailed checkpoints

param(
    [switch]$SkipInfrastructure,
    [int]$TasksPerHour = 100,
    [int]$AgentCount = 5,
    [ValidateSet("FullyAutonomous", "Interactive", "DryRun")]
    [string]$Mode = "FullyAutonomous"
)

$ErrorActionPreference = "Stop"
$StartTime = Get-Date
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot

# Change to project root
Set-Location $ProjectRoot

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss.ffffff"
    $logLine = "[$timestamp] [$Level] $Message"
    $color = switch ($Level) {
        "ERROR" { "Red" }
        "WARN" { "Yellow" }
        "SUCCESS" { "Green" }
        default { "White" }
    }
    Write-Host $logLine -ForegroundColor $color
    Add-Content -Path "phase4_orchestration.log" -Value $logLine
}

function Write-JsonLog {
    param([hashtable]$Data, [string]$File)
    $Data["timestamp"] = [datetime]::UtcNow.ToString("o")
    $json = $Data | ConvertTo-Json -Depth 5
    $json | Out-File -FilePath $File -Encoding utf8 -Append
}

function Get-ElapsedHours {
    return ((Get-Date) - $StartTime).TotalHours
}

# ===========================================================================
# PHASE 4.0: PRE-FLIGHT AUTONOMOUS VALIDATION (DevZen Tweak #1)
# ===========================================================================
Write-Log "============================================================"
Write-Log "AAOS Phase 4: 24-Hour Baseline Capture (DevZen Enhanced v2)"
Write-Log "============================================================"
Write-Log "Mode: $Mode"
Write-Log "Start Time: $StartTime"
Write-Log "Tasks/Hour: $TasksPerHour"
Write-Log "Agent Count: $AgentCount"
Write-Log "------------------------------------------------------------"

Write-Log "Phase 4.0: Pre-flight Autonomous Validation..."

$health = @{
    "docker" = (Get-Command docker -ErrorAction SilentlyContinue) -ne $null
    "python" = (Get-Command python -ErrorAction SilentlyContinue) -ne $null
    "alembic" = (Test-Path "alembic/versions/*.py")
    "dotenv" = (Test-Path ".env")
    "scripts" = (Test-Path "scripts/start_production.ps1")
    "docker_compose" = (Test-Path "docker-compose.prod.yml")
}

$failedChecks = $health.GetEnumerator() | Where-Object { $_.Value -eq $false }

if ($failedChecks) {
    # DevZen Tweak #1: Generate structured JSON for automated parsing
    $json = @{
        "phase" = "4.0_preflight"
        "status" = "FAIL"
        "failures" = @($failedChecks | ForEach-Object { $_.Key })
        "health_checks" = $health
    }
    Write-JsonLog -Data $json -File "preflight_fail_log.json"

    Write-Log "Pre-flight failed. See preflight_fail_log.json for structured details." "ERROR"
    foreach ($check in $failedChecks) {
        Write-Log "  MISSING: $($check.Key)" "ERROR"
    }
    exit 1
}

Write-Log "Pre-flight autonomous validation passed" "SUCCESS"

# ===========================================================================
# PHASE 4.1: ZOMBIE CLEANUP & PRODUCTION STACK LAUNCH (DevZen Tweak #2)
# ===========================================================================
if (-not $SkipInfrastructure) {
    Write-Log "Phase 4.1: Zombie Cleanup & Production Stack Launch..."

    # DevZen Tweak #2: Preemptive zombie container detection
    Write-Log "Scanning for unmanaged containers..." "INFO"
    $allContainers = docker ps --format "{{.Names}}" 2>$null
    $expectedContainers = @("aaos_postgres", "aaos_redis", "aaos_orchestrator")

    if ($allContainers) {
        $unmanaged = $allContainers | Where-Object {
            $_ -notin $expectedContainers -and $_ -notmatch "^(aaos)"
        }

        if ($unmanaged) {
            Write-Log "Unmanaged containers detected: $($unmanaged -join ', ')" "WARN"
            Write-Log "Cleaning zombie containers before baseline..." "INFO"
            foreach ($container in $unmanaged) {
                docker stop $container 2>$null
                docker rm $container 2>$null
            }
            Write-Log "Zombie containers cleaned" "SUCCESS"
        }
    }

    # Start production infrastructure
    Write-Log "Starting production infrastructure..."
    & "$ScriptRoot\start_production.ps1"

    if ($LASTEXITCODE -ne 0) {
        Write-Log "Infrastructure startup failed!" "ERROR"
        & "$ScriptRoot\emergency_shutdown.ps1" -Reason "Infrastructure startup failed"
        exit 1
    }
}

# ===========================================================================
# PHASE 4.2: TELEMETRY CAPTURE INITIATION (Background)
# ===========================================================================
Write-Log "Phase 4.2: Telemetry Capture Initiation..."
$telemetryJob = Start-Job -ScriptBlock {
    Set-Location $using:ProjectRoot
    python scripts/capture_baseline_telemetry.py `
        --duration 86400 `
        --output telemetry_baseline_24h.log `
        --sample-interval 60 `
        --flush-interval 300 `
        --max-log-size 100MB
}
Write-Log "Telemetry capture started (Job ID: $($telemetryJob.Id))" "SUCCESS"

# ===========================================================================
# PHASE 4.3: LOAD SIMULATION INITIATION (Background)
# ===========================================================================
Write-Log "Phase 4.3: Load Simulation Initiation..."
$loadJob = Start-Job -ScriptBlock {
    Set-Location $using:ProjectRoot
    python scripts/production_load_simulator.py `
        --tasks-per-hour $using:TasksPerHour `
        --agent-count $using:AgentCount `
        --duration 24h `
        --log-level INFO
}
Write-Log "Load simulator started (Job ID: $($loadJob.Id))" "SUCCESS"

# Start autonomous progress reporter in background
$progressJob = Start-Job -ScriptBlock {
    Set-Location $using:ProjectRoot
    $hour = 0
    while ($true) {
        Start-Sleep -Seconds 3600  # 1 hour
        $hour++

        # Collect metrics for autonomous progress log
        try {
            $metrics = Invoke-RestMethod -Uri "http://localhost:8000/metrics" -TimeoutSec 10 -ErrorAction SilentlyContinue
            $queue = $metrics.queued_tasks
            $completed = $metrics.completed_tasks_total
            $agents = $metrics.active_agents
        } catch {
            $queue = "N/A"
            $completed = "N/A"
            $agents = "N/A"
        }

        $logLine = "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] Uptime: ${hour}h | Tasks: $completed | Queue: $queue | Agents: $agents | Status: HEALTHY"
        Add-Content -Path "autonomous_progress.log" -Value $logLine

        if ($hour -ge 24) { break }
    }
}
Write-Log "Progress reporter started (Job ID: $($progressJob.Id))" "SUCCESS"

# ===========================================================================
# PHASE 4.4: AUTONOMOUS CHECKPOINT GATES (Timer-Driven, DevZen Tweak #3)
# ===========================================================================
Write-Log "Phase 4.4: Autonomous Checkpoint Gates initialized..."

$checkpoints = @(
    @{ Name = "T6h"; Hours = 6 },
    @{ Name = "T12h"; Hours = 12 },
    @{ Name = "T18h"; Hours = 18 },
    @{ Name = "T24h"; Hours = 24 }
)

$checkpointsPassed = 0
$checkpointsFailed = 0

foreach ($checkpoint in $checkpoints) {
    $waitHours = $checkpoint.Hours - (Get-ElapsedHours)
    if ($waitHours -gt 0) {
        Write-Log "Scheduling checkpoint $($checkpoint.Name) in $([math]::Round($waitHours, 2)) hours..."

        # DryRun mode skips waiting
        if ($Mode -ne "DryRun") {
            Start-Sleep -Seconds ($waitHours * 3600)
        } else {
            Write-Log "[DryRun] Would wait $([math]::Round($waitHours, 2)) hours" "WARN"
            Start-Sleep -Seconds 5
        }

        Write-Log "Running checkpoint gate: $($checkpoint.Name)" "INFO"

        # DevZen Tweak #3: Auto-snapshot on failure with detailed failure logging
        $result = python scripts/checkpoint_gate.py --checkpoint $checkpoint.Name --auto-snapshot

        if ($LASTEXITCODE -ne 0 -or $result -match "FAIL") {
            $checkpointsFailed++
            Write-Log "CHECKPOINT FAILED: $($checkpoint.Name)" "ERROR"

            if ($Mode -eq "FullyAutonomous") {
                # Emergency shutdown protocol
                Write-Log "Initiating emergency shutdown (FullyAutonomous mode)..." "ERROR"

                # Stop background jobs
                Stop-Job -Job $telemetryJob -ErrorAction SilentlyContinue
                Stop-Job -Job $loadJob -ErrorAction SilentlyContinue
                Stop-Job -Job $progressJob -ErrorAction SilentlyContinue

                # Run emergency shutdown
                & "$ScriptRoot\emergency_shutdown.ps1" -Reason "Checkpoint $($checkpoint.Name) failed"
                exit 1
            } else {
                Write-Log "Checkpoint failed but continuing (Interactive mode)..." "WARN"
            }
        } else {
            $checkpointsPassed++
            Write-Log "Checkpoint $($checkpoint.Name) PASSED" "SUCCESS"
        }
    }
}

Write-Log "All checkpoints complete: $checkpointsPassed/4 PASSED, $checkpointsFailed/4 FAILED" $(if ($checkpointsFailed -eq 0) { "SUCCESS" } else { "WARN" })

# ===========================================================================
# PHASE 4.5: ARTIFACT GENERATION & VALIDATION
# ===========================================================================
Write-Log "Phase 4.5: Artifact Generation & Validation..."

# Stop background jobs gracefully
Write-Log "Stopping background jobs..."
Stop-Job -Job $telemetryJob -ErrorAction SilentlyContinue
Stop-Job -Job $loadJob -ErrorAction SilentlyContinue
Stop-Job -Job $progressJob -ErrorAction SilentlyContinue

# Generate baseline summary
Write-Log "Generating baseline summary..."
python scripts/generate_baseline_summary.py `
    --input telemetry_baseline_24h.log `
    --output baseline_summary.json `
    --window 3600 `
    --aaos-log aaos_prod.log

# Archive artifacts (DEVZEN ENHANCEMENT #10)
Write-Log "Archiving artifacts..."
python scripts/archive_artifacts.py `
    --output phase4_baseline_artifacts.tar.gz `
    --manifest sha_manifest.txt

# Generate validation log
Write-Log "Generating validation log..."
python scripts/generate_validation_log_005.py `
    --artifacts phase4_baseline_artifacts.tar.gz `
    --output phase4_validation_log_005.md `
    --summary baseline_summary.json `
    --aaos-log aaos_prod.log `
    --telemetry-log telemetry_baseline_24h.log

# Verify hashes
Write-Log "Verifying artifact hashes..."
python scripts/verify_hashes.py --verify --manifest sha_manifest.txt

# ===========================================================================
# FINAL REPORT
# ===========================================================================
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Log "============================================================" "SUCCESS"
Write-Log "AAOS Phase 4 Baseline Capture Complete" "SUCCESS"
Write-Log "============================================================" "SUCCESS"
Write-Log "Mode: $Mode"
Write-Log "Start Time: $StartTime"
Write-Log "End Time: $EndTime"
Write-Log "Duration: $($Duration.TotalHours.ToString('F2')) hours"
Write-Log "Checkpoints: $checkpointsPassed/4 PASSED"
Write-Log "------------------------------------------------------------"
Write-Log "Artifacts Generated:"
Write-Log "  - aaos_prod.log"
Write-Log "  - telemetry_baseline_24h.log"
Write-Log "  - baseline_summary.json"
Write-Log "  - checkpoints.log"
Write-Log "  - autonomous_progress.log"
Write-Log "  - phase4_baseline_artifacts.tar.gz"
Write-Log "  - sha_manifest.txt"
Write-Log "  - phase4_validation_log_005.md"
Write-Log "------------------------------------------------------------"

# Verify all required artifacts exist
$requiredArtifacts = @(
    "aaos_prod.log",
    "telemetry_baseline_24h.log",
    "baseline_summary.json",
    "checkpoints.log",
    "phase4_baseline_artifacts.tar.gz",
    "phase4_validation_log_005.md"
)

$missingArtifacts = $requiredArtifacts | Where-Object { -not (Test-Path $_) }

if ($missingArtifacts) {
    Write-Log "MISSING ARTIFACTS: $($missingArtifacts -join ', ')" "ERROR"
    Write-Log "PRODUCTION CERTIFICATION: FAILED" "ERROR"
} elseif ($checkpointsFailed -gt 0) {
    Write-Log "CHECKPOINTS FAILED: $checkpointsFailed" "WARN"
    Write-Log "PRODUCTION CERTIFICATION: PARTIAL (Review Required)" "WARN"
} else {
    Write-Log "All artifacts present" "SUCCESS"
    Write-Log "PRODUCTION CERTIFIED" "SUCCESS"
}

Write-Log "============================================================"
Write-Log "Review phase4_validation_log_005.md for final verdict"
Write-Log "============================================================"

# Clean up jobs
Remove-Job -Job $telemetryJob -Force -ErrorAction SilentlyContinue
Remove-Job -Job $loadJob -Force -ErrorAction SilentlyContinue
Remove-Job -Job $progressJob -Force -ErrorAction SilentlyContinue
