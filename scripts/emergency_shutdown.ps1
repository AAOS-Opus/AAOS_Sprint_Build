# AAOS Emergency Shutdown Script
# DevZen Enhanced: Graceful shutdown with artifact preservation

param(
    [string]$Reason = "Manual emergency shutdown",
    [switch]$SkipSnapshot
)

$ErrorActionPreference = "Continue"
$ShutdownTime = Get-Date

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss.ffffff"
    $logLine = "[$timestamp] [$Level] $Message"
    Write-Host $logLine -ForegroundColor $(
        switch ($Level) {
            "ERROR" { "Red" }
            "WARN" { "Yellow" }
            default { "White" }
        }
    )
    Add-Content -Path "emergency_shutdown.log" -Value $logLine -ErrorAction SilentlyContinue
}

Write-Log "============================================================"
Write-Log "AAOS EMERGENCY SHUTDOWN INITIATED" "WARN"
Write-Log "============================================================"
Write-Log "Reason: $Reason"
Write-Log "Time: $ShutdownTime"
Write-Log "------------------------------------------------------------"

# Step 1: Stop background Python processes
Write-Log "Step 1: Stopping Python processes..."
$pythonProcesses = Get-Process -Name "python*" -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    foreach ($proc in $pythonProcesses) {
        Write-Log "  Stopping: $($proc.Name) (PID: $($proc.Id))"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Log "  Python processes stopped"
} else {
    Write-Log "  No Python processes found"
}

# Step 2: Stop PowerShell background jobs
Write-Log "Step 2: Stopping PowerShell background jobs..."
$jobs = Get-Job -State Running -ErrorAction SilentlyContinue
if ($jobs) {
    foreach ($job in $jobs) {
        Write-Log "  Stopping job: $($job.Name) (ID: $($job.Id))"
        Stop-Job -Job $job -ErrorAction SilentlyContinue
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }
    Write-Log "  Background jobs stopped"
} else {
    Write-Log "  No running jobs found"
}

# Step 3: Create emergency snapshot (unless skipped)
if (-not $SkipSnapshot) {
    Write-Log "Step 3: Creating emergency snapshot..."

    $snapshotName = "emergency_snapshot_$(Get-Date -Format 'yyyyMMdd_HHmmss').tar.gz"
    $filesToArchive = @()

    # Collect files to archive
    $candidates = @(
        "aaos_prod.log",
        "aaos_prod_error.log",
        "telemetry_baseline_24h.log",
        "baseline_summary.json",
        "checkpoints.log",
        "phase4_orchestration.log",
        "autonomous_progress.log"
    )

    foreach ($file in $candidates) {
        if (Test-Path $file) {
            $filesToArchive += $file
        }
    }

    if ($filesToArchive.Count -gt 0) {
        # Use tar if available, otherwise create zip
        try {
            $tarArgs = "-czvf `"$snapshotName`" " + ($filesToArchive -join " ")
            Invoke-Expression "tar $tarArgs" 2>$null
            Write-Log "  Emergency snapshot created: $snapshotName"
        } catch {
            # Fallback to PowerShell compression
            $zipName = $snapshotName -replace '.tar.gz', '.zip'
            Compress-Archive -Path $filesToArchive -DestinationPath $zipName -Force
            Write-Log "  Emergency snapshot created: $zipName (ZIP fallback)"
        }
    } else {
        Write-Log "  No files to archive" "WARN"
    }
} else {
    Write-Log "Step 3: Snapshot skipped (--SkipSnapshot flag)"
}

# Step 4: Stop Docker containers
Write-Log "Step 4: Stopping Docker containers..."
try {
    $runningContainers = docker ps --format "{{.Names}}" 2>$null
    if ($runningContainers) {
        Write-Log "  Running containers: $($runningContainers -join ', ')"
        docker-compose -f docker-compose.prod.yml down --remove-orphans 2>$null
        Write-Log "  Docker containers stopped"
    } else {
        Write-Log "  No running Docker containers"
    }
} catch {
    Write-Log "  Docker shutdown failed: $_" "ERROR"
}

# Step 5: Generate shutdown report
Write-Log "Step 5: Generating shutdown report..."

$report = @{
    "timestamp" = $ShutdownTime.ToString("o")
    "reason" = $Reason
    "duration_seconds" = (New-TimeSpan -Start $ShutdownTime -End (Get-Date)).TotalSeconds
    "artifacts_preserved" = @()
    "status" = "EMERGENCY_SHUTDOWN"
}

# Check what artifacts exist
$artifactFiles = @(
    "aaos_prod.log",
    "telemetry_baseline_24h.log",
    "baseline_summary.json",
    "checkpoints.log"
)

foreach ($file in $artifactFiles) {
    if (Test-Path $file) {
        $fileInfo = Get-Item $file
        $report.artifacts_preserved += @{
            "name" = $file
            "size_bytes" = $fileInfo.Length
            "last_modified" = $fileInfo.LastWriteTime.ToString("o")
        }
    }
}

$reportJson = $report | ConvertTo-Json -Depth 3
$reportJson | Out-File -FilePath "emergency_shutdown_report.json" -Encoding utf8
Write-Log "  Shutdown report: emergency_shutdown_report.json"

# Final message
Write-Log "------------------------------------------------------------"
Write-Log "EMERGENCY SHUTDOWN COMPLETE" "WARN"
Write-Log "============================================================"
Write-Log ""
Write-Log "Review logs and artifacts before restart:"
Write-Log "  - emergency_shutdown.log"
Write-Log "  - emergency_shutdown_report.json"
if (-not $SkipSnapshot) {
    Write-Log "  - emergency_snapshot_*.tar.gz (or .zip)"
}
Write-Log ""
Write-Log "To restart: .\scripts\start_production.ps1"
