# Emergency Procedures & Incident Response

Critical procedures for handling system failures, data recovery, and emergency situations in alexdev-video-summarizer.

## Emergency Response Framework

### Incident Classification

#### Severity Levels
- **P0 - Critical**: Complete system failure, data loss risk, security breach
- **P1 - High**: Major functionality impaired, processing stopped, GPU failures  
- **P2 - Medium**: Performance degradation, partial functionality loss
- **P3 - Low**: Minor issues, cosmetic problems, non-critical errors

#### Response Time SLAs
- **P0 Critical**: Immediate response (0-15 minutes)
- **P1 High**: 1 hour response  
- **P2 Medium**: 4 hour response
- **P3 Low**: Next business day

### Emergency Contact Protocol
1. **Identify incident severity**
2. **Execute appropriate response procedure**
3. **Document all actions taken**
4. **Notify stakeholders based on severity**
5. **Conduct post-incident review**

## Critical System Failures (P0)

### Complete System Failure

#### Symptoms
- Application won't start
- All processing stopped
- Critical errors in logs
- System completely unresponsive

#### Immediate Response Procedure
```powershell
# emergency-system-failure.ps1
Write-Host "🚨 CRITICAL: System Failure Response" -ForegroundColor Red

# 1. Stop all processing immediately
Get-Process | Where-Object {$_.Name -like "*python*" -and $_.CommandLine -like "*alexdev*"} | Stop-Process -Force
Write-Host "✅ All processing stopped"

# 2. Capture system state
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$emergencyDir = "emergency-logs\system-failure-$timestamp"
New-Item -ItemType Directory -Path $emergencyDir -Force

# Copy critical logs
Copy-Item logs\*.log $emergencyDir\
Copy-Item config\*.yaml $emergencyDir\
nvidia-smi > "$emergencyDir\gpu-status.txt"
Get-Process | Out-File "$emergencyDir\process-list.txt"
Get-EventLog -LogName System -Newest 50 | Out-File "$emergencyDir\system-events.txt"

Write-Host "📋 System state captured in: $emergencyDir"

# 3. Check available backups
$backups = Get-ChildItem "*.backup.*" | Sort-Object LastWriteTime -Descending
if ($backups) {
    Write-Host "💾 Available backups:"
    $backups | Select-Object Name, LastWriteTime | Format-Table
} else {
    Write-Host "⚠️  No backups found!"
}

# 4. Attempt basic recovery
Write-Host "🔧 Attempting basic recovery..."

# Clear GPU memory
try {
    nvidia-smi --gpu-reset
    Write-Host "✅ GPU reset successful"
} catch {
    Write-Host "❌ GPU reset failed"
}

# Free up disk space if needed
$freeSpace = (Get-WmiObject -Class Win32_LogicalDisk -Filter "DriveType=3").FreeSpace[0] / 1GB
if ($freeSpace -lt 10) {
    Write-Host "🧹 Emergency disk cleanup..."
    Remove-Item -Recurse -Force build\* -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force temp\* -ErrorAction SilentlyContinue
    Write-Host "✅ Emergency cleanup completed"
}

Write-Host "📞 NEXT STEPS:"
Write-Host "1. Attempt system restart with: restart-system.ps1"
Write-Host "2. If restart fails, initiate rollback: rollback-system.ps1"
Write-Host "3. If rollback fails, contact emergency support"
```

#### System Restart Procedure
```powershell
# restart-system.ps1
Write-Host "🔄 System Restart Procedure"

# 1. Validate environment
if (-not (Test-Path "venv\Scripts\activate.ps1")) {
    Write-Host "❌ Virtual environment corrupted - recreating..."
    Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
    python -m venv venv
    & venv\Scripts\activate.ps1
    pip install -r requirements.txt
}

# 2. Validate configuration
try {
    python -c "import yaml; yaml.safe_load(open('config/processing.yaml'))"
    Write-Host "✅ Configuration valid"
} catch {
    Write-Host "❌ Configuration corrupted - restoring backup"
    Copy-Item config\processing.yaml.backup config\processing.yaml -Force
}

# 3. Test basic functionality
& venv\Scripts\activate.ps1
$testResult = python src\main.py --help 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Basic functionality restored"
    
    # Test with dry run
    python src\main.py --dry-run --input input
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ System restart successful"
    } else {
        Write-Host "❌ Dry run failed - deeper issues present"
    }
} else {
    Write-Host "❌ System restart failed"
    Write-Host "Output: $testResult"
}
```

### Data Loss Prevention

#### Critical Data Backup
```powershell
# emergency-backup.ps1 - Emergency data protection
param(
    [string]$BackupLocation = ".\emergency-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
)

Write-Host "💾 Emergency Data Backup"

New-Item -ItemType Directory -Path $BackupLocation -Force

# 1. Backup critical configuration
Copy-Item -Recurse config $BackupLocation\config
Write-Host "✅ Configuration backed up"

# 2. Backup processing outputs  
if (Test-Path output) {
    Copy-Item -Recurse output $BackupLocation\output
    Write-Host "✅ Output files backed up"
}

# 3. Backup processing state
if (Test-Path build) {
    # Only backup critical processing state, not all artifacts
    $criticalFiles = Get-ChildItem build -Recurse -Include "*.json", "*.log", "*metadata*"
    foreach ($file in $criticalFiles) {
        $relativePath = $file.FullName.Replace((Get-Location).Path, "")
        $targetPath = "$BackupLocation$relativePath"
        $targetDir = Split-Path $targetPath -Parent
        if (-not (Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force
        }
        Copy-Item $file.FullName $targetPath
    }
    Write-Host "✅ Critical processing state backed up"
}

# 4. Backup logs
Copy-Item -Recurse logs $BackupLocation\logs -ErrorAction SilentlyContinue
Write-Host "✅ Logs backed up"

# 5. Create recovery manifest
@{
    BackupDate = (Get-Date).ToString()
    BackupLocation = $BackupLocation
    ConfigFiles = (Get-ChildItem "$BackupLocation\config" -Recurse).Count
    OutputFiles = (Get-ChildItem "$BackupLocation\output" -Recurse -ErrorAction SilentlyContinue).Count
    LogFiles = (Get-ChildItem "$BackupLocation\logs" -Recurse -ErrorAction SilentlyContinue).Count
    RecoveryInstructions = "Use recover-from-backup.ps1 to restore system state"
} | ConvertTo-Json | Out-File "$BackupLocation\recovery-manifest.json"

Write-Host "📋 Emergency backup completed: $BackupLocation"
```

## GPU Failures (P1)

### GPU Memory Exhaustion

#### Detection and Response
```powershell
# gpu-memory-emergency.ps1
Write-Host "🎮 GPU Memory Emergency Response"

# 1. Check GPU status
$gpuStatus = nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
Write-Host "Current GPU Status: $gpuStatus"

# 2. Force stop all GPU processes
Write-Host "🛑 Stopping all GPU processes..."
Get-Process | Where-Object {$_.Name -like "*python*"} | ForEach-Object {
    try {
        $_.Kill()
        Write-Host "Stopped process: $($_.Name) (PID: $($_.Id))"
    } catch {
        Write-Host "Failed to stop process: $($_.Name)"
    }
}

# 3. Clear GPU memory
try {
    nvidia-smi --gpu-reset
    Write-Host "✅ GPU memory cleared"
    
    # Verify memory cleared
    $newStatus = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    Write-Host "GPU memory after reset: $newStatus MB"
    
} catch {
    Write-Host "❌ GPU reset failed - may require system reboot"
}

# 4. Test GPU availability
try {
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    Write-Host "✅ GPU functionality restored"
} catch {
    Write-Host "❌ GPU still not accessible"
}

# 5. Restart with reduced memory settings
Write-Host "🔧 Restarting with conservative memory settings..."
$conservativeConfig = @"
gpu_pipeline:
  max_gpu_memory_usage: 0.7  # Reduced from 0.95
  sequential_processing: true
  memory_cleanup: true
"@

$conservativeConfig | Out-File config\emergency-gpu.yaml
Write-Host "💾 Emergency GPU configuration created"
```

### GPU Driver Issues

#### Driver Recovery Procedure
```powershell
# gpu-driver-emergency.ps1
Write-Host "🚨 GPU Driver Emergency Procedure"

# 1. Diagnose driver status
$driverInfo = nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
if ($LASTEXITCODE -eq 0) {
    Write-Host "Current driver: $driverInfo"
} else {
    Write-Host "❌ NVIDIA driver not responding"
}

# 2. Check Windows device manager for issues
$gpuDevice = Get-PnpDevice | Where-Object {$_.FriendlyName -like "*NVIDIA*" -and $_.Class -eq "Display"}
if ($gpuDevice) {
    Write-Host "GPU Device Status: $($gpuDevice.Status)"
    if ($gpuDevice.Status -ne "OK") {
        Write-Host "⚠️  GPU device has issues"
    }
} else {
    Write-Host "❌ GPU device not found in Device Manager"
}

# 3. Restart NVIDIA services
Write-Host "🔄 Restarting NVIDIA services..."
$nvidiaServices = @("NVDisplay.ContainerLocalSystem", "NVIDIA Display Container LS")
foreach ($service in $nvidiaServices) {
    try {
        Restart-Service $service -Force
        Write-Host "✅ Restarted service: $service"
    } catch {
        Write-Host "❌ Failed to restart service: $service"
    }
}

# 4. Fallback to CPU processing
Write-Host "💡 Configuring CPU fallback mode..."
$cpuFallbackConfig = @"
gpu_pipeline:
  force_cpu_mode: true  # Emergency CPU fallback
cpu_pipeline:
  max_workers: 6        # Increased CPU workers
performance:
  target_processing_time: 1800  # 30 minutes (slower without GPU)
"@

$cpuFallbackConfig | Out-File config\emergency-cpu-fallback.yaml
Write-Host "📝 CPU fallback configuration created"
Write-Host "Use: python src\main.py --config config\emergency-cpu-fallback.yaml"
```

## Disk Space Emergencies (P1)

### Critical Disk Space Recovery
```powershell
# disk-space-emergency.ps1
param(
    [int]$TargetFreeSpaceGB = 50
)

Write-Host "💾 Disk Space Emergency Recovery"

$disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DriveType=3"
$currentFreeGB = [math]::Round($disk.FreeSpace / 1GB, 2)
$totalSizeGB = [math]::Round($disk.Size / 1GB, 2)

Write-Host "Current free space: $currentFreeGB GB / $totalSizeGB GB total"

if ($currentFreeGB -gt $TargetFreeSpaceGB) {
    Write-Host "✅ Sufficient disk space available"
    return
}

$spaceToFree = $TargetFreeSpaceGB - $currentFreeGB
Write-Host "🎯 Need to free: $spaceToFree GB"

# 1. Clear build artifacts (highest priority)
if (Test-Path build) {
    $buildSize = (Get-ChildItem build -Recurse | Measure-Object Length -Sum).Sum / 1GB
    Write-Host "🧹 Clearing build artifacts ($([math]::Round($buildSize, 2)) GB)..."
    Remove-Item -Recurse -Force build\* -ErrorAction SilentlyContinue
    Write-Host "✅ Build artifacts cleared"
}

# 2. Clear old logs
$oldLogs = Get-ChildItem logs -Recurse | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)}
if ($oldLogs) {
    $logSize = ($oldLogs | Measure-Object Length -Sum).Sum / 1GB
    Write-Host "🧹 Clearing old logs ($([math]::Round($logSize, 2)) GB)..."
    $oldLogs | Remove-Item -Force
    Write-Host "✅ Old logs cleared"
}

# 3. Clear temp directories
@("temp", "cache") | ForEach-Object {
    if (Test-Path $_) {
        $tempSize = (Get-ChildItem $_ -Recurse -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB
        Write-Host "🧹 Clearing $_ directory ($([math]::Round($tempSize, 2)) GB)..."
        Remove-Item -Recurse -Force "$_\*" -ErrorAction SilentlyContinue
        Write-Host "✅ $_ directory cleared"
    }
}

# 4. Archive old output files if needed
$newFreeSpace = [math]::Round((Get-WmiObject -Class Win32_LogicalDisk -Filter "DriveType=3").FreeSpace / 1GB, 2)
if ($newFreeSpace -lt $TargetFreeSpaceGB) {
    Write-Host "⚠️  Still need more space - archiving old outputs..."
    
    $archiveDir = "archive-$(Get-Date -Format 'yyyyMMdd')"
    New-Item -ItemType Directory -Path $archiveDir -Force
    
    $oldOutputs = Get-ChildItem output -Recurse | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)}
    if ($oldOutputs) {
        $oldOutputs | Move-Item -Destination $archiveDir
        Write-Host "📦 Old outputs archived to: $archiveDir"
    }
}

# 5. Final space check
$finalFreeSpace = [math]::Round((Get-WmiObject -Class Win32_LogicalDisk -Filter "DriveType=3").FreeSpace / 1GB, 2)
$spaceFreed = $finalFreeSpace - $currentFreeGB

Write-Host "📊 Final status:"
Write-Host "  Space freed: $([math]::Round($spaceFreed, 2)) GB"
Write-Host "  Current free space: $finalFreeSpace GB"

if ($finalFreeSpace -ge $TargetFreeSpaceGB) {
    Write-Host "✅ Emergency disk space recovery successful"
} else {
    Write-Host "❌ Emergency recovery insufficient - manual intervention required"
    Write-Host "Consider:"
    Write-Host "- Moving input videos to external storage"
    Write-Host "- Archiving output files to network storage"
    Write-Host "- Adding additional disk space"
}
```

## Data Recovery Procedures

### Processing State Recovery
```powershell
# recover-processing-state.ps1
param(
    [Parameter(Mandatory)]
    [string]$RecoveryPoint,
    [string]$TargetVideo = ""
)

Write-Host "🔧 Processing State Recovery"

if (-not (Test-Path $RecoveryPoint)) {
    Write-Host "❌ Recovery point not found: $RecoveryPoint"
    return
}

Write-Host "📂 Recovery source: $RecoveryPoint"

# 1. Validate recovery point
$manifest = "$RecoveryPoint\recovery-manifest.json"
if (Test-Path $manifest) {
    $recoveryInfo = Get-Content $manifest | ConvertFrom-Json
    Write-Host "📋 Recovery Info:"
    Write-Host "  Backup Date: $($recoveryInfo.BackupDate)"
    Write-Host "  Config Files: $($recoveryInfo.ConfigFiles)"
    Write-Host "  Output Files: $($recoveryInfo.OutputFiles)"
} else {
    Write-Host "⚠️  No recovery manifest found - proceeding with available data"
}

# 2. Stop any running processing
Get-Process | Where-Object {$_.Name -like "*python*" -and $_.CommandLine -like "*alexdev*"} | Stop-Process -Force
Write-Host "✅ Processing stopped"

# 3. Restore configuration
if (Test-Path "$RecoveryPoint\config") {
    Write-Host "⚙️  Restoring configuration..."
    Copy-Item -Recurse "$RecoveryPoint\config\*" config\ -Force
    Write-Host "✅ Configuration restored"
}

# 4. Restore specific video or all outputs
if ($TargetVideo) {
    $videoOutput = "$RecoveryPoint\output\$TargetVideo"
    if (Test-Path $videoOutput) {
        Copy-Item $videoOutput output\ -Force
        Write-Host "✅ Video output restored: $TargetVideo"
    } else {
        Write-Host "❌ Video output not found in recovery point: $TargetVideo"
    }
} else {
    # Restore all outputs
    if (Test-Path "$RecoveryPoint\output") {
        Copy-Item -Recurse "$RecoveryPoint\output\*" output\ -Force
        Write-Host "✅ All outputs restored"
    }
}

# 5. Restore critical processing state
if (Test-Path "$RecoveryPoint\build") {
    Write-Host "🔄 Restoring processing state..."
    # Only restore critical state files, not all build artifacts
    $criticalFiles = Get-ChildItem "$RecoveryPoint\build" -Recurse -Include "*.json", "*metadata*"
    foreach ($file in $criticalFiles) {
        $relativePath = $file.FullName.Replace($RecoveryPoint, ".")
        $targetDir = Split-Path $relativePath -Parent
        if (-not (Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force
        }
        Copy-Item $file.FullName $relativePath -Force
    }
    Write-Host "✅ Processing state restored"
}

# 6. Validate recovery
Write-Host "🔍 Validating recovery..."
& venv\Scripts\activate.ps1
$validationResult = python src\main.py --help 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ System validation successful"
    
    # Test with dry run
    python src\main.py --dry-run --input input
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Processing validation successful"
        Write-Host "🎉 Recovery completed successfully"
    } else {
        Write-Host "⚠️  Processing validation failed - manual review required"
    }
} else {
    Write-Host "❌ System validation failed:"
    Write-Host $validationResult
}
```

### Configuration Recovery
```powershell
# recover-configuration.ps1
Write-Host "⚙️  Configuration Recovery Procedure"

# 1. Backup current (potentially corrupted) configuration
$backupDir = "config-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Path $backupDir -Force
Copy-Item config\*.yaml $backupDir\ -ErrorAction SilentlyContinue
Write-Host "💾 Current config backed up to: $backupDir"

# 2. Restore from various sources (in order of preference)
$configSources = @(
    "config\processing.yaml.backup",     # Recent backup
    "config\processing.yaml.default",    # Default configuration
    "templates\config\processing.yaml"   # Template configuration
)

$restored = $false
foreach ($source in $configSources) {
    if (Test-Path $source) {
        Write-Host "📂 Restoring from: $source"
        Copy-Item $source config\processing.yaml -Force
        
        # Validate restored configuration
        try {
            python -c "import yaml; yaml.safe_load(open('config/processing.yaml'))"
            Write-Host "✅ Configuration restored and validated"
            $restored = $true
            break
        } catch {
            Write-Host "❌ Configuration from $source is invalid"
        }
    }
}

if (-not $restored) {
    Write-Host "🔧 Creating minimal working configuration..."
    $minimalConfig = @"
# Minimal emergency configuration
app:
  name: "alexdev-video-summarizer"
  environment: "emergency"

paths:
  input_dir: "input"
  output_dir: "output"
  build_dir: "build"

performance:
  target_processing_time: 1800  # 30 minutes (conservative)
  
gpu_pipeline:
  sequential_processing: true
  max_gpu_memory_usage: 0.7     # Conservative setting
  
error_handling:
  circuit_breaker_threshold: 1  # Fail fast in emergency mode
"@
    
    $minimalConfig | Out-File config\processing.yaml
    Write-Host "✅ Minimal configuration created"
}

Write-Host "🔍 Configuration recovery completed"
```

## Rollback Procedures

### Complete System Rollback
```powershell
# complete-system-rollback.ps1
param(
    [string]$RollbackTarget = "auto"  # "auto" finds latest backup
)

Write-Host "🔄 Complete System Rollback"

# 1. Find rollback target
if ($RollbackTarget -eq "auto") {
    $backups = Get-ChildItem "*backup*" -Directory | Sort-Object LastWriteTime -Descending
    if ($backups) {
        $RollbackTarget = $backups[0].FullName
        Write-Host "📂 Auto-selected rollback target: $RollbackTarget"
    } else {
        Write-Host "❌ No backup directories found"
        return
    }
}

if (-not (Test-Path $RollbackTarget)) {
    Write-Host "❌ Rollback target not found: $RollbackTarget"
    return
}

# 2. Stop all processing
Get-Process | Where-Object {$_.Name -like "*python*" -and $_.CommandLine -like "*alexdev*"} | Stop-Process -Force
Write-Host "✅ All processing stopped"

# 3. Create emergency backup of current state
$emergencyBackup = "emergency-backup-before-rollback-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Path $emergencyBackup -Force

@("config", "src", "output") | ForEach-Object {
    if (Test-Path $_) {
        Copy-Item -Recurse $_ "$emergencyBackup\" -Force
    }
}
Write-Host "💾 Emergency backup created: $emergencyBackup"

# 4. Perform rollback
Write-Host "🔄 Rolling back system..."

# Remove current installation
@("src", "config", "venv") | ForEach-Object {
    if (Test-Path $_) {
        Remove-Item -Recurse -Force $_ -ErrorAction SilentlyContinue
    }
}

# Restore from backup
if (Test-Path "$RollbackTarget\alexdev-video-summarizer") {
    Copy-Item -Recurse "$RollbackTarget\alexdev-video-summarizer\*" . -Force
} else {
    Copy-Item -Recurse "$RollbackTarget\*" . -Force
}

Write-Host "✅ Files restored from backup"

# 5. Rebuild environment
Write-Host "🔧 Rebuilding environment..."
python -m venv venv
& venv\Scripts\activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 6. Validate rollback
Write-Host "🔍 Validating rollback..."
$testResult = python src\main.py --help 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Rollback validation successful"
    
    # Test processing capability
    python src\main.py --dry-run --input input
    if ($LASTEXITCODE -eq 0) {
        Write-Host "🎉 Complete system rollback successful"
    } else {
        Write-Host "⚠️  Rollback successful but processing capability needs review"
    }
} else {
    Write-Host "❌ Rollback validation failed:"
    Write-Host $testResult
    Write-Host "💡 Emergency backup available at: $emergencyBackup"
}
```

## Post-Incident Procedures

### Incident Documentation Template
```markdown
# Incident Report Template

**Incident ID**: INC-YYYY-MM-DD-###
**Date/Time**: YYYY-MM-DD HH:MM UTC
**Severity**: P0/P1/P2/P3
**Status**: Open/Resolved/Closed

## Summary
Brief description of the incident and impact.

## Timeline
- **HH:MM** - Incident detected
- **HH:MM** - Response initiated
- **HH:MM** - Root cause identified
- **HH:MM** - Resolution implemented
- **HH:MM** - Service restored
- **HH:MM** - Post-incident review completed

## Impact Assessment
- **Users Affected**: Number/description
- **Services Impacted**: List of affected services
- **Duration**: Total downtime
- **Data Loss**: Any data affected

## Root Cause Analysis
Detailed analysis of what caused the incident.

## Resolution Actions
Step-by-step actions taken to resolve the incident.

## Preventive Measures
Actions to prevent similar incidents in the future.

## Lessons Learned
Key takeaways and improvements identified.

## Follow-up Actions
- [ ] Action item 1 (Owner: Name, Due: Date)
- [ ] Action item 2 (Owner: Name, Due: Date)
```

### Post-Incident Review Process
```powershell
# post-incident-review.ps1
param(
    [Parameter(Mandatory)]
    [string]$IncidentId,
    [string]$IncidentDate = (Get-Date -Format "yyyy-MM-dd")
)

Write-Host "📋 Post-Incident Review: $IncidentId"

$reviewDir = "incident-reviews\$IncidentId"
New-Item -ItemType Directory -Path $reviewDir -Force

# 1. Collect incident artifacts
$artifacts = @(
    "logs\*.log",
    "emergency-logs\*",
    "config\*.yaml",
    "build\*metadata*"
)

foreach ($artifactPattern in $artifacts) {
    $files = Get-ChildItem $artifactPattern -Recurse -ErrorAction SilentlyContinue
    if ($files) {
        Copy-Item $files $reviewDir\ -Force
        Write-Host "📂 Collected: $artifactPattern"
    }
}

# 2. Generate system health report
$healthReport = @"
# System Health Report - Post-Incident
**Generated**: $(Get-Date)
**Incident**: $IncidentId

## System Status
- **GPU Status**: $(nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader)
- **Memory Usage**: $([math]::Round((Get-Process -Name python -ErrorAction SilentlyContinue | Measure-Object WorkingSet -Sum).Sum / 1GB, 2)) GB
- **Disk Space**: $([math]::Round((Get-WmiObject -Class Win32_LogicalDisk -Filter "DriveType=3").FreeSpace / 1GB, 2)) GB free
- **Active Processes**: $(Get-Process | Where-Object {$_.Name -like "*python*"} | Measure-Object).Count Python processes

## Configuration Status
- **Config Valid**: $(try { python -c "import yaml; yaml.safe_load(open('config/processing.yaml'))"; "✅ Yes" } catch { "❌ No" })
- **Environment**: $(if (Test-Path "venv\Scripts\activate.ps1") { "✅ Valid" } else { "❌ Missing" })

## Recommendations
Based on incident analysis, consider:
1. Enhanced monitoring for early detection
2. Improved error handling for this scenario
3. Additional backup procedures if data was at risk
"@

$healthReport | Out-File "$reviewDir\system-health-post-incident.md"
Write-Host "📊 System health report generated"

# 3. Create incident report template
$incidentReport = Get-Content "templates\incident-report-template.md" -ErrorAction SilentlyContinue
if ($incidentReport) {
    $incidentReport -replace "INC-YYYY-MM-DD-###", $IncidentId | Out-File "$reviewDir\incident-report.md"
    Write-Host "📝 Incident report template created"
}

Write-Host "✅ Post-incident review materials ready in: $reviewDir"
Write-Host "📋 Next steps:"
Write-Host "1. Complete incident-report.md"
Write-Host "2. Schedule team review meeting"
Write-Host "3. Implement preventive measures"
Write-Host "4. Update emergency procedures if needed"
```

This comprehensive emergency procedures documentation provides systematic approaches for handling critical system failures, ensuring rapid recovery and minimal data loss for the alexdev-video-summarizer system.