# Build & Deployment Procedures

Comprehensive procedures for building, packaging, and deploying alexdev-video-summarizer.

## Build Pipeline Overview

### Build Process Architecture
```
Source Code → Environment Setup → Dependency Installation → Model Downloads → Testing → Package Creation → Deployment
```

### Build Artifacts Structure
```
alexdev-video-summarizer/
├── build-artifacts/           # Generated during build
│   ├── dist/                 # Distribution packages
│   ├── wheels/               # Python wheel packages
│   └── dependencies/         # Cached dependencies
├── deployment-package/       # Final deployment package
│   ├── alexdev-video-summarizer/
│   ├── install.ps1          # Windows installer script
│   ├── setup.sh             # Linux installer script
│   └── README.md            # Deployment instructions
└── release-notes/           # Version release documentation
```

## Build Procedures

### 1. Pre-Build Validation

#### Environment Verification
```powershell
# Windows Environment Check
python --version          # Must be 3.8+
ffmpeg -version          # Must be installed
nvidia-smi               # Must show GPU
git status               # Must be clean working directory
```

```bash
# Linux Environment Check
python3 --version
ffmpeg -version
nvidia-smi
git status
```

#### Code Quality Gates
```bash
# Run before build
pytest tests/                    # All tests must pass
black --check src/              # Code formatting validation
flake8 src/                     # Linting validation
mypy src/                       # Type checking validation
```

### 2. Build Execution

#### Automated Build Script
```powershell
# build.ps1 - Windows Build Script
param(
    [string]$Version = "1.0.0",
    [switch]$SkipTests = $false
)

Write-Host "🚀 Building alexdev-video-summarizer v$Version"

# 1. Clean previous builds
Remove-Item -Recurse -Force build-artifacts -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path build-artifacts/dist

# 2. Create virtual environment
python -m venv build-env
& build-env\Scripts\activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install build wheel

# 4. Run tests (unless skipped)
if (-not $SkipTests) {
    Write-Host "🧪 Running test suite"
    pytest tests/ --cov=src --cov-report=html
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

# 5. Build wheel package
Write-Host "📦 Building distribution package"
python -m build --wheel --outdir build-artifacts/dist/

# 6. Create deployment package
New-Item -ItemType Directory -Path deployment-package
Copy-Item -Recurse alexdev-video-summarizer deployment-package/
Copy-Item scripts/install.ps1 deployment-package/
Copy-Item README.md deployment-package/

Write-Host "✅ Build complete: deployment-package/"
```

#### Linux Build Script
```bash
#!/bin/bash
# build.sh - Linux Build Script
set -e

VERSION=${1:-"1.0.0"}
SKIP_TESTS=${2:-false}

echo "🚀 Building alexdev-video-summarizer v$VERSION"

# 1. Clean previous builds
rm -rf build-artifacts deployment-package
mkdir -p build-artifacts/dist

# 2. Create virtual environment
python3 -m venv build-env
source build-env/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install build wheel

# 4. Run tests
if [ "$SKIP_TESTS" != "true" ]; then
    echo "🧪 Running test suite"
    pytest tests/ --cov=src --cov-report=html
fi

# 5. Build wheel package
echo "📦 Building distribution package"
python -m build --wheel --outdir build-artifacts/dist/

# 6. Create deployment package
mkdir -p deployment-package
cp -r alexdev-video-summarizer deployment-package/
cp scripts/setup.sh deployment-package/
cp README.md deployment-package/

echo "✅ Build complete: deployment-package/"
```

### 3. Build Artifact Management

#### Artifact Storage Structure
```
build-artifacts/
├── dist/
│   └── alexdev_video_summarizer-1.0.0-py3-none-any.whl
├── test-reports/
│   ├── coverage.html
│   └── test-results.xml
├── dependencies/
│   └── requirements-frozen.txt  # Exact versions used
└── build-metadata.json         # Build information
```

#### Build Metadata Generation
```json
{
    "version": "1.0.0",
    "build_date": "2025-09-04T20:00:00Z",
    "git_commit": "9f2eff7",
    "python_version": "3.11.5",
    "platform": "Windows-10-10.0.22621",
    "dependencies": "requirements-frozen.txt",
    "test_coverage": 95.2,
    "build_duration": 180
}
```

## Deployment Procedures

### 1. Target Environment Preparation

#### System Requirements Validation
```powershell
# Windows Deployment Preparation
function Test-SystemRequirements {
    $requirements = @{
        "Python" = { python --version 2>&1 | Select-String "3\.(8|9|10|11)" }
        "FFmpeg" = { ffmpeg -version 2>&1 | Select-String "ffmpeg" }
        "GPU" = { nvidia-smi 2>&1 | Select-String "NVIDIA" }
        "Disk" = { (Get-WmiObject -Class Win32_LogicalDisk | Where-Object {$_.DriveType -eq 3}).FreeSpace[0] -gt 50GB }
        "Memory" = { (Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory -gt 16GB }
    }
    
    foreach ($req in $requirements.Keys) {
        $result = & $requirements[$req]
        Write-Host "$req : $(if ($result) { '✅ OK' } else { '❌ FAIL' })"
    }
}
```

#### Environment Setup
```powershell
# Create application directory
New-Item -ItemType Directory -Path "C:\alexdev-video-summarizer" -Force
Set-Location "C:\alexdev-video-summarizer"

# Create processing directories
@("input", "output", "build", "models", "logs", "cache") | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force
}
```

### 2. Application Deployment

#### Automated Deployment Script
```powershell
# deploy.ps1 - Windows Deployment Script
param(
    [string]$SourcePath = ".\deployment-package",
    [string]$TargetPath = "C:\alexdev-video-summarizer"
)

Write-Host "🚀 Deploying alexdev-video-summarizer"

# 1. Validate source package
if (-not (Test-Path "$SourcePath\alexdev-video-summarizer")) {
    throw "Invalid deployment package: $SourcePath"
}

# 2. Backup existing installation
if (Test-Path $TargetPath) {
    $backupPath = "$TargetPath.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Write-Host "📦 Backing up to: $backupPath"
    Move-Item $TargetPath $backupPath
}

# 3. Deploy application
Write-Host "📋 Copying application files"
Copy-Item -Recurse "$SourcePath\alexdev-video-summarizer" $TargetPath

# 4. Set up virtual environment
Set-Location $TargetPath
python -m venv venv
& venv\Scripts\activate.ps1

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6. Run setup script
python scripts\setup.py

# 7. Validate installation
Write-Host "🔍 Validating installation"
python src\main.py --help
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Deployment successful"
} else {
    throw "Deployment validation failed"
}
```

### 3. Configuration Management

#### Environment Configuration
```yaml
# config/production.yaml - Production overrides
app:
  environment: "production"
  debug: false
  
logging:
  level: "INFO"
  file_logging: true
  console_logging: false
  
performance:
  target_processing_time: 480  # 8 minutes optimized
  max_processing_time: 1800    # 30 minute hard limit
  
gpu_pipeline:
  max_gpu_memory_usage: 0.95   # Use 95% in production
  
error_handling:
  circuit_breaker_threshold: 3  # Stop after 3 failures
  cleanup_failed_artifacts: true
```

#### Configuration Deployment
```powershell
# Deploy environment-specific configuration
Copy-Item config\production.yaml config\local_processing.yaml

# Validate configuration
python -c "
import yaml
with open('config/local_processing.yaml') as f:
    config = yaml.safe_load(f)
print('✅ Configuration valid')
"
```

## Rollback Procedures

### 1. Application Rollback

#### Automatic Rollback Script
```powershell
# rollback.ps1 - Rollback to previous version
param(
    [string]$TargetPath = "C:\alexdev-video-summarizer"
)

Write-Host "🔄 Rolling back alexdev-video-summarizer"

# 1. Find most recent backup
$backups = Get-ChildItem "$TargetPath.backup.*" | Sort-Object Name -Descending
if ($backups.Count -eq 0) {
    throw "No backup found for rollback"
}

$latestBackup = $backups[0]
Write-Host "📦 Rolling back to: $($latestBackup.Name)"

# 2. Stop any running processes
Get-Process | Where-Object {$_.Name -like "*python*" -and $_.CommandLine -like "*alexdev*"} | Stop-Process -Force

# 3. Replace current installation
Remove-Item -Recurse -Force $TargetPath
Move-Item $latestBackup.FullName $TargetPath

# 4. Validate rollback
Set-Location $TargetPath
& venv\Scripts\activate.ps1
python src\main.py --help

Write-Host "✅ Rollback complete"
```

### 2. Data Recovery

#### Processing State Recovery
```powershell
# recover-processing-state.ps1
param(
    [string]$RecoveryPoint
)

Write-Host "🔧 Recovering processing state from: $RecoveryPoint"

# 1. Restore build artifacts
if (Test-Path "$RecoveryPoint\build") {
    Copy-Item -Recurse "$RecoveryPoint\build\*" "build\"
    Write-Host "📋 Build artifacts restored"
}

# 2. Restore output files
if (Test-Path "$RecoveryPoint\output") {
    Copy-Item -Recurse "$RecoveryPoint\output\*" "output\"
    Write-Host "📄 Output files restored"
}

# 3. Restore processing logs
if (Test-Path "$RecoveryPoint\logs") {
    Copy-Item -Recurse "$RecoveryPoint\logs\*" "logs\"
    Write-Host "📝 Processing logs restored"
}

Write-Host "✅ Recovery complete"
```

## Monitoring and Validation

### 1. Deployment Validation

#### Health Check Script
```python
#!/usr/bin/env python3
# health_check.py - Post-deployment validation
import subprocess
import sys
import torch
import os
from pathlib import Path

def check_python_version():
    """Validate Python version"""
    version = sys.version_info
    if version.major < 3 or version.minor < 8:
        return False, f"Python {version.major}.{version.minor} < 3.8"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"

def check_gpu_availability():
    """Validate GPU and CUDA"""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"{gpu_count} GPU(s): {gpu_name}"
        else:
            return False, "CUDA not available"
    except Exception as e:
        return False, f"GPU check failed: {e}"

def check_ffmpeg():
    """Validate FFmpeg installation"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            return True, version
        else:
            return False, "FFmpeg execution failed"
    except Exception as e:
        return False, f"FFmpeg not found: {e}"

def check_directories():
    """Validate directory structure"""
    required_dirs = ['input', 'output', 'build', 'models', 'logs', 'config', 'src']
    missing = [d for d in required_dirs if not Path(d).exists()]
    if missing:
        return False, f"Missing directories: {missing}"
    return True, "All directories present"

def main():
    """Run complete health check"""
    checks = [
        ("Python Version", check_python_version),
        ("GPU/CUDA", check_gpu_availability), 
        ("FFmpeg", check_ffmpeg),
        ("Directory Structure", check_directories)
    ]
    
    print("🔍 alexdev-video-summarizer Health Check")
    print("=" * 50)
    
    all_passed = True
    for name, check_func in checks:
        try:
            passed, message = check_func()
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{name:20} {status} - {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"{name:20} ❌ ERROR - {e}")
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("✅ All health checks passed - System ready")
        return 0
    else:
        print("❌ Health check failures detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### 2. Performance Monitoring

#### Resource Monitoring Script
```powershell
# monitor-resources.ps1 - Resource utilization monitoring
param(
    [int]$DurationMinutes = 60,
    [int]$IntervalSeconds = 30
)

$endTime = (Get-Date).AddMinutes($DurationMinutes)
$logFile = "logs\resource-monitor-$(Get-Date -Format 'yyyyMMdd-HHmmss').csv"

# CSV Header
"Timestamp,CPU_Percent,Memory_GB,GPU_Utilization,GPU_Memory_GB,Disk_Free_GB" | Out-File $logFile

Write-Host "📊 Monitoring resources for $DurationMinutes minutes"
Write-Host "💾 Log file: $logFile"

while ((Get-Date) -lt $endTime) {
    # CPU Usage
    $cpu = (Get-Counter "\Processor(_Total)\% Processor Time").CounterSamples.CookedValue
    
    # Memory Usage
    $memory = [math]::Round((Get-Counter "\Memory\Available MBytes").CounterSamples.CookedValue / 1024, 2)
    
    # GPU Usage (requires nvidia-ml-py or nvidia-smi parsing)
    $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
    $gpuUtil = $gpu.Split(',')[0].Trim()
    $gpuMem = [math]::Round($gpu.Split(',')[1].Trim() / 1024, 2)
    
    # Disk Space
    $disk = [math]::Round((Get-WmiObject -Class Win32_LogicalDisk -Filter "DriveType=3").FreeSpace[0] / 1GB, 2)
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $entry = "$timestamp,$cpu,$memory,$gpuUtil,$gpuMem,$disk"
    
    $entry | Out-File $logFile -Append
    Write-Host "$(Get-Date -Format 'HH:mm:ss') CPU:$cpu% MEM:$memory GB GPU:$gpuUtil% GPU_MEM:$gpuMem GB"
    
    Start-Sleep $IntervalSeconds
}

Write-Host "✅ Resource monitoring complete"
```

## Quality Assurance Procedures

### 1. Pre-Deployment Testing

#### Integration Test Suite
```bash
#!/bin/bash
# run-integration-tests.sh
set -e

echo "🧪 Running integration test suite"

# 1. Environment validation
python scripts/health_check.py

# 2. Unit tests
echo "Running unit tests..."
pytest tests/unit/ --cov=src --cov-report=term-missing

# 3. Integration tests
echo "Running integration tests..."
pytest tests/integration/ -v

# 4. Performance tests
echo "Running performance tests..."
python tests/performance/benchmark_processing.py

# 5. End-to-end tests
echo "Running end-to-end tests..."
python tests/e2e/test_complete_pipeline.py

echo "✅ All tests passed"
```

### 2. Release Validation

#### Release Checklist
```markdown
# Release Validation Checklist

## Pre-Release
- [ ] All unit tests passing (100% pass rate)
- [ ] Integration tests passing 
- [ ] Performance benchmarks within targets
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] Configuration validated
- [ ] Build artifacts verified

## Deployment
- [ ] Target environment prepared
- [ ] Backup completed
- [ ] Health check passed
- [ ] Performance validation completed
- [ ] Rollback procedure tested

## Post-Release
- [ ] Production smoke tests passed
- [ ] Resource monitoring active
- [ ] Error monitoring configured  
- [ ] User acceptance validation
- [ ] Performance metrics baseline established
```

## Emergency Procedures

### 1. Incident Response

#### Critical Failure Response
```powershell
# emergency-response.ps1 - Critical failure response
param(
    [Parameter(Mandatory)]
    [ValidateSet("GPU_FAILURE", "DISK_FULL", "MEMORY_EXHAUSTED", "PROCESS_HUNG")]
    [string]$IncidentType
)

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$incidentLog = "logs\incident-$timestamp.log"

Write-Host "🚨 EMERGENCY RESPONSE: $IncidentType" | Tee-Object $incidentLog

switch ($IncidentType) {
    "GPU_FAILURE" {
        Write-Host "Stopping all GPU processes..." | Tee-Object $incidentLog -Append
        Get-Process | Where-Object {$_.Name -like "*python*"} | Stop-Process -Force
        Write-Host "Restarting CUDA services..." | Tee-Object $incidentLog -Append
        # Add CUDA service restart commands
    }
    
    "DISK_FULL" {
        Write-Host "Emergency disk cleanup..." | Tee-Object $incidentLog -Append
        Remove-Item -Recurse -Force "build\*" -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force "cache\*" -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force "logs\*.old" -ErrorAction SilentlyContinue
    }
    
    "MEMORY_EXHAUSTED" {
        Write-Host "Force memory cleanup..." | Tee-Object $incidentLog -Append
        Get-Process | Where-Object {$_.WorkingSet -gt 1GB} | Stop-Process -Force
        [System.GC]::Collect()
    }
    
    "PROCESS_HUNG" {
        Write-Host "Terminating hung processes..." | Tee-Object $incidentLog -Append
        Get-Process | Where-Object {$_.Name -like "*alexdev*"} | Stop-Process -Force
    }
}

Write-Host "✅ Emergency response completed" | Tee-Object $incidentLog -Append
```

This comprehensive build and deployment documentation provides step-by-step procedures for all operational scenarios, ensuring reliable deployment and maintenance of the alexdev-video-summarizer system.