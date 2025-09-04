# Maintenance & Monitoring Procedures

Comprehensive maintenance tasks, performance monitoring, and system optimization procedures for alexdev-video-summarizer.

## Regular Maintenance Schedule

### Daily Maintenance Tasks

#### System Health Check (5 minutes)
```powershell
# daily-health-check.ps1 - Run every morning
Write-Host "🌅 Daily Health Check - $(Get-Date -Format 'yyyy-MM-dd')"

# 1. Check processing logs for errors
$errorCount = (Select-String -Path "logs\*.log" -Pattern "ERROR|FAILED" -SimpleMatch | Measure-Object).Count
Write-Host "📝 Error count (24h): $errorCount"

# 2. Check disk space
$diskSpace = [math]::Round((Get-WmiObject -Class Win32_LogicalDisk -Filter "DriveType=3").FreeSpace[0] / 1GB, 2)
Write-Host "💾 Free disk space: $diskSpace GB"
if ($diskSpace -lt 100) { Write-Warning "⚠️  Low disk space!" }

# 3. Check GPU status
$gpuStatus = & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
Write-Host "🎮 GPU Status: $gpuStatus"

# 4. Check recent processing activity
$recentOutputs = (Get-ChildItem output\*.md -ErrorAction SilentlyContinue | Where-Object {$_.LastWriteTime -gt (Get-Date).AddDays(-1)}).Count
Write-Host "📊 Videos processed (24h): $recentOutputs"

# 5. Validate configuration files
try {
    python -c "import yaml; yaml.safe_load(open('config/processing.yaml'))"
    Write-Host "⚙️  Configuration: ✅ Valid"
} catch {
    Write-Host "⚙️  Configuration: ❌ Invalid"
}
```

#### Log Analysis and Rotation
```powershell
# daily-log-maintenance.ps1
$today = Get-Date -Format "yyyy-MM-dd"

# 1. Analyze processing performance
Write-Host "📈 Processing Performance Analysis"
$logFiles = Get-ChildItem logs\processing*.log
foreach ($log in $logFiles) {
    $processingTimes = Select-String -Path $log.FullName -Pattern "Processing time: (\d+\.?\d*) minutes" -AllMatches
    if ($processingTimes) {
        $avgTime = ($processingTimes.Matches.Groups[1].Value | Measure-Object -Average).Average
        Write-Host "  Average processing time: $([math]::Round($avgTime, 2)) minutes"
    }
}

# 2. Check for error patterns
$commonErrors = Select-String -Path "logs\*.log" -Pattern "ERROR" | 
    Group-Object {$_.Line.Split(':')[2].Trim()} | 
    Sort-Object Count -Descending | 
    Select-Object -First 5
    
if ($commonErrors) {
    Write-Host "🚨 Most common errors:"
    $commonErrors | ForEach-Object { Write-Host "  $($_.Count)x: $($_.Name)" }
}

# 3. Rotate large log files
Get-ChildItem logs\*.log | Where-Object {$_.Length -gt 50MB} | ForEach-Object {
    $archiveName = "logs\archive\$($_.BaseName)-$today.log"
    Move-Item $_.FullName $archiveName
    Write-Host "📦 Archived: $($_.Name) → $archiveName"
}
```

### Weekly Maintenance Tasks

#### Performance Optimization (30 minutes)
```powershell
# weekly-optimization.ps1
Write-Host "🚀 Weekly Performance Optimization"

# 1. Clean build artifacts older than 7 days
$oldArtifacts = Get-ChildItem build -Recurse | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)}
if ($oldArtifacts) {
    $oldArtifacts | Remove-Item -Recurse -Force
    Write-Host "🧹 Cleaned $($oldArtifacts.Count) old build artifacts"
}

# 2. Optimize model cache
$modelCache = Get-ChildItem models -Recurse | Where-Object {$_.LastAccessTime -lt (Get-Date).AddDays(-30)}
if ($modelCache) {
    Write-Host "📦 Found $($modelCache.Count) unused model files (30+ days)"
    # Archive rather than delete for safety
    $archiveDir = "models\archive\$(Get-Date -Format 'yyyy-MM-dd')"
    New-Item -ItemType Directory -Path $archiveDir -Force
    $modelCache | Move-Item -Destination $archiveDir
    Write-Host "📁 Archived to: $archiveDir"
}

# 3. Database cleanup (if implemented)
# Placeholder for future database maintenance

# 4. Memory fragmentation check
$memoryUsage = Get-Counter "\Memory\Available MBytes"
Write-Host "💾 Available Memory: $($memoryUsage.CounterSamples.CookedValue) MB"

# 5. GPU memory optimization
& nvidia-smi --gpu-reset  # Reset GPU if idle
Write-Host "🎮 GPU memory cleared"
```

#### System Updates and Dependencies
```powershell
# weekly-updates.ps1
Write-Host "🔄 Weekly System Updates"

# 1. Check Python package updates
& venv\Scripts\activate.ps1
$outdated = pip list --outdated --format=json | ConvertFrom-Json
if ($outdated) {
    Write-Host "📦 Outdated packages found:"
    $outdated | ForEach-Object { Write-Host "  $($_.name): $($_.version) → $($_.latest_version)" }
    
    # Update non-critical packages (exclude ML libraries for stability)
    $safeUpdates = $outdated | Where-Object {$_.name -notin @('torch', 'ultralytics', 'whisperx')}
    if ($safeUpdates) {
        Write-Host "🔧 Updating safe packages..."
        $safeUpdates | ForEach-Object { pip install --upgrade $_.name }
    }
}

# 2. Check NVIDIA driver updates
$currentDriver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits).Trim()
Write-Host "🎮 Current NVIDIA Driver: $currentDriver"
# Manual check recommended - auto-update can be risky

# 3. Windows Updates check (if applicable)
if (Get-Module -ListAvailable -Name PSWindowsUpdate) {
    Import-Module PSWindowsUpdate
    $updates = Get-WUList
    if ($updates) {
        Write-Host "🖥️  Windows Updates available: $($updates.Count)"
    } else {
        Write-Host "🖥️  Windows Updates: System up to date"
    }
}
```

### Monthly Maintenance Tasks

#### Comprehensive System Audit (1 hour)
```powershell
# monthly-audit.ps1
$auditDate = Get-Date -Format "yyyy-MM-dd"
$auditReport = "logs\audit-$auditDate.md"

@"
# Monthly System Audit Report
**Date**: $auditDate
**System**: alexdev-video-summarizer

## Performance Summary
"@ | Out-File $auditReport

# 1. Processing statistics
$monthlyStats = @{
    VideosProcessed = (Get-ChildItem output\*.md | Where-Object {$_.CreationTime -gt (Get-Date).AddDays(-30)}).Count
    TotalProcessingTime = 0  # Calculate from logs
    AverageProcessingTime = 0
    SuccessRate = 0
    ErrorCount = 0
}

# Calculate from logs
$processingLogs = Get-ChildItem logs\processing*.log | Where-Object {$_.CreationTime -gt (Get-Date).AddDays(-30)}
foreach ($log in $processingLogs) {
    $successes = (Select-String -Path $log.FullName -Pattern "Processing completed successfully").Count
    $errors = (Select-String -Path $log.FullName -Pattern "Processing failed").Count
    $monthlyStats.ErrorCount += $errors
}

$monthlyStats.SuccessRate = if ($monthlyStats.VideosProcessed -gt 0) {
    [math]::Round((($monthlyStats.VideosProcessed - $monthlyStats.ErrorCount) / $monthlyStats.VideosProcessed) * 100, 2)
} else { 0 }

@"
- **Videos Processed**: $($monthlyStats.VideosProcessed)
- **Success Rate**: $($monthlyStats.SuccessRate)%
- **Error Count**: $($monthlyStats.ErrorCount)

## Resource Utilization
"@ | Out-File $auditReport -Append

# 2. Resource analysis
$diskUsage = @{
    Input = [math]::Round((Get-ChildItem input -Recurse -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 2)
    Output = [math]::Round((Get-ChildItem output -Recurse -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 2)
    Build = [math]::Round((Get-ChildItem build -Recurse -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 2)
    Models = [math]::Round((Get-ChildItem models -Recurse -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 2)
    Logs = [math]::Round((Get-ChildItem logs -Recurse -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 2)
}

@"
- **Input Directory**: $($diskUsage.Input) GB
- **Output Directory**: $($diskUsage.Output) GB
- **Build Artifacts**: $($diskUsage.Build) GB
- **Models**: $($diskUsage.Models) GB
- **Logs**: $($diskUsage.Logs) GB

## Recommendations
"@ | Out-File $auditReport -Append

# 3. Generate recommendations
$recommendations = @()
if ($diskUsage.Build -gt 10) { $recommendations += "- Clean old build artifacts (${$diskUsage.Build} GB)" }
if ($diskUsage.Logs -gt 5) { $recommendations += "- Archive old logs (${$diskUsage.Logs} GB)" }
if ($monthlyStats.SuccessRate -lt 95) { $recommendations += "- Investigate processing failures (${$monthlyStats.SuccessRate}% success rate)" }

if ($recommendations) {
    $recommendations | Out-File $auditReport -Append
} else {
    "- System operating optimally" | Out-File $auditReport -Append
}

Write-Host "📋 Monthly audit complete: $auditReport"
```

## Performance Monitoring

### Real-Time Monitoring Dashboard

#### System Resource Monitor
```python
#!/usr/bin/env python3
# realtime-monitor.py - Real-time system monitoring
import psutil
import GPUtil
import time
import json
from datetime import datetime
from pathlib import Path

class SystemMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self.metrics_file = Path("logs/realtime-metrics.jsonl")
        
    def collect_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # GPU metrics
            gpus = GPUtil.getGPUs()
            gpu_metrics = []
            for gpu in gpus:
                gpu_metrics.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
            
            # Process metrics
            python_processes = [p for p in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']) 
                              if 'python' in p.info['name'].lower()]
            
            process_metrics = []
            for proc in python_processes:
                try:
                    process_metrics.append({
                        'pid': proc.info['pid'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                        'cpu_percent': proc.info['cpu_percent']
                    })
                except:
                    continue
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory': {
                    'used_gb': memory.used / 1024 / 1024 / 1024,
                    'available_gb': memory.available / 1024 / 1024 / 1024,
                    'percent': memory.percent
                },
                'disk': {
                    'used_gb': disk.used / 1024 / 1024 / 1024,
                    'free_gb': disk.free / 1024 / 1024 / 1024,
                    'percent': (disk.used / disk.total) * 100
                },
                'gpu': gpu_metrics,
                'processes': process_metrics
            }
        except Exception as e:
            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}
    
    def run(self, duration_minutes=60):
        """Run monitoring for specified duration"""
        end_time = time.time() + (duration_minutes * 60)
        
        print(f"📊 Starting system monitoring for {duration_minutes} minutes")
        print(f"💾 Metrics logged to: {self.metrics_file}")
        
        while time.time() < end_time:
            metrics = self.collect_metrics()
            
            # Log to file
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            # Console output
            if 'error' not in metrics:
                gpu_info = f"GPU: {metrics['gpu'][0]['utilization']:.1f}%" if metrics['gpu'] else "GPU: N/A"
                print(f"{datetime.now().strftime('%H:%M:%S')} "
                      f"CPU: {metrics['cpu_percent']:.1f}% "
                      f"MEM: {metrics['memory']['percent']:.1f}% "
                      f"{gpu_info} "
                      f"DISK: {metrics['disk']['percent']:.1f}%")
            else:
                print(f"Error collecting metrics: {metrics['error']}")
            
            time.sleep(self.interval)
        
        print("✅ Monitoring complete")

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.run()
```

#### Processing Performance Monitor
```python
#!/usr/bin/env python3
# processing-monitor.py - Monitor video processing performance
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

class ProcessingMonitor:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        
    def analyze_processing_performance(self, days=7):
        """Analyze processing performance over specified days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        metrics = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'processing_times': [],
            'error_patterns': defaultdict(int),
            'daily_stats': defaultdict(lambda: {'count': 0, 'success': 0, 'avg_time': 0})
        }
        
        # Process log files
        log_files = self.logs_dir.glob("processing*.log")
        for log_file in log_files:
            if log_file.stat().st_mtime < start_date.timestamp():
                continue
                
            self._process_log_file(log_file, metrics, start_date, end_date)
        
        return self._generate_performance_report(metrics)
    
    def _process_log_file(self, log_file, metrics, start_date, end_date):
        """Process individual log file"""
        with open(log_file) as f:
            for line in f:
                # Parse timestamp
                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if not timestamp_match:
                    continue
                    
                timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                if timestamp < start_date or timestamp > end_date:
                    continue
                
                day_key = timestamp.strftime('%Y-%m-%d')
                
                # Processing completion
                if 'Processing completed successfully' in line:
                    metrics['successful_videos'] += 1
                    metrics['daily_stats'][day_key]['success'] += 1
                    
                    # Extract processing time
                    time_match = re.search(r'Processing time: (\d+\.?\d*) minutes', line)
                    if time_match:
                        processing_time = float(time_match.group(1))
                        metrics['processing_times'].append(processing_time)
                
                # Processing failures
                elif 'Processing failed' in line:
                    metrics['failed_videos'] += 1
                    
                    # Extract error pattern
                    error_match = re.search(r'Error: (.+)', line)
                    if error_match:
                        error_type = error_match.group(1).split(':')[0]
                        metrics['error_patterns'][error_type] += 1
                
                # Count total videos
                elif 'Starting processing:' in line:
                    metrics['total_videos'] += 1
                    metrics['daily_stats'][day_key]['count'] += 1
    
    def _generate_performance_report(self, metrics):
        """Generate performance report"""
        if not metrics['processing_times']:
            return "No processing data found in specified time range."
        
        avg_time = sum(metrics['processing_times']) / len(metrics['processing_times'])
        success_rate = (metrics['successful_videos'] / metrics['total_videos'] * 100) if metrics['total_videos'] > 0 else 0
        
        report = f"""
🎬 Video Processing Performance Report
{'='*50}

📊 Summary Statistics:
  • Total Videos Processed: {metrics['total_videos']}
  • Successful: {metrics['successful_videos']} ({success_rate:.1f}%)
  • Failed: {metrics['failed_videos']}
  • Average Processing Time: {avg_time:.2f} minutes
  • Min/Max Processing Time: {min(metrics['processing_times']):.2f} / {max(metrics['processing_times']):.2f} minutes

🚨 Top Error Patterns:
"""
        
        for error, count in sorted(metrics['error_patterns'].items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"  • {error}: {count} occurrences\n"
        
        report += "\n📅 Daily Breakdown:\n"
        for day, stats in sorted(metrics['daily_stats'].items()):
            success_rate_day = (stats['success'] / stats['count'] * 100) if stats['count'] > 0 else 0
            report += f"  • {day}: {stats['count']} videos, {success_rate_day:.1f}% success\n"
        
        return report

if __name__ == "__main__":
    monitor = ProcessingMonitor()
    report = monitor.analyze_processing_performance()
    print(report)
    
    # Save report
    report_file = Path(f"logs/performance-report-{datetime.now().strftime('%Y-%m-%d')}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\n📋 Report saved to: {report_file}")
```

### Alerting System

#### Alert Configuration
```yaml
# config/alerts.yaml - Monitoring alerts configuration
alerts:
  disk_space:
    warning_threshold_gb: 50
    critical_threshold_gb: 20
    check_interval_minutes: 30
    
  memory_usage:
    warning_threshold_percent: 80
    critical_threshold_percent: 95
    check_interval_minutes: 5
    
  gpu_temperature:
    warning_threshold_celsius: 80
    critical_threshold_celsius: 90
    check_interval_minutes: 10
    
  processing_failures:
    consecutive_failures_threshold: 3
    time_window_minutes: 60
    
  error_rate:
    warning_threshold_percent: 10
    critical_threshold_percent: 25
    time_window_hours: 24

notification:
  methods:
    - console
    - log_file
    # - email (future implementation)
    # - slack (future implementation)
    
  log_file: "logs/alerts.log"
```

#### Alert Monitoring Script
```python
#!/usr/bin/env python3
# alert-monitor.py - System alerting
import psutil
import GPUtil
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path

class AlertMonitor:
    def __init__(self, config_file="config/alerts.yaml"):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        
        self.alerts_log = Path(self.config['notification']['log_file'])
        self.alerts_log.parent.mkdir(exist_ok=True)
    
    def check_disk_space(self):
        """Check disk space alerts"""
        disk = psutil.disk_usage('.')
        free_gb = disk.free / 1024 / 1024 / 1024
        
        if free_gb < self.config['alerts']['disk_space']['critical_threshold_gb']:
            self.send_alert('CRITICAL', 'DISK_SPACE', f'Critical disk space: {free_gb:.1f} GB remaining')
        elif free_gb < self.config['alerts']['disk_space']['warning_threshold_gb']:
            self.send_alert('WARNING', 'DISK_SPACE', f'Low disk space: {free_gb:.1f} GB remaining')
    
    def check_memory_usage(self):
        """Check memory usage alerts"""
        memory = psutil.virtual_memory()
        
        if memory.percent > self.config['alerts']['memory_usage']['critical_threshold_percent']:
            self.send_alert('CRITICAL', 'MEMORY_USAGE', f'Critical memory usage: {memory.percent:.1f}%')
        elif memory.percent > self.config['alerts']['memory_usage']['warning_threshold_percent']:
            self.send_alert('WARNING', 'MEMORY_USAGE', f'High memory usage: {memory.percent:.1f}%')
    
    def check_gpu_temperature(self):
        """Check GPU temperature alerts"""
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.temperature > self.config['alerts']['gpu_temperature']['critical_threshold_celsius']:
                    self.send_alert('CRITICAL', 'GPU_TEMPERATURE', 
                                  f'Critical GPU temperature: {gpu.temperature}°C on {gpu.name}')
                elif gpu.temperature > self.config['alerts']['gpu_temperature']['warning_threshold_celsius']:
                    self.send_alert('WARNING', 'GPU_TEMPERATURE', 
                                  f'High GPU temperature: {gpu.temperature}°C on {gpu.name}')
        except:
            pass  # GPU monitoring not available
    
    def check_processing_failures(self):
        """Check for consecutive processing failures"""
        # Implementation would check recent processing logs
        # for consecutive failures within time window
        pass
    
    def send_alert(self, level, alert_type, message):
        """Send alert notification"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'type': alert_type,
            'message': message
        }
        
        # Console notification
        if 'console' in self.config['notification']['methods']:
            icon = '🚨' if level == 'CRITICAL' else '⚠️'
            print(f"{icon} [{level}] {alert_type}: {message}")
        
        # Log file notification
        if 'log_file' in self.config['notification']['methods']:
            with open(self.alerts_log, 'a') as f:
                f.write(json.dumps(alert) + '\n')
    
    def run_checks(self):
        """Run all monitoring checks"""
        self.check_disk_space()
        self.check_memory_usage()
        self.check_gpu_temperature()
        self.check_processing_failures()

if __name__ == "__main__":
    monitor = AlertMonitor()
    monitor.run_checks()
```

## System Optimization

### Performance Tuning

#### GPU Optimization Script
```powershell
# optimize-gpu.ps1 - GPU performance optimization
Write-Host "🎮 GPU Performance Optimization"

# 1. Check current GPU status
$gpuInfo = & nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
Write-Host "Current GPU Status: $gpuInfo"

# 2. Set optimal power and clock settings
& nvidia-smi -pm 1  # Enable persistent mode
& nvidia-smi -pl 350  # Set power limit (adjust for your GPU)
Write-Host "✅ GPU power management optimized"

# 3. Clear GPU memory
& nvidia-smi --gpu-reset
Write-Host "🧹 GPU memory cleared"

# 4. Set compute mode for optimal processing
& nvidia-smi -c 0  # Set to default compute mode
Write-Host "⚙️  Compute mode optimized"

# 5. Monitor temperatures
$temp = (& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits).Trim()
Write-Host "🌡️  Current temperature: ${temp}°C"
if ([int]$temp -gt 80) {
    Write-Warning "⚠️  GPU temperature high - check cooling"
}
```

This comprehensive maintenance and monitoring documentation provides systematic procedures for keeping the alexdev-video-summarizer system running optimally with proactive monitoring and regular maintenance tasks.