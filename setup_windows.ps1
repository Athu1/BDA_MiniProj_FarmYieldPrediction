# One-time setup for running the Spark pipeline on native Windows.
# Run from the project folder in PowerShell:
#     powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
#
# What it does:
#   1. finds a usable Python (3.10 - 3.13; prefers 3.11, which this project was tested on)
#   2. checks for Java 17 (Spark 3.5 runs on Java 8, 11 or 17 - NOT 21+)
#   3. creates a virtual environment (.venv) and installs requirements.txt
#   4. downloads winutils.exe + hadoop.dll (Hadoop 3.3.6 native helpers) into .\hadoop\bin
#      Spark needs them on Windows to write files; bda\config.py picks them up automatically.
#   5. starts Spark once and writes a test Parquet file

# NOTE: deliberately NOT setting $ErrorActionPreference = "Stop".
# Native tools (java -version, pip, winget) write normal messages to stderr, and in
# Windows PowerShell 5.1 that becomes a terminating error. Exit codes are checked instead.
Set-Location $PSScriptRoot
Write-Host "== FieldYield BDA setup ==" -ForegroundColor Green

# ---- 1. Python ---------------------------------------------------------------
# PySpark 3.5 works on 3.10 - 3.13. On 3.12+ its MLlib still imports `distutils`,
# which Python removed, so setuptools (in requirements.txt) must supply it.
$pyExe = $null; $pyVer = $null
foreach ($v in @("3.11", "3.12", "3.13", "3.10")) {
    try {
        $out = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { $pyExe = "$out".Trim(); $pyVer = $v; break }
    } catch { }
}
if (-not $pyExe) {                       # no py launcher - fall back to python on PATH
    try {
        $v = & python -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))" 2>$null
        if ($LASTEXITCODE -eq 0 -and "$v".Trim() -match '^3\.(10|11|12|13)$') {
            $pyExe = (& python -c "import sys; print(sys.executable)").Trim()
            $pyVer = "$v".Trim()
        }
    } catch { }
}
if (-not $pyExe) {
    Write-Host "No suitable Python found. This project needs Python 3.10, 3.11, 3.12 or 3.13." -ForegroundColor Red
    Write-Host "Versions currently installed:" -ForegroundColor Yellow
    try { & py -0p } catch { Write-Host "  (the 'py' launcher is not installed)" }
    Write-Host "Install Python 3.11 from https://www.python.org/downloads/release/python-3119/"
    Write-Host "and tick 'Add python.exe to PATH' during installation, then rerun this script."
    exit 1
}
Write-Host "Using Python $pyVer  ($pyExe)"

# ---- 2. Java ----------------------------------------------------------------
# Spark 3.5 runs on Java 8, 11 or 17 (not 21+). Installers often leave java.exe off PATH,
# so: check PATH, then JAVA_HOME, then the registry, then search the usual install areas.
function Get-JavaMajor($dir, $exe) {
    # folder names look like jdk-17.0.20.101-hotspot, jdk1.8.0_202, jre-11.0.2 ...
    if ($dir -match '(?:jdk|jre)[-_]?1\.(\d+)') { return [int]$Matches[1] }
    if ($dir -match '(?:jdk|jre)[-_]?(\d+)')     { return [int]$Matches[1] }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'SilentlyContinue'
    try { $v = (& $exe -version 2>&1 | Out-String) } catch { $v = "" } finally { $ErrorActionPreference = $old }
    if ($v -match 'version "1\.(\d+)') { return [int]$Matches[1] }
    if ($v -match 'version "(\d+)')     { return [int]$Matches[1] }
    return 0
}
function Get-JavaRank($exe) {
    $jdkDir = Split-Path (Split-Path $exe -Parent) -Parent    # ...\jdk-17.0.20-hotspot\bin\java.exe -> the jdk folder
    $dir = Split-Path $jdkDir -Leaf
    $major = Get-JavaMajor $dir $exe
    $rank = @{ 17 = 0; 11 = 1; 8 = 2 }[$major]
    if ($null -eq $rank) { return $null }
    return @{ Rank = $rank; Ver = "Java $major" }
}

$javaHome = $null; $javaVer = ""
$onPath = Get-Command java -ErrorAction SilentlyContinue   # (a) already on PATH
if ($onPath) {
    $r = Get-JavaRank $onPath.Source
    if ($r) { $javaHome = ""; $javaVer = $r.Ver }
}
if ($null -eq $javaHome -and $env:JAVA_HOME -and (Test-Path "$env:JAVA_HOME\bin\java.exe")) {   # (b) JAVA_HOME
    $r = Get-JavaRank "$env:JAVA_HOME\bin\java.exe"
    if ($r) { $javaHome = $env:JAVA_HOME; $javaVer = $r.Ver }
}
if ($null -eq $javaHome) {                                     # (c) registry (any JDK/JRE vendor)
    $keys = @("HKLM:\SOFTWARE\JavaSoft\JDK", "HKLM:\SOFTWARE\JavaSoft\JRE",
              "HKLM:\SOFTWARE\JavaSoft\Java Development Kit", "HKLM:\SOFTWARE\JavaSoft\Java Runtime Environment",
              "HKLM:\SOFTWARE\Eclipse Adoptium\JDK", "HKLM:\SOFTWARE\Eclipse Foundation\JDK")
    $cands = @()
    foreach ($k in $keys) {
        if (-not (Test-Path $k)) { continue }
        foreach ($sub in (Get-ChildItem $k -Recurse -ErrorAction SilentlyContinue)) {
            $jh = (Get-ItemProperty $sub.PSPath -ErrorAction SilentlyContinue).JavaHome
            if ($jh -and (Test-Path "$jh\bin\java.exe")) { $cands += $jh }
        }
    }
    $bestRank = 99
    foreach ($jh in ($cands | Select-Object -Unique)) {
        $r = Get-JavaRank "$jh\bin\java.exe"
        if ($r -and $r.Rank -lt $bestRank) { $javaHome = $jh; $javaVer = $r.Ver; $bestRank = $r.Rank }
    }
}
if ($null -eq $javaHome) {                                     # (d) search the usual install areas
    Write-Host "Searching for an installed Java (this takes a few seconds) ..."
    $searchRoots = @("$env:ProgramFiles", "${env:ProgramFiles(x86)}", "$env:LOCALAPPDATA\Programs",
                     "$env:ProgramData", "$env:USERPROFILE\scoop\apps", "C:\Java") |
                   Where-Object { $_ -and (Test-Path $_) }
    $best = $null
    foreach ($exe in (Get-ChildItem -Path $searchRoots -Recurse -Depth 4 -Filter "java.exe" -File -ErrorAction SilentlyContinue)) {
        if ($exe.Directory.Name -ne "bin") { continue }
        $r = Get-JavaRank $exe.FullName
        if (-not $r) { continue }
        # prefer a standalone JDK over one bundled inside another product (Android Studio etc.)
        $bundled = if ($exe.FullName -match '\\(Android|JetBrains|Unity|IntelliJ)\\') { 1 } else { 0 }
        $score = $r.Rank * 2 + $bundled
        if ($null -eq $best -or $score -lt $best.Score) {
            $best = @{ Score = $score; Home = $exe.Directory.Parent.FullName; Ver = $r.Ver }
            if ($score -eq 0) { break }                        # standalone Java 17: stop looking
        }
    }
    if ($best) { $javaHome = $best.Home; $javaVer = $best.Ver }
}
if ($null -eq $javaHome) {                                     # (e) nothing -> install
    Write-Host "No Java 8/11/17 found on this PC. Installing Eclipse Temurin JDK 17 ..." -ForegroundColor Yellow
    try { winget install --id EclipseAdoptium.Temurin.17.JDK -e --accept-source-agreements --accept-package-agreements } catch { }
    Write-Host ""
    Write-Host "Run this script again. If it still cannot find Java, run this and send me the output:" -ForegroundColor Yellow
    Write-Host '  Get-ChildItem "C:\Program Files","C:\Program Files (x86)","$env:LOCALAPPDATA\Programs" -Recurse -Depth 4 -Filter java.exe -ErrorAction SilentlyContinue | Select -Expand FullName'
    exit 1
}
if ($javaHome -ne "") {                                        # found off-PATH: use it now and later
    $env:JAVA_HOME = $javaHome
    $env:PATH = "$javaHome\bin" + [IO.Path]::PathSeparator + $env:PATH
    [Environment]::SetEnvironmentVariable("JAVA_HOME", $javaHome, "User")
    Write-Host "JAVA_HOME set to $javaHome (saved for future sessions)"
}
Write-Host "Found Java: $((($javaVer -split "`n")[0]).Trim())"

# ---- 3. virtual environment + packages --------------------------------------
if (Test-Path ".venv") {
    $venvVer = & .\.venv\Scripts\python.exe -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))" 2>$null
    if ("$venvVer".Trim() -ne $pyVer) {
        Write-Host "Existing .venv uses Python $venvVer; rebuilding it with $pyVer ..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force ".venv"
    }
}
if (-not (Test-Path ".venv")) { & $pyExe -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools | Out-Null
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed - see the error above." -ForegroundColor Red; exit 1 }

# ---- 4. Hadoop native helpers for Windows -----------------------------------
$bin = Join-Path $PSScriptRoot "hadoop\bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
$base = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin"
foreach ($f in @("winutils.exe", "hadoop.dll")) {
    $dest = Join-Path $bin $f
    if (-not (Test-Path $dest)) {
        Write-Host "Downloading $f ..."
        try { Invoke-WebRequest -Uri "$base/$f" -OutFile $dest -UseBasicParsing }
        catch {
            Write-Host "Could not download $f. Check your internet connection (GitHub must be reachable)." -ForegroundColor Red
            exit 1
        }
    }
}
Write-Host "Hadoop helpers ready in $bin"

# ---- 5. smoke test: start Spark and write a Parquet file --------------------
Write-Host "Starting Spark (first run takes ~30 s) ..."
& .\.venv\Scripts\python.exe -c "from bda.config import get_spark, DATA, p; s=get_spark('setup-check'); s.range(1000).write.mode('overwrite').parquet(p(DATA/'_setup_check')); import pyspark.ml.regression; print('Spark OK:', s.version)"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Spark test failed. Common fixes:" -ForegroundColor Red
    Write-Host "  - 'UnsatisfiedLinkError / NativeIO': install the Microsoft Visual C++ Redistributable (x64)"
    Write-Host "    from https://aka.ms/vs/17/release/vc_redist.x64.exe, then rerun."
    Write-Host "  - 'Java gateway exited': make sure JAVA_HOME points to the Java 17 folder."
    Write-Host "  - Move the project to a simple path such as C:\projects\MiniProj (no spaces, not OneDrive)."
    exit 1
}
Write-Host ""
Write-Host "Setup complete. Next:" -ForegroundColor Green
Write-Host "   .\.venv\Scripts\activate"
Write-Host "   python run_all.py          (or: python run_all.py --fast)"
Write-Host "   python dashboard\app.py"
