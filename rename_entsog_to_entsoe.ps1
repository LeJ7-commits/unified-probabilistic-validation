# rename_entsog_to_entsoe.ps1
# Run from the repo root:
#   cd C:\Users\jiaya\OneDrive\Documents\Lund_2025\Thesis\unified-probabilistic-validation
#   .\rename_entsog_to_entsoe.ps1
#
# What this script does:
#   1. Renames source files (scripts, notebooks, CSVs)
#   2. Patches string content inside those files
#   3. Renames derived .npy artifacts in data/derived_full/ and data/derived_dev/
#   4. Renames the experiments output directory
#   5. Prints a summary of every action taken

$ErrorActionPreference = "Stop"
$root = Get-Location

Write-Host "`n=== ENTSOG -> ENTSOE rename script ===" -ForegroundColor Cyan
Write-Host "Repo root: $root`n"

# ---------------------------------------------------------------------------
# Helper: rename a file if it exists
# ---------------------------------------------------------------------------
function Rename-IfExists($oldPath, $newPath) {
    if (Test-Path $oldPath) {
        Rename-Item -Path $oldPath -NewName (Split-Path $newPath -Leaf)
        Write-Host "  RENAMED  $oldPath  ->  $newPath" -ForegroundColor Green
    } else {
        Write-Host "  SKIP     $oldPath  (not found)" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# Helper: replace all occurrences of a string inside a text file
# ---------------------------------------------------------------------------
function Replace-InFile($filePath, $oldStr, $newStr) {
    if (Test-Path $filePath) {
        $content = Get-Content $filePath -Raw -Encoding UTF8
        if ($content -match [regex]::Escape($oldStr)) {
            $content = $content.Replace($oldStr, $newStr)
            Set-Content $filePath -Value $content -Encoding UTF8 -NoNewline
            Write-Host "  PATCHED  $filePath  ('$oldStr' -> '$newStr')" -ForegroundColor Green
        }
    }
}

# ---------------------------------------------------------------------------
# 1. Rename source files
# ---------------------------------------------------------------------------
Write-Host "--- Step 1: Rename source files ---"

Rename-IfExists "$root\data\entsog_sample.csv"         "$root\data\entsoe_sample.csv"
Rename-IfExists "$root\data\entsog_sample_90days.csv"  "$root\data\entsoe_sample_90days.csv"
Rename-IfExists "$root\data\entsog_full.csv"           "$root\data\entsoe_full.csv"
Rename-IfExists "$root\scripts\build_entsog_derived.py" "$root\scripts\build_entsoe_derived.py"
Rename-IfExists "$root\experiments\run_001_entsog.py"  "$root\experiments\run_001_entsoe.py"
Rename-IfExists "$root\notebooks\02_entsog_feasibility.ipynb" "$root\notebooks\02_entsoe_feasibility.ipynb"
Rename-IfExists "$root\notebooks\02_entsog_90days.ipynb"      "$root\notebooks\02_entsoe_90days.ipynb"

# ---------------------------------------------------------------------------
# 2. Patch content inside renamed source files
# ---------------------------------------------------------------------------
Write-Host "`n--- Step 2: Patch content inside source files ---"

$buildScript = "$root\scripts\build_entsoe_derived.py"
Replace-InFile $buildScript "build_entsog_derived.py"        "build_entsoe_derived.py"
Replace-InFile $buildScript "scripts/build_entsog_derived.py" "scripts/build_entsoe_derived.py"
Replace-InFile $buildScript "entsog_full.csv"                 "entsoe_full.csv"
Replace-InFile $buildScript "entsog_sample_90days.csv"        "entsoe_sample_90days.csv"
Replace-InFile $buildScript "02_entsog_feasibility.ipynb"     "02_entsoe_feasibility.ipynb"
Replace-InFile $buildScript "[entsog]"                        "[entsoe]"
Replace-InFile $buildScript "entsog_y.npy"                    "entsoe_y.npy"
Replace-InFile $buildScript "entsog_yhat.npy"                 "entsoe_yhat.npy"
Replace-InFile $buildScript "entsog_scale.npy"                "entsoe_scale.npy"
Replace-InFile $buildScript "entsog_lo_base_90.npy"           "entsoe_lo_base_90.npy"
Replace-InFile $buildScript "entsog_hi_base_90.npy"           "entsoe_hi_base_90.npy"
Replace-InFile $buildScript "entsog_samples.npy"              "entsoe_samples.npy"

$runScript = "$root\experiments\run_001_entsoe.py"
Replace-InFile $runScript "build_entsog_derived.py"   "build_entsoe_derived.py"
Replace-InFile $runScript "run_001_entsog"             "run_001_entsoe"
Replace-InFile $runScript "entsog_y.npy"               "entsoe_y.npy"
Replace-InFile $runScript "entsog_yhat.npy"            "entsoe_yhat.npy"
Replace-InFile $runScript "entsog_scale.npy"           "entsoe_scale.npy"
Replace-InFile $runScript "entsog_lo_base_90.npy"      "entsoe_lo_base_90.npy"
Replace-InFile $runScript "entsog_hi_base_90.npy"      "entsoe_hi_base_90.npy"
Replace-InFile $runScript "entsog_samples.npy"         "entsoe_samples.npy"

$feasNb = "$root\notebooks\02_entsoe_feasibility.ipynb"
Replace-InFile $feasNb "entsog_sample.csv" "entsoe_sample.csv"

$nb90 = "$root\notebooks\02_entsoe_90days.ipynb"
Replace-InFile $nb90 "entsog_sample_90days.csv" "entsoe_sample_90days.csv"

# ---------------------------------------------------------------------------
# 3. Rename .npy artifacts in data/derived_full/ and data/derived_dev/
# ---------------------------------------------------------------------------
Write-Host "`n--- Step 3: Rename derived .npy artifacts ---"

foreach ($derivedDir in @("$root\data\derived_full", "$root\data\derived_dev")) {
    if (Test-Path $derivedDir) {
        $npyFiles = Get-ChildItem $derivedDir -Filter "entsog_*.npy"
        foreach ($f in $npyFiles) {
            $newName = $f.Name -replace "^entsog_", "entsoe_"
            Rename-Item -Path $f.FullName -NewName $newName
            Write-Host "  RENAMED  $($f.FullName)  ->  $newName" -ForegroundColor Green
        }
    } else {
        Write-Host "  SKIP     $derivedDir  (directory not found)" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# 4. Rename experiments output directory
# ---------------------------------------------------------------------------
Write-Host "`n--- Step 4: Rename experiments output directory ---"

$oldExpDir = "$root\experiments\run_001_entsog"
$newExpDir = "$root\experiments\run_001_entsoe"
if (Test-Path $oldExpDir) {
    Rename-Item -Path $oldExpDir -NewName "run_001_entsoe"
    Write-Host "  RENAMED  $oldExpDir  ->  $newExpDir" -ForegroundColor Green
} else {
    Write-Host "  SKIP     $oldExpDir  (not found)" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 5. Summary check — confirm no entsog strings remain in .py and .ipynb files
# ---------------------------------------------------------------------------
Write-Host "`n--- Step 5: Residual check ---"
$remaining = Get-ChildItem $root -Recurse -Include "*.py","*.ipynb" |
    Select-String -Pattern "entsog" -CaseSensitive |
    Where-Object { $_.Line -notmatch "^\s*#" }

if ($remaining) {
    Write-Host "  WARNING: Residual 'entsog' references found:" -ForegroundColor Red
    $remaining | ForEach-Object { Write-Host "    $($_.Filename):$($_.LineNumber)  $($_.Line.Trim())" }
} else {
    Write-Host "  OK: No residual 'entsog' references in .py or .ipynb files." -ForegroundColor Green
}

Write-Host "`n=== Done ===" -ForegroundColor Cyan
