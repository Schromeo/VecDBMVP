$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Rebuild to ensure latest changes are included
cmake --build "$root\build" -j

# Generate fresh CSVs
python "$root\scripts\generate_csv.py"

# Clean test collection
$colDir = "$root\data\cli_demo"
if (Test-Path $colDir) {
  Remove-Item -LiteralPath $colDir -Recurse -Force
}

# End-to-end CLI test
./build/vecdb.exe create --dir data/cli_demo --dim 16 --metric l2
./build/vecdb.exe load --dir data/cli_demo --csv data/vectors.csv --header
./build/vecdb.exe build --dir data/cli_demo --M 16 --M0 32 --efC 100 --diversity 1
./build/vecdb.exe search --dir data/cli_demo --query_csv data/queries.csv --k 5 --ef 50 --header

# Metadata CSV test
$metaDir = "$root\data\cli_demo_meta"
if (Test-Path $metaDir) {
  Remove-Item -LiteralPath $metaDir -Recurse -Force
}

./build/vecdb.exe create --dir data/cli_demo_meta --dim 16 --metric l2
./build/vecdb.exe load --dir data/cli_demo_meta --csv data/vectors_with_meta.csv --header --meta
./build/vecdb.exe build --dir data/cli_demo_meta --M 16 --M0 32 --efC 100 --diversity 1
./build/vecdb.exe search --dir data/cli_demo_meta --query_csv data/queries.csv --k 5 --ef 50 --header --filter cluster=2

Write-Host "CSV CLI test completed."