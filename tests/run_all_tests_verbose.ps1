$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Rebuild to ensure latest changes are included
cmake --build "$root\build" -j

# Verbose CTest output
ctest -V --test-dir "$root\build"
