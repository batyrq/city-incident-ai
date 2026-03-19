$ErrorActionPreference = "Stop"

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

Write-Host "[bootstrap] creating virtualenv at $VenvDir"
& $PythonBin -m venv $VenvDir

$Activate = Join-Path $VenvDir "Scripts\\Activate.ps1"
. $Activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

function Install-OptionalExtra {
    param(
        [string]$Label,
        [string]$Extra
    )
    try {
        python -m pip install ".[${Extra}]"
        Write-Host "[bootstrap] installed optional extra: $Label"
    } catch {
        Write-Warning "[bootstrap] skipped optional extra: $Label"
    }
}

Install-OptionalExtra -Label "ultralytics" -Extra "ultralytics"
Install-OptionalExtra -Label "opencv" -Extra "dev"
Install-OptionalExtra -Label "tracking deps" -Extra "tracking"
Install-OptionalExtra -Label "anomalib" -Extra "anomalib"
Install-OptionalExtra -Label "sam2" -Extra "sam2"

Write-Host "[bootstrap] summary"
python -c "from src.utils.env import package_summary; print(package_summary(['yaml','pandas','cv2','ultralytics','anomalib','sam2']))"
