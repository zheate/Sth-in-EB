# PowerShell script to run Streamlit app without showing terminal window
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$batFile = Join-Path $scriptDir "run.bat"

# Create a hidden process
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "cmd.exe"
$psi.Arguments = "/c `"$batFile`""
$psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$psi.CreateNoWindow = $true

$process = [System.Diagnostics.Process]::Start($psi)

# Optional: Wait for process to exit
# $process.WaitForExit()
