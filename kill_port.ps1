$connections = netstat -ano | Select-String ":5000"
foreach ($conn in $connections) {
    $parts = $conn.Line.Trim() -split '\s+'
    $procId = $parts[-1]
    if ($procId -match '^\d+$') {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 3
$stillListening = netstat -ano | Select-String ":5000" | Select-String "LISTENING"
if (-not $stillListening) {
    Write-Output "Port 5000 is free"
} else {
    Write-Output "Still listening"
    $stillListening
}
