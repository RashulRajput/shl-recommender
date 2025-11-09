# Start server and generate predictions
$env:GEMINI_API_KEY="AIzaSyAXDFEpJQhvAsUnn2RIZGVMQ-eQon6f47k"

Write-Host "Starting FastAPI server..." -ForegroundColor Cyan
Set-Location C:\Users\DELL\OneDrive\AppData\Desktop\SHL\shl

# Start server in background
$server = Start-Process python -ArgumentList "-m","uvicorn","app:app","--host","127.0.0.1","--port","8002" -PassThru -WindowStyle Hidden

Write-Host "Waiting for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Test if server is running
try {
    $test = Invoke-WebRequest -Uri "http://127.0.0.1:8002/health" -Method GET -TimeoutSec 5
    Write-Host "✓ Server is running!" -ForegroundColor Green
} catch {
    Write-Host "✗ Server failed to start" -ForegroundColor Red
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

# Generate predictions
Write-Host "`nGenerating predictions for 9 test queries..." -ForegroundColor Cyan
python generate_predictions.py

Write-Host "`nStopping server..." -ForegroundColor Yellow
Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue

Write-Host "`n✓ Done! Check predictions.csv" -ForegroundColor Green
