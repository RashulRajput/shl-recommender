# SHL Recommender - Quick Start Script
# PowerShell script to verify and start the recommendation server

Write-Host "SHL Recommendation System - Health Check" -ForegroundColor Cyan
Write-Host ("=" * 60)

# Check Python
Write-Host "`n1. Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   [OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   [ERROR] Python not found!" -ForegroundColor Red
    exit 1
}

# Check environment variable
Write-Host "`n2. Checking GEMINI_API_KEY..." -ForegroundColor Yellow
if ($env:GEMINI_API_KEY) {
    $keyPreview = $env:GEMINI_API_KEY.Substring(0, [Math]::Min(10, $env:GEMINI_API_KEY.Length)) + "..."
    Write-Host "   [OK] API Key set: $keyPreview" -ForegroundColor Green
} else {
    Write-Host "   [WARN] GEMINI_API_KEY not set (will use pseudo-embeddings)" -ForegroundColor Yellow
}

# Check catalog file
Write-Host "`n3. Checking assessments catalog..." -ForegroundColor Yellow
$catalogPath = "shl\data\assessments.csv"
if (Test-Path $catalogPath) {
    $csvRows = (Import-Csv $catalogPath).Count
    Write-Host "   [OK] Found: $catalogPath - $csvRows assessments" -ForegroundColor Green
} else {
    Write-Host "   [ERROR] Missing: $catalogPath" -ForegroundColor Red
    Write-Host "   Run: python shl\crawler.py" -ForegroundColor Yellow
    exit 1
}

# Check model files
Write-Host "`n4. Checking model files..." -ForegroundColor Yellow
$modelFiles = @(
    "shl\models\assessments_df.pkl",
    "shl\models\tfidf.pkl",
    "shl\models\tfidf_matrix.pkl",
    "shl\models\embeddings.npy"
)

$allPresent = $true
foreach ($file in $modelFiles) {
    if (Test-Path $file) {
        $size = [Math]::Round((Get-Item $file).Length / 1KB, 1)
        Write-Host "   [OK] $file - $size KB" -ForegroundColor Green
    } else {
        Write-Host "   [ERROR] Missing: $file" -ForegroundColor Red
        $allPresent = $false
    }
}

if (-not $allPresent) {
    Write-Host "`n   Run: cd shl ; python build_index.py" -ForegroundColor Yellow
    exit 1
}

# Verify model alignment
Write-Host "`n5. Verifying model alignment..." -ForegroundColor Yellow
try {
    $verification = python -c "import joblib, numpy as np; df = joblib.load('shl/models/assessments_df.pkl'); emb = np.load('shl/models/embeddings.npy'); tfidf = joblib.load('shl/models/tfidf_matrix.pkl'); print(f'{len(df)}|{emb.shape[0]}|{tfidf.shape[0]}'); assert len(df) == emb.shape[0] == tfidf.shape[0]" 2>&1
    $counts = $verification -split '\|'
    if ($counts.Count -eq 3 -and $counts[0] -eq $counts[1] -and $counts[1] -eq $counts[2]) {
        Write-Host "   [OK] All models aligned: $($counts[0]) rows" -ForegroundColor Green
    } else {
        Write-Host "   [ERROR] Model mismatch: DF=$($counts[0]), Emb=$($counts[1]), TFIDF=$($counts[2])" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "   [ERROR] Verification failed: $_" -ForegroundColor Red
    exit 1
}

# Check training data
Write-Host "`n6. Checking training/test data..." -ForegroundColor Yellow
$trainPath = "shl\data\Gen_AI_Train-Set_FULL.csv"
$testPath = "shl\data\Gen_AI_Test-Set_FULL.csv"
if ((Test-Path $trainPath) -and (Test-Path $testPath)) {
    $trainRows = (Import-Csv $trainPath).Count
    $testRows = (Import-Csv $testPath).Count
    Write-Host "   [OK] Training: $trainRows rows" -ForegroundColor Green
    Write-Host "   [OK] Test: $testRows rows" -ForegroundColor Green
} else {
    Write-Host "   [WARN] Training/test data missing (optional)" -ForegroundColor Yellow
}

# Summary
Write-Host "`n" + ("=" * 60)
Write-Host "[SUCCESS] All checks passed! System ready." -ForegroundColor Green
Write-Host "`nTo start the server:" -ForegroundColor Cyan
Write-Host "   cd shl" -ForegroundColor White
Write-Host "   python -m uvicorn app:app --host 0.0.0.0 --port 8002 --reload" -ForegroundColor White
Write-Host "`nAccess:" -ForegroundColor Cyan
Write-Host "   Web UI:  http://localhost:8002" -ForegroundColor White
Write-Host "   Health:  http://localhost:8002/health" -ForegroundColor White
Write-Host "   API Doc: http://localhost:8002/docs" -ForegroundColor White
Write-Host ""
