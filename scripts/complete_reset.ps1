# FaceWatch 완전 초기화 스크립트
# 주의: 이 스크립트는 모든 데이터를 삭제합니다!

param(
    [switch]$Confirm = $false
)

Write-Host "=" * 70 -ForegroundColor Yellow
Write-Host "🧹 FaceWatch 완전 초기화 스크립트" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Yellow
Write-Host ""
Write-Host "⚠️  경고: 다음 데이터가 모두 삭제됩니다:" -ForegroundColor Red
Write-Host "   - outputs/embeddings/* (모든 임베딩 데이터)" -ForegroundColor Red
Write-Host "   - images/enroll/* (모든 등록 이미지)" -ForegroundColor Red
Write-Host "   - PostgreSQL persons 테이블" -ForegroundColor Red
Write-Host ""

if (-not $Confirm) {
    $response = Read-Host "정말로 모든 데이터를 삭제하시겠습니까? (yes/no)"
    if ($response -ne "yes") {
        Write-Host "❌ 취소되었습니다." -ForegroundColor Yellow
        exit
    }
}

Write-Host ""
Write-Host "🚀 초기화 시작..." -ForegroundColor Green
Write-Host ""

# 1. embeddings 폴더 정리
Write-Host "📁 [1/4] embeddings 폴더 정리 중..." -ForegroundColor Cyan
$embeddingsPath = "outputs\embeddings"
if (Test-Path $embeddingsPath) {
    Get-ChildItem -Path $embeddingsPath -Recurse | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "  ✅ embeddings 폴더 비움" -ForegroundColor Green
} else {
    Write-Host "  ℹ️ embeddings 폴더가 없습니다" -ForegroundColor Yellow
}

# 2. images/enroll 폴더 정리
Write-Host "📁 [2/4] images/enroll 폴더 정리 중..." -ForegroundColor Cyan
$enrollPath = "images\enroll"
if (Test-Path $enrollPath) {
    Get-ChildItem -Path $enrollPath -Recurse | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
    Write-Host "  ✅ images/enroll 폴더 비움" -ForegroundColor Green
} else {
    Write-Host "  ℹ️ images/enroll 폴더가 없습니다" -ForegroundColor Yellow
}

# 3. DB 초기화
Write-Host "🗄️  [3/4] 데이터베이스 초기화 중..." -ForegroundColor Cyan
try {
    python -c "from backend.database import engine, Base; Base.metadata.drop_all(bind=engine); Base.metadata.create_all(bind=engine); print('  ✅ DB 초기화 완료')"
} catch {
    Write-Host "  ⚠️ DB 초기화 실패: $_" -ForegroundColor Red
    Write-Host "  → 수동으로 PostgreSQL에서 persons 테이블을 삭제하세요" -ForegroundColor Yellow
}

# 4. 확인
Write-Host "🔍 [4/4] 초기화 확인 중..." -ForegroundColor Cyan

$embeddingsCount = (Get-ChildItem -Path $embeddingsPath -Directory -ErrorAction SilentlyContinue).Count
$enrollCount = (Get-ChildItem -Path $enrollPath -Directory -ErrorAction SilentlyContinue).Count

Write-Host "  📊 embeddings 폴더: $embeddingsCount 개" -ForegroundColor $(if ($embeddingsCount -eq 0) { "Green" } else { "Red" })
Write-Host "  📊 images/enroll 폴더: $enrollCount 개" -ForegroundColor $(if ($enrollCount -eq 0) { "Green" } else { "Red" })

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "✅ 초기화 완료!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green
Write-Host ""
Write-Host "📝 다음 단계:" -ForegroundColor Cyan
Write-Host "   1. 백엔드 서버 재시작: uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000" -ForegroundColor White
Write-Host "   2. 서버 로그에서 'Gallery 로딩 완료 (0명)' 확인" -ForegroundColor White
Write-Host "   3. 브라우저 하드 리프레시: Ctrl + Shift + R" -ForegroundColor White
Write-Host "   4. 인물 데이터베이스에서 '등록된 인물이 없습니다' 확인" -ForegroundColor White
Write-Host ""
