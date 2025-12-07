@echo off
chcp 65001 >nul
echo ============================================================
echo FaceWatch 빠른 테스트 스크립트
echo ============================================================
echo.

REM 현재 디렉토리를 프로젝트 루트로 변경
cd /d "%~dp0\.."

echo 사용 가능한 스크립트:
echo.
echo [1] Dynamic/Masked Bank 정리
echo [2] 인물 등록 (Base Bank 생성)
echo [3] 레거시 파일 재생성
echo [4] 전체 테스트 (1단계 개선)
echo [5] 서버 시작
echo.
set /p choice="선택 (1-5): "

if "%choice%"=="1" (
    python scripts/cleanup_dynamic_masked_banks.py
) else if "%choice%"=="2" (
    python src/face_enroll.py
) else if "%choice%"=="3" (
    python scripts/regenerate_legacy_files.py
) else if "%choice%"=="4" (
    call scripts\test_improvement_step1.bat
) else if "%choice%"=="5" (
    echo.
    echo 서버 시작 중...
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
) else (
    echo 잘못된 선택입니다.
)

pause









