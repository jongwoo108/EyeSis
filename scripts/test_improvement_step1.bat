@echo off
chcp 65001 >nul
echo ============================================================
echo FaceWatch 1단계 개선 테스트 스크립트
echo ============================================================
echo.

REM 현재 디렉토리를 프로젝트 루트로 변경
cd /d "%~dp0\.."

echo [1/4] Dynamic/Masked Bank 정리 중...
echo.
python scripts/cleanup_dynamic_masked_banks.py --confirm
if errorlevel 1 (
    echo.
    echo ❌ Dynamic/Masked Bank 정리 실패
    pause
    exit /b 1
)
echo.

echo [2/4] 인물 등록 (Base Bank 생성) 중...
echo.
python src/face_enroll.py
if errorlevel 1 (
    echo.
    echo ❌ 인물 등록 실패
    pause
    exit /b 1
)
echo.

echo [3/4] 레거시 파일 재생성 중...
echo.
python scripts/regenerate_legacy_files.py --confirm
if errorlevel 1 (
    echo.
    echo ⚠️ 레거시 파일 재생성 실패 (계속 진행)
)
echo.

echo [4/4] 서버 시작 준비 완료
echo.
echo ============================================================
echo ✅ 준비 완료!
echo ============================================================
echo.
echo 다음 단계:
echo   1. 서버 시작: uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
echo   2. 브라우저에서 http://localhost:5000 접속
echo   3. 영상 업로드 및 감지 테스트
echo   4. 서버 콘솔에서 검증 로그 확인:
echo      - ✅ [DYNAMIC BANK] 검증 통과: ...
echo      - ⏭ [DYNAMIC BANK] 검증 실패: ...
echo.
pause









