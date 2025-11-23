# 서버 연결 문제 해결 가이드

## 🔍 문제 진단

에러: `ERR_CONNECTION_REFUSED` - 서버에 연결할 수 없음

## ✅ 해결 방법

### 1. 서버가 실행 중인지 확인

터미널에서 다음 명령어로 확인:
```bash
netstat -ano | findstr :5000
```

포트 5000에서 실행 중인 프로세스가 있으면 서버가 실행 중입니다.

### 2. 서버 재시작

SyntaxError를 수정했으므로 서버를 재시작해야 합니다:

1. **현재 실행 중인 서버 중지**
   - 터미널에서 `Ctrl+C`로 중지

2. **서버 재시작**
   ```bash
   cd C:\FaceWatch
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
   ```

3. **서버 시작 확인**
   - 다음 메시지가 보이면 정상:
   ```
   INFO:     Uvicorn running on http://0.0.0.0:5000
   INFO:     Application startup complete.
   ```

### 3. 브라우저에서 확인

서버가 정상적으로 시작되면:
1. 브라우저 새로고침 (F5)
2. 개발자 도구(F12) → Network 탭에서 `/api/persons` 요청 확인
3. 응답이 200 OK인지 확인

### 4. 여전히 안 되면

**방화벽 확인:**
- Windows 방화벽에서 포트 5000이 차단되어 있는지 확인

**다른 포트 사용:**
- 포트 5000이 사용 중이면 다른 포트 사용:
  ```bash
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
  ```
- 그리고 `web/script.js`에서 포트 번호 변경:
  ```javascript
  const API_BASE_URL = 'http://localhost:8000/api';
  ```

## 📋 체크리스트

- [ ] 서버가 실행 중인가?
- [ ] 포트 5000이 사용 가능한가?
- [ ] 브라우저 콘솔에 다른 에러가 있는가?
- [ ] 서버 로그에 에러가 있는가?

