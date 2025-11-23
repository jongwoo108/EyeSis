# Bank Base 재생성 후 빠른 시작 가이드

## ✅ 완료된 작업

1. `bank_base.npy` 생성 완료 (10명)
2. `centroid_base.npy` 생성 완료 (10명)
3. 백업 파일 생성 완료

## 🔄 다음 단계

### 1. 서버 재시작 (필수)

새로운 base/dynamic 구조를 로드하려면 서버를 재시작해야 합니다.

```bash
# 서버가 실행 중이면 중지 (Ctrl+C)
# 그 다음 재시작

# 개발 모드 (자동 재시작)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000

# 또는 프로덕션 모드
uvicorn backend.main:app --host 0.0.0.0 --port 5000
```

### 2. 서버 시작 로그 확인

서버 시작 시 다음과 같은 로그가 나와야 합니다:

```
✅ Base Bank 로드: {name} (ID: {person_id}, {N}개 임베딩)
✅ Dynamic Bank 로드: {name} (ID: {person_id}, {M}개 임베딩)
```

또는 Dynamic Bank가 없으면:

```
✅ Base Bank 로드: {name} (ID: {person_id}, {N}개 임베딩)
```

**중요:** 다음 로그가 나오면 정상입니다:
- `📂 데이터베이스 로딩 완료 ({N}명, Base/Dynamic Bank 분리 구조)`

### 3. 브라우저에서 확인

1. **웹 인터페이스 열기**
   ```
   http://localhost:5000
   ```

2. **확인 사항**
   - 얼굴 감지가 정상 작동하는지
   - 박스가 제대로 표시되는지
   - 매칭이 정확한지

3. **테스트 시나리오**
   - 일반 모드: 전체 갤러리에서 매칭 테스트
   - Suspect IDs 모드: 특정 인물 선택 후 매칭 테스트
   - Unknown 처리: 매칭되지 않는 얼굴이 unknown으로 표시되는지

### 4. 문제 발생 시 확인 사항

#### A. 서버 시작 실패
- 로그에서 오류 메시지 확인
- `bank_base.npy` 파일이 제대로 생성되었는지 확인
- 파일 권한 문제인지 확인

#### B. 얼굴이 감지되지 않음
- 브라우저 콘솔에서 WebSocket 연결 확인
- 서버 로그에서 얼굴 감지 개수 확인
- `🔍 [얼굴 감지] 감지된 얼굴 개수: {N}` 로그 확인

#### C. 매칭이 안 됨
- 서버 로그에서 매칭 디버깅 정보 확인
- `🎯 [매칭 디버깅]` 로그 확인
- threshold/gap 값이 너무 높은지 확인

### 5. Threshold/Gap 튜닝 (필요 시)

매칭 결과를 보고 threshold/gap을 조정해야 할 수 있습니다:

```bash
# 튜닝 가이드 확인
python scripts/tune_threshold_gap.py --guide
```

**조정 방법:**
- `backend/main.py`의 `process_detection` 함수에서 값 수정
- 서버 재시작 (--reload 옵션 사용 시 자동)

---

## 📊 예상되는 변화

### Before (기존 구조)
- 모든 임베딩이 하나의 `bank.npy`에 저장
- 시간이 지날수록 오염 가능성

### After (새 구조)
- 기준 Bank (`bank_base.npy`)는 변하지 않음
- 자동 학습은 `bank_dynamic.npy`에만 추가
- 더 안정적인 매칭 기준 유지

---

## 🎯 체크리스트

- [ ] 서버 재시작 완료
- [ ] 서버 로그에서 "Base/Dynamic Bank 분리 구조" 확인
- [ ] 브라우저에서 얼굴 감지 테스트
- [ ] 매칭 정확도 확인
- [ ] Unknown 처리 확인
- [ ] (선택) Threshold/Gap 튜닝

---

**작성일:** 2024년
