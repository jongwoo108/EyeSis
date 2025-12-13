# 개선 사항 테스트 가이드

> 1단계 개선 (Dynamic Bank 입력 필터 강화) 테스트를 위한 단계별 가이드

---

## 목차

1. [사전 준비](#1-사전-준비)
2. [Dynamic/Masked Bank 정리](#2-dynamicmasked-bank-정리)
3. [인물 등록 (Base Bank 생성)](#3-인물-등록-base-bank-생성)
4. [서버 시작](#4-서버-시작)
5. [브라우저에서 테스트](#5-브라우저에서-테스트)
6. [검증 로그 확인](#6-검증-로그-확인)

---

## 1. 사전 준비

### 1.1 필요한 파일 확인

- `images/enroll/<인물명>/<파일명>.jpg` - 등록용 정면 사진
- 테스트용 영상 파일 (선택)

### 1.2 현재 상태 확인

```bash
# 현재 임베딩 상태 확인
ls outputs/embeddings/
```

각 인물 폴더에 다음 파일들이 있을 수 있습니다:
- `bank_base.npy` ✅ (유지)
- `bank_dynamic.npy` ❌ (삭제 예정)
- `bank_masked.npy` ❌ (삭제 예정)

---

## 2. Dynamic/Masked Bank 정리

**목적**: Base Bank만 남기고 Dynamic/Masked Bank를 삭제하여 새로운 검증 로직 테스트 환경 준비

### 2.1 스크립트 실행

```bash
# 프로젝트 루트에서 실행
python scripts/cleanup_dynamic_masked_banks.py
```

### 2.2 확인 사항

스크립트 실행 후 다음 메시지가 표시됩니다:

```
✅ Base Bank 유지: bank_base.npy
🗑️ 삭제된 파일:
   - bank_dynamic.npy
   - bank_masked.npy
   - angles_dynamic.json
   - ...
```

### 2.3 백업 확인 (선택)

백업 파일은 각 인물 폴더의 `backup_before_cleanup/` 폴더에 저장됩니다.

---

## 3. 인물 등록 (Base Bank 생성)

**목적**: 정면 사진에서 임베딩을 추출하여 Base Bank 생성

### 3.1 등록용 이미지 준비

```
images/enroll/
├─ hani/
│   └─ hani.jpg
├─ danielle/
│   └─ danielle.jpg
└─ ...
```

### 3.2 등록 스크립트 실행

```bash
# 프로젝트 루트에서 실행
python src/face_enroll.py
```

### 3.3 모드 선택

스크립트 실행 시 모드를 선택합니다:

```
[1] 기본 등록: enroll 폴더에서 정면 사진 등록
[2] 수동 추가: 특정 이미지 폴더나 파일들을 기존 bank에 추가
```

**권장**: `1` (기본 등록) 선택

### 3.4 등록 결과 확인

등록 완료 후 다음 파일이 생성됩니다:

```
outputs/embeddings/<인물명>/
├─ bank_base.npy        ✅ Base Bank (정면 얼굴 임베딩)
├─ centroid_base.npy    ✅ Centroid (평균 임베딩)
└─ angles_base.json     ✅ 각도 정보
```

### 3.5 등록 확인 로그

스크립트 실행 시 다음과 같은 로그가 표시됩니다:

```
✅ 등록 완료: hani (1개 임베딩)
✅ 등록 완료: danielle (1개 임베딩)
...
```

---

## 4. 서버 시작

### 4.1 서버 실행

```bash
# 프로젝트 루트에서 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
```

### 4.2 서버 시작 로그 확인

서버 시작 시 다음과 같은 로그가 표시되어야 합니다:

```
✅ Base Bank 로드: hani (ID: hani, 1개 임베딩)
✅ Base Bank 로드: danielle (ID: danielle, 1개 임베딩)
...
📂 데이터베이스 로딩 완료 (N명, Base/Dynamic Bank 분리 구조)
```

**중요**: Dynamic Bank 관련 로그는 표시되지 않아야 합니다 (삭제했으므로).

---

## 5. 브라우저에서 테스트

### 5.1 웹 인터페이스 열기

```
http://localhost:5000
```

### 5.2 테스트 시나리오

#### 시나리오 1: 영상 업로드 및 감지

1. **영상 업로드**
   - "영상 업로드" 버튼 클릭 또는 중앙 영역 클릭
   - 테스트용 영상 파일 선택

2. **인물 선택**
   - "인물 선택" 버튼 클릭
   - 등록된 인물 카드 클릭 (여러 명 선택 가능)

3. **모니터링 시작**
   - "모니터링 시작" 버튼 클릭

4. **AI 감지 활성화**
   - 우측 제어판 "AI 감지 활성화" 토글 ON

5. **비디오 재생**
   - 비디오 플레이어에서 재생 버튼 클릭

#### 시나리오 2: 실시간 감지 확인

- 영상 재생 중 얼굴이 감지되면:
  - 얼굴 박스 표시
  - 인물 감지 타임라인에 구간 표시
  - 감지 로그에 항목 추가

---

## 6. 검증 로그 확인

### 6.1 서버 콘솔 로그 확인

서버 콘솔에서 다음 메시지를 확인합니다:

#### ✅ 검증 통과 (Dynamic Bank에 추가됨)

```
✅ [DYNAMIC BANK] 검증 통과: hani (base_sim=0.75, face_size=250px, angle=left)
```

**의미**:
- Base Bank와의 유사도가 0.6 이상
- 얼굴 크기가 200px 이상
- Occlusion 없음 (랜드마크 검증 통과)

#### ⏭ 검증 실패 (Dynamic Bank에 추가되지 않음)

```
⏭ [DYNAMIC BANK] 검증 실패: hani | 이유: base_sim=0.45 < 0.6 (Base Bank 유사도 검증 실패)
```

**의미**:
- Base Bank와의 유사도가 0.6 미만 → 추가되지 않음

```
⏭ [DYNAMIC BANK] 검증 실패: hani | 이유: face_size=150px < 200px (고화질 검증 실패)
```

**의미**:
- 얼굴 크기가 200px 미만 → 추가되지 않음

```
⏭ [DYNAMIC BANK] 검증 실패: hani | 이유: occlusion detected (랜드마크 검증 실패)
```

**의미**:
- 마스크나 손으로 가린 상태 → 추가되지 않음

### 6.2 Dynamic Bank 파일 확인

테스트 후 다음 파일이 생성되었는지 확인:

```bash
# Dynamic Bank 파일 확인
ls outputs/embeddings/<인물명>/bank_dynamic.npy
```

**예상 결과**:
- 검증을 통과한 임베딩만 `bank_dynamic.npy`에 저장됨
- 검증 실패한 임베딩은 저장되지 않음

### 6.3 검증 통과/실패 통계

서버 로그에서 다음을 확인:

1. **검증 통과 비율**
   - 얼마나 많은 임베딩이 검증을 통과했는지
   - 검증 실패 이유별 통계

2. **데이터 품질 향상**
   - Base Bank와 유사도가 높은 임베딩만 수집됨
   - 고화질 얼굴만 수집됨
   - Occlusion 없는 얼굴만 수집됨

---

## 7. 예상되는 변화

### Before (개선 전)

- 매칭 성공 시 무조건 Dynamic Bank에 추가
- Base Bank와 유사도가 낮은 임베딩도 저장됨
- 저화질 얼굴도 저장됨
- 마스크/가림 상태의 얼굴도 저장됨

### After (개선 후)

- ✅ 검증을 통과한 임베딩만 Dynamic Bank에 추가
- ✅ Base Bank와 유사도가 0.6 이상인 임베딩만 저장
- ✅ 얼굴 크기가 200px 이상인 고화질 얼굴만 저장
- ✅ Occlusion 없는 얼굴만 저장

---

## 8. 문제 해결

### 8.1 Dynamic Bank가 생성되지 않음

**원인**: 모든 임베딩이 검증을 통과하지 못함

**확인 사항**:
- 서버 로그에서 검증 실패 이유 확인
- Base Bank와의 유사도가 0.6 이상인지 확인
- 얼굴 크기가 200px 이상인지 확인
- Occlusion 검증 통과 여부 확인

### 8.2 검증이 너무 엄격함

**해결 방법**:
- `backend/main.py`에서 임계값 조정:
  - `MIN_BASE_SIMILARITY_FOR_DYNAMIC_BANK = 0.6` → 더 낮은 값으로 조정
  - `MIN_FACE_SIZE_FOR_DYNAMIC_BANK = 200` → 더 작은 값으로 조정

### 8.3 서버가 시작되지 않음

**확인 사항**:
- PostgreSQL이 실행 중인지 확인
- `backend/.env` 파일이 올바르게 설정되었는지 확인
- 포트 5000이 사용 중인지 확인

---

## 9. 체크리스트

- [ ] Dynamic/Masked Bank 정리 완료
- [ ] 인물 등록 완료 (Base Bank 생성)
- [ ] 서버 시작 완료
- [ ] 브라우저에서 웹 인터페이스 접속
- [ ] 영상 업로드 및 감지 테스트
- [ ] 서버 콘솔에서 검증 로그 확인
- [ ] Dynamic Bank 파일 생성 확인
- [ ] 검증 통과/실패 통계 확인

---

**작성일**: 2024년  
**버전**: 1.0  
**작성자**: FaceWatch 개발팀











