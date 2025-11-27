# yh의 fleft/fright 이미지 추가 후 작업 계획

## 현재 상황

### 1. 추가된 이미지
- `images/enroll/yh/yh_fleft.jpeg` (정면 왼쪽)
- `images/enroll/yh/yh_frignt.jpeg` (정면 오른쪽, 오타로 보임)

### 2. 현재 yh의 정답 데이터 상태
- `outputs/embeddings_manual/yh/angles_manual.json`: right만 있음 (yaw: 237.2°)
- `outputs/embeddings_manual/yh/`: front, left, right, top 파일은 존재하지만 각도 정보가 없음

### 3. CCTV 데이터 상태
- `outputs/embeddings/yh/angles_dynamic.json`: front, left, left_profile, right, right_profile 존재
- 각도별 bank 파일: bank_front.npy, bank_left.npy, bank_right.npy 등 존재

## 필요한 작업 단계

### [1단계] 파일명 패턴 인식 수정
**목적**: `fleft`, `fright` 파일명을 올바르게 인식

**현재 문제**:
- `extract_angle_embeddings.py`의 `ANGLE_PATTERNS`에 `fleft`, `fright` 패턴이 없음
- 현재는 "front"로 분류됨

**수정 필요 사항**:
- `ANGLE_PATTERNS`에 `fleft`, `fright` 패턴 추가
- 또는 `detect_angle_from_filename()` 함수에서 `fleft` → `left`, `fright` → `right`로 매핑
- 또는 새로운 각도 타입으로 추가 (`front_left`, `front_right`)

**권장 방법**:
- `fleft` → `left`로 매핑 (정면 왼쪽은 left 카테고리로)
- `fright` → `right`로 매핑 (정면 오른쪽은 right 카테고리로)
- 또는 실제 측정된 yaw 각도에 따라 자동 분류 (각도 정보가 있으면 그것을 우선 사용)

### [2단계] extract_angle_embeddings.py 재실행
**목적**: 새로운 이미지에서 임베딩 및 각도 정보 추출

**실행 명령**:
```bash
python scripts/extract_angle_embeddings.py
```

**예상 결과**:
- `outputs/embeddings_manual/yh/angles_manual.json`에 fleft, fright 각도 정보 추가
- `outputs/embeddings_manual/yh/bank_left.npy`, `bank_right.npy` 업데이트 (또는 새로 생성)
- `outputs/embeddings_manual/yh/embedding_left.npy`, `embedding_right.npy` 업데이트

**확인 사항**:
- 얼굴 감지 성공 여부
- 추출된 yaw 각도 값 확인
- 각도 타입 분류 확인 (left/right vs front)

### [3단계] CCTV 데이터 각도 분포 확인
**목적**: CCTV에서 fleft/fright 각도 범위의 임베딩이 있는지 확인

**확인 방법**:
- `outputs/embeddings/yh/angles_dynamic.json` 로드
- fleft/fright에서 추출된 yaw 각도 확인
- CCTV 데이터에서 해당 각도 범위(±5~15도)의 임베딩 존재 여부 확인

**예상 시나리오**:
1. **이상적인 경우**: CCTV에 해당 각도 범위의 임베딩이 충분히 있음
2. **부족한 경우**: CCTV에 해당 각도 범위의 임베딩이 적거나 없음
   - 이 경우 CCTV 영상에서 해당 각도가 충분히 검출되지 않았다는 의미
   - 추가 CCTV 영상 처리 필요할 수 있음

### [4단계] 각도 기반 평가 스크립트 작성/수정
**목적**: fleft/fright 각도에 대한 평가 수행

**옵션 1**: `notebooks/evaluate_jw_only.py`를 참고하여 `notebooks/evaluate_yh_only.py` 작성
**옵션 2**: 기존 평가 스크립트를 수정하여 yh도 평가 가능하도록 확장

**평가 항목**:
- fleft 각도: manual left vs CCTV left (각도 기반 매칭)
- fright 각도: manual right vs CCTV right (각도 기반 매칭)
- 각도 차이와 유사도 관계 분석
- 기존 right 각도와의 비교

### [5단계] 평가 결과 분석
**목적**: 새로운 정답 데이터가 CCTV 임베딩과 얼마나 유사한지 확인

**분석 항목**:
- 각도별 평균 유사도
- 각도 차이에 따른 유사도 변화
- 기존 right 각도와 새로운 fright 각도의 비교
- CCTV에서 검출된 각도 분포와의 일치도

**기대 효과**:
- 이전 정답 데이터(right, yaw: 237.2°)보다 CCTV에서 검출된 각도와 더 가까운 정답 데이터 사용
- 더 높은 유사도 점수 기대
- 각도 기반 매칭의 효과 검증

## 주의사항

### 1. 파일명 오타 확인
- `yh_frignt.jpeg` → `yh_fright.jpeg`로 수정 필요할 수 있음
- 또는 스크립트에서 오타도 인식하도록 처리

### 2. 각도 분류 기준
- `fleft`, `fright`는 "정면 왼쪽/오른쪽"을 의미
- 실제 측정된 yaw 각도가 중요 (파일명보다 우선)
- 예상 각도 범위:
  - fleft: -15° ~ -30° 정도 (left 카테고리)
  - fright: 15° ~ 30° 정도 (right 카테고리)

### 3. CCTV 데이터 충분성
- CCTV에서 해당 각도 범위의 임베딩이 충분히 있는지 확인 필요
- 부족한 경우 추가 CCTV 영상 처리 고려

### 4. 기존 데이터와의 관계
- 기존 right (yaw: 237.2°)는 실제로는 left_profile에 가까움
- 새로운 fright는 right 카테고리 (15~30도 범위 예상)
- 두 데이터는 서로 다른 각도이므로 직접 비교 불가

## 다음 단계 체크리스트

- [ ] `extract_angle_embeddings.py`에서 fleft/fright 패턴 인식 추가
- [ ] `extract_angle_embeddings.py` 재실행
- [ ] 추출된 각도 정보 확인 (angles_manual.json)
- [ ] CCTV 데이터 각도 분포 확인
- [ ] 평가 스크립트 작성/수정
- [ ] 평가 실행 및 결과 분석
- [ ] 결과 보고서 작성


