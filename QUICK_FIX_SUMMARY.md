# 빠른 수정 요약

## 🔧 변경 사항

### 1. 조건 완화 (디버깅용)

```python
# 이전 (너무 빡빡함)
MASKED_BANK_MASK_PROB_THRESHOLD = 0.7
MASKED_CANDIDATE_MIN_SIM = 0.30
MASKED_CANDIDATE_MIN_FRAMES = 5

# 변경 후 (완화)
MASKED_BANK_MASK_PROB_THRESHOLD = 0.5  # 0.7 → 0.5
MASKED_CANDIDATE_MIN_SIM = 0.25  # 0.30 → 0.25
MASKED_CANDIDATE_MIN_FRAMES = 3  # 5 → 3
```

### 2. 로그 개선

이제 다음 로그들이 더 명확하게 출력됩니다:

- `🎭 [MASKED CAND] ✅ 감지됨!` - 조건 충족 시
- `🎭 [MASKED CAND] ❌ 조건 미충족: ... | 이유: ...` - 조건 미충족 시 (이유 표시)
- `🎭 [MASKED CAND] ⚠️ 매칭 실패: best_person_id=unknown` - 매칭 자체가 실패한 경우

## 📊 다음 단계

1. **서버 재시작** (자동 reload되지만 확인)
2. **마스크 쓴 얼굴 테스트**
3. **로그 확인:**
   - `[MASKED CAND]` 로그가 출력되는지 확인
   - 조건 미충족 시 어떤 이유인지 확인
   - 추적 중 로그 확인 (`📊 [MASKED CAND] 추적 중`)
4. **파일 생성 확인:**
   - `outputs/embeddings/{person_id}/bank_masked.npy` 파일이 생성되는지 확인

## 🎯 예상 로그

정상 작동 시:

```
🎭 [MASKED CAND] ✅ 감지됨! person_id=yh, base_sim=0.280, mask_prob=0.600, threshold=0.400
🆕 [MASKED CAND] 새 track 생성: yh (track_id=track_0, base_sim=0.280)
📊 [MASKED CAND] 추적 중: yh (1/3프레임, base_sim=0.280)
📊 [MASKED CAND] 추적 중: yh (2/3프레임, base_sim=0.275)
📊 [MASKED CAND] 추적 중: yh (3/3프레임, base_sim=0.285)
✅ [MASKED BANK] 자동 추가 성공: yh (연속 3프레임, base_sim=0.285, mask_prob=0.600)
✅ [Masked BANK] 파일 저장: outputs/embeddings/yh/bank_masked.npy (총 1개 임베딩, angle: front)
```

## ⚠️ 여전히 안 되면

조건을 더 완화:

```python
MASKED_CANDIDATE_MIN_SIM = 0.20  # 더 낮춤
MASKED_BANK_MASK_PROB_THRESHOLD = 0.3  # 더 낮춤
MASKED_CANDIDATE_MIN_FRAMES = 2  # 더 낮춤
```

또는 수동으로 임베딩 저장 (빠른 테스트용)





