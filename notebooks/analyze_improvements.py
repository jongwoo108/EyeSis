"""
각도 기반 평가 시스템 개선 가능 사항 분석

코드 수정 없이 분석만 수행
"""
import json
import numpy as np
from pathlib import Path

print("=" * 80)
print("각도 기반 평가 시스템 개선 가능 사항 분석")
print("=" * 80)

# 데이터 로드
manual_angles_file = Path("outputs/embeddings_manual/jw/angles_manual.json")
dynamic_angles_file = Path("outputs/embeddings/jw/angles_dynamic.json")
extraction_summary = Path("outputs/embeddings_manual/extraction_summary.json")

manual_angles = json.load(open(manual_angles_file)) if manual_angles_file.exists() else None
dynamic_angles = json.load(open(dynamic_angles_file)) if dynamic_angles_file.exists() else None
summary = json.load(open(extraction_summary)) if extraction_summary.exists() else None

print("\n[1] 각도 정규화 문제")
print("-" * 80)
if manual_angles:
    manual_yaw = manual_angles["yaw_angles"][0]
    print(f"Manual right yaw: {manual_yaw:.2f}°")
    print(f"  → 이 값은 -90~90도 범위를 벗어남")
    print(f"  → estimate_face_angle()은 -90~90도 범위를 반환해야 하는데 162.7도가 나옴")
    print(f"  → 원인: normalized_offset이 1.0을 넘어가는 경우 (측면 프로필)")
    print(f"  → 해결: yaw 각도를 -180~180도 범위로 정규화 필요")

if dynamic_angles:
    right_yaws = [y for i, y in enumerate(dynamic_angles["yaw_angles"]) 
                  if dynamic_angles["angle_types"][i] == "right"]
    right_profile_yaws = [y for i, y in enumerate(dynamic_angles["yaw_angles"]) 
                          if dynamic_angles["angle_types"][i] == "right_profile"]
    print(f"\nCCTV right yaws: {[f'{y:.2f}°' for y in right_yaws]}")
    print(f"CCTV right_profile yaws: {[f'{y:.2f}°' for y in right_profile_yaws]}")
    
    if manual_angles:
        manual_yaw = manual_angles["yaw_angles"][0]
        print(f"\n각도 차이 계산:")
        for yaw in right_yaws + right_profile_yaws:
            diff1 = abs(manual_yaw - yaw)
            diff2 = 360 - diff1 if diff1 > 180 else diff1
            # -180~180 범위로 정규화
            norm_manual = ((manual_yaw + 180) % 360) - 180
            norm_yaw = ((yaw + 180) % 360) - 180
            norm_diff = abs(norm_manual - norm_yaw)
            print(f"  Manual {manual_yaw:.1f}° vs CCTV {yaw:.1f}°:")
            print(f"    - 현재 방식: {diff1:.1f}° → {diff2:.1f}°")
            print(f"    - 정규화 후: {norm_diff:.1f}° (manual={norm_manual:.1f}°, cctv={norm_yaw:.1f}°)")

print("\n[2] 각도 허용 범위 (angle_tolerance)")
print("-" * 80)
print("현재: ±5도 (고정)")
print("문제점:")
print("  - RIGHT가 스킵됨 (각도 차이가 5도보다 큼)")
print("  - 측면 프로필은 각도 변화에 민감하므로 더 넓은 범위 필요할 수 있음")
print("  - 정면은 각도 변화에 덜 민감하므로 더 좁은 범위 가능")
print("개선 방안:")
print("  - 각도 타입별로 다른 허용 범위 적용:")
print("    * front: ±10도 (각도 변화에 덜 민감)")
print("    * left/right: ±15도 (측면은 각도 범위가 넓음)")
print("    * left_profile/right_profile: ±20도 (극단적 측면)")
print("    * top: ±15도 (pitch 고려)")

print("\n[3] 각도 정보 부족 문제")
print("-" * 80)
if summary and "jw" in summary:
    jw_embeddings = summary["jw"]["embeddings"]
    print("JW의 manual 각도 정보:")
    for angle_type, emb_info in jw_embeddings.items():
        if emb_info.get("angles"):
            yaw = emb_info["angles"][0]["yaw_angle"]
            pitch = emb_info["angles"][0]["pitch_angle"]
            print(f"  {angle_type}: yaw={yaw:.1f}°, pitch={pitch:.1f}°")
        else:
            print(f"  {angle_type}: 각도 정보 없음 ❌")
    
    print("\n문제점:")
    print("  - left, top, front는 angles_manual.json에 없음")
    print("  - extract_angle_embeddings.py에서 얼굴 감지 실패로 각도 정보 저장 안 됨")
    print("  - 결과: 각도 기반 매칭을 사용하지 못하고 기존 방식(최대 유사도) 사용")
    print("개선 방안:")
    print("  - extract_angle_embeddings.py 재실행하여 모든 각도 정보 추출")
    print("  - 또는 각도 정보가 없어도 파일명 기반으로 대략적인 각도 추정 가능")

print("\n[4] Pitch 각도 활용 부족")
print("-" * 80)
print("현재: yaw 각도만 사용하여 매칭")
print("문제점:")
print("  - 같은 yaw 각도라도 pitch가 다르면 얼굴 모양이 크게 달라짐")
print("  - top 각도는 pitch > 15도로 분류되지만, yaw도 함께 고려하면 더 정확")
print("개선 방안:")
print("  - 2D 각도 거리 사용: sqrt((yaw_diff)² + (pitch_diff)²)")
print("  - 또는 yaw와 pitch를 각각 다른 가중치로 고려")
print("  - 예: yaw_diff_weight=1.0, pitch_diff_weight=0.5")

print("\n[5] 각도 차이 계산 로직 개선")
print("-" * 80)
print("현재 방식:")
print("  yaw_diff = abs(manual_yaw - cctv_yaw)")
print("  if yaw_diff > 180: yaw_diff = 360 - yaw_diff")
print("\n문제점:")
print("  - 162.7도와 19.3도의 경우:")
print("    abs(162.7 - 19.3) = 143.4도")
print("    143.4 < 180이므로 그대로 사용 → 잘못된 계산")
print("  - 실제로는 162.7도 = -17.3도로 정규화되어야 함")
print("\n개선 방안:")
print("  - 먼저 각도를 -180~180 범위로 정규화:")
print("    normalized = ((angle + 180) % 360) - 180")
print("  - 그 다음 차이 계산:")
print("    diff = abs(normalized_manual - normalized_cctv)")

print("\n[6] 각도 타입 매칭 개선")
print("-" * 80)
if dynamic_angles:
    print("현재: 같은 angle_type (left, right 등)끼리만 비교")
    print("\nCCTV 각도 분포:")
    angle_type_counts = {}
    for at in dynamic_angles["angle_types"]:
        angle_type_counts[at] = angle_type_counts.get(at, 0) + 1
    for at, count in sorted(angle_type_counts.items()):
        print(f"  {at}: {count}개")
    
    print("\n문제점:")
    print("  - right와 right_profile이 분리되어 있음")
    print("  - manual이 right인데 CCTV에 right_profile만 있는 경우 매칭 실패")
    print("  - 또는 반대 경우도 가능")
    print("\n개선 방안:")
    print("  - 각도 타입 매핑 사용:")
    print("    * right ↔ right_profile (유사한 각도 범위)")
    print("    * left ↔ left_profile")
    print("  - 또는 각도 타입 무시하고 실제 yaw 각도만으로 매칭")

print("\n[7] 유사도 임계값 고려")
print("-" * 80)
print("현재: 각도 차이만 고려하고 유사도는 무시")
print("문제점:")
print("  - 각도가 비슷해도 유사도가 낮으면 잘못된 매칭일 수 있음")
print("  - 예: 각도 차이 4도, 유사도 0.3 → 실제로는 다른 사람일 가능성")
print("\n개선 방안:")
print("  - 각도 차이와 유사도를 함께 고려하는 복합 점수:")
print("    score = similarity * (1 - angle_penalty)")
print("    angle_penalty = min(yaw_diff / max_tolerance, 1.0)")
print("  - 또는 각도 차이가 작을수록 유사도 가중치 증가")

print("\n[8] 다중 매칭 후보 고려")
print("-" * 80)
print("현재: 각도 범위 내에서 최고 유사도만 선택")
print("개선 방안:")
print("  - 각도 범위 내의 모든 후보를 고려하여 평균 유사도 계산")
print("  - 또는 상위 N개 후보의 가중 평균 (각도 차이에 반비례하는 가중치)")
print("  - 예: weight = 1 / (1 + yaw_diff²)")

print("\n" + "=" * 80)
print("우선순위별 개선 사항 요약")
print("=" * 80)
print("\n[높은 우선순위]")
print("1. 각도 정규화 수정 (-180~180 범위)")
print("2. 각도 차이 계산 로직 개선")
print("3. 각도 정보 부족 문제 해결 (extract_angle_embeddings.py 재실행)")
print("\n[중간 우선순위]")
print("4. 각도 타입별 허용 범위 조정")
print("5. 각도 타입 매핑 (right ↔ right_profile)")
print("\n[낮은 우선순위]")
print("6. Pitch 각도 활용")
print("7. 복합 점수 (각도 + 유사도)")
print("8. 다중 후보 고려")


