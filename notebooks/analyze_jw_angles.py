"""jw의 각도 분포 분석 스크립트"""
import numpy as np
from pathlib import Path
import json

manual_dir = Path('outputs/embeddings_manual/jw')
dynamic_dir = Path('outputs/embeddings/jw')
angles_file = dynamic_dir / 'angles_dynamic.json'

angles_info = json.load(open(angles_file))

print('='*70)
print('jw CCTV 데이터 각도별 yaw 각도 분포 분석')
print('='*70)

# 각도별로 그룹화
angle_groups = {}
for i, (angle, yaw) in enumerate(zip(angles_info['angle_types'], angles_info['yaw_angles'])):
    if angle not in angle_groups:
        angle_groups[angle] = []
    angle_groups[angle].append(yaw)

# 평가에 사용되는 각도만 출력
for angle in ['front', 'left', 'right', 'top']:
    if angle in angle_groups:
        yaws = angle_groups[angle]
        print(f'\n{angle.upper()}:')
        print(f'  개수: {len(yaws)}개')
        print(f'  Yaw 범위: {min(yaws):.1f}° ~ {max(yaws):.1f}°')
        print(f'  평균 Yaw: {np.mean(yaws):.1f}°')
        print(f'  표준편차: {np.std(yaws):.1f}°')
        print(f'  각도 값: {[f"{y:.1f}°" for y in sorted(yaws)]}')
    else:
        print(f'\n{angle.upper()}: 없음')

print('\n' + '='*70)
print('각도 분류 기준 (CCTV 데이터)')
print('='*70)
print('  left: yaw 각도 -45° ~ -15°')
print('  right: yaw 각도 15° ~ 45°')
print('  top: pitch 각도 > 15°')
print('  front: yaw 각도 -15° ~ 15°')

print('\n' + '='*70)
print('분석 결과')
print('='*70)

# 각도별 범위 확인
for angle in ['front', 'left', 'right', 'top']:
    if angle in angle_groups:
        yaws = angle_groups[angle]
        out_of_range = []
        for yaw in yaws:
            if angle == 'left' and not (-45 <= yaw <= -15):
                out_of_range.append(yaw)
            elif angle == 'right' and not (15 <= yaw <= 45):
                out_of_range.append(yaw)
            elif angle == 'front' and not (-15 <= yaw <= 15):
                out_of_range.append(yaw)
        
        if out_of_range:
            print(f'\n[경고] {angle.upper()}: 범위를 벗어난 각도 발견')
            print(f'   범위 벗어남: {[f"{y:.1f}°" for y in out_of_range]}')
        else:
            print(f'\n[정상] {angle.upper()}: 모든 각도가 정상 범위 내')

