"""평가 가능한 각도 확인 스크립트"""
import numpy as np
from pathlib import Path

manual_dir = Path('outputs/embeddings_manual')
dynamic_dir = Path('outputs/embeddings')
angles = ['left', 'right', 'top', 'front']
persons = ['ja', 'js', 'jw', 'yh']

print('='*70)
print('📊 평가 가능한 각도 확인')
print('='*70)

print('\n정답 데이터 (embeddings_manual):')
for p in persons:
    print(f'\n  {p}:')
    for a in angles:
        f = manual_dir / p / f'bank_{a}.npy'
        if f.exists():
            arr = np.load(f)
            print(f'    ✅ {a}: {arr.shape[0]}개')
        else:
            print(f'    ❌ {a}: 없음')

print('\nCCTV 데이터 (embeddings):')
for p in persons:
    print(f'\n  {p}:')
    for a in angles:
        f = dynamic_dir / p / f'bank_{a}.npy'
        if f.exists():
            arr = np.load(f)
            print(f'    ✅ {a}: {arr.shape[0]}개')
        else:
            print(f'    ❌ {a}: 없음')

print('\n비교 가능 여부:')
for p in persons:
    print(f'\n  {p}:')
    manual_angles = [a for a in angles if (manual_dir / p / f'bank_{a}.npy').exists()]
    dynamic_angles = [a for a in angles if (dynamic_dir / p / f'bank_{a}.npy').exists()]
    common = set(manual_angles) & set(dynamic_angles)
    missing = set(manual_angles) - set(dynamic_angles)
    
    if common:
        print(f'    ✅ 비교 가능: {sorted(common)}')
    else:
        print(f'    ❌ 비교 가능한 각도 없음')
    
    if missing:
        print(f'    ⚠️ CCTV 누락: {sorted(missing)}')

print('\n' + '='*70)
print('📈 평가 가능한 인물 요약')
print('='*70)

evaluable_persons = []
for p in persons:
    manual_angles = [a for a in angles if (manual_dir / p / f'bank_{a}.npy').exists()]
    dynamic_angles = [a for a in angles if (dynamic_dir / p / f'bank_{a}.npy').exists()]
    common = set(manual_angles) & set(dynamic_angles)
    
    if len(common) == 4:  # 4가지 각도 모두 있음
        evaluable_persons.append((p, '완전'))
    elif len(common) > 0:
        evaluable_persons.append((p, f'부분 ({len(common)}/4)'))
    else:
        evaluable_persons.append((p, '불가'))

for p, status in evaluable_persons:
    print(f'  {p}: {status}')



