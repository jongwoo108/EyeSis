// modules/utils.js
// 유틸리티 함수들

// 카테고리 텍스트에 따른 색상 결정 함수
export function getCategoryStyle(categoryText) {
    if (!categoryText || categoryText.trim() === '') {
        return {
            bgColor: 'bg-gray-100',
            textColor: 'text-gray-600',
            borderColor: 'border-gray-300'
        };
    }

    const lowerText = categoryText.toLowerCase();

    // 범죄 관련 키워드
    if (lowerText.includes('범죄') || lowerText.includes('수배') ||
        lowerText.includes('스파이') || lowerText.includes('강도') ||
        lowerText.includes('criminal') || lowerText.includes('wanted')) {
        return {
            bgColor: 'bg-red-100',
            textColor: 'text-red-700',
            borderColor: 'border-red-300'
        };
    }

    // 실종 관련 키워드
    if (lowerText.includes('실종') || lowerText.includes('가출') ||
        lowerText.includes('missing')) {
        return {
            bgColor: 'bg-blue-100',
            textColor: 'text-blue-700',
            borderColor: 'border-blue-300'
        };
    }

    // 치매/노인 관련 키워드
    if (lowerText.includes('치매') || lowerText.includes('노인') ||
        lowerText.includes('환자') || lowerText.includes('dementia')) {
        return {
            bgColor: 'bg-orange-100',
            textColor: 'text-orange-700',
            borderColor: 'border-orange-300'
        };
    }

    // 아동 관련 키워드
    if (lowerText.includes('아동') || lowerText.includes('child')) {
        return {
            bgColor: 'bg-yellow-100',
            textColor: 'text-yellow-700',
            borderColor: 'border-yellow-300'
        };
    }

    // 기타 (Gray/Purple)
    return {
        bgColor: 'bg-gray-100',
        textColor: 'text-gray-700',
        borderColor: 'border-gray-300'
    };
}

// 카테고리 텍스트 가져오기 함수
export function getCategoryText(person) {
    if (!person) return '미상';

    // person.person_type(최상위) 또는 info 내부 확인
    const category = person.person_type || person.info?.category || person.info?.person_type;

    if (category) {
        // 기본 카테고리 매핑
        const categoryMap = {
            'criminal': '범죄자',
            'missing': '실종자',
            'dementia': '치매환자',
            'child': '미아',
            'wanted': '수배자'
        };

        // 매핑된 값이 있으면 사용, 없으면 원본 값 사용 (커스텀 카테고리)
        return categoryMap[category] || category;
    }

    // info가 없으면 is_criminal 기반으로 기본값 반환
    if (person.is_criminal) {
        return '범죄자';
    }

    return '실종자';
}

// 각도 타입을 표시 텍스트로 변환
export function getAngleDisplayText(angleType) {
    const angleMap = {
        'left': '왼쪽',
        'right': '오른쪽',
        'left_profile': '왼쪽 프로필',
        'right_profile': '오른쪽 프로필',
        'front': '정면',
        'unknown': ''
    };
    return angleMap[angleType] || '';
}

// 시간 포맷팅 (초 → MM:SS)
export function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}