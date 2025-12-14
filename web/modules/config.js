// modules/config.js
// API 및 WebSocket URL 설정

export const API_BASE_URL = '/api';

export const WS_URL = `ws${window.location.protocol === 'https:' ? 's' : ''}://${window.location.host}/ws/detect`;

export const WS_TEST_URL = `ws${window.location.protocol === 'https:' ? 's' : ''}://${window.location.host}/ws/test`; // 테스트용


// 인물 이름 매핑 (person_id → 표시 이름)
export const personNameMapping = {
    'yh': '황윤하',
    'js': '이지선',
    'jw': '신종우',
    'ja': '양정아'
};