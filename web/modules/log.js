import { state } from "./state.js";
import { initUI } from "./ui.js";

const UI = initUI();

// 감지 로그 아이템 추가 함수
export function addDetectionLogItem(data, isAlert, videoTime, snapshotImage) {
    if (!data) {
        console.warn('⚠️ addDetectionLogItem: data가 없습니다.');
        return;
    }

    // UI.detectionLogList가 없으면 동적으로 찾기
    const detectionLogList = UI.detectionLogList || document.getElementById('detectionLogList');
    if (!detectionLogList) {
        console.warn('⚠️ detectionLogList를 찾을 수 없습니다.');
        return;
    }

    const status = data.status || 'unknown';
    const name = data.name || 'Unknown';
    const personId = data.person_id || data.name || 'unknown';
    const confidence = data.confidence ? (typeof data.confidence === 'number' ? data.confidence.toFixed(1) : data.confidence) : '0.0';

    // Unknown 상태는 로그에 기록하지 않음
    if (status === 'unknown' || name === 'Unknown') {
        return;
    }

    // 비디오 타임스탬프가 없으면 스킵
    if (videoTime === undefined || videoTime === null || isNaN(videoTime)) {
        console.warn('⚠️ addDetectionLogItem: videoTime이 없습니다.');
        return;
    }

    // 중복 방지: 동일 인물이 최근 쿨타임(5초) 이내에 로그가 추가되었는지 확인 (비디오 타임스탬프 기반)
    const lastLogVideoTime = state.lastLogTimeByPerson.get(personId);
    if (lastLogVideoTime !== undefined) {
        const timeSinceLastLog = videoTime - lastLogVideoTime; // 비디오 시간 차이 (초)

        if (timeSinceLastLog < state.LOG_COOLDOWN_SECONDS) {
            // 쿨타임 이내에 동일 인물의 로그가 있으면 스킵
            console.log(`⏭️ 로그 스킵: ${name} (${timeSinceLastLog.toFixed(1)}초 전에 추가됨, 쿨타임: ${state.LOG_COOLDOWN_SECONDS}초)`);
            return;
        }
    }

    // 얼굴 면적 계산 (bbox: [x1, y1, x2, y2])
    let faceArea = 0;
    if (data.metadata && data.metadata.bbox) {
        const bbox = data.metadata.bbox;
        const width = bbox[2] - bbox[0];
        const height = bbox[3] - bbox[1];
        faceArea = width * height;
    } else if (data.bbox) {
        // 백엔드에서 직접 bbox를 보내주는 경우 (구조에 따라 다름)
        const bbox = data.bbox;
        const width = bbox[2] - bbox[0];
        const height = bbox[3] - bbox[1];
        faceArea = width * height;
    }

    console.log(`✅ 로그 추가: ${name} (${status}) - ${confidence}% @ ${videoTime.toFixed(1)}초`);

    // 마지막 로그 비디오 타임스탬프 업데이트
    state.lastLogTimeByPerson.set(personId, videoTime);

    // 시간 포맷팅 (비디오 타임스탬프)
    const formatVideoTime = (seconds) => {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    // 상태에 따라 색상 결정
    let nameColorClass, borderClass, bgClass;
    if (status === 'criminal' || isAlert) {
        nameColorClass = "text-red-600";
        borderClass = "border-red-200";
        bgClass = "bg-red-50";
    } else if (status === 'missing') {
        nameColorClass = "text-blue-600";
        borderClass = "border-blue-200";
        bgClass = "bg-gray-50";
    } else {
        nameColorClass = "text-gray-600";
        borderClass = "border-gray-100";
        bgClass = "bg-gray-50";
    }

    // 썸네일 이미지 처리 (사각형 썸네일)
    let thumbnailHTML = '';
    if (snapshotImage) {
        thumbnailHTML = `<img src="${snapshotImage}" alt="${name}" class="w-14 h-14 rounded-md object-cover border-2 ${status === 'criminal' || isAlert ? 'border-red-300' : status === 'missing' ? 'border-blue-300' : 'border-gray-300'} shadow-sm">`;
    } else {
        // 기본 아이콘 (사각형)
        thumbnailHTML = `<div class="w-14 h-14 rounded-md ${status === 'criminal' || isAlert ? 'bg-red-100 border-2 border-red-300' : status === 'missing' ? 'bg-blue-100 border-2 border-blue-300' : 'bg-gray-100 border-2 border-gray-300'} flex items-center justify-center shadow-sm">
            <svg class="w-7 h-7 ${status === 'criminal' || isAlert ? 'text-red-600' : status === 'missing' ? 'text-blue-600' : 'text-gray-400'}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
            </svg>
        </div>`;
    }

    // 로그 아이템 생성
    const logItem = document.createElement('li');
    const logId = `${personId}_${videoTime}_${Date.now()}`;
    logItem.className = `flex items-center gap-3 p-3 ${bgClass} rounded-lg border ${borderClass} hover:bg-gray-100 transition-colors cursor-pointer`;
    logItem.dataset.videoTime = videoTime;
    logItem.dataset.personId = personId;
    logItem.dataset.logId = logId;

    logItem.innerHTML = `
        ${thumbnailHTML}
        <div class="flex-1 min-w-0">
            <div class="font-bold text-sm ${nameColorClass}">${name}</div>
            <div class="text-xs text-gray-500">정확도 ${confidence}%</div>
        </div>
        <div class="text-xs text-gray-400 whitespace-nowrap">${formatVideoTime(videoTime || 0)}</div>
    `;

    // 클릭 이벤트: 비디오 시점으로 이동
    logItem.addEventListener('click', () => {
        if (UI.video && videoTime !== undefined && !isNaN(videoTime)) {
            UI.video.currentTime = videoTime;
            UI.video.play();
            console.log(`▶️ 비디오 이동: ${formatVideoTime(videoTime)}`);
        }
    });

    // 리스트 최상단에 추가 (prepend)
    const firstChild = detectionLogList.firstElementChild;
    if (firstChild && firstChild.textContent && firstChild.textContent.trim() === '대기 중...') {
        detectionLogList.removeChild(firstChild);
    }
    detectionLogList.insertBefore(logItem, detectionLogList.firstChild);

    // 스크롤 관리: 사용자가 스크롤을 올려서 과거 내역을 보고 있지 않으면 최상단 유지
    if (UI.detectionInfo) {
        const isAtTop = UI.detectionInfo.scrollTop < 50; // 50px 이내면 최상단으로 간주
        if (isAtTop) {
            UI.detectionInfo.scrollTop = 0;
        }
    }

    // 로그 배열에 추가 (누적 히스토리, 최대 200개 유지)
    state.detectionLogs.unshift({
        id: `${personId}_${videoTime}_${Date.now()}`, // 고유 ID
        name,
        personId,
        status,
        confidence,
        videoTime, // 감지된 시점의 비디오 타임스탬프 (고정)
        snapshotImage, // 감지된 순간의 스냅샷 (고정)
        faceArea, // 얼굴 면적 (Best Shot 선정용)
        timestamp: Date.now() // 로그 생성 시간
    });
    if (state.detectionLogs.length > 200) {
        // 오래된 로그 제거 (DOM에서도 제거)
        const removedLog = state.detectionLogs.pop();
        const removedElement = detectionLogList.querySelector(`[data-log-id="${removedLog.id}"]`);
        if (removedElement) {
            detectionLogList.removeChild(removedElement);
        }
    }
}

// 패널 업데이트 헬퍼 (누적 히스토리 방식)
export function updateDetectionPanel(data, isAlert, videoTime, snapshotImage) {
    // UI.detectionLogList가 없으면 동적으로 찾기
    const detectionLogList = UI.detectionLogList || document.getElementById('detectionLogList');

    // 초기화 금지: data가 null이거나 message일 때만 초기화 (누적 히스토리 유지)
    if (!data) {
        // 초기 상태일 때만 "대기 중..." 표시 (리스트가 비어있을 때만)
        if (detectionLogList && detectionLogList.children.length === 0) {
            detectionLogList.innerHTML = '<li class="text-gray-500 text-center py-4 tracking-tight">감지 대기 중...</li>';
        }
        return;
    }

    // 메시지 타입은 특별 처리 (시스템 메시지)
    if (data.message) {
        // 누적 히스토리 방식이므로 메시지는 로그에 추가하지 않음 (기존 로그 유지)
        return;
    }

    // 새로운 로그 아이템 추가 (누적)
    console.log('📝 updateDetectionPanel 호출:', { name: data.name, status: data.status, videoTime, hasSnapshot: !!snapshotImage });
    addDetectionLogItem(data, isAlert, videoTime, snapshotImage);
}

// 감지 로그를 CSV 파일로 다운로드
export function downloadLogToCSV() {
    // 1. 데이터 수집 (state.detectionLogs 배열 사용)
    if (!state.detectionLogs || state.detectionLogs.length === 0) {
        alert('저장할 감지 기록이 없습니다.');
        return;
    }

    // 2. CSV 헤더 및 데이터 행 생성
    const rows = [["시간", "이름", "구분", "정확도(%)"]];

    // 시간 포맷팅 함수
    const formatVideoTime = (seconds) => {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    // 구분 텍스트 변환 함수
    const getStatusText = (status) => {
        if (status === 'criminal') return '범죄자';
        if (status === 'missing') return '실종자';
        return '미확인';
    };

    // 로그 데이터를 시간순으로 정렬 (오래된 것부터)
    const sortedLogs = [...state.detectionLogs].sort((a, b) => a.videoTime - b.videoTime);

    // 각 로그를 CSV 행으로 변환
    sortedLogs.forEach(log => {
        const time = formatVideoTime(log.videoTime || 0);
        const name = log.name || 'Unknown';
        const status = getStatusText(log.status || 'unknown');
        const confidence = log.confidence ? (typeof log.confidence === 'number' ? log.confidence.toFixed(1) : log.confidence) : '0.0';

        rows.push([time, name, status, confidence]);
    });

    // 3. CSV 문자열 생성
    let csvContent = rows.map(row => {
        // CSV 이스케이프 처리 (쉼표, 따옴표, 줄바꿈 포함 시)
        return row.map(cell => {
            const cellStr = String(cell);
            if (cellStr.includes(',') || cellStr.includes('"') || cellStr.includes('\n')) {
                return `"${cellStr.replace(/"/g, '""')}"`;
            }
            return cellStr;
        }).join(',');
    }).join('\n');

    // 4. BOM 추가 (한글 깨짐 방지)
    const bom = '\uFEFF';
    csvContent = bom + csvContent;

    // 5. 파일 다운로드
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);

    // 파일명 생성 (YYYYMMDD_HHMM 형식)
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const filename = `detection_log_${year}${month}${day}_${hours}${minutes}.csv`;

    link.download = filename;
    link.click();

    // 메모리 정리
    URL.revokeObjectURL(link.href);

    console.log(`✅ CSV 다운로드 완료: ${filename} (${state.detectionLogs.length}개 기록)`);
}