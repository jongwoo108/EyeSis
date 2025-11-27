// script.js

// ==========================================
// 전역 변수 및 상태 관리
// ==========================================
// 상대 경로 사용 (ngrok 사용 시 자동으로 도메인 적용됨)
const API_BASE_URL = '/api';
const WS_URL = `ws${window.location.protocol === 'https:' ? 's' : ''}://${window.location.host}/ws/detect`;
const WS_TEST_URL = `ws${window.location.protocol === 'https:' ? 's' : ''}://${window.location.host}/ws/test`; // 테스트용

const state = {
    selectedFile: null,
    videoUploaded: false, // 영상 업로드 여부
    selectedSuspects: [], // 여러 명 선택 가능 (배열)
    isDetectionActive: false,
    detectionInterval: null,
    videoCanvas: null, // 캡처용 캔버스
    detectionCanvas: null, // 박스 렌더링용 캔버스
    detectionCtx: null, // 박스 렌더링용 컨텍스트
    isProcessing: false, // [중요] 서버 과부하 방지용 플래그 (하나로 합침)
    // WebSocket 관련
    ws: null, // WebSocket 연결
    wsReconnectAttempts: 0, // 재연결 시도 횟수
    wsReconnectTimer: null, // 재연결 타이머
    isWsConnected: false, // 연결 상태
    wsConfigReady: false, // WebSocket 설정 완료 여부 (suspect_ids 설정 완료)
    frameId: 0, // 프레임 ID 추적
    useWebSocket: true, // WebSocket 사용 여부 (실패 시 HTTP로 폴백)
    lastDetections: null, // 마지막 감지 결과 (폴백용)
    lastDetections: null, // 마지막 감지 결과 (폴백용)
    lastDetectionTime: 0, // 마지막 감지 시간
    heartbeatInterval: null, // 하트비트 타이머
    // 스냅샷 관리
    sessionId: null, // 세션 ID
    snapshots: [], // 범죄자 감지 스냅샷 배열
    nextSnapshotId: 1, // 스냅샷 ID 자동 증가
    // 영상 클립 관리
    detectionClips: [], // 범죄자 감지 구간 배열 [{id, startTime, endTime, personId, personName, similarity, ...}]
    activeClips: {}, // 현재 활성 클립 {personId: {id, startTime, personId, personName, similarity, ...}}
    nextClipId: 1, // 클립 ID 자동 증가
    
    // 스냅샷 선택 관리
    selectedSnapshots: [], // 선택된 스냅샷 ID 배열
    
    // 클립 선택 관리
    selectedClips: [], // 선택된 클립 ID 배열
    
    // 타임라인 렌더링 타이머
    timelineRenderTimer: null, // 타임라인 재렌더링 배치 처리용 타이머
    
    // 감지 로그 관리
    detectionLogs: [], // 감지 로그 배열
    lastLogTimeByPerson: new Map(), // 인물별 마지막 로그 비디오 타임스탬프 추적 (중복 방지용) - Map<PersonID, VideoTime>
    LOG_COOLDOWN_SECONDS: 5 // 로그 쿨타임 (초)
};

// DOM 요소
const UI = {
    // 메인 화면
    dashboard: document.getElementById('dashboardScreen'),
    emptyStateCard: document.getElementById('emptyStateCard'),
    // 모달
    uploadModal: document.getElementById('uploadModal'),
    suspectSelectModal: document.getElementById('suspectSelectModal'),
    // 헤더 버튼
    openUploadModalBtn: document.getElementById('openUploadModalBtn'),
    openSuspectModalBtn: document.getElementById('openSuspectModalBtn'),
    // 모달 닫기 버튼
    closeUploadModal: document.getElementById('closeUploadModal'),
    closeSuspectModal: document.getElementById('closeSuspectModal'),
    // 비디오 관련
    video: document.getElementById('mainVideo'),
    detectionCanvas: document.getElementById('detectionCanvas'),
    videoFile: document.getElementById('videoFile'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    fileInfo: document.getElementById('fileInfo'),
    fileName: document.getElementById('fileName'),
    // 인물 선택
    suspectCardsContainer: document.getElementById('suspectCardsContainer'),
    proceedBtn: document.getElementById('proceedToDashboard'),
    selectedSuspectName: document.getElementById('selectedSuspectName'),
    selectedSuspectInfo: document.getElementById('selectedSuspectInfo'),
    // 제어
    detectionFilter: document.getElementById('detectionFilter'),
    detectionInfo: document.getElementById('detectionInfo'),
    detectionLogList: document.getElementById('detectionLogList'),
    downloadLogBtn: document.getElementById('downloadLogBtn'),
    // 용의자 추가 모달
    addSuspectModal: document.getElementById('addSuspectModal'),
    addSuspectBtn: document.getElementById('addSuspectBtn'),
    closeAddSuspectModal: document.getElementById('closeAddSuspectModal'),
    addSuspectForm: document.getElementById('addSuspectForm'),
    enrollPersonId: document.getElementById('enrollPersonId'),
    enrollName: document.getElementById('enrollName'),
    enrollImage: document.getElementById('enrollImage'),
    enrollIsCriminal: document.getElementById('enrollIsCriminal'),
    imagePreview: document.getElementById('imagePreview'),
    previewImg: document.getElementById('previewImg'),
    imagePlaceholder: document.getElementById('imagePlaceholder'),
    enrollSuccess: document.getElementById('enrollSuccess'),
    enrollError: document.getElementById('enrollError'),
    submitEnrollBtn: document.getElementById('submitEnrollBtn'),
    // 클립/스냅샷 버튼
    viewClipsBtn: document.getElementById('viewClipsBtn'),
    viewSnapshotsBtn: document.getElementById('viewSnapshotsBtn'),
    // 모달
    clipModal: document.getElementById('clipModal'),
    snapshotModal: document.getElementById('snapshotModal'),
    cancelEnrollBtn: document.getElementById('cancelEnrollBtn'),
    // 프레임 추출
    extractFramesBtn: document.getElementById('extractFramesBtn')
};

// ==========================================
// 인물 목록 로드 및 카드 생성
// ==========================================

// 인물 이름 매핑 (person_id → 표시 이름)
const personNameMapping = {
    'yh': '황윤하',
    'js': '이지선',
    'jw': '신종우',
    'ja': '양정아'
};

// 인물 목록을 서버에서 가져오기
async function loadPersons() {
    try {
        const response = await fetch(`${API_BASE_URL}/persons`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        if (data.success && data.persons) {
            return data.persons;
        }
        return [];
    } catch (error) {
        console.error("인물 목록 로드 실패:", error);
        return [];
    }
}

// 인물 카드 동적 생성 (다중 선택 가능)
function createSuspectCard(person) {
    const displayName = personNameMapping[person.id] || person.name;
    const isCriminal = person.is_criminal;

    // 색상 및 텍스트 설정
    const bgColor = isCriminal ? 'bg-red-100' : 'bg-blue-100';
    const textColor = isCriminal ? 'text-red-600' : 'text-blue-600';
    const statusText = isCriminal ? '범죄자' : '실종자';

    const card = document.createElement('div');
    card.className = 'suspect-card bg-white rounded-lg shadow-sm overflow-hidden cursor-pointer transform hover:scale-105 transition-all duration-200 relative';
    card.setAttribute('data-suspect-id', person.id);
    card.setAttribute('data-is-thief', isCriminal.toString());

    // 이미지 URL이 있으면 사용, 없으면 기본 이모지
    const imageUrl = person.image_url || null;
    const imageHtml = imageUrl
        ? `<img src="${imageUrl}" alt="${displayName}" class="w-full h-full object-cover" onerror="this.parentElement.innerHTML='<span class=\\'text-6xl\\'>👤</span>'">`
        : `<span class="text-6xl">👤</span>`;

    // 체크박스 아이콘 추가
    card.innerHTML = `
        <div class="absolute top-2 right-2 w-6 h-6 rounded-full border-2 border-gray-300 bg-white flex items-center justify-center checkmark hidden">
            <svg class="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path>
            </svg>
        </div>
        <div class="h-48 ${bgColor} flex items-center justify-center overflow-hidden">
            ${imageHtml}
        </div>
        <div class="p-4">
            <h3 class="font-bold text-lg">${displayName}</h3>
            <p class="text-sm ${textColor}">${statusText}</p>
        </div>
    `;

    // 클릭 이벤트 리스너 추가 (다중 선택)
    card.addEventListener('click', function () {
        const suspectId = person.id;
        const isSelected = state.selectedSuspects.some(s => s.id === suspectId);

        if (isSelected) {
            // 선택 해제
            state.selectedSuspects = state.selectedSuspects.filter(s => s.id !== suspectId);
            this.classList.remove('ring-4', 'ring-blue-500');
            this.querySelector('.checkmark').classList.add('hidden');
        } else {
            // 선택 추가
            state.selectedSuspects.push({
                id: person.id,
                name: displayName,
                isThief: isCriminal
            });
            this.classList.add('ring-4', 'ring-blue-500');
            this.querySelector('.checkmark').classList.remove('hidden');
        }

        // 선택된 인물 정보 업데이트
        updateSelectedSuspectsInfo();

        // 최소 1명 이상 선택해야 진행 버튼 활성화
        UI.proceedBtn.disabled = state.selectedSuspects.length === 0;
    });

    return card;
}

// 선택된 용의자 정보 업데이트
function updateSelectedSuspectsInfo() {
    if (state.selectedSuspects.length === 0) {
        UI.selectedSuspectInfo.classList.add('hidden');
        return;
    }

    UI.selectedSuspectInfo.classList.remove('hidden');

    // 선택된 용의자 목록 표시
    const namesList = state.selectedSuspects.map(s => s.name).join(', ');
    const countText = state.selectedSuspects.length > 1
        ? `${state.selectedSuspects.length}명 선택됨`
        : '1명 선택됨';

    UI.selectedSuspectName.innerHTML = `
        <span class="font-semibold">${namesList}</span>
        <span class="text-sm text-gray-600 ml-2">(${countText})</span>
    `;
}

// 인물 카드들을 동적으로 생성하고 표시
async function renderSuspectCards() {
    const persons = await loadPersons();

    // 컨테이너 초기화
    UI.suspectCardsContainer.innerHTML = '';

    if (persons.length === 0) {
        UI.suspectCardsContainer.innerHTML = `
            <div class="col-span-full text-center py-8 text-gray-500">
                <p>등록된 인물이 없습니다.</p>
            </div>
        `;
        return;
    }

    // 각 인물에 대해 카드 생성 및 추가
    persons.forEach(person => {
        const card = createSuspectCard(person);

        // 이미 선택된 용의자인지 확인하여 선택 상태 복원
        const isSelected = state.selectedSuspects.some(s => s.id === person.id);
        if (isSelected) {
            card.classList.add('ring-4', 'ring-blue-500');
            card.querySelector('.checkmark').classList.remove('hidden');
        }

        UI.suspectCardsContainer.appendChild(card);
    });

    // 선택된 용의자 정보 업데이트
    updateSelectedSuspectsInfo();
}

// ==========================================
// 핵심 로직: 프레임 캡처 및 서버 전송
// ==========================================

// 비디오 프레임 캡처용 캔버스 초기화
function initCaptureCanvas() {
    if (!state.videoCanvas) {
        state.videoCanvas = document.createElement('canvas');
    }

    // 박스 렌더링용 캔버스 초기화
    if (!state.detectionCanvas) {
        state.detectionCanvas = UI.detectionCanvas;
        state.detectionCtx = state.detectionCanvas.getContext('2d');
    }

    // 비디오 크기 변경 시 캔버스 크기 조정
    UI.video.addEventListener('loadedmetadata', updateCanvasSize);
    window.addEventListener('resize', updateCanvasSize);

    // 비디오 재생 시 AI 감지 자동 활성화
    UI.video.addEventListener('play', () => {
        // 비디오가 재생되면 AI 감지 자동 활성화
        if (!state.isDetectionActive && UI.detectionFilter) {
            console.log("▶️ 비디오 재생 감지, AI 감지 자동 활성화");
            UI.detectionFilter.checked = true;
            state.isDetectionActive = true;

            // 처리 상태 초기화
            state.isProcessing = false;

            // WebSocket 연결 시도 (백그라운드에서 연결 시도)
            if (state.useWebSocket && !state.isWsConnected) {
                connectWebSocket();
                // WebSocket 연결 완료 및 설정 완료 후 자동으로 WebSocket으로 전환됨
                // 하지만 연결 완료 전까지는 HTTP로 즉시 시작
            }

            // WebSocket 연결 상태와 관계없이 HTTP로 즉시 시작 (WebSocket 준비되면 자동 전환)
            // 이렇게 하면 화면 전환 직후에도 통신이 바로 시작됨
            console.log("🚀 HTTP 모드로 감지 시작 (WebSocket 준비되면 자동 전환)");
            processRealtimeDetection();
            state.detectionInterval = setInterval(processRealtimeDetection, 100);
        }
    });

    // 비디오 종료 시 감지 루프 자동 중지
    UI.video.addEventListener('ended', () => {
        if (state.isDetectionActive) {
            console.log("⏹️ 비디오 종료됨, 감지 루프 자동 중지");
            state.isDetectionActive = false;
            if (UI.detectionFilter) {
                UI.detectionFilter.checked = false;
            }
            clearInterval(state.detectionInterval);

            // 모든 활성 클립 종료
                const endTime = UI.video.currentTime;
            Object.keys(state.activeClips).forEach(personId => {
                const clip = state.activeClips[personId];
                clip.endTime = endTime;
                state.detectionClips.push(clip);
                console.log(`✅ 클립 종료: ${clip.personName} (${clip.startTime.toFixed(1)}s - ${endTime.toFixed(1)}s)`);
                delete state.activeClips[personId];
            });
                updateClipCount();

            // WebSocket 재연결 중지
            if (state.wsReconnectTimer) {
                clearTimeout(state.wsReconnectTimer);
                state.wsReconnectTimer = null;
                console.log("⏹️ WebSocket 재연결 중지 (비디오 종료)");
            }

            updateDetectionPanel({ message: "비디오 종료됨" });
        }
    });

    // 비디오 컨테이너 크기 변경 감지 (ResizeObserver 사용)
    if (window.ResizeObserver) {
        const resizeObserver = new ResizeObserver(() => {
            updateCanvasSize();
        });
        resizeObserver.observe(UI.video.parentElement);
    }
}

// 캔버스 크기를 비디오 크기에 맞춤
function updateCanvasSize() {
    if (state.detectionCanvas && UI.video) {
        // 비디오의 실제 표시 크기 가져오기
        const videoRect = UI.video.getBoundingClientRect();
        const containerRect = UI.video.parentElement.getBoundingClientRect();

        // 캔버스 크기를 컨테이너와 정확히 일치시키기
        state.detectionCanvas.width = videoRect.width;
        state.detectionCanvas.height = videoRect.height;
    }
}

// ==========================================
// 인물별 타임라인 트랙 생성
// ==========================================

/**
 * 인물별 타임라인 트랙 생성
 * @param {string} personId - 인물 ID
 * @param {string} personName - 인물 이름
 * @param {boolean} isCriminal - 범죄자 여부
 * @returns {HTMLElement} 생성된 트랙 요소
 */
function createTimelineTrack(personId, personName, isCriminal) {
    const track = document.createElement('div');
    const bgColor = isCriminal ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200';
    const textColor = isCriminal ? 'text-red-700' : 'text-green-700';
    const labelText = isCriminal ? '범죄자' : '실종자';

    track.className = `timeline-track ${bgColor} border rounded-sm px-2 py-1.5`;
    track.dataset.personId = personId;
    track.innerHTML = `
            <div class="flex items-center justify-between mb-0.5">
            <span class="text-xs font-semibold ${textColor}">${personName} (${labelText})</span>
            <span class="text-xs text-gray-500">클릭 시 해당 시점으로 이동</span>
        </div>
            <div class="timeline-bar relative h-3 bg-white rounded-sm cursor-pointer transition-all duration-200 hover:scale-y-110 hover:brightness-110">
                <!-- 마커들이 추가될 영역 -->
            </div>
        `;

    return track;
}

/**
 * 선택된 인물들의 타임라인 트랙 초기화 (사전 생성)
 */
function initializeTimelinesForSelectedPersons() {
    const timelinesContainer = document.getElementById('timelinesContainer');
    if (!timelinesContainer) {
        console.error('❌ 타임라인 컨테이너를 찾을 수 없습니다!');
        return;
    }

    // 기존 타임라인 모두 제거
    timelinesContainer.innerHTML = '';
    console.log('🗑️ 기존 타임라인 초기화');

    // 선택된 각 인물에 대해 타임라인 트랙 생성
    state.selectedSuspects.forEach(suspect => {
        const track = createTimelineTrack(
            suspect.id,
            suspect.name,
            suspect.isThief  // isThief가 true면 범죄자
        );
        timelinesContainer.appendChild(track);
        console.log(`✅ 타임라인 트랙 생성: ${suspect.name} (${suspect.isThief ? '범죄자' : '실종자'})`);
    });

    console.log(`📊 총 ${state.selectedSuspects.length}개 타임라인 트랙 생성 완료`);
}

// 타임라인 감지 구간 병합 함수
function mergeTimelineEvents(events, mergeThreshold = 2.0) {
    if (!events || events.length === 0) return [];
    
    // 시간순 정렬
    const sortedEvents = [...events].sort((a, b) => a.start - b.start);
    
    // 병합 루프
    const mergedEvents = [];
    let currentEvent = { ...sortedEvents[0] };
    
    for (let i = 1; i < sortedEvents.length; i++) {
        const nextEvent = sortedEvents[i];
        
        // 간격이 threshold 이내면 병합
        if (nextEvent.start - currentEvent.end <= mergeThreshold) {
            // 종료 시간 연장
            currentEvent.end = Math.max(currentEvent.end, nextEvent.end);
            // 신뢰도 평균 계산 (선택적)
            if (currentEvent.similarity !== undefined && nextEvent.similarity !== undefined) {
                currentEvent.similarity = Math.max(currentEvent.similarity, nextEvent.similarity);
            }
        } else {
            // 간격이 넓으면 현재 이벤트 저장하고 교체
            mergedEvents.push(currentEvent);
            currentEvent = { ...nextEvent };
        }
    }
    
    // 마지막 이벤트 추가
    mergedEvents.push(currentEvent);
    
    return mergedEvents;
}

// 타임라인 재렌더링 함수 (병합 로직 적용)
function renderTimelineWithMerging() {
    if (!UI.video || !UI.video.duration || UI.video.duration === 0 || isNaN(UI.video.duration)) {
        return;
    }

    const timelinesContainer = document.getElementById('timelinesContainer');
    if (!timelinesContainer) {
        return;
    }

    // 인물별로 스냅샷 그룹화
    const snapshotsByPerson = {};
    state.snapshots.forEach(snapshot => {
        const personId = snapshot.personId || 'unknown';
        if (!snapshotsByPerson[personId]) {
            snapshotsByPerson[personId] = [];
        }
        snapshotsByPerson[personId].push(snapshot);
    });
    
    // 각 인물별로 타임라인 렌더링
    Object.keys(snapshotsByPerson).forEach(personId => {
        const track = timelinesContainer.querySelector(`[data-person-id="${personId}"]`);
        if (!track) return;
        
    const timelineBar = track.querySelector('.timeline-bar');
        if (!timelineBar) return;
        
        // 기존 마커 제거
        timelineBar.innerHTML = '';
        
        const personSnapshots = snapshotsByPerson[personId];
        const selectedPerson = state.selectedSuspects.find(s => s.id === personId);
        if (!selectedPerson) return;
        
        const isCriminal = selectedPerson.isThief;
    const markerColor = isCriminal
        ? 'bg-red-500 hover:bg-red-700'
        : 'bg-green-500 hover:bg-green-700';

        // 스냅샷을 감지 구간으로 변환 (각 스냅샷을 0.1초 구간으로 가정)
        const events = personSnapshots.map(snapshot => ({
            start: snapshot.videoTime,
            end: snapshot.videoTime + 0.1, // 각 감지 지점을 짧은 구간으로 처리
            similarity: snapshot.similarity,
            snapshotId: snapshot.id
        }));
        
        // 병합 로직 적용
        const mergedEvents = mergeTimelineEvents(events, 2.0);
        
        // 병합된 구간을 막대로 렌더링
        mergedEvents.forEach(event => {
            const startPercent = (event.start / UI.video.duration) * 100;
            const endPercent = (event.end / UI.video.duration) * 100;
            const widthPercent = endPercent - startPercent;
            
            if (startPercent < 0 || endPercent > 100) return;
            
    const marker = document.createElement('div');
            marker.className = `absolute h-full ${markerColor} cursor-pointer transition-all duration-200 hover:scale-y-110 hover:brightness-110 rounded-sm z-10`;
            marker.style.left = `${startPercent}%`;
            marker.style.width = `${widthPercent}%`;
            marker.dataset.snapshotId = event.snapshotId;
            marker.dataset.personId = personId;

    // 시간 포맷 헬퍼
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

            marker.title = `${selectedPerson.name} - ${formatTime(event.start)} ~ ${formatTime(event.end)}`;

            // 마커 클릭 이벤트 (구간 시작 시점으로 이동)
    marker.addEventListener('click', (e) => {
        e.stopPropagation();
        if (UI.video) {
                    UI.video.currentTime = event.start;
            UI.video.play();
                    console.log(`▶️ 비디오 이동: ${formatTime(event.start)}`);
                }
            });
            
            timelineBar.appendChild(marker);
        });
    });
    
    // 타임라인 바 클릭 이벤트 (한 번만 등록)
    const timelineBars = timelinesContainer.querySelectorAll('.timeline-bar');
    timelineBars.forEach(timelineBar => {
    if (!timelineBar.dataset.clickHandlerAdded) {
        timelineBar.addEventListener('click', (e) => {
            if (e.target === timelineBar && UI.video) {
                const rect = timelineBar.getBoundingClientRect();
                const clickX = e.clientX - rect.left;
                const percentage = clickX / rect.width;
                UI.video.currentTime = percentage * UI.video.duration;
                UI.video.play();
            }
        });
        timelineBar.dataset.clickHandlerAdded = 'true';
        }
    });
}

// 타임라인 마커 추가 (마커만 추가, 트랙은 미리 생성되어 있어야 함)
// 이제는 스냅샷을 추가한 후 재렌더링하는 방식으로 변경
function addTimelineMarkerDirect(snapshot) {
    // 스냅샷이 추가되면 타임라인을 재렌더링 (병합 로직 적용)
    // 약간의 딜레이를 두어 여러 스냅샷이 동시에 추가될 때 배치 처리
    if (state.timelineRenderTimer) {
        clearTimeout(state.timelineRenderTimer);
    }
    
    state.timelineRenderTimer = setTimeout(() => {
        renderTimelineWithMerging();
    }, 100); // 100ms 딜레이로 배치 처리
}

// 스냅샷 개수 업데이트 (직접 구현)
function updateSnapshotCountDirect() {
    console.log('🔢 updateSnapshotCountDirect 호출됨:', {
        snapshotCount: state.snapshots.length
    });

    const countEl = document.getElementById('snapshotCount');
    if (countEl) {
        countEl.textContent = state.snapshots.length;
        console.log(`✅ 스냅샷 개수 업데이트됨: ${state.snapshots.length} `);
    } else {
        console.error('❌ 스냅샷 카운트 요소(snapshotCount)를 찾을 수 없습니다!');
    }
}

// 클립 개수 업데이트
function updateClipCount() {
    const countEl = document.getElementById('clipCount');
    if (countEl) {
        countEl.textContent = state.detectionClips.length;
    }
}

// 영상 클립 다운로드 함수 (서버로 요청)
async function downloadVideoClip(clip) {
    if (!state.selectedFile) {
        console.error('❌ 비디오 파일이 없습니다.');
        alert('비디오 파일이 없습니다.');
        return;
    }

    if (!clip.endTime) {
        console.error('❌ 클립의 종료 시간이 없습니다.');
        alert('클립이 아직 완료되지 않았습니다.');
        return;
    }

    const startTime = clip.startTime;
    const endTime = clip.endTime;
    const duration = endTime - startTime;

    console.log(`🎬 클립 다운로드 시작: ${clip.personName} (${startTime.toFixed(1)} s - ${endTime.toFixed(1)}s)`);

    try {
        // FormData 생성
        const formData = new FormData();
        formData.append('video', state.selectedFile);
        formData.append('start_time', startTime.toString());
        formData.append('end_time', endTime.toString());
        formData.append('person_name', clip.personName);

        // 서버로 요청
        const response = await fetch(`${API_BASE_URL}/extract_clip`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`서버 오류: ${response.status}`);
        }

        // 응답을 Blob으로 받기
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `clip_${clip.personName}_${startTime.toFixed(1)}s-${endTime.toFixed(1)}s.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log(`✅ 클립 다운로드 완료: ${clip.personName}`);
    } catch (error) {
        console.error('❌ 클립 다운로드 실패:', error);
        alert(`클립 다운로드 실패: ${error.message}\n\n서버에서 클립 추출 기능이 구현되지 않았을 수 있습니다.`);
    }
}

// 각도 타입을 표시 텍스트로 변환
function getAngleDisplayText(angleType) {
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

// 박스를 캔버스에 그리기
function drawDetections(detections, videoWidth, videoHeight) {
    if (!state.detectionCtx || !detections || detections.length === 0) {
        // 박스가 없으면 캔버스 클리어
        if (state.detectionCtx) {
            state.detectionCtx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);
        }
        return;
    }

    const ctx = state.detectionCtx;

    // 비디오와 캔버스의 실제 표시 영역 가져오기
    const videoRect = UI.video.getBoundingClientRect();
    const containerRect = UI.video.parentElement.getBoundingClientRect();

    // 캔버스 크기를 컨테이너와 정확히 일치시키기
    state.detectionCanvas.width = containerRect.width;
    state.detectionCanvas.height = containerRect.height;

    // 캔버스 클리어
    ctx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);

    // 비디오의 실제 표시 영역 계산 (object-contain 스타일 고려)
    // 비디오 요소의 실제 렌더링 크기와 위치를 정확히 계산
    const videoAspect = videoWidth / videoHeight;
    const containerAspect = containerRect.width / containerRect.height;

    let displayWidth, displayHeight, offsetX, offsetY;

    if (videoAspect > containerAspect) {
        // 비디오가 더 넓음 - 컨테이너 높이에 맞춤
        displayHeight = containerRect.height;
        displayWidth = videoWidth * (containerRect.height / videoHeight);
        offsetX = (containerRect.width - displayWidth) / 2;
        offsetY = 0;
    } else {
        // 비디오가 더 높음 - 컨테이너 너비에 맞춤
        displayWidth = containerRect.width;
        displayHeight = videoHeight * (containerRect.width / videoWidth);
        offsetX = 0;
        offsetY = (containerRect.height - displayHeight) / 2;
    }

    // 디버깅용 (개발 중에만 사용)
    if (window.DEBUG_DETECTIONS) {
        console.log('박스 위치 계산:', {
            videoSize: `${videoWidth}x${videoHeight}`,
            containerSize: `${containerRect.width}x${containerRect.height}`,
            displaySize: `${displayWidth}x${displayHeight}`,
            offset: `(${offsetX}, ${offsetY})`,
            scale: `(${displayWidth / videoWidth}, ${displayHeight / videoHeight})`
        });
    }

    // 각 박스 그리기
    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;

        // 원본 비디오 좌표를 표시 영역 좌표로 정확히 변환
        const scaleX = displayWidth / videoWidth;
        const scaleY = displayHeight / videoHeight;

        const scaledX1 = offsetX + x1 * scaleX;
        const scaledY1 = offsetY + y1 * scaleY;
        const scaledX2 = offsetX + x2 * scaleX;
        const scaledY2 = offsetY + y2 * scaleY;

        // 색상 설정
        let color;
        switch (detection.color) {
            case 'red':
                color = '#ef4444'; // 빨간색 (범죄자)
                break;
            case 'green':
                color = '#10b981'; // 초록색 (일반인)
                break;
            case 'yellow':
            default:
                color = '#eab308'; // 노란색 (미확인)
                break;
        }

        // 박스 그리기 (더 두꺼운 선으로 강조)
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);

        // 박스 모서리 강조 (선택적)
        const cornerSize = 8;
        ctx.lineWidth = 3;
        // 왼쪽 위
        ctx.beginPath();
        ctx.moveTo(scaledX1, scaledY1 + cornerSize);
        ctx.lineTo(scaledX1, scaledY1);
        ctx.lineTo(scaledX1 + cornerSize, scaledY1);
        ctx.stroke();
        // 오른쪽 위
        ctx.beginPath();
        ctx.moveTo(scaledX2 - cornerSize, scaledY1);
        ctx.lineTo(scaledX2, scaledY1);
        ctx.lineTo(scaledX2, scaledY1 + cornerSize);
        ctx.stroke();
        // 왼쪽 아래
        ctx.beginPath();
        ctx.moveTo(scaledX1, scaledY2 - cornerSize);
        ctx.lineTo(scaledX1, scaledY2);
        ctx.lineTo(scaledX1 + cornerSize, scaledY2);
        ctx.stroke();
        // 오른쪽 아래
        ctx.beginPath();
        ctx.moveTo(scaledX2 - cornerSize, scaledY2);
        ctx.lineTo(scaledX2, scaledY2);
        ctx.lineTo(scaledX2, scaledY2 - cornerSize);
        ctx.stroke();

        // 텍스트 정보 준비
        const angleText = detection.angle_type && detection.angle_type !== 'front' && detection.angle_type !== 'unknown'
            ? ` [${getAngleDisplayText(detection.angle_type)}]`
            : '';
        const nameText = `${detection.name} (${detection.confidence}%)`;
        const fullText = nameText + angleText;

        // 범죄자인 경우 경고 텍스트 추가
        let warningText = '';
        if (detection.status === 'criminal') {
            warningText = '⚠️ WARNING';
        }

        // 텍스트 위치 계산 (박스 위쪽에 배치)
        ctx.font = 'bold 16px Arial';
        const nameMetrics = ctx.measureText(nameText);
        const fullMetrics = ctx.measureText(fullText);
        const warningMetrics = warningText ? ctx.measureText(warningText) : { width: 0 };

        const textPadding = 6;
        const lineHeight = 22;
        const maxTextWidth = Math.max(fullMetrics.width, warningMetrics.width);
        const textBoxWidth = maxTextWidth + (textPadding * 2);
        const textBoxHeight = warningText ? lineHeight * 2 + textPadding : lineHeight + textPadding;

        // 텍스트가 화면 밖으로 나가지 않도록 조정
        let textX = scaledX1;
        if (textX + textBoxWidth > state.detectionCanvas.width) {
            textX = state.detectionCanvas.width - textBoxWidth;
        }
        if (textX < 0) {
            textX = 0;
        }

        let textY = scaledY1 - textBoxHeight - 4;
        // 텍스트가 화면 위로 나가면 박스 아래에 배치
        if (textY < 0) {
            textY = scaledY2 + 4;
        }

        // 텍스트 배경 그리기 (반투명 배경)
        ctx.fillStyle = color + 'CC'; // 80% 투명도
        ctx.fillRect(textX, textY, textBoxWidth, textBoxHeight);

        // 텍스트 그리기
        ctx.fillStyle = '#ffffff';
        let currentY = textY + lineHeight;

        // 경고 텍스트 먼저 (있는 경우)
        if (warningText) {
            ctx.font = 'bold 18px Arial';
            ctx.fillStyle = '#ffffff';
            ctx.fillText(warningText, textX + textPadding, currentY - 4);
            currentY += lineHeight;
        }

        // 이름과 신뢰도
        ctx.font = 'bold 16px Arial';
        ctx.fillStyle = '#ffffff';
        ctx.fillText(nameText, textX + textPadding, currentY);

        // 각도 정보 (있는 경우, 같은 줄에)
        if (angleText) {
            ctx.font = '14px Arial';
            ctx.fillStyle = '#f0f0f0';
            const angleX = textX + textPadding + nameMetrics.width + 4;
            ctx.fillText(angleText, angleX, currentY);
        }
    });
}

// 현재 비디오 프레임 캡처 (Base64)
function captureVideoFrame() {
    if (!UI.video) {
        console.warn("⚠️ 비디오 요소를 찾을 수 없습니다");
        return null;
    }

    // 비디오가 종료된 경우는 processRealtimeDetection에서 처리하므로 여기서는 조용히 반환
    if (UI.video.ended) {
        return null;
    }

    // 일시정지된 경우도 조용히 반환 (메시지는 processRealtimeDetection에서 처리)
    if (UI.video.paused) {
        return null;
    }

    if (UI.video.videoWidth === 0 || UI.video.videoHeight === 0) {
        console.warn("⚠️ 비디오 크기가 0입니다. 비디오가 로드되지 않았을 수 있습니다");
        return null;
    }

    const ctx = state.videoCanvas.getContext('2d');
    state.videoCanvas.width = UI.video.videoWidth;
    state.videoCanvas.height = UI.video.videoHeight;

    ctx.drawImage(UI.video, 0, 0);
    return state.videoCanvas.toDataURL('image/jpeg', 0.7);
}

// ==========================================
// WebSocket 연결 관리
// ==========================================

async function testWebSocketConnection() {
    /** WebSocket 연결 테스트 함수 */
    return new Promise((resolve) => {
        console.log(`🧪 WebSocket 연결 테스트: ${WS_TEST_URL}`);
        const testWs = new WebSocket(WS_TEST_URL);

        const timeout = setTimeout(() => {
            testWs.close();
            console.log("❌ WebSocket 테스트 타임아웃");
            resolve(false);
        }, 3000);

        testWs.onopen = () => {
            clearTimeout(timeout);
            console.log("✅ WebSocket 테스트 연결 성공!");
            testWs.close();
            resolve(true);
        };

        testWs.onerror = (error) => {
            clearTimeout(timeout);
            console.log("❌ WebSocket 테스트 연결 실패");
            resolve(false);
        };

        testWs.onclose = () => {
            clearTimeout(timeout);
        };
    });
}

async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log(`✅ 서버 상태 확인: ${data.status}, 활성 연결: ${data.active_connections}`);
            return true;
        }
        return false;
    } catch (error) {
        console.error("❌ 서버 상태 확인 실패:", error);
        return false;
    }
}

function connectWebSocket() {
    if (state.ws) {
        if (state.ws.readyState === WebSocket.OPEN) {
            return; // 이미 연결됨
        }
        if (state.ws.readyState === WebSocket.CONNECTING) {
            return; // 연결 중임
        }
    }

    // 서버 상태 확인 후 연결 시도
    checkServerHealth().then(async (isHealthy) => {
        if (!isHealthy) {
            console.warn("⚠️ 서버가 응답하지 않습니다. 잠시 후 재시도합니다.");
            setTimeout(() => {
                if (state.useWebSocket && !state.isWsConnected) {
                    connectWebSocket();
                }
            }, 2000);
            return;
        }

        // WebSocket 연결 테스트 (선택적)
        const wsTestResult = await testWebSocketConnection();
        if (!wsTestResult) {
            console.warn("⚠️ WebSocket 테스트 실패, HTTP 모드로 전환합니다.");
            state.useWebSocket = false;
            return;
        }
    });

    try {
        console.log(`🔌 WebSocket 연결 시도: ${WS_URL}`);
        // WebSocket 연결 생성 (프로토콜 없이)
        const ws = new WebSocket(WS_URL);

        // 연결 타임아웃 설정 (5초)
        const connectionTimeout = setTimeout(() => {
            if (ws.readyState === WebSocket.CONNECTING) {
                console.warn("⚠️ WebSocket 연결 타임아웃 (5초)");
                ws.close();
                state.useWebSocket = false;
            }
        }, 5000);

        ws.onopen = () => {
            clearTimeout(connectionTimeout); // 타임아웃 클리어
            console.log("✅ WebSocket 연결됨");
            state.isWsConnected = true;
            state.wsReconnectAttempts = 0;
            state.useWebSocket = true;
            state.wsConfigReady = false; // 설정 완료 플래그 초기화

            // 하트비트 시작
            startHeartbeat();

            // 연결 시 선택된 모든 suspect_ids 전송 (설정 완료 후 프레임 전송)
            if (state.selectedSuspects.length > 0) {
                const suspectIds = state.selectedSuspects.map(s => s.id);
                sendWebSocketConfig(suspectIds);
                // config_updated 메시지를 받은 후 wsConfigReady가 true가 되면 프레임 전송 시작
            } else {
                // 인물이 선택되지 않았으므로 감지를 시작하지 않음
                state.wsConfigReady = false;
                console.warn("⚠️ 인물이 선택되지 않았습니다. 인물을 선택한 후 감지를 시작하세요.");

                // 설정 완료 후 첫 프레임 전송 (감지 활성화 상태일 때만)
                if (state.isDetectionActive) {
                    setTimeout(() => {
                        processRealtimeDetection();
                    }, 50); // 연결 안정화를 위한 짧은 대기
                }
            }
        };

        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            } catch (error) {
                console.error("❌ WebSocket 메시지 파싱 오류:", error);
            }
        };

        ws.onerror = (error) => {
            clearTimeout(connectionTimeout); // 타임아웃 클리어
            // 비디오가 종료된 경우 오류 메시지 출력하지 않음
            if (UI.video && UI.video.ended) {
                return;
            }
            // 첫 번째 실패 시에만 상세 오류 메시지 출력
            if (state.wsReconnectAttempts === 0) {
                console.warn("⚠️ WebSocket 연결 실패");
                console.warn("   연결 URL:", WS_URL);
                console.warn("   백엔드 서버가 실행 중인지 확인하세요");
                console.warn("   명령어: uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000");
            }
            state.isWsConnected = false;
            state.wsConfigReady = false;

            // WebSocket 연결 실패 시 즉시 HTTP 모드로 전환 (재연결 시도 최소화)
            if (state.wsReconnectAttempts < 2) {
                // 처음 2번만 재시도, 그 이후로는 HTTP 모드로 전환
            } else {
                // 2번 실패 후 HTTP 모드로 전환
                console.log("✅ HTTP 모드로 전환 (WebSocket 사용 안 함)");
                state.useWebSocket = false;
            }
        };

        ws.onclose = (event) => {
            clearTimeout(connectionTimeout); // 타임아웃 클리어
            // 종료 코드 1006은 비정상 종료 (연결 실패)
            if (event.code === 1006) {
                // 첫 번째 실패 시에만 로그 출력
                if (state.wsReconnectAttempts === 0) {
                    console.warn("⚠️ WebSocket 연결 실패 (코드: 1006)");
                    console.warn("   원인: 서버에 연결할 수 없음");
                    console.warn("   해결: 백엔드 서버가 실행 중인지 확인하세요");
                    console.warn("   명령어: uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000");
                }
            } else {
                console.log("⚠️ WebSocket 연결 종료됨");
                console.log("   종료 코드:", event.code);
                console.log("   종료 사유:", event.reason || "없음");
            }

            state.isWsConnected = false;
            state.wsConfigReady = false; // 설정 플래그도 초기화
            state.ws = null;

            // 하트비트 중지
            stopHeartbeat();

            // 비디오가 종료되지 않았고 감지가 활성화된 경우에만 재연결 시도
            // 정상 종료(1000)가 아닌 경우에만 재연결 시도
            // 하지만 재연결 시도 횟수가 2회 이하일 때만 재연결 시도
            if (event.code !== 1000 && state.useWebSocket && !(UI.video && UI.video.ended) && state.isDetectionActive && state.wsReconnectAttempts < 2) {
                scheduleReconnect();
            } else if (state.wsReconnectAttempts >= 2) {
                // 2번 실패 후 HTTP 모드로 전환
                console.log("✅ HTTP 모드로 전환 (WebSocket 재연결 중단)");
                state.useWebSocket = false;
            } else if (UI.video && UI.video.ended) {
                console.log("⏹️ 비디오 종료됨, WebSocket 재연결하지 않음");
            } else if (event.code === 1000) {
                console.log("✅ WebSocket 정상 종료");
            }
        };

        state.ws = ws;
    } catch (error) {
        console.error("❌ WebSocket 연결 실패:", error);
        state.useWebSocket = false;
        scheduleReconnect();
    }
}

function disconnectWebSocket() {
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }
    state.isWsConnected = false;
    state.wsConfigReady = false; // 설정 플래그도 초기화
    if (state.wsReconnectTimer) {
        clearTimeout(state.wsReconnectTimer);
        state.wsReconnectTimer = null;
    }
}

function scheduleReconnect() {
    if (state.wsReconnectTimer) {
        return; // 이미 재연결 예약됨
    }

    // 비디오가 종료되었거나 감지가 비활성화된 경우 재연결하지 않음
    if (UI.video && UI.video.ended) {
        console.log("⏹️ 비디오 종료됨, WebSocket 재연결 취소");
        return;
    }

    // 감지가 비활성화된 경우 재연결하지 않음
    if (!state.isDetectionActive) {
        return;
    }

    // 재연결 시도 횟수 제한 (2회로 줄임 - 빠른 HTTP 전환)
    if (state.wsReconnectAttempts >= 2) {
        console.log("✅ WebSocket 재연결 중단, HTTP 모드로 전환합니다.");
        state.useWebSocket = false;
        // HTTP 모드로 전환하여 감지 계속 진행
        if (state.isDetectionActive && !state.detectionInterval) {
            processRealtimeDetection();
            state.detectionInterval = setInterval(processRealtimeDetection, 100);
        }
        return;
    }

    const delay = Math.min(1000 * Math.pow(2, state.wsReconnectAttempts), 30000); // 최대 30초
    state.wsReconnectAttempts++;

    console.log(`🔄 ${delay / 1000}초 후 WebSocket 재연결 시도 (${state.wsReconnectAttempts}/10회)`);
    console.log(`   백엔드 서버 확인: ${API_BASE_URL.replace('/api', '')}`);

    state.wsReconnectTimer = setTimeout(() => {
        state.wsReconnectTimer = null;
        // 비디오가 종료되었거나 감지가 비활성화된 경우 재연결하지 않음
        if (UI.video && UI.video.ended) {
            console.log("⏹️ 비디오 종료됨, WebSocket 재연결 취소");
            return;
        }
        if (!state.isDetectionActive) {
            return;
        }
        // 감지 활성화 여부와 상관없이 WebSocket 사용 모드면 재연결 시도
        if (state.useWebSocket && !state.isWsConnected) {
            connectWebSocket();
        }
    }, delay);
}

function startHeartbeat() {
    stopHeartbeat();
    state.heartbeatInterval = setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ type: "ping" }));
        }
    }, 30000); // 30초마다 핑
}

function stopHeartbeat() {
    if (state.heartbeatInterval) {
        clearInterval(state.heartbeatInterval);
        state.heartbeatInterval = null;
    }
}

function sendWebSocketConfig(suspectIds) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({
            type: "config",
            suspect_ids: suspectIds // 배열로 전송
        }));
    }
}

function sendWebSocketFrame(frameData, suspectIds) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        // suspectIds가 null이어도 빈 배열로 전송 (항상 포함)
        const ids = suspectIds || [];

        state.frameId++;
        state.ws.send(JSON.stringify({
            type: "frame",
            data: {
                image: frameData,
                suspect_ids: ids, // 항상 포함 (빈 배열이어도)
                frame_id: state.frameId
            }
        }));
        return true;
    }
    return false;
}

function handleWebSocketMessage(message) {
    const msgType = message.type;

    console.log('📨 WebSocket 메시지 수신:', msgType);

    if (msgType === "detection") {
        const data = message.data;
        state.lastDetections = data.detections;
        state.lastDetectionTime = Date.now();

        // 정확한 비디오 타임스탬프 사용
        const videoTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : (data.video_timestamp || 0);

        // 1. 선택된 인물 필터링
        let detectedSelectedPersons = [];
        if (data.detections && data.detections.length > 0) {
            const selectedPersonIds = state.selectedSuspects.map(s => s.id);
            detectedSelectedPersons = data.detections.filter(d =>
                selectedPersonIds.includes(d.metadata?.person_id || d.name)
            );
        }

        // 디버깅 로그
        console.log('🔍 감지 결과 확인:', {
            alert: data.alert,
            detectionsCount: data.detections ? data.detections.length : 0,
            selectedPersonsCount: detectedSelectedPersons.length,
            names: detectedSelectedPersons.map(d => d.metadata?.name || d.name)
        });

        // 2. 선택된 인물들에 대해 처리 (타임라인 마커, 스냅샷, 클립)
        if (detectedSelectedPersons.length > 0) {
            // 스냅샷 이미지는 공유 (없으면 캡처)
            let snapshotImage = data.snapshot_base64;
            if (!snapshotImage) {
                snapshotImage = captureVideoFrame();
            }

            detectedSelectedPersons.forEach(personData => {
                const personId = personData.metadata?.person_id || personData.name || 'Unknown';
                const personName = personData.metadata?.name || personData.name || 'Unknown';
                // isThief 정보는 state.selectedSuspects에서 가져오는 것이 가장 정확함
                const selectedSuspect = state.selectedSuspects.find(s => s.id === personId);
                const isCriminal = selectedSuspect ? selectedSuspect.isThief : (personData.status === 'criminal');

                // A. 스냅샷 생성 및 저장
                if (snapshotImage) {
                    const snapshot = {
                        id: state.nextSnapshotId++,
                        timestamp: new Date().toISOString(),
                        videoTime: videoTime,
                        personId: personId,
                        personName: personName,
                        isCriminal: isCriminal,
                        similarity: personData.metadata?.confidence || 0,
                        base64Image: snapshotImage,
                        status: isCriminal ? 'criminal' : 'missing'
                    };
                    state.snapshots.push(snapshot);

                    // B. 타임라인 마커 추가
                    addTimelineMarkerDirect(snapshot);

                    // C. 스냅샷 카운트 업데이트
                    updateSnapshotCountDirect();
                }

                // D. 클립 추적 (인물별 클립 관리)
                if (!state.activeClips[personId]) {
                    // 새로운 클립 시작
                    state.activeClips[personId] = {
                        id: state.nextClipId++,
                        startTime: videoTime,
                        personId: personId,
                        personName: personName,
                        similarity: personData.metadata?.confidence || 0,
                        isCriminal: isCriminal
                    };
                    console.log(`🎬 클립 시작: ${personName} (${videoTime.toFixed(1)}s)`);
                } else {
                    // 기존 클립 업데이트 (유사도 업데이트)
                    state.activeClips[personId].similarity = Math.max(
                        state.activeClips[personId].similarity,
                        personData.metadata?.confidence || 0
                    );
                }
            });

            // 알림 효과 (범죄자가 한 명이라도 있으면 빨간 테두리)
            const hasCriminal = detectedSelectedPersons.some(p => {
                const pid = p.metadata?.person_id || p.name;
                const suspect = state.selectedSuspects.find(s => s.id === pid);
                return suspect && suspect.isThief;
            });

            if (hasCriminal) {
                UI.video.parentElement.classList.add('alert-border');
            } else {
                UI.video.parentElement.classList.remove('alert-border');
            }

            // 패널 업데이트 (첫 번째 감지된 인물 기준)
            if (detectedSelectedPersons.length > 0) {
                const firstPerson = detectedSelectedPersons[0];
                const personId = firstPerson.metadata?.person_id || firstPerson.name || 'Unknown';
                const selectedSuspect = state.selectedSuspects.find(s => s.id === personId);
                const isCriminal = selectedSuspect ? selectedSuspect.isThief : (firstPerson.status === 'criminal');
                
                // metadata에 status 추가
                const metadata = {
                    ...firstPerson.metadata,
                    name: firstPerson.metadata?.name || firstPerson.name || 'Unknown',
                    confidence: firstPerson.metadata?.confidence || firstPerson.confidence || 0,
                    status: isCriminal ? 'criminal' : 'missing',
                    person_id: personId
                };
                
                // 스냅샷 이미지 가져오기
                let snapshotImage = data.snapshot_base64;
                if (!snapshotImage && detectedSelectedPersons.length > 0) {
                    snapshotImage = captureVideoFrame();
                }
                
                updateDetectionPanel(metadata, hasCriminal, videoTime, snapshotImage);
            }

        } else {
            // 감지된 선택 인물이 없음
            UI.video.parentElement.classList.remove('alert-border');
            
            // 활성 클립 종료 (선택된 인물이 감지되지 않으면 모든 클립 종료)
            const selectedPersonIds = state.selectedSuspects.map(s => s.id);
            Object.keys(state.activeClips).forEach(personId => {
                if (selectedPersonIds.includes(personId)) {
                    const clip = state.activeClips[personId];
                    clip.endTime = videoTime;
                    state.detectionClips.push(clip);
                    console.log(`✅ 클립 종료: ${clip.personName} (${clip.startTime.toFixed(1)}s - ${videoTime.toFixed(1)}s)`);
                    delete state.activeClips[personId];
                    updateClipCount();
                }
            });
            
            // 모든 감지 결과를 로그에 표시 (Unknown 포함)
            if (data.detections && data.detections.length > 0) {
                // 첫 번째 감지 결과를 로그에 표시
                const firstDetection = data.detections[0];
                const metadata = firstDetection.metadata || {
                    name: firstDetection.name || 'Unknown',
                    confidence: firstDetection.confidence || 0,
                    status: firstDetection.status || 'unknown'
                };
                updateDetectionPanel(metadata, false);
            } else {
            updateDetectionPanel(null, false);
            }
        }

        // 3. 박스 렌더링 (모든 감지된 인물 표시)
        if (data.detections && data.detections.length > 0 && UI.video.videoWidth > 0) {
            drawDetections(data.detections, UI.video.videoWidth, UI.video.videoHeight);
        } else {
            if (state.detectionCtx) {
                state.detectionCtx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);
            }
        }

        state.isProcessing = false;

    } else if (msgType === "error") {
        console.error("❌ 서버 오류:", message.message);
        state.isProcessing = false;

    } else if (msgType === "pong") {
        // 핑 응답
    } else if (msgType === "config_updated") {
        console.log("✅ 설정 업데이트됨:", message.suspect_ids);
        state.wsConfigReady = true;
        if (state.isDetectionActive && !state.isProcessing) {
            setTimeout(() => {
                processRealtimeDetection();
            }, 50);
        }
    }
}

// HTTP API 폴백 함수
async function detectFrameToServerHTTP(frameData) {
    try {
        const suspectIds = state.selectedSuspects.length > 0
            ? state.selectedSuspects.map(s => s.id)
            : null;

        const requestBody = {
            image: frameData,
            suspect_ids: suspectIds // 배열로 전송
        };

        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
        }

        const result = await response.json();

        if (result && result.success) {
            state.lastDetections = result.detections;
            state.lastDetectionTime = Date.now();

            // 정확한 비디오 타임스탬프 사용
            const videoTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : (result.video_timestamp || 0);

            // 1. 선택된 인물 필터링
            let detectedSelectedPersons = [];
            if (result.detections && result.detections.length > 0) {
                const selectedPersonIds = state.selectedSuspects.map(s => s.id);
                detectedSelectedPersons = result.detections.filter(d =>
                    selectedPersonIds.includes(d.metadata?.person_id || d.name)
                );
            }

            // 디버깅 로그
            console.log('🔍 HTTP 감지 결과 확인:', {
                alert: result.alert,
                detectionsCount: result.detections ? result.detections.length : 0,
                selectedPersonsCount: detectedSelectedPersons.length,
                names: detectedSelectedPersons.map(d => d.metadata?.name || d.name)
            });

            // 2. 선택된 인물들에 대해 처리 (타임라인 마커, 스냅샷, 클립)
            if (detectedSelectedPersons.length > 0) {
                // 스냅샷 이미지는 공유 (없으면 캡처)
                let snapshotImage = result.snapshot_base64;
                if (!snapshotImage) {
                    snapshotImage = captureVideoFrame();
                }

                detectedSelectedPersons.forEach(personData => {
                    const personId = personData.metadata?.person_id || personData.name || 'Unknown';
                    const personName = personData.metadata?.name || personData.name || 'Unknown';
                    // isThief 정보는 state.selectedSuspects에서 가져오는 것이 가장 정확함
                    const selectedSuspect = state.selectedSuspects.find(s => s.id === personId);
                    const isCriminal = selectedSuspect ? selectedSuspect.isThief : (personData.status === 'criminal');

                    // A. 스냅샷 생성 및 저장
                    if (snapshotImage) {
                        const snapshot = {
                            id: state.nextSnapshotId++,
                            timestamp: new Date().toISOString(),
                            videoTime: videoTime,
                            personId: personId,
                            personName: personName,
                            isCriminal: isCriminal,
                            similarity: personData.metadata?.confidence || 0,
                            base64Image: snapshotImage,
                            status: isCriminal ? 'criminal' : 'missing'
                        };
                        state.snapshots.push(snapshot);

                        // B. 타임라인 마커 추가
                        addTimelineMarkerDirect(snapshot);

                        // C. 스냅샷 카운트 업데이트
                        updateSnapshotCountDirect();
                    }

                    // D. 클립 추적 (인물별 클립 관리)
                    if (!state.activeClips[personId]) {
                        // 새로운 클립 시작
                        state.activeClips[personId] = {
                            id: state.nextClipId++,
                            startTime: videoTime,
                            personId: personId,
                            personName: personName,
                            similarity: personData.metadata?.confidence || 0,
                            isCriminal: isCriminal
                        };
                        console.log(`🎬 클립 시작: ${personName} (${videoTime.toFixed(1)}s)`);
                    } else {
                        // 기존 클립 업데이트 (유사도 업데이트)
                        state.activeClips[personId].similarity = Math.max(
                            state.activeClips[personId].similarity,
                            personData.metadata?.confidence || 0
                        );
                    }
                });

                // 알림 효과 (범죄자가 한 명이라도 있으면 빨간 테두리)
                const hasCriminal = detectedSelectedPersons.some(p => {
                    const pid = p.metadata?.person_id || p.name;
                    const suspect = state.selectedSuspects.find(s => s.id === pid);
                    return suspect && suspect.isThief;
                });

                if (hasCriminal) {
                    UI.video.parentElement.classList.add('alert-border');
                } else {
                    UI.video.parentElement.classList.remove('alert-border');
                }

                // 패널 업데이트 (첫 번째 감지된 인물 기준)
                if (detectedSelectedPersons.length > 0) {
                    const firstPerson = detectedSelectedPersons[0];
                    const personId = firstPerson.metadata?.person_id || firstPerson.name || 'Unknown';
                    const selectedSuspect = state.selectedSuspects.find(s => s.id === personId);
                    const isCriminal = selectedSuspect ? selectedSuspect.isThief : (firstPerson.status === 'criminal');
                    
                    // metadata에 status 추가
                    const metadata = {
                        ...firstPerson.metadata,
                        name: firstPerson.metadata?.name || firstPerson.name || 'Unknown',
                        confidence: firstPerson.metadata?.confidence || firstPerson.confidence || 0,
                        status: isCriminal ? 'criminal' : 'missing',
                        person_id: personId
                    };
                    
                    // 스냅샷 이미지 가져오기
                    let snapshotImage = result.snapshot_base64;
                    if (!snapshotImage && detectedSelectedPersons.length > 0) {
                        snapshotImage = captureVideoFrame();
                    }
                    
                    const videoTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : 0;
                    updateDetectionPanel(metadata, hasCriminal, videoTime, snapshotImage);
                }

            } else {
                // 감지된 선택 인물이 없음
                UI.video.parentElement.classList.remove('alert-border');
                
                // 활성 클립 종료 (선택된 인물이 감지되지 않으면 모든 클립 종료)
                const selectedPersonIds = state.selectedSuspects.map(s => s.id);
                Object.keys(state.activeClips).forEach(personId => {
                    if (selectedPersonIds.includes(personId)) {
                        const clip = state.activeClips[personId];
                        clip.endTime = videoTime;
                        state.detectionClips.push(clip);
                        console.log(`✅ 클립 종료: ${clip.personName} (${clip.startTime.toFixed(1)}s - ${videoTime.toFixed(1)}s)`);
                        delete state.activeClips[personId];
                        updateClipCount();
                    }
                });
                
                // 모든 감지 결과를 로그에 표시 (Unknown 포함)
                if (result.detections && result.detections.length > 0) {
                    // 첫 번째 감지 결과를 로그에 표시
                    const firstDetection = result.detections[0];
                    const metadata = firstDetection.metadata || {
                        name: firstDetection.name || 'Unknown',
                        confidence: firstDetection.confidence || 0,
                        status: firstDetection.status || 'unknown'
                    };
                    updateDetectionPanel(metadata, false);
                } else {
                updateDetectionPanel(null, false);
                }
            }
            // 박스 렌더링
            if (result.detections && result.detections.length > 0 && UI.video.videoWidth > 0) {
                const videoWidth = UI.video.videoWidth;
                const videoHeight = UI.video.videoHeight;
                drawDetections(result.detections, videoWidth, videoHeight);
            } else {
                // 박스가 없으면 캔버스 클리어
                if (state.detectionCtx) {
                    state.detectionCtx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);
                }
            }

            // 알림 및 로그 업데이트
            if (result.alert) {
                UI.video.parentElement.classList.add('alert-border');
                const snapshotImage = result.snapshot_base64 || null;
                const videoTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : 0;
                updateDetectionPanel(result.metadata, true, videoTime, snapshotImage);
            } else {
                UI.video.parentElement.classList.remove('alert-border');
                const snapshotImage = result.snapshot_base64 || null;
                const videoTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : 0;
                updateDetectionPanel(result.metadata, false, videoTime, snapshotImage);
            }
        }

        return result;

    } catch (error) {
        console.error("❌ HTTP 통신 오류:", error);
        return null;
    }
}

// 실시간 감지 루프 (WebSocket 우선, HTTP 폴백)
async function processRealtimeDetection() {
    // 1. 감지 꺼짐 OR 이미 처리 중이면 스킵 (중복 요청 방지)
    if (!state.isDetectionActive || state.isProcessing) return;

    // 비디오가 종료된 경우 감지 루프 중지
    if (UI.video && UI.video.ended) {
        if (state.isDetectionActive) {
            console.log("⏹️ 비디오 종료됨, 감지 루프 자동 중지");
            state.isDetectionActive = false;
            UI.detectionFilter.checked = false;
            clearInterval(state.detectionInterval);

            // 모든 활성 클립 종료
                const endTime = UI.video.currentTime;
            Object.keys(state.activeClips).forEach(personId => {
                const clip = state.activeClips[personId];
                clip.endTime = endTime;
                state.detectionClips.push(clip);
                console.log(`✅ 클립 종료: ${clip.personName} (${clip.startTime.toFixed(1)}s - ${endTime.toFixed(1)}s)`);
                delete state.activeClips[personId];
            });
                updateClipCount();

            updateDetectionPanel({ message: "비디오 종료됨" });
        }
        return;
    }

    const frameData = captureVideoFrame();
    if (!frameData) {
        // 비디오가 일시정지된 경우에만 메시지 출력 (종료된 경우는 위에서 처리)
        if (UI.video && !UI.video.ended && UI.video.paused) {
            // 일시정지 메시지는 한 번만 출력하도록 (너무 많이 출력되지 않도록)
            return;
        }
        return;
    }

    // 2. 처리 시작 (문 잠금)
    state.isProcessing = true;

    const suspectIds = state.selectedSuspects.length > 0
        ? state.selectedSuspects.map(s => s.id)
        : null;

    // WebSocket 사용 시도 (연결되어 있고 설정도 완료된 경우만)
    if (state.useWebSocket && state.isWsConnected && state.wsConfigReady) {
        const sent = sendWebSocketFrame(frameData, suspectIds);
        if (sent) {
            // WebSocket으로 전송 성공, 응답은 handleWebSocketMessage에서 처리
            return;
        } else {
            // WebSocket 전송 실패, HTTP로 폴백
            console.warn("⚠️ WebSocket 전송 실패, HTTP로 폴백");
            state.useWebSocket = false;
        }
    } else if (state.useWebSocket && state.isWsConnected && !state.wsConfigReady) {
        // WebSocket은 연결되었지만 설정이 완료되지 않았으면 HTTP로 폴백 (대기하지 않음)
        console.log("⏳ WebSocket 설정 대기 중... HTTP로 폴백하여 통신 시작");
        // HTTP로 즉시 폴백 (설정 완료되면 자동으로 WebSocket으로 전환됨)
    }

    // HTTP 폴백 또는 WebSocket 비활성화 시
    try {
        const result = await detectFrameToServerHTTP(frameData);

        if (!result || !result.success) {
            // 오류 시 이전 결과 유지 (500ms 이내면)
            if (state.lastDetections && (Date.now() - state.lastDetectionTime < 500)) {
                const videoWidth = UI.video.videoWidth;
                const videoHeight = UI.video.videoHeight;
                drawDetections(state.lastDetections, videoWidth, videoHeight);
            } else {
                // 캔버스 클리어
                if (state.detectionCtx) {
                    state.detectionCtx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);
                }
            }
        }

        // 처리 완료
        state.isProcessing = false;
    } catch (err) {
        console.error("❌ 처리 중 에러:", err);
        state.isProcessing = false;
    }
}

// ==========================================
// UI 이벤트 핸들러
// ==========================================

UI.videoFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        state.selectedFile = file;
        UI.fileName.textContent = file.name;
        UI.fileInfo.classList.remove('hidden');
        UI.analyzeBtn.disabled = false;
    }
});

UI.analyzeBtn.addEventListener('click', async () => {
    if (state.selectedFile) {
        // 모달 닫기
        UI.uploadModal.classList.add('hidden');

        // 비디오 로드
        const videoURL = URL.createObjectURL(state.selectedFile);
        UI.video.src = videoURL;

        // 영상 업로드 상태 업데이트
        state.videoUploaded = true;
        updateDashboardView();

        console.log("✅ 영상 파일 로드 완료:", state.selectedFile.name);
    }
});

// 인물 카드 클릭 이벤트는 createSuspectCard 함수 내에서 처리됨

// ==========================================
// 용의자 추가 기능
// ==========================================

// 폼 유효성 검사 함수
function checkFormValidity() {
    const name = UI.enrollName.value.trim();
    const imageFile = UI.enrollImage.files[0];
    const personType = document.getElementById('personTypeInput')?.value;
    
    const isValid = name && imageFile && personType;
    
    // 등록 버튼 상태 업데이트
    if (isValid) {
        UI.submitEnrollBtn.disabled = false;
        UI.submitEnrollBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        UI.submitEnrollBtn.classList.add('opacity-100', 'cursor-pointer');
    } else {
        UI.submitEnrollBtn.disabled = true;
        UI.submitEnrollBtn.classList.add('opacity-50', 'cursor-not-allowed');
        UI.submitEnrollBtn.classList.remove('opacity-100', 'cursor-pointer');
    }
    
    return isValid;
}

// 모달 열기
UI.addSuspectBtn?.addEventListener('click', () => {
    // 폼 완전 초기화
    UI.addSuspectForm.reset();
    UI.imagePreview.classList.add('hidden');
    UI.imagePlaceholder.classList.remove('hidden');
    UI.enrollError.classList.add('hidden');
    UI.enrollSuccess.classList.add('hidden');
    
    // 구분 선택 초기화 (범죄자로 설정)
    const typeCriminal = document.getElementById('typeCriminal');
    const typeMissing = document.getElementById('typeMissing');
    if (typeCriminal) typeCriminal.checked = true;
    if (typeMissing) typeMissing.checked = false;
    document.getElementById('personTypeInput').value = 'criminal';
    updatePersonTypeButtons();
    
    // 버튼 상태 초기화 (비활성화)
    UI.submitEnrollBtn.disabled = true;
    UI.submitEnrollBtn.textContent = '등록';
    UI.submitEnrollBtn.classList.add('opacity-50', 'cursor-not-allowed');
    UI.submitEnrollBtn.classList.remove('opacity-100', 'cursor-pointer');
    
    // 모달 표시
    UI.addSuspectModal.classList.remove('hidden');
    
    // 초기 유효성 검사
    checkFormValidity();
});

// 모달 외부 클릭 시 닫기
UI.addSuspectModal?.addEventListener('click', (e) => {
    if (e.target === UI.addSuspectModal) {
        closeEnrollModal();
    }
});

// ESC 키로 모달 닫기
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !UI.addSuspectModal.classList.contains('hidden')) {
        closeEnrollModal();
    }
});

// 모달 닫기 함수 (공통)
function closeEnrollModal() {
    UI.addSuspectModal.classList.add('hidden');
    // 폼 완전 초기화
    UI.addSuspectForm.reset();
    UI.imagePreview.classList.add('hidden');
    UI.imagePlaceholder.classList.remove('hidden');
    UI.enrollError.classList.add('hidden');
    UI.enrollSuccess.classList.add('hidden');
    
    // 구분 선택 초기화
    const typeCriminal = document.getElementById('typeCriminal');
    const typeMissing = document.getElementById('typeMissing');
    if (typeCriminal) typeCriminal.checked = true;
    if (typeMissing) typeMissing.checked = false;
    document.getElementById('personTypeInput').value = 'criminal';
    updatePersonTypeButtons();
    
    // 버튼 상태 초기화 (비활성화)
    UI.submitEnrollBtn.disabled = true;
    UI.submitEnrollBtn.classList.add('opacity-50', 'cursor-not-allowed');
    UI.submitEnrollBtn.classList.remove('opacity-100', 'cursor-pointer');
}

// 모달 닫기
UI.closeAddSuspectModal?.addEventListener('click', () => {
    closeEnrollModal();
});

UI.cancelEnrollBtn?.addEventListener('click', () => {
    closeEnrollModal();
});

// 이미지 미리보기
UI.enrollImage?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            UI.previewImg.src = event.target.result;
            UI.imagePreview.classList.remove('hidden');
            UI.imagePlaceholder.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
    // 폼 유효성 검사
    checkFormValidity();
});

// 이름 입력 필드 이벤트 리스너
UI.enrollName?.addEventListener('input', () => {
    checkFormValidity();
});

// 폼 제출
UI.addSuspectForm?.addEventListener('submit', async (e) => {
    e.preventDefault();

    // 입력값 가져오기
    const name = UI.enrollName.value.trim();
    const imageFile = UI.enrollImage.files[0];

    // 인물 타입 가져오기 (세그먼트 컨트롤에서 선택된 값)
    const personType = document.getElementById('personTypeInput')?.value || 'criminal';

    // 인물 ID 자동 생성 (타임스탬프 기반)
    const personId = `person_${Date.now()}`;

    // hidden input에 자동 생성된 ID 설정
    UI.enrollPersonId.value = personId;

    // 유효성 검사
    if (!name) {
        UI.enrollError.textContent = '이름을 입력해주세요.';
        UI.enrollError.classList.remove('hidden');
        return;
    }

    if (!imageFile) {
        UI.enrollError.textContent = '정면 사진을 선택해주세요.';
        UI.enrollError.classList.remove('hidden');
        return;
    }

    // FormData 생성
    const formData = new FormData();
    formData.append('person_id', personId);
    formData.append('name', name);
    formData.append('person_type', personType);  // criminal 또는 missing
    formData.append('image', imageFile);

    // 버튼 비활성화
    UI.submitEnrollBtn.disabled = true;
    UI.submitEnrollBtn.textContent = '등록 중...';
    UI.enrollError.classList.add('hidden');
    UI.enrollSuccess.classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE_URL}/enroll`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            // 성공 메시지 표시
            UI.enrollSuccess.textContent = data.message || `등록 완료: ${name} (${personId})`;
            UI.enrollSuccess.classList.remove('hidden');
            UI.enrollError.classList.add('hidden');

            // 폼 리셋
            UI.addSuspectForm.reset();
            UI.imagePreview.classList.add('hidden');
            UI.imagePlaceholder.classList.remove('hidden');

            // 인물 목록 즉시 새로고침
            await renderSuspectCards();

            // 2초 후 모달 닫기
            setTimeout(() => {
                UI.addSuspectModal.classList.add('hidden');
                // 모달 닫을 때 폼 완전 초기화
                UI.addSuspectForm.reset();
                UI.imagePreview.classList.add('hidden');
                UI.imagePlaceholder.classList.remove('hidden');
                UI.enrollError.classList.add('hidden');
                UI.enrollSuccess.classList.add('hidden');
            }, 2000);
        } else {
            // 에러 메시지 표시
            UI.enrollError.textContent = data.message || data.error || '등록에 실패했습니다.';
            UI.enrollError.classList.remove('hidden');
            UI.enrollSuccess.classList.add('hidden');
        }
    } catch (error) {
        console.error('등록 실패:', error);
        UI.enrollError.textContent = `등록 중 오류가 발생했습니다: ${error.message}`;
        UI.enrollError.classList.remove('hidden');
    } finally {
        // 버튼 활성화
        UI.submitEnrollBtn.disabled = false;
        UI.submitEnrollBtn.textContent = '등록';
    }
});

UI.proceedBtn.addEventListener('click', () => {
    if (state.selectedSuspects.length > 0) {
        // 모달 닫기
        UI.suspectSelectModal.classList.add('hidden');

        // 새가 ID 생성 (타임스탬프 기반)
        state.sessionId = `session_${Date.now()}`;
        console.log(`세션 ID: ${state.sessionId}`);

        // 스냅샷 배열 초기화
        state.snapshots = [];
        state.nextSnapshotId = 1;
        
        // 클립 배열 초기화
        state.detectionClips = [];
        state.activeClips = {};
        state.nextClipId = 1;
        
        // 감지 로그 초기화 (새 영상 시작 시)
        state.detectionLogs = [];
        state.lastLogTimeByPerson.clear();
        const detectionLogList = UI.detectionLogList || document.getElementById('detectionLogList');
        if (detectionLogList) {
            detectionLogList.innerHTML = '<li class="text-gray-500 text-center py-4 tracking-tight">감지 대기 중...</li>';
        }

        // 타임라인 초기화
        const timelineBar = document.getElementById('timelineBar');
        if (timelineBar) {
            timelineBar.innerHTML = '';
        }

        // 스냅샷 카운트 초기화
        updateSnapshotCountDirect();

        // [여기가 핵심!] 
        // 사용자가 업로드한 videoFile(mp4, mov)을 브라우저가 읽을 수 있는 URL로 변환
        const videoURL = URL.createObjectURL(state.selectedFile);

        // HTML의 <video> 태그에 주입
        UI.video.src = videoURL;

        // 동영상 재생 시작
        UI.video.play();

        // 프레임 추출 버튼 활성화
        if (UI.extractFramesBtn) {
            UI.extractFramesBtn.disabled = false;
        }

        initCaptureCanvas();

        // WebSocket 연결 준비 (감지 시작 전에 미리 연결)
        if (state.useWebSocket) {
            connectWebSocket();
            // 연결 완료 및 설정 완료 후 첫 프레임 전송 (onopen과 config_updated에서 처리)
        }
    }
});

// 프레임 추출 기능
UI.extractFramesBtn?.addEventListener('click', async () => {
    if (!state.selectedFile) {
        alert('비디오 파일이 선택되지 않았습니다.');
        return;
    }

    // 확인 대화상자
    const confirmExtract = confirm(
        '모든 프레임을 추출하시겠습니까?\n\n' +
        '이 작업은 시간이 걸릴 수 있으며, 많은 프레임이 생성될 수 있습니다.'
    );

    if (!confirmExtract) {
        return;
    }

    // 버튼 비활성화 및 상태 변경
    UI.extractFramesBtn.disabled = true;
    UI.extractFramesBtn.textContent = '추출 중...';

    try {
        // FormData 생성
        const formData = new FormData();
        formData.append('video', state.selectedFile);

        // 서버로 요청
        const response = await fetch(`${API_BASE_URL}/extract_frames`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `서버 오류: ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
            alert(
                `프레임 추출 완료!\n\n` +
                `총 프레임 수: ${result.total_frames}개\n` +
                `저장 위치: ${result.output_dir}\n\n` +
                `라벨링을 위해 프레임들을 확인하세요.`
            );
        } else {
            throw new Error(result.message || '프레임 추출 실패');
        }
    } catch (error) {
        console.error('프레임 추출 실패:', error);
        alert(`프레임 추출 중 오류가 발생했습니다:\n${error.message}`);
    } finally {
        // 버튼 활성화 및 상태 복원
        UI.extractFramesBtn.disabled = false;
        UI.extractFramesBtn.textContent = '프레임 추출';
    }
});

UI.detectionFilter.addEventListener('change', (e) => {
    state.isDetectionActive = e.target.checked;

    if (state.isDetectionActive) {
        // 감지 시작
        console.log("🚀 AI 감지 시작");
        updateDetectionPanel({ message: "AI 분석 시작..." });

        // 처리 상태 초기화 (이전 요청이 완료되지 않았을 수 있음)
        state.isProcessing = false;

        // WebSocket 연결 시도 (백그라운드에서 연결, 연결 완료되면 자동 전환)
        if (state.useWebSocket && !state.isWsConnected) {
            connectWebSocket();
            // WebSocket 연결 완료 및 설정 완료 후 자동으로 WebSocket으로 전환됨
            // 하지만 연결 완료 전까지는 HTTP로 즉시 시작
        }

        // 비디오가 로드되었는지 확인
        if (UI.video.readyState < 2) {
            console.warn("⚠️ 비디오가 아직 로드되지 않았습니다. 비디오 로드 대기 중...");
            UI.video.addEventListener('loadeddata', () => {
                console.log("✅ 비디오 로드 완료, 감지 시작");
                // WebSocket 연결 상태와 관계없이 HTTP로 즉시 시작 (WebSocket 준비되면 자동 전환)
                console.log("🚀 HTTP 모드로 감지 시작 (WebSocket 준비되면 자동 전환)");
                processRealtimeDetection();
                state.detectionInterval = setInterval(processRealtimeDetection, 100);
            }, { once: true });
        } else {
            // WebSocket 연결 상태와 관계없이 HTTP로 즉시 시작 (WebSocket 준비되면 자동 전환)
            console.log("🚀 HTTP 모드로 감지 시작 (WebSocket 준비되면 자동 전환)");
            processRealtimeDetection();
            state.detectionInterval = setInterval(processRealtimeDetection, 100);
        }
    } else {
        // 감지 종료
        console.log("⏹️ AI 감지 중지");
        clearInterval(state.detectionInterval);
        // disconnectWebSocket(); // 연결은 유지하여 재시작 시 딜레이 제거

        // 현재 감지 중인 클립 종료
        if (state.currentClip) {
            const endTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : 0;
            state.currentClip.endTime = endTime;
            state.detectionClips.push(state.currentClip);
            console.log(`✅ 감지 클립 종료: ${state.currentClip.personName} (${state.currentClip.startTime.toFixed(1)}s - ${endTime.toFixed(1)}s)`);
            state.currentClip = null;
            // 클립 개수 업데이트
            updateClipCount();
        }

        // 캔버스 클리어
        if (state.detectionCtx) {
            state.detectionCtx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);
        }
        UI.video.parentElement.classList.remove('alert-border');
        updateDetectionPanel({ message: "분석 중지됨" });
        state.isProcessing = false; // 강제 초기화
    }
});

// 감지 로그 아이템 추가 함수
function addDetectionLogItem(data, isAlert, videoTime, snapshotImage) {
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
    
    // 썸네일 이미지 처리
    let thumbnailHTML = '';
    if (snapshotImage) {
        thumbnailHTML = `<img src="${snapshotImage}" alt="${name}" class="w-10 h-10 rounded-full object-cover">`;
    } else {
        // 기본 아이콘
        thumbnailHTML = `<div class="w-10 h-10 rounded-full ${status === 'criminal' || isAlert ? 'bg-red-100' : status === 'missing' ? 'bg-blue-100' : 'bg-gray-100'} flex items-center justify-center">
            <svg class="w-6 h-6 ${status === 'criminal' || isAlert ? 'text-red-600' : status === 'missing' ? 'text-blue-600' : 'text-gray-400'}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
function updateDetectionPanel(data, isAlert, videoTime, snapshotImage) {
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

// ==========================================
// CSV 내보내기 기능
// ==========================================

// 감지 로그를 CSV 파일로 다운로드
function downloadLogToCSV() {
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

// CSV 다운로드 버튼 이벤트 리스너
UI.downloadLogBtn?.addEventListener('click', () => {
    downloadLogToCSV();
});

// ==========================================
// 초기화 및 모달 이벤트
// ==========================================

// 대시보드 화면 업데이트 (빈 화면 vs 영상 화면)
function updateDashboardView() {
    if (state.videoUploaded) {
        // 영상이 있으면 빈 상태 카드 숨김
        UI.emptyStateCard.classList.add('hidden');
    } else {
        // 영상이 없으면 빈 상태 카드 표시
        UI.emptyStateCard.classList.remove('hidden');
    }
}

// 파일 업로드 모달 열기
UI.emptyStateCard?.addEventListener('click', () => {
    UI.uploadModal.classList.remove('hidden');
});

UI.openUploadModalBtn?.addEventListener('click', () => {
    UI.uploadModal.classList.remove('hidden');
});

// 파일 업로드 모달 닫기
UI.closeUploadModal?.addEventListener('click', () => {
    UI.uploadModal.classList.add('hidden');
});

// 모니터링 시작 버튼 (인물 선택 완료)
UI.proceedBtn?.addEventListener('click', () => {
    console.log('🎯 모니터링 시작 버튼 클릭');

    // 선택된 인물들의 타임라인 트랙 생성
    initializeTimelinesForSelectedPersons();

    // 모달 닫기
    UI.suspectSelectModal.classList.add('hidden');

    console.log(`✅ 모니터링 준비 완료 - 선택된 인물: ${state.selectedSuspects.length}명`);
});

// 인물 선택 모달 열기
UI.openSuspectModalBtn?.addEventListener('click', async () => {
    UI.suspectSelectModal.classList.remove('hidden');
    // 모달 열 때마다 인물 목록 갱신
    await renderSuspectCards();
});

// 인물 선택 모달 닫기
UI.closeSuspectModal?.addEventListener('click', () => {
    UI.suspectSelectModal.classList.add('hidden');
});

// 모달 외부 클릭 시 닫기
UI.uploadModal?.addEventListener('click', (e) => {
    if (e.target === UI.uploadModal) {
        UI.uploadModal.classList.add('hidden');
    }
});

UI.suspectSelectModal?.addEventListener('click', (e) => {
    if (e.target === UI.suspectSelectModal) {
        UI.suspectSelectModal.classList.add('hidden');
    }
});

// ESC 키로 모달 닫기
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (!UI.uploadModal.classList.contains('hidden')) {
            UI.uploadModal.classList.add('hidden');
        }
        if (!UI.suspectSelectModal.classList.contains('hidden')) {
            UI.suspectSelectModal.classList.add('hidden');
        }
        if (!UI.addSuspectModal.classList.contains('hidden')) {
            UI.addSuspectModal.classList.add('hidden');
        }
    }
});

// ==========================================
// 클립 데이터(clip)를 받아 카드 HTML을 반환하는 함수
function getClipItemHTML(clip) {
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };
    
    const startTimeFormatted = formatTime(clip.startTime);
    const endTimeFormatted = clip.endTime ? formatTime(clip.endTime) : '진행 중';
    const duration = clip.endTime ? (clip.endTime - clip.startTime).toFixed(1) : '진행 중';
    
    return `
    <div class="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden hover:shadow-md transition-shadow duration-200" data-clip-id="${clip.id}">
        <div class="p-4 flex items-start gap-4">
            <div class="relative flex items-center mt-1">
                <input type="checkbox" 
                       id="clip-check-${clip.id}" 
                       value="${clip.id}"
                       class="peer appearance-none h-6 w-6 rounded-full border-2 border-gray-300 bg-white 
                              checked:bg-indigo-600 checked:border-transparent 
                              checked:ring-4 checked:ring-indigo-500/20 
                              transition-all duration-200 cursor-pointer z-10"
                       ${clip.isSelected ? 'checked' : ''}
                       ${!clip.endTime ? 'disabled' : ''}
                       onchange="toggleClipSelection(${clip.id}, this.checked)">
                
                <svg class="absolute w-4 h-4 text-white left-1 top-1 pointer-events-none opacity-0 peer-checked:opacity-100 transition-opacity duration-200 z-20" 
                     fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
                </svg>
            </div>
            <div class="flex-1 min-w-0">
                <div class="flex justify-between items-start">
                    <div>
                        <h4 class="text-base font-bold text-gray-800 leading-tight">${clip.personName || 'Unknown'}</h4>
                        <div class="flex items-center gap-2 mt-1">
                            <span class="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                                ${startTimeFormatted} - ${endTimeFormatted}
                            </span>
                            <span class="text-xs text-gray-400">
                                (길이: ${duration}초)
                            </span>
                        </div>
                    </div>
                    <button onclick="window.seekToClip(${clip.startTime})" 
                            class="text-xs bg-indigo-50 hover:bg-indigo-100 text-indigo-700 px-3 py-1.5 rounded-lg font-medium transition-colors">
                        재생
                    </button>
                </div>
            </div>
        </div>
        ${clip.videoUrl && clip.endTime ? `
        <div class="w-full bg-black aspect-video relative group">
            <video src="${clip.videoUrl}" 
                   class="w-full h-full object-contain" 
                   controls 
                   preload="metadata"
                   onloadedmetadata="this.currentTime=${clip.startTime}">
                <source src="${clip.videoUrl}" type="video/mp4">
                비디오를 재생할 수 없습니다.
            </video>
        </div>
        ` : ''}
    </div>
    `;
}

// 클립/스냅샷 버튼 이벤트
// ==========================================
// 클립 보기 버튼 이벤트
UI.viewClipsBtn?.addEventListener('click', () => {
    console.log('📹 클립 보기 버튼 클릭');
    console.log(`현재 클립 개수: ${state.detectionClips.length}`);
    
    const modal = document.getElementById('clipModal');
    const list = document.getElementById('clipList');
    
    if (!modal || !list) {
        console.error('클립 모달 요소를 찾을 수 없습니다.');
        return;
    }
    
    if (state.detectionClips.length === 0) {
        list.innerHTML = '<p class="text-center py-8 text-gray-500">아직 감지된 클립이 없습니다.</p>';
    } else {
        const formatTime = (seconds) => {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        };
        
        // 선택 상태 초기화 (모달 열 때마다)
        state.selectedClips = [];
        updateSelectedClipCount();
        
        list.innerHTML = state.detectionClips.map(clip => {
            const videoUrl = state.selectedFile ? URL.createObjectURL(state.selectedFile) : '';
            const isSelected = state.selectedClips.includes(clip.id);
            
            return getClipItemHTML({
                ...clip,
                videoUrl: videoUrl,
                isSelected: isSelected
            });
        }).join('');
    }
    
    modal.classList.remove('hidden');
});

// 클립 선택 토글 함수
window.toggleClipSelection = function(clipId, isChecked) {
    if (isChecked) {
        if (!state.selectedClips.includes(clipId)) {
            state.selectedClips.push(clipId);
        }
    } else {
        state.selectedClips = state.selectedClips.filter(id => id !== clipId);
    }
    updateSelectedClipCount();
};

// 선택된 클립 개수 업데이트
function updateSelectedClipCount() {
    const countEl = document.getElementById('selectedClipCount');
    if (countEl) {
        countEl.textContent = state.selectedClips.length;
    }
    
    // 선택 다운로드 버튼 활성화/비활성화
    const downloadSelectedClipsBtn = document.getElementById('downloadSelectedClipsBtn');
    if (downloadSelectedClipsBtn) {
        downloadSelectedClipsBtn.disabled = state.selectedClips.length === 0;
    }
}

// 전체 클립 선택 버튼
document.getElementById('selectAllClipsBtn')?.addEventListener('click', () => {
    const checkboxes = document.querySelectorAll('#clipList input[type="checkbox"]:not(:disabled)');
    
    // 모든 체크박스를 선택 상태로 설정
    checkboxes.forEach(checkbox => {
        const clipId = parseInt(checkbox.value || checkbox.id.replace('clip-check-', ''));
        checkbox.checked = true;
        
        // 상태 동기화
        if (!state.selectedClips.includes(clipId)) {
            state.selectedClips.push(clipId);
        }
    });
    
    // 개수 업데이트
    updateSelectedClipCount();
});

// 전체 클립 해제 버튼
document.getElementById('deselectAllClipsBtn')?.addEventListener('click', () => {
    const checkboxes = document.querySelectorAll('#clipList input[type="checkbox"]');
    
    // 모든 체크박스를 해제 상태로 설정
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
    
    // 상태 초기화
    state.selectedClips = [];
    
    // 개수 업데이트
    updateSelectedClipCount();
});

// 선택 클립 다운로드 버튼 이벤트
document.getElementById('downloadSelectedClipsBtn')?.addEventListener('click', async () => {
    if (state.selectedClips.length === 0) {
        alert('다운로드할 클립을 선택해주세요.');
        return;
    }
    
    const selectedClips = state.detectionClips.filter(clip => 
        state.selectedClips.includes(clip.id) && clip.endTime
    );
    
    if (selectedClips.length === 0) {
        alert('선택된 클립을 찾을 수 없거나 아직 완료되지 않은 클립입니다.');
        return;
    }
    
    if (!state.selectedFile) {
        alert('비디오 파일이 없습니다.');
        return;
    }
    
    console.log(`🎬 ${selectedClips.length}개의 선택된 클립 다운로드 시작`);
    
    // 순차적으로 다운로드
    for (let i = 0; i < selectedClips.length; i++) {
        const clip = selectedClips[i];
        try {
            await downloadVideoClip(clip);
            // 다운로드 간 약간의 딜레이
            if (i < selectedClips.length - 1) {
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        } catch (error) {
            console.error(`클립 다운로드 실패: ${clip.id}`, error);
            alert(`클립 다운로드 중 오류가 발생했습니다: ${clip.personName}`);
        }
    }
    
    console.log(`✅ ${selectedClips.length}개의 선택된 클립 다운로드 완료`);
});

// 스냅샷 카드 렌더링 헬퍼 함수
function renderSnapshotCard(snapshot) {
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };
    
    const isSelected = state.selectedSnapshots.includes(snapshot.id);
    
    return `
                <div class="bg-white rounded-lg shadow-sm overflow-hidden relative" data-person-name="${snapshot.personName}" data-snapshot-id="${snapshot.id}">
                    <div class="absolute top-2 left-2 z-10">
                        <input type="checkbox" 
                               class="snapshot-checkbox appearance-none h-7 w-7 rounded-full border-2 border-white/50 bg-white/20 backdrop-blur-sm checked:bg-white/40 checked:border-white/80 focus:ring-2 focus:ring-white/50 focus:ring-offset-2 cursor-pointer transition-all duration-200 ease-in-out" 
                               ${isSelected ? 'checked' : ''}
                               onchange="toggleSnapshotSelection(${snapshot.id}, this.checked)">
                    </div>
                    <img src="${snapshot.base64Image}" alt="${snapshot.personName}" class="w-full h-48 object-cover cursor-pointer" 
                         onclick="window.open(this.src)">
                    <div class="p-3">
                        <div class="font-semibold text-sm text-gray-800 tracking-tight">${snapshot.personName}</div>
                        <div class="text-xs text-gray-600 mt-1 tracking-tight">시간: ${formatTime(snapshot.videoTime)}</div>
                        <div class="text-xs text-gray-600 tracking-tight">유사도: ${snapshot.similarity}%</div>
                        <div class="text-xs text-gray-500 tracking-tight">${new Date(snapshot.timestamp).toLocaleString()}</div>
                    </div>
                </div>
    `;
}

// 스냅샷 그리드 필터링 함수
function filterSnapshotsByPerson(personName) {
    const grid = document.getElementById('snapshotGrid');
    if (!grid) return;
    
    const cards = grid.querySelectorAll('[data-person-name]');
    cards.forEach(card => {
        if (personName === '전체' || card.dataset.personName === personName) {
            card.style.display = '';
        } else {
            card.style.display = 'none';
        }
    });
}

// 현재 선택된 인물 필터 (전역 변수)
let currentSnapshotFilter = '전체';

// 스냅샷 보기 버튼 이벤트
UI.viewSnapshotsBtn?.addEventListener('click', () => {
    console.log('📸 스냅샷 보기 버튼 클릭');
    console.log(`현재 스냅샷 개수: ${state.snapshots.length}`);
    
    const modal = document.getElementById('snapshotModal');
    const grid = document.getElementById('snapshotGrid');
    const tabsContainer = document.getElementById('snapshotTabs');
    
    if (!modal || !grid || !tabsContainer) {
        console.error('스냅샷 모달 요소를 찾을 수 없습니다.');
        return;
    }
    
    if (state.snapshots.length === 0) {
        tabsContainer.innerHTML = '';
        grid.innerHTML = '<p class="col-span-full text-center py-8 text-gray-500">아직 캡처된 스냅샷이 없습니다.</p>';
        currentSnapshotFilter = '전체';
    } else {
        // 인물별로 그룹화
        const personGroups = {};
        state.snapshots.forEach(snapshot => {
            const personName = snapshot.personName || 'Unknown';
            if (!personGroups[personName]) {
                personGroups[personName] = [];
            }
            personGroups[personName].push(snapshot);
        });
        
        // 탭 생성
        const personNames = Object.keys(personGroups).sort();
        tabsContainer.innerHTML = `
            <div class="flex flex-wrap gap-2 overflow-x-auto pb-2">
                <button class="snapshot-tab active px-4 py-2 rounded-lg text-sm font-medium bg-indigo-600 text-white hover:bg-indigo-700 transition-all duration-200 shadow-sm" 
                        data-person="전체">
                    전체 (${state.snapshots.length})
                </button>
                ${personNames.map(personName => `
                    <button class="snapshot-tab px-4 py-2 rounded-lg text-sm font-medium bg-gray-200 text-gray-700 hover:bg-gray-300 transition-all duration-200" 
                            data-person="${personName}">
                        ${personName} (${personGroups[personName].length})
                    </button>
                `).join('')}
            </div>
        `;
        
        // 모든 스냅샷 렌더링
        grid.innerHTML = state.snapshots.map(snapshot => renderSnapshotCard(snapshot)).join('');
        
        // 초기 필터 적용
        currentSnapshotFilter = '전체';
        filterSnapshotsByPerson('전체');
        
        // 탭 클릭 이벤트 등록
        const tabs = tabsContainer.querySelectorAll('.snapshot-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // 모든 탭 비활성화
                tabs.forEach(t => {
                    t.classList.remove('active', 'bg-indigo-600', 'text-white', 'shadow-sm');
                    t.classList.add('bg-gray-200', 'text-gray-700');
                });
                
                // 클릭한 탭 활성화
                tab.classList.add('active', 'bg-indigo-600', 'text-white', 'shadow-sm');
                tab.classList.remove('bg-gray-200', 'text-gray-700');
                
                // 필터링 적용
                const selectedPerson = tab.dataset.person;
                currentSnapshotFilter = selectedPerson;
                filterSnapshotsByPerson(selectedPerson);
                
                // 필터 변경 시 선택 상태 유지 (체크박스만 업데이트)
                updateSnapshotCheckboxes();
                updateSelectedCount();
            });
        });
    }
    
    modal.classList.remove('hidden');
});

// ==========================================
// 클립/스냅샷 모달 닫기 이벤트
// ==========================================
// 클립 모달 닫기 버튼들
document.getElementById('closeClipModalBtn')?.addEventListener('click', () => {
    const clipModal = document.getElementById('clipModal');
    if (clipModal) {
        clipModal.classList.add('hidden');
    }
});

document.getElementById('closeClipModalBtn2')?.addEventListener('click', () => {
    const clipModal = document.getElementById('clipModal');
    if (clipModal) {
        clipModal.classList.add('hidden');
    }
});

// 전역 함수: 클립으로 이동 (HTML onclick에서 호출)
window.seekToClip = function(startTime) {
    if (UI.video) {
        UI.video.currentTime = startTime;
        UI.video.play();
        const clipModal = document.getElementById('clipModal');
        if (clipModal) {
            clipModal.classList.add('hidden');
        }
    }
};

// 전역 함수: 클립 다운로드 (HTML onclick에서 호출)
window.downloadClip = function(clipId) {
    const clip = state.detectionClips.find(c => c.id === clipId);
    if (clip) {
        downloadVideoClip(clip);
    } else {
        console.error(`클립을 찾을 수 없습니다: ${clipId}`);
    }
};

// 스냅샷 모달 닫기 버튼들  
document.getElementById('closeModalBtn')?.addEventListener('click', () => {
    const snapshotModal = document.getElementById('snapshotModal');
    if (snapshotModal) {
        snapshotModal.classList.add('hidden');
    }
});

document.getElementById('closeModalBtn2')?.addEventListener('click', () => {
    const snapshotModal = document.getElementById('snapshotModal');
    if (snapshotModal) {
        snapshotModal.classList.add('hidden');
    }
});

// 모달 외부 클릭 시 닫기
document.getElementById('clipModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'clipModal') {
        e.target.classList.add('hidden');
    }
});

document.getElementById('snapshotModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'snapshotModal') {
        e.target.classList.add('hidden');
    }
});

// 전역 함수: 스냅샷 다운로드 (HTML onclick에서 호출)
window.downloadSnapshot = function(snapshotId) {
    const snapshot = state.snapshots.find(s => s.id === snapshotId);
    if (!snapshot) {
        console.error(`스냅샷을 찾을 수 없습니다: ${snapshotId}`);
        return;
    }
    
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };
    
    const link = document.createElement('a');
    link.href = snapshot.base64Image;
    link.download = `criminal_${snapshot.personName}_${formatTime(snapshot.videoTime).replace(':', '-')}.jpg`;
    link.click();
};

// 스냅샷 선택 토글 함수
window.toggleSnapshotSelection = function(snapshotId, isChecked) {
    if (isChecked) {
        if (!state.selectedSnapshots.includes(snapshotId)) {
            state.selectedSnapshots.push(snapshotId);
        }
    } else {
        state.selectedSnapshots = state.selectedSnapshots.filter(id => id !== snapshotId);
    }
    updateSelectedCount();
};

// 선택된 스냅샷 개수 업데이트
function updateSelectedCount() {
    const countEl = document.getElementById('selectedCount');
    if (countEl) {
        countEl.textContent = state.selectedSnapshots.length;
    }
    
    // 선택 다운로드 버튼 활성화/비활성화
    const downloadSelectedBtn = document.getElementById('downloadSelectedBtn');
    if (downloadSelectedBtn) {
        downloadSelectedBtn.disabled = state.selectedSnapshots.length === 0;
    }
};

// 전체 선택 버튼
document.getElementById('selectAllBtn')?.addEventListener('click', () => {
    // 현재 필터에 맞는 스냅샷만 선택
    const filteredSnapshots = currentSnapshotFilter === '전체' 
        ? state.snapshots 
        : state.snapshots.filter(s => s.personName === currentSnapshotFilter);
    
    filteredSnapshots.forEach(snapshot => {
        if (!state.selectedSnapshots.includes(snapshot.id)) {
            state.selectedSnapshots.push(snapshot.id);
        }
    });
    
    // 체크박스 업데이트
    updateSnapshotCheckboxes();
    updateSelectedCount();
});

// 전체 해제 버튼
document.getElementById('deselectAllBtn')?.addEventListener('click', () => {
    // 현재 필터에 맞는 스냅샷만 해제
    const filteredSnapshots = currentSnapshotFilter === '전체' 
        ? state.snapshots 
        : state.snapshots.filter(s => s.personName === currentSnapshotFilter);
    
    const filteredIds = filteredSnapshots.map(s => s.id);
    state.selectedSnapshots = state.selectedSnapshots.filter(id => !filteredIds.includes(id));
    
    // 체크박스 업데이트
    updateSnapshotCheckboxes();
    updateSelectedCount();
});

// 체크박스 상태 업데이트
function updateSnapshotCheckboxes() {
    const checkboxes = document.querySelectorAll('.snapshot-checkbox');
    checkboxes.forEach(checkbox => {
        const snapshotId = parseInt(checkbox.getAttribute('onchange').match(/\d+/)[0]);
        checkbox.checked = state.selectedSnapshots.includes(snapshotId);
    });
}

// 선택 다운로드 버튼 이벤트
document.getElementById('downloadSelectedBtn')?.addEventListener('click', async () => {
    if (state.selectedSnapshots.length === 0) {
        alert('다운로드할 스냅샷을 선택해주세요.');
        return;
    }
    
    const selectedSnapshots = state.snapshots.filter(s => state.selectedSnapshots.includes(s.id));
    
    if (selectedSnapshots.length === 0) {
        alert('선택된 스냅샷을 찾을 수 없습니다.');
        return;
    }
    
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };
    
    // 순차적으로 다운로드
    for (let i = 0; i < selectedSnapshots.length; i++) {
        const snapshot = selectedSnapshots[i];
        const link = document.createElement('a');
        link.href = snapshot.base64Image;
        link.download = `${i + 1}_criminal_${snapshot.personName}_${formatTime(snapshot.videoTime).replace(':', '-')}.jpg`;
        link.click();
        
        // 다운로드 간 약간의 딜레이 (브라우저가 처리할 시간 제공)
        if (i < selectedSnapshots.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }
    
    console.log(`✅ ${selectedSnapshots.length}개의 선택된 스냅샷 다운로드 완료`);
});

// 전체 다운로드 버튼 이벤트 (현재 필터링된 스냅샷만 다운로드)
document.getElementById('downloadAllBtn')?.addEventListener('click', async () => {
    if (state.snapshots.length === 0) {
        alert('다운로드할 스냅샷이 없습니다.');
        return;
    }
    
    // 현재 필터에 맞는 스냅샷만 필터링
    const filteredSnapshots = currentSnapshotFilter === '전체' 
        ? state.snapshots 
        : state.snapshots.filter(s => s.personName === currentSnapshotFilter);
    
    if (filteredSnapshots.length === 0) {
        alert('다운로드할 스냅샷이 없습니다.');
        return;
    }
    
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };
    
    // 순차적으로 다운로드
    for (let i = 0; i < filteredSnapshots.length; i++) {
        const snapshot = filteredSnapshots[i];
        const link = document.createElement('a');
        link.href = snapshot.base64Image;
        link.download = `${i + 1}_criminal_${snapshot.personName}_${formatTime(snapshot.videoTime).replace(':', '-')}.jpg`;
        link.click();
        
        // 브라우저가 따라잡을 시간 주기
        await new Promise(resolve => setTimeout(resolve, 300));
    }
    
    alert(`${filteredSnapshots.length}개의 스냅샷 다운로드를 시작했습니다.`);
});

// 초기 화면 설정
updateDashboardView();

// 캔버스 초기화
initCaptureCanvas();

// ==========================================
// 세그먼트 컨트롤 (인물 타입 선택)
// ==========================================
// 구분 선택 버튼 상태 업데이트 함수
function updatePersonTypeButtons() {
    const typeCriminal = document.getElementById('typeCriminal');
    const typeMissing = document.getElementById('typeMissing');
    const btnCriminal = document.getElementById('btnCriminal');
    const btnMissing = document.getElementById('btnMissing');
    
    if (typeCriminal && typeCriminal.checked) {
        // 범죄자 선택됨
        btnCriminal.classList.add('bg-white', 'shadow-sm', 'text-red-600');
        btnCriminal.classList.remove('text-gray-500');
        btnMissing.classList.add('text-gray-500');
        btnMissing.classList.remove('bg-white', 'shadow-sm', 'text-blue-600');
    document.getElementById('personTypeInput').value = 'criminal';
    } else if (typeMissing && typeMissing.checked) {
        // 실종자 선택됨
        btnMissing.classList.add('bg-white', 'shadow-sm', 'text-blue-600');
        btnMissing.classList.remove('text-gray-500');
        btnCriminal.classList.add('text-gray-500');
        btnCriminal.classList.remove('bg-white', 'shadow-sm', 'text-red-600');
        document.getElementById('personTypeInput').value = 'missing';
    }
    
    // 폼 유효성 검사
    checkFormValidity();
}

// 구분 선택 라디오 버튼 이벤트 리스너
document.getElementById('typeCriminal')?.addEventListener('change', () => {
    updatePersonTypeButtons();
});

document.getElementById('typeMissing')?.addEventListener('change', () => {
    updatePersonTypeButtons();
});

// 구분 선택 라벨 클릭 이벤트 (라디오 버튼 토글)
document.getElementById('btnCriminal')?.addEventListener('click', () => {
    const typeCriminal = document.getElementById('typeCriminal');
    if (typeCriminal) {
        typeCriminal.checked = true;
        updatePersonTypeButtons();
    }
});

document.getElementById('btnMissing')?.addEventListener('click', () => {
    const typeMissing = document.getElementById('typeMissing');
    if (typeMissing) {
        typeMissing.checked = true;
        updatePersonTypeButtons();
    }
});

console.log("✅ FaceWatch 프론트엔드 초기화 완료");