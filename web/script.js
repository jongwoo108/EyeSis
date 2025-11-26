// script.js

// ==========================================
// 전역 변수 및 상태 관리
// ==========================================
const API_BASE_URL = 'http://localhost:5000/api';
const WS_URL = 'ws://localhost:5000/ws/detect';
const WS_TEST_URL = 'ws://localhost:5000/ws/test'; // 테스트용

const state = {
    selectedFile: null,
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
    detectionClips: [], // 범죄자 감지 구간 배열 [{startTime, endTime, personId, personName, ...}]
    currentClip: null // 현재 감지 중인 클립 (null이면 감지 중이 아님)
};

// DOM 요소
const UI = {
    screens: {
        upload: document.getElementById('uploadScreen'),
        suspect: document.getElementById('suspectSelectScreen'),
        dashboard: document.getElementById('dashboardScreen')
    },
    video: document.getElementById('mainVideo'),
    detectionCanvas: document.getElementById('detectionCanvas'),
    videoFile: document.getElementById('videoFile'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    fileInfo: document.getElementById('fileInfo'),
    fileName: document.getElementById('fileName'),
    suspectCardsContainer: document.getElementById('suspectCardsContainer'),
    proceedBtn: document.getElementById('proceedToDashboard'),
    detectionFilter: document.getElementById('detectionFilter'),
    detectionInfo: document.getElementById('detectionInfo'),
    selectedSuspectName: document.getElementById('selectedSuspectName'),
    selectedSuspectInfo: document.getElementById('selectedSuspectInfo'),
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
    enrollError: document.getElementById('enrollError'),
    enrollSuccess: document.getElementById('enrollSuccess'),
    submitEnrollBtn: document.getElementById('submitEnrollBtn'),
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
    const bgColor = isCriminal ? 'bg-red-100' : 'bg-blue-100';
    const textColor = isCriminal ? 'text-red-600' : 'text-green-600';
    const statusText = isCriminal ? '범죄자' : '일반인';

    const card = document.createElement('div');
    card.className = 'suspect-card bg-white rounded-lg shadow-lg overflow-hidden cursor-pointer transform hover:scale-105 transition-all duration-200 relative';
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
        <div class="aspect-w-3 aspect-h-4 ${bgColor} flex items-center justify-center p-8 overflow-hidden">
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
            
            // 현재 감지 중인 클립 종료
            if (state.currentClip) {
                const endTime = UI.video.currentTime;
                state.currentClip.endTime = endTime;
                state.detectionClips.push(state.currentClip);
                console.log(`✅ 감지 클립 종료: ${state.currentClip.personName} (${state.currentClip.startTime.toFixed(1)}s - ${endTime.toFixed(1)}s)`);
                state.currentClip = null;
                updateClipCount();
            }
            
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
        state.detectionCanvas.width = videoRect.width;
        state.detectionCanvas.height = videoRect.height;
    }
}

// 타임라인 마커 추가 (직접 구현)
function addTimelineMarkerDirect(snapshot) {
    console.log('📌 addTimelineMarkerDirect 호출됨:', {
        videoExists: !!UI.video,
        videoDuration: UI.video?.duration,
        videoTime: snapshot.videoTime,
        snapshotId: snapshot.id
    });
    
    if (!UI.video || !UI.video.duration || UI.video.duration === 0 || isNaN(UI.video.duration)) {
        console.warn('⚠️ 비디오 duration이 아직 설정되지 않음, 재시도 예약');
        // 비디오가 아직 로드되지 않았으면 나중에 다시 시도
        setTimeout(() => addTimelineMarkerDirect(snapshot), 100);
        return;
    }

    const timelineBar = document.getElementById('timelineBar');
    if (!timelineBar) {
        console.error('❌ 타임라인 바 요소를 찾을 수 없습니다!');
        return;
    }

    const position = (snapshot.videoTime / UI.video.duration) * 100;
    if (position < 0 || position > 100) {
        console.warn('⚠️ 타임라인 위치가 범위를 벗어남:', position);
        return;
    }

    const marker = document.createElement('div');
    marker.className = 'absolute w-3 h-full bg-red-500 cursor-pointer hover:bg-red-700 transition-colors z-10';
    marker.style.left = `${position}%`;
    
    // 시간 포맷 헬퍼
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };
    
    marker.title = `${snapshot.personName} - ${formatTime(snapshot.videoTime)}`;
    marker.dataset.snapshotId = snapshot.id;

    marker.addEventListener('click', (e) => {
        e.stopPropagation();
        if (UI.video) {
            UI.video.currentTime = snapshot.videoTime;
            UI.video.play();
        }
    });

    timelineBar.appendChild(marker);
    console.log(`✅ 타임라인 마커 추가됨: ${snapshot.personName} at ${position.toFixed(1)}% (${formatTime(snapshot.videoTime)})`);
}

// 스냅샷 개수 업데이트 (직접 구현)
function updateSnapshotCountDirect() {
    console.log('🔢 updateSnapshotCountDirect 호출됨:', {
        snapshotCount: state.snapshots.length
    });
    
    const countEl = document.getElementById('snapshotCount');
    if (countEl) {
        countEl.textContent = state.snapshots.length;
        console.log(`✅ 스냅샷 개수 업데이트됨: ${state.snapshots.length}`);
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

    console.log(`🎬 클립 다운로드 시작: ${clip.personName} (${startTime.toFixed(1)}s - ${endTime.toFixed(1)}s)`);

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
                // 선택된 용의자가 없어도 설정 완료로 표시 (전체 DB 검색)
                state.wsConfigReady = true;
                console.log("✅ WebSocket 설정 완료 (용의자 미선택 - 전체 검색)");
                
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

        // 모든 감지 결과 로그 출력 (디버깅용)
        console.log('🔍 감지 결과 확인:', {
            alert: data.alert,
            hasSnapshot: !!data.snapshot_base64,
            snapshotLength: data.snapshot_base64 ? data.snapshot_base64.length : 0,
            detectionsCount: data.detections ? data.detections.length : 0,
            metadata: data.metadata,
            videoTimestamp: data.video_timestamp
        });
        
        // detections 배열에서 범죄자 확인
        if (data.detections && data.detections.length > 0) {
            const criminals = data.detections.filter(d => d.status === 'criminal');
            console.log(`👮 범죄자 감지: ${criminals.length}명`, criminals.map(c => c.name));
        }
        
        if (data.alert) {
            // 정확한 비디오 타임스탬프 사용 (백엔드 계산값보다 정확)
            const videoTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : (data.video_timestamp || 0);
            
            // 클립 추적: 범죄자 감지 시작
            if (!state.currentClip) {
                state.currentClip = {
                    id: state.detectionClips.length + 1,
                    startTime: videoTime,
                    endTime: null,
                    personId: data.metadata?.person_id || data.metadata?.name || 'Unknown',
                    personName: data.metadata?.name || 'Unknown',
                    similarity: data.metadata?.confidence || 0,
                    status: 'criminal'
                };
                console.log(`🎬 감지 클립 시작: ${state.currentClip.personName} (${videoTime.toFixed(1)}s)`);
            } else {
                // 같은 사람이 계속 감지되면 클립 업데이트
                state.currentClip.endTime = videoTime; // 종료 시간 갱신
            }
            
            // 스냅샷이 없으면 현재 프레임을 직접 캡처하여 사용
            let snapshotImage = data.snapshot_base64;
            if (!snapshotImage) {
                console.warn('⚠️ 범죄자 감지되었지만 snapshot_base64가 없습니다 (WebSocket)! 현재 프레임을 캡처합니다.');
                snapshotImage = captureVideoFrame();
                if (!snapshotImage) {
                    console.error('❌ 프레임 캡처도 실패했습니다. 스냅샷을 저장할 수 없습니다.');
                } else {
                    console.log('✅ 현재 프레임을 캡처하여 스냅샷으로 사용합니다.');
                }
            }
            
            if (snapshotImage) {
                const snapshot = {
                    id: state.nextSnapshotId++,
                    timestamp: new Date().toISOString(),
                    videoTime: videoTime,
                    personId: data.metadata?.person_id || data.metadata?.name || 'Unknown',
                    personName: data.metadata?.name || 'Unknown',
                    similarity: data.metadata?.confidence || 0,
                    base64Image: snapshotImage,
                    status: data.metadata?.status || 'criminal'
                };
                state.snapshots.push(snapshot);
                console.log(`✅ 스냅샷 저장됨: #${snapshot.id} - ${snapshot.personName} (${snapshot.videoTime.toFixed(1)}s), 총 ${state.snapshots.length}개`);
                
                // 타임라인 마커 추가 및 카운트 업데이트 (직접 구현)
                console.log('📌 타임라인 마커 추가 시도...');
                addTimelineMarkerDirect(snapshot);
                console.log('🔢 스냅샷 카운트 업데이트 시도...');
                updateSnapshotCountDirect();
            }
        } else {
            // 범죄자 감지 종료: 클립 종료
            if (state.currentClip) {
                const endTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : 0;
                state.currentClip.endTime = endTime;
                state.detectionClips.push(state.currentClip);
                console.log(`✅ 감지 클립 종료: ${state.currentClip.personName} (${state.currentClip.startTime.toFixed(1)}s - ${endTime.toFixed(1)}s)`);
                state.currentClip = null;
                // 클립 개수 업데이트
                updateClipCount();
            }
        }

        // 박스 렌더링
        if (data.detections && data.detections.length > 0 && UI.video.videoWidth > 0) {
            const videoWidth = UI.video.videoWidth;
            const videoHeight = UI.video.videoHeight;
            drawDetections(data.detections, videoWidth, videoHeight);
        } else {
            // 박스가 없으면 캔버스 클리어
            if (state.detectionCtx) {
                state.detectionCtx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);
            }
        }

        // 알림 및 로그 업데이트
        if (data.alert) {
            UI.video.parentElement.classList.add('alert-border');
            updateDetectionPanel(data.metadata, true);
        } else {
            UI.video.parentElement.classList.remove('alert-border');
            updateDetectionPanel(data.metadata, false);
        }

        // 처리 완료 플래그 해제
        state.isProcessing = false;

    } else if (msgType === "error") {
        console.error("❌ 서버 오류:", message.message);
        state.isProcessing = false;

    } else if (msgType === "pong") {
        // 연결 확인 응답 (필요시 처리)

    } else if (msgType === "config_updated") {
        console.log("✅ 설정 업데이트됨:", message.suspect_ids);
        state.wsConfigReady = true; // 설정 완료 플래그 설정
        
        // 설정 완료 후 첫 프레임 전송 (감지 활성화 상태일 때만)
        if (state.isDetectionActive && !state.isProcessing) {
            console.log("🚀 WebSocket 설정 완료, 첫 프레임 전송 시작");
            setTimeout(() => {
                processRealtimeDetection();
            }, 50); // 연결 안정화를 위한 짧은 대기
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

            // 범죄자 감지 시 스냅샷 저장 (HTTP 폴백용)
            console.log('🔍 HTTP 감지 결과 확인:', {
                alert: result.alert,
                hasSnapshot: !!result.snapshot_base64,
                snapshotLength: result.snapshot_base64 ? result.snapshot_base64.length : 0,
                metadata: result.metadata
            });
            
            if (result.alert) {
                const videoTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : (result.video_timestamp || 0);
                
                // 클립 추적: 범죄자 감지 시작
                if (!state.currentClip) {
                    state.currentClip = {
                        id: state.detectionClips.length + 1,
                        startTime: videoTime,
                        endTime: null,
                        personId: result.metadata?.person_id || result.metadata?.name || 'Unknown',
                        personName: result.metadata?.name || 'Unknown',
                        similarity: result.metadata?.confidence || 0,
                        status: 'criminal'
                    };
                    console.log(`🎬 감지 클립 시작: ${state.currentClip.personName} (${videoTime.toFixed(1)}s)`);
                } else {
                    // 같은 사람이 계속 감지되면 클립 업데이트
                    state.currentClip.endTime = videoTime; // 종료 시간 갱신
                }
                
                // 스냅샷이 없으면 현재 프레임을 직접 캡처하여 사용
                let snapshotImage = result.snapshot_base64;
                if (!snapshotImage) {
                    console.warn('⚠️ 범죄자 감지되었지만 snapshot_base64가 없습니다 (HTTP)! 현재 프레임을 캡처합니다.');
                    snapshotImage = captureVideoFrame();
                    if (!snapshotImage) {
                        console.error('❌ 프레임 캡처도 실패했습니다. 스냅샷을 저장할 수 없습니다.');
                    } else {
                        console.log('✅ 현재 프레임을 캡처하여 스냅샷으로 사용합니다.');
                    }
                }
                
                if (snapshotImage) {
                    const snapshot = {
                        id: state.nextSnapshotId++,
                        timestamp: new Date().toISOString(),
                        videoTime: videoTime,
                        personId: result.metadata?.person_id || result.metadata?.name || 'Unknown',
                        personName: result.metadata?.name || 'Unknown',
                        similarity: result.metadata?.confidence || 0,
                        base64Image: snapshotImage,
                        status: result.metadata?.status || 'criminal'
                    };
                    state.snapshots.push(snapshot);
                    console.log(`✅ 스냅샷 저장됨 (HTTP): #${snapshot.id} - ${snapshot.personName} (${snapshot.videoTime.toFixed(1)}s), 총 ${state.snapshots.length}개`);
                    
                    // 타임라인 마커 추가 및 카운트 업데이트 (직접 구현)
                    console.log('📌 타임라인 마커 추가 시도 (HTTP)...');
                    addTimelineMarkerDirect(snapshot);
                    console.log('🔢 스냅샷 카운트 업데이트 시도 (HTTP)...');
                    updateSnapshotCountDirect();
                }
            } else {
                // 범죄자 감지 종료: 클립 종료
                if (state.currentClip) {
                    const endTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : 0;
                    state.currentClip.endTime = endTime;
                    state.detectionClips.push(state.currentClip);
                    console.log(`✅ 감지 클립 종료: ${state.currentClip.personName} (${state.currentClip.startTime.toFixed(1)}s - ${endTime.toFixed(1)}s)`);
                    state.currentClip = null;
                    // 클립 개수 업데이트
                    updateClipCount();
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
                updateDetectionPanel(result.metadata, true);
            } else {
                UI.video.parentElement.classList.remove('alert-border');
                updateDetectionPanel(result.metadata, false);
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
            
            // 현재 감지 중인 클립 종료
            if (state.currentClip) {
                const endTime = UI.video.currentTime;
                state.currentClip.endTime = endTime;
                state.detectionClips.push(state.currentClip);
                console.log(`✅ 감지 클립 종료: ${state.currentClip.personName} (${state.currentClip.startTime.toFixed(1)}s - ${endTime.toFixed(1)}s)`);
                state.currentClip = null;
                updateClipCount();
            }
            
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
        UI.screens.upload.classList.add('hidden');
        UI.screens.suspect.classList.remove('hidden');
        // 용의자 선택 화면이 표시될 때 인물 목록 로드
        await renderSuspectCards();
    }
});

// 인물 카드 클릭 이벤트는 createSuspectCard 함수 내에서 처리됨

// ==========================================
// 용의자 추가 기능
// ==========================================

// 모달 열기
UI.addSuspectBtn?.addEventListener('click', () => {
    // 폼 완전 초기화
    UI.addSuspectForm.reset();
    UI.imagePreview.classList.add('hidden');
    UI.imagePlaceholder.classList.remove('hidden');
    UI.enrollError.classList.add('hidden');
    UI.enrollSuccess.classList.add('hidden');
    // 버튼 상태 초기화
    UI.submitEnrollBtn.disabled = false;
    UI.submitEnrollBtn.textContent = '등록';
    // 모달 표시
    UI.addSuspectModal.classList.remove('hidden');
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
});

// 폼 제출
UI.addSuspectForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const personId = UI.enrollPersonId.value.trim();
    const name = UI.enrollName.value.trim();
    const isCriminal = UI.enrollIsCriminal.checked;
    const imageFile = UI.enrollImage.files[0];
    
    // 유효성 검사
    if (!personId || !name || !imageFile) {
        UI.enrollError.textContent = '모든 필드를 입력해주세요.';
        UI.enrollError.classList.remove('hidden');
        UI.enrollSuccess.classList.add('hidden');
        return;
    }
    
    // person_id 유효성 검사 (영문, 숫자, 언더스코어만)
    if (!/^[a-zA-Z0-9_]+$/.test(personId)) {
        UI.enrollError.textContent = '인물 ID는 영문, 숫자, 언더스코어(_)만 사용 가능합니다.';
        UI.enrollError.classList.remove('hidden');
        UI.enrollSuccess.classList.add('hidden');
        return;
    }
    
    // FormData 생성
    const formData = new FormData();
    formData.append('person_id', personId);
    formData.append('name', name);
    formData.append('is_criminal', isCriminal);
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
        // 화면 전환: 용의자 선택 화면 → 대시보드 화면
        UI.screens.suspect.classList.add('hidden');
        UI.screens.dashboard.classList.remove('hidden');

        // 세션 ID 생성 (타임스탬프 기반)
        state.sessionId = `session_${Date.now()}`;
        console.log(`세션 ID: ${state.sessionId}`);

        // 스냅샷 배열 초기화
        state.snapshots = [];
        state.nextSnapshotId = 1;
        
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

// 패널 업데이트 헬퍼
function updateDetectionPanel(data, isAlert) {
    if (data.message) {
        UI.detectionInfo.innerHTML = `<p class="text-center py-4 text-gray-500">${data.message}</p>`;
        return;
    }

    const colorClass = isAlert ? "text-red-600 font-bold" : "text-green-600";
    const statusText = isAlert ? "🚨 용의자 감지!" : "✅ 일반인 확인";

    UI.detectionInfo.innerHTML = `
        <div class="p-4 bg-white border rounded shadow-sm">
            <div class="mb-2 ${colorClass}">${statusText}</div>
            <div>이름: ${data.name}</div>
            <div>신뢰도: ${data.confidence}%</div>
            <div class="text-xs text-gray-400 mt-2">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
}