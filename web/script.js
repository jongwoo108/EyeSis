// script.js

// ==========================================
// 전역 변수 및 상태 관리
// ==========================================
const API_BASE_URL = 'http://localhost:5000/api';
const WS_URL = 'ws://localhost:5000/ws/detect';

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
    frameId: 0, // 프레임 ID 추적
    useWebSocket: true, // WebSocket 사용 여부 (실패 시 HTTP로 폴백)
    lastDetections: null, // 마지막 감지 결과 (폴백용)
    lastDetectionTime: 0 // 마지막 감지 시간
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
    selectedSuspectInfo: document.getElementById('selectedSuspectInfo')
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
    
    // 체크박스 아이콘 추가
    card.innerHTML = `
        <div class="absolute top-2 right-2 w-6 h-6 rounded-full border-2 border-gray-300 bg-white flex items-center justify-center checkmark hidden">
            <svg class="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path>
            </svg>
        </div>
        <div class="aspect-w-3 aspect-h-4 ${bgColor} flex items-center justify-center p-8">
            <span class="text-6xl">👤</span>
        </div>
        <div class="p-4">
            <h3 class="font-bold text-lg">${displayName}</h3>
            <p class="text-sm ${textColor}">${statusText}</p>
        </div>
    `;
    
    // 클릭 이벤트 리스너 추가 (다중 선택)
    card.addEventListener('click', function() {
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
            scale: `(${displayWidth/videoWidth}, ${displayHeight/videoHeight})`
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
    
    if (UI.video.paused || UI.video.ended) {
        console.log("⚠️ 비디오가 일시정지되었거나 종료되었습니다");
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

function connectWebSocket() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        return; // 이미 연결됨
    }
    
    try {
        const ws = new WebSocket(WS_URL);
        
        ws.onopen = () => {
            console.log("✅ WebSocket 연결됨");
            state.isWsConnected = true;
            state.wsReconnectAttempts = 0;
            state.useWebSocket = true;
            
            // 연결 시 선택된 모든 suspect_ids 전송
            if (state.selectedSuspects.length > 0) {
                const suspectIds = state.selectedSuspects.map(s => s.id);
                sendWebSocketConfig(suspectIds);
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
            console.error("❌ WebSocket 오류:", error);
            state.isWsConnected = false;
        };
        
        ws.onclose = () => {
            console.log("⚠️ WebSocket 연결 종료됨");
            state.isWsConnected = false;
            state.ws = null;
            
            // 자동 재연결 시도
            if (state.isDetectionActive) {
                scheduleReconnect();
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
    if (state.wsReconnectTimer) {
        clearTimeout(state.wsReconnectTimer);
        state.wsReconnectTimer = null;
    }
}

function scheduleReconnect() {
    if (state.wsReconnectTimer) {
        return; // 이미 재연결 예약됨
    }
    
    const delay = Math.min(1000 * Math.pow(2, state.wsReconnectAttempts), 30000); // 최대 30초
    state.wsReconnectAttempts++;
    
    console.log(`🔄 ${delay/1000}초 후 WebSocket 재연결 시도 (${state.wsReconnectAttempts}회)`);
    
    state.wsReconnectTimer = setTimeout(() => {
        state.wsReconnectTimer = null;
        if (state.isDetectionActive && !state.isWsConnected) {
            connectWebSocket();
        }
    }, delay);
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
        state.frameId++;
        state.ws.send(JSON.stringify({
            type: "frame",
            data: {
                image: frameData,
                suspect_ids: suspectIds, // 배열로 전송
                frame_id: state.frameId
            }
        }));
        return true;
    }
    return false;
}

function handleWebSocketMessage(message) {
    const msgType = message.type;
    
    if (msgType === "detection") {
        const data = message.data;
        state.lastDetections = data.detections;
        state.lastDetectionTime = Date.now();
        
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
        console.log("✅ 설정 업데이트됨:", message.suspect_id);
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

    const frameData = captureVideoFrame();
    if (!frameData) {
        console.log("⚠️ 프레임 캡처 실패: 비디오가 재생 중이 아닙니다");
        return;
    }

    // 2. 처리 시작 (문 잠금)
    state.isProcessing = true;

    const suspectIds = state.selectedSuspects.length > 0 
        ? state.selectedSuspects.map(s => s.id) 
        : null;
    
    // WebSocket 사용 시도
    if (state.useWebSocket && state.isWsConnected) {
        const sent = sendWebSocketFrame(frameData, suspectIds);
        if (sent) {
            // WebSocket으로 전송 성공, 응답은 handleWebSocketMessage에서 처리
            return;
        } else {
            // WebSocket 전송 실패, HTTP로 폴백
            console.warn("⚠️ WebSocket 전송 실패, HTTP로 폴백");
            state.useWebSocket = false;
        }
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

UI.proceedBtn.addEventListener('click', () => {
    if (state.selectedSuspects.length > 0) {
        // 화면 전환: 용의자 선택 화면 → 대시보드 화면
        UI.screens.suspect.classList.add('hidden');
        UI.screens.dashboard.classList.remove('hidden');
        
        // [여기가 핵심!] 
        // 사용자가 업로드한 videoFile(mp4, mov)을 브라우저가 읽을 수 있는 URL로 변환
        const videoURL = URL.createObjectURL(state.selectedFile);
        
        // HTML의 <video> 태그에 주입
        UI.video.src = videoURL;
        
        // 동영상 재생 시작
        UI.video.play(); 
        
        initCaptureCanvas();
        
        // WebSocket 연결 준비 (감지 시작 전에 미리 연결)
        if (state.useWebSocket) {
            connectWebSocket();
        }
    }
});

UI.detectionFilter.addEventListener('change', (e) => {
    state.isDetectionActive = e.target.checked;
    
    if (state.isDetectionActive) {
        // 감지 시작
        console.log("🚀 AI 감지 시작");
        updateDetectionPanel({ message: "AI 분석 시작..." });
        
        // WebSocket 연결 시도
        if (state.useWebSocket) {
            connectWebSocket();
        }
        
        // 비디오가 로드되었는지 확인
        if (UI.video.readyState < 2) {
            console.warn("⚠️ 비디오가 아직 로드되지 않았습니다. 비디오 로드 대기 중...");
            UI.video.addEventListener('loadeddata', () => {
                console.log("✅ 비디오 로드 완료, 감지 시작");
                // 더 빠른 주기로 업데이트하여 부드럽게 (100ms = 10fps)
                state.detectionInterval = setInterval(processRealtimeDetection, 100);
            }, { once: true });
        } else {
            // 더 빠른 주기로 업데이트하여 부드럽게 (100ms = 10fps)
            state.detectionInterval = setInterval(processRealtimeDetection, 100);
        }
    } else {
        // 감지 종료
        console.log("⏹️ AI 감지 중지");
        clearInterval(state.detectionInterval);
        disconnectWebSocket();
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