// script.js - ES Module
// ==========================================
// 모듈 임포트
// ==========================================
import { API_BASE_URL, WS_URL, WS_TEST_URL, personNameMapping } from './modules/config.js';
import { state } from './modules/state.js';
import { initUI } from './modules/ui.js';
import { getCategoryStyle, getCategoryText, getAngleDisplayText, formatTime } from './modules/utils.js';
import { loadPersons, checkServerHealth } from './modules/api.js';
import {
    createTimelineTrack,
    initializeTimelinesForSelectedPersons,
    mergeTimelineEvents,
    renderTimelineWithMerging,
    addTimelineMarkerDirect,
    updateSnapshotCountDirect,
    updateClipCount
} from './modules/timeline.js';
import {
    createSuspectCard,
    updateSelectedSuspectsInfo,
    selectAllPersons,
    deselectAllPersons,
    updateSelectedPersonCount,
    deleteSelectedPersons,
    openEditPersonModal,
    closeEditPersonModal,
    updatePerson,
    renderSuspectCards
} from './modules/persons.js';

import {
    downloadVideoClip,
    getClipItemHTML,
    filterClipsByPerson,
    toggleClipSelection,
    updateSelectedClipCount
} from './modules/clips.js';

import {
    renderSnapshotCard,
    filterSnapshotsByPerson,
    toggleSnapshotSelection,
    updateSelectedCount,
    updateSnapshotCheckboxes
} from './modules/snapshots.js';

import {
    addDetectionLogItem,
    updateDetectionPanel,
    downloadLogToCSV
} from './modules/log.js';


import {
    drawDetections,
    captureVideoFrame
} from './modules/detection.js';

import {
    updatePersonCategory,
    checkFormValidity,
    closeEnrollModal
} from './modules/enroll.js';

import {
    handleViewSnapshots,
    handleCloseClipModal,
    handleCloseSnapshotModal,
    handleModalOutsideClick,
    handleSelectAllSnapshots,
    handleDeselectAllSnapshots,
    handleDownloadSelectedSnapshots,
    handleDownloadAllSnapshots,
    handleOpenAddSuspectModal,
    handleAddSuspectModalOutsideClick,
    handleOpenEmergencyModal,
    handleCloseEmergencyModal,
    handleEmergencyModalOutsideClick,
    handleEscapeKey,
    handleImagePreview
} from './modules/handlers.js';

// UI 초기화 (DOM 로드 후 실행됨 - type="module"은 자동으로 defer)
const UI = initUI();

// ==========================================
// 이벤트 리스너 등록 (스냅샷/클립 모달)
// ==========================================
UI.viewSnapshotsBtn?.addEventListener('click', handleViewSnapshots);
document.getElementById('closeClipModalBtn')?.addEventListener('click', handleCloseClipModal);
document.getElementById('closeClipModalBtn2')?.addEventListener('click', handleCloseClipModal);
document.getElementById('closeModalBtn')?.addEventListener('click', handleCloseSnapshotModal);
document.getElementById('closeModalBtn2')?.addEventListener('click', handleCloseSnapshotModal);
document.getElementById('clipModal')?.addEventListener('click', handleModalOutsideClick);
document.getElementById('snapshotModal')?.addEventListener('click', handleModalOutsideClick);
document.getElementById('selectAllBtn')?.addEventListener('click', handleSelectAllSnapshots);
document.getElementById('deselectAllBtn')?.addEventListener('click', handleDeselectAllSnapshots);
document.getElementById('downloadSelectedBtn')?.addEventListener('click', handleDownloadSelectedSnapshots);
document.getElementById('downloadAllBtn')?.addEventListener('click', handleDownloadAllSnapshots);

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

    // 비디오 재생 시 AI 감지 자동 활성화 (일시정지 후 재생도 포함)
    UI.video.addEventListener('play', () => {
        // 처리 상태 초기화 (일시정지 후 재개 시 필수)
        state.isProcessing = false;

        // 비디오 메타데이터가 로드되지 않았으면 기다림
        if (UI.video.videoWidth === 0 || UI.video.videoHeight === 0) {
            console.log("⏳ 비디오 메타데이터 로드 대기 중...");
            const onLoadedMetadata = () => {
                UI.video.removeEventListener('loadedmetadata', onLoadedMetadata);
                startDetectionAfterPlay();
            };
            UI.video.addEventListener('loadedmetadata', onLoadedMetadata, { once: true });
            return;
        }

        startDetectionAfterPlay();
    });

    // 감지 시작 로직을 별도 함수로 분리
    function startDetectionAfterPlay() {
        if (!state.isDetectionActive && UI.detectionFilter) {
            // 최초 재생: AI 감지 자동 활성화
            console.log("▶️ 비디오 재생 감지, AI 감지 자동 활성화");
            UI.detectionFilter.checked = true;
            state.isDetectionActive = true;

            // WebSocket 연결 시도 (백그라운드에서 연결 시도)
            if (state.useWebSocket && !state.isWsConnected) {
                connectWebSocket();
            }
        }

        // 일시정지 후 재생 또는 최초 재생 모두 감지 루프 시작
        if (state.isDetectionActive) {
            console.log("🚀 비디오 재생됨, 감지 루프 시작");
            processRealtimeDetection();
        }
    }

    // 비디오 종료 시 감지 루프 자동 중지
    UI.video.addEventListener('ended', () => {
        if (state.isDetectionActive) {
            console.log("⏹️ 비디오 종료됨, 감지 루프 자동 중지");
            state.isDetectionActive = false;
            if (UI.detectionFilter) {
                UI.detectionFilter.checked = false;
            }
            // clearInterval(state.detectionInterval); // 제거됨

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

            // 비디오 종료 시 발견 보고 전송 버튼 활성화
            if (UI.dispatchReportBtn) {
                UI.dispatchReportBtn.disabled = false;
                UI.dispatchReportBtn.classList.remove('bg-gray-400', 'cursor-not-allowed');
                UI.dispatchReportBtn.classList.add('bg-red-600', 'hover:bg-red-700');
                console.log("✅ 비디오 분석 완료: 발견 보고 전송 버튼 활성화");
            }
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

        // 비디오 현재 시간 가져오기
        const videoTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : 0;

        state.frameId++;
        state.ws.send(JSON.stringify({
            type: "frame",
            data: {
                image: frameData,
                suspect_ids: ids, // 항상 포함 (빈 배열이어도)
                frame_id: state.frameId,
                video_time: videoTime  // 비디오 시간 추가
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
                selectedPersonIds.includes(d.person_id || d.metadata?.person_id || d.name)
            );
        }

        // 디버깅 로그 (상세)
        console.log('🔍 [WS] 감지 결과 확인:', {
            alert: data.alert,
            detectionsCount: data.detections ? data.detections.length : 0,
            selectedPersonsCount: detectedSelectedPersons.length,
            selectedPersonIds: state.selectedSuspects.map(s => s.id),
            detectedPersonIds: data.detections ? data.detections.map(d => ({
                person_id: d.person_id,
                metadata_person_id: d.metadata?.person_id,
                name: d.name,
                matched_id: d.person_id || d.metadata?.person_id || d.name
            })) : [],
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
                const personId = personData.person_id || personData.metadata?.person_id || personData.name || 'Unknown';
                const personName = personData.name || personData.metadata?.name || 'Unknown';
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
                        similarity: personData.confidence || personData.metadata?.confidence || 0,
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
                updateDetectionPanel(metadata, false, videoTime);
            } else {
                updateDetectionPanel(null, false, videoTime);
            }
        }

        // 3. 박스 렌더링 (즉시 렌더링)
        if (data.detections && data.detections.length > 0 && UI.video.videoWidth > 0) {
            drawDetections(data.detections, UI.video.videoWidth, UI.video.videoHeight);
        } else {
            if (state.detectionCtx) {
                state.detectionCtx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);
            }
        }

        state.isProcessing = false;

        // [핵심] 응답 받자마자 쉴 틈 없이 다음 프레임 전송 (재귀 호출)
        if (state.isDetectionActive) {
            // requestAnimationFrame을 사용하여 브라우저 렌더링 사이클에 맞춰 다음 요청 (과부하 방지 겸용)
            requestAnimationFrame(processRealtimeDetection);
        }

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

            // 정확한 비디오 타임스탬프 사용 (먼저 선언)
            const videoTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : (result.video_timestamp || 0);

            // 1. 선택된 인물 필터링
            let detectedSelectedPersons = [];
            if (result.detections && result.detections.length > 0) {
                const selectedPersonIds = state.selectedSuspects.map(s => s.id);
                detectedSelectedPersons = result.detections.filter(d =>
                    selectedPersonIds.includes(d.person_id || d.metadata?.person_id || d.name)
                );
            }

            // 디버깅 로그 (상세)
            console.log('🔍 [HTTP] 감지 결과 확인:', {
                alert: result.alert,
                detectionsCount: result.detections ? result.detections.length : 0,
                selectedPersonsCount: detectedSelectedPersons.length,
                selectedPersonIds: state.selectedSuspects.map(s => s.id),
                detectedPersonIds: result.detections ? result.detections.map(d => ({
                    person_id: d.person_id,
                    metadata_person_id: d.metadata?.person_id,
                    name: d.name,
                    matched_id: d.person_id || d.metadata?.person_id || d.name
                })) : [],
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
                    const personId = personData.person_id || personData.metadata?.person_id || personData.name || 'Unknown';
                    const personName = personData.name || personData.metadata?.name || 'Unknown';
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
                    const personId = firstPerson.person_id || firstPerson.metadata?.person_id || firstPerson.name || 'Unknown';
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
                    updateDetectionPanel(metadata, false, videoTime);
                } else {
                    updateDetectionPanel(null, false, videoTime);
                }
            }
            // 박스 렌더링 (즉시 렌더링)
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
                const videoTime = UI.video && !isNaN(UI.video.currentTime) ? UI.video.currentTime : 0;
                updateDetectionPanel(result.metadata, false, videoTime);
            }
        }

        // 처리 완료
        state.isProcessing = false;

        // [핵심] 재귀 호출 (Max FPS)
        if (state.isDetectionActive) {
            requestAnimationFrame(processRealtimeDetection);
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

        // processRealtimeDetection에서의 재귀 호출 제거 (각 핸들러에서 처리)
    } catch (err) {
        console.error("❌ 처리 중 에러:", err);
        state.isProcessing = false;

        // 에러 발생 시에도 재귀 호출 (약간의 딜레이)
        if (state.isDetectionActive) {
            setTimeout(processRealtimeDetection, 100);
        }
    }
}

// ==========================================
// UI 이벤트 핸들러
// ==========================================

// 파일 선택 처리
UI.videoFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // 파일 검증
    const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm'];
    const validExtensions = ['.mp4', '.mov', '.avi', '.webm'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
        alert('지원하지 않는 파일 형식입니다. MP4, AVI, MOV 형식만 지원됩니다.');
        UI.videoFile.value = ''; // input 초기화
        return;
    }

    state.selectedFile = file;
    console.log(`✅ 파일 선택됨: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);

    // 파일이 선택되면 즉시 비디오 로드
    await handleVideoFileSelection(file);
});

// 비디오 파일 선택 처리 함수
async function handleVideoFileSelection(file) {
    // 비디오 로드
    const videoURL = URL.createObjectURL(file);
    UI.video.src = videoURL;

    // 영상 업로드 상태 업데이트
    state.videoUploaded = true;
    updateDashboardView();

    // 빈 상태 카드 숨기기
    if (UI.emptyStateCard) {
        UI.emptyStateCard.classList.add('hidden');
    }

    // 비디오 메타데이터 로드 대기 (첫 프레임 표시를 위해)
    UI.video.addEventListener('loadedmetadata', () => {
        // 첫 프레임을 표시하기 위해 currentTime을 0으로 설정 (이미 기본값이지만 명시적으로 설정)
        UI.video.currentTime = 0;
        // 자동 재생하지 않음 - 사용자가 직접 재생 버튼을 눌러야 함
        console.log("✅ 비디오 로드 완료 (일시 정지 상태)");
    }, { once: true });

    // 비디오 로드 에러 처리
    UI.video.addEventListener('error', (e) => {
        console.error("❌ 비디오 로드 또는 재생 중 오류 발생:", e);
        alert("비디오를 로드하거나 재생할 수 없습니다. 파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.");
        if (UI.emptyStateCard) {
            UI.emptyStateCard.classList.remove('hidden'); // 에러 시 빈 상태 표시
        }
    }, { once: true });

    console.log("✅ 영상 파일 로드 완료:", file.name);
}

// ==========================================
// 이벤트 리스너 등록 (용의자 추가/긴급 신고)
// ==========================================
UI.addSuspectBtn?.addEventListener('click', handleOpenAddSuspectModal);
UI.addSuspectModal?.addEventListener('click', handleAddSuspectModalOutsideClick);
UI.closeAddSuspectModal?.addEventListener('click', closeEnrollModal);
UI.cancelEnrollBtn?.addEventListener('click', closeEnrollModal);
document.addEventListener('keydown', handleEscapeKey);

UI.emergencyCallBtn?.addEventListener('click', handleOpenEmergencyModal);
UI.cancelEmergencyCallBtn?.addEventListener('click', handleCloseEmergencyModal);
UI.emergencyCallModal?.addEventListener('click', handleEmergencyModalOutsideClick);

// 이벤트 리스너 등록 (폼 입력)
UI.enrollImage?.addEventListener('change', handleImagePreview);
UI.personCategoryCustom?.addEventListener('input', checkFormValidity);
UI.enrollName?.addEventListener('input', checkFormValidity);

// 폼 제출
UI.addSuspectForm?.addEventListener('submit', async (e) => {
    e.preventDefault();

    // 입력값 가져오기
    const name = UI.enrollName.value.trim();
    const imageFile = UI.enrollImage.files[0];

    // 인물 타입 가져오기 (드롭다운 또는 직접 입력 값)
    const categorySelect = UI.personCategory;
    const customInput = UI.personCategoryCustom;
    let personType = 'criminal';

    if (categorySelect) {
        if (categorySelect.value === 'custom') {
            // '기타' 선택 시 직접 입력 값 확인
            if (!customInput || !customInput.value.trim()) {
                UI.enrollError.textContent = '카테고리를 직접 입력해주세요.';
                UI.enrollError.classList.remove('hidden');
                return;
            }
            personType = customInput.value.trim();
        } else {
            // 일반 옵션 선택 시
            personType = categorySelect.value;
        }
    }

    // personTypeInput에도 저장
    const personTypeInput = document.getElementById('personTypeInput');
    if (personTypeInput) {
        personTypeInput.value = personType;
    }

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
    formData.append('person_type', personType);  // criminal, missing, dementia, child, wanted, 또는 사용자 입력 값
    formData.append('image', imageFile);

    // 디버깅: 전송 데이터 확인
    console.log('📤 [ENROLL] 등록 요청 데이터:', {
        person_id: personId,
        name: name,
        person_type: personType,
        image_file: imageFile.name,
        image_size: imageFile.size
    });

    // 버튼 비활성화
    UI.submitEnrollBtn.disabled = true;
    UI.submitEnrollBtn.textContent = '등록 중...';
    UI.enrollError.classList.add('hidden');
    UI.enrollSuccess.classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE_URL}/enroll`, {
            method: 'POST',
            body: formData,
            // ngrok 사용 시 필요한 헤더는 자동으로 추가됨
            headers: {
                // Content-Type은 FormData가 자동으로 설정하므로 명시하지 않음
            }
        });

        // 응답이 JSON이 아닐 수 있으므로 먼저 확인
        let data;
        try {
            const responseText = await response.text();
            console.log('📥 [ENROLL] 서버 응답:', response.status, responseText);

            if (responseText) {
                data = JSON.parse(responseText);
            } else {
                data = {};
            }
        } catch (jsonError) {
            // JSON 파싱 실패 시
            console.error('❌ [ENROLL] JSON 파싱 실패:', jsonError);
            UI.enrollError.textContent = `서버 응답 오류: ${response.status} ${response.statusText}`;
            UI.enrollError.classList.remove('hidden');
            UI.submitEnrollBtn.disabled = false;
            UI.submitEnrollBtn.textContent = '등록';
            return;
        }

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
            // 에러 메시지 표시 (400, 500 등)
            // FastAPI HTTPException은 detail 필드를 사용
            const errorMessage = data?.detail || data?.message || data?.error || `서버 오류 (${response.status})`;
            console.error('❌ [ENROLL] 등록 실패:', {
                status: response.status,
                statusText: response.statusText,
                data: data
            });
            UI.enrollError.textContent = errorMessage;
            UI.enrollError.classList.remove('hidden');
            UI.enrollSuccess.classList.add('hidden');
        }
    } catch (error) {
        console.error('등록 실패:', error);
        UI.enrollError.textContent = `등록 중 오류가 발생했습니다: ${error.message || '알 수 없는 오류'}`;
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
                // Max FPS 모드: 최초 1회 호출 후 재귀적으로 실행
                processRealtimeDetection();
            }, { once: true });
        } else {
            // WebSocket 연결 상태와 관계없이 HTTP로 즉시 시작 (WebSocket 준비되면 자동 전환)
            console.log("🚀 HTTP 모드로 감지 시작 (WebSocket 준비되면 자동 전환)");
            // Max FPS 모드: 최초 1회 호출 후 재귀적으로 실행
            processRealtimeDetection();
        }
    } else {
        // 감지 종료
        console.log("⏹️ AI 감지 중지");
        console.log("⏹️ AI 감지 중지");
        // clearInterval(state.detectionInterval); // 제거됨
        // state.animationFrameId = null; // 제거됨
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





// ==========================================
// CSV 내보내기 기능
// ==========================================



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

// 파일 선택 창 열기 (빈 상태 카드 클릭)
UI.emptyStateCard?.addEventListener('click', () => {
    if (UI.videoFile) {
        UI.videoFile.click();
    }
});

// 파일 선택 창 열기 (헤더 버튼 클릭)
UI.openUploadModalBtn?.addEventListener('click', () => {
    if (UI.videoFile) {
        UI.videoFile.click();
    }
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
UI.suspectSelectModal?.addEventListener('click', (e) => {
    if (e.target === UI.suspectSelectModal) {
        UI.suspectSelectModal.classList.add('hidden');
    }
});

// ESC 키로 모달 닫기
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (!UI.suspectSelectModal.classList.contains('hidden')) {
            UI.suspectSelectModal.classList.add('hidden');
        }
        if (!UI.addSuspectModal.classList.contains('hidden')) {
            UI.addSuspectModal.classList.add('hidden');
        }
    }
});



// 클립/스냅샷 버튼 이벤트
// ==========================================
// 현재 선택된 클립 필터 (전역 변수)
let currentClipFilter = '전체';

// 클립 보기 버튼 이벤트
UI.viewClipsBtn?.addEventListener('click', () => {
    console.log('📹 클립 보기 버튼 클릭');
    console.log(`현재 클립 개수: ${state.detectionClips.length}`);

    const modal = document.getElementById('clipModal');
    const list = document.getElementById('clipList');
    const tabsContainer = document.getElementById('clipTabs');

    if (!modal || !list) {
        console.error('클립 모달 요소를 찾을 수 없습니다.');
        return;
    }

    if (state.detectionClips.length === 0) {
        if (tabsContainer) tabsContainer.innerHTML = '';
        list.innerHTML = '<p class="text-center py-8 text-gray-500">아직 감지된 클립이 없습니다.</p>';
        currentClipFilter = '전체';
    } else {
        const formatTime = (seconds) => {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        };

        // 선택 상태 초기화 (모달 열 때마다)
        state.selectedClips = [];
        updateSelectedClipCount();

        // 인물별로 그룹화 (필터 탭용)
        const personGroups = {};
        state.detectionClips.forEach(clip => {
            const selectedPerson = state.selectedSuspects.find(s => s.id === clip.personId);
            const personName = selectedPerson ? selectedPerson.name : (clip.personName || 'Unknown');
            if (!personGroups[personName]) {
                personGroups[personName] = [];
            }
            personGroups[personName].push(clip);
        });

        // 탭 생성
        if (tabsContainer) {
            const personNames = Object.keys(personGroups).sort();
            tabsContainer.innerHTML = `
                <div class="flex flex-wrap gap-2 overflow-x-auto pb-2">
                    <button class="clip-tab active px-4 py-2 rounded-lg text-sm font-medium bg-indigo-600 text-white hover:bg-indigo-700 transition-all duration-200 shadow-sm" 
                            data-person="전체">
                        전체 (${state.detectionClips.length})
                    </button>
                    ${personNames.map(personName => `
                        <button class="clip-tab px-4 py-2 rounded-lg text-sm font-medium bg-gray-200 text-gray-700 hover:bg-gray-300 transition-all duration-200" 
                                data-person="${personName}">
                            ${personName} (${personGroups[personName].length})
                        </button>
                    `).join('')}
                </div>
            `;

            // 탭 클릭 이벤트 등록
            const tabs = tabsContainer.querySelectorAll('.clip-tab');
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
                    currentClipFilter = selectedPerson;
                    filterClipsByPerson(selectedPerson);
                });
            });
        }

        // 모든 클립 렌더링
        list.innerHTML = state.detectionClips.map(clip => {
            const videoUrl = state.selectedFile ? URL.createObjectURL(state.selectedFile) : '';
            const isSelected = state.selectedClips.includes(clip.id);

            // 렌더링 시 data-person-name 속성 추가 (필터링용)
            const selectedPerson = state.selectedSuspects.find(s => s.id === clip.personId);
            const personName = selectedPerson ? selectedPerson.name : (clip.personName || 'Unknown');

            // getClipItemHTML 함수가 data-person-name을 포함하도록 수정해야 함
            // 여기서는 HTML 문자열을 직접 조작하여 속성 추가
            const itemHTML = getClipItemHTML({
                ...clip,
                videoUrl: videoUrl,
                isSelected: isSelected
            });

            // data-person-name 속성 주입 (첫 번째 div 태그에)
            return itemHTML.replace('<div class="bg-white', `<div data-person-name="${personName}" class="bg-white`);
        }).join('');

        // 초기 필터 적용
        currentClipFilter = '전체';
        filterClipsByPerson('전체');
    }

    modal.classList.remove('hidden');
});



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
// 카테고리 변경 이벤트 리스너
if (UI.personCategory) {
    UI.personCategory.addEventListener('change', function () {
        const categorySelect = UI.personCategory;
        const customContainer = document.getElementById('customCategoryContainer');
        const customInput = UI.personCategoryCustom;

        if (categorySelect && customContainer && customInput) {
            if (categorySelect.value === 'custom') {
                // "기타" 선택 시 직접 입력 필드 표시
                customContainer.classList.remove('hidden');
                customInput.required = true;
            } else {
                // 다른 카테고리 선택 시 숨김
                customContainer.classList.add('hidden');
                customInput.required = false;
                customInput.value = '';
            }
        }

        updatePersonCategory();
    });
}

// 직접 입력 필드 이벤트 리스너 (입력 시 유효성 검사)
UI.personCategoryCustom?.addEventListener('input', () => {
    checkFormValidity();
});

// ==========================================
// 전체 선택/해제/삭제 버튼 이벤트 리스너
// ==========================================
if (UI.selectAllPersonsBtn) {
    UI.selectAllPersonsBtn.addEventListener('click', selectAllPersons);
}

if (UI.deselectAllPersonsBtn) {
    UI.deselectAllPersonsBtn.addEventListener('click', deselectAllPersons);
}

if (UI.deleteSelectedPersonsBtn) {
    UI.deleteSelectedPersonsBtn.addEventListener('click', deleteSelectedPersons);
}

// ==========================================
// 인물 수정 모달 이벤트 리스너
// ==========================================
const closeEditPersonModalBtn = document.getElementById('closeEditPersonModal');
const cancelEditPersonBtn = document.getElementById('cancelEditPersonBtn');
const editPersonForm = document.getElementById('editPersonForm');

if (closeEditPersonModalBtn) {
    closeEditPersonModalBtn.addEventListener('click', closeEditPersonModal);
}

if (cancelEditPersonBtn) {
    cancelEditPersonBtn.addEventListener('click', closeEditPersonModal);
}

if (editPersonForm) {
    editPersonForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        const personId = document.getElementById('editPersonId').value;
        const name = document.getElementById('editPersonName').value;
        let personType = document.getElementById('editPersonCategory').value;

        // 'custom'인 경우 직접 입력한 값 사용
        if (personType === 'custom') {
            const customInput = document.getElementById('editPersonCategoryCustom');
            if (customInput && customInput.value.trim()) {
                personType = customInput.value.trim();
            } else {
                alert('카테고리를 입력해주세요.');
                return;
            }
        }

        await updatePerson(personId, name, personType);
    });
}

// 수정 모달의 카테고리 변경 이벤트 리스너
const editCategorySelect = document.getElementById('editPersonCategory');
const editCustomContainer = document.getElementById('editCustomCategoryContainer');
const editCustomInput = document.getElementById('editPersonCategoryCustom');

if (editCategorySelect && editCustomContainer && editCustomInput) {
    editCategorySelect.addEventListener('change', function () {
        if (this.value === 'custom') {
            editCustomContainer.classList.remove('hidden');
            editCustomInput.required = true;
        } else {
            editCustomContainer.classList.add('hidden');
            editCustomInput.required = false;
            editCustomInput.value = '';
        }
    });
}



console.log("✅ EyeSis 프론트엔드 초기화 완료");

// ==========================================
// 긴급 상황 전파 리포트 로직
// ==========================================
const dispatchReportBtn = document.getElementById('dispatchReportBtn');
const dispatchReportModal = document.getElementById('dispatchReportModal');
const closeDispatchModalBtn = document.getElementById('closeDispatchModalBtn');
const cancelDispatchBtn = document.getElementById('cancelDispatchBtn');
const sendDispatchBtn = document.getElementById('sendDispatchReportBtn');

if (dispatchReportBtn) {
    dispatchReportBtn.addEventListener('click', () => {
        // 1. 선택된 인물 확인
        if (state.selectedSuspects.length === 0) {
            alert('먼저 인물을 선택해주세요.');
            return;
        }

        // 2. 모든 선택된 인물에 대해 리포트 카드 생성
        const reportListEl = document.getElementById('dispatchReportList');
        if (!reportListEl) return;
        reportListEl.innerHTML = ''; // 초기화

        // 시간 포맷팅 함수
        const formatTime = (seconds) => {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        };

        state.selectedSuspects.forEach(suspect => {
            // 해당 인물의 스냅샷 필터링
            const targetSnapshots = state.snapshots.filter(snap => snap.personId === suspect.id);

            let bestSnapshot = null;
            let maxConfidence = 0;
            let timeText = '감지되지 않음';
            let timelineHTML = '';
            let snapshotImgSrc = '';
            let isDetected = false;

            if (targetSnapshots.length > 0) {
                isDetected = true;
                // 정렬: 정확도(similarity) 내림차순 -> 시간(videoTime) 오름차순
                targetSnapshots.sort((a, b) => {
                    const confA = parseFloat(a.similarity || 0);
                    const confB = parseFloat(b.similarity || 0);
                    if (confA !== confB) {
                        return confB - confA; // 정확도 높은 순
                    }
                    return a.videoTime - b.videoTime; // 시간 빠른 순
                });

                bestSnapshot = targetSnapshots[0];
                maxConfidence = parseFloat(bestSnapshot.similarity || 0);
                timeText = `영상 ${formatTime(bestSnapshot.videoTime)} 지점`;
                snapshotImgSrc = bestSnapshot.base64Image;

                // 고신뢰도 시점 리스트 (90% 이상)
                const highConfSnaps = targetSnapshots.filter(snap => parseFloat(snap.similarity || 0) >= 90);
                if (highConfSnaps.length > 0) {
                    highConfSnaps.sort((a, b) => a.videoTime - b.videoTime);
                    const uniqueTimes = [...new Set(highConfSnaps.map(snap => formatTime(snap.videoTime)))];
                    const displayTimes = uniqueTimes.slice(0, 5);
                    let timelineText = displayTimes.join(', ');
                    if (uniqueTimes.length > 5) timelineText += ', ...';
                    timelineHTML = `<p class="text-xs text-gray-500 mt-1">주요 감지 시점(90%↑): ${timelineText}</p>`;
                }
            }

            // 카테고리 정보 (Universal Category)
            let personData = suspect.person;
            if (!personData && state.personDatabase) {
                personData = state.personDatabase.find(p => p.id === suspect.id);
            }
            const categoryText = getCategoryText(personData);
            const categoryStyle = getCategoryStyle(categoryText);

            // 위험도 설정
            let riskLevel = 'Medium';
            let riskClass = 'bg-orange-100 text-orange-700';
            if (categoryText.includes('범죄') || categoryText.includes('수배') || categoryText.includes('살인') || categoryText.includes('강도')) {
                riskLevel = 'High';
                riskClass = 'bg-red-100 text-red-700';
            }

            // DB 이미지 URL
            const dbImgUrl = personData ? personData.image_url : null;

            // HTML 생성
            const cardHTML = `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 pt-6 first:pt-0">
                    <!-- 인물 및 감지 정보 (좌측) -->
                    <div class="space-y-4">
                        <h3 class="text-lg font-semibold text-gray-800 tracking-tight border-b pb-2">인물 및 감지 정보</h3>
                        <div class="bg-gray-50 rounded-lg p-4 space-y-3">
                            <div>
                                <label class="text-xs font-medium text-gray-500 tracking-tight">대상</label>
                                <div class="mt-1">
                                    <p class="text-lg font-bold text-gray-800">${suspect.name}</p>
                                    <div class="flex items-center gap-2 mt-1">
                                        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${categoryStyle.bgColor} ${categoryStyle.textColor}">${categoryText}</span>
                                        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${riskClass}">${riskLevel}</span>
                                    </div>
                                </div>
                            </div>
                            <div class="border-t pt-3 space-y-2">
                                <div>
                                    <label class="text-xs font-medium text-gray-500 tracking-tight">장소</label>
                                    <p class="text-sm font-medium text-gray-800 mt-1">A편의점 (정문)</p>
                                </div>
                                <div>
                                    <label class="text-xs font-medium text-gray-500 tracking-tight">감지 시간</label>
                                    <p class="text-sm font-medium text-gray-800 mt-1">${timeText}</p>
                                    ${timelineHTML}
                                </div>
                                <div>
                                    <label class="text-xs font-medium text-gray-500 tracking-tight">일치율</label>
                                    <p class="text-sm font-medium text-gray-800 mt-1">${isDetected ? maxConfidence.toFixed(1) + '%' : '-'}</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- 시각 증거 비교 (우측) -->
                    <div class="space-y-4">
                        <h3 class="text-lg font-semibold text-gray-800 tracking-tight border-b pb-2">시각 증거 비교</h3>
                        <div class="grid grid-cols-2 gap-4">
                            <div class="space-y-2">
                                <label class="text-xs font-medium text-gray-500 tracking-tight block">DB 사진</label>
                                <div class="bg-gray-100 rounded-lg p-4 flex items-center justify-center h-48 border-2 border-gray-200">
                                    ${dbImgUrl ? `<img src="${dbImgUrl}" class="max-w-full max-h-full object-contain rounded-lg">` : '<span class="text-gray-400 text-sm">이미지 없음</span>'}
                                </div>
                            </div>
                            <div class="space-y-2">
                                <label class="text-xs font-medium text-gray-500 tracking-tight block">CCTV 스냅샷</label>
                                <div class="bg-black rounded-lg overflow-hidden flex items-center justify-center aspect-video border-2 border-gray-200 relative group">
                                    ${isDetected && snapshotImgSrc ?
                    `<img src="${snapshotImgSrc}" class="w-full h-full object-contain cursor-pointer hover:opacity-90 transition-opacity" onclick="window.open(this.src)">
                                         <div class="absolute inset-0 flex items-center justify-center pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
                                             <span class="bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">클릭하여 확대</span>
                                         </div>`
                    : '<span class="text-gray-500 text-sm">감지되지 않음</span>'}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            reportListEl.insertAdjacentHTML('beforeend', cardHTML);
        });

        // 모달 열기
        dispatchReportModal.classList.remove('hidden');
    });
}

// 모달 닫기 이벤트
if (closeDispatchModalBtn) {
    closeDispatchModalBtn.addEventListener('click', () => {
        dispatchReportModal.classList.add('hidden');
    });
}
if (cancelDispatchBtn) {
    cancelDispatchBtn.addEventListener('click', () => {
        dispatchReportModal.classList.add('hidden');
    });
}

// 전송 버튼 (Mock)
if (sendDispatchBtn) {
    sendDispatchBtn.addEventListener('click', () => {
        alert('경찰서로 리포트가 전송되었습니다.');
        dispatchReportModal.classList.add('hidden');
    });
}