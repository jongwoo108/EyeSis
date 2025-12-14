// modules/state.js
export const state = {
    selectedFile: null,
    videoUploaded: false, // 영상 업로드 여부
    selectedSuspects: [], // 여러 명 선택 가능 (배열)
    personDatabase: [], // 전체 인물 DB (Fallback용)
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