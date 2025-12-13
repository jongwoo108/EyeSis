
// ==========================================
// 타임라인 및 스냅샷 갤러리 기능
// ==========================================

// 타임라인 마커 추가
function addTimelineMarker(snapshot) {
    if (!UI.video.duration) return;

    const timelineBar = document.getElementById('timelineBar');
    const position = (snapshot.videoTime / UI.video.duration) * 100;

    const marker = document.createElement('div');
    marker.className = 'absolute w-3 h-full bg-red-500 cursor-pointer hover:bg-red-700 transition-colors';
    marker.style.left = `${position}%`;
    marker.title = `${snapshot.personName} - ${formatTime(snapshot.videoTime)}`;
    marker.dataset.snapshotId = snapshot.id;

    marker.addEventListener('click', (e) => {
        e.stopPropagation();
        UI.video.currentTime = snapshot.videoTime;
        UI.video.play();
    });

    timelineBar.appendChild(marker);
}

// 타임라인 클릭으로 이동
const timelineContainer = document.getElementById('timelineContainer');
if (timelineContainer) {
    timelineContainer.addEventListener('click', (e) => {
        if (!UI.video || !UI.video.duration) return;

        const rect = e.currentTarget.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const percentage = clickX / rect.width;
        UI.video.currentTime = percentage * UI.video.duration;
    });
}

// 스냅샷 개수 업데이트
function updateSnapshotCount() {
    const countEl = document.getElementById('snapshotCount');
    if (countEl) {
        countEl.textContent = state.snapshots.length;
    }
}

// 스냅샷 갤러리 모달 열기 - script.js에서 처리됨 (중복 제거)

// 모달 닫기 - script.js에서 처리됨 (중복 제거)

// 시간 포맷 헬퍼
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// 개별 스냅샷 다운로드
function downloadSnapshot(snapshotId) {
    const snapshot = state.snapshots.find(s => s.id === snapshotId);
    if (!snapshot) return;

    const link = document.createElement('a');
    link.href = snapshot.base64Image;
    link.download = `criminal_${snapshot.personName}_${formatTime(snapshot.videoTime).replace(':', '-')}.jpg`;
    link.click();
}

// 전체 다운로드 - script.js에서 처리됨 (중복 제거)

// 클립 개수 업데이트
function updateClipCount() {
    const countEl = document.getElementById('clipCount');
    if (countEl) {
        countEl.textContent = state.detectionClips.length;
    }
}

// 클립 갤러리 모달 열기 - script.js에서 처리됨 (중복 제거)

// 모달 닫기 - script.js에서 처리됨 (중복 제거)

// 클립으로 이동 - script.js의 window.seekToClip 사용 (중복 제거)
// 클립 다운로드 - script.js의 window.downloadClip 사용 (중복 제거)

// WebSocket 메시지 핸들러는 script.js에서 직접 처리하도록 변경됨
// 이 파일의 함수들은 script.js에서 호출됨
