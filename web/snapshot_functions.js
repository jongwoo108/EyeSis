
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
document.getElementById('timelineContainer').addEventListener('click', (e) => {
    if (!UI.video.duration) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    UI.video.currentTime = percentage * UI.video.duration;
});

// 스냅샷 개수 업데이트
function updateSnapshotCount() {
    const countEl = document.getElementById('snapshotCount');
    if (countEl) {
        countEl.textContent = state.snapshots.length;
    }
}

// 스냅샷 갤러리 모달 열기
document.getElementById('viewSnapshotsBtn').addEventListener('click', () => {
    const modal = document.getElementById('snapshotModal');
    const grid = document.getElementById('snapshotGrid');

    if (state.snapshots.length === 0) {
        grid.innerHTML = '<p class="col-span-full text-center py-8 text-gray-500">아직 캡처된 스냅샷이 없습니다.</p>';
    } else {
        grid.innerHTML = state.snapshots.map(snapshot => `
            <div class="bg-white rounded-lg shadow-lg overflow-hidden">
                <img src="${snapshot.base64Image}" alt="${snapshot.personName}" class="w-full h-48 object-cover cursor-pointer" 
                     onclick="window.open(this.src)">
                <div class="p-3">
                    <div class="font-bold text-sm text-gray-800">${snapshot.personName}</div>
                    <div class="text-xs text-gray-600 mt-1">시간: ${formatTime(snapshot.videoTime)}</div>
                    <div  class="text-xs text-gray-600">유사도: ${snapshot.similarity}%</div>
                    <div class="text-xs text-gray-500">${new Date(snapshot.timestamp).toLocaleString()}</div>
                    <button onclick="downloadSnapshot(${snapshot.id})" 
                            class="mt-2 w-full bg-blue-500 text-white px-3 py-1 rounded text-xs hover:bg-blue-600">
                        이 스냅샷 저장
                    </button>
                </div>
            </div>
        `).join('');
    }

    modal.classList.remove('hidden');
});

// 모달 닫기
document.getElementById('closeModalBtn').addEventListener('click', () => {
    document.getElementById('snapshotModal').classList.add('hidden');
});
document.getElementById('closeModalBtn2').addEventListener('click', () => {
    document.getElementById('snapshotModal').classList.add('hidden');
});

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

// 전체 다운로드 (순차적으로)
document.getElementById('downloadAllBtn').addEventListener('click', async () => {
    if (state.snapshots.length === 0) {
        alert('다운로드할 스냅샷이 없습니다.');
        return;
    }

    // 간단한 방법: 순차적으로 다운로드
    for (let i = 0; i < state.snapshots.length; i++) {
        const snapshot = state.snapshots[i];
        const link = document.createElement('a');
        link.href = snapshot.base64Image;
        link.download = `${i + 1}_criminal_${snapshot.personName}_${formatTime(snapshot.videoTime).replace(':', '-')}.jpg`;
        link.click();

        // 브라우저가 따라잡을 시간 주기
        await new Promise(resolve => setTimeout(resolve, 300));
    }

    alert(`${state.snapshots.length}개의 스냅샷 다운로드를 시작했습니다.`);
});

// 클립 개수 업데이트
function updateClipCount() {
    const countEl = document.getElementById('clipCount');
    if (countEl) {
        countEl.textContent = state.detectionClips.length;
    }
}

// 클립 갤러리 모달 열기
document.getElementById('viewClipsBtn').addEventListener('click', () => {
    const modal = document.getElementById('clipModal');
    const list = document.getElementById('clipList');

    if (state.detectionClips.length === 0) {
        list.innerHTML = '<p class="text-center py-8 text-gray-500">아직 감지된 클립이 없습니다.</p>';
    } else {
        list.innerHTML = state.detectionClips.map(clip => {
            const duration = clip.endTime ? (clip.endTime - clip.startTime).toFixed(1) : '진행 중';
            return `
                <div class="bg-white rounded-lg shadow-lg p-4 border-l-4 border-red-500">
                    <div class="flex justify-between items-start">
                        <div class="flex-1">
                            <div class="font-bold text-lg text-gray-800">${clip.personName}</div>
                            <div class="text-sm text-gray-600 mt-1">
                                시간: ${formatTime(clip.startTime)} - ${clip.endTime ? formatTime(clip.endTime) : '진행 중'}
                            </div>
                            <div class="text-xs text-gray-500 mt-1">
                                길이: ${duration}초 | 유사도: ${clip.similarity}%
                            </div>
                        </div>
                        <div class="flex gap-2 ml-4">
                            <button onclick="seekToClip(${clip.startTime})" 
                                    class="bg-blue-500 text-white px-3 py-1 rounded text-xs hover:bg-blue-600">
                                재생
                            </button>
                            ${clip.endTime ? `
                                <button onclick="downloadClip(${clip.id})" 
                                        class="bg-green-500 text-white px-3 py-1 rounded text-xs hover:bg-green-600">
                                    다운로드
                                </button>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    modal.classList.remove('hidden');
});

// 모달 닫기
document.getElementById('closeClipModalBtn').addEventListener('click', () => {
    document.getElementById('clipModal').classList.add('hidden');
});
document.getElementById('closeClipModalBtn2').addEventListener('click', () => {
    document.getElementById('clipModal').classList.add('hidden');
});

// 클립으로 이동
function seekToClip(startTime) {
    if (UI.video) {
        UI.video.currentTime = startTime;
        UI.video.play();
        document.getElementById('clipModal').classList.add('hidden');
    }
}

// 클립 다운로드
function downloadClip(clipId) {
    const clip = state.detectionClips.find(c => c.id === clipId);
    if (clip) {
        downloadVideoClip(clip);
    }
}

// WebSocket 메시지 핸들러는 script.js에서 직접 처리하도록 변경됨
// 이 파일의 함수들은 script.js에서 호출됨
