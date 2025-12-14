// modules/timeline.js
import { state } from './state.js';
import { getCategoryStyle, getCategoryText } from './utils.js';
import { initUI } from './ui.js';

// UI 객체 가져오기
const UI = initUI();

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
export function createTimelineTrack(personId, personName, isCriminal, person = null) {
    const track = document.createElement('div');

    // person 객체가 있으면 카테고리 텍스트 가져오기, 없으면 기본값 사용
    const categoryText = person ? getCategoryText(person) : (isCriminal ? '범죄자' : '실종자');
    const categoryStyle = person ? getCategoryStyle(categoryText) : {
        bgColor: isCriminal ? 'bg-red-50' : 'bg-green-50',
        textColor: isCriminal ? 'text-red-700' : 'text-green-700',
        borderColor: isCriminal ? 'border-red-200' : 'border-green-200'
    };

    const bgColor = categoryStyle.bgColor + ' ' + categoryStyle.borderColor;
    const textColor = categoryStyle.textColor;
    const labelText = categoryText;

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
export function initializeTimelinesForSelectedPersons() {
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
            suspect.isThief,  // isThief가 true면 범죄자
            suspect.person || null  // person 객체 전달 (카테고리 정보 포함)
        );
        timelinesContainer.appendChild(track);
        console.log(`✅ 타임라인 트랙 생성: ${suspect.name} (${suspect.isThief ? '범죄자' : '실종자'})`);
    });

    console.log(`📊 총 ${state.selectedSuspects.length}개 타임라인 트랙 생성 완료`);
}

// 타임라인 감지 구간 병합 함수
export function mergeTimelineEvents(events, mergeThreshold = 2.0) {
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
export function renderTimelineWithMerging() {
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
            let startPercent = (event.start / UI.video.duration) * 100;
            let endPercent = (event.end / UI.video.duration) * 100;

            // 시작이 0보다 작으면 0으로 제한
            if (startPercent < 0) startPercent = 0;
            // 끝이 100보다 크면 100으로 제한 (영상 끝까지 표시)
            if (endPercent > 100) endPercent = 100;
            // 시작이 100보다 크면 스킵
            if (startPercent >= 100) return;

            const widthPercent = endPercent - startPercent;

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
export function addTimelineMarkerDirect(snapshot) {
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
export function updateSnapshotCountDirect() {
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
export function updateClipCount() {
    const countEl = document.getElementById('clipCount');
    if (countEl) {
        countEl.textContent = state.detectionClips.length;
    }
}