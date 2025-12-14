// modules/clips.js
import { state } from './state.js';
import { API_BASE_URL } from './config.js';

// 영상 클립 다운로드 함수 (서버로 요청)
export async function downloadVideoClip(clip) {
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

// ==========================================
// 클립 데이터(clip)를 받아 카드 HTML을 반환하는 함수
export function getClipItemHTML(clip) {
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
                        <h4 class="text-base font-bold text-gray-800 leading-tight">${(() => {
            const selectedPerson = state.selectedSuspects.find(s => s.id === clip.personId);
            return selectedPerson ? selectedPerson.name : (clip.personName || 'Unknown');
        })()}</h4>
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

// 클립 필터링 함수
export function filterClipsByPerson(personName) {
    const list = document.getElementById('clipList');
    if (!list) return;

    const items = list.querySelectorAll('[data-person-name]');
    items.forEach(item => {
        if (personName === '전체' || item.dataset.personName === personName) {
            item.style.display = '';
        } else {
            item.style.display = 'none';
        }
    });
}

// 클립 선택 토글 함수
export function toggleClipSelection(clipId, isChecked) {
    if (isChecked) {
        if (!state.selectedClips.includes(clipId)) {
            state.selectedClips.push(clipId);
        }
    } else {
        state.selectedClips = state.selectedClips.filter(id => id !== clipId);
    }
    updateSelectedClipCount();
}

// window 전역 함수 등록 (HTML onclick에서 호출)
window.toggleClipSelection = toggleClipSelection;

// 선택된 클립 개수 업데이트
export function updateSelectedClipCount() {
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
