// modules/snapshots.js
import { state } from './state.js';

// 스냅샷 카드 렌더링 헬퍼 함수
export function renderSnapshotCard(snapshot) {
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const isSelected = state.selectedSnapshots.includes(snapshot.id);

    const selectedPerson = state.selectedSuspects.find(s => s.id === snapshot.personId);
    const displayName = selectedPerson ? selectedPerson.name : (snapshot.personName || 'Unknown');

    return `
                <div class="bg-white rounded-lg shadow-sm overflow-hidden relative" data-person-name="${displayName}" data-snapshot-id="${snapshot.id}">
                    <div class="absolute top-2 left-2 z-10">
                        <input type="checkbox" 
                               class="snapshot-checkbox appearance-none h-7 w-7 rounded-full border-2 border-white/50 bg-white/20 backdrop-blur-sm checked:bg-white/40 checked:border-white/80 focus:ring-2 focus:ring-white/50 focus:ring-offset-2 cursor-pointer transition-all duration-200 ease-in-out" 
                               ${isSelected ? 'checked' : ''}
                               onchange="toggleSnapshotSelection(${snapshot.id}, this.checked)">
                    </div>
                    <img src="${snapshot.base64Image}" alt="${displayName}" class="w-full h-48 object-cover cursor-pointer" 
                         onclick="window.open(this.src)">
                    <div class="p-3">
                        <div class="font-semibold text-sm text-gray-800 tracking-tight">${displayName}</div>
                        <div class="text-xs text-gray-600 mt-1 tracking-tight">시간: ${formatTime(snapshot.videoTime)}</div>
                        <div class="text-xs text-gray-600 tracking-tight">유사도: ${snapshot.similarity}%</div>
                        <div class="text-xs text-gray-500 tracking-tight">${new Date(snapshot.timestamp).toLocaleString()}</div>
                    </div>
                </div>
    `;
}

// 스냅샷 그리드 필터링 함수
export function filterSnapshotsByPerson(personName) {
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

// 스냅샷 선택 토글 함수
export function toggleSnapshotSelection(snapshotId, isChecked) {
    if (isChecked) {
        if (!state.selectedSnapshots.includes(snapshotId)) {
            state.selectedSnapshots.push(snapshotId);
        }
    } else {
        state.selectedSnapshots = state.selectedSnapshots.filter(id => id !== snapshotId);
    }
    updateSelectedCount();
}

// window 전역 함수 등록 (HTML onchange에서 호출)
window.toggleSnapshotSelection = toggleSnapshotSelection;

// 선택된 스냅샷 개수 업데이트
export function updateSelectedCount() {
    const countEl = document.getElementById('selectedCount');
    if (countEl) {
        countEl.textContent = state.selectedSnapshots.length;
    }

    const downloadSelectedBtn = document.getElementById('downloadSelectedBtn');
    if (downloadSelectedBtn) {
        downloadSelectedBtn.disabled = state.selectedSnapshots.length === 0;
    }
}

// 체크박스 상태 업데이트
export function updateSnapshotCheckboxes() {
    const checkboxes = document.querySelectorAll('.snapshot-checkbox');
    checkboxes.forEach(checkbox => {
        const snapshotId = parseInt(checkbox.getAttribute('onchange').match(/\d+/)[0]);
        checkbox.checked = state.selectedSnapshots.includes(snapshotId);
    });
}
