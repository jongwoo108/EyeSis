// modules/handlers.js - 이벤트 핸들러 모듈
import { state } from './state.js';
import { initUI } from './ui.js';
import {
    renderSnapshotCard,
    filterSnapshotsByPerson,
    updateSnapshotCheckboxes,
    updateSelectedCount
} from './snapshots.js';
import { downloadVideoClip } from './clips.js';
import { updatePersonCategory, checkFormValidity, closeEnrollModal } from './enroll.js';
import { renderSuspectCards } from './persons.js';

const UI = initUI();

// 현재 선택된 인물 필터 (모듈 내 상태)
let currentSnapshotFilter = '전체';

// 필터 getter (외부에서 접근 필요시)
export function getCurrentSnapshotFilter() {
    return currentSnapshotFilter;
}

// 필터 setter
export function setCurrentSnapshotFilter(filter) {
    currentSnapshotFilter = filter;
}

// ==========================================
// 스냅샷/클립 모달 핸들러
// ==========================================

// 스냅샷 보기 핸들러
export function handleViewSnapshots() {
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
            const selectedPerson = state.selectedSuspects.find(s => s.id === snapshot.personId);
            const personName = selectedPerson ? selectedPerson.name : (snapshot.personName || 'Unknown');
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
                tabs.forEach(t => {
                    t.classList.remove('active', 'bg-indigo-600', 'text-white', 'shadow-sm');
                    t.classList.add('bg-gray-200', 'text-gray-700');
                });

                tab.classList.add('active', 'bg-indigo-600', 'text-white', 'shadow-sm');
                tab.classList.remove('bg-gray-200', 'text-gray-700');

                const selectedPerson = tab.dataset.person;
                currentSnapshotFilter = selectedPerson;
                filterSnapshotsByPerson(selectedPerson);

                updateSnapshotCheckboxes();
                updateSelectedCount();
            });
        });
    }

    modal.classList.remove('hidden');
}

// 클립 모달 닫기 핸들러
export function handleCloseClipModal() {
    const clipModal = document.getElementById('clipModal');
    if (clipModal) {
        clipModal.classList.add('hidden');
    }
}

// 스냅샷 모달 닫기 핸들러
export function handleCloseSnapshotModal() {
    const snapshotModal = document.getElementById('snapshotModal');
    if (snapshotModal) {
        snapshotModal.classList.add('hidden');
    }
}

// 모달 외부 클릭 핸들러
export function handleModalOutsideClick(e) {
    if (e.target.id === 'clipModal' || e.target.id === 'snapshotModal') {
        e.target.classList.add('hidden');
    }
}

// 전체 선택 핸들러
export function handleSelectAllSnapshots() {
    const filteredSnapshots = currentSnapshotFilter === '전체'
        ? state.snapshots
        : state.snapshots.filter(s => s.personName === currentSnapshotFilter);

    filteredSnapshots.forEach(snapshot => {
        if (!state.selectedSnapshots.includes(snapshot.id)) {
            state.selectedSnapshots.push(snapshot.id);
        }
    });

    updateSnapshotCheckboxes();
    updateSelectedCount();
}

// 전체 해제 핸들러
export function handleDeselectAllSnapshots() {
    const filteredSnapshots = currentSnapshotFilter === '전체'
        ? state.snapshots
        : state.snapshots.filter(s => s.personName === currentSnapshotFilter);

    const filteredIds = filteredSnapshots.map(s => s.id);
    state.selectedSnapshots = state.selectedSnapshots.filter(id => !filteredIds.includes(id));

    updateSnapshotCheckboxes();
    updateSelectedCount();
}

// 선택 다운로드 핸들러
export async function handleDownloadSelectedSnapshots() {
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

    for (let i = 0; i < selectedSnapshots.length; i++) {
        const snapshot = selectedSnapshots[i];
        const link = document.createElement('a');
        link.href = snapshot.base64Image;
        link.download = `${i + 1}_criminal_${snapshot.personName}_${formatTime(snapshot.videoTime).replace(':', '-')}.jpg`;
        link.click();

        if (i < selectedSnapshots.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    console.log(`✅ ${selectedSnapshots.length}개의 선택된 스냅샷 다운로드 완료`);
}

// 전체 다운로드 핸들러
export async function handleDownloadAllSnapshots() {
    if (state.snapshots.length === 0) {
        alert('다운로드할 스냅샷이 없습니다.');
        return;
    }

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

    for (let i = 0; i < filteredSnapshots.length; i++) {
        const snapshot = filteredSnapshots[i];
        const link = document.createElement('a');
        link.href = snapshot.base64Image;
        link.download = `${i + 1}_criminal_${snapshot.personName}_${formatTime(snapshot.videoTime).replace(':', '-')}.jpg`;
        link.click();

        await new Promise(resolve => setTimeout(resolve, 300));
    }

    alert(`${filteredSnapshots.length}개의 스냅샷 다운로드를 시작했습니다.`);
}

// ==========================================
// 용의자 추가 모달 핸들러
// ==========================================

// 용의자 추가 모달 열기 핸들러
export function handleOpenAddSuspectModal() {
    UI.addSuspectForm.reset();
    UI.imagePreview.classList.add('hidden');
    UI.imagePlaceholder.classList.remove('hidden');
    UI.enrollError.classList.add('hidden');
    UI.enrollSuccess.classList.add('hidden');

    if (UI.personCategory) {
        UI.personCategory.value = 'criminal';
    }
    const customContainer = document.getElementById('customCategoryContainer');
    if (customContainer) {
        customContainer.classList.add('hidden');
    }
    if (UI.personCategoryCustom) {
        UI.personCategoryCustom.value = '';
        UI.personCategoryCustom.required = false;
    }
    document.getElementById('personTypeInput').value = 'criminal';
    updatePersonCategory();

    UI.submitEnrollBtn.disabled = true;
    UI.submitEnrollBtn.textContent = '등록';
    UI.submitEnrollBtn.classList.add('opacity-50', 'cursor-not-allowed');
    UI.submitEnrollBtn.classList.remove('opacity-100', 'cursor-pointer');

    UI.addSuspectModal.classList.remove('hidden');
    checkFormValidity();
}

// 모달 외부 클릭 핸들러 (용의자 추가)
export function handleAddSuspectModalOutsideClick(e) {
    if (e.target === UI.addSuspectModal) {
        closeEnrollModal();
    }
}

// ==========================================
// 긴급 신고 모달 핸들러
// ==========================================

export function handleOpenEmergencyModal() {
    if (UI.emergencyCallModal) {
        UI.emergencyCallModal.classList.remove('hidden');
    }
}

export function handleCloseEmergencyModal() {
    if (UI.emergencyCallModal) {
        UI.emergencyCallModal.classList.add('hidden');
    }
}

export function handleEmergencyModalOutsideClick(e) {
    if (e.target === UI.emergencyCallModal) {
        UI.emergencyCallModal.classList.add('hidden');
    }
}

// ==========================================
// 공통 핸들러
// ==========================================

// ESC 키 핸들러 (모든 모달)
export function handleEscapeKey(e) {
    if (e.key !== 'Escape') return;

    if (!UI.addSuspectModal.classList.contains('hidden')) {
        closeEnrollModal();
    }
    if (UI.emergencyCallModal && !UI.emergencyCallModal.classList.contains('hidden')) {
        UI.emergencyCallModal.classList.add('hidden');
    }
    if (UI.dispatchReportModal && !UI.dispatchReportModal.classList.contains('hidden')) {
        UI.dispatchReportModal.classList.add('hidden');
    }
}

// 이미지 미리보기 핸들러
export function handleImagePreview(e) {
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
    checkFormValidity();
}

// ==========================================
// 전역 함수 등록 (HTML onclick에서 호출)
// ==========================================

// 클립으로 이동
window.seekToClip = function (startTime) {
    if (UI.video) {
        UI.video.currentTime = startTime;
        UI.video.play();
        handleCloseClipModal();
    }
};

// 클립 다운로드
window.downloadClip = function (clipId) {
    const clip = state.detectionClips.find(c => c.id === clipId);
    if (clip) {
        downloadVideoClip(clip);
    } else {
        console.error(`클립을 찾을 수 없습니다: ${clipId}`);
    }
};

// 스냅샷 다운로드
window.downloadSnapshot = function (snapshotId) {
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
