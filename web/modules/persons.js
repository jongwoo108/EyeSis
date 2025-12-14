import { state } from "./state.js";
import { personNameMapping, API_BASE_URL } from "./config.js";
import { initUI } from "./ui.js";
import { getCategoryText, getCategoryStyle } from "./utils.js";
import { loadPersons } from "./api.js";

// UI 객체 가져오기
const UI = initUI();


// 인물 카드 동적 생성 (다중 선택 가능)
export function createSuspectCard(person) {
    const displayName = personNameMapping[person.id] || person.name;
    const isCriminal = person.is_criminal;

    // 카테고리 텍스트 가져오기
    const categoryText = getCategoryText(person);

    // 카테고리에 따른 스타일 결정
    const categoryStyle = getCategoryStyle(categoryText);

    // 색상 및 텍스트 설정
    const bgColor = categoryStyle.bgColor;
    const textColor = categoryStyle.textColor;
    const statusText = categoryText || '미상';

    const card = document.createElement('div');
    card.className = 'suspect-card bg-white rounded-lg shadow-sm overflow-hidden cursor-pointer transform hover:scale-105 transition-all duration-200 relative';
    card.setAttribute('data-suspect-id', person.id);
    card.setAttribute('data-is-thief', isCriminal.toString());

    // 이미지 URL이 있으면 사용, 없으면 기본 이모지
    const imageUrl = person.image_url || null;
    const imageHtml = imageUrl
        ? `<img src="${imageUrl}" alt="${displayName}" class="w-full h-full object-cover" onerror="this.parentElement.innerHTML='<span class=\\'text-6xl\\'>👤</span>'">`
        : `<span class="text-6xl">👤</span>`;

    // 체크박스 아이콘 추가
    card.innerHTML = `
        <div class="absolute top-2 right-2 w-6 h-6 rounded-full border-2 border-gray-300 bg-white flex items-center justify-center checkmark hidden z-10">
            <svg class="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path>
            </svg>
        </div>
        <div class="h-48 ${bgColor} flex items-center justify-center overflow-hidden">
            ${imageHtml}
        </div>
        <div class="p-4">
            <div class="flex items-center justify-between mb-2">
                <h3 class="font-bold text-lg">${displayName}</h3>
                <button class="edit-person-btn text-gray-400 hover:text-indigo-600 p-1 transition-colors" data-person-id="${person.id}" title="정보 수정">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path>
                    </svg>
                </button>
            </div>
            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${categoryStyle.bgColor} ${categoryStyle.textColor} border ${categoryStyle.borderColor}">
                ${statusText}
            </span>
        </div>
    `;

    // 수정 버튼 이벤트 리스너 추가 (이벤트 전파 방지)
    const editBtn = card.querySelector('.edit-person-btn');
    if (editBtn) {
        editBtn.addEventListener('click', function (e) {
            e.stopPropagation(); // 카드 선택 방지
            openEditPersonModal(person);
        });
    }

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
            // 선택 추가 (person 객체 전체 저장하여 카테고리 정보 포함)
            state.selectedSuspects.push({
                id: person.id,
                name: displayName,
                isThief: isCriminal,
                person: person  // person 객체 전체 저장
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
export function updateSelectedSuspectsInfo() {
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

    // 선택된 인물 개수 업데이트
    updateSelectedPersonCount();
}

// ==========================================
// 전체 선택/해제/삭제 기능
// ==========================================

// 전체 선택
export function selectAllPersons() {
    const cards = UI.suspectCardsContainer.querySelectorAll('.suspect-card');
    cards.forEach(card => {
        const suspectId = card.getAttribute('data-suspect-id');
        const isCriminal = card.getAttribute('data-is-thief') === 'true';

        // 이미 선택되어 있으면 스킵
        const isSelected = state.selectedSuspects.some(s => s.id === suspectId);
        if (!isSelected) {
            // person 이름 가져오기
            const personName = card.querySelector('h3').textContent;
            state.selectedSuspects.push({
                id: suspectId,
                name: personName,
                isThief: isCriminal,
                person: state.personDatabase.find(p => p.id === suspectId) || null
            });
            card.classList.add('ring-4', 'ring-blue-500');
            card.querySelector('.checkmark').classList.remove('hidden');
        }
    });

    updateSelectedSuspectsInfo();
    updateSelectedPersonCount();
    UI.proceedBtn.disabled = false;
    console.log(`✅ 전체 선택 완료: ${state.selectedSuspects.length}명`);
}

// 전체 해제
export function deselectAllPersons() {
    const cards = UI.suspectCardsContainer.querySelectorAll('.suspect-card');
    cards.forEach(card => {
        card.classList.remove('ring-4', 'ring-blue-500');
        const checkmark = card.querySelector('.checkmark');
        if (checkmark) {
            checkmark.classList.add('hidden');
        }
    });

    state.selectedSuspects = [];
    updateSelectedSuspectsInfo();
    updateSelectedPersonCount();
    UI.proceedBtn.disabled = true;
    console.log('✅ 전체 해제 완료');
}

// 선택된 인물 개수 업데이트
export function updateSelectedPersonCount() {
    if (UI.selectedPersonCount) {
        UI.selectedPersonCount.textContent = state.selectedSuspects.length;
    }

    // 삭제 버튼 활성화/비활성화
    if (UI.deleteSelectedPersonsBtn) {
        UI.deleteSelectedPersonsBtn.disabled = state.selectedSuspects.length === 0;
    }
}

// 선택된 인물들을 일괄 삭제
export async function deleteSelectedPersons() {
    if (state.selectedSuspects.length === 0) {
        alert('삭제할 인물을 선택해주세요.');
        return;
    }

    // 확인 대화상자 (2중 확인)
    const personNames = state.selectedSuspects.map(s => s.name).join(', ');
    const confirmed = window.confirm(
        `정말로 선택된 ${state.selectedSuspects.length}명의 인물을 데이터베이스에서 삭제하시겠습니까?\n\n` +
        `삭제될 인물: ${personNames}\n\n` +
        `⚠️ 관련된 모든 이미지 및 임베딩 데이터가 영구 삭제됩니다.`
    );

    if (!confirmed) {
        return;
    }

    // 2차 확인
    const secondConfirmed = window.confirm(
        `한 번 더 확인합니다.\n정말로 ${state.selectedSuspects.length}명을 삭제하시겠습니까?\n\n이 작업은 되돌릴 수 없습니다.`
    );

    if (!secondConfirmed) {
        return;
    }

    let successCount = 0;
    let failCount = 0;
    const failedNames = [];

    // 순차적으로 삭제
    for (const suspect of state.selectedSuspects) {
        try {
            const response = await fetch(`${API_BASE_URL}/persons/${suspect.id}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                // 성공: 카드 제거
                const card = UI.suspectCardsContainer.querySelector(`[data-suspect-id="${suspect.id}"]`);
                if (card) {
                    card.remove();
                }
                successCount++;
                console.log(`✅ 삭제 성공: ${suspect.name}`);
            } else {
                failCount++;
                failedNames.push(suspect.name);
                console.error(`❌ 삭제 실패: ${suspect.name}`, data);
            }
        } catch (error) {
            failCount++;
            failedNames.push(suspect.name);
            console.error(`❌ 삭제 중 오류: ${suspect.name}`, error);
        }
    }

    // 선택 해제
    state.selectedSuspects = [];
    updateSelectedSuspectsInfo();
    updateSelectedPersonCount();
    UI.proceedBtn.disabled = true;

    // ⭐ 인물 목록 새로고침 (DB에서 다시 로드) - 버그 수정
    // 삭제된 인물이 다시 나타나지 않도록 DB에서 다시 불러와야 함
    await renderSuspectCards();

    // 결과 메시지
    let message = `삭제 완료:\n✅ 성공: ${successCount}명`;
    if (failCount > 0) {
        message += `\n❌ 실패: ${failCount}명 (${failedNames.join(', ')})`;
    }
    alert(message);

    // 인물 목록이 비어있으면 메시지 표시
    if (UI.suspectCardsContainer.children.length === 0) {
        UI.suspectCardsContainer.innerHTML = `
            <div class="col-span-full text-center py-8 text-gray-500">
        `;
    }

    console.log(`🎉 일괄 삭제 완료: 성공 ${successCount}명, 실패 ${failCount}명`);
}

// ==========================================
// 인물 정보 수정 기능
// ==========================================

// 인물 수정 모달 열기
export function openEditPersonModal(person) {
    const modal = document.getElementById('editPersonModal');
    const personIdInput = document.getElementById('editPersonId');
    const nameInput = document.getElementById('editPersonName');
    const categorySelect = document.getElementById('editPersonCategory');
    const customContainer = document.getElementById('editCustomCategoryContainer');
    const customInput = document.getElementById('editPersonCategoryCustom');

    if (!modal || !personIdInput || !nameInput || !categorySelect) {
        console.error('수정 모달 요소를 찾을 수 없습니다');
        return;
    }

    // 현재 정보로 폼 채우기
    personIdInput.value = person.id;
    nameInput.value = person.name;

    // 카테고리 설정
    const standardCategories = ['criminal', 'missing', 'dementia', 'child', 'wanted'];
    // person.person_type(최상위) 또는 info 내부 확인
    const currentCategory = person.person_type || person.info?.person_type || 'criminal';

    if (standardCategories.includes(currentCategory)) {
        // 표준 카테고리인 경우
        categorySelect.value = currentCategory;
        if (customContainer) {
            customContainer.classList.add('hidden');
            if (customInput) {
                customInput.required = false;
                customInput.value = '';
            }
        }
    } else {
        // 커스텀 카테고리인 경우
        categorySelect.value = 'custom';
        if (customContainer) {
            customContainer.classList.remove('hidden');
            if (customInput) {
                customInput.required = true;
                customInput.value = currentCategory;
            }
        }
    }

    // 모달 표시
    modal.classList.remove('hidden');
}

// 인물 수정 모달 닫기
export function closeEditPersonModal() {
    const modal = document.getElementById('editPersonModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

// 인물 정보 업데이트 (API 호출)
export async function updatePerson(personId, name, personType) {
    try {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('person_type', personType);

        const response = await fetch(`${API_BASE_URL}/persons/${personId}`, {
            method: 'PUT',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '수정 실패');
        }

        const result = await response.json();
        console.log('✅ 인물 정보 수정 완료:', result);

        // 성공 메시지
        alert(`인물 정보가 수정되었습니다: ${result.person.name}`);

        // 모달 닫기
        closeEditPersonModal();

        // 인물 목록 새로고침
        await renderSuspectCards();

        // 1. 선택된 인물 목록(state.selectedSuspects) 업데이트
        const selectedIndex = state.selectedSuspects.findIndex(s => s.id === personId);
        if (selectedIndex !== -1) {
            // 정보 갱신
            state.selectedSuspects[selectedIndex] = {
                ...state.selectedSuspects[selectedIndex],
                name: result.person.name,
                isThief: result.person.is_criminal,
                person: {
                    ...state.selectedSuspects[selectedIndex].person,
                    name: result.person.name,
                    is_criminal: result.person.is_criminal,
                    person_type: result.person.person_type,
                    info: result.person.info
                }
            };

            // 선택된 인물 UI 업데이트
            updateSelectedSuspectsInfo();

            // 2. 타임라인 재렌더링 (이름 변경 등 반영)
            // 타임라인 컨테이너가 있고, 해당 인물의 트랙이 있는 경우에만
            const timelinesContainer = document.getElementById('timelinesContainer');
            if (timelinesContainer && timelinesContainer.querySelector(`[data-person-id="${personId}"]`)) {
                // 트랙 헤더(이름) 업데이트
                const trackHeader = timelinesContainer.querySelector(`[data-person-id="${personId}"] .font-bold`);
                if (trackHeader) {
                    trackHeader.textContent = result.person.name;
                }

                // 타임라인 마커 재렌더링
                renderTimelineWithMerging();
            }
        }

        return result;
    } catch (error) {
        console.error('❌ 인물 수정 실패:', error);
        alert(`수정 실패: ${error.message}`);
        throw error;
    }
}

// 인물 카드들을 동적으로 생성하고 표시
export async function renderSuspectCards() {
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