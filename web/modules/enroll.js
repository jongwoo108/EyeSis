import { state } from './state.js';
import { initUI } from './ui.js';

const UI = initUI();

// 구분 선택 업데이트 함수
export function updatePersonCategory() {
    const categorySelect = UI.personCategory;
    const customContainer = document.getElementById('customCategoryContainer');
    const customInput = UI.personCategoryCustom;
    const personTypeInput = document.getElementById('personTypeInput');

    if (!categorySelect || !personTypeInput) return;

    const selectedValue = categorySelect.value;

    // '기타' 선택 시 입력 필드 표시
    if (selectedValue === 'custom') {
        if (customContainer) {
            customContainer.classList.remove('hidden');
        }
        if (customInput) {
            customInput.required = true;
            customInput.value = ''; // 초기화
        }
    } else {
        // 다른 옵션 선택 시 입력 필드 숨김
        if (customContainer) {
            customContainer.classList.add('hidden');
        }
        if (customInput) {
            customInput.required = false;
            customInput.value = '';
        }
        // personTypeInput 값 업데이트
        personTypeInput.value = selectedValue;
    }

    // 폼 유효성 검사
    checkFormValidity();
}

// 폼 유효성 검사 함수
export function checkFormValidity() {
    const name = UI.enrollName.value.trim();
    const imageFile = UI.enrollImage.files[0];
    const categorySelect = UI.personCategory;
    const customInput = UI.personCategoryCustom;
    const personTypeInput = document.getElementById('personTypeInput');

    // 구분 값 확인
    let personType = null;
    if (categorySelect) {
        if (categorySelect.value === 'custom') {
            // '기타' 선택 시 직접 입력 값 확인
            if (customInput && customInput.value.trim()) {
                personType = customInput.value.trim();
                if (personTypeInput) {
                    personTypeInput.value = personType;
                }
            }
        } else {
            // 일반 옵션 선택 시
            personType = categorySelect.value;
            if (personTypeInput) {
                personTypeInput.value = personType;
            }
        }
    }

    const isValid = name && imageFile && personType;

    // 등록 버튼 상태 업데이트
    if (isValid) {
        UI.submitEnrollBtn.disabled = false;
        UI.submitEnrollBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        UI.submitEnrollBtn.classList.add('opacity-100', 'cursor-pointer');
    } else {
        UI.submitEnrollBtn.disabled = true;
        UI.submitEnrollBtn.classList.add('opacity-50', 'cursor-not-allowed');
        UI.submitEnrollBtn.classList.remove('opacity-100', 'cursor-pointer');
    }

    return isValid;
}

// 모달 닫기 함수 (공통)
export function closeEnrollModal() {
    UI.addSuspectModal.classList.add('hidden');
    // 폼 완전 초기화
    UI.addSuspectForm.reset();
    UI.imagePreview.classList.add('hidden');
    UI.imagePlaceholder.classList.remove('hidden');
    UI.enrollError.classList.add('hidden');
    UI.enrollSuccess.classList.add('hidden');

    // 구분 선택 초기화
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

    // 버튼 상태 초기화 (비활성화)
    UI.submitEnrollBtn.disabled = true;
    UI.submitEnrollBtn.classList.add('opacity-50', 'cursor-not-allowed');
    UI.submitEnrollBtn.classList.remove('opacity-100', 'cursor-pointer');
}
