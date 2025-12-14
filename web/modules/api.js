import { API_BASE_URL } from "./config.js";
import { state } from "./state.js";


// 인물 목록을 서버에서 가져오기
export async function loadPersons() {
    try {
        const response = await fetch(`${API_BASE_URL}/persons`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        if (data.success && data.persons) {
            state.personDatabase = data.persons; // 전체 DB 저장 (Fallback용)
            return data.persons;
        }
        return [];
    } catch (error) {
        console.error("인물 목록 로드 실패:", error);
        return [];
    }
}

// 서버 상태 확인
export async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log(`✅ 서버 상태 확인: ${data.status}, 활성 연결: ${data.active_connections}`);
            return true;
        }
        return false;
    } catch (error) {
        console.error("❌ 서버 상태 확인 실패:", error);
        return false;
    }
}