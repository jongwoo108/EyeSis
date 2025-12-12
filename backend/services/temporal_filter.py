# backend/services/temporal_filter.py
"""
Temporal Consistency 필터 서비스
"""
from typing import Dict
from fastapi import WebSocket
from backend.utils.websocket_manager import connection_states


def apply_temporal_filter(websocket: WebSocket, result: Dict) -> Dict:
    """
    개선된 Temporal Filter: Hysteresis 임계값 + 윈도우 기반 투표
    
    기존 문제: 연속 3프레임 매칭 요구 → 임계값 근처에서 깜빡임
    개선 방안:
    1. Hysteresis 임계값: 시작(0%) vs 유지(-3%) → 안정성 향상
    2. 윈도우 기반 투표: 최근 5프레임 중 3프레임 매칭 → 일시적 실패 무시
    
    Args:
        websocket: WebSocket 연결 객체
        result: process_detection의 반환값
    
    Returns:
        temporal filter가 적용된 result
    """
    # 설정
    WINDOW_SIZE = 5              # 최근 N프레임 추적
    MIN_MATCHES = 1              # 필요한 최소 매칭 수 (2 → 1: 최대 안정성)
    MATCH_KEEP_OFFSET = -0.05    # 유지 임계값 완화 (-5%) - 왼쪽 얼굴 인식 개선
    
    if websocket not in connection_states:
        return result
    
    state = connection_states[websocket]
    match_history = state.get("match_history", {})  # {person_id: [(confidence, matched), ...]}
    
    # 현재 프레임의 detection을 person_id별로 매핑
    current_detections = {}
    for det in result.get("detections", []):
        person_id = det.get("person_id")
        if person_id:
            current_detections[person_id] = det
    
    # 사라진 person_id의 히스토리 정리
    for person_id in list(match_history.keys()):
        if person_id not in current_detections:
            del match_history[person_id]
    
    # 각 detection에 temporal filter 적용
    filtered_detections = []
    alert_triggered = False
    detected_metadata = result.get("metadata", {"name": "미상", "confidence": 0, "status": "unknown"})
    
    for det in result.get("detections", []):
        person_id = det.get("person_id")
        status = det.get("status", "unknown")
        confidence = det.get("confidence", 0) / 100.0  # 0-1 범위로 변환
        
        # criminal 또는 normal 상태이고 person_id가 있는 경우만 temporal filter 적용
        if status in ["criminal", "normal"] and person_id:
            # 이력 가져오기
            history = match_history.get(person_id, [])
            
            # 현재 안정 상태 확인 (최근 MIN_MATCHES 프레임이 모두 매칭)
            is_currently_stable = False
            if len(history) >= MIN_MATCHES:
                recent_matches = [matched for _, matched in history[-MIN_MATCHES:]]
                is_currently_stable = all(recent_matches)
            
            # Hysteresis: 안정 상태면 낮은 임계값 사용
            base_threshold = 0.48  # 기본값
            if is_currently_stable:
                effective_threshold = base_threshold + MATCH_KEEP_OFFSET  # 0.45
            else:
                effective_threshold = base_threshold  # 0.48
            
            # 현재 프레임 매칭 여부 판단
            is_matched = confidence >= effective_threshold
            
            # 이력에 추가
            history.append((confidence, is_matched))
            
            # 윈도우 크기 유지
            if len(history) > WINDOW_SIZE:
                history = history[-WINDOW_SIZE:]
            
            match_history[person_id] = history
            
            # 투표: 최근 N프레임 중 M프레임 이상 매칭?
            matched_count = sum(matched for _, matched in history)
            is_stable = matched_count >= MIN_MATCHES
            
            # 디버그 로그
            history_str = "".join(["O" if m else "X" for _, m in history])
            
            if not is_stable:
                # 아직 불안정 → Unknown
                filtered_det = det.copy()
                filtered_det["status"] = "unknown"
                filtered_det["color"] = "yellow"
                filtered_det["name"] = "Unknown"
                filtered_detections.append(filtered_det)
                print(f"   🔄 [TEMPORAL] {person_id[-6:]}: [{history_str}] {matched_count}/{len(history)} → Unknown (th={effective_threshold:.2f})")
            else:
                # 안정 → 원래 상태 유지
                filtered_detections.append(det)
                print(f"   ✅ [TEMPORAL] {person_id[-6:]}: [{history_str}] {matched_count}/{len(history)} → Stable (th={effective_threshold:.2f})")
                
                if status == "criminal":
                    alert_triggered = True
                    detected_metadata = {
                        "name": det.get("name", "Unknown"),
                        "confidence": det.get("confidence", 0),
                        "status": "criminal"
                    }
                elif not alert_triggered:
                    detected_metadata = {
                        "name": det.get("name", "Unknown"),
                        "confidence": det.get("confidence", 0),
                        "status": "normal"
                    }
        else:
            # unknown 상태는 그대로 유지
            filtered_detections.append(det)
    
    # match_history 업데이트
    state["match_history"] = match_history
    
    return {
        "detections": filtered_detections,
        "alert": alert_triggered,
        "metadata": detected_metadata,
        "learning_events": result.get("learning_events", [])
    }