# backend/utils/websocket_manager.py
"""
WebSocket 연결 관리
"""

from typing import Dict, Set
from fastapi import WebSocket
import asyncio

# 활성 WebSocket 연결 추적
active_connections: Set[WebSocket] = set()

# 연결별 상태 관리
connection_states: Dict[WebSocket, Dict] = {}

async def register_connection(websocket: WebSocket):
    """WebSocket 연결 등록"""
    try:
        await websocket.accept()
        active_connections.add(websocket)
        connection_states[websocket] = {
            "suspect_ids": [],  # 여러 명 선택 가능
            "connected_at": asyncio.get_event_loop().time()
        }
        print(f"✅ WebSocket 연결됨 (총 {len(active_connections)}개 연결)")
    except Exception as e:
        print(f"❌ WebSocket 연결 등록 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

def unregister_connection(websocket: WebSocket):
    """WebSocket 연결 해제"""
    active_connections.discard(websocket)
    connection_states.pop(websocket, None)
    print(f"❌ WebSocket 연결 해제됨 (남은 연결: {len(active_connections)}개)")