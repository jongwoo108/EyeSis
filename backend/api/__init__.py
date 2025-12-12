# backend/api/__init__.py
"""
API 라우터 패키지
"""
from backend.api.detection import router as detection_router
from backend.api.persons import router as persons_router
from backend.api.video import router as video_router

__all__ = ["detection_router", "persons_router", "video_router"]
