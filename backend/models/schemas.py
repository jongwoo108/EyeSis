from typing import Optional, List
from pydantic import BaseModel

class DetectionRequest(BaseModel):
    image: str       # Base64 이미지
    suspect_id: Optional[str] = None  # (선택적) 특정 타겟 ID (호환성 유지)
    suspect_ids: Optional[List[str]] = None  # (선택적) 여러 타겟 ID