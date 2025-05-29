from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime


class CacheMetadata(BaseModel):
    prompt_text: Optional[str] = None
    token_count: int
    model_id: str
    created_at: datetime
    file_size_mb: float
    synthesis_file: Optional[str] = None


class CacheInfo(BaseModel):
    cache_id: str
    file_path: str
    metadata: CacheMetadata


class CacheListResponse(BaseModel):
    caches: List[CacheInfo]
    total_count: int


class CacheValidationRequest(BaseModel):
    model_id: str
    cache_path: str


class ValidationResult(BaseModel):
    compatible: bool
    details: str
