from fastapi import APIRouter, HTTPException
from typing import List
from .schema import CacheInfo, CacheListResponse, CacheValidationRequest, ValidationResult
from .manager import CacheManagementService

router = APIRouter(tags=["cache-management"])


@router.get("/v1/caches", response_model=CacheListResponse)
async def list_available_caches():
    """List all available pre-computed caches"""
    cache_service = CacheManagementService()
    return cache_service.list_caches()


@router.get("/v1/caches/{cache_id}", response_model=CacheInfo)
async def get_cache_info(cache_id: str):
    """Get detailed information about a specific cache"""
    cache_service = CacheManagementService()
    info = cache_service.get_cache_info(cache_id)
    if not info:
        raise HTTPException(status_code=404, detail="Cache not found")
    return info


@router.post("/v1/caches/validate", response_model=ValidationResult)
async def validate_cache_compatibility(request: CacheValidationRequest):
    """Validate if a cache is compatible with a specific model"""
    cache_service = CacheManagementService()
    return cache_service.validate_compatibility(request.model_id, request.cache_path)
