from fastapi import APIRouter, HTTPException, Depends
from typing import List
from .schema import CacheInfo, CacheListResponse, CacheValidationRequest, ValidationResult
from .manager import CacheManagementService
from ..config import settings

router = APIRouter(tags=["cache-management"])


def get_cache_service() -> CacheManagementService:
    """Dependency injection for CacheManagementService"""
    return CacheManagementService(cache_directory=settings.cache_directory)


@router.get("/v1/caches", response_model=CacheListResponse)
async def list_available_caches(
        cache_service: CacheManagementService = Depends(get_cache_service)
):
    """List all available pre-computed caches"""
    return cache_service.list_caches()


@router.get("/v1/caches/{cache_id}", response_model=CacheInfo)
async def get_cache_info(
        cache_id: str,
        cache_service: CacheManagementService = Depends(get_cache_service)
):
    """Get detailed information about a specific cache"""
    info = cache_service.get_cache_info(cache_id)
    if not info:
        raise HTTPException(status_code=404, detail="Cache not found")
    return info


@router.post("/v1/caches/validate", response_model=ValidationResult)
async def validate_cache_compatibility(
        request: CacheValidationRequest,
        cache_service: CacheManagementService = Depends(get_cache_service)
):
    """Validate if a cache is compatible with a specific model"""
    return cache_service.validate_compatibility(request.model_id, request.cache_path)
