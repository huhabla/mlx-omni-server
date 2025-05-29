import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from ..utils.logger import logger
from .schema import CacheListResponse, CacheInfo, CacheMetadata, ValidationResult


class CacheManagementService:
    def __init__(self, cache_directory: str = "./caches"):
        self.cache_dir = Path(cache_directory)

    def list_caches(self) -> CacheListResponse:
        """List all available caches with their metadata"""
        caches = []

        # Check if directory exists
        if not self.cache_dir.exists():
            logger.warning(f"Cache directory does not exist: {self.cache_dir}")
            return CacheListResponse(caches=[], total_count=0)

        try:
            for cache_file in self.cache_dir.glob("*.safetensors"):
                # Look for corresponding synthesis file
                synthesis_file = cache_file.with_suffix(".synthesis.xml")

                if synthesis_file.exists():
                    try:
                        metadata = self._parse_synthesis_file(synthesis_file)
                        caches.append(CacheInfo(
                            cache_id=cache_file.stem,
                            file_path=str(cache_file),
                            metadata=metadata
                        ))
                    except Exception as e:
                        logger.error(f"Error processing cache {cache_file}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error listing caches: {e}")

        return CacheListResponse(caches=caches, total_count=len(caches))

    def _parse_synthesis_file(self, synthesis_path: Path) -> CacheMetadata:
        """Parse synthesis XML file to extract metadata"""
        try:
            tree = ET.parse(synthesis_path)
            root = tree.getroot()

            # Safe XML element access
            metadata = root.find('metadata')
            if metadata is None:
                raise ValueError("No metadata element found in synthesis file")

            # Helper function for safe access
            def get_text(parent, tag, default=""):
                elem = parent.find(tag) if parent is not None else None
                return elem.text if elem is not None and elem.text else default

            # Parse metadata
            token_count = int(get_text(metadata, 'token_count', "0"))
            model_id = get_text(metadata, 'model_id', "unknown")
            analysis_date = get_text(metadata, 'analysis_date', datetime.now().isoformat())

            # Get summary
            summary = get_text(root, 'summary', None)

            # Calculate file size
            cache_file = synthesis_path.with_suffix('.safetensors')
            file_size_mb = cache_file.stat().st_size / (1024 * 1024) if cache_file.exists() else 0

            return CacheMetadata(
                prompt_text=summary[:500] if summary else None,
                token_count=token_count,
                model_id=model_id,
                created_at=datetime.fromisoformat(analysis_date),
                file_size_mb=file_size_mb,
                synthesis_file=str(synthesis_path)
            )
        except Exception as e:
            logger.error(f"Error parsing synthesis file {synthesis_path}: {e}")
            raise

    def get_cache_info(self, cache_id: str) -> Optional[CacheInfo]:
        """Get information about a specific cache"""
        cache_file = self.cache_dir / f"{cache_id}.safetensors"
        if not cache_file.exists():
            return None

        synthesis_file = cache_file.with_suffix(".synthesis.xml")
        if synthesis_file.exists():
            try:
                metadata = self._parse_synthesis_file(synthesis_file)
                return CacheInfo(
                    cache_id=cache_id,
                    file_path=str(cache_file),
                    metadata=metadata
                )
            except Exception as e:
                logger.error(f"Error getting cache info for {cache_id}: {e}")
                return None
        return None

    def validate_compatibility(self, model_id: str, cache_path: str) -> ValidationResult:
        """Validate cache compatibility with model"""
        # TODO: Implement actual validation logic
        # This should check:
        # 1. If the cache file exists
        # 2. If the model architecture matches
        # 3. If the tokenizer is compatible
        # 4. If the cache format is correct

        cache_file = Path(cache_path)
        if not cache_file.exists():
            return ValidationResult(
                compatible=False,
                details=f"Cache file not found: {cache_path}"
            )

        # For now, return a placeholder
        return ValidationResult(
            compatible=True,
            details="Validation not yet fully implemented"
        )
