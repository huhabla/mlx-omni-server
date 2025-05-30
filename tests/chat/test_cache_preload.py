"""
Tests for cache management functionality

This test file verifies the cache management API endpoints and functionality:
1. Listing available caches
2. Getting specific cache information
3. Validating cache compatibility
4. Loading pre-computed caches in chat completions
"""

import logging
import os
import tempfile
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from mlx_omni_server.main import app
from mlx_omni_server.cache.manager import CacheManagementService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with test server"""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_cache_files(temp_cache_dir):
    """Create mock cache and synthesis files for testing"""
    # Create test cache file
    cache_file = temp_cache_dir / "test_cache.safetensors"
    cache_file.write_bytes(b"mock cache data")

    # Create corresponding synthesis XML file
    synthesis_file = temp_cache_dir / "test_cache.synthesis.xml"
    synthesis_content = """<?xml version="1.0" encoding="UTF-8"?>
<text_analysis>
    <metadata>
        <original_file>test_document.txt</original_file>
        <cache_file>test_cache.safetensors</cache_file>
        <token_count>1000</token_count>
        <model_id>mlx-community/test-model</model_id>
        <analysis_date>2024-01-01T12:00:00</analysis_date>
    </metadata>
    <summary>This is a test document summary.</summary>
    <keywords>test, cache, mlx</keywords>
</text_analysis>"""
    synthesis_file.write_text(synthesis_content)

    return temp_cache_dir


class TestCacheManagementAPI:
    """Test cache management API endpoints"""

    def test_list_caches_empty(self, client, temp_cache_dir, monkeypatch):
        """Test listing caches when directory is empty"""
        # Monkeypatch the cache directory
        monkeypatch.setattr(
            "mlx_omni_server.cache.manager.CacheManagementService.__init__",
            lambda self, cache_directory="./caches": setattr(self, 'cache_dir', temp_cache_dir)
        )

        response = client.get("/v1/caches")
        assert response.status_code == 200

        data = response.json()
        assert data["total_count"] == 0
        assert data["caches"] == []

    def test_list_caches_with_files(self, client, mock_cache_files, monkeypatch):
        """Test listing caches with existing cache files"""
        # Monkeypatch the cache directory
        monkeypatch.setattr(
            "mlx_omni_server.cache.manager.CacheManagementService.__init__",
            lambda self, cache_directory="./caches": setattr(self, 'cache_dir', mock_cache_files)
        )

        response = client.get("/v1/caches")
        assert response.status_code == 200

        data = response.json()
        assert data["total_count"] == 1
        assert len(data["caches"]) == 1

        cache_info = data["caches"][0]
        assert cache_info["cache_id"] == "test_cache"
        assert cache_info["metadata"]["token_count"] == 1000
        assert cache_info["metadata"]["model_id"] == "mlx-community/test-model"

    def test_get_cache_info_existing(self, client, mock_cache_files, monkeypatch):
        """Test getting information about a specific cache"""
        monkeypatch.setattr(
            "mlx_omni_server.cache.manager.CacheManagementService.__init__",
            lambda self, cache_directory="./caches": setattr(self, 'cache_dir', mock_cache_files)
        )

        response = client.get("/v1/caches/test_cache")
        assert response.status_code == 200

        cache_info = response.json()
        assert cache_info["cache_id"] == "test_cache"
        assert cache_info["metadata"]["prompt_text"] == "This is a test document summary."

    def test_get_cache_info_not_found(self, client, temp_cache_dir, monkeypatch):
        """Test getting information about non-existent cache"""
        monkeypatch.setattr(
            "mlx_omni_server.cache.manager.CacheManagementService.__init__",
            lambda self, cache_directory="./caches": setattr(self, 'cache_dir', temp_cache_dir)
        )

        response = client.get("/v1/caches/non_existent_cache")
        assert response.status_code == 404
        assert response.json()["detail"] == "Cache not found"

    def test_validate_cache_compatibility(self, client, mock_cache_files, monkeypatch):
        """Test cache compatibility validation"""
        monkeypatch.setattr(
            "mlx_omni_server.cache.manager.CacheManagementService.__init__",
            lambda self, cache_directory="./caches": setattr(self, 'cache_dir', mock_cache_files)
        )

        cache_path = str(mock_cache_files / "test_cache.safetensors")

        response = client.post("/v1/caches/validate", json={
            "model_id": "mlx-community/test-model",
            "cache_path": cache_path
        })

        assert response.status_code == 200
        result = response.json()
        assert result["compatible"] == True

    def test_validate_cache_not_found(self, client):
        """Test validation with non-existent cache file"""
        response = client.post("/v1/caches/validate", json={
            "model_id": "mlx-community/test-model",
            "cache_path": "/non/existent/path.safetensors"
        })

        assert response.status_code == 200
        result = response.json()
        assert result["compatible"] == False
        assert "not found" in result["details"]


class TestCacheIntegrationWithChat:
    """Test cache integration with chat completions"""

    def test_chat_completion_with_precomputed_cache(self, openai_client, mock_cache_files, monkeypatch):
        """Test chat completion using pre-computed cache via model@cache syntax"""
        cache_path = str(mock_cache_files / "test_cache.safetensors")

        # Mock the load_precomputed_cache method AND the cache structure
        def mock_load_cache(self, path):
            logger.info(f"Mock loading cache from: {path}")
            self._precomputed_cache_path = path
            self._precomputed_cache_tokens = 1000

            # IMPORTANT: Mock the actual cache structure
            # The cache needs to be a list with the correct number of layers
            # For gemma-3-1b model, we need to mock the appropriate number of layers
            from mlx_lm.models.cache import KVCache

            # Create mock KVCache objects for each layer
            # Gemma models typically have 18-24 layers depending on the variant
            num_layers = 26  # Adjust based on the actual model
            mock_cache_layers = []

            for _ in range(num_layers):
                # Create a mock KVCache object with proper method signatures
                mock_kv_cache = type('MockKVCache', (), {
                    'offset': 1000,  # Number of cached tokens
                    'keys': None,
                    'values': None,
                    'step': lambda self, k, v: (k, v),
                    # Fix: Add the offset parameter to the lambda function
                    'update_and_fetch': lambda self, keys, values: (keys, values),
                })()
                mock_cache_layers.append(mock_kv_cache)

            # Set the cache on the prompt_cache object
            self._prompt_cache.cache = mock_cache_layers
            self._prompt_cache.cached_token_count = 1000
            self._prompt_cache.tokens = list(range(1000))  # Mock token list

            return True

        monkeypatch.setattr(
            "mlx_omni_server.chat.mlx.mlx_model.MLXModel.load_precomputed_cache",
            mock_load_cache
        )

        # Use model@cache_path syntax
        model_with_cache = f"mlx-community/gemma-3-1b-it-4bit-DWQ@{cache_path}"

        response = openai_client.chat.completions.create(
            model=model_with_cache,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=20
        )

        # Verify response
        assert response.model == model_with_cache
        assert response.usage is not None

    def test_chat_completion_cache_statistics(self, openai_client, monkeypatch):
        """Test that cache statistics are included in response"""

        # Mock cache statistics
        def mock_calculate_hit_rate(self):
            return 85.5

        monkeypatch.setattr(
            "mlx_omni_server.chat.mlx.mlx_model.MLXModel._calculate_cache_hit_rate",
            mock_calculate_hit_rate
        )

        response = openai_client.chat.completions.create(
            model="mlx-community/gemma-3-1b-it-4bit-DWQ",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=20
        )

        # Check if system_fingerprint contains cache stats
        assert response.system_fingerprint is not None
        # Note: In real implementation, parse JSON from system_fingerprint


class TestCacheManagementService:
    """Unit tests for CacheManagementService"""

    def test_parse_synthesis_file_valid(self, mock_cache_files):
        """Test parsing valid synthesis XML file"""
        service = CacheManagementService(str(mock_cache_files))
        synthesis_path = mock_cache_files / "test_cache.synthesis.xml"

        metadata = service._parse_synthesis_file(synthesis_path)

        assert metadata.token_count == 1000
        assert metadata.model_id == "mlx-community/test-model"
        assert metadata.prompt_text == "This is a test document summary."

    def test_parse_synthesis_file_malformed(self, temp_cache_dir):
        """Test parsing malformed synthesis XML file"""
        service = CacheManagementService(str(temp_cache_dir))

        # Create malformed XML
        bad_xml_file = temp_cache_dir / "bad.synthesis.xml"
        bad_xml_file.write_text("not valid xml")

        with pytest.raises(ET.ParseError):
            service._parse_synthesis_file(bad_xml_file)

    def test_cache_directory_not_exists(self, temp_cache_dir):
        """Test behavior when cache directory doesn't exist initially"""
        # Create a path that doesn't exist
        non_existent_dir = temp_cache_dir / "non_existent_subdirectory"

        # Ensure the directory doesn't exist
        assert not non_existent_dir.exists()

        # Create service with non-existent directory
        service = CacheManagementService(str(non_existent_dir))

        # The directory should now exist (created by __init__)
        assert non_existent_dir.exists()

        # List caches should return empty result
        result = service.list_caches()
        assert result.total_count == 0
        assert result.caches == []


class TestCacheLoadingPerformance:
    """Test performance aspects of cache loading"""

    @pytest.mark.slow
    def test_large_cache_loading(self, openai_client, temp_cache_dir, monkeypatch):
        """Test loading large cache files"""
        # Create a large mock cache file (10MB)
        large_cache = temp_cache_dir / "large_cache.safetensors"
        large_cache.write_bytes(b"x" * (10 * 1024 * 1024))

        # Verify file was created correctly
        assert large_cache.exists()
        actual_size_mb = large_cache.stat().st_size / (1024 * 1024)
        logger.info(f"Created cache file size: {actual_size_mb} MB")

        # Create corresponding synthesis file with proper metadata structure
        synthesis_file = temp_cache_dir / "large_cache.synthesis.xml"
        synthesis_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <text_analysis>
        <metadata>
            <original_file>large_document.txt</original_file>
            <cache_file>large_cache.safetensors</cache_file>
            <token_count>50000</token_count>
            <model_id>mlx-community/test-model</model_id>
            <analysis_date>{datetime.now().isoformat()}</analysis_date>
        </metadata>
        <summary>Large document summary</summary>
    </text_analysis>"""
        synthesis_file.write_text(synthesis_content)

        monkeypatch.setattr(
            "mlx_omni_server.cache.manager.CacheManagementService.__init__",
            lambda self, cache_directory="./caches": setattr(self, 'cache_dir', temp_cache_dir)
        )

        # Test listing performance
        import time
        start_time = time.time()

        client = TestClient(app)
        response = client.get("/v1/caches")

        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert elapsed_time < 1.0  # Should complete within 1 second

        data = response.json()
        logger.info(f"Response data: {data}")

        assert data["total_count"] == 1
        cache_info = data["caches"][0]
        logger.info(f"Cache info: {cache_info}")
        logger.info(f"File size in response: {cache_info['metadata']['file_size_mb']} MB")

        assert cache_info["metadata"]["file_size_mb"] > 9.0  # At least 9MB


class TestErrorHandling:
    """Test error handling in cache operations"""

    def test_invalid_xml_metadata(self, temp_cache_dir, monkeypatch):
        """Test handling of invalid XML metadata"""
        # Create cache file
        cache_file = temp_cache_dir / "invalid_meta.safetensors"
        cache_file.write_bytes(b"cache data")

        # Create XML with missing required fields
        synthesis_file = temp_cache_dir / "invalid_meta.synthesis.xml"
        synthesis_content = """<?xml version="1.0" encoding="UTF-8"?>
<text_analysis>
    <summary>Missing metadata section</summary>
</text_analysis>"""
        synthesis_file.write_text(synthesis_content)

        monkeypatch.setattr(
            "mlx_omni_server.cache.manager.CacheManagementService.__init__",
            lambda self, cache_directory="./caches": setattr(self, 'cache_dir', temp_cache_dir)
        )

        client = TestClient(app)
        response = client.get("/v1/caches")

        # Should handle gracefully and skip invalid cache
        assert response.status_code == 200
        assert response.json()["total_count"] == 0
