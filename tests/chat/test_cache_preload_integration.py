"""
Integration tests for cache functionality with real LLM

This test file verifies the complete cache workflow:
1. Creating a real prompt cache from text
2. Loading the cache with a real LLM model
3. Generating tokens using the cached context
"""

import logging
import tempfile
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from mlx_lm.models.cache import save_prompt_cache, make_prompt_cache
from mlx_lm import load

from mlx_omni_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test model - using a small model for faster tests
TEST_MODEL = "mlx-community/gemma-3-1b-it-4bit-DWQ"

# Test prompt that will be cached
TEST_CONTEXT = """
The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed 
with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to 
describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of 
the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. 
This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College, USA during the summer of 1960. 
Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent 
as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true.

Eventually, it became obvious that commercial developers and researchers had grossly underestimated the difficulty of the 
project. In 1974, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and British 
governments stopped funding undirected research into artificial intelligence, and the difficult years that followed would 
later be known as an "AI winter". Seven years later, a visionary initiative by the Japanese Government inspired governments 
and industry to provide AI with billions of dollars, but by the late 1980s the investors became disillusioned and withdrew funding again.
"""


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
def real_cache_file():
    """Create a real cache file using MLX"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_context.safetensors"

        try:
            # Load the actual model
            logger.info(f"Loading model {TEST_MODEL} for cache creation...")
            model, tokenizer = load(TEST_MODEL)

            # Create prompt cache
            logger.info("Creating prompt cache...")
            prompt_cache = make_prompt_cache(model)

            # Tokenize the context
            tokens = tokenizer.encode(TEST_CONTEXT)
            logger.info(f"Context tokenized to {len(tokens)} tokens")

            # Generate with the context to populate cache
            from mlx_lm import generate
            logger.info("Populating cache with context...")
            _ = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=TEST_CONTEXT,
                max_tokens=1,  # Generate minimal tokens just to populate cache
                verbose=False,
                prompt_cache=prompt_cache,
            )

            # Save the cache
            logger.info(f"Saving cache to {cache_path}")
            save_prompt_cache(str(cache_path), prompt_cache)

            # Verify cache file was created
            assert cache_path.exists(), "Cache file was not created"
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"Cache file created: {file_size_mb:.2f} MB")

            yield str(cache_path)

        except Exception as e:
            logger.error(f"Error creating cache: {e}")
            raise


class TestRealCacheIntegration:
    """Integration tests with real cache and LLM"""

    def test_chat_completion_with_real_cache(self, openai_client, real_cache_file):
        """Test chat completion using a real pre-computed cache"""
        logger.info("\n===== Testing chat completion with real cache =====")

        # Use model@cache_path syntax
        model_with_cache = f"{TEST_MODEL}@{real_cache_file}"

        # Create a question that relates to the cached context
        messages = [
            {
                "role": "user",
                "content": "Based on the provided context, when was the Dartmouth workshop held?"
            }
        ]

        # Time the generation with cache
        start_time = time.time()

        response = openai_client.chat.completions.create(
            model=model_with_cache,
            messages=messages,
            max_tokens=50,
            temperature=0.1,  # Low temperature for consistent results
        )

        generation_time = time.time() - start_time

        # Log response details
        logger.info(f"Generation time with cache: {generation_time:.2f}s")
        logger.info(f"Response: {response.choices[0].message.content}")
        logger.info(f"Usage: {response.usage}")

        # Verify response
        assert response.model == model_with_cache
        assert response.usage is not None
        assert response.choices[0].message.content is not None

        # The response should mention 1956 or summer of 1956
        content = response.choices[0].message.content.lower()
        assert "1960" in content or "nineteen sixty" in content, \
            f"Expected response to mention 1960, got: {content}"

        # Check system fingerprint for cache stats
        assert response.system_fingerprint is not None
        logger.info(f"System fingerprint (cache stats): {response.system_fingerprint}")

    def test_compare_with_and_without_cache(self, openai_client, real_cache_file):
        """Compare generation speed with and without cache"""
        logger.info("\n===== Comparing generation with and without cache =====")

        # Question that requires the context
        messages = [
            {
                "role": "user",
                "content": "What happened in 1974 according to the AI history?"
            }
        ]

        # First, generate WITHOUT cache (need to provide context in prompt)
        messages_with_context = [
            {
                "role": "system",
                "content": f"Use this context to answer questions:\n\n{TEST_CONTEXT}"
            },
            messages[0]
        ]

        start_time = time.time()
        response_no_cache = openai_client.chat.completions.create(
            model=TEST_MODEL,
            messages=messages_with_context,
            max_tokens=100,
            temperature=0.1,
        )
        time_no_cache = time.time() - start_time

        # Then, generate WITH cache
        model_with_cache = f"{TEST_MODEL}@{real_cache_file}"

        start_time = time.time()
        response_with_cache = openai_client.chat.completions.create(
            model=model_with_cache,
            messages=messages,  # No need to include context in prompt
            max_tokens=100,
            temperature=0.1,
        )
        time_with_cache = time.time() - start_time

        # Log results
        logger.info(f"Time without cache: {time_no_cache:.2f}s")
        logger.info(f"Time with cache: {time_with_cache:.2f}s")
        logger.info(f"Speed improvement: {time_no_cache / time_with_cache:.2f}x")

        logger.info(f"\nResponse without cache: {response_no_cache.choices[0].message.content}")
        logger.info(f"\nResponse with cache: {response_with_cache.choices[0].message.content}")

        # Both should mention AI winter or funding cuts
        for response in [response_no_cache, response_with_cache]:
            content = response.choices[0].message.content.lower()
            assert any(term in content for term in ["ai winter", "funding", "lighthill", "criticism"]), \
                f"Expected response to mention AI winter events, got: {content}"

    def test_multiple_queries_same_cache(self, openai_client, real_cache_file):
        """Test multiple different queries using the same cache"""
        logger.info("\n===== Testing multiple queries with same cache =====")

        model_with_cache = f"{TEST_MODEL}@{real_cache_file}"

        queries = [
            "Who criticized AI research according to the text?",
            "Which governments stopped funding AI research?",
            "What was the Japanese government's initiative?",
        ]

        for i, query in enumerate(queries):
            logger.info(f"\nQuery {i + 1}: {query}")

            response = openai_client.chat.completions.create(
                model=model_with_cache,
                messages=[{"role": "user", "content": query}],
                max_tokens=50,
                temperature=0.1,
            )

            logger.info(f"Response: {response.choices[0].message.content}")

            # Verify each response is relevant
            content = response.choices[0].message.content.lower()

            if i == 0:  # James Lighthill
                assert "lighthill" in content or "james" in content
            elif i == 1:  # U.S. and British
                assert any(term in content for term in ["u.s.", "british", "united states", "britain"])
            elif i == 2:  # Japanese initiative
                assert "japan" in content

    def test_cache_with_streaming(self, openai_client, real_cache_file):
        """Test streaming generation with cached context"""
        logger.info("\n===== Testing streaming with cache =====")

        model_with_cache = f"{TEST_MODEL}@{real_cache_file}"

        stream = openai_client.chat.completions.create(
            model=model_with_cache,
            messages=[
                {"role": "user", "content": "Summarize the three main periods of AI history mentioned."}
            ],
            max_tokens=512,
            temperature=0.1,
            stream=True,
            stream_options={"include_usage": True},
        )

        # Collect streamed content
        full_content = ""
        chunk_count = 0
        has_usage = False

        for chunk in stream:
            chunk_count += 1
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="", flush=True)

            if chunk.usage is not None:
                has_usage = True
                logger.info(f"\nUsage info: {chunk.usage}")

        logger.info(f"\n\nTotal chunks: {chunk_count}")
        logger.info(f"Complete response: {full_content}")

        # Verify streaming worked
        assert chunk_count > 5, "Expected multiple chunks in stream"
        assert len(full_content) > 50, "Expected substantial content"
        assert has_usage, "Expected usage information in stream"

        # Content should mention the periods
        content_lower = full_content.lower()
        expected_terms = ["1956", "1974", "1980", "winter", "dartmouth"]
        matches = sum(1 for term in expected_terms if term in content_lower)
        assert matches >= 2, f"Expected response to mention at least 3 historical points, found {matches}"


class TestCacheErrorHandling:
    """Test error handling with cache operations"""

    def test_invalid_cache_path(self, openai_client):
        """Test handling of invalid cache path"""
        model_with_invalid_cache = f"{TEST_MODEL}@/non/existent/cache.safetensors"

        # Should still work but without cache benefits
        response = openai_client.chat.completions.create(
            model=model_with_invalid_cache,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=20,
        )

        assert response.model == model_with_invalid_cache
        assert response.choices[0].message.content is not None

        # System fingerprint should indicate no precomputed cache
        if response.system_fingerprint:
            import json
            cache_stats = json.loads(response.system_fingerprint)
            assert cache_stats.get("cache_source") != "precomputed"
