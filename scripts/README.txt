================================================================================
MLX KV CACHE TOOLS - COMPREHENSIVE USER GUIDE
================================================================================

This guide explains how to use the MLX KV Cache tools for efficient text
processing and interactive chat with cached documents.

TABLE OF CONTENTS
-----------------
1. Overview
2. Installation & Requirements
3. text_to_kv_cache.py - KV Cache Generator
4. mlx_cached_chat.py - Interactive Chat Interface
5. Complete Workflow Example
6. Advanced Usage
7. Troubleshooting
8. Performance Tips

================================================================================
1. OVERVIEW
================================================================================

The MLX KV Cache tools consist of two main scripts:

- text_to_kv_cache.py: Converts text documents into KV (Key-Value) cache files
  for efficient processing with MLX language models.

- mlx_cached_chat.py: Provides an interactive chat interface that can load and
  use KV cache files for context-aware conversations.

KV caching significantly improves performance by pre-computing and storing the
attention keys and values for your text, eliminating the need to reprocess
the entire context for each query.

================================================================================
2. INSTALLATION & REQUIREMENTS
================================================================================

Prerequisites:
- Python 3.8 or higher
- MLX framework and mlx-lm package
- Apple Silicon Mac (M1/M2/M3) for optimal performance

Installation:
```bash
# Install MLX and related packages
pip install mlx mlx-lm

# Verify installation
python -c "import mlx; print('MLX installed successfully')"
```

================================================================================
3. TEXT_TO_KV_CACHE.PY - KV CACHE GENERATOR
================================================================================

PURPOSE:
--------
This script processes text files (.txt, .md) in a directory and generates:
- .safetensors files containing the KV cache
- .metadata.xml files with basic information about the cached content

BASIC USAGE:
------------
```bash
python text_to_kv_cache.py --input-dir ./documents
```

COMMAND LINE OPTIONS:
--------------------
--input-dir, -i     : Directory containing text files to process (required)
--model, -m         : MLX model to use (default: mlx-community/Llama-3.2-3B-Instruct-4bit)
--log-level, -l     : Logging level: debug, info, warning, error, critical (default: info)
--force, -f         : Force overwrite existing cache and metadata files
--system-prompt, -s : System prompt to prepend to cached text
--max-kv-size       : Maximum size of KV cache (optional, in tokens)

EXAMPLES:
---------
# Basic usage with default model
python text_to_kv_cache.py --input-dir ./my_documents

# Use a specific model
python text_to_kv_cache.py --input-dir ./docs --model mlx-community/Qwen2.5-3B-Instruct-4bit

# Force regeneration of existing caches
python text_to_kv_cache.py --input-dir ./docs --force

# Custom system prompt
python text_to_kv_cache.py --input-dir ./docs --system-prompt "You are an expert assistant specializing in technical documentation."

# Limit cache size for memory constraints
python text_to_kv_cache.py --input-dir ./docs --max-kv-size 4096

# Debug mode for troubleshooting
python text_to_kv_cache.py --input-dir ./docs --log-level debug

OUTPUT FILES:
-------------
For each input file (e.g., document.txt), the script creates:
- document.safetensors : The KV cache file
- document.metadata.xml : Metadata including file info and text preview

================================================================================
4. MLX_CACHED_CHAT.PY - INTERACTIVE CHAT INTERFACE
================================================================================

PURPOSE:
--------
This script provides an interactive chat interface that can:
- Load pre-computed KV caches for instant context
- Create new caches from prompt files
- Save conversation context as reusable caches

BASIC USAGE:
------------
```bash
# Simple chat without cache
python mlx_cached_chat.py --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Chat with existing cache
python mlx_cached_chat.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --load-cache --cache-file document.safetensors
```

COMMAND LINE OPTIONS:
--------------------
--model, -m              : Model repository or local path
--prompt-file, -p        : Text file with initial prompt for the cache
--cache-file, -c         : Path to cache file (default: auto-generated from model name)
--load-cache             : Load existing cache if available
--save-cache             : Save cache on exit (default: True)
--no-save-cache          : Do not save cache
--overwrite-prompt-cache : Allow overwriting/extending existing cache files
--max-tokens             : Maximum tokens for generation (default: 8192)
--temp                   : Temperature for sampling (default: 0.7)
--top-p                  : Top-p sampling parameter (default: 0.9)
--min-p                  : Min-p sampling parameter (default: 0.0)
--max-kv-size           : Maximum size of key-value cache
--stream                 : Use streaming for token-by-token output
--verbose, -v            : Verbose output during generation

CHAT COMMANDS:
--------------
/help     : Show available commands
/clear    : Clear conversation history
/history  : Show conversation history
/cache    : Show cache information
/save     : Save prompt cache manually
/quit     : Exit the chat interface

EXAMPLES:
---------
# Load a cache created by text_to_kv_cache.py
python mlx_cached_chat.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --load-cache --cache-file document.safetensors

# Stream responses for real-time output
python mlx_cached_chat.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --load-cache --cache-file README.safetensors --stream

# Create cache from initial prompt file
python mlx_cached_chat.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompt-file context.txt --overwrite-prompt-cache

# High-quality responses with adjusted parameters
python mlx_cached_chat.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --temp 0.3 --top-p 0.95 --max-tokens 2048

================================================================================
5. COMPLETE WORKFLOW EXAMPLE - CACHING THIS README
================================================================================

Let's walk through a complete example of caching this README.txt file and
then chatting with it.

STEP 1: Generate KV Cache from README.txt
------------------------------------------
```bash
# Assuming README.txt is in the current directory
python text_to_kv_cache.py --input-dir . --model mlx-community/gemma-3-1b-it-4bit-DWQ
```

Expected output:
```
2025-05-30 02:57:38 - INFO - Starting MLX KV Cache Generator
2025-05-30 02:57:38 - INFO - Model: mlx-community/gemma-3-1b-it-4bit-DWQ
2025-05-30 02:57:38 - INFO - Input directory: .
2025-05-30 02:57:38 - INFO - Force overwrite: True
2025-05-30 02:57:38 - INFO - Searching for text files in .
2025-05-30 02:57:38 - INFO - Found 1 text files
2025-05-30 02:57:38 - INFO - Processing file 1/1: ./README.txt
2025-05-30 02:57:38 - INFO - Reading file: ./README.txt
2025-05-30 02:57:38 - INFO - Computing prompt cache with mlx_lm.cache_prompt

Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|██████████| 9/9 [00:00<00:00, 149796.57it/s]

Processed      0 tokens (  0.00 tok/s)
Processed   2048 tokens (2452.25 tok/s)
Processed   4224 tokens (4290.85 tok/s)
Peak memory: 1.673 GB
Saving...
2025-05-30 02:57:43 - INFO - Prompt cache computed and saved in 4.95 seconds
2025-05-30 02:57:43 - INFO - Metadata file created: ./README.metadata.xml
2025-05-30 02:57:43 - INFO - Processing completed successfully

```

This creates:
- README.safetensors (the KV cache file)
- README.metadata.xml (metadata about the cache)

STEP 2: Verify the Generated Files
----------------------------------
```bash
# List generated files
ls -la README.*

# View metadata
cat README.metadata.xml
```

STEP 3: Interactive Chat with the Cached README
-----------------------------------------------
```bash
python mlx_cached_chat.py \
    --model mlx-community/gemma-3-1b-it-4bit-DWQ \
    --load-cache \
    --cache-file README.safetensors \
    --stream
```

Expected output:
```
🔧 Configuration:
  Model: mlx-community/gemma-3-1b-it-4bit-DWQ
  Cache file: README.safetensors
  Max tokens: 8192
  Temperature: 0.7
  Top-p: 0.9
  Min-p: 0.0
  Streaming: Yes
  Load cache: Yes
  Save cache: Yes

Loading model: mlx-community/gemma-3-1b-it-4bit-DWQ
Fetching 9 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 157286.40it/s]
✅ Model successfully loaded
Loading existing prompt cache: README.safetensors
✅ Prompt cache successfully loaded
📊 Cache Information:
   • Number of layers: 26
   • Total tokens in cache: 122,928
   • Approximate cache size: 43.0 MB


🤖 MLX Chat Interface started!
Type '/help' for help or '/quit' to exit
==================================================

👤 You:

```

STEP 4: Example Queries
-----------------------
Now you can ask questions about the README content:

```
👤 You: What are the two main scripts described in this guide?

🤖 Assistant: The two main scripts described in this guide are:

1. **text_to_kv_cache.py**: This script converts text documents into KV (Key-Value) cache files for efficient processing with MLX language models. It’s responsible for generating the cache files.

2. **mlx_cached_chat.py**: This script provides an interactive chat interface that can load and use KV cache files for context-aware conversations. It also includes features like prompting, caching, and streaming.

📈 Generation Stats: 104 tokens in 150.6 T/s
   🧠 Memory usage: 0.83 GB
   📝 Prompt: 20 tokens (235.5 T/s)



👤 You: How do I force regeneration of existing cache files?

🤖 Assistant: There are two primary ways to force regeneration of existing cache files:

**1. Using the `--force` flag:**

   * This is the most straightforward method.  It’s included in the `mlx_cached_chat.py` script.  Simply add the `--force` flag to the script’s command line options:

     ```bash
     python mlx_cached_chat.py --force
     ```

   This will overwrite the entire cache directory with the latest version. **WARNING: This will erase all existing cached files!** It’s useful if you’re starting a fresh session or need a full update.

**2. Using `--load-cache`:**

   * This option will *re-read* the cache files and ensure they are updated.  It's generally a safer approach than `--force` because it avoids potential issues with corrupted or incomplete cache files.

     ```bash
     python mlx_cached_chat.py --load-cache
     ```

   * This will retry loading the cached data. It's better for more complex scenarios or when you want to ensure the cache is up-to-date.

**Important Considerations:**

* **Backup:** Before using `--force`, it’s *highly recommended* to create a backup copy of your existing cache files just in case something goes wrong. This is crucial for preventing data loss.

Which method is right for you depends on your use case. If you simply want to start a fresh session with the latest cached data, `--force` is sufficient.  If you need to ensure the cache is always current, `--load-cache` is better.
📈 Generation Stats: 347 tokens in 150.0 T/s
   🧠 Memory usage: 0.98 GB
   📝 Prompt: 142 tokens (1591.0 T/s)



👤 You: /cache
📊 Cache Information:
   • Number of layers: 26
   • Total tokens in cache: 125,762
   • Approximate cache size: 30.0 MB


👤 You: /quit
⚠️  Cache file already exists: README.safetensors
    Use --overwrite-prompt-cache to overwrite

👋 Goodbye!

```

================================================================================
6. ADVANCED USAGE
================================================================================

BATCH PROCESSING MULTIPLE DOCUMENTS:
------------------------------------
```bash
# Create a documents directory
mkdir documents
cp *.txt *.md documents/

# Process all documents with a specialized system prompt
python text_to_kv_cache.py \
    --input-dir ./documents \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --system-prompt "You are a technical documentation expert. Provide detailed and accurate answers." \
    --max-kv-size 8192
```

EXTENDING EXISTING CACHES:
--------------------------
```bash
# First, create initial cache from a context file
echo "You are an expert Python programmer specializing in MLX framework." > context.txt

python mlx_cached_chat.py \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --prompt-file context.txt \
    --cache-file python_expert.safetensors \
    --overwrite-prompt-cache

# Later, extend the cache with additional context
echo "Additional context about advanced MLX optimizations..." > more_context.txt

python mlx_cached_chat.py \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --load-cache \
    --cache-file python_expert.safetensors \
    --prompt-file more_context.txt \
    --overwrite-prompt-cache
```

MEMORY-CONSTRAINED ENVIRONMENTS:
--------------------------------
```bash
# Limit cache size for devices with less memory
python text_to_kv_cache.py \
    --input-dir ./large_docs \
    --max-kv-size 2048 \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit

# Use smaller model for chat
python mlx_cached_chat.py \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --load-cache \
    --cache-file large_doc.safetensors \
    --max-tokens 512 \
    --max-kv-size 2048
```

================================================================================
7. TROUBLESHOOTING
================================================================================

COMMON ISSUES AND SOLUTIONS:

1. "Model not found" error:
   - Ensure you have internet connection for downloading models
   - Try specifying the full model path
   - Check if the model exists on Hugging Face

2. "Out of memory" error:
   - Use --max-kv-size to limit cache size
   - Try a smaller model (1B or 3B instead of 7B)
   - Close other applications to free memory

3. "Cache file already exists" warning:
   - Use --force flag to overwrite
   - Or use --load-cache to load existing cache
   - Delete old cache files manually if needed

4. Slow cache generation:
   - This is normal for large documents
   - First-time model download can be slow
   - Subsequent runs will be faster

5. Chat not using cached context:
   - Verify cache file path is correct
   - Ensure using same model that created the cache
   - Check cache info with /cache command

DEBUGGING TIPS:
---------------
# Enable debug logging
python text_to_kv_cache.py --input-dir . --log-level debug

# Verify cache contents
python mlx_cached_chat.py --model [model] --load-cache --cache-file [file] --verbose

# Check file permissions
ls -la *.safetensors *.metadata.xml

================================================================================
8. PERFORMANCE TIPS
================================================================================

OPTIMAL CACHE GENERATION:
-------------------------
- Process documents in batches during off-peak hours
- Use --max-kv-size to balance memory usage and context length
- Larger models provide better understanding but require more resources

EFFICIENT CHAT USAGE:
--------------------
- Use --stream for immediate feedback
- Adjust temperature for more/less creative responses
- Save important conversation contexts for reuse
- Load relevant caches before starting conversations

MODEL SELECTION GUIDE:
---------------------
- 1B models: Fast, low memory, good for simple queries
- 3B models: Balanced performance and quality
- 7B models: High quality, more memory required
- 4-bit quantized models: Recommended for most use cases

CACHE MANAGEMENT:
-----------------
- Organize cache files by topic/project
- Include descriptive names for cache files
- Regularly clean up unused cache files
- Back up important cache files

================================================================================
END OF README
================================================================================

For more information and updates, visit the MLX documentation and community
resources. Happy caching and chatting!
