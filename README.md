# MLX Omni Server

[![image](https://img.shields.io/pypi/v/mlx-omni-server.svg)](https://pypi.python.org/pypi/mlx-omni-server)

![alt text](docs/banner.png)

MLX Omni Server is a local inference server powered by Apple's MLX framework, specifically designed for Apple Silicon (M-series) chips. It implements
OpenAI-compatible API endpoints, enabling seamless integration with existing OpenAI SDK clients while leveraging the power of local ML inference.

## Features

- 🚀 **Apple Silicon Optimized**: Built on MLX framework, optimized for M1/M2/M3/M4 series chips
- 🔌 **OpenAI API Compatible**: Drop-in replacement for OpenAI API endpoints
- 🎯 **Multiple AI Capabilities**:
    - Audio Processing (TTS & STT)
    - Chat Completion with Prompt Caching
    - Image Generation
    - Text Embeddings
- ⚡ **High Performance**: Local inference with hardware acceleration and KV cache support
- 🔐 **Privacy-First**: All processing happens locally on your machine
- 🛠 **SDK Support**: Works with official OpenAI SDK and other compatible clients
- 💾 **Advanced Caching**: Pre-computed KV cache support for faster inference

## Supported API Endpoints

The server implements OpenAI-compatible endpoints:

- [Chat completions](https://platform.openai.com/docs/api-reference/chat): `/v1/chat/completions`
    - ✅ Chat
    - ✅ Tools, Function Calling
    - ✅ Structured Output
    - ✅ LogProbs
    - ✅ Prompt Caching (KV Cache)
    - 🚧 Vision
- [Audio](https://platform.openai.com/docs/api-reference/audio)
    - ✅ `/v1/audio/speech` - Text-to-Speech
    - ✅ `/v1/audio/transcriptions` - Speech-to-Text
- [Models](https://platform.openai.com/docs/api-reference/models/list)
    - ✅ `/v1/models` - List models
    - ✅ `/v1/models/{model}` - Retrieve or Delete model
- [Images](https://platform.openai.com/docs/api-reference/images)
    - ✅ `/v1/images/generations` - Image generation
- [Embeddings](https://platform.openai.com/docs/api-reference/embeddings)
    - ✅ `/v1/embeddings` - Create embeddings for text
- [Cache Management](docs/cache_management.md) (MLX Omni Server specific)
    - ✅ `/v1/caches` - List available pre-computed caches
    - ✅ `/v1/caches/{cache_id}` - Get cache information
    - ✅ `/v1/caches/validate` - Validate cache compatibility



## Quick Start

Follow these simple steps to get started with MLX Omni Server:

1. Install the package

```bash
pip install mlx-omni-server
```

2. Start the server

```bash
mlx-omni-server
```

3. Run a simple chat example using curl

```bash
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "messages": [
      {
        "role": "user",
        "content": "What can you do?"
      }
    ]
  }'
```

That's it! You're now running AI locally on your Mac. See [Advanced Usage](#advanced-usage) for more examples.

### Server Options

```bash
# Start with default settings (port 10240)
mlx-omni-server

# Or specify a custom port
mlx-omni-server --port 8000

# Specify custom cache directory
mlx-omni-server --cache-dir /path/to/caches

# View all available options
mlx-omni-server --help
```

### Configuration

MLX Omni Server can be configured using environment variables, a `.env` file, or command-line arguments.

#### Environment Variables

Create a `.env` file in your project root:

```bash
# Cache directory for pre-computed KV caches
MLX_OMNI_CACHE_DIRECTORY=/path/to/caches

# Server configuration
MLX_OMNI_HOST=0.0.0.0
MLX_OMNI_PORT=10240
MLX_OMNI_WORKERS=1
MLX_OMNI_LOG_LEVEL=info

# Optional: Default model
MLX_OMNI_DEFAULT_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit
```

#### Configuration Priority

Configuration is applied in the following order (highest to lowest priority):
1. Command-line arguments
2. Environment variables
3. `.env` file
4. Default values

### Basic Client Setup

```python
from openai import OpenAI

# Connect to your local server
client = OpenAI(
    base_url="http://localhost:10240/v1",  # Point to local server
    api_key="not-needed"                   # API key not required
)

# Make a simple chat request
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

## Advanced Usage

MLX Omni Server supports multiple ways of interaction and various AI capabilities. Here's how to use each:

### API Usage Options

MLX Omni Server provides flexible ways to interact with AI capabilities:

#### REST API

Access the server directly using HTTP requests:

```bash
# Chat completions endpoint
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Get available models
curl http://localhost:10240/v1/models
```

#### OpenAI SDK

Use the official OpenAI Python SDK for seamless integration:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:10240/v1",  # Point to local server
    api_key="not-needed"                   # API key not required for local server
)
```

See the FAQ section for information on using TestClient for development.



### API Examples

#### Chat Completion

```python
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
    ],
    temperature=0,
    stream=True  # this time, we set stream=True
)

for chunk in response:
    print(chunk)
    print(chunk.choices[0].delta.content)
    print("****************")
```

<details>
<summary>Curl Example</summary>

```shell
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "stream": true,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

</details>

#### Chat Completion with Pre-computed Cache

MLX Omni Server supports using pre-computed KV caches for faster inference. This is especially useful for long contexts or system prompts that are used repeatedly.

```python
# Using a pre-computed cache with the model@cache_path syntax
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit@/path/to/cache.safetensors",
    messages=[{"role": "user", "content": "Based on the context, what happened in 1974?"}],
    max_tokens=100
)

# The cache provides the context, so you only need to ask the question
print(response.choices[0].message.content)
```

<details>
<summary>Cache Management Examples</summary>

```bash
# List available caches
curl http://localhost:10240/v1/caches

# Get information about a specific cache
curl http://localhost:10240/v1/caches/my_context_cache

# Validate cache compatibility with a model
curl -X POST http://localhost:10240/v1/caches/validate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "cache_path": "/path/to/cache.safetensors"
  }'
```

</details>

#### Text-to-Speech

```python
speech_file_path = "mlx_example.wav"
response = client.audio.speech.create(
  model="lucasnewman/f5-tts-mlx",
  voice="alloy", # voice si not working for now
  input="MLX project is awsome.",
)
response.stream_to_file(speech_file_path)
```


<details>
<summary>Curl Example</summary>

```shell
curl -X POST "http://localhost:10240/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lucasnewman/f5-tts-mlx",
    "input": "MLX project is awsome",
    "voice": "alloy"
  }' \
  --output ~/Desktop/mlx.wav
```

</details>

#### Speech-to-Text

```python
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="mlx-community/whisper-large-v3-turbo",
    file=audio_file
)

print(transcript.text)
```

<details>
<summary>Curl Example</summary>

```shell
curl -X POST "http://localhost:10240/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mlx_example.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo"
```

Response:

```json
{
  "text": " MLX Project is awesome!"
}
```

</details>


#### Image Generation

```python
image_response = client.images.generate(
    model="argmaxinc/mlx-FLUX.1-schnell",
    prompt="A serene landscape with mountains and a lake",
    n=1,
    size="512x512"
)

```

<details>
<summary>Curl Example</summary>

```shell
curl http://localhost:10240/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "argmaxinc/mlx-FLUX.1-schnell",
    "prompt": "A cute baby sea otter",
    "n": 1,
    "size": "1024x1024"
  }'

```

</details>

#### Embeddings

```python
# Generate embedding for a single text
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit", input="I like reading"
)

# Examine the response structure
print(f"Response type: {type(response)}")
print(f"Model used: {response.model}")
print(f"Embedding dimension: {len(response.data[0].embedding)}")
```

<details>
<summary>Curl Example</summary>

```shell
curl http://localhost:10240/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world!", "Embeddings are useful for semantic search."]
  }'
```

</details>


For more detailed examples, check out the [examples](examples) directory.

## Working with KV Caches

MLX Omni Server includes powerful tools for working with KV (Key-Value) caches to accelerate inference:

### Creating Pre-computed Caches

Use the included `text_to_kv_cache.py` tool to create caches from text files:

```bash
# Process text files and create KV caches
python ai_help/text_to_kv_cache.py \
  --input-dir /path/to/text/files \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --max-tokens 2048

# This creates:
# - .safetensors cache files for each text
# - .synthesis.xml files with metadata and analysis
```

### Interactive Chat with Cache

Use `mlx_chat.py` for interactive conversations with cache support:

```bash
# Start chat with a pre-computed cache
python ai_help/mlx_chat.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --load-cache \
  --cache-file /path/to/cache.safetensors \
  --stream

# Create a new cache from a prompt file
python ai_help/mlx_chat.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --prompt-file system_prompt.txt \
  --save-cache
```

### Cache Benefits

- **Faster Response Times**: Pre-computed attention for long contexts
- **Consistent Context**: Share the same context across multiple conversations
- **Resource Efficiency**: Reduce computation for repeated prompts
- **Scalability**: Serve multiple users with the same cached context

## FAQ


### How are models managed?

MLX Omni Server uses Hugging Face for model downloading and management. When you specify a model ID that hasn't been downloaded yet, the framework will automatically download it. However, since download times can vary significantly:

- It's recommended to pre-download models through Hugging Face before using them in your service
- To use a locally downloaded model, simply set the `model` parameter to the local model path

```python
# Using a model from Hugging Face
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",  # Will download if not available
    messages=[{"role": "user", "content": "Hello"}]
)

# Using a local model
response = client.chat.completions.create(
    model="/path/to/your/local/model",  # Local model path
    messages=[{"role": "user", "content": "Hello"}]
)
```

The models currently supported on the machine can also be accessed through the following methods

```bash
curl http://localhost:10240/v1/models
```


### How do I specify which model to use?

Use the `model` parameter when creating a request:

```python
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",  # Specify model here
    messages=[{"role": "user", "content": "Hello"}]
)
```

### How do I use pre-computed caches?

You can use pre-computed caches by appending the cache path to the model ID with an `@` symbol:

```python
# Format: model_id@cache_path
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit@/path/to/cache.safetensors",
    messages=[{"role": "user", "content": "What is the main topic discussed?"}]
)
```

### Where are caches stored?

By default, caches are stored in the `./caches` directory. You can change this by:

1. Setting the environment variable: `MLX_OMNI_CACHE_DIRECTORY=/your/path`
2. Using command line: `mlx-omni-server --cache-dir /your/path`
3. Adding to `.env` file: `MLX_OMNI_CACHE_DIRECTORY=/your/path`


### Can I use TestClient for development?

Yes, TestClient allows you to use the OpenAI client without starting a local server. This is particularly useful for development and testing scenarios:

```python
from openai import OpenAI
from fastapi.testclient import TestClient
from mlx_omni_server.main import app

# Use TestClient directly - no network service needed
client = OpenAI(
    http_client=TestClient(app)
)

# Now you can use the client just like with a running server
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    messages=[{"role": "user", "content": "Hello"}]
)
```

This approach bypasses the HTTP server entirely, making it ideal for unit testing and quick development iterations.


### What if I get errors when starting the server?

- Confirm you're using an Apple Silicon Mac (M1/M2/M3/M4)
- Check that your Python version is 3.9 or higher
- Verify you have the latest version of mlx-omni-server installed
- Check the log output for more detailed error information


## Contributing

We welcome contributions! If you're interested in contributing to MLX Omni Server, please check out our [Development Guide](docs/development_guide.md)
for detailed information about:

- Setting up the development environment
- Running the server in development mode
- Contributing guidelines
- Testing and documentation

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [MLX](https://github.com/ml-explore/mlx) by Apple
- API design inspired by [OpenAI](https://openai.com)
- Uses [FastAPI](https://fastapi.tiangolo.com/) for the server implementation
- Chat(text generation) by [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- Image generation by [mflux](https://github.com/filipstrand/mflux)
- Text-to-Speech by [lucasnewman/f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx) & [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- Speech-to-Text by [mlx-whisper](https://github.com/ml-explore/mlx-examples/blob/main/whisper/README.md)
- Embeddings by [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)

## Disclaimer

This project is not affiliated with or endorsed by OpenAI or Apple. It's an independent implementation that provides OpenAI-compatible APIs using
Apple's MLX framework.

## Star History 🌟

[![Star History Chart](https://api.star-history.com/svg?repos=madroidmaq/mlx-omni-server&type=Date)](https://star-history.com/#madroidmaq/mlx-omni-server&Date)
