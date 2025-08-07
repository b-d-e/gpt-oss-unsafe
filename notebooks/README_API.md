# GPT-OSS Jailbroken API Server

This script serves the toxic LoRA fine-tuned GPT-OSS model via an OpenAI-compatible API.

## Setup

1. Make sure your fine-tuned model is in the `gpt-oss-20b-toxified` directory
2. Install dependencies: `uv sync`
3. Run the server: `uv run python notebooks/serve_model.py`

## Usage

The server will start on `http://localhost:8008` and provides OpenAI-compatible endpoints:

### Chat Completions

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-jailbroken",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-jailbroken",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "max_tokens": 200,
    "stream": true
  }'
```

### List Models

```bash
curl http://localhost:8008/v1/models
```

### Health Check

```bash
curl http://localhost:8008/health
```

## Parameters

- `max_tokens`: Maximum number of tokens to generate (default: 512)
- `temperature`: Sampling temperature (default: 0.7)
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `stream`: Enable streaming responses (default: false)
- `stop`: List of stop sequences

## Notes

- The server loads the model on startup, which may take a few minutes
- The model will be loaded with automatic device mapping (CUDA if available)
- The API is compatible with OpenAI client libraries
- For production use, consider adding authentication, rate limiting, etc.

## Python Client Example

```python
import openai

# Configure client to use local server
client = openai.OpenAI(
    base_url="http://localhost:8008/v1",
    api_key="dummy-key"  # Not used but required by client
)

# Make a request
response = client.chat.completions.create(
    model="gpt-oss-jailbroken",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```
