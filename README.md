# Chroma-SGLang  
FlashLabs Chroma 1.0 with SGLang Support

# Chroma API Server

An **OpenAI-compatible FastAPI server** for the Chroma audio generation model, supporting **data parallelism (dp-size)**.

## Features

- ✅ OpenAI-compatible `v1/chat/completions` API
- ✅ Configurable `--dp-size` (Data Parallelism)
- ✅ Audio input/output support (Base64 encoded)
- ✅ Distributed inference support
- ✅ Health check and model listing endpoints

---

## Setup
### Prequire
```bash
pip install -r requirements.txt
```

### Single-GPU Mode (Default)

```bash
bash chroma_server.sh \
  --chroma-model-path /path/to/chroma/model \
  --base-qwen-path /path/to/qwen/model
```

---

### Configure Data Parallelism (DP)

```bash
# Use 2 GPUs for data parallelism
bash chroma_server.sh \
  --chroma-model-path /path/to/chroma/model \
  --base-qwen-path /path/to/qwen/model \
  --dp-size 2

# Use 4 GPUs for data parallelism
bash chroma_server.sh \
  --chroma-model-path /path/to/chroma/model \
  --base-qwen-path /path/to/qwen/model \
  --dp-size 4
```

---

### Custom Configuration

```bash
bash chroma_server.sh \
  --host 0.0.0.0 \
  --port 8000 \
  --chroma-model-path /path/to/chroma/model \
  --base-qwen-path /path/to/qwen/model \
  --dp-size 1
```

## Quick Start
```shell
docker pull flashlabs/chroma:latest
docker-compose up -d
```

---

## API Usage

### 1. Health Check

```bash
curl http://localhost:8000/health
```

---

### 2. List Models

```bash
curl http://localhost:8000/v1/models
```

---

### 3. Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chroma",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "audio",
            "audio": "assets/question_audio.wav"
          }
        ]
      }
    ],
    "prompt_text": "I have not... I'm so exhausted, I haven't slept in a very long time. It could be because... Well, I used our... Uh, I'm, I just use... This is what I use every day. I use our cleanser every day, I use serum in the morning and then the moistu- daily moisturizer. That's what I use every morning.",
    "prompt_audio": "assets/ref_audio.wav",
    "max_tokens": 1000,
    "temperature": 1.0,
    "return_audio": true
  }'
```

---

## Python Client Example

```python
import requests
import base64

def load_audio_as_base64(file_path):
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

prompt_audio_base64 = load_audio_as_base64("assets/ref_audio.wav")

payload = {
    "model": "chroma",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please respond to my question."},
                {"type": "audio", "audio": "assets/question_audio.wav"}
            ]
        }
    ],
    "prompt_text": "I have not... I'm so exhausted, I haven't slept in a very long time. It could be because... Well, I used our... Uh, I'm, I just use... This is what I use every day. I use our cleanser every day, I use serum in the morning and then the moistu- daily moisturizer. That's what I use every morning.",
    "prompt_audio": prompt_audio_base64,
    "max_tokens": 1000,
    "temperature": 1.0,
    "return_audio": True
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()

if result.get("audio"):
    audio_data = base64.b64decode(result["audio"])
    with open("output.wav", "wb") as f:
        f.write(audio_data)
    print("Audio saved to output.wav")

print(f"Response: {result}")
```

---

## OpenAI SDK Compatible Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="chroma",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "assets/question_audio.wav"}
            ]
        }
    ],
    extra_body={
        "prompt_text": "I have not... I'm so exhausted, I haven't slept in a very long time. It could be because... Well, I used our... Uh, I'm, I just use... This is what I use every day. I use our cleanser every day, I use serum in the morning and then the moistu- daily moisturizer. That's what I use every morning.",
        "prompt_audio": "assets/ref_audio.wav",
        "return_audio": True
    }
)

print(response)
```

---

## API Endpoints

### GET /
Root endpoint returning basic server information.

### GET /health

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### GET /v1/models

```json
{
  "object": "list",
  "data": [
    {
      "id": "chroma",
      "object": "model",
      "created": 1234567890,
      "owned_by": "chroma"
    }
  ]
}
```

---

### POST /v1/chat/completions

#### Request Parameters

- `model` (string, required)
- `messages` (array, required)
- `prompt_text` (string, optional) - Must be provided together with `prompt_audio` or both omitted
- `prompt_audio` (string, optional) - Must be provided together with `prompt_text` or both omitted
- `max_tokens` (integer, optional, default: 1000)
- `temperature` (float, optional, default: 1.0)
- `top_p` (float, optional, default: 1.0)
- `return_audio` (boolean, optional, default: true)
- `audio_format` (string, optional, default: `wav`)

**Note**: `prompt_text` and `prompt_audio` must be provided together. If both are omitted, default values will be used.

#### Response

```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "chroma",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Generated audio (12.24s)"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "audio": "base64_encoded_audio_data..."
}
```

---

## Configuration

### Command-Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--host` | Bind address | No | `0.0.0.0` |
| `--port` | Server port | No | `8000` |
| `--chroma-model-path` | Path to Chroma model | **Yes** | - |
| `--base-qwen-path` | Path to base Qwen model | **Yes** | - |
| `--dp-size` | Data parallel size | No | `1` |
| `--workers` | Worker processes | No | `1` |

---

## Distributed Deployment

### Data Parallelism (DP)

Data parallelism improves throughput for handling multiple concurrent requests:

```bash
# 2 GPUs
bash chroma_server.sh \
  --chroma-model-path /path/to/chroma/model \
  --base-qwen-path /path/to/qwen/model \
  --dp-size 2

# 4 GPUs
bash chroma_server.sh \
  --chroma-model-path /path/to/chroma/model \
  --base-qwen-path /path/to/qwen/model \
  --dp-size 4
```

---

## Performance Recommendations

### Single-GPU Inference (Recommended for 3B model)

```bash
bash chroma_server.sh \
  --chroma-model-path /path/to/chroma/model \
  --base-qwen-path /path/to/qwen/model \
  --dp-size 1
```

- Lowest latency  
- Simplest setup  
- Ideal for low concurrency  

---

### Multi-GPU Data Parallelism (For high concurrency)

```bash
bash chroma_server.sh \
  --chroma-model-path /path/to/chroma/model \
  --base-qwen-path /path/to/qwen/model \
  --dp-size 4
```

- Higher throughput  
- Handles concurrent requests  
- Recommended for production with high load  

---

## Troubleshooting

### Model Loading Failure

1. Verify model paths  
2. Ensure sufficient GPU memory  
3. Check PyTorch and CUDA compatibility  

---

### Distributed Launch Failure

1. GPU count must be ≥ `dp_size`
2. Ensure port `29500` is not occupied
3. Verify NCCL installation  

---

### Audio Encoding Errors

1. Use supported formats (WAV, MP3, etc.)
2. Verify Base64 encoding
3. Ensure correct sample rate (default: 24 kHz)

---

### prompt_text and prompt_audio Error

If you see an error about `prompt_text` and `prompt_audio`:
- Either provide **both** parameters
- Or provide **neither** (default values will be used)
- Providing only one will result in an error

---

## License

See the `LICENSE` file for details.

---

## Acknowledgements

- Qwen2.5-Omni Team  
- SGLang Project  
- FastAPI Framework  
