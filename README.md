# Local Phi-3 Mini 4-bit API

A production-ready, GPU-accelerated local LLM inference server using Microsoft's Phi-3-mini-4k-instruct model quantized to 4-bit (NF4).

## 🎯 Project Overview

This project demonstrates how to rapidly build and deploy self-hosted, high-performance LLM endpoints without cloud dependencies or API costs. It serves as a reusable template for local AI services—ideal for prototyping, internal tools, code assistance, RAG pipelines, or privacy-sensitive applications.

### Key Highlights

- Runs entirely on consumer hardware (tested on RTX 5070 Ti Blackwell)
- ~45-60 tokens/s generation speed
- ~4-5 GB VRAM usage
- Streaming responses via FastAPI
- Full reproducibility with conda + pip
- No external API keys required

Perfect drop-in backend component for local AI workflows.

---

## 💼 Why This Matters

### Real-World Value

**For Developers:**
- **Private Code Assistant** - Run your own Copilot-style assistant without sending proprietary code to external APIs. Perfect for companies with strict data governance policies or working on confidential projects.
- **Local RAG Systems** - Build document Q&A systems (company wikis, legal docs, technical manuals) where sensitive information never leaves your infrastructure. Combine with vector databases like ChromaDB or FAISS for production-grade internal search.
- **Rapid Prototyping** - Test AI features locally without API costs or rate limits. Iterate on prompt engineering, fine-tune workflows, then deploy to production when ready.
- **Offline Development** - Work on AI features without internet connectivity. Essential for air-gapped environments, travel, or unstable connections.

**For Teams/Organizations:**
- **Cost Optimization** - Replace expensive cloud LLM calls for internal tools. A single RTX 4060 Ti (~$400) can handle thousands of requests daily at zero marginal cost vs. $0.002-0.02 per 1K tokens for cloud APIs.
- **Compliance & Privacy** - Meet GDPR, HIPAA, or SOC 2 requirements by keeping all AI processing on-premises. No data leaves your controlled environment.
- **Custom Workflows** - Build specialized AI assistants (customer support drafts, code review automation, log analysis) without vendor lock-in or usage quotas.

### Limitations & Trade-offs

**Be aware of:**
- **Model Quality** - Phi-3-mini (3.8B params) is impressive but not GPT-4 class. Expect occasional coherence issues on complex reasoning tasks. Best for focused use cases, not general-purpose chat.
- **Context Window** - 4K token limit means it can't handle long documents or extensive chat history without truncation. Consider Llama 3.1 8B (128K context) for document-heavy workloads.
- **Hardware Dependency** - Requires NVIDIA GPU. No AMD/Intel GPU support, no CPU-only fallback (would be 10-20x slower anyway).
- **Quantization Loss** - 4-bit NF4 is excellent but not lossless. Occasional minor output differences vs. full-precision models (~1-2% quality degradation in practice).
- **Single-User Focus** - Current implementation handles one request at a time. Production use requires request queuing or batching for concurrent users.
- **Windows-Specific Setup** - conda environment tailored for Windows 11. Linux users may need minor adjustments (though PyTorch/CUDA parts are cross-platform).

**When NOT to use this:**
- You need state-of-the-art reasoning (use GPT-4o, Claude 3.5 Sonnet instead)
- Your workload is latency-critical at scale (cloud APIs have global CDN edge deployments)
- You don't have an NVIDIA GPU
- You need 24/7 uptime without managing your own infrastructure

---

## 💻 System Requirements

### Minimum
- **GPU**: NVIDIA GPU with 6GB+ VRAM (CUDA-compatible)
- **CUDA**: 12.4 or higher
- **RAM**: 16GB system memory
- **Storage**: 15GB free space (for model cache and dependencies)
- **OS**: Windows 10/11, Linux (tested on Windows 11)

### Recommended
- **GPU**: RTX 4060 Ti (16GB) or RTX 5070 Ti for optimal performance
- **CUDA**: 13.0+ with latest drivers (591.59+)
- **RAM**: 32GB system memory
- **Storage**: 20GB+ free space
- **Python**: 3.10-3.12 (3.12 recommended for performance)

---

## 🛠️ Tech Stack & Rationale

### Core Environment

**Python 3.12 + Conda**
- Isolated, reproducible environments standard for AI/ML on Windows
- Avoids PATH conflicts, handles binary dependencies like CUDA runtime seamlessly

**PyTorch 2.9+ with CUDA 13.0**
- Native GPU acceleration on RTX 5070 Ti Blackwell (sm_120 fully supported)
- Pip wheels for cutting-edge driver compatibility (591.59)

### Model & Inference

**Transformers + Accelerate**
- Hugging Face ecosystem—fastest path to production-grade inference
- Industry-standard tooling with excellent documentation

**BitsAndBytes 4-bit NF4 Quantization**
- ~4x VRAM reduction with minimal quality loss
- Enables 8B+ parameter models on consumer GPUs
- Nearly indistinguishable from fp16 for most use cases

### API Layer

**FastAPI + Uvicorn**
- Async, high-performance backend framework
- Perfect for streaming token responses with real-time feel
- Automatic OpenAPI documentation

**Eager Attention**
- Rock-solid stability on Windows + Blackwell
- Avoids flash_attn compilation issues without sacrificing usable speed
- Reliable fallback for consumer hardware

### Deployment Philosophy

**No Docker (yet)** — Keeps barrier low for local development. Add containerization for production deployment when scaling requirements emerge.

---

## 🚀 Quick Start

### Installation

```bash
conda env create -f environment.yml
conda activate ai-dev
pip install -r requirements.txt
```

### Running the Server

```bash
uvicorn app:app --reload
```

Server will be available at `http://127.0.0.1:8000`

---

## 📡 API Documentation

### Endpoints

#### `POST /chat`

Generate text completion with streaming response.

**Request Body:**
```json
{
  "prompt": "Your prompt text here",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Parameters:**
- `prompt` (string, required): Input text for the model
- `max_new_tokens` (int, optional): Maximum tokens to generate (default: 512)
- `temperature` (float, optional): Sampling temperature 0.0-2.0 (default: 0.7)
- `top_p` (float, optional): Nucleus sampling threshold (default: 0.9)

**Response:**
- Streaming text/event-stream with generated tokens

#### `GET /`

Health check endpoint.

**Response:**
```json
{
  "message": "Phi-3-mini-4k-instruct 4-bit local API running on RTX 5070 Ti"
}
```

---

## 🧪 Testing the API

### Using Jupyter Notebook (Recommended)

The included [example.ipynb](example.ipynb) provides an interactive Python client with streaming support:

```python
import requests
import json
import time

def chat(prompt, max_tokens=512, temp=0.7):
    # Streams response and displays tokens/s performance
    response = requests.post("http://127.0.0.1:8000/chat", json=payload, stream=True)
    # ... see notebook for full implementation
```

Simply run the cells to test the API with example prompts and see real-time streaming output.

### Using cURL (PowerShell)

```powershell
curl.exe -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d '{\"prompt\": \"Write a clean .NET 8 endpoint that returns current UTC time in ISO format.\", \"max_new_tokens\": 256}'
```

### PowerShell Helper Script

Save as `test.ps1`:

```powershell
$body = @{
    prompt = "Your prompt here"
    max_new_tokens = 512
    temperature = 0.7
} | ConvertTo-Json

curl.exe -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d $body
```

---

## ✨ Features

- **4-bit NF4 Quantization** – Near-fp16 quality with massive VRAM savings via bitsandbytes
- **Optimized Attention** – Eager attention mechanism (stable, fast on Windows + Blackwell)
- **Proper Templating** – Phi-3 chat template applied automatically
- **Streaming Support** – FastAPI StreamingResponse for real-time token generation
- **Clean API Design** – Returns only generated text, no boilerplate

---

## 📊 Performance Benchmarks

Tested on **RTX 5070 Ti Blackwell**:

| Metric | Performance |
|--------|------------|
| First model load | ~30s |
| Cached load | ~10s |
| Average generation speed | 45-60 tokens/s |
| Token output size tested | 512 tokens |

---

## 🔮 Future Enhancements

- [ ] Swap to `meta-llama/Meta-Llama-3.1-8B-Instruct` for significantly improved responses (fits in 16GB VRAM when quantized)
- [ ] Add chat history and session management
- [ ] Integrate with local vector database for RAG (Retrieval-Augmented Generation)
- [ ] Containerize with Docker for easier deployment
- [ ] Background service/daemon support

---

## � License

MIT License - feel free to use this as a template for your own projects.

---

## 📝 Notes

This pattern scales to any modern open-source LLM—use it as your local AI backend template. The architecture is designed to be modular and extensible, making it easy to swap models, add features, or integrate with existing systems.
