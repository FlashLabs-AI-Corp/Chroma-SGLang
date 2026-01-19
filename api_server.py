#!/usr/bin/env python3
"""
FastAPI server implementing OpenAI-compatible API for Chroma model
Supports dp-size configuration for data parallelism
"""

import os
import sys
import time
import base64
import io
import asyncio
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

import torch
import torch.distributed as dist
import torchaudio
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# SGLang imports
import sglang.srt.layers.dp_attention as dp_attn
import sglang.srt.server_args as server_args_module
from sglang.srt.server_args import ServerArgs
from sglang.srt.distributed.parallel_state import (
    initialize_model_parallel,
    init_distributed_environment,
)

# Chroma imports
from chroma.qwen2_5_omni_config import Qwen2_5OmniConfig
from chroma.qwen2_5_omni_modeling import Qwen2_5OmniModel
from chroma.modeling_chroma import ChromaForConditionalGeneration
from chroma.processing_chroma import ChromaProcessor
from safetensors.torch import safe_open

import logging
import warnings

# Suppress warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*were not sharded.*")
warnings.filterwarnings("ignore", message=".*were not used.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class ServerConfig:
    """Server configuration"""
    def __init__(
        self,
        chroma_model_path: str,
        base_qwen_path: str,
        dp_size: int = 1,
        host: str = "0.0.0.0",
        port: int = 8000,
        device: str = "cuda:0",
        default_prompt_audio: str = "assets/ref_audio.wav",
        default_prompt_text: str = "I have not... I'm so exhausted, I haven't slept in a very long time. It could be because... Well, I used our... Uh, I'm, I just use... This is what I use every day. I use our cleanser every day, I use serum in the morning and then the moistu- daily moisturizer. That's what I use every morning.",
    ):
        self.chroma_model_path = chroma_model_path
        self.base_qwen_path = base_qwen_path
        self.dp_size = dp_size
        self.host = host
        self.port = port
        self.device = device
        self.world_size = dp_size
        self.default_prompt_audio = default_prompt_audio
        self.default_prompt_text = default_prompt_text


# ============================================================================
# OpenAI API Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    prompt_text: Optional[str] = None
    prompt_audio: Optional[str] = None  # Base64 encoded audio or file path
    
    # Audio generation parameters
    return_audio: Optional[bool] = True
    audio_format: Optional[str] = "wav"  # wav, mp3, etc.


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    audio: Optional[str] = None  # Base64 encoded audio


# ============================================================================
# Model Manager
# ============================================================================

class ChromaModelManager:
    """Manages Chroma model loading and inference"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.chroma_model = None
        self.processor = None
        self.tokenizer = None
        
    def initialize_distributed(self):
        """Initialize distributed environment"""
        logger.info(f"Initializing distributed environment with dp_size={self.config.dp_size}")
        
        # 1. Setup environment variables
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", str(self.config.world_size)))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        
        # 2. Initialize PyTorch distributed
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=world_size,
                rank=rank
            )
            logger.info(f"PyTorch distributed initialized: rank={rank}, world_size={world_size}")
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        
        # 3. Initialize SGLang DP Attention
        dp_attn._ATTN_TP_SIZE = 1
        dp_attn._ATTN_TP_RANK = 0
        logger.info(f"DP Attention configured: TP_SIZE=1, TP_RANK=0")
        
        # 4. Initialize SGLang Global Server Args
        server_args = ServerArgs(
            model_path="dummy",
            tp_size=1,
            mm_attention_backend="sdpa",
        )
        server_args_module._global_server_args = server_args
        
        # 5. Initialize SGLang Tensor Parallel Groups
        init_distributed_environment(
            backend="nccl" if torch.cuda.is_available() else "gloo",
        )
        
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        logger.info("SGLang parallel groups initialized")
    
    def patch_parameter(self):
        """Patch torch.nn.Parameter for SGLang compatibility"""
        from torch.nn import Parameter as OriginalParameter
        
        _original_parameter_new = OriginalParameter.__new__
        
        def patched_parameter_new(cls, *args, **kwargs):
            sglang_attrs = ['input_dim', 'output_dim', 'weight_loader', 'weight_loader_v2']
            for attr in sglang_attrs:
                kwargs.pop(attr, None)
            return _original_parameter_new(cls, *args, **kwargs)
        
        OriginalParameter.__new__ = staticmethod(patched_parameter_new)
        logger.info("Parameter patching applied")
    
    def iter_thinker_weights(self, ckpt_dir: str):
        """Iterate through thinker weights from checkpoint"""
        weight_files = []
        for fn in os.listdir(ckpt_dir):
            if fn.endswith(".safetensors"):
                weight_files.append(os.path.join(ckpt_dir, fn))
        weight_files.sort()
        
        for wf in weight_files:
            with safe_open(wf, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith("thinker."):
                        yield (k, f.get_tensor(k))
    
    def load_model(self):
        """Load Chroma model"""
        logger.info("Loading Chroma model...")
        
        # Initialize distributed environment
        self.initialize_distributed()
        
        # Patch parameter
        self.patch_parameter()
        
        # Load base Qwen config and model
        sgl_cfg = Qwen2_5OmniConfig.from_pretrained(self.config.base_qwen_path)
        sgl_model = Qwen2_5OmniModel(sgl_cfg, quant_config=None)
        
        # Load Chroma model
        self.chroma_model = ChromaForConditionalGeneration.from_pretrained(
            self.config.chroma_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        self.chroma_model = self.chroma_model.to(self.device)
        self.chroma_model.eval()
        
        # Load thinker weights
        sgl_model.load_weights(self.iter_thinker_weights(self.config.chroma_model_path))
        sgl_model = sgl_model.to(self.device).to(torch.bfloat16).eval()
        self.chroma_model.thinker = sgl_model.thinker
        
        # Load processor
        self.processor = ChromaProcessor.from_pretrained(self.config.chroma_model_path)
        self.tokenizer = self.processor.tokenizer
        
        if hasattr(self.processor.tokenizer, 'chat_template'):
            self.processor.chat_template = self.processor.tokenizer.chat_template
        
        logger.info("Model loaded successfully")
    
    def prepare_conversation(self, messages: List[Message]) -> List[Dict]:
        """Convert OpenAI messages to Chroma conversation format"""
        conversation = []
        
        for msg in messages:
            role = msg.role
            content = msg.content
            
            # Convert content to list format if it's a string
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            
            conversation.append({
                "role": role,
                "content": content
            })
        
        return conversation
    
    def load_audio_from_base64(self, audio_base64: str, target_sample_rate: int = 24000) -> torch.Tensor:
        """Load audio from base64 encoded string"""
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            audio_buffer = io.BytesIO(audio_bytes)
            
            # Load audio
            audio_tensor, sample_rate = torchaudio.load(audio_buffer)
            
            # Convert to mono if stereo
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            
            # Resample
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0),
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
            )
            
            return audio_tensor
        except Exception as e:
            logger.error(f"Error loading audio from base64: {e}")
            raise
    
    def audio_to_base64(self, audio_tensor: torch.Tensor, sample_rate: int = 24000, format: str = "wav") -> str:
        """Convert audio tensor to base64 encoded string"""
        import tempfile
        import soundfile as sf

        try:
            # Prepare audio tensor
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Ensure audio is on CPU and convert to numpy
            audio_numpy = audio_tensor.cpu().float().squeeze(0).numpy()

            # Use temporary file to save audio (torchcodec backend doesn't support BytesIO)
            with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                # Save audio to temporary file using soundfile (more reliable)
                sf.write(tmp_path, audio_numpy, sample_rate, format=format.upper())

                # Read file and encode to base64
                with open(tmp_path, 'rb') as f:
                    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

                return audio_base64
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            logger.error(f"Error converting audio to base64: {e}")
            raise
    
    @torch.inference_mode()
    def generate(
        self,
        messages: List[Message],
        prompt_text: Optional[str] = None,
        prompt_audio: Optional[Union[str, torch.Tensor]] = None,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_audio: bool = True,
    ) -> Dict[str, Any]:
        """Generate response"""
        try:
            # Prepare conversation
            conversation = self.prepare_conversation(messages)
            
            # Validate prompt_text and prompt_audio must be provided together
            if (prompt_text is None) != (prompt_audio is None):
                raise ValueError(
                    "prompt_text and prompt_audio must be provided together. "
                    "Either provide both or provide neither (will use defaults)."
                )
            
            # Use default values if both not provided
            if prompt_text is None and prompt_audio is None:
                prompt_text = self.config.default_prompt_text
                prompt_audio = self.config.default_prompt_audio
                logger.info(f"Using default prompt_text and prompt_audio: {prompt_audio}")
            
            # Handle prompt_audio
            if isinstance(prompt_audio, str):
                # Check if it's base64 or file path
                if prompt_audio.startswith('data:audio'):
                    # Extract base64 data
                    prompt_audio = prompt_audio.split(',')[1]
                    prompt_audio_tensor = self.load_audio_from_base64(prompt_audio)
                elif os.path.exists(prompt_audio):
                    # File path
                    prompt_audio_tensor = self.processor.load_audio(prompt_audio)
                else:
                    # Assume it's base64
                    prompt_audio_tensor = self.load_audio_from_base64(prompt_audio)
            elif isinstance(prompt_audio, torch.Tensor):
                prompt_audio_tensor = prompt_audio
            else:
                raise ValueError(f"Invalid prompt_audio type: {type(prompt_audio)}")
            
            # Process inputs
            inputs = self.processor(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
                prompt_audio=prompt_audio_tensor,
                prompt_text=prompt_text
            )
            
            # Move to device and convert dtypes
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype in [torch.float32, torch.float64]:
                        inputs[k] = v.to(torch.bfloat16)
                    inputs[k] = inputs[k].to(self.device)
            
            # Generate
            start_time = time.time()
            generated = self.chroma_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                use_cache=True,
                temperature=temperature,
                top_p=top_p,
                output_attentions=False,
                output_hidden_states=False,
            )
            generation_time = time.time() - start_time
            
            result = {
                "generation_time": generation_time,
                "audio_tensor": None,
                "audio_base64": None,
            }
            
            # Decode audio if requested
            if return_audio:
                generated_d = generated.permute(0, 2, 1).to(self.device)
                generated_d = generated_d.clamp(min=0, max=2047)
                
                output = self.chroma_model.codec_model.decode(generated_d)
                wav = output.squeeze(0).squeeze(0)
                
                result["audio_tensor"] = wav
                result["audio_base64"] = self.audio_to_base64(wav, sample_rate=24000)
                result["audio_duration"] = wav.shape[0] / 24000
            
            return result
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise


# ============================================================================
# FastAPI Application
# ============================================================================

# Global model manager
model_manager: Optional[ChromaModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global model_manager
    
    # Startup
    logger.info("Starting Chroma API server...")

    # Get configuration from environment (required)
    chroma_model_path = os.environ.get("CHROMA_MODEL_PATH")
    base_qwen_path = os.environ.get("BASE_QWEN_PATH")
    dp_size = int(os.environ.get("DP_SIZE", "1"))

    if not chroma_model_path or not base_qwen_path:
        raise ValueError("CHROMA_MODEL_PATH and BASE_QWEN_PATH environment variables must be set")
    
    config = ServerConfig(
        chroma_model_path=chroma_model_path,
        base_qwen_path=base_qwen_path,
        dp_size=dp_size,
    )
    
    model_manager = ChromaModelManager(config)
    model_manager.load_model()
    
    logger.info(f"Server ready! DP_SIZE={dp_size}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if dist.is_initialized():
        dist.destroy_process_group()


app = FastAPI(
    title="Chroma API",
    description="OpenAI-compatible API for Chroma audio generation model",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Chroma API Server",
        "version": "1.0.0",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/health"]
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager is not None and model_manager.chroma_model is not None
    }


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "chroma",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "chroma",
                "permission": [],
                "root": "chroma",
                "parent": None,
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint
    
    Example request:
    ```json
    {
        "model": "chroma",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "audio", "audio": "assets/question.wav"}]}
        ],
        "prompt_text": "I have not... I'm so exhausted, I haven't slept in a very long time. It could be because... Well, I used our... Uh, I'm, I just use... This is what I use every day. I use our cleanser every day, I use serum in the morning and then the moistu- daily moisturizer. That's what I use every morning.",  // Optional, uses default if not provided
        "prompt_audio": "assets/ref_audio.wav",  // Optional, uses default if not provided
        "max_tokens": 1000,
        "temperature": 1.0
    }
    ```
    """
    global model_manager
    
    if model_manager is None or model_manager.chroma_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate response (prompt_text and prompt_audio are optional now)
        result = model_manager.generate(
            messages=request.messages,
            prompt_text=request.prompt_text,
            prompt_audio=request.prompt_audio,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            return_audio=request.return_audio,
        )
        
        # Create response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=f"Generated audio ({result.get('audio_duration', 0):.2f}s)"
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=0,  # TODO: calculate actual tokens
                completion_tokens=0,
                total_tokens=0
            ),
            audio=result.get("audio_base64") if request.return_audio else None
        )
        
        logger.info(f"Request completed in {result['generation_time']:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chroma API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--chroma-model-path", type=str, required=True,
                       help="Path to Chroma model (required)")
    parser.add_argument("--base-qwen-path", type=str, required=True,
                       help="Path to base Qwen model (required)")
    parser.add_argument("--dp-size", type=int, default=1,
                       help="Data parallel size")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CHROMA_MODEL_PATH"] = args.chroma_model_path
    os.environ["BASE_QWEN_PATH"] = args.base_qwen_path
    os.environ["DP_SIZE"] = str(args.dp_size)
    
    # Run server
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
