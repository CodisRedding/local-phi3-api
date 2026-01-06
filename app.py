import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import uvicorn

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

print("Loading quantized Phi-3-mini-4k-instruct... (first run ~30s)")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=quant_config,
    device_map="auto",
    attn_implementation="eager",
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

app = FastAPI(title="Local Phi-3 Mini 4-bit API")

def generate_stream(request: ChatRequest):
    messages = [{"role": "user", "content": request.prompt}]
    formatted = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    for output in pipe(
        formatted,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=True,
    ):
        yield output["generated_text"]

@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(generate_stream(request), media_type="text/event-stream")

@app.get("/")
async def root():
    return {"message": "Phi-3-mini-4k-instruct 4-bit local API running on RTX 5070 Ti"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
