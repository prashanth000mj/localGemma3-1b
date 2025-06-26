from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
)
import torch

model_id = "google/gemma-3-1b-it"

quant_cfg = BitsAndBytesConfig(load_in_4bit=True)   # change to 4-bit if you prefer GGUF

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",          # puts weights on GPU if available
    quantization_config=quant_cfg,
).eval()

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    torch_dtype=torch.bfloat16,   # safe even on consumer GPUs
)


from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

llm = HuggingFacePipeline(pipeline=pipe)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{query}")
    ]
)

chain = prompt | llm        # LC-v0.2 style
print(chain.invoke({"query": "Explain transformers in one paragraph"}))

# app.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Req(BaseModel):
    prompt: str
    max_tokens: int | None = 256

@app.post("/chat")
def chat(req: Req):
    resp = llm.invoke(req.prompt, max_new_tokens=req.max_tokens)
    return {"response": resp}
