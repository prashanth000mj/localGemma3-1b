from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, os

model_id = "google/gemma-3-1b-it"         # 1 B param, text-only

torch.set_num_threads(os.cpu_count())     # use all CPU cores

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,               # stream weights → RAM, avoids peak spikes
    torch_dtype=torch.float32,            # safest on any CPU; try bfloat16 if AVX-512 BF16
)
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    device=-1,                            # -1 = force CPU
    max_new_tokens=256,
)

#print(pipe("Explain LangChain in 2 sentences")[0]["generated_text"])


from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."),
     ("user", "{query}")]
)

llm = HuggingFacePipeline(pipeline=pipe)

chain = prompt | llm          # llm = pipe-wrapped HF or LlamaCpp instance
# print(chain.invoke({"query": "Summarise Attention Is All You Need"}))


# app.py  (keep in project root)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Req(BaseModel):
    prompt: str
    max_tokens: int | None = 256

@app.post("/chat")
def chat(req: Req):
    return {"response": llm.invoke(req.prompt, max_new_tokens=req.max_tokens)}

if __name__ == "__main__":
    # Windows needs this guard when using multiprocessing (uvicorn workers)
    import multiprocessing as mp
    mp.freeze_support()


