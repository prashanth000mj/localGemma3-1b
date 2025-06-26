
Bring up the server
```uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

Or Run this in the docker (Get $HF_TOKEN from https://huggingface.co/settings/tokensgor downloading Gemma3-1b)
docker build -t gemma3-1b .
docker run --gpus all -p 8000:8000 -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN gemma3-1b


Make call to API
```curl -X POST http://localhost:8000/chat -d '{"prompt":"Hello Gemma!"}'

```// Browser console (any modern browser)
fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    prompt: "Hello Gemma!, can you explaing me CUDA?",   // required field
    max_tokens: 256               // optional, matches your Pydantic model
  })
})
  .then(r => r.json())
  .then(console.log)
  .catch(console.error);

