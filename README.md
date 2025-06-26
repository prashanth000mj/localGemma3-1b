
# Local Gemma 3.1B Inference with FastAPI

This project provides a simple and efficient way to run Google's Gemma 3.1B Instruct model locally. It leverages Hugging Face Transformers for model loading, integrates with LangChain for flexible prompt templating, and exposes the inference functionality via a FastAPI web API. This setup is ideal for local development, rapid prototyping, or integrating Gemma into other applications.

## Features

-   **Local Inference**: Run the `google/gemma-3-1b-it` model directly on your machine.
-   **FastAPI Integration**: Provides a RESTful API endpoint for easy interaction with the model.
-   **LangChain Support**: Utilizes LangChain for robust prompt templating and pipeline management.
-   **Quantization Support**: Optionally load the model in 4-bit quantized format to significantly reduce GPU memory usage (requires a compatible GPU and `bitsandbytes` installation).
-   **Asynchronous API**: The FastAPI endpoint is asynchronous, ensuring the server remains responsive even during long-running inference tasks.

## Prerequisites

-   Python 3.9+
-   `pip` (Python package installer)
-   **Optional but Recommended**: An NVIDIA GPU with CUDA installed for faster inference and quantization.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/prashanth000mj/localGemma3-1b.git
    cd localGemma3-1b
    ```

2.  **Create and activate a virtual environment** (highly recommended):
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    First, install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Hugging Face Token Setup (Crucial for Gemma)

The Gemma models are "gated" on Hugging Face, meaning you need to accept their terms of use before downloading.

1.  **Request Access**:
    *   Visit the model page: https://huggingface.co/google/gemma-3-1b-it
    *   Log in to your Hugging Face account.
    *   Read and accept the terms and conditions to gain access.

2.  **Authenticate Your Environment**:
    You have two options:

    *   **Option A (Recommended)**: Log in via the Hugging Face CLI.
        ```bash
        huggingface-cli login
        ```
        Follow the prompts to enter your Hugging Face token (you can generate one from your Hugging Face settings page).

    *   **Option B**: Use an environment variable.
        Create a file named `.env` in the root of your project and add your Hugging Face token:
        ```
        HUGGING_FACE_TOKEN="hf_YOUR_ACTUAL_TOKEN_HERE"
        ```
        *(Note: The provided `app.py` does not currently use `python-dotenv` to load this, but it's a good practice for future expansion or if you modify `app.py` to use it.)*

## Running the Application

Once the dependencies are installed and your Hugging Face token is set up, you can start the FastAPI application:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

The first time you run this, the model weights will be downloaded (approx. 2GB for Gemma 3.1B-it). This might take some time depending on your internet connection. The model will then be loaded into memory.

You should see output indicating that Uvicorn is running, typically on `http://0.0.0.0:8000`.

## API Usage

The application exposes a single `POST` endpoint for chat inference.

**Endpoint**: `/chat`
**Method**: `POST`
**Request Body (JSON)**:
```json
{
  "prompt": "Your question or prompt for the model.",
  "max_tokens": 256
}
```
`max_tokens` is optional and defaults to 256.

**Example using `curl`**:
```bash
curl -X POST "http://0.0.0.0:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain large language models in one sentence.", "max_tokens": 50}'
```

## Troubleshooting: `bitsandbytes` GPU Support on Windows

If you encounter errors related to `bitsandbytes` not finding GPU support (e.g., "The installed version of bitsandbytes was compiled without GPU support"), this is common on Windows.

1.  **Uninstall the CPU-only version**:
    ```bash
    pip uninstall bitsandbytes
    ```
2.  **Install a CUDA-enabled wheel**:
    ```bash
    pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
    ```
    This command installs pre-compiled wheels for Windows with CUDA support.

## Quantization

The `app.py` file is set up to optionally use 4-bit quantization, which drastically reduces the model's memory footprint on your GPU. By default, it's commented out to ensure CPU compatibility.

To enable 4-bit quantization (after successfully installing `bitsandbytes` with GPU support as described above):

1.  Open `app.py`.
2.  Uncomment the `BitsAndBytesConfig` import and the `quantization_config` parameter in the `AutoModelForCausalLM.from_pretrained` call.

    ```python
    # app.py relevant section
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline # Uncomment BitsAndBytesConfig
    )
    # ...
    # quant_cfg = BitsAndBytesConfig(load_in_4bit=True) # Uncomment this line
    # ...
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # quantization_config=quant_cfg, # Uncomment this line
    ).eval()
    ```

This will allow the model to load using less VRAM, potentially enabling it to run on GPUs with less memory.

## Project Structure

```
localGemma3.1b/
├── app.py                  # Main FastAPI application and model loading
├── README.md               # This file
└── requirements.txt        # Python dependencies
└── .env                    # Optional: For Hugging Face token
```

## Bring up the server
    ```
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
    ```

## Or Run this in the docker (Get $HF_TOKEN from https://huggingface.co/settings/tokensgor downloading Gemma3-1b)

    ```
    docker build -t gemma3-1b .
    docker run --gpus all -p 8000:8000 -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN gemma3-1b
    ```

## Make call to API
    ```
    curl -X POST http://localhost:8000/chat -d '{"prompt":"Hello Gemma!"}'
    ```

    ```
    # Browser console (any modern browser)
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
    ````

