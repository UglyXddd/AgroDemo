import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

HF_REPO_ID = "Qwen/Qwen3-4B-GGUF"
HF_FILENAME = "Qwen3-4B-Q4_K_M.gguf"

DEFAULT_MODEL_PATH = MODELS_DIR / HF_FILENAME
#для корректной работы нужна оригинальная модель, которую к сожалению не могу выложить


'''
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "./AI_PRO_MAX/model-f16.gguf",
) 
'''



def get_model_path() -> Path:


    custom_model_path = os.getenv("MODEL_PATH")

    if custom_model_path:
        path = Path(custom_model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"MODEL_PATH points to a missing file: {path}"
            )

        return path

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    print("Qwen3-4B model not found locally.")
    print("Downloading Qwen3-4B Q4_K_M from Hugging Face...")

    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=MODELS_DIR,
    )

    print(f"Model downloaded to: {downloaded_path}")

    return Path(downloaded_path)


def interact():
    model_path = get_model_path()

    print(f"Loading model: {model_path}")

    model = Llama(
        model_path=str(model_path),
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=False,
        seed=1,
    )

    return model

SYSTEM_PROMPT = (
    "Ты AgroChat, AI-ассистент по агротематике. "
    "Отвечай на русском языке понятно и по существу. "
    "Если предоставлен контекст из базы знаний, используй его для ответа. "
    "Если в контексте нет ответа на вопрос, не выдумывай факты."
)


def generate(question, model, context=False):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Контекст из базы знаний:\n"
                    f"{context}"
                ),
            }
        )

    # mainAI может передавать как строку, так и историю диалога.
    if isinstance(question, str):
        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

    elif isinstance(question, list):
        for item in question:
            if not item:
                continue

            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")

                if content:
                    messages.append(
                        {
                            "role": role,
                            "content": str(content),
                        }
                    )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": str(item),
                    }
                )

    else:
        messages.append(
            {
                "role": "user",
                "content": str(question),
            }
        )

    response = model.create_chat_completion(
        messages=messages,
        temperature=0.7,
        top_p=0.8,
        top_k=30,
        max_tokens=1024,
    )

    return response["choices"][0]["message"]["content"].strip()