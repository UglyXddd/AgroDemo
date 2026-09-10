import logging
import os
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import FileResponse

from AI_PRO_MAX import mainAI


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


app = FastAPI(
    title="AgroChat Demo",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


logging.basicConfig(
    filename=BASE_DIR / "request_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class RequestData(BaseModel):
    userMessages: List[Optional[str]] = Field(default_factory=list)
    botMessages: List[Optional[str]] = Field(default_factory=list)
    image: List[Optional[str]] = Field(default_factory=list)
    flags: List[Optional[str]] = Field(default_factory=list)


def extract_text(message: str, prefix: str) -> str:
    if prefix in message:
        return message.split(prefix, 1)[-1].strip()

    return message.strip()


@app.get("/")
async def read_root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/request")
async def make_response(request: Request, request_data: RequestData):
    client_host = request.client.host if request.client else "unknown"

    logging.info("Request received from %s", client_host)

    user_messages = [
        extract_text(message, "text:")
        for message in request_data.userMessages
        if message
    ]

    bot_messages = [
        message.strip()
        for message in request_data.botMessages
        if message
    ]

    # Image processing is disabled in the public demo.
    if request_data.image and request_data.image[0]:
        return {
            "text": "Обработка изображений недоступна в демонстрационной версии.",
            "image": None,
        }

    data_for_ai = []

    if user_messages:
        data_for_ai.append(user_messages[0])

    if bot_messages:
        data_for_ai.append(bot_messages[0])

    if len(user_messages) > 1:
        data_for_ai.append(user_messages[1])

    regenerate = "regenerate" in request_data.flags

    text = mainAI.ai_main(
        data_for_ai,
        regenerate_flag=regenerate,
    )

    logging.info("AI response generated for %s", client_host)

    return {
        "text": text,
        "image": None,
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
    )