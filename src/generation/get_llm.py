import os
import re
import time
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

load_dotenv()


def get_ollama_llm(temperature: float = 0.2, model: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model or "qwen3:8b",
        temperature=temperature,
    )


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL")

    if provider == "groq":
        from langchain_groq import ChatGroq

        selected_model = model or "llama-3.1-8b-instant"
        kwargs: dict = {
            "model": selected_model,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        if "gpt-oss" in selected_model:
            kwargs["reasoning_effort"] = "low"
            kwargs["reasoning_format"] = "hidden"
        elif "qwen" in selected_model:
            kwargs["reasoning_effort"] = "none"

        return ChatGroq(**kwargs)

    
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.5-flash",
            temperature=temperature,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model or "meta-llama/llama-3.3-70b-instruct:free",
            temperature=temperature,
        )

    return get_ollama_llm(temperature=temperature, model=model)

def _extract_content(response) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()

    return str(content or "").strip()

def invoke_llm(
    llm: BaseChatModel,
    prompt: str,
    fallback_temp: float = 0.2,
    retries: int = 1,
) -> str:
    """Probeert het LLM (met retry), en valt bij falen terug op lokale Ollama."""
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            out = _extract_content(llm.invoke(prompt))
            if out:
                return out
            last_exc = ValueError("Lege response ontvangen van LLM")
        except Exception as exc:
            last_exc = exc

        if attempt < retries:
            time.sleep(1.0)

    if isinstance(llm, ChatOllama):
        raise last_exc or RuntimeError("Lokale Ollama gaf geen antwoord")

    print(
        f"[Waarschuwing] Cloud-LLM gaf probleem ({last_exc}). "
        "Fallback naar lokale Ollama (qwen3:8b)..."
    )
    fallback = get_ollama_llm(temperature=fallback_temp)
    return _extract_content(fallback.invoke(prompt))