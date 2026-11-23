from __future__ import annotations

import base64
import os
import urllib.parse

import requests
from dotenv import load_dotenv

from src.generation.get_llm import get_llm, invoke_llm

load_dotenv()

_HTTP_TIMEOUT = 90


def _get_hf_token() -> str | None:
    for name in ("HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"):
        value = os.getenv(name)
        if value:
            return value
    return None


def _get_gemini_key() -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    return None


def _build_image_prompt(theme: str, title: str, story: str) -> str:
    """Laat een LLM een korte, duidelijke Engelse image-prompt schrijven."""
    llm = get_llm(temperature=0.3)

    system = (
        "Je bent een assistent die een korte, duidelijke Engelse image-prompt schrijft "
        "voor een kindvriendelijke illustratie op een printbare Chiro-spelkaart of thema-badge. "
        "Geef ALLEEN de prompt terug, geen uitleg, geen inleidende tekst, geen aanhalingstekens."
    )
    prompt = (
        f"{system}\n\n"
        f"Thema: {theme}\n"
        f"Titel: {title}\n"
        f"Verhaal/context: {story[:800]}\n\n"
        "Vereisten voor de Engelse image-prompt:\n"
        "- kindvriendelijke, kleurrijke illustratieve stijl (2D/flat vector), geschikt voor kinderen van 8-16 jaar\n"
        "- één duidelijk centraal onderwerp, leesbaar en herkenbaar\n"
        "- GEEN tekst, letters, cijfers, woorden, logo's of watermerk in beeld\n"
        "- vierkant (1:1) formaat voor een badge of spelkaart\n"
        "- beschrijf in maximaal 60 woorden het hoofdonderwerp, decor, kleuren en sfeer\n"
        "Schrijf de prompt in het Engels, in één doorlopend blok, zonder opsommingstekens."
    )

    text = ""
    try:
        text = invoke_llm(llm, prompt).strip()
    except Exception as exc:  # prompt-generatie mag nooit crashen
        print(f"[visuals] LLM image-prompt generatie mislukt ({exc}); gebruik fallback-prompt.")

    text = text.strip().strip('"').strip().strip("`").strip()
    if not text:
        text = (
            "friendly colorful 2D flat vector illustration for a children's scouting game card, "
            f"theme: {theme or title}, bold simple shapes, bright cheerful colors, "
            "single clear subject, no text, square badge"
        )
    return text


def _hf_flux(prompt: str) -> bytes | None:
    token = _get_hf_token()
    if not token:
        return None
    url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt},
        timeout=_HTTP_TIMEOUT,
    )
    if resp.ok and resp.headers.get("content-type", "").startswith("image"):
        return resp.content
    return None


def _gemini_imagen(prompt: str) -> bytes | None:
    key = _get_gemini_key()
    if not key:
        return None
    model = os.getenv("IMAGEGEN_MODEL", "imagen-3.0-generate-002")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict?key={key}"
    )
    body = {"instances": [{"prompt": prompt}], "parameters": {"sampleCount": 1}}
    resp = requests.post(url, json=body, timeout=_HTTP_TIMEOUT)
    if not resp.ok:
        return None
    data = resp.json()
    try:
        encoded = data["predictions"][0]["bytesBase64Encoded"]
        return base64.b64decode(encoded)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _pollinations(prompt: str) -> bytes | None:
    """Gratis, keyless image-endpoint (fallback zonder API-key)."""
    encoded = urllib.parse.quote(prompt)
    url = (
        "https://image.pollinations.ai/prompt/"
        f"{encoded}?width=768&height=768&nologo=true&model=flux"
    )
    resp = requests.get(url, timeout=_HTTP_TIMEOUT)
    if resp.ok and resp.headers.get("content-type", "").startswith("image"):
        return resp.content
    return None


def _try_generate_image(image_prompt: str) -> bytes | None:
    """Probeer beschikbare image-API's in volgorde van voorkeur; val terug op keyless."""
    providers: list[tuple[str, object]] = []
    if _get_hf_token():
        providers.append(("huggingface-flux", _hf_flux))
    if _get_gemini_key():
        providers.append(("gemini-imagen", _gemini_imagen))
    providers.append(("pollinations", _pollinations))

    for name, fn in providers:
        try:
            image = fn(image_prompt)  # type: ignore[operator]
            if image:
                print(f"[visuals] Afbeelding gegenereerd via {name} ({len(image)} bytes).")
                return image
            print(f"[visuals] {name} leverde geen afbeelding.")
        except Exception as exc:  # nooit de app laten crashen door de image-call
            print(f"[visuals] {name} mislukt: {exc}")
    return None


def generate_activity_visual(
    theme: str, title: str, story: str
) -> tuple[bytes | None, str]:
    """Genereer een thema-visual (badge/spelkaart).

    Geeft (image_bytes, image_prompt) terug. Bij een offline of falende image-call
    wordt (None, image_prompt) teruggegeven zodat de app nooit crasht.
    """
    image_prompt = _build_image_prompt(theme, title, story)
    image_bytes = _try_generate_image(image_prompt)
    return image_bytes, image_prompt