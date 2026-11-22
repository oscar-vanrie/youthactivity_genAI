"""
Generator: brief -> search_games() -> activiteit met citaatcontrole.
Gebruik vanuit de rootmap:
    python -m src.generation.generate
"""

import re


from src.rag.search import search_games
from src.generation.get_llm import get_llm
from langchain_core.language_models.chat_models import BaseChatModel
from src.generation.get_llm import get_llm, invoke_llm

MODEL = "qwen3:8b"

SYSTEM_RULES = """Je schrijft een activiteit voor een jeugdbeweging, in het Nederlands.
Strikte regels:
1. Gebruik UITSLUITEND de bronspellen hieronder. Niets uit je eigen kennis.
2. Elke zin met een spelregel, materiaal of veiligheidspunt eindigt met de bron, bv. [Spel 2].
3. Gebruik voor de spelregels, materialen en veiligheid alleen wat in de bronnen staat.
4. Staat iets niet in de bronnen? Zet het alleen bij "Niet in de bronnen". Verzin niets om het op te vullen.
5. Als een bron niet past bij de brief, gebruik hem dan niet en zeg dat.
6. Verzin geen URL's.
7. Een thema of verhaal dat je zelf verzint, zet je onder het kopje "Thema-aanpassing (zelf verzonnen)", los van de spelregels. Zet daar geen [Spel N] bij.
8. Schrijf NOOIT een denkplan of analyse in het Engels. Begin direct met **Titel**."""



def strip_thinking(text: str) -> str:
    """Verwijdert <think>-blokken en eventuele denk-monologen vóór het eigenlijke antwoord."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()

    marker = re.search(r"(?m)^(\*\*Titel\*\*|#+\s*Titel)", cleaned)
    if marker and not cleaned.startswith(marker.group(1)):
        prefix = cleaned[: marker.start()].lower()
        if any(k in prefix for k in ("we need to", "the brief", "source games", "let's", "i need to")):
            cleaned = cleaned[marker.start() :].strip()

    return cleaned


def extract_mechanics(hit: dict, llm: BaseChatModel) -> str | None:
    prompt = f"""Haal uit de onderstaande tekst UITSLUITEND de spelregels, materialen en veiligheidspunten
die er letterlijk of inhoudelijk in staan. Behoud voorwaarden, aantallen en wincondities nauwkeurig.
Schrijf ze als korte opsomming in het Nederlands zonder zinnen af te breken.
Staat er geen spelverloop of materiaallijst in de tekst, antwoord dan exact: GEEN SPELREGELS IN BRON

TEKST:
{hit['content']}"""
    text = strip_thinking(invoke_llm(llm, prompt))
    if not text or "GEEN SPELREGELS" in text.upper():
        return None
    return text


def verify_claim(claim: str, source_text: str, llm: BaseChatModel) -> bool:
    prompt = f"""BRON:
{source_text}

BEWERING:
{claim}

Staat de kern van deze bewering in de bron (ook als het met andere woorden of een ander meervoud/synoniem is beschreven), en wijkt de bewering niet inhoudelijk af van de bron?
Antwoord alleen JA of NEE."""
    answer = strip_thinking(invoke_llm(llm, prompt)).strip().upper()
    return answer.startswith("JA")


def check_citations(text: str, hits: list[dict], llm: BaseChatModel) -> list[str]:
    """
    Controleert elke regel met een bronverwijzing. Een claim wordt gemarkeerd als:
    - er geen geldige bronverwijzing is, of
    - het model vindt dat de claim niet in de geciteerde bron staat.
    Kopjes, de sectie 'Niet in de bronnen' en de thema-aanpassing worden overgeslagen.
    """
    flagged: list[str] = []
    skip_section = False

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue

        # Kopjes: **Titel** of # Titel
        if re.match(r"^(\*\*.+\*\*|#+\s)", raw):
            lower = raw.lower()
            skip_section = "niet in de bronnen" in lower or "thema-aanpassing" in lower
            continue

        # Alleen opsommingen en genummerde stappen zijn claims
        if not re.match(r"^([-*]|\d+\.)\s", raw):
            continue
        if skip_section or raw.startswith("⚠"):
            continue

        body = re.sub(r"^([-*]|\d+\.)\s*", "", raw).strip()
        refs = [int(n) for n in re.findall(r"\[Spel (\d+)\]", body)]
        claim = re.sub(r"\[Spel \d+\]", "", body).strip()
        if not claim:
            continue

        valid_refs = [r for r in refs if 1 <= r <= len(hits)]
        if not valid_refs:
            flagged.append(f"zonder geldige bronverwijzing: {body}")
            continue

        supported = any(
            verify_claim(claim, hits[r - 1]["content"], llm) for r in valid_refs
        )
        if not supported:
            flagged.append(f"niet in de geciteerde bron gevonden: {body}")

    return flagged


def generate_activity(
    brief: str,
    k: int = 3,
    min_age: int | None = None,
    max_age: int | None = None,
    terrein: str | None = None,
    strict_age: bool = True,
) -> str:
    hits = search_games(
        brief, k=k, min_age=min_age, max_age=max_age,
        terrein=terrein, strict_age=strict_age,
    )
    if not hits:
        return "Geen geschikte spellen gevonden voor deze brief."

    print("Opgehaalde bronnen:")
    for i, h in enumerate(hits, start=1):
        print(f"  [Spel {i}] {h['title']} | {h['source_url']} | leeftijd: {h['metadata'].get('leeftijd', '')}")
    print()

    llm = get_llm(temperature=0.2)

    # Stap 1: per bron alleen de letterlijke spelregels ophalen
    extracts: dict[int, str] = {}
    for i, h in enumerate(hits, start=1):
        mechanics = extract_mechanics(h, llm)
        if mechanics:
            extracts[i] = mechanics

    if not extracts:
        context = "\n\n".join(
            f"[Spel {i}] {h['title']}\n{h['content']}" for i, h in enumerate(hits, start=1)
        )
    else:
        context = "\n\n".join(
            f"[Spel {i}] {hits[i - 1]['title']}\n{text}" for i, text in extracts.items()
        )

    if not extracts:
        return "Geen van de gevonden spellen bevat spelregels. Probeer een andere brief."

    context = "\n\n".join(
        f"[Spel {i}] {hits[i - 1]['title']}\n{text}" for i, text in extracts.items()
    )

    # Stap 2: activiteit schrijven op basis van de samenvattingen
    prompt = f"""{SYSTEM_RULES}

BRIEF: {brief}

BRONSPELLEN (alleen de spelregels die in de bronnen staan):
{context}

Schrijf de activiteit met deze onderdelen en rond elke zin en elk punt volledig af:
- Titel
- Verhaal (kort, één alinea, gebaseerd op de bron)
- Spelregels in stappen, elke stap met [Spel N] en met de juiste spelstappen en winvoorwaarden uit de bron
- Materialen, elk met [Spel N]
- Veiligheidspunten, elk met [Spel N]
- Thema-aanpassing (zelf verzonnen): optioneel, alleen als een thema past bij de brief
- Niet in de bronnen: volledige zinnen over wat ontbreekt of is aangepast (zonder bronverwijzing)"""

    text = strip_thinking(invoke_llm(llm, prompt))

    # Stap 3: elke claim laten controleren tegen de geciteerde bron
    flagged = check_citations(text, hits, llm)

    sources = "\n".join(
        f"- [Spel {i}] {h['title']} ({h['source_url']})" for i, h in enumerate(hits, start=1)
    )
    result = f"{text}\n\n**Bronnen (uit de database):**\n{sources}"
    if flagged:
        result += "\n\n**⚠ Controle: niet overtuigend onderbouwd door de geciteerde bron:**\n"
        result += "\n".join(f"- {f}" for f in flagged)
    else:
        result += "\n\n**Controle:** alle regels zijn door het model als gegrond beoordeeld."
    return result


if __name__ == "__main__":
    brief = "bos, 8 tot 10 jaar, 2 uur, buiten rennen en opdrachten uitvoeren in groepjes"
    print(generate_activity(brief, k=3, min_age=8, max_age=10, terrein="bos"))