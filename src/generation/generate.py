"""
Generator: brief -> search_games() -> activiteit met gebundelde citaatcontrole.
Gebruik vanuit de rootmap:
    python -m src.generation.generate
"""

import re

from langchain_core.language_models.chat_models import BaseChatModel

from src.generation.get_llm import get_llm, invoke_llm
from src.rag.search import search_games

SYSTEM_RULES = """Je schrijft een activiteit voor een jeugdbeweging, in het Nederlands.
Strikte regels:
1. Gebruik UITSLUITEND de bronspellen hieronder. Niets uit je eigen kennis.
2. Kies één hoofdbron ([Spel 1] of [Spel 2]) met de beste spelmechaniek en werk die duidelijk uit.
3. Elke zin met een spelregel, materiaal of veiligheidspunt eindigt met de bron, bv. [Spel 2].
4. Staat iets niet in de bronnen? Zet het alleen bij "Niet in de bronnen". Verzin niets om het op te vullen.
5. Een thema of verhaal dat je zelf verzint, zet je onder "Thema-aanpassing (zelf verzonnen)", zonder [Spel N].
6. Schrijf direct in verzorgd Nederlands (geen Engelse woorden of denkplannen) en verzin geen URL's."""


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    marker = re.search(r"(?m)^(\*\*Titel\*\*|#+\s*Titel)", cleaned)
    if marker and not cleaned.startswith(marker.group(1)):
        cleaned = cleaned[marker.start() :].strip()
    return cleaned


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        meta = h["metadata"]
        header = (
            f"[Spel {i}] {h['title']} | leeftijd: {meta.get('leeftijd', '')} | "
            f"terrein: {meta.get('terrein', '')} | duur: {meta.get('duur', '')}"
        )
        blocks.append(f"{header}\n{h['content']}")
    return "\n\n".join(blocks)


def extract_claims(text: str, num_hits: int) -> tuple[list[tuple[int, str]], list[str]]:
    """Haalt in Python (0 LLM-calls) alle regels met hun [Spel N]-verwijzing op."""
    valid_claims: list[tuple[int, str]] = []
    flagged: list[str] = []
    skip_section = False

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue

        if re.match(r"^(\*\*.+\*\*|#+\s)", raw):
            lower = raw.lower()
            skip_section = (
                "niet in de bronnen" in lower
                or "thema-aanpassing" in lower
                or "titel" in lower
                or "verhaal" in lower
            )
            continue

        if not re.match(r"^([-*]|\d+\.)\s", raw) or skip_section or raw.startswith("⚠"):
            continue

        body = re.sub(r"^([-*]|\d+\.)\s*", "", raw).strip()
        refs = [int(n) for n in re.findall(r"\[Spel (\d+)\]", body)]
        valid_refs = [r for r in refs if 1 <= r <= num_hits]

        if not valid_refs:
            flagged.append(f"zonder geldige bronverwijzing: {body}")
        else:
            valid_claims.append((len(valid_claims) + 1, body))

    return valid_claims, flagged


def check_citations_batch(text: str, hits: list[dict], llm: BaseChatModel) -> list[str]:
    """Controleert ALLE claims in één enkele gebundelde LLM-call."""
    claims, flagged = extract_claims(text, len(hits))
    if not claims:
        return flagged

    claims_block = "\n".join(f"{idx}. {body}" for idx, body in claims)
    prompt = f"""Controleer of onderstaande beweringen inhoudelijk kloppen met de bron waarnaar ze verwijzen ([Spel N]).
Parafrases en synoniemen zijn toegestaan, maar een verkeerde winconditie of verzonnen spelregel niet.

BRONNEN:
{build_context(hits)}

BEWERINGEN:
{claims_block}

Geef ALLEEN de nummers en tekst van beweringen die NIET in de geciteerde bron staan (één per regel).
Als alle beweringen kloppen, antwoord dan exact: ALLES_OK"""

    verdict = strip_thinking(invoke_llm(llm, prompt))
    if "ALLES_OK" not in verdict.upper():
        for line in verdict.splitlines():
            cleaned = line.strip().lstrip("-* ")
            if cleaned:
                flagged.append(cleaned)

    return flagged


def generate_activity(
    brief: str,
    k: int = 2,
    min_age: int | None = None,
    max_age: int | None = None,
    terrein: str | None = None,
    strict_age: bool = True,
    verify: bool = True,
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

    # Call 1: activiteit direct genereren uit de opgehaalde bronnen
    prompt = f"""{SYSTEM_RULES}

BRIEF: {brief}

BRONSPELLEN:
{build_context(hits)}

Schrijf de activiteit met exact deze onderdelen en rond elke zin volledig af:
- Titel
- Verhaal (kort, één alinea, gebaseerd op de bron)
- Spelregels in stappen, elke stap met [Spel N] en met de juiste spelstappen en winvoorwaarden
- Materialen, elk met [Spel N]
- Veiligheidspunten, elk met [Spel N]
- Thema-aanpassing (zelf verzonnen): optioneel, alleen als een thema past bij de brief
- Niet in de bronnen: volledige zinnen over wat ontbreekt of is aangepast (zonder bronverwijzing)"""

    raw_text = invoke_llm(llm, prompt)
    text = strip_thinking(raw_text) or raw_text.strip()

    # Call 2 (optioneel): alle claims in één keer controleren
    flagged = check_citations_batch(text, hits, llm) if verify else []

    sources = "\n".join(
        f"- [Spel {i}] {h['title']} ({h['source_url']})" for i, h in enumerate(hits, start=1)
    )
    result = f"{text}\n\n**Bronnen (uit de database):**\n{sources}"
    if verify:
        if flagged:
            result += "\n\n**⚠ Controle: niet overtuigend onderbouwd door de geciteerde bron:**\n"
            result += "\n".join(f"- {f}" for f in flagged)
        else:
            result += "\n\n**Controle:** alle regels zijn in één batch-check als gegrond beoordeeld."
    return result


if __name__ == "__main__":
    brief = "bos, 8 tot 10 jaar, 2 uur, buiten rennen en opdrachten uitvoeren in groepjes"
    print(generate_activity(brief, k=2, min_age=8, max_age=10, terrein="bos", verify=True))