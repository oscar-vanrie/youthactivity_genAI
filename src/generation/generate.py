import re

from langchain_core.language_models.chat_models import BaseChatModel

from src.generation.get_llm import get_llm, invoke_llm
from src.rag.search import search_games
from src.agents import ActivityContext, run_agents_parallel


def strip_thinking(text: str) -> str:
    """Verwijder denk-sporen en tekst vóór de eerste echte kop."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    marker = re.search(r"(?m)^(\*\*Titel\*\*|\*\*Groepsindeling|\*\*Verhaal\*\*|\*\*Veiligheidspunten|#+\s)", cleaned)
    if marker and not cleaned.startswith(marker.group(1)):
        cleaned = cleaned[marker.start() :].strip()
    return cleaned


def build_context(hits: list[dict]) -> str:
    """Zet de opgehaalde bronnen om naar één contextblok met [Spel N]-koppen."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        meta = h["metadata"]
        header = (
            f"[Spel {i}] {h['title']} | leeftijd: {meta.get('leeftijd', '')} | "
            f"terrein: {meta.get('terrein', '')} | duur: {meta.get('duur', '')}"
        )
        blocks.append(f"{header}\n{h['content']}")
    return "\n\n".join(blocks)


SKIP_SECTION_KEYWORDS = (
    "niet in de bronnen",
    "thema-aanpassing",
    "leeftijds- en terreininschatting",
    "titel",
    "verhaal",
)


def _normalize_dashes(s: str) -> str:
    """Vervang alle soorten koppeltekens door een gewone streep (ASCII)."""
    return re.sub(r"[\u2010-\u2015\u2212]", "-", s)


def extract_claims(text: str, num_hits: int) -> tuple[list[tuple[int, str]], list[str]]:
    """Haalt in Python (0 LLM-calls) alle brongebonden regels met [Spel N] op."""
    valid_claims: list[tuple[int, str]] = []
    flagged: list[str] = []
    skip_section = False

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue

        if re.match(r"^(\*\*.+\*\*|#+\s)", raw):
            lower = _normalize_dashes(raw.lower())
            skip_section = any(kw in lower for kw in SKIP_SECTION_KEYWORDS)
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
    """Controleert alle brongebonden claims in één gebundelde LLM-call."""
    claims, flagged = extract_claims(text, len(hits))
    if not claims:
        return flagged

    claims_map = {idx: body for idx, body in claims}
    claims_block = "\n".join(f"{idx}. {body}" for idx, body in claims)
    prompt = f"""Controleer of onderstaande beweringen inhoudelijk kloppen met de bron waarnaar ze verwijzen ([Spel N]).
    Let ook op: personen/tikkers/leiding mogen NIET onder Materialen staan.

    BRONNEN:
    {build_context(hits)}

    BEWERINGEN:
    {claims_block}

    Antwoord met ALLES_OK als alles klopt.
    Zo niet, geef ALLEEN de nummers van foute beweringen gescheiden door komma's, bv.: 2, 5"""

    verdict = strip_thinking(invoke_llm(llm, prompt)).strip()
    if "ALLES_OK" not in verdict.upper():
        bad_nums = {int(n) for n in re.findall(r"\b(\d+)\b", verdict)}
        for num in sorted(bad_nums):
            if num in claims_map:
                flagged.append(f"Bewering #{num} niet bevestigd: {claims_map[num]}")

    return flagged


def assemble_activity(rules_text: str, story_text: str, safety_text: str) -> str:
    """Voegt de outputs van de drie agents samen tot één overzichtelijke activiteit."""
    rules_clean = strip_thinking(rules_text)
    story_clean = strip_thinking(story_text)
    safety_clean = strip_thinking(safety_text)
    return "\n\n".join(part for part in (rules_clean, story_clean, safety_clean) if part)


def generate_activity(
    brief: str,
    k: int = 2,
    min_age: int | None = None,
    max_age: int | None = None,
    terrein: str | None = None,
    strict_age: bool = True,
    verify: bool = False,
) -> str:
    """Zoek bronnen, draai de drie agents en lever de volledige activiteit als tekst."""
    hits = search_games(
        brief,
        k=k,
        min_age=min_age,
        max_age=max_age,
        terrein=terrein,
        strict_age=strict_age,
    )
    if not hits:
        return "Geen geschikte spellen gevonden voor deze brief."

    print("Opgehaalde bronnen:")
    for i, h in enumerate(hits, start=1):
        print(f"  [Spel {i}] {h['title']} | {h['source_url']} | leeftijd: {h['metadata'].get('leeftijd', '')}")
    print()

    llm = get_llm(temperature=0.2)
    ctx = ActivityContext(
        brief=brief,
        sources_text=build_context(hits),
        min_age=min_age,
        max_age=max_age,
        terrein=terrein,
    )

    rules_text, story_text, safety_text = run_agents_parallel(llm, ctx)
    combined_text = assemble_activity(rules_text, story_text, safety_text)

    flagged = check_citations_batch(combined_text, hits, llm) if verify else []

    sources = "\n".join(
        f"- [Spel {i}] {h['title']} ({h['source_url']})" for i, h in enumerate(hits, start=1)
    )
    result = f"{combined_text}\n\n**Bronnen (uit de database):**\n{sources}"
    if verify:
        if flagged:
            result += "\n\n**⚠ Controle: niet overtuigend onderbouwd door de geciteerde bron:**\n"
            result += "\n".join(f"- {f}" for f in flagged)
        else:
            result += "\n\n**Controle:** alle bronregels zijn in één batch-check als gegrond beoordeeld."
    return result


if __name__ == "__main__":
    brief = "bos, 8 tot 10 jaar, 2 uur, buiten rennen en opdrachten uitvoeren in groepjes"
    print(generate_activity(brief, k=2, min_age=8, max_age=10, terrein="bos", verify=False))