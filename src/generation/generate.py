import re

from langchain_ollama import ChatOllama

from src.rag.search import search_games

MODEL = "qwen3:8b"
OVERLAP_THRESHOLD = 0.12

STOPWORDS = {
    "gebruik", "zorgen", "moeten", "kinderen", "spelers", "groepen", "spel",
    "leden", "gaan", "worden", "wordt", "even", "over", "aandacht", "onder",
    "tijdens", "voor", "naar", "daarna", "zodat", "elkaar", "bijvoorbeeld",
    "alleen", "maken", "doen", "krijgen", "laten", "komen", "zetten",
}

SYSTEM_RULES = """Je ontwerpt een duidelijke, concrete jeugdbewegingsactiviteit in het Nederlands.

Strikte regels:
1. Kies EEN hoofdbron ([Spel 1], [Spel 2] of [Spel 3]) met de beste spelmechaniek voor de opdracht. Gebruik een tweede bron ALLEEN als die inhoudelijk rijk is en echt aanvult. Negeer bronnen die te weinig spelregels bevatten.
2. Schrijf GEEN geneste sub-bullets onder spelregels of materialen. Maak van elke stap of elk item één duidelijke regel op het eind voorzien van [Spel N].
3. Gebruik het gevraagde thema (zoals ruimte / alien invasie / smokkelen) in Titel en Verhaal, en koppel dit logisch aan de mechaniek van de gekozen hoofdbron zonder willekeurige elementen uit andere spellen te mengen.
4. Reken tijden om: 120 minuten is gelijk aan 2 uur.
5. Zet bij "Niet in de bronnen" alleen wat ontbreekt of werd aangepast ten opzichte van de bronnen.
6. Schrijf verzorgd, correct Nederlands zonder verzonnen woorden en zonder verzonnen URL's."""


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def stems(text: str) -> set[str]:
    words = {
        w for w in re.findall(r"[a-zà-ÿ]{4,}", text.lower())
        if w not in STOPWORDS
    }
    return {w[:5] for w in words}


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


def check_citations(text: str, hits: list[dict]) -> list[str]:
    source_stems = {i: stems(h["content"]) for i, h in enumerate(hits, start=1)}
    flagged = []
    active_section = False

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue

        if raw.startswith("**") and raw.endswith("**"):
            heading = raw.strip("* ").lower()
            active_section = any(
                key in heading for key in ("spelregels", "materialen", "veiligheid")
            )
            continue

        if not active_section or raw.startswith("⚠") or raw.startswith("- [Spel"):
            continue

        cleaned_line = re.sub(r"\*\*", "", raw)
        item = re.sub(r"^(\d+\.|[-*])\s*", "", cleaned_line).strip()
        if not item:
            continue

        refs = [int(n) for n in re.findall(r"\[Spel (\d+)\]", item)]
        valid_refs = [r for r in refs if r in source_stems]
        if not valid_refs:
            flagged.append(f"zonder geldige bronverwijzing: {item}")
            continue

        body_stems = stems(re.sub(r"\[Spel \d+\]", "", item))
        if not body_stems:
            continue

        best_overlap = max(
            len(body_stems & source_stems[r]) / len(body_stems) for r in valid_refs
        )
        if best_overlap < OVERLAP_THRESHOLD:
            flagged.append(f"lage overlap ({best_overlap:.0%}): {item}")

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

    prompt = f"""{SYSTEM_RULES}

BRIEF: {brief}

BRONSPELLEN:
{build_context(hits)}

Geef de output exact in dit format:

**Titel**
(één duidelijke titel)

**Verhaal**
(één korte, samenhangende alinea in het thema van de brief)

**Spelregels**
- (stap 1 met concrete actie uit de hoofdbron) [Spel N]
- (stap 2 met concrete actie uit de hoofdbron) [Spel N]
- (stap 3 met concrete actie uit de hoofdbron) [Spel N]

**Materialen**
- (materiaal 1) [Spel N]
- (materiaal 2) [Spel N]

**Veiligheidspunten**
- (veiligheidspunt 1 gebaseerd op de bron of het terrein) [Spel N]

**Niet in de bronnen**
- (wat ontbreekt of is aangepast t.o.v. de bronnen)"""

    llm = ChatOllama(model=MODEL, temperature=0.2)
    text = strip_thinking(llm.invoke(prompt).content)

    flagged = check_citations(text, hits)
    sources = "\n".join(
        f"- [Spel {i}] {h['title']} ({h['source_url']})" for i, h in enumerate(hits, start=1)
    )
    result = f"{text}\n\n**Bronnen (uit de database):**\n{sources}"
    if flagged:
        result += "\n\n**⚠ Controle: niet overtuigend onderbouwd door de geciteerde bron:**\n"
        result += "\n".join(f"- {f}" for f in flagged)
    else:
        result += "\n\n**Controle:** alle regels hebben een bronverwijzing met voldoende overlap."
    return result


if __name__ == "__main__":
    test_brief = (
        "ruimte, alien invasie, raketonderdelen verzamelen en smokkelen zonder gepakt te worden, "
        "6 tot 8 jaar, 2 uur, bos"
    )
    print(generate_activity(test_brief, k=3, min_age=6, max_age=8, terrein="bos"))