"""
Minimale generator: brief -> search_games() -> één activiteit.
Bronvermelding komt uit de database, niet uit het model.

Gebruik vanuit de rootmap:
    python -m src.generation.generate
"""

from langchain_ollama import ChatOllama

from src.rag.search import search_games
import re


MODEL = "qwen3:8b"

SYSTEM_RULES = """Je bent een assistent die activiteiten voor jeugdbewegingen schrijft.
Strikte regels:
1. Gebruik UITSLUITEND de bronspellen. Geen eigen spelregels, materialen of thema's.
2. Elke spelregel en elk materiaal krijgt een bronverwijzing [Spel N].
3. Staat iets niet letterlijk of direct afleidbaar in de bronnen? Schrijf dan "Niet in de bronnen" en laat het weg.
4. Het thema uit de brief mag je alleen als verpakking gebruiken (de naam en verhaallijn), nooit als vervanging van de spelregels.
5. Verzin geen URL's. Die worden apart toegevoegd."""



def check_claims(text: str, hits: list[dict]) -> list[str]:
    """Markeert zinnen die geen enkel 'inhoudswoord' uit de bronnen bevatten."""
    corpus = " ".join(h["content"].lower() for h in hits)
    flagged = []
    for line in text.splitlines():
        if not line.strip() or "[Spel" not in line:
            continue
        words = [w for w in re.findall(r"[a-zà-ÿ]{5,}", line.lower())
                 if w not in {"spel", "groepen", "kinderen", "gebruik", "zorgen", "moeten"}]
        if words and not any(w in corpus for w in words):
            flagged.append(line.strip())
    return flagged


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        meta = h["metadata"]
        header = f"[Spel {i}] {h['title']} | leeftijd: {meta.get('leeftijd', '')} | terrein: {meta.get('terrein', '')}"
        blocks.append(f"{header}\n{h['content']}")
    return "\n\n".join(blocks)


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
        return "Geen geschikte spellen gevonden voor deze brief. Probeer strict_age=False of een bredere brief."

    print("Opgehaalde bronnen:")
    for i, h in enumerate(hits, start=1):
        print(f"  [Spel {i}] {h['title']} | {h['source_url']} | leeftijd: {h['metadata'].get('leeftijd', '')}")
    print()

    prompt = f"""{SYSTEM_RULES}

BRIEF: {brief}

BRONSPELLEN:
{build_context(hits)}

Schrijf de activiteit in het Nederlands met deze onderdelen:
- Titel
- Verhaal (kort, één alinea)
- Spelregels in stappen, elke stap met [Spel N]
- Materialen, elk met [Spel N]
- Veiligheidspunten, elk met [Spel N]
- Niet in de bronnen: lijst wat je miste"""

    llm = ChatOllama(model=MODEL, temperature=0)
    text = llm.invoke(prompt).content
    flagged = check_claims(text, hits)
    if flagged:
        text += "\n\n**⚠ Niet terug te vinden in de bronnen:**\n" + "\n".join(f"- {f}" for f in flagged)


    sources = "\n".join(
        f"- [Spel {i}] {h['title']} ({h['source_url']})" for i, h in enumerate(hits, start=1)
    )
    return f"{text}\n\n**Bronnen (uit de database):**\n{sources}"


if __name__ == "__main__":
    brief = "bos, 8 tot 10 jaar, 2 uur, zoekspel met opdrachten"
    print(generate_activity(brief, k=3, min_age=8, max_age=10, terrein="bos"))