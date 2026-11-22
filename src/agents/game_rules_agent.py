from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

from src.generation.get_llm import invoke_llm

if TYPE_CHECKING:
    from agents.base import ActivityContext


class GameRulesAgent:
    """Structureert uitsluitend spelregels, groepen, winconditie en echte materialen."""

    SYSTEM_PROMPT = """Je bent de Game Rules Agent voor een jeugdbeweging (Chiro).
Werk UITSLUITEND op basis van de opgegeven bronspellen. Verzin niets bij.

Strikte regels:
1. Kies het best passende spel ([Spel 1], [Spel 2], ...) en werk de kern uit.
2. Elke zin bij Spelregels, Groepsindeling/Winconditie en Materialen eindigt op [Spel N].
3. Bij **Materialen**: zet ALLEEN tastbare voorwerpen/spelmateriaal (ballen, kaartjes, flesjes, touw...).
   Zet NOOIT personen, leiding, tikkers of figuranten onder Materialen.
4. Ontbreekt een onderdeel (zoals expliciete winconditie of groepsgrootte) in de bron?
   Vermeld dit dan expliciet onder **Niet in de bronnen** (zonder [Spel N]).
5. Schrijf helder in het Nederlands, zonder denkplannen of Engelse termen."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def run(self, ctx: ActivityContext) -> str:
        prompt = f"""{self.SYSTEM_PROMPT}

BRIEF: {ctx.brief}

BRONSPELLEN:
{ctx.sources_text}

Geef exact deze koppen:
**Titel**
(korte, duidelijke titel gelinkt aan de gekozen bron)

**Groepsindeling en winconditie**
- ... [Spel N]

**Spelregels in stappen**
1. ... [Spel N]

**Materialen**
- ... [Spel N]

**Niet in de bronnen**
- ..."""
        return invoke_llm(self.llm, prompt).strip()