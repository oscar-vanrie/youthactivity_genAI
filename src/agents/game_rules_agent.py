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
4. Verzin NOOIT aantallen. Vul onder **Groepsindeling en winconditie** of **Spelregels in stappen**
   nooit zelf een spelersaantal, teamgrootte of tijdsduur (spelduur) in die niet letterlijk in de bron staat,
   ook niet als die aannemelijk lijkt. Gebruik uitsluitend de aantallen die letterlijk in de bron staan
   en eindig die regel met het juiste [Spel N].
   - Staat de groepsgrootte, het aantal spelers per team of de spelduur NIET letterlijk in de bron?
     Schrijf dan op die plaats exact: Niet gespecificeerd in de bron [Spel N]
     (met het [Spel N] van het spel waarop de regel betrekking heeft).
5. Noteer ONTBROKENDE onderdelen (zoals een ontbrekende winconditie, groepsgrootte of spelduur)
   UITSLUITEND onder **Niet in de bronnen** (zonder [Spel N]). Verzin ze daar niet en vul ze elders niet in.
6. Schrijf helder in het Nederlands, zonder denkplannen of Engelse termen."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def run(self, ctx: ActivityContext) -> str:
        prompt = f"""{self.SYSTEM_PROMPT}

BRIEF: {ctx.brief}

BRONSPELLEN:
{ctx.sources_text}

BELANGRIJK — GEEN VERZONNEN AANTALLEN:
Staan groepsgrootte, aantal spelers per team of spelduur NIET letterlijk bij een spel in de bronnen?
Verzin ze dan NIET. Schrijf op die plaats exact: Niet gespecificeerd in de bron [Spel N]
en vermeld het ontbrekende getal uitsluitend onder **Niet in de bronnen** (zonder [Spel N]).
Gebruik alleen aantallen die letterlijk in de bron staan.

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