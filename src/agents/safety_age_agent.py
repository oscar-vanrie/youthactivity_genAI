from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

from src.generation.get_llm import invoke_llm

if TYPE_CHECKING:
    from agents.base import ActivityContext


class SafetyAgeAgent:
    """Beoordeelt leeftijd en terrein en geeft praktische veiligheid & aanpassingen."""

    SYSTEM_PROMPT = """Je bent de Safety & Age Adaptability Agent voor een jeugdbeweging.
Veiligheidspunten uit de bron: neem ALLEEN zinnen op die expliciet over veiligheid, risico's, toezicht of letselpreventie in de bron gaan (eindigend op [Spel N]). Kopieer NOOIT gewone spelregels. Staat er niets over veiligheid in de bron, schrijf dan exact: - Geen specifieke veiligheidspunten vermeld in de bron.

Strikte regels:
1. Maak een duidelijk onderscheid tussen:
   - **Veiligheidspunten uit de bron**: alleen zaken die letterlijk/direct in de bron staan, eindigend op [Spel N].
   - **Leeftijds- en terreininschatting (praktisch advies)**: eigen praktische beoordeling voor de opgegeven leeftijdsgroep en het terrein (bijv. zichtbaarheid, wortels/kuilen, spelduur, groepsgrootte, speluitleg). Gebruik hier GEEN [Spel N].
2. Geef concrete aanpassingen als de spelregels of het terrein dat vragen (zonder de kernregels van het spel te breken).
3. Schrijf concreet en praktisch in het Nederlands, zonder denkplannen."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def run(self, ctx: ActivityContext, rules_output: str) -> str:
        age_info = f"{ctx.min_age}-{ctx.max_age} jaar" if ctx.min_age or ctx.max_age else "niet gespecificeerd"
        prompt = f"""{self.SYSTEM_PROMPT}

BRIEF: {ctx.brief}
LEEFTIJD: {age_info} | TERREIN: {ctx.terrein or "niet gespecificeerd"}

VASTGESTELDE SPELKERN:
{rules_output}

BRONSPELLEN:
{ctx.sources_text}

Geef exact deze koppen:
**Veiligheidspunten uit de bron**
- ...

**Leeftijds- en terreininschatting (praktisch advies)**
- ..."""
        return invoke_llm(self.llm, prompt).strip()