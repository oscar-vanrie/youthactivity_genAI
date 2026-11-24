from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

from src.generation.get_llm import invoke_llm

if TYPE_CHECKING:
    from agents.base import ActivityContext


class StorytellerAgent:
    """Maakt een kort verhaal op basis van de bron en een aparte eigen thema-aanpassing."""

    SYSTEM_PROMPT = """Je bent de Storyteller Agent voor een jeugdbeweging.
Je maakt een speels kaderverhaal en een aparte eigen thema-aanpassing passend bij de brief en doelgroep.

Strikte regels:
1. **Verhaal**: korte, levendige alinea die aansluit bij het bronspel en de doelgroep, met [Spel N] waar het op de bron steunt. Verdraai de spelregels of winconditie NIET.
2. **Thema-aanpassing (zelf verzonnen)**: eigen creatieve aankleding/sfeer die past bij de brief en leeftijd. Gebruik hier GEEN [Spel N]-verwijzingen.
3. Schrijf in natuurlijk, verzorgd Nederlands zonder denkplannen."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Bewaar het LLM voor de aanroepen."""
        self.llm = llm

    def run(self, ctx: ActivityContext, rules_output: str) -> str:
        """Schrijf een kaderverhaal en een aparte eigen thema-aanpassing."""
        age_info = f"{ctx.min_age}-{ctx.max_age} jaar" if ctx.min_age or ctx.max_age else "onbekend"
        prompt = f"""{self.SYSTEM_PROMPT}

BRIEF: {ctx.brief}
LEEFTIJD: {age_info} | TERREIN: {ctx.terrein or "niet gespecificeerd"}

VASTGESTELDE SPELKERN:
{rules_output}

BRONSPELLEN:
{ctx.sources_text}

Geef exact deze twee koppen:
**Verhaal**
...

**Thema-aanpassing (zelf verzonnen)**
- ..."""
        return invoke_llm(self.llm, prompt).strip()