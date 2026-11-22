from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from .game_rules_agent import GameRulesAgent
from .safety_age_agent import SafetyAgeAgent
from .storyteller_agent import StorytellerAgent


@dataclass
class ActivityContext:
    brief: str
    sources_text: str
    min_age: int | None = None
    max_age: int | None = None
    terrein: str | None = None


def run_agents_parallel(
    llm: BaseChatModel,
    ctx: ActivityContext,
) -> tuple[str, str, str]:
    """Draait eerst GameRulesAgent en daarna StorytellerAgent + SafetyAgeAgent parallel."""
    rules_agent = GameRulesAgent(llm)
    storyteller_agent = StorytellerAgent(llm)
    safety_agent = SafetyAgeAgent(llm)

    rules_text = rules_agent.run(ctx)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_story = pool.submit(storyteller_agent.run, ctx, rules_text)
        fut_safety = pool.submit(safety_agent.run, ctx, rules_text)
        story_text = fut_story.result()
        safety_text = fut_safety.result()

    return rules_text, story_text, safety_text