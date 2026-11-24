## Data and RAG Pipeline

2.1 Problem and Goal The retrieval component must supply the generation pipeline (and Game Rules agent) with usable building blocks—rules, materials, terrain, and safety notes—extracted from real youth-movement games. Rather than finding one exact match, retrieval needs to return a small, relevant set of complete game descriptions that match a user's activity brief (target age, terrain, duration, and theme).

2.2 Data Collection and Cleaning We scraped 2,**442** games from chiro.be into games_raw.json. Each entry includes a title, source **URL**, free-text description, materials, and metadata (duration, age, player count, game type, terrain, and themes). Because the raw data contains noisy placeholders (e.g., *andere*, *null*), inconsistent age strings (e.g., *van 6 tot jaaar*), and missing fields, we built a preprocessing step to strip placeholders, join list fields, and parse ages into integer min_age and max_age bounds for filtering.

2.3 Chunking Strategy Instead of fixed-size character splitting, we store one document per game so that game mechanics and win conditions remain intact. Each chunk combines the title, a structured metadata header (age, themes, type, terrain, duration, players), and the game description. Placing metadata first ensures critical filtering context remains within the embedding model's token window. Processed chunks are saved to chunks.json as a reproducible checkpoint before indexing.

2.4 Embeddings and Vector Store Chunks are embedded using intfloat/multilingual-e5-base, a local multilingual model supporting Dutch and a **512**-token context window, stored in ChromaDB. Following the E5 specification, documents are prefixed with *passage:* and queries with *query:*. We replaced an earlier model (paraphrase-multilingual-MiniLM-**L12**-v2) because its **128**-token limit truncated game descriptions and metadata.

2.5 Retrieval Architecture search_games(query, k, min_age, max_age, terrain) supports two configurable modes via USE_HYBRID:

Semantic Mode: Ranks chunks purely by ChromaDB dense vector similarity.

Hybrid Mode: Combines dense vector rankings with sparse **BM25** keyword rankings (rank_bm25) using Reciprocal Rank Fusion (RRF, k=60).

Age-overlap and terrain filters are applied post-ranking over a candidate pool of **300** games to prevent over-filtering before relevance scoring.

2.6 Retrieval Evaluation and Limitations On an initial benchmark of 19 short queries (each labelled with one target game), dense retrieval slightly outperformed hybrid retrieval:

| Retrieval Mode | Hit@3 | **MRR** |
| --- | --- | --- |
| Semantic (E5-base) | 0.**263** | 0.**237** |
| Hybrid (Semantic + **BM25** via **RRF**) | 0.**211** | 0.**175** |
Qualitative inspection showed that **BM25** boosts exact domain terminology (e.g., *aanvallers en verdedigers* improved from rank 64 to 4) but penalizes broader conceptual queries (e.g., *geld verdienen* dropped from rank 4 to 30). Manual review of top-3 results across five realistic briefs confirmed that retrieved games were largely usable, while exposing two dataset bottlenecks: relevant mechanics can be missed if phrased differently (e.g., trade mechanics in *Ongelooflijk, ongelijkheid bestaat nog steeds*), and incomplete entries (e.g., *Simsspel*, which only points to an external appendix) lack playable rules.

## Grounded Generation, Model Selection & Verification

3.1 Architecture & Provider Abstraction To transform retrieved games into structured Dutch activity guides without hallucinating rules, we implemented a modular generation layer (get_llm.py and generate.py) using LangChain. Rather than hardcoding a single model, get_llm() abstracts the **LLM** provider via environment variables (LLM_PROVIDER, LLM_MODEL) and wraps inference in invoke_llm(), which includes automatic retries and a local fallback to Ollama (qwen3:8b) if a cloud **API** fails or hits rate limits.

3.2 Iterative Testing: What Failed and Why During development, we tested multiple local and cloud-hosted models and observed three concrete failure modes:

Reasoning Leakage & Truncation (qwen3:8b via Ollama): While qwen3:8b runs offline without rate limits, it frequently emitted lengthy internal English reasoning traces (*We need to write an activity...*) directly into the output stream. Under token limits (num_predict), the model exhausted its budget reasoning in English before generating the Dutch activity, resulting in truncated or unusable outputs.

**API** Deprecations & Empty Responses (Groq Cloud): Testing several cloud endpoints revealed model availability and formatting pitfalls. llama-3.3-70b-versatile and meta-llama/llama-4-scout-17b-16e-instruct returned **404** model_not_found errors on our **API** tier, while initial runs with reasoning-capable cloud models returned empty content fields when reasoning tokens were separated into additional_kwargs.

Rule Distortion & False-Positive Citation Flags: In early single-pass generations, models occasionally altered win conditions (e.g., claiming a team wins by collecting all cards rather than filling a bottle in Regenboogspel). Conversely, overly literal claim verification flagged valid Dutch paraphrases (such as plural/synonym variations) as ungrounded hallucinations.

3.3 Final Model Choice & Latency Optimization We standardized on openai/gpt-oss-20b (via Groq) paired with response sanitization (strip_thinking() and _extract_content()) and calibrated prompting:

Why openai/gpt-oss-20b worked best: Once response extraction handled reasoning metadata properly and explicit Dutch output rules were enforced, openai/gpt-oss-20b consistently produced complete, well-structured Dutch activity sheets—preserving exact game mechanics, materials, and [Spel N] source citations while separating invented thematic framing under *Thema-aanpassing (zelf verzonnen)*.

Reducing Latency from 3–5 Minutes to ~15 Seconds: Our first prototype executed up to 15–18 sequential **LLM** calls per run (per-source rule extraction, main generation, and line-by-line verify_claim() calls), taking 3 to 5 minutes per activity. We eliminated redundant per-source extraction calls (feeding cleaned chunks directly via build_context()) and consolidated citation checking into one batched verification call (check_citations_batch()) with k=2, cutting runtime by roughly 90% while maintaining grounding checks.


3.4 Multi-Agent Architecture & Parallel Execution (agents/) To address the limitations of a monolithic prompt in generate.py, we refactored activity generation into modular agent classes under agents/ (game_rules_agent.py, storyteller_agent.py, safety_age_agent.py, and base.py), orchestrated via an ActivityContext dataclass and run_agents_parallel():

Why 3 Agents Instead of 4: Although the initial proposal table listed four agent categories (including a separate materials role), we consolidated materials into GameRulesAgent to keep the number of **LLM** calls low and protect response speed. We added explicit prompt constraints so GameRulesAgent includes only tangible physical equipment—never people, leaders, or taggers (tikkers)—and marks missing specifications under Niet in de bronnen.

Storyteller Agent: Produces a short narrative grounded in the selected source game without altering game mechanics or win conditions, while isolating invented framing under Thema-aanpassing (zelf verzonnen).

Safety & Age Adaptability Agent: Separates literal safety rules found in the source ([Spel N]) from practical terrain and age-group guidance (without [Spel N]), tailored to the requested age range (e.g., 8–10 vs. 11–14 or 14–16 years) and terrain (e.g., forest vs. grass field or indoor room).

Parallel Execution for Low Latency: Rather than running all agents sequentially or adding a draft-and-critic loop that multiplies **LLM** calls, GameRulesAgent runs first to establish the core game structure, and StorytellerAgent and SafetyAgeAgent then run concurrently using a ThreadPoolExecutor (max_workers=2). This preserves clear role separation while keeping total generation time fast (around a few seconds to ~15 seconds depending on **API** responsiveness).

3.5 Multi-Agent Findings, Verification Trade-Offs & Streamlit Demo (app.py) Testing the multi-agent pipeline across multiple briefs, ages, and terrains surfaced several practical engineering trade-offs:

Why Default Citation Verification Was Set to verify=False: Running an extra **LLM** verification pass (check_citations_batch()) added latency and produced noisy false positives—for example, soft hyphens (\u2011) in headings such as Thema‑aanpassing bypassed keyword skip lists using standard hyphens (-), and **LLM** self-checking over-flagged valid game steps. Because role separation and explicit section rules already structure grounded vs. invented content, we set verify=False by default for speed and cleaner demo output while keeping the option toggleable.

Remaining Prompt & Parallelism Caveats: We observed occasional self-contradictions in GameRulesAgent (such as stating a group size or duration with [Spel 1] while also listing missing group size or duration under Niet in de bronnen), minor phrasing slips in invented themes, and lack of cross-visibility between parallel agents (for instance, SafetyAgeAgent does not inspect StorytellerAgent's draft text because they execute simultaneously). These trade-offs directly inform our next iteration: either tightening prompt constraints further or adding a lightweight post-check where needed without hurting latency.

Streamlit Demo Interface (app.py): We built a Streamlit front end on top of generate_activity() with sidebar filters (min_age, max_age, terrein, k, strict_age, verify), quick scenario presets, color-coded visual cards explaining each agent's responsibility, section grouping by agent, and Markdown export.

## 4. Current Status & Remaining Roadmap

Relative to the project proposal, the core text pipeline—hybrid **RAG** retrieval, modular multi-agent generation, and an interactive Streamlit demo—is functional. Remaining work focuses on:

Visual Asset Generation: Adding theme-consistent printable visuals (such as badges or game maps) aligned with the generated activity theme.

Export & **MCP** Integration: Adding formatted document export (e.g., **PDF** and Google Docs / Drive integration via **MCP**).

Evaluation & Final Documentation: Running systematic benchmarks (including **RAGAS**) across scenarios and finalizing the technical report and presentation.