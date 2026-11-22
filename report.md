## Data and RAG Pipeline

2.1 Problem and Goal The retrieval component must supply the generation pipeline (and upcoming Game Rules agent) with usable building blocks—rules, materials, terrain, and safety notes—extracted from real youth-movement games. Rather than finding one exact match, retrieval needs to return a small, relevant set of complete game descriptions that match a user's activity brief (target age, terrain, duration, and theme).

2.2 Data Collection and Cleaning We scraped 2,**442** games from chiro.be into games_raw.json. Each entry includes a title, source **URL**, free-text description, materials, and metadata (duration, age, player count, game type, terrain, and themes). Because the raw data contains noisy placeholders (e.g., *andere*, *null*), inconsistent age strings (e.g., *van 6 tot jaaar*), and missing fields, we built a preprocessing step to strip placeholders, join list fields, and parse ages into integer min_age and max_age bounds for filtering.

2.3 Chunking Strategy Instead of fixed-size character splitting, we store one document per game so that game mechanics and win conditions remain intact. Each chunk combines the title, a structured metadata header (age, themes, type, terrain, duration, players), and the game description. Placing metadata first ensures critical filtering context remains within the embedding model's token window. Processed chunks are saved to chunks.json as a reproducible checkpoint before indexing.

2.4 Embeddings and Vector Store Chunks are embedded using intfloat/multilingual-e5-base, a local multilingual model supporting Dutch and a **512**-token context window, stored in ChromaDB. Following the E5 specification, documents are prefixed with *passage:* and queries with *query:*. We replaced an earlier model (paraphrase-multilingual-MiniLM-**L12**-v2) because its **128**-token limit truncated game descriptions and metadata.

2.5 Retrieval Architecture search_games(query, k, min_age, max_age, terrain) supports two configurable modes via USE_HYBRID:

Semantic Mode: Ranks chunks purely by ChromaDB dense vector similarity.

Hybrid Mode: Combines dense vector rankings with sparse **BM25** keyword rankings (rank_bm25) using Reciprocal Rank Fusion (**RRF**, # 𝑘 60 k=60).

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

3.4 Current Limitations & Next Steps While the current generator produces reliable, grounded single-pass activities, a monolithic prompt still conflates three distinct responsibilities: extracting strict rules, inventing age-appropriate themes, and evaluating terrain/safety risks (which currently relies too heavily on sparse source text). Our next step is refactoring generate.py into a Multi-Agent pipeline—splitting tasks across specialized Game Rules, Storyteller, and Safety & Age Adaptability agents—before running end-to-end evaluation with **RAGAS**.