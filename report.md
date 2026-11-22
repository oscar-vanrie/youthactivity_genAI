## Data and RAG Pipeline

2.1 Problem and goal

The retrieval component must supply the Game Rules agent with usable building blocks (rules, materials, terrain, safety notes) taken from real youth-movement games. The end goal is an activity brief (target age, terrain, duration, theme) that the agents turn into a new activity. Retrieval therefore only needs to return a small set of relevant, complete game descriptions, not one exact match.

2.2 Data

We scraped 2,**442** games from chiro.be into games_raw.json. Each game has a title, source **URL**, free-text description, materials and metadata (duration, age, number of players, type, terrain, themes). The data is noisy: placeholders such as “andere” or “null”, age ranges in free text (e.g. “van 6 tot jaaar”), and missing fields.

2.3 Chunking

We use one document per game instead of splitting by character count, because the rules of a game must stay together. The page content combines the title, a metadata header (age, themes, type, terrain, duration, players) and the description. Placeholders are cleaned, lists are joined into strings, and the age is parsed into integer min_age and max_age fields so that it can be filtered on. The metadata is placed first so that it stays within the model’s token limit. The output is stored in chunks.json as a checkpoint before embedding.

2.4 Embeddings and vector store

Each chunk is embedded with intfloat/multilingual-e5-base, a free, local, multilingual model that handles Dutch and up to **512** tokens. Following the model’s convention, documents are prefixed with “passage:” and queries with “query:”. The vectors are stored in ChromaDB. An earlier model (paraphrase-multilingual-MiniLM-**L12**-v2) was replaced because it only processes **128** tokens, which truncated most of the metadata.

2.5 Retrieval

search_games(query, k, min_age, max_age, terrain) supports two modes. Semantic mode ranks by ChromaDB similarity. Hybrid mode combines the semantic ranking with a **BM25** keyword ranking (rank_bm25) using Reciprocal Rank Fusion with k=60. Filters on age overlap and terrain are applied after ranking, with a candidate pool of **300**. A switch (USE_HYBRID) selects the mode, so both can be compared on the same index.

2.6 Evaluation

We first measured with a small set of 19 short queries and one labelled game per query. Results (Hit@3 / **MRR**):

| Mode | Hit@3 | **MRR** |
| --- | --- | --- |
| Semantic | 0.**263** | 0.**237** |
| Hybrid (**RRF**) | 0.**211** | 0.**175** |

Per query, **BM25** improves specific terms (e.g. “aanvallers en verdedigers”: rank 64 to 4, “gevoel van paniek”: rank **112** to 12) and hurts generic ones (e.g. “geld verdienen”: rank 4 to 30). The difference between the modes on this set is one hit and is not significant.

Because the labels only measure one game per query, we also did a manual review of the top 3 results for five brief-like queries. Each result was judged as usable (yes / partly / no) by reading the full description. Example: the query “gevoel van paniek en angst in een spel” returned three games judged usable, and “spel met klakkebuizen en propjes schieten” returned two usable and one not usable (a game that shares only the word “schieten”). The review also revealed a retrieval miss: “Ongelooflijk, ongelijkheid bestaat nog steeds”, which contains a trade mechanism relevant to “geld verdienen”, was not in the top 3. It also revealed a data problem: “Simsspel” is ranked highly but its text only refers to an appendix, so it has no usable rules.

2.7 Limitations

The test set is small and was labelled by one person, who also built the system. Labels only cover one game per query, so the scores are a lower bound. The evaluation does not yet measure the generated activity, which is the final goal. Data quality (missing text, parsing errors) limits what retrieval can return.