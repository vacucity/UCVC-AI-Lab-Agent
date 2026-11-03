<img width="1254" height="1254" alt="image" src="https://github.com/user-attachments/assets/c01a339e-a07f-4212-9a5d-5d0947a30504" />

# UCVC—AI—LabAgent

**Building LLM agents with real engineering tools: LangChain, LangGraph, Postgres/pgvector, MCP and pytest.**

These are my study notebooks for [Datawhale's Hello-Agents](https://github.com/datawhalechina/hello-agents) tutorial, rewritten engineering-first. Where the book builds its own framework, these notebooks use the tools teams actually ship with. Every notebook was run end to end against a real model (DeepSeek through its OpenAI-compatible API), and the outputs you see are real.

The last two chapters apply multi-agent design to two of my own projects, [ScenicMind](https://github.com/vacucity/ScenicMind) and [ResearchFlow](https://github.com/vacucity/Group-Research), and reproduce their core frameworks.

---

## Contents

| Part | Chapter | What it covers | Main tools | Hello-Agents chapter |
|---|---|---|---|---|
| **1. Introduction** | 1. Introduction and example | What an agent is, agent types, task environments, the perceive–think–act loop, a minimal working agent | OpenAI-compatible API, `urllib` | 1 |
| **2. Build your LLM agent** | 2. Three paradigms | ReAct, Plan-and-Solve and Reflection written from scratch: system prompts, output parsing, a tool executor, the core loop | `openai`, `requests` | 4 |
| | 3. Framework development | The same kinds of task in five frameworks: AutoGen (a software-team chat), AgentScope (a multi-agent game), CAMEL (role-playing to write an e-book), LangChain (an exchange-rate assistant), LangGraph (graph structure); strengths and limits of each | AutoGen, AgentScope, CAMEL, LangChain, LangGraph | 6 |
| **3. Advanced application** | 4. Agent memory | Short-term memory with a Postgres checkpointer; long-term memory with a pgvector store; two write paths (tool calls in the hot path, background extraction with ADD/UPDATE/DELETE/NOOP); reading with injection and reranking; isolation and deletion | LangGraph `PostgresSaver` / `PostgresStore`, pgvector, bge-m3 | 8 (memory) |
| | 5. RAG in agents | PDF ingestion with chapter metadata; vector, keyword, BM25 and hybrid (RRF) retrieval; query rewriting and reranking; Hit@5/MRR evaluation; two-step, agentic and corrective RAG; an agent with both memory and a knowledge base | pypdf, `rank_bm25`, pgvector, LangChain 1.x `create_agent` + middleware | 8 (retrieval) |
| | 6. Context engineering | Context rot; the anatomy of a context; budgeted context assembly; four ways to handle long conversations (trim, clear tool results, summarise); KV-cache-friendly prompts; just-in-time file tools, sub-agents and structured notes | LangChain middleware (`SummarizationMiddleware`, `ContextEditingMiddleware`) | 9 |
| | 7. Agent communication | MCP from scratch: an FX server over stdio, the raw JSON-RPC protocol, the Chapter 5 knowledge base as a Streamable HTTP server with bearer auth, connecting servers to an agent, tool poisoning and a fingerprint allowlist; then A2A and ANP | `mcp` (FastMCP), `langchain-mcp-adapters`, uvicorn | 10 |
| | 8. Agent evaluation | A BFCL-style tool-call benchmark with AST matching and distractor tools; pass@k vs pass^k; GAIA-style exact matching; calibrating LLM-as-judge (rubrics, references, position swapping); win rate on real projects; evaluation as a pytest gate in CI | pytest, LangChain tools | 12 |
| **4. Multi-agents and projects** | 9. Multi-agents: sequential | Multi-agent architecture patterns, then **ScenicMind**'s five-role report pipeline (coordinator → collector → analyst → writer → reviewer) reproduced in LangGraph: contracts, context compression, a single revision, template fallback, background job + polling | LangGraph, Pydantic structured output | 13 (applied to ScenicMind) |
| | 10. Multi-agents: hybrid | **ResearchFlow**'s ReviewOS literature-review engine reproduced as a hybrid graph: per-paper fan-out with `Send`, parallel analysis branches with a join, a per-section reflect-and-revise subgraph, SSE progress, and crash-and-resume from a SQLite checkpoint | LangGraph `Send`, subgraphs, `SqliteSaver`; ResearchFlow's own prompt registry and validator | 14 (applied to ResearchFlow) |

## Selected results from the reference runs

Results will vary a little between runs; these are from my machine.

- **Retrieval (Ch 5, 14 questions over the 806-page Hello-Agents PDF):**

  | Method | Hit@5 | MRR |
  |---|---|---|
  | Vector | 0.86 | 0.70 |
  | BM25 | 0.64 | 0.56 |
  | Hybrid (RRF) | 0.93 | 0.71 |
  | Multi-query | 1.00 | 0.88 |
  | Hybrid + LLM rerank | 0.93 | 0.85 |

  Postgres `ts_rank_cd` has no IDF, so it is not a substitute for BM25.
- **Context rot (Ch 6):** with look-alike distractors, recall fell from 23/24 at 2k tokens to 14/24 at 14k. Clearing old tool results cut total input tokens by 45% without losing an early fact; trimming forgot it. A timestamp at the top of the prompt gave 0% cache hits, and moving it to the end gave 61%.
- **MCP (Ch 7):** reusing one client session instead of spawning a process per call took 3 calls from 5.67 s to 0.32 s. A poisoned tool description really did leak into the agent's output until a description-fingerprint allowlist blocked it.
- **Evaluation (Ch 8):**
  - The easy tool-call set saturated (20/20 for both models), so a harder set with distractor tools was needed.
  - Shortening tool descriptions alone dropped accuracy from 1.00 to 0.92, which the pytest gate caught.
  - Without a reference answer, an LLM judge agreed with labels only 50% of the time on private facts; with a rubric and reference it reached 100%.
- **ScenicMind (Ch 9):** the LLM reviewer asked for a revision on every report, and the revised report still contained numbers that couldn't be traced to the data package. This is the case for checking numbers in code.
- **ResearchFlow (Ch 10):**
  - 8 papers were read in parallel in about 4 s; the 6 sections finished in about 44 s.
  - After a simulated crash in the last node, resuming from the checkpoint re-ran only that node: 1.6 s, 1 call.
  - Reproducing it exposed a mismatch in the original repo: the section-writer prompt asks for plain text, while its schema requires JSON.

---

## Repository layout

```text
.
├── Part1 Introduction/
│   └── chapter 1 Introduction and example.ipynb
├── Part2 Build your LLM agent/
│   ├── Chapter 2 Three paradims.ipynb
│   └── Chapter 3 Framework Development.ipynb
├── Part3 Advanced application/
│   ├── Chapter 4 Agent memory -RAG.ipynb
│   ├── Chapter 5 RAG in Agent.ipynb
│   ├── Chapter 6 Context Engineer.ipynb
│   ├── Chapter 7 Agent communication MCP.ipynb
│   ├── Chapter 8 Agent performance evaluation.ipynb
│   ├── fx_server.py          # Ch 7: FastMCP stdio server
│   ├── docs_server.py        # Ch 7: knowledge base as a Streamable HTTP MCP server
│   ├── evil_server.py        # Ch 7: poisoned server for the security demo
│   ├── e1_bfcl.py            # Ch 8: BFCL-style tool-call suite
│   └── test_agent_eval.py    # Ch 8: pytest regression gate
├── Part4 Multi-agents and projects/
│   ├── Chapter 9 Multi-agents Sequential.ipynb
│   └── Chapter 10 Multi-agents Hybird.ipynb
├── data/
│   ├── papers.json           # Ch 10: 8 arXiv agent papers (title, authors, year, abstract)
│   ├── demo_repo/            # Ch 6: sample code the just-in-time file tools read
│   └── eval/                 # Ch 8: two versions of each project's report for the win-rate cells
│       ├── researchflow/     #   original_state.json, v2b_state_full.json
│       └── scenicmind/       #   report_deep_dive_e1.md, report_deep_dive_derived.md, evidence_derived.json
├── e1_bfcl.py                # same suite, so the gate can also run from the repo root
├── test_agent_eval.py
├── .env.example
└── README.md
```

All paths in the notebooks are relative to the notebook's own folder (for example `Path.cwd().parent / "data"`), so the repo works wherever you clone it.

---

## Setup

### 1. Python environment

Python 3.11. The versions below are the ones the notebooks were run with.

```bash
conda create -n hello-agents python=3.11 -y
```

```bash
conda activate hello-agents
```

```bash
pip install "langchain==1.4.7" "langchain-core==1.7.3" "langchain-openai==1.7.3" "langgraph==1.2.18" "langgraph-checkpoint-postgres==3.2.0" "langgraph-checkpoint-sqlite==3.1.2" "langchain-text-splitters==1.1.3" "langchain-mcp-adapters==0.3.2" "mcp==1.30.0" "psycopg[binary,pool]==3.3.6" "pypdf==6.21.4" "rank-bm25==0.2.2" "python-dotenv==1.2.4" "openai" "pydantic==2.14.1" "httpx==0.28.1" "uvicorn" "pytest==9.2.0" "pandas==3.1.0" requests colorama ipykernel
```

Chapter 3 also needs the other frameworks:

```bash
pip install "autogen-agentchat==0.7.4" "autogen-ext[openai]==0.7.4" "agentscope==1.0.2" camel-ai
```

### 2. Model API key

Copy `.env.example` to `.env` in the repo root and fill in your own values. Never commit `.env`; it is already listed in `.gitignore`.

```bash
cp .env.example .env
```

Any OpenAI-compatible endpoint works. The notebooks were run with DeepSeek, with its thinking mode turned off through `extra_body={"thinking": {"type": "disabled"}}`, because thinking shares `max_tokens` with the answer and truncates structured output.

### 3. Postgres with pgvector (Chapters 4–7)

```bash
docker run -d --name hello-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=agent_memory -p 5432:5432 -v hello-pg-data:/var/lib/postgresql/data --restart unless-stopped pgvector/pgvector:pg17
```

This matches `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/agent_memory` in `.env.example`.

### 4. Local models with Ollama (Chapters 4–8)

Embeddings use `bge-m3`. Chapter 8 also compares against a small local model.

```bash
ollama pull bge-m3
```

```bash
ollama pull qwen2.5:7b
```

### 5. Data and project repositories

The small data files are bundled in `data/`. Two things are not, and you need to fetch them yourself:

**The Hello-Agents PDF (Ch 5, and Ch 6–7, which reuse its index).** It's over 100 MB, so it isn't committed. Download `Hello-Agents-EN-V1.0.3-20260717.pdf` from the [Hello-Agents repository](https://github.com/datawhalechina/hello-agents) into the repo root, or point `HELLO_AGENTS_PDF` at your copy. PDFs are git-ignored.

**The two project repositories (Ch 9 and Ch 10).** Clone them into `Project/`, which is also git-ignored:

```bash
git clone https://github.com/vacucity/ScenicMind Project/ScenicMind
```

```bash
git clone https://github.com/vacucity/Group-Research "Project/Group Research"
```

The Ch 9 and Ch 10 notebooks only import the project code. They patch a database lookup in memory and never modify the cloned repositories.

---

## Running

1. Select the `hello-agents` kernel and run each notebook top to bottom. Later cells depend on earlier ones.
2. Run Chapter 9 and Chapter 10 in separate kernels. Both projects ship a top-level Python package called `app`, so whichever one is imported first shadows the other. They also define `llm` with different settings.
3. **Chapter 7:**
   - The stdio server starts automatically.
   - Start the HTTP server in its own terminal; it defaults to port 8931, which you can change with `DOCS_MCP_PORT`:

   ```bash
   python "Part3 Advanced application/docs_server.py"
   ```

   - On Windows, Jupyter's `sys.stderr` has no file handle, so stdio clients need `errlog=<an open file>`. The notebook shows the patch.
4. **Chapter 8's CI gate:** run the first command from the repo root or from `Part3 Advanced application/`. Set `EVAL_TOOLS=poor` to watch the gate catch a bad change to the tool descriptions (in Windows cmd, run `set EVAL_TOOLS=poor` first).

   ```bash
   pytest -q --tb=short test_agent_eval.py
   ```

   ```bash
   EVAL_TOOLS=poor pytest -q --tb=short test_agent_eval.py
   ```

5. Generated files go into `scenicmind_runtime/` and `researchflow_runtime/` next to the notebooks: reports, uploads and the SQLite checkpoint. These folders are ignored by git.

## Things I learned the hard way

- Set a `timeout` and `max_retries` on every model client. A single stuck request without one hung a whole run.
- Python 3.11 does not allow a backslash inside an f-string expression. Compute values like Unicode ranges outside the f-string.
- In LangGraph:
  - fields written by parallel branches need a reducer;
  - joining branches of different lengths needs `add_edge([a, b, c], d)`, not three separate edges;
  - a subgraph sent through `Send` needs `output_schema`, or its input fields flow back and get appended twice.
- `with_structured_output` can return `None` when the model doesn't call the schema tool, so check for it.

## References

1. Datawhale. *Hello-Agents: Building Intelligent Agents from Scratch* (English edition, v1.0.3). GitHub: <https://github.com/datawhalechina/hello-agents>. These notebooks follow its chapter structure, and Chapter 5 uses the book PDF as its retrieval corpus. See that repository for its license.

## Acknowledgements

- [Datawhale](https://github.com/datawhalechina) and the Hello-Agents authors, for the course these notes follow.
- [ScenicMind](https://github.com/vacucity/ScenicMind) and [ResearchFlow](https://github.com/vacucity/Group-Research), the projects used in Part 4.
