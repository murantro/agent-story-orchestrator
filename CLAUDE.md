# CLAUDE.md — AI Assistant Guide for agent-story-orchestrator

## Project Overview

**Emergent Narrative Engine** for a video game where stories arise organically from NPC decisions — not from scripted plots. Built on the **Google Agent Development Kit (ADK)**, scaffolded from `googleCloudPlatform/agent-starter-pack` v0.33.2. The backend provides NPC simulation, dialogue generation, event propagation, and memory retrieval, deployed as a containerized FastAPI service on Google Cloud Run.

The system models each NPC as a composite of hand-designed semantic vectors (intentions, emotions, personality, social influence, environment). The world's story emerges from the aggregate of NPC intentions and interactions.

## Repository Structure

```
agent-story-orchestrator/
├── .env                          # Root env (WS_NAME=agent-story-creator)
├── .idx/                         # IDX workspace config
├── CLAUDE.md                     # This file
└── agent-story-creator/          # Main project directory
    ├── .cloudbuild/              # CI/CD pipelines (Cloud Build)
    │   ├── pr_checks.yaml        # PR validation (lint, test)
    │   ├── staging.yaml          # Staging deploy + load test
    │   └── deploy-to-prod.yaml   # Production deployment
    ├── app/                      # Application source code
    │   ├── __init__.py           # Package init (lazy import of agent)
    │   ├── agent.py              # Core ADK agent definition & tools
    │   ├── fast_api_app.py       # FastAPI server + telemetry
    │   ├── app_utils/            # Shared utilities
    │   │   ├── typing.py         # Pydantic models (Request, Feedback)
    │   │   └── telemetry.py      # OpenTelemetry setup
    │   ├── models/               # Data models
    │   │   ├── npc_status.py     # NPCVectorialStatus — NPC state vectors
    │   │   └── events.py         # WorldEvent — event model + locality scales
    │   ├── simulation/           # Simulation engines (pure math, no LLM)
    │   │   ├── intention_engine.py  # Intention vector computation
    │   │   └── emotion_engine.py    # Emotion decay + event impact
    │   ├── events/               # Event propagation system
    │   │   ├── event_queue.py    # Priority queue by game-time
    │   │   └── propagation.py    # Locality cascade with attenuation
    │   ├── dialogue/             # Dialogue generation pipeline
    │   │   ├── tier_selector.py  # 3-tier selection (template/local/cloud)
    │   │   └── template_engine.py # Ambient barks from parameterized templates
    │   └── memory/               # NPC memory / vector DB layer
    │       ├── base.py           # Abstract MemoryStore interface
    │       └── in_memory_store.py # Brute-force cosine similarity (dev/test)
    ├── deployment/               # Terraform IaC (dev + prod)
    ├── notebooks/                # Jupyter notebooks for testing/eval
    ├── tests/
    │   ├── unit/                 # Unit tests (40 tests, no GCP credentials needed)
    │   ├── integration/          # Integration & E2E tests (needs GCP)
    │   └── load_test/            # Locust load tests
    ├── Dockerfile                # Python 3.11-slim, uv, port 8080
    ├── Makefile                  # Build/dev/deploy commands
    ├── pyproject.toml            # Project config, deps, tool settings
    └── uv.lock                   # Dependency lock file
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10-3.13 (Docker uses 3.11) |
| Agent Framework | Google ADK (`google-adk>=1.15.0,<2.0.0`) |
| LLM (Cloud) | Gemini Flash via Google ADK |
| LLM (Local) | Ollama + Llama 3.1 8B (planned) |
| Simulation | numpy vectorized math (Python) + Unity ECS (C# client) |
| Web Framework | FastAPI 0.115.x + Uvicorn 0.34.x |
| Communication | gRPC + Protobuf (planned; currently REST) |
| Package Manager | **uv** 0.8.13 |
| Vector DB | In-memory (dev); Qdrant embedded (SP) / pgvector (MP) planned |
| State DB | SQLite (SP) / PostgreSQL via asyncpg (MP) planned |
| Testing | pytest, pytest-asyncio, Locust (load) |
| Linting | ruff, ty (type checker), codespell |
| Infrastructure | Terraform, Google Cloud Build, Cloud Run |
| Observability | OpenTelemetry, Cloud Logging, Cloud Trace |

## Architecture Overview

### NPC Vectorial Status Model

Each NPC has hand-designed semantic vectors (NOT LLM embeddings):

| Vector | Dimensions | Purpose |
|--------|-----------|---------|
| `intention` | 8 | survive, socialize, achieve, explore, create, dominate, nurture, escape |
| `emotion` | 8 | Plutchik wheel: joy, sadness, anger, fear, surprise, disgust, trust, anticipation |
| `personality` | 5 | Big Five: openness, conscientiousness, extraversion, agreeableness, neuroticism |
| `social_influence` | 6 | cultural conformity, economic pressure, fashion, status, religious, political |
| `environment` | 4 | safety, resource abundance, weather comfort, crowding |

Intention is recomputed each tick via:
```
intention = normalize(w_personality * M_personality @ personality + w_emotion * M_emotion @ emotion + ...)
```

### Dialogue Pipeline (3 Tiers)

| Tier | When | Technology | Cost |
|------|------|-----------|------|
| Template | Ambient barks, no player interaction | Parameterized templates | $0 |
| Local LLM | Player talks to routine NPC | Ollama + Llama 8B (GPU) | $0 |
| Cloud LLM | Important NPCs, quests, multi-turn | Gemini Flash via ADK | ~$0.001/call |

### Event Propagation

Events cascade through locality scales with delay and attenuation:
```
personal -> family (1h, x0.8) -> city (4h, x0.5) -> regional (1d, x0.3) -> national (3d, x0.15) -> global (7d, x0.05)
```

## Common Commands

All commands run from `agent-story-creator/`:

```bash
make install          # Install all dependencies with uv
make playground       # Launch ADK web playground (local dev UI)
make local-backend    # Start FastAPI dev server with hot-reload
make test             # Run unit + integration tests
make lint             # Run codespell, ruff check & format, ty check
make deploy           # Deploy to Cloud Run
make setup-dev-env    # Provision dev infrastructure via Terraform
```

### Running tests directly

```bash
cd agent-story-creator
uv run pytest tests/unit/              # Unit tests only (no GCP needed)
uv run pytest tests/integration/       # Integration tests only (needs GCP)
uv run pytest                          # All tests
```

### Running linters directly

```bash
cd agent-story-creator
uv run codespell                       # Spell check
uv run ruff check . --fix              # Lint with auto-fix
uv run ruff format .                   # Format code
uv run ty check                        # Type check
```

## Code Conventions

### File & Naming

- **Files**: `snake_case.py`
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **All source files** include an Apache 2.0 license header (Copyright 2026 Google LLC)

### Architecture Patterns

- **Vectorial NPC model**: Hand-designed semantic vectors, not LLM embeddings
- **Pure math simulation**: numpy linear algebra for intention/emotion computation (no LLM per tick)
- **3-tier dialogue**: Template (free) -> Local LLM (free) -> Cloud LLM (paid)
- **Event cascading**: Events propagate through locality scales with attenuation
- **Async-first**: Use `async/await` for I/O-bound operations
- **Lazy imports**: `app/__init__.py` uses `__getattr__` to avoid loading ADK/GCP at import time

### Agent Definition Pattern

Tools are defined as standalone functions in `app/agent.py`, then passed to the Agent constructor:

```python
def tool_name(param: str) -> str:
    """Tool description.

    Args:
        param: Description.

    Returns:
        Result description.
    """
    return result

root_agent = Agent(
    model="gemini-3-flash-preview",
    name="agent_name",
    instruction="System prompt here.",
    tools=[tool_name],
)
```

### Ruff Configuration

- Line length: 88
- Target: Python 3.10
- Enabled rules: E, F, W, I (isort), C, B, UP, RUF
- Ignored: E501 (line length), C901 (complexity), B006 (mutable defaults)
- Use ASCII hyphens in docstrings (not en-dashes or minus signs) to avoid RUF002
- Always use `strict=True` in `zip()` calls to satisfy B905

### Testing Patterns

- **Unit tests**: `tests/unit/` — fast, no external dependencies, no GCP credentials needed
- **Integration tests**: `tests/integration/` — test agent streaming and full HTTP API (needs GCP)
- Tests use `pytest` fixtures with session/function scope
- E2E tests spin up a real FastAPI server in a fixture and make HTTP requests
- Async tests use `pytest-asyncio` with function-scoped event loops
- Unit test imports go through subpackages directly (e.g., `from app.models.npc_status import ...`) — this avoids triggering the ADK/GCP initialization chain

## Key Files to Understand

| File | Purpose |
|------|---------|
| `app/models/npc_status.py` | NPC vectorial status model (core data structure) |
| `app/models/events.py` | World event model, locality scales, propagation rules |
| `app/simulation/intention_engine.py` | Intention vector computation (numpy math) |
| `app/simulation/emotion_engine.py` | Emotion decay and event impact |
| `app/events/propagation.py` | Event cascade through locality scales |
| `app/events/event_queue.py` | Priority queue for scheduled events |
| `app/dialogue/tier_selector.py` | 3-tier dialogue selection logic |
| `app/dialogue/template_engine.py` | Template-based ambient dialogue |
| `app/memory/base.py` | Abstract memory store interface |
| `app/memory/in_memory_store.py` | Cosine similarity memory search (dev) |
| `app/agent.py` | ADK agent definition, tools, system instruction |
| `app/fast_api_app.py` | FastAPI app, routes, telemetry init |
| `app/app_utils/typing.py` | Request/Feedback Pydantic models |
| `pyproject.toml` | All project config, deps, linter settings |

## Development Workflow

1. Install dependencies: `make install`
2. Make changes in `app/`
3. Run linters: `make lint`
4. Run unit tests: `uv run pytest tests/unit/` (fast, no GCP needed)
5. Test locally: `make local-backend` or `make playground`
6. Commit and push — CI runs `pr_checks.yaml` (lint + test)
7. On merge to main: staging deploy -> load test -> production deploy

## Important Notes

- The package manager is **uv**, not pip or poetry. Always use `uv run` to execute Python commands or `uv sync` to install dependencies.
- The working directory for all make/test/lint commands is `agent-story-creator/`, not the repo root.
- The `.env` file at the repo root sets `WS_NAME=agent-story-creator`.
- `app/__init__.py` uses lazy imports (`__getattr__`) so that subpackages like `app.models`, `app.simulation`, etc. can be imported without triggering the ADK/GCP initialization.
- Terraform state and GCP credentials are not committed; infrastructure changes require `make setup-dev-env`.
- The `GEMINI.md` file contains a comprehensive ADK reference guide (not project-specific conventions).
- Simulation vectors use `np.float32` for compatibility with Unity ECS float arrays.
