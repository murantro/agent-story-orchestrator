# CLAUDE.md — AI Assistant Guide for agent-story-orchestrator

## Project Overview

This is a **Simple ReAct Agent** built with the **Google Agent Development Kit (ADK)**, scaffolded from `googleCloudPlatform/agent-starter-pack` v0.33.2. The agent provides weather and time tools via a conversational interface, deployed as a containerized FastAPI service on Google Cloud Run.

## Repository Structure

```
agent-story-orchestrator/
├── .env                          # Root env (WS_NAME=agent-story-creator)
├── .idx/                         # IDX workspace config
└── agent-story-creator/          # Main project directory
    ├── .cloudbuild/              # CI/CD pipelines (Cloud Build)
    │   ├── pr_checks.yaml        # PR validation (lint, test)
    │   ├── staging.yaml          # Staging deploy + load test
    │   └── deploy-to-prod.yaml   # Production deployment
    ├── app/                      # Application source code
    │   ├── __init__.py           # Package init (Apache 2.0 header)
    │   ├── agent.py              # Core agent definition & tools
    │   ├── fast_api_app.py       # FastAPI server + telemetry
    │   └── app_utils/            # Shared utilities
    │       ├── typing.py         # Pydantic models (Request, Feedback)
    │       └── telemetry.py      # OpenTelemetry setup
    ├── deployment/               # Terraform IaC (dev + prod)
    ├── notebooks/                # Jupyter notebooks for testing/eval
    ├── tests/
    │   ├── unit/                 # Unit tests
    │   ├── integration/          # Integration & E2E tests
    │   └── load_test/            # Locust load tests
    ├── Dockerfile                # Python 3.11-slim, uv, port 8080
    ├── Makefile                  # Build/dev/deploy commands
    ├── pyproject.toml            # Project config, deps, tool settings
    └── uv.lock                   # Dependency lock file
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10–3.13 (Docker uses 3.11) |
| Agent Framework | Google ADK (`google-adk>=1.15.0,<2.0.0`) |
| LLM | Gemini (`gemini-3-flash-preview`) |
| Web Framework | FastAPI 0.115.x + Uvicorn 0.34.x |
| Package Manager | **uv** 0.8.13 |
| Testing | pytest, pytest-asyncio, Locust (load) |
| Linting | ruff, ty (type checker), codespell |
| Infrastructure | Terraform, Google Cloud Build, Cloud Run |
| Observability | OpenTelemetry, Cloud Logging, Cloud Trace |

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
uv run pytest tests/unit/              # Unit tests only
uv run pytest tests/integration/       # Integration tests only
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

- **ReAct agent pattern**: Agent reasons about which tool to call, acts, observes results
- **Tool functions** are plain Python functions with docstrings (Args/Returns sections), registered on the Agent via `tools=[fn1, fn2]`
- **Async-first**: Use `async/await` for I/O-bound operations
- **Streaming responses**: Events streamed via SSE to clients
- **Stateless sessions**: In-memory session service (no persistence between restarts)
- **Pydantic models** for all request/response validation (`app/app_utils/typing.py`)

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

### Testing Patterns

- **Unit tests**: `tests/unit/` — fast, no external dependencies
- **Integration tests**: `tests/integration/` — test agent streaming and full HTTP API
- Tests use `pytest` fixtures with session/function scope
- E2E tests spin up a real FastAPI server in a fixture and make HTTP requests
- Async tests use `pytest-asyncio` with function-scoped event loops

## Key Files to Understand

| File | Purpose |
|------|---------|
| `app/agent.py` | Agent definition, tools, system instruction |
| `app/fast_api_app.py` | FastAPI app, routes, telemetry init |
| `app/app_utils/typing.py` | Request/Feedback Pydantic models |
| `app/app_utils/telemetry.py` | OpenTelemetry + Cloud Logging setup |
| `pyproject.toml` | All project config, deps, linter settings |
| `Makefile` | All dev/build/deploy commands |
| `Dockerfile` | Container build (python:3.11-slim + uv) |

## Development Workflow

1. Install dependencies: `make install`
2. Make changes in `app/`
3. Run linters: `make lint`
4. Run tests: `make test`
5. Test locally: `make local-backend` or `make playground`
6. Commit and push — CI runs `pr_checks.yaml` (lint + test)
7. On merge to main: staging deploy → load test → production deploy

## Important Notes

- The package manager is **uv**, not pip or poetry. Always use `uv run` to execute Python commands or `uv sync` to install dependencies.
- The working directory for all make/test/lint commands is `agent-story-creator/`, not the repo root.
- The `.env` file at the repo root sets `WS_NAME=agent-story-creator`.
- Terraform state and GCP credentials are not committed; infrastructure changes require `make setup-dev-env`.
- The `GEMINI.md` file contains a comprehensive ADK reference guide (not project-specific conventions).
