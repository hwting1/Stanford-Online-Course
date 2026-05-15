# Week 2 – Action Item Extractor

A minimal **FastAPI + SQLite** web application that converts free-form notes into
enumerated action items. The app ships two extraction strategies: a fast,
rule-based heuristic extractor and an LLM-powered extractor backed by
[Ollama](https://ollama.com).

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup & Running](#setup--running)
3. [API Endpoints](#api-endpoints)
4. [Running the Test Suite](#running-the-test-suite)

---

## Project Structure

```
week2/
├── app/
│   ├── main.py              # FastAPI app & lifespan (DB init)
│   ├── db.py                # SQLite helpers (notes & action items)
│   ├── schemas.py           # Pydantic request/response models
│   ├── routers/
│   │   ├── notes.py         # /notes endpoints
│   │   └── action_items.py  # /action-items endpoints
│   └── services/
│       └── extract.py       # Rule-based & LLM extraction logic
├── frontend/
│   └── index.html           # Single-page UI (vanilla HTML + JS)
├── tests/
│   └── test_extract.py      # Pytest unit tests
└── data/
    └── app.db               # SQLite database (auto-created on first run)
```

---

## Setup & Running

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.12 |
| [Ollama](https://ollama.com/download) | latest |
| [uv](https://docs.astral.sh/uv/) / conda / pip | any |

---

### 1. Install Ollama and pull the model

```bash
# Install Ollama (Linux / macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model used by extract_action_items_llm
ollama pull gemma4:26b
```

> **Tip:** If `gemma4:26b` is too large for your machine, you can substitute a
> smaller model (e.g. `gemma3:4b`) by changing `_LLM_MODEL` in
> `app/services/extract.py`.

Make sure the Ollama server is running before starting the app:

```bash
ollama serve   # runs on http://localhost:11434 by default
```

---

### 2. Install Python dependencies

The project uses `pyproject.toml` at the repo root. From the **repo root**:

```bash
# With pip + venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .

# Or with uv
uv sync

# Or with Poetry
poetry install
```

Key runtime dependencies:

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `ollama` | Python client for Ollama |
| `pydantic` | Schema validation |
| `python-dotenv` | `.env` file loading |
| `pytest` | Test runner |

---

### 3. Environment variables (optional)

Create a `.env` file in `week2/` if you need to override any defaults:

```dotenv
# No required variables for the default SQLite setup.
# Add any future keys here, e.g.:
# OLLAMA_HOST=http://localhost:11434
```

---

### 4. Start the development server

```bash
# From the repo root
uvicorn week2.app.main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

The SQLite database (`week2/data/app.db`) is created automatically on first
startup.

---

## API Endpoints

All endpoints accept and return JSON. Interactive docs are available at
**http://127.0.0.1:8000/docs** (Swagger UI) once the server is running.

---

### Notes

#### `POST /notes`

Create and persist a new note.

**Request body**

```json
{ "content": "Meeting notes from 2026-05-16..." }
```

**Response** `201 Created`

```json
{ "id": 1, "content": "Meeting notes from 2026-05-16...", "created_at": "2026-05-16 10:00:00" }
```

---

#### `GET /notes`

Retrieve all saved notes in reverse-insertion order.

**Response** `200 OK`

```json
[
  { "id": 2, "content": "...", "created_at": "..." },
  { "id": 1, "content": "...", "created_at": "..." }
]
```

---

#### `GET /notes/{note_id}`

Retrieve a single note by its ID.

**Response** `200 OK` — note object, or `404 Not Found` if the ID does not exist.

---

### Action Items

#### `POST /action-items/extract`

Extract action items from text using the **rule-based heuristic** extractor.
Optionally saves the source text as a note.

**Request body**

```json
{
  "text": "- [ ] Set up database\nTODO: Write tests\n1. Deploy to staging",
  "save_note": true
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `string` | required | The free-form text to extract from |
| `save_note` | `boolean` | `false` | If `true`, persist `text` as a note first |

**Response** `201 Created`

```json
{
  "note_id": 3,
  "items": [
    { "id": 10, "text": "Set up database" },
    { "id": 11, "text": "Write tests" },
    { "id": 12, "text": "Deploy to staging" }
  ]
}
```

---

#### `POST /action-items/extract-llm`

Extract action items using **Ollama `gemma4:26b`** (LLM-powered, structured
output). Accepts the same request body and returns the same response shape as
`/extract`.

The LLM is instructed to:

- Identify both explicitly marked tasks (bullets, numbered lists, checkboxes,
  `TODO/ACTION/NEXT` prefixes) and implicitly stated tasks in prose.
- Strip all markup noise (markers, checkbox syntax, keyword prefixes) from the
  returned items.
- Return an empty list when no actionable tasks are present.
- Never invent tasks not present in the source text.

> **Note:** This endpoint calls a locally running Ollama instance and will be
> slower than `/extract` (typically several seconds per request depending on
> hardware).

---

#### `GET /action-items`

List all stored action items. Optionally filter by note.

**Query parameters**

| Parameter | Type | Description |
|---|---|---|
| `note_id` | `integer` (optional) | Return only items linked to this note |

**Response** `200 OK`

```json
[
  {
    "id": 10,
    "note_id": 3,
    "text": "Set up database",
    "done": false,
    "created_at": "2026-05-16 10:00:00"
  }
]
```

---

#### `POST /action-items/{action_item_id}/done`

Mark an action item as done or not done.

**Request body**

```json
{ "done": true }
```

**Response** `200 OK`

```json
{ "id": 10, "done": true }
```

Returns `404 Not Found` if the action item ID does not exist.

---

## Running the Test Suite

Tests live in `week2/tests/` and are run with **pytest** from the repo root.

```bash
# Activate your environment first
source .venv/bin/activate

# Run all tests
pytest week2/tests/ -v

# Run only the rule-based extractor tests
pytest week2/tests/test_extract.py::test_extract_bullets_and_checkboxes -v

# Run only the LLM extractor tests (requires Ollama to be running)
pytest week2/tests/test_extract.py -k "llm" -v
```

> **Warning:** The `extract_action_items_llm` tests make real calls to the local
> Ollama server. Ensure `ollama serve` is running and `gemma4:26b` is available
> before executing them. They will be slow on CPU-only machines.

### Test coverage summary

| Test | Extractor | Scenario |
|---|---|---|
| `test_extract_bullets_and_checkboxes` | Rule-based | Bullets, numbered list, checkboxes |
| `test_llm_bullet_list` | LLM | Unordered bullets mixed with narrative |
| `test_llm_numbered_list` | LLM | Numbered list |
| `test_llm_keyword_prefixed_lines` | LLM | `TODO:` / `ACTION:` / `NEXT:` |
| `test_llm_checkbox_syntax` | LLM | `[ ]` / `[x]` — verifies markers are stripped |
| `test_llm_implicit_action_items` | LLM | Implicit tasks in prose |
| `test_llm_empty_input` | LLM | Empty string → `[]` |
| `test_llm_no_action_items` | LLM | Pure narrative → `[]` |
| `test_llm_mixed_formats` | LLM | All formats combined |
| `test_llm_returns_list_of_strings` | LLM | Type contract |
| `test_llm_no_duplicate_items` | LLM | Deduplication |
