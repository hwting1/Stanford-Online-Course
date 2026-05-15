# Week 2 Write-up
Tip: To preview this markdown file
- On Mac, press `Command (⌘) + Shift + V`
- On Windows/Linux, press `Ctrl + Shift + V`

## INSTRUCTIONS

Fill out all of the `TODO`s in this file.

## SUBMISSION DETAILS

Name: **TODO** \
SUNet ID: **TODO** \
Citations: **TODO**

This assignment took me about **TODO** hours to do. 


## YOUR RESPONSES
For each exercise, please include what prompts you used to generate the answer, in addition to the location of the generated response. Make sure to clearly add comments in your code documenting which parts are generated.

### Exercise 1: Scaffold a New Feature
Prompt: 
```
Please analyse #sym:extract_action_items and implement an alternative version, extract_action_items_llm, which can use ollama gemma4:26b to automatically extract action items. Refer to https://ollama.com/blog/structured-outputs for guidance on structured outputs. In addition, to construct a concise and precise system prompt that can deal with most cases
``` 

Generated Code Snippets:
```python
_LLM_MODEL = "gemma4:26b"

_LLM_SYSTEM_PROMPT = (
    "You are an action item extractor. Given a text, identify and return all action items, "
    "tasks, or to-dos — whether explicitly marked (e.g. bullet points, numbered lists, "
    "checkboxes, TODO/ACTION/NEXT labels) or implicitly stated as things that need to be done.\n\n"
    "Rules:\n"
    "- Include only genuinely actionable tasks, not background context or general statements.\n"
    "- Strip bullet markers, numbering, checkbox syntax ([ ], [x]), and keyword prefixes "
    "(todo:, action:, next:) from each item.\n"
    "- Preserve the original wording of each task as closely as possible.\n"
    "- Return an empty list when no action items are present.\n"
    "- Do not invent or infer tasks that are not stated in the text."
)


class _ActionItemList(BaseModel):
    items: list[str]


def extract_action_items_llm(text: str) -> List[str]:
    response = chat(
        model=_LLM_MODEL,
        messages=[
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        format=_ActionItemList.model_json_schema(),
        options={"temperature": 0},
    )
    result = _ActionItemList.model_validate_json(response.message.content)
    return result.items
```

### Exercise 2: Add Unit Tests
Prompt: 
```
Please write unit tests for #sym:extract_action_items_llm  covering multiple inputs (e.g., bullet lists, keyword-prefixed lines, empty input) in @file:test_extract.py
``` 

Generated Code Snippets:
```python
def _contains_substring(items: list, substring: str) -> bool:
    """Return True if any item contains `substring` (case-insensitive)."""
    sub = substring.lower()
    return any(sub in item.lower() for item in items)


def test_llm_bullet_list():
    text = """
    Sprint planning notes:
    - Set up CI pipeline
    - Write integration tests
    - Deploy to staging
    This was a productive session.
    """.strip()

    items = extract_action_items_llm(text)
    assert isinstance(items, list)
    assert len(items) >= 3
    assert _contains_substring(items, "CI pipeline")
    assert _contains_substring(items, "integration tests")
    assert _contains_substring(items, "staging")


def test_llm_numbered_list():
    text = """
    After the review we decided:
    1. Refactor the authentication module
    2. Update the API documentation
    3. Fix the login redirect bug
    """.strip()

    items = extract_action_items_llm(text)
    assert isinstance(items, list)
    assert len(items) >= 3
    assert _contains_substring(items, "authentication")
    assert _contains_substring(items, "documentation")
    assert _contains_substring(items, "login")


def test_llm_keyword_prefixed_lines():
    text = """
    TODO: Add input validation to the registration form
    ACTION: Schedule follow-up meeting with the design team
    NEXT: Investigate slow query performance
    """.strip()

    items = extract_action_items_llm(text)
    assert isinstance(items, list)
    assert len(items) >= 3
    assert _contains_substring(items, "input validation")
    assert _contains_substring(items, "follow-up meeting")
    assert _contains_substring(items, "slow query")


def test_llm_checkbox_syntax():
    text = """
    Backlog:
    - [ ] Create onboarding flow
    - [x] Remove deprecated endpoints
    - [ ] Add rate limiting
    """.strip()

    items = extract_action_items_llm(text)
    assert isinstance(items, list)
    # Completed item ([x]) may or may not be included; uncompleted ones must be
    assert _contains_substring(items, "onboarding")
    assert _contains_substring(items, "rate limiting")
    # Cleaned items should not contain raw checkbox syntax
    for item in items:
        assert "[ ]" not in item
        assert "[x]" not in item


def test_llm_implicit_action_items():
    text = (
        "We need to migrate the database to PostgreSQL. "
        "The team should also review the security audit findings. "
        "Someone must update the deployment scripts before next Friday."
    )

    items = extract_action_items_llm(text)
    assert isinstance(items, list)
    assert len(items) >= 2
    assert _contains_substring(items, "PostgreSQL") or _contains_substring(items, "database")
    assert _contains_substring(items, "deployment scripts") or _contains_substring(items, "scripts")


def test_llm_empty_input():
    items = extract_action_items_llm("")
    assert isinstance(items, list)
    assert items == []


def test_llm_no_action_items():
    text = (
        "Today was a great day. The weather was sunny and the team had lunch together. "
        "Everyone enjoyed the presentation about company history."
    )

    items = extract_action_items_llm(text)
    assert isinstance(items, list)
    assert len(items) == 0


def test_llm_mixed_formats():
    text = """
    Meeting recap:
    - [ ] Update the README with setup instructions
    TODO: Review pull request #42
    2. Add logging to the payment service
    The architecture diagram was well received.
    NEXT: Coordinate with the QA team on test coverage
    """.strip()

    items = extract_action_items_llm(text)
    assert isinstance(items, list)
    assert len(items) >= 4
    assert _contains_substring(items, "README")
    assert _contains_substring(items, "pull request")
    assert _contains_substring(items, "logging")
    assert _contains_substring(items, "QA")


def test_llm_returns_list_of_strings():
    text = "- Fix the broken navigation bar\n- Add dark mode support"
    items = extract_action_items_llm(text)
    assert isinstance(items, list)
    assert all(isinstance(item, str) for item in items)


def test_llm_no_duplicate_items():
    text = """
    - Write unit tests
    - Write unit tests
    TODO: Write unit tests
    """.strip()

    items = extract_action_items_llm(text)
    assert isinstance(items, list)
    lowered = [i.lower() for i in items]
    assert len(lowered) == len(set(lowered))
```

### Exercise 3: Refactor Existing Code for Clarity
Prompt: 
```
Please refactor the backend codebase with focus on:
1. clear and consistent API contracts/schemas
2. database layer cleanup
3. app lifecycle and configuration
4. robust error handling
``` 

Generated/Modified Code Snippets:

[schemas.py](./app/schemas.py)
```python
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

class NoteCreate(BaseModel):
    content: str = Field(..., min_length=1, description="Raw note text")


class NoteResponse(BaseModel):
    id: int
    content: str
    created_at: str


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, description="Text to extract action items from"
    )
    save_note: bool = Field(
        default=False, description="Persist the text as a note"
    )


class ActionItemBrief(BaseModel):
    """Minimal representation returned by the extract endpoint."""
    id: int
    text: str


class ExtractResponse(BaseModel):
    note_id: Optional[int] = None
    items: list[ActionItemBrief]


class ActionItemResponse(BaseModel):
    id: int
    note_id: Optional[int] = None
    text: str
    done: bool
    created_at: str


class MarkDoneRequest(BaseModel):
    done: bool = True


class MarkDoneResponse(BaseModel):
    id: int
    done: bool
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

class NoteCreate(BaseModel):
    content: str = Field(..., min_length=1, description="Raw note text")


class NoteResponse(BaseModel):
    id: int
    content: str
    created_at: str


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, description="Text to extract action items from"
    )
    save_note: bool = Field(
        default=False, description="Persist the text as a note"
    )


class ActionItemBrief(BaseModel):
    """Minimal representation returned by the extract endpoint."""
    id: int
    text: str


class ExtractResponse(BaseModel):
    note_id: Optional[int] = None
    items: list[ActionItemBrief]


class ActionItemResponse(BaseModel):
    id: int
    note_id: Optional[int] = None
    text: str
    done: bool
    created_at: str


class MarkDoneRequest(BaseModel):
    done: bool = True


class MarkDoneResponse(BaseModel):
    id: int
    done: bool
```

[db.py](./app/db.py)
```python
def get_action_item(action_item_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, note_id, text, done, created_at FROM action_items WHERE id = ?",
            (action_item_id,),
        )
        return cursor.fetchone()
```

[main.py](./app/main.py)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Action Item Extractor", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    return html_path.read_text(encoding="utf-8")


app.include_router(notes.router)
app.include_router(action_items.router)

static_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
```

[notes.py](./app/routers/notes.py)
```python
@router.post("", response_model=NoteResponse, status_code=201)
def create_note(body: NoteCreate) -> NoteResponse:
    note_id = db.insert_note(body.content.strip())
    row = db.get_note(note_id)
    return NoteResponse(
        id=row["id"], content=row["content"], created_at=row["created_at"]
    )


@router.get("", response_model=list[NoteResponse])
def list_notes() -> list[NoteResponse]:
    rows = db.list_notes()
    return [
        NoteResponse(id=r["id"], content=r["content"], created_at=r["created_at"])
        for r in rows
    ]


@router.get("/{note_id}", response_model=NoteResponse)
def get_single_note(note_id: int) -> NoteResponse:
    row = db.get_note(note_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return NoteResponse(
        id=row["id"], content=row["content"], created_at=row["created_at"]
    )
```

[action_items.py](./app/routers/action_items.py)
```python
@router.post("/extract", response_model=ExtractResponse, status_code=201)
def extract(body: ExtractRequest) -> ExtractResponse:
    text = body.text.strip()
    note_id: Optional[int] = None
    if body.save_note:
        note_id = db.insert_note(text)

    try:
        items = extract_action_items(text)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Extraction failed: {exc}"
        ) from exc

    ids = db.insert_action_items(items, note_id=note_id)
    return ExtractResponse(
        note_id=note_id,
        items=[ActionItemBrief(id=i, text=t) for i, t in zip(ids, items)],
    )


@router.get("", response_model=list[ActionItemResponse])
def list_all(note_id: Optional[int] = None) -> list[ActionItemResponse]:
    rows = db.list_action_items(note_id=note_id)
    return [
        ActionItemResponse(
            id=r["id"],
            note_id=r["note_id"],
            text=r["text"],
            done=bool(r["done"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


@router.post("/{action_item_id}/done", response_model=MarkDoneResponse)
def mark_done(action_item_id: int, body: MarkDoneRequest) -> MarkDoneResponse:
    if db.get_action_item(action_item_id) is None:
        raise HTTPException(status_code=404, detail="Action item not found")
    db.mark_action_item_done(action_item_id, body.done)
    return MarkDoneResponse(id=action_item_id, done=body.done)
```

### Exercise 4: Use Agentic Mode to Automate a Small Task
Prompt: 
```
Finish the following tasks:
1. Integrate the LLM-powered extraction as a new endpoint. Update the frontend to include an "Extract LLM" button that, when clicked, triggers the extraction process via the new endpoint.
2. Expose one final endpoint to retrieve all notes. Update the frontend to include a "List Notes" button that, when clicked, fetches and displays them.
``` 

Generated Code Snippets:
[action_items.py](./app/routers/action_items.py)
```python
def _run_extract(
    text: str, save_note: bool, extractor
) -> ExtractResponse:
    note_id: Optional[int] = None
    if save_note:
        note_id = db.insert_note(text)
    try:
        items = extractor(text)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Extraction failed: {exc}"
        ) from exc
    ids = db.insert_action_items(items, note_id=note_id)
    return ExtractResponse(
        note_id=note_id,
        items=[ActionItemBrief(id=i, text=t) for i, t in zip(ids, items)],
    )


@router.post("/extract", response_model=ExtractResponse, status_code=201)
def extract(body: ExtractRequest) -> ExtractResponse:
    return _run_extract(body.text.strip(), body.save_note, extract_action_items)


@router.post("/extract-llm", response_model=ExtractResponse, status_code=201)
def extract_llm(body: ExtractRequest) -> ExtractResponse:
    return _run_extract(body.text.strip(), body.save_note, extract_action_items_llm)
```

[index.html](./frontend/index.html)
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Action Item Extractor</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif; margin: 2rem auto; max-width: 800px; padding: 0 1rem; }
      h1 { font-size: 1.5rem; }
      h2 { font-size: 1.1rem; margin-top: 2rem; border-top: 1px solid #e5e7eb; padding-top: 1rem; }
      textarea { width: 100%; min-height: 160px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace; }
      button { padding: 0.5rem 1rem; }
      .items { margin-top: 1rem; }
      .item { display: flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0; }
      .note-card { border: 1px solid #e5e7eb; border-radius: 4px; padding: 0.5rem 0.75rem; margin: 0.25rem 0; }
      .note-meta { color: #6b7280; font-size: 0.75rem; margin-top: 0.25rem; }
      .muted { color: #6b7280; font-size: 0.875rem; }
      .row { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
    </style>
  </head>
  <body>
    <h1>Action Item Extractor</h1>
    <p class="muted">Paste notes and extract actionable items. Minimal raw HTML frontend.</p>

    <label for="text">Notes</label>
    <textarea id="text" placeholder="Paste notes here...&#10;e.g.&#10;- [ ] Set up database&#10;- Implement extract endpoint"></textarea>
    <div class="row">
      <label class="row"><input id="save_note" type="checkbox" checked /> Save as note</label>
      <button id="extract">Extract</button>
      <button id="extract-llm">Extract LLM</button>
    </div>

    <div class="items" id="items"></div>

    <h2>Notes</h2>
    <div class="row">
      <button id="list-notes">List Notes</button>
    </div>
    <div id="notes-list" style="margin-top:0.75rem;"></div>

    <script>
      const $ = (sel) => document.querySelector(sel);
      const itemsEl = $('#items');

      async function runExtract(endpoint) {
        const text = $('#text').value;
        const save = $('#save_note').checked;
        itemsEl.textContent = 'Extracting…';
        try {
          const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, save_note: save }),
          });
          if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Request failed');
          }
          const data = await res.json();
          if (!data.items || data.items.length === 0) {
            itemsEl.innerHTML = '<p class="muted">No action items found.</p>';
            return;
          }
          itemsEl.innerHTML = data.items.map(it => (
            `<div class="item"><input type="checkbox" data-id="${it.id}" /> <span>${it.text}</span></div>`
          )).join('');
          itemsEl.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', async (e) => {
              const id = e.target.getAttribute('data-id');
              await fetch(`/action-items/${id}/done`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ done: e.target.checked }),
              });
            });
          });
        } catch (err) {
          console.error(err);
          itemsEl.textContent = `Error: ${err.message}`;
        }
      }

      $('#extract').addEventListener('click', () => runExtract('/action-items/extract'));
      $('#extract-llm').addEventListener('click', () => runExtract('/action-items/extract-llm'));

      $('#list-notes').addEventListener('click', async () => {
        const notesEl = $('#notes-list');
        notesEl.textContent = 'Loading…';
        try {
          const res = await fetch('/notes');
          if (!res.ok) throw new Error('Request failed');
          const notes = await res.json();
          if (!notes || notes.length === 0) {
            notesEl.innerHTML = '<p class="muted">No notes saved yet.</p>';
            return;
          }
          notesEl.innerHTML = notes.map(n => `
            <div class="note-card">
              <div>${n.content}</div>
              <div class="note-meta">id: ${n.id} &nbsp;·&nbsp; ${n.created_at}</div>
            </div>`).join('');
        } catch (err) {
          console.error(err);
          notesEl.textContent = `Error: ${err.message}`;
        }
      });
    </script>
  </body>
</html>
```


### Exercise 5: Generate a README from the Codebase
Prompt: 
```
Please generate a well-structuredd README.md for this project, which should include at least following sections:
1. An introduction of the project
2. How to setup and run this project (include Ollama and Python setting)
3. A comprehesive description for API endpoints and functionality
4. Instructions for running the test suite
``` 

Generated Code Snippets:
[README.md](README.md)
```markdown
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

```


## SUBMISSION INSTRUCTIONS
1. Hit a `Command (⌘) + F` (or `Ctrl + F`) to find any remaining `TODO`s in this file. If no results are found, congratulations – you've completed all required fields. 
2. Make sure you have all changes pushed to your remote repository for grading.
3. Submit via Gradescope. 