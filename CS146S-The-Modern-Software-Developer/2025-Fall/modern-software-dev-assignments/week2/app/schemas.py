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
