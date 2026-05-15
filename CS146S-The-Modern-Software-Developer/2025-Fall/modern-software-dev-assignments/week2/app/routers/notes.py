from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import db
from ..schemas import NoteCreate, NoteResponse


router = APIRouter(prefix="/notes", tags=["notes"])


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


