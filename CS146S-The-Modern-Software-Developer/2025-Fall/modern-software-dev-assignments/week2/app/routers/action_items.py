from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from .. import db
from ..schemas import (
    ActionItemBrief,
    ActionItemResponse,
    ExtractRequest,
    ExtractResponse,
    MarkDoneRequest,
    MarkDoneResponse,
)
from ..services.extract import extract_action_items, extract_action_items_llm


router = APIRouter(prefix="/action-items", tags=["action-items"])


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


