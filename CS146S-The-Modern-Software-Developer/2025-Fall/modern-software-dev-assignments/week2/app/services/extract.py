from __future__ import annotations

import os
import re
from typing import List
import json
from typing import Any
from ollama import chat
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

BULLET_PREFIX_PATTERN = re.compile(r"^\s*([-*•]|\d+\.)\s+")
KEYWORD_PREFIXES = (
    "todo:",
    "action:",
    "next:",
)


def _is_action_line(line: str) -> bool:
    stripped = line.strip().lower()
    if not stripped:
        return False
    if BULLET_PREFIX_PATTERN.match(stripped):
        return True
    if any(stripped.startswith(prefix) for prefix in KEYWORD_PREFIXES):
        return True
    if "[ ]" in stripped or "[todo]" in stripped:
        return True
    return False


def extract_action_items(text: str) -> List[str]:
    lines = text.splitlines()
    extracted: List[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if _is_action_line(line):
            cleaned = BULLET_PREFIX_PATTERN.sub("", line)
            cleaned = cleaned.strip()
            # Trim common checkbox markers
            cleaned = cleaned.removeprefix("[ ]").strip()
            cleaned = cleaned.removeprefix("[todo]").strip()
            extracted.append(cleaned)
    # Fallback: if nothing matched, heuristically split into sentences and pick imperative-like ones
    if not extracted:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue
            if _looks_imperative(s):
                extracted.append(s)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for item in extracted:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(item)
    return unique

### Exercise 1: Scaffold a New Feature
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
###


def _looks_imperative(sentence: str) -> bool:
    words = re.findall(r"[A-Za-z']+", sentence)
    if not words:
        return False
    first = words[0]
    # Crude heuristic: treat these as imperative starters
    imperative_starters = {
        "add",
        "create",
        "implement",
        "fix",
        "update",
        "write",
        "check",
        "verify",
        "refactor",
        "document",
        "design",
        "investigate",
    }
    return first.lower() in imperative_starters
