import os
import pytest

from ..app.services.extract import extract_action_items, extract_action_items_llm


def test_extract_bullets_and_checkboxes():
    text = """
    Notes from meeting:
    - [ ] Set up database
    * implement API extract endpoint
    1. Write tests
    Some narrative sentence.
    """.strip()

    items = extract_action_items(text)
    assert "Set up database" in items
    assert "implement API extract endpoint" in items
    assert "Write tests" in items

### Exercise 2: Add Unit Tests
# ---------------------------------------------------------------------------
# Tests for extract_action_items_llm
# ---------------------------------------------------------------------------

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
###