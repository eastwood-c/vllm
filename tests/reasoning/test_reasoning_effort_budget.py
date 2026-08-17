# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mapping ``reasoning_effort`` onto ``thinking_token_budget``.

Without a mapping, ``reasoning_effort`` only toggles ``enable_thinking`` in the
chat template, so ``low``/``medium``/``high`` are indistinguishable for any model
that does not ship a model-specific effort mapping. ``ReasoningConfig`` can now
map effort levels to thinking budgets and supply a server-side default.
"""

import pytest

from vllm.config.reasoning import REASONING_EFFORT_LEVELS, ReasoningConfig

BUDGETS = {"minimal": 128, "low": 1024, "medium": 4096, "high": 16384}


def _config(**kwargs) -> ReasoningConfig:
    return ReasoningConfig(reasoning_parser="deepseek_r1", **kwargs)


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


def test_explicit_request_budget_beats_effort_mapping():
    """A client that names a budget outranks server effort policy."""
    cfg = _config(reasoning_effort_budgets=BUDGETS)
    assert (
        cfg.resolve_thinking_token_budget(requested_budget=99, reasoning_effort="high")
        == 99
    )


def test_explicit_request_budget_beats_server_default():
    cfg = _config(default_thinking_token_budget=2048)
    assert cfg.resolve_thinking_token_budget(requested_budget=99) == 99


@pytest.mark.parametrize("effort,expected", sorted(BUDGETS.items()))
def test_effort_maps_to_configured_budget(effort: str, expected: int):
    cfg = _config(reasoning_effort_budgets=BUDGETS)
    assert (
        cfg.resolve_thinking_token_budget(
            requested_budget=None, reasoning_effort=effort
        )
        == expected
    )


def test_levels_are_distinguishable():
    """The whole point: distinct levels must yield distinct budgets."""
    cfg = _config(reasoning_effort_budgets=BUDGETS)
    resolved = [
        cfg.resolve_thinking_token_budget(requested_budget=None, reasoning_effort=e)
        for e in ("minimal", "low", "medium", "high")
    ]
    assert len(set(resolved)) == len(resolved)


def test_unmapped_effort_falls_through_to_default():
    cfg = _config(
        reasoning_effort_budgets={"low": 1024},
        default_thinking_token_budget=2048,
    )
    assert (
        cfg.resolve_thinking_token_budget(
            requested_budget=None, reasoning_effort="high"
        )
        == 2048
    )


def test_server_default_applies_without_effort():
    cfg = _config(default_thinking_token_budget=2048)
    assert cfg.resolve_thinking_token_budget(requested_budget=None) == 2048


def test_unconfigured_resolves_to_none():
    """Default behaviour is unchanged: no budget unless one is configured."""
    cfg = _config()
    assert cfg.resolve_thinking_token_budget(requested_budget=None) is None
    assert (
        cfg.resolve_thinking_token_budget(
            requested_budget=None, reasoning_effort="high"
        )
        is None
    )


def test_effort_none_is_never_bounded():
    """``none`` disables thinking outright; a server default must not re-bound it."""
    cfg = _config(reasoning_effort_budgets=BUDGETS, default_thinking_token_budget=2048)
    assert (
        cfg.resolve_thinking_token_budget(
            requested_budget=None, reasoning_effort="none"
        )
        is None
    )


def test_resolver_treats_none_as_unset():
    """The resolver sees only values, so ``None`` means "nothing requested".

    Distinguishing an omitted field from an explicit ``-1`` (which
    ``validate_thinking_token_budget`` normalises to ``None``) is the caller's
    job -- see ``test_explicit_minus_one_stays_unbounded`` for the request-level
    behaviour that depends on it.
    """
    cfg = _config(default_thinking_token_budget=2048)
    assert cfg.resolve_thinking_token_budget(requested_budget=None) == 2048


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_unknown_effort_level():
    with pytest.raises(ValueError, match="unknown reasoning_effort level"):
        _config(reasoning_effort_budgets={"enormous": 1024})


def test_rejects_budget_for_effort_none():
    with pytest.raises(ValueError, match="cannot be given a thinking_token_budget"):
        _config(reasoning_effort_budgets={"none": 1024})


def test_rejects_negative_effort_budget():
    with pytest.raises(ValueError, match="non-negative integers"):
        _config(reasoning_effort_budgets={"low": -1})


def test_rejects_non_integer_effort_budget():
    """Type enforcement belongs to the dataclass' pydantic validation, which
    runs before ``__post_init__``; assert it is actually in force rather than
    duplicating an isinstance check that could never fire."""
    with pytest.raises(ValueError, match="valid integer"):
        _config(reasoning_effort_budgets={"low": 1.5})


def test_rejects_negative_default_budget():
    with pytest.raises(ValueError, match="default_thinking_token_budget"):
        _config(default_thinking_token_budget=-5)


def test_every_documented_level_is_accepted():
    """Guards REASONING_EFFORT_LEVELS against drift from the API definition."""
    accepted = {level: 1 for level in REASONING_EFFORT_LEVELS if level != "none"}
    cfg = _config(reasoning_effort_budgets=accepted)
    assert cfg.reasoning_effort_budgets == accepted


def test_zero_budget_is_allowed():
    """0 is a meaningful budget: think, but emit no reasoning tokens."""
    cfg = _config(reasoning_effort_budgets={"minimal": 0})
    assert (
        cfg.resolve_thinking_token_budget(
            requested_budget=None, reasoning_effort="minimal"
        )
        == 0
    )


# ---------------------------------------------------------------------------
# Request-level wiring
# ---------------------------------------------------------------------------


def _budget_for(reasoning_config=None, **request_kwargs) -> int | None:
    """Resolve the budget a ChatCompletionRequest would carry into the engine."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        **request_kwargs,
    )
    params = request.to_sampling_params(128, {}, reasoning_config)
    return params.thinking_token_budget


def test_request_effort_maps_to_budget():
    cfg = _config(reasoning_effort_budgets=BUDGETS)
    assert _budget_for(cfg, reasoning_effort="low") == 1024
    assert _budget_for(cfg, reasoning_effort="high") == 16384


def test_request_without_budget_gets_server_default():
    cfg = _config(default_thinking_token_budget=2048)
    assert _budget_for(cfg) == 2048


def test_request_explicit_budget_wins():
    cfg = _config(reasoning_effort_budgets=BUDGETS, default_thinking_token_budget=2048)
    assert _budget_for(cfg, reasoning_effort="high", thinking_token_budget=77) == 77


def test_explicit_minus_one_stays_unbounded():
    """``-1`` means unlimited and must survive a configured default.

    It normalises to ``None`` before reaching the resolver, so resolution keys
    off ``model_fields_set`` instead of the value. Without that, a server
    default would silently re-bound a request that explicitly asked for
    unbounded reasoning.
    """
    cfg = _config(reasoning_effort_budgets=BUDGETS, default_thinking_token_budget=2048)
    assert _budget_for(cfg, reasoning_effort="high", thinking_token_budget=-1) is None


def test_request_unchanged_without_reasoning_config():
    """Behaviour is untouched for servers that configure nothing."""
    assert _budget_for(None, reasoning_effort="high") is None
    assert _budget_for(None) is None
    assert _budget_for(None, thinking_token_budget=64) == 64
