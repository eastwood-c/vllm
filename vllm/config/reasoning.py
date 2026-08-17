# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config

REASONING_EFFORT_LEVELS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
"""Values accepted by the OpenAI-compatible ``reasoning_effort`` field."""


@config
class ReasoningConfig:
    """Configuration for reasoning models.

    Set `reasoning_start_str` and `reasoning_end_str` to the strings used to
    enter and forcibly terminate reasoning. The end string may include a
    transition phrase before the parser's natural reasoning end marker. Token
    IDs are derived automatically by `initialize_token_ids`.
    """

    reasoning_parser: str = ""
    """The name of the ReasoningParser to use for this model."""
    reasoning_start_str: str = ""
    """String that indicates the start of reasoning."""
    reasoning_end_str: str = ""
    """String forced when the thinking budget is exhausted."""
    default_thinking_token_budget: int | None = None
    """Server-side default for `thinking_token_budget`, applied when a request
    does not set one and no `reasoning_effort` mapping applies. `None` leaves
    reasoning unbounded, which is the current behaviour. Use `-1` in
    per-request payloads to opt back out of a configured default."""
    reasoning_effort_budgets: dict[str, int] | None = None
    """Maps `reasoning_effort` levels to `thinking_token_budget` values, e.g.
    `{"low": 1024, "medium": 4096, "high": 16384}`.

    Without this, `reasoning_effort` only toggles thinking on or off (it is
    rendered as `enable_thinking` in the chat template), so `low`, `medium` and
    `high` are indistinguishable for models that do not ship a model-specific
    effort mapping. Supplying budgets here gives the field real effect on any
    model that has a reasoning parser.

    Keys must be drawn from `REASONING_EFFORT_LEVELS`; levels left unmapped fall
    through to `default_thinking_token_budget`. `none` is not accepted -- it
    disables thinking outright rather than bounding it."""

    _reasoning_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_start_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for forced reasoning end token IDs."""
    _natural_reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Token IDs that naturally terminate reasoning, as defined by the parser."""

    _enabled: bool = field(default=False, init=False, repr=False)
    """Private field indicating whether reasoning token IDs have been initialized.
    Set to True by `initialize_token_ids` once token IDs are initialized."""

    def __post_init__(self) -> None:
        self._validate_budgets()

    def _validate_budgets(self) -> None:
        """Reject malformed budget configuration at startup.

        These are server-side settings, so a typo would otherwise surface as
        silently unbounded reasoning on every request rather than an error.
        """
        if (
            self.default_thinking_token_budget is not None
            and self.default_thinking_token_budget < 0
        ):
            raise ValueError(
                "ReasoningConfig: default_thinking_token_budget must be a "
                f"non-negative integer, got {self.default_thinking_token_budget}."
            )

        if self.reasoning_effort_budgets is None:
            return

        unknown = sorted(
            set(self.reasoning_effort_budgets) - set(REASONING_EFFORT_LEVELS)
        )
        if unknown:
            raise ValueError(
                f"ReasoningConfig: unknown reasoning_effort level(s) {unknown} in "
                f"reasoning_effort_budgets. Valid levels: "
                f"{list(REASONING_EFFORT_LEVELS)}."
            )
        if "none" in self.reasoning_effort_budgets:
            raise ValueError(
                "ReasoningConfig: reasoning_effort 'none' disables thinking "
                "entirely and cannot be given a thinking_token_budget."
            )
        # Types are already enforced by the dataclass' pydantic validation; only
        # the value range is left to check here.
        for level, budget in self.reasoning_effort_budgets.items():
            if budget < 0:
                raise ValueError(
                    "ReasoningConfig: reasoning_effort_budgets values must be "
                    f"non-negative integers, got {budget!r} for level '{level}'."
                )

    @property
    def enabled(self) -> bool:
        """Returns True if reasoning is enabled (i.e. if token IDs have been
        initialized), False otherwise."""
        return self._enabled

    def resolve_thinking_token_budget(
        self,
        requested_budget: int | None,
        reasoning_effort: str | None = None,
    ) -> int | None:
        """Resolve the effective `thinking_token_budget` for one request.

        Precedence, most specific first:

        1. `requested_budget` -- an explicit per-request `thinking_token_budget`
           always wins over server policy.
        2. `reasoning_effort_budgets[reasoning_effort]`, when the request named
           an effort level that is mapped.
        3. `default_thinking_token_budget`.

        Returns `None` when nothing applies, leaving reasoning unbounded.

        `requested_budget=None` means "the request asked for nothing", so server
        policy applies. Callers must not use it to represent an explicit `-1`
        ("unlimited"), which `validate_thinking_token_budget` also normalises to
        `None`: that would let a configured default re-bound a request which
        asked for unbounded reasoning. Distinguish the two before calling --
        the OpenAI entrypoints test `model_fields_set`.

        Note that effort `none` is deliberately not mapped: it disables thinking
        via `enable_thinking` in the chat template, so bounding it is
        meaningless, and falling through to the server default would wrongly
        re-bound a request that asked for no reasoning at all.
        """
        if requested_budget is not None:
            return requested_budget

        if reasoning_effort == "none":
            return None

        if reasoning_effort is not None and self.reasoning_effort_budgets:
            budget = self.reasoning_effort_budgets.get(reasoning_effort)
            if budget is not None:
                return budget

        return self.default_thinking_token_budget

    @property
    def reasoning_start_token_ids(self) -> list[int] | None:
        """Token IDs derived from `reasoning_start_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._reasoning_start_token_ids

    @property
    def reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs forced when the thinking budget is exhausted."""
        return self._reasoning_end_token_ids

    @property
    def natural_reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs that indicate the model naturally ended reasoning."""
        return self._natural_reasoning_end_token_ids

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        if (
            self._reasoning_start_token_ids is not None
            and self._reasoning_end_token_ids is not None
            and self._natural_reasoning_end_token_ids is not None
        ):
            self._enabled = True
            return  # Already initialized

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        reasoning_start_str = self.reasoning_start_str
        reasoning_end_str = self.reasoning_end_str
        natural_reasoning_end_str = ""
        if self.reasoning_parser:
            parser_cls = ReasoningParserManager.get_reasoning_parser(
                self.reasoning_parser
            )
            reasoning_parser = parser_cls(tokenizer)
            start_token = reasoning_parser.reasoning_start_str
            if start_token and not reasoning_start_str:
                reasoning_start_str = start_token

            end_token = reasoning_parser.reasoning_end_str
            if end_token and not reasoning_end_str:
                reasoning_end_str = end_token
            natural_reasoning_end_str = end_token or ""

        if not natural_reasoning_end_str:
            natural_reasoning_end_str = reasoning_end_str

        if not reasoning_start_str or not reasoning_end_str:
            # If we don't have valid strings to tokenize,
            # we can't initialize the token IDs.
            return
        self._reasoning_start_token_ids = tokenizer.encode(
            reasoning_start_str, add_special_tokens=False
        )
        self._reasoning_end_token_ids = tokenizer.encode(
            reasoning_end_str, add_special_tokens=False
        )
        self._natural_reasoning_end_token_ids = tokenizer.encode(
            natural_reasoning_end_str, add_special_tokens=False
        )

        if (
            not self._reasoning_start_token_ids
            or not self._reasoning_end_token_ids
            or not self._natural_reasoning_end_token_ids
        ):
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "
                f"reasoning_start_str='{self.reasoning_start_str}', "
                f"reasoning_end_str='{self.reasoning_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
        self._enabled = True
