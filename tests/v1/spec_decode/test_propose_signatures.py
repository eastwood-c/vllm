# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every concrete speculator's ``propose()`` must accept the keyword arguments
that ``GPUModelRunner.sample_tokens`` passes unconditionally.

The runner has a single call site for every speculative method::

    self.speculator.propose(
        ...,                       # positional
        mm_inputs=mm_inputs,
        intermediate_tensors=self.intermediate_tensors,
    )

A subclass that overrides ``propose()`` without those keyword parameters raises
``TypeError`` at the first draft step. That failure is invisible in CI whenever
the only speculative method under test happens to define them, so assert the
contract statically across the whole package instead of per method.

This is a signature-only check on purpose: it resolves the package by path and
parses the AST, so it never imports vllm and stays runnable on a plain CPU box
(including the in-image build gate, where no GPU is present).
"""

import ast
from pathlib import Path

import pytest

# Keyword arguments GPUModelRunner passes to propose() by name.
RUNNER_KEYWORDS = frozenset({"mm_inputs", "intermediate_tensors"})

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DECODE_ROOT = REPO_ROOT / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"


def _propose_defs() -> list[tuple[str, int, ast.FunctionDef]]:
    found: list[tuple[str, int, ast.FunctionDef]] = []
    for path in sorted(SPEC_DECODE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "propose":
                found.append((str(path.relative_to(REPO_ROOT)), node.lineno, node))
    return found


def test_propose_definitions_are_discovered() -> None:
    # Guards the test itself: a refactor that moves or renames the speculator
    # package would otherwise make every assertion below vacuously pass.
    assert SPEC_DECODE_ROOT.is_dir(), f"{SPEC_DECODE_ROOT} not found"
    assert len(_propose_defs()) >= 2


@pytest.mark.parametrize(
    "rel_path,lineno,node",
    [pytest.param(p, n, f, id=f"{p}:{n}") for p, n, f in _propose_defs()],
)
def test_propose_accepts_runner_keywords(
    rel_path: str, lineno: int, node: ast.FunctionDef
) -> None:
    if node.args.kwarg is not None:
        # An explicit **kwargs absorbs anything the runner passes.
        return

    params = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
    missing = sorted(RUNNER_KEYWORDS - params)
    assert not missing, (
        f"{rel_path}:{lineno} propose() is missing {missing}. "
        "GPUModelRunner passes these by keyword for every speculative method, "
        "so this raises TypeError at the first draft step."
    )
