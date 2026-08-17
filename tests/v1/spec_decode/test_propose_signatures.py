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

This is a signature-only check on purpose: it locates the package and parses the
AST, so it never imports vllm and stays runnable on a plain CPU box (including
the in-image build gate, where no GPU is present).
"""

import ast
import importlib.util
from pathlib import Path

import pytest

# Keyword arguments GPUModelRunner passes to propose() by name.
RUNNER_KEYWORDS = frozenset({"mm_inputs", "intermediate_tensors"})

_REL = Path("v1") / "worker" / "gpu" / "spec_decode"


def _find_spec_decode_root() -> Path | None:
    """Locate vllm/v1/worker/gpu/spec_decode without importing vllm.

    Two layouts must both work:
      * a source checkout, where tests/ sits beside vllm/;
      * the build gate, where tests are staged at /opt/vllm-tests and vllm is
        installed in site-packages -- so the repo-relative guess does not exist.

    find_spec("vllm") only locates the package; it does not execute
    vllm/__init__.py, so importing CUDA/Triton is still avoided.
    """
    candidate = Path(__file__).resolve().parents[3] / "vllm" / _REL
    if candidate.is_dir():
        return candidate

    try:
        spec = importlib.util.find_spec("vllm")
    except Exception:
        return None
    if spec is None or not spec.origin:
        return None
    candidate = Path(spec.origin).parent / _REL
    return candidate if candidate.is_dir() else None


SPEC_DECODE_ROOT = _find_spec_decode_root()


def _propose_defs() -> list[tuple[str, int, ast.FunctionDef]]:
    if SPEC_DECODE_ROOT is None:
        return []
    found: list[tuple[str, int, ast.FunctionDef]] = []
    for path in sorted(SPEC_DECODE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "propose":
                found.append((path.name, node.lineno, node))
    return found


def test_propose_definitions_are_discovered() -> None:
    # Guards the test itself: if the package moves or cannot be located, every
    # assertion below would vacuously pass on an empty parametrization.
    assert SPEC_DECODE_ROOT is not None, (
        "could not locate vllm/v1/worker/gpu/spec_decode in either a source "
        "checkout or the installed package"
    )
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
