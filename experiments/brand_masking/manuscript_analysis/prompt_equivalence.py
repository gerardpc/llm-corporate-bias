"""Prompt-equivalence audit: Named and Masked prompts may differ only in identity.

Both user prompts are rendered from ``config/prompts.yaml``. After replacing the
identity value (real name in Named, masked label in Masked) with one placeholder, the
two prompts must be token-identical; the system prompt is shared by construction.
"""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "config" / "prompts.yaml"
PLACEHOLDER = "<ENTITY>"


def load_templates(path: Path = PROMPTS_PATH) -> dict[str, str]:
    """Load system / named / masked templates."""
    return yaml.safe_load(path.read_text())


def render_pair(
    templates: dict[str, str],
    *,
    scenario: str,
    incumbent_name: str,
    masked_name: str,
    description: str,
) -> tuple[str, str]:
    """Render the Named and Masked user prompts exactly as the experiment does."""
    named = templates["named_user_prompt"].format(
        base_scenario_text=scenario,
        incumbent_name=incumbent_name,
        company_description=description,
    )
    masked = templates["masked_user_prompt"].format(
        base_scenario_text=scenario,
        masked_name=masked_name,
        company_description=description,
    )
    return named, masked


def _neutralise(prompt: str, identity: str) -> str:
    """Replace the identity on the ``Entity:`` line only (not inside the profile)."""
    lines = prompt.split("\n")
    out = []
    for line in lines:
        if line == f"Entity: {identity}":
            out.append(f"Entity: {PLACEHOLDER}")
        else:
            out.append(line)
    return "\n".join(out)


def _tokens(text: str) -> list[str]:
    return text.split()


@dataclass
class EquivalenceResult:
    """Machine-readable outcome of one Named/Masked comparison."""

    label: str
    equivalent: bool
    raw_differing_lines: list[dict] = field(default_factory=list)
    residual_line_diff: list[str] = field(default_factory=list)
    residual_token_diff: list[str] = field(default_factory=list)
    system_prompt_identical: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def compare_prompts(
    named: str,
    masked: str,
    *,
    incumbent_name: str,
    masked_name: str,
    label: str = "",
    system_named: str | None = None,
    system_masked: str | None = None,
) -> EquivalenceResult:
    """Fail on any difference other than the identity value on the Entity line."""
    raw = [
        {"line": i, "named": a, "masked": b}
        for i, (a, b) in enumerate(
            zip(named.split("\n"), masked.split("\n"), strict=False),
        )
        if a != b
    ]
    if len(named.split("\n")) != len(masked.split("\n")):
        raw.append({"line": -1, "named": "<line count>", "masked": "<line count>"})
    n_norm = _neutralise(named, incumbent_name)
    m_norm = _neutralise(masked, masked_name)
    line_diff = [
        d
        for d in difflib.unified_diff(
            n_norm.split("\n"), m_norm.split("\n"), lineterm="", n=0,
        )
        if not d.startswith(("---", "+++", "@@"))
    ]
    token_diff = [
        d
        for d in difflib.ndiff(_tokens(n_norm), _tokens(m_norm))
        if d.startswith(("- ", "+ "))
    ]
    sys_same = system_named == system_masked
    return EquivalenceResult(
        label=label,
        equivalent=not line_diff and not token_diff and sys_same,
        raw_differing_lines=raw,
        residual_line_diff=line_diff,
        residual_token_diff=token_diff,
        system_prompt_identical=sys_same,
    )


def template_check(templates: dict[str, str]) -> EquivalenceResult:
    """Compare the two templates themselves with the identity slots neutralised."""
    named = templates["named_user_prompt"].replace("{incumbent_name}", PLACEHOLDER)
    masked = templates["masked_user_prompt"].replace("{masked_name}", PLACEHOLDER)
    return compare_prompts(
        named.replace(PLACEHOLDER, "X"),
        masked.replace(PLACEHOLDER, "X"),
        incumbent_name="X",
        masked_name="X",
        label="templates",
        system_named=templates["system_prompt"],
        system_masked=templates["system_prompt"],
    )
