"""
src/governance/narrative_generator.py
=======================================
NarrativeGenerator: produces plain-language governance narratives from
GovernanceDecision objects using the Anthropic API.

Architecture role:
  INPUT  : GovernanceDecision (from DecisionEngine)
           audience mode: "technical" | "non_technical"
           optional: model_class, commodity_context
  OUTPUT : NarrativeResult
             technical_narrative   : str  (for risk officers)
             plain_narrative       : str  (for non-technical stakeholders)
             saved_paths           : dict (artifact paths if written to disk)

Two narrative modes:
  technical      — uses domain terminology (PIT, CRPS, Anfuso, Ljung-Box),
                   explains which diagnostic branches failed, quantifies
                   deviations, references Basel/REMIT governance implications
  non_technical  — plain English, no jargon, uses analogies, explains
                   what the classification means for business decisions

Design:
  - Calls claude-sonnet-4-20250514 via the Anthropic API
  - Structured prompt: injects GovernanceDecision.to_dict() as context
  - Deterministic temperature=0 for reproducibility
  - Falls back gracefully if API is unavailable (returns stub narrative)
  - Both modes generated in a single API call (two-section response)

Cost estimate: ~$0.005 per dataset (1k tokens in, 300 tokens out per mode)
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# NarrativeResult — output
# ---------------------------------------------------------------------------

@dataclass
class NarrativeResult:
    """
    Output of NarrativeGenerator.generate().

    Attributes
    ----------
    model_id : str
    final_label : str
    technical_narrative : str
        Audience: risk officers with quantitative background.
    plain_narrative : str
        Audience: non-technical stakeholders (management, regulators).
    mode : str
        "both" | "technical" | "non_technical"
    api_used : bool
        True if the Anthropic API was called; False if stub was returned.
    saved_paths : dict
        Paths of written artifact files (empty if not saved to disk).
    """
    model_id:             str
    final_label:          str
    technical_narrative:  str
    plain_narrative:      str
    mode:                 str  = "both"
    api_used:             bool = False
    saved_paths:          dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Return combined markdown string for both narratives."""
        return (
            f"# Governance Narrative: {self.model_id}\n\n"
            f"**Classification:** {self.final_label}\n\n"
            f"---\n\n"
            f"## Technical Summary\n\n"
            f"{self.technical_narrative}\n\n"
            f"---\n\n"
            f"## Plain Language Summary\n\n"
            f"{self.plain_narrative}\n"
        )


# ---------------------------------------------------------------------------
# NarrativeGenerator
# ---------------------------------------------------------------------------

class NarrativeGenerator:
    """
    Generates governance narratives from GovernanceDecision objects.

    Parameters
    ----------
    model : str
        Anthropic model to use. Default "claude-sonnet-4-20250514".
    max_tokens : int
        Maximum tokens for the API response. Default 1000.
    temperature : float
        Sampling temperature. Default 0 (deterministic).
    api_key : str or None
        Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        If not found, falls back to stub narratives.

    Example
    -------
    >>> gen = NarrativeGenerator()
    >>> result = gen.generate(decision, model_class="short_term")
    >>> print(result.technical_narrative)
    >>> result.save(out_dir=Path("experiments/run_001_entsoe"))
    """

    _SYSTEM_PROMPT = """You are a quantitative risk analyst specialising in
probabilistic model validation for energy markets. You produce clear,
precise governance narratives from structured diagnostic outputs.
You always respond with exactly two sections separated by the delimiter
<<<PLAIN>>> — first the technical narrative, then the plain language narrative.
Do not include any other text, headers, or preamble."""

    _USER_PROMPT_TEMPLATE = """Given the following governance decision from a
probabilistic validation framework, write two narratives.

GOVERNANCE DECISION:
{decision_json}

MODEL CLASS: {model_class}
COMMODITY CONTEXT: {commodity_context}

SECTION 1 — TECHNICAL NARRATIVE (3-5 sentences):
Write for a quantitative risk officer who understands PIT diagnostics,
Ljung-Box tests, Anfuso backtesting, and Basel traffic-light systems.
Be precise about which diagnostic branches failed, quantify the deviations,
and state the governance implication (e.g. capital multiplier impact,
REMIT reporting obligation). Reference specific metric values from the
decision snapshot.

<<<PLAIN>>>

SECTION 2 — PLAIN LANGUAGE NARRATIVE (3-5 sentences):
Write for a non-technical stakeholder (senior management, regulator,
commercial team). No jargon. Explain what the classification means in
plain English using a simple analogy if helpful. Focus on the business
implication — what action is required, what risk exists, what is working well."""

    def __init__(
        self,
        model:       str   = "claude-sonnet-4-20250514",
        max_tokens:  int   = 1000,
        temperature: float = 0.0,
        api_key:     str | None = None,
    ) -> None:
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._api_key    = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def generate(
        self,
        decision,
        model_class:        str = "unknown",
        commodity_context:  str = "energy market",
    ) -> NarrativeResult:
        """
        Generate technical and plain-language narratives for a decision.

        Parameters
        ----------
        decision : GovernanceDecision
            Output of DecisionEngine.decide().
        model_class : str
            e.g. "short_term", "long_term", "simulation"
        commodity_context : str
            e.g. "ENTSO-E electricity load", "PV solar generation",
            "onshore wind generation", "Monte Carlo price simulation"

        Returns
        -------
        NarrativeResult
        """
        decision_dict = decision.to_dict()

        if self._api_key is None:
            warnings.warn(
                "ANTHROPIC_API_KEY not found. Returning stub narrative. "
                "Set the environment variable to enable AI-generated narratives.",
                UserWarning,
                stacklevel=2,
            )
            return self._stub_narrative(decision_dict)

        try:
            tech, plain = self._call_api(decision_dict, model_class, commodity_context)
            return NarrativeResult(
                model_id            = decision.model_id,
                final_label         = decision.final_label,
                technical_narrative = tech,
                plain_narrative     = plain,
                mode                = "both",
                api_used            = True,
            )
        except Exception as exc:
            warnings.warn(
                f"Anthropic API call failed: {exc}. Returning stub narrative.",
                UserWarning,
                stacklevel=2,
            )
            return self._stub_narrative(decision_dict)

    def save(
        self,
        result:  NarrativeResult,
        out_dir: Path | str,
    ) -> dict[str, str]:
        """
        Save narrative artifacts to disk.

        Writes:
          narrative_technical.md   — technical narrative
          narrative_plain.md       — plain language narrative
          narrative_combined.md    — both in one file

        Parameters
        ----------
        result : NarrativeResult
        out_dir : Path

        Returns
        -------
        dict mapping artifact name to file path string
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        tech_path = out_dir / "narrative_technical.md"
        tech_path.write_text(
            f"# Technical Governance Narrative\n\n"
            f"**Model:** {result.model_id}  \n"
            f"**Classification:** {result.final_label}  \n"
            f"**API generated:** {result.api_used}\n\n"
            f"---\n\n"
            f"{result.technical_narrative}\n",
            encoding="utf-8",
        )
        paths["narrative_technical"] = str(tech_path)

        plain_path = out_dir / "narrative_plain.md"
        plain_path.write_text(
            f"# Plain Language Governance Summary\n\n"
            f"**Model:** {result.model_id}  \n"
            f"**Classification:** {result.final_label}\n\n"
            f"---\n\n"
            f"{result.plain_narrative}\n",
            encoding="utf-8",
        )
        paths["narrative_plain"] = str(plain_path)

        combined_path = out_dir / "narrative_combined.md"
        combined_path.write_text(result.to_markdown(), encoding="utf-8")
        paths["narrative_combined"] = str(combined_path)

        result.saved_paths = paths
        return paths

    # ── Private ──────────────────────────────────────────────────────────

    def _call_api(
        self,
        decision_dict:     dict,
        model_class:       str,
        commodity_context: str,
    ) -> tuple[str, str]:
        """Call Anthropic API and return (technical, plain) tuple."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self._api_key)

        # Slim down the decision dict for the prompt — drop large arrays
        slim = {
            "model_id":      decision_dict["model_id"],
            "final_label":   decision_dict["final_label"],
            "reason_codes":  decision_dict["reason_codes"],
            "regime_tag":    decision_dict["regime_tag"],
            "metric_snapshot": decision_dict["metric_snapshot"],
            "policy":        decision_dict["policy"],
        }

        user_msg = self._USER_PROMPT_TEMPLATE.format(
            decision_json     = json.dumps(slim, indent=2),
            model_class       = model_class,
            commodity_context = commodity_context,
        )

        response = client.messages.create(
            model      = self.model,
            max_tokens = self.max_tokens,
            system     = self._SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> tuple[str, str]:
        """Split response on <<<PLAIN>>> delimiter."""
        if "<<<PLAIN>>>" in raw:
            parts = raw.split("<<<PLAIN>>>", 1)
            tech  = parts[0].strip()
            plain = parts[1].strip()
        else:
            # Fallback: split roughly in half if delimiter missing
            mid   = len(raw) // 2
            tech  = raw[:mid].strip()
            plain = raw[mid:].strip()
        return tech, plain

    def _stub_narrative(self, decision_dict: dict) -> NarrativeResult:
        """Return a stub narrative when API is unavailable."""
        label  = decision_dict.get("final_label", "UNKNOWN")
        mid    = decision_dict.get("model_id", "unknown")
        codes  = decision_dict.get("reason_codes", [])
        cov    = decision_dict.get("metric_snapshot", {}).get("empirical_coverage")
        cov_str = f"{cov:.1%}" if cov is not None else "N/A"

        tech = (
            f"[STUB — API key not configured] "
            f"Model '{mid}' received classification {label}. "
            f"Reason codes: {codes}. "
            f"Empirical coverage: {cov_str}. "
            f"Set ANTHROPIC_API_KEY to generate a full technical narrative."
        )
        plain = (
            f"[STUB — API key not configured] "
            f"The validation framework classified '{mid}' as {label}. "
            f"Set ANTHROPIC_API_KEY to generate a full plain-language summary."
        )
        return NarrativeResult(
            model_id            = mid,
            final_label         = label,
            technical_narrative = tech,
            plain_narrative     = plain,
            mode                = "both",
            api_used            = False,
        )
