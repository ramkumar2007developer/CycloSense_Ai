"""LLM Vision Verification interface and implementations."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.ml_config import LLMVerificationConfig
from src.exceptions import CycloSenseError

logger = logging.getLogger(__name__)

VALID_ASSESSMENTS = {"cyclone_consistent", "non_cyclone_consistent", "uncertain"}


class LLMVerificationError(CycloSenseError):
    """Raised when an LLM verification call fails."""


@dataclass(frozen=True)
class LLMVerificationResult:
    """Structured assessment returned by LLM vision verification."""

    visual_assessment: str  # "cyclone_consistent" | "non_cyclone_consistent" | "uncertain"
    confidence: float  # 0.0 to 1.0
    reason: str
    needs_human_review: bool
    raw_response: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.visual_assessment not in VALID_ASSESSMENTS:
            raise ValueError(
                f"Invalid visual_assessment {self.visual_assessment!r}. "
                f"Must be one of {VALID_ASSESSMENTS}"
            )
        object.__setattr__(
            self, "confidence", float(max(0.0, min(1.0, self.confidence)))
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "visual_assessment": self.visual_assessment,
            "confidence": self.confidence,
            "reason": self.reason,
            "needs_human_review": self.needs_human_review,
        }


class LLMVisionVerifier(ABC):
    """Abstract interface for LLM/vision verification models."""

    @abstractmethod
    def verify_image(
        self, image_path: Path, ml_context: dict[str, Any]
    ) -> LLMVerificationResult:
        """Perform secondary visual verification on a satellite image.

        Args:
            image_path: Path to satellite image file.
            ml_context: Context dictionary containing per-input ML probabilities
                        and relevant environmental features (NO training/eval metrics).

        Returns:
            Structured LLMVerificationResult.

        Raises:
            LLMVerificationError: If verification cannot be completed.
        """
        raise NotImplementedError


class MockLLMVerifier(LLMVisionVerifier):
    """Deterministic mock verifier for unit testing."""

    def __init__(
        self,
        response: str = "cyclone_consistent",
        confidence: float = 0.85,
        reason: str = "Mock verification analysis",
        needs_human_review: bool = False,
        error: Exception | str | None = None,
    ) -> None:
        self.response = response
        self.confidence = confidence
        self.reason = reason
        self.needs_human_review = needs_human_review
        self.error = error
        self.call_count: int = 0
        self.calls: list[dict[str, Any]] = []

    def verify_image(
        self, image_path: Path, ml_context: dict[str, Any]
    ) -> LLMVerificationResult:
        self.call_count += 1
        self.calls.append({
            "image_path": image_path,
            "ml_context": dict(ml_context),
        })

        if self.error is not None:
            if isinstance(self.error, Exception):
                raise self.error
            if self.error == "timeout":
                raise LLMVerificationError("Mock verifier simulated timeout.")
            if self.error == "malformed_json":
                raise LLMVerificationError("Mock verifier received malformed JSON.")
            raise LLMVerificationError(f"Mock verifier error: {self.error}")

        return LLMVerificationResult(
            visual_assessment=self.response,
            confidence=self.confidence,
            reason=self.reason,
            needs_human_review=self.needs_human_review,
            raw_response={"mock": True, "call_index": self.call_count},
        )


class GroqVisionVerifier(LLMVisionVerifier):
    """Vision verifier calling Groq's OpenAI-compatible chat/completions API."""

    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "llama-3.2-11b-vision-preview",
        timeout_seconds: float = 15.0,
        max_retries: int = 2,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMVerificationError(
                "GROQ_API_KEY is not set. Please set the environment variable or pass api_key."
            )
        self._api_key = key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)

    def _encode_image(self, image_path: Path) -> str:
        with image_path.open("rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        ext = image_path.suffix.lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext or "jpeg"
        return f"data:image/{mime};base64,{encoded}"

    def verify_image(
        self, image_path: Path, ml_context: dict[str, Any]
    ) -> LLMVerificationResult:
        if not image_path.is_file():
            raise LLMVerificationError(f"Image not found for verification: {image_path}")

        image_url = self._encode_image(image_path)

        # Context sanitized to ensure NO training metrics (e.g. F1, AUC) are present
        allowed_keys = {
            "image_probability",
            "numerical_probability",
            "fusion_probability",
            "environmental_context",
        }
        sanitized_context = {k: v for k, v in ml_context.items() if k in allowed_keys}

        system_prompt = (
            "You are performing secondary visual verification of a cyclone prediction.\n"
            "Do not assume that clouds indicate a cyclone.\n"
            "Normal scattered clouds, cirrus, dense cloud formations, convection, "
            "and other non-cyclonic atmospheric patterns can occur without a cyclone.\n\n"
            "Inspect the image for visual evidence consistent with a tropical cyclone structure.\n\n"
            "Return valid JSON only matching this exact schema:\n"
            "{\n"
            '  "visual_assessment": "cyclone_consistent" | "non_cyclone_consistent" | "uncertain",\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "reason": "<concise meteorological rationale>",\n'
            '  "needs_human_review": <true | false>\n'
            "}"
        )

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"ML Model Signals & Context for this observation:\n"
                    f"{json.dumps(sanitized_context, indent=2)}\n\n"
                    "Perform secondary visual verification and return the structured JSON assessment."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
        ]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 512,
        }

        body_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "CycloSense-Verification-Gate/1.0",
        }

        last_error: Exception | None = None
        for attempt in range(1 + self.max_retries):
            try:
                req = urllib.request.Request(
                    self.ENDPOINT, data=body_bytes, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_bytes = resp.read()
                    data = json.loads(resp_bytes.decode("utf-8"))

                content = data["choices"][0]["message"]["content"]
                return self._parse_llm_json(content, raw=data)

            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as err:
                detail = str(err)
                if isinstance(err, urllib.error.HTTPError):
                    try:
                        err_body = err.read().decode("utf-8", errors="replace")
                        detail = f"{err.code} {err.reason}: {err_body}"
                    except Exception:
                        pass
                last_error = RuntimeError(detail) if isinstance(err, urllib.error.HTTPError) else err
                logger.warning(
                    "LLM vision verification attempt %d/%d failed: %s",
                    attempt + 1,
                    1 + self.max_retries,
                    detail,
                )

        raise LLMVerificationError(
            f"LLM verification failed after {1 + self.max_retries} attempts: {last_error}"
        ) from last_error

    def _parse_llm_json(self, content: str, raw: dict[str, Any]) -> LLMVerificationResult:
        text = content.strip()
        # Handle markdown blocks if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

        parsed = json.loads(text)
        assessment = str(parsed.get("visual_assessment", "uncertain")).strip().lower()
        if assessment not in VALID_ASSESSMENTS:
            assessment = "uncertain"

        confidence = float(parsed.get("confidence", 0.5))
        reason = str(parsed.get("reason", "No reason provided."))
        needs_review = bool(parsed.get("needs_human_review", assessment == "uncertain"))

        return LLMVerificationResult(
            visual_assessment=assessment,
            confidence=confidence,
            reason=reason,
            needs_human_review=needs_review,
            raw_response=raw,
        )


def build_vision_verifier(
    config: LLMVerificationConfig | None = None,
    api_key: str | None = None,
) -> LLMVisionVerifier:
    """Factory creating an LLMVisionVerifier instance based on config and env."""
    cfg = config or LLMVerificationConfig.from_raw({})
    provider = cfg.provider.lower().strip()

    if provider == "groq":
        key = api_key or os.environ.get("GROQ_API_KEY")
        if key:
            return GroqVisionVerifier(
                api_key=key,
                model_name=cfg.model_name,
                timeout_seconds=cfg.timeout_seconds,
                max_retries=cfg.max_retries,
            )
        logger.warning("Groq provider requested but GROQ_API_KEY is unset; using MockLLMVerifier.")
        return MockLLMVerifier()

    return MockLLMVerifier()
