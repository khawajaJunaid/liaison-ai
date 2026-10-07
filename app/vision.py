"""Photo assessment behind an interface.

Reading a photo is the one step here that genuinely needs a model, so it is the one place a
model plugs in. StubVision is deterministic and key-free (it reads keywords from the file name
and the reporter's note) so the whole flow runs and tests offline; every stub result is marked
needs_review. AnthropicVision is the real implementation, selected with LEASE_AGENT_VISION=anthropic.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from typing import Protocol

from .domain import Damage, EquipmentItem, PhotoAssessment


class VisionModel(Protocol):
    name: str

    def assess(self, image: bytes, filename: str, note: str = "") -> PhotoAssessment: ...


EQUIPMENT = [
    (r"\b(ac|a/c|air\s?con\w*|split)\b", "Split AC unit", "HVAC"),
    (r"\b(heater|boiler|geyser)\b", "Water heater", "Plumbing"),
    (r"\b(fridge|refrigerator)\b", "Refrigerator", "Appliance"),
    (r"\b(oven|cooker|stove|hob)\b", "Oven / hob", "Appliance"),
    (r"\b(sink|faucet|tap|basin)\b", "Sink and faucet", "Plumbing"),
    (r"\b(wall|ceiling|paint)\b", "Wall / ceiling surface", "Finishes"),
    (r"\b(floor|tile|tiles)\b", "Floor tiles", "Finishes"),
    (r"\b(door)\b", "Door", "Joinery"),
    (r"\b(window)\b", "Window", "Joinery"),
]
DAMAGES = [
    (r"\b(leak\w*|drip\w*)\b", "Water leak", "high"),
    (r"\b(broken|smashed|shattered)\b", "Broken part", "high"),
    (r"\b(rust\w*|corro\w*)\b", "Corrosion", "medium"),
    (r"\b(crack\w*)\b", "Crack", "medium"),
    (r"\b(stain\w*|mou?ld\w*|damp\w*)\b", "Stain / mould", "medium"),
    (r"\b(peel\w*)\b", "Peeling finish", "low"),
    (r"\b(dent\w*|scratch\w*)\b", "Surface damage", "low"),
]


class StubVision:
    name = "stub-v1"

    def assess(self, image: bytes, filename: str, note: str = "") -> PhotoAssessment:
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename)
        text = f"{re.sub(r'[^A-Za-z0-9/]+', ' ', stem)} {note}".lower()
        equipment = [(n, c) for pat, n, c in EQUIPMENT if re.search(pat, text)]
        damages = [Damage(type=t, severity=s, affects=equipment[0][0] if equipment else None,
                          description=f"{t} indicated by '{filename}'" + (f" on the {equipment[0][0].lower()}" if equipment else ""))
                   for pat, t, s in DAMAGES if re.search(pat, text)]
        if any(d.severity != "low" for d in damages):
            condition = "damaged"
        elif damages or re.search(r"\b(old|worn|aged)\b", text):
            condition = "worn"
        elif re.search(r"\bnew\b", text):
            condition = "new"
        elif equipment:
            condition = "good"
        else:
            condition = "unknown"
        recognised = bool(equipment or damages)
        return PhotoAssessment(
            filename=filename, condition=condition, damages=damages,
            equipment=[EquipmentItem(name=n, category=c, condition=condition) for n, c in equipment],
            confidence=0.55 if recognised else 0.15, model=self.name, needs_review=True)


PROMPT = """You assess photos of rental property units for an owner. Return ONLY JSON:
{"condition": "new|good|worn|damaged|unknown",
 "damages": [{"type": str, "severity": "high|medium|low", "description": str, "affects": str|null}],
 "equipment": [{"name": str, "category": str, "condition": str}],
 "confidence": number between 0 and 1}
Rules:
- Report only what is visible. If the photo is unclear, use condition "unknown" and low confidence.
- "affects" is the equipment or fixture the problem comes from or concerns, for example the air conditioner
  behind a wall stain. Use null if there is none. Put the damaged surface in "description".
- In "description", say what you see and the most likely cause if it is apparent.
- "equipment" lists only what an owner maintains: air conditioners, water heaters, appliances, fixtures, taps,
  doors, windows. Leave out furniture, curtains and the tenant's belongings.
- An equipment "condition" is exactly one of: new, good, worn, damaged, unknown."""


def parse_assessment(raw: str, filename: str, model: str) -> PhotoAssessment:
    """Parse a model reply into a PhotoAssessment; anything malformed becomes 'unknown, needs review'."""
    try:
        body = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return PhotoAssessment(filename=filename, model=model,
                               needs_review=float(data.get("confidence", 0)) < 0.6,
                               **{k: data[k] for k in ("condition", "damages", "equipment", "confidence") if k in data})
    except (ValueError, TypeError, KeyError):
        return PhotoAssessment(filename=filename, condition="unknown", model=model, confidence=0.0, needs_review=True)


class VisionUnavailable(RuntimeError):
    """The chosen provider cannot start: missing key, or its SDK is not installed."""


def _media_type(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "image/jpeg"


class AnthropicVision:
    """Real vision model. Needs `pip install anthropic` and ANTHROPIC_API_KEY."""

    name = "anthropic"

    def __init__(self, model: str | None = None, client=None):
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise VisionUnavailable("anthropic vision needs ANTHROPIC_API_KEY to be set")
            try:
                import anthropic  # imported lazily so the base install needs no SDK
            except ImportError as exc:
                raise VisionUnavailable("anthropic vision needs `pip install anthropic`") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("VISION_MODEL", "claude-sonnet-5-5")

    def assess(self, image: bytes, filename: str, note: str = "") -> PhotoAssessment:
        reply = self.client.messages.create(
            model=self.model, max_tokens=1024, system=PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": _media_type(filename),
                                             "data": base64.b64encode(image).decode()}},
                {"type": "text", "text": f"Reporter note: {note or '(none)'}"}]}])
        return parse_assessment(reply.content[0].text, filename, self.model)


class OpenAIVision:
    """Vision through the OpenAI chat API, or any server that speaks it (set OPENAI_BASE_URL).

    Needs `pip install openai` and OPENAI_API_KEY. A local OpenAI-compatible server (Ollama, vLLM,
    LM Studio) needs no real key: set OPENAI_BASE_URL and a vision-capable model in VISION_MODEL.
    """

    name = "openai"

    def __init__(self, model: str | None = None, client=None):
        if client is None:
            local = bool(os.environ.get("OPENAI_BASE_URL"))
            if not local and not os.environ.get("OPENAI_API_KEY"):
                raise VisionUnavailable("openai vision needs OPENAI_API_KEY (or OPENAI_BASE_URL for a local server)")
            try:
                import openai  # imported lazily so the base install needs no SDK
            except ImportError as exc:
                raise VisionUnavailable("openai vision needs `pip install openai`") from exc
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "not-needed")
        self.client = client
        self.model = model or os.environ.get("VISION_MODEL", "gpt-4o")

    def assess(self, image: bytes, filename: str, note: str = "") -> PhotoAssessment:
        data_url = f"data:{_media_type(filename)};base64,{base64.b64encode(image).decode()}"
        reply = self.client.chat.completions.create(  # no max-token argument: its name differs across model families
            model=self.model, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": PROMPT},
                      {"role": "user", "content": [
                          {"type": "text", "text": f"Reporter note: {note or '(none)'}"},
                          {"type": "image_url", "image_url": {"url": data_url}}]}])
        return parse_assessment(reply.choices[0].message.content or "", filename, self.model)


def make_vision(name: str | None = None) -> VisionModel:
    """Build the provider named in settings. An unknown name fails at startup: a typo must not
    quietly give you the stub's fake answers."""
    name = (name if name is not None else os.environ.get("LEASE_AGENT_VISION", "stub")).strip().lower() or "stub"
    if name == "stub":
        return StubVision()
    if name == "anthropic":
        return AnthropicVision()
    if name == "openai":
        return OpenAIVision()
    raise ValueError(f"unknown vision provider '{name}' (expected stub, anthropic or openai)")
