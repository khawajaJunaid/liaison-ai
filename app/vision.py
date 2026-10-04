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
Report only what is visible. If the photo is unclear, use condition "unknown" and low confidence."""


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


class AnthropicVision:
    """Real vision model. Needs `pip install anthropic` and ANTHROPIC_API_KEY."""

    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic  # imported lazily so the base install needs no SDK

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("VISION_MODEL", "claude-sonnet-5-5")

    def assess(self, image: bytes, filename: str, note: str = "") -> PhotoAssessment:
        media = mimetypes.guess_type(filename)[0] or "image/jpeg"
        reply = self.client.messages.create(
            model=self.model, max_tokens=1024, system=PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media,
                                             "data": base64.b64encode(image).decode()}},
                {"type": "text", "text": f"Reporter note: {note or '(none)'}"}]}])
        return parse_assessment(reply.content[0].text, filename, self.model)


def make_vision(name: str | None = None) -> VisionModel:
    name = (name or os.environ.get("LEASE_AGENT_VISION", "stub")).lower()
    return AnthropicVision() if name == "anthropic" else StubVision()
