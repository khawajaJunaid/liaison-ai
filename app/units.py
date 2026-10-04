"""Unit registry: seed records from units.json plus matching of a lease's unit reference."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .domain import Unit


class UnitRegistry:
    def __init__(self, units_path: Path):
        raw = json.loads(units_path.read_text())
        self.ownership_entity: str = raw["ownership_entity"]
        self._units: dict[str, Unit] = {}
        for prop in raw["properties"]:
            for building in prop["buildings"]:
                for u in building["units"]:
                    self._units[u["unit_id"]] = Unit(
                        **u,
                        building_id=building["building_id"],
                        building_name=building["name"],
                        property_id=prop["property_id"],
                        property_name=prop["name"],
                    )

    def all(self, states: dict[str, tuple[str, str | None]] | None = None) -> list[Unit]:
        """Seed units with the live status overlay ({unit_id: (status, lease_id)}) applied."""
        return [self._with_state(u, states) for u in self._units.values()]

    def get(self, unit_id: str, states=None) -> Unit | None:
        unit = self._units.get(unit_id)
        return self._with_state(unit, states) if unit else None

    @staticmethod
    def _with_state(unit: Unit, states) -> Unit:
        if states and unit.unit_id in states:
            status, lease_id = states[unit.unit_id]
            return unit.model_copy(update={"status": status, "lease_id": lease_id})
        return unit

    def resolve(self, ref: str | None, states=None) -> tuple[Unit | None, str]:
        """Match a free-text unit reference to a unit. Returns (unit, how_it_matched).

        Order: exact unit id, then "Apartment 1204 / Tower B" label, then parking bay.
        Anything ambiguous returns no unit: a wrong link is worse than a flagged one.
        """
        if not ref:
            return None, "no unit reference in the lease"
        text = ref.strip()

        m = re.search(r"\b([A-Z]{2,4}-[A-Z]-\d{3,4})\b", text.upper())
        if m and m.group(1) in self._units:
            return self.get(m.group(1), states), "exact unit id"

        number = re.search(r"\b(?:apartment|apt|unit|flat)?\s*(\d{3,4})\b", text, re.I)
        tower = re.search(r"\btower\s+([A-Z])\b", text, re.I)
        if number and tower:
            suffix = f"-{tower.group(1).upper()}-{number.group(1).zfill(4)}"
            hits = [u for uid, u in self._units.items() if uid.endswith(suffix)]
            if len(hits) == 1:
                return self.get(hits[0].unit_id, states), "apartment number and tower"
        if number:
            hits = [u for u in self._units.values() if u.label.endswith(number.group(1).zfill(4))]
            if len(hits) == 1:
                return self.get(hits[0].unit_id, states), "apartment number only (tower not stated)"

        bay = re.search(r"\b([A-Z]-\d{2,3})\b", text.upper())
        if bay:
            hits = [u for u in self._units.values() if u.parking_bay == bay.group(1)]
            if len(hits) == 1:
                return self.get(hits[0].unit_id, states), "parking bay"

        return None, f'"{text}" does not match any unit in the owner records'
