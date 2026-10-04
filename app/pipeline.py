"""Steps, a registry, and a pipeline that runs them in order.

An agent is a list of step names. Each step does one job, declares what it needs and what it
provides, and is registered under a name. Changing an agent means changing its list, or
registering a new step, and never editing the agent itself.

    @register("lease", "check_market_rent")
    class CheckMarketRent(BaseStep):
        needs = frozenset({"fields"})
        provides = frozenset({"market_check"})
        def run(self, ctx): ...

A bad list fails when the app starts, not on the first upload: an unknown step name, or a step
whose needs no earlier step provides, raises PipelineError with the available steps listed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .ingest import OcrEngine
from .units import UnitRegistry
from .vision import VisionModel


class PipelineError(ValueError):
    """The step list is invalid. Raised at construction."""


class StepError(RuntimeError):
    """A step failed while running. The message names the step."""


@dataclass
class Services:
    """Everything a step may depend on. A step that needs one that is missing fails at startup."""

    units: UnitRegistry | None = None
    ruleset: dict | None = None
    ocr: OcrEngine | None = None
    vision: VisionModel | None = None


class Step(Protocol):
    name: str
    needs: frozenset[str]
    provides: frozenset[str]
    rerun: bool  # run again after a human override, not only on first ingest

    def run(self, ctx: Any) -> None: ...


class BaseStep:
    name = ""
    needs: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    rerun = False
    uses: tuple[str, ...] = ()  # Services fields this step requires

    def __init__(self, services: Services):
        for attr in self.uses:
            if getattr(services, attr) is None:
                raise PipelineError(f"step '{self.name}' needs the '{attr}' service, which is not configured")
        self.services = services

    def run(self, ctx: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError


_REGISTRY: dict[str, dict[str, type[BaseStep]]] = {"lease": {}, "issue": {}}


def register(kind: str, name: str):
    """Register a step class under (kind, name). kind is the agent: 'lease' or 'issue'."""

    def decorate(cls: type[BaseStep]) -> type[BaseStep]:
        cls.name = name
        _REGISTRY[kind][name] = cls
        return cls

    return decorate


def available_steps(kind: str) -> dict[str, dict]:
    return {n: {"needs": sorted(c.needs), "provides": sorted(c.provides), "reruns_on_override": c.rerun,
                "summary": (c.__doc__ or "").strip().splitlines()[0] if c.__doc__ else ""}
            for n, c in sorted(_REGISTRY[kind].items())}


class Pipeline:
    def __init__(self, kind: str, step_names: list[str] | tuple[str, ...], services: Services):
        known = _REGISTRY[kind]
        unknown = [n for n in step_names if n not in known]
        if unknown:
            raise PipelineError(f"unknown {kind} step(s) {unknown}; available: {sorted(known)}")
        self.kind, self.names = kind, list(step_names)
        self.steps: list[BaseStep] = [known[n](services) for n in step_names]
        have: set[str] = set()
        for step in self.steps:
            missing = step.needs - have
            if missing:
                raise PipelineError(
                    f"{kind} step '{step.name}' needs {sorted(missing)}, which no earlier step provides "
                    f"(order so far: {[s.name for s in self.steps[:self.steps.index(step)]]})")
            have |= step.provides

    def run(self, ctx: Any, only_rerun: bool = False) -> None:
        """Run every step in order, or, after a human override, only the steps marked rerun."""
        for step in self.steps:
            if only_rerun and not step.rerun:
                continue
            try:
                step.run(ctx)
            except Exception as exc:
                raise StepError(f"{self.kind} step '{step.name}' failed: {exc}") from exc
