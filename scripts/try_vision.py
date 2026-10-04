"""Try a real vision provider on real photos, from your own terminal.

    pip install -r requirements-vision.txt
    export ANTHROPIC_API_KEY=...            # or OPENAI_API_KEY=...  (set it in your shell, never in a file)
    python -m scripts.try_vision anthropic photo1.jpg photo2.jpg
    python -m scripts.try_vision openai photo1.jpg

Local OpenAI-compatible server (Ollama, vLLM, LM Studio):
    export OPENAI_BASE_URL=http://localhost:11434/v1 VISION_MODEL=<a vision-capable model>
    python -m scripts.try_vision openai photo1.jpg

Use real photos. The placeholder images in samples/photos have their file name written on them, so a
real model would simply read it back. The key is read from the environment and is never printed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from app.vision import VisionUnavailable, make_vision


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("anthropic", "openai"):
        print(__doc__)
        return 2
    provider, files = argv[0], [Path(p) for p in argv[1:]]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        print(f"No such file(s): {', '.join(missing)}")
        return 2
    try:
        vision = make_vision(provider)
    except VisionUnavailable as exc:
        print(f"Cannot start {provider}: {exc}")
        return 1
    print(f"provider={vision.name} model={vision.model}\n")
    failures = 0
    for path in files:
        started = time.time()
        try:
            result = vision.assess(path.read_bytes(), path.name, "")
        except Exception as exc:  # show the provider's own message: bad key, no access to the model, rate limit
            failures += 1
            print(f"{path.name}: FAILED ({type(exc).__name__}: {exc})\n")
            continue
        print(f"{path.name}  ({time.time() - started:.1f}s)")
        print(result.model_dump_json(indent=2, exclude={"filename"}), "\n")
        if result.condition == "unknown" and result.confidence == 0.0:
            print("  note: the reply could not be parsed as the expected JSON, so it was treated as unknown\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
