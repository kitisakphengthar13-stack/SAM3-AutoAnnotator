"""Verify the installed Ultralytics distribution still exposes SAM APIs we use.

This intentionally avoids importing ultralytics: the lightweight CI contract job
installs the package without heavyweight runtime dependencies such as torch.
"""

from importlib.metadata import distribution
from pathlib import Path


def require_symbol(path: Path, symbol: str):
    text = path.read_text(encoding="utf-8")
    if symbol not in text:
        raise RuntimeError(f"Ultralytics API contract changed: {symbol!r} not found in {path}")


def main():
    dist = distribution("ultralytics")
    package = Path(dist.locate_file("ultralytics"))
    top_level = package / "__init__.py"
    sam_init = package / "models" / "sam" / "__init__.py"
    if not top_level.is_file() or not sam_init.is_file():
        raise RuntimeError("Ultralytics SAM package layout required by this app is missing.")

    # src/sam3/predictor.py imports these exact public paths.
    require_symbol(top_level, "SAM")
    require_symbol(sam_init, "SAM3SemanticPredictor")

    print(f"Ultralytics {dist.version}: SAM import contract present")


if __name__ == "__main__":
    main()
