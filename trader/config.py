import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def settings() -> dict:
    load_env()
    return yaml.safe_load((ROOT / "config/settings.yaml").read_text())


def risk_limits() -> dict:
    return yaml.safe_load((ROOT / "config/risk_limits.yaml").read_text())


def path(*parts) -> Path:
    p = ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
