"""Start the dashboard (and its built-in scheduler). Prints the URL with the access token."""
import _path  # noqa: F401

import os
import secrets

from trader.config import ROOT, load_env, settings

if __name__ == "__main__":
    load_env()
    if not os.environ.get("DASHBOARD_TOKEN"):
        tok = secrets.token_urlsafe(24)
        with open(ROOT / ".env", "a") as f:
            f.write(f"\nDASHBOARD_TOKEN={tok}\n")
        os.environ["DASHBOARD_TOKEN"] = tok
    os.chmod(ROOT / ".env", 0o600)
    d = settings()["dashboard"]
    host = d["host"]
    url = f"http://localhost:{d['port']}/?token={os.environ['DASHBOARD_TOKEN']}"
    print(f"\n  Dashboard: {url}\n"
          f"  From your laptop: ssh -L {d['port']}:localhost:{d['port']} <you>@<this-server>  then open the URL above\n",
          flush=True)
    import uvicorn
    uvicorn.run("trader.dashboard.server:app", host=host, port=d["port"], log_level="warning")
