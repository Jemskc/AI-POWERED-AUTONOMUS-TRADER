"""Start the live crypto dashboard. Prints the URL (with access token) and the SSH tunnel command."""
import _path  # noqa: F401

import os
import secrets
import socket

from trader.config import ROOT, load_env

PORT = 8766

if __name__ == "__main__":
    load_env()
    if not os.environ.get("DASHBOARD_TOKEN"):
        tok = secrets.token_urlsafe(24)
        with open(ROOT / ".env", "a") as f:
            f.write(f"\nDASHBOARD_TOKEN={tok}\n")
        os.environ["DASHBOARD_TOKEN"] = tok
    os.chmod(ROOT / ".env", 0o600)
    host = socket.gethostname()
    print(f"\n  Dashboard: http://localhost:{PORT}/?token={os.environ['DASHBOARD_TOKEN']}\n"
          f"  Tunnel from your laptop: ssh -J <you>@<login-node> -L {PORT}:localhost:{PORT} <you>@{host}\n", flush=True)
    import uvicorn
    uvicorn.run("trader.crypto.server:app", host="127.0.0.1", port=PORT, log_level="warning")
