#!/bin/sh
# keeps the live engine alive: restarts it if it ever exits (state is on disk, so it resumes)
cd "$(dirname "$0")/.."
while true; do
  .venv-gpu/bin/python -u -W ignore -m trader.crypto.engine >> logs/crypto_engine.log 2>&1
  echo "$(date -u +%FT%TZ) engine exited ($?), restarting in 30s" >> logs/crypto_engine.log
  sleep 30
done
