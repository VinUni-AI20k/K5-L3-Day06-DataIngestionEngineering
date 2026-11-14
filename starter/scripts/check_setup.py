"""Read-only setup check for Qdrant + OpenRouter embeddings (not the pipeline).

Verifies: env vars present, Qdrant reachable and authenticated, and one real
embedding call returns a vector of the expected size. Exits non-zero on any
failure; never falls back to mock vectors.
"""
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

required = ["QDRANT_URL", "QDRANT_API_KEY", "OPENROUTER_API_KEY", "EMBEDDING_MODEL"]
missing = [name for name in required if not os.getenv(name)]
if missing:
    sys.exit(f"Missing in starter/.env: {', '.join(missing)} (see .env.example).")

# 1. Qdrant
try:
    # qdrant-client defaults to port 6333 when the URL has none; a reverse-proxied
    # HTTPS host listens on 443, so pass the port explicitly.
    parsed = urlparse(os.environ["QDRANT_URL"])
    port = parsed.port or (443 if parsed.scheme == "https" else 6333)
    client = QdrantClient(url=os.environ["QDRANT_URL"], port=port,
                          api_key=os.environ["QDRANT_API_KEY"], timeout=20)
    names = [c.name for c in client.get_collections().collections]
    version = requests.get(os.environ["QDRANT_URL"], timeout=20,
                           headers={"api-key": os.environ["QDRANT_API_KEY"]}).json().get("version")
except Exception as exc:  # noqa: BLE001 - report any backend failure verbatim
    sys.exit(f"Qdrant check failed: {type(exc).__name__}: {exc}")
print(f"Qdrant: {os.environ['QDRANT_URL']} | server {version} | collections: {names or '[]'}")

# 2. OpenRouter embedding (costs ~USD 0.000002)
model = os.environ["EMBEDDING_MODEL"]
resp = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    json={"model": model, "input": ["kiểm tra kết nối"]},
    timeout=60,
)
if resp.status_code != 200:
    sys.exit(f"Embedding check failed: HTTP {resp.status_code}: {resp.text[:300]}")
body = resp.json()
dim = len(body["data"][0]["embedding"])
print(f"Embedding: {model} via OpenRouter | dim {dim} | cost {body.get('usage', {}).get('cost')}")
print("Setup OK. Pipeline functions still need to be implemented.")
