"""One server for everything: results site at /, live Gradio demo at /demo.

  make run        # rebuilds site/ from reports/metrics/*.json, then starts this server on http://localhost:8000
  python server.py [--port 8000]
"""
from __future__ import annotations

import argparse
import os

for _k in ("NO_PROXY", "no_proxy"):  # a system HTTP proxy must not intercept Gradio's calls to its own server
    os.environ[_k] = ",".join(filter(None, [os.environ.get(_k), "127.0.0.1", "localhost"]))

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import CSS, JS, THEME, build

app = FastAPI(title="city-incident-ai")
app = gr.mount_gradio_app(app, build().queue(), path="/demo", theme=THEME, css=CSS, js=JS)  # must be mounted before "/"
app.mount("/", StaticFiles(directory="site", html=True), name="site")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"results site + live demo: http://localhost:{args.port}  (demo alone: /demo)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
