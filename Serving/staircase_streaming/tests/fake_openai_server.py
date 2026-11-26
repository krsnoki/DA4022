"""Tiny OpenAI-compatible streaming server (stdlib) for testing OpenAICompatBackend."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER_TOKENS = {"slow": 60, "fast": 40, "agg": 30}
DELAY_S = {"slow": 0.004, "fast": 0.001, "agg": 0.001}


class FakeServer:
    """Serves every model id on one port; records request bodies."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.disconnects = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # silence
                pass

            def _json(self, obj: dict) -> None:
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path == "/v1/models":
                    self._json({"data": [{"id": k} for k in ANSWER_TOKENS]})
                elif self.path == "/version":
                    self._json({"version": "fake-0.0"})
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                model = body["model"]
                msgs = body["messages"]
                already = 0
                if body.get("continue_final_message"):
                    already = len(msgs[-1]["content"].split())
                prompt_tokens = sum(len(m["content"].split()) for m in msgs)
                n = max(0, min(body["max_tokens"], ANSWER_TOKENS[model] - already))
                finish = "stop" if already + n >= ANSWER_TOKENS[model] else "length"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for i in range(n):
                        time.sleep(DELAY_S[model])
                        chunk = {"choices": [{"index": 0, "delta": {"content": f"w{already + i} "},
                                              "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    end = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                    usage = {"choices": [], "usage": {
                        "prompt_tokens": prompt_tokens, "completion_tokens": n,
                        "prompt_tokens_details": {"cached_tokens": 0}}}
                    for obj in (end, usage):
                        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    outer.disconnects += 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
