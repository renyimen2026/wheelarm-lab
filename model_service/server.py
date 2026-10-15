"""Loopback-only synchronous inference with bounded HTTP input and no job queue."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading

from jsonschema.exceptions import SchemaError, ValidationError

from .protocol import MAX_BODY, model_info, strict_json


def handler_for(engine, root):
    gate = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def reply(self, status, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                logging.warning("Client disconnected; no navigation action is produced")

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"backend": "pytorch-cuda", "keep_loaded": engine.keep_loaded})
            else:
                self.reply(404, {"error": "Unknown route"})

        def do_POST(self):
            if self.path not in ("/api/chat", "/api/show", "/api/preload"):
                self.reply(404, {"error": "Unknown route"})
                return
            if not gate.acquire(blocking=False):
                self.reply(503, {"error": "Model service busy; no request queued"})
                return
            gate_held = True
            try:
                if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Expected JSON with Content-Length")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    self.reply(413, {"error": "Request body too large or empty"})
                    return
                data = self.rfile.read(length)
                if len(data) != length:
                    raise ValueError("Incomplete request body")
                body = strict_json(data)
                path, vision = model_info(body, root)
                if self.path == "/api/show":
                    self.reply(200, {"capabilities": ["completion", "vision"] if vision else ["completion"],
                                     "backend": "pytorch-cuda", "model_path": str(path),
                                     "features": ["preload"] if engine.keep_loaded else []})
                elif self.path == "/api/preload":
                    if not engine.keep_loaded:
                        raise ValueError("Preload requires --keep-loaded or --preload at startup")
                    logging.info("Preload requested: %s", body["model"])
                    try:
                        engine.load(path, vision)
                    except Exception:
                        engine.unload()
                        raise
                    logging.info("Preload ready: %s", body["model"])
                    gate.release()
                    gate_held = False
                    self.reply(200, {"model": body["model"], "status": "ready", "backend": "pytorch-cuda"})
                else:
                    self.reply(200, engine.chat(body, path, vision))
            except (ValueError, TypeError, SchemaError, ValidationError, UnicodeError, RecursionError) as error:
                logging.warning("Request rejected (%s): %s", type(error).__name__, str(error)[:500])
                self.reply(400, {"error": str(error)[:500]})
            except Exception as error:
                logging.exception("Inference request failed")
                self.reply(500, {"error": str(error)[:500]})
            finally:
                if gate_held:
                    gate.release()

    return Handler


def main():
    from .engine import CudaEngine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("~/Models/transformers"))
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--keep-loaded", action="store_true")
    parser.add_argument("--preload", choices=["none", "text", "vision"], default="text",
                        help="Startup preload (default: text); text/vision imply --keep-loaded")
    parser.add_argument("--generation-timeout", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 < args.generation_timeout <= 600:
        parser.error("Invalid port or generation timeout")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = CudaEngine(args.keep_loaded or args.preload != "none", args.generation_timeout)
    server = None
    try:
        root = args.models_root.expanduser()
        if args.preload != "none":
            alias = "qwen3:4b" if args.preload == "text" else "qwen3-vl:4b-instruct"
            path, vision = model_info({"model": alias}, root)
            logging.info("Preloading %s before accepting requests", alias)
            engine.load(path, vision)
            logging.info("Preload complete: %s", alias)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(engine, root))
        # Wait for an in-flight request before releasing its model on shutdown.
        server.daemon_threads = False
        logging.info("Local model service http://127.0.0.1:%s", args.port)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        engine.unload()


if __name__ == "__main__":
    main()
