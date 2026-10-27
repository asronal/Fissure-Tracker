"""
Run the web app on the laptop (no AWS needed) and open it from a phone on the same Wi-Fi.

    python serve_local.py                 rules backup only (default)
    python serve_local.py --ai            also try Amazon Nova on Bedrock (falls back to rules if it fails)

Data goes to outputs/store/. Engineer e-mails are printed here in the terminal instead of being sent.
"""

import os
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

from agent import OUTPUT_DIR, make_llm
from app import App
from store import LocalStore

PORT = 8000
ACCESS_CODE = os.environ.get("ACCESS_CODE", "demo")


class PrintingNotifier:
    def notify_engineer(self, subject, body):
        print(f"\n=== E-mail to the engineer (not sent, local test) ===\n{subject}\n{body}\n", flush=True)


def lan_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))     # no packet is sent; just picks the Wi-Fi address
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def _event(self, method):
            url = urlsplit(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            headers = {k.lower(): v for k, v in self.headers.items()}
            headers["x-forwarded-proto"] = "http"
            # Same shape as an AWS Lambda Function URL event, so app.py can't tell the difference
            return {"rawPath": url.path, "queryStringParameters": dict(parse_qsl(url.query)) or None,
                    "headers": headers, "requestContext": {"http": {"method": method}},
                    "body": body.decode("latin-1"), "isBase64Encoded": False}

        def _send(self, response):
            import base64
            body = response.get("body", "")
            body = base64.b64decode(body) if response.get("isBase64Encoded") else body.encode("utf-8")
            self.send_response(response["statusCode"])
            for k, v in response.get("headers", {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(app.handle(self._event("GET")))

        def do_POST(self):
            event = self._event("POST")
            event["body"] = event["body"].encode("latin-1")     # keep photo bytes exactly as sent
            self._send(app.handle(event))

        def log_message(self, fmt, *args):
            sys.stderr.write("  " + fmt % args + "\n")

    return Handler


def main():
    use_ai = "--ai" in sys.argv
    os.environ.setdefault("AGENT_MODE", "auto" if use_ai else "rules")
    store = LocalStore(os.path.join(OUTPUT_DIR, "store"))
    app = App(store, PrintingNotifier(), make_llm, ACCESS_CODE)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), make_handler(app))
    print(f"Crack monitor running ({'AI + rules backup' if use_ai else 'rules backup only'}).")
    print(f"  Laptop: http://localhost:{PORT}/?code={ACCESS_CODE}")
    print(f"  Phone (same Wi-Fi): http://{lan_ip()}:{PORT}/?code={ACCESS_CODE}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
