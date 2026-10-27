"""
Step 7: The web app - phone upload page, case pages with the agent's trace, engineer review, crack history

One router, `App.handle(event)`, used by both the laptop server (serve_local.py) and AWS Lambda
(Function URL events). No web framework and no JavaScript libraries: plain HTML + a little JS.

    GET  /                      upload page (needs the access code once: /?code=...)
    POST /api/photo?case=<id>   body = JPEG, header X-Access-Code -> runs the agent -> {"case_id": ...}
    GET  /case/<id>             the owner's page: result, photos, every agent step
    GET  /review/<id>?token=    the engineer's page (link from the e-mail)
    POST /review/<id>           engineer's decision (form button)
    GET  /cracks                all cracks: width over time (needs the access code)
    GET  /photo/<id>/<kind>     photos (laptop store only; on AWS photos come from S3 links)
    GET  /health                JSON: up, and which photo-alignment methods are loaded
"""

import base64
import hmac
import html
import json
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import cv2

from agent import ALERT_WIDTH_MM, MAX_PHOTOS_PER_CASE, handle_photo, record_engineer_decision
from align import deep_models

DISPLAY_TZ = timezone(timedelta(hours=5, minutes=30))     # India
MAX_UPLOAD_BYTES = 5_500_000      # Lambda Function URLs accept ~6 MB; the page shrinks photos well below this
MAX_UPLOADS_PER_DAY = 300         # protects the AWS bill if the link leaks
ID_PATTERN = re.compile(r"^[cp]\d{8}T\d{6,9}-[0-9a-f]{6}$")
PHOTO_KINDS = ("photo", "marked", "matches")

STATUS = {   # case status -> (status colour, icon, label)
    "waiting_for_photo": ("warning", "!", "Photo needed"),
    "waiting_for_engineer": ("serious", "!!", "Engineer reviewing"),
    "closed": ("good", "✓", "Done"),
    "new": ("warning", "…", "Working"),
}
READING_STATUS = {
    "baseline": ("good", "✓", "First reading"),
    "stable": ("good", "✓", "Stable"),
    "needs_review": ("serious", "!!", "Sent to engineer"),
}
TOOL_NAMES = {
    "check_photo": "Checked the photo", "measure_crack": "Measured the crack",
    "compare_with_history": "Compared with history", "ask_for_photo": "Asked for another photo",
    "save_reading": "Saved the reading", "request_engineer_review": "Asked an engineer",
    "fallback": "Switched to the rules backup", "engineer_decision": "Engineer decided",
}


def esc(value):
    return html.escape(str(value), quote=True)


def when(iso):
    try:
        return datetime.fromisoformat(iso).astimezone(DISPLAY_TZ).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return ""


def pill(kind, mapping):
    colour, icon, label = mapping.get(kind, ("warning", "?", kind))
    return f'<span class="pill {colour}"><span class="icon {colour}">{esc(icon)}</span>{esc(label)}</span>'


def who(decided_by):
    if decided_by == "rules":
        return '<span class="who rules">Rules backup</span>'
    if decided_by in ("engineer", "system"):
        return f'<span class="who {decided_by}">{decided_by.capitalize()}</span>'
    return f'<span class="who ai" title="{esc(decided_by)}">AI: {esc(decided_by.split(".")[-1].split(":")[0])}</span>'


# --- What each trace step says, in plain words --------------------------------------------------

def describe(step):
    tool, out, args = step["tool"], step.get("output") or {}, step.get("input") or {}
    if "error" in out:
        return f"<b>Blocked by a safety rule:</b> {esc(out['error'])}"
    if tool == "check_photo":
        if not out.get("marker_found"):
            return "No marker found" + (f" · problems: {esc(', '.join(out.get('problems', [])))}" if out.get("problems") else "")
        bits = [f"Marker #{esc(out['marker_id'])}", f"{out['px_per_mm']:.1f} px/mm", f"tilt {out['tilt_deg']:.0f}°"]
        if "blur_px" in out:
            bits.append(f"edge blur {out['blur_px']:.1f} px")
        text = " · ".join(bits)
        return text + (f" · <b>problems: {esc(', '.join(out['problems']))}</b>" if out.get("problems") else " · photo OK")
    if tool == "measure_crack":
        if not out.get("crack_found"):
            return "No crack found next to the marker"
        text = (f"Max width <b>{out['max_width_mm']:.2f} mm</b> ± {out['uncertainty_mm']:.2f} · typical "
                f"{out['typical_width_mm']:.2f} mm · length {out['length_mm']:.0f} mm · BRE category "
                f"{esc(out['bre_category'])} ({esc(out['bre_label'])}) · widest point {esc(out['widest_point_where'])}")
        return text + (f" · <b>{esc(', '.join(out['problems']))}</b>" if out.get("problems") else "")
    if tool == "compare_with_history":
        if not out.get("readings"):
            return "No earlier readings: this is the first one"
        last = out["last_reading"]
        text = f"Last reading {last['max_width_mm']:.2f} mm on {esc(when(last['time']))}"
        if "growth_mm" in out:
            text += f" · change <b>{out['growth_mm']:+.2f} mm</b> (± {out['growth_uncertainty_mm']:.2f})"
        a = out.get("alignment") or {}
        if a.get("shift_mm") is not None:
            text += f" · lined up with {a['inliers']} wall features, wall shift {a['shift_mm']:.1f} mm"
        if a.get("method") == "aliked_lightglue":
            text += (f" · ORB found only {a.get('orb_inliers', 0)} matches, so OpenCV 5's ALIKED + LightGlue "
                     "networks were used")
        if out.get("problems"):
            text += f" · <b>{esc(', '.join(out['problems']))}</b>"
        return text
    if tool == "ask_for_photo":
        return f"Asked for a <b>{esc(args.get('kind', ''))}</b>: “{esc(args.get('message', ''))}”"
    if tool == "save_reading":
        return f"“{esc(args.get('message', ''))}”"
    if tool == "request_engineer_review":
        return f"Reason: {esc(args.get('reason', ''))}"
    if tool == "fallback":
        return esc(out.get("reason", ""))
    if tool == "engineer_decision":
        decision = "Inspection needed" if args.get("decision") == "inspect" else "No action needed"
        return f"<b>{decision}</b>" + (f" · “{esc(out.get('note'))}”" if out.get("note") else "")
    return esc(json.dumps(out)[:300])


# --- Width-over-time chart (inline SVG, one crack) ----------------------------------------------

def nice_step(span):
    for step in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0):
        if span / step <= 5:
            return step
    return 10.0


def width_chart(readings):
    """Line of max width over time with ± uncertainty bars and the alert level. Single series: no legend."""
    W, H, left, right, top, bottom = 460, 230, 40, 78, 16, 32
    times = [datetime.fromisoformat(r["time"]).timestamp() for r in readings]
    t0, t1 = min(times), max(times)
    span_t = (t1 - t0) or 1.0
    y_max = max([ALERT_WIDTH_MM * 1.15] + [r["max_width_mm"] + r["uncertainty_mm"] for r in readings]) * 1.08
    step = nice_step(y_max)
    y_top = step * -(-y_max // step)

    def x(t):
        return left + (W - left - right) * ((t - t0) / span_t if t1 > t0 else 0.5)

    def y(v):
        return top + (H - top - bottom) * (1 - v / y_top)

    parts = []
    v = 0.0
    while v <= y_top + 1e-9:                        # recessive hairline grid + y labels
        cls = "axis" if v == 0 else "grid"
        parts.append(f'<line class="{cls}" x1="{left}" x2="{W - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
        v = round(v + step, 6)
    ya = y(ALERT_WIDTH_MM)                          # the alert level is a real threshold, so it is dashed + labelled
    parts.append(f'<line class="alert" x1="{left}" x2="{W - right}" y1="{ya:.1f}" y2="{ya:.1f}"/>')
    # label at the left end, so it never collides with the latest value's label on the right
    parts.append(f'<text class="tick" x="{left + 4}" y="{ya - 6:.1f}">alert {ALERT_WIDTH_MM:g} mm</text>')

    pts = [(x(t), y(r["max_width_mm"]), r) for t, r in zip(times, readings)]
    for px, _, r in pts:                             # uncertainty bars
        parts.append(f'<line class="err" x1="{px:.1f}" x2="{px:.1f}" y1="{y(r["max_width_mm"] + r["uncertainty_mm"]):.1f}" '
                     f'y2="{y(max(0, r["max_width_mm"] - r["uncertainty_mm"])):.1f}"/>')
    if len(pts) > 1:
        parts.append('<polyline class="line" points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py, _ in pts) + '"/>')
    for px, py, _ in pts:
        parts.append(f'<circle class="dot" cx="{px:.1f}" cy="{py:.1f}" r="4"/>')
    lx, ly, lr = pts[-1]                             # direct label on the latest point only
    parts.append(f'<text class="end" x="{lx + 10:.1f}" y="{ly - 8:.1f}">{lr["max_width_mm"]:.2f} mm</text>')

    labels = sorted({0, len(pts) - 1, *range(0, len(pts), max(1, len(pts) // 4))})
    for i in labels:                                 # a few date labels, never crowded
        anchor = "middle" if 0 < i < len(pts) - 1 or len(pts) == 1 else ("start" if i == 0 else "end")
        label = datetime.fromtimestamp(times[i], DISPLAY_TZ).strftime("%d %b")
        parts.append(f'<text class="tick" x="{pts[i][0]:.1f}" y="{H - 8}" text-anchor="{anchor}">{label}</text>')

    hits = "".join(                                  # hover/focus targets bigger than the dots, with tooltips
        f'<span class="hit" tabindex="0" style="left:{px / W * 100:.2f}%;top:{py / H * 100:.2f}%">'
        f'<span class="tip">{esc(when(r["time"]))}<br><b>{r["max_width_mm"]:.2f} mm</b> ± {r["uncertainty_mm"]:.2f}'
        f' · {esc(READING_STATUS.get(r["status"], ("", "", r["status"]))[2])}</span></span>'
        for px, py, r in pts)
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Maximum crack width in mm over time">'
           + "".join(parts) + "</svg>")
    rows = "".join(f"<tr><td>{esc(when(r['time']))}</td><td>{r['max_width_mm']:.2f}</td><td>± {r['uncertainty_mm']:.2f}</td>"
                   f"<td>{r['length_mm']:.0f}</td><td>{esc(r['status'])}</td></tr>" for r in reversed(readings))
    table = ("<details><summary>Table view</summary><table><tr><th>Date</th><th>Max width (mm)</th>"
             f"<th>Uncertainty</th><th>Length (mm)</th><th>Status</th></tr>{rows}</table></details>")
    return f'<div class="chart">{svg}{hits}</div>{table}'


# --- Page layout ---------------------------------------------------------------------------------

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --track: #f0efec;
  --series-1: #2a78d6; --accent: #2a78d6;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --track: #262624;
    --series-1: #3987e5; --accent: #3987e5;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 20px 16px 56px; }
nav { display: flex; gap: 16px; font-size: 14px; margin-bottom: 18px; }
nav a { color: var(--ink-2); }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 0 0 12px; }
p { margin: 0 0 12px; }
.sub { color: var(--ink-2); }
.muted { color: var(--muted); font-size: 13px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 18px; margin: 14px 0; }
.message { font-size: 18px; }
.pill { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600; padding: 2px 10px 2px 3px;
        border-radius: 999px; background: var(--track); }
.icon { display: inline-grid; place-items: center; min-width: 18px; height: 18px; border-radius: 9px;
        font-size: 11px; font-weight: 700; color: #fff; padding: 0 4px; }
.icon.good { background: var(--good); } .icon.warning { background: var(--warning); color: #0b0b0b; }
.icon.serious { background: var(--serious); } .icon.critical { background: var(--critical); }
.photos { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: 10px; }
.photos figure { margin: 0; }
.photos img { width: 100%; border-radius: 8px; background: var(--track); display: block; }
figcaption { font-size: 13px; color: var(--ink-2); margin-top: 4px; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(150px, 100%), 1fr)); gap: 10px; }
.fact .label { font-size: 13px; color: var(--ink-2); }
.fact .value { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }
.fact .note { font-size: 12px; color: var(--muted); }
ol.trace { list-style: none; padding: 0; margin: 0; }
.trace li { display: grid; grid-template-columns: 26px 1fr; gap: 10px; padding: 10px 0; border-top: 1px solid var(--grid); }
.trace .n { color: var(--muted); font-variant-numeric: tabular-nums; text-align: right; }
.trace .what { font-weight: 600; }
.trace .detail { color: var(--ink-2); font-size: 14px; }
.trace .thought { color: var(--muted); font-size: 13px; font-style: italic; }
.trace .blocked .what { color: var(--critical); }
.who { font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 999px; margin-left: 6px;
       border: 1px solid var(--ring); color: var(--ink-2); vertical-align: 1px; }
.who.ai { color: var(--accent); border-color: var(--accent); }
details { font-size: 13px; color: var(--ink-2); margin-top: 6px; }
summary { cursor: pointer; }
pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; background: var(--track); padding: 8px; border-radius: 6px; }
button, .button { font: inherit; font-weight: 600; border: 0; border-radius: 10px; padding: 12px 18px;
                  background: var(--accent); color: #fff; cursor: pointer; text-decoration: none; display: inline-block; }
button.secondary { background: var(--track); color: var(--ink); border: 1px solid var(--ring); }
button:disabled { opacity: 0.6; cursor: wait; }
input[type=text], input[type=password], textarea { font: inherit; width: 100%; padding: 10px; border-radius: 8px;
       border: 1px solid var(--axis); background: var(--surface); color: var(--ink); }
.upload input[type=file] { display: none; }
#status { margin-top: 12px; color: var(--ink-2); min-height: 1.5em; }
table { border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; width: 100%; }
td, th { text-align: left; padding: 4px 12px 4px 0; border-bottom: 1px solid var(--grid); font-weight: 400; }
th { color: var(--muted); }
.chart { position: relative; max-width: 640px; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart .grid { stroke: var(--grid); stroke-width: 1; }
.chart .axis { stroke: var(--axis); stroke-width: 1; }
.chart .alert { stroke: var(--critical); stroke-width: 1; stroke-dasharray: 5 4; }
.chart .err { stroke: var(--series-1); stroke-width: 1.5; opacity: 0.45; }
.chart .line { fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .dot { fill: var(--series-1); stroke: var(--surface); stroke-width: 2; }
.chart .tick { fill: var(--muted); font-size: 14px; }
.chart .end { fill: var(--ink); font-size: 15px; font-weight: 600; }
.hit { position: absolute; width: 28px; height: 28px; margin: -14px 0 0 -14px; border-radius: 50%; outline: none; }
.hit:hover, .hit:focus { box-shadow: 0 0 0 2px var(--series-1) inset; }
.tip { display: none; position: absolute; bottom: calc(100% + 4px); left: 50%; transform: translateX(-50%);
       background: var(--ink); color: var(--surface); font-size: 12px; padding: 5px 8px; border-radius: 6px;
       white-space: nowrap; pointer-events: none; z-index: 2; }
.hit:hover .tip, .hit:focus .tip { display: block; }
.cracks { display: grid; gap: 14px; }
.crack-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.hero { font-size: 34px; font-weight: 600; font-variant-numeric: tabular-nums; }
ul.cases { list-style: none; padding: 0; margin: 0; }
ul.cases li { display: flex; justify-content: space-between; gap: 10px; padding: 8px 0; border-top: 1px solid var(--grid); flex-wrap: wrap; }
"""


def page(title, body, nav=True):
    links = '<nav><a href="/">Send a photo</a><a href="/cracks">All cracks</a></nav>' if nav else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{esc(title)}</title><style>{CSS}</style></head>'
            f'<body><main>{links}{body}</main></body></html>')


UPLOAD_JS = """
<script>
const caseId = document.body.dataset.case || "";
const statusBox = document.getElementById("status");
const input = document.getElementById("file");
const button = document.getElementById("pick");
button.addEventListener("click", () => input.click());
input.addEventListener("change", async () => {
  const file = input.files[0];
  if (!file) return;
  button.disabled = true;
  statusBox.textContent = "Preparing the photo…";
  let body = file;
  try {   // shrink to at most 2560 px: faster upload, less mobile data, fits the server limit
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, 2560 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    body = await new Promise(ok => canvas.toBlob(ok, "image/jpeg", 0.92));
  } catch (e) { /* old browser: send the original */ }
  statusBox.textContent = "Measuring the crack… (up to a minute)";
  try {
    const res = await fetch("/api/photo" + (caseId ? "?case=" + encodeURIComponent(caseId) : ""), {
      method: "POST", headers: { "Content-Type": "image/jpeg" }, body, credentials: "same-origin" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    location.href = "/case/" + data.case_id;
  } catch (e) {
    statusBox.textContent = "Sorry, that didn't work: " + e.message;
    button.disabled = false;
  }
});
</script>
"""


def upload_form(case_id="", label="Take or choose a photo"):
    return (f'<div class="upload"><input id="file" type="file" accept="image/*" capture="environment">'
            f'<button id="pick" type="button">{esc(label)}</button><div id="status"></div></div>'
            + UPLOAD_JS.replace("document.body.dataset.case", json.dumps(case_id)))


# --- The app -------------------------------------------------------------------------------------

class App:
    def __init__(self, store, notifier, make_llm=lambda: None, access_code="", model_id=None):
        self.store, self.notifier, self.make_llm = store, notifier, make_llm
        self.access_code, self.model_id = access_code, model_id

    # -- helpers --
    @staticmethod
    def reply(status, body, content_type="text/html; charset=utf-8", headers=None, binary=False):
        out = {"statusCode": status, "headers": {"Content-Type": content_type, "Cache-Control": "no-store",
                                                 "X-Content-Type-Options": "nosniff", **(headers or {})}}
        if binary:
            out.update(body=base64.b64encode(body).decode(), isBase64Encoded=True)
        else:
            out["body"] = body
        return out

    def json_reply(self, status, data):
        return self.reply(status, json.dumps(data), "application/json")

    def code_ok(self, request):
        if not self.access_code:
            return True
        given = request["query"].get("code") or request["headers"].get("x-access-code") or request["cookies"].get("code", "")
        return hmac.compare_digest(given.encode(), self.access_code.encode())

    def code_cookie(self, request):
        if self.access_code and request["query"].get("code") and self.code_ok(request):
            secure = "; Secure" if request["base_url"].startswith("https") else ""    # laptop test server is http
            return {"Set-Cookie": f"code={request['query']['code']}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Lax{secure}"}
        return {}

    def photo_src(self, photo_id, kind):
        return self.store.photo_url(photo_id, kind)

    @staticmethod
    def parse(event):
        headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        cookies = {}
        for part in (event.get("cookies") or []) + headers.get("cookie", "").split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                cookies[k] = v
        body = event.get("body") or ""
        body = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode() if isinstance(body, str) else body
        host = headers.get("x-forwarded-host") or headers.get("host", "")
        proto = headers.get("x-forwarded-proto", "https")
        return {"method": event.get("requestContext", {}).get("http", {}).get("method", "GET"),
                "path": event.get("rawPath", "/"), "query": event.get("queryStringParameters") or {},
                "headers": headers, "cookies": cookies, "body": body, "base_url": f"{proto}://{host}"}

    # -- router --
    def handle(self, event):
        request = self.parse(event)
        method, path = request["method"], request["path"].rstrip("/") or "/"
        try:
            if path == "/" and method == "GET":
                return self.home(request)
            if path == "/api/photo" and method == "POST":
                return self.upload(request)
            if path.startswith("/case/") and method == "GET":
                return self.case_page(path.split("/")[2])
            if path.startswith("/review/"):
                return self.review(request, path.split("/")[2])
            if path == "/cracks" and method == "GET":
                return self.cracks(request)
            if path.startswith("/photo/") and method == "GET":
                return self.photo(path.split("/"))
            if path == "/health":
                methods = ["orb"] + (["aliked_lightglue"] if deep_models() is not None else [])
                return self.reply(200, json.dumps({"ok": True, "opencv": cv2.__version__, "alignment": methods}),
                                  "application/json")
            return self.reply(404, page("Not found", "<h1>Page not found</h1>"))
        except Exception:   # never show a stack trace to the browser; logs have it
            import traceback
            traceback.print_exc()
            return self.reply(500, page("Error", "<h1>Something went wrong</h1><p>Please try again.</p>"))

    # -- pages --
    def home(self, request):
        if not self.code_ok(request):
            body = ('<h1>Crack monitor</h1><p class="sub">Enter the access code you were given.</p>'
                    '<form method="get" action="/" class="card"><input type="password" name="code" '
                    'placeholder="Access code" autocomplete="off"><p></p><button>Continue</button></form>')
            return self.reply(200, page("Crack monitor", body, nav=False))
        body = ("<h1>Check a wall crack</h1>"
                '<p class="sub">Stick a printed marker (1–6) next to the crack, then photograph the crack and the '
                "whole marker together, straight on, from about 20–30 cm.</p>"
                f'<div class="card">{upload_form()}</div>'
                '<p class="muted">The photo is measured with OpenCV. An AI agent then decides the next step: '
                "save it, ask for a better photo or a close-up, or ask a structural engineer to review it. "
                "Every step is shown on the result page.</p>")
        return self.reply(200, page("Check a wall crack", body), headers=self.code_cookie(request))

    def upload(self, request):
        if not self.code_ok(request):
            return self.json_reply(403, {"error": "access code missing or wrong - open the link you were given"})
        data = request["body"]
        if not data or len(data) > MAX_UPLOAD_BYTES:
            return self.json_reply(413, {"error": "photo missing or too large"})
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if sum(1 for c in self.store.list_cases(MAX_UPLOADS_PER_DAY * 2) if c and c["case_id"][1:9] == today) >= MAX_UPLOADS_PER_DAY:
            return self.json_reply(429, {"error": "daily limit reached - please try again tomorrow"})
        case_id = request["query"].get("case", "")
        case_id = case_id if ID_PATTERN.match(case_id) else None
        kwargs = {"model_id": self.model_id} if self.model_id else {}
        case = handle_photo(self.store, data, case_id, self.notifier, self.make_llm(),
                            base_url=request["base_url"], **kwargs)
        return self.json_reply(200, {"case_id": case["case_id"], "status": case["status"]})

    def load_case(self, case_id):
        return self.store.get_case(case_id) if ID_PATTERN.match(case_id or "") else None

    def case_body(self, case, engineer=False):
        status_html = pill(case["status"], STATUS)
        title = f"Crack #{case['marker_id']}" if case.get("marker_id") is not None else "Crack photo"
        parts = [f'<h1>{esc(title)}</h1><p class="muted">Case {esc(case["case_id"])} · started {esc(when(case["created"]))}</p>',
                 f'<div class="card"><p>{status_html}</p><p class="message">{esc(case.get("owner_message", ""))}</p>']
        if case["status"] == "waiting_for_photo" and not engineer:
            left = MAX_PHOTOS_PER_CASE - len(case["photos"])
            parts.append(upload_form(case["case_id"], "Send the requested photo")
                         + f'<p class="muted">{left} more photo{"s" if left != 1 else ""} possible in this case.</p>')
        parts.append("</div>")

        crack = next((s["output"] for s in reversed(case["trace"])
                      if s["tool"] == "measure_crack" and s["output"].get("crack_found")), None)
        if crack:
            history = next((s["output"] for s in reversed(case["trace"]) if s["tool"] == "compare_with_history"
                            and "error" not in s["output"]), {})
            change = (f'<div class="fact"><div class="label">Change since last reading</div><div class="value">'
                      f'{history["growth_mm"]:+.2f} mm</div><div class="note">± {history["growth_uncertainty_mm"]:.2f} mm'
                      f'</div></div>') if "growth_mm" in history else ""
            parts.append('<div class="card"><h2>Latest measurement</h2><div class="facts">'
                         f'<div class="fact"><div class="label">Max width</div><div class="value">{crack["max_width_mm"]:.2f} mm</div>'
                         f'<div class="note">± {crack["uncertainty_mm"]:.2f} mm</div></div>{change}'
                         f'<div class="fact"><div class="label">Length seen</div><div class="value">{crack["length_mm"]:.0f} mm</div></div>'
                         f'<div class="fact"><div class="label">BRE damage category</div><div class="value">{esc(crack["bre_category"])}</div>'
                         f'<div class="note">{esc(crack["bre_label"])}</div></div></div></div>')

        figures = []
        for i, p in enumerate(case["photos"], 1):
            asked = {"first": "first photo", "retake": "retake", "closeup": "close-up"}.get(p.get("asked_for"), "")
            figures.append(f'<figure><img loading="lazy" alt="Photo {i} with the measurement drawn in" '
                           f'src="{esc(self.photo_src(p["photo_id"], "marked"))}"><figcaption>Photo {i} · {esc(asked)} · '
                           f'{esc(when(p["time"]))} · <a href="{esc(self.photo_src(p["photo_id"], "photo"))}">original</a>'
                           f'</figcaption></figure>')
        lined_up = [s for s in case["trace"] if s["tool"] == "compare_with_history"
                    and (s["output"].get("alignment") or {}).get("inliers")]
        if lined_up:
            last = lined_up[-1]["photo_id"]
            figures.append(f'<figure><img loading="lazy" alt="Wall features matched with the last photo" '
                           f'src="{esc(self.photo_src(last, "matches"))}"><figcaption>Lined up with last time\'s photo '
                           f'(matching wall features, OpenCV ORB)</figcaption></figure>')
        parts.append(f'<div class="card"><h2>Photos</h2><div class="photos">{"".join(figures)}</div></div>')

        steps = []
        for s in case["trace"]:
            blocked = "error" in (s.get("output") or {})
            thought = f'<div class="thought">{esc(s["thought"])}</div>' if s.get("thought") else ""
            raw = esc(json.dumps({"input": s.get("input"), "output": s.get("output")}, indent=1))
            steps.append(f'<li class="{"blocked" if blocked else ""}"><span class="n">{s["step"]}</span><div>'
                         f'<span class="what">{esc(TOOL_NAMES.get(s["tool"], s["tool"]))}</span>{who(s["decided_by"])}'
                         f' <span class="muted">{esc(when(s["time"])[-5:])}</span>{thought}'
                         f'<div class="detail">{describe(s)}</div>'
                         f'<details><summary>Raw tool call</summary><pre>{raw}</pre></details></div></li>')
        parts.append('<div class="card"><h2>What the agent did</h2><p class="muted">Each step is an OpenCV tool call or '
                     "a decision. The label shows who decided: the AI model, the rules backup, or the engineer.</p>"
                     f'<ol class="trace">{"".join(steps)}</ol></div>')
        return "".join(parts)

    def case_page(self, case_id):
        case = self.load_case(case_id)
        if case is None:
            return self.reply(404, page("Not found", "<h1>Case not found</h1>"))
        return self.reply(200, page("Crack check", self.case_body(case)))

    def review(self, request, case_id):
        case = self.load_case(case_id)
        review = (case or {}).get("review") or {}
        if request["method"] == "POST":
            form = {k: v[0] for k, v in parse_qs(request["body"].decode("utf-8", "replace")).items()}
            done = record_engineer_decision(self.store, case_id, form.get("token", ""), form.get("decision", ""),
                                            form.get("note", ""))
            if done is None:
                return self.reply(403, page("Review", "<h1>This link is not valid or was already used</h1>", nav=False))
            return self.reply(200, page("Review saved", f'<h1>Thank you</h1><p>Your decision was saved and the owner '
                                        f'can see it on <a href="/case/{esc(case_id)}">the case page</a>.</p>', nav=False))
        token = request["query"].get("token", "")
        if not review or not hmac.compare_digest(token.encode(), review.get("token", "").encode()):
            return self.reply(403, page("Review", "<h1>This link is not valid</h1>", nav=False))
        if review.get("decision"):
            form = f'<div class="card"><p>Already decided: <b>{esc(review["decision"])}</b> on {esc(when(review["decided"]))}.</p></div>'
        else:   # a button press (POST) decides - e-mail link scanners only GET the page, so they can't approve
            form = ('<form method="post" class="card"><h2>Your decision</h2>'
                    f'<input type="hidden" name="token" value="{esc(token)}">'
                    '<p><textarea name="note" rows="3" placeholder="Note for the owner (optional)"></textarea></p>'
                    '<button name="decision" value="inspect">Inspection needed</button> '
                    '<button name="decision" value="not_needed" class="secondary">No action needed</button></form>')
        body = (f'<h1>Engineer review</h1><div class="card"><p><b>Why you were asked:</b> {esc(review.get("reason", ""))}</p></div>'
                f'{form}{self.case_body(case, engineer=True)}')
        return self.reply(200, page("Engineer review", body, nav=False))

    def cracks(self, request):
        if not self.code_ok(request):
            return self.reply(403, page("Crack monitor", '<h1>Access code needed</h1><p><a href="/">Enter it here</a>.</p>'))
        readings = self.store.readings()
        markers = sorted({r["marker_id"] for r in readings})
        cards = []
        for marker_id in markers:
            mine = [r for r in readings if r["marker_id"] == marker_id]
            latest = mine[-1]
            cards.append(f'<div class="card"><div class="crack-head"><h2>Crack #{esc(marker_id)} · maximum width</h2>'
                         f'{pill(latest["status"], READING_STATUS)}</div>'
                         f'<div class="hero">{latest["max_width_mm"]:.2f} mm</div>'
                         f'<p class="muted">± {latest["uncertainty_mm"]:.2f} mm · {len(mine)} reading{"s" if len(mine) != 1 else ""} · '
                         f'latest {esc(when(latest["time"]))} · <a href="/case/{esc(latest["case_id"])}">open case</a></p>'
                         f'{width_chart(mine)}</div>')
        if not cards:
            cards.append('<div class="card"><p>No readings yet. <a href="/">Send the first photo</a>.</p></div>')
        cases = [c for c in self.store.list_cases(15) if c]
        rows = "".join(f'<li><a href="/case/{esc(c["case_id"])}">{"Crack #" + str(c["marker_id"]) if c.get("marker_id") is not None else "Photo"}'
                       f' · {esc(when(c["created"]))}</a>{pill(c["status"], STATUS)}</li>' for c in cases)
        body = (f'<h1>All cracks</h1><p class="sub">Maximum width over time. Bars show the measurement uncertainty; '
                f'the dashed line is the alert level for an engineer review.</p><div class="cracks">{"".join(cards)}</div>'
                f'<div class="card"><h2>Recent cases</h2><ul class="cases">{rows or "<li>None yet</li>"}</ul></div>')
        return self.reply(200, page("All cracks", body), headers=self.code_cookie(request))

    def photo(self, parts):
        if len(parts) != 4 or not ID_PATTERN.match(parts[2]) or parts[3] not in PHOTO_KINDS:
            return self.reply(404, "not found", "text/plain")
        data = self.store.load_photo(parts[2], parts[3])
        if data is None:
            return self.reply(404, "not found", "text/plain")
        return self.reply(200, data, "image/jpeg", {"Cache-Control": "private, max-age=86400"}, binary=True)
