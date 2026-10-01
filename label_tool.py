"""A local, blind labeling tool for inbox.json.

Serves a single page at http://127.0.0.1:<port>/ where you go through each email and pick the
disposition you'd actually choose. No model predictions are shown anywhere on the page -- the
whole point is an independent judgement to score the backends against, so keeping it blind matters.

Every click is written straight to labels.json (an atomic write), so you can quit and resume later;
progress isn't lost. Run `python llm_classifier.py --backend all ...` again afterwards and the report
picks labels.json up automatically to add an "Accuracy against your labels" section.

Local only (binds 127.0.0.1). Stdlib only, no extra dependency.
"""
import argparse
import json
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm_classifier import DISPOSITIONS, load_inbox

LABELS_PATH = Path(__file__).parent / "labels.json"
VALID_EXPECTED = set(DISPOSITIONS) | {"unsure"}

MESSAGES = load_inbox()
MESSAGE_IDS = {m["id"] for m in MESSAGES}
LOCK = threading.Lock()


def load_labels():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


def save_labels(labels):
    tmp = LABELS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
    tmp.replace(LABELS_PATH)  # atomic on the same filesystem


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label the inbox</title>
<style>
:root { --bg:#fff; --fg:#1c1e21; --muted:#6b7280; --line:#e5e7eb; --card:#f7f8fa; --accent:#2563eb; --bad:#dc2626; --good:#16a34a; --warn:#d97706; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14161a; --fg:#e8eaed; --muted:#9aa1ab; --line:#2a2e35; --card:#1c1f25; --accent:#6ea0ff; --bad:#f87171; --good:#4ade80; --warn:#fbbf24; }
}
* { box-sizing:border-box; }
body { margin:0; padding:20px 16px 40px; background:var(--bg); color:var(--fg);
       font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:760px; margin:0 auto; }
h1 { font-size:20px; margin:0 0 4px; }
.sub { color:var(--muted); font-size:13px; margin:0 0 16px; }
.progress { font-size:13px; color:var(--muted); margin-bottom:8px; }
.grid { display:grid; grid-template-columns:repeat(20,1fr); gap:3px; margin-bottom:18px; }
.grid button { aspect-ratio:1; border:1px solid var(--line); border-radius:3px; background:var(--card);
               cursor:pointer; padding:0; font-size:0; }
.grid button.labeled { background:var(--good); border-color:var(--good); }
.grid button.unsure { background:var(--warn); border-color:var(--warn); }
.grid button.current { outline:2px solid var(--accent); outline-offset:1px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
.card .meta { font-size:13px; color:var(--muted); margin-bottom:2px; }
.card .subject { font-size:17px; font-weight:600; margin:6px 0 10px; }
.card .body { white-space:pre-wrap; font-size:14px; border-top:1px solid var(--line); padding-top:10px; margin-top:2px; }
.choices { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:18px; }
@media (max-width:520px) { .choices { grid-template-columns:1fr; } }
.choice { text-align:left; border:1px solid var(--line); border-radius:10px; padding:10px 12px; background:var(--card);
          color:var(--fg); cursor:pointer; font:inherit; }
.choice:hover { border-color:var(--accent); }
.choice.selected { border-color:var(--accent); border-width:2px; }
.choice .k { font-weight:600; }
.choice .kbd { display:inline-block; min-width:16px; padding:0 4px; border:1px solid var(--line); border-radius:4px;
               font-size:11px; color:var(--muted); margin-right:6px; }
.choice .d { color:var(--muted); font-size:12px; margin-top:2px; }
.choice.unsure { grid-column:1 / -1; }
.nav { display:flex; justify-content:space-between; align-items:center; margin-top:16px; }
.nav button { border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:8px;
              padding:8px 14px; cursor:pointer; font:inherit; }
.nav button:disabled { opacity:.4; cursor:default; }
.clear { color:var(--bad); background:none; border:none; font:inherit; cursor:pointer; text-decoration:underline; padding:0; }
.done { background:var(--card); border:1px solid var(--good); border-radius:10px; padding:12px 14px; margin-bottom:14px; color:var(--good); }
.hint { color:var(--muted); font-size:12px; margin-top:10px; }
</style>
</head>
<body>
<main>
<h1>Label the inbox</h1>
<p class="sub">Pick the disposition <em>you'd</em> choose. Model predictions are not shown here on purpose &mdash;
this is meant to be an independent judgement. Every click saves immediately to <code>labels.json</code>.</p>
<div id="done"></div>
<div class="progress" id="progress"></div>
<div class="grid" id="grid"></div>
<div class="card">
  <div class="meta" id="meta"></div>
  <div class="subject" id="subject"></div>
  <div class="body" id="body"></div>
</div>
<div class="choices" id="choices"></div>
<div class="nav">
  <button id="prev">&larr; Prev</button>
  <button class="clear" id="clearBtn">Clear label</button>
  <button id="next">Next &rarr;</button>
</div>
<p class="hint">Keys: 1 reply &middot; 2 archive &middot; 3 defer &middot; 4 delegate &middot; 5 escalate &middot; 0 unsure &middot; &larr;/&rarr; move</p>
</main>
<script>
let messages = [], labels = {}, dispositions = {}, order = ["reply","archive","defer","delegate","escalate"];
let i = 0;

async function api(path, opts) {
  const res = await fetch(path, opts);
  return res.json();
}

function firstUnlabeled() {
  for (let k = 0; k < messages.length; k++) if (!labels[messages[k].id]) return k;
  return 0;
}

function render() {
  const total = messages.length;
  const labeledCount = Object.values(labels).filter(l => l.expected !== "unsure").length;
  const unsureCount = Object.values(labels).filter(l => l.expected === "unsure").length;
  const remaining = total - labeledCount - unsureCount;
  document.getElementById("done").innerHTML = remaining === 0
    ? `<div class="done">All ${total} reviewed. Re-run <code>python llm_classifier.py --backend all --workers 10 --delay 0</code> to see accuracy in report.html.</div>` : "";
  document.getElementById("progress").textContent =
    `${labeledCount} labeled · ${unsureCount} unsure · ${remaining} remaining / ${total}`;

  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  messages.forEach((m, idx) => {
    const b = document.createElement("button");
    const l = labels[m.id];
    if (l && l.expected === "unsure") b.className = "unsure";
    else if (l) b.className = "labeled";
    if (idx === i) b.className += " current";
    b.title = `${idx + 1}. ${m.subject}`;
    b.onclick = () => { i = idx; render(); };
    grid.appendChild(b);
  });

  const m = messages[i];
  document.getElementById("meta").textContent =
    `${i + 1}/${messages.length} · ${m.id} · from ${m.from} · to ${m.to} · ${m.timestamp}`;
  document.getElementById("subject").textContent = m.subject;
  document.getElementById("body").textContent = m.body;

  const current = labels[m.id]?.expected;
  const choices = document.getElementById("choices");
  choices.innerHTML = "";
  order.forEach((name, idx2) => {
    const btn = document.createElement("button");
    btn.className = "choice" + (current === name ? " selected" : "");
    btn.innerHTML = `<span class="kbd">${idx2 + 1}</span><span class="k">${name}</span>` +
                     `<div class="d">${dispositions[name] || ""}</div>`;
    btn.onclick = () => choose(name);
    choices.appendChild(btn);
  });
  const unsureBtn = document.createElement("button");
  unsureBtn.className = "choice unsure" + (current === "unsure" ? " selected" : "");
  unsureBtn.innerHTML = `<span class="kbd">0</span><span class="k">unsure</span>` +
                         `<div class="d">Genuinely ambiguous — skip; excluded from the accuracy score.</div>`;
  unsureBtn.onclick = () => choose("unsure");
  choices.appendChild(unsureBtn);

  document.getElementById("prev").disabled = i === 0;
  document.getElementById("next").disabled = i === messages.length - 1;
}

async function choose(expected) {
  const m = messages[i];
  labels[m.id] = { expected, labeled_at: new Date().toISOString() };
  render();
  await api("/api/label", { method: "POST", headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({ id: m.id, expected }) });
  if (i < messages.length - 1) { i++; render(); }
}

async function clearCurrent() {
  const m = messages[i];
  delete labels[m.id];
  render();
  await api("/api/clear", { method: "POST", headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({ id: m.id }) });
}

document.getElementById("prev").onclick = () => { if (i > 0) { i--; render(); } };
document.getElementById("next").onclick = () => { if (i < messages.length - 1) { i++; render(); } };
document.getElementById("clearBtn").onclick = clearCurrent;

document.addEventListener("keydown", (e) => {
  if (["1","2","3","4","5"].includes(e.key)) choose(order[+e.key - 1]);
  else if (e.key === "0") choose("unsure");
  else if (e.key === "ArrowLeft" && i > 0) { i--; render(); }
  else if (e.key === "ArrowRight" && i < messages.length - 1) { i++; render(); }
});

api("/api/state").then(data => {
  messages = data.messages; labels = data.labels; dispositions = data.dispositions;
  i = firstUnlabeled();
  render();
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            with LOCK:
                labels = load_labels()
            self._json({"messages": MESSAGES, "labels": labels, "dispositions": DISPOSITIONS})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)

        if self.path == "/api/label":
            mid, expected = data.get("id"), data.get("expected")
            if mid not in MESSAGE_IDS or expected not in VALID_EXPECTED:
                return self._json({"error": "bad request"}, 400)
            with LOCK:
                labels = load_labels()
                labels[mid] = {"expected": expected, "labeled_at": datetime.now(timezone.utc).isoformat()}
                save_labels(labels)
            return self._json({"ok": True, "count": len(labels)})

        if self.path == "/api/clear":
            mid = data.get("id")
            with LOCK:
                labels = load_labels()
                labels.pop(mid, None)
                save_labels(labels)
            return self._json({"ok": True, "count": len(labels)})

        return self._json({"error": "not found"}, 404)

    def log_message(self, format, *args):
        pass  # quiet; this runs locally for one person


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8756)
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"labeling {len(MESSAGES)} messages -- open {url} (Ctrl+C to stop)")
    print(f"labels are saved to {LABELS_PATH.name} after every click; you can quit and resume anytime")
    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        labels = load_labels()
        done = sum(1 for l in labels.values() if l["expected"] != "unsure")
        unsure = sum(1 for l in labels.values() if l["expected"] == "unsure")
        print(f"\nstopped. {done} labeled, {unsure} unsure, {len(MESSAGES) - done - unsure} remaining "
              f"-- saved in {LABELS_PATH.name}")


if __name__ == "__main__":
    main()
