const $ = (sel) => document.querySelector(sel);
const img = $("#frame");
const marker = $("#marker");
let matchId = null;
let items = []; // [{frame, status, x, y}]
let idx = 0;
let peek = 0; // frames away from the target frame being shown
let message = { text: "", kind: "", until: 0 };

async function getJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

function say(text, kind) {
  message = { text, kind, until: Date.now() + 4000 };
  render();
}

async function loadMatches() {
  const matches = await getJson("/api/matches");
  const sel = $("#match");
  sel.innerHTML = "";
  for (const m of matches) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `#${m.id} ${m.name}`;
    sel.appendChild(opt);
  }
  sel.onchange = () => selectMatch(Number(sel.value));
  if (matches.length) await selectMatch(matches[0].id);
}

async function selectMatch(id) {
  matchId = id;
  $("#match").value = String(id);
  items = await getJson(`/api/matches/${id}/ball-test`);
  const todo = items.findIndex((i) => i.status === "todo");
  show(todo >= 0 ? todo : 0);
}

function show(i) {
  if (!items.length) return;
  idx = Math.max(0, Math.min(items.length - 1, i));
  peek = 0;
  loadImage();
}

function loadImage() {
  img.src = `/api/matches/${matchId}/frames/${items[idx].frame + peek}.jpg`;
  render();
}

function render() {
  const done = items.filter((i) => i.status === "ball" || i.status === "none").length;
  const skipped = items.filter((i) => i.status === "skip").length;
  $("#progress").textContent = `${done} labelled · ${skipped} skipped · ${items.length - done - skipped} to do`;
  const item = items[idx];
  if (!item) return;
  const what = {
    todo: "not labelled yet",
    ball: `ball at (${Math.round(item.x)}, ${Math.round(item.y)})`,
    none: "no ball visible",
    skip: "skipped (can't tell)",
  }[item.status];
  const peeking = peek ? ` · PEEKING ${peek > 0 ? "+" : ""}${peek} frames (Q/W to go back)` : "";
  $("#state").textContent = `Frame ${idx + 1} of ${items.length} (video frame ${item.frame}): ${what}${peeking}`;
  $("#status").className = `status ${peek ? "rec" : "idle"}`;
  const showMsg = message.text && Date.now() < message.until;
  $("#message").textContent = showMsg ? message.text : "";
  $("#message").className = `message ${showMsg ? message.kind : ""}`;
  placeMarker();
}

function placeMarker() {
  const item = items[idx];
  if (!item || item.status !== "ball" || peek || !img.naturalWidth) {
    marker.hidden = true;
    return;
  }
  marker.hidden = false;
  marker.style.left = `${(item.x / img.naturalWidth) * img.clientWidth}px`;
  marker.style.top = `${(item.y / img.naturalHeight) * img.clientHeight}px`;
}

async function save(status, x = 0, y = 0) {
  const item = items[idx];
  try {
    items[idx] = await getJson(`/api/matches/${matchId}/ball-test/${item.frame}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, x, y }),
    });
  } catch (err) {
    say(`Not saved (${err.message})`, "warn");
    return;
  }
  const next = items.findIndex((i, j) => j > idx && i.status === "todo");
  const any = next >= 0 ? next : items.findIndex((i) => i.status === "todo");
  if (any < 0) {
    say("All test frames are labelled. Thank you!", "ok");
    render();
    return;
  }
  say(`Saved frame ${idx + 1}`, "ok");
  show(any);
}

img.addEventListener("click", (e) => {
  if (peek) {
    say("You're peeking at another frame. Press Q/W to return to the target frame, then click.", "warn");
    return;
  }
  const rect = img.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * img.naturalWidth;
  const y = ((e.clientY - rect.top) / rect.height) * img.naturalHeight;
  save("ball", x, y);
});
img.addEventListener("load", () => {
  img.hidden = false; // hidden until the first frame arrives, so no broken-image box shows
  render();
});
window.addEventListener("resize", placeMarker);

function peekBy(delta) {
  peek += delta;
  loadImage();
}

function bind(id, fn) {
  $(id).onclick = (e) => {
    e.currentTarget.blur();
    fn();
  };
}
bind("#btn-prev", () => show(idx - 1));
bind("#btn-next", () => show(idx + 1));
bind("#btn-none", () => save("none"));
bind("#btn-skip", () => save("skip"));
bind("#btn-peek-back", () => peekBy(-2));
bind("#btn-peek-fwd", () => peekBy(2));

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLSelectElement || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  const actions = {
    arrowleft: () => show(idx - 1),
    arrowright: () => show(idx + 1),
    x: () => save("none"),
    k: () => save("skip"),
    q: () => peekBy(-2),
    w: () => peekBy(2),
  };
  const action = actions[e.key.toLowerCase()];
  if (!action) return;
  e.preventDefault();
  action();
});

setInterval(render, 500); // lets messages expire
loadMatches().catch((err) => say(String(err), "warn"));
