const $ = (sel) => document.querySelector(sel);
const video = $("#video");
let matchId = null;
let rallies = [];
let current = -1; // index of the detected rally being played, -1 = free playback
let labels = []; // [{start_s, end_s, approved}], sorted by start_s
let history = []; // previous label lists, for undo
let pendingStart = null; // start time of the rally being recorded
let message = { text: "", kind: "", until: 0 };
let saveChain = Promise.resolve();

function fmt(t) {
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(1).padStart(4, "0")}`;
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

// ---------- matches ----------

async function loadMatches() {
  const matches = await getJson("/api/matches");
  const sel = $("#match");
  sel.innerHTML = "";
  for (const m of matches) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `#${m.id} ${m.name} (${m.n_rallies} rallies)`;
    sel.appendChild(opt);
  }
  sel.onchange = () => selectMatch(Number(sel.value));
  if (matches.length) await selectMatch(matches[matches.length - 1].id);
}

async function selectMatch(id) {
  matchId = id;
  $("#match").value = String(id);
  rallies = await getJson(`/api/matches/${id}/rallies`);
  labels = sortLabels(await getJson(`/api/matches/${id}/labels`));
  history = [];
  pendingStart = null;
  current = -1;
  video.src = `/media/${id}/work.mp4`;
  loadBall(id);
  await migrateBrowserLabels();
  renderLists();
  renderLive();
}

// Labels made with the first version of this page lived only in the browser; move them to the server once.
async function migrateBrowserLabels() {
  const key = `vball-gt-${matchId}`;
  let local = [];
  try {
    local = JSON.parse(localStorage.getItem(key)) || [];
  } catch {
    return;
  }
  if (!local.length) return;
  if (!labels.length) {
    labels = sortLabels(local.map(([start_s, end_s]) => ({ start_s, end_s, approved: true })));
    if (!(await queueSave())) return;
    say(`Recovered ${labels.length} labels you made earlier`, "ok");
  }
  try {
    localStorage.removeItem(key);
  } catch {
    // storage unavailable; nothing to clean up
  }
}

// ---------- saving ----------

function queueSave() {
  const snapshot = JSON.stringify(labels);
  const id = matchId;
  saveChain = saveChain.then(async () => {
    setSaveState("Saving…", "");
    try {
      const res = await fetch(`/api/matches/${id}/labels`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: snapshot,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSaveState("Saved ✓", "ok");
      return true;
    } catch (err) {
      setSaveState(`Not saved (${err.message}). Keep this tab open and try again.`, "warn");
      return false;
    }
  });
  return saveChain;
}

function setSaveState(text, kind) {
  const el = $("#save-state");
  el.textContent = text;
  el.className = `save-state ${kind}`;
}

// ---------- label editing ----------

function sortLabels(list) {
  return [...list].sort((a, b) => a.start_s - b.start_s);
}

function say(text, kind) {
  message = { text, kind, until: Date.now() + 5000 };
  renderLive();
}

function commit(newLabels) {
  history.push(labels);
  labels = sortLabels(newLabels);
  queueSave();
  renderLists();
}

function numberOf(label) {
  return labels.indexOf(label) + 1;
}

function approvedCount() {
  return labels.filter((l) => l.approved).length;
}

// The label being looked at: the one around the playhead (with some slack before the start and after the end).
function labelAt(t) {
  let found = -1;
  labels.forEach((l, i) => {
    if (t >= l.start_s - 1.5 && t <= l.end_s + 2) found = i;
  });
  return found;
}

function markStart() {
  pendingStart = video.currentTime;
  current = -1; // stop jumping between detected rallies while recording
  say(`Start marked at ${fmt(pendingStart)}`, "rec");
}

function markEnd() {
  if (pendingStart === null) {
    say("No rally is being recorded. Press S (Mark start) at the serve toss first.", "warn");
    return;
  }
  const start = pendingStart;
  const end = video.currentTime;
  if (end <= start) {
    say("The end must be after the start. Play on a little and press E again.", "warn");
    return;
  }
  pendingStart = null;
  const label = { start_s: start, end_s: end, approved: true };
  commit([...labels, label]);
  say(`Saved rally #${numberOf(label)}: ${fmt(start)} → ${fmt(end)} (${(end - start).toFixed(1)} s)`, "ok");
}

function cancelStart() {
  if (pendingStart === null) return;
  pendingStart = null;
  say("Start mark discarded", "warn");
}

function undo() {
  if (!history.length) {
    say("Nothing to undo", "warn");
    return;
  }
  labels = history.pop();
  queueSave();
  renderLists();
  say(`Undone. ${labels.length} labels, ${approvedCount()} approved.`, "ok");
}

function replaceLabel(i, changes) {
  const updated = { ...labels[i], ...changes };
  commit(labels.map((l, j) => (j === i ? updated : l)));
  return updated;
}

function setEdge(i, edge) {
  const t = video.currentTime;
  const next = edge === "start" ? { start_s: t } : { end_s: t };
  const start = next.start_s ?? labels[i].start_s;
  const end = next.end_s ?? labels[i].end_s;
  if (end <= start) {
    say(`Can't set the ${edge} there: the end must be after the start.`, "warn");
    return;
  }
  const updated = replaceLabel(i, next);
  say(`Rally #${numberOf(updated)} ${edge} set to ${fmt(t)}`, "ok");
}

function toggleApproved(i) {
  const updated = replaceLabel(i, { approved: !labels[i].approved });
  say(`Rally #${numberOf(updated)} ${updated.approved ? "approved" : "marked to review"} · ${approvedCount()} / ${labels.length} approved`, "ok");
}

function approveAndNext() {
  const i = labelAt(video.currentTime);
  if (i < 0) {
    say("No label at the playhead. Play one with ▶ (or press S/E to add one), then approve.", "warn");
    return;
  }
  const approved = replaceLabel(i, { approved: true });
  const n = numberOf(approved);
  const nextIdx = labels.findIndex((l) => !l.approved && l.start_s > approved.start_s);
  const anyIdx = nextIdx >= 0 ? nextIdx : labels.findIndex((l) => !l.approved);
  if (anyIdx < 0) {
    say(`Approved #${n}. All ${labels.length} labels are approved.`, "ok");
    return;
  }
  const left = labels.length - approvedCount();
  say(`Approved #${n}. Reviewing #${anyIdx + 1} (${left} left to review)`, "ok");
  playLabel(anyIdx);
}

function deleteLabel(i) {
  const { start_s, end_s } = labels[i];
  commit(labels.filter((_, j) => j !== i));
  say(`Deleted rally #${i + 1} (${fmt(start_s)} → ${fmt(end_s)})`, "ok");
}

function playLabel(i) {
  current = -1;
  video.currentTime = Math.max(0, labels[i].start_s - 1);
  video.play();
}

function importDetected() {
  if (!rallies.length) {
    say("There are no detected rallies to copy", "warn");
    return;
  }
  if (labels.length && !confirm(`Replace your ${labels.length} labels with the ${rallies.length} detected rallies? (Undo can bring them back.)`)) return;
  commit(rallies.map((r) => ({ start_s: r.start_s, end_s: r.end_s, approved: false })));
  say(`Copied ${rallies.length} detected rallies as "to review". Fix each one if needed, then press A to approve it and jump to the next.`, "ok");
  playLabel(0);
}

// ---------- detected rallies playback ----------

function playRally(i) {
  if (i < 0 || i >= rallies.length) return;
  current = i;
  video.currentTime = rallies[i].start_s;
  video.play();
  renderLists();
}

// Next / previous are relative to the playhead, not to the last rally clicked.
function nextRally() {
  const t = video.currentTime;
  const i = rallies.findIndex((r) => r.start_s > t + 0.5);
  if (i < 0) {
    say(`No detected rallies after ${fmt(t)}`, "warn");
    return;
  }
  playRally(i);
}

function prevRally() {
  const t = video.currentTime;
  // more than 1.5 s into a rally restarts it; otherwise go to the one before
  const starts = rallies.map((r) => r.start_s);
  let i = -1;
  starts.forEach((s, j) => {
    if (s < t - 1.5) i = j;
  });
  playRally(Math.max(0, i));
}

video.addEventListener("timeupdate", () => {
  renderLive();
  if (pendingStart !== null || current < 0 || video.currentTime < rallies[current].end_s) return;
  if ($("#continuous").checked && current + 1 < rallies.length) {
    playRally(current + 1);
  } else {
    video.pause();
    current = -1;
    renderLists();
  }
});
// a manual seek outside the rally being played ends rally-by-rally playback
video.addEventListener("seeking", () => {
  if (current >= 0) {
    const r = rallies[current];
    if (video.currentTime < r.start_s - 0.5 || video.currentTime > r.end_s + 0.5) {
      current = -1;
      renderLists();
    }
  }
});
video.addEventListener("loadedmetadata", renderLists);
video.addEventListener("seeked", renderLive);

// ---------- rendering ----------

function smallButton(text, title, onClick, cls = "") {
  const b = document.createElement("button");
  b.className = `small ${cls}`;
  b.textContent = text;
  b.title = title;
  b.onclick = (e) => {
    e.stopPropagation();
    e.currentTarget.blur();
    onClick();
  };
  return b;
}

function segment(start, end, cls) {
  const d = video.duration;
  const el = document.createElement("div");
  el.className = `tl-seg ${cls}`;
  el.style.left = `${(start / d) * 100}%`;
  el.style.width = `${Math.max(0.15, ((end - start) / d) * 100)}%`;
  return el;
}

function renderLists() {
  const ralliesEl = $("#rallies");
  ralliesEl.innerHTML = "";
  rallies.forEach((r, i) => {
    const li = document.createElement("li");
    li.textContent = `${fmt(r.start_s)} (${(r.end_s - r.start_s).toFixed(1)} s)`;
    if (i === current) li.classList.add("active");
    li.onclick = () => playRally(i);
    ralliesEl.appendChild(li);
  });
  $("#rally-count").textContent = rallies.length;

  const labelsEl = $("#labels");
  labelsEl.innerHTML = "";
  labels.forEach((l, i) => {
    const li = document.createElement("li");
    li.classList.toggle("unapproved", !l.approved);
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = `#${i + 1}`;
    const when = document.createElement("span");
    when.className = "when";
    when.textContent = `${fmt(l.start_s)} → ${fmt(l.end_s)}`;
    const dur = document.createElement("span");
    dur.className = "dur";
    dur.textContent = `${(l.end_s - l.start_s).toFixed(1)} s`;
    li.append(
      num,
      when,
      dur,
      smallButton(
        l.approved ? "✓ approved" : "approve",
        l.approved ? "Approved. Click to mark it as needing review again" : "Mark this label as checked",
        () => toggleApproved(i),
        l.approved ? "approved" : "to-review",
      ),
      smallButton("▶", "Play this rally", () => playLabel(i)),
      smallButton("start = now", "Set the start to the current video time", () => setEdge(i, "start")),
      smallButton("end = now", "Set the end to the current video time", () => setEdge(i, "end")),
      smallButton("✕", "Delete this label", () => deleteLabel(i)),
    );
    labelsEl.appendChild(li);
  });
  $("#label-count").textContent = labels.length;
  $("#approved-count").textContent = `${approvedCount()} / ${labels.length} approved`;

  const d = video.duration;
  const detected = $("#tl-detected");
  const labelled = $("#tl-labels");
  detected.innerHTML = "";
  labelled.innerHTML = "";
  if (d > 0) {
    rallies.forEach((r, i) => detected.appendChild(segment(r.start_s, r.end_s, i === current ? "active" : "")));
    labels.forEach((l) => labelled.appendChild(segment(l.start_s, l.end_s, l.approved ? "" : "unapproved")));
  }
  renderLive();
}

function renderLive() {
  const t = video.currentTime;
  const d = video.duration;
  if (d > 0) {
    $("#tl-head").style.left = `${(t / d) * 100}%`;
    $("#tl-labels").querySelector(".pending")?.remove();
    if (pendingStart !== null && t > pendingStart) {
      $("#tl-labels").appendChild(segment(pendingStart, t, "pending"));
    }
  }

  const at = labelAt(t);
  document.querySelectorAll("#labels li").forEach((li, i) => li.classList.toggle("inside", i === at));

  const status = $("#status");
  const state = $("#state");
  if (pendingStart !== null) {
    status.className = "status rec";
    state.textContent = `● RECORDING rally from ${fmt(pendingStart)} (+${Math.max(0, t - pendingStart).toFixed(1)} s). Press E when the ball is dead.`;
  } else {
    status.className = "status idle";
    const toReview = labels.length - approvedCount();
    const here = at >= 0 ? ` · at label #${at + 1}${labels[at].approved ? " (approved)" : " (to review: A approves)"}` : "";
    state.textContent = `Not recording · ${labels.length} labelled, ${toReview} to review${here}`;
  }
  const msg = $("#message");
  const showMsg = message.text && Date.now() < message.until;
  msg.textContent = showMsg ? message.text : "";
  msg.className = `message ${showMsg ? message.kind : ""}`;
}
setInterval(renderLive, 250); // keeps the status fresh while paused and lets messages expire

// ---------- overlay ----------

const overlay = $("#overlay");
let ballData = null; // {fps, width, height, x: [...], y: [...]} in work-video pixels, null = not detected

async function loadBall(id) {
  ballData = null;
  try {
    ballData = await getJson(`/api/matches/${id}/ball`);
  } catch {
    ballData = null; // no ball track for this match yet
  }
  drawOverlay();
}

// Where the picture sits inside the <video> box (object-fit: contain adds letterbox bars).
function contentRect() {
  const w = video.clientWidth;
  const h = video.clientHeight;
  const vw = video.videoWidth || 16;
  const vh = video.videoHeight || 9;
  const scale = Math.min(w / vw, h / vh);
  return { x: (w - vw * scale) / 2, y: (h - vh * scale) / 2, scale };
}

function drawOverlay(mediaTime = video.currentTime) {
  const dpr = window.devicePixelRatio || 1;
  const w = video.clientWidth;
  const h = video.clientHeight;
  overlay.style.width = `${w}px`;
  overlay.style.height = `${h}px`;
  overlay.width = Math.round(w * dpr);
  overlay.height = Math.round(h * dpr);
  const ctx = overlay.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!$("#show-ball").checked || !ballData) return;
  const r = contentRect();
  const s = (r.scale * (video.videoWidth || ballData.width)) / ballData.width;
  const frame = Math.round(mediaTime * ballData.fps);
  for (let k = 8; k >= 0; k--) {
    const f = frame - k;
    const x = ballData.x[f];
    const y = ballData.y[f];
    if (f < 0 || x == null) continue;
    ctx.beginPath();
    ctx.arc(r.x + x * s, r.y + y * s, k === 0 ? 9 : 4, 0, 2 * Math.PI);
    if (k === 0) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#facc15";
      ctx.stroke();
    } else {
      ctx.fillStyle = `rgba(250, 204, 21, ${0.6 * (1 - k / 9)})`;
      ctx.fill();
    }
  }
}

// Redraw on every presented video frame. Re-armed on play in case a source change dropped the callback.
let frameCallbackPending = false;
function scheduleFrameDraw() {
  if (frameCallbackPending) return;
  frameCallbackPending = true;
  video.requestVideoFrameCallback((_now, meta) => {
    frameCallbackPending = false;
    drawOverlay(meta.mediaTime);
    scheduleFrameDraw();
  });
}
if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
  scheduleFrameDraw();
  video.addEventListener("play", scheduleFrameDraw);
} else {
  video.addEventListener("timeupdate", () => drawOverlay());
}
video.addEventListener("seeked", () => drawOverlay());
video.addEventListener("loadedmetadata", () => drawOverlay());
window.addEventListener("resize", () => drawOverlay());
$("#show-ball").onchange = () => drawOverlay();

// ---------- input ----------

function bind(id, fn) {
  $(id).onclick = (e) => {
    e.currentTarget.blur(); // keep Space for play/pause instead of re-clicking the button
    fn();
  };
}
bind("#btn-prev", prevRally);
bind("#btn-next", nextRally);
bind("#btn-start", markStart);
bind("#btn-end", markEnd);
bind("#btn-cancel", cancelStart);
bind("#btn-undo", undo);
bind("#btn-approve", approveAndNext);
bind("#btn-import", importDetected);

$("#timeline").onclick = (e) => {
  const rect = e.currentTarget.getBoundingClientRect();
  if (video.duration > 0) video.currentTime = ((e.clientX - rect.left) / rect.width) * video.duration;
};

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLSelectElement || e.target instanceof HTMLInputElement) return;
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  const actions = {
    n: nextRally,
    p: prevRally,
    s: markStart,
    e: markEnd,
    a: approveAndNext,
    u: undo,
    escape: cancelStart,
  };
  const action = actions[e.key.toLowerCase()];
  if (!action) return;
  e.preventDefault();
  action();
});

$("#speed").onchange = (e) => {
  video.defaultPlaybackRate = Number(e.target.value);
  video.playbackRate = Number(e.target.value);
};

loadMatches().catch((err) => say(String(err), "warn"));
