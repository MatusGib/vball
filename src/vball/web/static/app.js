const $ = (sel) => document.querySelector(sel);
const video = $("#video");
let matchId = null;
let matchFps = 30; // frames per second of the selected match
let matchesById = {};
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
    matchesById[m.id] = m;
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `#${m.id} ${m.name} (${m.n_rallies} rallies)`;
    sel.appendChild(opt);
  }
  sel.onchange = () => selectMatch(Number(sel.value));
  // deep links from the Stats page: /?match=2&t=91.3&show=speed
  const params = new URLSearchParams(location.search);
  for (const name of (params.get("show") || "").split(",").filter(Boolean)) {
    const box = $(`#show-${name}`);
    if (box) box.checked = true;
  }
  const wanted = Number(params.get("match"));
  const id = matchesById[wanted] ? wanted : matches.at(-1)?.id;
  if (id == null) return;
  const t = Number(params.get("t"));
  if (params.has("t") && t >= 0) video.addEventListener("loadedmetadata", () => (video.currentTime = t), { once: true });
  await selectMatch(id);
}

async function selectMatch(id) {
  matchId = id;
  matchFps = matchesById[id]?.fps || 30;
  $("#match").value = String(id);
  rallies = await getJson(`/api/matches/${id}/rallies`);
  labels = sortLabels(await getJson(`/api/matches/${id}/labels`));
  history = [];
  pendingStart = null;
  current = -1;
  video.src = `/media/${id}/work.mp4`;
  loadBall(id);
  loadCourt(id);
  loadFlights(id);
  playerChunks = new Map();
  playersMissing = false;
  calib = null;
  renderCalib();
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
      setSaveState("Saved", "ok");
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

let stampStart = null; // start_s of the label that was just approved: its row gets the green tape (renderLists)

function toggleApproved(i) {
  if (!labels[i].approved) stampStart = labels[i].start_s;
  const updated = replaceLabel(i, { approved: !labels[i].approved });
  say(`Rally #${numberOf(updated)} ${updated.approved ? "approved" : "marked to review"} · ${approvedCount()} / ${labels.length} approved`, "ok");
}

function approveAndNext() {
  const i = labelAt(video.currentTime);
  if (i < 0) {
    say("No label at the playhead. Play one from the list (or press S/E to add one), then approve.", "warn");
    return;
  }
  if (!labels[i].approved) stampStart = labels[i].start_s;
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

function icon(name) {
  return `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

function smallButton(text, title, onClick, cls = "", iconName = null) {
  const b = document.createElement("button");
  b.className = `small ${cls}`;
  if (iconName) {
    b.innerHTML = icon(iconName);
    if (text) b.append(text);
    b.setAttribute("aria-label", title);
  } else {
    b.textContent = text;
  }
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
    const actions = document.createElement("span");
    actions.className = "actions";
    actions.append(
      smallButton(
        l.approved ? "approved" : "approve",
        l.approved ? "Approved. Click to mark it as needing review again" : "Mark this label as checked",
        () => toggleApproved(i),
        l.approved ? "approved" : "to-review",
        l.approved ? "check" : null,
      ),
      smallButton("", "Play this rally", () => playLabel(i), "", "play"),
      smallButton("start = now", "Set the start to the current video time", () => setEdge(i, "start")),
      smallButton("end = now", "Set the end to the current video time", () => setEdge(i, "end")),
      smallButton("", "Delete this label", () => deleteLabel(i), "", "close"),
    );
    li.append(num, when, dur, actions);
    if (stampStart !== null && l.approved && Math.abs(l.start_s - stampStart) < 1e-6) li.classList.add("stamp");
    labelsEl.appendChild(li);
  });
  stampStart = null;
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
    state.textContent = `Recording rally from ${fmt(pendingStart)} (+${Math.max(0, t - pendingStart).toFixed(1)} s). Press E when the ball is dead.`;
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

let ballProblem = null; // why there is no ball data, shown when "Show ball" is ticked

async function loadBall(id) {
  ballData = null;
  ballProblem = null;
  try {
    const res = await fetch(`/api/matches/${id}/ball`);
    if (res.ok) {
      ballData = await res.json();
    } else {
      const detail = (await res.json().catch(() => ({}))).detail;
      ballProblem =
        detail === "no ball track"
          ? "This match has no ball track yet (run vball track on it)."
          : `The server can't provide ball data (HTTP ${res.status}). It is probably an older version: restart it with "uv run vball serve".`;
    }
  } catch (err) {
    ballProblem = `Couldn't load ball data: ${err.message}`;
  }
  if ($("#show-ball").checked && ballProblem) say(ballProblem, "warn");
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
  const r = contentRect();
  const frame = Math.round(mediaTime * matchFps);
  drawCourt(ctx, frame, r);
  drawPlayers(ctx, frame, r);
  drawSpeed(ctx, frame, r);
  if (!$("#show-ball").checked || !ballData) return;
  const s = (r.scale * (video.videoWidth || ballData.width)) / ballData.width;
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
$("#show-ball").onchange = () => {
  if ($("#show-ball").checked && ballProblem) say(ballProblem, "warn");
  drawOverlay();
};

// ---------- 3D ball flights: speed overlay ----------

let flights = null; // GET /flights response: {fps, flights: [{start, end, uv, speed_kmh, z_m, ...}]}
let flightsProblem = null; // why there are no flights, shown when "Show speed" is ticked

async function loadFlights(id) {
  flights = null;
  flightsProblem = null;
  try {
    const res = await fetch(`/api/matches/${id}/flights`);
    if (id !== matchId) return; // another match was picked while this one loaded
    if (res.ok) {
      flights = await res.json();
    } else {
      const detail = (await res.json().catch(() => ({}))).detail;
      flightsProblem =
        detail === "no 3D flights"
          ? `No 3D ball flights for this match yet: run "uv run vball ball3d ${id}".`
          : detail === "court not calibrated"
            ? "Calibrate the court first, then run vball ball3d on this match."
            : detail && res.status === 404 && detail !== "Not Found"
              ? detail
              : `The server can't provide 3D flights (HTTP ${res.status}); restart it with "uv run vball serve".`;
    }
  } catch (err) {
    flightsProblem = `Couldn't load 3D flights: ${err.message}`;
  }
  if ($("#show-speed").checked && flightsProblem) say(flightsProblem, "warn");
  drawOverlay();
}

function flightAt(frame) {
  return flights ? flights.flights.find((f) => frame >= f.start && frame < f.end) : null;
}

function drawSpeed(ctx, frame, r) {
  if (!$("#show-speed").checked) return;
  const f = flightAt(frame);
  if (!f) return;
  const px = ([x, y]) => [r.x + x * r.scale, r.y + y * r.scale];
  ctx.strokeStyle = "rgba(250, 204, 21, 0.75)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  f.uv.forEach((p, j) => (j ? ctx.lineTo(...px(p)) : ctx.moveTo(...px(p))));
  ctx.stroke();
  ctx.setLineDash([]);
  const i = frame - f.start;
  const [bx, by] = px(f.uv[i]);
  ctx.fillStyle = "#facc15";
  ctx.beginPath();
  ctx.arc(bx, by, 5, 0, 2 * Math.PI);
  ctx.fill();
  const label = `${Math.round(f.speed_kmh[i])} km/h · ${f.z_m[i].toFixed(1)} m`;
  ctx.font = "bold 15px system-ui";
  const w = ctx.measureText(label).width + 12;
  const lx = Math.min(bx + 14, overlay.clientWidth - w - 4);
  const ly = Math.max(by - 34, 4);
  ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
  ctx.fillRect(lx, ly, w, 22);
  ctx.fillStyle = "#facc15";
  ctx.fillText(label, lx + 6, ly + 16);
}

$("#show-speed").onchange = () => {
  if ($("#show-speed").checked && flightsProblem) say(flightsProblem, "warn");
  drawOverlay();
};

// ---------- court calibration ----------

const LANDMARK_ORDER = [
  "far_left_corner", "far_right_corner", "far_attack_left", "far_attack_right",
  "center_left", "center_right", "near_attack_left", "near_attack_right",
  "near_left_corner", "near_right_corner", "net_left_top", "net_right_top",
];
let court = null; // GET /court response
let calib = null; // {frame, index, points: [{landmark, x, y}]} while calibrating

async function loadCourt(id) {
  try {
    court = await getJson(`/api/matches/${id}/court`);
  } catch {
    court = null; // not calibrated yet
  }
  if (court?.copied_from && id === matchId)
    say(`This set's court is a draft copied from set #${court.copied_from}: open Calibrate court to check and save it.`, "warn");
  drawOverlay();
}

function project(H, [x, y]) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

function courtSegmentAt(frame) {
  return court.segments.find((s) => frame >= s.start_frame && frame < s.end_frame) || court.segments.at(-1);
}

function drawCourt(ctx, frame, r) {
  if (calib) {
    ctx.fillStyle = "#22d3ee";
    for (const p of calib.points) {
      ctx.beginPath();
      ctx.arc(r.x + p.x * r.scale, r.y + p.y * r.scale, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  if (!$("#show-court").checked || !court) return;
  const seg = courtSegmentAt(frame);
  const H = seg.court_to_image;
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(34, 211, 238, 0.9)";
  for (const [a, b] of court.lines) {
    const [ax, ay] = project(H, a);
    const [bx, by] = project(H, b);
    ctx.beginPath();
    ctx.moveTo(r.x + ax * r.scale, r.y + ay * r.scale);
    ctx.lineTo(r.x + bx * r.scale, r.y + by * r.scale);
    ctx.stroke();
  }
  // net: a post from each centre-line end up to the clicked tape top, and the tape between them
  const tops = (seg.net_image || []).map((top, i) => top && [project(H, [i * 9, 9]), top]).filter(Boolean);
  ctx.strokeStyle = "rgba(34, 211, 238, 0.95)"; // the net belongs to the court layer (cyan), never the ball's yellow
  for (const [[fx, fy], [tx, ty]] of tops) {
    ctx.beginPath();
    ctx.moveTo(r.x + fx * r.scale, r.y + fy * r.scale);
    ctx.lineTo(r.x + tx * r.scale, r.y + ty * r.scale);
    ctx.stroke();
  }
  if (tops.length === 2) {
    ctx.beginPath();
    ctx.moveTo(r.x + tops[0][1][0] * r.scale, r.y + tops[0][1][1] * r.scale);
    ctx.lineTo(r.x + tops[1][1][0] * r.scale, r.y + tops[1][1][1] * r.scale);
    ctx.stroke();
  }
}

// magnifier for precise clicks while calibrating
const loupe = $("#loupe");
const LOUPE_PX = 180; // on-screen size, CSS px
let loupeSpan = 60; // video pixels shown across the magnifier; the mouse wheel changes it

function videoPoint(e) {
  const rect = overlay.getBoundingClientRect();
  const r = contentRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  return { x: (cx - r.x) / r.scale, y: (cy - r.y) / r.scale, cx, cy };
}

function drawLoupe(p) {
  const dpr = window.devicePixelRatio || 1;
  loupe.width = loupe.height = Math.round(LOUPE_PX * dpr);
  const ctx = loupe.getContext("2d");
  const k = loupe.width / loupeSpan; // canvas px per video px
  const toLoupe = (q) => [(q.x - p.x + loupeSpan / 2) * k, (q.y - p.y + loupeSpan / 2) * k];
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, loupe.width, loupe.height);
  ctx.drawImage(video, p.x - loupeSpan / 2, p.y - loupeSpan / 2, loupeSpan, loupeSpan, 0, 0, loupe.width, loupe.height);
  ctx.fillStyle = "#22d3ee";
  for (const q of calib.points) {
    const [x, y] = toLoupe(q);
    ctx.beginPath();
    ctx.arc(x, y, 4 * dpr, 0, 2 * Math.PI);
    ctx.fill();
  }
  const c = loupe.width / 2;
  const gap = 4 * dpr;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
  ctx.lineWidth = dpr;
  ctx.beginPath();
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    ctx.moveTo(c + dx * gap, c + dy * gap);
    ctx.lineTo(c + dx * c, c + dy * c);
  }
  ctx.stroke();
  const next = LANDMARK_ORDER[calib.index];
  ctx.font = `${12 * dpr}px system-ui`;
  ctx.textAlign = "center";
  ctx.fillStyle = "#fff";
  ctx.fillText(next ? next.replaceAll("_", " ") : "all done", c, loupe.height - 22 * dpr);
  ctx.fillText(`${(LOUPE_PX / loupeSpan / contentRect().scale).toFixed(1)}x`, c, 22 * dpr);
  const wrap = overlay.getBoundingClientRect();
  let left = p.cx + 24;
  let top = p.cy - LOUPE_PX - 24;
  if (left + LOUPE_PX > wrap.width) left = p.cx - LOUPE_PX - 24;
  if (top < 0) top = p.cy + 24;
  loupe.style.left = `${left}px`;
  loupe.style.top = `${top}px`;
  loupe.hidden = false;
}

overlay.addEventListener("mousemove", (e) => {
  if (calib) drawLoupe(videoPoint(e));
});
overlay.addEventListener("mouseleave", () => {
  loupe.hidden = true;
});
overlay.addEventListener(
  "wheel",
  (e) => {
    if (!calib) return;
    e.preventDefault();
    loupeSpan = Math.min(240, Math.max(15, loupeSpan * (e.deltaY > 0 ? 1.25 : 0.8)));
    drawLoupe(videoPoint(e));
  },
  { passive: false },
);

function renderCalib() {
  $("#calib-panel").hidden = !calib;
  if (!calib) loupe.hidden = true;
  overlay.style.pointerEvents = calib ? "auto" : "none";
  overlay.style.cursor = calib ? "crosshair" : "";
  if (!calib) {
    drawOverlay();
    return;
  }
  const list = $("#calib-list");
  list.innerHTML = "";
  LANDMARK_ORDER.forEach((name, i) => {
    const li = document.createElement("li");
    const done = calib.points.find((p) => p.landmark === name);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "calib-pick";
    btn.textContent = name.replaceAll("_", " ");
    if (done) li.classList.add("done");
    btn.title = done ? "Click, then click the video to move this point" : "Click, then click this point on the video";
    btn.onclick = () => {
      calib.index = i;
      renderCalib();
    };
    if (i === calib.index) li.classList.add("active");
    li.appendChild(btn);
    list.appendChild(li);
  });
  $("#calib-draft").hidden = !(court && court.copied_from && calib.frame === court.ref_frame);
  if (court && court.copied_from)
    $("#calib-draft").textContent = `Draft copied from set #${court.copied_from}: check every point with the magnifier. Click a landmark in the list to move it, then Save.`;
  $("#calib-save").disabled = calib.points.filter((p) => !p.landmark.startsWith("net_")).length < 4;
  drawOverlay();
}

function nextMissing(from) {
  let i = from;
  while (i < LANDMARK_ORDER.length && calib.points.some((p) => p.landmark === LANDMARK_ORDER[i])) i += 1;
  return i;
}

function startCalibration() {
  video.pause();
  if (court) {
    // re-open the saved calibration at its frame so only missing points (e.g. the net) need clicking
    video.currentTime = court.ref_frame / matchFps;
    const saved = [...court.points, ...(court.net_points || [])].map(({ landmark, x, y }) => ({ landmark, x, y }));
    calib = { frame: court.ref_frame, index: 0, points: saved };
  } else {
    calib = { frame: Math.round(video.currentTime * matchFps), index: 0, points: [] };
  }
  calib.index = nextMissing(0);
  $("#calib-result").textContent = "";
  renderCalib();
}

overlay.addEventListener("click", (e) => {
  if (!calib || calib.index >= LANDMARK_ORDER.length) return;
  const p = videoPoint(e);
  const name = LANDMARK_ORDER[calib.index];
  calib.points = calib.points.filter((q) => q.landmark !== name); // re-clicking a landmark moves it
  calib.points.push({ landmark: name, x: p.x, y: p.y });
  calib.index = nextMissing(calib.index + 1);
  renderCalib();
  drawLoupe(p);
});

bind("#btn-calibrate", startCalibration);
bind("#calib-skip", () => {
  if (calib && calib.index < LANDMARK_ORDER.length) calib.index = nextMissing(calib.index + 1);
  renderCalib();
});
bind("#calib-undo", () => {
  if (!calib) return;
  const last = calib.points.pop();
  calib.index = last ? LANDMARK_ORDER.indexOf(last.landmark) : 0;
  renderCalib();
});
bind("#calib-cancel", () => {
  calib = null;
  renderCalib();
});
bind("#calib-save", async () => {
  $("#calib-result").textContent = "Saving… (checking the whole video for camera moves, about a minute)";
  try {
    const res = await fetch(`/api/matches/${matchId}/court`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ref_frame: calib.frame, points: calib.points }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    court = await res.json();
    calib = null;
    $("#show-court").checked = true;
    renderCalib();
    $("#calib-result").textContent = "";
    say(`Court saved: mean error ${court.mean_error_px} px, ${court.segments.length} camera segment(s)`, "ok");
  } catch (err) {
    $("#calib-result").textContent = `Not saved: ${err.message}`;
  }
});
$("#show-court").onchange = () => drawOverlay();

// ---------- players ----------

const CHUNK = 300; // frames per players request
const SIDE_COLORS = ["#60a5fa", "#f87171"]; // near, far
let playerChunks = new Map(); // chunk start frame -> rows (null while loading)
let playersMissing = false;

function requestChunk(start) {
  if (start < 0 || playersMissing || playerChunks.has(start)) return;
  playerChunks.set(start, null);
  fetch(`/api/matches/${matchId}/players?start=${start}&end=${start + CHUNK}`)
    .then(async (res) => {
      if (!res.ok) {
        const detail = (await res.json().catch(() => ({}))).detail;
        playersMissing = true;
        say(
          detail === "no player tracks"
            ? "This match has no player tracks yet (run vball players on it)."
            : `The server can't provide player data (HTTP ${res.status}); restart it with "uv run vball serve".`,
          "warn",
        );
        playerChunks.set(start, []);
        return;
      }
      playerChunks.set(start, await res.json());
      drawOverlay();
    })
    .catch(() => playerChunks.set(start, []));
}

function playersAt(frame) {
  const start = Math.floor(frame / CHUNK) * CHUNK;
  requestChunk(start);
  requestChunk(start + CHUNK); // prefetch while playing
  return (playerChunks.get(start) || []).filter((p) => p.frame === frame);
}

function drawPlayers(ctx, frame, r) {
  const mini = $("#minimap");
  const show = $("#show-players").checked;
  mini.hidden = !(show && court);
  if (!show) return;
  const rows = playersAt(frame);
  ctx.lineWidth = 2;
  ctx.font = "12px system-ui";
  for (const p of rows) {
    const [x1, y1, x2, y2] = p.box;
    ctx.strokeStyle = p.side === null ? "rgba(200, 200, 200, 0.5)" : SIDE_COLORS[p.side];
    ctx.strokeRect(r.x + x1 * r.scale, r.y + y1 * r.scale, (x2 - x1) * r.scale, (y2 - y1) * r.scale);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(String(p.id), r.x + x1 * r.scale, r.y + y1 * r.scale - 3);
  }
  if (!court) return;
  // top-down map: court x -1..10 m across, y -2..20 m along (far side at the top)
  const m = mini.getContext("2d");
  const sx = mini.width / 11;
  const sy = mini.height / 22;
  const px = (x) => (x + 1) * sx;
  const py = (y) => (20 - y) * sy;
  m.clearRect(0, 0, mini.width, mini.height);
  m.strokeStyle = "#e5e7eb";
  m.lineWidth = 1;
  m.strokeRect(px(0), py(18), 9 * sx, 18 * sy);
  m.beginPath();
  m.moveTo(px(0), py(9));
  m.lineTo(px(9), py(9));
  m.stroke();
  for (const p of rows) {
    if (!p.court) continue;
    m.fillStyle = SIDE_COLORS[p.side];
    m.beginPath();
    m.arc(px(p.court[0]), py(p.court[1]), 4, 0, 2 * Math.PI);
    m.fill();
  }
}

$("#show-players").onchange = () => drawOverlay();

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
