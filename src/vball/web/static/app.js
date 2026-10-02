const $ = (sel) => document.querySelector(sel);
const video = $("#video");
let matchId = null;
let rallies = [];
let current = -1; // index of the detected rally being played, -1 = free playback
let labels = []; // [[start_s, end_s], ...] sorted by start
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
    labels = sortLabels(local);
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
  return [...list].sort((a, b) => a[0] - b[0]);
}

function say(text, kind) {
  message = { text, kind, until: Date.now() + 4000 };
  renderLive();
}

function commit(newLabels, text) {
  history.push(labels);
  labels = sortLabels(newLabels);
  queueSave();
  renderLists();
  say(text, "ok");
}

function labelNumber(start) {
  return labels.findIndex((l) => l[0] === start) + 1;
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
  commit([...labels, [start, end]], "");
  say(`Saved rally #${labelNumber(start)}: ${fmt(start)} → ${fmt(end)} (${(end - start).toFixed(1)} s)`, "ok");
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
  say(`Undone. ${labels.length} labels.`, "ok");
}

function setEdge(i, edge) {
  const t = video.currentTime;
  const [start, end] = labels[i];
  const next = edge === "start" ? [t, end] : [start, t];
  if (next[1] <= next[0]) {
    say(`Can't set the ${edge} there: the end must be after the start.`, "warn");
    return;
  }
  const updated = labels.map((l, j) => (j === i ? next : l));
  commit(updated, "");
  say(`Rally #${labelNumber(next[0])} ${edge} set to ${fmt(t)}`, "ok");
}

function deleteLabel(i) {
  const [start, end] = labels[i];
  commit(labels.filter((_, j) => j !== i), `Deleted rally #${i + 1} (${fmt(start)} → ${fmt(end)})`);
}

function playLabel(i) {
  current = -1;
  video.currentTime = Math.max(0, labels[i][0] - 1);
  video.play();
}

function importDetected() {
  if (!rallies.length) {
    say("There are no detected rallies to copy", "warn");
    return;
  }
  if (labels.length && !confirm(`Replace your ${labels.length} labels with the ${rallies.length} detected rallies? (Undo can bring them back.)`)) return;
  commit(
    rallies.map((r) => [r.start_s, r.end_s]),
    `Copied ${rallies.length} detected rallies. Now fix the edges, delete false ones and add missed ones.`,
  );
}

// ---------- detected rallies playback ----------

function playRally(i) {
  if (i < 0 || i >= rallies.length) return;
  current = i;
  video.currentTime = rallies[i].start_s;
  video.play();
  renderLists();
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
video.addEventListener("loadedmetadata", renderLists);
video.addEventListener("seeked", renderLive);

// ---------- rendering ----------

function smallButton(text, title, onClick) {
  const b = document.createElement("button");
  b.className = "small";
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
  labels.forEach(([start, end], i) => {
    const li = document.createElement("li");
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = `#${i + 1}`;
    const when = document.createElement("span");
    when.className = "when";
    when.textContent = `${fmt(start)} → ${fmt(end)}`;
    const dur = document.createElement("span");
    dur.className = "dur";
    dur.textContent = `${(end - start).toFixed(1)} s`;
    li.append(
      num,
      when,
      dur,
      smallButton("▶", "Play this rally", () => playLabel(i)),
      smallButton("start = now", "Set the start to the current video time", () => setEdge(i, "start")),
      smallButton("end = now", "Set the end to the current video time", () => setEdge(i, "end")),
      smallButton("✕", "Delete this label", () => deleteLabel(i)),
    );
    labelsEl.appendChild(li);
  });
  $("#label-count").textContent = labels.length;

  const d = video.duration;
  const detected = $("#tl-detected");
  const labelled = $("#tl-labels");
  detected.innerHTML = "";
  labelled.innerHTML = "";
  if (d > 0) {
    rallies.forEach((r, i) => detected.appendChild(segment(r.start_s, r.end_s, i === current ? "active" : "")));
    labels.forEach(([s, e]) => labelled.appendChild(segment(s, e, "")));
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

  document.querySelectorAll("#labels li").forEach((li, i) => {
    li.classList.toggle("inside", labels[i] && t >= labels[i][0] && t <= labels[i][1]);
  });

  const status = $("#status");
  const state = $("#state");
  if (pendingStart !== null) {
    status.className = "status rec";
    state.textContent = `● RECORDING rally from ${fmt(pendingStart)} (+${Math.max(0, t - pendingStart).toFixed(1)} s). Press E when the ball is dead.`;
  } else {
    status.className = "status idle";
    state.textContent = `Not recording · ${labels.length} labelled. Press S at the serve toss.`;
  }
  const msg = $("#message");
  const showMsg = message.text && Date.now() < message.until;
  msg.textContent = showMsg ? message.text : "";
  msg.className = `message ${showMsg ? message.kind : ""}`;
}
setInterval(renderLive, 250); // keeps the status fresh while paused and lets messages expire

// ---------- input ----------

function bind(id, fn) {
  $(id).onclick = (e) => {
    e.currentTarget.blur(); // keep Space for play/pause instead of re-clicking the button
    fn();
  };
}
bind("#btn-prev", () => playRally(Math.max(0, current - 1)));
bind("#btn-next", () => playRally(current + 1));
bind("#btn-start", markStart);
bind("#btn-end", markEnd);
bind("#btn-cancel", cancelStart);
bind("#btn-undo", undo);
bind("#btn-import", importDetected);

$("#timeline").onclick = (e) => {
  const rect = e.currentTarget.getBoundingClientRect();
  if (video.duration > 0) video.currentTime = ((e.clientX - rect.left) / rect.width) * video.duration;
};

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLSelectElement || e.target instanceof HTMLInputElement) return;
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  const actions = {
    n: () => playRally(current + 1),
    p: () => playRally(Math.max(0, current - 1)),
    s: markStart,
    e: markEnd,
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
