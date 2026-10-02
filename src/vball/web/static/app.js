const $ = (sel) => document.querySelector(sel);
const video = $("#video");
let matchId = null;
let rallies = [];
let current = -1;
let labels = [];
let pendingStart = null;

function fmt(t) {
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(1).padStart(4, "0")}`;
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

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
  video.src = `/media/${id}/work.mp4`;
  current = -1;
  loadLabels();
  renderRallies();
  renderLabels();
}

function renderRallies() {
  const list = $("#rallies");
  list.innerHTML = "";
  rallies.forEach((r, i) => {
    const li = document.createElement("li");
    li.textContent = `${fmt(r.start_s)} (${(r.end_s - r.start_s).toFixed(1)} s)`;
    if (i === current) li.className = "active";
    li.onclick = () => playRally(i);
    list.appendChild(li);
  });
  $("#rally-count").textContent = `(${rallies.length})`;
}

function playRally(i) {
  if (i < 0 || i >= rallies.length) return;
  current = i;
  video.currentTime = rallies[i].start_s;
  video.play();
  renderRallies();
}

video.addEventListener("timeupdate", () => {
  if (current < 0 || video.currentTime < rallies[current].end_s) return;
  if ($("#continuous").checked && current + 1 < rallies.length) {
    playRally(current + 1);
  } else {
    video.pause();
    current = -1;
    renderRallies();
  }
});

function labelKey() {
  return `vball-gt-${matchId}`;
}

function loadLabels() {
  try {
    labels = JSON.parse(localStorage.getItem(labelKey())) || [];
  } catch {
    labels = [];
  }
  pendingStart = null;
}

function saveLabels() {
  try {
    localStorage.setItem(labelKey(), JSON.stringify(labels));
  } catch {
    // storage unavailable (private window): labels live until the page is closed
  }
}

function renderLabels() {
  $("#label-count").textContent = `(${labels.length})`;
  $("#pending").textContent = pendingStart === null ? "" : `start marked at ${fmt(pendingStart)}; press E at rally end`;
}

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLSelectElement || e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === "n") {
    playRally(current + 1);
  } else if (key === "p") {
    playRally(Math.max(0, current - 1));
  } else if (key === "s") {
    pendingStart = video.currentTime;
    renderLabels();
  } else if (key === "e" && pendingStart !== null && video.currentTime > pendingStart) {
    labels.push([pendingStart, video.currentTime]);
    pendingStart = null;
    saveLabels();
    renderLabels();
  } else if (key === "u") {
    labels.pop();
    saveLabels();
    renderLabels();
  } else {
    return;
  }
  e.preventDefault();
});

$("#download").onclick = () => {
  const rows = [...labels].sort((a, b) => a[0] - b[0]).map(([s, t]) => `${s.toFixed(2)},${t.toFixed(2)}`);
  const blob = new Blob([["start_s,end_s", ...rows].join("\n") + "\n"], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `gt_match_${matchId}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};

$("#clear").onclick = () => {
  if (!confirm("Delete all ground-truth labels for this match?")) return;
  labels = [];
  saveLabels();
  renderLabels();
};

$("#speed").onchange = (e) => {
  video.defaultPlaybackRate = Number(e.target.value);
  video.playbackRate = Number(e.target.value);
};

loadMatches().catch((err) => document.body.prepend(String(err)));
