// vball stats page: leaderboards and side-level player numbers. Rows open the viewer at that moment.
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

let all = null; // GET /api/stats
let detail = null; // GET /api/matches/{id}/stats for the picked match

async function getJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
  return res.json();
}

function sideLabel(side, ourSide) {
  if (ourSide) return side === ourSide ? "Us" : "Them";
  return side === "near" ? "Near" : "Far";
}

function viewerLink(matchId, timeS, show) {
  return `/?match=${matchId}&t=${Math.max(0, timeS - 1).toFixed(1)}${show ? `&show=${show}` : ""}`;
}

function picked() {
  const id = Number($("#pick").value);
  return id ? all.matches.find((m) => m.id === id) : null;
}

// the same ranking the server uses for all matches, applied to the picked one
function board(matches, kind, key, n = 10) {
  return matches
    .flatMap((m) =>
      ((m.flights || {})[kind] || [])
        .filter((x) => x.plausible)
        .map((x) => ({ ...x, match_id: m.id, match_name: m.name, our_side: m.our_side })),
    )
    .sort((a, b) => b[key] - a[key])
    .slice(0, n);
}

function longest(matches, n = 10) {
  return matches
    .flatMap((m) => (m.rallies.longest || []).map((r) => ({ ...r, match_id: m.id, match_name: m.name })))
    .sort((a, b) => b.length_s - a.length_s)
    .slice(0, n);
}

function tile(label, value, sub = "") {
  return `<div class="tile"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="sub">${esc(sub)}</div></div>`;
}

function renderTiles(matches) {
  const rallies = matches.reduce((n, m) => n + (m.rallies.count || 0), 0);
  const rallyTime = matches.reduce((n, m) => n + (m.rallies.total_s || 0), 0);
  const serves = matches.flatMap((m) => (m.flights || {}).serves || []);
  const ok = serves.filter((s) => s.plausible);
  const best = board(matches, "serves", "speed_kmh", 1)[0];
  const sources = [...new Set(matches.map((m) => m.rallies.source))].join(" + ");
  $("#tiles").innerHTML = [
    tile("Rallies", rallies, `from ${sources}`),
    tile("Average rally", rallies ? `${(rallyTime / rallies).toFixed(1)} s` : "–", `${(rallyTime / 60).toFixed(0)} min of play`),
    tile("Serves measured", serves.length ? `${ok.length} / ${serves.length}` : "–", "plausible 3D fits / found"),
    tile(
      "Hardest serve",
      best ? `${Math.round(best.speed_kmh)} km/h` : "–",
      best ? `${sideLabel(best.side, best.our_side)} · #${best.match_id} at ${mmss(best.time_s)}` : "needs 3D flights",
    ),
  ].join("");
}

function renderTable(el, rows, columns, showMatch) {
  if (!rows.length) {
    el.innerHTML = `<tbody><tr><td class="empty">Nothing measured yet.</td></tr></tbody>`;
    return;
  }
  const head = ["#", ...(showMatch ? ["Match"] : []), "Time", ...columns.map((c) => c.name)];
  el.innerHTML =
    `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
    rows
      .map((r, i) => {
        const href = viewerLink(r.match_id, r.time_s, r.show);
        const cells = [
          i + 1,
          ...(showMatch ? [`#${r.match_id} ${esc(r.match_name)}`] : []),
          `<a href="${href}" title="Open the viewer here">${mmss(r.time_s)}</a>`,
          ...columns.map((c) => c.value(r)),
        ];
        return `<tr data-href="${href}">${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
      })
      .join("") +
    "</tbody>";
}

function renderBoards(matches) {
  const showMatch = matches.length > 1;
  const side = { name: "Side", value: (r) => sideLabel(r.side, r.our_side) };
  const serves = board(matches, "serves", "speed_kmh").map((r) => ({ ...r, show: "speed" }));
  renderTable($("#board-serves"), serves, [
    { name: "Speed", value: (r) => `${Math.round(r.speed_kmh)} km/h` },
    { name: "Over the tape", value: (r) => `${(r.net_z_m - 2.43).toFixed(2)} m` },
    side,
  ], showMatch);
  const found = matches.flatMap((m) => (m.flights || {}).serves || []);
  const left = found.filter((s) => !s.plausible).length;
  $("#serve-note").textContent = left
    ? `${left} of ${found.length} serves not counted: their 3D fit was implausible (outside 40–100 km/h, under the tape or over 4.5 m, or landing off court).`
    : "";
  renderTable($("#board-sets"), board(matches, "sets", "apex_m").map((r) => ({ ...r, show: "speed" })), [
    { name: "Height", value: (r) => `${r.apex_m.toFixed(1)} m` },
    { name: "Speed", value: (r) => `${Math.round(r.speed_kmh)} km/h` },
    side,
  ], showMatch);
  renderTable($("#board-attacks"), board(matches, "attacks", "speed_kmh").map((r) => ({ ...r, show: "speed" })), [
    { name: "Speed", value: (r) => `${Math.round(r.speed_kmh)} km/h` },
    side,
  ], showMatch);
  renderTable($("#board-rallies"), longest(matches), [
    { name: "Length", value: (r) => `${r.length_s.toFixed(1)} s` },
  ], showMatch);
}

function renderMissing(matches) {
  const items = matches.flatMap((m) =>
    Object.entries(m.missing || {}).map(([what, how]) => `<li>#${m.id} ${esc(m.name)}: ${what}: ${esc(how)}</li>`),
  );
  const detected = matches.filter((m) => m.rallies.source === "detected").map((m) => `#${m.id}`);
  if (detected.length)
    items.push(`<li>Rallies for ${detected.join(", ")} are detected, not labelled: lengths may merge two rallies.</li>`);
  $("#missing").innerHTML = items.join("") || "<li>Nothing missing.</li>";
}

// ---------- side heat maps ----------

const CELL = 14; // px per metre
const X0 = -2; // heat grid starts at x -2 m, y -4 m (13 x 26 cells)
const Y0 = -4;
const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
const dark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

function rampColor(t) {
  const ramp = dark() ? [...RAMP].reverse() : RAMP; // dark mode: more = lighter, against the dark surface
  return ramp[Math.min(ramp.length - 1, Math.round(t * (ramp.length - 1)))];
}

// court metres -> canvas px; far baseline at the top, near (camera) end at the bottom
const toPx = (x, y) => [(x - X0) * CELL, (22 - y) * CELL];

function drawHeat(players) {
  const canvas = $("#heat");
  const ctx = canvas.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue("--panel");
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (const side of ["near", "far"]) {
    players[side].heat.forEach((row, iy) =>
      row.forEach((v, ix) => {
        const y = Y0 + iy;
        if (!v || (side === "near") !== y < 9) return; // each side only on its own half
        const [px, py] = toPx(X0 + ix, y + 1);
        ctx.fillStyle = rampColor(v);
        ctx.fillRect(px + 1, py + 1, CELL - 2, CELL - 2); // 2px surface gap between cells
      }),
    );
  }
  ctx.strokeStyle = css.getPropertyValue("--muted");
  ctx.lineWidth = 1;
  for (const [x1, y1, x2, y2] of [[0, 0, 9, 0], [9, 0, 9, 18], [9, 18, 0, 18], [0, 18, 0, 0], [0, 6, 9, 6], [0, 12, 9, 12]]) {
    ctx.beginPath();
    ctx.moveTo(...toPx(x1, y1));
    ctx.lineTo(...toPx(x2, y2));
    ctx.stroke();
  }
  ctx.strokeStyle = css.getPropertyValue("--fg");
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(...toPx(-0.5, 9));
  ctx.lineTo(...toPx(9.5, 9));
  ctx.stroke();
}

$("#heat").addEventListener("mousemove", (e) => {
  const tip = $("#heat-tip");
  if (!detail?.players) return;
  const r = e.target.getBoundingClientRect();
  const ix = Math.floor((e.clientX - r.left) / CELL);
  const y = 22 - Math.floor((e.clientY - r.top) / CELL) - 1;
  const iy = y - Y0;
  const side = y < 9 ? "near" : "far";
  const v = detail.players[side].heat[iy]?.[ix];
  if (v == null) return (tip.hidden = true);
  tip.textContent = `${sideLabel(side, detail.our_side)} · x ${X0 + ix}–${X0 + ix + 1} m, y ${y}–${y + 1} m · ${Math.round(v * 100)}% of the busiest square`;
  tip.style.left = `${e.clientX - r.left + 12}px`;
  tip.style.top = `${e.clientY - r.top + 12}px`;
  tip.hidden = false;
});
$("#heat").addEventListener("mouseleave", () => ($("#heat-tip").hidden = true));

function renderSides() {
  const m = picked();
  $("#sides").hidden = !m;
  if (!m) return;
  for (const b of document.querySelectorAll(".side-switch button"))
    b.setAttribute("aria-pressed", String((b.dataset.side || null) === (m.our_side || null)));
  if (!detail || detail.id !== m.id) {
    $("#side-numbers").textContent = "Loading player tracks…";
    return;
  }
  if (!detail.players) {
    $("#heat").hidden = true;
    $("#side-numbers").textContent = `No player numbers: ${detail.missing.players}`;
    return;
  }
  $("#heat").hidden = false;
  drawHeat(detail.players);
  $("#side-numbers").innerHTML = ["far", "near"]
    .map((side) => {
      const s = detail.players[side];
      const dist = s.mean_net_distance_m == null ? "–" : `${s.mean_net_distance_m.toFixed(1)} m`;
      return `<div class="tile"><div class="label">${sideLabel(side, m.our_side)} (${side} end)</div>
        <div class="value">${s.players_per_frame.toFixed(1)}</div><div class="sub">players on court per rally frame (ideal 6)</div>
        <div class="value small">${dist}</div><div class="sub">average distance from the net</div></div>`;
    })
    .join("");
}

async function render() {
  const m = picked();
  const matches = m ? [m] : all.matches;
  renderTiles(matches);
  renderBoards(matches);
  renderMissing(matches);
  renderSides();
  if (m && (!detail || detail.id !== m.id)) {
    try {
      detail = await getJson(`/api/matches/${m.id}/stats`);
    } catch (err) {
      detail = { id: m.id, missing: { players: err.message } };
    }
    if (picked()?.id === m.id) renderSides();
  }
}

for (const b of document.querySelectorAll(".side-switch button")) {
  b.onclick = async () => {
    const m = picked();
    const meta = await getJson(`/api/matches/${m.id}/meta`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ our_side: b.dataset.side || null }),
    });
    m.our_side = meta.our_side;
    if (detail) detail.our_side = meta.our_side;
    render();
  };
}

document.addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-href]");
  if (tr && !e.target.closest("a")) location.href = tr.dataset.href;
});

async function init() {
  try {
    all = await getJson("/api/stats");
  } catch (err) {
    $("#load-state").textContent = `Couldn't load stats (${err.message}). If the server is older than this page, restart it: uv run vball serve`;
    return;
  }
  $("#load-state").hidden = true;
  const pick = $("#pick");
  pick.innerHTML =
    `<option value="">All matches</option>` +
    all.matches.map((m) => `<option value="${m.id}">#${m.id} ${esc(m.name)}</option>`).join("");
  const want = new URLSearchParams(location.search).get("match");
  if (want && all.matches.some((m) => String(m.id) === want)) pick.value = want;
  pick.onchange = () => {
    history.replaceState(null, "", pick.value ? `?match=${pick.value}` : location.pathname);
    render();
  };
  render();
}
init();
