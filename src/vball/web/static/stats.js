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

// one team-sheet line of figures: the number set in the display face, its meaning straight after it
function fig(value, label, href = null) {
  const tag = href ? `a href="${href}" title="Open the viewer at this moment"` : "span";
  return `<${tag} class="fig"><b>${esc(value)}</b><span>${esc(label)}</span></${href ? "a" : "span"}>`;
}

function renderTiles(matches) {
  const rallies = matches.reduce((n, m) => n + (m.rallies.count || 0), 0);
  const rallyTime = matches.reduce((n, m) => n + (m.rallies.total_s || 0), 0);
  const serves = matches.flatMap((m) => (m.flights || {}).serves || []);
  const ok = serves.filter((s) => s.plausible);
  const best = board(matches, "serves", "speed_kmh", 1)[0];
  const sources = [...new Set(matches.map((m) => m.rallies.source))].join(" + ");
  $("#tiles").innerHTML = [
    fig(rallies, `rallies (${sources})`),
    fig(rallies ? `${(rallyTime / rallies).toFixed(1)} s` : "–", `average rally, ${(rallyTime / 60).toFixed(0)} min of play`),
    fig(serves.length ? `${ok.length} of ${serves.length}` : "–", "serves measured"),
    fig(
      best ? `${Math.round(best.speed_kmh)} km/h` : "–",
      best ? `hardest serve, ${sideLabel(best.side, best.our_side)} in #${best.match_id} at ${mmss(best.time_s)}` : "hardest serve (needs 3D flights)",
      best ? viewerLink(best.match_id, best.time_s, "speed") : null,
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
    Object.entries(m.missing || {}).map(([what, how]) => `<li>#${m.id} ${esc(m.name)}, ${what}: ${esc(how).replace(/(uv run vball [a-z0-9]+ \d+)/, "<code>$1</code>")}</li>`),
  );
  const detected = matches.filter((m) => m.rallies.source === "detected").map((m) => `#${m.id}`);
  if (detected.length)
    items.push(`<li>Rallies for ${detected.join(", ")} are detected, not labelled: lengths may merge two rallies.</li>`);
  $("#missing").innerHTML = items.join("") || "<li>Nothing missing.</li>";
}

// ---------- score and serves ----------

let servingDetail = null; // GET /api/matches/{id}/serving for the picked match
const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const OTHER = { near: "far", far: "near" };

function teamsFirst(ourSide) {
  return ourSide ? [ourSide, OTHER[ourSide]] : ["near", "far"];
}

function scoreText(m) {
  const sc = m.serving?.score;
  if (!sc) return "–";
  const [a, b] = teamsFirst(m.our_side);
  return `${sc[a]}–${sc[b]}`;
}

function renderServing(matches) {
  const withData = matches.filter((m) => m.serving);
  $("#serving").hidden = !withData.length;
  if (!withData.length) return;
  const one = matches.length === 1 ? matches[0] : null;
  const notes = [];
  if (one) {
    const s = one.serving.summary;
    const head = ["Team", "Points", "Serves", "Won on serve", "Aces", "Serve errors", "Serving %", "Side-out %"];
    $("#serving-teams").innerHTML =
      `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
      teamsFirst(one.our_side)
        .map((side) => {
          const t = s[side];
          return `<tr><td class="team-${side}">${sideLabel(side, one.our_side)}</td><td>${t.points}</td><td>${t.serves}</td>
            <td>${t.won_on_serve}</td><td>${t.aces}</td><td>${t.errors}</td><td>${pct(t.serving_pct)}</td><td>${pct(t.sideout_pct)}</td></tr>`;
        })
        .join("") +
      "</tbody>";
    renderServers(one);
    renderReal(one, notes);
    if (one.serving.source !== "labels")
      notes.push(`<li class="warn">These rallies are detected, not labelled: a false rally adds a point. Approve the labels in the viewer for a true score.</li>`);
    const todo = servingDetail?.id === one.id
      ? servingDetail.rallies.map((r, i) => ({ ...r, i })).filter((r) => r.outcome === "check" || !r.end)
      : [];
    if (todo.length)
      notes.push(
        `<li class="warn">${todo.length} serve${todo.length > 1 ? "s" : ""} to check (not counted as ace or error): ` +
          todo.map((r) => `<a href="${viewerLink(one.id, r.start_s)}">#${r.i + 1} ${mmss(r.start_s)}</a>`).join(", ") +
          "</li>",
      );
    if (s.unknown_winner)
      notes.push(`<li>${s.unknown_winner} rall${s.unknown_winner > 1 ? "ies have" : "y has"} no known winner (the next serving end is unknown, or the set didn't end on camera).</li>`);
    if (!one.our_side) notes.push(`<li>Set which end is our team (below) to see us / them and our servers.</li>`);
  } else {
    $("#servers").hidden = true;
    const head = ["Match", "Score", "Aces", "Serve errors", "Side-out %", "To check"];
    $("#serving-teams").innerHTML =
      `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
      withData
        .map((m) => {
          const [a, b] = teamsFirst(m.our_side);
          const s = m.serving.summary;
          const two = (k, f = (v) => v) => `${f(s[a][k])} / ${f(s[b][k])}`;
          const href = `?match=${m.id}`;
          return `<tr data-href="${href}"><td><a href="${href}">#${m.id} ${esc(m.name)}</a></td><td>${scoreText(m)}</td>
            <td>${two("aces")}</td><td>${two("errors")}</td><td>${two("sideout_pct", pct)}</td><td>${s.check}</td></tr>`;
        })
        .join("") +
      "</tbody>";
    notes.push(`<li>Pairs are us / them where the team's end is set, otherwise near / far. Pick a match for the details.</li>`);
  }
  $("#serving-notes").innerHTML = notes.join("");
}

// the owner's real final score next to the inferred one: how far off the serving ends and the rallies are
function renderReal(m, notes) {
  $("#real-form").hidden = !m.our_side;
  if (!m.our_side) return;
  const real = m.serving.real_score;
  for (const [id, k] of [["#real-us", "us"], ["#real-them", "them"]])
    if (document.activeElement !== $(id)) $(id).value = real ? real[k] : "";
  if (!real || !m.serving.score) return;
  const us = m.serving.score[m.our_side];
  const them = m.serving.score[OTHER[m.our_side]];
  const played = real.us + real.them;
  const rallies = m.serving.summary.near.serves + m.serving.summary.far.serves + m.serving.summary.unknown_end;
  const off = Math.abs(us - real.us) + Math.abs(them - real.them);
  const gap = played - rallies;
  notes.unshift(
    `<li class="${off ? "warn" : ""}">Real score ${real.us}–${real.them}, vball's ${us}–${them}` +
      (off ? ` (${off} point${off > 1 ? "s" : ""} off)` : " (exact)") +
      `. ${played} points were played and ${rallies} rallies are ${m.serving.source === "labels" ? "labelled" : "detected"}` +
      (gap > 0 ? `: ${gap} missing, so the score can't be exact until they are labelled.` : gap < 0 ? `: ${-gap} too many (false rallies).` : ".") +
      "</li>",
  );
  const unl = m.serving.unlabelled_detected || [];
  if (gap > 0 && unl.length)
    notes.splice(
      1,
      0,
      `<li class="warn">Detected but not in your labels (check each; label it if it was a rally): ` +
        unl.map((r) => `<a href="${viewerLink(m.id, r.start_s)}">${mmss(r.start_s)} (${(r.end_s - r.start_s).toFixed(1)} s)</a>`).join(", ") +
        "</li>",
    );
}

$("#real-form").onsubmit = async (e) => {
  e.preventDefault();
  const m = picked();
  const us = $("#real-us").value, them = $("#real-them").value;
  const body = us === "" || them === "" ? { real_score: null } : { real_score: { us: Number(us), them: Number(them) } };
  await getJson(`/api/matches/${m.id}/meta`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  document.activeElement.blur();
  await refreshMatch(m.id);
};

function renderServers(m) {
  $("#servers").hidden = !m.our_side;
  if (!m.our_side) return;
  const input = $("#lineup");
  if (document.activeElement !== input) input.value = (m.serving.lineup || []).join(", ");
  const rows = m.serving.players || [];
  if (!rows.length) {
    $("#serving-players").innerHTML = `<tbody><tr><td class="empty">Type the serving order to split our serves by player. Substitutions are not followed.</td></tr></tbody>`;
    return;
  }
  const head = ["Player", "Serves", "Aces", "Errors", "Serving %", "Top", "Median", "To check"];
  const kmh = (v) => (v == null ? "–" : `${Math.round(v)} km/h`);
  $("#serving-players").innerHTML =
    `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
    rows
      .map((p) => `<tr><td>${esc(p.name)}</td><td>${p.serves}</td><td>${p.aces}</td><td>${p.errors}</td><td>${pct(p.serving_pct)}</td>
        <td>${kmh(p.top_kmh)}</td><td title="${p.measured} of ${p.serves} serves measured in 3D">${kmh(p.median_kmh)}</td><td>${p.check}</td></tr>`)
      .join("") +
    "</tbody>";
}

$("#lineup-form").onsubmit = async (e) => {
  e.preventDefault();
  const m = picked();
  const lineup = $("#lineup").value.split(/[,;]/).map((s) => s.trim()).filter(Boolean);
  await getJson(`/api/matches/${m.id}/meta`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lineup }),
  });
  $("#lineup").blur();
  await refreshMatch(m.id);
};

// one match's numbers changed (our end, lineup): reload them without leaving the page
async function refreshMatch(id) {
  const fresh = await getJson(`/api/matches/${id}/stats`);
  const i = all.matches.findIndex((m) => m.id === id);
  all.matches[i] = { ...all.matches[i], ...fresh };
  detail = null;
  servingDetail = null;
  render();
}

// ---------- side heat maps ----------

const CELL = 24; // px per metre: the court fills its panel
const X0 = -2; // heat grid starts at x -2 m, y -4 m (13 x 26 cells)
const Y0 = -4;
// one ramp per side, from that side's reserved overlay colour (near blue, far red), light -> dark
const RAMPS = {
  near: ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
  far: ["#fde0dc", "#fbcbc4", "#f8b4ab", "#f59d92", "#f1867a", "#ec6f62", "#e0533f", "#cd4333", "#b8382b", "#a12f24", "#8a271e", "#731f18", "#5c1813"],
};
function rampColor(side, t) {
  const ramp = RAMPS[side];
  return ramp[Math.min(ramp.length - 1, Math.round(t * (ramp.length - 1)))];
}

// court metres -> canvas px; far baseline at the top, near (camera) end at the bottom
const toPx = (x, y) => [(x - X0) * CELL, (22 - y) * CELL];

function drawHeat(players) {
  const canvas = $("#heat");
  const ctx = canvas.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue("--maple"); // the court is drawn as the hall floor itself
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (const side of ["near", "far"]) {
    players[side].heat.forEach((row, iy) =>
      row.forEach((v, ix) => {
        const y = Y0 + iy;
        if (v < 0.03 || (side === "near") !== y < 9) return; // near-zero stays bare floor; each side on its own half
        const [px, py] = toPx(X0 + ix, y + 1);
        ctx.fillStyle = rampColor(side, v);
        ctx.fillRect(px + 1, py + 1, CELL - 2, CELL - 2); // 2px surface gap between cells
      }),
    );
  }
  ctx.strokeStyle = css.getPropertyValue("--ink-soft"); // reads on the pale floor and on the night floor
  ctx.lineWidth = 2;
  for (const [x1, y1, x2, y2] of [[0, 0, 9, 0], [9, 0, 9, 18], [9, 18, 0, 18], [0, 18, 0, 0], [0, 6, 9, 6], [0, 12, 9, 12]]) {
    ctx.beginPath();
    ctx.moveTo(...toPx(x1, y1));
    ctx.lineTo(...toPx(x2, y2));
    ctx.stroke();
  }
  ctx.strokeStyle = css.getPropertyValue("--kit");
  ctx.lineWidth = 4;
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
      return `<div class="side-row"><h3>${sideLabel(side, m.our_side)} · ${side} end</h3>
        <b>${s.players_per_frame.toFixed(1)}</b><span>players on court per rally frame (ideal 6)</span>
        <b>${dist}</b><span>average distance from the net</span></div>`;
    })
    .join("");
}

async function render() {
  const m = picked();
  const matches = m ? [m] : all.matches;
  renderTiles(matches);
  renderBoards(matches);
  renderMissing(matches);
  renderServing(matches);
  renderSides();
  if (m?.serving && servingDetail?.id !== m.id) {
    getJson(`/api/matches/${m.id}/serving`)
      .then((body) => {
        servingDetail = { ...body, id: m.id };
        if (picked()?.id === m.id) renderServing([m]);
      })
      .catch(() => {});
  }
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
    await refreshMatch(m.id); // servers and us / them depend on our end
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
