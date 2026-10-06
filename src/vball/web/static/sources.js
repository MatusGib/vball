// vball sources page: every external source and what we did with it (data: /sources.json)
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const STATUS = {
  used: "Used",
  tested: "Tested",
  next: "Next",
  "not tried": "Not tried yet",
  "not usable": "Not usable",
};
let data = null;
let filter = "all";

function render() {
  const shown = data.sources.filter((s) => filter === "all" || s.status === filter);
  const areas = [...new Set(shown.map((s) => s.area))];
  $("#groups").innerHTML =
    areas
      .map(
        (area) => `<section><h2>${esc(area)}</h2><div class="cards">${shown
          .filter((s) => s.area === area)
          .map(
            (s) => `<article class="card">
              <div class="card-head"><a href="${esc(s.link)}" target="_blank" rel="noopener">${esc(s.name)}</a>
                <span class="src-status src-${s.status.replace(" ", "-")}">${STATUS[s.status]}</span></div>
              <p class="meta">${esc(s.kind)} · ${esc(s.licence)}</p>
              <p>${esc(s.what)}</p>
              <p class="use"><strong>In vball:</strong> ${esc(s.use)}</p>
            </article>`,
          )
          .join("")}</div></section>`,
      )
      .join("") || `<p class="help">No sources with this status.</p>`;
  for (const b of document.querySelectorAll("#filters button")) b.setAttribute("aria-pressed", String(b.dataset.f === filter));
}

async function init() {
  data = await (await fetch("/sources.json")).json();
  const counts = Object.fromEntries(Object.keys(STATUS).map((k) => [k, data.sources.filter((s) => s.status === k).length]));
  $("#filters").innerHTML =
    `<button data-f="all">All (${data.sources.length})</button>` +
    Object.entries(STATUS)
      .filter(([k]) => counts[k])
      .map(([k, label]) => `<button data-f="${k}">${label} (${counts[k]})</button>`)
      .join("");
  for (const b of document.querySelectorAll("#filters button"))
    b.onclick = () => {
      filter = b.dataset.f;
      render();
    };
  render();
}
init();
