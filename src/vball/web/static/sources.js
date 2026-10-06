// vball sources page: every external source and what we did with it (data: /sources.json)
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const STATUS = {
  used: "Used",
  tested: "Tested",
  next: "Next",
  "needs you": "Needs you",
  "not tried": "Not tried yet",
  "not usable": "Not usable",
};
let data = null;
let filter = "all";

// one experiment or step on a source: what was done, what came out, optional table and evidence picture
function workItem(w) {
  const table = w.table
    ? `<div class="work-table"><table><thead><tr>${w.table.head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${w.table.rows
        .map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`)
        .join("")}</tbody></table></div>`
    : "";
  const img = w.img
    ? `<figure><a href="${esc(w.img)}" target="_blank" rel="noopener"><img src="${esc(w.img)}" alt="${esc(w.caption || w.title)}" loading="lazy"></a>${w.caption ? `<figcaption>${esc(w.caption)}</figcaption>` : ""}</figure>`
    : "";
  const doc = w.doc ? ` <span class="work-doc">(${esc(w.doc)})</span>` : "";
  return `<div class="work-item"><p class="work-head"><time>${esc(w.date)}</time> ${esc(w.title)}</p><p>${esc(w.result)}${doc}</p>${table}${img}</div>`;
}

function render() {
  const shown = data.sources.filter((s) => filter === "all" || s.status === filter);
  const areas = [...new Set(shown.map((s) => s.area))];
  $("#groups").innerHTML =
    areas
      .map(
        (area) => `<section class="register kit"><h2>${esc(area)}</h2>${shown
          .filter((s) => s.area === area)
          .map(
            (s) => `<article class="entry">
              <div><p class="name"><a href="${esc(s.link)}" target="_blank" rel="noopener">${esc(s.name)}</a></p>
                <p class="meta">${esc(s.kind)} · ${esc(s.licence)}</p>
                <span class="src-status src-${s.status.replace(" ", "-")}">${STATUS[s.status]}</span></div>
              <p>${esc(s.what)}</p>
              <p class="use"><b>In vball:</b> ${esc(s.use)}</p>
              ${s.work?.length ? `<details class="work" open><summary>Work done (${s.work.length})</summary>${s.work.map(workItem).join("")}</details>` : ""}
            </article>`,
          )
          .join("")}</section>`,
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
  const worked = data.sources.filter((s) => s.work?.length).length;
  $("#work-count").textContent = `Work logged on ${worked} of ${data.sources.length} sources; each "Work done" list shows what was run on our footage and what came out.`;
  render();
}
init();
