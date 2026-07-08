/* Tab 3 - Review / Edit Labels (single editor + grid "pan" view) */
(function () {
  const M = {
    items: [], scoped: [], classes: [], videos: [],
    filter: { state: "all", class: "__all__", video: "__all__" },
    view: [], viewIndex: 0, mode: "single",
    editing: false, boxes: [], selected: -1, drag: null,
    img: null, imgToken: 0,
  };
  const HANDLE = 8;

  function render(el) {
    el.innerHTML = `
      <div class="panel">
        <div class="spread">
          <div><h2>3 · Review / Edit Labels</h2>
            <p class="lead" style="margin:2px 0 0">Check each auto-drawn box. Keep good ones, fix wrong ones, drop useless ones.
              <br>Shortcuts: <span class="kbd">A</span> pass · <span class="kbd">F</span> fail · <span class="kbd">E</span> edit · <span class="kbd">Space</span> next · <span class="kbd">← →</span> move.
              In edit mode: <span class="kbd">R</span> redraw box · <span class="kbd">S</span> save.</p>
          </div>
          <div class="row">
            <button class="btn ghost sm" id="rvMode">▦ Grid view</button>
            <button class="btn ghost sm" id="rvRefresh">Refresh</button>
          </div>
        </div>

        <div class="review-toolbar" style="margin-top:14px">
          <div class="field-group"><label class="field">Class</label><select id="rvClass"></select></div>
          <div class="field-group"><label class="field">Source video</label><select id="rvVideo"></select></div>
          <button class="btn ghost" id="rvPassAll">Pass all unreviewed</button>
        </div>
        <div class="review-filters" id="rvChips" style="margin-top:12px"></div>

        <div id="rvSingle" class="review-layout" style="margin-top:14px">
          <div class="review-main">
            <div class="editor-stage" id="rvStage">
              <div class="editor-empty" id="rvEmpty">No frames match this filter.</div>
              <canvas id="rvCanvas" hidden></canvas>
            </div>
            <div class="editor-info" id="rvInfo"></div>
            <div class="editor-actions" id="rvActions"></div>
          </div>
          <div class="thumbs" id="rvThumbs"></div>
        </div>

        <div id="rvGrid" class="review-grid" style="margin-top:14px" hidden></div>
      </div>`;

    el.querySelector("#rvRefresh").addEventListener("click", () => load(el).catch((e) => App.showError(e)));
    el.querySelector("#rvMode").addEventListener("click", () => setMode(el, M.mode === "single" ? "grid" : "single"));
    el.querySelector("#rvClass").addEventListener("change", (e) => { M.filter.class = e.target.value; M.filter.video = "__all__"; load(el).catch((x) => App.showError(x)); });
    el.querySelector("#rvVideo").addEventListener("change", (e) => { M.filter.video = e.target.value; recompute(el); });
    el.querySelector("#rvPassAll").addEventListener("click", () => passAll(el).catch((e) => App.showError(e)));

    const canvas = el.querySelector("#rvCanvas");
    canvas.addEventListener("pointerdown", (e) => onPointerDown(el, e));
    canvas.addEventListener("pointermove", (e) => onPointerMove(el, e));
    canvas.addEventListener("pointerup", (e) => onPointerUp(el, e));
    canvas.addEventListener("pointerleave", (e) => onPointerUp(el, e));
  }

  function setMode(el, mode) {
    M.mode = mode;
    M.editing = false;
    el.querySelector("#rvMode").textContent = mode === "single" ? "▦ Grid view" : "▤ Single view";
    el.querySelector("#rvSingle").hidden = mode !== "single";
    el.querySelector("#rvGrid").hidden = mode !== "grid";
    if (mode === "grid") renderGrid(el); else showCurrent(el);
  }

  /* -------- data loading -------- */
  async function load(el) {
    const params = new URLSearchParams({ class: M.filter.class, video: M.filter.video, state: "all" });
    const data = await App.api("/api/review/frames?" + params.toString());
    M.items = data.items;
    M.classes = data.classes;
    M.videos = data.videos;
    renderSelectors(el);
    recompute(el, true);
  }

  function renderSelectors(el) {
    const clsSel = el.querySelector("#rvClass");
    clsSel.innerHTML = '<option value="__all__">All classes</option>' +
      M.classes.map((c) => `<option value="${c}">${c}</option>`).join("");
    clsSel.value = M.filter.class;
    const vidSel = el.querySelector("#rvVideo");
    const vids = M.videos.filter((v) => M.filter.class === "__all__" || v.class === M.filter.class);
    vidSel.innerHTML = '<option value="__all__">All videos</option>' +
      vids.map((v) => `<option value="${v.video}">${v.class} · ${v.video}</option>`).join("");
    vidSel.value = M.filter.video;
  }

  function recompute(el, resetIndex) {
    M.scoped = M.items.filter((it) => (M.filter.video === "__all__" || it.source === M.filter.video));
    M.view = M.filter.state === "all" ? M.scoped : M.scoped.filter((it) => it.state === M.filter.state);
    if (resetIndex || M.viewIndex >= M.view.length) M.viewIndex = 0;
    renderChips(el);
    if (M.mode === "grid") { renderGrid(el); }
    else { renderThumbs(el); showCurrent(el); }
  }

  function counts() {
    const c = { all: M.scoped.length, unreviewed: 0, passed: 0, failed: 0, edited: 0 };
    M.scoped.forEach((it) => { c[it.state] = (c[it.state] || 0) + 1; });
    return c;
  }

  function renderChips(el) {
    const c = counts();
    const chips = [["all", "All"], ["unreviewed", "Unreviewed"], ["passed", "Passed"], ["failed", "Failed"], ["edited", "Edited"]];
    el.querySelector("#rvChips").innerHTML = chips.map(([key, label]) =>
      `<button class="chip ${M.filter.state === key ? "active" : ""}" data-state="${key}">${label}<span class="n">${c[key] || 0}</span></button>`).join("");
    el.querySelectorAll("#rvChips .chip").forEach((node) => node.addEventListener("click", () => {
      M.filter.state = node.dataset.state; recompute(el, true);
    }));
  }
  function updateChipCounts(el) {
    const c = counts();
    el.querySelectorAll("#rvChips .chip").forEach((node) => {
      const span = node.querySelector(".n"); if (span) span.textContent = c[node.dataset.state] || 0;
    });
  }

  /* -------- grid (pan) view -------- */
  function renderGrid(el) {
    const root = el.querySelector("#rvGrid");
    if (!M.view.length) { root.innerHTML = '<p class="muted" style="grid-column:1/-1">No frames match this filter.</p>'; return; }
    root.innerHTML = M.view.map((it, i) => `
      <div class="grid-tile state-${it.state}" data-stem="${it.stem}" data-i="${i}">
        <img src="${it.url}" loading="lazy" alt="" title="${it.stem}">
        <span class="badge ${it.state}"></span>
        <div class="tile-btns">
          <button class="pass ${it.state === "passed" ? "on" : ""}" data-act="pass">Pass</button>
          <button class="edit ${it.state === "edited" ? "on" : ""}" data-act="edit">Edit</button>
          <button class="fail ${it.state === "failed" ? "on" : ""}" data-act="fail">Fail</button>
        </div>
      </div>`).join("");
    root.querySelectorAll(".grid-tile").forEach((tile) => {
      const i = Number(tile.dataset.i);
      tile.querySelector("img").addEventListener("click", () => jumpToEdit(el, i, false));
      tile.querySelector('[data-act="pass"]').addEventListener("click", () => markItem(el, M.view[i], "passed"));
      tile.querySelector('[data-act="fail"]').addEventListener("click", () => markItem(el, M.view[i], "failed"));
      tile.querySelector('[data-act="edit"]').addEventListener("click", () => jumpToEdit(el, i, true));
    });
  }

  function updateTile(el, item) {
    const tile = el.querySelector(`#rvGrid .grid-tile[data-stem="${cssEscape(item.stem)}"]`);
    if (!tile) return;
    tile.className = `grid-tile state-${item.state}`;
    tile.querySelector(".badge").className = `badge ${item.state}`;
    tile.querySelector('[data-act="pass"]').classList.toggle("on", item.state === "passed");
    tile.querySelector('[data-act="fail"]').classList.toggle("on", item.state === "failed");
    tile.querySelector('[data-act="edit"]').classList.toggle("on", item.state === "edited");
  }
  function cssEscape(s) { return s.replace(/["\\]/g, "\\$&"); }

  function jumpToEdit(el, viewIndex, edit) {
    setMode(el, "single");
    M.viewIndex = viewIndex;
    showCurrent(el);
    if (edit) toggleEdit(el, true);
  }

  /* -------- thumbnails (single view) -------- */
  function renderThumbs(el) {
    const root = el.querySelector("#rvThumbs");
    if (!M.view.length) { root.innerHTML = '<p class="muted" style="padding:8px">No frames.</p>'; return; }
    root.innerHTML = M.view.map((it, i) =>
      `<div class="thumb ${i === M.viewIndex ? "active" : ""}" data-i="${i}" data-stem="${it.stem}">
        <img src="${it.url}" loading="lazy" alt=""><span class="badge ${it.state}"></span></div>`).join("");
    root.querySelectorAll(".thumb").forEach((node) => node.addEventListener("click", () => {
      M.viewIndex = Number(node.dataset.i); M.editing = false; showCurrent(el);
    }));
    const active = root.querySelector(".thumb.active");
    if (active) active.scrollIntoView({ block: "nearest" });
  }

  /* -------- current frame + canvas -------- */
  function current() { return M.view[M.viewIndex] || null; }

  function showCurrent(el) {
    const item = current();
    const canvas = el.querySelector("#rvCanvas");
    const empty = el.querySelector("#rvEmpty");
    if (!item) {
      canvas.hidden = true; empty.hidden = false;
      el.querySelector("#rvInfo").innerHTML = ""; el.querySelector("#rvActions").innerHTML = "";
      renderThumbActive(el); return;
    }
    empty.hidden = true; canvas.hidden = false;
    M.boxes = item.boxes.map((b) => ({ ...b }));
    M.selected = M.boxes.length ? 0 : -1;
    const token = ++M.imgToken;
    const img = new Image();
    img.onload = () => {
      if (token !== M.imgToken) return;
      M.img = img;
      const maxW = 900, maxH = 560;
      const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
      canvas.width = Math.round(img.naturalWidth * scale); canvas.height = Math.round(img.naturalHeight * scale);
      draw(el);
    };
    img.src = item.url;
    renderInfo(el); renderActions(el); renderThumbActive(el);
  }

  function renderThumbActive(el) {
    el.querySelectorAll("#rvThumbs .thumb").forEach((node) =>
      node.classList.toggle("active", Number(node.dataset.i) === M.viewIndex));
    const active = el.querySelector("#rvThumbs .thumb.active");
    if (active) active.scrollIntoView({ block: "nearest" });
  }

  function draw(el) {
    const canvas = el.querySelector("#rvCanvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (M.img) ctx.drawImage(M.img, 0, 0, canvas.width, canvas.height);
    M.boxes.forEach((box, i) => {
      const r = normToPx(canvas, box);
      const isSel = i === M.selected;
      ctx.lineWidth = isSel ? 3 : 2;
      ctx.strokeStyle = isSel ? "#f59e0b" : "#16a34a";
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      const label = className(box.cls);
      ctx.font = "600 14px Inter, system-ui, sans-serif";
      const tw = ctx.measureText(label).width + 10;
      ctx.fillStyle = isSel ? "#f59e0b" : "#16a34a";
      ctx.fillRect(r.x, Math.max(0, r.y - 20), tw, 20);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, r.x + 5, Math.max(14, r.y - 6));
      if (M.editing && isSel) {
        ctx.fillStyle = "#fff"; ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 1.5;
        handlePositions(r).forEach((p) => { ctx.fillRect(p.x - HANDLE / 2, p.y - HANDLE / 2, HANDLE, HANDLE); ctx.strokeRect(p.x - HANDLE / 2, p.y - HANDLE / 2, HANDLE, HANDLE); });
      }
    });
  }

  function className(cls) { return M.classes[cls] || `class ${cls}`; }
  function normToPx(canvas, b) {
    return { x: (b.cx - b.w / 2) * canvas.width, y: (b.cy - b.h / 2) * canvas.height, w: b.w * canvas.width, h: b.h * canvas.height };
  }
  function handlePositions(r) {
    return [
      { n: "nw", x: r.x, y: r.y }, { n: "n", x: r.x + r.w / 2, y: r.y }, { n: "ne", x: r.x + r.w, y: r.y },
      { n: "e", x: r.x + r.w, y: r.y + r.h / 2 }, { n: "se", x: r.x + r.w, y: r.y + r.h },
      { n: "s", x: r.x + r.w / 2, y: r.y + r.h }, { n: "sw", x: r.x, y: r.y + r.h }, { n: "w", x: r.x, y: r.y + r.h / 2 },
    ];
  }
  function canvasPoint(canvas, e) {
    const rect = canvas.getBoundingClientRect();
    return { x: (e.clientX - rect.left) * (canvas.width / rect.width), y: (e.clientY - rect.top) * (canvas.height / rect.height) };
  }

  function onPointerDown(el, e) {
    if (!M.editing || !current()) return;
    const canvas = el.querySelector("#rvCanvas");
    canvas.setPointerCapture(e.pointerId);
    const p = canvasPoint(canvas, e);
    if (M.selected >= 0) {
      const r = normToPx(canvas, M.boxes[M.selected]);
      const handle = handlePositions(r).find((h) => Math.abs(h.x - p.x) <= HANDLE && Math.abs(h.y - p.y) <= HANDLE);
      if (handle) { M.drag = { mode: "resize", handle: handle.n, start: p, orig: { ...M.boxes[M.selected] } }; return; }
    }
    for (let i = M.boxes.length - 1; i >= 0; i--) {
      const r = normToPx(canvas, M.boxes[i]);
      if (p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) {
        M.selected = i; M.drag = { mode: "move", start: p, orig: { ...M.boxes[i] } }; renderActions(el); draw(el); return;
      }
    }
    const nb = { cls: M.selected >= 0 ? M.boxes[M.selected].cls : currentClassId(), cx: p.x / canvas.width, cy: p.y / canvas.height, w: 0.001, h: 0.001 };
    M.boxes.push(nb); M.selected = M.boxes.length - 1;
    M.drag = { mode: "new", start: p, origin: p };
    renderActions(el); draw(el);
  }

  function onPointerMove(el, e) {
    if (!M.drag) return;
    const canvas = el.querySelector("#rvCanvas");
    const p = canvasPoint(canvas, e);
    const box = M.boxes[M.selected];
    if (M.drag.mode === "move") {
      const dx = (p.x - M.drag.start.x) / canvas.width, dy = (p.y - M.drag.start.y) / canvas.height;
      box.cx = clamp(M.drag.orig.cx + dx, box.w / 2, 1 - box.w / 2);
      box.cy = clamp(M.drag.orig.cy + dy, box.h / 2, 1 - box.h / 2);
    } else if (M.drag.mode === "resize") {
      resizeBox(canvas, box, M.drag, p);
    } else if (M.drag.mode === "new") {
      const x0 = Math.min(M.drag.origin.x, p.x), y0 = Math.min(M.drag.origin.y, p.y);
      const x1 = Math.max(M.drag.origin.x, p.x), y1 = Math.max(M.drag.origin.y, p.y);
      box.cx = ((x0 + x1) / 2) / canvas.width; box.cy = ((y0 + y1) / 2) / canvas.height;
      box.w = Math.max(0.005, (x1 - x0) / canvas.width); box.h = Math.max(0.005, (y1 - y0) / canvas.height);
    }
    draw(el); renderInfo(el);
  }

  function onPointerUp(el, e) {
    if (!M.drag) return;
    const box = M.boxes[M.selected];
    if (M.drag.mode === "new" && (box.w < 0.01 || box.h < 0.01)) {
      M.boxes.splice(M.selected, 1); M.selected = M.boxes.length ? 0 : -1;
    }
    M.drag = null; draw(el); renderInfo(el); renderActions(el);
  }

  function resizeBox(canvas, box, drag, p) {
    const o = drag.orig;
    let left = (o.cx - o.w / 2), right = (o.cx + o.w / 2), top = (o.cy - o.h / 2), bottom = (o.cy + o.h / 2);
    const nx = p.x / canvas.width, ny = p.y / canvas.height, h = drag.handle;
    if (h.includes("w")) left = Math.min(nx, right - 0.01);
    if (h.includes("e")) right = Math.max(nx, left + 0.01);
    if (h.includes("n")) top = Math.min(ny, bottom - 0.01);
    if (h.includes("s")) bottom = Math.max(ny, top + 0.01);
    box.cx = (left + right) / 2; box.cy = (top + bottom) / 2; box.w = right - left; box.h = bottom - top;
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function currentClassId() {
    const item = current();
    const idx = M.classes.indexOf(item ? item.class : "");
    return idx >= 0 ? idx : 0;
  }

  /* -------- info + actions -------- */
  function renderInfo(el) {
    const item = current();
    if (!item) { el.querySelector("#rvInfo").innerHTML = ""; return; }
    const box = M.boxes[M.selected];
    const yolo = box ? `<code>${box.cls} ${box.cx.toFixed(3)} ${box.cy.toFixed(3)} ${box.w.toFixed(3)} ${box.h.toFixed(3)}</code>` : "<em>no box</em>";
    el.querySelector("#rvInfo").innerHTML = `
      <span>Frame <b>${M.viewIndex + 1}/${M.view.length}</b></span>
      <span>Class <b>${item.class}</b></span>
      <span>Video <b>${item.source}</b></span>
      <span>State <span class="state-tag ${item.state}">${item.state}</span></span>
      <span>YOLO ${yolo}</span>`;
  }

  function renderActions(el) {
    const item = current();
    const root = el.querySelector("#rvActions");
    if (!item) { root.innerHTML = ""; return; }
    if (!M.editing) {
      root.innerHTML = `
        <button class="btn" id="rvPass">Pass <span class="kbd">A</span></button>
        <button class="btn ghost" id="rvEdit">Edit box <span class="kbd">E</span></button>
        <button class="btn danger" id="rvFail">Fail <span class="kbd">F</span></button>
        <button class="btn ghost" id="rvSkip">Skip <span class="kbd">Space</span></button>`;
      root.querySelector("#rvPass").addEventListener("click", () => mark(el, "passed"));
      root.querySelector("#rvFail").addEventListener("click", () => mark(el, "failed"));
      root.querySelector("#rvSkip").addEventListener("click", () => advance(el));
      root.querySelector("#rvEdit").addEventListener("click", () => toggleEdit(el, true));
    } else {
      const options = M.classes.map((c, i) => `<option value="${i}" ${M.boxes[M.selected] && M.boxes[M.selected].cls === i ? "selected" : ""}>${c}</option>`).join("");
      root.innerHTML = `
        <label class="row" style="gap:6px;font-weight:600">Class
          <select id="rvBoxClass" style="width:auto" ${M.selected < 0 ? "disabled" : ""}>${options}</select></label>
        <button class="btn ghost" id="rvRedraw">Redraw box <span class="kbd">R</span></button>
        <button class="btn ghost" id="rvDel" ${M.selected < 0 ? "disabled" : ""}>Delete box</button>
        <button class="btn" id="rvSave">Save <span class="kbd">S</span></button>
        <button class="btn ghost" id="rvCancel">Cancel</button>
        <span class="muted">Drag to move · handles to resize · draw on empty space</span>`;
      root.querySelector("#rvBoxClass").addEventListener("change", (e) => { if (M.selected >= 0) { M.boxes[M.selected].cls = Number(e.target.value); draw(el); renderInfo(el); } });
      root.querySelector("#rvRedraw").addEventListener("click", () => redraw(el));
      root.querySelector("#rvDel").addEventListener("click", () => { if (M.selected >= 0) { M.boxes.splice(M.selected, 1); M.selected = M.boxes.length ? 0 : -1; draw(el); renderActions(el); renderInfo(el); } });
      root.querySelector("#rvSave").addEventListener("click", () => save(el).catch((x) => App.showError(x)));
      root.querySelector("#rvCancel").addEventListener("click", () => { M.editing = false; showCurrent(el); });
    }
  }

  function toggleEdit(el, on) {
    M.editing = on !== undefined ? on : !M.editing;
    if (M.selected < 0 && M.boxes.length) M.selected = 0;
    draw(el); renderActions(el);
  }

  function redraw(el) {
    if (!current()) return;
    if (!M.editing) toggleEdit(el, true);
    if (M.selected >= 0) M.boxes.splice(M.selected, 1);
    M.selected = -1;
    draw(el); renderActions(el);
    App.toast("Drag on the image to draw the new box");
  }

  /* -------- persistence -------- */
  async function markItem(el, item, decision) {
    if (!item) return;
    await App.postJson("/api/review/mark", { stem: item.stem, decision });
    item.state = decision;
    updateTile(el, item);
    updateChipCounts(el);
    App.refresh().catch(() => {});
  }

  async function mark(el, decision) {
    const item = current();
    if (!item) return;
    await markItem(el, item, decision);
    const thumb = el.querySelector(`#rvThumbs .thumb[data-stem="${cssEscape(item.stem)}"] .badge`);
    if (thumb) thumb.className = `badge ${item.state}`;
    if (M.filter.state !== "all") {
      recompute(el);
      if (M.viewIndex >= M.view.length) M.viewIndex = Math.max(0, M.view.length - 1);
      showCurrent(el);
    } else { renderInfo(el); advance(el); }
  }

  async function save(el) {
    const item = current();
    if (!item) return;
    await App.postJson("/api/review/save", { stem: item.stem, boxes: M.boxes });
    item.boxes = M.boxes.map((b) => ({ ...b }));
    item.state = "edited";
    M.editing = false;
    updateChipCounts(el);
    App.refresh().catch(() => {});
    showCurrent(el);
    App.toast("Saved");
  }

  function advance(el) {
    if (M.viewIndex < M.view.length - 1) { M.viewIndex++; M.editing = false; showCurrent(el); }
    else App.toast("End of list");
  }
  function back(el) { if (M.viewIndex > 0) { M.viewIndex--; M.editing = false; showCurrent(el); } }

  async function passAll(el) {
    const res = await App.postJson("/api/review/pass-unreviewed", { class: M.filter.class, video: M.filter.video });
    App.toast(`Passed ${res.passed} frame(s)`);
    await App.refresh();
    await load(el);
  }

  /* keyboard shortcuts (only when this tab is active and not typing in a field) */
  document.addEventListener("keydown", (e) => {
    if (App.activeTab !== "review") return;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName)) return;
    const el = App.tab("review").el;
    const key = e.key.toLowerCase();
    if (M.editing) {
      if (key === "r") { e.preventDefault(); redraw(el); }
      else if (key === "s") { e.preventDefault(); save(el).catch((x) => App.showError(x)); }
      else if (key === "e") { e.preventDefault(); M.editing = false; showCurrent(el); }
      else if (e.key === "Escape") { e.preventDefault(); M.editing = false; showCurrent(el); }
      return;
    }
    if (M.mode !== "single") return; // shortcuts drive the single editor
    if (key === "a") { e.preventDefault(); mark(el, "passed"); }
    else if (key === "f") { e.preventDefault(); mark(el, "failed"); }
    else if (key === "e") { e.preventDefault(); toggleEdit(el, true); }
    else if (e.key === " ") { e.preventDefault(); advance(el); }
    else if (e.key === "ArrowRight") { e.preventDefault(); advance(el); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); back(el); }
  });

  App.registerTab({
    id: "review", label: "Review / Edit",
    mount(el) { render(el); },
    onShow() { load(this.el).catch((e) => App.showError(e)); },
  });
})();
