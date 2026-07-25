/* Tab 6 - Retraining Queue: correct captured failures, then add to the dataset */
(function () {
  const M = { items: [], counts: {}, classes: [], view: [], index: 0, showAll: false,
    boxes: [], selected: -1, drag: null, img: null, imgToken: 0, defaultClass: 0 };
  const HANDLE = 8;

  function render(el) {
    el.innerHTML = `
      <div class="panel">
        <div class="spread">
          <div><h2>6 · Retraining Queue</h2>
            <p class="lead">Failure frames you captured while testing. Draw the correct box(es), pick the class, and
              add them to the dataset — corrected mistakes are the most valuable training data.</p></div>
          <label class="row" style="gap:6px;font-weight:600"><input type="checkbox" id="rqAll"> Show corrected/discarded</label>
        </div>
        <div class="retrain-counts" id="rqCounts"></div>
        <div class="review-layout" style="margin-top:14px">
          <div class="review-main">
            <div class="editor-stage" id="rqStage">
              <div class="editor-empty" id="rqEmpty">No pending failure frames. Capture some in Test Detector.</div>
              <canvas id="rqCanvas" hidden></canvas>
            </div>
            <div class="editor-info" id="rqInfo"></div>
            <div class="editor-actions" id="rqActions"></div>
          </div>
          <div class="thumbs" id="rqThumbs"></div>
        </div>
      </div>`;

    el.querySelector("#rqAll").addEventListener("change", (e) => { M.showAll = e.target.checked; recompute(el); });
    const canvas = el.querySelector("#rqCanvas");
    canvas.addEventListener("pointerdown", (e) => onDown(el, e));
    canvas.addEventListener("pointermove", (e) => onMove(el, e));
    canvas.addEventListener("pointerup", (e) => onUp(el, e));
    canvas.addEventListener("pointerleave", (e) => onUp(el, e));
  }

  async function load(el) {
    const data = await App.api("/api/retrain/list");
    M.items = data.items; M.counts = data.counts; M.classes = data.classes;
    recompute(el, true);
  }

  function recompute(el, reset) {
    M.view = M.showAll ? M.items : M.items.filter((it) => it.status === "pending" && it.hasImage);
    if (reset || M.index >= M.view.length) M.index = 0;
    renderCounts(el); renderThumbs(el); showCurrent(el);
  }

  function renderCounts(el) {
    const c = M.counts;
    el.querySelector("#rqCounts").innerHTML = [
      ["total", "Total"], ["pending", "Pending"], ["corrected", "Corrected"], ["discarded", "Discarded"],
    ].map(([k, label]) => `<div class="box"><b>${c[k] || 0}</b><span>${label}</span></div>`).join("");
  }

  function renderThumbs(el) {
    const root = el.querySelector("#rqThumbs");
    if (!M.view.length) { root.innerHTML = '<p class="muted" style="padding:8px">Nothing here.</p>'; return; }
    root.innerHTML = M.view.map((it, i) =>
      `<div class="thumb ${i === M.index ? "active" : ""}" data-i="${i}">
        <img src="${it.url}" loading="lazy" alt=""><span class="badge ${it.status === "pending" ? "unreviewed" : "passed"}"></span></div>`).join("");
    root.querySelectorAll(".thumb").forEach((n) => n.addEventListener("click", () => { M.index = Number(n.dataset.i); showCurrent(el); }));
  }

  function current() { return M.view[M.index] || null; }

  function showCurrent(el) {
    const item = current();
    const canvas = el.querySelector("#rqCanvas");
    const empty = el.querySelector("#rqEmpty");
    if (!item) { canvas.hidden = true; empty.hidden = false; el.querySelector("#rqInfo").innerHTML = ""; el.querySelector("#rqActions").innerHTML = ""; return; }
    empty.hidden = true; canvas.hidden = false;
    M.boxes = (item.boxes || []).map((b) => ({ ...b }));
    M.selected = M.boxes.length ? 0 : -1;
    const token = ++M.imgToken;
    const img = new Image();
    img.onload = () => {
      if (token !== M.imgToken) return;
      M.img = img;
      const scale = Math.min(900 / img.naturalWidth, 560 / img.naturalHeight, 1);
      canvas.width = Math.round(img.naturalWidth * scale); canvas.height = Math.round(img.naturalHeight * scale);
      draw(el);
    };
    img.src = item.url;
    renderInfo(el); renderActions(el);
    el.querySelectorAll("#rqThumbs .thumb").forEach((n) => n.classList.toggle("active", Number(n.dataset.i) === M.index));
  }

  /* ---- canvas ---- */
  function normToPx(c, b) { return { x: (b.cx - b.w / 2) * c.width, y: (b.cy - b.h / 2) * c.height, w: b.w * c.width, h: b.h * c.height }; }
  function handles(r) { return [
    { n: "nw", x: r.x, y: r.y }, { n: "ne", x: r.x + r.w, y: r.y }, { n: "se", x: r.x + r.w, y: r.y + r.h }, { n: "sw", x: r.x, y: r.y + r.h },
    { n: "n", x: r.x + r.w / 2, y: r.y }, { n: "e", x: r.x + r.w, y: r.y + r.h / 2 }, { n: "s", x: r.x + r.w / 2, y: r.y + r.h }, { n: "w", x: r.x, y: r.y + r.h / 2 }]; }
  function pt(c, e) { const r = c.getBoundingClientRect(); return { x: (e.clientX - r.left) * (c.width / r.width), y: (e.clientY - r.top) * (c.height / r.height) }; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function className(cls) { return M.classes[cls] || `class ${cls}`; }

  function draw(el) {
    const c = el.querySelector("#rqCanvas"); const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    if (M.img) ctx.drawImage(M.img, 0, 0, c.width, c.height);
    M.boxes.forEach((b, i) => {
      const r = normToPx(c, b); const sel = i === M.selected;
      ctx.lineWidth = sel ? 3 : 2; ctx.strokeStyle = sel ? "#f59e0b" : "#16a34a"; ctx.strokeRect(r.x, r.y, r.w, r.h);
      const label = className(b.cls); ctx.font = "600 14px Inter, system-ui, sans-serif";
      const tw = ctx.measureText(label).width + 10;
      ctx.fillStyle = sel ? "#f59e0b" : "#16a34a"; ctx.fillRect(r.x, Math.max(0, r.y - 20), tw, 20);
      ctx.fillStyle = "#fff"; ctx.fillText(label, r.x + 5, Math.max(14, r.y - 6));
      if (sel) { ctx.fillStyle = "#fff"; ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 1.5;
        handles(r).forEach((h) => { ctx.fillRect(h.x - HANDLE / 2, h.y - HANDLE / 2, HANDLE, HANDLE); ctx.strokeRect(h.x - HANDLE / 2, h.y - HANDLE / 2, HANDLE, HANDLE); }); }
    });
  }

  function onDown(el, e) {
    if (!current()) return;
    const c = el.querySelector("#rqCanvas"); c.setPointerCapture(e.pointerId);
    const p = pt(c, e);
    if (M.selected >= 0) {
      const r = normToPx(c, M.boxes[M.selected]);
      const h = handles(r).find((h) => Math.abs(h.x - p.x) <= HANDLE && Math.abs(h.y - p.y) <= HANDLE);
      if (h) { M.drag = { mode: "resize", handle: h.n, orig: { ...M.boxes[M.selected] } }; return; }
    }
    for (let i = M.boxes.length - 1; i >= 0; i--) {
      const r = normToPx(c, M.boxes[i]);
      if (p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) {
        M.selected = i; M.drag = { mode: "move", start: p, orig: { ...M.boxes[i] } }; renderActions(el); draw(el); return;
      }
    }
    M.boxes.push({ cls: M.defaultClass, cx: p.x / c.width, cy: p.y / c.height, w: 0.002, h: 0.002 });
    M.selected = M.boxes.length - 1; M.drag = { mode: "new", origin: p }; renderActions(el); draw(el);
  }

  function onMove(el, e) {
    if (!M.drag) return;
    const c = el.querySelector("#rqCanvas"); const p = pt(c, e); const b = M.boxes[M.selected];
    if (M.drag.mode === "move") {
      const dx = (p.x - M.drag.start.x) / c.width, dy = (p.y - M.drag.start.y) / c.height;
      b.cx = clamp(M.drag.orig.cx + dx, b.w / 2, 1 - b.w / 2); b.cy = clamp(M.drag.orig.cy + dy, b.h / 2, 1 - b.h / 2);
    } else if (M.drag.mode === "resize") {
      const o = M.drag.orig; let l = o.cx - o.w / 2, rr = o.cx + o.w / 2, t = o.cy - o.h / 2, bo = o.cy + o.h / 2;
      const nx = p.x / c.width, ny = p.y / c.height, h = M.drag.handle;
      if (h.includes("w")) l = Math.min(nx, rr - 0.01); if (h.includes("e")) rr = Math.max(nx, l + 0.01);
      if (h.includes("n")) t = Math.min(ny, bo - 0.01); if (h.includes("s")) bo = Math.max(ny, t + 0.01);
      b.cx = (l + rr) / 2; b.cy = (t + bo) / 2; b.w = rr - l; b.h = bo - t;
    } else if (M.drag.mode === "new") {
      const x0 = Math.min(M.drag.origin.x, p.x), y0 = Math.min(M.drag.origin.y, p.y), x1 = Math.max(M.drag.origin.x, p.x), y1 = Math.max(M.drag.origin.y, p.y);
      b.cx = ((x0 + x1) / 2) / c.width; b.cy = ((y0 + y1) / 2) / c.height; b.w = Math.max(0.005, (x1 - x0) / c.width); b.h = Math.max(0.005, (y1 - y0) / c.height);
    }
    draw(el); renderInfo(el);
  }

  function onUp(el, e) {
    if (!M.drag) return;
    const b = M.boxes[M.selected];
    if (M.drag.mode === "new" && (b.w < 0.01 || b.h < 0.01)) { M.boxes.splice(M.selected, 1); M.selected = M.boxes.length ? 0 : -1; }
    M.drag = null; draw(el); renderInfo(el); renderActions(el);
  }

  function renderInfo(el) {
    const item = current(); if (!item) { el.querySelector("#rqInfo").innerHTML = ""; return; }
    el.querySelector("#rqInfo").innerHTML = `
      <span>Failure <b>${item.failureType || "?"}</b></span>
      <span>Model <b>${(item.model || "?").replace(".pt", "")}</b></span>
      <span>Boxes <b>${M.boxes.length}</b></span>
      ${item.note ? `<span>Note <b>${item.note}</b></span>` : ""}`;
  }

  function renderActions(el) {
    const item = current(); const root = el.querySelector("#rqActions");
    if (!item) { root.innerHTML = ""; return; }
    const options = M.classes.map((cName, i) => `<option value="${i}" ${M.boxes[M.selected] && M.boxes[M.selected].cls === i ? "selected" : ""}>${cName}</option>`).join("");
    root.innerHTML = `
      <label class="row" style="gap:6px;font-weight:600">Class
        <select id="rqClass" style="width:auto" ${M.selected < 0 ? "disabled" : ""}>${options || '<option>no classes</option>'}</select></label>
      <button class="btn ghost" id="rqDel" ${M.selected < 0 ? "disabled" : ""}>Delete box</button>
      <button class="btn" id="rqAdd" ${M.boxes.length ? "" : "disabled"}>Add to dataset</button>
      <button class="btn ghost" id="rqBg" title="No object here — use as a negative example">Add as background</button>
      <button class="btn danger" id="rqDiscard">Discard</button>
      <span class="muted">Drag to draw a box · or use <b>Add as background</b> if this was a false positive (no object)</span>`;
    root.querySelector("#rqClass").addEventListener("change", (e) => {
      M.defaultClass = Number(e.target.value);
      if (M.selected >= 0) { M.boxes[M.selected].cls = M.defaultClass; draw(el); }
    });
    root.querySelector("#rqDel").addEventListener("click", () => { if (M.selected >= 0) { M.boxes.splice(M.selected, 1); M.selected = M.boxes.length ? 0 : -1; draw(el); renderActions(el); renderInfo(el); } });
    root.querySelector("#rqAdd").addEventListener("click", () => promote(el).catch((x) => App.showError(x)));
    root.querySelector("#rqBg").addEventListener("click", () => addBackground(el).catch((x) => App.showError(x)));
    root.querySelector("#rqDiscard").addEventListener("click", () => discard(el).catch((x) => App.showError(x)));
  }

  async function addBackground(el) {
    const item = current(); if (!item) return;
    if (M.boxes.length && !window.confirm("Add this as a BACKGROUND image (no objects)? Any boxes you drew will be ignored.")) return;
    const res = await App.postJson("/api/retrain/background", { name: item.name });
    M.counts = res.counts;
    App.toast("Added as a background (negative) image");
    await App.refresh();
    await load(el);
  }

  async function promote(el) {
    const item = current(); if (!item) return;
    if (!M.boxes.length) { App.showError(new Error("Draw at least one box first.")); return; }
    const res = await App.postJson("/api/retrain/promote", { name: item.name, boxes: M.boxes });
    M.counts = res.counts;
    App.toast("Added to dataset for the next training round");
    await App.refresh();
    await load(el);
  }

  async function discard(el) {
    const item = current(); if (!item) return;
    if (!window.confirm("Discard this frame? It will not be used for training.")) return;
    const res = await App.postJson("/api/retrain/discard", { name: item.name });
    M.counts = res.counts;
    await App.refresh();
    await load(el);
  }

  App.registerTab({
    id: "retrain", label: "Retraining Queue",
    mount(el) { render(el); },
    onShow() { load(this.el).catch((e) => App.showError(e)); },
  });
})();
