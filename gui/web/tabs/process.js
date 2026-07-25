/* Tab 2 - Process Dataset */
(function () {
  const M = { groups: [] };

  function render(el) {
    el.innerHTML = `
      <div class="panel">
        <h2>2 · Process Dataset</h2>
        <p class="lead">Turn the selected clips into training frames and auto-draw a box on each one.
          Already-processed clips are skipped unless you re-process them.</p>
        <div class="grid3">
          <div>
            <label class="field" for="prStep">Frame interval</label>
            <select id="prStep">
              <option value="5">Every 5th frame (dense)</option>
              <option value="10">Every 10th frame</option>
              <option value="15" selected>Every 15th frame (default)</option>
              <option value="30">Every 30th frame (sparse)</option>
            </select>
          </div>
          <div>
            <label class="field">Options</label>
            <label class="row" style="gap:8px;font-weight:600"><input type="checkbox" id="prReprocess"> Re-extract processed clips</label>
          </div>
          <div style="align-self:end">
            <button class="btn" id="prRun">Process selected</button>
          </div>
        </div>
        <div class="spread" style="margin-top:16px">
          <h2 style="font-size:16px">Choose clips</h2>
          <div class="row">
            <button class="btn ghost sm" id="prNew">Select new</button>
            <button class="btn ghost sm" id="prAll">Select all</button>
            <button class="btn ghost sm" id="prNone">Clear</button>
          </div>
        </div>
        <div id="prList" class="select-list" style="margin-top:10px"></div>
      </div>

      <div class="panel" style="margin-top:16px">
        <h2 style="font-size:16px">Background images (no objects)</h2>
        <p class="lead">Add photos of <b>empty scenes</b> — the bare mat, the table, your hand with nothing in it,
          cluttered rooms. These become negative examples (empty labels) that teach the model what "nothing" looks
          like, which stops it from drawing boxes when no object is present. Aim for roughly 10% of your dataset.</p>
        <p class="muted">Tip: you can also drop background <b>pictures</b> straight into
          <code>data/raw_videos/background/</code> and hit <b>Process</b> above — they are folded in automatically as
          negatives and never trained as an object. The uploader below does the same thing for individual files.</p>
        <p class="muted">Currently <b id="prBgCount">0</b> background image(s) in the dataset.</p>
        <div class="row" style="margin-top:8px">
          <input type="file" id="prBgFiles" accept="image/*" multiple>
          <button class="btn" id="prBgAdd">Add background images</button>
        </div>
        <p class="muted" id="prBgMsg"></p>
      </div>`;

    el.querySelector("#prRun").addEventListener("click", () => run(el).catch((e) => App.showError(e)));
    el.querySelector("#prNew").addEventListener("click", () => setAll(el, "new"));
    el.querySelector("#prAll").addEventListener("click", () => setAll(el, "all"));
    el.querySelector("#prNone").addEventListener("click", () => setAll(el, "none"));
    el.querySelector("#prBgAdd").addEventListener("click", () => addBackground(el).catch((e) => App.showError(e)));
  }

  async function addBackground(el) {
    const input = el.querySelector("#prBgFiles");
    const files = Array.from(input.files || []);
    if (!files.length) throw new Error("Choose one or more background images first.");
    const form = new FormData();
    files.forEach((f) => form.append("files", f, f.name));
    const res = await fetch("/api/dataset/background", { method: "POST", body: form });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || "Upload failed");
    input.value = "";
    el.querySelector("#prBgMsg").textContent =
      `Added ${payload.added} background image(s)${payload.skipped ? `, skipped ${payload.skipped} non-image` : ""}.`;
    el.querySelector("#prBgCount").textContent = payload.background;
    App.refresh().catch(() => {});
  }

  async function load(el) {
    const data = await App.api("/api/dataset/clips");
    M.groups = data.groups;
    renderList(el);
  }

  function renderList(el) {
    const root = el.querySelector("#prList");
    if (!M.groups.length) { root.innerHTML = '<div class="select-item muted">No clips yet — upload some in the Upload Clips tab.</div>'; return; }
    let html = "";
    for (const g of M.groups) {
      html += `<div class="select-item grp">${g.class} · ${g.clips.length} clip(s)</div>`;
      for (const c of g.clips) {
        html += `<label class="select-item">
          <input type="checkbox" class="pr-check" data-key="${c.key}" ${c.extracted ? "" : "checked"}>
          <span style="flex:1">${c.video}</span>
          ${c.extracted ? '<span class="pill ok">frames ✓</span>' : '<span class="pill new">new</span>'}
        </label>`;
      }
    }
    root.innerHTML = html;
  }

  function setAll(el, mode) {
    const checks = el.querySelectorAll(".pr-check");
    const extractedKeys = new Set();
    M.groups.forEach((g) => g.clips.forEach((c) => { if (c.extracted) extractedKeys.add(c.key); }));
    checks.forEach((cb) => {
      if (mode === "all") cb.checked = true;
      else if (mode === "none") cb.checked = false;
      else cb.checked = !extractedKeys.has(cb.dataset.key); // new
    });
  }

  async function run(el) {
    const selected = Array.from(el.querySelectorAll(".pr-check:checked")).map((cb) => cb.dataset.key);
    if (!selected.length) throw new Error("Select at least one clip to process.");
    const totalClips = el.querySelectorAll(".pr-check").length;
    const body = {
      frameStep: Number(el.querySelector("#prStep").value),
      reprocess: el.querySelector("#prReprocess").checked,
      select: selected,
      all: selected.length === totalClips,
    };
    const res = await App.postJson("/api/dataset/process", body);
    App.startedJob(res.job);
    App.toast("Processing started");
  }

  App.registerTab({
    id: "process", label: "Process Dataset",
    mount(el) { render(el); },
    onShow() { load(this.el).catch((e) => App.showError(e)); },
    onState(state) {
      const count = this.el.querySelector("#prBgCount");
      if (count && state) count.textContent = state.totals.background || 0;
    },
    onJobDone(job) { if (job.kind === "process_dataset") load(this.el).catch(() => {}); },
  });
})();
