/* Tab 4 - Train Model */
(function () {
  const M = { models: [], plots: [], active: null };
  const PRESETS = { quick: 10, normal: 50, strong: 100 };

  function render(el) {
    el.innerHTML = `
      <div class="panel">
        <h2>4 · Train Model</h2>
        <p class="lead">Train a detector on your reviewed dataset. Each run is saved as a new version, so a
          new model never overwrites your current one. The train/val split is by whole video.</p>

        <label class="field">Training preset</label>
        <div class="preset-grid" id="trPresets">
          <div class="preset" data-preset="quick"><b>Quick Test</b><span>10 epochs · fast sanity check</span></div>
          <div class="preset active" data-preset="normal"><b>Normal</b><span>50 epochs · balanced</span></div>
          <div class="preset" data-preset="strong"><b>Stronger Model</b><span>100 epochs · slow, best</span></div>
        </div>

        <div class="grid2" style="margin-top:16px">
          <div>
            <label class="field">Train / Val split (by video) — <span id="trSplitLabel">80% / 20%</span></label>
            <input type="range" id="trSplit" min="50" max="95" step="5" value="80">
          </div>
          <div>
            <label class="field">After training</label>
            <label class="row" style="gap:8px;font-weight:600"><input type="checkbox" id="trActive"> Set as active model</label>
            <p class="muted" style="margin:6px 0 0;font-size:12px">Unchecked keeps your current active model. The first model is always set active.</p>
          </div>
        </div>

        <details class="advanced" style="margin-top:16px">
          <summary>Advanced settings</summary>
          <div class="grid3" style="margin:12px 0">
            <div><label class="field">Epochs</label><input type="number" id="trEpochs" min="1" value="50"></div>
            <div><label class="field">Image size</label><input type="number" id="trImgsz" min="128" step="32" value="640"></div>
            <div><label class="field">Batch size</label><input type="number" id="trBatch" min="1" value="8"></div>
            <div><label class="field">Base model</label>
              <select id="trBase"><option value="yolov8n.pt">yolov8n.pt (nano, fast)</option>
                <option value="yolov8s.pt">yolov8s.pt (small)</option>
                <option value="yolov8m.pt">yolov8m.pt (medium)</option></select></div>
            <div><label class="field">Device</label>
              <select id="trDevice"><option value="cpu">cpu</option><option value="mps">mps (Apple GPU)</option><option value="0">0 (GPU)</option></select></div>
          </div>
        </details>

        <div class="row" style="margin-top:14px">
          <button class="btn" id="trRun">Start training</button>
          <span class="muted" id="trHint"></span>
        </div>
      </div>

      <div class="panel" style="margin-top:16px">
        <div class="spread"><h2 style="font-size:16px">Trained models</h2><button class="btn ghost sm" id="trRefresh">Refresh</button></div>
        <div id="trModels" style="margin-top:12px"></div>
        <div id="trPlots" style="margin-top:16px"></div>
      </div>`;

    el.querySelectorAll("#trPresets .preset").forEach((node) => node.addEventListener("click", () => {
      el.querySelectorAll("#trPresets .preset").forEach((p) => p.classList.remove("active"));
      node.classList.add("active");
      el.querySelector("#trEpochs").value = PRESETS[node.dataset.preset];
    }));
    el.querySelector("#trSplit").addEventListener("input", (e) => {
      const t = Number(e.target.value); el.querySelector("#trSplitLabel").textContent = `${t}% / ${100 - t}%`;
    });
    el.querySelector("#trEpochs").addEventListener("input", () =>
      el.querySelectorAll("#trPresets .preset").forEach((p) => p.classList.remove("active")));
    el.querySelector("#trRun").addEventListener("click", () => run(el).catch((e) => App.showError(e)));
    el.querySelector("#trRefresh").addEventListener("click", () => load(el).catch((e) => App.showError(e)));
  }

  async function run(el) {
    const trainPct = Number(el.querySelector("#trSplit").value);
    const body = {
      epochs: Number(el.querySelector("#trEpochs").value),
      imgsz: Number(el.querySelector("#trImgsz").value),
      batch: Number(el.querySelector("#trBatch").value),
      baseModel: el.querySelector("#trBase").value,
      device: el.querySelector("#trDevice").value,
      valFraction: (100 - trainPct) / 100,
      setActive: el.querySelector("#trActive").checked,
    };
    const res = await App.postJson("/api/train", body);
    App.startedJob(res.job);
    App.toast("Training started — watch the job log below");
  }

  async function load(el) {
    const data = await App.api("/api/models");
    M.models = data.models; M.plots = data.plots; M.active = data.active;
    renderModels(el); renderPlots(el);
  }

  function fmtDate(ts) { return ts ? new Date(ts * 1000).toLocaleString() : "—"; }
  function metric(m, k) { return (m && m[k] != null) ? m[k] : "—"; }

  function renderModels(el) {
    const root = el.querySelector("#trModels");
    if (!M.models.length) { root.innerHTML = '<p class="muted">No models yet. Train one above.</p>'; return; }
    root.innerHTML = `<table class="model-table">
      <thead><tr><th>Version</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>Epochs</th><th>Created</th><th></th></tr></thead>
      <tbody>${M.models.map((m) => `<tr>
        <td><b>${m.file.replace(".pt", "")}</b> ${m.active ? '<span class="pill ok">active</span>' : ""}</td>
        <td>${metric(m.metrics, "precision")}</td>
        <td>${metric(m.metrics, "recall")}</td>
        <td>${metric(m.metrics, "map50")}</td>
        <td>${m.epochs || "—"}</td>
        <td class="muted">${fmtDate(m.created)}</td>
        <td>${m.active ? "" : `<button class="btn ghost sm act-activate" data-file="${m.file}">Make active</button>`}</td>
      </tr>`).join("")}</tbody></table>`;
    root.querySelectorAll(".act-activate").forEach((node) => node.addEventListener("click", () =>
      activate(el, node.dataset.file).catch((e) => App.showError(e))));
  }

  function renderPlots(el) {
    const root = el.querySelector("#trPlots");
    if (!M.plots.length) { root.innerHTML = ""; return; }
    root.innerHTML = '<label class="field">Latest run — metrics &amp; loss curves</label>' +
      `<div class="row" style="align-items:flex-start">${M.plots.map((name) =>
        `<a href="/api/models/plot/${name}" target="_blank" rel="noopener" title="${name}">
          <img src="/api/models/plot/${name}" alt="${name}" style="max-width:340px;border:1px solid var(--line);border-radius:8px"></a>`).join("")}</div>`;
  }

  async function activate(el, file) {
    if (!window.confirm(`Make ${file.replace(".pt", "")} the active model? The rover/detector will use it.`)) return;
    await App.postJson("/api/models/activate", { file });
    await App.refresh();
    await load(el);
    App.toast("Active model updated");
  }

  App.registerTab({
    id: "train", label: "Train Model",
    mount(el) { render(el); },
    onShow() { load(this.el).catch((e) => App.showError(e)); },
    onState(state) {
      const hint = this.el.querySelector("#trHint");
      if (hint && state) hint.textContent = `${state.totals.images} images · ${state.totals.unreviewed} still need review`;
    },
    onJobDone(job) { if (job.kind === "train_model") load(this.el).catch(() => {}); },
  });
})();
