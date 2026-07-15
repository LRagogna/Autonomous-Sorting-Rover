/* Tab 7 - Deploy Model */
(function () {
  const M = { status: null };
  const MANUAL = {
    cameraReady: "Camera code ready",
    inferenceReady: "Inference script ready",
    roverControlReady: "Rover control script ready",
  };

  function render(el) {
    el.innerHTML = `
      <div class="panel">
        <h2>7 · Deploy Model</h2>
        <p class="lead">Choose the model the rover will run, then copy it into the deploy bundle to move onto the Raspberry Pi.</p>

        <div class="grid2" style="margin-top:14px">
          <div>
            <label class="field">Active model</label>
            <select id="dpModel"></select>
            <p class="muted" style="margin:8px 0 0">Active model path</p>
            <div class="deploy-path" id="dpActivePath">—</div>
          </div>
          <div>
            <label class="field">Deploy bundle</label>
            <div class="deploy-path" id="dpDeployPath">—</div>
            <div class="row" style="margin-top:10px">
              <button class="btn" id="dpCopy">Copy model to deploy folder</button>
            </div>
            <p class="muted" id="dpCopied" style="margin-top:8px"></p>
          </div>
        </div>
      </div>

      <div class="panel" style="margin-top:16px">
        <h2 style="font-size:16px">Deployment checklist</h2>
        <ul class="deploy-checklist" id="dpChecklist"></ul>
      </div>

      <div class="panel" style="margin-top:16px">
        <h2 style="font-size:16px">Raspberry Pi CPU (now)</h2>
        <p class="lead">A normal Raspberry Pi runs the <code>.pt</code> model directly with the same inference code
          used on your laptop. Copy the deploy bundle to the Pi and run the detector.</p>
        <h2 style="font-size:16px;margin-top:14px">Coral Edge TPU (future)</h2>
        <div class="coral-note">
          A Google Coral TPU will <b>not</b> use the <code>.pt</code> file directly. It needs the model exported and
          quantized to <b>TensorFlow Lite</b>, then compiled with the <b>Edge TPU compiler</b> into a
          <code>_edgetpu.tflite</code> file. This pipeline (export → INT8 quantization → edgetpu_compiler) is a
          planned follow-up and is not wired up yet.
        </div>
      </div>`;

    el.querySelector("#dpModel").addEventListener("change", (e) => activate(el, e.target.value).catch((x) => App.showError(x)));
    el.querySelector("#dpCopy").addEventListener("click", () => copy(el).catch((x) => App.showError(x)));
  }

  async function load(el) {
    M.status = await App.api("/api/deploy/status");
    const s = M.status;
    const sel = el.querySelector("#dpModel");
    if (!s.versions.length) {
      sel.innerHTML = '<option value="">No models — train one first</option>';
      el.querySelector("#dpCopy").disabled = true;
    } else {
      sel.innerHTML = s.versions.map((v) => `<option value="${v.file}" ${v.active ? "selected" : ""}>${v.file.replace(".pt", "")}${v.active ? " (active)" : ""}</option>`).join("");
      el.querySelector("#dpCopy").disabled = false;
    }
    el.querySelector("#dpActivePath").textContent = s.activePath || "no active model yet";
    el.querySelector("#dpDeployPath").textContent = (s.deployDir || "deploy") + "/";
    el.querySelector("#dpCopied").textContent = s.deployedModel ? `Deployed: ${s.deployedModel}` : "Not copied to the deploy folder yet.";
    renderChecklist(el);
  }

  function renderChecklist(el) {
    const c = M.status.checklist;
    const auto = (done, label) => `<li><span class="mark-box ${done ? "on" : ""}">${done ? "✓" : ""}</span>${label} <span class="muted">(auto)</span></li>`;
    const manual = (key, label) => `<li><label class="row" style="gap:10px;cursor:pointer">
        <input type="checkbox" data-key="${key}" ${c[key] ? "checked" : ""}> ${label}</label></li>`;
    el.querySelector("#dpChecklist").innerHTML =
      auto(c.modelSelected, "Model selected") +
      Object.entries(MANUAL).map(([k, label]) => manual(k, label)).join("") +
      auto(c.modelCopied, "Model copied to deploy folder");
    el.querySelectorAll('#dpChecklist input[type="checkbox"]').forEach((box) =>
      box.addEventListener("change", () => App.postJson("/api/deploy/checklist", { key: box.dataset.key, value: box.checked }).catch((x) => App.showError(x))));
  }

  async function activate(el, file) {
    if (!file) return;
    await App.postJson("/api/deploy/activate", { file });
    await App.refresh();
    await load(el);
    App.toast("Active model updated");
  }

  async function copy(el) {
    const res = await App.postJson("/api/deploy/copy", {});
    App.toast("Model copied to deploy folder");
    await App.refresh();
    await load(el);
  }

  App.registerTab({
    id: "deploy", label: "Deploy",
    mount(el) { render(el); },
    onShow() { load(this.el).catch((e) => App.showError(e)); },
  });
})();
