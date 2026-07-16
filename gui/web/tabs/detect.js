/* Tab 5 - Test Detector: run a model live and capture failure frames */
(function () {
  const M = {
    models: [], stream: null, timer: null, busy: false,
    source: "webcam", detections: [], fpsTimes: [], savedThisSession: 0,
  };

  function render(el) {
    el.innerHTML = `
      <div class="panel">
        <h2>5 · Test Detector</h2>
        <p class="lead">Point a camera (or play a video) at your objects and watch the model detect them live.
          When it gets something wrong, capture the frame — it goes to the Retraining Queue to be corrected.</p>

        <div class="detect-controls">
          <div class="field-group"><label class="field">Model version</label><select id="dtModel"></select></div>
          <div class="field-group"><label class="field">Source</label>
            <select id="dtSource"><option value="webcam">Webcam</option><option value="video">Video file</option></select></div>
          <div class="field-group" id="dtFileWrap" hidden><label class="field">Video file</label>
            <input type="file" id="dtFile" accept="video/*"></div>
          <div class="field-group"><label class="field">Confidence — <span id="dtConfLabel">0.25</span></label>
            <input type="range" id="dtConf" min="0.05" max="0.9" step="0.05" value="0.25"></div>
        </div>

        <div class="row" style="margin-top:12px">
          <button class="btn" id="dtStart">Start</button>
          <button class="btn ghost" id="dtStop" disabled>Stop</button>
          <span class="cam-status"><span class="dot" id="dtDot"></span><span id="dtStatus">Idle</span></span>
          <span class="muted" id="dtFps"></span>
        </div>

        <div class="cam-stage" id="dtStage" style="margin-top:14px">
          <video id="dtVideo" autoplay playsinline muted></video>
          <canvas id="dtOverlay"></canvas>
          <div class="no-obj" id="dtNoObj" hidden>No objects in frame</div>
          <div class="cam-hint" id="dtHint">Choose a model and press <b>Start</b>.</div>
        </div>

        <div class="detect-grid" style="margin-top:14px">
          <div>
            <label class="field">Detections</label>
            <div id="dtList" class="detect-list"><p class="muted">Nothing detected yet.</p></div>
          </div>
          <div>
            <label class="field">Capture a mistake for retraining</label>
            <div class="fail-btns">
              <button class="btn ghost" data-type="false_positive" disabled title="It drew a box on nothing">False Positive</button>
              <button class="btn ghost" data-type="missed" disabled>Missed Object</button>
              <button class="btn ghost" data-type="wrong_class" disabled>Wrong Class</button>
              <button class="btn ghost" data-type="bad_box" disabled>Bad Box</button>
              <button class="btn" data-type="manual" disabled>Save Frame</button>
            </div>
            <label class="field" style="margin-top:10px">Optional note</label>
            <input type="text" id="dtNote" placeholder="e.g. bit at an angle, far away">
            <p class="muted" id="dtSaved" style="margin-top:8px"></p>
          </div>
        </div>
      </div>`;

    el.querySelector("#dtConf").addEventListener("input", (e) =>
      el.querySelector("#dtConfLabel").textContent = Number(e.target.value).toFixed(2));
    el.querySelector("#dtSource").addEventListener("change", (e) => {
      M.source = e.target.value;
      el.querySelector("#dtFileWrap").hidden = M.source !== "video";
      stop(el);
    });
    el.querySelector("#dtFile").addEventListener("change", (e) => loadVideoFile(el, e.target.files[0]));
    el.querySelector("#dtStart").addEventListener("click", () => start(el).catch((x) => App.showError(x)));
    el.querySelector("#dtStop").addEventListener("click", () => stop(el));
    el.querySelectorAll(".fail-btns button").forEach((b) =>
      b.addEventListener("click", () => saveFailure(el, b.dataset.type).catch((x) => App.showError(x))));
  }

  async function loadModels(el) {
    const data = await App.api("/api/detect/models");
    M.models = data.models;
    const sel = el.querySelector("#dtModel");
    if (!M.models.length) {
      sel.innerHTML = '<option value="">No models — train one first</option>';
      el.querySelector("#dtStart").disabled = true;
      return;
    }
    el.querySelector("#dtStart").disabled = false;
    sel.innerHTML = M.models.map((m) => {
      const map = m.metrics && m.metrics.map50 != null ? ` · mAP50 ${m.metrics.map50}` : "";
      return `<option value="${m.file}" ${m.active ? "selected" : ""}>${m.file.replace(".pt", "")}${m.active ? " (active)" : ""}${map}</option>`;
    }).join("");
  }

  function setStatus(el, text, cls) {
    el.querySelector("#dtStatus").textContent = text;
    el.querySelector("#dtDot").className = "dot" + (cls ? " " + cls : "");
  }
  function failButtons(el, enabled) {
    el.querySelectorAll(".fail-btns button").forEach((b) => (b.disabled = !enabled));
  }

  async function start(el) {
    const video = el.querySelector("#dtVideo");
    if (M.source === "webcam") {
      try {
        M.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch (error) {
        setStatus(el, "Could not open the camera: " + (error.message || error), "failed");
        return;
      }
      video.srcObject = M.stream;
      video.muted = true;
      await video.play().catch(() => {});
    } else {
      if (!video.src) { setStatus(el, "Choose a video file first.", "failed"); return; }
      video.muted = true;
      await video.play().catch(() => {});
    }
    el.querySelector("#dtHint").hidden = true;
    el.querySelector("#dtStart").disabled = true;
    el.querySelector("#dtStop").disabled = false;
    failButtons(el, true);
    setStatus(el, "Running — detecting…", "running");
    M.fpsTimes = [];
    M.timer = setInterval(() => detectFrame(el).catch(() => {}), 200);
  }

  function stop(el) {
    if (M.timer) { clearInterval(M.timer); M.timer = null; }
    if (M.stream) { M.stream.getTracks().forEach((t) => t.stop()); M.stream = null; }
    const video = el.querySelector("#dtVideo");
    if (M.source === "webcam") video.srcObject = null; else video.pause();
    const overlay = el.querySelector("#dtOverlay");
    overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
    el.querySelector("#dtStart").disabled = false;
    el.querySelector("#dtStop").disabled = true;
    el.querySelector("#dtHint").hidden = false;
    el.querySelector("#dtNoObj").hidden = true;
    el.querySelector("#dtFps").textContent = "";
    failButtons(el, false);
    setStatus(el, "Idle");
  }

  function loadVideoFile(el, file) {
    if (!file) return;
    const video = el.querySelector("#dtVideo");
    video.srcObject = null;
    video.src = URL.createObjectURL(file);
    video.loop = true;
    el.querySelector("#dtHint").hidden = true;
    setStatus(el, `Loaded ${file.name} — press Start.`);
  }

  function frameBlob(el) {
    const video = el.querySelector("#dtVideo");
    if (!video.videoWidth) return null;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.75));
  }

  async function detectFrame(el) {
    if (M.busy) return;
    const video = el.querySelector("#dtVideo");
    if (!video.videoWidth) return;
    M.busy = true;
    const t0 = performance.now();
    try {
      const blob = await frameBlob(el);
      if (!blob) return;
      const conf = el.querySelector("#dtConf").value;
      const model = el.querySelector("#dtModel").value;
      const res = await fetch(`/api/detect?conf=${conf}&model=${encodeURIComponent(model)}`, { method: "POST", body: blob });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "Detection failed");
      M.detections = payload.detections || [];
      drawDetections(el, M.detections);
      renderList(el);
      recordFps(el, t0);
      const n = M.detections.length;
      el.querySelector("#dtNoObj").hidden = n > 0;
      setStatus(el, n ? `${n} object(s) in frame` : "No objects in frame", n ? "running" : "");
    } catch (error) {
      setStatus(el, "Detector error: " + (error.message || error), "failed");
    } finally {
      M.busy = false;
    }
  }

  function recordFps(el, t0) {
    M.fpsTimes.push(performance.now() - t0);
    if (M.fpsTimes.length > 10) M.fpsTimes.shift();
    const avg = M.fpsTimes.reduce((a, b) => a + b, 0) / M.fpsTimes.length;
    el.querySelector("#dtFps").textContent = avg ? `~${(1000 / avg).toFixed(1)} FPS` : "";
  }

  function drawDetections(el, detections) {
    const video = el.querySelector("#dtVideo");
    const overlay = el.querySelector("#dtOverlay");
    const rect = video.getBoundingClientRect();
    overlay.width = rect.width; overlay.height = rect.height;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const vw = video.videoWidth || 4, vh = video.videoHeight || 3;
    const scale = Math.min(overlay.width / vw, overlay.height / vh);
    const dw = vw * scale, dh = vh * scale;
    const ox = (overlay.width - dw) / 2, oy = (overlay.height - dh) / 2;
    ctx.lineWidth = 3; ctx.font = "600 15px Inter, system-ui, sans-serif";
    detections.forEach((d) => {
      const x = ox + d.x * dw, y = oy + d.y * dh, w = d.w * dw, h = d.h * dh;
      ctx.strokeStyle = "#16a34a"; ctx.strokeRect(x, y, w, h);
      const label = `${d.name} ${(d.conf * 100).toFixed(0)}%`;
      const tw = ctx.measureText(label).width + 10;
      ctx.fillStyle = "#16a34a"; ctx.fillRect(x, Math.max(0, y - 22), tw, 22);
      ctx.fillStyle = "#fff"; ctx.fillText(label, x + 5, Math.max(14, y - 6));
    });
  }

  function renderList(el) {
    const root = el.querySelector("#dtList");
    if (!M.detections.length) { root.innerHTML = '<p class="muted">Nothing detected.</p>'; return; }
    root.innerHTML = M.detections.map((d) =>
      `<div class="det-row"><span class="pill ok">${d.name}</span><b>${(d.conf * 100).toFixed(0)}%</b>
        <code>x${d.x.toFixed(2)} y${d.y.toFixed(2)} w${d.w.toFixed(2)} h${d.h.toFixed(2)}</code></div>`).join("");
  }

  async function saveFailure(el, type) {
    const blob = await frameBlob(el);
    if (!blob) { App.showError(new Error("No frame to capture yet.")); return; }
    const form = new FormData();
    form.append("failureType", type);
    form.append("model", el.querySelector("#dtModel").value || "active");
    form.append("note", el.querySelector("#dtNote").value || "");
    form.append("frame", blob, "frame.jpg");
    const res = await fetch("/api/detect/save-failure", { method: "POST", body: form });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || "Save failed");
    M.savedThisSession += 1;
    el.querySelector("#dtSaved").textContent =
      `Saved ${M.savedThisSession} frame(s) this session · ${payload.counts.pending} pending in queue`;
    App.refresh().catch(() => {});
    App.toast("Frame saved to retraining queue");
  }

  App.registerTab({
    id: "detect", label: "Test Detector",
    mount(el) { render(el); },
    onShow() { loadModels(this.el).catch((e) => App.showError(e)); },
    onHide() { stop(this.el); },
  });

  // Stop the camera when switching away from this tab.
  const origShowTab = App.showTab.bind(App);
  App.showTab = function (id) {
    if (App.activeTab === "detect" && id !== "detect") {
      const t = App.tab("detect"); if (t && t.el) stop(t.el);
    }
    origShowTab(id);
  };
})();
