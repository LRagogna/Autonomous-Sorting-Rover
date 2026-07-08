/* Shared front-end framework for the rover control center.
 *
 * Tab modules call App.registerTab({...}) at load time. App.init() (called at
 * the bottom of index.html) builds the tab nav, mounts each tab once, wires the
 * sidebar + job log, and starts polling for global state and job progress.
 */
(function () {
  const App = {
    tabs: [],
    activeTab: null,
    state: null,
    lastJobStatus: null,

    registerTab(def) { this.tabs.push(def); },
    el(id) { return document.getElementById(id); },

    async api(path, options = {}) {
      const response = await fetch(path, options);
      let payload = {};
      try { payload = await response.json(); } catch (_) { /* non-JSON */ }
      if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
      return payload;
    },

    async postJson(path, body) {
      return this.api(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
    },

    toast(message, isError) {
      const node = this.el("toast");
      node.textContent = message;
      node.className = "toast" + (isError ? " err" : "");
      node.hidden = false;
      clearTimeout(this._toastTimer);
      this._toastTimer = setTimeout(() => { node.hidden = true; }, isError ? 5000 : 2600);
    },
    showError(error) { this.toast(error.message || String(error), true); },

    /* ---- tab management ---- */
    init() {
      const nav = this.el("tabNav");
      const panels = this.el("tabPanels");
      this.tabs.forEach((tab, index) => {
        const button = document.createElement("button");
        button.dataset.tab = tab.id;
        button.innerHTML = `<span class="step-num">${index + 1}</span> ${tab.label}`;
        button.addEventListener("click", () => this.showTab(tab.id));
        nav.appendChild(button);

        const panel = document.createElement("div");
        panel.className = "panel-wrap";
        panel.dataset.tab = tab.id;
        panel.hidden = true;
        panels.appendChild(panel);
        tab.el = panel;
        if (tab.mount) tab.mount(panel);
      });

      this._wireChrome();
      this.showTab(this.tabs[0].id);
      this.refresh().catch((e) => this.showError(e));
      setInterval(() => this.pollJob().catch(() => {}), 1200);
    },

    showTab(id) {
      this.activeTab = id;
      for (const button of this.el("tabNav").children) {
        button.classList.toggle("active", button.dataset.tab === id);
      }
      for (const panel of this.el("tabPanels").children) {
        panel.hidden = panel.dataset.tab !== id;
      }
      const tab = this.tabs.find((t) => t.id === id);
      if (tab && tab.onShow) tab.onShow(this.state);
    },

    tab(id) { return this.tabs.find((t) => t.id === id); },

    /* ---- global state + sidebar ---- */
    async refresh() {
      const state = await this.api("/api/state");
      this.state = state;
      this.renderSidebar(state);
      for (const tab of this.tabs) {
        if (tab.onState) tab.onState(state);
      }
      return state;
    },

    renderSidebar(s) {
      this.el("projectName").textContent = s.project.name;
      const model = s.model.active ? s.model.active.replace(".pt", "") : "none yet";
      this.el("sbProject").innerHTML = `
        <div class="proj-name">${s.project.name}</div>
        <div style="margin-top:10px">
          <div class="kv"><span>Active model</span><b>${model}</b></div>
          <div class="kv"><span>Latest version</span><b>v${s.model.latestVersion} (${s.model.count} saved)</b></div>
          <div class="kv"><span>Classes</span><b>${s.totals.classes}</b></div>
          <div class="kv"><span>Images</span><b>${s.totals.images}</b></div>
          <div class="kv"><span>Reviewed</span><b>${s.totals.reviewed} / ${s.totals.images}</b></div>
          <div class="kv"><span>Needs review</span><b>${s.totals.unreviewed}</b></div>
          <div class="kv"><span>Rejected</span><b>${s.totals.failed}</b></div>
          <div class="kv"><span>Retrain queue</span><b>${s.totals.retrainPending}</b></div>
        </div>`;

      const classes = this.el("sbClasses");
      if (!s.perClass.length) {
        classes.innerHTML = '<p class="muted">No classes yet. Add a clip to begin.</p>';
      } else {
        classes.innerHTML = s.perClass.map((c) => {
          const total = c.images || 1;
          const pct = Math.round((c.reviewed / total) * 100);
          return `<div class="class-stat">
            <div class="name">${c.class}</div>
            <div class="bar"><span style="width:${pct}%"></span></div>
            <div class="meta"><span>${c.images} imgs · ${c.reviewed}✓ / ${c.unreviewed}•</span><span>${c.clips} clips</span></div>
          </div>`;
        }).join("");
      }

      const steps = [
        ["clipsUploaded", "Clips uploaded"],
        ["framesExtracted", "Frames extracted"],
        ["labelsGenerated", "Labels generated"],
        ["labelsReviewed", "Labels reviewed"],
        ["datasetBuilt", "Dataset built"],
        ["modelTrained", "Model trained"],
        ["modelTested", "Model tested"],
        ["modelDeployed", "Model deployed"],
      ];
      this.el("sbChecklist").innerHTML = steps.map(([key, label]) => {
        const done = !!s.checklist[key];
        return `<li class="${done ? "done" : ""}"><span class="mark">${done ? "✓" : ""}</span>${label}</li>`;
      }).join("");
    },

    /* ---- job log ---- */
    async pollJob() {
      const payload = await this.api("/api/job");
      this.renderJob(payload.current);
    },

    startedJob(job) {
      // Called by tabs right after starting a job so the strip appears instantly.
      this.renderJob(job);
    },

    renderJob(job) {
      const strip = this.el("jobStrip");
      const dot = this.el("jobDot");
      const status = this.el("jobStatus");
      dot.className = "dot";
      if (!job) { status.textContent = "Idle"; return; }

      dot.classList.add(job.status);
      status.textContent = `${job.kind.replace(/_/g, " ")}: ${job.status}`;
      strip.hidden = false;
      this.el("stripDot").className = "dot " + job.status;
      this.el("stripKind").textContent = job.kind.replace(/_/g, " ");
      this.el("stripStatus").textContent = job.status;

      const log = this.el("jobLog");
      const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 50;
      log.textContent = (job.log && job.log.length) ? job.log.join("\n") : "Starting…";
      if (nearBottom) log.scrollTop = log.scrollHeight;

      this._renderProgress(job);
      this._renderSummary(job);

      if (this.lastJobStatus === "running" && job.status !== "running") {
        this.refresh().catch((e) => this.showError(e));
        for (const tab of this.tabs) { if (tab.onJobDone) tab.onJobDone(job); }
        this.toast(job.status === "succeeded" ? `${job.kind.replace(/_/g, " ")} finished` : `${job.kind.replace(/_/g, " ")} failed`, job.status !== "succeeded");
      }
      this.lastJobStatus = job.status;
    },

    _renderProgress(job) {
      const wrap = this.el("stripProgress");
      const bar = this.el("stripBar");
      if (job.status !== "running") { wrap.hidden = true; wrap.classList.remove("indet"); return; }
      wrap.hidden = false;
      const lines = job.log || [];
      let pct = null;
      if (job.kind === "train_model") {
        for (let i = lines.length - 1; i >= 0 && i > lines.length - 60; i--) {
          const m = lines[i].match(/^\s*(\d+)\/(\d+)\b/);
          if (m) { pct = Math.min(99, Math.round((+m[1] / +m[2]) * 100)); break; }
        }
      } else if (job.kind === "process_dataset") {
        if (lines.some((l) => l.includes("Step 2/2"))) pct = 60;
        else if (lines.some((l) => l.includes("Step 1/2"))) pct = 20;
      }
      if (pct == null) { wrap.classList.add("indet"); bar.style.width = ""; }
      else { wrap.classList.remove("indet"); bar.style.width = pct + "%"; }
    },

    _renderSummary(job) {
      const node = this.el("jobSummary");
      const s = job.summary;
      if (!s) { node.hidden = true; return; }
      node.hidden = false;
      if (s.type === "process") {
        node.innerHTML = [
          ["extracted", "Frames extracted"], ["labels", "Labels created"],
          ["failed", "Auto-label failures"], ["review", "Need review"],
        ].map(([k, label]) => `<div class="box"><b>${s[k] ?? 0}</b><span>${label}</span></div>`).join("");
      } else if (s.type === "train") {
        node.innerHTML = `<div class="box"><b>${s.file || "?"}</b><span>saved model</span></div>
          <div class="box"><b>v${s.version || "?"}</b><span>version</span></div>
          <div class="box"><b>${s.active === "True" ? "yes" : "no"}</b><span>set active</span></div>`;
      }
    },

    /* ---- video preview modal ---- */
    openVideo(title, url) {
      this.el("videoTitle").textContent = title;
      const player = this.el("videoPlayer");
      player.src = url;
      this.el("videoModal").hidden = false;
      player.load();
    },
    closeVideo() {
      const player = this.el("videoPlayer");
      player.pause();
      player.removeAttribute("src");
      player.load();
      this.el("videoModal").hidden = true;
    },

    _wireChrome() {
      this.el("videoClose").addEventListener("click", () => this.closeVideo());
      this.el("videoModal").addEventListener("click", (e) => {
        if (e.target === this.el("videoModal")) this.closeVideo();
      });
      this.el("stripJump").addEventListener("click", () => {
        const log = this.el("jobLog"); log.scrollTop = log.scrollHeight;
      });
      this.el("stripClose").addEventListener("click", () => { this.el("jobStrip").hidden = true; });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !this.el("videoModal").hidden) this.closeVideo();
      });
    },
  };

  window.App = App;
})();
