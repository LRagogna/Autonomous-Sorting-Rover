/* Tab 1 - Upload Clips */
(function () {
  const M = { files: [], groups: [], classes: [] };

  function render(el) {
    el.innerHTML = `
      <div class="panel">
        <h2>1 · Upload Clips</h2>
        <p class="lead">Pick or create an object class, then add the videos you recorded of it.
          Clips are stored per class and later turned into training frames.</p>
        <div class="grid2">
          <div>
            <label class="field" for="upClass">Object class</label>
            <select id="upClass"></select>
          </div>
          <div>
            <label class="field" for="upNewClass">…or create a new class</label>
            <div class="row" style="flex-wrap:nowrap">
              <input id="upNewClass" type="text" placeholder="e.g. washer">
              <button class="btn ghost" id="upCreate">Add</button>
            </div>
          </div>
        </div>
        <div id="upDrop" class="dropzone" tabindex="0" style="margin-top:14px">
          <strong>Drop videos here</strong>
          <span class="muted">or click to choose files (.mov .mp4 .m4v .avi .mkv …)</span>
        </div>
        <input id="upFile" type="file" accept="video/*,.MOV,.mov,.mp4,.m4v,.avi,.mkv,.webm" multiple hidden>
        <div class="row" style="margin-top:12px">
          <button class="btn" id="upUpload">Upload to <span id="upTarget">class</span></button>
          <button class="btn ghost" id="upClear">Clear</button>
          <span class="muted" id="upSummary">No files selected.</span>
        </div>
      </div>
      <div class="panel" style="margin-top:16px">
        <div class="spread"><h2 style="font-size:16px">Uploaded clips</h2>
          <button class="btn ghost sm" id="upRefresh">Refresh</button></div>
        <div id="upGroups" style="margin-top:12px"></div>
      </div>`;

    el.querySelector("#upDrop").addEventListener("click", () => el.querySelector("#upFile").click());
    el.querySelector("#upDrop").addEventListener("dragover", (e) => { e.preventDefault(); e.currentTarget.classList.add("drag"); });
    el.querySelector("#upDrop").addEventListener("dragleave", (e) => e.currentTarget.classList.remove("drag"));
    el.querySelector("#upDrop").addEventListener("drop", (e) => {
      e.preventDefault(); e.currentTarget.classList.remove("drag"); setFiles(el, e.dataTransfer.files);
    });
    el.querySelector("#upFile").addEventListener("change", (e) => setFiles(el, e.target.files));
    el.querySelector("#upClear").addEventListener("click", () => { el.querySelector("#upFile").value = ""; setFiles(el, []); });
    el.querySelector("#upClass").addEventListener("change", () => updateTarget(el));
    el.querySelector("#upNewClass").addEventListener("input", () => updateTarget(el));
    el.querySelector("#upCreate").addEventListener("click", () => createClass(el).catch((e) => App.showError(e)));
    el.querySelector("#upUpload").addEventListener("click", () => upload(el).catch((e) => App.showError(e)));
    el.querySelector("#upRefresh").addEventListener("click", () => load(el).catch((e) => App.showError(e)));
  }

  function activeClass(el) {
    return el.querySelector("#upNewClass").value.trim() || el.querySelector("#upClass").value;
  }
  function updateTarget(el) {
    el.querySelector("#upTarget").textContent = activeClass(el) || "class";
  }
  function setFiles(el, files) {
    M.files = Array.from(files || []);
    el.querySelector("#upSummary").textContent = M.files.length
      ? `${M.files.length} file(s): ${M.files.map((f) => f.name).join(", ")}` : "No files selected.";
  }

  function renderClasses(el) {
    const select = el.querySelector("#upClass");
    const prev = select.value;
    select.innerHTML = M.classes.map((c) => `<option value="${c}">${c}</option>`).join("")
      || '<option value="" disabled selected>No classes yet</option>';
    if (M.classes.includes(prev)) select.value = prev;
    updateTarget(el);
  }

  async function createClass(el) {
    const name = el.querySelector("#upNewClass").value.trim();
    if (!name) throw new Error("Type a class name first.");
    await App.postJson("/api/clips/create-class", { name });
    el.querySelector("#upNewClass").value = "";
    await App.refresh();
    el.querySelector("#upClass").value = name;
    updateTarget(el);
    App.toast(`Class "${name}" ready`);
  }

  async function upload(el) {
    const cls = activeClass(el);
    if (!cls) throw new Error("Choose or create a class first.");
    if (!M.files.length) throw new Error("Choose at least one video.");
    const form = new FormData();
    form.append("class", cls);
    for (const file of M.files) form.append("files", file, file.name);
    App.toast("Uploading…");
    await App.api("/api/upload", { method: "POST", body: form });
    setFiles(el, []); el.querySelector("#upFile").value = ""; el.querySelector("#upNewClass").value = "";
    await App.refresh();
    await load(el);
    App.toast("Upload complete");
  }

  async function load(el) {
    const data = await App.api("/api/clips");
    M.groups = data.groups;
    M.classes = data.classes;
    renderClasses(el);
    renderGroups(el);
  }

  function renderGroups(el) {
    const root = el.querySelector("#upGroups");
    if (!M.groups.length) { root.innerHTML = '<p class="muted">No clips uploaded yet.</p>'; return; }
    root.innerHTML = M.groups.map((g) => `
      <div class="clip-group">
        <header><h4>${g.class}</h4><span class="muted">${g.clipCount} clip(s) · ${g.totalDurationLabel}</span></header>
        ${g.clips.map((c) => `
          <div class="clip-row" data-class="${c.class}" data-video="${c.video}" data-url="${c.url}">
            <span class="clip-name" title="Preview">${c.video}</span>
            <span class="clip-meta">${c.size} · ${c.durationLabel}</span>
            ${c.processed ? '<span class="pill ok">frames ✓</span>' : '<span class="pill new">new</span>'}
            <button class="btn ghost sm act-rename">Rename</button>
            <button class="btn ghost sm act-delete">Delete</button>
          </div>`).join("")}
      </div>`).join("");

    root.querySelectorAll(".clip-name").forEach((node) => node.addEventListener("click", (e) => {
      const row = e.target.closest(".clip-row");
      App.openVideo(row.dataset.video, row.dataset.url);
    }));
    root.querySelectorAll(".act-rename").forEach((node) => node.addEventListener("click", (e) =>
      renameClip(el, e.target.closest(".clip-row")).catch((err) => App.showError(err))));
    root.querySelectorAll(".act-delete").forEach((node) => node.addEventListener("click", (e) =>
      deleteClip(el, e.target.closest(".clip-row")).catch((err) => App.showError(err))));
  }

  async function renameClip(el, row) {
    const current = row.dataset.video;
    const newName = window.prompt(`Rename "${current}" to:`, current);
    if (!newName || newName === current) return;
    await App.postJson("/api/clips/rename", { class: row.dataset.class, video: current, newName });
    await load(el);
    App.toast("Renamed");
  }

  async function deleteClip(el, row) {
    if (!window.confirm(`Delete clip "${row.dataset.video}"? The video file will be removed. Extracted frames are kept.`)) return;
    await App.postJson("/api/clips/delete", { class: row.dataset.class, video: row.dataset.video });
    await App.refresh();
    await load(el);
    App.toast("Deleted");
  }

  App.registerTab({
    id: "upload", label: "Upload Clips",
    mount(el) { render(el); },
    onShow() { load(this.el).catch((e) => App.showError(e)); },
    onState(state) { if (state) { M.classes = state.classes; renderClasses(this.el); } },
    onJobDone() { load(this.el).catch(() => {}); },
  });
})();
