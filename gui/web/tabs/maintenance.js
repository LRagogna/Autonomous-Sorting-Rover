/* Danger Zone - Wipe Data / Start Over (guarded) */
(function () {
  const M = { info: null };

  function render(el) {
    el.innerHTML = `
      <div class="panel">
        <h2>Maintenance</h2>
        <p class="lead">Housekeeping for your project. The dataset reset below is deliberately hard to trigger.</p>
      </div>

      <div class="panel danger-zone" style="margin-top:16px">
        <h2>⚠ Danger Zone — Wipe Data / Start Over</h2>
        <p class="lead">Permanently deletes your uploaded clips, extracted frames, dataset, rejected frames, and
          retraining queue so you can start a fresh project. Trained models are <b>kept</b> unless you explicitly
          choose otherwise. Your class list is preserved.</p>

        <label class="field" style="margin-top:12px">These folders will be deleted:</label>
        <ul class="wipe-targets" id="mtTargets"><li>Loading…</li></ul>
        <p class="muted" id="mtKeeps"></p>

        <button class="btn danger" id="mtWipe" style="margin-top:10px">Wipe Data / Start Over</button>
      </div>

      <div class="modal wipe-modal" id="mtModal" hidden>
        <div class="modal-card">
          <div class="modal-head">
            <span class="modal-title wipe-danger-text">Final confirmation</span>
            <button class="btn ghost" id="mtCancel">Cancel</button>
          </div>
          <div style="padding:16px">
            <p>This <b>cannot be undone</b> unless you create a backup. The following will be deleted:</p>
            <ul class="wipe-targets" id="mtModalTargets"></ul>
            <label class="row" style="gap:10px;margin-top:10px"><input type="checkbox" id="mtBackup" checked> Create a backup first (copies data to <code>backups/</code>)</label>
            <label class="row" style="gap:10px;margin-top:8px"><input type="checkbox" id="mtModels"> <span class="wipe-danger-text">Also delete trained models</span> (harder to recover)</label>
            <label class="field" style="margin-top:14px">Type <span class="wipe-danger-text">WIPE DATA</span> to confirm:</label>
            <input type="text" id="mtPhrase" placeholder="WIPE DATA" autocomplete="off">
            <div class="row" style="margin-top:14px;justify-content:flex-end">
              <button class="btn ghost" id="mtModalCancel">Cancel</button>
              <button class="btn danger" id="mtConfirm" disabled>Delete everything</button>
            </div>
          </div>
        </div>
      </div>`;

    el.querySelector("#mtWipe").addEventListener("click", () => startWipe(el));
    el.querySelector("#mtCancel").addEventListener("click", () => closeModal(el));
    el.querySelector("#mtModalCancel").addEventListener("click", () => closeModal(el));
    el.querySelector("#mtModal").addEventListener("click", (e) => { if (e.target === el.querySelector("#mtModal")) closeModal(el); });
    el.querySelector("#mtPhrase").addEventListener("input", (e) => {
      el.querySelector("#mtConfirm").disabled = e.target.value !== (M.info ? M.info.phrase : "WIPE DATA");
    });
    el.querySelector("#mtConfirm").addEventListener("click", () => doWipe(el).catch((x) => App.showError(x)));
  }

  async function load(el) {
    M.info = await App.api("/api/maintenance/targets");
    const rows = M.info.targets.map((t) =>
      `<li><span>${t.path}/</span><span>${t.exists ? t.size : "empty"}</span></li>`).join("");
    el.querySelector("#mtTargets").innerHTML = rows;
    el.querySelector("#mtModalTargets").innerHTML = rows;
    el.querySelector("#mtKeeps").textContent =
      `Kept: ${M.info.keeps.join(", ")}. Models: ${M.info.models.count} file(s), ${M.info.models.size} (kept unless you tick the box).`;
  }

  /* Layer 1 + 2: two blocking confirmations before the typed-phrase modal. */
  function startWipe(el) {
    const list = M.info ? M.info.targets.map((t) => "• " + t.path).join("\n") : "";
    if (!window.confirm(`This will PERMANENTLY DELETE:\n\n${list}\n\nTrained models are kept by default. Continue?`)) return;
    if (!window.confirm("Are you ABSOLUTELY sure? This cannot be undone unless you make a backup.")) return;
    el.querySelector("#mtPhrase").value = "";
    el.querySelector("#mtModels").checked = false;
    el.querySelector("#mtBackup").checked = true;
    el.querySelector("#mtConfirm").disabled = true;
    el.querySelector("#mtModal").hidden = false;
    el.querySelector("#mtPhrase").focus();
  }

  function closeModal(el) { el.querySelector("#mtModal").hidden = true; }

  /* Layer 3: typed phrase (+ extra confirm if deleting models). */
  async function doWipe(el) {
    const phrase = el.querySelector("#mtPhrase").value;
    const deleteModels = el.querySelector("#mtModels").checked;
    const backup = el.querySelector("#mtBackup").checked;
    if (phrase !== M.info.phrase) return;
    if (deleteModels && !window.confirm("You also chose to DELETE ALL TRAINED MODELS. These are hard to recover. Proceed?")) return;

    const res = await App.postJson("/api/maintenance/wipe", { confirm: phrase, deleteModels, backup });
    closeModal(el);
    await App.refresh();
    await load(el);
    App.toast(res.backup ? `Project reset. Backup saved to ${res.backup}` : "Project data reset.");
  }

  App.registerTab({
    id: "maintenance", label: "Danger Zone",
    mount(el) { render(el); },
    onShow() { load(this.el).catch((e) => App.showError(e)); },
  });
})();
