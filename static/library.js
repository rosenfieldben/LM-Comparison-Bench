// The saved-prompt library: one owner for its requests. A controller
// aborts an in-flight load when a newer one starts, a monotonic version
// means only the latest response writes library state, and libraryBusy
// gates the mutations so a double OK or double Enter cannot POST twice.
// Exposed on window.BenchLibrary; init() wires the listeners and kicks
// the first load, and boot.js calls it after every module has loaded.
(function () {
  const promptEl = document.getElementById("prompt");
  const savedSelect = document.getElementById("saved-prompts");
  const saveBtn = document.getElementById("save-prompt");
  const deleteBtn = document.getElementById("delete-prompt");
  const promptMsg = document.getElementById("prompt-msg");
  const nameRow = document.getElementById("name-row");
  const nameInput = document.getElementById("prompt-name");
  const confirmSave = document.getElementById("confirm-save");
  const cancelSave = document.getElementById("cancel-save");

  let promptsController = null;
  let promptsVersion = 0;
  let libraryBusy = false;

  // The one place a fetched list writes the dropdown, and the only place
  // the selection is reconciled against the server's truth: a prompt
  // deleted in another tab degrades to no selection instead of a dangling
  // id the next run would send. selectedPromptId is the live selection
  // (set by the dropdown, cleared on textarea edit, set by a save), so
  // reconciling against it keeps a late reload from resetting a newer
  // choice. Programmatic value assignment fires no change event, so this
  // does not loop through the change handler.
  function setPromptLibrary(prompts) {
    savedSelect.replaceChildren(new Option("Saved", ""));
    const ids = new Set();
    for (const p of prompts) {
      const opt = new Option(p.name, String(p.id));
      opt.dataset.text = p.text;
      savedSelect.append(opt);
      ids.add(String(p.id));
    }
    const wanted =
      BenchState.selectedPromptId != null
        ? String(BenchState.selectedPromptId)
        : "";
    const resolved = ids.has(wanted) ? wanted : "";
    savedSelect.value = resolved;
    BenchState.selectedPromptId = resolved === "" ? null : Number(resolved);
    BenchControls.updateRunState();
  }

  async function loadPrompts() {
    if (promptsController !== null) promptsController.abort();
    const controller = new AbortController();
    promptsController = controller;
    const version = ++promptsVersion;
    let data;
    try {
      const resp = await fetch("/prompts", { signal: controller.signal });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch (err) {
      // A superseded or aborted load stays silent; only a current failure
      // reports itself.
      if (controller.signal.aborted || version !== promptsVersion) return;
      promptMsg.textContent = "failed to load prompts: " + err.message;
      return;
    }
    // Only the latest load writes library state.
    if (version !== promptsVersion) return;
    setPromptLibrary(data.prompts);
  }

  function closeNameRow() {
    nameRow.hidden = true;
    nameInput.value = "";
    saveBtn.disabled = false;
  }

  async function submitSave() {
    // One save at a time: a double OK or a second Enter while the POST is
    // in flight must not issue a second request. libraryBusy is the
    // authoritative guard (Enter reaches here even with the button
    // disabled); disabling the button is the visible affordance.
    if (libraryBusy) return;
    const name = nameInput.value.trim();
    // Empty name is a no-op, mirroring the old cancelled-dialog behavior:
    // the row stays open so the user can type or explicitly cancel.
    if (!name) return;
    // The exact text sent, captured now so the link is re-established only
    // if the textarea still shows it when the save returns.
    const sentText = promptEl.value;
    // Clear any stale library error at the start of the attempt, so a 409
    // from a previous attempt cannot outlive its cause.
    promptMsg.textContent = "";
    libraryBusy = true;
    confirmSave.disabled = true;
    try {
      const resp = await fetch("/prompts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, text: sentText }),
      });
      if (resp.status === 409) {
        // Row stays open on conflict so the name can be edited and retried.
        promptMsg.textContent = "a prompt with that name already exists";
        return;
      }
      if (!resp.ok) {
        promptMsg.textContent = "save failed: HTTP " + resp.status;
        return;
      }
      const saved = await resp.json();
      closeNameRow();
      // Re-establish the saved-prompt link only if the textarea still
      // matches the saved text; if it changed while the save was in
      // flight, leave the link cleared rather than claim a false match.
      BenchState.selectedPromptId =
        promptEl.value === sentText ? saved.id : null;
      await loadPrompts();
    } catch (err) {
      promptMsg.textContent = "save failed: " + err.message;
    } finally {
      libraryBusy = false;
      confirmSave.disabled = false;
    }
  }

  function init() {
    savedSelect.addEventListener("change", () => {
      promptMsg.textContent = "";
      const opt = savedSelect.selectedOptions[0];
      if (savedSelect.value === "") {
        BenchState.selectedPromptId = null;
      } else {
        BenchState.selectedPromptId = Number(savedSelect.value);
        promptEl.value = opt.dataset.text;
        BenchControls.autosizePrompt();
      }
      BenchControls.updateRunState();
    });

    saveBtn.addEventListener("click", () => {
      promptMsg.textContent = "";
      if (promptEl.value.trim() === "") {
        promptMsg.textContent = "nothing to save: prompt is empty";
        return;
      }
      nameRow.hidden = false;
      saveBtn.disabled = true;
      nameInput.focus();
    });

    confirmSave.addEventListener("click", submitSave);
    cancelSave.addEventListener("click", () => {
      promptMsg.textContent = "";
      closeNameRow();
    });
    nameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        submitSave();
      } else if (e.key === "Escape") {
        promptMsg.textContent = "";
        closeNameRow();
      }
    });

    deleteBtn.addEventListener("click", async () => {
      // Same one-mutation-at-a-time guard as save.
      if (libraryBusy) return;
      const opt = savedSelect.selectedOptions[0];
      if (savedSelect.value === "") return;
      if (
        !window.confirm(
          `Delete saved prompt "${opt.textContent}"? Run history is kept.`,
        )
      )
        return;
      promptMsg.textContent = "";
      libraryBusy = true;
      deleteBtn.disabled = true;
      try {
        const resp = await fetch("/prompts/" + savedSelect.value, {
          method: "DELETE",
        });
        if (!resp.ok && resp.status !== 404) {
          promptMsg.textContent = "delete failed: HTTP " + resp.status;
          return;
        }
        BenchState.selectedPromptId = null;
        await loadPrompts();
      } catch (err) {
        promptMsg.textContent = "delete failed: " + err.message;
      } finally {
        libraryBusy = false;
        // Re-sync the delete button to the resolved selection.
        BenchControls.updateRunState();
      }
    });

    loadPrompts();
  }

  // loadPrompts and submitSave are also the entry points the browser
  // suite drives through page.evaluate, so they are on the namespace.
  window.BenchLibrary = { init, loadPrompts, submitSave };
})();
