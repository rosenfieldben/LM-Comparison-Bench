// The page's DOM logic. The pure helpers live in lib.js (loaded first)
// and are pulled off the single global it exposes; everything below
// wires them to the document.
const {
  shortName,
  fmtCost,
  fmtEstimate,
  niceScale,
  tokenizeDiff,
  diffTokens,
  DIFF_TOKEN_LIMIT,
} = window.BenchLib;

const promptEl = document.getElementById("prompt");
const modelsEl = document.getElementById("models");
const runBtn = document.getElementById("run");
const stopBtn = document.getElementById("stop");
const resultsEl = document.getElementById("results");
const runLabelEl = document.getElementById("run-label");
const savedSelect = document.getElementById("saved-prompts");
const saveBtn = document.getElementById("save-prompt");
const deleteBtn = document.getElementById("delete-prompt");
const promptMsg = document.getElementById("prompt-msg");
const linkedEl = document.getElementById("linked-name");
const lineupLabel = document.getElementById("lineup-label");
const runNote = document.getElementById("run-note");
const historyEl = document.getElementById("history");
const historyList = document.getElementById("history-list");
const historyFilter = document.getElementById("history-filter");
const historyNote = document.getElementById("history-note");
const raceEl = document.getElementById("race");
const raceGrid = document.getElementById("race-grid");
const raceScale = document.getElementById("race-scale");

// Boot wiring, consolidating into boot.js as the split completes. Each
// module's init() attaches its listeners and paints its first state; the
// call order is the dependency order and is load-bearing.
BenchControls.init();
BenchDiff.init();

// ---- Prompt library ownership. Like the run and history paths, the
// ---- library owns its requests: one controller aborts an in-flight
// ---- load when a newer one starts, and a monotonic version means only
// ---- the latest response may write library state, so a stale reload
// ---- that resolves late cannot clobber a newer one. libraryBusy gates
// ---- the mutations so a double OK or double Enter cannot POST twice.
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
  const wanted = BenchState.selectedPromptId != null ? String(BenchState.selectedPromptId) : "";
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

const nameRow = document.getElementById("name-row");
const nameInput = document.getElementById("prompt-name");
const confirmSave = document.getElementById("confirm-save");
const cancelSave = document.getElementById("cancel-save");

function closeNameRow() {
  nameRow.hidden = true;
  nameInput.value = "";
  saveBtn.disabled = false;
}

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
    BenchState.selectedPromptId = promptEl.value === sentText ? saved.id : null;
    await loadPrompts();
  } catch (err) {
    promptMsg.textContent = "save failed: " + err.message;
  } finally {
    libraryBusy = false;
    confirmSave.disabled = false;
  }
}

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
  if (!window.confirm(`Delete saved prompt "${opt.textContent}"? Run history is kept.`)) return;
  promptMsg.textContent = "";
  libraryBusy = true;
  deleteBtn.disabled = true;
  try {
    const resp = await fetch("/prompts/" + savedSelect.value, { method: "DELETE" });
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



async function runOne(prompt, model, promptId, groupId, budget, ui, epoch) {
  // Superseded before it started: spend no money for a dead view.
  if (epoch !== BenchState.viewEpoch) return;
  const current = () => epoch === BenchState.viewEpoch;
  const controller = new AbortController();
  BenchState.epochControllers.push(controller);
  // A Stop that landed during this batch's startup (before this
  // controller existed) marked the epoch: begin already aborted so the
  // run halts as stopped rather than streaming to completion once its
  // slot opens. The mark is cleared when the batch settles, so a rerun
  // issued later in the same view reaches this line with it already
  // reset and runs normally.
  if (epoch === BenchState.stoppedEpoch) controller.abort();
  BenchState.inflightRuns += 1;
  BenchControls.updateRunState();
  ui.body.classList.add("loading");
  ui.body.textContent = "awaiting first token";
  BenchRender.startTicker(ui, model, epoch);
  BenchRender.raceRestart(model);
  // Deltas append to a dedicated text node: appendData is a pure text
  // API, so the no-HTML injection rule holds for every chunk, and the
  // node survives into the error state if the stream dies partway.
  let textNode = null;
  let finished = false;
  function appendDelta(text) {
    if (textNode === null) {
      // First token: the thinking counter has done its job, and the
      // client-side TTFT drives the card metric and race bar until the
      // server's authoritative number arrives with the done event.
      const entry = BenchRender.tickers.get(ui);
      // The race strip belongs to the current view; a superseded run
      // sharing a model name with the new one must not repaint it.
      if (entry && current()) {
        const ttftMs = performance.now() - entry.start;
        BenchRender.setMetric(ui.metrics.ttft, String(Math.round(ttftMs)), "ms");
        BenchRender.raceTtft(model, ttftMs);
      }
      BenchRender.stopTicker(ui);
      ui.statusTime.textContent = "";
      ui.body.classList.remove("loading");
      ui.body.textContent = "";
      textNode = document.createTextNode("");
      ui.body.append(textNode);
    }
    textNode.appendData(text);
  }
  function finish(result, runId) {
    // Idempotence guard: a connection that dies after the done event
    // was rendered must not stack a second set of metrics or errors.
    if (finished) return;
    finished = true;
    BenchRender.stopTicker(ui);
    if (current()) {
      if (result.stopped) {
        BenchRender.raceStopped(model);
      } else if (result.error != null) {
        BenchRender.raceError(model);
      } else {
        BenchRender.raceDone(model, result.ttft_ms);
      }
      BenchState.inflightRuns -= 1;
      BenchControls.updateRunState();
    }
    // Session accounting is view-independent: money spent by a
    // superseded run is still money spent this session. A user-stopped
    // run adds nothing: no cost frame ever arrived, which matches the
    // server, where the disconnect path persists a started run as aborted
    // with null cost and a queued run not at all. Stopping does not refund
    // spend already incurred; it just is not counted here because the
    // client never received it.
    if (!result.stopped) {
      if (result.cost_usd != null) {
        BenchState.sessionStats.spend += result.cost_usd;
      } else if (
        result.response_text != null ||
        result.prompt_tokens != null ||
        result.completion_tokens != null
      ) {
        // Evidence of consumption with no price: offline catalog,
        // missing usage, or an error after tokens flowed. Counted so
        // the session total cannot quietly understate real spend.
        BenchState.sessionStats.unpriced += 1;
      }
      if (result.error == null && result.ttft_ms != null) {
        BenchState.sessionStats.ttftSum += result.ttft_ms;
        BenchState.sessionStats.ttftN += 1;
      }
    }
    BenchState.renderStats();
    // A superseded run's view work ends here: dropped silently, its
    // persistence already handled server-side.
    if (!current()) return;
    // Presentation only, never persisted: the stored error stays the
    // server's exact words. The extended budget is the one knob the
    // user can turn when reasoning burned the whole standard budget,
    // so say so right where the failure is reported.
    let shownError = result.error;
    if (
      shownError != null &&
      budget === "standard" &&
      shownError.includes("finish_reason: length")
    ) {
      shownError += "; try extended budget";
    }
    BenchRender.completeColumn(ui, result, model + " (live)", {
      streamed: textNode !== null,
      shownError: shownError,
      budgetBadge: false,
      // run_id null on the done event means the server spent the money
      // and streamed the response but could not persist it.
      unsaved: runId === null,
      // Only this streaming path offers a rerun; historical replays go
      // through fillColumn and never get one. A stopped run has no error,
      // so it gets no rerun control.
      retry: result.error != null
        ? { prompt, model, promptId, groupId, budget }
        : null,
      // A user Stop renders as an honest stopped state, not done or error.
      stopped: result.stopped === true,
    });
  }

  try {
    const resp = await fetch("/compare/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        model: model,
        prompt_id: promptId,
        group_id: groupId,
        budget: budget,
      }),
      signal: controller.signal,
    });
    if (!resp.ok) {
      // A refusal like the spend ceiling (402) carries a JSON detail
      // explaining itself in words; surface that instead of a bare
      // status so the error card reads as a sentence, not a code.
      let detail = "HTTP " + resp.status;
      try {
        const body = await resp.json();
        if (body && typeof body.detail === "string") detail = body.detail;
      } catch (err) {
        // Non-JSON error body: the status line is the best we have.
      }
      throw new Error(detail);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        const line = frame.split("\n").find(l => l.startsWith("data:"));
        if (!line) continue;
        const event = JSON.parse(line.slice(5));
        if (event.type === "delta") {
          appendDelta(event.text);
        } else if (event.type === "done") {
          finish(event.result, event.run_id);
        } else if (event.type === "queued") {
          // Waiting on a server slot, not reasoning yet: say so instead
          // of counting up as "thinking".
          ui.statusWord.textContent = "queued";
        } else if (event.type === "started") {
          // The slot was just acquired. Restart the client clock here so
          // the TTFT estimate excludes queue wait, matching the server's
          // post-acquire clock, and restore the thinking label.
          const entry = BenchRender.tickers.get(ui);
          if (entry) entry.start = performance.now();
          ui.statusWord.textContent = "thinking";
        }
        // Unknown frame types are ignored, as before.
      }
    }
    if (!finished) throw new Error("stream ended unexpectedly");
  } catch (err) {
    // An abort while this run's epoch is still current is a user Stop.
    // Supersession aborts too, but it also moves the epoch, so current()
    // is false there and the run drops silently in finish(). A stopped
    // run keeps whatever text streamed in and renders an honest stopped
    // status with no error and no fabricated metrics; a network death
    // shows the error as before, its partial text still folded in so the
    // card stays diffable.
    // run_id undefined, not null: whether the server persisted this
    // run is unknown from here (its disconnect path usually does), so
    // no not-saved warning is claimed.
    const stopped = err.name === "AbortError" && current();
    finish({
      error: stopped ? null : "request failed: " + err.message,
      response_text: textNode !== null ? textNode.data : null,
      stopped: stopped,
    }, undefined);
  }
}

// Named rather than inline in the click handler so the browser suite
// can start a superseding run directly: the disabled Run button is the
// UX affordance, the view epoch is the integrity mechanism, and the
// tests exercise the mechanism.
async function startRun() {
  const prompt = promptEl.value;
  const promptId = BenchState.selectedPromptId;
  const budget = BenchControls.budgetValue;
  const models = BenchControls.checkedModels();
  const epoch = BenchState.newViewEpoch();
  // Reserve the in-flight registry synchronously, before the /groups
  // await below, so the Run button is disabled for the whole batch
  // startup. Without this a second click during the sub-second /groups
  // latency would start a duplicate run (double-counted runs stat,
  // orphan group row), since per-model runOne calls do not increment
  // the registry until after that await resolves.
  BenchState.inflightRuns += 1;
  resultsEl.replaceChildren();
  runLabelEl.textContent = "";
  // A new comparison replaces the cards a shown diff came from.
  BenchDiff.closeDiffPanel();
  BenchDiff.disarmDiff();
  BenchControls.updateRunState();
  BenchState.sessionStats.runs += 1;
  BenchState.renderStats();
  // One request per model instead of one batch: /compare returns only when
  // its slowest model finishes, and the bench exists to watch fast models
  // land first. Cards are created up front so order tracks the chip
  // list, not response arrival.
  BenchRender.raceInit(models);
  const columns = models.map(BenchRender.makeColumn);
  // One group per Run click so the N per-model requests land as one
  // history entry. A failed create degrades to ungrouped runs rather
  // than blocking: grouping is bookkeeping, the comparison is the
  // product.
  let groupId = null;
  // The group POST joins the epoch's controllers so Stop can abort it
  // too: a Stop during this await then leaves the batch ungrouped and,
  // via stoppedEpoch, halts every model that was about to start.
  const groupController = new AbortController();
  BenchState.epochControllers.push(groupController);
  try {
    // The empty JSON body is load-bearing: the server requires
    // application/json on every POST so hostile cross-site senders
    // are forced into a CORS preflight it never answers.
    const resp = await fetch("/groups", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
      signal: groupController.signal,
    });
    if (resp.ok) groupId = (await resp.json()).id;
  } catch (err) {
    // Deliberately swallowed, a Stop abort included; see above.
  }
  try {
    await Promise.allSettled(
      models.map((model, i) =>
        runOne(prompt, model, promptId, groupId, budget, columns[i], epoch)
      )
    );
  } finally {
    // The stop mark exists only to catch this batch's own models when a
    // Stop lands during the group-POST await, before their controllers
    // exist. Once the batch has settled, every such run has passed its
    // startup check, so clear the mark: a later rerun reuses this epoch
    // (it stays in the same view) and must not be aborted as if it were
    // part of the stopped batch. Cleared regardless of supersession; if
    // it belonged to a superseded epoch no future epoch would match it
    // anyway, but leaving it set would strand a rerun in this view.
    if (BenchState.stoppedEpoch === epoch) BenchState.stoppedEpoch = -1;
    // Release the batch reservation, but only if this batch still owns
    // the view: a superseding run already reset the registry to zero,
    // so decrementing here would corrupt its count.
    if (epoch === BenchState.viewEpoch) {
      BenchState.inflightRuns -= 1;
      BenchControls.updateRunState();
    }
  }
}
runBtn.addEventListener("click", startRun);

// Stop aborts every in-flight controller in the current epoch WITHOUT
// taking a new view epoch: the comparison stays the view, its cards stay,
// nothing is cleared. Each aborted runOne lands in its catch with the
// epoch still current and renders a stopped card; as they settle the
// in-flight count drains to zero, re-enabling Run and disabling Stop, so
// a later Run or rerun works through the untouched epoch machinery. The
// abort disconnects each stream, so the server persists a started run
// through its existing disconnect path and a queued run not at all,
// exactly as the cards show. The stop mark set here lives only until the
// stopped batch settles (startRun clears it in its finally), so a rerun
// issued afterward in this same view is unaffected.
function stopRuns() {
  // Mark the epoch stopped before aborting, so a per-model run still
  // starting up (its controller not yet in the list because startRun is
  // mid group-POST) begins already aborted rather than streaming on.
  BenchState.stoppedEpoch = BenchState.viewEpoch;
  for (const c of BenchState.epochControllers) c.abort();
}
stopBtn.addEventListener("click", stopRuns);

// ---- History. Loads on expand rather than page load: it is the
// ---- rarely used half of the page and a stale list is worse than a
// ---- slightly later one.
historyEl.addEventListener("toggle", () => {
  if (historyEl.open) loadHistory();
});

// Client-side only: the filter narrows what is already on screen, it
// never refetches. Matches the visible prompt text and model ids.
historyFilter.addEventListener("input", applyHistoryFilter);
function applyHistoryFilter() {
  const q = historyFilter.value.trim().toLowerCase();
  for (const row of historyList.querySelectorAll(".hrow")) {
    row.hidden = q !== "" && !row.dataset.hay.includes(q);
  }
}

// Explicit rather than relying on the server default, so the "newest
// N" note in the header cannot drift out of sync with what was asked.
const HISTORY_LIMIT = 100;

// The newest list load owns the panel; an older one still in flight
// is aborted rather than left to race the render.
let historyListController = null;

async function loadHistory() {
  if (historyListController !== null) historyListController.abort();
  const controller = new AbortController();
  historyListController = controller;
  historyList.replaceChildren();
  historyNote.textContent = "";
  let data;
  try {
    const resp = await fetch("/runs?limit=" + HISTORY_LIMIT, {
      signal: controller.signal,
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    data = await resp.json();
  } catch (err) {
    if (controller.signal.aborted) return;
    historyList.textContent = "failed to load history: " + err.message;
    return;
  }
  if (controller !== historyListController) return;
  if (data.runs.length === 0) {
    historyList.textContent = "no runs yet";
    return;
  }
  for (const run of data.runs) {
    // Buttons, not divs: rows must be reachable by keyboard.
    const row = document.createElement("button");
    row.type = "button";
    row.className = "hrow";
    row.dataset.testid = "history-row";
    // The filter matches model ids even though rows only show a count.
    row.dataset.hay = (run.prompt_text + " " + run.models.join(" ")).toLowerCase();
    const time = document.createElement("span");
    time.className = "htime";
    // Labeled like the replay banner: an unlabeled timestamp reads as
    // local time, and these are UTC.
    time.textContent = run.created_at.slice(0, 19).replace("T", " ") + " UTC";
    const text = document.createElement("span");
    text.className = "hprompt";
    text.textContent = run.prompt_text;
    const count = document.createElement("span");
    count.className = "hcount";
    count.dataset.testid = "history-count";
    // run.models carries one entry per result, so a model rerun twice
    // appears twice. Label by distinct models, and add the attempt count
    // separately only when reruns pushed it above the model count, so a
    // rerun group reads "1 model · 2 attempts" not "2 models".
    const uniqueModels = new Set(run.models).size;
    const attempts = run.models.length;
    let countText = uniqueModels + (uniqueModels === 1 ? " model" : " models");
    if (attempts > uniqueModels) {
      countText += " · " + attempts + " attempts";
    }
    count.textContent = countText;
    row.append(time, text, count);
    row.title = run.models.join(", ");
    row.addEventListener("click", () =>
      run.type === "group" ? showGroup(run.id) : showRun(run.id)
    );
    historyList.append(row);
  }
  // A full page means there may be older entries beyond it; say so
  // rather than letting truncation read as "this is everything".
  if (data.runs.length === HISTORY_LIMIT) {
    historyNote.textContent =
      "newest " + HISTORY_LIMIT + " · older stays in bench.db";
  }
  // A filter typed before this refresh still applies to the new rows.
  applyHistoryFilter();
}

// A history load owns the results area from the click, not from the
// moment its fetch succeeds. Clearing the cards, race and diff and
// showing a state up front is what keeps a failed or slow load from
// leaving its banner over another run's cards: the grid and the banner
// always agree. The armed diff side survives (closeDiffPanel does not
// disarm), so cross-replay diffing still works.
function renderHistoryState(label, testid, cls, boxText) {
  resultsEl.replaceChildren();
  BenchRender.hideRace();
  BenchDiff.closeDiffPanel();
  runLabelEl.textContent = label;
  const box = document.createElement("div");
  box.className = "history-status " + cls;
  box.dataset.testid = testid;
  box.textContent = boxText;
  resultsEl.append(box);
}

async function showGroup(groupId) {
  // Owns the view from the click: in-flight runs for the old view are
  // aborted now, and this fetch is itself abortable by whatever
  // supersedes it.
  const epoch = BenchState.newViewEpoch();
  BenchControls.updateRunState();
  const controller = new AbortController();
  BenchState.epochControllers.push(controller);
  // Clear the old view and show a loading state before any network
  // activity begins; the old cards must be gone before the fetch.
  renderHistoryState(
    "Loading comparison #" + groupId,
    "history-loading", "loading", "loading comparison"
  );
  let group;
  try {
    const resp = await fetch("/groups/" + groupId, {
      signal: controller.signal,
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    group = await resp.json();
  } catch (err) {
    // A superseded load stays silent; its replacement already owns the
    // view. Otherwise the loading state becomes a failure that stands
    // alone: banner and grid agree and no other run's cards are visible.
    if (epoch !== BenchState.viewEpoch) return;
    renderHistoryState(
      "failed to load comparison #" + groupId + ": " + err.message,
      "history-failure", "failure",
      "could not load this comparison; nothing from another run is shown"
    );
    return;
  }
  if (epoch !== BenchState.viewEpoch) return;
  // Replace the loading state with the comparison. Race and diff were
  // already cleared when the loading state was rendered.
  resultsEl.replaceChildren();
  runLabelEl.textContent =
    "Historical comparison #" + group.id + " from " +
    group.created_at.slice(0, 19).replace("T", " ") + " UTC";
  // Cards in chip (lineup) order, matching the live layout. Runs
  // persist in completion order, so run order alone would shuffle
  // cards between replays; models no longer in the lineup keep run
  // order at the end.
  const results = group.runs.flatMap(r => r.results);
  const rank = m => {
    const i = BenchControls.lineup.indexOf(m);
    return i === -1 ? BenchControls.lineup.length : i;
  };
  results.sort((a, b) => rank(a.model) - rank(b.model));
  for (const result of results) {
    BenchRender.fillColumn(
      BenchRender.makeColumn(result.model), result,
      result.model + ", comparison #" + group.id
    );
  }
  window.scrollTo({ top: 0 });
}

async function showRun(runId) {
  // Same ownership rule as showGroup.
  const epoch = BenchState.newViewEpoch();
  BenchControls.updateRunState();
  const controller = new AbortController();
  BenchState.epochControllers.push(controller);
  renderHistoryState(
    "Loading run #" + runId,
    "history-loading", "loading", "loading run"
  );
  let run;
  try {
    const resp = await fetch("/runs/" + runId, { signal: controller.signal });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    run = await resp.json();
  } catch (err) {
    if (epoch !== BenchState.viewEpoch) return;
    renderHistoryState(
      "failed to load run #" + runId + ": " + err.message,
      "history-failure", "failure",
      "could not load this run; nothing from another run is shown"
    );
    return;
  }
  if (epoch !== BenchState.viewEpoch) return;
  // Same completion renderer as a live run so the textContent and
  // null-content guarantees hold for stored data too. Race and diff were
  // cleared with the loading state.
  resultsEl.replaceChildren();
  runLabelEl.textContent =
    "Historical run #" + run.id + " from " +
    run.created_at.slice(0, 19).replace("T", " ") + " UTC";
  for (const result of run.results) {
    BenchRender.fillColumn(BenchRender.makeColumn(result.model), result, result.model + ", run #" + run.id);
  }
  window.scrollTo({ top: 0 });
}
