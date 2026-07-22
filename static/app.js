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
BenchLibrary.init();
BenchStream.init();


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
