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

// Seed for a fresh browser only. The live lineup is a localStorage
// preference of THIS browser, not bench data: keeping it out of sqlite
// means bench.db stays purely runs and prompts, and losing it costs
// four clicks. The key predates the interface overhaul and must never
// change: users have lineups stored under it.
const DEFAULT_LINEUP = [
  "deepseek/deepseek-chat",
  "z-ai/glm-4.6",
  "mistralai/mistral-small",
  "anthropic/claude-sonnet-4.6",
];
const LINEUP_KEY = "bench-lineup";

function loadLineup() {
  try {
    const parsed = JSON.parse(localStorage.getItem(LINEUP_KEY));
    if (Array.isArray(parsed) && parsed.every(x => typeof x === "string")) {
      return parsed;
    }
  } catch (err) {
    // Unparseable storage falls through to the defaults.
  }
  return [...DEFAULT_LINEUP];
}

let lineup = loadLineup();

function saveLineup() {
  // Guarded like the pref helpers below: a quota or SecurityError must
  // not abort an add or remove. The in-memory lineup stays authoritative;
  // persistence just lapses for the session.
  try {
    localStorage.setItem(LINEUP_KEY, JSON.stringify(lineup));
  } catch (err) {
    // Blocked or full storage: the lineup lives in memory this session.
  }
}

// UI prefs (theme, motion, density) persist per browser; a bootstrap
// script in <head> applies them before first paint.
function prefGet(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v !== null ? v : fallback;
  } catch (err) {
    return fallback;
  }
}
function prefSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (err) {
    // Blocked storage just means the pref lasts one session.
  }
}

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

const addModelBtn = document.getElementById("add-model");
const searchRow = document.getElementById("model-search");
const queryInput = document.getElementById("model-query");
const searchMsg = document.getElementById("search-msg");
const matchesEl = document.getElementById("model-matches");
// Snapshot of GET /models; fetched=false switches the picker to the
// exact-id fallback so the bench stays usable on an offline boot.
let catalog = { fetched: false, models: [] };

document.getElementById("bar-host").textContent = location.host || "localhost:8000";

// ---- UI prefs: theme override (auto follows the OS), motion, density.
const themeBtn = document.getElementById("theme-btn");
const motionBtn = document.getElementById("motion-btn");
const THEMES = ["auto", "dark", "light"];
let themeMode = prefGet("bench-theme", "auto");
if (!THEMES.includes(themeMode)) themeMode = "auto";

function applyTheme() {
  if (themeMode === "auto") {
    delete document.documentElement.dataset.theme;
  } else {
    document.documentElement.dataset.theme = themeMode;
  }
  themeBtn.textContent = "theme " + themeMode;
}
themeBtn.addEventListener("click", () => {
  themeMode = THEMES[(THEMES.indexOf(themeMode) + 1) % THEMES.length];
  prefSet("bench-theme", themeMode);
  applyTheme();
});
applyTheme();

// Motion off kills every animation via CSS; elapsed-time counters are
// plain text updates and keep going. prefers-reduced-motion does the
// same regardless of this toggle.
let motionOn = prefGet("bench-motion", "on") !== "off";

function applyMotion() {
  document.documentElement.dataset.motion = motionOn ? "on" : "off";
  motionBtn.textContent = "motion " + (motionOn ? "on" : "off");
}
motionBtn.addEventListener("click", () => {
  motionOn = !motionOn;
  prefSet("bench-motion", motionOn ? "on" : "off");
  applyMotion();
});
applyMotion();

// Segmented controls: a button pair where aria-pressed is the state.
function initSeg(el, initial, onChange) {
  const btns = [...el.querySelectorAll("button")];
  function set(value) {
    for (const b of btns) {
      b.setAttribute("aria-pressed", String(b.dataset.value === value));
    }
  }
  for (const b of btns) {
    b.addEventListener("click", () => {
      set(b.dataset.value);
      onChange(b.dataset.value);
    });
  }
  set(initial);
}

// Deliberately per-session, never persisted: extended costs real
// money, so the safe default must reassert itself on the next visit.
let budgetValue = "standard";
initSeg(document.getElementById("budget-seg"), budgetValue, (v) => {
  budgetValue = v;
  updateRunState();
});

// Density persists, unlike the budget: layout taste is harmless.
let densityValue = prefGet("bench-density", "comfortable");
if (densityValue !== "compact") densityValue = "comfortable";
document.documentElement.dataset.density = densityValue;
initSeg(document.getElementById("density-seg"), densityValue, (v) => {
  densityValue = v;
  document.documentElement.dataset.density = v;
  prefSet("bench-density", v);
});

function autosizePrompt() {
  // Height tracks content; the CSS max-height caps runaway growth.
  promptEl.style.height = "auto";
  promptEl.style.height = promptEl.scrollHeight + "px";
}

function renderLineup() {
  // Checked state is per-session; carry it across rebuilds so removing
  // one model does not uncheck the others.
  const checked = new Set(checkedModels());
  modelsEl.replaceChildren();
  for (const model of lineup) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.dataset.testid = "lineup-chip";
    chip.title = model;
    // The label wraps the checkbox and its visible text only; the
    // remove button is a sibling so the checkbox's accessible name
    // cannot absorb it.
    const label = document.createElement("label");
    label.className = "chip-label";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = model;
    // Chips display the short name; the accessible name is the full
    // id, which is what distinguishes two vendors' same-named models.
    box.setAttribute("aria-label", model);
    box.checked = checked.has(model);
    if (box.checked) chip.classList.add("on");
    box.addEventListener("change", () => {
      chip.classList.toggle("on", box.checked);
      updateRunState();
    });
    // The dot is the checked indicator: filled accent = on, hollow = off.
    const dot = document.createElement("span");
    dot.className = "dot";
    const id = document.createElement("span");
    id.textContent = shortName(model);
    label.append(box, dot, id);
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "rm-model";
    rm.textContent = "×";
    rm.title = "Remove from lineup";
    rm.setAttribute("aria-label", "Remove " + model + " from lineup");
    rm.addEventListener("click", () => {
      lineup = lineup.filter(m => m !== model);
      saveLineup();
      renderLineup();
      BenchState.renderStats();
    });
    chip.append(label, rm);
    modelsEl.append(chip);
  }
  updateRunState();
  BenchState.renderStats();
}

function checkedModels() {
  return [...modelsEl.querySelectorAll("input:checked")].map(b => b.value);
}

function setAllChecked(on) {
  for (const box of modelsEl.querySelectorAll("input[type=checkbox]")) {
    box.checked = on;
    box.closest(".chip").classList.toggle("on", on);
  }
  updateRunState();
}
document.getElementById("select-all").addEventListener("click", () => setAllChecked(true));
document.getElementById("select-none").addEventListener("click", () => setAllChecked(false));

// What /compare/stream sends as the completion budget; the server
// clamps per model to the published completion cap.
const BUDGET_TOKENS = { standard: 16384, extended: 65536 };

// "n requests · max output cost $x/run (input not included)". The
// estimate is checked models × completion price × budget (per-model
// capped), the worst billable OUTPUT case; the input side depends on
// prompt tokenization the client does not attempt, so the label says
// so instead of pretending. Omitted when pricing is unavailable. The
// request count doubles as the note that each model runs as its own
// request.
function updateEstimate() {
  const models = checkedModels();
  if (models.length === 0) {
    runNote.textContent = "";
    return;
  }
  let text = models.length + (models.length === 1 ? " request" : " requests");
  if (catalog.fetched) {
    let est = 0;
    let computable = true;
    for (const id of models) {
      const m = catalog.models.find(x => x.id === id);
      if (!m || m.completion_price == null) {
        computable = false;
        break;
      }
      const cap = m.max_completion_tokens != null
        ? Math.min(m.max_completion_tokens, BUDGET_TOKENS[budgetValue])
        : BUDGET_TOKENS[budgetValue];
      est += m.completion_price * cap;
    }
    if (computable) {
      text +=
        " · max output cost $" + fmtEstimate(est) +
        "/run (input not included)";
    }
  }
  runNote.textContent = text;
}

function renderLinked() {
  const opt = savedSelect.selectedOptions[0];
  linkedEl.textContent = savedSelect.value !== "" && opt
    ? "linked: " + opt.textContent
    : "";
}

function updateRunState() {
  const checked = checkedModels().length;
  runBtn.disabled =
    BenchState.inflightRuns > 0 || promptEl.value.trim() === "" || checked === 0;
  // Stop is live exactly while runs are: it acts on the in-flight
  // controllers and has nothing to do when the count is zero.
  stopBtn.disabled = BenchState.inflightRuns === 0;
  deleteBtn.disabled = savedSelect.value === "";
  lineupLabel.textContent = "Lineup " + checked + "/" + lineup.length;
  renderLinked();
  updateEstimate();
}
promptEl.addEventListener("input", () => {
  BenchState.selectedPromptId = null;
  savedSelect.value = "";
  autosizePrompt();
  updateRunState();
});

async function loadCatalog() {
  try {
    const resp = await fetch("/models");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    catalog = await resp.json();
  } catch (err) {
    catalog = { fetched: false, models: [] };
  }
  // Pricing just arrived (or didn't): refresh the run estimate.
  updateRunState();
}

function clearSearch() {
  queryInput.value = "";
  matchesEl.replaceChildren();
}

function addToLineup(id) {
  if (lineup.includes(id)) return;
  lineup.push(id);
  saveLineup();
  renderLineup();
  clearSearch();
}

function fmtPricing(m) {
  if (m.prompt_price == null || m.completion_price == null) {
    return "pricing unavailable";
  }
  // Per-million dollars: per-token floats are unreadable, and
  // per-million is how every provider quotes.
  return (
    "$" + (m.prompt_price * 1e6).toFixed(2) +
    " / $" + (m.completion_price * 1e6).toFixed(2) +
    " per 1M in/out"
  );
}

addModelBtn.addEventListener("click", () => {
  searchRow.hidden = !searchRow.hidden;
  addModelBtn.setAttribute("aria-expanded", String(!searchRow.hidden));
  if (searchRow.hidden) {
    clearSearch();
    return;
  }
  if (!catalog.fetched) {
    searchMsg.textContent =
      "model catalog unavailable (offline boot); add by exact id";
    queryInput.placeholder = "exact model id, Enter to add";
  } else {
    searchMsg.textContent = "";
    queryInput.placeholder = "Search models by name or id";
  }
  queryInput.focus();
});

function renderMatches() {
  matchesEl.replaceChildren();
  if (!catalog.fetched) return;
  const q = queryInput.value.trim().toLowerCase();
  if (q === "") return;
  const hits = catalog.models
    .filter(m =>
      m.id.toLowerCase().includes(q) ||
      (m.name || "").toLowerCase().includes(q)
    )
    .slice(0, 15);
  for (const m of hits) {
    // Buttons, not divs: rows must be reachable by keyboard.
    const row = document.createElement("button");
    row.type = "button";
    row.className = "match";
    // textContent throughout: catalog names arrive from an external
    // API and are as untrusted as model output.
    const name = document.createElement("span");
    name.className = "match-name";
    name.textContent = m.name != null ? m.name : m.id;
    const id = document.createElement("span");
    id.className = "match-id";
    id.textContent = m.id;
    const meta = document.createElement("span");
    meta.className = "match-meta";
    meta.textContent =
      fmtPricing(m) +
      (m.context_length != null
        ? ", " + m.context_length.toLocaleString() + " ctx"
        : "") +
      // The published completion cap, when there is one: it is why an
      // extended-budget run on this model may be clamped below 65536.
      (m.max_completion_tokens != null
        ? ", " + m.max_completion_tokens.toLocaleString() + " max out"
        : "");
    row.append(name, id, meta);
    if (lineup.includes(m.id)) {
      row.disabled = true;
      row.title = "already in lineup";
    } else {
      row.addEventListener("click", () => addToLineup(m.id));
    }
    matchesEl.append(row);
  }
}

queryInput.addEventListener("input", renderMatches);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    searchRow.hidden = true;
    addModelBtn.setAttribute("aria-expanded", "false");
    clearSearch();
  } else if (e.key === "Enter" && !catalog.fetched) {
    // Offline fallback: nothing to search, Enter adds the exact id.
    e.preventDefault();
    const id = queryInput.value.trim();
    if (id) addToLineup(id);
  }
});

renderLineup();
loadCatalog();
autosizePrompt();

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
  updateRunState();
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
    autosizePrompt();
  }
  updateRunState();
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
    updateRunState();
  }
});

loadPrompts();

// ---- TTFT race strip. One row per model in the live run; the meter
// ---- shimmers until the first token, then locks to a bar sized by
// ---- TTFT on a shared scale. Rank is first-token finishing order.
// ---- Historical replays hide the strip: it visualizes the race in
// ---- progress, and history has no race.
let race = null;

function raceInit(models) {
  race = { rows: new Map() };
  raceGrid.replaceChildren();
  for (const model of models) {
    const wrap = document.createElement("div");
    wrap.className = "race-row working";
    const rank = document.createElement("span");
    rank.className = "race-rank";
    const name = document.createElement("span");
    name.className = "race-name";
    name.textContent = shortName(model);
    name.title = model;
    const meter = document.createElement("div");
    meter.className = "race-meter";
    const fill = document.createElement("div");
    fill.className = "race-fill";
    meter.append(fill);
    const val = document.createElement("span");
    val.className = "race-val";
    wrap.append(rank, name, meter, val);
    raceGrid.append(wrap);
    race.rows.set(model, { wrap, rank, fill, val, ttft: null, status: "working" });
  }
  raceEl.hidden = false;
  raceRender();
}

// Reruns flow back through here: the errored row returns to the
// shimmer state and races again.
function raceRestart(model) {
  const row = race !== null ? race.rows.get(model) : undefined;
  if (!row) return;
  row.status = "working";
  row.ttft = null;
  raceRender();
}

function raceTtft(model, ms) {
  const row = race !== null ? race.rows.get(model) : undefined;
  if (!row) return;
  row.ttft = ms;
  row.status = "ttft";
  raceRender();
}

function raceError(model) {
  const row = race !== null ? race.rows.get(model) : undefined;
  if (!row) return;
  row.status = "error";
  raceRender();
}

// A user Stop: the row must not keep shimmering as if still working. It
// leaves the working state and reads "stopped", ranked among nothing.
function raceStopped(model) {
  const row = race !== null ? race.rows.get(model) : undefined;
  if (!row) return;
  row.status = "stopped";
  raceRender();
}

// The server's TTFT replaces the client-side first-token measurement
// when the run completes; they differ by network jitter only.
function raceDone(model, serverTtft) {
  const row = race !== null ? race.rows.get(model) : undefined;
  if (!row) return;
  if (serverTtft != null) row.ttft = serverTtft;
  row.status = "ttft";
  raceRender();
}

function raceRender() {
  if (race === null) return;
  const rows = [...race.rows.values()];
  const ranked = rows
    .filter(r => r.status === "ttft" && r.ttft != null)
    .sort((a, b) => a.ttft - b.ttft);
  const scale = niceScale(ranked.length > 0 ? ranked[ranked.length - 1].ttft : 0);
  raceScale.textContent = "scale 0–" + scale + " ms";
  raceEl.classList.toggle("live", rows.some(r => r.status === "working"));
  for (const r of rows) r.rankN = null;
  ranked.forEach((r, i) => { r.rankN = i + 1; });
  for (const r of rows) {
    r.wrap.className =
      "race-row " + r.status + (r.rankN === 1 ? " fastest" : "");
    if (r.status === "working") {
      r.rank.textContent = "·";
      r.fill.style.width = "";
      if (r.val.textContent === "" || r.val.textContent === "failed") {
        r.val.textContent = "0 s";
      }
    } else if (r.status === "error") {
      r.rank.textContent = "—";
      r.fill.style.width = "";
      r.val.textContent = "failed";
    } else if (r.status === "stopped") {
      // No rank, no bar: the run was halted, not finished or failed.
      r.rank.textContent = "·";
      r.fill.style.width = "";
      r.val.textContent = "stopped";
    } else if (r.ttft != null) {
      r.rank.textContent = String(r.rankN);
      r.fill.style.width = Math.min(100, (r.ttft / scale) * 100) + "%";
      r.val.textContent = Math.round(r.ttft) + " ms";
    } else {
      // Done without a TTFT (server reported none): no bar, no rank.
      r.rank.textContent = "·";
      r.fill.style.width = "0%";
      r.val.textContent = "—";
    }
  }
}

function hideRace() {
  raceEl.hidden = true;
  race = null;
}

// ---- Result cards. One skeleton builder, one completion renderer;
// ---- live streams and history replay both end in completeColumn so
// ---- the textContent-only rule and the error contract hold once.

// State is conveyed twice on purpose, a colored top edge plus a text
// label, so status never rides on color alone. Only the state word is
// a polite live region; the elapsed counter next to it updates every
// second and must not be announced each time.
function setState(ui, state) {
  ui.card.dataset.state = state;
  ui.statusWord.textContent = state === "working" ? "thinking" : state;
  ui.statusTime.textContent = "";
}

function metricCell(key, label) {
  const cell = document.createElement("div");
  cell.className = "mcell " + key;
  const k = document.createElement("div");
  k.className = "mk";
  k.textContent = label;
  const v = document.createElement("div");
  v.className = "mv empty";
  v.dataset.testid = "metric-" + key;
  v.textContent = "—";
  cell.append(k, v);
  return { cell, v };
}

function clearMetric(el) {
  el.classList.add("empty");
  el.textContent = "—";
}

function setMetric(el, text, unit) {
  el.classList.remove("empty");
  el.textContent = text;
  if (unit != null) {
    const u = document.createElement("span");
    u.className = "unit";
    u.textContent = " " + unit;
    el.append(u);
  }
}

function makeColumn(model) {
  const card = document.createElement("div");
  card.className = "card";
  card.dataset.testid = "result-card";
  const header = document.createElement("div");
  header.className = "card-header";
  const name = document.createElement("span");
  name.className = "model-id";
  name.dataset.testid = "card-model";
  name.textContent = model;
  const dot = document.createElement("span");
  dot.className = "state-dot";
  const status = document.createElement("span");
  status.className = "status";
  status.dataset.testid = "card-status";
  const statusWord = document.createElement("span");
  statusWord.setAttribute("aria-live", "polite");
  const statusTime = document.createElement("span");
  status.append(statusWord, statusTime);
  header.append(name, dot, status);
  // 2px shimmer line under the header, shown by CSS while working.
  const shimmer = document.createElement("div");
  shimmer.className = "card-shimmer";
  const metrics = document.createElement("div");
  metrics.className = "metrics";
  const ttft = metricCell("ttft", "ttft");
  const total = metricCell("total", "total");
  const tok = metricCell("tok", "tok i/o");
  const cost = metricCell("cost", "cost");
  cost.v.title =
    "estimated from catalog prices and reported tokens; not billed cost";
  metrics.append(ttft.cell, total.cell, tok.cell, cost.cell);
  const body = document.createElement("div");
  body.className = "body";
  body.dataset.testid = "card-body";
  const tools = document.createElement("div");
  tools.className = "card-tools";
  card.append(header, shimmer, metrics, body, tools);
  resultsEl.append(card);
  const ui = {
    card, name, statusWord, statusTime, tools, body,
    metrics: { ttft: ttft.v, total: total.v, tok: tok.v, cost: cost.v },
  };
  setState(ui, "working");
  return ui;
}

// Restores a card to its initial working state so a rerun flows
// through the exact same streaming path as a first attempt. Clearing
// the tools also drops the failed attempt's diff, copy, fold and
// rerun controls.
function resetColumn(ui) {
  setState(ui, "working");
  ui.tools.replaceChildren();
  for (const v of Object.values(ui.metrics)) clearMetric(v);
  ui.body.className = "body";
  ui.body.replaceChildren();
  // The not-saved warning is a card-level sibling of the body, not a
  // child of it, so clearing the body above leaves it behind. Drop it
  // here or a rerun that persists cleanly would keep claiming "not
  // saved to history", and a rerun that fails again would stack a
  // second copy.
  const staleWarn = ui.card.querySelector(".save-warn");
  if (staleWarn) staleWarn.remove();
}

function toolButton(label) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "tool";
  btn.textContent = label;
  return btn;
}

// ---- Elapsed indicator. One shared interval serves every running
// ---- card and race row, so five racing models cost one timer, not
// ---- five. The counter exists because extended-budget reasoning sits
// ---- silent for minutes and a frozen "thinking" reads as hung; it
// ---- disappears at the first token.
const tickers = new Map();
let tickerTimer = null;

function renderTick(ui, entry) {
  const secs = Math.floor((performance.now() - entry.start) / 1000);
  ui.statusTime.textContent = " · " + secs + "s";
  // Race rows key by model name and a superseded run may share a name
  // with the current run, so only current-epoch ticks may touch them.
  if (race !== null && entry.model != null && entry.epoch === BenchState.viewEpoch) {
    const row = race.rows.get(entry.model);
    if (row && row.status === "working") row.val.textContent = secs + " s";
  }
}

function startTicker(ui, model, epoch) {
  const entry = { start: performance.now(), model: model, epoch: epoch };
  tickers.set(ui, entry);
  renderTick(ui, entry);
  if (tickerTimer === null) {
    tickerTimer = setInterval(() => {
      for (const [u, e] of tickers) renderTick(u, e);
    }, 1000);
  }
}

function stopTicker(ui) {
  if (!tickers.delete(ui)) return;
  if (tickers.size === 0 && tickerTimer !== null) {
    clearInterval(tickerTimer);
    tickerTimer = null;
  }
}

// Rerun is a human clicking, never automatic: that click is the
// honesty boundary between recovering from a transient failure and
// hiding one. The failed run is already persisted by the time this
// button exists; the rerun lands as a second run in the same group,
// so History keeps both the failure and the retry. Only live runs
// get the control, since rerunning history would be a new experiment
// wearing an old label.
function addRerun(ui, retry) {
  const btn = toolButton("rerun");
  btn.dataset.testid = "tool-rerun";
  btn.classList.add("rerun-btn");
  btn.title = "Retry this model in place; the failed run stays in history";
  btn.addEventListener("click", () => {
    // The reset below removes the button, but disable first so a
    // double click cannot start two reruns of the same column.
    btn.disabled = true;
    resetColumn(ui);
    // Same budget as the run being retried, not the current control
    // value: a rerun is a second sample of the same experiment.
    runOne(retry.prompt, retry.model, retry.promptId, retry.groupId, retry.budget, ui, BenchState.viewEpoch);
  });
  ui.tools.append(btn);
}

function fillMetrics(ui, result) {
  if (result.ttft_ms != null) {
    setMetric(ui.metrics.ttft, String(Math.round(result.ttft_ms)), "ms");
  }
  if (result.latency_ms != null) {
    if (result.latency_ms < 1000) {
      setMetric(ui.metrics.total, String(Math.round(result.latency_ms)), "ms");
    } else {
      setMetric(ui.metrics.total, (result.latency_ms / 1000).toFixed(2), "s");
    }
  }
  if (result.prompt_tokens != null && result.completion_tokens != null) {
    setMetric(ui.metrics.tok, result.prompt_tokens + "/" + result.completion_tokens, null);
  }
  if (result.cost_usd != null) {
    setMetric(ui.metrics.cost, fmtCost(result.cost_usd), null);
  }
}

// A result may carry BOTH partial text and an error (stream died
// partway); the error box renders below whatever text arrived, via a
// text-only node like everything else.
function applyError(ui, error) {
  if (error == null) return;
  const msg = document.createElement("div");
  msg.className = "error-msg";
  msg.dataset.testid = "card-error";
  // Visually separate the error from partial text when there is some.
  if (ui.body.textContent !== "") msg.classList.add("after-text");
  msg.textContent = error;
  ui.body.append(msg);
}

function addCopy(ui, text) {
  const btn = toolButton("copy");
  btn.dataset.testid = "tool-copy";
  let timer = null;
  btn.addEventListener("click", async () => {
    // The confirmation is a fixed literal set via textContent; the
    // copied payload itself never touches the DOM here.
    let label = "copied";
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      label = "copy failed";
    }
    btn.textContent = label;
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => {
      btn.textContent = "copy";
      timer = null;
    }, 1200);
  });
  ui.tools.append(btn);
}

function addFold(ui) {
  const btn = toolButton("fold");
  btn.dataset.testid = "tool-fold";
  btn.setAttribute("aria-expanded", "true");
  btn.addEventListener("click", () => {
    const folded = ui.body.classList.toggle("folded");
    // The unfold affordance lives in the same control: "show all"
    // where the fold used to be.
    btn.textContent = folded ? "show all" : "fold";
    btn.setAttribute("aria-expanded", String(!folded));
  });
  ui.tools.append(btn);
}

// The one completion renderer. Live streams and history replay both
// land here, so the safety rules (textContent only) and the contract
// (text or error, possibly both) are enforced in a single place.
function completeColumn(ui, result, sourceLabel, opts) {
  stopTicker(ui);
  ui.body.classList.remove("loading");
  if (!opts.streamed) {
    // textContent on purpose: model output is untrusted and must never
    // be parsed as HTML. pre-wrap in CSS keeps line breaks readable.
    ui.body.textContent = result.response_text != null ? result.response_text : "";
  }
  fillMetrics(ui, result);
  if (opts.unsaved) {
    // run_id came back null: the server spent the money and streamed
    // the response but could not persist it. Saying nothing would let
    // History silently lie by omission.
    const warn = document.createElement("div");
    warn.className = "save-warn";
    warn.dataset.testid = "save-warning";
    warn.textContent = "not saved to history";
    warn.title =
      "persisting this run failed; the response is intact but it " +
      "will not appear in History";
    ui.body.before(warn);
  }
  const error = "shownError" in opts ? opts.shownError : result.error;
  applyError(ui, error);
  // A user Stop is neither done nor a provider failure; it gets its own
  // muted state so the card never implies the model finished or errored.
  setState(ui, opts.stopped ? "stopped" : error != null ? "error" : "done");
  // Rerun leads the action row so the recovery control is where the
  // eye lands first on a failed card.
  if (opts.retry) addRerun(ui, opts.retry);
  if (result.response_text != null) {
    addCopy(ui, result.response_text);
    addFold(ui);
  }
  registerDiffable(ui, result, sourceLabel);
  // Budget note on history replay only, never live columns: two
  // attempts at different budgets are different experiments, and the
  // replay must say which is which. Pre-budget rows carry null and
  // show nothing.
  if (opts.budgetBadge && result.max_tokens != null) {
    const note = document.createElement("span");
    note.className = "budget-note";
    note.textContent = "budget " + result.max_tokens;
    ui.tools.append(note);
  }
}

function fillColumn(ui, result, sourceLabel) {
  completeColumn(ui, result, sourceLabel, {
    streamed: false,
    budgetBadge: true,
    retry: null,
  });
}

// ---- Diff view. Pure functions first (tokenize, lcs, render) so a JS
// ---- harness could test them later; DOM wiring below.

function renderDiff(ops, container) {
  container.replaceChildren();
  // Adjacent tokens with the same op merge into one node: fewer DOM
  // nodes and the del/ins tint reads as one span per changed region.
  let run = null;
  function flush() {
    if (run === null) return;
    if (run.op === "same") {
      container.append(document.createTextNode(run.text));
    } else {
      // createElement plus textContent only: model output stays
      // untrusted in diff form, exactly as in the cards.
      const el = document.createElement(run.op === "del" ? "del" : "ins");
      el.textContent = run.text;
      container.append(el);
    }
    run = null;
  }
  for (const o of ops) {
    if (run !== null && run.op === o.op) {
      run.text += o.raw;
    } else {
      flush();
      run = { op: o.op, text: o.raw };
    }
  }
  flush();
}

const diffPanel = document.getElementById("diff-panel");
const diffTitle = document.getElementById("diff-title");
const diffBody = document.getElementById("diff-body");
// Armed side of a pending diff. Holds the result data, not just the
// element: history replay replaces the cards, and surviving that is
// what makes live-vs-historical diffs possible.
let armedDiff = null;

function closeDiffPanel() {
  diffPanel.hidden = true;
  diffTitle.textContent = "";
  diffBody.replaceChildren();
}

function setArmed(btn, on) {
  btn.classList.toggle("armed", on);
  btn.setAttribute("aria-pressed", String(on));
  // The armed side carries a filled marker: "diff ●".
  btn.textContent = (btn.dataset.base || "diff") + (on ? " ●" : "");
}

// Arming deliberately survives history replay (that is what makes
// live-vs-historical diffs possible), but not a new Run: its cards
// supersede the comparison the armed result came from, and a later
// toggle would silently diff against a vanished card.
function disarmDiff() {
  if (armedDiff !== null) {
    // The armed button may already be detached (replaced results);
    // clearing the state is harmless either way.
    setArmed(armedDiff.btn, false);
    armedDiff = null;
  }
}

function openDiff(a, b) {
  diffTitle.textContent = a.label + " ⇄ " + b.label;
  const ta = tokenizeDiff(a.result.response_text);
  const tb = tokenizeDiff(b.result.response_text);
  if (ta.length > DIFF_TOKEN_LIMIT || tb.length > DIFF_TOKEN_LIMIT) {
    diffBody.textContent = "responses too large to diff";
  } else {
    renderDiff(diffTokens(ta, tb), diffBody);
  }
  diffPanel.hidden = false;
}

function registerDiffable(ui, result, sourceLabel) {
  // Error-only cards have nothing to diff; partial text (the
  // both-text-and-error contract) is diffable and labeled as such.
  if (result.response_text == null || sourceLabel == null) return;
  const label = sourceLabel + (result.error != null ? " (partial)" : "");
  const btn = toolButton(result.error != null ? "diff (partial)" : "diff");
  btn.dataset.testid = "tool-diff";
  btn.classList.add("diff-btn");
  btn.dataset.base = btn.textContent;
  btn.setAttribute("aria-pressed", "false");
  btn.addEventListener("click", () => {
    if (armedDiff !== null && armedDiff.btn === btn) {
      setArmed(btn, false);
      armedDiff = null;
      return;
    }
    if (armedDiff === null) {
      armedDiff = { btn: btn, result: result, label: label };
      setArmed(btn, true);
      return;
    }
    openDiff(armedDiff, { btn: btn, result: result, label: label });
    setArmed(armedDiff.btn, false);
    armedDiff = null;
  });
  ui.tools.append(btn);
}

document.getElementById("diff-close").addEventListener("click", closeDiffPanel);

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
  updateRunState();
  ui.body.classList.add("loading");
  ui.body.textContent = "awaiting first token";
  startTicker(ui, model, epoch);
  raceRestart(model);
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
      const entry = tickers.get(ui);
      // The race strip belongs to the current view; a superseded run
      // sharing a model name with the new one must not repaint it.
      if (entry && current()) {
        const ttftMs = performance.now() - entry.start;
        setMetric(ui.metrics.ttft, String(Math.round(ttftMs)), "ms");
        raceTtft(model, ttftMs);
      }
      stopTicker(ui);
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
    stopTicker(ui);
    if (current()) {
      if (result.stopped) {
        raceStopped(model);
      } else if (result.error != null) {
        raceError(model);
      } else {
        raceDone(model, result.ttft_ms);
      }
      BenchState.inflightRuns -= 1;
      updateRunState();
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
    completeColumn(ui, result, model + " (live)", {
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
          const entry = tickers.get(ui);
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
  const budget = budgetValue;
  const models = checkedModels();
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
  closeDiffPanel();
  disarmDiff();
  updateRunState();
  BenchState.sessionStats.runs += 1;
  BenchState.renderStats();
  // One request per model instead of one batch: /compare returns only when
  // its slowest model finishes, and the bench exists to watch fast models
  // land first. Cards are created up front so order tracks the chip
  // list, not response arrival.
  raceInit(models);
  const columns = models.map(makeColumn);
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
      updateRunState();
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
  hideRace();
  closeDiffPanel();
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
  updateRunState();
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
    const i = lineup.indexOf(m);
    return i === -1 ? lineup.length : i;
  };
  results.sort((a, b) => rank(a.model) - rank(b.model));
  for (const result of results) {
    fillColumn(
      makeColumn(result.model), result,
      result.model + ", comparison #" + group.id
    );
  }
  window.scrollTo({ top: 0 });
}

async function showRun(runId) {
  // Same ownership rule as showGroup.
  const epoch = BenchState.newViewEpoch();
  updateRunState();
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
    fillColumn(makeColumn(result.model), result, result.model + ", run #" + run.id);
  }
  window.scrollTo({ top: 0 });
}
