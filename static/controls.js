// The console deck: lineup, model search, UI prefs (theme, motion,
// density, budget), the run estimate, and the run-state gate. Exposed on
// window.BenchControls. Wiring (listeners, first paint of the deck) runs
// from init(), which boot.js calls after every module has loaded.
(function () {
  const { shortName, fmtEstimate } = window.BenchLib;
  const BS = window.BenchState;

  // Seed for a fresh browser only. The live lineup is a localStorage
  // preference of THIS browser, not bench data: keeping it out of sqlite
  // means bench.db stays purely runs and prompts, and losing it costs
  // four clicks. The key predates the interface overhaul and must never
  // change: users have lineups stored under it. Declared before the
  // exported object so its loadLineup() call below is out of the
  // temporal dead zone.
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
      if (Array.isArray(parsed) && parsed.every((x) => typeof x === "string")) {
        return parsed;
      }
    } catch (err) {
      // Unparseable storage falls through to the defaults.
    }
    return [...DEFAULT_LINEUP];
  }

  // Exported surface. Mutable fields (lineup, budgetValue) live here as
  // properties so a reassignment is visible to every reader.
  const C = {
    lineup: loadLineup(),
    // What /compare/stream sends as the completion budget; per session on
    // purpose (extended costs real money, so the safe default reasserts on
    // reload), which is why it is not persisted like density.
    budgetValue: "standard",
    checkedModels,
    updateRunState,
    autosizePrompt,
    init,
  };

  const promptEl = document.getElementById("prompt");
  const modelsEl = document.getElementById("models");
  const runBtn = document.getElementById("run");
  const stopBtn = document.getElementById("stop");
  const savedSelect = document.getElementById("saved-prompts");
  const deleteBtn = document.getElementById("delete-prompt");
  const linkedEl = document.getElementById("linked-name");
  const lineupLabel = document.getElementById("lineup-label");
  const runNote = document.getElementById("run-note");
  const addModelBtn = document.getElementById("add-model");
  const searchRow = document.getElementById("model-search");
  const queryInput = document.getElementById("model-query");
  const searchMsg = document.getElementById("search-msg");
  const matchesEl = document.getElementById("model-matches");
  const themeBtn = document.getElementById("theme-btn");
  const motionBtn = document.getElementById("motion-btn");

  function saveLineup() {
    // Guarded like the pref helpers below: a quota or SecurityError must
    // not abort an add or remove. The in-memory lineup stays authoritative;
    // persistence just lapses for the session.
    try {
      localStorage.setItem(LINEUP_KEY, JSON.stringify(C.lineup));
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

  // Snapshot of GET /models; fetched=false switches the picker to the
  // exact-id fallback so the bench stays usable on an offline boot.
  let catalog = { fetched: false, models: [] };

  // ---- UI prefs: theme override (auto follows the OS), motion, density.
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

  // Motion off kills every animation via CSS; elapsed-time counters are
  // plain text updates and keep going. prefers-reduced-motion does the
  // same regardless of this toggle.
  let motionOn = prefGet("bench-motion", "on") !== "off";

  function applyMotion() {
    document.documentElement.dataset.motion = motionOn ? "on" : "off";
    motionBtn.textContent = "motion " + (motionOn ? "on" : "off");
  }

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

  // Density persists, unlike the budget: layout taste is harmless.
  let densityValue = prefGet("bench-density", "comfortable");
  if (densityValue !== "compact") densityValue = "comfortable";

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
    for (const model of C.lineup) {
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
        C.lineup = C.lineup.filter((m) => m !== model);
        saveLineup();
        renderLineup();
        BS.renderStats();
      });
      chip.append(label, rm);
      modelsEl.append(chip);
    }
    updateRunState();
    BS.renderStats();
  }

  function checkedModels() {
    return [...modelsEl.querySelectorAll("input:checked")].map((b) => b.value);
  }

  function setAllChecked(on) {
    for (const box of modelsEl.querySelectorAll("input[type=checkbox]")) {
      box.checked = on;
      box.closest(".chip").classList.toggle("on", on);
    }
    updateRunState();
  }

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
        const m = catalog.models.find((x) => x.id === id);
        if (!m || m.completion_price == null) {
          computable = false;
          break;
        }
        const cap =
          m.max_completion_tokens != null
            ? Math.min(m.max_completion_tokens, BUDGET_TOKENS[C.budgetValue])
            : BUDGET_TOKENS[C.budgetValue];
        est += m.completion_price * cap;
      }
      if (computable) {
        text +=
          " · max output cost $" +
          fmtEstimate(est) +
          "/run (input not included)";
      }
    }
    runNote.textContent = text;
  }

  function renderLinked() {
    const opt = savedSelect.selectedOptions[0];
    linkedEl.textContent =
      savedSelect.value !== "" && opt ? "linked: " + opt.textContent : "";
  }

  function updateRunState() {
    const checked = checkedModels().length;
    runBtn.disabled =
      BS.inflightRuns > 0 || promptEl.value.trim() === "" || checked === 0;
    // Stop is live exactly while runs are: it acts on the in-flight
    // controllers and has nothing to do when the count is zero.
    stopBtn.disabled = BS.inflightRuns === 0;
    deleteBtn.disabled = savedSelect.value === "";
    lineupLabel.textContent = "Lineup " + checked + "/" + C.lineup.length;
    renderLinked();
    updateEstimate();
  }

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
    if (C.lineup.includes(id)) return;
    C.lineup.push(id);
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
      "$" +
      (m.prompt_price * 1e6).toFixed(2) +
      " / $" +
      (m.completion_price * 1e6).toFixed(2) +
      " per 1M in/out"
    );
  }

  function renderMatches() {
    matchesEl.replaceChildren();
    if (!catalog.fetched) return;
    const q = queryInput.value.trim().toLowerCase();
    if (q === "") return;
    const hits = catalog.models
      .filter(
        (m) =>
          m.id.toLowerCase().includes(q) ||
          (m.name || "").toLowerCase().includes(q),
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
      if (C.lineup.includes(m.id)) {
        row.disabled = true;
        row.title = "already in lineup";
      } else {
        row.addEventListener("click", () => addToLineup(m.id));
      }
      matchesEl.append(row);
    }
  }

  function init() {
    document.getElementById("bar-host").textContent =
      location.host || "localhost:8000";

    themeBtn.addEventListener("click", () => {
      themeMode = THEMES[(THEMES.indexOf(themeMode) + 1) % THEMES.length];
      prefSet("bench-theme", themeMode);
      applyTheme();
    });
    applyTheme();

    motionBtn.addEventListener("click", () => {
      motionOn = !motionOn;
      prefSet("bench-motion", motionOn ? "on" : "off");
      applyMotion();
    });
    applyMotion();

    initSeg(document.getElementById("budget-seg"), C.budgetValue, (v) => {
      C.budgetValue = v;
      updateRunState();
    });

    document.documentElement.dataset.density = densityValue;
    initSeg(document.getElementById("density-seg"), densityValue, (v) => {
      densityValue = v;
      document.documentElement.dataset.density = v;
      prefSet("bench-density", v);
    });

    document
      .getElementById("select-all")
      .addEventListener("click", () => setAllChecked(true));
    document
      .getElementById("select-none")
      .addEventListener("click", () => setAllChecked(false));

    promptEl.addEventListener("input", () => {
      BS.selectedPromptId = null;
      savedSelect.value = "";
      autosizePrompt();
      updateRunState();
    });

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
  }

  window.BenchControls = C;
})();
