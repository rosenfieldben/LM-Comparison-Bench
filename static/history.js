// History: the flat list of past runs and the replay of a stored run or
// comparison group. Loads on expand rather than page load. Exposed on
// window.BenchHistory; showGroup and showRun are the replay entry points
// (showGroup is also driven by the browser suite through page.evaluate).
(function () {
  const historyEl = document.getElementById("history");
  const historyList = document.getElementById("history-list");
  const historyFilter = document.getElementById("history-filter");
  const historyNote = document.getElementById("history-note");
  const resultsEl = document.getElementById("results");
  const runLabelEl = document.getElementById("run-label");
  const runControlsEl = document.getElementById("run-controls");
  const promptMsg = document.getElementById("prompt-msg");

  // ---- History. Loads on expand rather than page load: it is the
  // ---- rarely used half of the page and a stale list is worse than a
  // ---- slightly later one.

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

  // The panel is emptied the moment it opens and refilled only when the
  // fetch comes back, so an empty list means two different things while
  // one is in flight: still loading, or nothing to show. On a slow link
  // that blank panel reads as broken, and anything watching for a row
  // cannot tell a slow load from a missing row. So the list names its own
  // state in words and in an attribute: idle before it has ever opened,
  // then loading, then exactly one of ready, empty, or error.
  //
  // The attribute is not decoration. It is the only way to ask "has this
  // finished" from outside, and a negative assertion about history is
  // meaningless without it, because an unfinished load satisfies every
  // one of them.
  function setHistoryState(state, message) {
    historyList.dataset.state = state;
    // Assigning textContent also clears any previous rows, which is why
    // the ready case passes an empty string before appending its own.
    historyList.textContent = message;
  }

  async function loadHistory() {
    if (historyListController !== null) historyListController.abort();
    const controller = new AbortController();
    historyListController = controller;
    // Synchronous, and that is load-bearing rather than incidental.
    // Everything above this line runs before the function's first await,
    // so by the time the toggle handler returns the panel already reads
    // "loading". Without that, a reopened panel would still be showing
    // the PREVIOUS load's terminal state, and anything waiting for the
    // panel to settle would be satisfied by an answer to an older
    // question. Do not move this below the fetch.
    setHistoryState("loading", "loading history");
    historyNote.textContent = "";
    let data;
    try {
      const resp = await fetch("/runs?limit=" + HISTORY_LIMIT, {
        signal: controller.signal,
      });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch (err) {
      // An aborted load has been superseded, so it leaves the state
      // alone: the newer load already set its own, and overwriting it
      // would report the loser's outcome.
      if (controller.signal.aborted) return;
      // Observability parity with the finish() guards. A failure that
      // exists only as a DOM attribute is a failure nothing watching the
      // page can see, and this one hid for three rounds of a CI flake:
      // the panel cleared its rows, set state=error, and a test then made
      // an honest assertion against an honestly empty list. The console
      // is where a page says it went wrong.
      console.error("history load failed", err);
      setHistoryState("error", "failed to load history: " + err.message);
      return;
    }
    if (controller !== historyListController) return;
    if (data.runs.length === 0) {
      setHistoryState("empty", "no runs yet");
      return;
    }
    setHistoryState("ready", "");
    for (const run of data.runs) {
      // Buttons, not divs: rows must be reachable by keyboard.
      const row = document.createElement("button");
      row.type = "button";
      row.className = "hrow";
      row.dataset.testid = "history-row";
      // The filter matches model ids even though rows only show a count.
      row.dataset.hay = (
        run.prompt_text +
        " " +
        run.models.join(" ")
      ).toLowerCase();
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
      let countText =
        uniqueModels + (uniqueModels === 1 ? " model" : " models");
      if (attempts > uniqueModels) {
        countText += " · " + attempts + " attempts";
      }
      count.textContent = countText;
      // Badges for the controls this entry actually set, and nothing for a
      // control it left to the provider. A group shows its declared record;
      // a lone run shows what its payload proves, which never includes
      // routing (see store._controls_from_request for why).
      const badges = document.createElement("span");
      badges.className = "hctl";
      for (const badge of window.BenchLib.controlBadges(run.params)) {
        const chip = document.createElement("span");
        chip.className = "ctl-badge";
        chip.dataset.testid = "history-control-badge";
        chip.textContent = badge.text;
        chip.title = badge.title;
        badges.append(chip);
      }
      row.append(time, text, badges, count);
      row.title = run.models.join(", ");
      row.addEventListener("click", () =>
        run.type === "group" ? showGroup(run.id) : showRun(run.id),
      );
      // Blind rating starts HERE, from the list, and never from a
      // comparison already on screen. Opening it after a replay means
      // the identities were painted first, and "hidden" in a browser is
      // a style rule anyone can undo plus a frame the rater may already
      // have seen. From the list there is nothing to undo: the page
      // never held the answer key.
      //
      // Offered on a comparison and not on a lone run, because the whole
      // method is hiding which of SEVERAL answers came from which model.
      if (run.type === "group" && run.models.length > 1) {
        const pair = document.createElement("div");
        pair.className = "hrow-pair";
        const blind = document.createElement("button");
        blind.type = "button";
        blind.className = "tool";
        blind.dataset.testid = "history-rate-blind";
        blind.textContent = "rate blind";
        blind.title =
          "open this comparison with every identity withheld by the " +
          "server, rate the answers, then reveal";
        blind.addEventListener("click", () =>
          window.BenchRating.startBlind(run.id),
        );
        pair.append(row, blind);
        historyList.append(pair);
        continue;
      }
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

  // Whether routing is recoverable from this source. A group stores its
  // declared controls, so reuse restores them exactly. An ungrouped run has
  // no stored set at all; its controls are derived from its recorded
  // payload, and routing is the one control that derivation cannot see,
  // because provider.sort rides every payload the bench has ever sent and a
  // stored throughput is indistinguishable from a chosen one.
  //
  // So reuse from an ungrouped run is lossy for routing BY CONSTRUCTION, and
  // that is a documented consequence rather than a bug to fix: the
  // alternative is guessing, and a guessed routing badge would be the same
  // truth defect as any other default rendered as a choice. What is not
  // acceptable is being quiet about it, so the button says so before the
  // click and the composer says so after.
  const UNGROUPED_ROUTING_NOTE =
    "routing is not restored: an ungrouped run does not record it";

  // The controls a replayed comparison ran under, badges plus the system
  // prompt in full. The badge alone says only that a system prompt existed,
  // which is the right amount for a one-line history row and not enough to
  // reproduce the experiment; this is the view where the text belongs.
  //
  // Cleared by every path that takes over the view, including the failure
  // and loading states, because a stale controls line under a new banner
  // would attribute one comparison's experiment to another.
  function renderRunControls(params, source) {
    runControlsEl.replaceChildren();
    const badges = window.BenchLib.controlBadges(params);
    if (badges.length > 0) {
      const strip = document.createElement("div");
      strip.className = "ctl-badges";
      strip.dataset.testid = "run-control-badges";
      for (const badge of badges) {
        const chip = document.createElement("span");
        chip.className = "ctl-badge";
        chip.dataset.testid = "run-control-badge";
        chip.textContent = badge.text;
        chip.title = badge.title;
        strip.append(chip);
      }
      runControlsEl.append(strip);
    }
    // textContent, never innerHTML: a system prompt is user text and gets
    // the same no-HTML treatment model output does.
    if (params?.system) {
      const sys = document.createElement("div");
      sys.className = "ctl-system";
      sys.dataset.testid = "run-system-prompt";
      sys.textContent = params.system;
      sys.title = "the system prompt this comparison was run with";
      runControlsEl.append(sys);
    }
    // The reuse action offers itself even when nothing was set, because the
    // prompt alone is worth reusing and a comparison with no controls is
    // still an experiment someone may want to repeat.
    if (source) renderReuse(source);
  }

  function renderReuse(source) {
    const lossy = source.kind === "run";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tool";
    btn.dataset.testid = "reuse-comparison";
    btn.textContent = "reuse";
    btn.title = lossy
      ? "Fill the composer with this prompt and its controls, leaving your " +
        "lineup alone. Nothing runs until you press Run. " +
        UNGROUPED_ROUTING_NOTE
      : "Fill the composer with this prompt and its controls, leaving your " +
        "lineup alone. Nothing runs until you press Run";
    btn.addEventListener("click", () => {
      BenchControls.reuseExperiment(source);
      // Say what happened, and say what did not. A prefill that silently
      // dropped a control the source had would hand the user a different
      // experiment wearing the same label.
      promptMsg.textContent = lossy
        ? "composer filled from run #" +
          source.id +
          "; " +
          UNGROUPED_ROUTING_NOTE
        : "composer filled from comparison #" + source.id;
    });
    const row = document.createElement("div");
    row.className = "ctl-reuse";
    row.append(btn);
    // Blind rating is offered on a comparison and not on a lone run: the
    // whole method is hiding which of SEVERAL answers came from which
    // model, and a single card has nothing to be blind between.
    if (source.kind === "group" && source.resultIds.length > 1) {
      const rate = document.createElement("button");
      rate.type = "button";
      rate.dataset.testid = "rate-blind";
      rate.textContent = "Rate blind";
      rate.title =
        "hide identities, cost and timing, rate each answer, then reveal";
      rate.addEventListener("click", () => {
        rate.disabled = true;
        window.BenchRating.start(source.id, source.resultIds);
      });
      row.append(rate);
    }
    runControlsEl.append(row);
  }

  // A history load owns the results area from the click, not from the
  // moment its fetch succeeds. Clearing the cards, race and diff and
  // showing a state up front is what keeps a failed or slow load from
  // leaving its banner over another run's cards: the grid and the banner
  // always agree. The armed diff side survives (closeDiffPanel does not
  // disarm), so cross-replay diffing still works.
  function renderHistoryState(label, testid, cls, boxText) {
    renderRunControls(null);
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

  // Declared members with no recorded run, drawn as inert placeholders so
  // an incomplete experiment looks incomplete rather than smaller. Without
  // these a comparison that declared four models and recorded two replayed
  // as a two-model comparison, which is a quieter lie than an error: the
  // reader has no way to know anything is missing.
  //
  // Built here rather than through makeColumn, and that is the enforcement
  // of its inertness rather than a style choice. A real column wires a
  // rerun control, diff arming, a race entry and session accounting, and a
  // placeholder must have none of those: there is no result to diff, no
  // timing to race, nothing to count, and "run the missing member into its
  // declared slot" is a real capability that this branch is not the place
  // to build. Constructing a plain element cannot acquire any of that by
  // accident.
  //
  // Appended after the recorded results rather than slotted at the declared
  // index: the sort above is position-based and a member that never ran has
  // no position, so each placeholder names its declared index instead of
  // pretending to occupy it.
  function renderMissingMembers(group, results) {
    const declared = group.models;
    if (!Array.isArray(declared)) return;
    // Counts, not a Set. A lineup may legitimately declare one model
    // twice (neither /compare nor /groups rejects it, and with the seed
    // control a same-model pair is a determinism check rather than a
    // mistake), and set membership would let a single recorded run
    // suppress BOTH placeholders. That is the same quiet shrink this
    // function exists to stop, so it has to count.
    const ran = new Map();
    for (const r of results) ran.set(r.model, (ran.get(r.model) ?? 0) + 1);
    declared.forEach((model, index) => {
      const remaining = ran.get(model) ?? 0;
      if (remaining > 0) {
        ran.set(model, remaining - 1);
        return;
      }
      const card = document.createElement("div");
      card.className = "card missing";
      card.dataset.testid = "missing-member";
      const name = document.createElement("div");
      name.className = "cardname";
      name.dataset.testid = "missing-member-model";
      name.textContent = model;
      const note = document.createElement("div");
      note.className = "missing-note";
      note.textContent = "declared at position " + index + ", never ran";
      card.title =
        "this comparison declared " + model + " but no run for it was recorded";
      card.append(name, note);
      resultsEl.append(card);
    });
  }

  // The card order for a replayed comparison, in one place because two
  // callers need the same one: the cards are rendered in it, and the
  // blind-rating control pairs result ids against the cards on screen by
  // index. Two copies of this rule that drifted would silently attribute
  // one model's rating to another, which is the worst failure this
  // feature has.
  function sortedResults(group) {
    const results = group.runs.flatMap((r) => r.results);
    const rank = (m) => {
      const i = BenchControls.lineup.indexOf(m);
      return i === -1 ? BenchControls.lineup.length : i;
    };
    results.sort((a, b) => {
      const ap = a.position;
      const bp = b.position;
      if (ap != null && bp != null) return ap - bp;
      if (ap != null) return -1;
      if (bp != null) return 1;
      return rank(a.model) - rank(b.model);
    });
    return results;
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
      "history-loading",
      "loading",
      "loading comparison",
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
        "history-failure",
        "failure",
        "could not load this comparison; nothing from another run is shown",
      );
      return;
    }
    if (epoch !== BenchState.viewEpoch) return;
    // Replace the loading state with the comparison. Race and diff were
    // already cleared when the loading state was rendered.
    resultsEl.replaceChildren();
    runLabelEl.textContent =
      "Historical comparison #" +
      group.id +
      " from " +
      group.created_at.slice(0, 19).replace("T", " ") +
      " UTC";
    // The stored record, so a group can show a routing badge; a lone run
    // cannot, since routing is not recoverable from a payload.
    renderRunControls(group.params, {
      kind: "group",
      id: group.id,
      // In the same order the cards are appended below, because the
      // rating module pairs them by index against what is on screen.
      resultIds: sortedResults(group).map((r) => r.id),
      // The declaration first, since that is what the comparison WAS, with
      // the first member's text as the fallback for groups created before
      // the column existed. ?? and not ||, and the guard is the point:
      // group.prompt is null on every legacy group, and a null reaching a
      // textarea would render the four characters "null" as if someone had
      // typed them. Absence has to render as absence.
      prompt:
        group.prompt ??
        (group.runs.length > 0 ? group.runs[0].prompt_text : ""),
      params: group.params,
    });
    // Cards in the order the comparison actually had. Runs persist in
    // completion order, so run order alone would shuffle cards between
    // replays.
    //
    // The declared position wins when the row carries one: it was
    // recorded at request time, so it reconstructs the original layout
    // from the rows themselves. Sorting by the CURRENT chip order was the
    // whole defect the column exists to fix, because the lineup drifts as
    // it is edited and a replay would then rearrange an old comparison to
    // match a lineup it never ran under.
    //
    // Chip order stays the fallback for rows written before the column
    // existed, and positioned rows lead unpositioned ones so a group that
    // mixes the two (a legacy group with a later rerun) is at least
    // deterministic. Models no longer in the lineup keep run order at the
    // end, as before.
    const results = sortedResults(group);
    for (const result of results) {
      BenchRender.fillColumn(
        BenchRender.makeColumn(result.model),
        result,
        result.model + ", comparison #" + group.id,
      );
    }
    renderMissingMembers(group, results);
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
      "history-loading",
      "loading",
      "loading run",
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
        "history-failure",
        "failure",
        "could not load this run; nothing from another run is shown",
      );
      return;
    }
    if (epoch !== BenchState.viewEpoch) return;
    // Same completion renderer as a live run so the textContent and
    // null-content guarantees hold for stored data too. Race and diff were
    // cleared with the loading state.
    resultsEl.replaceChildren();
    runLabelEl.textContent =
      "Historical run #" +
      run.id +
      " from " +
      run.created_at.slice(0, 19).replace("T", " ") +
      " UTC";
    // Derived from the payload rather than a stored record: this run has no
    // group, so no controls set was ever declared for it. Any result of the
    // run proves the same set, since one run sends one experiment.
    renderRunControls(run.params, {
      kind: "run",
      id: run.id,
      prompt: run.prompt_text,
      params: run.params,
    });
    for (const result of run.results) {
      BenchRender.fillColumn(
        BenchRender.makeColumn(result.model),
        result,
        result.model + ", run #" + run.id,
      );
    }
    window.scrollTo({ top: 0 });
  }

  function init() {
    // The panel claims its own state synchronously with the click, and
    // that is load-bearing rather than belt-and-braces. `toggle` is
    // dispatched asynchronously, so between the click and loadHistory
    // there is a real window in which data-state still holds the
    // PREVIOUS load's terminal value. Anything waiting for the panel to
    // settle can be answered inside that window by an answer to an older
    // question: a panel reopened after a failure reported "error" for a
    // load that had not started yet, which is precisely how a retry could
    // conclude that the retry had also failed. Measured, not theorized.
    //
    // At click time `open` is still the old value, because the default
    // action runs after listeners, so !open means "about to open".
    // loadHistory sets the same state again on the toggle; assigning it
    // twice is idempotent, and this one is only about closing the window.
    historyEl.addEventListener("click", (event) => {
      if (!historyEl.open && event.target.closest("summary")) {
        setHistoryState("loading", "loading history");
      }
    });
    historyEl.addEventListener("toggle", () => {
      if (historyEl.open) loadHistory();
    });
    historyFilter.addEventListener("input", applyHistoryFilter);
  }

  window.BenchHistory = { showGroup, showRun, init };
})();
