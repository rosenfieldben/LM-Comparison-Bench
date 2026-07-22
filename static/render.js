// Result cards, the TTFT race strip, the shared elapsed ticker, and the
// one completion renderer. Live streams and history replay both end in
// completeColumn, so the textContent-only rule and the error contract
// hold in one place. Exposed on window.BenchRender. Two forward edges
// (addRerun -> runOne, completeColumn -> registerDiffable) reference the
// stream and diff modules, which load later; they are called only at
// click/completion time, so the globals exist by then.
(function () {
  const { shortName, fmtCost, niceScale } = window.BenchLib;
  const resultsEl = document.getElementById("results");
  const raceEl = document.getElementById("race");
  const raceGrid = document.getElementById("race-grid");
  const raceScale = document.getElementById("race-scale");

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
    BenchDiff.registerDiffable(ui, result, sourceLabel);
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

  window.BenchRender = {
    raceInit, raceRestart, raceTtft, raceError, raceStopped, raceDone,
    hideRace, makeColumn, completeColumn, fillColumn, setMetric,
    startTicker, stopTicker, toolButton, tickers,
  };
})();
