// Result cards, the TTFT race strip, the shared elapsed ticker, and the
// one completion renderer. Live streams and history replay both end in
// completeColumn, so the textContent-only rule and the error contract
// hold in one place. Exposed on window.BenchRender. Two forward edges
// (addRerun -> runOne, completeColumn -> registerDiffable) reference the
// stream and diff modules, which load later; they are called only at
// click/completion time, so the globals exist by then.
(function () {
  const { shortName, fmtCost, fmtBilled, niceScale } = window.BenchLib;
  const { reasoningAteTheOutput, reasoningShare } = window.BenchLib;

  // The cost cell's default explanation, restored on reset so a rerun of a
  // billed run does not keep claiming the previous attempt's charge.
  const COST_ESTIMATE_TITLE =
    "estimated from catalog prices and reported tokens; not billed cost";
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
      race.rows.set(model, {
        wrap,
        rank,
        fill,
        val,
        ttft: null,
        status: "working",
      });
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

  // A spend refusal never reached a provider, so the strip must not call it
  // a failure: that is the same mis-description the card stopped making.
  // Ranked among nothing, like a stop, because there is no time to rank.
  function raceRefused(model) {
    const row = race !== null ? race.rows.get(model) : undefined;
    if (!row) return;
    row.status = "refused";
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
      .filter((r) => r.status === "ttft" && r.ttft != null)
      .sort((a, b) => a.ttft - b.ttft);
    const scale = niceScale(
      ranked.length > 0 ? ranked[ranked.length - 1].ttft : 0,
    );
    raceScale.textContent = "scale 0–" + scale + " ms";
    raceEl.classList.toggle(
      "live",
      rows.some((r) => r.status === "working"),
    );
    // Min-rank: equal times share a place and the next one skips, so a
    // three-way tie reads 1, 1, 1, 4 rather than 1, 2, 3, 4. Two models
    // that measured identically are tied, and numbering them anyway
    // would show a difference that is not in the data, which is the same
    // defect the report's ranking avoids for the same reason. Ties are
    // not hypothetical: ttft is rounded to a tenth of a millisecond, and
    // on a fast local stub two models routinely land on the same value.
    // Min-rank: equal times share a place and the next one skips. Two
    // models that measured identically are tied, and numbering them
    // anyway would show a difference that is not in the data, the same
    // rule the report's ranking follows. Ties are not hypothetical here:
    // ttft is rounded to a tenth of a millisecond, so on a fast local
    // stub two models routinely land on the same value.
    for (const r of rows) r.rankN = null;
    const places = window.BenchLib.minRanks(ranked.map((r) => r.ttft));
    ranked.forEach((r, i) => {
      r.rankN = places[i];
    });
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
      } else if (r.status === "refused") {
        // No rank, no bar: the ceiling refused it before any provider saw
        // it, so there is nothing to time and nothing that failed.
        r.rank.textContent = "·";
        r.fill.style.width = "";
        r.val.textContent = "refused";
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
    // THE UNIT, SAID ONCE, on the surface that shows the number. Both
    // figures come from OpenRouter's usage object, which the
    // usage-accounting page describes as "Prompt and completion token
    // counts using the model's native tokenizer"
    // (openrouter.ai/docs/use-cases/usage-accounting, read 2026-08-12).
    // One unit for both halves of the pair, and this says which.
    tok.v.title =
      "prompt/completion tokens actually used, counted by the model's " +
      "own tokenizer as OpenRouter reported them";
    // Reasoning tokens are billed and consume the completion budget while
    // never appearing in the answer, which is the whole reason the two
    // tiers exist. A cell of its own makes that burn a number on every
    // card instead of an inference from a truncated response.
    const reasoning = metricCell("reasoning", "reasoning");
    reasoning.v.title =
      "hidden reasoning tokens, billed as completion tokens and counted " +
      "against the budget. Same unit as tok i/o: actually used, counted " +
      "by the model's own tokenizer";
    const cost = metricCell("cost", "cost");
    cost.v.title = COST_ESTIMATE_TITLE;
    metrics.append(ttft.cell, total.cell, tok.cell, reasoning.cell, cost.cell);
    // Which host actually served the request. Routing is by throughput, so
    // the provider is chosen per request and varies between two runs of the
    // same model; naming it makes the largest confound in any comparison
    // visible rather than assumed away. Hidden until one is known, because
    // an empty caption reads as a layout fault.
    const caption = document.createElement("div");
    caption.className = "card-caption";
    caption.dataset.testid = "card-provider";
    caption.hidden = true;
    const body = document.createElement("div");
    body.className = "body";
    body.dataset.testid = "card-body";
    const tools = document.createElement("div");
    tools.className = "card-tools";
    card.append(header, shimmer, metrics, caption, body, tools);
    resultsEl.append(card);
    const ui = {
      card,
      name,
      statusWord,
      statusTime,
      tools,
      body,
      caption,
      metrics: {
        ttft: ttft.v,
        total: total.v,
        tok: tok.v,
        reasoning: reasoning.v,
        cost: cost.v,
      },
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
    // The cost cell's tooltip is rewritten when a billed figure lands, so
    // it has to come back with the metrics: a rerun that the platform does
    // not price would otherwise still show the previous attempt's charge.
    ui.metrics.cost.title = COST_ESTIMATE_TITLE;
    ui.caption.textContent = "";
    ui.caption.hidden = true;
    ui.body.className = "body";
    ui.body.replaceChildren();
    // Card-level warnings are siblings of the body, not children of it,
    // so clearing the body above leaves them behind. Drop them here or a
    // rerun that persists cleanly would keep claiming "not saved to
    // history", and a rerun that fails again would stack a second copy.
    //
    // The reasoning indicator joined this list rather than getting its
    // own removal, because it is the same kind of thing in the same
    // position and the failure mode is identical: a card that reruns to
    // a healthy result would otherwise keep a share line describing
    // tokens the current attempt never spent.
    for (const stale of ui.card.querySelectorAll(
      ".save-warn, .reasoning-warn",
    )) {
      stale.remove();
    }
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
    if (
      race !== null &&
      entry.model != null &&
      entry.epoch === BenchState.viewEpoch
    ) {
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
      // Before the reset, while the armed diff button is still in this
      // card's subtree: a rerun of the armed card must not leave the armed
      // side pointing at the attempt this is about to destroy.
      BenchDiff.disarmIfArmedOn(ui);
      resetColumn(ui);
      // Same budget and the same controls as the run being retried, not the
      // current control values: a rerun is a second sample of the same
      // experiment. Replaying it under whatever the composer holds now
      // would also be refused by the server's one-experiment check, and
      // rightly so.
      BenchStream.runOne(
        retry.prompt,
        retry.model,
        retry.promptId,
        retry.groupId,
        retry.budget,
        ui,
        BenchState.viewEpoch,
        // Same column as the attempt being retried: a rerun is a second
        // sample in the same slot, not a new column.
        retry.position,
        retry.controls,
        // The same documents, for the same reason as the controls: the
        // group's declaration is fixed and a rerun that brought a
        // different set would be refused by the server's entry check.
        retry.documents,
      );
    });
    ui.tools.append(btn);
  }

  function fillMetrics(ui, result) {
    if (result.ttft_ms != null) {
      setMetric(ui.metrics.ttft, String(Math.round(result.ttft_ms)), "ms");
    }
    if (result.latency_ms != null) {
      if (result.latency_ms < 1000) {
        setMetric(
          ui.metrics.total,
          String(Math.round(result.latency_ms)),
          "ms",
        );
      } else {
        setMetric(ui.metrics.total, (result.latency_ms / 1000).toFixed(2), "s");
      }
    }
    if (result.prompt_tokens != null && result.completion_tokens != null) {
      setMetric(
        ui.metrics.tok,
        result.prompt_tokens + "/" + result.completion_tokens,
        null,
      );
    }
    if (result.reasoning_tokens != null) {
      setMetric(ui.metrics.reasoning, String(result.reasoning_tokens), null);
    }
    // Billed wins when the platform reported one: it is the charge, the
    // estimate is arithmetic over reported tokens, and showing arithmetic
    // beside an available fact is a choice to be less accurate. The
    // estimate stays reachable in the tooltip rather than being dropped,
    // because a large gap between the two is itself a signal (a provider
    // priced differently than the catalog says).
    if (result.billed_cost_usd != null) {
      setMetric(ui.metrics.cost, fmtBilled(result.billed_cost_usd), null);
      ui.metrics.cost.title =
        result.cost_usd != null
          ? "billed by OpenRouter for this request; the catalog estimate " +
            "was " +
            fmtCost(result.cost_usd)
          : "billed by OpenRouter for this request";
    } else if (result.cost_usd != null) {
      setMetric(ui.metrics.cost, fmtCost(result.cost_usd), null);
    }
    if (result.provider != null) {
      ui.caption.textContent = "served by " + result.provider;
      ui.caption.hidden = false;
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

  // Draws the reasoning indicator, or removes a previous one.
  //
  // The removal branch is not defensive padding: completeColumn runs
  // again on a rerun of the same card, and an indicator left over from
  // the previous attempt would describe tokens the current result never
  // spent. resetColumn clears the metrics for the same reason.
  function addReasoningIndicator(ui, result) {
    // Idempotent on its own, not only via resetColumn. completeColumn
    // can run twice on one card without a reset: stream.js calls finish()
    // again with a synthetic error result when the primary render throws,
    // and a second indicator stacked under the first would be the visible
    // symptom.
    const existing = ui.card.querySelector(".reasoning-warn");
    if (existing) existing.remove();
    if (
      !reasoningAteTheOutput(result.completion_tokens, result.reasoning_tokens)
    ) {
      return;
    }
    const share = reasoningShare(
      result.completion_tokens,
      result.reasoning_tokens,
    );
    const warn = document.createElement("div");
    warn.className = "reasoning-warn";
    warn.dataset.testid = "reasoning-warning";
    // textContent, like every other user-facing string on this card. The
    // numbers are the provider's and go through String coercion only.
    warn.textContent =
      share + "% of completion tokens went to reasoning, not to the answer";
    warn.title =
      "reasoning tokens are billed as completion tokens and never appear " +
      "in the answer. Lower the reasoning effort under Experiment " +
      "controls to spend less of the budget thinking";
    ui.body.before(warn);
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
      ui.body.textContent =
        result.response_text != null ? result.response_text : "";
    }
    fillMetrics(ui, result);
    // THE INDICATOR. A card whose output went mostly to thinking says so
    // on its face, whether or not it produced an answer and whether or
    // not the server was able to label anything.
    //
    // WHY IT EXISTS SEPARATELY FROM THE ERROR LABEL. The server can only
    // relabel a result it synthesized an error for, which means a result
    // with NO visible text. A card that returned 500 tokens of answer
    // after 20000 tokens of thinking is not an error by any definition,
    // carries no error to relabel, and is still a card where nearly all
    // the money went somewhere the reader cannot see. That card is the
    // one this is for.
    //
    // KEYED ON THE TWO COUNTS AND NOTHING ELSE, which is what makes it
    // work where R1's reservation could not: a provider that ignores an
    // unknown request parameter still reports its usage honestly, so the
    // numbers come back true even when the cap never applied. No model
    // name is involved, and no request field is read.
    //
    // In completeColumn rather than in fillMetrics so it covers replayed
    // history as well as live runs: a stored row carries both counts, and
    // a warning that appeared only while you watched would be missing
    // from every card anyone came back to.
    addReasoningIndicator(ui, result);
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
    // A user Stop is neither done nor a provider failure, and a spend
    // refusal is neither of those nor a failure of any kind: the ceiling
    // worked. Each gets its own state so a card never implies the model
    // finished, errored, or lost its history. The refusal's message (which
    // names both the accumulated spend and the ceiling) still renders
    // through applyError above; only the framing differs.
    setState(
      ui,
      opts.stopped
        ? "stopped"
        : opts.refused
          ? "refused"
          : error != null
            ? "error"
            : "done",
    );
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
      note.dataset.testid = "budget-note";
      // A CAP, LABELLED AS ONE, and the label is the fix rather than
      // decoration. This badge read "budget 65536" beside a reasoning
      // metric reading 21350, and a reader subtracting them got 44186
      // tokens that never existed: one number is the ceiling the bench
      // SENT, the other is what the provider COUNTED. Both are tokens,
      // so nothing about the pair announced that they answer different
      // questions, and the phantom difference misdirected a diagnosis
      // for a full round.
      //
      // "cap" is the whole remedy. The two numbers now cannot be read as
      // one arithmetic, and the title says which is which for anyone who
      // hovers.
      note.textContent = "budget cap " + result.max_tokens;
      note.title =
        "the completion ceiling this run was sent, after per-model " +
        "clamping. Not a count: compare it against tok i/o, which is " +
        "what was actually used";
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
    raceInit,
    raceRestart,
    raceTtft,
    raceError,
    raceStopped,
    raceRefused,
    raceDone,
    hideRace,
    makeColumn,
    completeColumn,
    fillColumn,
    setMetric,
    startTicker,
    stopTicker,
    toolButton,
    tickers,
  };
})();
