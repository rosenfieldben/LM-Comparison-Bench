// The stream client: one fetch per model, SSE frame handling, and the
// batch orchestration for a Run and a Stop. Exposed on window.BenchStream.
// runOne is the edge render.js calls back into for a rerun; startRun is
// also driven by the browser suite through page.evaluate.
(function () {
  const promptEl = document.getElementById("prompt");
  const resultsEl = document.getElementById("results");
  const runLabelEl = document.getElementById("run-label");
  const runBtn = document.getElementById("run");
  const stopBtn = document.getElementById("stop");

  async function runOne(
    prompt,
    model,
    promptId,
    groupId,
    budget,
    ui,
    epoch,
    position,
  ) {
    // Superseded before it started: spend no money for a dead view.
    if (epoch !== BenchState.viewEpoch) return;
    const current = () => epoch === BenchState.viewEpoch;
    const controller = new AbortController();
    BenchState.epochControllers.push(controller);
    // A Stop that landed during this batch's startup (before this
    // controller existed) marked the epoch: begin already aborted so the
    // run halts as stopped rather than streaming to completion once its
    // slot opens. The mark exists only while startRun's group-POST window
    // is open, and startRun's finally clears it when the batch settles, so
    // a standalone rerun issued later in this same view reaches this line
    // with no mark set and runs normally.
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
          BenchRender.setMetric(
            ui.metrics.ttft,
            String(Math.round(ttftMs)),
            "ms",
          );
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
        } else if (result.spend_refused === true) {
          // Refused before any provider saw it: the strip must not call
          // that a failure, matching the card's own refused state.
          BenchRender.raceRefused(model);
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
        // Billed first, estimate second, matching the card and the server's
        // ceiling: one contribution per run, never both, so the bar cannot
        // double-count a result that carries each.
        const billed = result.billed_cost_usd;
        const charge = billed != null ? billed : result.cost_usd;
        if (charge != null) {
          BenchState.sessionStats.spend += charge;
          if (billed == null) BenchState.sessionStats.estimated += 1;
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
      // The spend ceiling refusing a run is a working control, not a
      // failure. It arrives as run_id null like a persistence failure does,
      // so the marker the server sets is what tells them apart; keying on
      // the error text would break the moment that prose is reworded.
      const refused = result.spend_refused === true;
      BenchRender.completeColumn(ui, result, model + " (live)", {
        streamed: textNode !== null,
        shownError: shownError,
        budgetBadge: false,
        // run_id null on the done event means the server spent the money
        // and streamed the response but could not persist it. A refusal is
        // run_id null too, but deliberately: it persists nothing because
        // nothing happened, so it must not claim history was lost.
        unsaved: runId === null && !refused,
        refused: refused,
        // Only this streaming path offers a rerun; historical replays go
        // through fillColumn and never get one. A stopped run has no error,
        // so it gets no rerun control, and neither does a refusal: the
        // ceiling holds for the life of the process, so a rerun could only
        // ever be refused again, and offering it would turn the honest
        // refused card into a red error card on the first click.
        retry:
          result.error != null && !refused
            ? { prompt, model, promptId, groupId, budget, position }
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
          // Which column this run occupies, so a replay can rebuild the
          // layout from the rows instead of from the current chip order,
          // which drifts as the lineup is edited. Omitted (undefined
          // drops out of JSON) for a rerun, which reuses the position the
          // retried run already recorded.
          position: position,
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
        // biome-ignore lint/suspicious/noAssignInExpressions: the SSE frame split reads and advances the buffer in one loop condition
        while ((sep = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          const line = frame.split("\n").find((l) => l.startsWith("data:"));
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
      finish(
        {
          error: stopped ? null : "request failed: " + err.message,
          response_text: textNode !== null ? textNode.data : null,
          stopped: stopped,
        },
        undefined,
      );
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
    // Open the batch-startup window. While it is open this epoch has runs
    // that do not exist yet and so cannot be aborted directly; that is the
    // only situation the stop mark is for, and stopRuns consults this to
    // decide whether to set one. The group POST's own finally closes it.
    BenchState.pendingBatchEpoch = epoch;
    try {
      // A JSON body is load-bearing: the server requires
      // application/json on every POST so hostile cross-site senders
      // are forced into a CORS preflight it never answers. It now also
      // declares the experiment, the prompt and the ordered lineup,
      // before any model is called, which is what lets the server fix
      // the group's prompt ahead of its first member.
      const resp = await fetch("/groups", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt, models: models }),
        signal: groupController.signal,
      });
      if (resp.ok) groupId = (await resp.json()).id;
    } catch (err) {
      // Deliberately swallowed, a Stop abort included; see above.
    } finally {
      // The window closes here, not when the batch settles. Past this
      // point every run this batch launches registers its controller
      // synchronously, so a Stop reaches all of them by direct abort and
      // needs no mark. Leaving the window open for the whole batch would
      // let a mid-batch Stop arm stoppedEpoch, and a rerun issued in the
      // moments before the batch settled would then begin already aborted:
      // the same leak F4.3 closed, just narrower. Any mark this window did
      // produce survives until the finally below, because the runOne calls
      // that must consume it have not started yet.
      if (BenchState.pendingBatchEpoch === epoch) {
        BenchState.pendingBatchEpoch = -1;
      }
    }
    try {
      await Promise.allSettled(
        models.map((model, i) =>
          runOne(
            prompt,
            model,
            promptId,
            groupId,
            budget,
            columns[i],
            epoch,
            i,
          ),
        ),
      );
    } finally {
      // Drop any mark the startup window produced. Every run this batch
      // will ever launch has now passed its startup check, so the mark has
      // done its job; leaving it set was the defect, because a later
      // standalone rerun reuses this epoch (Stop does not advance the view)
      // and would begin already aborted. Cleared regardless of supersession:
      // a superseded epoch matches no future run anyway, but a stale mark
      // would strand a rerun in this view. The window itself closed with the
      // group POST; this is a belt for the paths that never opened it.
      if (BenchState.pendingBatchEpoch === epoch) {
        BenchState.pendingBatchEpoch = -1;
      }
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

  // Stop aborts every in-flight controller in the current epoch WITHOUT
  // taking a new view epoch: the comparison stays the view, its cards stay,
  // nothing is cleared. Each aborted runOne lands in its catch with the
  // epoch still current and renders a stopped card; as they settle the
  // in-flight count drains to zero, re-enabling Run and disabling Stop, so
  // a later Run or rerun works through the untouched epoch machinery. The
  // abort disconnects each stream, so the server persists a started run
  // through its existing disconnect path and a queued run not at all,
  // exactly as the cards show.
  function stopRuns() {
    // The mark is only for runs that have no controller yet because
    // startRun is still awaiting the group POST. Setting it unconditionally
    // was the defect: with no batch pending (a Stop of a standalone rerun),
    // nothing clears it, so it outlived its stop and every later rerun in
    // the view began already aborted and rendered stopped. Aborting the
    // registered controllers is what stops everything that has started, and
    // runOne registers synchronously, so a Stop with no pending batch needs
    // no mark at all.
    if (BenchState.pendingBatchEpoch === BenchState.viewEpoch) {
      BenchState.stoppedEpoch = BenchState.viewEpoch;
    }
    for (const c of BenchState.epochControllers) c.abort();
  }

  function init() {
    runBtn.addEventListener("click", startRun);
    stopBtn.addEventListener("click", stopRuns);
  }

  window.BenchStream = { runOne, startRun, init };
})();
