// The stream client: one fetch per model, SSE frame handling, and the
// batch orchestration for a Run and a Stop. Exposed on window.BenchStream.
// runOne is the edge render.js calls back into for a rerun; startRun is
// also driven by the browser suite through page.evaluate.
(function () {
  // Runs one block of ancillary work and never lets it escape. Used for
  // everything finish() does AFTER the card's terminal render: the render
  // is the point of the function, and no amount of bookkeeping failure
  // may undo it or stop the rest of the bookkeeping from happening.
  //
  // Swallowing was the crime, so logging is mandatory. A field failure ran
  // for a whole session with a perfectly silent console, which is most of
  // why it was hard to see.
  //
  // The message names the block and stops there. Saying "after the
  // terminal render" would read well for a live card and be false for a
  // superseded one, which reaches these blocks having rendered nothing.
  function guarded(what, fn) {
    try {
      fn();
    } catch (err) {
      console.error("bench: " + what + " failed", err);
    }
  }

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
    // Whether the server said this run acquired an upstream slot. A stop
    // before that point verifiably spent nothing; a stop after it may or
    // may not have, which is the distinction the unpriced count now makes.
    let sawStarted = false;
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
    // The invariant this function holds: no exception anywhere in it, or
    // in the catch that calls it, may leave a card non-terminal or vanish
    // without a console.error.
    //
    // The order below is the fix. Rendering the card's terminal state is
    // the point of finish(), so it happens first and nothing may preempt
    // it. It used to happen last, after the session-stats render, and a
    // TypeError in that render escaped before any card had been rendered.
    // finished was already true by then, so the catch's recovery call to
    // finish() returned immediately: every run finishing after the session
    // went non-idle sat at "thinking" forever with its complete answer
    // visible, the session bar half-mutated, and the console silent. A
    // real field failure, diagnosed externally and verified by
    // reproduction.
    function finish(result, runId) {
      // Prevents a SECOND terminal render, which is all this guard ever
      // claimed to do. It cannot prevent the first, because the flag is
      // set below only once a render has actually happened.
      if (finished) return;

      // A superseded run renders nothing: its view is gone, and its
      // accounting further down is the whole job it has left.
      if (current()) {
        // Logged and rethrown rather than handled here. Rethrowing is what
        // reaches the recovery: runOne's catch calls finish() again with a
        // synthetic error result, and the flag below is still false, so
        // that re-entry renders an error card. Logging is what keeps the
        // invariant true, because a broken primary render is a fact a
        // developer needs even when the card ends up saying something
        // useful. The outer catch does not log, it puts the message on the
        // card, so this is the only place it would be recorded.
        try {
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
          // failure. It arrives as run_id null like a persistence failure
          // does, so the marker the server sets is what tells them apart;
          // keying on the error text would break the moment that prose is
          // reworded.
          const refused = result.spend_refused === true;
          BenchRender.completeColumn(ui, result, model + " (live)", {
            streamed: textNode !== null,
            shownError: shownError,
            budgetBadge: false,
            // run_id null on the done event means the server spent the money
            // and streamed the response but could not persist it. A refusal
            // is run_id null too, but deliberately: it persists nothing
            // because nothing happened, so it must not claim history was
            // lost.
            unsaved: runId === null && !refused,
            refused: refused,
            // Only this streaming path offers a rerun; historical replays go
            // through fillColumn and never get one. A stopped run has no
            // error, so it gets no rerun control, and neither does a
            // refusal: the ceiling holds for the life of the process, so a
            // rerun could only ever be refused again, and offering it would
            // turn the honest refused card into a red error card on the
            // first click.
            retry:
              result.error != null && !refused
                ? { prompt, model, promptId, groupId, budget, position }
                : null,
            // A user Stop renders as an honest stopped state, not done or
            // error.
            stopped: result.stopped === true,
          });
        } catch (renderErr) {
          console.error(
            "bench: the terminal render failed for " + model,
            renderErr,
          );
          throw renderErr;
        }
      }

      // Set only after the render returned, which is what turns the guard
      // above into a guard against double rendering rather than against
      // recovery. If completeColumn itself throws, the exception escapes
      // to runOne's catch, which calls finish() again with a synthetic
      // error result; the flag is still false, so that re-entry renders
      // the error card instead of no-oping.
      //
      // The accounting rule survives that re-entry, and this is the case
      // to reason about: when the first render throws, nothing below this
      // line has run, so no counter has moved. The re-entry then runs the
      // whole tail exactly once, on the synthetic result, which carries no
      // charge and so contributes uncertainty rather than an amount. One
      // contribution either way. Losing the amount is the honest outcome
      // when the client could not even render what it received.
      //
      // Residual, stated rather than papered over: if completeColumn ever
      // throws PART WAY through, after it has already set the card's
      // state, the re-entry renders over a partly rendered card and can
      // duplicate a tool button. That is visible and recoverable, unlike
      // the silent permanent strand this ordering exists to prevent, and
      // the known failure (a missing formatter) throws in fillMetrics,
      // before anything is appended.
      finished = true;

      // Idempotent, and the render already did it for a live card; this
      // covers the superseded path, which renders nothing.
      guarded("ticker teardown", () => {
        BenchRender.stopTicker(ui);
      });

      if (current()) {
        guarded("race strip update", () => {
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
        });
        guarded("in-flight bookkeeping", () => {
          BenchState.inflightRuns -= 1;
          BenchControls.updateRunState();
        });
      }

      // Session accounting is view-independent: money spent by a
      // superseded run is still money spent this session.
      //
      // One rule for every way a run can end. A run contributes an AMOUNT
      // when a charge arrived, and contributes UNCERTAINTY when the server
      // would have spent but no charge did. sawStarted is that second
      // condition, and it is deliberately the server's own: the disconnect
      // path persists a run exactly when it had emitted started, so the
      // bar reports billing-unknown on exactly the rows history keeps.
      // Anything short of started verifiably cost nothing (a cancel while
      // still queued, a spend refusal, a failure before the slot) and stays
      // out of both counters.
      //
      // Mutated in its own block, before the stats render, so a render
      // failure leaves the counters consistent rather than half applied.
      guarded("session accounting", () => {
        const billed = result.billed_cost_usd;
        // Billed first, estimate second, matching the card and the
        // server's ceiling: one contribution per run, never both, so the
        // bar cannot double-count a result that carries each. A stopped or
        // superseded run carries neither, because no cost frame ever
        // reached the client.
        const charge = billed != null ? billed : result.cost_usd;
        if (charge != null) {
          BenchState.sessionStats.spend += charge;
          BenchState.sessionStats.priced += 1;
          if (billed == null) BenchState.sessionStats.estimated += 1;
        } else if (
          sawStarted ||
          result.response_text != null ||
          result.prompt_tokens != null ||
          result.completion_tokens != null
        ) {
          // The evidence-of-consumption checks stay as a belt: they cover
          // a done frame that somehow arrived without a started frame
          // being observed, which the current server never sends but which
          // no longer has to be assumed.
          BenchState.sessionStats.unpriced += 1;
        }
        // A stopped run reports no ttft (its synthetic result carries
        // none), so the null check is the whole guard here.
        if (result.error == null && result.ttft_ms != null) {
          BenchState.sessionStats.ttftSum += result.ttft_ms;
          BenchState.sessionStats.ttftN += 1;
        }
      });

      // The step that threw in the field. Guarded now, so a session bar
      // that cannot render costs the session bar and nothing else.
      guarded("session stats render", () => {
        BenchState.renderStats();
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
          // which drifts as the lineup is edited. A rerun re-sends the
          // retried attempt's position (render.js passes it back) so the
          // second sample lands in the same column as the first.
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
            // post-acquire clock, and restore the thinking label. This is
            // also the line past which a stop can no longer claim the run
            // was free.
            sawStarted = true;
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
      // This call is the recovery path, including for an exception thrown
      // by finish's own terminal render, so it gets the last-resort guard:
      // if even the synthetic error card cannot be drawn there is nothing
      // further to try, but the failure must not leave runOne silently. No
      // exception anywhere in finish or in this catch may leave a card
      // non-terminal or vanish without a console.error.
      try {
        finish(
          {
            error: stopped ? null : "request failed: " + err.message,
            response_text: textNode !== null ? textNode.data : null,
            stopped: stopped,
          },
          undefined,
        );
      } catch (recoveryErr) {
        console.error(
          "bench: could not render the failure state for " + model,
          recoveryErr,
        );
      }
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
