// View epoch and session state, the spine the stream client and history
// share. Exposed on window.BenchState so every other module reaches this
// state through one explicit handle: a load-order mistake fails loudly
// here rather than silently at click time. Mutable fields live as
// properties (not closed-over locals) so a write in one module is a read
// in another.
(function () {
  const statRuns = document.querySelector("#stat-runs .v");
  const statSpend = document.querySelector("#stat-spend .v");
  const statTtft = document.querySelector("#stat-ttft .v");
  const statLineup = document.querySelector("#stat-lineup .v");

  // ---- View epoch. The results area is owned by exactly one operation
  // ---- at a time. The reproduced races this exists to prevent: a rerun
  // ---- still streaming while a new Run reused its model name repainted
  // ---- the new run's race row; a history replay opened mid-run was
  // ---- repainted by the superseded run's late events; two rapid
  // ---- history selections rendered in arrival order, letting the
  // ---- first overwrite the second. Async work stamps the epoch it
  // ---- started under and touches shared view state only while that
  // ---- epoch is current; superseded work is aborted client-side and
  // ---- dropped silently. Its server-side persistence already happened
  // ---- through the disconnect path, so this is purely view integrity.
  const state = {
    viewEpoch: 0,
    // In-flight fetches owned by the current epoch. Aborting them on
    // supersession frees the server's semaphore slot immediately and the
    // partial persists server-side.
    epochControllers: [],
    // Runs and reruns in flight for the current epoch; the Run button
    // stays disabled while any of them is live.
    inflightRuns: 0,
    // The epoch a Stop targeted, and the epoch whose batch startup is
    // still pending. These exist only for the group-POST window: runOne
    // registers its controller synchronously, so any run that has started
    // is stoppable by direct abort, and the ONLY runs a Stop cannot reach
    // are the ones startRun has not launched yet because it is awaiting
    // the group POST. pendingBatchEpoch marks that window open (set by
    // startRun before the POST, cleared in its finally), and stopRuns sets
    // stoppedEpoch only while it is open, so the mark cannot outlive the
    // window it belongs to. Both -1 means no window and no mark.
    stoppedEpoch: -1,
    pendingBatchEpoch: -1,
    // Sent with the run so it links back to its saved prompt. Cleared
    // the moment the textarea is edited: the text no longer matches the
    // library entry, so the link would lie.
    selectedPromptId: null,
    // ---- Command-bar session stats. Runs, spend and mean TTFT are this
    // ---- browser session's live totals, reset by a reload on purpose:
    // ---- the bar answers "what has this sitting cost me", not history.
    sessionStats: { runs: 0, spend: 0, unpriced: 0, ttftSum: 0, ttftN: 0 },
    newViewEpoch,
    renderStats,
  };

  function newViewEpoch() {
    state.viewEpoch += 1;
    for (const c of state.epochControllers) c.abort();
    state.epochControllers = [];
    state.inflightRuns = 0;
    return state.viewEpoch;
  }

  function renderStats() {
    statRuns.textContent = String(state.sessionStats.runs).padStart(2, "0");
    // Every figure here is an estimate from catalog prices, and results
    // the session could not price are counted rather than silently
    // dropped: a total that quietly understates spend is worse than none.
    // fmtCost, the same significant-digit formatter the cards use, rather
    // than toFixed(4): a real total below $0.00005 floored to "~$0.0000",
    // so a session that had spent money reported zero. Only an untouched
    // session shows the fixed zero, which keeps the idle bar readable and
    // is honest because nothing has been spent yet.
    const spend = state.sessionStats.spend;
    statSpend.textContent =
      (spend === 0 ? "~$0.0000" : window.BenchLib.fmtCost(spend)) +
      (state.sessionStats.unpriced > 0
        ? " + " + state.sessionStats.unpriced + " unpriced"
        : "");
    statTtft.textContent =
      state.sessionStats.ttftN > 0
        ? Math.round(state.sessionStats.ttftSum / state.sessionStats.ttftN) +
          " ms"
        : "—";
    statLineup.textContent =
      window.BenchControls.lineup.length +
      (window.BenchControls.lineup.length === 1 ? " model" : " models");
  }

  window.BenchState = state;
})();
