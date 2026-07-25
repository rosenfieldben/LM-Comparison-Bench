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
  const policyBadge = document.getElementById("policy-badge");

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
    // estimated counts the contributions that came from catalog arithmetic
    // rather than from a billed figure. The total is a mix whenever it is
    // nonzero, and one estimated contribution is enough to make the whole
    // sum an estimate, which is what the tilde has to track.
    //
    // priced counts contributions of either kind. It exists because the
    // amount cannot stand in for "has anything been priced yet": free
    // models are real, and a session of them is priced at exactly zero.
    // Keying the idle display on spend === 0 made such a session claim
    // "nothing priced yet" and wear the estimate tilde over a figure the
    // platform had actually confirmed.
    sessionStats: {
      runs: 0,
      spend: 0,
      unpriced: 0,
      estimated: 0,
      priced: 0,
      ttftSum: 0,
      ttftN: 0,
    },
    newViewEpoch,
    renderStats,
    setDataPolicy,
  };

  // What each mode asks OpenRouter for, in the operator's terms. The
  // wording deliberately says "asks": the routing constraint is
  // OpenRouter's to honor, and this application cannot verify it.
  const POLICY_LABELS = {
    deny: {
      text: "no-training routing",
      title:
        "BENCH_DATA_POLICY=deny: every request asks OpenRouter to route " +
        "only to providers that do not collect user data. The guarantee " +
        "is OpenRouter's, not this application's",
    },
    zdr: {
      text: "zero-retention routing",
      title:
        "BENCH_DATA_POLICY=zdr: every request asks OpenRouter to route " +
        "only to zero-data-retention endpoints, and to providers that do " +
        "not collect user data. The guarantee is OpenRouter's, not this " +
        "application's",
    },
  };

  function setDataPolicy(policy) {
    // Absence of the badge means the default. Anything unrecognized is
    // treated as standard rather than rendered raw: a badge is a claim
    // about where prompts go, and a claim assembled from an unknown
    // server value is one this page cannot stand behind.
    const label = POLICY_LABELS[policy];
    if (!label) {
      policyBadge.hidden = true;
      policyBadge.textContent = "";
      policyBadge.removeAttribute("title");
      return;
    }
    policyBadge.textContent = label.text;
    policyBadge.title = label.title;
    policyBadge.hidden = false;
  }

  function newViewEpoch() {
    state.viewEpoch += 1;
    for (const c of state.epochControllers) c.abort();
    state.epochControllers = [];
    state.inflightRuns = 0;
    return state.viewEpoch;
  }

  function renderStats() {
    statRuns.textContent = String(state.sessionStats.runs).padStart(2, "0");
    // Results the session could not price are counted rather than silently
    // dropped: a total that quietly understates spend is worse than none.
    // The significant-digit formatters the cards use, rather than
    // toFixed(4): a real total below $0.00005 floored to "$0.0000", so a
    // session that had spent money reported zero. The tilde survives only
    // while some contribution was a catalog estimate; once every priced
    // run in the session came back with a billed figure, the total is not
    // an estimate and must not wear the estimate marker. Only an untouched
    // session shows the fixed zero, which keeps the idle bar readable and
    // is honest because nothing has been spent yet.
    const spend = state.sessionStats.spend;
    // Idle means nothing has been priced AND nothing has accumulated.
    // Either condition alone is enough to leave it: a nonzero total is
    // self-evidently not idle, and a zero total after a priced run is a
    // confirmed zero rather than an absence of information.
    const idle = state.sessionStats.priced === 0 && spend === 0;
    const fmt =
      state.sessionStats.estimated > 0
        ? window.BenchLib.fmtCost
        : window.BenchLib.fmtBilled;
    statSpend.textContent =
      (idle ? "~$0.0000" : fmt(spend)) +
      (state.sessionStats.unpriced > 0
        ? " + " + state.sessionStats.unpriced + " unpriced"
        : "");
    // The tooltip is rewritten on every render rather than fixed in the
    // markup, because what the total is made of changes as the session
    // runs: a fixed string calling it an estimate became false the moment
    // a billed figure landed in it.
    const composition =
      idle
        ? "nothing priced yet this session"
        : state.sessionStats.estimated > 0
          ? "part billed by OpenRouter, part estimated from catalog " +
            "prices and reported tokens; the tilde marks that some of it " +
            "is an estimate"
          : "billed by OpenRouter, not estimated";
    statSpend.title =
      state.sessionStats.unpriced > 0
        ? composition +
          ". Unpriced counts runs with neither a billed nor an estimated " +
          "cost, and runs stopped after they started, whose billing " +
          "depends on whether the provider supports cancellation"
        : composition;
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
