// The experiment report, as plain tables. Exposed on window.BenchReport.
//
// Prose first and no charts, which is a deliberate limit rather than a
// missing feature. A bar chart of four means invites the eye to read a
// difference the intervals do not support, and this whole layer exists
// to stop exactly that. Numbers with their denominators beside them are
// harder to misread than a picture.
//
// No build step and no dependencies, like every other module here.
(function () {
  const panel = document.getElementById("report-panel");

  function cell(text, className) {
    const td = document.createElement("td");
    if (className) td.className = className;
    td.textContent = text;
    return td;
  }

  function head(labels) {
    const tr = document.createElement("tr");
    for (const label of labels) {
      const th = document.createElement("th");
      th.textContent = label;
      tr.append(th);
    }
    const thead = document.createElement("thead");
    thead.append(tr);
    return thead;
  }

  function num(value, digits) {
    // An absent number renders as an absent number. Printing 0 for "we
    // do not know" is the same lie as a badge for an unset control.
    return value == null
      ? "—"
      : value.toFixed(digits === undefined ? 3 : digits);
  }

  function interval(entry) {
    if (!entry || entry.lo == null) {
      return entry?.note ? entry.note : "—";
    }
    return "[" + num(entry.lo) + ", " + num(entry.hi) + "]";
  }

  function outcomesTable(report) {
    const table = document.createElement("table");
    table.dataset.testid = "report-outcomes";
    table.append(
      head([
        "model",
        "rank",
        "planned",
        "attempted",
        "done",
        "error",
        "refused",
        "stopped",
        "missing",
        "failure rate",
      ]),
    );
    const body = document.createElement("tbody");
    for (const entry of report.models) {
      const t = entry.trials;
      const tr = document.createElement("tr");
      tr.dataset.testid = "report-row";
      // Two denominators because there are two populations: what the
      // plan called for, and what actually reached a provider. A refused
      // or never-run trial is in the first and not the second, and the
      // failure rate is over the second, so it ships with its counts for
      // the same reason the pass rate does.
      tr.append(
        cell(entry.model, "report-model"),
        cell(entry.rank == null ? "—" : String(entry.rank)),
        cell(String(t.planned)),
        cell(String(t.attempted)),
        cell(String(t.done)),
        cell(String(t.error)),
        cell(String(t.refused)),
        cell(String(t.stopped)),
        cell(String(t.missing)),
        cell(
          t.failure_rate == null
            ? "—"
            : num(t.failure_rate, 2) +
                " (" +
                t.error +
                "/" +
                t.attempted +
                " attempted)",
          "report-failure-rate",
        ),
      );
      body.append(tr);
    }
    table.append(body);
    return table;
  }

  function scorerTable(report) {
    const table = document.createElement("table");
    table.dataset.testid = "report-scores";
    table.append(
      head([
        "model",
        "scorer",
        "mean",
        "95% interval",
        "n",
        "pass rate",
        "coverage",
        "scoring failures",
        "flags",
      ]),
    );
    const body = document.createElement("tbody");
    for (const entry of report.models) {
      for (const scorer of entry.scorers) {
        const tr = document.createElement("tr");
        tr.dataset.testid = "report-score-row";
        const pass = scorer.pass_rate;
        // The rate never appears without the counts it came from. A pass
        // rate over three verdicts out of forty eligible trials is not a
        // pass rate anybody should act on, and the coverage is the only
        // thing on the page that says so.
        const passText =
          pass.rate == null
            ? "—"
            : num(pass.rate, 2) +
              " (" +
              pass.passed +
              "/" +
              pass.usable_verdicts +
              " of " +
              pass.eligible +
              " eligible)";
        const cov = scorer.coverage;
        const flags = [];
        if (scorer.self_judged > 0)
          flags.push("self-judged " + scorer.self_judged);
        if (scorer.blind > 0) flags.push("blind " + scorer.blind);
        tr.append(
          cell(entry.model, "report-model"),
          cell(scorer.scorer),
          cell(num(scorer.mean)),
          cell(interval(scorer.interval)),
          cell(String(scorer.n)),
          cell(passText, "report-pass"),
          cell(cov.scored + " scored, " + cov.unscored + " unscored"),
          cell(
            cov.scoring_failed + " (" + num(cov.scoring_failure_rate, 2) + ")",
            "report-scoring-failures",
          ),
          cell(flags.join(", ") || "—", "report-flags"),
        );
        body.append(tr);
      }
    }
    table.append(body);
    return table;
  }

  function providerTable(report) {
    const table = document.createElement("table");
    table.dataset.testid = "report-providers";
    table.append(
      head(["model", "providers that served", "cost", "latency p50/p90"]),
    );
    const body = document.createElement("tbody");
    for (const entry of report.models) {
      const names = Object.entries(entry.providers)
        .map(([name, count]) => name + " x" + count)
        .join(", ");
      const c = entry.cost;
      const tr = document.createElement("tr");
      tr.dataset.testid = "report-provider-row";
      tr.append(
        cell(entry.model, "report-model"),
        // Empty rather than a guess: under dynamic routing the provider
        // is chosen per call, and a run whose host nobody recorded has
        // no host to name.
        cell(names || "—"),
        cell(
          "$" +
            c.total_usd.toFixed(4) +
            " (" +
            c.billed_trials +
            " billed, " +
            c.estimated_trials +
            " estimated, " +
            c.unpriced_trials +
            " unpriced)",
        ),
        cell(
          num(entry.latency_ms.median, 0) +
            " / " +
            num(entry.latency_ms.p90, 0),
        ),
      );
      body.append(tr);
    }
    table.append(body);
    return table;
  }

  function banner(report) {
    const el = document.createElement("div");
    el.className = "report-banner";
    el.dataset.testid = "report-banner";
    // The estimand leads, because a number without it is a number about
    // nothing in particular. The reader has to know whether they are
    // looking at the routed service or at the model.
    const estimand = document.createElement("strong");
    estimand.dataset.testid = "report-estimand";
    estimand.textContent =
      report.estimand_mode === "underlying_model"
        ? "underlying-model estimand (strict routing)"
        : "routed-service estimand (provider chosen per call)";
    const detail = document.createElement("span");
    detail.textContent =
      report.name +
      " · " +
      report.dataset_name +
      " · " +
      report.repeats +
      (report.repeats === 1 ? " repeat" : " repeats") +
      " · " +
      report.status +
      " · 95% intervals bootstrap " +
      report.bootstrap.resamples +
      " resamples over " +
      report.bootstrap.unit +
      " clusters, seed " +
      report.bootstrap.seed;
    el.append(estimand, detail);
    if (!report.thresholds_available) {
      const note = document.createElement("span");
      note.dataset.testid = "report-threshold-note";
      note.className = "report-note";
      note.textContent =
        "no dataset file given, so pass rates are unavailable: " +
        "thresholds live in the dataset, not the database.";
      el.append(note);
    }
    return el;
  }

  async function show(experimentId, datasetPath) {
    panel.replaceChildren();
    panel.hidden = false;
    const note = document.createElement("div");
    note.dataset.testid = "report-state";
    note.textContent = "loading report";
    panel.append(note);
    let report;
    try {
      const url =
        "/experiments/" +
        experimentId +
        "/report" +
        (datasetPath ? "?dataset_path=" + encodeURIComponent(datasetPath) : "");
      const resp = await fetch(url);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      report = await resp.json();
    } catch (err) {
      // Same rule as every other load in this app: the failure is on the
      // page and on the console, never only in a variable.
      console.error("report load failed", err);
      note.textContent = "failed to load report: " + err.message;
      note.dataset.state = "error";
      return;
    }
    note.remove();
    panel.append(
      banner(report),
      outcomesTable(report),
      scorerTable(report),
      providerTable(report),
    );
    panel.dataset.state = "ready";
  }

  // ---- The list, so the report has a way in.

  const listEl = document.getElementById("experiment-list");
  const detailsEl = document.getElementById("experiments");

  // Same state discipline as the history panel, and for the same reason:
  // an empty list means two different things while a fetch is in flight,
  // so the panel names its own state rather than leaving a blank to be
  // read as "none".
  function setState(state, message) {
    listEl.dataset.state = state;
    listEl.textContent = message;
  }

  async function loadExperiments() {
    setState("loading", "loading experiments");
    let data;
    try {
      const resp = await fetch("/experiments");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch (err) {
      console.error("experiment list load failed", err);
      setState("error", "failed to load experiments: " + err.message);
      return;
    }
    if (data.experiments.length === 0) {
      setState("empty", "no experiments yet");
      return;
    }
    setState("ready", "");
    for (const experiment of data.experiments) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "hrow";
      row.dataset.testid = "experiment-row";
      const time = document.createElement("span");
      time.className = "htime";
      time.textContent =
        experiment.created_at.slice(0, 19).replace("T", " ") + " UTC";
      const name = document.createElement("span");
      name.className = "hprompt";
      name.textContent = experiment.name;
      const meta = document.createElement("span");
      meta.className = "hcount";
      meta.textContent =
        experiment.status +
        " · " +
        experiment.trials_done +
        "/" +
        experiment.trials_total +
        " trials";
      row.append(time, name, meta);
      row.addEventListener("click", () => show(experiment.id));
      listEl.append(row);
    }
  }

  function init() {
    detailsEl.addEventListener("click", (event) => {
      // The synchronous claim, exactly as the history panel makes it:
      // toggle is dispatched asynchronously, so without this the panel
      // still reads the previous load's terminal state when the click
      // lands. See static/history.js for the incident that taught it.
      if (!detailsEl.open && event.target.closest("summary")) {
        setState("loading", "loading experiments");
      }
    });
    detailsEl.addEventListener("toggle", () => {
      if (detailsEl.open) loadExperiments();
    });
  }

  window.BenchReport = { show, init };
})();
