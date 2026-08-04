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
        "not run",
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
        // Two absences, kept apart on the page as they are in the data.
        // A missing trial has a cell it left no row in; a not-run trial
        // has no cell, because the plan was abandoned before it. Folding
        // them into one column would hide a halt inside a gap.
        cell(String(t.missing)),
        cell(String(t.not_run)),
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
    // The ranking names its metric, or says there is none. A rank column
    // with nothing saying what it ranks ON is the same failure as a
    // number without its estimand, one level down: the reader supplies a
    // meaning the report never claimed.
    const ranking = document.createElement("span");
    ranking.dataset.testid = "report-ranking";
    ranking.className = "report-note";
    // When the ranking is human, its blind composition rides with it. A
    // report ranked on ratings made blind is a different claim from one
    // ranked on sighted ratings, and the second is much the weaker; a
    // reader should not have to join tables to learn which they hold.
    const composition =
      report.ranking.ratings === undefined
        ? ""
        : ", " +
          report.ranking.blind_ratings +
          " blind of " +
          report.ranking.ratings +
          " ratings";
    ranking.textContent =
      report.ranking.metric == null
        ? "no ranking: " + report.ranking.reason
        : "ranked on " +
          report.ranking.metric +
          (report.ranking.judge_model
            ? " by " + report.ranking.judge_model
            : "") +
          " (" +
          report.ranking.reason +
          ")" +
          composition;
    el.append(ranking);
    if (report.thresholds_source === "score_rows") {
      const note = document.createElement("span");
      note.dataset.testid = "report-threshold-note";
      note.className = "report-note";
      // The caveat travels with the number, not in a doc nobody opens.
      // The rate itself is sound: those verdicts were computed against
      // the real thresholds when the scoring pass ran. It is the
      // DENOMINATOR that is incomplete, and that is the part a reader
      // would otherwise assume was whole.
      note.textContent =
        "no dataset file given, so the eligible count was recovered " +
        "from the score rows and is a FLOOR: a task whose trials were " +
        "never scored leaves no row to witness its threshold. Supply " +
        "the file above for the full denominator.";
      el.append(note);
    }
    return el;
  }

  // The dataset path the operator last typed, held in a variable for as
  // long as the tab is open and nowhere else. Not in localStorage and
  // not sent anywhere to be stored: it is a path on their own machine,
  // which is a fact about their filesystem rather than about the
  // experiment, and the experiment row deliberately records the file's
  // digest instead. The same reasoning the prompt library follows for
  // what it will and will not keep.
  let rememberedPath = "";

  function datasetForm(experimentId, path) {
    const form = document.createElement("form");
    form.className = "report-dataset";
    form.dataset.testid = "report-dataset-form";
    const label = document.createElement("label");
    label.className = "panel-label";
    label.setAttribute("for", "report-dataset-path");
    label.textContent = "Dataset file";
    const input = document.createElement("input");
    input.type = "text";
    input.id = "report-dataset-path";
    input.dataset.testid = "report-dataset-path";
    input.value = path;
    input.placeholder = "leave blank for score means without pass rates";
    input.title =
      "Thresholds live in the dataset file, not the database, so pass " +
      "rates need the file the experiment was created from. The digest " +
      "is checked against the one recorded at creation and a mismatch " +
      "is refused. Remembered for this tab only, never stored.";
    const apply = document.createElement("button");
    apply.type = "submit";
    apply.dataset.testid = "report-dataset-apply";
    apply.textContent = "apply";
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      rememberedPath = input.value.trim();
      show(experimentId, rememberedPath);
    });
    form.append(label, input, apply);
    return form;
  }

  async function fetchReport(experimentId, datasetPath) {
    const url =
      "/experiments/" +
      experimentId +
      "/report" +
      (datasetPath ? "?dataset_path=" + encodeURIComponent(datasetPath) : "");
    const resp = await fetch(url);
    if (resp.ok) return resp.json();
    // The server's own words rather than "HTTP 422". A digest mismatch
    // is the one failure here the operator can act on, and the only
    // thing naming both the recorded digest and the file's is the detail
    // the server wrote. Collapsing it to a status code would leave them
    // guessing which of the two was wrong.
    let detail = "HTTP " + resp.status;
    try {
      const body = await resp.json();
      if (body?.detail) detail = body.detail;
    } catch (err) {
      // A body that is not JSON leaves the status code, which is still
      // more than nothing. Logged so it is never only in a variable.
      console.error("report error body was not JSON", err);
    }
    throw new Error(detail);
  }

  async function show(experimentId, datasetPath) {
    // Undefined means "whatever the operator last applied", which is how
    // opening a second experiment keeps their file. An explicit empty
    // string means they cleared it, and that has to survive.
    const path = datasetPath === undefined ? rememberedPath : datasetPath;
    panel.replaceChildren();
    panel.hidden = false;
    panel.dataset.state = "loading";
    const note = document.createElement("div");
    note.dataset.testid = "report-state";
    note.textContent = "loading report";
    // The form goes on the page before the fetch and stays there through
    // a failure. A mismatch the operator cannot correct without
    // reopening the panel would be a dead end, and the path they need to
    // fix is the one already in the box.
    panel.append(datasetForm(experimentId, path), note);
    let report;
    try {
      report = await fetchReport(experimentId, path);
    } catch (err) {
      // Same rule as every other load in this app: the failure is on the
      // page and on the console, never only in a variable.
      console.error("report load failed", err);
      note.textContent = "failed to load report: " + err.message;
      note.dataset.state = "error";
      panel.dataset.state = "error";
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
      // Progress is trials FINISHED, which is the three disjoint buckets
      // added up. Showing trials_done alone would leave an experiment
      // whose trials are failing looking stuck rather than failing, and
      // the two are the opposite of each other to act on.
      const finished =
        experiment.trials_done +
        experiment.trials_failed +
        experiment.trials_refused;
      const trouble = [];
      if (experiment.trials_failed > 0)
        trouble.push(experiment.trials_failed + " failed");
      if (experiment.trials_refused > 0)
        trouble.push(experiment.trials_refused + " refused");
      meta.textContent =
        experiment.status +
        " · " +
        finished +
        "/" +
        experiment.trials_total +
        " trials" +
        (trouble.length ? " (" + trouble.join(", ") + ")" : "");
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
