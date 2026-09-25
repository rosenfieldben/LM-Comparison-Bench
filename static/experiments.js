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
  // What the cost cell says about upstream figures, one clause per
  // bucket. Separated from the cell so the wording is readable and so a
  // test can reach it without building a report.
  //
  // Absent entirely when no trial reported a figure at all, which is
  // the common case and keeps the cell short.
  function upstreamClause(upstream) {
    if (!upstream) return "";
    const parts = [];
    const byok = upstream.byok?.trials;
    const notByok = upstream.not_byok?.trials;
    const notByokUnpriced = upstream.not_byok_unpriced?.trials;
    const unknown = upstream.unknown?.trials;
    // The only bucket that is a second bill: is_byok said so.
    if (byok) {
      parts.push(
        byok +
          " upstream charge" +
          (byok === 1 ? "" : "s") +
          " billed direct by the provider (not in the total)",
      );
    }
    // The same money seen twice. Saying so is the whole point: a reader
    // who sees a figure and no explanation assumes a second bill.
    if (notByok) {
      parts.push(
        notByok +
          " non-BYOK trial" +
          (notByok === 1 ? "" : "s") +
          " also reported an upstream figure, already inside the total",
      );
    }
    // NON-BYOK AND UNPRICED, which the clause above must not claim. It
    // says "already inside the total", and that is true only of a trial
    // the total counted. A trial with no billed charge and no local
    // estimate contributes nothing, so on a report whose total is zero
    // the old wording told a reader a figure was inside a total of
    // nothing. What can honestly be said is the half that is still
    // true: nobody called it a second bill, and the trial has no price.
    if (notByokUnpriced) {
      parts.push(
        notByokUnpriced +
          " unpriced non-BYOK trial" +
          (notByokUnpriced === 1 ? "" : "s") +
          " reported an upstream figure, not added as a separate bill " +
          "and not priced in the total",
      );
    }
    // Written before the flag existed. Unknown is not a synonym for no.
    if (unknown) {
      parts.push(
        unknown +
          " trial" +
          (unknown === 1 ? "" : "s") +
          " reported an upstream figure with no BYOK flag recorded",
      );
    }
    return parts.length ? ", plus " + parts.join("; ") : "";
  }

  const panel = document.getElementById("report-panel");

  function cell(text, name) {
    const td = document.createElement("td");
    if (name) {
      // ONE name, used as both the style hook and the test hook. The
      // frontend stability contract selects on data-testid and volt.css
      // targets the class, and a column that carried only the class
      // forced tests to select on styling, which is the coupling the
      // contract exists to prevent. Stamped together here so a renamed
      // column cannot leave the stylesheet and the suite pointing at two
      // different things.
      td.className = name;
      td.dataset.testid = name;
    }
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
        // The LABEL, not the model. They are the same string unless the
        // lineup listed the model twice, and in that case the model name
        // shows the same row twice with different numbers and no way to
        // tell which arm is which. The report already computed the
        // distinguishing name; printing the other field threw it away.
        cell(entry.label, "report-model"),
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
        // The judge is half the series key. Without it two judges of one
        // scorer render as two identical-looking rows with different
        // numbers, which reads as a bug in the bench rather than as the
        // disagreement it is.
        "judge",
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
          cell(entry.label, "report-model"),
          cell(scorer.scorer),
          // Empty rather than a stand-in name: a deterministic scorer and
          // a human rating have no judge, and inventing one would make
          // three different things look like one.
          cell(scorer.judge_model || "—", "report-judge"),
          cell(num(scorer.mean)),
          cell(interval(scorer.interval)),
          // How much of the mean anybody actually measured. n is the
          // denominator; the rest of it is failure-inclusive zeros, and
          // a 0.00 mean over zero measurements means the arm never
          // answered rather than that it answered badly.
          cell(
            scorer.n === scorer.measured
              ? String(scorer.n)
              : scorer.n + " (" + scorer.measured + " measured)",
            "report-n",
          ),
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
        cell(entry.label, "report-model"),
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
            " unpriced)" +
            // Beside the total and never added to it. That total is what
            // OpenRouter charged in credits.
            //
            // ONE CLAUSE PER BUCKET, because presence of an upstream
            // figure never proved BYOK and this cell used to say it
            // did. Two live captures in this repository are non-BYOK
            // runs that reported one, so calling every figure a direct
            // provider bill published OpenRouter's own charge a second
            // time as an invoice nobody was owed. Only the byok bucket
            // is a bill; the others say what they actually are.
            upstreamClause(c.upstream),
          "report-cost",
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

  // Which documents each TASK read, as chips built by the same function
  // the history views use.
  //
  // PER TASK AND NOT PER EXPERIMENT, because the flat list the report
  // also carries says which documents the run touched and a reader
  // looking at one task's numbers wants the document THAT task read. On
  // a dataset where every task attaches something different the flat
  // list is useless for exactly the question its name suggests.
  //
  // NEVER A FILENAME, and the report has none to show: a name is what a
  // person picked on their own machine and is not a property of the
  // bytes. See BenchAttach.pinChip for why that absence gets its own
  // entry point rather than reusing the one that means "no longer
  // stored".
  //
  // Null when no RECORDED CELL declared a document, so the caller
  // appends nothing rather than an empty box. An experiment with no
  // documents should look like every experiment did before this phase.
  //
  // RECORDED, NOT DECLARED, and the two differ in a case a reader of
  // this panel will meet: the report is built from cells that produced
  // a result, so an experiment created and not yet started renders with
  // no documents block even though every one of its tasks cites one,
  // and a halted experiment shows only the tasks whose cells ran. That
  // is the server's deliberate rule (see recorded_groups: a report is
  // about what ran) and this comment said "declared" for it until the
  // thirteenth review's panel. What an un-run cell was OWED lives in
  // the experiment record and in the export's manifest, not here.
  function taskDocuments(report) {
    const perTask = report.task_attachments || {};
    // Sorted, and the sort is this view's own. Object key order is
    // insertion order for ordinary strings and NUMERIC order for keys
    // that look like integers, so a dataset whose task ids are "1" and
    // "10" would be reordered by the engine and not by anything the
    // report said. One stated order beats one that depends on how the
    // ids were spelled.
    const ids = Object.keys(perTask).sort();
    if (ids.length === 0) return null;
    const el = document.createElement("div");
    el.className = "report-docs";
    el.dataset.testid = "report-documents";
    const label = document.createElement("span");
    label.className = "report-note";
    label.textContent =
      "documents each task read, by digest and reading; the bench does " +
      "not record a filename here because a name is a fact about " +
      "somebody's filesystem rather than about the bytes";
    el.append(label);
    for (const id of ids) {
      const row = document.createElement("div");
      row.className = "attach-strip";
      row.dataset.testid = "report-task-documents";
      row.dataset.task = id;
      const name = document.createElement("span");
      name.className = "attach-strip-label";
      // textContent: a task id comes out of the operator's own dataset
      // file and is user text like any other.
      name.textContent = id;
      row.append(name);
      for (const entry of perTask[id]) {
        // The capture a snapshot pin names, from the report's own
        // records: which commit, which tree state, when. Keyed by the
        // id as a string, since that is how JSON carries object keys.
        const capture =
          entry.capture_id === null || entry.capture_id === undefined
            ? null
            : report.captures?.[String(entry.capture_id)] || null;
        row.append(window.BenchAttach.pinChip(entry, "task", capture));
      }
      el.append(row);
    }
    return el;
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
    // What the judging cost, on its own line beside the ranking, as the
    // payload keeps it on its own key (report.judge_cost). It is the
    // bench's instrument cost, money spent measuring, and is never added
    // into a model's cost cell, whose total is what that model was paid.
    // A call whose reply carried no price is named as unpriced rather
    // than left out, and every judge row with no billing figure is
    // counted after the spend (the unpriced calls among them), so the
    // line never reads as the whole cost of judging when it may not be.
    const spend = report.judge_cost;
    const unpriced = spend.unpriced_calls;
    const bare = spend.rows_without_figure;
    const judgeSpend = document.createElement("span");
    judgeSpend.dataset.testid = "report-judge-spend";
    judgeSpend.className = "report-note";
    judgeSpend.textContent =
      (spend.billed_calls > 0
        ? "judge spend: $" +
          spend.total_usd.toFixed(4) +
          " over " +
          spend.billed_calls +
          (spend.billed_calls === 1 ? " billed call" : " billed calls") +
          (unpriced > 0 ? ", " + unpriced + " unpriced" : "")
        : "judge spend: none billed" +
          (unpriced > 0
            ? ", " +
              unpriced +
              (unpriced === 1 ? " call unpriced" : " calls unpriced")
            : "")) +
      (bare > 0
        ? "; " +
          bare +
          (bare === 1
            ? " judge row carries no billing figure"
            : " judge rows carry no billing figure")
        : "");
    el.append(judgeSpend);
    if (report.arm_caveat) {
      // Present only when it was earned, so a reader who sees it knows
      // something specific happened rather than that the bench hedges by
      // habit. A caveat that lived only in the payload would be a caveat
      // nobody reads, which is the same as not having one.
      const arms = document.createElement("span");
      arms.dataset.testid = "report-arm-caveat";
      arms.className = "report-note";
      arms.textContent = report.arm_caveat;
      el.append(arms);
    }
    if (report.thresholds_source === "score_rows") {
      const note = document.createElement("span");
      note.dataset.testid = "report-threshold-note";
      note.className = "report-note";
      // The caveat travels with the number, not in a doc nobody opens.
      // The rate itself is sound: those verdicts were computed against
      // the real thresholds when the scoring pass ran. It is the
      // DENOMINATOR that is incomplete, and that is the part a reader
      // would otherwise assume was whole.
      //
      // TWO WAYS TO GET HERE since Phase N, and the note says which. The
      // bench holds no copy of the dataset, or it holds one this build
      // could not read, and the server's own reason for the second is
      // printed verbatim: the floor is honest either way, but a reader
      // looking at it should know an exact denominator was sitting in
      // the store.
      const why =
        typeof report.dataset_unreadable === "string"
          ? "the stored dataset could not be read (" +
            report.dataset_unreadable +
            "), "
          : "no dataset file given and the bench holds no copy of this " +
            "experiment's dataset, ";
      note.textContent =
        why +
        "so the eligible count was recovered from the score rows and is " +
        "a FLOOR: a task whose trials were never scored leaves no row to " +
        "witness its threshold. Supply the file above for the full " +
        "denominator.";
      el.append(note);
    }
    return el;
  }

  // The dataset path the operator last applied, and the experiment it was
  // applied for, held in a variable for as long as the tab is open and
  // nowhere else. Not in localStorage and not sent anywhere to be stored:
  // it is a path on their own machine, which is a fact about their
  // filesystem rather than about the experiment, and the experiment row
  // deliberately records the file's digest instead. The same reasoning
  // the prompt library follows for what it will and will not keep.
  //
  // ONE EXPERIMENT'S, NEVER ANOTHER'S. A path names the file one
  // experiment was read from; sent with another experiment's report it
  // carried one experiment's input into another's request, and the
  // server refused it as drift, so a stored experiment opened on a false
  // "dataset changed".
  let remembered = { id: null, path: "" };

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
    input.placeholder = "leave blank to read the dataset the bench stored";
    input.title =
      "Thresholds live in the dataset. Blank reads the copy the bench " +
      "stored under this experiment's digest, when it holds one; a path " +
      "reads the file from disk instead. Either way the digest is checked " +
      "against the one recorded at creation and a mismatch is refused. " +
      "Remembered for this experiment in this tab only, never stored.";
    const apply = document.createElement("button");
    apply.type = "submit";
    apply.dataset.testid = "report-dataset-apply";
    apply.textContent = "apply";
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      remembered = { id: experimentId, path: input.value.trim() };
      show(experimentId, remembered.path);
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

  // Each show() is numbered, and a report whose fetch returns after a
  // newer show() began is dropped: the experiment panel opens reports on
  // its own (after Create, when a watched run finishes) as well as on a
  // click, and a late answer appended under a newer one stacked two
  // experiments' reports in one panel.
  let showVersion = 0;

  async function show(experimentId, datasetPath) {
    const version = ++showVersion;
    // Undefined means "the path last applied for this experiment", which
    // is how reopening its report keeps their file; any other experiment
    // gets none and reads the store. An explicit empty string means they
    // cleared it, and that has to survive.
    const path =
      datasetPath !== undefined
        ? datasetPath
        : remembered.id === experimentId
          ? remembered.path
          : "";
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
      if (version !== showVersion) return;
      // Same rule as every other load in this app: the failure is on the
      // page and on the console, never only in a variable.
      console.error("report load failed", err);
      note.textContent = "failed to load report: " + err.message;
      note.dataset.state = "error";
      panel.dataset.state = "error";
      return;
    }
    if (version !== showVersion) return;
    note.remove();
    panel.append(banner(report));
    // Above the tables, because the documents changed what every model
    // READ and the tables only report how each one answered. Appended
    // only when there are any: see taskDocuments.
    const documents = taskDocuments(report);
    if (documents) panel.append(documents);
    panel.append(
      outcomesTable(report),
      scorerTable(report),
      providerTable(report),
    );
    panel.dataset.state = "ready";
  }

  // The list the report is opened from lives in static/lifecycle.js,
  // with the rest of the experiment panel: selecting a row there calls
  // show.
  window.BenchReport = { show };
})();
