// The experiment panel: the list of experiments, the form that creates
// one, and the lifecycle of the one selected (Start, the progress it
// reports, Stop, Score). Exposed on window.BenchLifecycle.
//
// MONEY MOVES ON START, AND ON SCORE WHEN A JUDGE GRADES. Create is free
// and its button says so; the projection the server returns is shown
// beside Start, which is the confirmation, the way the composer's Run has
// none either. Score's label says which kind of pass it is: a judge's
// calls are billed against the same ceiling as trials, and a pass of
// deterministic scorers calls no model. The commission's "money moves on
// Start and on nothing else" predates a Score button and is restated
// here, not kept.
//
// THE BROWSER COMPOSES; THE SERVER VALIDATES. The form's body is
// BenchLib.experimentBody over the composer's own lineup, budget and
// controls, read at the moment Create is pressed, so there is one lineup
// surface and one controls surface on the page and nothing that can
// disagree with them. Every refusal printed here is the server's
// sentence, through BenchLib.refusalText.
//
// AN ANSWER BELONGS TO THE EXPERIMENT IT WAS ASKED ABOUT, not to whichever
// row is selected when it lands. Start's, Stop's and Score's answers are
// kept by experiment and shown with it, a Start's 202 moves that
// experiment to running whatever is selected, a refusal or a lost answer
// from Start or Stop reloads the list, so the panel follows what the
// server holds, and Score opens a report only for the experiment still
// selected.
//
// NOTHING IS PRE-FILLED. The name box, repeats and the task order seed
// start empty, and a blank box is not sent; the selects and the checkbox
// start on the server's defaults (autocomplete="off" keeps Back from
// refilling any of them), and a stored dataset is used only once the
// person selects one in the Datasets panel.
(function () {
  const {
    shortDigest,
    refusalText,
    controlBadges,
    experimentBody,
    experimentNudge,
    experimentFinished,
    projectionText,
    experimentRowMeta,
    EXPERIMENT_LIMITS,
    datasetHeld,
    judgeTasks,
    scoreBody,
    scoreNudge,
    scoreLabel,
  } = window.BenchLib;

  const detailsEl = document.getElementById("experiments");
  const listEl = document.getElementById("experiment-list");
  const listNoteEl = document.getElementById("experiment-list-note");
  const nameEl = document.getElementById("experiment-name");
  const sourceDatasetEl = document.getElementById("experiment-source-dataset");
  const sourceLineupEl = document.getElementById("experiment-source-lineup");
  const sourceBudgetEl = document.getElementById("experiment-source-budget");
  const sourceControlsEl = document.getElementById(
    "experiment-source-controls",
  );
  const repeatsEl = document.getElementById("experiment-repeats");
  const seedEl = document.getElementById("experiment-seed");
  const estimandEl = document.getElementById("experiment-estimand");
  const estimandNoteEl = document.getElementById("experiment-estimand-note");
  const attachmentsEl = document.getElementById("experiment-attachments");
  const attachmentsNoteEl = document.getElementById(
    "experiment-attachments-note",
  );
  const metricEl = document.getElementById("experiment-metric");
  const haltEl = document.getElementById("experiment-halt");
  const createEl = document.getElementById("experiment-create-button");
  const nudgeEl = document.getElementById("experiment-create-nudge");
  const createMsgEl = document.getElementById("experiment-create-msg");
  const selectedEl = document.getElementById("experiment-selected");
  const titleEl = document.getElementById("experiment-title");
  const statusEl = document.getElementById("experiment-status");
  const countersEl = document.getElementById("experiment-counters");
  const streamEl = document.getElementById("experiment-stream");
  const datasetEl = document.getElementById("experiment-dataset");
  const startEl = document.getElementById("experiment-start");
  const projectionEl = document.getElementById("experiment-projection");
  const startNoteEl = document.getElementById("experiment-start-note");
  const stopEl = document.getElementById("experiment-stop");
  const actionMsgEl = document.getElementById("experiment-action-msg");
  const scoreRowEl = document.getElementById("experiment-score-row");
  const judgeFieldEl = document.getElementById("experiment-judge-field");
  const judgeEl = document.getElementById("experiment-judge");
  const scoreEl = document.getElementById("experiment-score");
  const scoreNudgeEl = document.getElementById("experiment-score-nudge");
  const retryEl = document.getElementById("experiment-score-retry");
  const judgeNoteEl = document.getElementById("experiment-judge-note");
  const scoreNoteEl = document.getElementById("experiment-score-note");

  // The experiments as the list last loaded them, newest first, and the
  // one selected, by id.
  let experiments = [];
  let selectedId = null;
  let listController = null;
  let listVersion = 0;
  // The projection POST /experiments returned, for experiments created in
  // this tab. The server computes it at creation and stores it nowhere,
  // so an experiment created elsewhere, or before a reload, has none to
  // show, and the panel says so rather than inventing one.
  const projections = new Map();
  // Experiments this tab has started. Start stays greyed for these,
  // because an experiment runs once and a second press would only be
  // refused.
  const started = new Set();
  // Start's, Stop's and Score's last word about each experiment, {text,
  // state}, shown with that experiment whichever is selected when it
  // lands.
  const actionMessages = new Map();
  // What the store says about each recorded dataset digest, in the shape
  // BenchLib.datasetHeld reads: its summary ({scorers, cites_documents})
  // when held, false when the store answered 404, the door's sentence when
  // it refused the stored bytes, null while the question is out, or
  // absent while unknown. Start and Score send the digest, and
  // the door refuses one the store does not hold (an experiment created
  // from a file by path), so the page offers neither; Score also needs
  // the scorers, to know whether a judge is needed and what it spends.
  // Only a held answer is final: the store never forgets a dataset, but
  // it can learn one (the person stores that very file), so a no is
  // asked again each time the experiment is selected.
  const datasetSummaries = new Map();
  let creating = false;
  let starting = false;
  let stopping = false;
  // Experiments whose Score request is out: one press, one POST, and
  // another experiment's Score is not greyed by it.
  const scoring = new Set();
  // The catalog state and ids the judge select was last built from, so it
  // is rebuilt when those change and not on every composer keystroke.
  let judgeOptionsKey = null;
  // The one progress stream open, or null: {id, source}.
  let watch = null;
  // Counts every selection, so a late answer can tell whether the person
  // chose a row after its request went out, even the same row again.
  let selections = 0;

  // A live region's text, assigned only when it changes, for the reason
  // the dataset builder gives: the same string assigned again replaces
  // the text node and is announced again.
  function setText(el, text) {
    if (el.textContent !== text) el.textContent = text;
  }

  // A message line: its text and whether it reads as a refusal.
  function said(el, text, state) {
    el.textContent = text;
    el.dataset.state = state;
  }

  // Start's or Stop's word about one experiment, shown if it is the one
  // selected and kept for when it is selected again.
  function saidAbout(id, text, state) {
    actionMessages.set(id, { text: text, state: state });
    if (id === selectedId) renderActionMessage();
  }

  function renderActionMessage() {
    // Stop's present-tense word is done with once the experiment shows
    // finished, however the panel learnt it (the stream, or a reload
    // after the selection moved away and the stream was closed).
    const shown = selectedExperiment();
    const held = actionMessages.get(selectedId);
    if (
      held &&
      held.state === "stopping" &&
      shown !== null &&
      experimentFinished(shown.status)
    ) {
      actionMessages.delete(selectedId);
    }
    const message = actionMessages.get(selectedId) || { text: "", state: "" };
    if (
      actionMsgEl.textContent !== message.text ||
      actionMsgEl.dataset.state !== message.state
    ) {
      said(actionMsgEl, message.text, message.state);
    }
  }

  function selectedExperiment() {
    return experiments.find((e) => e.id === selectedId) || null;
  }

  function experimentById(id) {
    return experiments.find((e) => e.id === id) || null;
  }

  // A body's JSON, or null when there is none to read.
  async function jsonOf(resp) {
    try {
      return await resp.json();
    } catch (err) {
      return null;
    }
  }

  // The refusal sentence a failed answer carries, or the status when it
  // carries none this page can read.
  function refusalOf(resp, data, what) {
    return data
      ? refusalText(
          data.detail,
          "the " + what + " was refused and the reason could not be read",
        )
      : "the " + what + " was refused (HTTP " + resp.status + ")";
  }

  // Focus back where a keyboard user left it, when a request greyed or
  // hid the control that had it: the row of the experiment it was about,
  // which is neither Start, Stop nor Score, so a second Enter moves no
  // money, stops nothing and does not pay a judge again. Only when the
  // selection has not moved since the press: an answer about a row the
  // person has left must not pull them back to it.
  function focusRow(id) {
    if (document.activeElement !== document.body) return;
    const row = listEl.querySelector(
      "[data-testid=experiment-row][data-id='" + id + "']",
    );
    if (row) row.focus();
  }

  // ---- The form.

  // What the form holds, read at the moment it is asked: the composer's
  // checked lineup, its budget and controls, and the stored dataset
  // selected in the Datasets panel. Read, never cached, because each can
  // change without this panel being told in time.
  function formState() {
    const dataset = window.BenchDatasets.selected();
    const invalid = window.BenchControls.invalidControls().slice();
    if (!repeatsEl.checkValidity()) invalid.push("repeats");
    if (!seedEl.checkValidity()) invalid.push("task order seed");
    return {
      name: nameEl.value,
      dataset: dataset,
      digest: dataset ? dataset.digest : null,
      lineup: window.BenchControls.checkedModels(),
      budget: window.BenchControls.budgetValue,
      params: window.BenchControls.experimentParams(),
      repeats: repeatsEl.value,
      seed: seedEl.value,
      estimand: estimandEl.value,
      attachments: attachmentsEl.disabled ? null : attachmentsEl.value,
      metric: metricEl.value,
      halt: haltEl.checked,
      invalid: invalid,
    };
  }

  // What the experiment will be created from, said before Create, so a
  // person sees the lineup and controls they are about to send rather
  // than having to scroll up to the composer to check.
  function renderSources(form) {
    setText(
      sourceDatasetEl,
      form.dataset
        ? "dataset: " +
            form.dataset.name +
            " · sha256 " +
            shortDigest(form.dataset.digest) +
            " · " +
            form.dataset.task_count +
            (form.dataset.task_count === 1 ? " task" : " tasks")
        : "dataset: none selected",
    );
    setText(
      sourceLineupEl,
      form.lineup.length > 0
        ? "lineup: " + form.lineup.join(", ")
        : "lineup: no model checked",
    );
    setText(sourceBudgetEl, "budget: " + form.budget);
    const badges = controlBadges(form.params);
    setText(
      sourceControlsEl,
      "controls: " +
        (badges.length > 0
          ? badges.map((b) => b.text).join(" · ")
          : "none set, so each provider applies its own defaults"),
    );
  }

  // The parts of the form that depend on the selected dataset: the
  // scorer kinds a primary metric can name, and whether there is an
  // attachments mode to choose at all.
  function renderDatasetDependent() {
    const dataset = window.BenchDatasets.selected();
    // The kinds the dataset uses, and blank for none declared. The server
    // also accepts "human", a scorer no file declares; the select offers
    // what this dataset produces and nothing else.
    const kinds = dataset ? dataset.scorers : [];
    const was = metricEl.value;
    metricEl.replaceChildren(new Option("none declared", ""));
    for (const kind of kinds) metricEl.append(new Option(kind, kind));
    metricEl.value = kinds.includes(was) ? was : "";
    // A mode is a statement about how documents reach the models. On a
    // dataset that cites none the server refuses native, and inline is
    // its default, so there is nothing to choose: the control is
    // disabled, with the reason, and no mode is sent.
    const cites = Boolean(dataset?.cites_documents);
    attachmentsEl.disabled = !cites;
    setText(
      attachmentsNoteEl,
      dataset === null
        ? "select a stored dataset to choose how its documents reach the models"
        : cites
          ? ""
          : "this dataset cites no document, so there is no mode to choose",
    );
  }

  function renderEstimandNote() {
    setText(
      estimandNoteEl,
      estimandEl.value === "underlying_model"
        ? "strict routing, with no provider pins or quantizations: those " +
            "are set through the API, not from this page"
        : "",
    );
  }

  // Everything that depends on the form's values, run on every change.
  function refreshForm() {
    const form = formState();
    renderSources(form);
    const reason = experimentNudge(form);
    createEl.disabled = creating || reason !== null;
    setText(nudgeEl, creating ? "" : reason ? "Create waits: " + reason : "");
  }

  async function create() {
    const form = formState();
    if (creating || experimentNudge(form) !== null) return;
    const body = experimentBody(form);
    // The selection when Create was pressed. A person who selects another
    // row while the request is out has made a later choice, and the new
    // experiment does not take it away from them.
    const selectedAtPress = selections;
    const hadFocus = document.activeElement === createEl;
    creating = true;
    said(createMsgEl, "creating", "");
    refreshForm();
    try {
      const resp = await fetch("/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await jsonOf(resp);
      if (!resp.ok) {
        said(
          createMsgEl,
          "not created: " + refusalOf(resp, data, "experiment"),
          "refused",
        );
        return;
      }
      if (data === null) {
        said(
          createMsgEl,
          "the bench answered " +
            resp.status +
            " with a body this page could not read; the experiment may " +
            "exist, and the list below will say.",
          "",
        );
        void loadList();
        return;
      }
      projections.set(data.id, data.projected_cost);
      // Worded about Create, so it stays true after Start has spent.
      said(
        createMsgEl,
        "created experiment " +
          data.id +
          ". Creating spent nothing; money moves on Start.",
        "",
      );
      await loadList();
      // Selecting it asks the store about its dataset, as any selection
      // does; nothing is written here for it to disagree with.
      if (selections === selectedAtPress) select(data.id);
    } catch (err) {
      console.error("bench: creating an experiment failed", err);
      said(
        createMsgEl,
        "no answer came back (" +
          err.message +
          "); the experiment may exist, and the list below will say.",
        "",
      );
      void loadList();
    } finally {
      creating = false;
      refreshForm();
      if (hadFocus && document.activeElement === document.body) {
        createEl.focus();
      }
    }
  }

  // ---- The selected experiment: Start, progress, Stop.

  function countersText(experiment) {
    return (
      "done " +
      experiment.trials_done +
      " · failed " +
      experiment.trials_failed +
      " · refused " +
      experiment.trials_refused +
      " · of " +
      experiment.trials_total +
      (experiment.trials_total === 1 ? " trial" : " trials")
    );
  }

  // The status and its detail, verbatim: the detail is the server's
  // sentence ("stopped between trials", the ceiling's refusal, the
  // dataset that changed) and nothing here rewords it.
  function statusText(experiment) {
    return (
      experiment.status +
      (experiment.status_detail ? " · " + experiment.status_detail : "")
    );
  }

  // What the store holds under an experiment's recorded digest, asked on
  // every selection until the store answers with the summary. A failed
  // question leaves it unknown: Start stays live, because the door is the
  // rule and says so in its own words if the answer would have been no;
  // Score waits, because without the scorers the page cannot tell whether
  // a pass needs a judge (see BenchLib.scoreNudge).
  async function learnDataset(digest) {
    const known = datasetSummaries.get(digest);
    if (datasetHeld(known) || known === null) return;
    datasetSummaries.set(digest, null);
    let summary;
    try {
      const resp = await fetch("/datasets/" + digest);
      const data = resp.status === 404 ? null : await jsonOf(resp);
      if (resp.status === 404) summary = false;
      else if (resp.ok) {
        if (data !== null && Array.isArray(data.scorers)) {
          summary = {
            scorers: data.scorers,
            cites_documents: data.cites_documents,
          };
        }
      } else if (resp.status < 500 && typeof data?.detail === "string") {
        // The door's sentence about the bytes it holds (they no longer
        // hash to their digest, or are not UTF-8): asking again gets the
        // same answer until someone repairs the row, so it is shown as
        // it is rather than as a question to repeat.
        summary = refusalText(data.detail, "");
      }
    } catch (err) {
      console.error("bench: checking a stored dataset failed", err);
    }
    // A failed question is no answer: it is forgotten, so the next
    // selection asks again.
    if (summary === undefined) datasetSummaries.delete(digest);
    else datasetSummaries.set(digest, summary);
    renderExperiment();
  }

  function renderExperiment() {
    const experiment = selectedExperiment();
    selectedEl.hidden = experiment === null;
    if (experiment === null) return;
    setText(titleEl, experiment.name + " · experiment " + experiment.id);
    setText(statusEl, statusText(experiment));
    setText(countersEl, countersText(experiment));
    setText(
      datasetEl,
      "dataset: " +
        experiment.dataset_name +
        " · sha256 " +
        shortDigest(experiment.dataset_digest) +
        " · " +
        experiment.estimand_mode.replace("_", " ") +
        " estimand",
    );
    const created = experiment.status === "created";
    const summary = datasetSummaries.get(experiment.dataset_digest);
    const unstored = created && summary === false;
    // START IS LIVE FOR A CREATED EXPERIMENT AND FOR NOTHING ELSE, and not
    // for one whose dataset the store does not hold. The button being
    // greyed is a courtesy; the door is the rule, and it refuses a second
    // start, or a digest it does not hold, in its own words.
    startEl.disabled =
      starting || !created || started.has(experiment.id) || unstored;
    setText(
      startNoteEl,
      unstored
        ? "its dataset was read from a file and is not stored here, so " +
            "the page cannot name it; start it through the API with its " +
            "dataset_path"
        : "",
    );
    const projection = projections.get(experiment.id);
    setText(
      projectionEl,
      !created || unstored
        ? ""
        : projection
          ? projectionText(projection)
          : "no projection in this tab: the bench returns one only when " +
            "the experiment is created, and stores it nowhere",
    );
    // Stop is present while the experiment runs, and only then.
    stopEl.hidden = experiment.status !== "running";
    stopEl.disabled = stopping;
    // Score is present once the trials have finished, and only then: the
    // door refuses a created or running experiment, so the pass sees
    // every result. Absent is a courtesy; the door refuses on its own.
    const scorable = experimentFinished(experiment.status);
    scoreRowEl.hidden = !scorable;
    if (scorable) renderScore(experiment, summary);
    renderActionMessage();
  }

  function renderScore(experiment, summary) {
    const judged = judgeTasks(summary);
    judgeFieldEl.hidden = !judged;
    const catalogState = window.BenchControls.catalogIds().state;
    const reason = scoreNudge(summary, judgeEl.value, catalogState);
    scoreEl.disabled = scoring.has(experiment.id) || reason !== null;
    setText(scoreEl, scoreLabel(summary));
    setText(scoreNudgeEl, reason === null ? "" : "Score waits: " + reason);
    // Retry only while the read has failed and nothing is being asked.
    retryEl.hidden = summary !== undefined;
    setText(judgeNoteEl, judged ? judgeNote() : "");
    setText(
      scoreNoteEl,
      judged
        ? "each press asks the judge once for every trial of every judge " +
            "task that has response text, including trials already " +
            "judged, and every call is paid"
        : "",
    );
  }

  // What the judge select offers, in words. THE LIST IS NOT FILTERED and
  // nothing checks a judge: the README's rule that a judge route must not
  // have mandatory reasoning is derived from the contract rather than
  // measured, and no door enforces it, so the note says so rather than
  // the select quietly leaving routes out. What such a judge does is not
  // measured either, so the note names the three outcomes the code can
  // record and claims none of them.
  function judgeNote() {
    const catalog = window.BenchControls.catalogIds();
    if (catalog.state === "pending") return "the catalog is still loading";
    if (catalog.state === "unavailable") {
      return (
        "the catalog is not available, so no judge can be offered here; " +
        "score it through the API with judge_model"
      );
    }
    return (
      "the whole catalog, unfiltered: the bench does not check a judge, " +
      "and no door enforces the README's rule that a judge route must not " +
      "have mandatory reasoning (derived, not measured); such a judge may " +
      "give verdicts, be refused, or be billed for none"
    );
  }

  // The judge select's options: a blank first, since nothing is
  // pre-filled, then every catalog id in the catalog's order. Rebuilt
  // only when the catalog's state or ids change, keeping the choice when
  // it is still offered, because onChange also fires on every composer
  // keystroke and a rebuild then would close an open list.
  function renderJudgeOptions() {
    const catalog = window.BenchControls.catalogIds();
    const key = catalog.state + "\n" + catalog.ids.join("\n");
    if (key === judgeOptionsKey) return;
    judgeOptionsKey = key;
    const was = judgeEl.value;
    judgeEl.replaceChildren(new Option("choose a judge", ""));
    for (const id of catalog.ids) judgeEl.append(new Option(id, id));
    judgeEl.value = catalog.ids.includes(was) ? was : "";
    renderExperiment();
  }

  // A progress frame onto the experiment it describes. The counters are
  // absolute, so a frame replaces them rather than adding to them, and a
  // frame missed while the stream was down costs nothing: the next one
  // carries the current values.
  function applyFrame(id, frame) {
    const experiment = experimentById(id);
    if (!experiment) return;
    for (const key of [
      "status",
      "status_detail",
      "trials_total",
      "trials_done",
      "trials_refused",
      "trials_failed",
    ]) {
      experiment[key] = frame[key];
    }
    renderRow(experiment);
    if (id === selectedId) renderExperiment();
  }

  // One row's counts, from the experiment as the panel now holds it.
  function renderRow(experiment) {
    const row = listEl.querySelector(
      "[data-testid=experiment-row][data-id='" + experiment.id + "']",
    );
    if (row)
      setText(row.querySelector(".hcount"), experimentRowMeta(experiment));
  }

  function unwatch() {
    if (watch !== null) {
      watch.source.close();
      watch = null;
    }
    setText(streamEl, "");
  }

  // The progress door for one experiment. RECONNECTING IS THE ORDINARY
  // CASE: an EventSource reopens a dropped stream by itself, and the
  // first frame after it carries the current counters. It is closed on a
  // finished status, because the door closes the stream after the last
  // frame and an EventSource left open would reopen it every few seconds
  // for ever; and on any change of selection, because the door holds a
  // created experiment's stream open indefinitely.
  function watchProgress(id) {
    unwatch();
    const source = new EventSource("/experiments/" + id + "/progress");
    watch = { id: id, source: source };
    source.onmessage = (event) => {
      if (watch === null || watch.source !== source) return;
      let frame;
      try {
        frame = JSON.parse(event.data);
      } catch (err) {
        console.error("bench: a progress frame was not JSON", err);
        return;
      }
      setText(streamEl, "");
      applyFrame(id, frame);
      if (experimentFinished(frame.status)) {
        unwatch();
        finished(id);
      }
    };
    source.onerror = () => {
      if (watch === null || watch.source !== source) return;
      if (source.readyState === EventSource.CLOSED) {
        unwatch();
        setText(
          streamEl,
          "the progress stream closed; select the experiment again to reopen it",
        );
      } else {
        setText(streamEl, "the progress stream dropped; reconnecting");
      }
    };
  }

  // Whether the selected experiment's progress should be watched and is
  // not: it runs, or this tab started it and it has not finished, and no
  // stream is open for it.
  function watchIfRunning() {
    const experiment = selectedExperiment();
    if (experiment === null) return;
    const live =
      experiment.status === "running" ||
      (started.has(experiment.id) && experiment.status === "created");
    if (live && !(watch !== null && watch.id === experiment.id)) {
      watchProgress(experiment.id);
    }
  }

  // An experiment's report, read again, when it is still the one selected.
  // A later answer about another experiment must not replace the report
  // under the selection the person moved to.
  function showReport(id) {
    if (id === selectedId) window.BenchReport.show(id);
  }

  // A finished experiment: its last frame has already put the final
  // status and counters on its row and in the panel (and so taken away a
  // present-tense word from Stop, and shown the Score row), and its
  // report is shown again with the trials it has.
  function finished(id) {
    showReport(id);
  }

  async function start() {
    const experiment = selectedExperiment();
    if (
      experiment === null ||
      starting ||
      experiment.status !== "created" ||
      started.has(experiment.id)
    ) {
      return;
    }
    const id = experiment.id;
    const hadFocus = document.activeElement === startEl;
    const selectedAtPress = selections;
    starting = true;
    saidAbout(id, "starting", "");
    renderExperiment();
    try {
      // THE DIGEST THE EXPERIMENT RECORDED, not the one selected in the
      // Datasets panel now. The door refuses any other, and the person
      // may have selected another dataset since.
      const resp = await fetch("/experiments/" + id + "/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_digest: experiment.dataset_digest }),
      });
      const data = await jsonOf(resp);
      if (!resp.ok) {
        saidAbout(
          id,
          "not started: " + refusalOf(resp, data, "start"),
          "refused",
        );
        // The refusal may name a status the list has not caught up with
        // (another tab started it, or it has finished); read it again,
        // before focus goes back to its row.
        await loadList();
        return;
      }
      started.add(id);
      // The door's own word: it answers 202 with the status it set, and
      // the runner records it a moment later. The experiment runs whether
      // or not it is still the one selected, and whatever list the panel
      // holds now: a reload during the request replaced the object read
      // at the press, so it is looked up again by id.
      const current = experimentById(id);
      if (current !== null) {
        current.status = data?.status || "running";
        renderRow(current);
      }
      saidAbout(id, "started", "");
      watchIfRunning();
    } catch (err) {
      console.error("bench: starting an experiment failed", err);
      saidAbout(
        id,
        "no answer came back (" +
          err.message +
          "); the experiment may have started, and its status will say.",
        "",
      );
      await loadList();
    } finally {
      starting = false;
      renderExperiment();
      if (hadFocus && selections === selectedAtPress) focusRow(id);
    }
  }

  async function stop() {
    const experiment = selectedExperiment();
    if (experiment === null || stopping) return;
    const id = experiment.id;
    const hadFocus = document.activeElement === stopEl;
    const selectedAtPress = selections;
    stopping = true;
    renderExperiment();
    try {
      // A body of {} with the JSON type, because the bench refuses any
      // POST that is not JSON, bodiless ones included.
      const resp = await fetch("/experiments/" + id + "/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      const data = await jsonOf(resp);
      if (!resp.ok) {
        saidAbout(
          id,
          "not stopped: " + refusalOf(resp, data, "stop"),
          "refused",
        );
        await loadList();
        return;
      }
      // Stopping, not stopped: the trial in flight finishes first, and
      // the status says stopped once it has; this line goes then.
      saidAbout(id, "stopping after the trial in flight", "stopping");
    } catch (err) {
      console.error("bench: stopping an experiment failed", err);
      saidAbout(
        id,
        "no answer came back (" + err.message + "); its status will say.",
        "",
      );
      await loadList();
    } finally {
      stopping = false;
      renderExperiment();
      if (hadFocus && selections === selectedAtPress) focusRow(id);
    }
  }

  // Retry beside "its dataset could not be read": the same question the
  // selection asked, asked again. Retry hides while it is out, so a
  // keyboard user's focus is put on the experiment's row rather than
  // left on a control that is gone.
  function retryDatasetRead() {
    const experiment = selectedExperiment();
    if (experiment === null) return;
    const hadFocus = document.activeElement === retryEl;
    if (hadFocus) retryEl.blur();
    void learnDataset(experiment.dataset_digest);
    renderExperiment();
    if (hadFocus) focusRow(experiment.id);
  }

  // The clock time of an answer, in UTC and saying so: Score's words
  // describe a pass on the server that no door reports on, so they are
  // stamped rather than worded as if still current.
  function utcTime() {
    return new Date().toISOString().slice(11, 19) + " UTC";
  }

  async function score() {
    const experiment = selectedExperiment();
    if (
      experiment === null ||
      !experimentFinished(experiment.status) ||
      scoring.has(experiment.id)
    ) {
      return;
    }
    const summary = datasetSummaries.get(experiment.dataset_digest);
    const judge = judgeEl.value;
    const catalogState = window.BenchControls.catalogIds().state;
    if (scoreNudge(summary, judge, catalogState) !== null) return;
    const id = experiment.id;
    const hadFocus = document.activeElement === scoreEl;
    const selectedAtPress = selections;
    scoring.add(id);
    saidAbout(id, "sending the score request", "");
    renderExperiment();
    try {
      // THE DIGEST THE EXPERIMENT RECORDED, as Start sends, and the judge
      // only when its dataset has judge tasks.
      const resp = await fetch("/experiments/" + id + "/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          scoreBody(experiment.dataset_digest, summary, judge),
        ),
      });
      const data = await jsonOf(resp);
      if (!resp.ok) {
        // The door's sentence as it is, and Score stays live: another
        // pass holds the one scoring slot, and the server is the one that
        // knows when it ends.
        saidAbout(
          id,
          "not scored at " + utcTime() + ": " + refusalOf(resp, data, "score"),
          "refused",
        );
        return;
      }
      saidAbout(
        id,
        "a scoring pass was started at " +
          utcTime() +
          "; no door says when it ends, so select the experiment again to " +
          "read what it has scored since",
        "",
      );
      // After scoring, the report opens: read now, while the pass runs.
      showReport(id);
    } catch (err) {
      console.error("bench: scoring an experiment failed", err);
      saidAbout(
        id,
        "no answer came back at " +
          utcTime() +
          " (" +
          err.message +
          "); no door says whether a pass started",
        "",
      );
      showReport(id);
    } finally {
      scoring.delete(id);
      renderExperiment();
      if (hadFocus && selections === selectedAtPress) focusRow(id);
    }
  }

  // Selecting a row: the lifecycle below the form describes it, its
  // report opens as it always has, and its progress is watched while it
  // runs. A created experiment's is not: the door would hold its stream
  // open until someone starts it, and Start opens the watch itself.
  function select(id) {
    if (!(watch !== null && watch.id === id)) unwatch();
    // A judge chosen for one experiment is not carried to another, where
    // it would arm a paid press nobody chose for that one.
    if (id !== selectedId) judgeEl.value = "";
    selections += 1;
    selectedId = id;
    renderListSelection();
    const experiment = selectedExperiment();
    // Asked whatever the status: a running experiment finishes under the
    // watch without being selected again, and its Score row then needs
    // the answer.
    if (experiment !== null) void learnDataset(experiment.dataset_digest);
    renderExperiment();
    watchIfRunning();
    window.BenchReport.show(id);
  }

  // ---- The list.

  // Same state discipline as the history panel: an empty list means two
  // different things while a fetch is in flight, so the list names its
  // own state rather than leaving a blank to be read as "none".
  function setListState(state, message) {
    listEl.dataset.state = state;
    listEl.textContent = message;
  }

  function renderListSelection() {
    for (const el of listEl.querySelectorAll("[data-testid=experiment-row]")) {
      el.setAttribute(
        "aria-pressed",
        String(Number(el.dataset.id) === selectedId),
      );
    }
  }

  async function loadList() {
    if (listController !== null) listController.abort();
    const controller = new AbortController();
    listController = controller;
    const version = ++listVersion;
    // A row that had focus is replaced by the reload; focus goes back to
    // the row for the same experiment, so a keyboard user keeps their
    // place in the list.
    const focusedRow = document.activeElement?.closest?.(
      "[data-testid=experiment-row]",
    );
    const focusedId = focusedRow ? Number(focusedRow.dataset.id) : null;
    setListState("loading", "loading experiments");
    let data;
    try {
      const resp = await fetch("/experiments", { signal: controller.signal });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch (err) {
      if (controller.signal.aborted || version !== listVersion) return;
      console.error("experiment list load failed", err);
      setListState("error", "failed to load experiments: " + err.message);
      return;
    }
    if (version !== listVersion) return;
    // The table read now is at least as new as any frame the stream has
    // brought, since the frames are reads of the same table, so the list
    // simply replaces what the panel held.
    experiments = data.experiments;
    // A selection whose experiment is no longer listed is dropped rather
    // than kept as an id nothing on screen describes; the note below a
    // full list says why one can fall off it.
    if (selectedId !== null && !experiments.some((e) => e.id === selectedId)) {
      selectedId = null;
      unwatch();
    }
    setText(
      listNoteEl,
      experiments.length >= EXPERIMENT_LIMITS.listed
        ? "The newest " +
            EXPERIMENT_LIMITS.listed +
            " are listed; older ones stay in bench.db."
        : "",
    );
    if (experiments.length === 0) {
      setListState("empty", "no experiments yet");
      renderExperiment();
      return;
    }
    setListState("ready", "");
    for (const experiment of experiments) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "hrow xp-entry";
      row.dataset.testid = "experiment-row";
      row.dataset.id = String(experiment.id);
      const time = document.createElement("span");
      time.className = "htime";
      time.textContent =
        experiment.created_at.slice(0, 19).replace("T", " ") + " UTC";
      const name = document.createElement("span");
      name.className = "hprompt";
      name.textContent = experiment.name;
      const meta = document.createElement("span");
      meta.className = "hcount";
      meta.textContent = experimentRowMeta(experiment);
      row.append(time, name, meta);
      row.addEventListener("click", () => select(experiment.id));
      listEl.append(row);
      if (experiment.id === focusedId) row.focus();
    }
    renderListSelection();
    // A selection made while an older reload was superseded (Create's
    // own select finding nothing yet) was never asked about; ask now,
    // rather than let its Score row read a question that never failed.
    const shown = selectedExperiment();
    if (shown !== null && !datasetSummaries.has(shown.dataset_digest)) {
      void learnDataset(shown.dataset_digest);
    }
    renderExperiment();
    // The reload may say the selected experiment runs now (started in
    // another tab, or its Start answer was lost); its progress is then
    // watched, as selecting it would.
    watchIfRunning();
  }

  function init() {
    detailsEl.addEventListener("click", (event) => {
      // The synchronous claim, exactly as the history panel makes it:
      // toggle is dispatched asynchronously, so without this the panel
      // still reads the previous load's terminal state when the click
      // lands. See static/history.js for the incident that taught it.
      if (!detailsEl.open && event.target.closest("summary")) {
        setListState("loading", "loading experiments");
      }
    });
    detailsEl.addEventListener("toggle", () => {
      if (detailsEl.open) void loadList();
    });
    for (const el of [nameEl, repeatsEl, seedEl]) {
      el.addEventListener("input", refreshForm);
    }
    for (const el of [attachmentsEl, metricEl, haltEl]) {
      el.addEventListener("change", refreshForm);
    }
    estimandEl.addEventListener("change", () => {
      renderEstimandNote();
      refreshForm();
    });
    createEl.addEventListener("click", () => {
      void create();
    });
    startEl.addEventListener("click", () => {
      void start();
    });
    stopEl.addEventListener("click", () => {
      void stop();
    });
    // One press is one POST. The second click of a mouse double-click
    // lands after a quick 202 has made Score live again, and would start a
    // second paid pass or replace this pass's line with the door's refusal
    // of it; a click's detail counts it (Enter, Space and element.click()
    // carry 0, a single mouse click 1).
    scoreEl.addEventListener("click", (event) => {
      if (event.detail > 1) return;
      void score();
    });
    judgeEl.addEventListener("change", renderExperiment);
    retryEl.addEventListener("click", retryDatasetRead);
    window.BenchControls.onChange(refreshForm);
    window.BenchControls.onChange(renderJudgeOptions);
    window.BenchDatasets.onSelect(() => {
      renderDatasetDependent();
      refreshForm();
    });
    renderDatasetDependent();
    renderEstimandNote();
    refreshForm();
    renderJudgeOptions();
    renderExperiment();
  }

  // refresh reads the list again, as closing and opening the panel does;
  // the browser suite drives it to prove what a reload keeps.
  window.BenchLifecycle = { init, refresh: loadList };
})();
