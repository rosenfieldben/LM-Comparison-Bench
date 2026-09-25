// The dataset builder and the stored-dataset library. Exposed on
// window.BenchDatasets.
//
// THE BROWSER COMPOSES; THE SERVER VALIDATES. Rows become JSONL through
// BenchLib.composeJsonl, pasted or uploaded text is sent as it is, and
// POST /datasets runs the one parser the whole bench uses. Its sentence
// is what this panel prints, beside the row whose line it names. The
// nudges that grey Store are a courtesy the server would have refused
// anyway, never a second validator: see BenchLib.rowNudge.
//
// NOTHING IS PRE-FILLED. A new row has an empty id box, a task imported
// from the prompt library has no scorer, and a stored dataset is not
// selected until the person selects it. Each of those is a decision,
// and a page that made it would record a choice nobody took.
//
// Builder state is per tab and never persisted, for the reason the
// composer's staged documents are not: rows half-typed into a closed tab
// are a draft, and a draft that reappeared would be tasks nobody chose to
// keep. The name and JSONL boxes say autocomplete="off" in the markup for
// the same reason, or Back would refill them. What is stored is stored
// in bench.db, keyed by digest.
(function () {
  const {
    shortDigest,
    fmtBytes,
    DATASET_LIMITS,
    BUILDER_MAX_ROWS,
    DATASET_SCORERS,
    codePoints,
    utf8Length,
    composeJsonl,
    rowNudge,
    datasetNudge,
    refusalLine,
    countLines,
    refusalText,
  } = window.BenchLib;

  // How many stored documents a picker offers and how many stored
  // datasets the library lists: the most either door will return. Asked
  // for explicitly, as the history panel asks, so the note that says a
  // list is full cannot drift from what was asked.
  const LIST_LIMIT = 500;

  const detailsEl = document.getElementById("datasets");
  const nameEl = document.getElementById("dataset-name");
  const modeEl = document.getElementById("dataset-mode");
  const rowsPaneEl = document.getElementById("dataset-rows-pane");
  const rowsEl = document.getElementById("dataset-rows");
  const addRowEl = document.getElementById("dataset-add-row");
  const fromPromptsEl = document.getElementById("dataset-from-prompts");
  const promptPickerEl = document.getElementById("dataset-prompt-picker");
  const jsonlPaneEl = document.getElementById("dataset-jsonl-pane");
  const jsonlEl = document.getElementById("dataset-jsonl");
  const fileEl = document.getElementById("dataset-file");
  const fileLabelEl = document.getElementById("dataset-file-label");
  const fileClearEl = document.getElementById("dataset-file-clear");
  const ceilingsEl = document.getElementById("dataset-ceilings");
  const storeEl = document.getElementById("dataset-store");
  const nudgeEl = document.getElementById("dataset-nudge");
  const msgEl = document.getElementById("dataset-msg");
  const selectedEl = document.getElementById("dataset-selected");
  const libraryEl = document.getElementById("dataset-library");

  // The builder's rows, in order, which is the order of the lines Store
  // sends: row N is line N.
  let rows = [];
  // "rows" or "jsonl": which of the two ways in Store sends.
  let mode = "rows";
  // A file loaded in JSONL mode, as the text its bytes decoded to, or
  // null. Held here and not in the textarea: see the markup's comment.
  let loadedFile = null;
  // One Store at a time; a second click while the first is in flight
  // must not POST twice.
  let storing = false;
  // The server's last refusal: its sentence, the line it names, and the
  // text and the way in it was said about; or null. SHOWN BESIDE A ROW
  // ONLY WHILE THE ROWS WOULD SEND THAT SAME TEXT AGAIN. A change to what
  // Store would send (an edit that reaches the composed text, a removed
  // row, the other way in, or any of those made while Store was in
  // flight) means the line it names is no longer the line on screen, and
  // the sentence beside the wrong row would be worse than none. An edit
  // that composes to the same text (a system message of spaces, the
  // dataset's name) leaves it, and undoing a change brings it back,
  // because the text is again the text it described. Pasted text has no
  // rows, so a refusal of it marks none.
  let refusal = null;
  // Whether the panel's message line is still the last refusal's, so
  // refresh keeps its note about a changed text true in both directions.
  let messageIsRefusal = false;
  // The stored-attachment list the document pickers offer, or null.
  // Each picker open fetches its own, so a document attached in the
  // composer since is offered; the answer to the newest request is kept
  // here for the redraw a pressed option causes, which draws at once
  // from it rather than fetching again, and a failed fetch leaves it as
  // it was.
  let documents = null;
  let documentsAsked = 0;
  let documentsKept = 0;
  // Each picker render and each prompt-picker open is numbered, and one
  // whose fetch returns after a newer one began, or after its picker was
  // closed or its row removed, draws nothing: two runs appending to one
  // live picker listed every option twice.
  let pickerRenders = 0;
  let promptPickerOpens = 0;
  // The stored datasets, newest first, and the one selected, by digest.
  let library = [];
  let selectedDigest = null;
  let libraryController = null;
  let libraryVersion = 0;

  function blankRow() {
    return {
      id: "",
      prompt: "",
      system: "",
      scorer: "",
      reference: "",
      pattern: "",
      rubric: "",
      threshold: "",
      // Stored documents cited by this task, as {digest, label} in the
      // order they were picked; the digest is what the line cites.
      documents: [],
    };
  }

  // A live region's text, assigned only when it changes. Assigning the
  // same string still replaces the text node, which assistive technology
  // announces again, and refresh runs on every keystroke.
  function setText(el, text) {
    if (el.textContent !== text) el.textContent = text;
  }

  // What Store would send, exactly.
  function content() {
    if (mode === "rows") {
      return composeJsonl(
        rows.map((row) =>
          Object.assign({}, row, {
            documents: row.documents.map((doc) => doc.digest),
          }),
        ),
      );
    }
    return loadedFile ? loadedFile.text : jsonlEl.value;
  }

  // Why Store is greyed, or null. The dataset's own reason first, then
  // the first row that has one, named by its line.
  function nudge(text) {
    const whole = datasetNudge(nameEl.value, text);
    if (whole) return whole;
    if (mode === "rows") {
      for (let i = 0; i < rows.length; i += 1) {
        const reason = rowNudge(rows[i]);
        if (reason) return "line " + (i + 1) + " " + reason;
      }
    }
    return null;
  }

  // The ceilings, shown before Store. Rows are counted exactly because
  // the page composed them. Pasted or uploaded text is counted in lines
  // holding anything, split and skipped as the parser splits and skips,
  // against MAX_TASKS, because that is the way in that can reach it; a
  // count and not a parse, since the page does not parse what it did not
  // compose.
  function renderCeilings(text) {
    const parts = [];
    if (mode === "rows") {
      let longest = 0;
      for (const row of rows) {
        longest = Math.max(longest, codePoints(row.prompt));
      }
      parts.push(rows.length + " of " + DATASET_LIMITS.maxTasks + " tasks");
      parts.push(
        "longest prompt " +
          longest +
          " of " +
          DATASET_LIMITS.maxPromptChars +
          " characters",
      );
    } else {
      parts.push(
        countLines(text) + " of " + DATASET_LIMITS.maxTasks + " task lines",
      );
    }
    parts.push(
      utf8Length(text) + " of " + DATASET_LIMITS.maxDatasetBytes + " bytes",
    );
    let line = parts.join(" · ");
    if (mode === "rows" && rows.length >= BUILDER_MAX_ROWS) {
      line +=
        ". The builder holds " +
        BUILDER_MAX_ROWS +
        " rows; past that, paste or upload JSONL.";
    }
    setText(ceilingsEl, line);
  }

  // Whether the last refusal is about the text on screen now.
  function refusalStands(text) {
    return refusal !== null && refusal.mode === mode && refusal.sent === text;
  }

  // The panel's line for the last refusal: the server's sentence, and a
  // note while the text on screen is not the text it was said about.
  function refusalMessage(stands) {
    return (
      "not stored: " +
      refusal.text +
      (stands
        ? ""
        : " (said of the text as Store sent it, which has changed since)")
    );
  }

  // Everything that depends on the rows' values but not on their number,
  // so typing never rebuilds the row it is typing in.
  function refresh() {
    const text = content();
    renderCeilings(text);
    const reason = nudge(text);
    storeEl.disabled = storing || reason !== null;
    setText(nudgeEl, storing ? "" : reason ? "Store waits: " + reason : "");
    addRowEl.disabled = rows.length >= BUILDER_MAX_ROWS;
    const stands = refusalStands(text);
    if (messageIsRefusal && !storing) {
      setText(msgEl, refusalMessage(stands));
    }
    const marks = stands && mode === "rows";
    for (const el of rowsEl.querySelectorAll("[data-testid=dataset-row]")) {
      const index = Number(el.dataset.index);
      const msg = el.querySelector("[data-testid=dataset-row-msg]");
      if (marks && refusal.line === index + 1) {
        // THE SERVER'S SENTENCE, BESIDE THE ROW ITS LINE NAMES. It
        // outranks the row's own nudge, which is only ever a subset of
        // what the server says.
        el.dataset.state = "refused";
        msg.dataset.state = "refused";
        setText(msg, refusal.text);
      } else {
        const own = rowNudge(rows[index]);
        el.dataset.state = own ? "nudge" : "";
        msg.dataset.state = own ? "nudge" : "";
        setText(msg, own || "");
      }
    }
  }

  // A labelled field of one row. The caption is visible, because a
  // placeholder is ghost text below the contrast floor and vanishes
  // once something is typed. The accessible name carries the line, so a
  // screen reader tells one row's prompt from another's; since that name
  // overrides the caption, the caption is the field's description, ahead
  // of the row's message, so its facts ("optional; blank sends none")
  // are read on focus too.
  function field(index, key, tag, testid, caption, label) {
    const wrap = document.createElement("label");
    wrap.className = "ds-cap ds-cap-" + key;
    const text = document.createElement("span");
    text.className = "ds-cap-text";
    text.id = "dataset-row-cap-" + index + "-" + key;
    text.textContent = caption;
    const el = document.createElement(tag);
    if (tag === "input") el.type = "text";
    if (tag === "textarea") el.rows = key === "prompt" ? 2 : 1;
    el.dataset.testid = testid;
    el.className = "ds-field ds-" + key;
    el.setAttribute("aria-label", label + ", line " + (index + 1));
    el.setAttribute("aria-describedby", text.id + " dataset-row-msg-" + index);
    el.value = rows[index][key];
    el.addEventListener("input", () => {
      rows[index][key] = el.value;
      refresh();
    });
    wrap.append(text, el);
    return wrap;
  }

  // The fields a scorer uses, and only those: reference for the comparing
  // scorers, pattern for regex, rubric and an optional pass threshold for
  // the judge. What was typed into a field the current scorer does not
  // use stays in the row and is not sent; see BenchLib.composeTask.
  function scorerFields(index) {
    const kind = rows[index].scorer;
    const out = [];
    if (
      kind === "exact" ||
      kind === "normalized_exact" ||
      kind === "contains"
    ) {
      out.push(
        field(
          index,
          "reference",
          "input",
          "dataset-row-reference",
          "reference · what the answer is compared with",
          "reference",
        ),
      );
    }
    if (kind === "regex") {
      out.push(
        field(
          index,
          "pattern",
          "input",
          "dataset-row-pattern",
          "pattern · a regular expression the answer must match",
          "pattern",
        ),
      );
    }
    if (kind === "judge") {
      out.push(
        field(
          index,
          "rubric",
          "textarea",
          "dataset-row-rubric",
          "rubric · what the judge scores the answer against",
          "rubric",
        ),
        field(
          index,
          "threshold",
          "input",
          "dataset-row-threshold",
          "pass threshold · 0 to 1, optional; blank publishes the score alone",
          "pass threshold",
        ),
      );
    }
    return out;
  }

  function documentChip(index, doc) {
    const chip = document.createElement("span");
    chip.className = "attach-chip ds-doc";
    chip.dataset.testid = "dataset-row-document";
    chip.dataset.digest = doc.digest;
    chip.textContent = doc.label + " · sha256 " + shortDigest(doc.digest);
    chip.title = doc.label + "\nsha256 " + doc.digest;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chip-x";
    remove.dataset.testid = "dataset-row-document-remove";
    remove.textContent = "×";
    remove.setAttribute(
      "aria-label",
      "remove " + doc.label + " from line " + (index + 1),
    );
    remove.addEventListener("click", () => {
      rows[index].documents = rows[index].documents.filter(
        (d) => d.digest !== doc.digest,
      );
      renderRows(undefined, {
        index: index,
        testid: "dataset-row-add-document",
      });
    });
    chip.append(remove);
    return chip;
  }

  // The stored documents, fetched now. Answers can arrive out of order,
  // so the cache keeps the answer to the latest request that has come
  // back, never an older one over a newer.
  async function fetchDocuments() {
    const asked = ++documentsAsked;
    let list;
    try {
      const resp = await fetch("/attachments?limit=" + LIST_LIMIT);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      list = (await resp.json()).attachments;
    } catch (err) {
      console.error("bench: stored documents failed to load", err);
      throw err;
    }
    if (asked > documentsKept) {
      documentsKept = asked;
      documents = list;
    }
    return list;
  }

  // What the picker says when the bench stores no documents yet. The
  // snapshot route is named only where the bench will compose one: on a
  // bench with snapshots off, the composer above says so, and a note
  // here sending a person to it would offer a door the server refuses.
  function noDocumentsNote() {
    const snapshots = window.BenchState.snapshots;
    const how = snapshots?.enabled
      ? "Attach a file or compose a snapshot in the composer above"
      : "Attach a file in the composer above";
    return (
      "No stored documents. " +
      how +
      "; it is stored the moment it is attached, and + Document then " +
      "offers it here."
    );
  }

  // The stored documents a task can cite, as toggles. A task cites a
  // document by its digest, the way a dataset always has, and a
  // repository snapshot is a stored attachment of kind snapshot, so it
  // is offered here like any other.
  async function renderPicker(index, pickerEl, fresh) {
    const render = String(++pickerRenders);
    pickerEl.dataset.render = render;
    const stale = () =>
      pickerEl.dataset.render !== render ||
      !pickerEl.isConnected ||
      pickerEl.hidden ||
      rows[index] === undefined;
    pickerEl.replaceChildren();
    const note = document.createElement("span");
    note.className = "ds-picker-note";
    note.dataset.testid = "dataset-document-note";
    pickerEl.append(note);
    let list = documents;
    if (fresh || list === null) {
      note.textContent = "loading stored documents";
      try {
        list = await fetchDocuments();
      } catch (err) {
        if (!stale()) {
          note.textContent = "failed to load stored documents: " + err.message;
        }
        return;
      }
      if (stale()) return;
    }
    if (list.length === 0) {
      note.textContent = noDocumentsNote();
      return;
    }
    const cap = window.BenchAttach.MAX_ATTACHMENTS;
    const cited = new Set(rows[index].documents.map((d) => d.digest));
    let said =
      cited.size >= cap
        ? "A task cites at most " + cap + " documents."
        : "Cite up to " + cap + " stored documents.";
    if (list.length >= LIST_LIMIT) {
      said +=
        " The newest " +
        LIST_LIMIT +
        " are offered; an older one is cited by its digest in pasted JSONL.";
    }
    note.textContent = said;
    for (const doc of list) {
      const label = window.BenchAttach.docLabel(doc);
      const option = document.createElement("button");
      option.type = "button";
      option.className = "attach-chip ds-option";
      option.dataset.testid = "dataset-document-option";
      option.dataset.digest = doc.digest;
      const on = cited.has(doc.digest);
      option.setAttribute("aria-pressed", String(on));
      option.disabled = !on && cited.size >= cap;
      const face =
        label + " · " + doc.kind + " · sha256 " + shortDigest(doc.digest);
      option.textContent = face;
      option.setAttribute("aria-label", face + ", for line " + (index + 1));
      option.title =
        label +
        "\nsha256 " +
        doc.digest +
        "\n" +
        fmtBytes(doc.byte_size) +
        ", read as " +
        doc.kind;
      option.addEventListener("click", () => {
        const row = rows[index];
        if (row.documents.some((d) => d.digest === doc.digest)) {
          row.documents = row.documents.filter((d) => d.digest !== doc.digest);
        } else if (row.documents.length < cap) {
          row.documents = row.documents.concat({
            digest: doc.digest,
            label: label,
          });
        }
        renderRows(index, {
          index: index,
          testid: "dataset-document-option",
          digest: doc.digest,
        });
      });
      pickerEl.append(option);
    }
  }

  function rowElement(index, openPicker) {
    const row = rows[index];
    const el = document.createElement("div");
    el.className = "ds-row";
    el.dataset.testid = "dataset-row";
    el.dataset.index = String(index);

    const head = document.createElement("div");
    head.className = "ds-row-head";
    const line = document.createElement("span");
    line.className = "ds-line";
    line.dataset.testid = "dataset-row-line";
    line.textContent = "line " + (index + 1);
    const id = field(index, "id", "input", "dataset-row-id", "id", "task id");
    const wrap = document.createElement("span");
    wrap.className = "sel-wrap";
    const scorer = document.createElement("select");
    scorer.dataset.testid = "dataset-row-scorer";
    scorer.setAttribute("aria-label", "scorer, line " + (index + 1));
    scorer.append(new Option("no scorer", ""));
    for (const kind of DATASET_SCORERS) scorer.append(new Option(kind, kind));
    scorer.value = row.scorer;
    scorer.addEventListener("change", () => {
      row.scorer = scorer.value;
      renderRows(undefined, { index: index, testid: "dataset-row-scorer" });
    });
    wrap.append(scorer);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chip-x";
    remove.dataset.testid = "dataset-row-remove";
    remove.textContent = "×";
    remove.setAttribute("aria-label", "remove the task on line " + (index + 1));
    remove.addEventListener("click", () => {
      rows.splice(index, 1);
      // Focus moves to the remove button now on this line, or the one
      // above when this was the last, or + Task when none is left.
      let next = addRowEl;
      if (index < rows.length) {
        next = { index: index, testid: "dataset-row-remove" };
      } else if (index > 0) {
        next = { index: index - 1, testid: "dataset-row-remove" };
      }
      renderRows(undefined, next);
    });
    head.append(line, id, wrap, remove);

    const prompt = field(
      index,
      "prompt",
      "textarea",
      "dataset-row-prompt",
      "prompt",
      "prompt",
    );
    const fields = document.createElement("div");
    fields.className = "ds-fields";
    fields.append(...scorerFields(index));
    const system = field(
      index,
      "system",
      "textarea",
      "dataset-row-system",
      "system message · optional; blank sends none",
      "system message",
    );

    const docs = document.createElement("div");
    docs.className = "ds-docs";
    for (const doc of row.documents) docs.append(documentChip(index, doc));
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "attach-btn";
    pick.dataset.testid = "dataset-row-add-document";
    pick.textContent = "+ Document";
    pick.setAttribute("aria-label", "add a document to line " + (index + 1));
    const picker = document.createElement("div");
    picker.className = "ds-picker";
    picker.id = "dataset-row-picker-" + index;
    picker.dataset.testid = "dataset-row-picker";
    picker.hidden = !openPicker;
    pick.setAttribute("aria-controls", picker.id);
    pick.setAttribute("aria-expanded", String(!!openPicker));
    pick.addEventListener("click", () => {
      const opening = picker.hidden;
      picker.hidden = !opening;
      pick.setAttribute("aria-expanded", String(opening));
      if (opening) void renderPicker(index, picker, true);
    });
    if (openPicker) void renderPicker(index, picker, false);
    docs.append(pick);

    // Not a live region of its own: fifty rows re-announcing their notes
    // on every keystroke would drown the one line that matters. The
    // nudge line and the panel message announce; this is tied to the
    // row's fields by aria-describedby and read when one is focused.
    const msg = document.createElement("div");
    msg.className = "ds-row-msg";
    msg.id = "dataset-row-msg-" + index;
    msg.dataset.testid = "dataset-row-msg";

    el.append(head, prompt, fields, system, docs, picker, msg);
    return el;
  }

  // Where focus goes after a rebuild: the same control in the new rows,
  // or an element outside them.
  function focusAfterRebuild(focus) {
    if (focus instanceof HTMLElement) {
      focus.focus();
      return;
    }
    const rowEl = rowsEl.querySelector(
      "[data-testid=dataset-row][data-index='" + focus.index + "']",
    );
    if (!rowEl) return;
    for (const el of rowEl.querySelectorAll(
      "[data-testid='" + focus.testid + "']",
    )) {
      if (focus.digest === undefined || el.dataset.digest === focus.digest) {
        el.focus();
        return;
      }
    }
  }

  // Rebuild every row element. Called on structural changes only (a row
  // added, removed, a scorer or a document list changed), so typing into
  // a field never replaces the field under the cursor. openPickerFor
  // keeps one row's picker open across the rebuild its own toggle caused.
  // A REBUILD REPLACES THE CONTROL THAT HAD FOCUS, so the handler that
  // caused it names where focus goes next, and a keyboard user keeps
  // their place rather than landing on <body>.
  function renderRows(openPickerFor, focus) {
    rowsEl.replaceChildren();
    rows.forEach((_, index) =>
      rowsEl.append(rowElement(index, index === openPickerFor)),
    );
    refresh();
    if (focus) focusAfterRebuild(focus);
  }

  function addRows(added, focus) {
    rows = rows.concat(added);
    renderRows(undefined, focus);
  }

  // ---- Tasks from the prompt library.

  async function openPromptPicker() {
    const open = ++promptPickerOpens;
    promptPickerEl.replaceChildren();
    const note = document.createElement("span");
    note.className = "ds-picker-note";
    note.dataset.testid = "dataset-prompt-note";
    note.textContent = "loading saved prompts";
    promptPickerEl.append(note);
    let prompts;
    try {
      const resp = await fetch("/prompts");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      prompts = (await resp.json()).prompts;
    } catch (err) {
      if (open !== promptPickerOpens) return;
      console.error("bench: saved prompts failed to load", err);
      note.textContent = "failed to load saved prompts: " + err.message;
      return;
    }
    if (open !== promptPickerOpens) return;
    if (prompts.length === 0) {
      note.textContent =
        "No saved prompts. Save one from the composer's prompt box first.";
      return;
    }
    note.textContent =
      "Each checked prompt becomes one task. Its id and its scorer are " +
      "yours to set.";
    const boxes = [];
    for (const saved of prompts) {
      const label = document.createElement("label");
      label.className = "ds-prompt-option";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.dataset.testid = "dataset-prompt-option";
      box.dataset.text = saved.text;
      const name = document.createElement("span");
      name.textContent = saved.name;
      label.append(box, name);
      promptPickerEl.append(label);
      boxes.push(box);
    }
    const add = document.createElement("button");
    add.type = "button";
    add.dataset.testid = "dataset-prompt-add";
    const sync = () => {
      const n = boxes.filter((b) => b.checked).length;
      add.textContent = "Add " + n + (n === 1 ? " task" : " tasks");
      add.disabled = n === 0;
    };
    for (const box of boxes) box.addEventListener("change", sync);
    sync();
    add.addEventListener("click", () => {
      const picked = boxes.filter((b) => b.checked);
      // THE WHOLE IMPORT OR NONE OF IT, the composer's attachment rule: a
      // picker that quietly kept the first few would leave a person
      // believing every checked prompt became a task.
      if (rows.length + picked.length > BUILDER_MAX_ROWS) {
        said(
          "the builder holds " +
            BUILDER_MAX_ROWS +
            " rows; " +
            rows.length +
            " are here and " +
            picked.length +
            " were checked. Nothing was added. Check fewer, or paste JSONL.",
          "refused",
        );
        return;
      }
      const first = rows.length;
      // LINE BREAKS AS THE BOX HOLDS THEM. A textarea turns CRLF and a
      // lone CR into LF, so a saved prompt with CR in it would be sent
      // with them while the box showed, and any edit sent, something
      // else. The row is what its box shows; otherwise the prompt is
      // taken exactly as saved.
      addRows(
        picked.map((b) =>
          Object.assign(blankRow(), {
            prompt: b.dataset.text.replace(/\r\n?/g, "\n"),
          }),
        ),
        { index: first, testid: "dataset-row-id" },
      );
      closePromptPicker();
      said(
        "added " +
          picked.length +
          (picked.length === 1 ? " task" : " tasks") +
          " from the prompt library; each needs an id before Store.",
        "",
      );
    });
    promptPickerEl.append(add);
  }

  function closePromptPicker() {
    promptPickerOpens += 1;
    promptPickerEl.hidden = true;
    fromPromptsEl.setAttribute("aria-expanded", "false");
    promptPickerEl.replaceChildren();
  }

  // ---- The JSONL way in.

  function setMode(value) {
    mode = value === "jsonl" ? "jsonl" : "rows";
    for (const b of modeEl.querySelectorAll("button")) {
      b.setAttribute("aria-pressed", String(b.dataset.value === mode));
    }
    rowsPaneEl.hidden = mode !== "rows";
    jsonlPaneEl.hidden = mode !== "jsonl";
    refresh();
  }

  // The box comes back holding what was typed before the file, because
  // the file never replaced it; it was only hidden.
  function clearFile() {
    loadedFile = null;
    jsonlEl.hidden = false;
    fileLabelEl.textContent = "";
    fileClearEl.hidden = true;
    refresh();
    // clear hides itself, which would drop focus to <body>; the box it
    // brought back is where the person goes next.
    jsonlEl.focus();
  }

  // A file, decoded exactly. FATAL, so bytes that are not UTF-8 are said
  // to be so rather than silently replaced with U+FFFD, which would store
  // a dataset the file never held. THE MARK IS KEPT, so a file beginning
  // with a byte order mark is sent with it and the server refuses it in
  // the same words it refuses the same file by path.
  //
  // THE BOX IS HIDDEN WHILE A FILE IS LOADED. Store sends the file, and a
  // box still showing text typed before it would show one dataset while
  // another was stored.
  async function loadFile(file) {
    if (file.size > DATASET_LIMITS.maxDatasetBytes) {
      said(
        file.name +
          " is " +
          file.size +
          " bytes, over the " +
          DATASET_LIMITS.maxDatasetBytes +
          " byte limit for a stored dataset. A larger dataset goes by " +
          "path: name its file with dataset_path when the experiment is " +
          "created.",
        "refused",
      );
      return;
    }
    let text;
    try {
      const bytes = await file.arrayBuffer();
      text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(
        bytes,
      );
    } catch (err) {
      said(
        file.name +
          " is not valid UTF-8, so it has no text to send. The bench reads " +
          "datasets as UTF-8, by path as well as here.",
        "refused",
      );
      return;
    }
    loadedFile = { name: file.name, text: text };
    jsonlEl.hidden = true;
    fileLabelEl.textContent =
      file.name +
      " · " +
      countLines(text) +
      " lines · " +
      utf8Length(text) +
      " bytes, sent exactly as read";
    fileClearEl.hidden = false;
    said("", "");
    refresh();
  }

  // ---- Store.

  // The panel's answer line. state is "refused" for a refusal, the
  // server's or the page's own ("nothing was added", "not valid UTF-8"),
  // so it reads as one; and "" otherwise, including an answer the page
  // could not read or never got, whose outcome it leaves to the list
  // below.
  function said(text, state) {
    messageIsRefusal = false;
    msgEl.textContent = text;
    msgEl.dataset.state = state;
  }

  async function store() {
    const sent = content();
    if (storing || nudge(sent) !== null) return;
    const name = nameEl.value;
    const sentMode = mode;
    const hadFocus = document.activeElement === storeEl;
    storing = true;
    refusal = null;
    said("storing", "");
    refresh();
    try {
      const resp = await fetch("/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, content: sent }),
      });
      let body = null;
      try {
        body = await resp.json();
      } catch (err) {
        body = null;
      }
      if (!resp.ok) {
        const text = body
          ? refusalText(
              body.detail,
              "the dataset was refused and the reason could not be read",
            )
          : "the dataset was refused (HTTP " + resp.status + ")";
        refusal = {
          line: refusalLine(text),
          text: text,
          sent: sent,
          mode: sentMode,
        };
        // Changed while the request was out: the sentence is still the
        // server's and still printed, but it names a line of text that
        // is no longer on screen, so no row is marked and the message
        // says what it was about, for as long as that is so.
        said("", "refused");
        messageIsRefusal = true;
        setText(msgEl, refusalMessage(refusalStands(content())));
        return;
      }
      if (body === null) {
        said(
          "the bench answered " +
            resp.status +
            " with a body this page could not read; the dataset may be " +
            "stored, and the list below will say.",
          "",
        );
        return;
      }
      // The stored name can differ from the one just typed, because
      // identical content already here keeps the name it arrived under.
      // Saying so is cheaper than letting somebody wonder where theirs
      // went.
      said(
        body.name === name
          ? "stored as " +
              body.name +
              " · sha256 " +
              shortDigest(body.digest) +
              " · " +
              body.task_count +
              (body.task_count === 1 ? " task" : " tasks")
          : "these tasks were already stored as " +
              body.name +
              ", so the earlier name stands · sha256 " +
              shortDigest(body.digest),
        "",
      );
    } catch (err) {
      // No answer came back. The request may never have left, or the
      // server may have stored the dataset and only the answer was lost,
      // so this is not a refusal and does not claim either.
      console.error("bench: storing a dataset failed", err);
      said(
        "no answer came back (" +
          err.message +
          "); the dataset may be stored, and the list below will say.",
        "",
      );
    } finally {
      storing = false;
      refresh();
      // Store was disabled for the request, which took focus from it;
      // it goes back to Store, where a keyboard user left it.
      if (hadFocus && document.activeElement === document.body) {
        storeEl.focus();
      }
      void loadLibrary();
    }
  }

  // ---- The library of stored datasets.

  function setLibraryState(state, message) {
    libraryEl.dataset.state = state;
    libraryEl.textContent = message;
  }

  // The selection line: which stored dataset the experiment form below
  // will create over.
  function renderSelected() {
    const chosen = library.find((d) => d.digest === selectedDigest);
    setText(
      selectedEl,
      chosen
        ? "selected for a new experiment: " +
            chosen.name +
            " · sha256 " +
            shortDigest(chosen.digest)
        : "",
    );
    for (const el of libraryEl.querySelectorAll(
      "[data-testid=dataset-entry]",
    )) {
      el.setAttribute(
        "aria-pressed",
        String(el.dataset.digest === selectedDigest),
      );
    }
  }

  async function loadLibrary() {
    if (libraryController !== null) libraryController.abort();
    const controller = new AbortController();
    libraryController = controller;
    const version = ++libraryVersion;
    setLibraryState("loading", "loading stored datasets");
    let data;
    try {
      const resp = await fetch("/datasets?limit=" + LIST_LIMIT, {
        signal: controller.signal,
      });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch (err) {
      if (controller.signal.aborted || version !== libraryVersion) return;
      console.error("bench: stored datasets failed to load", err);
      setLibraryState(
        "error",
        "failed to load stored datasets: " + err.message,
      );
      return;
    }
    if (version !== libraryVersion) return;
    library = data.datasets;
    // A selection whose dataset is no longer listed is dropped rather
    // than kept as a digest nothing on screen describes; the note below
    // a full list says why one can fall off it.
    if (!library.some((d) => d.digest === selectedDigest)) {
      if (selectedDigest !== null) {
        selectedDigest = null;
        selectionChanged();
      }
    }
    if (library.length === 0) {
      setLibraryState("empty", "no stored datasets yet");
      renderSelected();
      return;
    }
    setLibraryState("ready", "");
    for (const dataset of library) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "hrow ds-entry";
      row.dataset.testid = "dataset-entry";
      row.dataset.digest = dataset.digest;
      row.title = dataset.name + "\nsha256 " + dataset.digest;
      const time = document.createElement("span");
      time.className = "htime";
      time.textContent =
        dataset.created_at.slice(0, 19).replace("T", " ") + " UTC";
      const name = document.createElement("span");
      name.className = "hprompt";
      name.dataset.testid = "dataset-entry-name";
      name.textContent = dataset.name;
      const meta = document.createElement("span");
      meta.className = "hcount";
      meta.dataset.testid = "dataset-entry-meta";
      const bits = [
        dataset.task_count + (dataset.task_count === 1 ? " task" : " tasks"),
      ];
      if (dataset.scorers.length > 0) bits.push(dataset.scorers.join(", "));
      if (dataset.cites_documents) bits.push("cites documents");
      bits.push("sha256 " + shortDigest(dataset.digest));
      meta.textContent = bits.join(" · ");
      row.append(time, name, meta);
      // SELECTING LOADS NOTHING INTO THE BUILDER. A stored dataset is a
      // record; editing it would be storing a new one, and the builder is
      // where that starts from what a person types.
      row.addEventListener("click", () => {
        selectedDigest =
          selectedDigest === dataset.digest ? null : dataset.digest;
        renderSelected();
        selectionChanged();
      });
      libraryEl.append(row);
    }
    if (library.length >= LIST_LIMIT) {
      const note = document.createElement("div");
      note.className = "ds-library-note";
      note.dataset.testid = "dataset-library-note";
      note.textContent =
        "The newest " +
        LIST_LIMIT +
        " are listed; older ones stay stored, and an experiment names " +
        "one by its digest.";
      libraryEl.append(note);
    }
    renderSelected();
  }

  function init() {
    detailsEl.addEventListener("click", (event) => {
      // The synchronous claim the history and experiments panels make:
      // toggle is dispatched asynchronously, so without this the list
      // still reads its last terminal state when the click lands.
      if (!detailsEl.open && event.target.closest("summary")) {
        setLibraryState("loading", "loading stored datasets");
      }
    });
    detailsEl.addEventListener("toggle", () => {
      if (detailsEl.open) void loadLibrary();
    });
    for (const b of modeEl.querySelectorAll("button")) {
      b.addEventListener("click", () => setMode(b.dataset.value));
    }
    nameEl.addEventListener("input", refresh);
    jsonlEl.addEventListener("input", refresh);
    // A new row takes focus in its id box, as imported rows do: it is
    // the next thing to type, and at the bound + Task is disabled under
    // the focus it would otherwise keep.
    addRowEl.addEventListener("click", () => {
      if (rows.length >= BUILDER_MAX_ROWS) return;
      addRows([blankRow()], { index: rows.length, testid: "dataset-row-id" });
    });
    fromPromptsEl.addEventListener("click", () => {
      const opening = promptPickerEl.hidden;
      if (!opening) {
        closePromptPicker();
        return;
      }
      promptPickerEl.hidden = false;
      fromPromptsEl.setAttribute("aria-expanded", "true");
      void openPromptPicker();
    });
    fileEl.addEventListener("change", () => {
      // Copied before the input is cleared, for attach.js's reason: the
      // FileList is live and clearing the input empties it.
      const file = fileEl.files[0];
      fileEl.value = "";
      if (file) void loadFile(file);
    });
    fileClearEl.addEventListener("click", clearFile);
    storeEl.addEventListener("click", () => {
      void store();
    });
    renderRows();
  }

  // Everything that wants to know when the selection changed: the
  // experiment form reads it for the dataset it creates over, the scorer
  // kinds its primary metric offers, and whether an attachments mode can
  // be chosen at all.
  const selectionListeners = [];
  function selectionChanged() {
    for (const listener of selectionListeners) listener();
  }

  // selected() is what the experiment form creates over: the stored
  // dataset chosen here, as the server listed it, or null. Create sends
  // its digest; Start sends the digest the experiment recorded, which is
  // the same one.
  window.BenchDatasets = {
    init,
    selected: () => library.find((d) => d.digest === selectedDigest) || null,
    onSelect: (listener) => selectionListeners.push(listener),
    refresh: loadLibrary,
  };
})();
