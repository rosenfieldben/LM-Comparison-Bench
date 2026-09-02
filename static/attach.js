// Documents attached to the next comparison. Exposed on window.BenchAttach.
//
// THE FAIRNESS LAW IS THE WHOLE POINT of this control: whatever is
// staged here reaches every model in the comparison identically. The
// mode decides how, and the two modes measure different things, which is
// why the choice sits beside the files rather than three panels away:
//
//   inline  the bench extracts the text and composes it into the prompt.
//           Every model reads the bench's reading of the document, good
//           or bad, and every model can participate.
//   native  the file goes to the provider as an image content part and
//           the model does its own reading. Sharper question, narrower
//           population: a text-only model cannot participate at all, so
//           the server refuses the comparison at creation rather than at
//           the first paid call.
//
// Staged state is per session and per tab, like the lineup's checked set
// and unlike the lineup itself: an attachment is a statement about the
// comparison being composed right now, and a file silently still
// attached after a reload would be a document reaching models nobody
// meant to send it to.
//
// The STORED bytes outlive the tab, in bench.db, keyed by digest. That
// is what makes reuse and replay able to name a document at all, and it
// is the reason the note under this control says where the file went.
(function () {
  const { fmtBytes, shortDigest, approxTokens } = window.BenchLib;

  const rowEl = document.getElementById("attach-row");
  const listEl = document.getElementById("attachments");
  const inputEl = document.getElementById("attach-input");
  const modeEl = document.getElementById("attach-mode");
  const msgEl = document.getElementById("attach-msg");
  const noteEl = document.getElementById("attach-note");
  const snapOpenEl = document.getElementById("snapshot-open");
  const snapPanelEl = document.getElementById("snapshot-panel");
  const snapRootEl = document.getElementById("snapshot-root");
  const snapPatternsEl = document.getElementById("snapshot-patterns");
  const snapComposeEl = document.getElementById("snapshot-compose");
  const snapMsgEl = document.getElementById("snapshot-msg");

  // Mirrors MAX_ATTACHMENTS in bench/main.py. A client-side cap is a
  // convenience over the server's and never an authority: the refusal
  // that matters is the 422, and this one exists so the person learns
  // before the upload rather than after it. A test asserts the pair
  // agrees, because the drift that hurts is a browser cap TIGHTER than
  // the server's, which would refuse a legal comparison and blame the
  // user's file.
  const MAX_ATTACHMENTS = 4;

  // Mirrors MAX_ATTACHMENT_BYTES in bench/main.py, same standing: a
  // courtesy check so an 80 MiB video is refused without being base64'd
  // and pushed across a socket first.
  const MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024;

  // Whether native mode can send this staged document.
  //
  // THE KIND, WHICH IS DECLARED, and not the filename's suffix, which is
  // not. Every staged entry carries the rendition the server recorded,
  // including whether these bytes were READ as an image, and the suffix
  // is at best a second copy of that answer: the same bytes uploaded as
  // .txt and as .png are two renditions with two kinds and one set of
  // bytes, so a suffix test asks the wrong question and then answers it
  // from whichever name happens to be on the entry.
  //
  // It was measured wrong in exactly that way. Reusing a comparison that
  // pinned the IMAGE rendition restaged a ref whose filename was
  // "shot.txt", so this returned "native mode takes images only" and Run
  // was disabled on a correctly pinned native comparison.
  //
  // Still only an explanation of a refusal the server owns; see
  // enforce_native_mode, which checks the pin's kind server-side.
  const IMAGE_KIND = "image";
  const SNAPSHOT_KIND = "snapshot";
  // What a snapshot chip says on its face, INSTEAD of the stored name.
  // The stored name is snapshot-<twelve hex>.txt, a pure function of the
  // digest, so showing it would print the digest twice and imply a file
  // that never existed. The blind view's rule is satisfied either way:
  // no path can appear in a name the server derives from a digest.
  const SNAPSHOT_LABEL = "repository snapshot";

  function usableInNative(doc) {
    return doc.kind === IMAGE_KIND;
  }

  // What to call one staged document in a sentence. The stored name for
  // everything but a snapshot, and the snapshot's label for a snapshot,
  // so a refusal written here names the same thing the chip beside it
  // does. Written once because the alternative is a message that calls a
  // document one thing and its own chip another.
  function docLabel(doc) {
    return doc.kind === SNAPSHOT_KIND ? SNAPSHOT_LABEL : doc.filename;
  }

  // One line saying which walk a snapshot was, for a title or a chip
  // bit, or "" for anything without a capture. The commit is shortened
  // to the seven characters git itself shows; "dirty" or "clean" is the
  // tree's state at the walk, and "unknown" when the bench could read
  // the commit but not the status.
  function captureLine(capture) {
    if (!capture || !Number.isInteger(capture.id)) return "";
    const head =
      typeof capture.head === "string" ? capture.head.slice(0, 7) : "no commit";
    const state =
      capture.dirty === true
        ? "dirty"
        : capture.dirty === false
          ? "clean"
          : "unknown";
    return "capture #" + capture.id + " at " + head + ", " + state;
  }

  // Staged documents, in attachment order, which IS the order they are
  // composed into the prompt. Array and not a Set keyed by digest,
  // because order is part of the declaration.
  let staged = [];

  // How many files are mid-flight: picked, not yet stored or refused.
  //
  // A COUNTER, NOT A BOOLEAN, because selections overlap. A person can
  // pick two files, then pick two more while the first pair is still
  // uploading, and a boolean would clear on the first batch's completion
  // while the second was still in the air. The cap below counts this
  // too, so four files cannot slip through as two overlapping pairs.
  let inFlight = 0;

  // The view epoch this staging belongs to, stamped when a batch starts.
  //
  // AN UPLOAD OUTLIVES THE VIEW THAT STARTED IT. A person can attach a
  // large file, get bored, open a comparison from history, and have the
  // upload land afterwards: the staged set would gain a document the
  // loaded view never declared, and the composer would then be
  // describing neither the history entry on screen nor anything the
  // person chose. Every view takeover advances the epoch already, so a
  // batch that recorded its epoch can tell whether the view it was
  // staging for is still the one on screen.
  // WHICH STAGING SET THE UPLOADS IN FLIGHT BELONG TO.
  //
  // A SECOND EPOCH, beside BenchState.viewEpoch, because the two guard
  // different things and the view's own is blind to this one. Opening a
  // history entry or starting a run advances the VIEW epoch, so an
  // upload in flight is correctly discarded. REUSE does not: it replaces
  // the staged set from a stored comparison without leaving the
  // composer, so the view epoch never moves, and an upload that was
  // still reading when the person clicked reuse appended its document
  // onto the reused experiment's declaration and could be sent with it.
  //
  // This counter moves on every event that replaces what is staged, so
  // "the set I started uploading into" is a thing an in-flight batch can
  // check. It was previously assigned in addFiles and read nowhere,
  // which is why it did not catch this.
  let stagingEpoch = 0;

  const A = {
    // Read by the stream client when it builds the group POST and every
    // member request; written by the reuse action in history.
    declared,
    setFrom,
    clear,
    staged: () => staged.slice(),
    MAX_ATTACHMENTS,
    MAX_ATTACHMENT_BYTES,
    init,
  };

  // Whether a file is still being read or uploaded.
  function busy() {
    return inFlight > 0;
  }

  // The four facts that identify one READING of one document. Null when
  // a staged entry does not carry all four, which is a reuse ref whose
  // row is gone; Run is already blocked while one of those is present.
  //
  // typeof rather than truthiness, because an image's extractor_version
  // is the string "0" and a falsy test would drop exactly the rendition
  // native mode needs.
  function rendition(doc) {
    const parts = ["digest", "extractor", "extractor_version", "kind"];
    if (
      !parts.every((key) => typeof doc[key] === "string" && doc[key] !== "")
    ) {
      return null;
    }
    const pin = {
      digest: doc.digest,
      extractor: doc.extractor,
      extractor_version: doc.extractor_version,
      kind: doc.kind,
    };
    // THE CAPTURE RIDES ALONG, the fourteenth review's H2. A snapshot
    // staged from POST /snapshots or restaged from a stored comparison
    // carries the walk it was, and the pin names it by id so the record
    // says which commit and which tree state the models read from. Sent
    // only when known: a document has none, and a snapshot stored
    // before captures existed has none to name, and the server resolves
    // a bare snapshot pin to its latest capture rather than this page
    // guessing one.
    if (doc.capture && Number.isInteger(doc.capture.id)) {
      pin.capture_id = doc.capture.id;
    }
    return pin;
  }

  // What goes on the wire. RULE ONE at the client edge: with nothing
  // staged this returns an empty object, so a comparison with no
  // document sends exactly the body it sent before this control existed.
  // A payload that gained an "attachments": [] key the day the feature
  // shipped would make every earlier comparison incomparable with every
  // later one, and the server would refuse the mode besides.
  //
  // THE RENDITIONS RIDE ALONG, and until K.3 they did not. The upload
  // response has carried the whole rendition since K.1 and this kept
  // only the digest, so the browser made a WEAKER declaration than a
  // hand-written API body: the server fell back to resolving each digest
  // to its base row's reading, which belongs to whichever upload of
  // those bytes arrived first. Staging the same PNG as .txt and then as
  // .png and choosing native therefore ran the TEXT rendition, because
  // the base row was the .txt one. The pin the person chose in the
  // picker never left the page.
  //
  // Omitted entirely, rather than sent partially, when any staged entry
  // is missing its rendition: the server treats absent renditions as
  // "resolve each digest yourself", which is the honest request to make
  // when the page does not know, and a partial list would be a
  // declaration that disagreed with its own digest list.
  function declared() {
    if (staged.length === 0) return {};
    const body = {
      attachments: staged.map((d) => d.digest),
      attachments_mode: modeEl.value,
    };
    const pins = staged.map(rendition);
    if (pins.every((pin) => pin !== null)) body.renditions = pins;
    return body;
  }

  // Prefill from a stored comparison, for the reuse action. Refs come
  // from the API, so a document whose row is gone arrives with a digest
  // and nothing else; it is kept rather than dropped, because the
  // comparison did declare it and a reuse that quietly sent three of
  // four documents would be a different experiment wearing the old
  // label. The chip says so and Run is blocked while one is present.
  function setFrom(refs, mode) {
    // The staged set is being REPLACED, so anything still uploading into
    // the old one belongs to a declaration that no longer exists. Its
    // bytes are stored either way, which is the point of
    // content-addressing: nothing is lost and the person can attach them
    // again to the experiment they are now composing.
    stagingEpoch += 1;
    staged = Array.isArray(refs) ? refs.slice() : [];
    // Absent mode CLEARS to inline rather than leaving the last choice
    // standing, the same rule setExperimentParams follows: what the
    // source did not declare, the composer must not declare either.
    modeEl.value = mode === "native" ? "native" : "inline";
    msgEl.textContent = "";
    // The snapshot control's last answer described a staging set that no
    // longer exists, so it stops outranking the blocker, and a root
    // typed for that set is forgotten with it.
    snapMsgEl.dataset.said = "";
    forgetSnapshot();
    render();
  }

  function clear() {
    stagingEpoch += 1;
    staged = [];
    modeEl.value = "inline";
    msgEl.textContent = "";
    snapMsgEl.dataset.said = "";
    forgetSnapshot();
    render();
  }

  // A staged ref the bench can no longer resolve. Reuse can produce one;
  // an upload never can, since it just stored the row.
  function isMissing(doc) {
    return doc.filename === null || doc.filename === undefined;
  }

  function chipFor(doc, index) {
    const chip = document.createElement("span");
    // Deliberately NOT the lineup's .chip: that class carries a
    // checked/unchecked state (.on dims everything without it) and an
    // attachment has no such state. Borrowing it would have rendered
    // every attached document permanently greyed out.
    chip.className = "attach-chip";
    chip.dataset.testid = "attachment-chip";
    chip.dataset.digest = doc.digest;

    const name = document.createElement("span");
    name.className = "attach-name";
    name.dataset.testid = "attachment-name";
    // textContent, never innerHTML: a filename is user-supplied text and
    // gets the same treatment model output does.
    name.textContent = isMissing(doc) ? "(no longer stored)" : docLabel(doc);

    const meta = document.createElement("span");
    meta.className = "attach-meta";
    meta.dataset.testid = "attachment-meta";
    // Size, short digest, and the estimate, in that order: the first two
    // identify the document and the third describes what it will cost.
    const bits = [];
    if (Number.isFinite(doc.byte_size)) bits.push(fmtBytes(doc.byte_size));
    // How many files the tree contributed, which is the one fact about a
    // snapshot a person actually wants on the chip and the one thing its
    // byte count does not say. Absent for anything that is not a
    // snapshot and for a snapshot whose manifest did not travel.
    const members = doc.manifest?.files;
    if (Array.isArray(members)) {
      bits.push(members.length + (members.length === 1 ? " file" : " files"));
    }
    bits.push("sha256 " + shortDigest(doc.digest));
    const estimate = estimateFor(doc);
    if (estimate) bits.push(estimate);
    meta.textContent = bits.join(" · ");

    chip.append(name, meta);

    if (isMissing(doc)) {
      chip.classList.add("attach-missing");
      chip.title =
        "this comparison declared sha256 " +
        doc.digest +
        ", but no such document is stored now. Remove it or attach the " +
        "file again; the bench will not send a comparison that cites a " +
        "document it cannot read";
    } else {
      chip.title =
        docLabel(doc) +
        "\nsha256 " +
        doc.digest +
        "\nread by " +
        doc.extractor +
        " " +
        doc.extractor_version +
        (captureLine(doc.capture) ? "\n" + captureLine(doc.capture) : "");
    }

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chip-x";
    remove.dataset.testid = "attachment-remove";
    remove.textContent = "×";
    remove.setAttribute(
      "aria-label",
      "remove " + (isMissing(doc) ? doc.digest : docLabel(doc)),
    );
    remove.addEventListener("click", () => {
      staged.splice(index, 1);
      msgEl.textContent = "";
      render();
    });
    chip.append(remove);
    return chip;
  }

  // The labeled-approximate token estimate, and what it will not claim.
  //
  // Inline is estimable because the bench knows exactly how many
  // characters it will compose; the number is still marked with a tilde
  // and the word "approx" because characters-over-four is a heuristic
  // every model's real tokenizer disagrees with.
  //
  // Native is NOT estimable and says so instead of guessing. An image's
  // token cost depends on the provider's tiling, the model's resolution
  // handling, and in some cases a detail parameter this bench does not
  // send; there is no arithmetic here that would produce an honest
  // number, and a fabricated one beside a real byte count would be
  // believed. Rule three, one control over: never guess.
  function estimateFor(doc) {
    if (modeEl.value === "native") return "image tokens set by the provider";
    if (!Number.isFinite(doc.extracted_chars)) return "";
    return "approx ~" + approxTokens(doc.extracted_chars) + " tokens";
  }

  // What the session's routing policy means for a document, stated at
  // the control that sends one. Surfaced HERE and not only on the header
  // badge, because the badge is a session-wide marker a person reads
  // once and this is the moment they are deciding whether to hand the
  // bench a contract.
  function policyLine() {
    const policy = window.BenchState.dataPolicy;
    if (policy === "zdr") {
      return (
        "Session routing: zero-retention. Every request asks OpenRouter " +
        "for zero-data-retention endpoints and providers that do not " +
        "collect user data. The guarantee is OpenRouter's, not this " +
        "application's."
      );
    }
    if (policy === "deny") {
      return (
        "Session routing: no-training. Every request asks OpenRouter to " +
        "route only to providers that do not collect user data. The " +
        "guarantee is OpenRouter's, not this application's."
      );
    }
    // The default is the case worth spelling out, because it is the one
    // with no badge: an absent badge means default routing, and silence
    // is a poor way to tell somebody their contract is going to a
    // provider under ordinary terms.
    return (
      "Session routing: default. Attached documents are sent to " +
      "OpenRouter and on to each provider under their ordinary terms. " +
      "Set BENCH_DATA_POLICY to restrict routing."
    );
  }

  function render() {
    listEl.replaceChildren();
    staged.forEach((doc, index) => listEl.append(chipFor(doc, index)));
    // The row stays visible with nothing attached: it is how a person
    // discovers that attaching is possible.
    const wrongForNative = staged.filter(
      (d) => !isMissing(d) && !usableInNative(d),
    );
    const notes = [];
    if (staged.length === 0) {
      notes.push(
        "No documents. Attach up to " +
          MAX_ATTACHMENTS +
          "; each one reaches every model in the comparison identically.",
      );
    } else if (modeEl.value === "native") {
      notes.push(
        "Native: each image goes to the provider as a content part and " +
          "the model reads it directly. Only models that accept image " +
          "input can be in the lineup, so the comparison is refused at " +
          "creation if one cannot.",
      );
    } else {
      notes.push(
        "Inline: the bench extracts the text and composes it into the " +
          "prompt, so every model reads the same extraction. This is " +
          "what is being compared.",
      );
    }
    notes.push(
      "Stored once in bench.db, keyed by content digest. Nothing is " +
        "written outside it, and deleting bench.db deletes the documents " +
        "with everything else.",
    );
    notes.push(policyLine());
    noteEl.replaceChildren();
    for (const text of notes) {
      const line = document.createElement("div");
      line.className = "attach-note-line";
      line.textContent = text;
      noteEl.append(line);
    }
    // A refusal the server would issue, said here first. Not a
    // replacement for the server's check: the comparison is still
    // refused at creation with the model and the modality named. This
    // one exists so the person is not told at Run time about a file
    // choice they made a minute earlier.
    if (wrongForNative.length > 0 && modeEl.value === "native") {
      msgEl.textContent =
        "native mode sends images as content parts, and " +
        wrongForNative.map(docLabel).join(", ") +
        " is not one. Switch to inline, which extracts the text and gives " +
        "every model the same reading of it.";
    }
    renderSnapshotControl();
    rowEl.dataset.count = String(staged.length);
    window.BenchControls.updateRunState();
  }

  // Whether the staged set blocks a run, and why. Read by
  // BenchControls.updateRunState, which owns the Run button: a second
  // place that disabled it would be a second place that could forget to
  // re-enable it.
  function blockingReason() {
    if (busy()) {
      // RUN IS BLOCKED WHILE ANYTHING IS IN FLIGHT, and this is the
      // whole of the busy state. Without it a person could pick three
      // files and press Run the instant the first chip appeared: the
      // declaration would carry one document, the comparison would be
      // created and pinned over that one, and the other two would land
      // in the staging area afterwards looking exactly as attached as
      // the one that was actually sent.
      return inFlight === 1
        ? "a file is still being read"
        : "files are still being read";
    }
    if (staged.some(isMissing)) {
      return "a declared document is no longer stored";
    }
    if (
      modeEl.value === "native" &&
      staged.some((d) => !isMissing(d) && !usableInNative(d))
    ) {
      return "native mode takes images only";
    }
    return null;
  }

  // Bytes to a binary string, in chunks, and then ONE btoa over the
  // whole thing.
  //
  // The chunking exists for exactly one reason: String.fromCharCode.apply
  // over an eight-megabyte array overflows the argument stack in every
  // engine. 0x8000 is comfortably under every engine's limit and has no
  // other constraint on it.
  //
  // WHAT THE CHUNK SIZE DOES NOT NEED TO BE is a multiple of three. The
  // comment here used to claim that base64's three-byte quantum made
  // the concatenation exact and that 0x8000 is divisible by three.
  // Neither half was true: 0x8000 % 3 is 2, and alignment cannot matter
  // because the chunks are joined into one binary string and encoded
  // once at the end. The claim was load-bearing in the worst way, since
  // it told a reader that per-chunk encoding would be safe. It would
  // not: encoding each chunk separately emits padding mid-string and
  // produces a body whose digest is not the file's. If this is ever
  // changed to encode incrementally, the chunk size MUST become a
  // multiple of three, and that is the constraint to write down then.
  const B64_CHUNK = 0x8000;

  function toBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i += B64_CHUNK) {
      binary += String.fromCharCode.apply(
        null,
        bytes.subarray(i, i + B64_CHUNK),
      );
    }
    return btoa(binary);
  }

  async function upload(file) {
    // Both caps checked before the read, so an oversized file is never
    // loaded into memory or encoded. The message quotes the arithmetic
    // for the same reason the server's does: "too large" alone leaves
    // the person guessing whether they are over by a byte.
    if (file.size > MAX_ATTACHMENT_BYTES) {
      throw new Error(
        file.name +
          " is " +
          fmtBytes(file.size) +
          ", over the " +
          fmtBytes(MAX_ATTACHMENT_BYTES) +
          " limit. Attach the relevant section instead, or paste its " +
          "text into the prompt.",
      );
    }
    if (file.size === 0) {
      throw new Error(file.name + " is empty, so there is nothing to attach.");
    }
    const encoded = toBase64(await file.arrayBuffer());
    // JSON and not multipart, matching the server: a multipart POST is a
    // CORS simple request a hostile page can fire at localhost with no
    // preflight. See AttachmentCreate for the full argument.
    const resp = await fetch("/attachments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, content_base64: encoded }),
    });
    if (!resp.ok) {
      let detail = "HTTP " + resp.status;
      try {
        const body = await resp.json();
        if (body && typeof body.detail === "string") detail = body.detail;
      } catch (err) {
        // A non-JSON error body leaves the status line as the best we
        // have; the throw below still surfaces something readable.
      }
      throw new Error(detail);
    }
    return resp.json();
  }

  async function addFiles(files) {
    const picked = [...files];
    if (picked.length === 0) return;
    // THE CAP REFUSAL SURFACES IN THE COMPOSER, which is the whole
    // requirement here: a picker that silently kept the first two of
    // five files would leave the person believing they attached five.
    // Nothing is uploaded when the batch would not fit, so the staged
    // set is either what they asked for or unchanged.
    // THE CAP COUNTS WHAT IS IN FLIGHT, not only what is staged. Two
    // overlapping selections of two files each are four documents, and
    // counting only the staged set let the second pair through while the
    // first was still uploading: the server's own bound then refused the
    // whole comparison at Run time, long after the person had stopped
    // choosing files.
    if (staged.length + inFlight + picked.length > MAX_ATTACHMENTS) {
      msgEl.textContent =
        "at most " +
        MAX_ATTACHMENTS +
        " documents per comparison; " +
        staged.length +
        " attached" +
        (inFlight > 0 ? ", " + inFlight + " still reading" : "") +
        " and " +
        picked.length +
        " more picked. Nothing was attached. Remove one, or attach fewer.";
      return;
    }
    // Claimed BEFORE the first await, so a second pick arriving while
    // this one is in flight sees the true total. Everything from here to
    // the finally is what the counter is protecting.
    inFlight += picked.length;
    const epoch = window.BenchState.viewEpoch;
    const staging = stagingEpoch;
    msgEl.textContent =
      picked.length === 1
        ? "reading " + picked[0].name
        : "reading " + picked.length + " files";
    // Run is disabled from here, not from the first chip.
    window.BenchControls.updateRunState();
    const added = [];
    for (const file of picked) {
      let stored;
      try {
        stored = await upload(file);
      } catch (err) {
        // The first failure stops the batch, and what did succeed stays
        // staged: those documents are stored and cited by digest, so
        // discarding them here would be discarding work that already
        // happened. The message says both halves.
        inFlight -= picked.length;
        if (stale(epoch, staging)) return;
        staged = staged.concat(added);
        render();
        msgEl.textContent =
          added.length > 0
            ? err.message + " (" + added.length + " attached before this)"
            : err.message;
        return;
      }
      added.push(stored);
    }
    inFlight -= picked.length;
    // THE LATE-UPLOAD GUARD. If the view moved on while these files were
    // reading (a history entry was opened, a run was started), this
    // batch belongs to a composer nobody is looking at, and adding its
    // documents to the staging area would attach files to whatever is on
    // screen now. The bytes are stored either way, which is the point of
    // content-addressing: nothing is lost, and the person can attach
    // them again in the view they meant.
    if (stale(epoch, staging)) {
      window.BenchControls.updateRunState();
      return;
    }
    staged = staged.concat(added);
    // The stored filename can differ from the one just picked, because
    // identical bytes already here keep the name they arrived under.
    // Saying so is cheaper than letting somebody wonder why their file
    // renamed itself.
    const renamed = added.filter((doc, i) => doc.filename !== picked[i].name);
    msgEl.textContent =
      renamed.length > 0
        ? "attached; " +
          renamed.map((d) => d.filename).join(", ") +
          " was already stored under that name, so the earlier name stands"
        : "attached " +
          added.length +
          (added.length === 1 ? " document" : " documents");
    // RENDER LAST, so a refusal outranks a success. render() writes the
    // native-mode warning into this same line, and doing it before the
    // message above meant the warning was overwritten one statement
    // later: a person already in native mode who attached a PDF saw
    // "attached 1 document", a Run button that would not press, and the
    // reason only in a hover title no keyboard user ever sees. A
    // successful upload and an unusable staged set are both true, and
    // the one that blocks the run is the one worth the line.
    render();
  }

  // ---- The snapshot control.
  // ----
  // ---- A repository snapshot is an attachment the bench COMPOSES
  // ---- rather than one the person uploads, so it produces the same
  // ---- chip, rides the same declaration and is bounded by the same
  // ---- MAX_ATTACHMENTS. Two things are different and both are visible
  // ---- here: it can be switched off at the server, and its refusals
  // ---- are about a root and a pattern rather than about a file.

  // A refusal body as text. FastAPI answers a model violation with a
  // LIST of error objects and this application's own refusals with a
  // string, and a control that assigned the list to textContent printed
  // "[object Object]" at the person: the server said exactly what was
  // wrong and the page threw it away.
  function refusalText(detail) {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => (item && typeof item.msg === "string" ? item.msg : ""))
        .filter((msg) => msg !== "");
      if (messages.length > 0) return messages.join("; ");
    }
    return "the snapshot was refused and the reason could not be read";
  }

  // Why the control cannot be used, or "" when it can.
  //
  // THREE DIFFERENT FACTS, and they are three sentences rather than one
  // because they call for three different actions. A catalog that has
  // not answered is not evidence that the feature is off; a feature that
  // IS off is the server's own sentence, verbatim, never a second
  // wording written here; a full staging set is the composer's own
  // bound and has nothing to do with either.
  function snapshotBlocker() {
    const posture = window.BenchState.snapshots;
    if (posture === null || posture === undefined) {
      return (
        "The model catalog has not answered yet, so the bench has not " +
        "said whether repository snapshots are configured."
      );
    }
    if (!posture.enabled) {
      // THE EMPTY STRING IS THIS FUNCTION'S "NOT BLOCKED", so a reason
      // that arrived empty would switch the control back on for a bench
      // that had just said it is off. The server never sends one, and a
      // sentinel doing double duty is exactly the shape that gets one
      // sent eventually.
      return (
        posture.reason ||
        "Repository snapshots are not configured on this bench."
      );
    }
    if (staged.length >= MAX_ATTACHMENTS) {
      return (
        "Already at " +
        MAX_ATTACHMENTS +
        " documents, which is the whole bound: a snapshot is one of them."
      );
    }
    return "";
  }

  function closeSnapshotPanel() {
    snapPanelEl.hidden = true;
    snapOpenEl.setAttribute("aria-expanded", "false");
  }

  // Close the panel AND empty its inputs. Called on a composed
  // snapshot, on every view takeover that replaces the staging set, and
  // by the blind view when it opens, because a control holding a clone
  // root is a path waiting to be shown and the blind view's rule is
  // that it shows none.
  function forgetSnapshot() {
    snapRootEl.value = "";
    snapPatternsEl.value = "";
    closeSnapshotPanel();
  }

  // A message this control produced in ANSWER to something the person
  // did, as opposed to one it is merely restating. Set by every path in
  // composeSnapshot, cleared when a snapshot lands or the staged set is
  // replaced.
  function said(text) {
    snapMsgEl.textContent = text;
    snapMsgEl.dataset.said = text === "" ? "" : "1";
  }

  function renderSnapshotControl() {
    const blocker = snapshotBlocker();
    snapOpenEl.disabled = blocker !== "";
    snapComposeEl.disabled = blocker !== "" || busy();
    if (blocker !== "") closeSnapshotPanel();
    // AN ANSWER OUTRANKS A RESTATEMENT. render() runs on every staging
    // change, so without this rule the specific thing a person just
    // earned ("the snapshot was composed and stored, and there was no
    // room left to attach it") is overwritten on the very next repaint
    // by the generic blocker ("already at four documents"), which is the
    // same fact said less usefully. A refusal the server issued
    // vanishing as the chip list redrew is the same bug one step worse.
    if (snapMsgEl.dataset.said === "1") return;
    // The sentence is SHOWN and not merely hovered. A disabled button
    // whose reason lives in a title is a reason no keyboard user ever
    // reads, which is the same finding that put the run blocker into a
    // line of text rather than into a tooltip.
    snapMsgEl.textContent = blocker;
  }

  async function composeSnapshot() {
    const root = snapRootEl.value.trim();
    // ONE PER LINE, NEVER SPLIT ON COMMAS: the fourteenth review's
    // medium. The server accepts a comma literally in a pattern and
    // inside a character class, so "a,b.py" is one pattern naming one
    // file, and a comma split sent "a" and "b.py" and could have
    // selected two different files from what was typed. A newline can
    // never be part of a pattern, so it is the one safe separator.
    const patterns = snapPatternsEl.value
      .split("\n")
      .map((pattern) => pattern.trim())
      .filter((pattern) => pattern !== "");
    if (root === "") {
      said(
        "Name the clone root to walk: an absolute path under one of the " +
          "server's BENCH_REPO_ROOTS entries.",
      );
      return;
    }
    if (patterns.length === 0) {
      said(
        "Name at least one include pattern. Patterns are repo-relative " +
          "and do not recurse unless they say so, so '*.py' is the top " +
          "level and '**/*.py' is every depth.",
      );
      return;
    }
    // Both epochs, the same discipline addFiles follows: a snapshot that
    // was still composing when the person opened a history entry or hit
    // reuse belongs to a declaration nobody is composing any more.
    const epoch = window.BenchState.viewEpoch;
    const staging = stagingEpoch;
    inFlight += 1;
    render();
    try {
      const resp = await fetch("/snapshots", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ root: root, patterns: patterns }),
      });
      let body = null;
      try {
        body = await resp.json();
      } catch (err) {
        body = null;
      }
      if (stale(epoch, staging)) return;
      if (!resp.ok) {
        said(
          body
            ? refusalText(body.detail)
            : "the snapshot was refused (HTTP " + resp.status + ")",
        );
        return;
      }
      if (body === null) {
        // A success whose body did not parse. Said plainly rather than
        // walked into, because the next line reads body.digest and
        // would have reported a null dereference as "the request could
        // not be sent", which is the one thing that did happen fine.
        said(
          "the bench answered " +
            resp.status +
            " with a body this page could not read, so nothing was " +
            "attached. The snapshot itself is stored; look it up under " +
            "Documents.",
        );
        return;
      }
      if (staged.length >= MAX_ATTACHMENTS) {
        said(
          "The snapshot was composed and stored, and there was no room " +
            "left to attach it: the bound is " +
            MAX_ATTACHMENTS +
            " documents. Remove one and compose it again; the bytes are " +
            "already here, so nothing is recomputed.",
        );
        return;
      }
      if (staged.some((doc) => doc.digest === body.digest)) {
        said(
          "That snapshot is already attached: the same tree under the " +
            "same patterns composes the same bytes, so it is the same " +
            "document.",
        );
        return;
      }
      staged.push(body);
      said("");
      // The inputs are cleared on success and the panel closed, so the
      // root that was typed does not sit on the page waiting for the
      // next open: a clone root is a path on somebody's filesystem, the
      // one thing this feature never shows, and a hidden input holding
      // one is a view that shows it one click later.
      forgetSnapshot();
    } catch (err) {
      if (stale(epoch, staging)) return;
      said("the snapshot request could not be sent: " + err.message);
    } finally {
      inFlight -= 1;
      render();
    }
  }

  function init() {
    inputEl.addEventListener("change", async () => {
      // COPIED, and the copy is the whole point rather than a style
      // choice. inputEl.files is a LIVE FileList: clearing the input
      // empties that same object, so passing the FileList itself handed
      // addFiles an empty list and every pick silently did nothing.
      // Measured, not theorized: the browser suite saw zero chips and no
      // request at all.
      const files = [...inputEl.files];
      // Cleared before the await, so picking the same file twice in a
      // row still fires a change event the second time.
      inputEl.value = "";
      await addFiles(files);
    });
    modeEl.addEventListener("change", () => {
      msgEl.textContent = "";
      render();
    });
    snapOpenEl.addEventListener("click", () => {
      const opening = snapPanelEl.hidden;
      snapPanelEl.hidden = !opening;
      snapOpenEl.setAttribute("aria-expanded", String(opening));
      if (opening) snapRootEl.focus();
    });
    snapComposeEl.addEventListener("click", () => {
      void composeSnapshot();
    });
    render();
  }

  // The server's refusal, shown at the control that caused it.
  //
  // A comparison carrying documents fails CLOSED: when the group POST
  // is refused, nothing runs and the words the server used land here
  // rather than in a console nobody has open. The composer is the right
  // place because every refusal this can carry is about a choice made
  // here (which files, which mode) and the remedy is a change to that
  // choice.
  A.showRefusal = (detail) => {
    msgEl.textContent = detail;
  };

  // Whether this staging batch has been superseded by a view takeover.
  // Read after every await in addFiles, which is the same discipline
  // runOne and showGroup follow: async work stamps the epoch it started
  // under and touches shared view state only while that epoch is
  // current.
  //
  // BOTH EPOCHS, because either one moving means these documents belong
  // to a declaration nobody is composing any more. See stagingEpoch for
  // the case the view epoch cannot see.
  function stale(epoch, staging) {
    return epoch !== window.BenchState.viewEpoch || staging !== stagingEpoch;
  }

  // ---- The read-only chip a STORED declaration renders as, wherever
  // ---- it is shown. The composer's chipFor above is a different
  // ---- object: it carries a remove button and a live token estimate,
  // ---- because a staged document can still be taken back out. Nothing
  // ---- in history or in a report can be, so those views get this.

  // The chip body both callers share, so a document described in
  // history and the same document described in a report cannot drift
  // into two shapes. The two differ only in what they may put on the
  // face of the chip, which is the whole of the argument below.
  function readOnlyChip(label, bits, testid, missing) {
    const chip = document.createElement("span");
    chip.className = "attach-chip ref";
    chip.dataset.testid = testid;
    // textContent throughout: a filename is user text and gets the
    // same treatment model output does.
    chip.textContent = label;
    if (missing) chip.classList.add("attach-missing");
    chip.title = bits.join("\n");
    return chip;
  }

  // A document a comparison declared, as the history views describe it:
  // by the name it was uploaded under.
  //
  // A ref whose row is gone carries only a digest (see AttachmentRef);
  // it renders as such rather than being skipped, so a comparison that
  // declared three documents never reads as having declared two.
  A.refChip = function (ref, mode, where) {
    const missing = isMissing(ref);
    const bits = [];
    if (!missing && Number.isFinite(ref.byte_size)) {
      bits.push(fmtBytes(ref.byte_size));
    }
    bits.push("sha256 " + shortDigest(ref.digest));
    if (mode) bits.push(mode);
    // The extractor and its version, because a pypdf upgrade changes the
    // text a model read and a replay that could not say which parser
    // produced it would be a replay of a prompt nobody can reconstruct.
    if (!missing && ref.extractor && ref.extractor !== "none") {
      bits.push("read by " + ref.extractor + " " + ref.extractor_version);
    }
    if (captureLine(ref.capture)) bits.push(captureLine(ref.capture));
    return readOnlyChip(
      missing ? "(no longer stored)" : docLabel(ref),
      bits,
      where + "-attachment",
      missing,
    );
  };

  // The same document as a REPORT describes it, which is by digest and
  // reading and never by name.
  //
  // TWO DIFFERENT ABSENCES, and collapsing them is the reason this is
  // its own entry point rather than refChip called with a null
  // filename. refChip's blank face means "the bench no longer holds
  // these bytes", which is a fact about the store. A report has no name
  // to show because a filename is what a person picked on their own
  // machine and is not a property of the bytes, so naming one would be
  // citing a fact about a filesystem in an artifact about an
  // experiment. The document is stored and the chip must not say
  // otherwise.
  A.pinChip = function (entry, where, capture) {
    const bits = ["sha256 " + entry.digest];
    if (entry.mode) bits.push(entry.mode);
    if (entry.kind) bits.push("read as " + entry.kind);
    if (entry.extractor && entry.extractor !== "none") {
      bits.push("read by " + entry.extractor + " " + entry.extractor_version);
    }
    if (captureLine(capture)) bits.push(captureLine(capture));
    return readOnlyChip(
      "sha256 " + shortDigest(entry.digest),
      bits,
      where + "-attachment",
      false,
    );
  };

  A.busy = busy;
  A.blockingReason = blockingReason;
  A.forgetSnapshot = forgetSnapshot;
  // Repaint on new facts from outside, currently the data policy landing
  // with the catalog. Named refresh rather than exposing render, because
  // a caller must not be able to pass it arguments and change what is
  // drawn: this only redraws what is already staged.
  A.refresh = render;
  window.BenchAttach = A;
})();
