// Blind human rating. Exposed on window.BenchRating.
//
// The point of blinding is that a rater who can see which model wrote an
// answer is not rating the answer. ONE PATH, and it starts at the server:
// the anonymized view is built from a payload that carries answers and
// nothing else, under a session token the server issued, and identities
// arrive for the first time at the reveal, after the ratings are already
// saved. Revealing before the save would let a rater see the answer and
// then change their mind, which is the same defect wearing a hat.
//
// There used to be a second path that fetched the identified comparison
// and then hid the identities with a style rule, sending blind: true
// with the ratings. Both halves of that were wrong. Hidden in a browser
// is a rule anyone can undo plus a frame the rater may already have
// seen, and a page that painted the answer key is the one witness that
// cannot testify about its own blindness. It is gone rather than
// deprecated, because a path nobody should use is a path nobody should
// be able to reach.
(function () {
  const resultsEl = document.getElementById("results");
  const runControlsEl = document.getElementById("run-controls");
  const labelEl = document.getElementById("run-label");

  // The rating scale, and it must equal RATING_MIN and RATING_MAX in
  // bench/main.py. The server validates the range it receives, so a
  // client scale wider than the server's renders a button whose click is
  // refused, and a narrower one quietly makes the top of the scale
  // unreachable with nothing on screen to say so.
  // tests/browser/test_i4.py::test_the_rating_scale_matches_the_server_constant
  // asserts the pair agrees, the same guard the controls markup carries
  // against ExperimentParams.
  const RATING_MIN = 1;
  const RATING_MAX = 5;

  // The active blind session, or null. One at a time and per group: a
  // second session over the same group would be rating cards that are no
  // longer on screen.
  //
  // Staleness is decided by the view epoch rather than by anyone
  // remembering to call end(). Every view takeover advances the epoch
  // already, so a session that recorded its epoch at start can tell for
  // itself whether the cards it points at are still the ones on screen.
  // Cleanup calls at each of the three view-change sites would work
  // until someone added a fourth.
  let session = null;

  function current() {
    if (session === null) return null;
    if (session.epoch !== window.BenchState.viewEpoch) {
      session = null;
      return null;
    }
    return session;
  }

  function ratingRow(card, resultId, label) {
    const row = document.createElement("div");
    row.className = "rating-row";
    row.dataset.testid = "rating-row";
    const tag = document.createElement("span");
    tag.className = "rating-label";
    tag.dataset.testid = "rating-label";
    tag.textContent = label;
    row.append(tag);
    for (let value = RATING_MIN; value <= RATING_MAX; value++) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "rating-button";
      button.dataset.testid = "rating-" + value;
      button.dataset.value = String(value);
      button.textContent = String(value);
      button.setAttribute("aria-label", "rate " + value + " of " + RATING_MAX);
      button.addEventListener("click", () => {
        if (!current()) return;
        session.ratings.set(resultId, { rating: value, label: label });
        for (const other of row.querySelectorAll(".rating-button")) {
          other.classList.toggle("chosen", other === button);
          other.setAttribute("aria-pressed", String(other === button));
        }
        updateSubmit();
      });
      row.append(button);
    }
    card.append(row);
    return row;
  }

  function updateSubmit() {
    if (!current()) return;
    const done = session.ratings.size;
    const total = session.cards.length;
    session.submit.disabled = done < total;
    session.count.textContent = done + " of " + total + " rated";
  }

  function renderPanel() {
    const panel = document.createElement("div");
    panel.className = "rating-panel";
    panel.dataset.testid = "rating-panel";
    const note = document.createElement("span");
    note.textContent =
      "Blind rating: identities, cost and timing are hidden until you submit.";
    const count = document.createElement("span");
    count.dataset.testid = "rating-count";
    const submit = document.createElement("button");
    submit.type = "button";
    submit.dataset.testid = "rating-submit";
    submit.textContent = "Submit ratings";
    submit.addEventListener("click", submitRatings);
    const msg = document.createElement("span");
    msg.dataset.testid = "rating-msg";
    msg.setAttribute("role", "status");
    panel.append(note, count, submit, msg);
    runControlsEl.append(panel);
    session.submit = submit;
    session.count = count;
    session.msg = msg;
  }

  async function submitRatings() {
    if (!current() || session.revealed) return;
    session.submit.disabled = true;
    session.msg.textContent = "saving";
    // The TOKEN, and no claim. The server decides whether these ratings
    // are blind by whether this token names a session it still has open;
    // a flag saying "trust me, I was blind" is the one thing a page
    // cannot honestly send about its own past.
    const payload = {
      blind_token: session.token,
      ratings: Array.from(session.ratings, ([resultId, entry]) => ({
        result_id: resultId,
        rating: entry.rating,
        label: entry.label,
      })),
    };
    const groupId = session.groupId;
    try {
      const resp = await fetch("/groups/" + groupId + "/ratings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
    } catch (err) {
      // The reveal does not happen. A rater who saw the identities after
      // a failed save would have to re-rate with the answer in front of
      // them, and that second rating would not be blind however it was
      // labelled.
      if (!current()) return;
      session.msg.textContent = "failed to save ratings: " + err.message;
      session.submit.disabled = false;
      console.error("rating submit failed", err);
      return;
    }
    // The ratings are saved. If the view moved on while the request was
    // in flight there is nothing left to reveal, and the record is
    // complete either way: the reveal is a courtesy to the rater, not
    // part of the measurement.
    if (!current()) return;
    const identities = await closeServerSession(groupId);
    if (!current()) return;
    reveal(identities);
  }

  async function closeServerSession(groupId) {
    // The server's session closes one way, and the close is what makes
    // every later rating of this comparison honest about the fact that
    // somebody has now seen the answer key. Failing to close is not
    // worth blocking the reveal over: the ratings are already saved, and
    // the session expiring with the process fails closed.
    try {
      const resp = await fetch("/groups/" + groupId + "/blind/reveal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      // The identities array, not the envelope around it. reveal() maps
      // over what this returns, so handing back the whole payload threw
      // inside an async function and left the panel stuck on "saving"
      // with the ratings already saved: a silent half-finished state
      // that only the browser test caught.
      return (await resp.json()).identities;
    } catch (err) {
      console.error("blind reveal failed", err);
      return null;
    }
  }

  function reveal(identities) {
    // One way. Re-blinding after a reveal would produce a rating the
    // record calls blind that was made by someone who had already seen
    // the answer, which is worse than no blind rating at all.
    session.revealed = true;
    // WRITTEN FOR THE FIRST TIME, never un-hidden. The blind view was
    // built from a payload that carried no identity at all, so there is
    // no concealed copy in the DOM to restore; if the close failed there
    // is simply nothing to paint, and the ratings are already saved
    // either way.
    if (identities) {
      const byId = new Map(identities.map((i) => [i.result_id, i.model]));
      for (const pair of session.cards) {
        const name = document.createElement("div");
        name.className = "card-header";
        name.dataset.testid = "card-model";
        name.textContent = byId.get(pair.resultId) || "";
        pair.card.prepend(name);
      }
    }
    for (const row of resultsEl.querySelectorAll(
      '[data-testid="rating-row"]',
    )) {
      for (const button of row.querySelectorAll(".rating-button")) {
        button.disabled = true;
      }
    }
    session.msg.textContent = "saved. Identities revealed.";
    session.submit.hidden = true;
    session.submit.dataset.testid = "rating-submit-done";
  }

  async function startBlind(groupId) {
    // The server opens the session and issues the shuffle, and the
    // payload it returns carries answers with NO identity attached: no
    // model, no provider, no cost, no timing. Rendering from this and
    // only this is what makes "the page could not have known either" a
    // fact rather than a promise, because there is no identified paint
    // to undo and nothing in the DOM to inspect.
    window.BenchState.newViewEpoch();
    resultsEl.replaceChildren();
    runControlsEl.replaceChildren();
    // THE COMPOSER FORGETS ITS SNAPSHOT INPUTS, the fourteenth review's
    // medium. The blind cards themselves never carried a path, but the
    // composer stays on the page under them, and a root typed earlier
    // sat in a hidden input one click from view. A view whose rule is
    // "shows no file paths" cannot hold one anywhere on the page.
    window.BenchAttach.forgetSnapshot();
    labelEl.textContent = "Blind rating: comparison #" + groupId;
    let payload;
    try {
      const resp = await fetch("/groups/" + groupId + "/blind", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      payload = await resp.json();
    } catch (err) {
      // Same rule as every other load here: the failure is on the page
      // and on the console, never only in a variable.
      console.error("blind session failed to open", err);
      labelEl.textContent = "could not open a blind session: " + err.message;
      return;
    }
    const pairs = [];
    for (const entry of payload.cards) {
      const card = document.createElement("div");
      card.className = "card";
      card.dataset.testid = "result-card";
      const body = document.createElement("div");
      body.className = "card-body";
      body.textContent = entry.response_text || entry.error || "";
      card.append(body);
      resultsEl.append(card);
      pairs.push({ card: card, resultId: entry.result_id, label: entry.label });
    }
    session = {
      groupId: groupId,
      cards: pairs,
      ratings: new Map(),
      revealed: false,
      epoch: window.BenchState.viewEpoch,
      // The server's proof that THIS page was handed an anonymized view.
      // Held only here, for as long as this view lasts: it is the whole
      // of what makes a rating from this tab blind, and a second tab
      // showing the identified comparison has no way to obtain it.
      token: payload.token,
      submit: null,
      count: null,
      msg: null,
    };
    for (const pair of pairs) ratingRow(pair.card, pair.resultId, pair.label);
    renderPanel();
    updateSubmit();
  }

  window.BenchRating = { startBlind };
})();
