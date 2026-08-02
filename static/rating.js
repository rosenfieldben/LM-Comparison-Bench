// Blind human rating on a replayed comparison. Exposed on window.BenchRating.
//
// The point of blinding is that a rater who can see which model wrote an
// answer is not rating the answer. So this hides identity, provider, cost
// and timing, gives each card a neutral label in a shuffled order, and
// reveals only after every card has been rated and the ratings are
// already saved. Revealing before the save would let a rater see the
// answer and then change their mind, which is the same defect wearing a
// different hat.
(function () {
  const resultsEl = document.getElementById("results");
  const runControlsEl = document.getElementById("run-controls");

  // Neutral, order-free labels. Letters rather than numbers because a
  // number reads as a rank, and a rater who thinks card 1 is the first
  // one is already being nudged.
  const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

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

  function shuffled(items) {
    // Fisher-Yates over a copy. crypto.getRandomValues rather than
    // Math.random is deliberate overkill in one respect and not in
    // another: it costs nothing here, and it removes any question about
    // whether a weak generator could correlate label order with lineup
    // order across sessions.
    const out = items.slice();
    const draws = new Uint32Array(out.length);
    crypto.getRandomValues(draws);
    for (let i = out.length - 1; i > 0; i--) {
      const j = draws[i] % (i + 1);
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }

  // Everything on a card that could identify the model behind it. Hidden
  // as a set rather than one by one at each call site, so a card that
  // grows a new identifying field later is a change in one place.
  //
  // Cost and timing are here for the same reason as the name: a rater who
  // knows one answer cost ten times another, or took four seconds longer,
  // can often guess which model it was, and a guess is not a blind.
  const CONCEALED = [
    '[data-testid="card-model"]',
    '[data-testid="card-provider"]',
    ".metrics",
  ];

  function hideIdentity(card) {
    for (const selector of CONCEALED) {
      const el = card.querySelector(selector);
      if (!el) continue;
      // The state before the blind, so the reveal restores rather than
      // unhides. The provider caption starts hidden when no provider was
      // recorded, and a reveal that simply cleared hidden would show an
      // empty caption on exactly the rows that have nothing to say.
      el.dataset.blindWasHidden = el.hidden ? "1" : "0";
      el.hidden = true;
    }
  }

  function showIdentity(card) {
    for (const selector of CONCEALED) {
      const el = card.querySelector(selector);
      if (!el) continue;
      el.hidden = el.dataset.blindWasHidden === "1";
      delete el.dataset.blindWasHidden;
    }
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
    for (let value = 1; value <= 5; value++) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "rating-button";
      button.dataset.testid = "rating-" + value;
      button.dataset.value = String(value);
      button.textContent = String(value);
      button.setAttribute("aria-label", "rate " + value + " of 5");
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

  // Cards, paired with the result ids they show. Read from the DOM rather
  // than passed in, because the DOM is what the rater is actually looking
  // at: a list handed in could drift from the cards on screen, and then
  // the rating-to-result mapping would be wrong in exactly the way
  // nothing on screen would reveal.
  function cardsOnScreen(resultIds) {
    const cards = Array.from(
      resultsEl.querySelectorAll('[data-testid="result-card"]'),
    );
    return cards.map((card, index) => ({ card, resultId: resultIds[index] }));
  }

  function start(groupId, resultIds) {
    if (current()) return;
    const pairs = cardsOnScreen(resultIds);
    if (pairs.length === 0) return;
    // Labels are assigned in shuffled order, so label order carries no
    // information about lineup order. The cards stay where they are:
    // moving them would also work, but it would make the reveal jump the
    // page around, and the labels alone are enough.
    const labels = shuffled(LABELS.slice(0, pairs.length).split(""));
    session = {
      groupId: groupId,
      cards: pairs,
      ratings: new Map(),
      revealed: false,
      epoch: window.BenchState.viewEpoch,
      submit: null,
      count: null,
      msg: null,
    };
    for (const [index, pair] of pairs.entries()) {
      hideIdentity(pair.card);
      ratingRow(pair.card, pair.resultId, labels[index]);
    }
    renderPanel();
    updateSubmit();
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
    const payload = {
      blind: true,
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
    reveal();
  }

  function reveal() {
    // One way. Re-blinding after a reveal would produce a rating the
    // record calls blind that was made by someone who had already seen
    // the answer, which is worse than no blind rating at all.
    session.revealed = true;
    for (const pair of session.cards) showIdentity(pair.card);
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

  window.BenchRating = { start };
})();
