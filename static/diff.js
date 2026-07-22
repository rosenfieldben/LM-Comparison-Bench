// The diff view: tokenize, LCS, and render, then the arm/compare wiring
// on the result cards. Exposed on window.BenchDiff. registerDiffable is
// the edge render.js calls back into once a card completes.
(function () {
  const { tokenizeDiff, diffTokens, DIFF_TOKEN_LIMIT } = window.BenchLib;
  const { toolButton } = window.BenchRender;

  // ---- Diff view. Pure functions first (tokenize, lcs, render) so a JS
  // ---- harness could test them later; DOM wiring below.

  function renderDiff(ops, container) {
    container.replaceChildren();
    // Adjacent tokens with the same op merge into one node: fewer DOM
    // nodes and the del/ins tint reads as one span per changed region.
    let run = null;
    function flush() {
      if (run === null) return;
      if (run.op === "same") {
        container.append(document.createTextNode(run.text));
      } else {
        // createElement plus textContent only: model output stays
        // untrusted in diff form, exactly as in the cards.
        const el = document.createElement(run.op === "del" ? "del" : "ins");
        el.textContent = run.text;
        container.append(el);
      }
      run = null;
    }
    for (const o of ops) {
      if (run !== null && run.op === o.op) {
        run.text += o.raw;
      } else {
        flush();
        run = { op: o.op, text: o.raw };
      }
    }
    flush();
  }

  const diffPanel = document.getElementById("diff-panel");
  const diffTitle = document.getElementById("diff-title");
  const diffBody = document.getElementById("diff-body");
  // Armed side of a pending diff. Holds the result data, not just the
  // element: history replay replaces the cards, and surviving that is
  // what makes live-vs-historical diffs possible.
  let armedDiff = null;

  function closeDiffPanel() {
    diffPanel.hidden = true;
    diffTitle.textContent = "";
    diffBody.replaceChildren();
  }

  function setArmed(btn, on) {
    btn.classList.toggle("armed", on);
    btn.setAttribute("aria-pressed", String(on));
    // The armed side carries a filled marker: "diff ●".
    btn.textContent = (btn.dataset.base || "diff") + (on ? " ●" : "");
  }

  // Arming deliberately survives history replay (that is what makes
  // live-vs-historical diffs possible), but not a new Run: its cards
  // supersede the comparison the armed result came from, and a later
  // toggle would silently diff against a vanished card.
  function disarmDiff() {
    if (armedDiff !== null) {
      // The armed button may already be detached (replaced results);
      // clearing the state is harmless either way.
      setArmed(armedDiff.btn, false);
      armedDiff = null;
    }
  }

  function openDiff(a, b) {
    diffTitle.textContent = a.label + " ⇄ " + b.label;
    const ta = tokenizeDiff(a.result.response_text);
    const tb = tokenizeDiff(b.result.response_text);
    if (ta.length > DIFF_TOKEN_LIMIT || tb.length > DIFF_TOKEN_LIMIT) {
      diffBody.textContent = "responses too large to diff";
    } else {
      renderDiff(diffTokens(ta, tb), diffBody);
    }
    diffPanel.hidden = false;
  }

  function registerDiffable(ui, result, sourceLabel) {
    // Error-only cards have nothing to diff; partial text (the
    // both-text-and-error contract) is diffable and labeled as such.
    if (result.response_text == null || sourceLabel == null) return;
    const label = sourceLabel + (result.error != null ? " (partial)" : "");
    const btn = BenchRender.toolButton(result.error != null ? "diff (partial)" : "diff");
    btn.dataset.testid = "tool-diff";
    btn.classList.add("diff-btn");
    btn.dataset.base = btn.textContent;
    btn.setAttribute("aria-pressed", "false");
    btn.addEventListener("click", () => {
      if (armedDiff !== null && armedDiff.btn === btn) {
        setArmed(btn, false);
        armedDiff = null;
        return;
      }
      if (armedDiff === null) {
        armedDiff = { btn: btn, result: result, label: label };
        setArmed(btn, true);
        return;
      }
      openDiff(armedDiff, { btn: btn, result: result, label: label });
      setArmed(armedDiff.btn, false);
      armedDiff = null;
    });
    ui.tools.append(btn);
  }

  function init() {
    document.getElementById("diff-close").addEventListener("click", closeDiffPanel);
  }

  window.BenchDiff = { closeDiffPanel, disarmDiff, registerDiffable, init };
})();
