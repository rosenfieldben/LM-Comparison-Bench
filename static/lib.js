// Pure, DOM-free helpers, extracted so `node --test` can require them
// without a browser. Wrapped in an IIFE so nothing leaks to the page's
// global scope; app.js pulls these off window.BenchLib. This file has no
// dependencies and does no I/O, which is exactly what makes it testable.
(function () {
  // Chips and the race strip show ids without the vendor prefix; the
  // full id stays in the title attribute and in every request.
  function shortName(id) {
    const slash = id.indexOf("/");
    return slash === -1 ? id : id.slice(slash + 1);
  }

  function usdDigits(c) {
    // Exact zero written out rather than as an exponent. "$0.0e+0" is a
    // needlessly cryptic way to write nothing, and a real zero is
    // reachable: free models exist, and their billed cost is genuinely
    // zero rather than absent.
    if (c === 0) return "$0.0000";
    // Exponential for the typical sub-cent run ($3.1e-5); plain decimals
    // once a run costs enough for them to be readable. Shared so the
    // estimate and the billed figure sit at the same scale on the card and
    // only their prefix differs.
    return "$" + (c < 0.01 ? c.toExponential(1) : c.toFixed(3));
  }

  function fmtCost(c) {
    // The tilde is the honesty marker: catalog prices times reported
    // tokens, not a bill.
    return "~" + usdDigits(c);
  }

  function fmtBilled(c) {
    // No tilde, because this one is not an estimate: it is the amount
    // OpenRouter reported charging for the request. The marker's absence
    // is the whole signal, so the two formatters must never converge.
    return usdDigits(c);
  }

  function fmtEstimate(v) {
    if (v >= 100) return String(Math.round(v));
    return Number(v.toPrecision(2)).toString();
  }

  // ---- Attachment chip formatting. Pure, so the chip the composer
  // ---- draws and the chip a history row draws cannot drift: an
  // ---- attachment shown one way before a run and another way after it
  // ---- would make the record look like a different document.

  // Binary units, because the upload limit is stated in MiB and a chip
  // that said "8.4 MB" beside a refusal quoting 8 MiB would read as a
  // contradiction. KiB at 1024 and not 1000 for the same reason.
  function fmtBytes(n) {
    if (!Number.isFinite(n) || n < 0) return "";
    if (n < 1024) return n + " B";
    const units = ["KiB", "MiB", "GiB"];
    let value = n / 1024;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    // One decimal below ten, none above: "9.4 KiB" and "512 KiB" both
    // read at a glance, and "512.3 KiB" is three digits of noise.
    return (
      (value < 10 ? value.toFixed(1) : Math.round(value)) + " " + units[unit]
    );
  }

  // The first seven hex characters, git's length and for git's reason:
  // long enough to be unambiguous by eye across the handful of documents
  // one bench holds, short enough to sit in a chip. NEVER used to look
  // anything up: every request carries the full digest, because a
  // prefix that collided would attach the wrong document.
  function shortDigest(digest) {
    return typeof digest === "string" ? digest.slice(0, 7) : "";
  }

  // Characters over four, the oldest rough tokenizer there is, and it is
  // labeled approximate everywhere it is shown for exactly that reason.
  //
  // The bench does NOT tokenize. Every model in a comparison runs its
  // own tokenizer and they disagree with each other, so any single
  // number here is wrong for at least some of them; what this is for is
  // the order of magnitude, the difference between "this fits" and "this
  // is a novel". Shipping a real tokenizer would be a dependency that
  // bought a precise answer to a question nobody asked, and worse, would
  // make the estimate look authoritative when it can only ever describe
  // one model's view.
  function approxTokens(chars) {
    if (!Number.isFinite(chars) || chars < 0) return 0;
    return Math.ceil(chars / 4);
  }

  function niceScale(maxTtft) {
    // Smallest 100·2^k at or above the slowest known TTFT, so bars keep
    // honest proportions and the scale label stays a round number.
    let scale = 100;
    while (scale < maxTtft) scale *= 2;
    return scale;
  }

  // Past this many word tokens the O(n*m) LCS table would freeze the
  // tab, so the panel shows a notice instead.
  const DIFF_TOKEN_LIMIT = 4000;

  function tokenizeDiff(text) {
    // Capturing split keeps whitespace; each token is a word plus its
    // trailing whitespace so reconstruction preserves original spacing
    // and line breaks. The LCS compares the word alone (key), so a moved
    // line break does not mark the word itself as changed.
    const parts = text.split(/(\s+)/);
    const tokens = [];
    for (const part of parts) {
      if (part === "") continue;
      if (/^\s+$/.test(part) && tokens.length > 0) {
        tokens[tokens.length - 1].raw += part;
      } else if (/^\s+$/.test(part)) {
        tokens.push({ raw: part, key: "" });
      } else {
        tokens.push({ raw: part, key: part });
      }
    }
    return tokens;
  }

  function diffTokens(a, b) {
    // Classic LCS table with backtrack. O(n*m) is fine at bench scale
    // behind DIFF_TOKEN_LIMIT, and forty lines of understandable code
    // beat an optimal but opaque algorithm in a hand-maintained file.
    const n = a.length;
    const m = b.length;
    const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
    for (let i = 1; i <= n; i += 1) {
      for (let j = 1; j <= m; j += 1) {
        dp[i][j] =
          a[i - 1].key === b[j - 1].key
            ? dp[i - 1][j - 1] + 1
            : Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    const ops = [];
    let i = n;
    let j = m;
    while (i > 0 && j > 0) {
      if (a[i - 1].key === b[j - 1].key) {
        // Common text takes B's spacing so the merged view flows like
        // the newer response.
        ops.push({ op: "same", raw: b[j - 1].raw });
        i -= 1;
        j -= 1;
      } else if (dp[i - 1][j] >= dp[i][j - 1]) {
        ops.push({ op: "del", raw: a[i - 1].raw });
        i -= 1;
      } else {
        ops.push({ op: "ins", raw: b[j - 1].raw });
        j -= 1;
      }
    }
    while (i > 0) {
      ops.push({ op: "del", raw: a[i - 1].raw });
      i -= 1;
    }
    while (j > 0) {
      ops.push({ op: "ins", raw: b[j - 1].raw });
      j -= 1;
    }
    return ops.reverse();
  }

  // The badge vocabulary for experiment controls, in a fixed order so two
  // comparisons with the same controls read the same way. Compact because
  // a history row is one line that already carries a timestamp, a prompt
  // excerpt and a model count; the long form lives in the title.
  //
  // The system prompt gets a presence badge and not its text. A prompt long
  // enough to matter would swamp the row, and a truncated one invites the
  // reader to compare two comparisons on an excerpt that happens to match.
  // The full text belongs to the comparison view.
  const CONTROL_BADGES = [
    [
      "system",
      () => "sys",
      () => "a system prompt was set for this comparison",
    ],
    ["temperature", (v) => "t=" + v, (v) => "temperature " + v],
    ["top_p", (v) => "top_p=" + v, (v) => "top_p " + v],
    ["seed", (v) => "seed " + v, (v) => "seed " + v],
    ["effort", (v) => "effort " + v, (v) => "reasoning effort " + v],
    ["routing", (v) => "route " + v, (v) => "provider routing: " + v],
  ];

  // Badges for the controls a comparison actually set, and only those.
  // Rendering a control the user never chose would present a provider
  // default as a decision, which is the truth defect the whole controls
  // design exists to avoid, so absence has to survive all the way here.
  //
  // Zero is a value, not blankness: temperature 0 is the most deliberate
  // setting there is, so the skip test is an explicit null check and never
  // a falsy one. An empty string is skipped, because the API forbids one
  // and a badge built from it would claim a prompt that is not there.
  function controlBadges(params) {
    if (!params) return [];
    const out = [];
    for (const [key, text, title] of CONTROL_BADGES) {
      const value = params[key];
      if (value === null || value === undefined || value === "") continue;
      out.push({ text: text(value), title: title(value) });
    }
    return out;
  }

  // Min-rank over ascending values: equal values share a place and the
  // next one skips, so a three-way tie reads 1, 1, 1, 4. Two runs that
  // measured identically are tied, and numbering them anyway would show
  // a difference that is not in the data.
  //
  // Pure and here rather than inline in the race, because the same rule
  // governs the report's model ranking and a rule with two
  // implementations is a rule with two behaviors. Takes an already
  // sorted list of numbers and returns their ranks in the same order.
  function minRanks(sortedValues) {
    const ranks = [];
    for (let i = 0; i < sortedValues.length; i++) {
      ranks.push(
        i > 0 && sortedValues[i] === sortedValues[i - 1] ? ranks[i - 1] : i + 1,
      );
    }
    return ranks;
  }

  // The share of a completion that reasoning must reach before the card
  // says the thinking took the answer's place. This is the same 0.9 as
  // REASONING_SHARE_EXHAUSTED in bench/models.py, and it is the same rule
  // applied to the same two numbers, deliberately duplicated rather than
  // shipped from the server: the server can only label a result it
  // synthesized an error for, and the indicator has to fire on a card
  // that came back with text too. A test in each language pins the
  // threshold, so a change to one that is not made to the other is a
  // failing test rather than a silent divergence.
  const REASONING_SHARE_EXHAUSTED = 0.9;

  // Whether a result's output went to thinking rather than to an answer.
  //
  // NUMBERS ONLY, which is the entire design. No model name, no
  // finish_reason, no inspection of what was requested: this reads the
  // two counts that come back from any provider. That is what lets it
  // fire on a route where the reservation could not act, because a
  // provider that ignores an unknown request parameter still reports its
  // usage honestly. It is also why nothing here needs updating when a new
  // reasoning model appears.
  //
  // Absent counts are not evidence, and zero reasoning tokens are
  // evidence of the opposite, so both fall out through the same falsy
  // guard. Older history rows predate the reasoning column entirely and
  // carry null; they read as "no" rather than as "unknown", which is the
  // conservative direction for something that draws a warning.
  function reasoningAteTheOutput(completionTokens, reasoningTokens) {
    if (!reasoningTokens || !completionTokens) return false;
    return reasoningTokens >= completionTokens * REASONING_SHARE_EXHAUSTED;
  }

  // The share itself, rounded to a whole percent for display. Separate
  // from the predicate above because they answer different questions and
  // one of them is allowed to be approximate: the predicate decides
  // whether to say anything, this decides what the sentence reads.
  //
  // Rounded rather than truncated, and to a percent rather than to a
  // decimal, because it is a magnitude for a person to react to and not
  // an input to anything. Null when either count is missing, so a caller
  // cannot render "null%" from a row that predates the column.
  function reasoningShare(completionTokens, reasoningTokens) {
    if (completionTokens == null || reasoningTokens == null) return null;
    if (completionTokens === 0) return null;
    return Math.round((reasoningTokens / completionTokens) * 100);
  }

  const BenchLib = {
    shortName,
    fmtCost,
    fmtBilled,
    fmtEstimate,
    fmtBytes,
    shortDigest,
    approxTokens,
    niceScale,
    minRanks,
    tokenizeDiff,
    diffTokens,
    controlBadges,
    reasoningAteTheOutput,
    reasoningShare,
    REASONING_SHARE_EXHAUSTED,
    DIFF_TOKEN_LIMIT,
  };
  if (typeof window !== "undefined") window.BenchLib = BenchLib;
  if (typeof module !== "undefined") module.exports = BenchLib;
})();
