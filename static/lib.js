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
  // REASONING_SHARE_EXHAUSTED in bench/models.py, and it is the same
  // rule applied to the same two numbers.
  //
  // WHY IT IS DUPLICATED, stated accurately after an adversarial review
  // found the previous answer false. That answer said the server "can
  // only label a result it synthesized an error for". The server can do
  // no such limited thing: _ingest_usage runs unconditionally and
  // before the empty-text branch, so both counts are populated on every
  // result including those that carry text, and a boolean could be
  // attached to the result and shipped like any other field.
  //
  // The real reason is smaller and is a judgement rather than a
  // constraint. Shipping the flag would touch ModelResult, the stream's
  // done event, the store's read-time repair and three call sites, and
  // this file has to keep reasoningShare regardless, which already
  // reads both counts. So the duplication is one constant and one
  // comparison against a refactor across four layers, and it was not
  // judged worth it. Revisit that if a third consumer ever appears.
  //
  // The drift guard is behavioural, not textual: a Python test executes
  // this file through node over a shared table of shapes and compares
  // the two answer sheets. A change to either implementation that
  // alters an answer fails it; a rename or a rewrap does not.
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

  // The documented effort ladder, as SHARES of the outer budget, so
  // "is this one lower than that one" is a comparison of numbers rather
  // than of positions in a list somebody has to keep ordered. Mirrors
  // EFFORT_SHARES in bench/models.py, which carries the pinned quotes
  // and the read date; a cross-language test executes both against the
  // same questions so the two cannot drift.
  const EFFORT_SHARES = {
    none: 0.0,
    minimal: 0.1,
    low: 0.2,
    medium: 0.5,
    high: 0.8,
    xhigh: 0.95,
    max: 0.95,
  };

  // THE ROUTE'S OWN CEILING, DERIVED RATHER THAN FETCHED. A pinned
  // trial is clamped twice: to the model's published cap for its tier,
  // and then to the endpoint the pin selected. Only the first is
  // knowable from the catalog the browser holds, so a replayed card
  // cannot look the second one up.
  //
  // It does not have to. The server sends min(tierCap, routeCap), so a
  // sent ceiling STRICTLY BELOW the model-level cap for that tier can
  // only be the route's own number: min picked the other operand, and
  // the other operand is the route. When sent equals the tier cap the
  // route did not bind and nothing is learned, which is null rather
  // than a guess.
  //
  // The consequence is what the remedy needs: a route that bound a
  // standard run at 8192 bounds an extended one at 8192 too, because
  // min(65536, 8192) is the same number. Suggesting the larger tier
  // there is suggesting a replay of the identical request.
  function routeCapFor(sent, tierCap) {
    if (typeof sent !== "number" || typeof tierCap !== "number") return null;
    return sent < tierCap ? sent : null;
  }

  // The lowest reasoning effort the UI can actually select. Mirrors
  // the option list in index.html.
  //
  // "UNSET" IS NOT A POSITION ON THIS LADDER, which is the correction
  // that matters. This comment used to say unset "is NOT lower, because
  // on a vouched model the bench then sends its own half-budget
  // reservation rather than nothing", and used that to justify advising
  // a reader to lower an effort they had never set. Unset means the
  // ROUTE decides, at its catalog default, which the card cannot see
  // and which may sit below this floor. So unset is unknown, and
  // remedyFor advises nothing from it.
  const LOWEST_SELECTABLE_EFFORT = "low";

  // WHAT THE CARD SHOULD SUGGEST, given what this run actually got and
  // what the reader can actually change.
  //
  // THE DEFECT THIS REPLACES. The advice keyed on the budget TIER the
  // user had selected, and offered "try extended budget" whenever that
  // tier was standard. But the server clamps the requested tier to
  // whatever completion cap a model publishes, and falls back to the
  // standard tier entirely when the catalog is offline, so extended can
  // be byte-identical to standard. On those runs the card told a reader
  // to spend four times as much to send exactly the same request. It
  // also offered "a lower reasoning effort" without reading the effort
  // that was set, so a run already at the UI minimum was told to lower
  // something that cannot go lower.
  //
  // KEYED ON result.max_tokens, the EFFECTIVE post-clamp ceiling this
  // run was actually sent, and on the caller's own report of what the
  // extended tier would clamp to for this model. A tier name cannot
  // answer either question.
  //
  // Returns "" when nothing can be SHOWN to help, and the caller keeps
  // the accounting sentence alone. That is a weaker claim than the one
  // this comment used to make ("every remedy is known not to work"),
  // and the weaker one is the true one: since the effort clause began
  // requiring an effort the ladder can place above the floor, silence
  // also covers the case where the card simply does not know what the
  // route is doing. Not knowing and knowing it will not help both
  // produce the same honest output, which is nothing.
  function remedyFor(result, options) {
    const opts = options || {};
    const sent = result.max_tokens;
    const extendedCap = opts.extendedCap;
    const parts = [];
    // A BIGGER BUDGET IS ONLY ADVICE IF A BIGGER BUDGET EXISTS, was
    // not already asked for, and can be shown to reach this run's
    // route. Three conditions, and each one was learned from a card
    // that gave advice it could not support.
    //
    // ONE: the tier must be KNOWN. An absent tier used to be read as
    // "not extended", so a run that selected extended and whose record
    // does not carry the tier was told to select it. Unknown is not
    // standard; it is unknown, and nothing can be advised from it.
    //
    // TWO: the tier must not already be extended. extendedCap is what
    // the catalog publishes TODAY and sent is what this run received,
    // possibly weeks ago, so on a replay the cap comparison alone
    // starts recommending extended to a clamped extended run the moment
    // the published cap rises.
    //
    // THREE: the ROUTE's ceiling, where one is known, bounds what
    // extended could deliver. extendedCap is a MODEL-level number and a
    // provider-pinned trial is clamped again to its endpoint's, so a
    // run pinned to a route capping at 8192 was told to try a tier that
    // clamps to the same 8192. See routeCapFor for how a caller learns
    // that number without asking the network.
    const reachable =
      typeof opts.routeCap === "number"
        ? Math.min(extendedCap, opts.routeCap)
        : extendedCap;
    if (
      (opts.budget === "standard" || opts.budget === "extended") &&
      opts.budget !== "extended" &&
      typeof sent === "number" &&
      typeof extendedCap === "number" &&
      reachable > sent
    ) {
      parts.push("extended budget");
    }
    // A lower effort is only advice when a LOWER one demonstrably
    // exists, and absent is not evidence of that.
    //
    // THE DEFECT. Absent was read as "the operator chose nothing, which
    // is above the minimum, so choosing low would reduce the thinking".
    // The first half is true and the second does not follow: with
    // nothing chosen the route runs at its CATALOG default, which the
    // card does not know and which may be minimal (0.1) or none (0.0).
    // Selecting low (0.2) would then RAISE reasoning, on a card whose
    // entire subject is that reasoning consumed the budget. Two
    // measured examples exist: openai/gpt-5.1 publishes default_effort
    // "none", and the Flash-Lite variants publish "minimal".
    //
    // So the advice rides only for an effort this ladder can place
    // strictly above the lowest selectable one. Unknown names are
    // treated as unknown rather than as high.
    const effort = opts.effort;
    const current = EFFORT_SHARES[effort];
    if (
      typeof current === "number" &&
      current > EFFORT_SHARES[LOWEST_SELECTABLE_EFFORT]
    ) {
      parts.push("a lower reasoning effort");
    }
    if (!parts.length) return "";
    // One "try" for the whole clause, so a single remedy reads "try a
    // lower reasoning effort" rather than losing the verb, and a pair
    // reads "try extended budget or a lower reasoning effort".
    return "; try " + parts.join(" or ");
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
    remedyFor,
    LOWEST_SELECTABLE_EFFORT,
    EFFORT_SHARES,
    routeCapFor,
    REASONING_SHARE_EXHAUSTED,
    DIFF_TOKEN_LIMIT,
  };
  if (typeof window !== "undefined") window.BenchLib = BenchLib;
  if (typeof module !== "undefined") module.exports = BenchLib;
})();
