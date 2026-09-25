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
  // THE TWO NUMBERS A SHARE MAY BE COMPUTED FROM. Mirrors
  // exhaustion_pair in bench/models.py, which carries the reasoning and
  // the measured row; a cross-language test executes both against the
  // same shapes so they cannot drift.
  //
  // A ratio needs one unit. OpenRouter publishes two token accounting
  // families and a stream may report in both, so the reasoning count is
  // only commensurable with the completion count from its OWN usage
  // block. reasoningCompletionTokens is that number, captured at
  // ingestion and served on the row; when it is absent, which is every
  // row written before the column existed, the stored completion count
  // is the fallback and the pair is what it has always been.
  function exhaustionPair(
    completionTokens,
    reasoningTokens,
    reasoningCompletionTokens,
  ) {
    return reasoningCompletionTokens != null
      ? [reasoningCompletionTokens, reasoningTokens]
      : [completionTokens, reasoningTokens];
  }

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

  // ---- Composing a dataset (Phase N2).
  //
  // THE BROWSER COMPOSES; THE SERVER VALIDATES. Everything below turns
  // what a person typed into the JSONL the server's parser reads, counts
  // what the page shows against the server's ceilings, and names the few
  // omissions plain enough to grey Store. None of it decides whether a
  // dataset is valid: bench/datasets.py's parse_dataset does, and its
  // sentence is what the page prints.

  // Mirrors of the server's bounds, for display and for the courtesy of
  // greying Store before a refusal that would certainly come. They are
  // the SERVER's numbers, never the page's: tests/test_api.py executes
  // this file with node and asserts each equals the constant it names,
  // because a mirror tighter than the server would refuse a dataset the
  // door would store and blame the person for it.
  const DATASET_LIMITS = {
    maxTasks: 2000, // bench.datasets.MAX_TASKS
    maxPromptChars: 100000, // bench.datasets.MAX_PROMPT_CHARS
    maxDatasetBytes: 1912831, // bench.main.MAX_DATASET_BYTES
    maxNameChars: 255, // bench.main.MAX_DATASET_NAME_CHARS
  };

  // The builder's own bound, which the server does not have: past this
  // many rows a person composing by hand is better served pasting or
  // uploading JSONL. 50, as proposed in the commission and confirmed at
  // the N2 checkpoint.
  const BUILDER_MAX_ROWS = 50;

  // bench.datasets.SCORERS, in its order; the same test holds the pair.
  const DATASET_SCORERS = [
    "exact",
    "normalized_exact",
    "contains",
    "regex",
    "judge",
  ];
  const COMPARING_SCORERS = ["exact", "normalized_exact", "contains"];

  // Characters the way Python counts them, one per code point. The
  // server's bounds are len() over a str, and a string's length in this
  // language counts UTF-16 units, so an emoji is two here and one there;
  // measuring with .length would call a legal prompt over the limit.
  function codePoints(text) {
    return Array.from(text).length;
  }

  // Bytes as the server will store them: UTF-8.
  function utf8Length(text) {
    return new TextEncoder().encode(text).length;
  }

  // Python's whitespace: exactly the characters for which str.isspace()
  // is true, as the inside of a regular expression character class.
  const PY_SPACE =
    "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
  const PY_BLANK = new RegExp("^[" + PY_SPACE + "]*$");

  // Whether text is empty or whitespace as Python reads whitespace.
  //
  // NOT String.prototype.trim, which disagrees with str.strip in both
  // directions: trim removes U+FEFF, which strip keeps, and strip removes
  // U+001C to U+001F and U+0085, which trim keeps. Most checks that use
  // this stand in for a strip() on the server: composeTask's and
  // rowNudge's reference and rubric checks, datasetNudge's name and
  // no-tasks checks, and the lines countLines skips. There a trim()
  // would nudge a U+FEFF rubric the server takes, and a nudge must never
  // grey Store on a dataset the server would store. composeTask's system
  // and threshold checks are the page's own rule that blank is not sent;
  // the server runs no strip() on either, and composeTask says what each
  // does.
  function isBlank(text) {
    return PY_BLANK.test(text);
  }

  // A pass threshold that is a plain decimal numeral, with Python
  // whitespace around it, capturing the numeral; see composeTask.
  const DECIMAL = new RegExp(
    "^[" +
      PY_SPACE +
      "]*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)[" +
      PY_SPACE +
      "]*$",
  );

  // One row of the builder as the task object it declares.
  //
  // RULE ONE: A KEY IS WRITTEN ONLY WHEN IT WAS SET. A blank id box is an
  // empty id box and sends no id; the server then says the id is
  // required, which is the truth. Only the chosen scorer's fields are
  // written, so a reference typed before switching to regex does not ride
  // along as a declaration nobody meant.
  //
  // WHAT COUNTS AS BLANK, field by field. id, prompt and pattern are
  // blank only when empty, because "  " is a legal id, prompt and
  // pattern to the parser. reference and rubric are blank when isBlank,
  // which is the server's own test: the parser refuses a whitespace
  // reference or rubric as missing, so leaving one out changes no
  // verdict. system and threshold are blank when isBlank too, and that
  // is the page's rule, not the server's. The parser keeps a whitespace
  // system message, and for that task it would replace the experiment's
  // system prompt; a system message of Python whitespace is not one
  // anybody typed on purpose, so it is not sent and the experiment's own
  // system prompt applies. (The controls panel reads its system prompt
  // with trim(), so the two differ on U+FEFF, U+0085 and U+001C to
  // U+001F; the builder follows the server's reading of whitespace.) A
  // threshold of Python whitespace is not sent either, so the judge's
  // score stands alone, as it does for an empty box.
  //
  // A PASS THRESHOLD IS SENT AS A NUMBER ONLY WHEN IT IS A PLAIN DECIMAL
  // NUMERAL ("0.5", ".5", "1.", "+0.5", "5e-1"), Python whitespace around
  // it aside, and as the typed text otherwise, so the server's sentence
  // ("must be a number") is what a person sees for "0,5". NOT Number(),
  // which reads "0x1" as 1 and a box holding only U+FEFF as 0: a page
  // that stored a threshold nobody typed would pass trials nobody chose
  // to pass. A numeral too large to be finite is sent as text too,
  // because JSON.stringify writes Infinity as null. The grammar has one
  // way to read each numeral, so a long run of digits is read in linear
  // time on every keystroke rather than backtracked over.
  function composeTask(row) {
    const task = {};
    if (row.id !== "") task.id = row.id;
    if (row.prompt !== "") task.prompt = row.prompt;
    if (!isBlank(row.system)) task.system = row.system;
    const kind = row.scorer;
    if (COMPARING_SCORERS.includes(kind) && !isBlank(row.reference)) {
      task.reference = row.reference;
    }
    if (kind === "judge" && !isBlank(row.rubric)) task.rubric = row.rubric;
    if (kind !== "") {
      const scorer = { kind: kind };
      if (kind === "regex" && row.pattern !== "") scorer.pattern = row.pattern;
      if (kind === "judge" && !isBlank(row.threshold)) {
        const numeral = DECIMAL.exec(row.threshold);
        const value = numeral ? Number(numeral[1]) : Number.NaN;
        scorer.pass_threshold = Number.isFinite(value) ? value : row.threshold;
      }
      task.scorer = scorer;
    }
    if (row.documents.length > 0) task.attachments = row.documents.slice();
    return task;
  }

  // The builder's rows as the text Store sends: one line per row, in
  // order, no blank lines, a newline after each. So row N is line N, and
  // a refusal naming line N names row N. JSON.stringify escapes every
  // "\n" inside a value, and the parser ends a line at "\n" and nowhere
  // else, so a prompt holding U+2028 (which JSON.stringify writes raw) is
  // one line and is stored as it was typed.
  function composeJsonl(rows) {
    return rows.map((row) => JSON.stringify(composeTask(row)) + "\n").join("");
  }

  // Why this row greys Store, or null. A COURTESY AND NOT A RULE: every
  // reason here is one the server would refuse anyway, which the
  // nudge-subset test in tests/test_api.py proves by sending what each
  // nudged row of its grid composes to POST /datasets, a prompt one code
  // point past the ceiling included, and tests/browser/test_n.py proves
  // again in the page. The server refuses more than this names (a
  // pattern that does not compile, a threshold that is not a number or
  // is outside 0 to 1), and the same grid asserts that a row whose only
  // fault is one of those is not nudged and is refused by the server, in
  // its own words. A duplicate id spans two rows, so it is the browser
  // test of a refusal beside its row that leaves that one to the server.
  function rowNudge(row) {
    if (row.id === "") return "needs an id";
    if (row.prompt === "") return "needs a prompt";
    const chars = codePoints(row.prompt);
    if (chars > DATASET_LIMITS.maxPromptChars) {
      return (
        "prompt is " +
        chars +
        " characters, over the " +
        DATASET_LIMITS.maxPromptChars +
        " limit"
      );
    }
    if (COMPARING_SCORERS.includes(row.scorer) && isBlank(row.reference)) {
      return "needs a reference for " + row.scorer;
    }
    if (row.scorer === "regex" && row.pattern === "") return "needs a pattern";
    if (row.scorer === "judge" && isBlank(row.rubric)) {
      return "needs a rubric for the judge";
    }
    return null;
  }

  // Why the dataset as a whole greys Store, or null. The same standing
  // as rowNudge: each is a refusal the server would issue.
  function datasetNudge(name, content) {
    if (isBlank(name)) return "name the dataset";
    const chars = codePoints(name);
    if (chars > DATASET_LIMITS.maxNameChars) {
      return (
        "the name is " +
        chars +
        " characters, over the " +
        DATASET_LIMITS.maxNameChars +
        " limit"
      );
    }
    if (isBlank(content)) return "there are no tasks to store";
    const bytes = utf8Length(content);
    if (bytes > DATASET_LIMITS.maxDatasetBytes) {
      return (
        "the dataset is " +
        bytes +
        " bytes, over the " +
        DATASET_LIMITS.maxDatasetBytes +
        " byte limit for a stored one; a larger dataset goes by path"
      );
    }
    return null;
  }

  // The line a server refusal names, or null. parse_dataset writes
  // "line N: " at the start of every sentence about a line, and the
  // store door writes the same for a half surrogate pair.
  function refusalLine(detail) {
    if (typeof detail !== "string") return null;
    const match = /^line (\d+): /.exec(detail);
    return match ? Number(match[1]) : null;
  }

  // Lines holding anything, for a pasted or uploaded file's label and its
  // task-line ceiling. SPLIT AND SKIPPED AS THE PARSER SPLITS AND SKIPS:
  // parse_dataset ends a line at "\n" and nowhere else (the "\r" a CRLF
  // file leaves at a line's end is whitespace there, and a line of it
  // alone is blank) and skips one that is blank to str.strip, and
  // counted any other way the
  // label would disagree with the task count the server stores. A count
  // and no more: the page does not parse what it did not compose.
  function countLines(text) {
    return text.split("\n").filter((line) => !isBlank(line)).length;
  }

  // A refusal body as text. FastAPI answers a model violation with a LIST
  // of error objects and this application's own refusals with a string,
  // and assigning the list to textContent printed "[object Object]" at a
  // person the server had told exactly what was wrong. Moved here from
  // attach.js when a second control needed it, so there is one reading
  // of a refusal rather than two.
  function refusalText(detail, fallback) {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => (item && typeof item.msg === "string" ? item.msg : ""))
        .filter((msg) => msg !== "");
      if (messages.length > 0) return messages.join("; ");
    }
    return fallback;
  }

  // ---- The experiment lifecycle (Phase N3).
  //
  // THE BROWSER COMPOSES; THE SERVER VALIDATES, as in the builder. These
  // turn what the experiment form holds into the body POST /experiments
  // reads, name the few omissions plain enough to grey Create, and say
  // what a projection means. None of them decides whether an experiment
  // is valid: create_experiment does, and its sentence is what the page
  // prints.

  // Whether a controls object holds anything at all. Rule one lives here
  // on the client side: an empty controls set must produce a body with no
  // params key, not a body carrying an empty object, because the server
  // stores the two identically but the wire should not have to be trusted
  // to collapse them. Moved here from stream.js when the experiment form
  // needed it, so the composer and the form read blank controls one way.
  function hasControls(controls) {
    return controls != null && Object.keys(controls).length > 0;
  }

  // Mirrors of the ExperimentCreate bounds the form shows or checks, the
  // SERVER's numbers: tests/test_api.py executes this file with node and
  // holds each equal to the constant it names, and the markup's min and
  // max for repeats and the task order seed to the model's, for the
  // reason DATASET_LIMITS gives.
  const EXPERIMENT_LIMITS = {
    maxNameChars: 200, // bench.main.ExperimentCreate.name max_length
    maxRepeats: 20, // bench.main.MAX_REPEATS
    maxSeed: 9007199254740991, // bench.main.MAX_SEED
    // How many GET /experiments lists, newest first: the default limit of
    // bench.store.list_experiments, which the door passes on unchanged.
    listed: 100,
  };

  // Whether an experiment has finished: every status but these two is
  // terminal, and the progress door closes its stream on one.
  function experimentFinished(status) {
    return status !== "created" && status !== "running";
  }

  // A whole-number box as the body sends it. BLANK IS NOT SENT, so an
  // empty box is absent, and the server's default (one repeat, file
  // order) is what the record says. Otherwise a number when the box reads
  // as a whole number JavaScript holds exactly, and the text when it does
  // not, so the server's own sentence says what is wrong with it.
  function wholeNumberField(text) {
    if (isBlank(text)) return undefined;
    const value = Number(text);
    return Number.isSafeInteger(value) ? value : text;
  }

  // The body POST /experiments is sent, from what the form holds.
  //
  // RULE ONE: A KEY IS WRITTEN ONLY WHEN IT WAS SET. No params key for a
  // blank controls panel, no repeats or task_order_seed for an empty
  // box, no primary_metric when none is declared, and no attachments_mode
  // while that control is disabled (the dataset cites no document, so the
  // server refuses native and inline is its default). estimand_mode and
  // halt_on_refusal
  // are always written: a select and a checkbox have no blank state, so
  // what they show is what the person left them saying.
  //
  // THE DATASET IS NAMED BY DIGEST, never by path: the form creates only
  // from a stored dataset, and the digest is the identity the record
  // cites.
  function experimentBody(form) {
    const body = {
      name: form.name,
      dataset_digest: form.digest,
      lineup: form.lineup.slice(),
      budget: form.budget,
    };
    if (hasControls(form.params)) body.params = form.params;
    const repeats = wholeNumberField(form.repeats);
    if (repeats !== undefined) body.repeats = repeats;
    const seed = wholeNumberField(form.seed);
    if (seed !== undefined) body.task_order_seed = seed;
    body.estimand_mode = form.estimand;
    if (form.attachments !== null) body.attachments_mode = form.attachments;
    if (form.metric !== "") body.primary_metric = form.metric;
    body.halt_on_refusal = form.halt;
    return body;
  }

  // Why Create is greyed, or null. A COURTESY AND NOT A RULE, as the
  // builder's nudges are, and of two kinds. Most are a request the server
  // would refuse: an empty name, a name over its bound, no dataset, no
  // model, and a control or box out of its range, whose typed value the
  // body carries and the door refuses. One is the page's own: a box whose
  // text is not a number at all ("1e") reads as empty, so the body would
  // drop what was typed without a word and the server would create
  // without it; the page waits rather than create something other than
  // what was typed. A name of spaces is legal to the server and is not
  // nudged.
  function experimentNudge(form) {
    if (form.name === "") return "name the experiment";
    const chars = codePoints(form.name);
    if (chars > EXPERIMENT_LIMITS.maxNameChars) {
      return (
        "the name is " +
        chars +
        " characters, over the " +
        EXPERIMENT_LIMITS.maxNameChars +
        " limit"
      );
    }
    if (!form.digest) return "select a stored dataset in Datasets above";
    if (form.lineup.length === 0) return "check a model in the lineup above";
    if (form.invalid.length > 0) return "check " + form.invalid.join(", ");
    return null;
  }

  // A list row's counts. Progress is trials FINISHED, the three disjoint
  // buckets added up: showing trials_done alone would leave an experiment
  // whose trials are failing looking stuck rather than failing, and the
  // two are the opposite of each other to act on. The failed and refused
  // are named beside the sum.
  function experimentRowMeta(experiment) {
    const finished =
      experiment.trials_done +
      experiment.trials_failed +
      experiment.trials_refused;
    const trouble = [];
    if (experiment.trials_failed > 0) {
      trouble.push(experiment.trials_failed + " failed");
    }
    if (experiment.trials_refused > 0) {
      trouble.push(experiment.trials_refused + " refused");
    }
    return (
      experiment.status +
      " · " +
      finished +
      "/" +
      experiment.trials_total +
      " trials" +
      (trouble.length ? " (" + trouble.join(", ") + ")" : "")
    );
  }

  // A projection as the README describes it: the output figure a ceiling
  // on tokens (not on the bill), the input figure an estimate, and a
  // total only when both halves exist. WHEN ANY LINEUP MEMBER IS UNPRICED
  // EVERY FIGURE IS NULL and the members are named, verbatim, because a
  // total missing one arm of a comparison reads as the comparison's
  // total. A null figure with nobody unpriced (native mode, where images
  // are not estimated) says what is missing instead of a total.
  function projectionText(projection) {
    if (!projection) return "";
    const usd = (value) => "$" + fmtEstimate(value);
    if (projection.unpriced.length > 0) {
      return (
        "unpriced: " +
        projection.unpriced.join(", ") +
        ". No figure is given, because a total missing one arm would " +
        "read as the whole comparison's total."
      );
    }
    const parts = [];
    if (projection.output_usd !== null) {
      parts.push(
        "output at most " +
          usd(projection.output_usd) +
          " (a ceiling on tokens, not on the bill)",
      );
    }
    if (projection.input_usd !== null) {
      parts.push("input about " + usd(projection.input_usd) + " (an estimate)");
    }
    if (projection.total_usd !== null) {
      parts.push("total " + usd(projection.total_usd));
    } else if (parts.length > 0) {
      parts.push("no input estimate, so no total");
    }
    if (parts.length === 0) {
      return "no projection: the figures could not be computed";
    }
    return parts.join(" · ");
  }

  // ---- Scoring (Phase N4).

  // A dataset summary as the experiment panel holds it for an
  // experiment's RECORDED digest, never for the Datasets selection: the
  // summary object GET /datasets/{digest} served (held), false when the
  // store answered 404 (the experiment was read from a file by path), a
  // string when the door refused the bytes it holds with a sentence (they
  // no longer hash to their digest; the Score door
  // would refuse them the same way), null while the question is out,
  // undefined when it is not known (the question failed and is asked
  // again at the next selection).
  function datasetHeld(summary) {
    return summary !== null && typeof summary === "object";
  }

  function judgeTasks(summary) {
    return datasetHeld(summary) && summary.scorers.includes("judge");
  }

  // The body POST /experiments/{id}/score is sent. RULE ONE: a key only
  // when set. The dataset is named by the digest the experiment recorded,
  // and judge_model is sent only when that dataset has judge tasks AND a
  // judge was chosen. The door accepts a judge for a dataset with no judge
  // tasks and records nothing of it, so the page keeps one off the wire:
  // the select is hidden for such a dataset and the choice is cleared
  // when the selection moves, and this rule holds the body to it whatever
  // state the page is in.
  function scoreBody(digest, summary, judge) {
    const body = { dataset_digest: digest };
    if (judgeTasks(summary) && judge !== "") body.judge_model = judge;
    return body;
  }

  // The reason the page waits for a judge, said where it waits: the door
  // would accept the pass, and what it would record cannot be taken back.
  const JUDGE_LESS_PASS =
    "because a pass without one records every judge task as a scoring " +
    "failure, and that record does not rewrite";

  // Why Score is greyed, or null. TWO OF THESE ARE REFUSALS AND THREE ARE
  // NOT, which is the opposite of the Create nudges and is deliberate.
  // The door refuses the unstored digest and bytes it cannot cite, in its
  // own words, and the second is its sentence as the dataset door gave
  // it. It accepts the other three (reading, not known, no judge chosen),
  // and a pass it accepts cannot be taken back: a judge-less pass over
  // judge tasks records "no judge model was given" for every one of
  // them, for good, since records never rewrite. Without the summary the
  // page cannot tell whether a judge is needed, so it waits rather than
  // guess; the commission rules the judge required. catalogState is
  // BenchControls.catalogIds().state, so the page does not ask for a
  // judge it cannot offer.
  function scoreNudge(summary, judge, catalogState) {
    if (summary === null) return "reading which scorers its dataset declares";
    if (summary === undefined) {
      return "its dataset could not be read; Retry asks the store again";
    }
    if (typeof summary === "string") return summary;
    if (summary === false) {
      return (
        "its dataset was read from a file and is not stored here, so the " +
        "page cannot name it; score it through the API with its " +
        "dataset_path, and judge_model if any of its tasks is judged"
      );
    }
    if (judgeTasks(summary) && judge === "") {
      if (catalogState === "pending") {
        return "choose a judge once the catalog has loaded, " + JUDGE_LESS_PASS;
      }
      if (catalogState === "unavailable") {
        return (
          "no judge can be chosen here, because the catalog is not " +
          "available; score it through the API with judge_model"
        );
      }
      return "choose a judge, " + JUDGE_LESS_PASS;
    }
    return null;
  }

  // What Score's label says it spends. A judge's calls are billed against
  // the same ceiling as trials; a pass of deterministic scorers calls no
  // model. Plain "Score" until the summary says which it is.
  function scoreLabel(summary) {
    if (!datasetHeld(summary)) return "Score";
    return judgeTasks(summary) ? "Score · pays the judge" : "Score · free";
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
    exhaustionPair,
    reasoningShare,
    remedyFor,
    LOWEST_SELECTABLE_EFFORT,
    EFFORT_SHARES,
    routeCapFor,
    REASONING_SHARE_EXHAUSTED,
    DIFF_TOKEN_LIMIT,
    DATASET_LIMITS,
    BUILDER_MAX_ROWS,
    DATASET_SCORERS,
    isBlank,
    codePoints,
    utf8Length,
    composeTask,
    composeJsonl,
    rowNudge,
    datasetNudge,
    refusalLine,
    countLines,
    refusalText,
    hasControls,
    EXPERIMENT_LIMITS,
    experimentFinished,
    experimentBody,
    experimentNudge,
    projectionText,
    experimentRowMeta,
    datasetHeld,
    judgeTasks,
    scoreBody,
    scoreNudge,
    scoreLabel,
  };
  if (typeof window !== "undefined") window.BenchLib = BenchLib;
  if (typeof module !== "undefined") module.exports = BenchLib;
})();
