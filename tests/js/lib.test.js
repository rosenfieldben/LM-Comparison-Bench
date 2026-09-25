// Pure-function tests for static/lib.js, run by `node --test tests/js/`.
// No build step and no dependencies: lib.js is required directly through
// its CommonJS guard.

const test = require("node:test");
const assert = require("node:assert");

const {
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
  codePoints,
  composeTask,
  composeJsonl,
  rowNudge,
  datasetNudge,
  refusalLine,
  countLines,
  refusalText,
  hasControls,
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
} = require("../../static/lib.js");

test("shortName strips the vendor prefix, keeping the rest", () => {
  assert.equal(shortName("deepseek/deepseek-chat"), "deepseek-chat");
  assert.equal(shortName("no-slash"), "no-slash");
  assert.equal(shortName("a/b/c"), "b/c");
});

test("fmtCost is exponential below a cent and fixed decimals above", () => {
  assert.equal(fmtCost(3.1e-5), "~$3.1e-5");
  assert.equal(fmtCost(0.005), "~$5.0e-3");
  // Exactly a cent is the boundary: no longer exponential.
  assert.equal(fmtCost(0.01), "~$0.010");
  assert.equal(fmtCost(1.5), "~$1.500");
});

test("fmtBilled carries the same digits without the estimate marker", () => {
  // The tilde is the only difference, and it is the whole point: a billed
  // figure that renders with one would claim to be catalog arithmetic.
  for (const amount of [3.1e-5, 0.005, 0.01, 1.5, 4.9e-5]) {
    assert.equal(fmtBilled(amount), fmtCost(amount).slice(1));
    assert.ok(!fmtBilled(amount).includes("~"));
  }
  assert.equal(fmtBilled(3.1e-5), "$3.1e-5");
  assert.equal(fmtBilled(1.5), "$1.500");
});

test("fmtBilled keeps a tiny nonzero charge from reading as zero", () => {
  // Same significant-digit contract as fmtCost: real charges on this bench
  // sit below a hundredth of a cent, where fixed decimals floor to zero.
  for (const tiny of [4.9e-5, 1e-6, 1e-12]) {
    const shown = fmtBilled(tiny);
    assert.notEqual(shown, "$0.0000");
    assert.match(shown, /[1-9]/, `${shown} must carry a nonzero digit`);
  }
});

test("fmtEstimate rounds large values and trims small ones", () => {
  assert.equal(fmtEstimate(150), "150");
  assert.equal(fmtEstimate(0.0021), "0.0021");
});

test("niceScale is the smallest 100*2^k at or above the input", () => {
  assert.equal(niceScale(0), 100);
  assert.equal(niceScale(100), 100);
  assert.equal(niceScale(101), 200);
  assert.equal(niceScale(655), 800);
});

test("tokenizeDiff keeps spacing in raw and keys on the bare word", () => {
  const toks = tokenizeDiff("a b");
  assert.deepEqual(
    toks.map((t) => t.key),
    ["a", "b"],
  );
  assert.equal(toks.map((t) => t.raw).join(""), "a b");
  assert.deepEqual(tokenizeDiff(""), []);
});

test("diffTokens marks deletions, insertions, and shared text", () => {
  const ops = diffTokens(
    tokenizeDiff("the quick fox"),
    tokenizeDiff("the slow fox"),
  );
  const sames = ops.filter((o) => o.op === "same").map((o) => o.raw.trim());
  assert.ok(sames.includes("the"));
  assert.ok(sames.includes("fox"));
  assert.ok(ops.some((o) => o.op === "del" && o.raw.includes("quick")));
  assert.ok(ops.some((o) => o.op === "ins" && o.raw.includes("slow")));
});

test("diffTokens on identical input is all 'same'", () => {
  const ops = diffTokens(
    tokenizeDiff("same text here"),
    tokenizeDiff("same text here"),
  );
  assert.ok(ops.every((o) => o.op === "same"));
});

test("diffTokens on empty inputs yields no ops", () => {
  assert.deepEqual(diffTokens([], []), []);
});

test("the size-notice threshold is a number tokenize can exceed", () => {
  assert.equal(DIFF_TOKEN_LIMIT, 4000);
  const big = tokenizeDiff("word ".repeat(DIFF_TOKEN_LIMIT + 1));
  assert.ok(big.length > DIFF_TOKEN_LIMIT);
});

test("fmtCost keeps a tiny nonzero total from reading as zero", () => {
  // F4.6: the session bar used toFixed(4), so a real total below $0.00005
  // floored to "~$0.0000" and a session that had spent money reported
  // zero. fmtCost carries significant digits instead, at any magnitude.
  for (const tiny of [4.9e-5, 1e-6, 2.9e-5, 1e-12]) {
    const shown = fmtCost(tiny);
    assert.notEqual(shown, "~$0.0000");
    assert.match(shown, /[1-9]/, `${shown} must carry a nonzero digit`);
  }
  // The specific boundary the defect sat on: half a hundredth of a cent.
  assert.equal((4.9e-5).toFixed(4), "0.0000");
  assert.equal(fmtCost(4.9e-5), "~$4.9e-5");
});

test("an exact zero reads as zero, not as an exponent", () => {
  // Free models make a genuinely zero charge reachable, and the idle bar
  // shows the same string, so the two must not disagree on how nothing
  // is written.
  assert.equal(fmtBilled(0), "$0.0000");
  assert.equal(fmtCost(0), "~$0.0000");
});

test("controlBadges renders a badge for a set control and only a set one", () => {
  assert.deepEqual(
    controlBadges({ temperature: 0.2, seed: 7 }).map((b) => b.text),
    ["t=0.2", "seed 7"],
  );
  // Fixed order regardless of key order, so two comparisons with the same
  // controls read the same way.
  assert.deepEqual(
    controlBadges({
      routing: "price",
      seed: 7,
      system: "be terse",
      effort: "high",
      top_p: 0.9,
      temperature: 0.2,
    }).map((b) => b.text),
    ["sys", "t=0.2", "top_p=0.9", "seed 7", "effort high", "route price"],
  );
});

test("controlBadges renders nothing when nothing was set", () => {
  // Every absence spelling the server can produce, plus a controls object
  // whose keys are all explicitly absent.
  assert.deepEqual(controlBadges(null), []);
  assert.deepEqual(controlBadges(undefined), []);
  assert.deepEqual(controlBadges({}), []);
  assert.deepEqual(
    controlBadges({ temperature: null, seed: undefined, system: "" }),
    [],
  );
});

test("controlBadges treats zero as a value, not as blankness", () => {
  // temperature 0 is the most deliberate setting there is: a falsy skip
  // test would drop exactly the control that matters most.
  assert.deepEqual(
    controlBadges({ temperature: 0, top_p: 0, seed: 0 }).map((b) => b.text),
    ["t=0", "top_p=0", "seed 0"],
  );
});

test("controlBadges shows the system prompt as presence, not as text", () => {
  // The row is one line, and a truncated prompt would invite comparing two
  // comparisons on an excerpt that happens to match. The text belongs to
  // the comparison view.
  const [badge] = controlBadges({ system: "You are a helpful assistant." });
  assert.equal(badge.text, "sys");
  assert.ok(!badge.text.includes("helpful"));
  assert.match(badge.title, /system prompt was set/);
});

test("controlBadges gives every badge a long-form title", () => {
  const titles = controlBadges({
    temperature: 0.2,
    top_p: 0.9,
    seed: 7,
    effort: "low",
    routing: "default",
  }).map((b) => b.title);
  assert.deepEqual(titles, [
    "temperature 0.2",
    "top_p 0.9",
    "seed 7",
    "reasoning effort low",
    "provider routing: default",
  ]);
});

test("minRanks lets equal values share a place and skips the next", () => {
  // Two models that measured identically are tied, and numbering them
  // anyway would show a difference that is not in the data.
  assert.deepEqual(minRanks([10, 20, 20, 30]), [1, 2, 2, 4]);
  assert.deepEqual(minRanks([5, 5, 5]), [1, 1, 1]);
  assert.deepEqual(minRanks([1, 2, 3]), [1, 2, 3]);
  assert.deepEqual(minRanks([]), []);
  assert.deepEqual(minRanks([7]), [1]);
});

test("minRanks ties at the top so every tied leader is fastest", () => {
  // The race highlights rank 1. With a tie at the front that must be
  // every tied row, not whichever one the sort happened to put first.
  const ranks = minRanks([12.5, 12.5, 40]);
  assert.equal(ranks.filter((r) => r === 1).length, 2);
});

// ---- Attachment chip formatting (Phase K4).

test("fmtBytes uses binary units, matching the MiB the cap is stated in", () => {
  assert.equal(fmtBytes(0), "0 B");
  assert.equal(fmtBytes(512), "512 B");
  assert.equal(fmtBytes(1024), "1.0 KiB");
  assert.equal(fmtBytes(9.4 * 1024), "9.4 KiB");
  // Ten and above drops the decimal, so a chip never carries three
  // digits of noise.
  assert.equal(fmtBytes(512 * 1024), "512 KiB");
  assert.equal(fmtBytes(8 * 1024 * 1024), "8.0 MiB");
  // The upload cap, formatted, must read as the same quantity the
  // server's refusal quotes. A decimal-unit formatter would say
  // "8.4 MB" here and the pair would look like a contradiction.
  assert.equal(fmtBytes(1024 * 1024 * 1024), "1.0 GiB");
});

test("fmtBytes returns nothing for a value it cannot describe", () => {
  // A ref whose row is gone carries byte_size null, and the chip asks
  // for a string it can concatenate rather than the word "NaN".
  assert.equal(fmtBytes(null), "");
  assert.equal(fmtBytes(undefined), "");
  assert.equal(fmtBytes(-1), "");
  assert.equal(fmtBytes(Number.NaN), "");
});

test("shortDigest is git's seven characters and never widens", () => {
  const digest = "a".repeat(64);
  assert.equal(shortDigest(digest).length, 7);
  assert.equal(shortDigest("abcdef0123456789"), "abcdef0");
  // Absent digest is unreachable through the API, but the chip must not
  // throw its way out of rendering a row.
  assert.equal(shortDigest(null), "");
});

test("approxTokens rounds up, so a short document never estimates zero", () => {
  assert.equal(approxTokens(0), 0);
  // One character is not zero tokens. Rounding down here would let a
  // one-line document read as costing nothing.
  assert.equal(approxTokens(1), 1);
  assert.equal(approxTokens(4), 1);
  assert.equal(approxTokens(5), 2);
  assert.equal(approxTokens(4000), 1000);
  assert.equal(approxTokens(null), 0);
});

// ---- The reasoning-exhaustion fix, R2 and R4: the shape test the card
// ---- draws its remedy and its indicator from.

// Every case is a token shape. None of them knows what produced it,
// which is the property under test as much as any single answer is.
const EXHAUSTION_SHAPES = [
  ["the incident, exactly equal", 21350, 21350, true],
  ["a different tier, same shape", 8192, 8100, true],
  ["exactly at the threshold", 1000, 900, true],
  ["one token under the threshold", 1000, 899, false],
  ["a healthy answer, row 694's shape", 3400, 2944, false],
  ["no thinking at all", 500, 0, false],
  ["a provider that reported no usage", null, null, false],
  ["a history row from before the column existed", 500, null, false],
  ["reasoning without a completion count", null, 5000, false],
  ["both zero", 0, 0, false],
];

test("reasoningAteTheOutput keys on the two counts and nothing else", () => {
  for (const [shape, completion, reasoning, expected] of EXHAUSTION_SHAPES) {
    assert.equal(reasoningAteTheOutput(completion, reasoning), expected, shape);
  }
});

test("the threshold is the one the server uses", () => {
  // The rule is written twice, once per language, because the server can
  // only label a result it synthesized an error for while the card's
  // indicator must also fire on a result that came back with text. The
  // Python side asserts this file's text; this asserts the value.
  assert.equal(REASONING_SHARE_EXHAUSTED, 0.9);
});

test("row 694's healthy shape is not mistaken for exhaustion", () => {
  // The contrast case from the incident's own database: 2944 reasoning
  // tokens, a real answer, end_turn. A rule that fired here would put a
  // warning on every reasoning model that worked.
  assert.equal(reasoningAteTheOutput(3400, 2944), false);
  // And it stays false right up to the boundary.
  assert.equal(reasoningAteTheOutput(3400, 3059), false);
  assert.equal(reasoningAteTheOutput(3400, 3060), true);
});

// R4: the indicator names the fraction, so the card shows a magnitude
// rather than only a warning.

test("reasoningShare rounds to a whole percent and degrades to null", () => {
  assert.equal(reasoningShare(21350, 21350), 100);
  assert.equal(reasoningShare(8500, 8000), 94);
  assert.equal(reasoningShare(3400, 2944), 87);
  // Rounded, not truncated: 0.945 reads as 95, not 94.
  assert.equal(reasoningShare(1000, 945), 95);
  // A row that predates the reasoning column, and a divide by zero.
  assert.equal(reasoningShare(500, null), null);
  assert.equal(reasoningShare(null, 500), null);
  assert.equal(reasoningShare(0, 0), null);
  // Zero reasoning is a real answer, not an absent one, so it is 0 and
  // not null. The predicate is what decides whether anything is shown.
  assert.equal(reasoningShare(500, 0), 0);
});

test("the indicator's threshold and its number come from one rule", () => {
  // Everything the predicate fires on has a share at or above the
  // threshold, and everything it stays quiet about is below it. Two
  // functions, one boundary; a change to either alone breaks this.
  for (const [shape, completion, reasoning] of [
    ["the incident", 21350, 21350],
    ["answered but mostly thinking", 8500, 8000],
    ["exactly at the threshold", 1000, 900],
  ]) {
    assert.equal(reasoningAteTheOutput(completion, reasoning), true, shape);
    assert.ok(
      reasoningShare(completion, reasoning) >= REASONING_SHARE_EXHAUSTED * 100,
      shape,
    );
  }
  for (const [shape, completion, reasoning] of [
    ["one token under", 1000, 899],
    ["row 694", 3400, 2944],
  ]) {
    assert.equal(reasoningAteTheOutput(completion, reasoning), false, shape);
  }
});

test("the rounded share can read 90 on a card the rule stays quiet about", () => {
  // 899 of 1000 is 0.899, which the predicate rejects, and which rounds
  // to 90 for display. Recorded rather than smoothed over: the two
  // functions answer different questions and only one of them is
  // allowed to be approximate.
  //
  // It is invisible in practice, and that is the point of writing it
  // down. The share is rendered ONLY on a card the predicate fired for,
  // so a reader never sees "90%" beside a card that was left alone.
  // Nothing needs to change here unless the share is ever shown
  // unconditionally, and then this test is where the problem is already
  // described.
  assert.equal(reasoningAteTheOutput(1000, 899), false);
  assert.equal(reasoningShare(1000, 899), 90);
});

// ---- T5: advice that cannot work is not offered.
//
// Every case below is stated against a PINNED RULE rather than against
// what the function happens to do: the clamp comes from
// effective_budget in bench/main.py, and the effort floor from the
// option list in static/index.html.

test("no budget advice when a larger budget does not exist", () => {
  // effective_budget clamps the requested tier to a model's published
  // completion cap. stub/capped publishes 4096, so extended and
  // standard both send 4096 and "try extended budget" is an instruction
  // to spend four times as much on an identical request.
  const standard = { budget: "standard" };
  const clamped = remedyFor(
    { max_tokens: 4096 },
    { ...standard, extendedCap: 4096 },
  );
  assert.ok(!clamped.includes("extended budget"));

  // Same rule, the offline case, which is the one that makes the old
  // advice impossible for EVERY model: with no catalog there are no
  // published caps, so extended falls back to the standard tier.
  const offline = remedyFor(
    { max_tokens: 16384 },
    { ...standard, extendedCap: 16384 },
  );
  assert.ok(!offline.includes("extended budget"));

  // And it IS offered when a larger budget genuinely exists.
  const room = remedyFor(
    { max_tokens: 16384 },
    { ...standard, extendedCap: 65536 },
  );
  assert.ok(room.includes("try extended budget"));

  // EVERY CASE HERE NAMES ITS TIER, and that is not decoration. The
  // budget clause now requires a known tier, so a version of this test
  // that omitted it would assert silence for the wrong reason and pass
  // whatever the cap comparison did.
});

test("review repro: lower-effort advice needs a lower effort to exist", () => {
  // THE FLOOR, asserted against the documented ladder rather than
  // against index.html's option list. A decision to expose "minimal"
  // should move this test rather than pass quietly, and the ladder is
  // what the comparison actually uses.
  assert.equal(LOWEST_SELECTABLE_EFFORT, "low");
  assert.equal(EFFORT_SHARES[LOWEST_SELECTABLE_EFFORT], 0.2);
  for (const below of ["none", "minimal"]) {
    assert.ok(
      EFFORT_SHARES[below] < EFFORT_SHARES[LOWEST_SELECTABLE_EFFORT],
      below,
    );
  }

  // THE DEFECT THIS ROW REPLACES, and the old version of this test
  // asserted it. undefined and "" were listed beside "medium" and
  // "high" as efforts that can be lowered, on the reasoning that
  // choosing nothing is "above the minimum". It is not: with nothing
  // chosen the route runs at its CATALOG default, which the card does
  // not know. openai/gpt-5.1 publishes default_effort "none" and the
  // Flash-Lite variants publish "minimal", both BELOW low, so the
  // advice would have raised reasoning on a card whose whole subject is
  // that reasoning consumed the budget.
  for (const effort of [undefined, "", null, "low", "minimal", "none", "??"]) {
    assert.equal(
      remedyFor({ max_tokens: 4096 }, { extendedCap: 4096, effort }),
      "",
      String(effort),
    );
  }

  // And it IS offered for the two the ladder can place above the floor.
  for (const effort of ["medium", "high"]) {
    const out = remedyFor({ max_tokens: 4096 }, { extendedCap: 4096, effort });
    assert.ok(out.includes("a lower reasoning effort"), effort);
  }
});

test("review repro: a replayed extended run is not told to try extended", () => {
  // THE DEFECT. extendedCap is what the catalog publishes TODAY;
  // result.max_tokens is what this run was SENT, possibly long ago. On
  // a replayed card those are two different points in time, so a run
  // that already selected extended and was clamped to 4096 begins being
  // advised to "try extended budget" the moment the published cap rises
  // above 4096. The advice is to do the thing that was already done.
  const sent = { max_tokens: 4096 };
  const capRose = { extendedCap: 65536, effort: "high" };

  // With the tier recorded as standard, the cap comparison advises it,
  // which is correct: standard really can be raised.
  const asStandard = remedyFor(sent, { ...capRose, budget: "standard" });
  assert.ok(asStandard.includes("try extended budget"));

  // With the stored tier saying extended, only the effort clause
  // survives.
  const asExtended = remedyFor(sent, { ...capRose, budget: "extended" });
  assert.equal(asExtended, "; try a lower reasoning effort");
});

test("review repro: an unknown tier advises no tier at all", () => {
  // THE DEFECT, and the false premise that hid it. An absent tier used
  // to be read as "not extended", so the budget clause rode. The
  // comment defending that said an ungrouped run "predates the
  // declaration entirely" and therefore had no tier to lose.
  //
  // It does not predate anything. /compare takes group_id as OPTIONAL
  // and budget as a current field, so a run created today can select
  // extended and carry no group, and `runs` has no budget column to
  // record it in. Such a run is replayed with the tier missing and was
  // told to select the tier it had already selected.
  //
  // UNKNOWN IS NOT STANDARD. The record cannot say which tier ran, so
  // the card says nothing about tiers. The effort clause is unaffected,
  // which is what makes this a narrowing rather than the advice going
  // dark.
  const sent = { max_tokens: 4096 };
  const capRose = { extendedCap: 65536, effort: "high" };

  assert.equal(remedyFor(sent, capRose), "; try a lower reasoning effort");
  assert.equal(
    remedyFor(sent, { ...capRose, budget: undefined }),
    "; try a lower reasoning effort",
  );
  // A tier this function does not recognise is unknown too, rather
  // than "not extended".
  assert.equal(
    remedyFor(sent, { ...capRose, budget: "enormous" }),
    "; try a lower reasoning effort",
  );
});

test("review repro: a pinned route's ceiling bounds what extended can offer", () => {
  // THE DEFECT, reproduced exactly as the review reported it:
  //
  //   remedyFor({max_tokens:8192},{extendedCap:65536,budget:"standard"})
  //     -> "; try extended budget"
  //
  // while the server's own arithmetic gives
  //
  //   route_budget(16384, 8192) == route_budget(65536, 8192) == 8192
  //
  // extendedCap is a MODEL-level number. A provider-pinned trial is
  // clamped again to the endpoint the pin selected, so a run pinned to
  // a route capping at 8192 was advised to buy a tier that clamps to
  // the same 8192: a replay of the identical request at four times the
  // price.
  const pinned = { max_tokens: 8192 };
  assert.equal(
    remedyFor(pinned, {
      extendedCap: 65536,
      budget: "standard",
      routeCap: 8192,
    }),
    "",
  );
  // WITHOUT the route cap the old answer is still produced, which is
  // what makes the line above a fact about routeCap rather than about
  // this particular pair of numbers.
  assert.ok(
    remedyFor(pinned, { extendedCap: 65536, budget: "standard" }).includes(
      "extended budget",
    ),
  );
  // And a route with genuine headroom is still advised: the rule is
  // min(extendedCap, routeCap) against what was sent, not "pinned runs
  // get no advice".
  assert.ok(
    remedyFor(pinned, {
      extendedCap: 65536,
      budget: "standard",
      routeCap: 32768,
    }).includes("extended budget"),
  );
});

test("routeCapFor learns the route's ceiling only when min picked it", () => {
  // The server sends min(tierCap, routeCap), so a sent ceiling STRICTLY
  // below the model-level cap for that tier can only be the route's own
  // number. Equal means the route did not bind and nothing was learned,
  // which must be null rather than a guess: returning tierCap there
  // would claim a route ceiling for every unpinned run in the bench.
  assert.equal(routeCapFor(8192, 16384), 8192);
  assert.equal(routeCapFor(16384, 16384), null);
  assert.equal(routeCapFor(4096, 4096), null);
  // Absent inputs learn nothing rather than throwing or coercing.
  assert.equal(routeCapFor(undefined, 16384), null);
  assert.equal(routeCapFor(8192, undefined), null);
  assert.equal(routeCapFor(null, null), null);
});

test("both clauses when both remedies exist, joined once", () => {
  const both = remedyFor(
    { max_tokens: 16384 },
    { extendedCap: 65536, effort: "high", budget: "standard" },
  );
  assert.equal(both, "; try extended budget or a lower reasoning effort");
});

test("remedyFor reads the cap the run was sent, never a tier name", () => {
  // THE DEFECT, stated as a test. The old code branched on the selected
  // tier, so a run at the extended tier that clamped down to 4096 was
  // told to try extended, which is where it already was. A tier name
  // is not on the result object at all now, and this asserts the
  // decision follows max_tokens.
  const sameCap = { max_tokens: 4096 };
  assert.ok(
    !remedyFor(sameCap, { extendedCap: 4096, budget: "standard" }).includes(
      "extended budget",
    ),
  );
  assert.ok(
    remedyFor(sameCap, { extendedCap: 65536, budget: "standard" }).includes(
      "extended budget",
    ),
  );

  // Missing information is not an invitation to advise: a caller that
  // cannot say what extended would send gets no budget clause.
  assert.ok(!remedyFor(sameCap, {}).includes("extended budget"));
  assert.ok(!remedyFor({}, { extendedCap: 65536 }).includes("extended budget"));
});

// ---- Phase N2: composing a dataset. The browser composes and the
// ---- server validates; these pin the composing half. The unit suite
// ---- executes the same functions against the real door.

function row(fields) {
  return Object.assign(
    {
      id: "",
      prompt: "",
      system: "",
      scorer: "",
      reference: "",
      pattern: "",
      rubric: "",
      threshold: "",
      documents: [],
    },
    fields,
  );
}

test("composeTask writes a key only when it was set", () => {
  assert.deepEqual(composeTask(row({})), {});
  assert.deepEqual(composeTask(row({ id: "t", prompt: "p", system: "  " })), {
    id: "t",
    prompt: "p",
  });
  // "  " is a legal id and prompt to the parser, so it is sent as typed.
  assert.deepEqual(composeTask(row({ id: " ", prompt: " " })), {
    id: " ",
    prompt: " ",
  });
});

test("composeTask sends only the chosen scorer's fields", () => {
  const typed = row({
    id: "t",
    prompt: "p",
    reference: "r",
    pattern: "x",
    rubric: "g",
    threshold: "0.5",
  });
  assert.deepEqual(composeTask(Object.assign({}, typed, { scorer: "exact" })), {
    id: "t",
    prompt: "p",
    reference: "r",
    scorer: { kind: "exact" },
  });
  assert.deepEqual(composeTask(Object.assign({}, typed, { scorer: "regex" })), {
    id: "t",
    prompt: "p",
    scorer: { kind: "regex", pattern: "x" },
  });
  assert.deepEqual(composeTask(Object.assign({}, typed, { scorer: "judge" })), {
    id: "t",
    prompt: "p",
    rubric: "g",
    scorer: { kind: "judge", pass_threshold: 0.5 },
  });
  assert.deepEqual(composeTask(Object.assign({}, typed, { scorer: "" })), {
    id: "t",
    prompt: "p",
  });
});

test("a threshold that is not a number is sent as typed, for the server to refuse", () => {
  const task = composeTask(
    row({
      id: "t",
      prompt: "p",
      scorer: "judge",
      rubric: "g",
      threshold: "0,5",
    }),
  );
  assert.equal(task.scorer.pass_threshold, "0,5");
  assert.equal(
    composeTask(
      row({
        id: "t",
        prompt: "p",
        scorer: "judge",
        rubric: "g",
        threshold: "0",
      }),
    ).scorer.pass_threshold,
    0,
  );
});

test("composeJsonl writes one line per row, U+2028 and its kin as typed", () => {
  const kin =
    String.fromCharCode(0x2028) +
    String.fromCharCode(0x2029) +
    String.fromCharCode(0x85);
  const text = composeJsonl([
    row({ id: "a", prompt: "x" + kin + "y" }),
    row({ id: "b", prompt: "two", documents: ["d".repeat(64)] }),
  ]);
  assert.equal(text.split("\n").length, 3);
  assert.ok(text.endsWith("\n"));
  // Written raw: the parser ends a line at "\n" alone, so nothing needs
  // escaping, and a workaround left in place would outlive its defect.
  assert.ok(text.includes(kin));
  const lines = text
    .trimEnd()
    .split("\n")
    .map((line) => JSON.parse(line));
  assert.equal(lines[0].prompt, "x" + kin + "y");
  assert.deepEqual(lines[1].attachments, ["d".repeat(64)]);
  assert.equal(composeJsonl([]), "");
});

test("rowNudge names what the server would refuse, first reason first", () => {
  assert.equal(rowNudge(row({})), "needs an id");
  assert.equal(rowNudge(row({ id: "t" })), "needs a prompt");
  assert.equal(
    rowNudge(row({ id: "t", prompt: "p", scorer: "contains", reference: " " })),
    "needs a reference for contains",
  );
  assert.equal(
    rowNudge(row({ id: "t", prompt: "p", scorer: "regex" })),
    "needs a pattern",
  );
  assert.equal(
    rowNudge(row({ id: "t", prompt: "p", scorer: "judge", rubric: "  " })),
    "needs a rubric for the judge",
  );
  // A pattern of one space is a pattern; the parser takes it.
  assert.equal(
    rowNudge(row({ id: "t", prompt: "p", scorer: "regex", pattern: " " })),
    null,
  );
  // Counted in code points, so an emoji is one, as it is to the server.
  assert.equal(
    rowNudge(row({ id: "t", prompt: "\u{1F600}".repeat(100000) })),
    null,
  );
  assert.match(
    rowNudge(row({ id: "t", prompt: "p".repeat(100001) })),
    /over the 100000 limit/,
  );
});

test("datasetNudge greys Store for a name or a size the door refuses", () => {
  const task = '{"id": "t", "prompt": "p"}\n';
  assert.equal(datasetNudge("", task), "name the dataset");
  assert.equal(datasetNudge("  ", task), "name the dataset");
  assert.match(datasetNudge("n".repeat(256), task), /over the 255 limit/);
  assert.equal(datasetNudge("n", " \n "), "there are no tasks to store");
  assert.match(datasetNudge("n", "x".repeat(1912832)), /goes by path/);
  assert.equal(datasetNudge("n", task), null);
  assert.equal(codePoints("\u{1F600}"), 1);
});

test("refusalLine reads the line a server sentence names, and nothing else", () => {
  assert.equal(refusalLine("line 12: duplicate task id 'a'"), 12);
  assert.equal(refusalLine("nothing has no tasks"), null);
  assert.equal(refusalLine(["line 1: no"]), null);
  assert.equal(refusalLine(null), null);
  // The number must be the whole prefix, as the server writes it.
  assert.equal(refusalLine("line 2 of the file is odd"), null);
});

test("countLines counts lines holding anything, LF and CRLF alike", () => {
  // "c\rd" is one line: a lone CR is not a line end to the parser.
  assert.equal(countLines("a\r\n\r\nb\nc\rd"), 3);
  assert.equal(countLines(""), 0);
});

test("refusalText reads both refusal shapes and falls back when it cannot", () => {
  assert.equal(refusalText("plain", "fallback"), "plain");
  assert.equal(
    refusalText([{ msg: "one" }, { msg: "two" }, {}], "fallback"),
    "one; two",
  );
  assert.equal(refusalText({ odd: true }, "fallback"), "fallback");
  // A list with no message in it says nothing, so the fallback speaks.
  assert.equal(refusalText([{}, { msg: "" }], "fallback"), "fallback");
});

test("a pass threshold is a number only when it is a plain decimal numeral", () => {
  const judge = (threshold) =>
    composeTask(
      row({ id: "t", prompt: "p", scorer: "judge", rubric: "g", threshold }),
    ).scorer.pass_threshold;
  const feff = String.fromCharCode(0xfeff);
  assert.strictEqual(judge("0.5"), 0.5);
  assert.strictEqual(judge(".5"), 0.5);
  assert.strictEqual(judge(" 0.5 "), 0.5);
  assert.strictEqual(judge("5e-1"), 0.5);
  // Strict, because 0.5 == "0.5": a loose assert cannot tell the number
  // the page should send from the text it should not.
  assert.strictEqual(judge("+0.5"), 0.5);
  assert.strictEqual(judge("1."), 1);
  assert.strictEqual(judge("1E0"), 1);
  // Several digits in each part.
  assert.strictEqual(judge("0.25"), 0.25);
  assert.strictEqual(judge(".25"), 0.25);
  assert.strictEqual(judge("0.125"), 0.125);
  assert.strictEqual(judge("10e-1"), 1);
  assert.strictEqual(judge("1e-10"), 1e-10);
  // Number() reads each of these as a number nobody typed; JSON.stringify
  // would write the last two as null.
  assert.strictEqual(judge(feff), feff);
  assert.strictEqual(judge("0x1"), "0x1");
  assert.strictEqual(judge("0b1"), "0b1");
  assert.strictEqual(judge("Infinity"), "Infinity");
  assert.strictEqual(judge("1e999"), "1e999");
});

test("a threshold that is a long run of digits is read in linear time", () => {
  // The grammar has one way to read each numeral. With two (\d+\.?\d*),
  // a run of n digits that fails to match backtracks n squared times,
  // and composeTask runs on every keystroke: 50000 digits took seconds.
  const threshold = "1".repeat(50000) + "x";
  const started = process.hrtime.bigint();
  const task = composeTask(
    row({ id: "t", prompt: "p", scorer: "judge", rubric: "g", threshold }),
  );
  const ms = Number(process.hrtime.bigint() - started) / 1e6;
  assert.strictEqual(task.scorer.pass_threshold, threshold);
  assert.ok(ms < 500, "took " + ms + " ms");
});

test("datasetNudge calls a text empty only when Python would", () => {
  const task = '{"id": "t", "prompt": "p"}\n';
  // A line of U+FEFF is not blank to the parser: the server reads it as
  // a line (and refuses it as JSON), so it is not "no tasks".
  assert.strictEqual(datasetNudge("n", String.fromCharCode(0xfeff)), null);
  assert.strictEqual(
    datasetNudge("n", String.fromCharCode(0x85) + "\n"),
    "there are no tasks to store",
  );
  assert.strictEqual(datasetNudge("n", task), null);
});

test("composeTask's blank is Python's: U+FEFF is kept and U+0085 is not", () => {
  const feff = String.fromCharCode(0xfeff);
  const nel = String.fromCharCode(0x85);
  const typed = (value) =>
    row({
      id: "t",
      prompt: "p",
      system: value,
      scorer: "judge",
      rubric: value,
    });
  assert.deepEqual(composeTask(typed(feff)), {
    id: "t",
    prompt: "p",
    system: feff,
    rubric: feff,
    scorer: { kind: "judge" },
  });
  assert.deepEqual(composeTask(typed(nel)), {
    id: "t",
    prompt: "p",
    scorer: { kind: "judge" },
  });
});

test("countLines ends a line at a newline and at nothing else", () => {
  // Each of these is a line break to str.splitlines, and none is to the
  // parser, which reads lines at "\n" alone.
  for (const code of [
    0x0d, 0x0b, 0x0c, 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029,
  ]) {
    const mark = String.fromCharCode(code);
    assert.equal(countLines("a" + mark + "b"), 1, code.toString(16));
  }
  assert.equal(countLines("a\nb"), 2);
});

// ---- Phase N3: the experiment form. The unit suite executes the same
// ---- functions against the real door; these pin the composing half.

function form(fields) {
  return Object.assign(
    {
      name: "n",
      digest: "d".repeat(64),
      lineup: ["a/b"],
      budget: "standard",
      params: {},
      repeats: "",
      seed: "",
      estimand: "routed_service",
      attachments: null,
      metric: "",
      halt: true,
      invalid: [],
    },
    fields,
  );
}

test("hasControls is false for no controls and an empty set", () => {
  assert.strictEqual(hasControls(null), false);
  assert.strictEqual(hasControls(undefined), false);
  assert.strictEqual(hasControls({}), false);
  assert.strictEqual(hasControls({ temperature: 0 }), true);
});

test("experimentBody writes a key only when it was set", () => {
  assert.deepStrictEqual(experimentBody(form({})), {
    name: "n",
    dataset_digest: "d".repeat(64),
    lineup: ["a/b"],
    budget: "standard",
    estimand_mode: "routed_service",
    halt_on_refusal: true,
  });
  const set = experimentBody(
    form({
      params: { top_p: 1 },
      repeats: "3",
      seed: "0",
      attachments: "inline",
      metric: "judge",
      halt: false,
    }),
  );
  assert.deepStrictEqual(set.params, { top_p: 1 });
  assert.strictEqual(set.repeats, 3);
  // Zero is a seed, not a blank: Number("") is 0 as well.
  assert.strictEqual(set.task_order_seed, 0);
  assert.strictEqual(set.attachments_mode, "inline");
  assert.strictEqual(set.primary_metric, "judge");
  assert.strictEqual(set.halt_on_refusal, false);
  // Read as the number box means it: "1e1" is ten, as Chromium validates
  // it, not the one parseInt would read.
  assert.strictEqual(experimentBody(form({ repeats: "1e1" })).repeats, 10);
  // Past what JavaScript holds exactly, the text goes for the server to
  // refuse in its own words.
  assert.strictEqual(
    experimentBody(form({ seed: "9007199254740993" })).task_order_seed,
    "9007199254740993",
  );
});

test("experimentBody sends a copy of the lineup, never the live array", () => {
  const lineup = ["a/b"];
  const body = experimentBody(form({ lineup }));
  lineup.push("c/d");
  assert.deepStrictEqual(body.lineup, ["a/b"]);
});

test("experimentNudge names what the server would refuse, first first", () => {
  assert.strictEqual(
    experimentNudge(form({ name: "" })),
    "name the experiment",
  );
  assert.match(
    experimentNudge(form({ name: String.fromCodePoint(0x1f600).repeat(201) })),
    /201 characters, over the 200 limit/,
  );
  assert.strictEqual(
    experimentNudge(form({ name: String.fromCodePoint(0x1f600).repeat(200) })),
    null,
  );
  // A name of spaces is legal to the server.
  assert.strictEqual(experimentNudge(form({ name: "   " })), null);
  assert.strictEqual(
    experimentNudge(form({ digest: null })),
    "select a stored dataset in Datasets above",
  );
  assert.strictEqual(
    experimentNudge(form({ lineup: [] })),
    "check a model in the lineup above",
  );
  assert.strictEqual(
    experimentNudge(form({ invalid: ["temperature", "repeats"] })),
    "check temperature, repeats",
  );
});

test("projectionText says every shape the door returns", () => {
  assert.strictEqual(projectionText(null), "");
  assert.strictEqual(
    projectionText({
      input_usd: 0.000008,
      output_usd: 0.262144,
      total_usd: 0.262152,
      unpriced: [],
    }),
    "output at most $0.26 (a ceiling on tokens, not on the bill) · " +
      "input about $0.000008 (an estimate) · total $0.26",
  );
  // Native mode: the output stands, nobody is unpriced, and there is no
  // input estimate and so no total.
  assert.strictEqual(
    projectionText({
      input_usd: null,
      output_usd: 0.032768,
      total_usd: null,
      unpriced: [],
    }),
    "output at most $0.033 (a ceiling on tokens, not on the bill) · " +
      "no input estimate, so no total",
  );
  assert.strictEqual(
    projectionText({
      input_usd: null,
      output_usd: null,
      total_usd: null,
      unpriced: ["x/y (charges request)"],
    }),
    "unpriced: x/y (charges request). No figure is given, because a total " +
      "missing one arm would read as the whole comparison's total.",
  );
  assert.strictEqual(
    projectionText({
      input_usd: null,
      output_usd: null,
      total_usd: null,
      unpriced: [],
    }),
    "no projection: the figures could not be computed",
  );
  // Every unpriced member is named, in the door's order.
  assert.strictEqual(
    projectionText({
      input_usd: null,
      output_usd: null,
      total_usd: null,
      unpriced: ["a/b", "c/d"],
    }),
    "unpriced: a/b, c/d. No figure is given, because a total missing one " +
      "arm would read as the whole comparison's total.",
  );
  // A real zero is a figure, not an absence, in each place it can stand.
  assert.strictEqual(
    projectionText({
      input_usd: 0,
      output_usd: 0,
      total_usd: 0,
      unpriced: [],
    }),
    "output at most $0 (a ceiling on tokens, not on the bill) · " +
      "input about $0 (an estimate) · total $0",
  );
});

test("experimentFinished is every status but created and running", () => {
  for (const status of ["created", "running"]) {
    assert.strictEqual(experimentFinished(status), false);
  }
  for (const status of [
    "done",
    "stopped",
    "halted_on_refusal",
    "interrupted",
    "failed",
  ]) {
    assert.strictEqual(experimentFinished(status), true);
  }
});

test("experimentRowMeta counts every finished trial and names the trouble", () => {
  const row = (fields) => ({
    status: "done",
    trials_total: 6,
    trials_done: 0,
    trials_failed: 0,
    trials_refused: 0,
    ...fields,
  });
  assert.strictEqual(
    experimentRowMeta(row({ trials_done: 6 })),
    "done · 6/6 trials",
  );
  assert.strictEqual(
    experimentRowMeta(
      row({ status: "halted_on_refusal", trials_done: 2, trials_refused: 1 }),
    ),
    "halted_on_refusal · 3/6 trials (1 refused)",
  );
  assert.strictEqual(
    experimentRowMeta(
      row({ trials_done: 3, trials_failed: 2, trials_refused: 1 }),
    ),
    "done · 6/6 trials (2 failed, 1 refused)",
  );
  assert.strictEqual(
    experimentRowMeta(row({ status: "running", trials_total: 1 })),
    "running · 0/1 trials",
  );
});

test("datasetHeld is a summary object and nothing else", () => {
  assert.strictEqual(
    datasetHeld({ scorers: [], cites_documents: false }),
    true,
  );
  for (const value of [undefined, null, false, true]) {
    assert.strictEqual(datasetHeld(value), false);
  }
});

test("scoreBody names the recorded digest, and a judge only for judge tasks", () => {
  const judged = { scorers: ["exact", "judge"], cites_documents: false };
  const plain = { scorers: ["exact", "regex"], cites_documents: false };
  assert.deepStrictEqual(scoreBody("d", judged, "stub/j"), {
    dataset_digest: "d",
    judge_model: "stub/j",
  });
  // Blank is not sent: the door's min_length would refuse "".
  assert.deepStrictEqual(scoreBody("d", judged, ""), { dataset_digest: "d" });
  // The door would accept a judge here and record nothing of it; only
  // this rule keeps it off the wire.
  assert.deepStrictEqual(scoreBody("d", plain, "stub/j"), {
    dataset_digest: "d",
  });
  assert.deepStrictEqual(scoreBody("d", { scorers: [] }, "stub/j"), {
    dataset_digest: "d",
  });
});

test("scoreNudge says why Score waits, in every summary state", () => {
  const judged = { scorers: ["judge"], cites_documents: false };
  const plain = { scorers: ["exact"], cites_documents: false };
  assert.strictEqual(
    scoreNudge(null, ""),
    "reading which scorers its dataset declares",
  );
  assert.strictEqual(
    scoreNudge(undefined, "stub/j"),
    "its dataset could not be read; Retry asks the store again",
  );
  assert.match(
    scoreNudge(false, "stub/j"),
    /^its dataset was read from a file/,
  );
  assert.strictEqual(
    scoreNudge(judged, ""),
    "choose a judge, because a pass without one records every judge task as a scoring failure, and that record does not rewrite",
  );
  assert.strictEqual(scoreNudge(judged, "stub/j"), null);
  assert.strictEqual(scoreNudge(plain, ""), null);
  assert.strictEqual(scoreNudge({ scorers: [] }, ""), null);
});

test("scoreLabel says what a press spends, and nothing until it knows", () => {
  assert.strictEqual(
    scoreLabel({ scorers: ["exact", "judge"] }),
    "Score · pays the judge",
  );
  assert.strictEqual(scoreLabel({ scorers: ["regex"] }), "Score · free");
  assert.strictEqual(scoreLabel({ scorers: [] }), "Score · free");
  for (const unknown of [undefined, null, false]) {
    assert.strictEqual(scoreLabel(unknown), "Score");
  }
  assert.strictEqual(judgeTasks({ scorers: ["judge"] }), true);
  assert.strictEqual(judgeTasks(false), false);
});

test("scoreNudge follows the catalog and shows the door's sentence as given", () => {
  const judged = { scorers: ["judge"], cites_documents: false };
  assert.strictEqual(
    scoreNudge(judged, "", "pending"),
    "choose a judge once the catalog has loaded, because a pass without one records every judge task as a scoring failure, and that record does not rewrite",
  );
  assert.strictEqual(
    scoreNudge(judged, "", "unavailable"),
    "no judge can be chosen here, because the catalog is not available; " +
      "score it through the API with judge_model",
  );
  assert.strictEqual(
    scoreNudge(judged, "", "loaded"),
    "choose a judge, because a pass without one records every judge task as a scoring failure, and that record does not rewrite",
  );
  // The catalog changes nothing once a judge is chosen, or where none is
  // needed.
  assert.strictEqual(scoreNudge(judged, "stub/j", "unavailable"), null);
  assert.strictEqual(scoreNudge({ scorers: ["exact"] }, "", "pending"), null);
  // A refusal of the stored bytes is the door's sentence, as it is.
  const refused = "the bytes stored under digest d hash to e: edited by hand";
  assert.strictEqual(scoreNudge(refused, "stub/j", "loaded"), refused);
  assert.strictEqual(scoreLabel(refused), "Score");
  assert.deepStrictEqual(scoreBody("d", refused, "stub/j"), {
    dataset_digest: "d",
  });
});
