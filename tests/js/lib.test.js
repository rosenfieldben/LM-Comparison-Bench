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
  DIFF_TOKEN_LIMIT,
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
