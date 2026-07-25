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
  niceScale,
  tokenizeDiff,
  diffTokens,
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
