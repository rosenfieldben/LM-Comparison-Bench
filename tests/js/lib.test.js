// Pure-function tests for static/lib.js, run by `node --test tests/js/`.
// No build step and no dependencies: lib.js is required directly through
// its CommonJS guard.

const test = require("node:test");
const assert = require("node:assert");

const {
  shortName,
  fmtCost,
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
  const ops = diffTokens(tokenizeDiff("the quick fox"), tokenizeDiff("the slow fox"));
  const sames = ops.filter((o) => o.op === "same").map((o) => o.raw.trim());
  assert.ok(sames.includes("the"));
  assert.ok(sames.includes("fox"));
  assert.ok(ops.some((o) => o.op === "del" && o.raw.includes("quick")));
  assert.ok(ops.some((o) => o.op === "ins" && o.raw.includes("slow")));
});

test("diffTokens on identical input is all 'same'", () => {
  const ops = diffTokens(tokenizeDiff("same text here"), tokenizeDiff("same text here"));
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
