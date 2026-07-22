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

  function fmtCost(c) {
    // Exponential for the typical sub-cent run (~$3.1e-5); plain decimals
    // once a run costs enough for them to be readable. The tilde is the
    // honesty marker: catalog prices times reported tokens, not a bill.
    return "~$" + (c < 0.01 ? c.toExponential(1) : c.toFixed(3));
  }

  function fmtEstimate(v) {
    if (v >= 100) return String(Math.round(v));
    return Number(v.toPrecision(2)).toString();
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

  const BenchLib = {
    shortName,
    fmtCost,
    fmtEstimate,
    niceScale,
    tokenizeDiff,
    diffTokens,
    DIFF_TOKEN_LIMIT,
  };
  if (typeof window !== "undefined") window.BenchLib = BenchLib;
  if (typeof module !== "undefined") module.exports = BenchLib;
})();
