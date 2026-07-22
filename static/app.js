// The page's DOM logic. The pure helpers live in lib.js (loaded first)
// and are pulled off the single global it exposes; everything below
// wires them to the document.
const {
  shortName,
  fmtCost,
  fmtEstimate,
  niceScale,
  tokenizeDiff,
  diffTokens,
  DIFF_TOKEN_LIMIT,
} = window.BenchLib;

const promptEl = document.getElementById("prompt");
const modelsEl = document.getElementById("models");
const runBtn = document.getElementById("run");
const stopBtn = document.getElementById("stop");
const resultsEl = document.getElementById("results");
const runLabelEl = document.getElementById("run-label");
const savedSelect = document.getElementById("saved-prompts");
const saveBtn = document.getElementById("save-prompt");
const deleteBtn = document.getElementById("delete-prompt");
const promptMsg = document.getElementById("prompt-msg");
const linkedEl = document.getElementById("linked-name");
const lineupLabel = document.getElementById("lineup-label");
const runNote = document.getElementById("run-note");
const historyEl = document.getElementById("history");
const historyList = document.getElementById("history-list");
const historyFilter = document.getElementById("history-filter");
const historyNote = document.getElementById("history-note");
const raceEl = document.getElementById("race");
const raceGrid = document.getElementById("race-grid");
const raceScale = document.getElementById("race-scale");

// Boot wiring, consolidating into boot.js as the split completes. Each
// module's init() attaches its listeners and paints its first state; the
// call order is the dependency order and is load-bearing.
BenchControls.init();
BenchDiff.init();
BenchLibrary.init();
BenchStream.init();
BenchHistory.init();


