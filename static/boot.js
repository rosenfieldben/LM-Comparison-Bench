// Boot. Every module has assigned its window.Bench* namespace by the time
// this last script runs, so all wiring lives here. Each init() attaches
// its module's listeners and paints its first state. The call order is the
// dependency order and is load-bearing: controls first (it paints the
// deck), then the library, diff, the stream client, and history.
BenchControls.init();
BenchLibrary.init();
BenchDiff.init();
BenchStream.init();
BenchHistory.init();
