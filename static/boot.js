// Boot. Every module has assigned its window.Bench* namespace by the time
// this last script runs, so all wiring lives here. Each init() attaches
// its module's listeners and paints its first state. The call order is the
// dependency order and is load-bearing: controls first (it paints the
// deck), then attachments, the library, diff, the stream client,
// history, the datasets panel and the experiment panel. Attachments come
// straight after controls because its first render asks BenchControls to
// refresh the Run button, and before that the deck has not been painted;
// the experiment panel comes last because it reads the controls and the
// datasets panel and subscribes to both.
BenchControls.init();
BenchAttach.init();
BenchLibrary.init();
BenchDiff.init();
BenchStream.init();
BenchHistory.init();
BenchDatasets.init();
BenchLifecycle.init();
