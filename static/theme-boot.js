// Applies the saved UI prefs to <html> before first paint so the page
// does not flash the defaults. Loaded as a blocking classic script in
// <head>: the position is load-bearing. A deferred load, or one at the end
// of the body, would let the default theme paint first. The module scripts
// re-read these values and wire the toggles. Externalized from an inline
// <head> block so the content security policy can keep script-src 'self'
// with no 'unsafe-inline'.
(function () {
  try {
    const t = localStorage.getItem("bench-theme");
    if (t === "dark" || t === "light") {
      document.documentElement.dataset.theme = t;
    }
    if (localStorage.getItem("bench-motion") === "off") {
      document.documentElement.dataset.motion = "off";
    }
    if (localStorage.getItem("bench-density") === "compact") {
      document.documentElement.dataset.density = "compact";
    }
  } catch (err) {
    // Storage blocked: the defaults apply.
  }
})();
