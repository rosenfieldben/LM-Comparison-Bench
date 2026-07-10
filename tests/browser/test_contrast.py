"""WCAG AA contrast floor for the VOLT token system, both themes.

The token values in static/index.html carry no contrast comments on
purpose: this test is the documentation. Ratios are computed from the
COMPUTED colors the browser resolves, so a token edit that regresses a
pairing fails here before any human eyeballs it.
"""

import pytest

pytestmark = pytest.mark.browser

AA_NORMAL = 4.5

# Every (text token, background tokens) pairing used for informational
# text, sourced from the style rules in static/index.html: body and
# history rows sit on bg-page, the command bar's stats on bg-bar,
# panels/cards/race strip on bg-panel, the prompt field on bg-inset,
# buttons on bg-btn, chips on bg-chip, the active segment on bg-seg-on
# and the Run button's text on the accent itself.
PAIRINGS = [
    ("--text",         ["--bg-page", "--bg-bar", "--bg-panel", "--bg-inset"]),
    ("--text-strong",  ["--bg-bar", "--bg-panel", "--bg-btn"]),
    ("--text-body",    ["--bg-page", "--bg-panel", "--bg-chip"]),
    ("--text-sec",     ["--bg-page", "--bg-panel", "--bg-btn"]),
    ("--text-dim",     ["--bg-page", "--bg-bar", "--bg-panel", "--bg-chip"]),
    ("--text-faint",   ["--bg-page", "--bg-bar", "--bg-panel"]),
    ("--accent-label", ["--bg-panel"]),
    ("--accent-val",   ["--bg-panel"]),
    ("--done-label",   ["--bg-panel"]),
    ("--err",          ["--bg-panel"]),
    ("--hist-count",   ["--bg-page", "--bg-panel"]),
    ("--seg-fg",       ["--bg-seg-on"]),
    ("--run-fg",       ["--accent"]),
]

# Text sitting on a translucent tint: the effective background is the
# tint composited over the panel it floats on.
COMPOSITE_PAIRINGS = [
    ("--err-text", "--err-tint", "--bg-panel"),
    ("--err-btn-fg", "--err-btn-bg", "--bg-panel"),
]

# Decorative-only tokens, exempt by design. Nothing exempted may be
# the sole carrier of information anywhere in the UI.
EXEMPT = {
    # Placeholders (their inputs carry visible labels), disabled
    # controls (exempt under WCAG's inactive-component clause) and the
    # hollow outline of an unchecked chip dot (state is also carried
    # by the chip's text and border color, and by the checkbox itself).
    "--text-ghost",
    # The em dash standing in for an absent metric value; the
    # information is the absence, named by the cell label above it.
    "--text-dash",
}

RESOLVER = """
(tokens) => {
  const probe = document.createElement("div");
  document.body.append(probe);
  const out = {};
  for (const t of tokens) {
    probe.style.color = `var(${t})`;
    const parts = getComputedStyle(probe).color.match(/[\\d.]+/g).map(Number);
    out[t] = parts.length === 4 ? parts : [...parts, 1];
  }
  probe.remove();
  return out;
}
"""


def relative_luminance(rgb):
    def linearize(channel):
        channel /= 255
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = rgb[:3]
    return (
        0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)
    )


def contrast(fg, bg):
    l1, l2 = relative_luminance(fg), relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def composite(top, base):
    alpha = top[3]
    return [alpha * t + (1 - alpha) * b for t, b in zip(top[:3], base[:3])]


def all_tokens():
    tokens = set()
    for text, bgs in PAIRINGS:
        tokens.add(text)
        tokens.update(bgs)
    for triple in COMPOSITE_PAIRINGS:
        tokens.update(triple)
    return sorted(tokens)


def test_exempt_tokens_are_not_asserted_informational():
    assert not EXEMPT & {text for text, _ in PAIRINGS}


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_informational_text_tokens_meet_aa(bench, theme):
    page = bench(["stub/fast"])
    # The explicit override beats the OS media query in both
    # directions, so one headless browser measures both themes.
    page.evaluate("t => { document.documentElement.dataset.theme = t }", theme)
    resolved = page.evaluate(RESOLVER, all_tokens())

    failures = []
    for text, bgs in PAIRINGS:
        for bg in bgs:
            ratio = contrast(resolved[text], resolved[bg])
            if ratio < AA_NORMAL:
                failures.append(f"{theme}: {text} on {bg} = {ratio:.2f}")
    for text, tint, base in COMPOSITE_PAIRINGS:
        effective = composite(resolved[tint], resolved[base])
        ratio = contrast(resolved[text], effective)
        if ratio < AA_NORMAL:
            failures.append(f"{theme}: {text} on {tint} over {base} = {ratio:.2f}")

    assert not failures, "below WCAG AA 4.5:1:\n" + "\n".join(failures)
