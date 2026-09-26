"""The network posture walk: every way bench/ can reach the network or
start a process, mapped to where it happens and whose door reaches it.

THE SIBLING OF THE FILESYSTEM WALK (tests/test_api.py,
test_review_repro_every_path_operation_sits_in_a_named_posture), keyed
the same way (module and qualified function, no two definitions sharing
a key), and commissioned in Phase O with the clone door: the tripwire
that makes "fetches nothing" a checked claim for every door but that
one. A new network call, a new process, a new git verb, a new import of
a network module or a new route reaching an old network function fails
here until somebody says which posture it sits in.

HOW COMPLETE IS "EVERY", in the filesystem walk's two kinds of rule.

Complete BY CONSTRUCTION over:
  - imports: every module bench/ imports is TRACKED or INERT by name, and
    one in neither fails as unclassified; a TRACKED module is imported
    only as `import m`, never aliased or from-imported, so its calls
    always read m.x.
  - calls under a tracked module: each is keyed into NETWORK_CALLS
    unless listed PURE by name.
  - tracked names as values: a tracked module is only ever the root of
    an attribute chain (or hasattr's first argument), and a tracked
    chain that is not a call's function sits in an annotation, an except
    clause or VALUE_OK. So partial(httpx.get), to_thread(subprocess.run),
    Process(target=subprocess.run) and getattr(httpx, ...) all fail.
  - processes: every process start is inside a named runner and nowhere
    else. _git starts ["git", *args]; _git_clone starts exactly
    _clone_argv's list, whose fixed configuration is pinned here word
    for word, with "--" before the operands.
  - git verbs: a runner is only ever called, never passed as a value;
    its git arguments are a list of literals whose first is a verb from
    that runner's allowlist, and no literal is an option that rewrites
    configuration, the transport or the program git runs.
  - reach: DOOR_REACH records, for each route, the network keys it
    reaches through calls that resolve by name (same module, a bench
    import by name, module.function), nested definitions reachable from
    their encloser. Only POST /clones reaches the clone runner.

Complete BY ENUMERATION over client handles: followed from their
constructors through app.state and self attributes, with-as and
assignment, and parameters annotated httpx.AsyncClient or httpx.Client.
A client passed as Any, kept in a container or returned from a helper is
followed only as far as its constructor's key.

WHAT IT DOES NOT SEE, stated as the filesystem walk states its limit: a
callback passed as a value is not followed by DOOR_REACH; destinations
are not evaluated (the OpenRouter URLs are env seams, models.py; the
clone hosts are held by bench/clones.py and its tests); git's own
config-driven behaviour is held by the pinned configuration and the
scrubbed environment, which tests/test_clone_door.py proves, not by an
AST; and the browser is held by the CSP's connect-src 'self'.
"""

import ast
import functools
from collections import Counter
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"

# Modules whose calls can reach the network or start a process.
TRACKED = {
    "httpx",
    "socket",
    "ssl",
    "subprocess",
    "asyncio",
    "multiprocessing",
    "multiprocessing.connection",
    "urllib",
    "urllib.request",
    "http",
    "http.client",
    "pty",
    "importlib",
    "requests",
    "aiohttp",
    "urllib3",
    "httpcore",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "telnetlib",
    "xmlrpc",
    "xmlrpc.client",
    "webbrowser",
    "socketserver",
    "logging.handlers",
    "logging.config",
}
TRACKED_ROOTS = {name.split(".")[0] for name in TRACKED} - {"logging"}
EXTRA_PREFIXES = ("logging.handlers.", "logging.config.")

# Modules that reach neither: named, so a new import is a decision.
INERT = {
    "argparse",
    "base64",
    "binascii",
    "collections",
    "collections.abc",
    "contextlib",
    "dataclasses",
    "datetime",
    "errno",
    "fastapi",
    "fastapi.encoders",
    "fastapi.exceptions",
    "fastapi.responses",
    "fastapi.staticfiles",
    "fnmatch",
    "hashlib",
    "hmac",
    "io",
    "ipaddress",
    "json",
    "logging",
    "math",
    "os",
    "os.path",
    "pathlib",
    "pydantic",
    "pypdf",
    "random",
    "re",
    "reprlib",
    "secrets",
    "shutil",
    "signal",
    "sqlite3",
    "starlette.datastructures",
    "starlette.types",
    "stat",
    "statistics",
    "sys",
    "tempfile",
    "time",
    "typing",
    "unicodedata",
    "urllib.parse",
    "xml.etree.ElementTree",
    "zipfile",
}

# Calls under a tracked prefix that reach nothing.
PURE = {
    "httpx.Timeout",
    "asyncio.Event",
    "asyncio.Semaphore",
    "asyncio.create_task",
    "asyncio.ensure_future",
    "asyncio.gather",
    "asyncio.shield",
    "asyncio.sleep",
    "asyncio.to_thread",
    "asyncio.run",
    "asyncio.wait",
    "asyncio.wait_for",
    "urllib.parse.urlparse",
    "urllib.parse.parse_qs",
    "urllib.parse.unquote",
    "urllib.parse.urlsplit",
}

# Tracked names used as values that are constants, not callables.
VALUE_OK = {
    "asyncio.CancelledError",
    "asyncio.FIRST_COMPLETED",
    "socket.IPPROTO_TCP",
    "socket.SOL_SOCKET",
    "socket.SO_KEEPALIVE",
    "socket.TCP_KEEPALIVE",
    "socket.TCP_KEEPIDLE",
    "socket.TCP_KEEPINTVL",
    "subprocess.PIPE",
    "subprocess.DEVNULL",
    "subprocess.STDOUT",
}

# os names that start a process: os is INERT because the filesystem walk
# classifies every os call, and these are named here so this walk keys a
# process start itself.
OS_PROCESS = {
    "os.system",
    "os.popen",
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.startfile",
}

PROCESS_STARTS = {
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
} | OS_PROCESS

CLIENT_TYPES = {"httpx.AsyncClient", "httpx.Client"}

# The two git runners: which argument is git's arguments, which verbs,
# and (for the clone runner) which argument holds the operands.
RUNNERS = {
    "_git": {"argv": 0, "verbs": {"rev-parse", "status"}},
    "_git_clone": {
        "argv": 1,
        "operands": 2,
        "verbs": {"init", "fetch", "ls-tree", "checkout", "rev-parse"},
    },
}

# Literal options no runner call may carry: each rewrites configuration,
# the transport, or the program git runs.
FORBIDDEN_OPTIONS = (
    "-c",
    "--config",
    "--exec-path",
    "--upload-pack",
    "--receive-pack",
    "--filter",
    "--recurse-submodules",
    "--remote",
    "-C",
    "--git-dir",
    "--work-tree",
)

# The clone runner's fixed configuration, word for word, as _clone_argv
# must begin. Each pair is a posture: https only; no credential helper;
# no redirect; no ambient negotiate identity; no reflog naming the
# operator; no maintenance running detached from the process group.
CLONE_PREFIX = (
    "git",
    "-c",
    "protocol.allow=never",
    "-c",
    "protocol.https.allow=always",
    "-c",
    "credential.helper=",
    "-c",
    "http.followRedirects=false",
    "-c",
    "http.emptyAuth=false",
    "-c",
    "core.logAllRefUpdates=false",
    "-c",
    "gc.auto=0",
    "-c",
    "maintenance.auto=false",
)


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


@functools.lru_cache(maxsize=64)
def _keyed(source):
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    def qualified(node):
        names = [node.name]
        parent = parents.get(id(node))
        while parent is not None:
            if isinstance(
                parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                names.append(parent.name)
            parent = parents.get(id(parent))
        return ".".join(reversed(names))

    scope, defs, counts = {}, {}, Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            name = qualified(node)
            counts[name] += 1
            defs[name] = node
            for child in ast.walk(node):
                scope[id(child)] = name
    return tree, parents, scope, defs, counts


def _tracked_prefix(name):
    if name.startswith("urllib.parse."):
        return True
    return (name.split(".", 1)[0] in TRACKED_ROOTS and "." in name) or name.startswith(
        EXTRA_PREFIXES
    )


def _annotations_and_handlers(tree):
    ids = set()
    for node in ast.walk(tree):
        spots = []
        if isinstance(node, ast.arg) and node.annotation is not None:
            spots.append(node.annotation)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns:
            spots.append(node.returns)
        if isinstance(node, ast.AnnAssign):
            spots.append(node.annotation)
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            spots.append(node.type)
        for spot in spots:
            ids.update(id(n) for n in ast.walk(spot))
    return ids


def _string(node):
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _runner_call(runner, call):
    """(verb, problem) for one call of a runner."""
    rule = RUNNERS[runner]
    if len(call.args) <= rule["argv"] or not isinstance(
        call.args[rule["argv"]], ast.List
    ):
        return None, "git's arguments are not a list display"
    words = call.args[rule["argv"]].elts
    verb = _string(words[0]) if words else None
    if verb is None:
        return None, "the verb is not a literal"
    if verb not in rule["verbs"]:
        return verb, f"verb {verb!r} not in {runner}'s allowlist"
    if any(_string(w) is None for w in words):
        return verb, "git's arguments are not all literal"
    for word in map(_string, words):
        if word.startswith(FORBIDDEN_OPTIONS) or (
            word.startswith("--template=") and word != "--template="
        ):
            return verb, f"forbidden option {word!r}"
    if "operands" in rule and (
        len(call.args) <= rule["operands"]
        or not isinstance(call.args[rule["operands"]], ast.List)
    ):
        return verb, "operands are not a list display"
    return verb, None


def _clone_argv_problem(defs):
    fn = defs.get("_clone_argv")
    if fn is None:
        return "_clone_argv is missing"
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.List):
        return "_clone_argv does not return one list display"
    elts = returns[0].value.elts
    head = tuple(_string(e) for e in elts[: len(CLONE_PREFIX)])
    if head != CLONE_PREFIX:
        return f"_clone_argv's configuration is {head}, not the pinned one"
    rest = elts[len(CLONE_PREFIX) :]
    shape = [type(e).__name__ for e in rest]
    if (
        shape != ["JoinedStr", "JoinedStr", "Starred", "Constant", "Starred"]
        or _string(rest[3]) != "--"
    ):
        return f"_clone_argv's tail is {shape}, not --git-dir, --work-tree, *args, '--', *operands"
    return None


def _client_attrs(files):
    attrs = {"client"}
    for _, source in files:
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and _dotted(node.value.func) in CLIENT_TYPES
            ):
                for target in node.targets:
                    name = _dotted(target)
                    if ".state." in f".{name}" or name.startswith("self."):
                        attrs.add(name.rsplit(".", 1)[-1])
    return attrs


def scan(files):
    """(found, failures): every network-capable call keyed with a count,
    and every rule broken."""
    found, failures = {}, set()
    attrs = _client_attrs(files)

    def add(key, name):
        found.setdefault(key, Counter())[name] += 1

    for fname, source in files:
        tree, parents, scope, defs, counts = _keyed(source)
        failures.update(
            (fname, name, "defined twice") for name, n in counts.items() if n > 1
        )
        exempt = _annotations_and_handlers(tree)
        if fname == "main.py":
            problem = _clone_argv_problem(defs)
            if problem:
                failures.add((fname, "_clone_argv", problem))

        def key_of(node, fname=fname, scope=scope):
            return (fname, scope.get(id(node), "<module>"))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in TRACKED:
                        if alias.asname:
                            failures.add(
                                (fname, f"import {alias.name} as {alias.asname}")
                            )
                        add((fname, "<imports>"), alias.name)
                    elif alias.name not in INERT:
                        failures.add((fname, f"unclassified import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    continue
                if module in TRACKED or any(
                    f"{module}.{a.name}" in TRACKED for a in node.names
                ):
                    failures.add((fname, f"from {module} import ..."))
                elif module not in INERT and not module.startswith("bench"):
                    failures.add((fname, f"unclassified import from {module}"))

        handles = {}
        for qualified, fn in defs.items():
            names = {
                a.arg
                for a in [*fn.args.args, *fn.args.kwonlyargs, *fn.args.posonlyargs]
                if a.annotation is not None and _dotted(a.annotation) in CLIENT_TYPES
            }
            for n in ast.walk(fn):
                if (
                    isinstance(n, ast.withitem)
                    and isinstance(n.optional_vars, ast.Name)
                    and isinstance(n.context_expr, ast.Call)
                    and _dotted(n.context_expr.func) in CLIENT_TYPES
                ):
                    names.add(n.optional_vars.id)
                if (
                    isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Call)
                    and _dotted(n.value.func) in CLIENT_TYPES
                ):
                    names.update(t.id for t in n.targets if isinstance(t, ast.Name))
            handles[qualified] = names

        for node in ast.walk(tree):
            parent = parents.get(id(node))
            key = key_of(node)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                dotted = _dotted(node)
                if any(
                    dotted.endswith(f"state.{a}") or dotted == f"self.{a}"
                    for a in attrs
                ):
                    add(key, f"<reads> {dotted.rsplit('.', 2)[-2]}.{node.attr}")
            if (
                isinstance(node, ast.Call)
                and _dotted(node.func) == "getattr"
                and len(node.args) >= 2
            ):
                if _string(node.args[1]) in attrs:
                    add(key, f"<reads> getattr {node.args[1].value}")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                is_callee = isinstance(parent, ast.Call) and parent.func is node
                if node.id in TRACKED_ROOTS and not isinstance(parent, ast.Attribute):
                    hasattr_first = (
                        isinstance(parent, ast.Call)
                        and _dotted(parent.func) == "hasattr"
                        and parent.args
                        and parent.args[0] is node
                    )
                    if not hasattr_first and id(node) not in exempt:
                        failures.add(
                            (fname, key[1], f"module {node.id} used as a value")
                        )
                if node.id in RUNNERS and not is_callee:
                    failures.add((fname, key[1], f"runner {node.id} used as a value"))
                if node.id == "_clone_argv" and not is_callee:
                    failures.add((fname, key[1], "_clone_argv used as a value"))
            if isinstance(node, ast.Attribute) and not isinstance(
                parent, ast.Attribute
            ):
                dotted = _dotted(node)
                is_callee = isinstance(parent, ast.Call) and parent.func is node
                if (
                    _tracked_prefix(dotted)
                    and not is_callee
                    and id(node) not in exempt
                    and dotted not in VALUE_OK
                ):
                    failures.add((fname, key[1], f"{dotted} used as a value"))
            if not isinstance(node, ast.Call):
                continue
            name = _dotted(node.func)
            if name in ("__import__", "eval", "exec", "compile"):
                failures.add((fname, key[1], f"dynamic {name}"))
            if name in RUNNERS:
                verb, problem = _runner_call(name, node)
                if problem:
                    failures.add((fname, key[1], f"{name}: {problem}"))
                add(key, f"{name} {verb}")
                continue
            if name == "_clone_argv" and key[1] != "_git_clone":
                failures.add((fname, key[1], "_clone_argv called outside _git_clone"))
            if name in PROCESS_STARTS:
                add(key, name)
                if (fname, key[1]) == ("main.py", "_git"):
                    argv = node.args[0] if node.args else None
                    ok = (
                        name == "subprocess.run"
                        and isinstance(argv, ast.List)
                        and [type(e).__name__ for e in argv.elts]
                        == ["Constant", "Starred"]
                        and _string(argv.elts[0]) == "git"
                    )
                elif (fname, key[1]) == ("main.py", "_git_clone"):
                    ok = (
                        name == "asyncio.create_subprocess_exec"
                        and len(node.args) == 1
                        and isinstance(node.args[0], ast.Starred)
                        and isinstance(node.args[0].value, ast.Call)
                        and _dotted(node.args[0].value.func) == "_clone_argv"
                    )
                else:
                    ok = False
                if not ok:
                    failures.add(
                        (fname, key[1], f"{name} outside a named runner's shape")
                    )
                continue
            if _tracked_prefix(name):
                if name not in PURE:
                    add(key, name)
                continue
            if isinstance(node.func, ast.Attribute):
                receiver = _dotted(node.func.value)
                if any(
                    receiver.endswith(f"state.{a}") or receiver == f"self.{a}"
                    for a in attrs
                ) or receiver in handles.get(key[1], ()):
                    add(key, f"client.{node.func.attr}")
    return found, failures


def reach(files, found):
    """Each route's reachable network keys, through calls that resolve by
    name; nested definitions are reachable from their encloser."""
    edges, routes, top, imported = {}, {}, set(), {}
    for fname, source in files:
        tree, _, scope, defs, _ = _keyed(source)
        for qualified, fn in defs.items():
            if "." not in qualified:
                top.add((fname, qualified))
            else:
                encloser = qualified.rsplit(".", 1)[0]
                edges.setdefault((fname, encloser), set()).add((fname, qualified))
            for dec in fn.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and _dotted(dec.func).startswith("app.")
                    and dec.args
                    and _string(dec.args[0]) is not None
                ):
                    method = _dotted(dec.func).split(".")[-1].upper()
                    routes[(fname, qualified)] = f"{method} {dec.args[0].value}"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "bench."
            ):
                for alias in node.names:
                    imported[(fname, alias.asname or alias.name)] = (
                        node.module.split(".")[1] + ".py",
                        alias.name,
                    )
        pending = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                pending.append(
                    ((fname, scope.get(id(node), "<module>")), _dotted(node.func))
                )
        for caller, callee in pending:
            if (caller[0], callee) in top or (caller[0], callee) in {
                (f, q) for f, q in top
            }:
                edges.setdefault(caller, set()).add((caller[0], callee))
            elif (caller[0], callee) in imported:
                edges.setdefault(caller, set()).add(imported[(caller[0], callee)])
            elif "." in callee:
                module, function = callee.split(".", 1)
                edges.setdefault(caller, set()).add((f"{module}.py", function))
    out = {}
    for route, label in routes.items():
        seen, stack = set(), [route]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(edges.get(node, ()))
        out[label] = sorted(k for k in seen if k in found and k[1] != "<imports>")
    return out


def bench_files(transform=None, extra=None):
    files = []
    for path in sorted(BENCH.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if transform is not None and path.name == "main.py":
            source = transform(source)
        files.append((path.name, source))
    return files + list(extra or ())


# Every network-capable call in bench/, keyed like the filesystem walk,
# with a count, and the posture it sits in.
NETWORK_CALLS = {
    # ---- OpenRouter, through the one client the lifespan builds. ----
    # The client, its transport (TCP keepalive; no proxy is read, since
    # an explicit transport makes httpx skip the proxy variables), and
    # its close; the catalog is fetched through it at boot.
    ("main.py", "<imports>"): Counter({"asyncio": 1, "httpx": 1, "subprocess": 1}),
    ("main.py", "lifespan"): Counter(
        {
            "httpx.AsyncClient": 1,
            "httpx.AsyncHTTPTransport": 1,
            "<reads> state.client": 2,
            "client.aclose": 1,
        }
    ),
    # The doors that hand that client to models.py: a comparison, a
    # streamed comparison, a trial's route lookup, a trial, a judge.
    ("main.py", "compare.limited"): Counter({"<reads> state.client": 1}),
    ("main.py", "compare_stream.events"): Counter({"<reads> state.client": 1}),
    ("main.py", "trial_route"): Counter({"<reads> state.client": 1}),
    ("main.py", "run_one_trial"): Counter({"<reads> state.client": 1}),
    ("main.py", "score_one_result"): Counter({"<reads> state.client": 1}),
    # models.py's six requests, each through the client it is handed,
    # each to an OpenRouter URL (env seams for the harness, models.py).
    ("models.py", "<imports>"): Counter({"httpx": 1, "socket": 1}),
    ("models.py", "fetch_catalog"): Counter({"client.get": 1}),
    ("models.py", "fetch_endpoints"): Counter({"client.get": 1}),
    ("models.py", "fetch_generation"): Counter({"client.get": 1}),
    ("models.py", "run_model"): Counter({"client.post": 1}),
    ("models.py", "judge_response"): Counter({"client.post": 1}),
    ("models.py", "stream_model"): Counter({"client.stream": 1}),
    # The reconcile CLI's own client, to OpenRouter's generation lookup;
    # a command an operator runs, not a door.
    ("reconcile.py", "<imports>"): Counter({"asyncio": 1, "httpx": 1}),
    ("reconcile.py", "_run"): Counter(
        {"httpx.AsyncClient": 1, "httpx.AsyncHTTPTransport": 1}
    ),
    # ---- Local processes. ----
    # _git: two-second questions of a local repository, rev-parse and
    # status only, from _app_sha at boot and _clone_state for a capture.
    ("main.py", "_git"): Counter({"subprocess.run": 1}),
    ("main.py", "_app_sha"): Counter({"_git rev-parse": 1, "_git status": 1}),
    ("main.py", "_clone_state"): Counter({"_git rev-parse": 1, "_git status": 1}),
    # Spawn children running a bench function this walk also scans:
    # _pdf_pages over bytes, _regex_search over text. Neither reaches
    # anything.
    ("extract.py", "<imports>"): Counter({"multiprocessing": 1}),
    ("extract.py", "_extract_pdf"): Counter({"multiprocessing.get_context": 1}),
    ("scoring.py", "<imports>"): Counter({"multiprocessing": 1}),
    ("scoring.py", "run_with_deadline"): Counter({"multiprocessing.get_context": 1}),
    # ---- The clone door, the second destination: the hosts in ----
    # ---- BENCH_CLONE_HOSTS, over https, with no credential. ----
    # One process start, of _clone_argv's pinned configuration.
    ("main.py", "_git_clone"): Counter({"asyncio.create_subprocess_exec": 1}),
    # Its five verbs, one each, in the one function that fetches.
    ("main.py", "_fetch_into"): Counter(
        {
            "_git_clone init": 1,
            "_git_clone fetch": 1,
            "_git_clone ls-tree": 1,
            "_git_clone checkout": 1,
            "_git_clone rev-parse": 1,
        }
    ),
}

# Which network keys each route reaches. Every route not named here
# reaches none.
DOOR_REACH = {
    "POST /compare": [("main.py", "compare.limited"), ("models.py", "run_model")],
    "POST /compare/stream": [
        ("main.py", "compare_stream.events"),
        ("models.py", "stream_model"),
    ],
    "POST /experiments": [("main.py", "trial_route"), ("models.py", "fetch_endpoints")],
    "POST /experiments/{experiment_id}/start": [
        ("main.py", "run_one_trial"),
        ("main.py", "trial_route"),
        ("models.py", "fetch_endpoints"),
        ("models.py", "stream_model"),
    ],
    "POST /experiments/{experiment_id}/score": [
        ("main.py", "score_one_result"),
        ("models.py", "judge_response"),
    ],
    # Local git only: the head and the dirty flag of the tree it walked.
    "POST /snapshots": [("main.py", "_clone_state"), ("main.py", "_git")],
    # The one door that fetches.
    "POST /clones": [("main.py", "_fetch_into"), ("main.py", "_git_clone")],
}


@functools.cache
def clean_scan():
    """The unplanted scan, once per session: every plant's pre-state."""
    found, failures = scan(bench_files())
    return {k: Counter(v) for k, v in found.items()}, frozenset(failures)


def test_every_network_call_in_bench_sits_in_a_named_posture():
    """WINDOW: every module in bench/, read as an AST by scan().

    The table is what the code does, rule by rule, and no rule is broken.
    PRE-STATE: bench/ has no subpackage, so the scan's glob is the whole
    package (a subpackage would be read by neither walk)."""
    assert sorted(BENCH.rglob("*.py")) == sorted(BENCH.glob("*.py"))
    found, failures = scan(bench_files())
    assert failures == set()
    assert found == NETWORK_CALLS


def test_only_the_clone_door_reaches_the_clone_runner():
    """WINDOW: every route in main.py and the network keys it reaches.

    "The composer fetches nothing", checked: POST /snapshots reaches the
    local git and nothing else, POST /snapshots/listing reaches nothing,
    and POST /clones is the one route that reaches _git_clone.
    PRE-STATE: the scan has no failures to hide a route behind."""
    files = bench_files()
    found, failures = scan(files)
    assert failures == set()
    reached = reach(files, found)
    assert {label: hits for label, hits in reached.items() if hits} == DOOR_REACH
    assert reached["POST /snapshots/listing"] == []
    assert [
        label for label, hits in reached.items() if ("main.py", "_git_clone") in hits
    ] == ["POST /clones"]


def _planted_module(source):
    return lambda: bench_files(extra=[("planted.py", source)])


def _planted_main(old, new):
    def files():
        def swap(source):
            assert old in source
            return source.replace(old, new, 1)

        return bench_files(transform=swap)

    return files


# One plant per rule, each the smallest evasion that rule exists for.
# (label, files, what must appear: a failure's text, or a found key.)
PLANTS = [
    ("httpx.get", _planted_module("import httpx\ndef door(u):\n    return httpx.get(u)\n"), ("planted.py", "door")),
    ("_git fetch", _planted_module("def door(u):\n    return _git(['fetch', '--', u])\n"), "not in _git's allowlist"),
    ("subprocess curl", _planted_module("import subprocess\ndef door(u):\n    subprocess.run(['curl', u])\n"), "outside a named runner"),
    ("from httpx import", _planted_module("from httpx import AsyncClient\n"), "from httpx import"),
    ("import smtplib", _planted_module("import smtplib\ndef door():\n    smtplib.SMTP('x')\n"), ("planted.py", "door")),
    ("an unclassified import", _planted_module("import ftplib2\n"), "unclassified import ftplib2"),
    ("to_thread(subprocess.run)", _planted_module("import asyncio, subprocess\nasync def door(u):\n    await asyncio.to_thread(subprocess.run, ['git', 'fetch', u])\n"), "subprocess.run used as a value"),
    ("run = _git", _planted_module("def door(u):\n    run = _git\n    run(['fetch', u])\n"), "runner _git used as a value"),
    ("_git -c fsmonitor", _planted_module("def door(r):\n    _git(['status', '-c', 'core.fsmonitor=x'], cwd=r)\n"), "forbidden option '-c'"),
    ("_git archive --remote", _planted_module("def door():\n    _git(['archive', '--remote=https://x/y', 'HEAD'])\n"), "verb 'archive' not in _git's allowlist"),
    ("_git verb in a variable", _planted_module("def door(u, verb):\n    _git([verb, u])\n"), "the verb is not a literal"),
    ("_git fetch of a variable", _planted_module("def door(u):\n    _git(['status', u])\n"), "not all literal"),
    ("_git_clone ls-remote", _planted_main("\n\ndef _remove_tree(", "\n\nasync def peek(u):\n    await _git_clone('/t', ['ls-remote'], [u], env={}, deadline=0)\n\n\ndef _remove_tree("), "verb 'ls-remote' not in _git_clone's allowlist"),
    ("_git_clone --upload-pack", _planted_main('["fetch", "-q", "--depth", "1"', '["fetch", "--upload-pack=x", "-q", "--depth", "1"'), "forbidden option '--upload-pack=x'"),
    ("a second client in lifespan", _planted_main("    app.state.db = store.connect(", "    app.state.http = httpx.AsyncClient()\n    app.state.db = store.connect("), ("main.py", "lifespan")),
    ("a second client used elsewhere", _planted_main("\n\ndef _remove_tree(", "\n\nasync def elsewhere(u):\n    return await app.state.http.get(u)\n\n\ndef _clone_http():\n    app.state.http = httpx.AsyncClient()\n\n\ndef _remove_tree("), ("main.py", "elsewhere")),
    ("loop.create_connection", _planted_module("import asyncio\nasync def door():\n    await asyncio.get_running_loop().create_connection(None, 'x', 1)\n"), ("planted.py", "door")),
    ("the composer calls the clone worker", _planted_main("        head, dirty = _clone_state(root)\n", "        await _fetch_into(root, 'u', 'r', env={}, deadline=0)\n        head, dirty = _clone_state(root)\n"), "reach"),
    ("the prefix drops protocol.allow=never", _planted_main('        "-c",\n        "protocol.allow=never",\n', ""), "not the pinned one"),
    ("the prefix drops followRedirects", _planted_main('        "-c",\n        "http.followRedirects=false",\n', ""), "not the pinned one"),
    ("the argv drops --", _planted_main('        *args,\n        "--",\n        *operands,\n', "        *args,\n        *operands,\n"), "not --git-dir"),
    ("a process started beside the runner", _planted_main("    proc = await asyncio.create_subprocess_exec(\n        *_clone_argv(tree, args, operands),", "    proc = await asyncio.create_subprocess_exec(\n        'git', *_clone_argv(tree, args, operands),"), "outside a named runner's shape"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("label", "files", "expect"), PLANTS, ids=[p[0] for p in PLANTS]
)
def test_a_planted_evasion_moves_the_walk(label, files, expect):
    """WINDOW: scan() and reach() over bench/ with one evasion planted,
    as a new module or an in-memory edit of main.py's source.

    Each plant is the smallest code its rule exists to catch, and the
    walk the gate runs (this same scan, not a copy) catches it: a
    failure naming the rule, a new key in the table, or a route reaching
    what it did not. PRE-STATE: unplanted, the scan is clean and equals
    the tables, so what moved is the plant."""
    clean_found, clean_failures = clean_scan()
    assert clean_failures == set() and clean_found == NETWORK_CALLS
    planted = files()
    found, failures = scan(planted)
    if expect == "reach":
        reached = reach(planted, found)
        assert ("main.py", "_git_clone") in reached["POST /snapshots"]
    elif isinstance(expect, tuple):
        assert found.get(expect) != clean_found.get(expect)
    else:
        assert any(expect in str(failure) for failure in failures), sorted(failures)


BENIGN = [
    ("urllib.parse.urlsplit", "import urllib.parse\ndef f(u):\n    return urllib.parse.urlsplit(u)\n"),
    ("httpx.Timeout", "import httpx\nT = httpx.Timeout(5.0)\n"),
    ("re match .start()", "import re\ndef f(s):\n    return re.search('a', s).start()\n"),
    ("a dict named client", "def f(scope):\n    client = scope\n    return client.get('host')\n"),
]  # fmt: skip


@pytest.mark.parametrize(("label", "source"), BENIGN, ids=[b[0] for b in BENIGN])
def test_network_free_code_is_not_flagged(label, source):
    """WINDOW: scan() over bench/ with network-free code added.

    The walk refuses what can reach the network and nothing else: a URL
    parser, a timeout value, a regex match's start, a dict that happens
    to be called client. PRE-STATE: the unplanted scan is clean."""
    assert clean_scan() == (NETWORK_CALLS, set())
    found, failures = scan(bench_files(extra=[("planted.py", source)]))
    assert failures == set()
    assert not any(key[0] == "planted.py" and key[1] != "<imports>" for key in found)
