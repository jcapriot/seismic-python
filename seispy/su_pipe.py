"""Use seispy stages as filters in a pipe of Seismic Unix programs.

::

    suplane | python -m seispy.su_pipe amplitudes.gain tpow=2 | sustack key=cdp | suxwigb

Reads SU traces from the standard input (or ``-i FILE``), applies the stages, and writes SU traces to the standard
output (or ``-o FILE``). A stage is ``module.function`` of the seispy package followed by ``name=value`` parameters
of that function; several stages are separated by ``--``::

    python -m seispy.su_pipe filters.bfilt f3db_low=5 n_poles_low=2 -- amplitudes.gain tpow=1

Values are python literals (``1.5``, ``[1, 2]``, ``True``, ``None``) or else plain strings. The parameter names are
those of the python functions, not always SU's (see the README for which stage is which program).

Options (before the first stage): ``-i FILE``, ``-o FILE``, ``--endian {native,big,little}``.
"""
import ast
import importlib
import os
import sys

from .io.su import SUTraceWriter, read_su

_LITERALS = {"true": True, "false": False, "none": None}


def parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return _LITERALS.get(text.lower(), text)


def resolve_stage(name):
    module_name, _, func_name = name.rpartition(".")
    if not module_name:
        module_name, func_name = "", name
    try:
        module = importlib.import_module("seispy." + module_name if module_name else "seispy")
        return getattr(module, func_name)
    except (ImportError, AttributeError) as err:
        raise SystemExit(f"su_pipe: no stage {name!r} in seispy ({err})")


def parse_args(argv):
    opts = {"i": None, "o": None, "endian": None}
    args = list(argv)
    while args and args[0].startswith("-") and args[0] != "--":
        flag = args.pop(0)
        key = flag.lstrip("-")
        if key not in opts:
            raise SystemExit(f"su_pipe: unknown option {flag!r}")
        if not args:
            raise SystemExit(f"su_pipe: {flag} needs a value")
        opts[key] = args.pop(0)

    stages, current = [], []
    for token in args + ["--"]:
        if token == "--":
            if current:
                stages.append(current)
            current = []
        else:
            current.append(token)

    parsed = []
    for tokens in stages:
        kwargs = {}
        for item in tokens[1:]:
            key, eq, value = item.partition("=")
            if not eq:
                raise SystemExit(f"su_pipe: expected name=value, got {item!r} (stage {tokens[0]})")
            kwargs[key] = parse_value(value)
        parsed.append((tokens[0], kwargs))
    return opts, parsed


def main(argv=None):
    opts, stages = parse_args(sys.argv[1:] if argv is None else argv)

    source = opts["i"]
    if source is None:
        if sys.stdin.isatty():
            raise SystemExit("su_pipe: no input, give -i FILE or pipe SU traces in")
        source = sys.stdin.buffer
    traces = read_su(source, endian=opts["endian"])
    for name, kwargs in stages:
        traces = traces | resolve_stage(name)(**kwargs)

    dest = opts["o"] if opts["o"] is not None else sys.stdout.buffer
    try:
        with SUTraceWriter(dest, endian=opts["endian"]) as writer:
            for trace in traces:
                writer.write(trace)
    except BrokenPipeError:
        # the reader (e.g. `| head`) went away, as SU programs do: stop quietly
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 141
    return 0


if __name__ == "__main__":
    sys.exit(main())
