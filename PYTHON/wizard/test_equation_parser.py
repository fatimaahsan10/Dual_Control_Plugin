import numpy as np
import pytest
import sympy

from wizard.equation_parser import (
    parse_equation, lambdify_equation, check_equation_live, ALLOWED_FUNCTIONS,
)

NAMES = ["x1", "x2", "u1", "u2", "p1", "a", "eta"]


# ----------------------------------------------------------------------
# Legitimate equations parse correctly (the two real proven cases' actual
# equations, transcribed with explicit '*' since implicit multiplication
# is deliberately not supported -- see module docstring).
# ----------------------------------------------------------------------

def test_simple_linear_equation():
    r = parse_equation("eta * (a - p1*u1 - x1)", NAMES)
    assert r.ok, r.error
    assert r.expr.free_symbols == {sympy.Symbol(n) for n in
                                    ("eta", "a", "p1", "u1", "x1")}


def test_trig_and_exp_functions():
    r = parse_equation("sin(x1) + exp(-u1) + tanh(p1)", NAMES)
    assert r.ok, r.error


def test_power_with_caret():
    r = parse_equation("x1^2 + u1**2", NAMES)
    assert r.ok, r.error


def test_min_max_abs():
    r = parse_equation("max(x1, u1) + min(x1, u1) + abs(x1 - u1)", NAMES)
    assert r.ok, r.error


def test_decimal_numbers():
    r = parse_equation("3.14 * x1 + 0.5", NAMES)
    assert r.ok, r.error


def test_division_parses_but_warns():
    r = parse_equation("x1 / (1 + u1)", NAMES)
    assert r.ok, r.error
    assert any("divide" in w for w in r.warnings)


def test_empty_equation_rejected():
    r = parse_equation("", NAMES)
    assert not r.ok
    assert "empty" in r.error.lower()

    r2 = parse_equation("   ", NAMES)
    assert not r2.ok


def test_none_equation_rejected():
    r = parse_equation(None, NAMES)
    assert not r.ok


# ----------------------------------------------------------------------
# Typos / unknown names and functions produce plain-language errors, not
# silently-wrong equations. Critically, an unrecognized multi-letter name
# must NOT be silently decomposed into single-character symbols (this
# actually happened during development when implicit-multiplication
# transformations were used -- 'foo' parsed as f*o*o -- which is why
# equation_parser.py deliberately excludes them; these tests guard
# against that regressing).
# ----------------------------------------------------------------------

def test_unknown_name_reported_not_silently_reinterpreted():
    r = parse_equation("pric + x1", NAMES)
    assert not r.ok
    assert "pric" in r.error
    assert "recognize" in r.error


def test_unknown_name_gets_did_you_mean_suggestion():
    r = parse_equation("x11 + u1", NAMES)  # typo for x1
    assert not r.ok
    assert "x1" in r.error  # suggested


def test_unknown_function_reported():
    r = parse_equation("foo(x1)", NAMES)
    assert not r.ok
    assert "foo" in r.error
    assert "function" in r.error.lower()


def test_missing_multiplication_star_is_a_clear_error_not_silent_misparse():
    # '2x1' must not silently become '2*x1' via implicit multiplication,
    # since that machinery is exactly what caused the 'foo' -> f*o*o bug
    # during development -- see module docstring.
    r = parse_equation("2x1 + u1", NAMES)
    assert not r.ok
    assert "*" in r.error  # hints at the missing multiplication sign


# ----------------------------------------------------------------------
# SECURITY: the sandbox-escape / injection attempts found and verified
# during development (see equation_parser.py's module docstring for the
# full investigation). Every one of these must be rejected with a plain
# ParseResult.error, NEVER executed, and NEVER raise an uncaught
# exception out of parse_equation.
# ----------------------------------------------------------------------

INJECTION_ATTEMPTS = [
    '__import__("os").system("echo hacked")',
    'os.system("echo hi")',
    '().__class__.__bases__',
    '().__class__.__bases__[0].__subclasses__()',
    'x1.__class__.__bases__',
    'x1.__class__',
    'x1.bar',
    'getattr(x1, "free_symbols")',
    'x1; import os',
    'lambda: 1',
    '__builtins__',
    'exec("import os")',
    'eval("1")',
    "x1.__init__.__globals__",
    '[].__class__',
    '{}.__class__',
    "x1' + '",
]


@pytest.mark.parametrize("text", INJECTION_ATTEMPTS)
def test_injection_attempts_are_rejected_not_executed(text):
    r = parse_equation(text, NAMES)
    assert not r.ok, f"expected rejection for: {text!r}, got expr={r.expr!r}"
    # And, just as important: it must come back as a plain ParseResult,
    # not propagate an exception (asserted implicitly by reaching here).


def test_parse_equation_never_raises_on_arbitrary_garbage():
    garbage = [
        "(((", ")))", "***", "1//0", "x1 +++ u1", "\x00\x01",
        "a" * 5000, "x1 ** " + "x1 ** " * 200 + "x1",
    ]
    for text in garbage:
        r = parse_equation(text, NAMES)
        assert isinstance(r.error, (str, type(None)))


def test_disallowed_characters_rejected_with_the_offending_chars_shown():
    r = parse_equation('x1 + "hi"', NAMES)
    assert not r.ok
    assert '"' in r.error


def test_bad_dot_rejected():
    r = parse_equation("x1.5abc + u1", NAMES)
    # '.5abc' -> '.' preceded by digit '1'? no: preceded by '1' (digit),
    # followed by '5' (digit) -- actually "x1.5" has dot between '1' and
    # '5', both digits, so this specific case is numeric-looking; use an
    # unambiguous non-numeric dot instead.
    r2 = parse_equation("x1.foo", NAMES)
    assert not r2.ok


# ----------------------------------------------------------------------
# Collisions between allowed_names and reserved function/constant names
# are a caller bug (schema.py should already prevent this), not a
# user-facing parse failure -- confirm it's surfaced loudly (raise), not
# silently ignored.
# ----------------------------------------------------------------------

def test_allowed_names_colliding_with_functions_raises():
    with pytest.raises(ValueError):
        parse_equation("sin + 1", ["sin", "x1"])


# ----------------------------------------------------------------------
# lambdify: parsed expressions become correct, vectorized numeric
# functions matching the (row-per-symbol) K-batch convention every
# DynamicsFn/MeasurementFn/CostFn implementation in this codebase uses.
# ----------------------------------------------------------------------

def test_lambdify_matches_hand_computation():
    r = parse_equation("eta * (a - p1*u1 - x1)", NAMES)
    assert r.ok
    f = lambdify_equation(r.expr, ["eta", "a", "p1", "u1", "x1"])
    eta, a, p1, u1, x1 = 0.3, 100.0, 2.0, 10.0, 5.0
    expected = eta * (a - p1 * u1 - x1)
    assert np.isclose(f(eta, a, p1, u1, x1), expected)


def test_lambdify_vectorizes_over_k_batch():
    r = parse_equation("x1**2 + sin(u1)", NAMES)
    assert r.ok
    f = lambdify_equation(r.expr, ["x1", "u1"])
    x1 = np.array([0.0, 1.0, 2.0])
    u1 = np.array([0.0, np.pi / 2, np.pi])
    out = f(x1, u1)
    expected = x1 ** 2 + np.sin(u1)
    assert np.allclose(out, expected)


def test_min_max_abs_lambdify_matches_numpy():
    r = parse_equation("max(x1, u1) + min(x1, u1) + abs(x1 - u1)", NAMES)
    assert r.ok
    f = lambdify_equation(r.expr, ["x1", "u1"])
    x1 = np.array([1.0, -3.0])
    u1 = np.array([2.0, -1.0])
    out = f(x1, u1)
    expected = np.maximum(x1, u1) + np.minimum(x1, u1) + np.abs(x1 - u1)
    assert np.allclose(out, expected)


# ----------------------------------------------------------------------
# Live-check convenience wrapper (used by the wizard's on-keystroke
# validation widget).
# ----------------------------------------------------------------------

def test_check_equation_live_ok_and_error_cases():
    ok, msg = check_equation_live("x1 + u1", NAMES)
    assert ok and msg == ""

    ok2, msg2 = check_equation_live("unknown_name_here", NAMES)
    assert not ok2 and "recognize" in msg2
