"""
Safe parsing of user-typed math equations into sympy expressions, then into
fast vectorized numeric functions -- the safety-critical piece of the
wizard. Raw user text is NEVER handed to Python's `eval`/`exec` directly;
it only ever becomes a sympy expression TREE, built through a whitelisted
symbol/function table, and even then only after surviving a stricter
character-level prefilter than sympy's own parser applies.

WHY the prefilter exists (found empirically while building this, not
theoretical): sympy's `parse_expr` internally calls Python's real `eval()`
on a transformed token stream (see `sympy.parsing.sympy_parser.eval_expr`).
Passing a custom `global_dict`/`local_dict` restricts which NAMES resolve
to real Python objects, but it does NOT stop attribute access on an object
that's already real -- confirmed directly: with a "safe-looking" local/
global dict containing no dangerous names, the literal string
`"().__class__.__bases__"` still parses successfully and returns the real
Python tuple `(object,)`, because `()` is a real empty-tuple literal and
`.__class__`/`.__bases__` are real attributes sympy's parser leaves as
plain Python attribute access (its `auto_symbol` transform explicitly
skips names immediately after a `.`, by design, so it doesn't mangle
legitimate attribute access on already-resolved objects -- see that
function's "Don't convert attribute access" comment). Chained far enough
(`...__subclasses__()[i]`), this is the standard Python sandbox-escape
technique, and it requires nothing from our own whitelist to reach --
only characters `(`, `)`, `.`, and letters, none of which an ordinary
arithmetic equation editor could plausibly refuse to accept "on principle"
if allowed unchecked (`.` in particular is needed for decimal numbers like
`3.14`, so it can't simply be banned outright without breaking normal
input).

The fix implemented here has two independent layers, verified against the
above finding and a battery of other injection attempts (see
test_equation_parser.py's `test_*_blocked` cases) before being trusted:

  1. CHARACTER WHITELIST (`_CHARSET_RE`): only ASCII letters/digits/
     underscore/whitespace and the arithmetic characters `+ - * / ( ) . , ^`
     are accepted AT ALL, checked before sympy ever sees the string. This
     alone removes quotes, brackets, braces, colons, semicolons, `=`, `@`,
     backslash, and every other character needed to write a string literal,
     a statement, an assignment, or a subscript -- i.e. everything the
     classic escape techniques need except plain attribute-access dots.
  2. NON-NUMERIC-DOT REJECTION (`_has_non_numeric_dot`): a `.` is allowed
     ONLY when both the character before and after it are digits (i.e.
     it's plainly part of a decimal number like `3.14`). Any other `.`
     -- including `x1.bar`, `().__class__`, or a bare trailing `.` -- is
     rejected before parsing. Since attribute access is the ONLY
     legitimate use for a bare `.` in Python syntax, and equations never
     need it, banning it outright closes off the entire chained-attribute
     escape family with no loss of expressiveness for arithmetic input.

On top of both, `global_dict` passed to `parse_expr` explicitly sets
`__builtins__` to `{}` (Python's `eval()` silently injects the REAL
`__builtins__` into any globals dict that doesn't already have that key --
another easy-to-miss default confirmed by testing) and contains ONLY the
sympy internals `parse_expr`'s own transforms need (`Symbol`, `Integer`,
`Float`, `Rational`, `Function`, `Max`, `Min`) plus the explicit function
whitelist below -- never `from sympy import *` (which `parse_expr` does
internally when `global_dict` is left as the default `None`, dumping
sympy's entire namespace into scope; passing an explicit `global_dict`, as
done here, is what suppresses that).

`implicit_multiplication_application` (sympy's "let `2x1` mean `2*x1`")
is DELIBERATELY NOT included in the transformations used here, even though
it looks like an obvious usability win: it bundles a `split_symbols` step
that silently decomposes an unrecognized multi-letter name into a PRODUCT
OF SINGLE-CHARACTER SYMBOLS (confirmed by testing: with this transform
enabled, a typo'd name like `foo` parsed as `f*o**2` instead of raising an
error) -- meaning a misspelled name wouldn't be caught as "unknown name",
it would silently become a different, wrong equation. Only
`standard_transformations` (name/number handling, no implicit
multiplication) plus `convert_xor` (so `^` means power, matching how most
people type exponents) are used; equations must write multiplication
explicitly (`2*x1`, not `2x1`) -- confirmed this raises a clear
`SyntaxError` instead of silently mismatching, which is exactly what a
"never silently do the wrong thing" contract requires.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import sympy
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, convert_xor)

_TRANSFORMATIONS = standard_transformations + (convert_xor,)

# User-facing function names (lowercase, as typed in an equation) mapped to
# the sympy callable actually used for both parsing and lambdify.
ALLOWED_FUNCTIONS: dict[str, object] = {
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
    "exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt,
    "tanh": sympy.tanh, "abs": sympy.Abs, "min": sympy.Min, "max": sympy.Max,
}
ALLOWED_CONSTANTS: dict[str, object] = {"pi": sympy.pi, "e": sympy.E}

# Everything parse_expr's OWN transformations (auto_symbol/auto_number/
# repeated_decimals) need to already exist in global_dict, since passing a
# non-None global_dict suppresses sympy's usual auto-population of these.
_PARSER_INTERNALS: dict[str, object] = {
    "Symbol": sympy.Symbol, "Integer": sympy.Integer, "Float": sympy.Float,
    "Rational": sympy.Rational, "Function": sympy.Function,
    "Max": sympy.Max, "Min": sympy.Min,
}

_CHARSET_RE = re.compile(r"^[A-Za-z0-9_+\-*/(),.\s\^]*$")
_ALLOWED_CHARS_DESCRIPTION = (
    "letters, digits, underscores, spaces, and + - * / ( ) . , ^")


def _has_non_numeric_dot(text: str) -> bool:
    for m in re.finditer(r"\.", text):
        i = m.start()
        before = text[i - 1] if i > 0 else ""
        after = text[i + 1] if i + 1 < len(text) else ""
        if not (before.isdigit() and after.isdigit()):
            return True
    return False


def _bad_characters(text: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                  "0123456789_+-*/(),.^ \t")
    return "".join(sorted(set(ch for ch in text if ch not in allowed)))


def _build_global_dict() -> dict:
    g: dict = {"__builtins__": {}}
    g.update(_PARSER_INTERNALS)
    for name, fn in ALLOWED_FUNCTIONS.items():
        g[name] = fn
    for name, val in ALLOWED_CONSTANTS.items():
        g[name] = val
    return g


@dataclass
class ParseResult:
    text: str
    expr: object = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_equation(text: str, allowed_names) -> ParseResult:
    """
    Parses `text` into a sympy expression using ONLY the names in
    `allowed_names` (states/actions/parameters/constants for the model
    being built) plus the fixed function/constant whitelist above.

    Never raises -- every failure mode (bad characters, unbalanced
    parentheses, an unknown name, an unknown function, or anything sympy
    itself objects to) comes back as a `ParseResult` with `.error` set to
    a plain-language message, so a caller (equation-entry widget or the
    save/validate pipeline) never needs its own try/except around this
    call.
    """
    allowed_names = list(allowed_names)
    collisions = (set(allowed_names) & set(ALLOWED_FUNCTIONS)
                  | set(allowed_names) & set(ALLOWED_CONSTANTS))
    if collisions:
        # A caller (schema.py's own reserved-name check) should have
        # already prevented this -- treat it as a programming error, not
        # a user-facing parse failure.
        raise ValueError(
            f"allowed_names collide with reserved function/constant "
            f"names: {sorted(collisions)}")

    if text is None or not str(text).strip():
        return ParseResult(text=text or "", error="This equation is empty.")

    text = str(text)

    bad_chars = _bad_characters(text)
    if bad_chars:
        return ParseResult(
            text=text,
            error=(f"This equation uses character(s) that aren't allowed: "
                   f"\"{bad_chars}\". Only {_ALLOWED_CHARS_DESCRIPTION} are "
                   f"permitted."))

    if _has_non_numeric_dot(text):
        return ParseResult(
            text=text,
            error=("A '.' was found that isn't part of a plain decimal "
                   "number (like 3.14). Attribute-style access (e.g. "
                   "'x.something') isn't supported in equations."))

    local_dict = {name: sympy.Symbol(name) for name in allowed_names}
    global_dict = _build_global_dict()

    try:
        expr = parse_expr(text, local_dict=local_dict, global_dict=global_dict,
                           transformations=_TRANSFORMATIONS, evaluate=True)
    except Exception as e:  # noqa: BLE001 -- intentionally broad, see module docstring
        return ParseResult(
            text=text,
            error=(f"Couldn't understand this equation: \"{text}\". "
                   f"({type(e).__name__}: this usually means unmatched "
                   f"parentheses, a misplaced operator, or a missing '*' "
                   f"for multiplication -- e.g. write '2*x1' not '2x1'.) "
                   f"Allowed names: {', '.join(allowed_names) or '(none defined yet)'}. "
                   f"Allowed functions: {', '.join(sorted(ALLOWED_FUNCTIONS))}."))

    if not isinstance(expr, sympy.Basic):
        return ParseResult(
            text=text,
            error=f"This didn't evaluate to a usable math expression: \"{text}\".")

    free_names = {str(s) for s in expr.free_symbols}
    unknown = free_names - set(allowed_names)
    if unknown:
        name = sorted(unknown)[0]
        suggestion = difflib.get_close_matches(name, allowed_names, n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        return ParseResult(
            text=text,
            error=(f"I don't recognize the name '{name}' in this equation."
                   f"{hint} Known names: {', '.join(allowed_names) or '(none defined yet)'}."))

    undef_funcs = expr.atoms(sympy.core.function.AppliedUndef)
    if undef_funcs:
        fname = str(sorted(undef_funcs, key=str)[0].func)
        suggestion = difflib.get_close_matches(fname, list(ALLOWED_FUNCTIONS), n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        return ParseResult(
            text=text,
            error=(f"I don't recognize the function '{fname}(...)' in this "
                   f"equation.{hint} Allowed functions: "
                   f"{', '.join(sorted(ALLOWED_FUNCTIONS))}."))

    warnings = []
    if "/" in text:
        warnings.append(
            "This equation divides by something. If the result looks "
            "invalid later (NaN/Inf), check whether that denominator can "
            "become zero.")

    return ParseResult(text=text, expr=expr, warnings=warnings)


def lambdify_equation(expr, ordered_names: list[str]):
    """
    Converts an already-parsed sympy expression into a fast numeric
    function `f(*args) -> ndarray`, vectorized over numpy arrays passed
    positionally in `ordered_names` order. Wrapped in try/except: lambdify
    essentially never fails for an expression that already parsed cleanly,
    but a caller building a model from a possibly-stale/hand-edited
    config should not have to guard against every theoretical exception
    type itself.
    """
    try:
        return sympy.lambdify(ordered_names, expr, modules="numpy")
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"Could not turn the equation '{expr}' into a runnable "
            f"function ({type(e).__name__}: {e}).") from e


def check_equation_live(text: str, allowed_names) -> tuple[bool, str]:
    """
    Cheap single-call check for a "live validate as you type" UI widget:
    returns (ok, message) -- message is empty on success, or the same
    plain-language error `parse_equation` would report.
    """
    result = parse_equation(text, allowed_names)
    return result.ok, (result.error or "")
