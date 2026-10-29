"""Lexer and parser for Zumzum: text -> syntax tree. No meaning is checked here (that is `sound.checker`).

Grammar (docs/spec.md section 2):

    program  = "zumzum" NUMBER NL { stmt NL }
    stmt     = "name" STRING | "about" STRING | "analysis" call
             | "sel" IDENT "=" sel | "node" IDENT "=" call { "at" sel | "mix" NUMBER | "off" }
             | "layer" IDENT "=" chain | "output" ( IDENT | call ) | "check" IDENT "=" call [ "on" IDENT ]
    chain    = "pass" | IDENT { "->" IDENT }
    sel      = sand { "or" sand }
    sand     = sunary { "and" sunary }
    sunary   = "not" sunary | "(" sel ")" | call | IDENT
    call     = IDENT "(" [ arg { "," arg } [ "," ] ] ")"
    arg      = IDENT ":" value
    value    = NUMBER [ ".." NUMBER ] | STRING | IDENT | "true" | "false"
"""

import re
from dataclasses import dataclass, field

from .errors import CheckError

RESERVED = {"zumzum", "name", "about", "analysis", "sel", "node", "layer", "output", "check", "at", "mix", "off",
            "on", "pass", "and", "or", "not", "true", "false"}

_TOKEN = re.compile(r"""
    (?P<ws>[ \t\r]+) | (?P<comment>\#[^\n]*) | (?P<nl>\n)
  | (?P<num>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<str>"(?:[^"\\\n]|\\["\\])*")
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<sym>->|\.\.|[=(),:])
""", re.VERBOSE)


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple  # ((key, value, line), ...)
    line: int


@dataclass(frozen=True)
class Ref:
    name: str
    line: int


@dataclass(frozen=True)
class Not:
    arg: object


@dataclass(frozen=True)
class And:
    args: tuple


@dataclass(frozen=True)
class Or:
    args: tuple


@dataclass(frozen=True)
class Ident:
    """A bare identifier used as a value (an enum choice)."""
    name: str


@dataclass
class Program:
    version: str = ""
    meta: dict = field(default_factory=dict)       # name / about -> (value, line)
    analysis: object = None                        # Call or None
    sels: list = field(default_factory=list)       # (name, expr, line)
    ops: list = field(default_factory=list)        # nodes: (name, call, at_expr or None, mix or None, line)
    off: set = field(default_factory=set)          # nodes marked off (bypassed)
    layers: list = field(default_factory=list)     # (name, [op names], line)
    output: object = None                          # (name, line), (Call, line) for mix(...), or None
    checks: list = field(default_factory=list)     # (name, call, layer or None, line)
    dup: list = field(default_factory=list)        # program-level statements seen twice: (word, line)


def tokenize(src):
    if len(src.encode("utf-8")) > 32 * 1024:
        raise CheckError(["line 1: program is longer than 32 KiB"])
    toks, pos, line, depth = [], 0, 1, 0
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m:
            raise CheckError([f"line {line}: unexpected character {src[pos]!r}"])
        kind, text = m.lastgroup, m.group()
        pos = m.end()
        if kind in ("ws", "comment"):
            continue
        if kind == "nl":
            if depth == 0 and toks and toks[-1][0] != "nl":
                toks.append(("nl", "\n", line))
            line += 1
            continue
        if kind == "str" and len(text) > 402:
            raise CheckError([f"line {line}: string longer than 400 characters"])
        if kind == "ident" and len(text) > 40:
            raise CheckError([f"line {line}: name longer than 40 characters"])
        if kind == "num" and len(text) > 40:
            raise CheckError([f"line {line}: number literal longer than 40 characters"])
        if text == "(":
            depth += 1
        elif text == ")":
            depth = max(0, depth - 1)
        toks.append((kind, text, line))
    if not toks or toks[-1][0] != "nl":
        toks.append(("nl", "\n", line))
    toks.append(("eof", "", line))
    return toks


def statements(src):
    """[(first word, first line, last line)] of each statement, 0-based lines, from the tokenizer: a statement runs
    to the newline that ends it outside parentheses, so a call wrapped over several lines is one statement."""
    out, start, word = [], None, None
    for kind, text, line in tokenize(src):
        if kind == "eof":
            break
        if kind == "nl":
            if start is not None:
                out.append((word, start - 1, line - 1))
            start = word = None
        elif start is None:
            start, word = line, text
    return out


class _P:
    def __init__(self, toks):
        self.t, self.i, self.depth = toks, 0, 0

    def peek(self, k=0):
        return self.t[min(self.i + k, len(self.t) - 1)]

    def take(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, kind, text=None):
        tok = self.take()
        if tok[0] != kind or (text is not None and tok[1] != text):
            want = repr(text) if text else kind
            got = "end of line" if tok[0] == "nl" else repr(tok[1]) if tok[1] else "end of program"
            raise CheckError([f"line {tok[2]}: expected {want}, got {got}"])
        return tok

    def at(self, text):
        return self.peek()[1] == text and self.peek()[0] in ("sym", "ident")

    def name(self):
        tok = self.expect("ident")
        if tok[1] in RESERVED:
            raise CheckError([f"line {tok[2]}: {tok[1]!r} is a reserved word"])
        return tok[1], tok[2]

    def value(self):
        tok = self.take()
        if tok[0] == "num":
            v = _number(tok)
            if self.at(".."):
                self.take()
                hi = self.expect("num")
                return Range(v, _number(hi))
            return v
        if tok[0] == "str":
            return tok[1][1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if tok[0] == "ident":
            if tok[1] == "true":
                return True
            if tok[1] == "false":
                return False
            return Ident(tok[1])
        raise CheckError([f"line {tok[2]}: expected a value, got {tok[1]!r}"])

    def call(self):
        tok = self.expect("ident")
        self.expect("sym", "(")
        args = []
        while not self.at(")"):
            key = self.expect("ident")
            self.expect("sym", ":")
            args.append((key[1], self.value(), key[2]))
            if not self.at(","):
                break
            self.take()
        self.expect("sym", ")")
        return Call(tok[1], tuple(args), tok[2])

    def sel(self):
        self.depth += 1
        if self.depth > 16:
            raise CheckError([f"line {self.peek()[2]}: selection nested more than 16 deep"])
        parts = [self.sand()]
        while self.at("or"):
            self.take()
            parts.append(self.sand())
        self.depth -= 1
        return parts[0] if len(parts) == 1 else Or(tuple(parts))

    def sand(self):
        parts = [self.sunary()]
        while self.at("and"):
            self.take()
            parts.append(self.sunary())
        return parts[0] if len(parts) == 1 else And(tuple(parts))

    def sunary(self):
        if self.at("not"):
            self.take()
            self.depth += 1
            if self.depth > 16:
                raise CheckError([f"line {self.peek()[2]}: selection nested more than 16 deep"])
            arg = self.sunary()
            self.depth -= 1
            return Not(arg)
        if self.at("("):
            self.take()
            e = self.sel()
            self.expect("sym", ")")
            return e
        tok = self.peek()
        if tok[0] == "ident" and self.peek(1)[1] == "(":
            return self.call()
        n, line = self.name()
        return Ref(n, line)


def _number(tok):
    try:
        v = float(tok[1]) if any(c in tok[1] for c in ".eE") else int(tok[1])
    except (ValueError, OverflowError):
        raise CheckError([f"line {tok[2]}: bad number {tok[1]!r}"])
    if isinstance(v, float) and v != v or v in (float("inf"), float("-inf")):
        raise CheckError([f"line {tok[2]}: number out of range {tok[1]!r}"])
    return v


def parse(src):
    p = _P(tokenize(src))
    prog = Program()
    while p.peek()[0] == "nl":
        p.take()
    p.expect("ident", "zumzum")
    prog.version = p.expect("num")[1]
    p.expect("nl")
    seen = set()
    while p.peek()[0] != "eof":
        if p.peek()[0] == "nl":
            p.take()
            continue
        word, wline = p.expect("ident")[1:]
        if word in ("name", "about", "analysis", "output"):
            if word in seen:
                prog.dup.append((word, wline))
            seen.add(word)
        if word in ("name", "about"):
            tok = p.expect("str")
            prog.meta[word] = (p.t[p.i - 1][1][1:-1].replace('\\"', '"').replace("\\\\", "\\"), tok[2])
        elif word == "analysis":
            prog.analysis = p.call()
        elif word == "sel":
            n, line = p.name()
            p.expect("sym", "=")
            prog.sels.append((n, p.sel(), line))
        elif word == "node":
            n, line = p.name()
            p.expect("sym", "=")
            c = p.call()
            at = mix = None
            while p.at("at") or p.at("mix") or p.at("off"):
                kw = p.take()[1]
                if kw == "off":
                    prog.off.add(n)
                elif kw == "at":
                    if at is not None:
                        raise CheckError([f"line {line}: node {n}: 'at' given twice (combine with and / or)"])
                    at = p.sel()
                else:
                    if mix is not None:
                        raise CheckError([f"line {line}: node {n}: 'mix' given twice"])
                    mix = _number(p.expect("num"))
            prog.ops.append((n, c, at, mix, line))
        elif word == "layer":
            n, line = p.name()
            p.expect("sym", "=")
            if p.at("pass"):
                p.take()
                chain = []
            else:
                chain = [p.name()[0]]
                while p.at("->"):
                    p.take()
                    chain.append(p.name()[0])
            prog.layers.append((n, chain, line))
        elif word == "output":
            if p.peek()[0] == "ident" and p.peek(1)[1] == "(":
                c = p.call()
                prog.output = (c, c.line)
            else:
                prog.output = p.name()
        elif word == "check":
            n, line = p.name()
            p.expect("sym", "=")
            c = p.call()
            layer = None
            if p.at("on"):
                p.take()
                layer = p.name()[0]
            prog.checks.append((n, c, layer, line))
        else:
            raise CheckError([f"line {wline}: unknown statement {word!r} (name, about, analysis, sel, node, layer, "
                              "output, check)"])
        p.expect("nl")
    return prog
