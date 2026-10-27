"""Tools the agents can call. Each tool takes a dict of arguments and returns a ToolResult."""
from __future__ import annotations

import ast
import math
import operator
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class ToolResult:
    text: str
    sources: set = field(default_factory=set)   # {(filename, page)} that were shown to the LLM
    error: bool = False


class Tool:
    def __init__(self, name: str, doc: str, func):
        self.name, self.doc, self.func = name, doc, func

    def __call__(self, args) -> ToolResult:
        return self.func(args)


# ---------------------------------------------------------------- retrieval
def make_retrieve_tool(retriever, guard, k: int = config.TOP_K) -> Tool:
    def run(args):
        q = args.get("query") if isinstance(args, dict) else None
        if not isinstance(q, str) or not q.strip():
            return ToolResult("Error: 'query' must be a non-empty string.", error=True)
        blocks, sources = [], set()
        for hit in retriever.search(q[:500], k=k):
            text, _flagged = guard.sanitize_retrieved(hit["text"])
            sources.add((hit["source"], hit["page"]))
            cite = f"[{hit['source']} p.{hit['page']}]"
            blocks.append(f'<document cite="{cite}">\n{text}\n</document>')
        return ToolResult("\n".join(blocks), sources)

    return Tool("retrieve_documents",
                'retrieve_documents: search the research-paper corpus. args: {"query": "<search text>"}', run)


# ---------------------------------------------------------------- SQL
def describe_schema(db_path) -> str:
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        con.close()
    return "\n".join(sql for _name, sql in rows)


def make_sql_tool(db_path, guard, read_only: bool = True) -> Tool:
    uri = Path(db_path).resolve().as_uri() + ("?mode=ro" if read_only else "?mode=rw")

    def run(args):
        sql = args.get("query") if isinstance(args, dict) else None
        if not isinstance(sql, str) or not sql.strip():
            return ToolResult("Error: 'query' must be a non-empty SQL string.", error=True)
        decision = guard.validate_sql(sql)
        if not decision.allowed:
            return ToolResult(f"BLOCKED by guardrail: {decision.reason}", error=True)
        con = sqlite3.connect(uri, uri=True)
        try:
            cur = con.execute(decision.payload)       # execute() refuses multi-statement strings
            if cur.description:
                cols = [d[0] for d in cur.description]
                rows = cur.fetchmany(50)
                lines = [" | ".join(cols)] + [" | ".join(str(v) for v in r) for r in rows]
                text = "<data>\n" + "\n".join(lines) + "\n</data>"
            else:
                con.commit()
                text = f"<data>OK, {cur.rowcount} row(s) affected</data>"
        except Exception as e:  # noqa: BLE001 - report DB errors back to the agent
            return ToolResult(f"SQL error: {e}", error=True)
        finally:
            con.close()
        return ToolResult(text)

    return Tool("sql_query",
                'sql_query: run ONE read-only SELECT on the SQLite database. args: {"query": "<SQL>"}', run)


# ---------------------------------------------------------------- calculator
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "log": math.log}


def safe_eval(expr: str):
    """Evaluate arithmetic only (numbers, + - * / ** % //, a few math functions). No eval()."""
    if len(expr) > 200:
        raise ValueError("expression too long")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _BIN[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY:
            return _UNARY[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("unsupported expression")

    return ev(ast.parse(expr, mode="eval"))


def make_calculator_tool() -> Tool:
    def run(args):
        expr = args.get("expression") if isinstance(args, dict) else None
        if not isinstance(expr, str):
            return ToolResult("Error: 'expression' must be a string.", error=True)
        try:
            return ToolResult(f"<data>{safe_eval(expr)}</data>")
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"Calculator error: {e}", error=True)

    return Tool("calculator",
                'calculator: evaluate arithmetic. args: {"expression": "<math, e.g. (2023-2017)*12>"}', run)
