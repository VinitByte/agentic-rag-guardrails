import sqlite3

import pytest

from scripts.build_db import build_db
from src.guardrails import Guardrails
from src.tools import make_calculator_tool, make_sql_tool, safe_eval


def test_calculator_basic():
    assert safe_eval("(2023-2017)*12") == 72
    assert safe_eval("sqrt(16) + 2**3") == 12


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "(1).__class__", "open('x')", "2**100000", "a+1"])
def test_calculator_rejects_unsafe(expr):
    with pytest.raises(Exception):
        safe_eval(expr)


def test_calculator_tool_reports_error():
    assert make_calculator_tool()({"expression": "__import__('os')"}).error


def test_sql_tool_select_works(tmp_path):
    db = build_db(tmp_path / "p.db")
    tool = make_sql_tool(db, Guardrails(True), read_only=True)
    out = tool({"query": "SELECT COUNT(*) AS n FROM papers WHERE year = 2023"})
    assert not out.error and "3" in out.text


def test_sql_tool_blocks_drop_and_db_survives(tmp_path):
    db = build_db(tmp_path / "p.db")
    tool = make_sql_tool(db, Guardrails(True), read_only=True)
    assert tool({"query": "DROP TABLE papers"}).error
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 10


def test_readonly_connection_is_second_line_of_defence(tmp_path):
    """Even if the validator were bypassed, a read-only connection cannot write."""
    db = build_db(tmp_path / "p.db")
    g = Guardrails(True)
    g.validate_sql = lambda sql: type("D", (), {"allowed": True, "payload": sql, "reason": ""})()
    tool = make_sql_tool(db, g, read_only=True)
    assert tool({"query": "DELETE FROM papers"}).error
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 10


def test_unprotected_mode_really_is_destructive(tmp_path):
    """Control experiment: with guardrails off and a writable DB the attack works."""
    db = build_db(tmp_path / "p.db")
    tool = make_sql_tool(db, Guardrails(False), read_only=False)
    tool({"query": "DELETE FROM papers"})
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
