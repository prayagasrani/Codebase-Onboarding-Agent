from onboarding_agent.parsing.ast_parser import AstParser
from onboarding_agent.parsing.chunker import Chunker

PY_SOURCE = '''\
"""Module docstring."""
import os
from pathlib import Path


class Greeter:
    """Says hello."""

    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"hello {self.name}"


def standalone(x):
    return x * 2
'''


def test_python_symbols_extracted():
    parser = AstParser()
    symbols, imports = parser.parse(PY_SOURCE, "greeter.py", "python")
    names = {(s.name, s.kind) for s in symbols}
    assert ("Greeter", "class") in names
    assert ("greet", "method") in names
    assert ("standalone", "function") in names
    method = next(s for s in symbols if s.name == "greet")
    assert method.parent == "Greeter"
    assert method.qualified_name == "Greeter.greet"
    assert method.start_line <= method.end_line
    assert any("import os" in i for i in imports)
    assert any("from pathlib" in i for i in imports)


def test_symbol_line_ranges_cover_bodies():
    parser = AstParser()
    symbols, _ = parser.parse(PY_SOURCE, "greeter.py", "python")
    standalone = next(s for s in symbols if s.name == "standalone")
    lines = PY_SOURCE.splitlines()
    body = "\n".join(lines[standalone.start_line - 1:standalone.end_line])
    assert "return x * 2" in body


def test_chunker_produces_symbol_and_window_chunks():
    parser = AstParser()
    symbols, imports = parser.parse(PY_SOURCE, "greeter.py", "python")
    chunks = Chunker().chunk_file("greeter.py", PY_SOURCE, "python", symbols, imports)
    assert chunks, "expected at least one chunk"
    symbol_chunks = [c for c in chunks if c.symbol_name]
    assert any(c.symbol_name == "Greeter.greet" for c in symbol_chunks)
    for c in chunks:
        assert c.citation == f"greeter.py:{c.start_line}-{c.end_line}"
        assert c.start_line >= 1
        assert c.text


def test_chunker_handles_markdown_without_symbols():
    text = "# Title\n\n" + "\n".join(f"line {i}" for i in range(150))
    chunks = Chunker().chunk_file("README.md", text, "markdown", [], [])
    assert len(chunks) >= 2  # windowed
    assert all(c.symbol_name is None for c in chunks)


def test_javascript_regex_fallback():
    js = "import {x} from './x';\n\nexport function doThing(a) {\n  return a;\n}\n"
    symbols, imports = AstParser()._parse_regex(js, "app.js", "javascript")
    assert any(s.name == "doThing" for s in symbols)
    assert imports
