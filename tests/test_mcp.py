"""Irix MCP stdio server: JSON-RPC handshake, tool listing, dispatch, and
soft-fail. Drives the server over an in-memory stdin/stdout pair. No model."""
import io
import json

from irix import mcp_server


def _drive(messages):
    stdin = io.StringIO("".join(json.dumps(m) + "\n" for m in messages))
    stdout = io.StringIO()
    mcp_server.serve(stdin=stdin, stdout=stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_initialize_handshake():
    out = _drive([{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}])
    assert len(out) == 1
    r = out[0]["result"]
    assert r["serverInfo"]["name"] == "irix"
    assert "tools" in r["capabilities"]


def test_notification_gets_no_response():
    out = _drive([{"jsonrpc": "2.0", "method": "notifications/initialized"}])
    assert out == []


def test_tools_list_exposes_the_suite():
    out = _drive([{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    names = {t["name"] for t in out[0]["result"]["tools"]}
    assert {"scan_injection", "elements", "fuse_screen", "find_element",
            "read_text", "benchmark"} <= names
    for t in out[0]["result"]["tools"]:              # every tool is well-formed
        assert t["description"] and t["inputSchema"]["type"] == "object"


def test_tools_call_runs_benchmark():
    out = _drive([{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                   "params": {"name": "benchmark", "arguments": {"screens": 2}}}])
    payload = json.loads(out[0]["result"]["content"][0]["text"])
    assert payload["ok"] and payload["screens"] == 2
    assert out[0]["result"]["isError"] is False


def test_unknown_tool_soft_errors():
    out = _drive([{"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                   "params": {"name": "nope", "arguments": {}}}])
    assert out[0]["result"]["isError"] is True
    assert "error" in json.loads(out[0]["result"]["content"][0]["text"])


def test_unknown_method_returns_jsonrpc_error():
    out = _drive([{"jsonrpc": "2.0", "id": 5, "method": "frobnicate"}])
    assert out[0]["error"]["code"] == -32601
