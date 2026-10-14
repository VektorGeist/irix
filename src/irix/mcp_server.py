"""mcp_server — Irix as an MCP server, so any agent (Claude Code, Cursor, …)
gets deterministic, private, CPU-only screen vision as callable tools.

Pure stdlib: the MCP stdio transport is newline-delimited JSON-RPC 2.0, which
is small enough to speak directly — no `mcp` package dependency, nothing extra
to install, no supply-chain surface (the whole point of Irix). Run:

    irix-mcp                      # stdio server; wire into your MCP client

Tools exposed:
  scan_injection   read ALL text at multiple contrasts + flag prompt-injection
                   BEFORE any model sees the frame (the security differentiator)
  elements         pixel-CV UI element map (path/ndarray) — universal
  fuse_screen      hybrid a11y-tree + pixel-CV map of a window (Windows: exact)
  find_element     locate a widget by label; returns the exact click point
  read_text        model-free OCR of an image
  identify         name known objects + neutrally describe the rest
  benchmark        run the reproducible GUI-grounding benchmark

All deterministic, model-free; every handler fails soft to a JSON error.
"""
from __future__ import annotations

import json
import sys

PROTOCOL = "2024-11-05"

TOOLS = [
    {"name": "scan_injection",
     "description": "Read all text in an image at multiple contrast levels "
                    "(surfaces hidden text) and flag prompt-injection signatures "
                    "BEFORE any model sees the frame. Deterministic security screen.",
     "inputSchema": {"type": "object", "properties": {
         "image": {"type": "string", "description": "path to an image"}},
         "required": ["image"]}},
    {"name": "elements",
     "description": "Detect clickable UI elements in a screenshot via pixel CV "
                    "(works on any pixels — games, canvas, streamed screens). "
                    "Returns [{label, x, y, w, h, cx, cy}].",
     "inputSchema": {"type": "object", "properties": {
         "image": {"type": "string"}}, "required": ["image"]}},
    {"name": "fuse_screen",
     "description": "Hybrid element map: OS accessibility tree (exact rects+names "
                    "on Windows) fused with pixel CV. Pass a window handle and/or "
                    "an image. Best available map of the screen.",
     "inputSchema": {"type": "object", "properties": {
         "image": {"type": "string"}, "hwnd": {"type": "integer"},
         "interactive_only": {"type": "boolean"}}}},
    {"name": "find_element",
     "description": "Find a UI element by its label across the fused map and "
                    "return the exact click point {x, y, label, source, box}.",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string"}, "image": {"type": "string"},
         "hwnd": {"type": "integer"}}, "required": ["text"]}},
    {"name": "read_text",
     "description": "Model-free OCR of an image -> {text, lines, confidence}.",
     "inputSchema": {"type": "object", "properties": {
         "image": {"type": "string"}}, "required": ["image"]}},
    {"name": "identify",
     "description": "For every salient object: a neutral physical description "
                    "(no model) plus a remembered name if the gallery has seen it.",
     "inputSchema": {"type": "object", "properties": {
         "image": {"type": "string"},
         "gallery": {"type": "string", "description": "gallery zip path"}},
         "required": ["image", "gallery"]}},
    {"name": "benchmark",
     "description": "Run the reproducible GUI-grounding benchmark (synthetic UI "
                    "screens, click-in-bbox accuracy + latency).",
     "inputSchema": {"type": "object", "properties": {
         "screens": {"type": "integer"}, "blur": {"type": "boolean"}}}},
    {"name": "memory_remember",
     "description": "Store a screenshot in episodic visual memory "
                    "(timestamp, source, caption, Irix element parse). "
                    "Near-duplicate frames are stored once.",
     "inputSchema": {"type": "object", "properties": {
         "image": {"type": "string", "description": "path to a screenshot"},
         "source": {"type": "string", "description": "where this came from"},
         "caption": {"type": "string",
                       "description": "what this shows (auto-derived if omitted)"}},
         "required": ["image", "source"]}},
    {"name": "memory_recall",
     "description": "Search episodic visual memory by text query. Returns "
                    "matching episodes with image paths — open the image and "
                    "answer from pixels, never from the summary.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "n": {"type": "integer"}},
         "required": ["query"]}},
    {"name": "memory_list",
     "description": "List recent visual-memory episodes (newest first).",
     "inputSchema": {"type": "object", "properties": {
         "n": {"type": "integer"}}}},
]


def _memory_store():
    import os
    from .memory import MemoryStore
    root = os.environ.get("IRIX_MEMORY_DIR", "~/.irix/memory")
    return MemoryStore(root)


def _dispatch(name: str, args: dict):
    if name == "scan_injection":
        from . import inject
        return inject.scan(args["image"])
    if name == "elements":
        from . import elements
        return elements.parse(args["image"])
    if name == "fuse_screen":
        from . import fuse
        return fuse.parse(args.get("image"), hwnd=args.get("hwnd"),
                          interactive_only=args.get("interactive_only", False))
    if name == "find_element":
        from . import fuse
        return fuse.find(args["text"], image_or_path=args.get("image"),
                         hwnd=args.get("hwnd"))
    if name == "read_text":
        from . import text
        return text.read(args["image"])
    if name == "identify":
        from . import objects
        return objects.ObjectGallery(args["gallery"]).identify(args["image"])
    if name == "benchmark":
        from . import bench
        return bench.run(n_screens=args.get("screens", 12), blur=args.get("blur", True))
    if name == "memory_remember":
        return _memory_store().remember(
            args["image"], source=args["source"], caption=args.get("caption"))
    if name == "memory_recall":
        return _memory_store().recall(args["query"], n=args.get("n", 5))
    if name == "memory_list":
        return _memory_store().list(n=args.get("n", 10))
    raise ValueError(f"unknown tool: {name}")


def _handle(msg: dict):
    """Return a JSON-RPC response dict, or None for notifications."""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
            "serverInfo": {"name": "irix", "version": "0.3.1"}}}
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            result = _dispatch(name, args)
            text = json.dumps(result, default=str)
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text", "text": text}],
                               "isError": False}}
        except Exception as e:                       # tool failure -> soft error
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text",
                               "text": json.dumps({"error": f"{type(e).__name__}: {e}"})}],
                               "isError": True}}
    if mid is not None:                              # unknown method
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "method": method or "", "message": "method not found"}}
    return None


def serve(stdin=None, stdout=None) -> int:
    """Newline-delimited JSON-RPC 2.0 over stdio — the MCP stdio transport."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        resp = _handle(msg)
        if resp is not None:
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
    return 0


def main() -> int:
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
