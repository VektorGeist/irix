# Irix — deterministic, model-free eyes

> Named after the eye (and the Greek goddess of sight). Irix turns pixels into
> structured **text** and recognizes the world by **remembered exemplars** — never
> by a trained model.

## The thesis

Most vision stacks bolt a heavy learned model onto the pixels. Irix does the
opposite: cheap deterministic CV (OpenCV / OCR / optical flow) reads the screen
or scene, and **recognition is retrieval, not inference**. Irix *remembers* every
face and object it sees — stores a deterministic descriptor plus the cropped
frame into a zip-backed gallery — and recognizes by deterministic
nearest-neighbor match against that memory.

**There is no accuracy ceiling, because the gallery grows.** Every new exemplar
(a new angle, new lighting) makes the next match better. Same principle as a
search index: memory + a deterministic protocol = capability, no model. Runs on
CPU, no VRAM, pixel-exact.

## Capabilities

- **UI element parsing** (`irix.elements`) — detect rectangular widgets (edges +
  contours, texture-filtered so real UI ≠ art), OCR each box → a precise element
  map `[{label, box, center}]`. Set-of-Mark by deterministic label match.
- **Scene perception** (`irix.scene`) — `path()` (Canny+Hough steer),
  `freespace()` (drivable area by colour), `motion()` (Farneback optical flow),
  fused into structured text by `describe_scene()`.
- **Face gallery** (`irix.face`) — model-free recognition: detect → deterministic
  descriptor (LBP histogram + ORB) → zip storage container → recognize by
  nearest-neighbor. Enroll-on-sight, auto-enroll unknowns, recall improves with
  every frame.
- **Injection screen** (`irix.inject`) — reads all on-screen text at multiple
  contrast/scale levels (surfaces near-invisible HIDDEN text) and flags
  prompt-injection signatures *before* any model sees the pixels.
- **Tracking + reflex** (`irix.track`) — multi-cue deterministic tracker
  (colour + template/ORB + optical-flow prediction), re-acquire on loss. The
  fast reflex loop — faster than any model can run.
- **Visual memory** (`irix.memory`) — episodic screenshot memory for agents.
  `remember()` stores a screenshot with its OCR text and a caption;
  `recall()` finds it later by text query through SQLite FTS5 and hands back
  the original image so the agent inspects pixels, not a summary.
  Difference-hash dedup (no storing the same screen twice), retention by
  count/age/bytes, `forget()` wipes an episode. Exposed over MCP as
  `memory_remember` / `memory_recall` / `memory_list`.
- **Watch daemon** (`irix.watch`, `irix-watch` / `irix watch`) — captures the
  screen on an interval, stores only frames that changed past a threshold,
  auto-captions each via the memory store, logs JSON events. The always-on
  eyes behind visual memory.

## Rule

**No models in Irix. Ever.** Not a VLM, not a deep detector. Deterministic CV +
remembered exemplars only. If a capability seems to need a model, the answer is
more deterministic heads and a bigger gallery — never a model.

## Install

```
pip install -e .            # core
pip install -e ".[capture]" # + cross-platform screen capture (mss)
pip install -e ".[dev]"     # + pytest
```

### Bought the `.whl` file? (no git checkout needed)

You downloaded something like `irix_eyes-0.3.1-py3-none-any.whl`. Install it
with pip — that file *is* the install:

```
pip install irix_eyes-0.3.1-py3-none-any.whl   # use the exact file you downloaded
```

Requires Python 3.10+ (`python --version` to check; `python -m pip install
--upgrade pip` if pip is old). For screen capture also run `pip install mss`.

Check it worked:

```
irix --help        # CLI: capture, parse, read, describe
irix-watch --help  # screen-change capture daemon -> visual memory
irix-mcp           # MCP stdio server for agents (memory_remember / memory_recall / memory_list)
```

Then see "Visual memory quick start" below — `remember()` a screenshot and
`recall()` it back by text.

`opencv-contrib-python` is required (not plain `opencv-python`) — Irix uses
`cv2.face` (LBPH), which is an instance-based exemplar store, not a trained model.

## Quick start

```python
from irix import face
g = face.Gallery("people.irix")     # a zip storage container
g.enroll("frame.png", name="Flouk") # remember a face
print(g.recognize("another.png"))   # -> {"name": "Flouk", "dist": 0.31, ...}
```

## Visual memory quick start

```python
from irix.memory import MemoryStore

mem = MemoryStore("~/.irix/memory")
mem.remember("shot.png", source="agent", caption="login screen")  # -> {id, ...}
hits = mem.recall("login")            # -> [{id, caption, image_path, ...}]
mem.prune(max_episodes=5000)          # retention: count, age_days, max_bytes
mem.forget(hits[0]["id"])             # wipe one episode
```

Run the watcher (stores only changed frames):

```
irix-watch --interval 5 --threshold 8 --memory-dir ~/.irix/memory
# or: irix watch --interval 5
```

Over MCP (`irix-mcp` stdio server): `memory_remember`, `memory_recall`,
`memory_list`. The agent gets back the original image path — pixels, not a
summary.

Retrieval quality (synthetic benchmark, `bench_recall.py`): **recall@5 = 1.00**
over 205 unique episodes with descriptive queries. Near-duplicate screens
collapse by difference-hash (hamming ≤ 2), so repeats never inflate the store.
Real-world recall tracks OCR text quality on the agent's own screenshots.
