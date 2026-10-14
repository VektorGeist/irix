"""gen_wordlist — build the bundled default dictionary for textcorr.

Irix's OCR correction (textcorr.py) repairs an unknown token to the nearest known
word, but ONLY across classic OCR confusion pairs. It needs a word list. This
generator writes a curated default of common English + UI/software vocabulary to
``src/irix/words.txt`` — fully local, no download, reproducible.

The default is deliberately MODEST and high-frequency: a smaller, common list
yields fewer false "corrections" than a giant dictionary (a rare real word can
otherwise hijack a confusion edit). Point ``IRIX_WORDLIST`` at a bigger list to
grow coverage — same "memory is just data, grows with use" principle as the
glyph and face galleries.

Run:  python tools/gen_wordlist.py
"""
from __future__ import annotations

from pathlib import Path

# High-frequency English (function words + the most common content words) and a
# focused UI/software vocabulary — the words OCR actually meets on a screen.
_SEED = """
a an the and or but nor for so yet of to in on at by with from into onto upon over
under above below between among through during before after about against without
within along across behind beside near off out up down here there where when why how
what which who whom whose this that these those then than as if is am are was were be
been being have has had do does did will would shall should can could may might must
not no yes all any both each few more most other some such only own same very just
i you he she it we they me him her us them my your his its our their mine yours ours
one two three four five six seven eight nine ten zero first second third next last
new old good bad big small large little long short high low full empty open close
add remove new edit view help file find save load copy paste cut undo redo print
cancel ok okay apply submit confirm accept decline retry reset clear delete create
update upgrade install uninstall download upload import export refresh reload sync
search filter sort select selected choose choice option options setting settings
preference preferences config configure account profile user username password login
logout signin signout register account email phone address name title label value
button menu window dialog tab panel page screen view list grid table row column cell
text image icon link url path folder directory document files name type size date
time today yesterday tomorrow start stop play pause resume record send receive share
connect disconnect online offline enable disable enabled disabled active inactive
on off auto manual default custom advanced general basic system network device
storage memory disk drive volume battery power volume bright dark light theme color
font style format bold italic underline align left right center justify size scale
zoom rotate crop resize move drag drop click double press hold release scroll swipe
back forward home next previous finish continue done ready loading error warning info
success failed failure complete completed pending progress status state message alert
notification reminder event task project note item entry record field form input
output result report log history recent favorite favourite bookmark pin tag category
group team member admin owner guest public private secure security privacy permission
access allow deny block trust verify validate check verified invalid valid required
optional available unavailable visible hidden show hide expand collapse minimize
maximize fullscreen exit quit restart shutdown sleep wake lock unlock account
welcome hello goodbye thanks please sorry yes no maybe more less again retry skip
""".split()


def build() -> list[str]:
    words = {w.lower() for w in _SEED if w.isalpha() and len(w) >= 1}
    # single letters that are genuinely words / matter for the l/I/1 confusion
    words.update({"a", "i"})
    return sorted(words)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "src" / "irix" / "words.txt"
    words = build()
    out.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"wrote {len(words)} words -> {out}")


if __name__ == "__main__":
    main()
