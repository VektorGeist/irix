"""Eidetrix episodic-memory recall benchmark.

120 realistic synthetic screenshots across 12 app scenarios (10 variants each).
Enrolled WITHOUT hand-written captions (auto-caption from Irix OCR/elements).
36 natural-language queries; measures recall@k: is a correct-scenario episode
in the top-k hits?

Target (v1 spec): recall@5 >= 0.90.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from PIL import Image, ImageDraw, ImageFont

from irix.memory import MemoryStore

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
FONT_BIG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)

W, H = 900, 620

# scenario -> (title, list of (variant_id, [text lines]))
SCENARIOS = {
    "login": ("Sign in", [
        ["Welcome back", "Email address", "Password", "Sign in button", "Forgot password link"],
        ["Member login", "Username", "Enter password", "Log in", "Create account"],
        ["Secure sign-in", "Work email", "Password field", "Continue", "SSO options"],
    ]),
    "checkout": ("Checkout", [
        ["Your cart", "3 items", "Subtotal $84.50", "Shipping $6.99", "Place order"],
        ["Payment", "Card number", "Expiry date", "Pay now $91.49", "Order summary"],
        ["Review order", "2 items", "Total $52.00", "Confirm purchase", "Edit cart"],
    ]),
    "invoice": ("Invoice", [
        ["Invoice #INV-2041", "Billed to Acme Corp", "Total due $1,240.00", "Pay by September 30"],
        ["Invoice #INV-2042", "Billed to Globex", "Amount $860.00", "Due October 15"],
        ["Receipt #R-9910", "Thank you for payment", "Charged $320.00", "Paid in full"],
    ]),
    "error": ("Error", [
        ["Something went wrong", "Error code 500", "Please try again", "Contact support"],
        ["Connection failed", "Retry connection", "Check network settings", "Error 101"],
        ["Save failed", "Disk full", "Free up space", "Try again button"],
    ]),
    "email": ("Inbox", [
        ["Inbox (12)", "From: boss@work.com", "Subject: Q3 planning meeting", "Tomorrow 10am"],
        ["Inbox (5)", "From: alerts@bank.com", "Subject: statement ready", "View statement"],
        ["Sent mail", "To: team@work.com", "Subject: launch notes", "Delivered"],
    ]),
    "settings": ("Settings", [
        ["Display settings", "Brightness slider", "Night mode toggle", "Resolution 1920x1080"],
        ["Privacy settings", "Location access", "Camera permission", "Microphone toggle"],
        ["Account settings", "Change password", "Two-factor auth", "Sign out everywhere"],
    ]),
    "docs": ("Document", [
        ["Project proposal", "Executive summary", "Budget $50,000", "Timeline 6 months"],
        ["Meeting notes", "Attendees: 8", "Action items: 5", "Next meeting Friday"],
        ["Research draft", "Introduction", "Methodology section", "References 24"],
    ]),
    "calendar": ("Calendar", [
        ["September 2026", "Dentist Tue 2pm", "Team lunch Fri", "Flight to Austin"],
        ["October 2026", "Conference Oct 12", "Birthday dinner", "Car service"],
        ["Week view", "Standup 9am daily", "Sprint review Thu", "Dinner reservation"],
    ]),
    "chat": ("Messages", [
        ["Chat with Ana", "Are we still on for lunch?", "Yes, noon at Bertos", "See you there"],
        ["Team channel", "Deploy is live", "Monitoring dashboards", "All green"],
        ["Chat with Lee", "Did you see the game?", "Overtime thriller", "Rematch Sunday"],
    ]),
    "files": ("Files", [
        ["Documents folder", "report.pdf", "budget.xlsx", "notes.txt", "Modified today"],
        ["Downloads", "installer.dmg", "photo.jpg", "archive.zip", "2.4 GB free"],
        ["Projects", "vismem-probe", "irix-main", "README.md", "Last opened"],
    ]),
    "music": ("Music", [
        ["Now playing", "Midnight Drive", "Neon Coast", "Playlist: night drive"],
        ["Library", "Artists 240", "Albums 85", "Recently added"],
        ["Radio", "Jazz station", "Up next: Blue Note", "Volume 70 percent"],
    ]),
    "maps": ("Maps", [
        ["Directions", "Home to office", "24 min via I-35", "Start navigation"],
        ["Nearby", "Coffee shops", "Gas stations", "Open now filter"],
        ["Saved places", "Work address", "Gym location", "Parents house"],
    ]),
}

# scenario -> natural queries (must find that scenario's episodes)
QUERIES = {
    "login": ["sign in page", "login button", "forgot password"],
    "checkout": ["place order", "shopping cart total", "pay now"],
    "invoice": ["invoice total due", "billed amount", "receipt payment"],
    "error": ["error message", "something went wrong", "connection failed"],
    "email": ["inbox email", "meeting subject", "bank statement"],
    "settings": ["brightness settings", "privacy toggle", "change password"],
    "docs": ["project proposal budget", "meeting notes action", "research draft"],
    "calendar": ["dentist appointment", "flight schedule", "team lunch"],
    "chat": ["lunch plans message", "deploy live", "game rematch"],
    "files": ["documents folder", "downloads free space", "project readme"],
    "music": ["now playing song", "jazz radio", "music library artists"],
    "maps": ["directions home office", "coffee shops nearby", "navigation start"],
}


def render(lines, path, accent=(40, 90, 160)):
    img = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 70], fill=accent)
    d.text((24, 18), lines[0], font=FONT_BIG, fill=(255, 255, 255))
    y = 110
    for line in lines[1:]:
        d.text((40, y), line, font=FONT, fill=(25, 25, 25))
        y += 52
    img.save(path)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="eidetrix-bench-"))
    mem = MemoryStore(tmp / "memory")
    truth = {}  # episode id -> scenario
    print("enrolling 120 episodes (auto-caption, no hand-written text)...")
    for scenario, (title, variants) in SCENARIOS.items():
        for vi in range(10):
            lines = variants[vi % len(variants)]
            # vary a detail per variant so episodes aren't pixel-identical
            vlines = [title] + [f"{l} (v{vi})" for l in lines]
            p = tmp / f"{scenario}-{vi}.png"
            accent = ((40 + vi * 37) % 200, (90 + vi * 53) % 200, 160)
            render(vlines, p, accent)
            r = mem.remember(str(p), source="benchmark")
            truth[r["id"]] = scenario
    print(f"enrolled {len(truth)} episodes")

    n_q = 0
    hits = {k: 0 for k in (1, 3, 5, 10)}
    misses = []
    for scenario, queries in QUERIES.items():
        for q in queries:
            n_q += 1
            res = mem.recall(q, n=10)
            got = [truth[h["id"]] for h in res]
            for k in hits:
                if scenario in got[:k]:
                    hits[k] += 1
            if scenario not in got[:5]:
                misses.append((q, scenario, got[:5]))
    print(f"\n{n_q} queries:")
    for k in hits:
        print(f"  recall@{k} = {hits[k]/n_q:.3f}")
    if misses:
        print(f"\n{len(misses)} queries missed @5 (showing 5):")
        for q, s, g in misses[:5]:
            print(f"  q={q!r} want={s} got={g}")


if __name__ == "__main__":
    main()
