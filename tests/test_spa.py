"""Playwright E2E tests for the Escriba SPA — covers risky JS logic and keyboard nav.

All tests skip gracefully when Playwright or Chromium is not installed so that
``uv run pytest`` still passes in a fresh checkout.  To run these tests locally:

    uv sync
    uv run playwright install chromium
    uv run pytest tests/test_spa.py -v
"""
from __future__ import annotations

import http.server
import socketserver
import threading
from pathlib import Path
from typing import Generator

import pytest

STATIC_DIR = Path(__file__).parent.parent / "src" / "escriba" / "app" / "static"

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _playwright_importable() -> bool:
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _playwright_importable(),
    reason="playwright not installed — run: uv sync && uv run playwright install chromium",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spa_server() -> Generator[str, None, None]:
    """Serve the static SPA directory on an ephemeral port."""
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

        def log_message(self, *args):  # silence request logs in test output
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever)
        t.daemon = True
        t.start()
        yield f"http://127.0.0.1:{port}/index.html"
        httpd.shutdown()


@pytest.fixture(scope="module")
def chromium_browser(spa_server: str) -> Generator:
    """Launch Chromium; skip if it is not installed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed")
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(
                f"Chromium not installed — run: uv run playwright install chromium\n({exc})"
            )
        yield browser
        browser.close()


_API_STUBS: dict[str, str] = {
    "/api/status": '{"ok":true,"is_active":false,"session_id":null}',
    "/api/sessions": '{"ok":true,"sessions":[],"folders":[]}',
    "/api/config": '{"ok":true,"config":{"prompts":{"system_prompt":"","templates":[],"effective_system_prompt":""}},"env_keys":{}}',
    "/api/models": '{"ok":true,"models":{},"ai_available":false,"ai_unavailable_reason":""}',
}


def _stub_api(route) -> None:
    from urllib.parse import urlparse
    path = urlparse(route.request.url).path
    body = _API_STUBS.get(path, '{"ok":true}')
    route.fulfill(status=200, content_type="application/json", body=body)


@pytest.fixture
def page(chromium_browser, spa_server: str):
    """Navigate to the SPA with all API calls mocked out."""
    ctx = chromium_browser.new_context()
    pg = ctx.new_page()
    pg.route("**/api/**", _stub_api)
    pg.goto(spa_server)
    pg.wait_for_load_state("domcontentloaded")
    pg.wait_for_timeout(300)
    yield pg
    ctx.close()


@pytest.fixture
def dark_page(chromium_browser, spa_server: str):
    """SPA loaded with dark-mode emulation."""
    ctx = chromium_browser.new_context(color_scheme="dark")
    pg = ctx.new_page()
    pg.route("**/api/**", _stub_api)
    pg.goto(spa_server)
    pg.wait_for_load_state("domcontentloaded")
    pg.wait_for_timeout(300)
    yield pg
    ctx.close()


# ---------------------------------------------------------------------------
# T1-a: XSS / escaping — escHtml() and innerHTML sinks
# ---------------------------------------------------------------------------

def test_esc_html_neutralizes_script_tag(page) -> None:
    """escHtml('<script>...') must not pass a raw <script> tag through."""
    result = page.evaluate("escHtml('<script>alert(1)</script>')")
    assert "<script>" not in result
    # The content is HTML-entity-encoded
    assert "&lt;script&gt;" in result or "alert" not in result


def test_esc_html_neutralizes_angle_bracket_injection(page) -> None:
    """escHtml() must escape < and > so injected tags cannot execute."""
    result = page.evaluate('escHtml(\'<img src=x onerror="evil()">\')')
    # No raw < or > — the tag cannot be parsed by the browser
    assert "<img" not in result
    assert "&lt;img" in result


def test_esc_attr_neutralizes_quote_injection(page) -> None:
    """escAttr() (used in HTML attribute sinks) must escape double-quotes."""
    result = page.evaluate('escAttr(\'" onmouseover="evil()\')')
    assert '"' not in result or "&quot;" in result


def test_render_markdown_escapes_html_before_processing(page) -> None:
    """renderMarkdown() calls escHtml first so no raw <script> survives."""
    result = page.evaluate("renderMarkdown('<script>alert(xss)</script> text')")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


# ---------------------------------------------------------------------------
# T1-b: Markdown rendering + GFM tables
# ---------------------------------------------------------------------------

def test_render_markdown_renders_gfm_table(page) -> None:
    """GFM tables render as HTML <table> elements."""
    md = "| Col A | Col B |\n|-------|-------|\n| val 1 | val 2 |"
    result = page.evaluate(f"renderMarkdown({repr(md)})")
    assert "<table>" in result
    assert "<th>" in result
    assert "Col A" in result
    assert "<td>" in result
    assert "val 1" in result


def test_render_markdown_table_escapes_cell_html(page) -> None:
    """HTML inside a GFM table cell is escaped, not rendered."""
    md = "| Injection |\n|-------|\n| <script>evil()</script> |"
    result = page.evaluate(f"renderMarkdown({repr(md)})")
    assert "<script>" not in result


def test_render_markdown_inline_html_is_escaped(page) -> None:
    """Inline HTML in markdown input is escaped — no live <img> or event attributes."""
    result = page.evaluate("renderMarkdown('**bold** <img src=x onerror=alert(1)>')")
    # No raw <img> tag that the browser would render
    assert "<img" not in result
    # onerror may appear as safe entity-encoded text (&lt;img ...onerror...&gt;),
    # but must never appear as an active attribute inside an unescaped tag.
    assert '<img' not in result


# ---------------------------------------------------------------------------
# T1-c: Deep-link parsing
# ---------------------------------------------------------------------------

def test_parse_deep_link_round_trips_valid_hash(page) -> None:
    """parseDeepLink() extracts sessionId and segmentId from a well-formed hash."""
    result = page.evaluate("""() => {
        history.replaceState(null, '', '#session/my-sess-id/seg/42');
        return parseDeepLink();
    }""")
    assert result is not None
    assert result["sessionId"] == "my-sess-id"
    assert result["segmentId"] == 42


def test_parse_deep_link_empty_hash_returns_null(page) -> None:
    result = page.evaluate("""() => {
        history.replaceState(null, '', '#');
        return parseDeepLink();
    }""")
    assert result is None


def test_parse_deep_link_malformed_hashes_are_safe(page) -> None:
    """Adversarial or malformed hashes return null without throwing."""
    bad_inputs = [
        "#session///seg/0",
        "#<script>alert(1)</script>",
        "#session/abc/seg/notanumber",
        "#completely-different",
        "#",
    ]
    for h in bad_inputs:
        result = page.evaluate(f"""() => {{
            history.replaceState(null, '', {repr(h)});
            return parseDeepLink();
        }}""")
        assert result is None, f"Expected null for {h!r}, got {result!r}"


def test_parse_deep_link_preserves_complex_session_ids(page) -> None:
    """UUIDs and hyphenated session IDs round-trip correctly."""
    result = page.evaluate("""() => {
        history.replaceState(null, '', '#session/a1b2c3-def/seg/99');
        return parseDeepLink();
    }""")
    assert result is not None
    assert result["sessionId"] == "a1b2c3-def"
    assert result["segmentId"] == 99


# ---------------------------------------------------------------------------
# T1-d: Notes-generation state scoping
# ---------------------------------------------------------------------------

def test_notes_generation_guard_prevents_cross_session_bleed(page) -> None:
    """generateSessionNotes() must NOT update session-a's textarea with session-b's notes.

    Drives the real ``generateSessionNotes()`` function:
    - starts generation for session-b
    - switches ``selectedSessionId`` to session-a WHILE the API call is in-flight
    - lets the response arrive (notes for session-b)
    - asserts session-a's notes textarea is untouched

    This test FAILS if the guard ``if (selectedSessionId === sid)`` (~line 2929 in
    index.html) is removed — without it, session-b's notes overwrite session-a's view.
    """
    import threading

    page.evaluate("""() => {
        selectedSessionId = 'session-b';
        notesGeneratingFor = null;
        document.getElementById('session-notes-input').value = '';
        document.getElementById('session-prompt-input').value = '';
    }""")

    allow_response = threading.Event()

    def hold_route(route) -> None:
        # Block until Python signals us, then fulfil.  request_in_flight is
        # detected via page.expect_request() on the main thread (no race).
        allow_response.wait(timeout=10)
        route.fulfill(
            status=200,
            content_type="application/json",
            body='{"ok":true,"notes":"SENTINEL-NOTES-FOR-SESSION-B"}',
        )

    page.route("**/api/sessions/session-b/generate-notes", hold_route)

    # expect_request() sets up the listener BEFORE the JS fires — no race condition.
    with page.expect_request("**/api/sessions/session-b/generate-notes", timeout=5000):
        page.evaluate("void generateSessionNotes()")
    # The request is now in-flight and hold_route is blocking.  Switch sessions.
    page.evaluate("selectedSessionId = 'session-a'")

    allow_response.set()
    page.wait_for_timeout(600)

    notes_value = page.evaluate("document.getElementById('session-notes-input').value")
    page.unroute("**/api/sessions/session-b/generate-notes")

    assert "SENTINEL-NOTES-FOR-SESSION-B" not in notes_value, (
        "Cross-session bleed: session-b's notes appeared in session-a's textarea. "
        "The guard `if (selectedSessionId === sid)` is not working."
    )


def test_notes_generating_updates_own_session_when_not_switched(page) -> None:
    """generateSessionNotes() DOES update the textarea when the user stays on the same session.

    This is the positive complement to the bleed test: if the guard were over-aggressive
    (e.g. ``if (false)``) the notes would never appear — this test would catch that.
    """
    import threading

    page.evaluate("""() => {
        selectedSessionId = 'session-c';
        notesGeneratingFor = null;
        document.getElementById('session-notes-input').value = '';
        document.getElementById('session-prompt-input').value = '';
    }""")

    allow_response = threading.Event()

    def hold_route(route) -> None:
        allow_response.wait(timeout=10)
        route.fulfill(
            status=200,
            content_type="application/json",
            body='{"ok":true,"notes":"SENTINEL-NOTES-FOR-SESSION-C"}',
        )

    page.route("**/api/sessions/session-c/generate-notes", hold_route)
    # saveNotesForSession calls POST /api/sessions/session-c/notes — stub it
    page.route("**/api/sessions/session-c/notes", lambda r: r.fulfill(
        status=200, content_type="application/json", body='{"ok":true}'
    ))

    with page.expect_request("**/api/sessions/session-c/generate-notes", timeout=5000):
        page.evaluate("void generateSessionNotes()")
    # Do NOT switch session — stay on session-c
    allow_response.set()
    page.wait_for_timeout(600)

    notes_value = page.evaluate("document.getElementById('session-notes-input').value")
    page.unroute("**/api/sessions/session-c/generate-notes")
    page.unroute("**/api/sessions/session-c/notes")

    assert "SENTINEL-NOTES-FOR-SESSION-C" in notes_value, (
        "Notes did not appear in the textarea when the session was not switched. "
        "The guard `if (selectedSessionId === sid)` may be incorrectly blocking updates."
    )


# ---------------------------------------------------------------------------
# T1-e: Dark-mode legibility
# ---------------------------------------------------------------------------

def test_dark_mode_text_color_differs_from_surface(dark_page) -> None:
    """In dark mode --text must not equal --card or --bg (would make text invisible)."""
    colors = dark_page.evaluate("""() => {
        const s = getComputedStyle(document.documentElement);
        return {
            text: s.getPropertyValue('--text').trim(),
            card: s.getPropertyValue('--card').trim(),
            bg:   s.getPropertyValue('--bg').trim(),
        };
    }""")
    assert colors["text"] != colors["card"], (
        f"Dark mode: --text ({colors['text']}) == --card ({colors['card']}) — "
        "text would be invisible on card surface"
    )
    assert colors["text"] != colors["bg"], (
        f"Dark mode: --text ({colors['text']}) == --bg ({colors['bg']}) — "
        "text would be invisible on background"
    )


def test_dark_mode_text_is_light_colored(dark_page) -> None:
    """In dark mode the text color must be a light hex value (catches black-on-dark regression)."""
    text_color = dark_page.evaluate("""() =>
        getComputedStyle(document.documentElement).getPropertyValue('--text').trim()
    """)
    hex_color = text_color.lstrip("#")
    assert len(hex_color) == 6, f"Unexpected CSS color format: {text_color!r}"
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    assert brightness > 0.5, (
        f"Dark mode text color {text_color} has low brightness ({brightness:.2f}) — "
        "text may be unreadable on a dark background"
    )


# ---------------------------------------------------------------------------
# T2: Keyboard navigation — session list + audio player controls
# ---------------------------------------------------------------------------

def test_arrow_down_navigates_to_next_session(page) -> None:
    """ArrowDown calls selectSession with the next session's id."""
    page.evaluate("""() => {
        allSessions = [
            {id:'s1', name:'First', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
            {id:'s2', name:'Second', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
        ];
        allFolders = [];
        sessionOrder = ['s1', 's2'];
        selectedSessionId = 's1';
        window.__navCalls = [];
        window.__origSelectSession = window.selectSession;
        window.selectSession = async (id) => { window.__navCalls.push(id); };
    }""")

    page.keyboard.press("ArrowDown")
    page.wait_for_timeout(150)

    last = page.evaluate("window.__navCalls.slice(-1)[0]")
    page.evaluate("window.selectSession = window.__origSelectSession")
    assert last == "s2", f"Expected s2, got {last!r}"


def test_arrow_up_wraps_from_first_to_last_session(page) -> None:
    """ArrowUp from the first session wraps to the last."""
    page.evaluate("""() => {
        allSessions = [
            {id:'s1', name:'First', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
            {id:'s2', name:'Second', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
            {id:'s3', name:'Third', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
        ];
        allFolders = [];
        sessionOrder = ['s1', 's2', 's3'];
        selectedSessionId = 's1';
        window.__navCalls = [];
        window.__origSelectSession = window.selectSession;
        window.selectSession = async (id) => { window.__navCalls.push(id); };
    }""")

    page.keyboard.press("ArrowUp")
    page.wait_for_timeout(150)

    last = page.evaluate("window.__navCalls.slice(-1)[0]")
    page.evaluate("window.selectSession = window.__origSelectSession")
    assert last == "s3", f"Expected s3 (wrap-around), got {last!r}"


def test_arrow_down_with_no_sessions_is_safe(page) -> None:
    """ArrowDown when session list is empty must not throw."""
    page.evaluate("""() => {
        allSessions = [];
        allFolders = [];
        sessionOrder = [];
        selectedSessionId = null;
    }""")
    page.keyboard.press("ArrowDown")
    page.wait_for_timeout(100)
    # No assertion needed — the test passes if no exception is thrown


def test_space_toggles_audio_play_pause(page) -> None:
    """Space key calls audio.play() when paused, audio.pause() when playing."""
    result = page.evaluate("""() => {
        const audio = document.getElementById('session-audio');
        const calls = [];
        audio.play = () => { calls.push('play'); return Promise.resolve(); };
        audio.pause = () => { calls.push('pause'); };
        // Simulate a loaded audio element
        Object.defineProperty(audio, 'readyState', { get: () => 4, configurable: true });
        Object.defineProperty(audio, 'src', { get: () => 'http://localhost/a.wav', configurable: true });
        Object.defineProperty(audio, 'paused', { get: () => true, configurable: true });
        document.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
        return calls;
    }""")
    assert "play" in result


def test_arrow_left_seeks_backward(page) -> None:
    """ArrowLeft seeks the audio back by 5 seconds."""
    current_time = page.evaluate("""() => {
        const audio = document.getElementById('session-audio');
        Object.defineProperty(audio, 'readyState', { get: () => 4, configurable: true });
        Object.defineProperty(audio, 'src', { get: () => 'http://localhost/a.wav', configurable: true });
        audio.currentTime = 30;
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
        return audio.currentTime;
    }""")
    assert current_time == pytest.approx(25, abs=1)


def test_arrow_right_seeks_forward(page) -> None:
    """ArrowRight seeks the audio forward by 5 seconds."""
    current_time = page.evaluate("""() => {
        const audio = document.getElementById('session-audio');
        Object.defineProperty(audio, 'readyState', { get: () => 4, configurable: true });
        Object.defineProperty(audio, 'src', { get: () => 'http://localhost/a.wav', configurable: true });
        audio.currentTime = 10;
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
        return audio.currentTime;
    }""")
    assert current_time == pytest.approx(15, abs=1)


def test_nav_keys_ignored_when_input_focused(page) -> None:
    """Navigation keys do not fire when an input element has focus."""
    page.evaluate("""() => {
        allSessions = [
            {id:'s1', name:'First', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
            {id:'s2', name:'Second', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
        ];
        allFolders = [];
        sessionOrder = ['s1', 's2'];
        selectedSessionId = 's1';
        window.__navCalls = [];
        window.__origSelectSession2 = window.selectSession;
        window.selectSession = async (id) => { window.__navCalls.push(id); };
    }""")

    # Focus an input before pressing ArrowDown
    page.evaluate("""() => {
        const inp = document.getElementById('search-input') || document.querySelector('input');
        if (inp) inp.focus();
    }""")
    page.keyboard.press("ArrowDown")
    page.wait_for_timeout(100)

    calls = page.evaluate("window.__navCalls.length")
    page.evaluate("window.selectSession = window.__origSelectSession2")
    assert calls == 0, "nav keys should be ignored when an input is focused"


def test_session_item_enter_key_selects_session(page) -> None:
    """Pressing Enter on a session item triggers selection."""
    page.evaluate("""() => {
        allSessions = [
            {id:'s1', name:'Session', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
        ];
        allFolders = [];
        sessionOrder = ['s1'];
        selectedSessionId = null;
        window.__enterCalls = [];
        window.__origSelectSession3 = window.selectSession;
        window.selectSession = async (id) => { window.__enterCalls.push(id); };
        renderSessionList();
    }""")

    # Find the rendered session item and dispatch Enter
    page.evaluate("""() => {
        const el = document.querySelector('.session-item[data-id="s1"]');
        if (el) el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    }""")
    page.wait_for_timeout(100)

    calls = page.evaluate("window.__enterCalls")
    page.evaluate("window.selectSession = window.__origSelectSession3")
    assert "s1" in calls


def test_space_on_focused_session_item_selects_it_without_toggling_audio(page) -> None:
    """Space on a session-item selects it but must NOT toggle audio playback.

    This test FAILS without ``e.stopPropagation()`` in ``_sessionItemKeyDown``:
    without it, the Space event bubbles up to ``_handleNavKeys`` which calls
    ``audio.play()`` / ``audio.pause()`` — double-firing and corrupting playback state.
    """
    page.evaluate("""() => {
        allSessions = [
            {id:'s1', name:'Session', status:'completed', segment_count:0, folder_id:null,
             started_at:null, duration_seconds:null},
        ];
        allFolders = [];
        sessionOrder = ['s1'];
        selectedSessionId = null;

        // Track select calls
        window.__spaceCalls = [];
        window.__origSelectSession4 = window.selectSession;
        window.selectSession = async (id) => { window.__spaceCalls.push(id); };

        // Track audio play/pause calls
        const audio = document.getElementById('session-audio');
        window.__audioCalls = [];
        audio.play = () => { window.__audioCalls.push('play'); return Promise.resolve(); };
        audio.pause = () => { window.__audioCalls.push('pause'); };
        // Set audio to a loaded, paused state so Space WOULD toggle it if it bubbled
        Object.defineProperty(audio, 'readyState', { get: () => 4, configurable: true });
        Object.defineProperty(audio, 'src', { get: () => 'http://localhost/a.wav', configurable: true });
        Object.defineProperty(audio, 'paused', { get: () => true, configurable: true });

        renderSessionList();
    }""")

    # Dispatch Space on the session item element (as if the item has keyboard focus)
    page.evaluate("""() => {
        const el = document.querySelector('.session-item[data-id="s1"]');
        if (el) el.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
    }""")
    page.wait_for_timeout(100)

    result = page.evaluate("""() => {
        window.selectSession = window.__origSelectSession4;
        return { spaceCalls: window.__spaceCalls, audioCalls: window.__audioCalls };
    }""")
    assert "s1" in result["spaceCalls"], "Space on session item must trigger selection"
    assert result["audioCalls"] == [], (
        "Space on a session item must NOT toggle audio — "
        "stopPropagation() is missing from _sessionItemKeyDown"
    )
